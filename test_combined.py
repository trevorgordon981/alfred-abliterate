#!/usr/bin/env python3
"""
Test the combined effect of TWO refusal directions (e.g. Chinese-political +
Western-safety) applied simultaneously with Gram-Schmidt orthogonalization.

Usage:
  python test_combined.py --model <path> \
    --directions refusal_direction.npz,safety_direction.npz \
    --prompts prompts_non_cn.json --top-k 20 --max-prompts 4
"""

import argparse
import json
import time

import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate

from project_inference import attach_projection
from sweep_layers import detach_projection, render_prompt
from test_abliteration import refusal_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--directions", required=True,
                   help="Comma-separated list of direction .npz files (first is primary)")
    p.add_argument("--prompts", required=True)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--max-prompts", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=300)
    args = p.parse_args()

    direction_files = [f.strip() for f in args.directions.split(",") if f.strip()]
    if len(direction_files) < 1:
        raise SystemExit("Need at least one direction file")

    # Load all direction arrays
    direction_arrays = []
    for f in direction_files:
        d = np.load(f)
        direction_arrays.append(d["directions"])
    primary = direction_arrays[0]
    extras = direction_arrays[1:] if len(direction_arrays) > 1 else None

    # Use top-K by combined magnitude across all directions
    # (sum of magnitudes per layer = where the most refusal signal lives)
    magnitudes_per_layer = np.zeros(primary.shape[0])
    for f in direction_files:
        magnitudes_per_layer += np.load(f)["magnitudes"]
    order = np.argsort(-magnitudes_per_layer)
    target_layers = set(order[: args.top_k].tolist())

    prompts = json.loads(open(args.prompts).read())[: args.max_prompts]

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)
    rendered = [render_prompt(tokenizer, p) for p in prompts]

    print(f"\nDirections: {direction_files}")
    print(f"Top-{args.top_k} combined-magnitude layers: {sorted(target_layers)}")

    # BASELINE
    detach_projection()
    print("\n=== BASELINE ===")
    baseline = []
    for i, (raw, rp) in enumerate(zip(prompts, rendered), 1):
        t0 = time.time()
        text = generate(model, tokenizer, prompt=rp, max_tokens=args.max_tokens, verbose=False)
        score = refusal_score(text)
        print(f"  [{i}/{len(prompts)}] {time.time()-t0:.1f}s  score={score:.2f}  {raw[:60]!r}")
        baseline.append((raw, score, text))

    # SINGLE direction (Chinese political only)
    detach_projection()
    attach_projection(model, primary, target_layers, extra_directions=None)
    print("\n=== SINGLE DIRECTION (Chinese political only) ===")
    single = []
    for i, (raw, rp) in enumerate(zip(prompts, rendered), 1):
        t0 = time.time()
        text = generate(model, tokenizer, prompt=rp, max_tokens=args.max_tokens, verbose=False)
        score = refusal_score(text)
        print(f"  [{i}/{len(prompts)}] {time.time()-t0:.1f}s  score={score:.2f}  {raw[:60]!r}")
        single.append((raw, score, text))

    # COMBINED (Gram-Schmidt orthogonalized)
    detach_projection()
    attach_projection(model, primary, target_layers, extra_directions=extras)
    print(f"\n=== COMBINED ({len(direction_files)} directions, Gram-Schmidt) ===")
    combined = []
    for i, (raw, rp) in enumerate(zip(prompts, rendered), 1):
        t0 = time.time()
        text = generate(model, tokenizer, prompt=rp, max_tokens=args.max_tokens, verbose=False)
        score = refusal_score(text)
        print(f"  [{i}/{len(prompts)}] {time.time()-t0:.1f}s  score={score:.2f}  {raw[:60]!r}")
        combined.append((raw, score, text))

    detach_projection()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  baseline:  mean={np.mean([s for _,s,_ in baseline]):.3f}")
    print(f"  single-dir: mean={np.mean([s for _,s,_ in single]):.3f}")
    print(f"  combined:  mean={np.mean([s for _,s,_ in combined]):.3f}")

    # Full outputs
    print("\n" + "=" * 70)
    print("FULL OUTPUTS")
    print("=" * 70)
    for i, prompt in enumerate(prompts):
        print(f"\n### PROMPT {i+1}: {prompt}")
        for label, data in (("baseline", baseline), ("single", single), ("combined", combined)):
            _, score, text = data[i]
            print(f"\n--- {label} (score={score:.2f}) ---")
            print(text[:800] + ("..." if len(text) > 800 else ""))


if __name__ == "__main__":
    main()
