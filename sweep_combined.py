#!/usr/bin/env python3
"""
Sweep top-K with COMBINED directions (Gram-Schmidt orthogonalized).
Tests CN-political + non-CN + benign prompts simultaneously.

Usage:
  python sweep_combined.py --model <path> \
    --directions refusal_direction.npz,safety_direction.npz \
    --prompts prompts_mixed.json --top-ks 0,20,24,28,32
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
    p.add_argument("--directions", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--top-ks", default="0,20,24,28")
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=250)
    args = p.parse_args()

    direction_files = [f.strip() for f in args.directions.split(",") if f.strip()]
    direction_arrays = [np.load(f)["directions"] for f in direction_files]
    magnitudes_per_layer = sum(np.load(f)["magnitudes"] for f in direction_files)
    order = np.argsort(-magnitudes_per_layer)

    prompts = json.loads(open(args.prompts).read())
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    top_ks = [int(k) for k in args.top_ks.split(",")]

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)
    rendered = [render_prompt(tokenizer, p) for p in prompts]

    primary = direction_arrays[0]
    extras = direction_arrays[1:] if len(direction_arrays) > 1 else None

    all_results = {}
    for k in top_ks:
        detach_projection()
        if k > 0:
            target = set(order[:k].tolist())
            attach_projection(model, primary, target, extra_directions=extras)
            label = f"top-{k}"
        else:
            label = "baseline"
        print(f"\n=== {label} ===")
        run_results = []
        for i, (raw, rp) in enumerate(zip(prompts, rendered), 1):
            t0 = time.time()
            text = generate(model, tokenizer, prompt=rp,
                            max_tokens=args.max_tokens, verbose=False)
            score = refusal_score(text)
            print(f"  [{i}/{len(prompts)}] {time.time()-t0:.1f}s  score={score:.2f}  {raw[:55]!r}")
            run_results.append((raw, score, text))
        all_results[k] = run_results

    detach_projection()

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY (combined directions)")
    print("=" * 70)
    for k in top_ks:
        scores = [s for _, s, _ in all_results[k]]
        label = f"top-{k}" if k > 0 else "baseline"
        print(f"  {label:<12} mean={np.mean(scores):.3f}  min={min(scores):.2f}  max={max(scores):.2f}")

    # Per-prompt side-by-side
    print("\n" + "=" * 70)
    print("OUTPUTS (first 600 chars each)")
    print("=" * 70)
    for i, prompt in enumerate(prompts):
        print(f"\n### PROMPT {i+1}: {prompt}")
        for k in top_ks:
            _, score, text = all_results[k][i]
            label = f"top-{k}" if k > 0 else "baseline"
            print(f"\n--- {label} (score={score:.2f}) ---")
            print(text[:600] + ("..." if len(text) > 600 else ""))


if __name__ == "__main__":
    main()
