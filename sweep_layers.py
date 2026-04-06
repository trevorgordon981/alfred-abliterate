#!/usr/bin/env python3
"""
Sweep across different top-k layer counts and compare refusal scores.
Loads the model ONCE, then toggles the projection hook between configurations.

Usage:
  python sweep_layers.py --model <path> --direction refusal_direction.npz \
    --prompts prompts_refusal.json --top-ks 0,4,8,12 --max-prompts 3
"""

import argparse
import json
import time

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate

from test_abliteration import refusal_score
from project_inference import attach_projection


def detach_projection():
    """Restore DecoderLayer.__call__ to its original (undo attach_projection)."""
    from mlx_lm.models.qwen3_5 import DecoderLayer
    cur = DecoderLayer.__call__
    orig = getattr(cur, "_original", None)
    if orig is not None:
        DecoderLayer.__call__ = orig


def render_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--top-ks", default="0,4,8,12", help="Comma-separated top-K values to sweep (0 = baseline)")
    p.add_argument("--max-prompts", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=400)
    args = p.parse_args()

    prompts = json.loads(open(args.prompts).read())[: args.max_prompts]
    d = np.load(args.direction)
    directions = d["directions"]
    best_layers = d["best_layers"].tolist()
    top_ks = [int(k) for k in args.top_ks.split(",")]

    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    rendered = [render_prompt(tokenizer, p) for p in prompts]

    all_results = {}  # k -> list of (prompt, score, text)
    for k in top_ks:
        detach_projection()
        if k > 0:
            target = set(best_layers[:k])
            attach_projection(model, directions, target)
            label = f"top-{k} (layers {sorted(target)})"
        else:
            label = "BASELINE (no projection)"
        print(f"\n=== {label} ===")
        run_results = []
        for i, (raw, rp) in enumerate(zip(prompts, rendered), 1):
            t_start = time.time()
            resp = generate(model, tokenizer, prompt=rp,
                            max_tokens=args.max_tokens, verbose=False)
            elapsed = time.time() - t_start
            score = refusal_score(resp)
            print(f"  [{i}/{len(prompts)}] {elapsed:.1f}s  score={score:.2f}  {raw[:60]!r}")
            run_results.append((raw, score, resp))
        all_results[k] = run_results

    # Restore original
    detach_projection()

    # Summary table
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'config':<15} {'mean_score':<12} {'min':<6} {'max':<6}")
    for k in top_ks:
        scores = [s for _, s, _ in all_results[k]]
        mean = np.mean(scores)
        label = f"top-{k}" if k > 0 else "baseline"
        print(f"{label:<15} {mean:<12.3f} {min(scores):<6.2f} {max(scores):<6.2f}")

    # Full outputs per prompt, each config's output side-by-side
    print("\n" + "=" * 70)
    print("FULL OUTPUTS (per prompt, per config)")
    print("=" * 70)
    for i, prompt in enumerate(prompts):
        print(f"\n### PROMPT {i+1}: {prompt}")
        for k in top_ks:
            _, score, text = all_results[k][i]
            label = f"top-{k}" if k > 0 else "baseline"
            print(f"\n--- {label} (score={score:.2f}) ---")
            print(text[:1200] + ("..." if len(text) > 1200 else ""))


if __name__ == "__main__":
    main()
