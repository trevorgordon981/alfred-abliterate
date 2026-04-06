#!/usr/bin/env python3
"""
Run inference on an mlx_lm model with refusal-direction projection applied.

Monkey-patches each DecoderLayer's __call__ to subtract the refusal direction's
projection from the layer output before it's returned. This abliterates the
model at inference time without modifying any weights on disk.

Usage:
  python project_inference.py \
    --model ~/models/qwen3.5-122b-a10b-4bit \
    --direction refusal_direction.npz \
    --layers 20,24,28,32,36,40 \
    --prompt "What happened at Tiananmen in 1989?"

If --layers is omitted, the script auto-selects the top-K layers by magnitude
(default K=8) from the direction file's best_layers array.

Tips:
  - Start with 4-8 layers. More layers = more aggressive abliteration but
    higher risk of capability degradation.
  - Concentrate on middle-late layers (30-70% depth) where the refusal
    signal typically peaks.
  - If the model still refuses after 8-layer projection, increase to 12-16
    or add layers specifically from the top-magnitude list.
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate


def gram_schmidt(vecs: list[np.ndarray]) -> list[np.ndarray]:
    """Orthonormalize a list of vectors. Each input should already be unit norm."""
    out = []
    for v in vecs:
        u = v.astype(np.float32).copy()
        for w in out:
            u = u - np.dot(u, w) * w
        n = np.linalg.norm(u)
        if n < 1e-6:
            continue  # linearly dependent with previous vectors, drop
        out.append(u / n)
    return out


def attach_projection(model, directions: np.ndarray, target_layers: set[int],
                      extra_directions: list[np.ndarray] | None = None) -> None:
    """
    Monkey-patch DecoderLayer.__call__ at the CLASS level to project out one
    or more refusal directions for targeted layer indices.

    Each extra_directions entry is an [num_layers, hidden_dim] array, parallel
    to the main `directions`. All directions for a given layer are
    Gram-Schmidt orthonormalized, then projected out of the layer output
    sequentially. This lets you compose multiple refusal-subspace removals
    (e.g. Chinese-political + Western-safety).

    Instance-level __call__ overrides don't work because Python looks up
    __call__ on the type, not the instance — so we patch at the class level
    and use id(layer_instance) to dispatch per-layer directions.
    """
    tm = getattr(model, "language_model", model)
    tm = getattr(tm, "model", tm)
    layers = tm.layers

    from mlx_lm.models.qwen3_5 import DecoderLayer

    all_direction_sets = [directions]
    if extra_directions:
        all_direction_sets.extend(extra_directions)

    # Map each targeted layer's instance id to a LIST of mx arrays (one per
    # orthonormalized direction for that layer).
    direction_map: dict[int, list] = {}
    for i in target_layers:
        per_layer_vecs = [ds[i] for ds in all_direction_sets]
        orth = gram_schmidt(per_layer_vecs)
        direction_map[id(layers[i])] = [mx.array(v, dtype=mx.float32) for v in orth]

    original_call = DecoderLayer.__call__

    def wrapped_call(self, x, mask=None, cache=None):
        out = original_call(self, x, mask=mask, cache=cache)
        dirs = direction_map.get(id(self))
        if dirs is not None:
            for d in dirs:
                d_cast = d.astype(out.dtype)
                proj_coef = mx.sum(out * d_cast, axis=-1, keepdims=True)
                out = out - proj_coef * d_cast
        return out

    DecoderLayer.__call__ = wrapped_call
    wrapped_call._original = original_call

    num_dirs = 1 + (len(extra_directions) if extra_directions else 0)
    print(f"Attached projection hooks to {len(target_layers)} layers, "
          f"{num_dirs} direction(s) each: {sorted(target_layers)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--direction", required=True, help=".npz from compute_direction.py")
    p.add_argument("--layers", default="", help="Comma-separated layer indices (overrides top-K)")
    p.add_argument("--top-k", type=int, default=8, help="Fallback: use top-K layers by magnitude")
    p.add_argument("--prompt", required=True, help="Prompt to test abliterated model on")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.7)
    args = p.parse_args()

    d = np.load(args.direction)
    directions = d["directions"]   # [L, D]
    best_layers = d["best_layers"].tolist()

    if args.layers:
        target_layers = set(int(x) for x in args.layers.split(",") if x.strip())
    else:
        target_layers = set(best_layers[: args.top_k])

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    attach_projection(model, directions, target_layers)

    # Apply chat template
    messages = [{"role": "user", "content": args.prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = args.prompt

    print(f"\n=== PROMPT ===\n{args.prompt}")
    print(f"\n=== OUTPUT (abliterated, temp={args.temp}) ===")
    response = generate(
        model, tokenizer,
        prompt=prompt_text,
        max_tokens=args.max_tokens,
        verbose=False,
        sampler=None,  # default
    )
    print(response)


if __name__ == "__main__":
    main()
