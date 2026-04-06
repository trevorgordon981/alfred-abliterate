#!/usr/bin/env python3
"""
Capture residual-stream activations from an mlx_lm model for a set of prompts.

For each prompt, monkey-patches each DecoderLayer's __call__ to record the
layer's output (the residual stream after the full attn+MLP block), runs the
model's native forward pass (which handles masks correctly), and records the
LAST-TOKEN activation at each layer.

Output: numpy .npz file with array of shape [num_prompts, num_layers, hidden_dim].

Usage:
  python capture_activations.py \
    --model ~/models/qwen3.5-122b-a10b-4bit \
    --prompts prompts_refusal.json \
    --out acts_refusal.npz

Designed for the qwen3_5 model family (hybrid GatedDeltaNet + Attention + MoE).
Works on the 4-bit quantized MLX weights. Run with vMLX stopped to free Metal
budget.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return prompt


def get_text_model(model):
    """Walk through wrappers to find the text model with .layers."""
    # Model -> language_model -> model
    tm = getattr(model, "language_model", model)
    tm = getattr(tm, "model", tm)
    return tm


def capture_for_prompt(model, tokenizer, prompt: str, DecoderLayer):
    """
    Run one forward pass through the model's own __call__, capturing last-token
    residual activation at each decoder layer via a class-level monkey-patch
    on DecoderLayer.__call__ (instance-level doesn't work because Python looks
    up __call__ on the type, not the instance).
    Returns [num_layers, hidden_dim] float32 numpy array.
    """
    tm = get_text_model(model)
    layers = tm.layers

    captured: list[np.ndarray] = []
    original_call = DecoderLayer.__call__

    def wrapped_call(self, x, mask=None, cache=None):
        out = original_call(self, x, mask=mask, cache=cache)
        # Cast to float32 on MLX side before numpy conversion (bfloat16 buffers
        # don't bridge directly through numpy's buffer protocol).
        last = out[0, -1, :].astype(mx.float32)
        mx.eval(last)
        captured.append(np.array(last))
        return out

    DecoderLayer.__call__ = wrapped_call
    try:
        text = apply_chat_template(tokenizer, prompt)
        tokens = mx.array([tokenizer.encode(text)])
        # Build a fresh cache using the model's own layout (ArraysCache + KVCache)
        try:
            cache = model.language_model.make_cache()
        except AttributeError:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(model)
        # Run full forward. mlx_lm handles masks internally (including the
        # None-mask case for GatedDeltaNet layers on fresh cache).
        _ = model(tokens, cache=cache)
    finally:
        DecoderLayer.__call__ = original_call

    assert len(captured) == len(layers), \
        f"Expected {len(layers)} layer captures, got {len(captured)}"
    return np.stack(captured, axis=0)  # [num_layers, hidden_dim]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-prompts", type=int, default=0)
    args = p.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    print(f"Loading model from {args.model}...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Import DecoderLayer class for the monkey-patch. Assumes qwen3_5 family.
    from mlx_lm.models.qwen3_5 import DecoderLayer

    all_acts = []
    for i, prompt in enumerate(prompts, 1):
        t0 = time.time()
        acts = capture_for_prompt(model, tokenizer, prompt, DecoderLayer)
        print(f"  [{i}/{len(prompts)}] {time.time() - t0:.1f}s  "
              f"layers={acts.shape[0]} dim={acts.shape[1]}  "
              f"{prompt[:60]!r}")
        all_acts.append(acts)

    stacked = np.stack(all_acts, axis=0)
    np.savez_compressed(args.out, activations=stacked,
                         prompts=np.array(prompts, dtype=object))
    print(f"Saved {stacked.shape} to {args.out}")


if __name__ == "__main__":
    main()
