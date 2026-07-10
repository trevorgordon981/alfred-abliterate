#!/usr/bin/env python3
"""
Capture residual-stream activations from an mlx_vlm model for a set of prompts.

Uses mlx_vlm's generate() with max_tokens=1 to handle all cache/position_ids
setup correctly, while hooking DecoderLayer.__call__ to capture activations.

Supports two prompt formats:
  - Flat strings (legacy): ["prompt1", "prompt2", ...]
  - Conversation lists (context-heavy): [[{"role":"user","content":"..."},...]...]

Output: numpy .npz file with array of shape [num_prompts, num_layers, hidden_dim].

Usage:
  # Legacy flat prompts
  python capture_activations_vlm.py \
    --model ~/models/qwen3.5-397b-a17b-4bit \
    --prompts prompts_refusal.json \
    --out acts_refusal_397b_4bit.npz

  # Context-heavy conversation prompts
  python capture_activations_vlm.py \
    --model ~/models/qwen3.5-397b-a17b-4bit \
    --prompts prompts_refusal_ctx.json \
    --out acts_refusal_ctx_397b_4bit.npz
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def get_text_model(model):
    tm = getattr(model, "language_model", model)
    tm = getattr(tm, "model", tm)
    return tm


def conversation_to_prompt(processor, conversation: list[dict]) -> str:
    """Apply the model's chat template to a conversation list.

    Args:
        processor: The mlx_vlm processor (has a tokenizer with chat template)
        conversation: List of {"role": "...", "content": "..."} dicts

    Returns:
        Formatted prompt string ready for the model
    """
    tokenizer = getattr(processor, "tokenizer", processor)

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: manual Qwen/ChatML format
    parts = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def detect_prompt_format(prompts) -> str:
    """Detect whether prompts are flat strings or conversation lists.

    Returns: 'flat' or 'conversation'
    """
    if not prompts:
        return "flat"
    first = prompts[0]
    if isinstance(first, str):
        return "flat"
    if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
        return "conversation"
    raise ValueError(
        f"Unrecognized prompt format. Expected list of strings or list of "
        f"conversation lists, got list of {type(first).__name__}"
    )


def capture_for_prompt(model, processor, prompt, DecoderLayer):
    """
    Use mlx_vlm's generate with max_tokens=1 to run the prefill, capturing
    last-token residual activation at each decoder layer.

    Args:
        prompt: Either a string (flat) or a list of message dicts (conversation)

    Returns [num_layers, hidden_dim] float32 numpy array.
    """
    from mlx_vlm.generate import generate

    # Convert conversation to string if needed
    if isinstance(prompt, list):
        prompt_str = conversation_to_prompt(processor, prompt)
    else:
        prompt_str = prompt

    captured: list[np.ndarray] = []
    original_call = DecoderLayer.__call__

    def wrapped_call(self, *args, **kwargs):
        out = original_call(self, *args, **kwargs)
        last = out[0, -1, :].astype(mx.float32)
        mx.eval(last)
        captured.append(np.array(last))
        return out

    DecoderLayer.__call__ = wrapped_call
    try:
        # generate with max_tokens=1 does the full prefill + 1 decode step
        # This handles cache, position_ids, masks correctly
        generate(model, processor, prompt_str, max_tokens=1, temperature=0.0)
    finally:
        DecoderLayer.__call__ = original_call

    tm = get_text_model(model)
    n_layers = len(tm.layers)

    # The generate call runs layers multiple times (prefill chunks + 1 decode).
    # We want the LAST full pass through all layers (the decode step),
    # which captures the accumulated residual stream state.
    # Take the last n_layers captures.
    if len(captured) < n_layers:
        raise RuntimeError(
            f"Expected at least {n_layers} captures, got {len(captured)}. "
            f"Model may have errored during generation."
        )

    last_pass = captured[-n_layers:]
    return np.stack(last_pass, axis=0)


def get_prompt_preview(prompt, max_len=60) -> str:
    """Get a short preview of a prompt for logging."""
    if isinstance(prompt, str):
        return repr(prompt[:max_len])
    if isinstance(prompt, list):
        # For conversations, show the last user message
        for msg in reversed(prompt):
            if msg.get("role") == "user":
                return f"[{len(prompt)} msgs] {repr(msg['content'][:max_len])}"
        return f"[{len(prompt)} msgs]"
    return repr(str(prompt)[:max_len])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-prompts", type=int, default=0)
    args = p.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    fmt = detect_prompt_format(prompts)
    print(f"Prompt format: {fmt} ({len(prompts)} prompts)")

    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    # For conversation format, estimate token count of first prompt
    if fmt == "conversation" and prompts:
        first_chars = sum(len(m.get("content", "")) for m in prompts[0])
        est_tokens = first_chars // 3
        print(f"Estimated tokens per prompt: ~{est_tokens:,} "
              f"({len(prompts[0])} messages, {first_chars:,} chars)")

    print(f"Loading model from {args.model}...")
    t0 = time.time()
    from mlx_vlm.utils import load
    model, processor = load(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    tm = get_text_model(model)
    DecoderLayer = type(tm.layers[0])
    print(f"DecoderLayer: {DecoderLayer.__module__}.{DecoderLayer.__name__}")
    print(f"Layers: {len(tm.layers)}, Hidden dim: {tm.layers[0].input_layernorm.weight.shape[0]}")

    # For conversation prompts, pre-render the shared prefix once to verify
    # the chat template works
    if fmt == "conversation" and prompts:
        test_prompt = prompts[0]
        if isinstance(test_prompt, list):
            test_rendered = conversation_to_prompt(processor, test_prompt)
            print(f"Chat template test: {len(test_rendered)} chars rendered OK")
            print(f"  Starts with: {repr(test_rendered[:100])}")
            print(f"  Ends with:   {repr(test_rendered[-100:])}")

    all_acts = []
    # Collect flat prompt strings for metadata (last user message for conversations)
    prompt_labels = []
    for i, prompt in enumerate(prompts, 1):
        t0 = time.time()
        acts = capture_for_prompt(model, processor, prompt, DecoderLayer)
        elapsed = time.time() - t0
        preview = get_prompt_preview(prompt)
        print(f"  [{i}/{len(prompts)}] {elapsed:.1f}s  "
              f"layers={acts.shape[0]} dim={acts.shape[1]}  "
              f"{preview}")
        all_acts.append(acts)

        # Extract label for metadata
        if isinstance(prompt, str):
            prompt_labels.append(prompt)
        elif isinstance(prompt, list):
            # Use last user message as label
            for msg in reversed(prompt):
                if msg.get("role") == "user":
                    prompt_labels.append(msg["content"])
                    break
            else:
                prompt_labels.append(f"conversation_{i}")

        mx.clear_cache()

    stacked = np.stack(all_acts, axis=0)
    np.savez_compressed(args.out, activations=stacked,
                        prompts=np.array(prompt_labels, dtype=object))
    print(f"Saved {stacked.shape} to {args.out}")


if __name__ == "__main__":
    main()
