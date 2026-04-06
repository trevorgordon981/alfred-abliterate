#!/usr/bin/env python3
"""
Launch mlx_lm's OpenAI-compatible HTTP server with refusal-direction
abliteration applied to the loaded model.

Wraps `mlx_lm.server` by monkey-patching ModelProvider.load to call
attach_projection() after the underlying model is loaded. All other server
features (streaming, concurrency, prompt cache) stay intact.

Usage:
  python abliterated_server.py \
    --model ~/models/qwen3.5-122b-a10b-4bit \
    --directions refusal_direction.npz,safety_direction.npz \
    --top-k 20 \
    --host 127.0.0.1 --port 8082

Any additional flags are passed through to mlx_lm.server unchanged. Env vars:
  ABLITERATE_DIRECTIONS  (comma-separated .npz paths; same as --directions)
  ABLITERATE_TOP_K       (same as --top-k; default 20)

Intended to run under launchd as a drop-in replacement for the vMLX server
on port 8082 for this model. vMLX is a different engine — this server uses
mlx_lm instead, so some features (vMLX prefix cache, tool-call parsing) are
not available. Those were handled by blockops-proxy anyway.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure this dir is importable so project_inference is found under launchd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mlx_lm.server as mlx_server

from project_inference import attach_projection


def main():
    # Strip our wrapper-specific args before handing off to mlx_lm.server
    # (mlx_lm's parser doesn't know about --directions / --top-k).
    ablit_parser = argparse.ArgumentParser(add_help=False)
    ablit_parser.add_argument("--directions",
        default=os.environ.get("ABLITERATE_DIRECTIONS", ""))
    ablit_parser.add_argument("--top-k", type=int,
        default=int(os.environ.get("ABLITERATE_TOP_K", "20")))
    ablit_args, passthrough = ablit_parser.parse_known_args()

    direction_files = [f.strip() for f in ablit_args.directions.split(",") if f.strip()]
    if not direction_files:
        raise SystemExit(
            "ERROR: --directions (or ABLITERATE_DIRECTIONS env var) is required.\n"
            "Provide comma-separated .npz paths, e.g. --directions refusal.npz,safety.npz"
        )

    # Load direction arrays
    direction_arrays = [np.load(f)["directions"] for f in direction_files]
    magnitudes = sum(np.load(f)["magnitudes"] for f in direction_files)
    order = np.argsort(-magnitudes)
    target_layers = set(order[: ablit_args.top_k].tolist())
    primary = direction_arrays[0]
    extras = direction_arrays[1:] if len(direction_arrays) > 1 else None

    print(f"[abliterated_server] Loaded {len(direction_files)} direction file(s): {direction_files}", flush=True)
    print(f"[abliterated_server] Top-{ablit_args.top_k} combined-magnitude layers: {sorted(target_layers)}", flush=True)

    # Monkey-patch ModelProvider.load to attach projection after model loads.
    # Idempotency: only attach once per ModelProvider instance, even if load()
    # is called repeatedly (cache-hit path returns the same model object).
    original_load = mlx_server.ModelProvider.load
    _attached_to = set()  # id(model) values we've already patched

    def patched_load(self, model_path, adapter_path=None, draft_model_path=None):
        model, tokenizer = original_load(self, model_path, adapter_path, draft_model_path)
        if id(model) not in _attached_to:
            print(f"[abliterated_server] Attaching projection hooks...", flush=True)
            attach_projection(model, primary, target_layers, extra_directions=extras)
            _attached_to.add(id(model))
        return model, tokenizer

    mlx_server.ModelProvider.load = patched_load

    # Hand off remaining argv to mlx_lm.server's main()
    sys.argv = [sys.argv[0]] + passthrough
    mlx_server.main()


if __name__ == "__main__":
    main()
