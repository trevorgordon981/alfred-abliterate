#!/usr/bin/env python3
"""
Runtime abliteration hook for vLLM on NVIDIA GPUs.

PyTorch equivalent of the MLX inference-time projection hook. Patches
DecoderLayer.__call__ (or .forward) to project out refusal directions
from the residual stream at targeted layers.

Usage:
    # As a standalone wrapper around vllm serve:
    python vllm_abliterate_hook.py \
        --model /path/to/model \
        --directions refusal_direction.npz,safety_direction.npz \
        --top-k 16 \
        --port 8082 \
        -- [extra vllm args]

    # Or import and call attach_projection() after model load.

The direction .npz files are architecture-independent (just numpy arrays
of shape [num_layers, hidden_dim]). Generate them on any platform using
capture_activations.py + compute_direction.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def gram_schmidt(vecs: list[np.ndarray]) -> list[np.ndarray]:
    """Orthonormalize a list of vectors."""
    out = []
    for v in vecs:
        u = v.astype(np.float32).copy()
        for w in out:
            u = u - np.dot(u, w) * w
        n = np.linalg.norm(u)
        if n < 1e-6:
            continue
        out.append(u / n)
    return out


def attach_projection(model, directions: np.ndarray, target_layers: set[int],
                      extra_directions: list[np.ndarray] | None = None) -> None:
    """
    Monkey-patch decoder layers to project out refusal directions from
    the residual stream. Works with any HuggingFace-style model where
    the decoder layers are in model.model.layers (or model.layers).

    Each targeted layer's forward output gets the refusal direction(s)
    projected out before returning.
    """
    # Find the decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Cannot find decoder layers in model")

    # Get the DecoderLayer class
    DecoderLayer = type(layers[0])

    # Build per-layer orthonormalized direction sets
    all_direction_sets = [directions]
    if extra_directions:
        all_direction_sets.extend(extra_directions)

    # Determine device from model
    device = next(model.parameters()).device

    for i in target_layers:
        if i >= len(layers):
            continue
        per_layer_vecs = [ds[i] for ds in all_direction_sets]
        orth = gram_schmidt(per_layer_vecs)
        # Store as instance attribute on the layer
        layers[i]._abliterate_dirs = [
            torch.tensor(v, dtype=torch.float32, device=device) for v in orth
        ]

    # Guard against double-patching
    existing = DecoderLayer.forward
    if hasattr(existing, '_abliterate_original'):
        original_forward = existing._abliterate_original
    else:
        original_forward = existing

    def patched_forward(self, *args, **kwargs):
        out = original_forward(self, *args, **kwargs)
        dirs = getattr(self, '_abliterate_dirs', None)
        if dirs is not None:
            # out can be a tuple (hidden_states, ...) or just hidden_states
            if isinstance(out, tuple):
                hidden = out[0]
                for d in dirs:
                    d_cast = d.to(hidden.dtype)
                    proj_coef = torch.sum(hidden * d_cast, dim=-1, keepdim=True)
                    hidden = hidden - proj_coef * d_cast
                out = (hidden,) + out[1:]
            else:
                for d in dirs:
                    d_cast = d.to(out.dtype)
                    proj_coef = torch.sum(out * d_cast, dim=-1, keepdim=True)
                    out = out - proj_coef * d_cast
        return out

    DecoderLayer.forward = patched_forward
    patched_forward._abliterate_original = original_forward

    count = sum(1 for l in layers if hasattr(l, '_abliterate_dirs'))
    n_dirs = 1 + (len(extra_directions) if extra_directions else 0)
    print(f"ABLITERATE: attached projection to {count} layers, "
          f"{n_dirs} direction(s) each", flush=True)


def load_and_attach(model, config_path: str | None = None,
                    direction_files: list[str] | None = None,
                    top_k: int = 16) -> None:
    """
    Load directions from a config file or explicit paths and attach hooks.

    Config file format (JSON):
        {"directions": ["path1.npz", "path2.npz"], "top_k": 16}
    """
    if config_path is None and direction_files is None:
        # Try default config location
        default = Path.home() / ".hermes" / "abliterate.json"
        if default.exists():
            config_path = str(default)
        else:
            return  # No config, nothing to do

    if config_path:
        config = json.loads(Path(config_path).read_text())
        direction_files = config.get("directions", [])
        top_k = config.get("top_k", top_k)

    if not direction_files:
        return

    # Load direction arrays (once each)
    loaded = [np.load(f) for f in direction_files]
    direction_arrays = [d["directions"] for d in loaded]
    magnitudes = sum(d["magnitudes"] for d in loaded)

    # Validate matching shapes
    if not all(d.shape == direction_arrays[0].shape for d in direction_arrays):
        print("ABLITERATE: skipping, direction files have mismatched shapes")
        return

    # Check layer count matches model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n_model_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        n_model_layers = len(model.layers)
    else:
        print("ABLITERATE: skipping, cannot find decoder layers")
        return

    n_dir_layers = direction_arrays[0].shape[0]
    if n_model_layers != n_dir_layers:
        print(f"ABLITERATE: skipping, layer count mismatch "
              f"(model={n_model_layers}, directions={n_dir_layers})")
        return

    order = np.argsort(-magnitudes)
    target_layers = set(order[:top_k].tolist())

    primary = direction_arrays[0]
    extras = direction_arrays[1:] if len(direction_arrays) > 1 else None

    attach_projection(model, primary, target_layers, extra_directions=extras)


def main():
    """Wrapper: start vllm serve with abliteration hooks injected."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", default="",
                        help="Comma-separated .npz direction files")
    parser.add_argument("--config", default=None,
                        help="Path to abliterate.json config")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", default="0.0.0.0")
    args, extra = parser.parse_known_args()

    direction_files = [f.strip() for f in args.directions.split(",") if f.strip()]

    # Monkey-patch vllm's model loader to attach hooks after load
    from vllm import LLM

    original_init = LLM.__init__

    def patched_init(self, *a, **kw):
        original_init(self, *a, **kw)
        # Access the underlying model
        model = self.llm_engine.model_executor.driver_worker.model_runner.model
        load_and_attach(
            model,
            config_path=args.config,
            direction_files=direction_files or None,
            top_k=args.top_k,
        )

    LLM.__init__ = patched_init

    # Build vllm serve command
    serve_args = [
        "vllm", "serve", args.model,
        "--port", str(args.port),
        "--host", args.host,
    ] + extra

    sys.argv = serve_args
    from vllm.entrypoints.openai.api_server import run_server
    run_server(serve_args[1:])


if __name__ == "__main__":
    main()
