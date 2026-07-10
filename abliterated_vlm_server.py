#!/usr/bin/env python3
"""
Drop-in replacement for `python -m mlx_vlm.server` with abliteration hooks.
Patches Qwen3_5MoeModel.__call__ to project out refusal directions from the
residual stream during the layer loop - this is more robust than patching
individual DecoderLayer.__call__ methods.
"""

import json
import sys
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np
import mlx_vlm.server as vlm_server


def apply_abliteration(model):
    """Apply abliteration by patching the Model's layer loop directly."""
    config_path = Path.home() / ".hermes" / "abliterate.json"
    if not config_path.exists():
        print("ABLITERATE: no config, skipping", flush=True)
        return

    try:
        config = json.loads(config_path.read_text())
        direction_files = config.get("directions", [])
        top_k = config.get("top_k", 16)
        if not direction_files:
            return

        # Load all direction files
        all_dirs = {}  # layer_idx -> list of mx direction arrays
        for fpath in direction_files:
            npz = np.load(fpath)
            dirs = npz["directions"]
            mags = npz["magnitudes"]
            order = np.argsort(-mags)
            for i in order[:top_k]:
                v = dirs[i].astype(np.float32)
                n = np.linalg.norm(v)
                if n > 1e-6:
                    all_dirs.setdefault(int(i), []).append(
                        mx.array(v / n, dtype=mx.float32)
                    )

        # Find the Qwen3_5MoeModel (or Qwen3_5Model) which has the layer loop
        tm = getattr(model, "language_model", model)
        inner = getattr(tm, "model", tm)
        ModelClass = type(inner)

        original_model_call = ModelClass.__call__

        _call_count = [0]
        def patched_model_call(self, inputs=None, inputs_embeds=None,
                               mask=None, cache=None, position_ids=None,
                               **kwargs):
            _call_count[0] += 1
            if _call_count[0] <= 3:
                print(f"ABLITERATE_CALL: type(self)={type(self).__name__} "
                      f"id={id(self)} inner_id={id(inner)}", flush=True)
            # Replicate the original __call__ but with projection after each layer
            if inputs_embeds is None:
                h = self.embed_tokens(inputs)
            else:
                h = inputs_embeds

            if cache is None:
                cache = [None] * len(self.layers)

            fa_mask = None
            ssm_mask = None
            # Build masks lazily
            try:
                from mlx_vlm.models.qwen3_5.language import (
                    create_attention_mask, create_ssm_mask
                )
                fa_mask = create_attention_mask(h, cache[self.fa_idx])
                ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
            except Exception:
                pass

            for i, (layer, c) in enumerate(zip(self.layers, cache)):
                if fa_mask is not None and ssm_mask is not None:
                    m = ssm_mask if layer.is_linear else fa_mask
                else:
                    m = mask
                h = layer(h, m, c, position_ids)

                # Project out refusal directions for targeted layers
                dirs = all_dirs.get(i)
                if dirs is not None:
                    for d in dirs:
                        dc = d.astype(h.dtype)
                        h = h - mx.sum(h * dc, axis=-1, keepdims=True) * dc
                    mx.eval(h)

            return self.norm(h)

        ModelClass.__call__ = patched_model_call

        count = len(all_dirs)
        total_dirs = sum(len(v) for v in all_dirs.values())
        print(f"ABLITERATE: patched model layer loop, {count} layers, "
              f"{total_dirs} total projections: "
              f"{[Path(f).name for f in direction_files]}", flush=True)

    except Exception as e:
        print(f"ABLITERATE: failed: {e}", flush=True)
        traceback.print_exc()


# Patch load_model_resources
_orig_load = vlm_server.load_model_resources

def patched_load(model_path, adapter_path=None):
    result = _orig_load(model_path, adapter_path)
    apply_abliteration(result[0])
    return result

vlm_server.load_model_resources = patched_load

vlm_server.main()
