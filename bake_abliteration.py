#!/usr/bin/env python3
"""
Bake refusal-direction abliteration into model weights, producing a new
MLX-quantized model directory that can be served directly (no inference-time
hook required).

Approach (FailSpy weight-orthogonalization):
  - Pick a single "global" direction per input direction file (from the layer
    with highest magnitude).
  - Gram-Schmidt orthonormalize all directions into an orthonormal basis.
  - Orthogonalize (project out) that basis from:
      * embed_tokens output direction (row-wise: each token embedding)
      * each layer's attention output projection (o_proj / out_proj)
      * each layer's MLP output projection (down_proj, shared_expert.down_proj)
  - This removes the refusal subspace from every contribution to the
    residual stream, in every layer, at weight-load time.

The resulting model loads and serves exactly like the original (same
architecture, same config, same quantization format), just with the refusal
directions zeroed out of the weights that write to the residual stream.

Usage:
  python bake_abliteration.py \
    --model ~/models/qwen3.5-397b-a17b-4bit \
    --directions refusal_direction_397b.npz,safety_direction_397b.npz \
    --output ~/models/qwen3.5-397b-a17b-4bit-abliterated
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load


def gram_schmidt(vecs):
    """Orthonormalize a list of numpy vectors; drop near-zero results."""
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


def orthogonalize_weight_mx(W: mx.array, directions: list[mx.array]) -> mx.array:
    """
    W shape: (out_features, in_features) or (experts, out_features, in_features).
    Project out each unit-direction d (length out_features) from the OUTPUT axis.
    For each d: W_new = W - d outer (d^T @ W)
    """
    result = W
    for d in directions:
        # d: (out_features,)
        if result.ndim == 2:
            # (out, in): project along dim 0
            dW = d[None, :] @ result   # hmm, wrong axis
            # Actually: d^T @ W requires d to match W's first axis
            # d: (out_features,) — matches dim 0 of W (out, in)
            # d @ W would need W to be (out, in) and d to be (out,), gives (in,)
            coeffs = d[None, :] @ result   # shape (1, in)
            result = result - d[:, None] @ coeffs  # (out, in)
        elif result.ndim == 3:
            # (experts, out, in): same projection per expert
            # d: (out,). Need coeffs shape (experts, 1, in)
            coeffs = (d[None, None, :] @ result)  # (1, 1, in) broadcast to (e, 1, in) after matmul? Actually mx matmul handles batching
            # d[None, :] @ result: shape broadcasting: d[None,:] is (1, out). result is (experts, out, in).
            # matmul treats last 2 dims: (1, out) @ (experts, out, in) → (experts, 1, in)
            # Then d[:, None] @ coeffs: (out, 1) @ (experts, 1, in) → (experts, out, in)
            d2 = d[None, :]             # (1, out)
            coeffs = d2 @ result        # (experts, 1, in)
            d_col = d[:, None]          # (out, 1)
            result = result - (d_col @ coeffs)  # (experts, out, in)
        else:
            raise ValueError(f"Unsupported weight ndim: {result.ndim}")
    return result


def dequantize_layer_weight(layer):
    """Given a QuantizedLinear or QuantizedSwitchLinear, return dequantized
    fp32 weight tensor plus the quant params needed to requantize."""
    w = layer.weight
    scales = layer.scales
    biases = getattr(layer, "biases", None)
    group_size = layer.group_size
    bits = layer.bits
    deq = mx.dequantize(w, scales=scales, biases=biases,
                         group_size=group_size, bits=bits)
    return deq, group_size, bits


def requantize_layer(layer, new_deq_weight: mx.array) -> None:
    """Quantize new fp32 weight back into the layer's quantization format."""
    group_size = layer.group_size
    bits = layer.bits
    w_q, scales, biases = mx.quantize(new_deq_weight,
                                       group_size=group_size, bits=bits)
    layer.weight = w_q
    layer.scales = scales
    if biases is not None:
        layer.biases = biases


def orthogonalize_quantized_layer(layer, directions: list[mx.array]) -> None:
    """Dequantize → orthogonalize against all directions → requantize in place."""
    deq, _, _ = dequantize_layer_weight(layer)
    modified = orthogonalize_weight_mx(deq, directions)
    requantize_layer(layer, modified)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--directions", required=True,
                   help="Comma-separated .npz files from compute_direction.py")
    p.add_argument("--output", required=True, help="Output model directory")
    p.add_argument("--source-layer", type=int, default=-1,
                   help="Layer index to pull the global direction from "
                        "(default: pick from middle of top-K in directions file). "
                        "Late layers (e.g. final) encode response-formation signal "
                        "and will break the model if orthogonalized.")
    p.add_argument("--target-layers", default="",
                   help="Comma-separated layer indices to orthogonalize "
                        "(default: apply to all). Use 'top-K' range from sweep results.")
    p.add_argument("--skip-embed", action="store_true",
                   help="Skip embed_tokens orthogonalization (safer)")
    p.add_argument("--per-layer", action="store_true",
                   help="Use per-layer direction from each directions file "
                        "(layer i's output projections get direction[i]). "
                        "Matches the inference-time hook's behavior.")
    args = p.parse_args()

    direction_files = [f.strip() for f in args.directions.split(",") if f.strip()]
    direction_arrays = [np.load(f) for f in direction_files]

    # Pick the direction to use as the global basis vector per file.
    # Late layers encode response-formation signal; using layer N-1 breaks the
    # model. Default: use the 8th-best layer (middle of top-16) — strong enough
    # for refusal signal, late enough to not dominate generation.
    global_vecs = []
    for i, d in enumerate(direction_arrays):
        if args.source_layer >= 0:
            src_layer = args.source_layer
        else:
            # 8th-best layer by magnitude (middle of the effective top-16)
            src_layer = int(d["best_layers"][7])
        vec = d["directions"][src_layer]
        global_vecs.append(vec)
        print(f"Direction {i} ({direction_files[i]}): using layer {src_layer}, "
              f"magnitude {d['magnitudes'][src_layer]:.2f}")

    # Orthonormalize the global basis
    orth_basis_np = gram_schmidt(global_vecs)
    print(f"Orthonormal basis size: {len(orth_basis_np)}")

    # Convert to mx arrays (we'll cast per-layer as needed)
    orth_basis_mx = [mx.array(v, dtype=mx.float32) for v in orth_basis_np]

    # Load model
    print(f"\nLoading model from {args.model}...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.time() - t0:.1f}s")

    tm = model.language_model.model
    layers = tm.layers
    print(f"Model has {len(layers)} decoder layers")

    # Determine target layers (which decoder layers to orthogonalize)
    if args.target_layers:
        target_layer_set = set(int(x) for x in args.target_layers.split(",") if x.strip())
    else:
        # Default: top-16 strongest layers (combined-magnitude across direction files)
        combined_mag = sum(d["magnitudes"] for d in direction_arrays)
        order = np.argsort(-combined_mag)
        target_layer_set = set(int(i) for i in order[:16])
    print(f"Target layers ({len(target_layer_set)}): {sorted(target_layer_set)}")

    # Optional: orthogonalize embed_tokens. Aggressive (every token embedding
    # loses its projection onto the direction), but useful if the refusal
    # manifold leaks in through the initial embedding.
    if not args.skip_embed:
        print("\nOrthogonalizing embed_tokens...")
        embed = tm.embed_tokens
        if hasattr(embed, "scales"):
            deq, _, _ = dequantize_layer_weight(embed)
        else:
            deq = embed.weight.astype(mx.float32)
        for d in orth_basis_mx:
            coeffs = deq @ d
            deq = deq - coeffs[:, None] @ d[None, :]
        if hasattr(embed, "scales"):
            requantize_layer(embed, deq)
        else:
            embed.weight = deq.astype(embed.weight.dtype)
    else:
        print("\nSkipping embed_tokens orthogonalization")

    # Orthogonalize per-layer output projections (only targeted layers)
    t0 = time.time()
    for i, layer in enumerate(layers):
        if i not in target_layer_set:
            continue
        # Choose directions for this layer: either the global orth basis, or
        # per-layer directions (one from each direction file), orthonormalized.
        if args.per_layer:
            layer_vecs = [d["directions"][i] for d in direction_arrays]
            layer_orth_np = gram_schmidt(layer_vecs)
            layer_dirs = [mx.array(v, dtype=mx.float32) for v in layer_orth_np]
        else:
            layer_dirs = orth_basis_mx
        # Attention output projection (self_attn.o_proj OR linear_attn.out_proj)
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            orthogonalize_quantized_layer(layer.self_attn.o_proj, layer_dirs)
        if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
            orthogonalize_quantized_layer(layer.linear_attn.out_proj, layer_dirs)
        # MLP: switch_mlp.down_proj + shared_expert.down_proj
        if hasattr(layer, "mlp"):
            if hasattr(layer.mlp, "switch_mlp") and hasattr(layer.mlp.switch_mlp, "down_proj"):
                orthogonalize_quantized_layer(layer.mlp.switch_mlp.down_proj, layer_dirs)
            if hasattr(layer.mlp, "shared_expert") and hasattr(layer.mlp.shared_expert, "down_proj"):
                orthogonalize_quantized_layer(layer.mlp.shared_expert.down_proj, layer_dirs)
            # Fallback: plain MLP (non-MoE layers)
            if hasattr(layer.mlp, "down_proj") and not hasattr(layer.mlp, "switch_mlp"):
                orthogonalize_quantized_layer(layer.mlp.down_proj, layer_dirs)
        if (i + 1) % 10 == 0:
            print(f"  orthogonalized {i+1}/{len(layers)} layers in {time.time()-t0:.1f}s")
    print(f"All layers done in {time.time() - t0:.1f}s")

    # Save the modified model
    out_path = Path(args.output).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving modified weights to {out_path}...")
    t0 = time.time()
    # Save with {"format": "mlx"} metadata so mlx_vlm recognizes this as
    # already-sanitized and skips the HF→MLX weight renaming. Without this
    # metadata, mlx_vlm's sanitize() tries to pop HF-format keys and crashes.
    from mlx.utils import tree_flatten
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out_path / "model.safetensors"), weights,
                         metadata={"format": "mlx"})
    print(f"Weights saved in {time.time() - t0:.1f}s")

    # Copy config, tokenizer, chat template from source
    src = Path(args.model).expanduser()
    copied = []
    for name in ["config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja",
                 "special_tokens_map.json", "merges.txt", "vocab.json"]:
        s = src / name
        if s.exists():
            shutil.copy(s, out_path / name)
            copied.append(name)
    print(f"Copied metadata files: {copied}")

    print(f"\nDone. Abliterated model at: {out_path}")
    print(f"Test with: python -c \"from mlx_lm import load, generate; "
          f"m,t=load('{out_path}'); print(generate(m,t,'What happened at Tiananmen in 1989? Answer briefly.', max_tokens=200))\"")


if __name__ == "__main__":
    main()
