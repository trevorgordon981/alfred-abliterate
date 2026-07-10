#!/usr/bin/env python3
"""
Bake refusal-direction abliteration into MiniMax-M3 (minimax_m3_vl) weights,
producing a new MLX-quantized model directory servable exactly like the
input by mlx_vlm (m3_serve_batched.py / `from mlx_vlm import load`).

This is an M3-VLM port of ~/alfred-abliterate-public/bake_abliteration.py
(built for Qwen3.5-397B / mlx_lm text-only models). Differences from that
script, and WHY:

  1. Load via `mlx_vlm.utils.load()` (returns `(model, processor)`), not
     `mlx_lm.load()` — M3 here is a minimax_m3_vl VLM. The text decoder
     tree lives at `model.language_model.model` (MiniMaxM3Model), not at
     `model` directly.

  2. MoE shared-expert attribute is PLURAL: `layer.mlp.shared_experts`
     (M3MoE, see ~/dev/mlx-vlm/mlx_vlm/models/minimax_m3/language.py).
     The Qwen-397B version of this script looked for the singular
     `shared_expert` — that check silently no-ops on M3 and would leave
     the shared-expert contribution to the residual stream un-ablated.
     Confirmed against the model source AND against the *live* prod
     quantization config (~/models/MiniMax-M3-v3-6bit-mixed/config.json
     has keys like `language_model.model.layers.54.mlp.shared_experts.down_proj`).

  3. Dense vs MoE layers: M3 has `first_k_dense` leading dense layers
     (3 of 60 in the current config, via `text_config.moe_layer_freq`)
     whose `layer.mlp` is a plain `M3MLP` (has `.down_proj` directly, NO
     `.switch_mlp`). Layers >= first_k_dense have `layer.mlp` as `M3MoE`
     (has BOTH `.switch_mlp.down_proj` [3D, experts x out x in] AND
     `.shared_experts.down_proj` [2D]). Detection mirrors the original
     script's existing fallback: `hasattr(mlp, 'down_proj') and not
     hasattr(mlp, 'switch_mlp')` for the dense case — this already works
     unchanged for M3, no fix needed there.

  4. Attention output projection is `self_attn.o_proj` (2D QuantizedLinear)
     — same attribute name as the original script already checks. M3 has
     no `linear_attn` variant; that branch is dead code here but kept for
     parity/future-proofing (harmless no-op via hasattr).

  5. SAVE PATH — the hard part. The public/397B script did a single
     `mx.save_safetensors(out/"model.safetensors", weights)`. That does
     NOT work for M3: the 6-bit M3 VLM is ~327GB across ~86 safetensors
     shards with a `model.safetensors.index.json`, plus vision tower /
     multimodal projector tensors, tokenizer/processor/config files, and
     a couple of custom `trust_remote_code` .py files. A single-file save
     would (a) blow past any sane single-shard size, (b) omit the index
     mlx_vlm's loader expects, (c) drop every non-weight file needed to
     load at all.

     Instead this script mirrors `mlx_vlm.convert.convert()`'s own save
     path exactly (~/dev/mlx-vlm/mlx_vlm/convert.py, and the prod-recipe
     script ~/pipeline-automation/m3_requant_6bit.py which calls the same
     `mlx_vlm.convert.convert()` to build the model this script's input
     descends from):

       - `mlx_vlm.utils.save_weights(out_path, model, donate_weights=True)`
         — flattens `model.parameters()` (i.e. the WHOLE VLM tree: text
         decoder + vision_tower + multi_modal_projector + patch_merge_mlp
         + lm_head + embed_tokens, since `model` here is the full VLM
         object returned by `mlx_vlm.utils.load()`), auto-shards via
         `make_shards()` (5GB/shard default), writes
         `model-XXXXX-of-YYYYY.safetensors` + `model.safetensors.index.json`
         with `metadata={"format": "mlx"}` per shard. This is the EXACT
         function `mlx_vlm.convert.convert()` calls to produce
         MiniMax-M3-v3-6bit-mixed in the first place, so the output is
         byte-layout-compatible with what m3_serve_batched.py loads.
       - Then copy `*.py` / `*.json` / `*.jinja` files from the source dir
         (config.json, generation_config.json, tokenizer.json,
         tokenizer_config.json, special_tokens_map.json, added_tokens.json,
         vocab.json, preprocessor_config.json, processor_config.json,
         chat_template.jinja, configuration_minimax_m3_vl.py,
         image_processor.py, processing_minimax.py, video_processor.py),
         SKIPPING model.safetensors.index.json since save_weights() just
         regenerated the correct one for OUR shard layout (source has 86
         shards; if any target layer's dequant/requant changes a tensor's
         byte size — it shouldn't, same group_size/bits round-trip — shard
         boundaries could differ slightly; using our own fresh index avoids
         a stale mapping either way). Any subdirectories under the source
         are copied as-is (there were none in the prod 6-bit dir at
         inspection time, but convert.py does this and so do we, for
         parity in case an assets/ dir shows up).
       - config.json is copied VERBATIM (not regenerated via save_config)
         because we do not change the quantization scheme — every ablated
         layer is dequantized and requantized with its OWN existing
         `group_size`/`bits` (read off the layer itself), so the
         `quantization` block in config.json stays valid unchanged.

     THIS SAVE PATH IS UNVERIFIED BY EXECUTION (GPU busy, per instructions —
     static/structural inspection only). Confirmed via source reading:
     `save_weights` signature at ~/dev/mlx-vlm/mlx_vlm/utils.py:1112,
     `make_shards` at :981, and `mlx_vlm.convert.convert()`'s copy logic
     (~/dev/mlx-vlm/mlx_vlm/convert.py) which this script's copy step
     replicates line-for-line minus the `*.jinja` glob (convert.py relies
     on `processor.save_pretrained()` to rewrite the chat template; this
     script does NOT call `processor.save_pretrained()` since we never
     touch processor state, so `*.jinja` is copied explicitly instead).
     See the runbook / final report for the specific uncertainty flags.

Usage:
  python bake_abliteration_m3.py \
    --model ~/models/MiniMax-M3-6bit-v1 \
    --directions cn_refusal_direction_m3v1.npz,safety_direction_m3v1.npz \
    --output ~/models/MiniMax-M3-6bit-v1-abliterated

  # Print the plan without loading anything (no model, no GPU):
  python bake_abliteration_m3.py \
    --model ~/models/MiniMax-M3-6bit-v1 \
    --directions cn_refusal_direction_m3v1.npz,safety_direction_m3v1.npz \
    --output ~/models/MiniMax-M3-6bit-v1-abliterated \
    --dry-structural
"""

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


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


def orthogonalize_weight_mx(W, directions):
    """
    W shape: (out_features, in_features) or (experts, out_features, in_features).
    Project out each unit-direction d (length out_features) from the OUTPUT axis.
    For each d: W_new = W - d outer (d^T @ W)
    """
    import mlx.core as mx  # noqa: F401  (imported lazily; not needed in --dry-structural)

    result = W
    for d in directions:
        # d: (out_features,)
        if result.ndim == 2:
            # (out, in): project along dim 0
            coeffs = d[None, :] @ result   # shape (1, in)
            result = result - d[:, None] @ coeffs  # (out, in)
        elif result.ndim == 3:
            # (experts, out, in): same projection per expert.
            # d[None,:] is (1, out). result is (experts, out, in).
            # matmul treats last 2 dims: (1, out) @ (experts, out, in) -> (experts, 1, in)
            # then (out, 1) @ (experts, 1, in) -> (experts, out, in)
            d2 = d[None, :]              # (1, out)
            coeffs = d2 @ result         # (experts, 1, in)
            d_col = d[:, None]           # (out, 1)
            result = result - (d_col @ coeffs)  # (experts, out, in)
        else:
            raise ValueError(f"Unsupported weight ndim: {result.ndim}")
    return result


def dequantize_layer_weight(layer):
    """Given a QuantizedLinear/QuantizedSwitchLinear, return dequantized fp32
    weight plus the quant params needed to requantize.

    SCALES GUARD: if the module is NOT quantized (bf16 target, e.g. a future
    build that leaves some projection unquantized), fall through to a plain
    fp32 view with bits=None so requantize_layer stores it back verbatim as
    bf16 instead of AttributeError-crashing mid-bake after writing GBs."""
    import mlx.core as mx

    if not hasattr(layer, "scales"):
        # Unquantized module — no dequant needed; signal fp path via bits=None.
        return layer.weight.astype(mx.float32), None, None

    w = layer.weight
    scales = layer.scales
    biases = getattr(layer, "biases", None)
    group_size = layer.group_size
    bits = layer.bits
    deq = mx.dequantize(w, scales=scales, biases=biases,
                         group_size=group_size, bits=bits)
    return deq, group_size, bits


def requantize_layer(layer, new_deq_weight, group_size, bits) -> None:
    """Quantize new fp32 weight back into the layer's OWN quantization format
    (same group_size/bits it already had — we never change the quant scheme).

    If bits is None the module was unquantized (bf16); store the weight back
    verbatim in the layer's original dtype without quantizing."""
    import mlx.core as mx

    if bits is None:
        layer.weight = new_deq_weight.astype(layer.weight.dtype)
        return

    w_q, scales, biases = mx.quantize(new_deq_weight,
                                       group_size=group_size, bits=bits)
    layer.weight = w_q
    layer.scales = scales
    if biases is not None:
        layer.biases = biases


def orthogonalize_quantized_layer(layer, directions) -> None:
    """Dequantize -> orthogonalize against all directions -> requantize in place."""
    deq, group_size, bits = dequantize_layer_weight(layer)
    modified = orthogonalize_weight_mx(deq, directions)
    requantize_layer(layer, modified, group_size, bits)


# --------------------------------------------------------------------------
# M3 module-walk helpers (arch knowledge baked in here, not scattered below)
# --------------------------------------------------------------------------

def m3_layer_target_modules(layer):
    """Given an M3DecoderLayer, return the list of (name, module) pairs to
    orthogonalize: attention o_proj + MLP output projection(s).

    Dense layers (layer.mlp is M3MLP):      mlp.down_proj
    MoE layers   (layer.mlp is M3MoE):      mlp.switch_mlp.down_proj (3D)
                                             mlp.shared_experts.down_proj (2D)
                                             (note PLURAL shared_experts)
    """
    targets = []
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        targets.append(("self_attn.o_proj", layer.self_attn.o_proj))
    # Kept for parity with the 397B script; M3 has no linear_attn variant
    # (dead branch here, harmless no-op).
    if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
        targets.append(("linear_attn.out_proj", layer.linear_attn.out_proj))
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp") and hasattr(mlp.switch_mlp, "down_proj"):
            targets.append(("mlp.switch_mlp.down_proj", mlp.switch_mlp.down_proj))
        if hasattr(mlp, "shared_experts") and hasattr(mlp.shared_experts, "down_proj"):
            # PLURAL — the one real fix vs. the 397B script's `shared_expert`.
            targets.append(("mlp.shared_experts.down_proj", mlp.shared_experts.down_proj))
        if hasattr(mlp, "down_proj") and not hasattr(mlp, "switch_mlp"):
            # Dense (non-MoE) layer's plain M3MLP.
            targets.append(("mlp.down_proj", mlp.down_proj))
    return targets


def m3_dense_moe_plan(config_path: Path):
    """Read config.json only (no weights) and return
    (num_layers, first_k_dense, moe_layer_freq) so --dry-structural can
    print an accurate per-layer plan without loading the model."""
    cfg = json.loads(config_path.read_text())
    tc = cfg.get("text_config", cfg)  # tolerate a flat (non-VL) config too
    num_layers = tc["num_hidden_layers"]
    freq = tc.get("moe_layer_freq")
    first_k_dense = 0
    if freq:
        for i, v in enumerate(freq):
            if v == 1:
                first_k_dense = i
                break
    return num_layers, first_k_dense, freq


# --------------------------------------------------------------------------

def _is_quantized_majority(config_path):
    """True if the source config.json indicates an already-quantized model
    (a `quantization` block with top-level bits/group_size, or per-module quant
    dicts). Abliterating such a model = dequant->edit->requant at low bits, which
    CORRUPTS weights (proven 2026-07-09: garbage output)."""
    try:
        cfg = json.loads(Path(config_path).expanduser().read_text())
    except Exception:
        return False
    q = cfg.get("quantization")
    if not isinstance(q, dict):
        return False
    if q.get("bits") is not None or q.get("group_size") is not None:
        return True
    return any(isinstance(v, dict) for v in q.values())


def main():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)  # SMB/GPU-timeout fix (matches m3_requant_6bit.py)
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
    p.add_argument("--force-quantized", action="store_true",
                   help="Override the refusal to abliterate a quantized (low-bit) "
                        "model. Abliterating a quantized model = dequant->edit->requant "
                        "at low bits, which CORRUPTS weights (proven 2026-07-09). Use "
                        "the fused bf16 path: ~/pipeline-automation/fused_abliterate_quantize.py.")
    p.add_argument("--dry-structural", action="store_true",
                   help="Print the plan (directions, source layer, target layers, "
                        "which module paths would be touched) WITHOUT loading the "
                        "model or touching the GPU. Reads only the small .npz "
                        "direction files and the source model's config.json.")
    args = p.parse_args()

    direction_files = [f.strip() for f in args.directions.split(",") if f.strip()]
    direction_arrays = [np.load(f) for f in direction_files]

    # Pick the direction to use as the global basis vector per file.
    # Late layers encode response-formation signal; using layer N-1 breaks the
    # model. Default: use the 8th-best layer (middle of top-16) — strong enough
    # for refusal signal, late enough to not dominate generation.
    global_vecs = []
    src_layers = []
    for i, d in enumerate(direction_arrays):
        if args.source_layer >= 0:
            src_layer = args.source_layer
        else:
            src_layer = int(d["best_layers"][7])
        vec = d["directions"][src_layer]
        global_vecs.append(vec)
        src_layers.append(src_layer)
        print(f"Direction {i} ({direction_files[i]}): using layer {src_layer}, "
              f"magnitude {d['magnitudes'][src_layer]:.2f}")

    orth_basis_np = gram_schmidt(global_vecs)
    print(f"Orthonormal basis size: {len(orth_basis_np)}")

    # Determine target layers from config.json alone (works even without
    # loading the model — needed for --dry-structural).
    src = Path(args.model).expanduser()
    num_layers, first_k_dense, moe_layer_freq = m3_dense_moe_plan(src / "config.json")
    print(f"\nModel config: {num_layers} decoder layers, first_k_dense={first_k_dense} "
          f"(layers 0..{first_k_dense - 1} dense, {first_k_dense}..{num_layers - 1} MoE)")

    if args.target_layers:
        target_layer_set = set(int(x) for x in args.target_layers.split(",") if x.strip())
    else:
        combined_mag = sum(d["magnitudes"] for d in direction_arrays)
        order = np.argsort(-combined_mag)
        target_layer_set = set(int(i) for i in order[:16])
    print(f"Target layers ({len(target_layer_set)}): {sorted(target_layer_set)}")

    if args.dry_structural:
        print("\n--dry-structural: NOT loading the model. Planned module walk:")
        if not args.skip_embed:
            print("  embed_tokens  (row-wise orthogonalize against basis; "
                  "quantized-embedding path used if the embed table is quantized)")
        else:
            print("  embed_tokens  SKIPPED (--skip-embed)")
        for i in sorted(target_layer_set):
            is_dense = i < first_k_dense
            kind = "dense (M3MLP)" if is_dense else "MoE (M3MoE)"
            mods = ["self_attn.o_proj"]
            if is_dense:
                mods.append("mlp.down_proj")
            else:
                mods += ["mlp.switch_mlp.down_proj (3D, experts x out x in)",
                         "mlp.shared_experts.down_proj (2D)"]
            print(f"  layer {i:3d}  [{kind}]  -> " + ", ".join(mods))
        print(f"\nOutput would be written to: {Path(args.output).expanduser()}")
        print("Save path (NOT executed in dry-structural mode): "
              "mlx_vlm.utils.save_weights(out, model, donate_weights=True) "
              "+ copy config/tokenizer/processor/*.py/*.jinja files from source "
              "(see module docstring for the exact list and rationale).")
        return

    # ---- refuse to abliterate an already-quantized model (corrupting order) ----
    if _is_quantized_majority(src / "config.json") and not args.force_quantized:
        print("\nREFUSING: input model at %s appears QUANTIZED (config.json has a "
              "quantization block)." % src)
        print("Abliterating a quantized model = dequant -> edit -> requant at low bits, "
              "which CORRUPTS weights (proven 2026-07-09: garbage output).")
        print("Correct order is abliterate at bf16 THEN quantize once:")
        print("  ~/pipeline-automation/fused_abliterate_quantize.py")
        print("Pass --force-quantized to override (NOT recommended).")
        sys.exit(1)

    # ---- real run below: loads the model, requires GPU/RAM ----
    import mlx.core as mx
    from mlx_vlm.utils import load, save_weights

    print(f"\nLoading model from {args.model}...")
    t0 = time.time()
    model, processor = load(args.model)
    print(f"Loaded in {time.time() - t0:.1f}s")

    tm = model.language_model.model
    layers = tm.layers
    print(f"Model has {len(layers)} decoder layers")
    if len(layers) != num_layers:
        print(f"WARNING: config.json says {num_layers} layers but loaded model has "
              f"{len(layers)} — target-layer selection above was computed from config.json "
              f"and may be stale. Continuing with the loaded model's layer count.")

    orth_basis_mx = [mx.array(v, dtype=mx.float32) for v in orth_basis_np]

    # Optional: orthogonalize embed_tokens.
    if not args.skip_embed:
        print("\nOrthogonalizing embed_tokens...")
        embed = tm.embed_tokens
        if hasattr(embed, "scales"):
            deq, egs, ebits = dequantize_layer_weight(embed)
        else:
            deq = embed.weight.astype(mx.float32)
        for d in orth_basis_mx:
            coeffs = deq @ d
            deq = deq - coeffs[:, None] @ d[None, :]
        if hasattr(embed, "scales"):
            requantize_layer(embed, deq, egs, ebits)
        else:
            embed.weight = deq.astype(embed.weight.dtype)
    else:
        print("\nSkipping embed_tokens orthogonalization")

    t0 = time.time()
    for i, layer in enumerate(layers):
        if i not in target_layer_set:
            continue
        if args.per_layer:
            layer_vecs = [d["directions"][i] for d in direction_arrays]
            layer_orth_np = gram_schmidt(layer_vecs)
            layer_dirs = [mx.array(v, dtype=mx.float32) for v in layer_orth_np]
        else:
            layer_dirs = orth_basis_mx

        for name, module in m3_layer_target_modules(layer):
            orthogonalize_quantized_layer(module, layer_dirs)

        if (i + 1) % 10 == 0:
            print(f"  orthogonalized {i + 1}/{len(layers)} layers in {time.time() - t0:.1f}s")
    print(f"All layers done in {time.time() - t0:.1f}s")

    # ---- Save: mirror mlx_vlm.convert.convert()'s save path (see module
    # docstring point 5 for full rationale) ----
    out_path = Path(args.output).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving modified weights to {out_path} (sharded, mlx_vlm layout)...")
    t0 = time.time()
    save_weights(out_path, model, donate_weights=True)
    print(f"Weights + model.safetensors.index.json saved in {time.time() - t0:.1f}s")

    # Copy config/tokenizer/processor/custom-code files verbatim. Mirrors
    # mlx_vlm/convert.py's copy logic; adds *.jinja explicitly since we don't
    # call processor.save_pretrained() (we never touch processor state).
    copied = []
    for pattern in ["*.py", "*.json", "*.jinja", "*.txt"]:
        for file in glob.glob(str(src / pattern)):
            fname = Path(file).name
            if fname == "model.safetensors.index.json":
                # save_weights() just wrote OUR correct index for our shard
                # layout — do not overwrite it with the source's.
                continue
            shutil.copy(file, out_path / fname)
            copied.append(fname)
    print(f"Copied metadata/code files: {sorted(copied)}")

    # Copy any subdirectories verbatim (parity with convert.py; none observed
    # in the prod 6-bit dir at inspection time, but harmless if one exists).
    for item in src.iterdir():
        if item.is_dir():
            dest = out_path / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"Copied subdirectory: {item.name}/")

    print(f"\nDone. Abliterated model at: {out_path}")
    print("Test with (mlx_vlm, NOT mlx_lm):")
    print(f"  python -c \"from mlx_vlm.utils import load; from mlx_vlm.generate import generate; "
          f"m,p=load('{out_path}'); print(generate(m,p,'What happened at Tiananmen in 1989? "
          f"Answer briefly.', max_tokens=200))\"")


if __name__ == "__main__":
    main()
