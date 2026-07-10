#!/usr/bin/env python3
"""Full-precision MiniMax-M3 abliteration helper.

This module is imported, byte-bound, and called by
``pipeline-automation/fused_abliterate_quantize.py``.  It intentionally has no
standalone build path: the canonical builder edits a bf16 source in memory and
then quantizes exactly once to the reviewed production recipe.

The historical low-bit path was unsafe.  Dequantizing an already-quantized M3,
editing it, and requantizing the touched weights produced corrupt output on
2026-07-09.  Consequently:

* direct execution always stops before importing MLX or loading a model;
* any module carrying low-bit attributes is rejected before it is edited; and
* the legacy helper export names remain only because the receipt-bound fused
  builder imports those exact symbols.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CANONICAL_BUILDER = "~/pipeline-automation/fused_abliterate_quantize.py"
CANONICAL_WRAPPER = "~/pipeline-automation/build_gate_promote_abliterated_m3.sh build"
_QUANTIZATION_KEYS = frozenset({"quantization", "quantization_config"})
_LOW_BIT_MODULE_ATTRIBUTES = ("scales", "bits", "group_size")


def _numpy():
    """Import numpy only when the receipt-bound builder calls numeric helpers."""
    import numpy as np

    return np


def gram_schmidt(vecs):
    """Orthonormalize numpy vectors; drop near-zero residuals."""
    np = _numpy()
    out = []
    for vector in vecs:
        residual = np.asarray(vector, dtype=np.float32).copy()
        for basis_vector in out:
            residual = residual - np.dot(residual, basis_vector) * basis_vector
        magnitude = np.linalg.norm(residual)
        if magnitude < 1e-6:
            continue
        out.append(residual / magnitude)
    return out


def orthogonalize_weight_mx(weight, directions):
    """Project each full-precision output-axis direction out of a weight."""
    result = weight
    for direction in directions:
        if result.ndim == 2:
            coefficients = direction[None, :] @ result
            result = result - direction[:, None] @ coefficients
        elif result.ndim == 3:
            coefficients = direction[None, :] @ result
            result = result - direction[:, None] @ coefficients
        else:
            raise ValueError(f"unsupported weight ndim: {result.ndim}")
    return result


def _assert_full_precision_module(module) -> None:
    present = [name for name in _LOW_BIT_MODULE_ATTRIBUTES if hasattr(module, name)]
    if present:
        raise RuntimeError(
            "refusing already-quantized M3 module; low-bit attributes present: "
            + ", ".join(present)
            + f". Use {CANONICAL_BUILDER} with a bf16 source."
        )


def dequantize_layer_weight(layer):
    """Return a full-precision view, rejecting every quantized module.

    The legacy function name is part of the fused builder's bound helper ABI.
    This function performs no dequantization and can never return quantization
    parameters.
    """
    _assert_full_precision_module(layer)
    import mlx.core as mx

    return layer.weight.astype(mx.float32), None, None


def requantize_layer(layer, new_weight, group_size, bits) -> None:
    """Store an edited full-precision weight; never quantize it here.

    Quantization belongs exclusively to the fused builder's final, whole-model
    quantization step.  The legacy name is retained for its helper ABI only.
    """
    _assert_full_precision_module(layer)
    if group_size is not None or bits is not None:
        raise RuntimeError(
            "refusing low-bit writeback; the M3 helper accepts full-precision weights only"
        )
    layer.weight = new_weight.astype(layer.weight.dtype)


def orthogonalize_quantized_layer(layer, directions) -> None:
    """Edit a full-precision module, rejecting quantized modules first.

    The legacy name is retained for the receipt-bound helper ABI.  There is no
    dequantize/edit/requantize branch.
    """
    weight, group_size, bits = dequantize_layer_weight(layer)
    modified = orthogonalize_weight_mx(weight, directions)
    requantize_layer(layer, modified, group_size, bits)


def m3_layer_target_modules(layer):
    """Return the residual-writing projections for one M3 decoder layer."""
    targets = []
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        targets.append(("self_attn.o_proj", layer.self_attn.o_proj))
    if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
        targets.append(("linear_attn.out_proj", layer.linear_attn.out_proj))
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp") and hasattr(mlp.switch_mlp, "down_proj"):
            targets.append(("mlp.switch_mlp.down_proj", mlp.switch_mlp.down_proj))
        if hasattr(mlp, "shared_experts") and hasattr(mlp.shared_experts, "down_proj"):
            targets.append(("mlp.shared_experts.down_proj", mlp.shared_experts.down_proj))
        if hasattr(mlp, "down_proj") and not hasattr(mlp, "switch_mlp"):
            targets.append(("mlp.down_proj", mlp.down_proj))
    return targets


def m3_dense_moe_plan(config_path: Path):
    """Read config metadata only and return layer/dense/MoE layout."""
    with Path(config_path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    text_config = config.get("text_config", config)
    num_layers = text_config["num_hidden_layers"]
    frequency = text_config.get("moe_layer_freq")
    first_dense_count = 0
    if frequency:
        for index, value in enumerate(frequency):
            if value == 1:
                first_dense_count = index
                break
    return num_layers, first_dense_count, frequency


def _quantization_declarations(value, path="config"):
    """Return every config path that declares a quantization field."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in _QUANTIZATION_KEYS:
                found.append(child_path)
            found.extend(_quantization_declarations(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_quantization_declarations(child, f"{path}[{index}]"))
    return found


def require_full_precision_source_config(model_path) -> dict:
    """Validate a readable config and fail closed on any quantization marker."""
    config_path = Path(model_path).expanduser() / "config.json"
    try:
        if config_path.is_symlink() or not config_path.is_file():
            raise ValueError("config.json must be a direct regular file")
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot prove a full-precision M3 source: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("cannot prove a full-precision M3 source: config root is not an object")
    declarations = _quantization_declarations(config)
    if declarations:
        raise ValueError(
            "source declares quantization at "
            + ", ".join(declarations)
            + "; low-bit M3 weights must never be edited"
        )
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-layer", type=int, default=-1)
    parser.add_argument("--target-layers", default="")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument("--dry-structural", action="store_true")
    args = parser.parse_args(argv)

    try:
        require_full_precision_source_config(args.model)
    except ValueError as exc:
        parser.exit(
            2,
            f"REFUSING: {exc}. Correct path: {CANONICAL_BUILDER} "
            "(bf16 edit, then one quantization).\n",
        )

    parser.exit(
        2,
        "REFUSING: standalone M3 baking is disabled. Use the receipt-bound "
        f"{CANONICAL_WRAPPER}, which calls {CANONICAL_BUILDER} on a bf16 source "
        "and quantizes exactly once.\n",
    )


if __name__ == "__main__":
    sys.exit(main())
