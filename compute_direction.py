#!/usr/bin/env python3
"""
Compute the refusal direction from captured activations.

Given two .npz files (refusal-triggering prompt activations and benign-prompt
activations), compute the per-layer mean-difference direction:

    direction[L] = normalize(mean(refusal_acts[:, L, :]) - mean(benign_acts[:, L, :]))

Saves: refusal_direction.npz with arrays:
  - directions:    [num_layers, hidden_dim]  (unit vectors)
  - magnitudes:    [num_layers]              (raw pre-normalization L2 norms)
  - best_layers:   [k] indices sorted by magnitude desc

Also prints a table of per-layer magnitudes so you can pick which layers to
project. Typically middle-late layers (30-70% depth) have the strongest
refusal signal.

Usage:
  python compute_direction.py \
    --refusal acts_refusal.npz \
    --benign acts_benign.npz \
    --out refusal_direction.npz \
    --top-k 10
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refusal", required=True, help="Activations .npz from refusal prompts")
    p.add_argument("--benign", required=True, help="Activations .npz from benign prompts")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--top-k", type=int, default=10, help="Print top-K layers by magnitude")
    args = p.parse_args()

    ref = np.load(args.refusal, allow_pickle=True)["activations"]   # [N_r, L, D]
    ben = np.load(args.benign, allow_pickle=True)["activations"]    # [N_b, L, D]

    if ref.shape[1:] != ben.shape[1:]:
        raise SystemExit(f"Shape mismatch: refusal {ref.shape} vs benign {ben.shape}")

    mean_ref = ref.mean(axis=0)   # [L, D]
    mean_ben = ben.mean(axis=0)   # [L, D]
    diff = mean_ref - mean_ben    # [L, D]

    magnitudes = np.linalg.norm(diff, axis=1)  # [L]
    # Normalize per-layer (avoid div-by-zero on pathological layers)
    normed = diff / np.clip(magnitudes[:, None], 1e-8, None)

    order = np.argsort(-magnitudes)  # largest first
    best = order[: args.top_k]

    print("Per-layer refusal-direction magnitudes:")
    print(f"  total layers: {len(magnitudes)}")
    print(f"  max:  layer {int(order[0])} mag={magnitudes[order[0]]:.3f}")
    print(f"  min:  layer {int(order[-1])} mag={magnitudes[order[-1]]:.3f}")
    print(f"  median: {float(np.median(magnitudes)):.3f}")
    print(f"\nTop {args.top_k} layers by magnitude:")
    for idx in best:
        depth_pct = 100 * int(idx) / (len(magnitudes) - 1)
        print(f"  layer {int(idx):3d}  ({depth_pct:5.1f}%)  mag={magnitudes[idx]:.3f}")

    np.savez_compressed(
        args.out,
        directions=normed.astype(np.float32),
        magnitudes=magnitudes.astype(np.float32),
        best_layers=best.astype(np.int64),
    )
    print(f"\nSaved {normed.shape[0]} layer directions (dim={normed.shape[1]}) to {args.out}")


if __name__ == "__main__":
    main()
