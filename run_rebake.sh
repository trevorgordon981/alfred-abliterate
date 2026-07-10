#!/bin/bash
set -euo pipefail
cd ~/alfred-abliterate

if pgrep -f 'm3_serve[^ ]*\.py' >/dev/null 2>&1; then
  echo "REFUSING: the custom Python serving engine is running" >&2
  exit 2
fi
if [ "${ABLITERATE_OFFLINE_CONFIRMED:-}" != "1" ]; then
  echo "REFUSING: set ABLITERATE_OFFLINE_CONFIRMED=1 in an approved maintenance window" >&2
  exit 2
fi

MODEL=~/models/qwen3.5-397b-a17b-4bit
OUT_MODEL=~/models/qwen3.5-397b-a17b-4bit-abliterated

echo "=== Step 1a: Capture safety activations ==="
python3 capture_activations.py \
  --model "$MODEL" \
  --prompts prompts_safety.json \
  --out acts_safety_397b_v2.npz

echo ""
echo "=== Step 1b: Capture benign activations ==="
python3 capture_activations.py \
  --model "$MODEL" \
  --prompts prompts_benign.json \
  --out acts_benign_397b_v2.npz

echo ""
echo "=== Step 2a: Compute safety direction ==="
python3 compute_direction.py \
  --refusal acts_safety_397b_v2.npz \
  --benign acts_benign_397b_v2.npz \
  --out safety_direction_397b_v2.npz \
  --top-k 16

echo ""
echo "=== Step 2b: Capture refusal (PRC) activations ==="
python3 capture_activations.py \
  --model "$MODEL" \
  --prompts prompts_refusal.json \
  --out acts_refusal_397b_v2.npz

echo ""
echo "=== Step 2c: Compute refusal direction ==="
python3 compute_direction.py \
  --refusal acts_refusal_397b_v2.npz \
  --benign acts_benign_397b_v2.npz \
  --out refusal_direction_397b_v2.npz \
  --top-k 16

echo ""
echo "=== Step 3: Bake abliteration (per-layer, top-16) ==="
python3 bake_abliteration.py \
  --model "$MODEL" \
  --directions refusal_direction_397b_v2.npz,safety_direction_397b_v2.npz \
  --output "$OUT_MODEL" \
  --per-layer

echo ""
echo "=== Done: offline model candidate built ==="
echo "Do not restart production; custom-engine integration requires a separate review."
