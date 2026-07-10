#!/bin/bash
set -euo pipefail

# Offline abliteration pipeline: capture and compute candidate directions.
# It never stops, starts, signals, purges, or rewrites a serving process.

WORKDIR="$HOME/alfred-abliterate"
PYTHON="$HOME/.pyenv/versions/3.12.13/bin/python3"
MODEL="$HOME/models/qwen3.5-397b-a17b-4bit"
OUTPUT_DIR="$WORKDIR/outputs"
CONFIG="$OUTPUT_DIR/abliterate.candidate.json"
LOG="$WORKDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== ABLITERATION PIPELINE START ==="

if pgrep -f 'm3_serve[^ ]*\.py' >/dev/null 2>&1; then
    log "REFUSING: the custom Python serving engine is running"
    exit 2
fi
if [ "${ABLITERATE_OFFLINE_CONFIRMED:-}" != "1" ]; then
    log "REFUSING: set ABLITERATE_OFFLINE_CONFIRMED=1 in an approved maintenance window"
    exit 2
fi
log "Offline guard passed; no serving process will be modified"

# Step 1: Build context-heavy prompts (if script exists and ctx files don't)
if [ -f "$WORKDIR/build_context_prompts.py" ] && [ ! -f "$WORKDIR/prompts_refusal_ctx.json" ]; then
    log "Building context-heavy prompts..."
    $PYTHON "$WORKDIR/build_context_prompts.py" 2>&1 | tee -a "$LOG"
    log "Context prompts built"
fi

# Step 2: Capture activations
for CATEGORY in refusal safety benign; do
    PROMPTS="$WORKDIR/prompts_${CATEGORY}_ctx.json"
    if [ ! -f "$PROMPTS" ]; then
        PROMPTS="$WORKDIR/prompts_${CATEGORY}.json"
    fi
    OUT="$WORKDIR/acts_${CATEGORY}_397b_ctx.npz"

    if [ -f "$OUT" ]; then
        log "Skipping $CATEGORY capture (already exists)"
        continue
    fi

    log "Capturing $CATEGORY activations from $PROMPTS..."
    $PYTHON "$WORKDIR/capture_activations_vlm.py" \
        --model "$MODEL" \
        --prompts "$PROMPTS" \
        --out "$OUT" 2>&1 | tee -a "$LOG"
    log "$CATEGORY capture complete"
done

# Step 3: Compute directions
log "Computing refusal direction..."
$PYTHON "$WORKDIR/compute_direction.py" \
    --refusal "$WORKDIR/acts_refusal_397b_ctx.npz" \
    --benign "$WORKDIR/acts_benign_397b_ctx.npz" \
    --out "$WORKDIR/refusal_direction_397b_ctx.npz" \
    --top-k 16 2>&1 | tee -a "$LOG"

log "Computing safety direction..."
$PYTHON "$WORKDIR/compute_direction.py" \
    --refusal "$WORKDIR/acts_safety_397b_ctx.npz" \
    --benign "$WORKDIR/acts_benign_397b_ctx.npz" \
    --out "$WORKDIR/safety_direction_397b_ctx.npz" \
    --top-k 16 2>&1 | tee -a "$LOG"

# Step 4: Write a candidate config under the ignored output directory.
log "Writing candidate config..."
cat > "$CONFIG" << JSONEOF
{
  "directions": [
    "$WORKDIR/refusal_direction_397b_ctx.npz",
    "$WORKDIR/safety_direction_397b_ctx.npz"
  ],
  "top_k": 16
}
JSONEOF
log "Config updated"

log "=== PIPELINE COMPLETE ==="
log "Candidate artifacts are offline only; custom-engine integration requires a separate review"
log "Log saved to $LOG"
