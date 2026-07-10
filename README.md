# alfred-abliterate

> **Deployment status:** Studio production is served by Trevor's custom Python
> engine, not vMLX. The `abliterated_vlm_server.py` and
> `vllm_abliterate_hook.py` files are retained as historical experiments only;
> they are not an activation path. Runtime integration is deferred until the
> live Byron benchmark is finished.

> **M3 weight-build safety:** Never modify an already-quantized MiniMax-M3.
> `bake_abliteration_m3.py` is helper-only and its direct entry point is
> permanently disabled. The sole M3 build path is
> `~/pipeline-automation/fused_abliterate_quantize.py`, invoked by
> `build_gate_promote_abliterated_m3.sh build`: edit the full-VL bf16 source,
> then quantize the complete candidate exactly once. See
> `M3_ABLITERATE_RUNBOOK.md` for the receipt-bound command.

Residual-stream refusal-direction projection for Qwen3.5-A10B-4bit running under mlx_lm on Mac Studio. Uses the standard abliteration approach (FailSpy, etc.) adapted for the qwen3_5 hybrid MoE architecture and a quantized MLX base model.

**All steps run on Mac Studio. No DGX needed. No fp16 download needed.**

## Architecture-specific notes

- Model is Qwen3.5-122B-A10B-4bit (hybrid GatedDeltaNet + Attention + SparseMoeBlock, 128 experts).
- `mlx_lm.models.qwen3_5.DecoderLayer` returns `out = h + mlp(norm(h))` where `h = x + attn(norm(x))`. That `out` IS the residual stream after the full layer. We hook at this return.
- Because the residual stream is full-precision floats regardless of weight quantization, the refusal direction is discoverable in 4-bit activations.
- **MoE caveat**: routing decisions happen inside `mlp` before the residual add, so the router is not affected by post-block projection. If a "refusal expert" handles a category (e.g., Chinese political topics), residual-only projection may remove the expert's OUTPUT but not its ROUTING. That would surface as partial abliteration where specific topic categories still refuse. See "Expert-level fallback" below if you hit this.

## Pipeline

```
Step 1: Use an approved maintenance window with the custom engine stopped
Step 2: capture_activations.py  (two runs: refusal + benign prompt sets)
Step 3: compute_direction.py    (mean-diff per layer, identify strongest layers)
Step 4: test_abliteration.py    (A/B vs baseline; read the outputs yourself)
Step 5: project_inference.py    (single-prompt inference with projection on)
Step 6: validate a separately reviewed custom-engine integration
```

## Step 1: Isolate the research run

Do not stop, signal, restart, or purge the production custom Python serving
engine from this repository. Confirm that Byron and every other inference job
has finished, schedule a maintenance window, and use a separate working model
copy. Verify Metal capacity before loading the research model.

## Step 2: Capture activations

```bash
PY=/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python
MODEL=~/models/qwen3.5-122b-a10b-4bit

$PY capture_activations.py --model $MODEL \
  --prompts prompts_refusal.json \
  --out acts_refusal.npz

$PY capture_activations.py --model $MODEL \
  --prompts prompts_benign.json \
  --out acts_benign.npz
```

Each run: ~60s to load the 65GB model + ~1-2s per prompt × 32 prompts ≈ 2 min per file. Total ~5 min for both.

Output: two `.npz` files each containing `[32, num_layers, hidden_dim]` float32 arrays.

## Step 3: Compute direction

```bash
$PY compute_direction.py \
  --refusal acts_refusal.npz \
  --benign acts_benign.npz \
  --out refusal_direction.npz \
  --top-k 12
```

Prints per-layer magnitude table. Expect strongest magnitudes in middle-late layers (40-70% depth). Save the layer indices of the top 8-12 for the next step.

## Step 4: A/B test

```bash
$PY test_abliteration.py --model $MODEL \
  --direction refusal_direction.npz \
  --prompts prompts_refusal.json \
  --top-k 8 \
  --max-prompts 10
```

Runs the first 10 refusal-triggering prompts twice (baseline + abliterated) and prints side-by-side outputs plus a heuristic refusal-score delta. **Read the outputs yourself.** The score is a rough signal, not ground truth.

If score delta is > 0.3 (i.e., baseline = 0.7, abliterated = 0.4), you have meaningful signal. If < 0.1, increase `--top-k` or target specific layers via `--layers 24,28,32,36`.

## Step 5: Single-prompt inference

```bash
$PY project_inference.py --model $MODEL \
  --direction refusal_direction.npz \
  --top-k 8 \
  --prompt "What happened at Tiananmen in 1989?"
```

Quick way to iterate on layer selection without the full A/B harness.

## Step 6: Custom-engine integration gate

This repository deliberately does not provide a production activation command.
Once the offline A/B result is acceptable, implement projection support in a
separate branch of Trevor's custom Python engine, add baseline/projection parity
tests, and review the exact deployment diff. Activation requires Byron to be
finished, an engine rollback copy, and a post-restart identity plus completion
smoke test. Legacy vMLX/vLLM adapters are not valid deployment instructions.

## Expert-level fallback (if residual-only is insufficient)

If specific topic categories still refuse after 12-16 layer residual projection:

1. Log expert-selection patterns from `SparseMoeBlock.gate` output during refusal-heavy prompts vs. benign prompts.
2. Identify experts with > 2x activation rate on refusal prompts (candidate "safety experts").
3. Options:
   - Reduce router logits for those experts by a negative bias at inference
   - Average their weights with neighboring experts (destroys the specialization)
   - Train small LoRA adapters on just those experts with compliance examples

This is research-grade work with no turnkey tool. Start with residual-only, move here only if needed.

## Tuning tips

- **Too aggressive** (model becomes incoherent, repeats, adds random refusals elsewhere): reduce `--top-k` or narrow `--layers` to a smaller window.
- **Not aggressive enough**: increase `--top-k`, or target specific layers identified by magnitude (the top-K usually has diminishing returns after 8-12).
- **Uneven by category**: some topics abliterate cleanly, others don't. That's the MoE router effect. See expert-level fallback.

## Files

| File | Purpose |
|---|---|
| `prompts_refusal.json` | 32 refusal-triggering prompts (Chinese political topics, overcautious-refusal patterns) |
| `prompts_benign.json` | 32 matched-length benign prompts |
| `capture_activations.py` | Forward-pass residual stream capture |
| `compute_direction.py` | Mean-difference direction computation + per-layer magnitude report |
| `project_inference.py` | Single-prompt inference with projection hook |
| `test_abliteration.py` | A/B baseline vs abliterated harness with refusal-score heuristic |

## Restoration

The research pipeline never mutates the base MLX weights at
`~/models/qwen3.5-122b-a10b-4bit/`. If a later custom-engine integration is
activated, rollback means restoring the reviewed pre-integration engine file
and restarting only that engine during a maintenance window.
