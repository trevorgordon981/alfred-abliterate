# alfred-abliterate

Residual-stream refusal-direction projection for Qwen3.5-A10B-4bit running under mlx_lm on Mac Studio. Uses the standard abliteration approach (FailSpy, etc.) adapted for the qwen3_5 hybrid MoE architecture and a quantized MLX base model.

**All steps run on Mac Studio. No DGX needed. No fp16 download needed.**

## Architecture-specific notes

- Model is Qwen3.5-122B-A10B-4bit (hybrid GatedDeltaNet + Attention + SparseMoeBlock, 128 experts).
- `mlx_lm.models.qwen3_5.DecoderLayer` returns `out = h + mlp(norm(h))` where `h = x + attn(norm(x))`. That `out` IS the residual stream after the full layer. We hook at this return.
- Because the residual stream is full-precision floats regardless of weight quantization, the refusal direction is discoverable in 4-bit activations.
- **MoE caveat**: routing decisions happen inside `mlp` before the residual add, so the router is not affected by post-block projection. If a "refusal expert" handles a category (e.g., Chinese political topics), residual-only projection may remove the expert's OUTPUT but not its ROUTING. That would surface as partial abliteration where specific topic categories still refuse. See "Expert-level fallback" below if you hit this.

## Pipeline

```
Step 1: Stop vMLX  (free Metal budget)
Step 2: capture_activations.py  (two runs: refusal + benign prompt sets)
Step 3: compute_direction.py    (mean-diff per layer, identify strongest layers)
Step 4: test_abliteration.py    (A/B vs baseline; read the outputs yourself)
Step 5: project_inference.py    (single-prompt inference with projection on)
Step 6: (optional) bake projection into live vMLX serving chain
```

## Step 1: Free Metal budget

```bash
launchctl unload ~/Library/LaunchAgents/ai.alfred.mlx-vlm-server.plist
launchctl unload ~/Library/LaunchAgents/ai.alfred.mlx-vlm-coder.plist
launchctl unload ~/Library/LaunchAgents/ai.alfred.mlx-vlm-reason.plist
```

Confirm freed: `~/.hermes/skills/devops/metal-memory/metal-memory.py --compact`

## Step 2: Capture activations

```bash
PY=python3
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

## Step 6: (optional) Bake into live serving

Once you're happy with a layer selection:

1. Copy the `attach_projection()` function from `project_inference.py` into your vMLX server startup.
2. Load `refusal_direction.npz` at serve-time, attach hooks to the chosen layers after `load()`.
3. Restart vMLX.

This is reversible: remove the hook call, restart, back to baseline.

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

To revert: remove any hooks / restart vMLX. The base MLX weights at `~/models/qwen3.5-122b-a10b-4bit/` are untouched by this pipeline.

## Related projects

Part of a self-hosted LLM operations toolkit:

- [blockops-proxy](https://github.com/trevorgordon981/blockops-proxy) — tool-call-translating proxy that fronts the abliterated model for OpenAI-compatible clients
- [llm-otel-proxy](https://github.com/trevorgordon981/llm-otel-proxy) — OTel metrics proxy for token/cost/latency tracking
- [alfred-infra](https://github.com/trevorgordon981/alfred-infra) — monitoring + backup infrastructure for the serving cluster
- [alfred-rag](https://github.com/trevorgordon981/alfred-rag) — hybrid RAG stack (dense + BM25 + rerank)
- [context-bench](https://github.com/trevorgordon981/context-bench) — context-window throughput benchmark

## License

MIT
