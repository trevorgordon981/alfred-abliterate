# M3 abliteration runbook

## Non-negotiable weight-order invariant

Never edit an already-low-bit MiniMax-M3. The former direct bake sequence—load
6-bit weights, expand selected projections, edit them, then write them back at
6-bit—produced corrupt output on 2026-07-09 and is permanently retired. There is
no force flag or emergency override.

The only supported M3 build is:

1. start from the full-VL bf16 source;
2. remove the selected directions while the weights are full precision;
3. quantize the complete model exactly once to the reviewed mixed 6/8-bit recipe;
4. scan, evaluate, receipt-bind, and leave production untouched.

`bake_abliteration_m3.py` is now a helper library for that builder. Direct
execution stops before any MLX import or model load, and its helper functions
reject modules carrying low-bit attributes.

## Canonical build

The authoritative implementation is
`~/pipeline-automation/fused_abliterate_quantize.py`. Invoke it through the
receipt-building wrapper so source, helper, ordered directions, recipe, result,
and output are all bound together:

```bash
PA=~/pipeline-automation

bash "$PA/build_gate_promote_abliterated_m3.sh" build \
  --ver abl-vN \
  --model /absolute/path/to/MiniMax-M3-full-vl-bf16 \
  --output /absolute/path/to/fresh/MiniMax-M3-abl-vN-mixed-6bit \
  --dirs-root /absolute/path/to/alfred-abliterate \
  --direction cn_refusal_direction_m3v1.npz \
  --direction safety_direction_m3v1.npz \
  --baseline-model /absolute/path/to/lucidity-baseline \
  --recipe-reference /absolute/path/to/reviewed-production-recipe
```

The wrapper calls `automation/fused_abliterate_quantize.py`; it refuses a
source that declares quantization before loading the model, performs the edit in
bf16, applies one whole-model quantization, validates the saved candidate, runs
the lucidity gate, and creates an immutable artifact receipt.

The build requires exclusive Studio memory and must not run during training or
while Trevor's custom Python M3 engine is resident. Nothing in this repository
may stop either process to make room.

## Direction preparation

Activation capture and direction computation remain research inputs. They do
not mutate weights, but capture still loads a model and therefore needs an
explicit maintenance window. Use an isolated model/port, never production.

```bash
cd ~/alfred-abliterate
PY=/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3

PYTHONPATH=~/dev/mlx-vlm:~ "$PY" capture_activations_vlm.py \
  --model /absolute/path/to/read-only-reference-model \
  --prompts prompts_benign.json --out acts_benign_m3v1.npz

PYTHONPATH=~/dev/mlx-vlm:~ "$PY" capture_activations_vlm.py \
  --model /absolute/path/to/read-only-reference-model \
  --prompts prompts_cn_refusal.json --out acts_cn_refusal_m3v1.npz

"$PY" compute_direction.py \
  --refusal acts_cn_refusal_m3v1.npz --benign acts_benign_m3v1.npz \
  --out cn_refusal_direction_m3v1.npz --top-k 16
```

Inspect direction magnitudes and layer selection before a build. A non-finite,
zero, layer-0, or final-layer-dominated result is not an input to trust.

## Verification and activation boundary

A successful build is not authorization to activate the candidate. Keep the
custom Python serving engine and production pointer unchanged until all current
pipeline gates are satisfied, including the exact Gordon candidate/baseline
decision and portfolio-ledger requirements documented in the training-pipeline
runbook. Activation and recovery use the separately reviewed promotion command;
this repository intentionally contains no production cutover shortcut.

For an offline evaluation, use a non-production port and compare the candidate
against its bound baseline. Read full completions as well as aggregate refusal
and coherence scores. Stop on any repetition, garbage output, reasoning damage,
or capability regression.

## Fail-closed recovery

If the fused build is interrupted, do not reuse a partial output directory.
Follow `automation/RECOVERY_AND_ACTIVATION.md` and use the builder's explicit
`recover-build` or `cleanup-build` operation against the exact result receipt.
Those commands reconcile the durable build journal; manual relabeling or copying
of a candidate is not a recovery mechanism.
