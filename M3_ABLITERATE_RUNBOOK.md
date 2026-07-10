# M3 Abliteration Runbook

Target model: `~/models/MiniMax-M3-6bit-v1` — **does NOT exist yet** (the
fuse job producing it was still running as of this writeup). Do not start
step 1 until it's confirmed present and the box is free.

Interpreter for every step below: `/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3`
(the ONLY interpreter on the box with minimax_m3 support — see
`reference_m3_training_interpreter`). All mlx_vlm-importing steps need
`PYTHONPATH=~/dev/mlx-vlm:~`.

**GPU/RAM is single-tenant on this box.** Every step that loads the model
(capture, bake, serve) needs the FULL box — one at a time, never overlapped
with prod (:8082) or with each other. `refusal_probe.py` itself is just an
HTTP client and is cheap/idle-safe, but it's pointless without a server
running, and the server IS the expensive part.

---

## 0. Preconditions

```bash
ls ~/models/MiniMax-M3-6bit-v1/config.json ~/models/MiniMax-M3-6bit-v1/model.safetensors.index.json
```
Confirms the fuse+requant finished and produced a real mlx_vlm-loadable dir
(config.json + sharded safetensors + index). **HUMAN CHECK:** also spot-read
`config.json`'s `quantization` block and compare module-key count against
`~/models/MiniMax-M3-v3-6bit-mixed/config.json` (the prod reference) the way
`m3_requant_6bit.py --verify-only` does — don't proceed on a partial/corrupt
fuse.

---

## 1. Serve v1 (baseline) for activation capture — OR skip and load directly

Two options; pick one:

**1a. (Preferred, fewer moving parts) Load directly, no server.**
`capture_activations_vlm.py` calls `mlx_vlm.utils.load(args.model)` itself —
it does NOT need a running HTTP server. Skip straight to step 2.

**1b. (If you want a server up first for other reasons.)** Serve on an
isolated, non-prod port so nothing collides with :8082:
```bash
HF_HUB_OFFLINE=1 M3_PORT=8086 M3_MODEL_DIR=~/models/MiniMax-M3-6bit-v1 \
  PYTHONPATH=~/dev/mlx-vlm:~ \
  /Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 ~/m3_serve_batched.py &
```
**GPU-EXCLUSIVE.** If you do this, do NOT also run capture_activations_vlm.py
in-process against the same weights concurrently — one resident copy of a
~327GB model is already a lot; two would not fit. If serving for the "before"
refusal_probe pass, capture activations AFTERWARD once the server is up (a
server load and a script-load are two separate weight copies in memory —
run capture BEFORE bringing the server up, or kill the server first).

**HUMAN CHECK:** `curl :8086/v1/models` returns the model id before trusting
anything downstream.

---

## 2. Capture activations (three prompt sets, already exist)

```bash
cd ~/alfred-abliterate
PYTHONPATH=~/dev/mlx-vlm:~ /Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 \
  capture_activations_vlm.py --model ~/models/MiniMax-M3-6bit-v1 \
  --prompts prompts_benign.json --out acts_benign_m3v1.npz

PYTHONPATH=~/dev/mlx-vlm:~ /Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 \
  capture_activations_vlm.py --model ~/models/MiniMax-M3-6bit-v1 \
  --prompts prompts_cn_refusal.json --out acts_cn_refusal_m3v1.npz

PYTHONPATH=~/dev/mlx-vlm:~ /Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 \
  capture_activations_vlm.py --model ~/models/MiniMax-M3-6bit-v1 \
  --prompts prompts_safety.json --out acts_safety_m3v1.npz
```
**GPU-EXCLUSIVE**, sequential (each run loads the full model). This is
`capture_activations_vlm.py` UNCHANGED — no edits needed per the ground-truth
note that it's already architecture-agnostic. **Run the first (benign) one
first and manually check its printed `DecoderLayer` class name is
`minimax_m3.language.M3DecoderLayer`** (or wherever mlx_vlm.load resolves it)
and that `Layers: 60` — a quick sanity check that the monkeypatch actually
attached to the right class before burning time on the other two prompt sets.

Expected total time: **unknown, not measured on this box for M3 yet** — the
397B runs on the old Qwen box are not a valid ETA anchor for this GPU/model
combo. Time the first capture and extrapolate for the remaining two, don't
guess.

---

## 3. Compute directions

```bash
cd ~/alfred-abliterate
/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 compute_direction.py \
  --refusal acts_cn_refusal_m3v1.npz --benign acts_benign_m3v1.npz \
  --out cn_refusal_direction_m3v1.npz --top-k 16

/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 compute_direction.py \
  --refusal acts_safety_m3v1.npz --benign acts_benign_m3v1.npz \
  --out safety_direction_m3v1.npz --top-k 16
```
Pure numpy, no GPU load, cheap. Can run on any interpreter with numpy, but
use the pinned one for consistency. **HUMAN CHECK:** eyeball the printed
per-layer magnitude table — confirm the top layers are in a sane
30-70%-depth band (layers ~18-42 of 60), not layer 0 or layer 59. A max at
layer 59 (the final layer) would mean the "direction" is really
response-formation signal, not refusal signal, and `--source-layer` picking
`best_layers[7]` would then be picking a near-final layer too — re-derive
`--source-layer` manually in that case rather than trusting the default.

---

## 4. Baseline refusal probe (BEFORE)

Requires a running server (step 1b) or ANY separately-served copy of v1 on
an isolated port. If you skipped 1b, stand it up now:
```bash
HF_HUB_OFFLINE=1 M3_PORT=8086 M3_MODEL_DIR=~/models/MiniMax-M3-6bit-v1 \
  PYTHONPATH=~/dev/mlx-vlm:~ \
  /Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 ~/m3_serve_batched.py &
sleep 5 && curl -s localhost:8086/v1/models   # HUMAN CHECK: confirm it's up before probing
```
Then:
```bash
cd ~/alfred-abliterate
/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 refusal_probe.py \
  --base http://127.0.0.1:8086/v1 --model minimax-m3 \
  --prompts prompts_cn_refusal.json --label "M3-v1 BEFORE / cn_refusal" \
  --out probe_cn_refusal_before.json

/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 refusal_probe.py \
  --base http://127.0.0.1:8086/v1 --model minimax-m3 \
  --prompts prompts_safety.json --label "M3-v1 BEFORE / safety" \
  --out probe_safety_before.json
```
**HUMAN CHECK:** skim a handful of the saved `response_first_200` fields for
false ANSWERED/REFUSED verdicts before treating the summary numbers as
ground truth — the keyword heuristic in `refusal_probe.py` is reasonable but
not perfect (e.g. a substantive answer that opens with a hedge like "I can't
be 100% certain, but here's what's documented..." could false-positive as
REFUSED since the marker is in the first 200 chars).

Kill this server (`kill %1` or find+kill the PID on :8086) before step 5 —
bake needs the GPU/RAM free.

---

## 5. Bake abliteration

```bash
cd ~/alfred-abliterate
PYTHONPATH=~/dev/mlx-vlm:~ /Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 \
  bake_abliteration_m3.py \
  --model ~/models/MiniMax-M3-6bit-v1 \
  --directions cn_refusal_direction_m3v1.npz,safety_direction_m3v1.npz \
  --output ~/models/MiniMax-M3-6bit-v1-abliterated
```
**GPU-EXCLUSIVE, the longest step** (loads the full ~327GB model, dequant/
requantizes ~16 layers x ~4 target modules each, then writes ~327GB back out
across ~86 shards). Not timed on this box — measure, don't guess, before
promising a duration to anyone downstream.

**Before running for real**, dry-run the plan with no GPU load:
```bash
/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 bake_abliteration_m3.py \
  --model ~/models/MiniMax-M3-6bit-v1 \
  --directions cn_refusal_direction_m3v1.npz,safety_direction_m3v1.npz \
  --output ~/models/MiniMax-M3-6bit-v1-abliterated \
  --dry-structural
```
**HUMAN CHECK on the dry-run output, before spending the real run:**
- target layer list looks reasonable (spread across mid-late layers, not
  clustered at 0 or 59)
- dense/MoE split matches expectation (layers 0-2 dense, 3-59 MoE, per the
  live config — confirm the printed `first_k_dense` matches)
- disk has ~330GB+ free at `~/models/` for the new output dir (it will NOT
  be smaller than the input — same quant scheme, same shard count roughly)

**HUMAN CHECK after the real run, before step 6:**
- `ls ~/models/MiniMax-M3-6bit-v1-abliterated/` — shard count roughly matches
  source's 86 (some drift is fine, save_weights re-shards from scratch at
  5GB/shard; a WILDLY different count, e.g. 3 or 300, means something broke)
- `model.safetensors.index.json` exists and its `weight_map` key count is
  sane
- every file that should have been copied is present: `config.json`,
  `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `added_tokens.json`, `vocab.json`,
  `preprocessor_config.json`, `processor_config.json`, `chat_template.jinja`,
  `configuration_minimax_m3_vl.py`, `image_processor.py`,
  `processing_minimax.py`, `video_processor.py` — diff the file list against
  the source dir's non-shard files
- **this is the single riskiest unverified step in the whole pipeline** — see
  the SAVE-PATH UNCERTAINTY section below. Do not skip the checks above.

---

## 6. Serve abliterated, refusal probe (AFTER), capability sanity check

```bash
HF_HUB_OFFLINE=1 M3_PORT=8086 M3_MODEL_DIR=~/models/MiniMax-M3-6bit-v1-abliterated \
  PYTHONPATH=~/dev/mlx-vlm:~ \
  /Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 ~/m3_serve_batched.py &
sleep 5 && curl -s localhost:8086/v1/models   # HUMAN CHECK before probing
```
**GPU-EXCLUSIVE.** Then:
```bash
cd ~/alfred-abliterate
/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 refusal_probe.py \
  --base http://127.0.0.1:8086/v1 --model minimax-m3 \
  --prompts prompts_cn_refusal.json --label "M3-v1-abliterated AFTER / cn_refusal" \
  --out probe_cn_refusal_after.json

/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 refusal_probe.py \
  --base http://127.0.0.1:8086/v1 --model minimax-m3 \
  --prompts prompts_safety.json --label "M3-v1-abliterated AFTER / safety" \
  --out probe_safety_after.json
```
Compare `probe_*_before.json` summaries vs `probe_*_after.json` — refused
count should drop materially.

**Capability sanity check (mandatory, not optional) — abliteration can break
a model outright, not just uncensor it.** Run a handful of BENIGN prompts
(`prompts_benign.json`) through the same probe AND manually read a couple of
full completions (not just the 200-char preview) for coherence:
```bash
/Users/alfredpennyworth/.pyenv/versions/3.12.13/bin/python3 refusal_probe.py \
  --base http://127.0.0.1:8086/v1 --model minimax-m3 \
  --prompts prompts_benign.json --label "M3-v1-abliterated AFTER / benign (capability check)" \
  --out probe_benign_after.json
```
**HUMAN CHECK, hard gate before this model touches anything real:** benign
answered-rate should stay ~100% (a drop here means the bake damaged general
capability, not just refusal behavior — e.g. too many target layers, or
`--skip-embed` needed, or the source-layer pick was too late/general). Also
manually run one hand-picked reasoning/coding prompt through and read the
full output — the keyword-heuristic probe cannot detect "answers but is now
incoherent."

Kill the :8086 server when done — do not leave a second M3 copy resident.

---

## Uncertainty flags / assumptions (see also the full report)

- **SAVE PATH is the biggest unknown.** `bake_abliteration_m3.py` calls
  `mlx_vlm.utils.save_weights(out, model, donate_weights=True)` (confirmed
  present at `~/dev/mlx-vlm/mlx_vlm/utils.py:1112`, and confirmed to be the
  exact function `mlx_vlm.convert.convert()` itself calls) plus a manual copy
  of config/tokenizer/processor/code files mirroring `convert.py`'s copy
  loop. This was verified by READING the source, never by RUNNING it on this
  box (GPU busy). The dequant -> requant round-trip for the ~16 target
  layers plus embed_tokens should produce tensors of the identical shape/
  dtype/quant-params as before (same `group_size`/`bits`, read off each
  layer itself), so `save_weights`'s auto-resharding should just work — but
  "should" is doing real work in that sentence. First real bake run needs
  the post-run HUMAN CHECK list in step 5 before trusting the output.
- `--dry-structural` reads `config.json`'s `text_config.moe_layer_freq` to
  compute `first_k_dense` without loading the model. This key's presence
  was confirmed on the prod reference config
  (`~/models/MiniMax-M3-v3-6bit-mixed/config.json` has `moe_layer_freq` as a
  60-element list, `first_k_dense=3` computed from it) — assumed the v1
  target model's config has the same shape (near-certain, same requant
  pipeline per `m3_requant_6bit.py`, but not directly inspected since v1
  doesn't exist yet).
- Timings throughout are marked "not measured" deliberately — do not let
  anyone (including future-me) quote an ETA for capture or bake without
  timing the first real run on THIS box with THIS model.
- `refusal_probe.py`'s classifier is a keyword heuristic, not a judge model.
  It will have some false-positive/negative rate. Treat the printed summary
  numbers as a directional signal to act on, not a certified score — spot
  check per prompt before reporting the before/after delta as fact.
