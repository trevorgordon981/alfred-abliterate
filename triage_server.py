#!/usr/bin/env python3
"""
Abliterated triage + simple inference server for Qwen3.5-9B-AWQ.

Uses transformers + auto-awq for loading, FastAPI for serving.
OpenAI-compatible /v1/chat/completions endpoint.
Applies runtime refusal-direction projection hooks.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

MODEL_PATH = "/home/node2/models/qwen3.5-9b"
ABLITERATE_CONFIG = "/home/node2/models/abliterate.json"
PORT = 8082
MAX_NEW_TOKENS = 4096
DEVICE = "cuda"

app = FastAPI(title="alfred-triage")

model = None
tokenizer = None


def gram_schmidt(vecs):
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


def attach_abliteration(model, config_path):
    config = json.loads(Path(config_path).read_text())
    direction_files = config.get("directions", [])
    top_k = config.get("top_k", 16)
    if not direction_files:
        return

    loaded = [np.load(f) for f in direction_files]
    direction_arrays = [d["directions"] for d in loaded]
    magnitudes = sum(d["magnitudes"] for d in loaded)

    if not all(d.shape == direction_arrays[0].shape for d in direction_arrays):
        print("ABLITERATE: mismatched shapes, skipping")
        return

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        print("ABLITERATE: cannot find decoder layers")
        return

    if len(layers) != direction_arrays[0].shape[0]:
        print(f"ABLITERATE: layer mismatch (model={len(layers)}, dir={direction_arrays[0].shape[0]})")
        return

    order = np.argsort(-magnitudes)
    target_layers = set(order[:top_k].tolist())
    device = next(model.parameters()).device

    for i in target_layers:
        if i >= len(layers):
            continue
        per_layer_vecs = [ds[i] for ds in direction_arrays]
        orth = gram_schmidt(per_layer_vecs)
        layers[i]._abliterate_dirs = [
            torch.tensor(v, dtype=torch.float32, device=device) for v in orth
        ]

    DecoderLayer = type(layers[0])
    existing = DecoderLayer.forward
    original_forward = getattr(existing, '_abliterate_original', existing)

    def patched_forward(self, *args, **kwargs):
        out = original_forward(self, *args, **kwargs)
        dirs = getattr(self, '_abliterate_dirs', None)
        if dirs is not None:
            if isinstance(out, tuple):
                hidden = out[0]
                for d in dirs:
                    d_cast = d.to(hidden.dtype)
                    proj = torch.sum(hidden * d_cast, dim=-1, keepdim=True)
                    hidden = hidden - proj * d_cast
                out = (hidden,) + out[1:]
            else:
                for d in dirs:
                    d_cast = d.to(out.dtype)
                    proj = torch.sum(out * d_cast, dim=-1, keepdim=True)
                    out = out - proj * d_cast
        return out

    DecoderLayer.forward = patched_forward
    patched_forward._abliterate_original = original_forward

    count = sum(1 for l in layers if hasattr(l, '_abliterate_dirs'))
    print(f"ABLITERATE: attached to {count} layers from {len(direction_files)} direction(s)", flush=True)


def load_model():
    global model, tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {MODEL_PATH}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    if Path(ABLITERATE_CONFIG).exists():
        attach_abliteration(model, ABLITERATE_CONFIG)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok" if model else "loading"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_PATH, "object": "model", "created": int(time.time())}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", MAX_NEW_TOKENS)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    # Strip tools/tool_choice - this model doesn't support them
    body.pop("tools", None)
    body.pop("tool_choice", None)
    # Clean messages: remove tool_calls, tool results, and non-string content
    clean_messages = []
    for m in messages:
        role = m.get("role", "")
        if role == "tool":
            continue
        msg = {"role": role}
        content = m.get("content", "")
        if isinstance(content, list):
            # Extract text parts only
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            msg["content"] = "\n".join(text_parts) or ""
        elif isinstance(content, str):
            msg["content"] = content
        else:
            msg["content"] = str(content) if content else ""
        if msg["content"]:
            clean_messages.append(msg)
    messages = clean_messages or [{"role": "user", "content": "hello"}]

    enable_thinking = body.get("enable_thinking", False)
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    gen_time = time.time() - t0
    prompt_tokens = inputs["input_ids"].shape[-1]
    completion_tokens = len(new_tokens)

    if stream:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        def sse_generator():
            # Send the full response as a single chunk (fake streaming)
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_PATH,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": response_text},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            # Send finish chunk
            finish = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_PATH,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(finish)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    return {
        "model": MODEL_PATH,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": response_text},
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
