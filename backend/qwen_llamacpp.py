"""
Run the fine-tuned Qwen on the CPU through llama.cpp's llama-server (no GPU, no cloud).

The model is the same Qwen2.5-7B + LoRA, merged and compressed to a 4-bit GGUF file
(made once with qwen_local/Make_Qwen_CPU_Model_Colab.ipynb). llama-server runs it
locally; this module sends it exactly the prompt used in fine-tuning.
"""

import math

import httpx

import config
from qwen_service.qwen_core import CLASSES, make_messages, parse_prediction

MODEL_LABEL = "Qwen2.5-7B (fine-tuned)"


def render_prompt(patient: dict, extra: list | None = None) -> str:
    """The Qwen2.5 chat template (ChatML) applied to the fine-tuning messages,
    with the assistant turn opened - identical to tokenizer.apply_chat_template(add_generation_prompt=True)."""
    out = ""
    for m in make_messages(patient, extra):
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out + "<|im_start|>assistant\n"


def _confidence(body: dict) -> float | None:
    """Probability of the generated answer = product of its token probabilities.
    Handles both the newer (logprob) and older (probs) llama-server formats."""
    items = body.get("completion_probabilities") or []
    if not items:
        return None
    total = 0.0
    for it in items:
        if "logprob" in it:
            total += float(it["logprob"])
        elif "probs" in it:
            chosen = it.get("content")
            p = next((c.get("prob") for c in it["probs"] if c.get("tok_str") == chosen), None)
            if p is None and it["probs"]:
                p = it["probs"][0].get("prob")
            if not p:
                return None
            total += math.log(float(p))
        elif "prob" in it:
            total += math.log(float(it["prob"]))
        else:
            return None
    return round(min(math.exp(total), 1.0) * 100, 2)


def _unavailable(reason: str) -> dict:
    return {"available": False, "model": MODEL_LABEL, "error": reason}


async def predict(patient: dict, extra: list | None = None) -> dict:
    payload = {
        "prompt": render_prompt(patient, extra),
        "n_predict": 16,
        "temperature": 0,
        "top_k": 1,
        "n_probs": 1,
        "cache_prompt": True,  # the long system prompt is reused between requests
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    try:
        async with httpx.AsyncClient(timeout=config.QWEN_TIMEOUT_SECONDS) as client:
            r = await client.post(f"{config.QWEN_API_URL}/completion", json=payload)
        r.raise_for_status()
        body = r.json()
    except httpx.TimeoutException:
        return _unavailable("Qwen (on this computer) did not answer in time.")
    except (httpx.HTTPError, ValueError):
        return _unavailable("Qwen is not running on this computer. Start it with START.bat.")

    raw = (body.get("content") or "").strip()
    disease = parse_prediction(raw)
    if disease not in CLASSES:
        return _unavailable("Qwen returned an answer outside the 10 classes.")
    confidence = _confidence(body)
    return {
        "available": True,
        "model": MODEL_LABEL,
        "disease": disease,
        "confidence": confidence,
        "probabilities": [{"disease": disease, "probability": confidence}] if confidence is not None else [],
        "raw_output": raw,
    }


async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{config.QWEN_API_URL}/health")
        return {"configured": True, "online": r.status_code == 200, "local": True}
    except httpx.HTTPError:
        return {"configured": True, "online": False, "local": True}
