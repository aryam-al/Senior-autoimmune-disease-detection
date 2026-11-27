"""
"Ask about this result" chat, answered by the same fine-tuned Qwen running on this computer
(llama-server, see qwen_local/). Answers are streamed piece by piece.

Qwen is given the patient's values, both models' results and XGBoost's explanation,
and strict rules: explain, don't diagnose or treat, and send the user to a doctor.
"""

import json

import httpx

import config
from features import BINARY, DISPLAY_NAMES

NORMAL_RANGES = {"ESR": (0, 20, "mm/hr"), "CRP": (0, 5, "mg/L"), "C3": (90, 180, "mg/dL"), "C4": (10, 40, "mg/dL")}

MAX_TURNS = 6           # earlier messages are dropped to stay inside the model's memory
MAX_MESSAGE_CHARS = 1000
MAX_ANSWER_TOKENS = 350

RULES = """You are the ImmunoAI assistant inside a research app for autoimmune disease screening.
You help the user understand ONE lab analysis, described below.

Rules:
- Explain in simple, friendly words. Keep answers short: at most 120 words.
- Reply in the same language as the user's last message (Arabic or English).
- You may explain what each lab test measures, what positive/negative or high/low means,
  why the models chose their result, and what to ask a doctor.
- Never give a diagnosis, never confirm that the user has a disease, and never suggest
  treatment, medication or doses. The results come from research models, not a doctor.
- If something is outside this analysis or outside medicine, say you can only help with this report.
- End medical answers by reminding the user to discuss the results with their doctor."""


def _context(report: dict) -> str:
    inp = report["input"]
    sex = inp.get("Gender") or "sex not given"
    age = f"{int(round(float(inp['Age'])))} years old" if inp.get("Age") is not None else "age not given"
    lines = [f"Patient: {sex}, {age}.", "", "Blood tests:"]
    for name, (low, high, unit) in NORMAL_RANGES.items():
        if inp.get(name) is None:
            lines.append(f"- {DISPLAY_NAMES[name]}: not tested")
            continue
        v = float(inp[name])
        status = "high" if v > high else "low" if v < low else "normal"
        lines.append(f"- {DISPLAY_NAMES[name]}: {v:g} {unit} (normal {low}-{high}, {status})")
    positives = [DISPLAY_NAMES[m] for m in BINARY if inp.get(m) is not None and int(inp[m]) == 1]
    negatives = [DISPLAY_NAMES[m] for m in BINARY if inp.get(m) is not None and int(inp[m]) == 0]
    untested = [DISPLAY_NAMES[m] for m in BINARY if inp.get(m) is None]
    lines += ["", "Antibody tests:",
              f"- Positive: {', '.join(positives) if positives else 'none'}",
              f"- Negative: {', '.join(negatives) if negatives else 'none'}"]
    if untested:
        lines.append(f"- Not tested (unknown, not negative): {', '.join(untested)}")
    extra = report.get("extra") or []
    if extra:
        lines += ["", "Other lab results from the reports (not used by XGBoost):"]
        for e in extra:
            unit = f" {e['unit']}" if e.get("unit") else ""
            flag = f" ({e['flag']})" if e.get("flag") else ""
            lines.append(f"- {e['name']}: {e['value']}{unit}{flag}")
    lines += ["", "Model results:"]

    xgb, qwen = report.get("xgboost") or {}, report.get("qwen") or {}
    if xgb.get("available"):
        lines.append(f"- XGBoost: {xgb['disease']} ({xgb.get('confidence', 0):.0f}% confidence)")
        others = [p for p in xgb.get("probabilities", []) if p["disease"] != xgb["disease"]][:2]
        if others:
            lines.append("  Next most likely: " + ", ".join(f"{p['disease']} ({p['probability']:.1f}%)" for p in others))
    if qwen.get("available"):
        conf = f" ({qwen['confidence']:.0f}% confidence)" if qwen.get("confidence") is not None else ""
        lines.append(f"- Qwen: {qwen['disease']}{conf}")
    elif qwen.get("skipped"):
        lines.append("- Qwen: not selected by the user for this analysis")
    else:
        lines.append("- Qwen: did not run for this analysis")
    if xgb.get("skipped"):
        lines.append("- XGBoost: not selected by the user for this analysis")
    if report.get("agreement") is not None:
        lines.append("- The two models " + ("agree." if report["agreement"] else "disagree."))

    feats = xgb.get("top_features") or []
    if feats:
        lines += ["", "Values that influenced XGBoost the most (explainable AI / SHAP):"]
        for f in feats:
            name = f["feature"]
            if f["value"] is None:
                continue
            value = ("positive" if int(f["value"]) == 1 else "negative") if name in BINARY else f["value"]
            direction = "pushed towards the result" if f["impact"] >= 0 else "pushed away from the result"
            lines.append(f"- {DISPLAY_NAMES.get(name, name)} = {value}: {direction}")
    if report.get("missing"):
        lines.append(f"- {len(report['missing'])} of the 23 standard values were not given, so the results are less certain.")
    lines += ["", "The models were trained on a limited research dataset (accuracy about 93-94%) and can be wrong."]
    return "\n".join(lines)


def build_prompt(report: dict, messages: list[dict]) -> str:
    system = RULES + "\n\n--- The analysis ---\n" + _context(report)
    out = f"<|im_start|>system\n{system}<|im_end|>\n"
    for m in messages[-MAX_TURNS:]:
        role = "assistant" if m["role"] == "assistant" else "user"
        out += f"<|im_start|>{role}\n{m['content'][:MAX_MESSAGE_CHARS].strip()}<|im_end|>\n"
    return out + "<|im_start|>assistant\n"


def available() -> bool:
    return config.QWEN_MODE == "local" and bool(config.QWEN_API_URL)


async def stream_answer(report: dict, messages: list[dict]):
    """Yields the answer as it is written. Raises httpx errors if Qwen can't be reached."""
    payload = {
        "prompt": build_prompt(report, messages),
        "n_predict": MAX_ANSWER_TOKENS,
        "temperature": 0.3,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "cache_prompt": True,
        "stream": True,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    timeout = httpx.Timeout(config.QWEN_TIMEOUT_SECONDS, connect=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{config.QWEN_API_URL}/completion", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("content"):
                    yield chunk["content"]
                if chunk.get("stop"):
                    break
