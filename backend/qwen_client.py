"""Client for the Qwen2.5-7B fine-tuned model, which runs as a separate GPU service.

The backend keeps working when Qwen is not reachable: the Qwen result is then
returned with "available": false and an error message, and XGBoost still answers.
"""

import httpx

import config


def _unavailable(reason: str) -> dict:
    return {"available": False, "model": "Qwen2.5-7B (fine-tuned)", "error": reason}


def _headers() -> dict:
    return {"X-API-Key": config.QWEN_API_KEY} if config.QWEN_API_KEY else {}


async def predict(patient: dict, extra: list | None = None) -> dict:
    if not config.QWEN_API_URL:
        return _unavailable("Qwen service is not configured (QWEN_API_URL is empty).")
    if config.QWEN_MODE == "local":
        import qwen_llamacpp
        return await qwen_llamacpp.predict(patient, extra)
    try:
        async with httpx.AsyncClient(timeout=config.QWEN_TIMEOUT_SECONDS) as client:
            r = await client.post(f"{config.QWEN_API_URL}/predict", json={"patient": patient, "extra": extra or []}, headers=_headers())
        r.raise_for_status()
        data = r.json()
    except httpx.TimeoutException:
        return _unavailable("Qwen service did not respond in time.")
    except httpx.HTTPStatusError as e:
        return _unavailable(f"Qwen service returned an error ({e.response.status_code}).")
    except (httpx.HTTPError, ValueError):
        return _unavailable("Qwen service is not reachable.")

    if "disease" not in data:
        return _unavailable("Qwen service returned an unexpected response.")
    return {"available": True, "model": "Qwen2.5-7B (fine-tuned)", **data}


async def health() -> dict:
    if not config.QWEN_API_URL:
        return {"configured": False, "online": False}
    if config.QWEN_MODE == "local":
        import qwen_llamacpp
        return await qwen_llamacpp.health()
    if config.QWEN_ON_DEMAND:
        # Pinging would start (and bill) the GPU just to show a status dot.
        return {"configured": True, "online": True, "on_demand": True}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{config.QWEN_API_URL}/health", headers=_headers())
        return {"configured": True, "online": r.status_code == 200}
    except httpx.HTTPError:
        return {"configured": True, "online": False}
