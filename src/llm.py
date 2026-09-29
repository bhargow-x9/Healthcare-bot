"""Thin client for Llama models served by Ollama (local) or Groq (hosted).

Both backends support text, JSON mode and images, so the rest of the app
does not care which one is configured.
"""
import base64
import json
import logging
import re
import time

import requests

from . import config

log = logging.getLogger(__name__)
GROQ_MAX_RETRIES = 6


class LLMError(RuntimeError):
    """The model backend is unreachable, misconfigured or returned an error."""


def complete(messages, *, model=None, json_mode=False, temperature=0.1, images=None) -> str:
    """Send chat messages and return the reply text.

    `images` (list of JPEG bytes) are attached to the last message.
    """
    model = model or config.LLM_MODEL
    if config.LLM_PROVIDER == "groq":
        return _groq(messages, model, json_mode, temperature, images)
    if config.LLM_PROVIDER == "ollama":
        return _ollama(messages, model, json_mode, temperature, images)
    raise LLMError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}' (use 'ollama' or 'groq').")


def complete_json(system: str, user: str, *, model=None, images=None, temperature=0.1) -> dict:
    """Like `complete`, but parses the reply as a JSON object (one retry on bad JSON)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    reply = complete(messages, model=model, json_mode=True, temperature=temperature, images=images)
    parsed = _parse_json(reply)
    if parsed is None:
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
        ]
        parsed = _parse_json(complete(messages, model=model, json_mode=True, temperature=0))
    if parsed is None:
        raise ValueError("Model did not return valid JSON")
    return parsed


def _parse_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _ollama(messages, model, json_mode, temperature, images) -> str:
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["images"] = [base64.b64encode(img).decode() for img in images]
    payload = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": config.LLM_NUM_CTX},
    }
    if json_mode:
        payload["format"] = "json"
    try:
        resp = requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=config.LLM_TIMEOUT)
    except requests.ConnectionError as exc:
        raise LLMError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Install Ollama and make sure it is running."
        ) from exc
    except requests.Timeout as exc:
        raise LLMError(f"Model '{model}' took longer than {config.LLM_TIMEOUT}s to answer.") from exc
    if resp.status_code == 404:
        raise LLMError(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}")
    if not resp.ok:
        raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("message", {}).get("content", "")


def _groq(messages, model, json_mode, temperature, images) -> str:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. Add it to .env or switch LLM_PROVIDER to 'ollama'.")
    msgs = [dict(m) for m in messages]
    if images:
        parts = [{"type": "text", "text": msgs[-1]["content"]}]
        parts += [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}}
            for img in images
        ]
        msgs[-1]["content"] = parts
    payload = {"model": model, "messages": msgs, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"  # fewer tokens against free-tier limits
    for attempt in range(GROQ_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{config.GROQ_BASE_URL}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                timeout=config.LLM_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Cannot reach Groq: {exc}") from exc
        if resp.status_code != 429 or attempt == GROQ_MAX_RETRIES:
            break
        delay = _retry_delay(resp)
        log.info("Groq rate limit hit, waiting %.1fs (attempt %d)", delay, attempt + 1)
        time.sleep(delay)
    if resp.status_code == 429:
        raise LLMError("Groq rate limit reached (free plan). Please wait a minute and try again.")
    if not resp.ok:
        raise LLMError(f"Groq error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def _retry_delay(resp) -> float:
    """Seconds to wait after a 429, from the Retry-After header or 'try again in 1m2.5s'."""
    try:
        return min(float(resp.headers["retry-after"]) + 0.5, 60)
    except (KeyError, ValueError):
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", resp.text)
    if not m:
        return 10
    seconds = int(m.group(1) or 0) * 60 + float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    return min(seconds + 0.5, 60)


def health() -> dict:
    """Report whether the backend is reachable and the configured models exist."""
    info = {
        "provider": config.LLM_PROVIDER,
        "models": {"chat": config.LLM_MODEL, "medical": config.MEDICAL_MODEL, "vision": config.VISION_MODEL},
        "ok": False,
        "message": "",
    }
    if config.LLM_PROVIDER == "groq":
        info["ok"] = bool(config.GROQ_API_KEY)
        info["message"] = "Groq API key set" if info["ok"] else "GROQ_API_KEY missing"
        return info
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        installed = {m["name"] for m in resp.json().get("models", [])}
    except (requests.RequestException, ValueError):
        info["message"] = f"Ollama not reachable at {config.OLLAMA_BASE_URL}"
        return info

    def present(name):
        return name in installed or f"{name}:latest" in installed

    missing = sorted({m for m in info["models"].values() if not present(m)})
    info["ok"] = not missing
    info["message"] = "All models installed" if not missing else "Missing: " + ", ".join(
        f"ollama pull {m}" for m in missing
    )
    return info
