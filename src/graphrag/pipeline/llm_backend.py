"""Chat completions: Groq (free tier), local Ollama, or optional Gemini."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from config import settings
logger = logging.getLogger(__name__)

def _groq_complete(system: str, user: str, json_mode: bool) -> str:
    """OpenAI-compatible chat against Groq's free API."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install openai for Groq: pip install openai") from exc
    key = (settings.groq_api_key or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    kwargs: dict[str, Any] = {
        "model": settings.groq_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        if json_mode:
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return (resp.choices[0].message.content or "").strip()

def _ollama_complete(system: str, user: str, json_mode: bool) -> str:
    """Chat against a local Ollama server (/api/chat)."""
    base = settings.ollama_base_url.rstrip("/")
    body: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if json_mode:
        body["format"] = "json"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    return (out.get("message") or {}).get("content", "").strip()

def _gemini_complete(system: str, user: str, json_mode: bool) -> str:
    """Legacy Gemini path when LLM_PROVIDER=gemini."""
    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-generativeai required for Gemini LLM") from exc
    if not (settings.google_api_key or "").strip():
        raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
    genai.configure(api_key=settings.google_api_key)
    if json_mode:
        gen_cfg = genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        )
    else:
        gen_cfg = genai.GenerationConfig(temperature=0.2)
    model = genai.GenerativeModel(
        settings.gemini_model,
        system_instruction=system,
        generation_config=gen_cfg,
    )
    resp = model.generate_content(user)
    return (resp.text or "").strip()

def complete_chat(system: str, user: str, *, json_mode: bool = False) -> str:
    """
    Run one chat completion for the configured LLM_PROVIDER.

    json_mode: ask for JSON object output (Groq/Ollama/Gemini where supported).
    """
    prov = (settings.llm_provider or "groq").lower().strip()
    try:
        if prov == "groq":
            return _groq_complete(system, user, json_mode)
        if prov == "ollama":
            return _ollama_complete(system, user, json_mode)
        if prov == "gemini":
            return _gemini_complete(system, user, json_mode)
        raise RuntimeError(f"Unknown LLM_PROVIDER={prov!r} (use groq, ollama, gemini)")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.exception("LLM backend error: %s", exc)
        raise RuntimeError(f"LLM request failed: {exc}") from exc
