"""Check the configured Gemini model and its Ollama fallback without printing credentials.

If the preferred model is unavailable to the configured account, the script selects a
generateContent-capable model returned by Gemini's authenticated model metadata endpoint.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import re
import sys
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.llm import LLMError, build_llm  # noqa: E402


def _gemini_api_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/openai"):
        root = root[: -len("/openai")]
    return root


def available_text_models(models: list[dict[str, Any]]) -> list[str]:
    """Return account-listed text models that support generateContent."""
    available: list[str] = []
    for item in models:
        name = str(item.get("name") or "")
        methods = item.get("supportedGenerationMethods")
        if any(kind in name.lower() for kind in ("tts", "live", "image", "transcribe")):
            continue
        if name.startswith("models/") and isinstance(methods, list) and "generateContent" in methods:
            available.append(name.removeprefix("models/"))
        elif name and isinstance(methods, list) and "generateContent" in methods:
            available.append(name)

    stable_flash = [
        name for name in available
        if "flash" in name.lower()
        and "preview" not in name.lower()
        and "experimental" not in name.lower()
    ]
    preview_flash = [name for name in available if "flash" in name.lower()]
    stable = [name for name in available if "preview" not in name.lower() and "experimental" not in name.lower()]
    return stable_flash or preview_flash or stable or available


def select_text_model(models: list[dict[str, Any]], preferred: str) -> str:
    """Choose the preferred accessible text model, or an API-listed text model."""
    available = available_text_models(models)
    preferred = preferred.removeprefix("models/")
    if preferred in available:
        return preferred
    if not available:
        raise RuntimeError("Gemini account exposes no model supporting generateContent.")
    return available[0]


def list_models(settings: Settings) -> list[dict[str, Any]]:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is missing; add a fresh Gemini key to .env locally.")
    try:
        response = httpx.get(
            f"{_gemini_api_root(settings.openai_base_url)}/models",
            headers={"x-goog-api-key": settings.gemini_api_key},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Gemini model metadata request failed (HTTP {exc.response.status_code}).") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Could not retrieve Gemini model metadata ({type(exc).__name__}).") from exc

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("Gemini model metadata response did not contain a models list.")
    return [model for model in models if isinstance(model, dict)]


def save_model_to_env(env_path: Path, model: str) -> None:
    """Persist only the selected model setting; preserve any local API key untouched."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    setting = f"OPENAI_MODEL={model}"
    for index, line in enumerate(lines):
        if re.match(r"^\s*OPENAI_MODEL\s*=", line):
            lines[index] = setting
            break
    else:
        lines.append(setting)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _probe_gemini(settings: Settings) -> str | None:
    primary_settings = replace(settings, llm_fallback_provider="")
    primary = build_llm(primary_settings)
    try:
        await primary.generate_json(
            system="You are checking model connectivity. Return a single JSON object.",
            user='Return exactly {"ready": true}.',
            label="connectivity",
        )
        return None
    except LLMError as primary_error:
        return str(primary_error)


async def _probe_fallback(settings: Settings) -> str | None:
    if not settings.llm_fallback_provider:
        return "no fallback provider is configured"
    fallback_settings = replace(settings, llm_provider=settings.llm_fallback_provider, llm_fallback_provider="")
    fallback = build_llm(fallback_settings)
    try:
        await fallback.generate_json(
            system="You are checking model connectivity. Return a single JSON object.",
            user='Return exactly {"ready": true}.',
            label="connectivity:fallback",
        )
        return None
    except LLMError as exc:
        return str(exc)


async def check_provider(settings: Settings) -> None:
    primary_error = await _probe_gemini(settings)
    if primary_error is None:
        print(f"Gemini reachable: {settings.openai_model}")
        return
    fallback_error = await _probe_fallback(settings)
    if fallback_error is not None:
        raise RuntimeError(
            f"Neither configured provider is available. Gemini: {primary_error}; "
            f"{settings.llm_fallback_provider}: {fallback_error}"
        )
    print(f"Gemini unavailable: {primary_error}")
    print(f"Fallback reachable: {settings.llm_fallback_provider} ({settings.ollama_model})")


async def main() -> int:
    settings = Settings.from_env()
    if settings.llm_provider not in {"openai", "openai-compatible", "gemini"}:
        raise RuntimeError("Set LLM_PROVIDER=openai and configure the Gemini endpoint in .env first.")
    if not settings.gemini_api_key:
        print("Gemini account access cannot be verified without GEMINI_API_KEY in the local .env.", file=sys.stderr)
        await check_provider(settings)
        return 0

    try:
        models = list_models(settings)
    except RuntimeError as metadata_error:
        print(f"Gemini model metadata check failed: {metadata_error}", file=sys.stderr)
        await check_provider(settings)
        return 0
    candidates = available_text_models(models)
    if not candidates:
        no_text_model = "Gemini account exposes no model supporting generateContent."
        fallback_error = await _probe_fallback(settings)
        if fallback_error is not None:
            raise RuntimeError(
                f"Neither configured provider is available. Gemini: {no_text_model}; "
                f"{settings.llm_fallback_provider}: {fallback_error}"
            )
        print(f"Gemini unavailable: {no_text_model}")
        print(f"Fallback reachable: {settings.llm_fallback_provider} ({settings.ollama_model})")
        return 0
    selected = select_text_model(models, settings.openai_model)
    ordered_models = [selected, *[model for model in candidates if model != selected]]
    last_error = "no Gemini model was tested"
    for model in ordered_models:
        candidate = replace(settings, openai_model=model)
        last_error = await _probe_gemini(candidate) or ""
        if not last_error:
            if model != settings.openai_model:
                save_model_to_env(ROOT / ".env", model)
                print(f"Configured model was unavailable; selected account-accessible model: {model}")
                print("Restart the API after this check so it loads the updated .env model.")
            print(f"Gemini reachable: {model}")
            return 0
        print(f"Gemini model {model} unavailable: {last_error}")

    fallback_error = await _probe_fallback(settings)
    if fallback_error is not None:
        raise RuntimeError(
            f"Neither configured provider is available. Gemini models: {last_error}; "
            f"{settings.llm_fallback_provider}: {fallback_error}"
        )
    print(f"Fallback reachable: {settings.llm_fallback_provider} ({settings.ollama_model})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except RuntimeError as exc:
        print(f"Provider check failed: {exc}", file=sys.stderr)
        sys.exit(1)
