"""Resilient LLM layer: retries with backoff, JSON repair and optional provider fallback.

Everything above this module only sees `generate_json(system=..., user=..., label=...) -> dict`
and `LLMError`, so providers can be swapped without touching agent logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterator, Protocol

import httpx

from .config import Settings

log = logging.getLogger("agent.llm")


class LLMError(RuntimeError):
    """The LLM could not produce a usable response (after retries / fallback)."""


@dataclass
class LLMStats:
    calls: int = 0
    retries: int = 0
    json_repairs: int = 0
    fallbacks: int = 0
    latency_ms_total: float = 0.0
    by_label: Counter = field(default_factory=Counter)

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "json_repairs": self.json_repairs,
            "fallbacks": self.fallbacks,
            "latency_ms_total": round(self.latency_ms_total),
            "calls_by_stage": dict(self.by_label),
        }


# Usage counters for the request being served. asyncio tasks inherit the context, so the
# parallel task coroutines of one request all report into the same object.
_scoped_stats: ContextVar[LLMStats | None] = ContextVar("llm_stats", default=None)


def current_stats(default: LLMStats) -> LLMStats:
    return _scoped_stats.get() or default


@contextmanager
def request_stats() -> Iterator[LLMStats]:
    stats = LLMStats()
    token = _scoped_stats.set(stats)
    try:
        yield stats
    finally:
        _scoped_stats.reset(token)


class LLMClient(Protocol):
    async def generate_json(self, *, system: str, user: str, label: str = "llm") -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating fences, prose around it and trailing commas."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty response")
    cleaned = _FENCE.sub("", text.strip()).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        for variant in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                value = json.loads(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("response was not a valid JSON object")


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
Sleep = Callable[[float], Awaitable[None]]


class _ChatClient:
    provider = "base"

    def __init__(
        self,
        settings: Settings,
        *,
        stats: LLMStats | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ):
        self.settings = settings
        self.stats = stats or LLMStats()
        self._transport = transport
        self._sleep = sleep

    @property
    def model(self) -> str:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model}

    async def _chat(self, http: httpx.AsyncClient, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def _http_hint(self, code: int) -> str:
        if code in (401, 403):
            return f"{self.provider} rejected the credentials (HTTP {code}); check the API key."
        if code == 404:
            return f"{self.provider} returned 404: model '{self.model}' or endpoint not found."
        return f"{self.provider} returned HTTP {code}."

    async def generate_json(self, *, system: str, user: str, label: str = "llm") -> dict[str, Any]:
        stats = current_stats(self.stats)
        stats.calls += 1
        stats.by_label[label.split(":")[0] or "unlabelled"] += 1
        messages = [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]
        attempts = max(1, self.settings.llm_max_retries)
        last = "no attempt made"
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds, transport=self._transport) as http:
                for attempt in range(attempts):
                    try:
                        text = await self._chat(http, messages)
                    except httpx.HTTPStatusError as exc:
                        code = exc.response.status_code
                        last = f"HTTP {code}"
                        if code not in RETRYABLE_STATUS:  # auth / bad model name: retrying cannot help
                            raise LLMError(self._http_hint(code)) from exc
                    except httpx.HTTPError as exc:  # timeouts, refused connections, ...
                        last = self._safe_error(f"{type(exc).__name__}: {exc}")
                    else:
                        try:
                            return extract_json(text)
                        except ValueError as exc:
                            # Ask the model to repair its own output instead of failing the whole run.
                            stats.json_repairs += 1
                            last = str(exc)
                            messages = messages + [
                                {"role": "assistant", "content": text[:4000]},
                                {
                                    "role": "user",
                                    "content": "That was not a single valid JSON object. Reply again with ONLY the JSON object, no markdown fences, no commentary.",
                                },
                            ]
                            continue
                    stats.retries += 1
                    if attempt < attempts - 1:
                        await self._sleep(self.settings.llm_backoff_seconds * (2**attempt))
        finally:
            stats.latency_ms_total += (time.perf_counter() - started) * 1000
        raise LLMError(f"{self.provider} ({self.model}) failed after {attempts} attempts: {last}")

    def _safe_error(self, message: str) -> str:
        for secret in (self.settings.openai_api_key, self.settings.gemini_api_key):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message


class OllamaClient(_ChatClient):
    provider = "ollama"

    @property
    def model(self) -> str:
        return self.settings.ollama_model

    async def _chat(self, http: httpx.AsyncClient, messages: list[dict[str, str]]) -> str:
        response = await http.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
            },
        )
        response.raise_for_status()
        try:
            return response.json()["message"]["content"]
        except (KeyError, TypeError, ValueError):
            return ""


class OpenAICompatClient(_ChatClient):
    """Any OpenAI-compatible chat API: Groq, Gemini's OpenAI endpoint, LM Studio, vLLM, OpenAI."""

    provider = "openai-compatible"

    @property
    def model(self) -> str:
        return self.settings.openai_model

    async def _chat(self, http: httpx.AsyncClient, messages: list[dict[str, str]]) -> str:
        is_gemini = "generativelanguage.googleapis.com" in self.settings.openai_base_url.lower()
        api_key = self.settings.gemini_api_key if is_gemini else self.settings.openai_api_key
        key_name = "GEMINI_API_KEY" if is_gemini else "OPENAI_API_KEY"
        if not api_key:
            raise LLMError(f"{key_name} is not set for the configured provider.")
        response = await http.post(
            f"{self.settings.openai_base_url}/chat/completions",
            headers={"Authorization": "Bearer " + api_key},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, ValueError):
            return ""


class FallbackLLM:
    """Try the primary provider; if it is exhausted, try the secondary one."""

    def __init__(self, primary: _ChatClient, secondary: _ChatClient, stats: LLMStats):
        self.primary, self.secondary, self.stats = primary, secondary, stats

    async def generate_json(self, *, system: str, user: str, label: str = "llm") -> dict[str, Any]:
        try:
            return await self.primary.generate_json(system=system, user=user, label=label)
        except LLMError as primary_error:
            current_stats(self.stats).fallbacks += 1
            log.warning("primary LLM failed (%s); trying fallback provider", self.primary._safe_error(str(primary_error)))
            try:
                return await self.secondary.generate_json(system=system, user=user, label=label)
            except LLMError as secondary_error:
                primary_message = self.primary._safe_error(str(primary_error))
                fallback_message = self.secondary._safe_error(str(secondary_error))
                raise LLMError(
                    f"primary failed: {primary_message} | fallback failed: {fallback_message}"
                ) from secondary_error

    def describe(self) -> dict[str, Any]:
        info = self.primary.describe()
        info["fallback"] = self.secondary.describe()
        return info


def _make(kind: str, settings: Settings, stats: LLMStats, transport: httpx.AsyncBaseTransport | None) -> _ChatClient:
    if kind == "ollama":
        return OllamaClient(settings, stats=stats, transport=transport)
    if kind in ("openai", "openai-compatible", "groq", "gemini"):
        return OpenAICompatClient(settings, stats=stats, transport=transport)
    raise ValueError(f"Unknown LLM provider '{kind}'. Use 'ollama' or 'openai'.")


def build_llm(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> LLMClient:
    stats = LLMStats()
    primary = _make(settings.llm_provider, settings, stats, transport)
    fallback = settings.llm_fallback_provider
    if not fallback or fallback == settings.llm_provider:
        return primary
    return FallbackLLM(primary, _make(fallback, settings, stats, transport), stats)
