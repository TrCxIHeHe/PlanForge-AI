"""A scriptable fake LLM so the whole agent can be tested offline and deterministically."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from app.config import Settings
from app.llm import LLMError, LLMStats, current_stats

GOOD_PLAN = {
    "title": "Support Response Proposal",
    "document_type": "proposal",
    "purpose": "Reduce support response time within 90 days.",
    "assumptions": ["Team of 6 agents (illustrative)"],
    "requirements": ["Compare at least two approaches", "Include a costed budget", "List risks", "Provide a timeline"],
    "tasks": [
        {"id": "T1", "title": "Executive summary", "objective": "Summarise", "section_title": "Executive Summary", "depends_on": ["T2", "T3", "T4"], "satisfies": [1]},
        {"id": "T2", "title": "Options and budget", "objective": "Compare options with costs", "section_title": "Options and Budget", "satisfies": [1, 2],
         "tools": [{"tool": "calculate", "args": {"expression": "4*2,500 + 12%*8000"}}]},
        {"id": "T3", "title": "Risks", "objective": "Risk register", "section_title": "Risks", "satisfies": [3],
         "tools": [{"tool": "risk_score", "args": {"risks": [{"name": "Churn", "likelihood": 2, "impact": 5}]}}]},
        {"id": "T4", "title": "Timeline", "objective": "Phased timeline", "section_title": "Timeline", "satisfies": [4],
         "tools": [{"tool": "schedule", "args": {"phases": [{"name": "Pilot", "weeks": 4}, {"name": "Rollout", "weeks": 8}], "start": "2026-11-02"}}]},
    ],
}

GOOD_SECTION = {
    "blocks": [
        {"type": "paragraph", "text": "This section explains the approach in enough detail to be useful to a reader and satisfy validation."},
        {"type": "table", "caption": "Options", "headers": ["Option", "Cost"], "rows": [["A", "10,960"], ["B", "8,000"]]},
        {"type": "bullets", "items": ["First point", "Second point"]},
    ],
    "key_points": ["Option A costs 10,960 (illustrative)"],
}


class FakeLLM:
    """handlers: label-prefix -> dict | Exception | callable(label, system, user, n) -> dict | Exception."""

    def __init__(self, handlers: dict[str, Any] | None = None, delay: float = 0.0):
        self.handlers = {"plan": GOOD_PLAN, "task": GOOD_SECTION, "review": None, **(handlers or {})}
        self.delay = delay
        self.calls: list[tuple[str, str]] = []  # (label, user prompt)
        self.counts: dict[str, int] = {}
        self.stats = LLMStats()
        self.running = self.peak = 0

    def describe(self) -> dict[str, Any]:
        return {"provider": "fake", "model": "fake-model"}

    async def generate_json(self, *, system: str, user: str, label: str = "llm") -> dict[str, Any]:
        stats = current_stats(self.stats)
        stats.calls += 1
        stats.by_label[label.split(":")[0]] += 1
        self.calls.append((label, user))
        key = label.split(":")[0]
        n = self.counts[label] = self.counts.get(label, 0) + 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            handler = self.handlers.get(label, self.handlers.get(key))
            if key == "review" and handler is None:  # default reviewer: everything met
                return {"checks": [{"id": f"R{i}", "met": True, "issue": ""} for i in range(1, 9)], "summary": "All requirements covered."}
            out = handler(label, system, user, n) if callable(handler) else handler
            if isinstance(out, Exception):
                raise out
            return out
        finally:
            self.running -= 1


def make_settings(tmp_path, **kw) -> Settings:
    base = dict(output_dir=tmp_path, llm_backoff_seconds=0.0, task_max_attempts=2, max_revision_rounds=1, include_document_base64=True)
    base.update(kw)
    return Settings(**base)


__all__ = ["FakeLLM", "GOOD_PLAN", "GOOD_SECTION", "make_settings", "LLMError"]
