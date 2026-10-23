"""Orchestration: validate -> plan -> execute (DAG + tools) -> review -> targeted revision -> Word document."""
from __future__ import annotations

import base64
import logging
import re
import uuid
from typing import Any

from .config import Settings
from .document import build_docx
from .executor import TaskExecutor
from .llm import LLMClient, build_llm, request_stats
from .planner import create_plan
from .reviewer import review
from .schemas import AgentResponse, DocumentInfo, ExecutionStep, QualityReport

log = logging.getLogger("agent")


class RequestRejected(ValueError):
    """The request failed input guardrails (maps to HTTP 422)."""


class AgentExecutionError(RuntimeError):
    """The run could not finish for a reason other than the LLM being unavailable (maps to HTTP 502)."""


def validate_request(text: str, settings: Settings) -> str:
    text = (text or "").strip()
    if len(text) < settings.min_request_chars:
        raise RequestRejected(f"Request is too short to plan from (minimum {settings.min_request_chars} characters). Describe the document you need.")
    if len(text) > settings.max_request_chars:
        raise RequestRejected(f"Request cannot exceed {settings.max_request_chars} characters.")
    letters = sum(c.isalpha() for c in text)
    if letters < 0.5 * len(text.replace(" ", "")) or len(re.findall(r"[A-Za-z]{3,}", text)) < 3:
        raise RequestRejected("Request does not look like a natural-language description of a document.")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # strip control characters


class DocumentAgent:
    def __init__(self, settings: Settings, llm: LLMClient | None = None, tools: Any = None):
        from .tools import ToolRegistry

        self.settings = settings
        self.llm = llm or build_llm(settings)
        self.tools = tools or ToolRegistry()
        self.executor = TaskExecutor(self.llm, self.tools, max_concurrency=settings.max_concurrency, max_attempts=settings.task_max_attempts)

    async def run(self, request_text: str) -> AgentResponse:
        request_text = validate_request(request_text, self.settings)
        with request_stats() as stats:  # per-request LLM usage, also for the parallel task coroutines
            return await self._run(request_text, stats)

    async def _run(self, request_text: str, stats: Any) -> AgentResponse:
        request_id = uuid.uuid4().hex
        log.info("[%s] planning", request_id[:8])
        plan = await create_plan(self.llm, request_text, self.tools)
        log.info("[%s] plan: %d tasks, %d requirements (%s)", request_id[:8], len(plan.tasks), len(plan.requirements), plan.source)

        sections = await self.executor.run(plan, request_text)

        result = await review(self.llm, plan, sections, request_text, round_no=0)
        revised: list[str] = []
        rounds = 0
        while result.feedback and rounds < self.settings.max_revision_rounds:
            rounds += 1
            targets = set(result.feedback)
            log.info("[%s] revision round %d for %s", request_id[:8], rounds, sorted(targets))
            sections = await self.executor.run(plan, request_text, only=targets, feedback=result.feedback, existing=sections)
            revised += sorted(t for t in targets if t not in revised)
            result = await review(self.llm, plan, sections, request_text, round_no=rounds)
        quality: QualityReport = result.report
        quality.revision_rounds, quality.revised_tasks = rounds, revised

        warnings = self._warnings(plan, sections, quality, stats)
        llm_info = {**self.llm.describe(), **stats.snapshot()}
        filename, path = build_docx(
            request_id=request_id, original_request=request_text, plan=plan,
            sections=[sections[t.id] for t in plan.tasks], quality=quality, warnings=warnings, output_dir=self.settings.output_dir,
        )
        data = path.read_bytes()
        steps = [
            ExecutionStep(
                task_id=t.id, title=t.title, status=sections[t.id].status, depends_on=t.depends_on, attempts=sections[t.id].attempts,
                tools_used=sorted({c["tool"] for c in sections[t.id].tool_log if "result" in c}), note=sections[t.id].error,
            )
            for t in plan.tasks
        ]
        return AgentResponse(
            request_id=request_id,
            status="completed_with_warnings" if warnings else "completed",
            summary=self._summary(plan, sections, quality),
            warnings=warnings,
            llm=llm_info,
            plan=plan,
            execution=steps,
            quality_check=quality,
            document=DocumentInfo(
                filename=filename, download_path=f"/documents/{filename}", size_bytes=len(data),
                base64_content=base64.b64encode(data).decode() if self.settings.include_document_base64 else None,
            ),
        )

    @staticmethod
    def _warnings(plan, sections, quality: QualityReport, stats) -> list[str]:
        out: list[str] = []
        if plan.source == "fallback":
            out.append("The model could not produce a valid custom plan; a standard document structure was used.")
        out += [f"Section '{s.title}' failed to generate and needs manual completion." for s in sections.values() if s.status == "failed"]
        out += [f"Requirement {c.requirement_id} not confirmed as met: {c.text}" for c in quality.checks if c.status == "not_met"]
        out += [f"Unresolved content issue in {i}" for i in quality.open_issues]
        if quality.status == "unverified":
            out.append("Requirement coverage could not be verified automatically.")
        if stats.fallbacks:
            out.append("The primary LLM failed and the fallback provider was used.")
        failed_tools = set()
        for section in sections.values():
            succeeded = {c["tool"] for c in section.tool_log if "result" in c}
            failed_tools.update(
                (section.task_id, c["tool"])
                for c in section.tool_log
                if "error" in c and c["tool"] not in succeeded
            )
        out += [f"Tool '{tool}' failed for {tid}; figures from it are marked as to be confirmed." for tid, tool in sorted(failed_tools)]
        return out

    @staticmethod
    def _summary(plan, sections, quality: QualityReport) -> str:
        done = sum(s.status != "failed" for s in sections.values())
        met = sum(c.status == "met" for c in quality.checks)
        tools = sum(1 for s in sections.values() for c in s.tool_log if "result" in c)
        text = f"Created '{plan.title}' ({plan.document_type}): planned {len(plan.tasks)} tasks, {done} sections completed; {met}/{len(plan.requirements)} requirements verified."
        if tools:
            text += f" {tools} deterministic tool call(s) supplied exact figures."
        if quality.revised_tasks:
            text += f" Self-review triggered revision of {', '.join(quality.revised_tasks)}."
        return text
