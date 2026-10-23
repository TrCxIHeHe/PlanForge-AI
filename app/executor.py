"""Dependency-aware task execution.

Tasks whose dependencies are finished run concurrently (bounded by a semaphore). A task that
fails after its retries becomes a clearly marked placeholder section; it never aborts the run,
and its dependants are told its output is unavailable.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .blocks import ContentError, parse_section_payload
from .llm import LLMClient, LLMError
from .prompts import WRITER_SYSTEM, writer_user
from .schemas import AgentPlan, Block, PlanTask, Section
from .tools import ToolRegistry

log = logging.getLogger("agent.executor")
MAX_TOOL_REQUESTS = 3
MAX_TOOL_ROUNDS = 2


class TaskExecutor:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, *, max_concurrency: int = 3, max_attempts: int = 2):
        self.llm, self.tools = llm, tools
        self.max_concurrency, self.max_attempts = max(1, max_concurrency), max(1, max_attempts)

    async def run(
        self,
        plan: AgentPlan,
        request_text: str,
        *,
        only: set[str] | None = None,
        feedback: dict[str, str] | None = None,
        existing: dict[str, Section] | None = None,
    ) -> dict[str, Section]:
        """Execute `only` (default: all) tasks; `existing` supplies finished dependencies for revisions."""
        results: dict[str, Section] = dict(existing or {})
        todo = [t for t in plan.tasks if only is None or t.id in only]
        events = {t.id: asyncio.Event() for t in todo}
        gate = asyncio.Semaphore(self.max_concurrency)
        feedback = feedback or {}

        async def worker(task: PlanTask) -> None:
            try:
                await asyncio.gather(*(events[d].wait() for d in task.depends_on if d in events))
                async with gate:
                    previous = (existing or {}).get(task.id)
                    section = await self._run_task(plan, task, request_text, results, feedback.get(task.id))
                    if previous is not None and section.status == "failed":
                        section = previous  # keep the earlier good draft rather than regress to a placeholder
                    elif previous is not None:
                        section.status = "revised"
                    results[task.id] = section
            except Exception as exc:  # last-resort isolation: a bug in one task must not kill siblings
                log.exception("task %s crashed", task.id)
                results[task.id] = self._failed(task, f"unexpected error: {type(exc).__name__}", 0)
            finally:
                events[task.id].set()

        await asyncio.gather(*(worker(t) for t in todo))
        return results

    # ------------------------------------------------------------------
    async def _run_task(self, plan: AgentPlan, task: PlanTask, request_text: str, results: dict[str, Section], feedback: str | None) -> Section:
        tool_log = [self._call(c.tool, c.args) for c in task.tools]
        req_texts = [r.text for r in plan.requirements if r.id in task.satisfies]
        last_error = "no attempt made"
        attempts, tool_rounds = 0, 0
        while attempts < self.max_attempts:
            prompt = writer_user(
                request_text=request_text, plan=plan, task=task, requirement_texts=req_texts,
                dependencies=[results.get(d) for d in task.depends_on], dependency_ids=task.depends_on,
                tool_log=tool_log, feedback=feedback,
            )
            try:
                raw = await self.llm.generate_json(system=WRITER_SYSTEM, user=prompt, label=f"task:{task.id}")
                requests = raw.get("tool_requests")
                if isinstance(requests, list) and requests and not raw.get("blocks") and tool_rounds < MAX_TOOL_ROUNDS:
                    # Retry once when the writer can correct invalid tool arguments.
                    tool_rounds += 1
                    for call in requests[:MAX_TOOL_REQUESTS]:
                        if isinstance(call, dict):
                            tool_log.append(self._call(str(call.get("tool")), call.get("args")))
                    continue
                attempts += 1
                blocks, key_points = parse_section_payload(raw)
                return Section(task_id=task.id, title=task.section_title, blocks=blocks, key_points=key_points, attempts=attempts, tool_log=tool_log)
            except ContentError as exc:
                last_error = str(exc)  # attempts already counted above
                feedback = f"Your previous answer was unusable ({exc}). Return the JSON with substantive blocks."
            except LLMError as exc:
                attempts += 1
                last_error = str(exc)
            log.warning("task %s attempt %d failed: %s", task.id, attempts, last_error)
        return self._failed(task, last_error, attempts, tool_log)

    def _call(self, name: str, args: Any) -> dict[str, Any]:
        outcome = self.tools.run(name, args)
        entry: dict[str, Any] = {"tool": name, "args": args}
        entry.update({"result": outcome.result} if outcome.ok else {"error": outcome.error})
        return entry

    @staticmethod
    def _failed(task: PlanTask, reason: str, attempts: int, tool_log: list[dict[str, Any]] | None = None) -> Section:
        note = f"This section could not be generated automatically ({task.objective[:120]}). It needs manual completion."
        return Section(
            task_id=task.id, title=task.section_title, blocks=[Block(type="paragraph", text=note)],
            status="failed", attempts=attempts, tool_log=tool_log or [], error=reason[:300],
        )
