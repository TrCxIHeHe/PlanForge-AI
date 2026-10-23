"""Planning: LLM-generated plan, normalised into a valid requirements-traced task DAG.

Model output is never trusted. `normalise_plan` repairs what it can (ids, dependencies, cycles,
unknown tools, uncovered requirements) and records every repair. If the model's plan is unusable
after one self-repair round, a deterministic fallback plan keeps the run going.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .blocks import clean_text
from .llm import LLMClient
from .prompts import planner_system, planner_user
from .schemas import AgentPlan, PlanTask, Requirement, ToolCall
from .tools import ToolRegistry

log = logging.getLogger("agent.planner")
MAX_TASKS, MAX_REQUIREMENTS = 8, 8


class PlanError(ValueError):
    """The model's plan cannot be repaired into a valid one."""


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())}


def _reaches(deps: dict[str, list[str]], start: str, target: str) -> bool:
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(deps.get(node, []))
    return False


def _req_texts(raw: Any) -> list[str]:
    out = []
    for item in raw if isinstance(raw, list) else []:
        text = item.get("text") or item.get("requirement") or "" if isinstance(item, dict) else item
        text = clean_text(text)
        if len(text) >= 5:
            out.append(text[:300])
    return out[:MAX_REQUIREMENTS]


def normalise_plan(raw: dict[str, Any], request_text: str, tools: ToolRegistry) -> AgentPlan:
    repairs: list[str] = []
    raw_tasks = [t for t in raw.get("tasks") or [] if isinstance(t, dict)][:MAX_TASKS]
    if len(raw_tasks) < 2:
        raise PlanError("the plan needs at least 2 tasks (one section each)")

    req_texts = _req_texts(raw.get("requirements"))
    if not req_texts:
        req_texts = [f"Fulfil the request: {clean_text(request_text)[:200]}"]
        repairs.append("no requirements given; derived one from the request")
    requirements = [Requirement(id=f"R{i}", text=t) for i, t in enumerate(req_texts, 1)]
    req_ids = {r.id for r in requirements}

    # Re-number tasks T1..Tn; remember how the model referred to each one.
    alias: dict[str, str] = {}
    for i, t in enumerate(raw_tasks, 1):
        new = f"T{i}"
        alias[new.lower()] = new
        alias[str(t.get("id", "")).strip().lower()] = alias.get(str(t.get("id", "")).strip().lower(), new)
        alias[str(i)] = new

    tasks: list[PlanTask] = []
    graph: dict[str, list[str]] = {}
    for i, t in enumerate(raw_tasks, 1):
        tid = f"T{i}"
        title = clean_text(t.get("title") or t.get("description") or t.get("section_title"))[:140]
        section = clean_text(t.get("section_title") or title)[:100]
        if not title or not section:
            raise PlanError(f"task {i} has no title or section_title")
        objective = clean_text(t.get("objective") or title)[:500]

        deps: list[str] = []
        for d in t.get("depends_on") or []:
            target = alias.get(str(d).strip().lower())
            if not target or target == tid or target in deps:
                repairs.append(f"{tid}: dropped invalid dependency '{d}'")
                continue
            if _reaches(graph, target, tid):  # adding tid -> target would close a cycle
                repairs.append(f"{tid}: dropped dependency {target} (would create a cycle)")
                continue
            deps.append(target)
        graph[tid] = deps

        satisfies: list[str] = []
        for r in t.get("satisfies") or []:
            rid = r if isinstance(r, str) and r.upper().startswith("R") else f"R{r}"
            rid = str(rid).upper()
            if rid in req_ids and rid not in satisfies:
                satisfies.append(rid)

        calls: list[ToolCall] = []
        for call in t.get("tools") or []:
            name = call.get("tool") if isinstance(call, dict) else None
            if name in tools.names and isinstance(call.get("args", {}), dict):
                calls.append(ToolCall(tool=name, args=call.get("args", {})))
            else:
                repairs.append(f"{tid}: dropped unknown tool call {name!r}")
        tasks.append(PlanTask(id=tid, title=title, objective=objective, section_title=section, depends_on=deps, satisfies=satisfies, tools=calls[:3]))

    # Traceability: every requirement must be owned by at least one task.
    covered = {rid for task in tasks for rid in task.satisfies}
    for req in requirements:
        if req.id in covered:
            continue
        rw = _words(req.text)
        best = max(tasks, key=lambda t: (len(rw & _words(f"{t.title} {t.objective} {t.section_title}")), t.id != tasks[0].id))
        best.satisfies.append(req.id)
        repairs.append(f"{req.id} was not assigned to any task; attached to {best.id}")

    assumptions = [clean_text(a)[:300] for a in raw.get("assumptions") or [] if clean_text(a)][:10]
    return AgentPlan(
        title=clean_text(raw.get("title"))[:120] or "Business Document",
        document_type=clean_text(raw.get("document_type"))[:80] or "business document",
        purpose=clean_text(raw.get("purpose"))[:600] or clean_text(request_text)[:300],
        assumptions=assumptions,
        requirements=requirements,
        tasks=tasks,
        source="repaired" if repairs else "llm",
        repairs=repairs,
    )


def fallback_plan(request_text: str) -> AgentPlan:
    """Deterministic plan used only when the model cannot produce a valid one."""
    brief = clean_text(request_text)
    reqs = [
        f"Address the request: {brief[:160]}",
        "State assumptions explicitly and label illustrative figures",
        "Describe the approach, options considered and a recommendation",
        "Give a phased plan with timeline and owners",
        "Identify key risks with mitigations",
        "Define measurable success criteria",
    ]

    def task(i: int, title: str, objective: str, deps: list[str], sat: list[str]) -> PlanTask:
        return PlanTask(id=f"T{i}", title=title, objective=objective, section_title=title, depends_on=deps, satisfies=sat)

    tasks = [
        task(1, "Executive Summary", "Summarise purpose, recommendation and next steps in plain language", ["T2", "T3", "T4", "T5", "T6"], ["R1"]),
        task(2, "Background and Assumptions", "Context for the request and every assumption made, flagged as illustrative where invented", [], ["R2"]),
        task(3, "Approach and Options", "Compare realistic options and recommend one with reasons", [], ["R3"]),
        task(4, "Implementation Plan", "Phases, timeline and owners as a table", [], ["R4"]),
        task(5, "Risks and Mitigations", "Key risks with likelihood, impact and mitigation", [], ["R5"]),
        task(6, "Success Metrics", "Measurable criteria with targets and how they are tracked", [], ["R6"]),
    ]
    return AgentPlan(
        title=(brief[:70].rstrip(" .,;:") or "Business Document"),
        document_type="business document",
        purpose=brief[:300] or "Business document",
        assumptions=["The model could not produce a valid custom plan, so a standard business-document structure was used."],
        requirements=[Requirement(id=f"R{i}", text=t) for i, t in enumerate(reqs, 1)],
        tasks=tasks,
        source="fallback",
    )


async def create_plan(llm: LLMClient, request_text: str, tools: ToolRegistry) -> AgentPlan:
    """Plan -> validate -> (feed problems back once) -> fallback. LLMError propagates: no model, no document."""
    system = planner_system(tools.describe())
    problems: list[str] | None = None
    for attempt in (1, 2):
        raw = await llm.generate_json(system=system, user=planner_user(request_text, problems), label=f"plan:{attempt}")
        try:
            plan = normalise_plan(raw, request_text, tools)
            if attempt == 2 and plan.source == "llm":
                plan.source = "repaired"
                plan.repairs.append("plan was regenerated after validation feedback")
            return plan
        except (PlanError, ValueError) as exc:
            problems = [str(exc)]
            log.warning("plan attempt %d rejected: %s", attempt, exc)
    log.warning("using fallback plan")
    return fallback_plan(request_text)
