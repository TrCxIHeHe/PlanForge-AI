"""Reflection: check the draft against the plan's requirements, then drive targeted revision.

The LLM verdict is combined with deterministic checks (failed sections, placeholder text).
Only sections tied to unmet requirements are rewritten, which is cheaper and safer than
regenerating the whole document.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from .blocks import placeholder_hits
from .llm import LLMClient, LLMError
from .prompts import REVIEWER_SYSTEM, reviewer_user
from .schemas import AgentPlan, QualityReport, RequirementCheck, Section

log = logging.getLogger("agent.reviewer")


@dataclass
class ReviewResult:
    report: QualityReport
    feedback: dict[str, str] = field(default_factory=dict)  # task id -> what to fix


def deterministic_feedback(sections: dict[str, Section]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tid, s in sections.items():
        if s.status == "failed":
            out[tid] = "The previous attempt failed to produce this section; write it in full."
        elif hits := placeholder_hits(s.blocks):
            out[tid] = f"Remove placeholder text ({', '.join(hits)}) and write real content."
        else:
            failed_tools = {
                c["tool"] for c in s.tool_log if "error" in c
            } - {c["tool"] for c in s.tool_log if "result" in c}
            if failed_tools:
                out[tid] = f"Required tool call(s) failed: {', '.join(sorted(failed_tools))}. Correct the tool arguments, use successful results, and revise this section."
    return out


def _section_text(section: Section) -> str:
    return " ".join(
        [b.text for b in section.blocks]
        + [item for b in section.blocks for item in b.items]
        + [cell for b in section.blocks for row in b.rows for cell in row]
    )


def _unrecovered_tools(section: Section) -> set[str]:
    attempted = {str(c.get("tool")) for c in section.tool_log if "error" in c}
    succeeded = {str(c.get("tool")) for c in section.tool_log if "result" in c}
    return attempted - succeeded


def _contains_past_calendar_date(text: str) -> bool:
    matches = re.findall(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if not matches:
        return True
    today = date.today()
    for value in matches:
        try:
            parsed = date.fromisoformat(value) if "-" in value else datetime.strptime(value, "%d %b %Y").date()
        except ValueError:
            try:
                parsed = datetime.strptime(value, "%d %B %Y").date()
            except ValueError:
                return True
        if parsed < today:
            return True
    return False


def _unsupported_verified_facts(section: Section, request_text: str) -> list[str]:
    request_numbers = {
        _normalise_number(value)
        for value in re.findall(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?", request_text)
    }
    claims = [block.text for block in section.blocks if block.type == "paragraph"]
    claims.extend(
        " | ".join(row)
        for block in section.blocks
        if block.type == "table"
        for row in block.rows
    )
    unsupported: set[str] = set()
    absent_data_claim = re.compile(
        r"\bno verified (?:operational )?(?:metrics|data|records|facts?)\b.{0,100}\b(?:exist|available|present)\b",
        re.IGNORECASE,
    )
    request_has_absence = bool(absent_data_claim.search(request_text))
    for claim in claims:
        if absent_data_claim.search(claim) and not request_has_absence:
            unsupported.add("unsupported claim that operational data or records do not exist")
        if not re.search(r"\bverified\s+facts?\b|\bfact\s*:\s*verified\b", claim, re.IGNORECASE):
            continue
        claim_numbers = re.findall(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?", claim)
        unsupported.update(
            value
            for value in claim_numbers
            if _normalise_number(value) not in request_numbers
        )
    return sorted(unsupported)


def _normalise_number(value: str) -> str:
    normalized = value.replace("$", "").replace(",", "").replace("%", "")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


async def review(llm: LLMClient, plan: AgentPlan, sections: dict[str, Section], request_text: str, *, round_no: int = 0) -> ReviewResult:
    ordered = [sections[t.id] for t in plan.tasks if t.id in sections]
    feedback = deterministic_feedback(sections)
    hard_failures = dict(feedback)  # known-bad content: a positive LLM review must never override these
    verdict: dict[str, tuple[bool, str]] = {}
    summary = ""
    reviewer_ok = True
    try:
        raw = await llm.generate_json(system=REVIEWER_SYSTEM, user=reviewer_user(request_text, plan, ordered), label=f"review:{round_no}")
        for item in raw.get("checks") or []:
            if isinstance(item, dict) and "met" in item:
                rid = str(item.get("id") or item.get("requirement_id") or "").strip().upper()
                rid = rid if rid.startswith("R") else f"R{rid}"
                verdict[rid] = (bool(item["met"]) if not isinstance(item["met"], str) else item["met"].lower() == "true", str(item.get("issue") or "")[:300])
        summary = str(raw.get("summary") or "")[:400]
    except LLMError as exc:
        reviewer_ok = False
        summary = "Automated review was unavailable, so requirement coverage is unverified."
        log.warning("reviewer unavailable: %s", exc)

    failed_tasks = {tid for tid, s in sections.items() if s.status == "failed"}
    deterministic_issues: dict[str, str] = {}
    for tid, section in sections.items():
        if failed := _unrecovered_tools(section):
            message = f"required tool call(s) failed: {', '.join(sorted(failed))}"
            deterministic_issues[tid] = message
            hard_failures[tid] = message
        if unverified := _unsupported_verified_facts(section, request_text):
            message = f"unsupported fact claim(s): {', '.join(unverified)}"
            deterministic_issues[tid] = message
            feedback[tid] = f"Remove unsupported fact claims ({', '.join(unverified)}); clearly label assumptions as illustrative and say missing data was not provided."
            hard_failures[tid] = message
    for req in plan.requirements:
        if not re.search(r"\b(?:future|calendar dates?|real dates?|specific dates?)\b", req.text, re.IGNORECASE):
            continue
        schedule_owners = [
            task.id
            for task in plan.tasks
            if req.id in task.satisfies
            and re.search(
                r"\b(?:schedule|timeline|rollout|calendar)\b",
                f"{task.title} {task.section_title} {task.objective}",
                re.IGNORECASE,
            )
        ]
        for owner in schedule_owners:
            section = sections.get(owner)
            if section and _contains_past_calendar_date(_section_text(section)):
                message = "the phased schedule does not contain valid future calendar dates"
                deterministic_issues[owner] = message
                feedback[owner] = f"Requirement {req.id} is not met: {message}. Use the successful schedule tool output or request the schedule tool with valid phase arguments."
                hard_failures[owner] = message
    checks: list[RequirementCheck] = []
    for req in plan.requirements:
        owners = [t.id for t in plan.tasks if req.id in t.satisfies]
        forced_issue = next((deterministic_issues[o] for o in owners if o in deterministic_issues), "")
        if req.id in verdict:
            met, issue = verdict[req.id]
            if forced_issue:
                met, issue = False, forced_issue
            if met and owners and all(o in failed_tasks for o in owners):
                met, issue = False, "the section meant to cover this failed to generate"
            status = "met" if met else "not_met"
            if not met:
                for owner in owners:
                    feedback.setdefault(owner, f"Requirement not met: {req.text}. Reviewer note: {issue or 'not covered'}")
            checks.append(RequirementCheck(requirement_id=req.id, text=req.text, status=status, note=issue))
        elif owners and all(o in failed_tasks for o in owners):
            checks.append(RequirementCheck(requirement_id=req.id, text=req.text, status="not_met", note="section failed to generate"))
        elif forced_issue:
            for owner in owners:
                feedback.setdefault(owner, f"Requirement not met: {req.text}. {forced_issue}")
            checks.append(RequirementCheck(requirement_id=req.id, text=req.text, status="not_met", note=forced_issue))
        else:  # reviewer skipped it or was unavailable: say so rather than guess either way
            checks.append(RequirementCheck(requirement_id=req.id, text=req.text, status="unverified", note="" if reviewer_ok else "reviewer unavailable"))

    if any(c.status == "not_met" for c in checks) or failed_tasks or hard_failures:
        status = "needs_attention"
    elif not reviewer_ok or any(c.status == "unverified" for c in checks):
        status = "unverified"
    else:
        status = "passed"
    issues = [f"{tid}: {msg}" for tid, msg in hard_failures.items() if tid not in failed_tasks]
    return ReviewResult(QualityReport(status=status, checks=checks, summary=summary, open_issues=issues), feedback)
