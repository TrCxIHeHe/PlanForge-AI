"""Validated API and agent data contracts."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class AgentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    request: str = Field(min_length=8, max_length=5000, description="Natural-language task for the agent")


# ---------------------------------------------------------------- planning
class Requirement(BaseModel):
    id: str  # R1, R2 ... - testable statement the finished document must satisfy
    text: str


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class PlanTask(BaseModel):
    id: str  # T1, T2 ...
    title: str
    objective: str
    section_title: str
    depends_on: list[str] = Field(default_factory=list)
    satisfies: list[str] = Field(default_factory=list)  # requirement ids this task is responsible for
    tools: list[ToolCall] = Field(default_factory=list)


class AgentPlan(BaseModel):
    title: str
    document_type: str
    purpose: str
    assumptions: list[str] = Field(default_factory=list)
    requirements: list[Requirement]
    tasks: list[PlanTask]
    source: Literal["llm", "repaired", "fallback"] = "llm"
    repairs: list[str] = Field(default_factory=list)  # what normalisation had to fix


# ---------------------------------------------------------------- content
class Block(BaseModel):
    type: Literal["paragraph", "bullets", "numbered", "table"]
    text: str = ""
    items: list[str] = Field(default_factory=list)
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str = ""


class Section(BaseModel):
    task_id: str
    title: str
    blocks: list[Block]
    key_points: list[str] = Field(default_factory=list)
    status: Literal["completed", "revised", "failed"] = "completed"
    attempts: int = 1
    tool_log: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------- review
class RequirementCheck(BaseModel):
    requirement_id: str
    text: str
    status: Literal["met", "not_met", "unverified"]
    note: str = ""


class QualityReport(BaseModel):
    status: Literal["passed", "needs_attention", "unverified"]
    checks: list[RequirementCheck] = Field(default_factory=list)
    summary: str = ""
    revision_rounds: int = 0
    revised_tasks: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)  # deterministic failures an LLM verdict cannot override


# ---------------------------------------------------------------- API response
class ExecutionStep(BaseModel):
    task_id: str
    title: str
    status: Literal["completed", "revised", "failed"]
    depends_on: list[str] = Field(default_factory=list)
    attempts: int = 1
    tools_used: list[str] = Field(default_factory=list)
    note: str | None = None


class DocumentInfo(BaseModel):
    filename: str
    download_path: str
    content_type: str = DOCX_MIME
    size_bytes: int
    base64_content: str | None = None


class AgentResponse(BaseModel):
    request_id: str
    status: Literal["completed", "completed_with_warnings"]
    summary: str
    warnings: list[str] = Field(default_factory=list)
    llm: dict[str, Any]
    plan: AgentPlan
    execution: list[ExecutionStep]
    quality_check: QualityReport
    document: DocumentInfo
