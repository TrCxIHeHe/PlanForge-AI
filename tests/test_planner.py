import copy

import pytest

from app.planner import PlanError, create_plan, fallback_plan, normalise_plan
from app.tools import ToolRegistry
from tests.fakes import FakeLLM, GOOD_PLAN

TOOLS = ToolRegistry()


def test_good_plan_passes_through_unchanged():
    plan = normalise_plan(GOOD_PLAN, "request", TOOLS)
    assert plan.source == "llm" and [t.id for t in plan.tasks] == ["T1", "T2", "T3", "T4"]
    assert plan.tasks[0].depends_on == ["T2", "T3", "T4"]


def test_every_requirement_is_traced_to_a_task():
    raw = copy.deepcopy(GOOD_PLAN)
    for t in raw["tasks"]:
        t["satisfies"] = []
    plan = normalise_plan(raw, "request", TOOLS)
    covered = {r for t in plan.tasks for r in t.satisfies}
    assert covered == {r.id for r in plan.requirements} and plan.source == "repaired"
    budget_req = next(r for r in plan.requirements if "budget" in r.text)
    assert budget_req.id in next(t for t in plan.tasks if t.id == "T2").satisfies  # matched by wording


def test_cycles_unknown_deps_and_tools_are_repaired():
    raw = {"tasks": [
        {"id": "a", "title": "One", "depends_on": ["b", "ghost", "a"]},
        {"id": "b", "title": "Two", "depends_on": ["a"], "tools": [{"tool": "rm -rf"}, {"tool": "calculate", "args": {"expression": "1+1"}}]},
    ]}
    plan = normalise_plan(raw, "do the thing please", TOOLS)
    assert plan.tasks[0].depends_on == ["T2"] and plan.tasks[1].depends_on == []  # second edge would close a cycle
    assert [c.tool for c in plan.tasks[1].tools] == ["calculate"]
    assert any("cycle" in r for r in plan.repairs)


def test_too_few_tasks_rejected():
    with pytest.raises(PlanError):
        normalise_plan({"tasks": [{"title": "only one"}]}, "x", TOOLS)


async def test_invalid_plan_is_regenerated_with_feedback():
    seq = lambda label, s, user, n: {"tasks": []} if label == "plan:1" else GOOD_PLAN
    llm = FakeLLM({"plan": seq})
    plan = await create_plan(llm, "make a proposal", TOOLS)
    assert plan.source == "repaired" and len(llm.calls) == 2
    assert "rejected" in llm.calls[1][1]  # the validation problem was fed back to the model


async def test_hopeless_plan_falls_back_to_deterministic_structure():
    plan = await create_plan(FakeLLM({"plan": {"nonsense": True}}), "make a proposal about onboarding", TOOLS)
    assert plan.source == "fallback" and len(plan.tasks) == 6


def test_fallback_plan_is_itself_valid():
    plan = fallback_plan("Write an SOP for incident response")
    assert normalise_plan(plan.model_dump(), "x", TOOLS).source == "llm"
