from datetime import date

import pytest

from app.tools import ToolError, ToolRegistry, safe_eval

REG = ToolRegistry(today=lambda: date(2026, 10, 23))  # a Friday


@pytest.mark.parametrize("expr,value", [("2+2*3", 8), ("4*2,500 + 12%*8000", 10960), ("10 % 3", 1), ("round(10/3, 2)", 3.33), ("sum([1,2,3])", 6), ("-(2**3)", -8), ("Approach_B_Total = 49 * 12 + 150", 738)])
def test_safe_eval_values(expr, value):
    assert safe_eval(expr) == value


@pytest.mark.parametrize("expr", ["__import__('os').system('id')", "open('x')", "9**9**9", "1/0", "", "a+1", "(lambda: 1)()", "1;2", "x" * 300])
def test_safe_eval_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        safe_eval(expr)


def test_calculate_single_and_many():
    assert REG.run("calculate", {"expression": "2+2"}).result == {"expression": "2+2", "value": 4}
    many = REG.run("calculate", {"expressions": ["1+1", "3*3"]}).result
    assert [r["value"] for r in many] == [2, 9]


def test_calculate_supports_references_to_prior_labeled_results():
    many = REG.run(
        "calculate",
        {
            "expressions": [
                "approach_a_cost = 0 + 12*50",
                "approach_b_cost = 1500 + 12*20",
                "variance = approach_b_cost - approach_a_cost",
            ]
        },
    )
    assert many.ok
    assert [result["value"] for result in many.result] == [600, 1740, 1140]
    assert not REG.run("calculate", {"expression": "undefined_cost + 10"}).ok


def test_calculate_supports_safe_semicolon_separated_assignments():
    result = REG.run(
        "calculate",
        {"expression": "target_csat = 85; current_csat = 70; improvement = target_csat - current_csat"},
    )
    assert result.ok
    assert result.result["value"] == 15
    assert not REG.run("calculate", {"expression": "1+1; 2+2"}).ok


def test_schedule_defaults_to_next_monday_and_chains_phases():
    out = REG.run("schedule", {"phases": [{"name": "A", "weeks": 2}, {"name": "B", "weeks": 1}]}).result
    assert out["start"] == "26 Oct 2026"
    assert out["phases"][0]["end"] == "08 Nov 2026" and out["phases"][1]["start"] == "09 Nov 2026"
    assert out["total_weeks"] == 3.0


def test_schedule_treats_format_placeholder_as_omitted_start():
    out = REG.run(
        "schedule",
        {"phases": [{"name": "A", "weeks": 1}], "start": "YYYY-MM-DD"},
    )
    assert out.ok and out.result["start"] == "26 Oct 2026"


def test_schedule_replaces_past_start_with_future_monday_and_discloses_adjustment():
    out = REG.run(
        "schedule",
        {"phases": [{"name": "A", "weeks": 1}], "start": "2023-11-01"},
    )
    assert out.ok
    assert out.result["start"] == "26 Oct 2026"
    assert "2023-11-01 was in the past" in out.result["note"]


def test_schedule_validation():
    assert not REG.run("schedule", {"phases": []}).ok
    assert not REG.run("schedule", {"phases": [{"name": "A", "weeks": 0}]}).ok
    assert not REG.run("schedule", {"phases": [{"name": "A", "weeks": 1}], "start": "tomorrow"}).ok
    assert "omit it" in REG.run(
        "schedule",
        {"phases": [{"name": "A", "weeks": 1}], "start": "next week"},
    ).error


def test_risk_score_ranks_and_rates():
    out = REG.run("risk_score", {"risks": [{"name": "x", "likelihood": 1, "impact": 2}, {"name": "y", "likelihood": 5, "impact": 5}]}).result
    assert out["risks"][0]["name"] == "y" and out["risks"][0]["rating"] == "Critical" and out["risks"][1]["rating"] == "Low"
    assert not REG.run("risk_score", {"risks": [{"name": "x", "likelihood": 9, "impact": 5}]}).ok


def test_unknown_tool_and_bad_args_never_raise():
    assert not REG.run("nope", {}).ok
    assert not REG.run("calculate", None).ok
    assert not REG.run("calculate", {"expressions": "not a list"}).ok
