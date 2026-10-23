"""Deterministic tools the agent can call instead of trusting the LLM with arithmetic and dates.

Tools never raise: they return a ToolOutcome so one bad argument cannot crash a run.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

MAX_EXPR_CHARS = 200


@dataclass
class ToolOutcome:
    ok: bool
    result: Any = None
    error: str | None = None


# ---------------------------------------------------------------- safe calculator
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {"round": round, "min": min, "max": max, "sum": sum, "abs": abs}


class ToolError(ValueError):
    pass


def _normalise(expression: str) -> str:
    text = expression.strip().replace("\u00d7", "*").replace("\u00f7", "/").replace("^", "**")
    text = re.sub(r"[$\u20b9\u20ac\u00a3]", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 2,500 -> 2500
    return re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(a-zA-Z.])", r"(\1/100)", text)  # 12% -> (12/100), 10 % 3 stays modulo


def _eval(node: ast.AST, variables: dict[str, float | int] | None = None) -> Any:
    variables = variables or {}
    if isinstance(node, ast.Expression):
        return _eval(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left, variables), _eval(node.right, variables)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e9):
            raise ToolError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, variables))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, variables) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a, variables) for a in node.args])
    raise ToolError("unsupported expression; use numbers, + - * / ** % ( ) and round/min/max/sum/abs")


def safe_eval(expression: str, variables: dict[str, float | int] | None = None) -> float | int:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression must be a non-empty string")
    if len(expression) > MAX_EXPR_CHARS:
        raise ToolError("expression too long")
    source = expression.strip()
    assignment = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)", source)
    if assignment:
        source = assignment.group(1)
    try:
        value = _eval(ast.parse(_normalise(source), mode="eval"), variables)
    except ToolError:
        raise
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except (SyntaxError, TypeError, ValueError, OverflowError, RecursionError):
        raise ToolError("could not evaluate expression") from None
    if isinstance(value, list) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
        raise ToolError("expression did not produce a finite number")
    if isinstance(value, float):
        value = round(value, 6)
        return int(value) if value.is_integer() else value
    return value


# ---------------------------------------------------------------- tools
def _calculate(args: dict[str, Any], _today: date) -> Any:
    expressions = args.get("expressions")
    if expressions is None:
        expressions = [args.get("expression")]
        single = True
    else:
        single = False
    if not isinstance(expressions, list) or not expressions or len(expressions) > 20:
        raise ToolError("provide 'expression' or up to 20 'expressions'")
    variables: dict[str, float | int] = {}
    results = []
    for expression in expressions:
        source = str(expression)
        statements = source.split(";")
        if len(statements) > 1:
            value = None
            for statement in statements:
                label = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", statement)
                if not label:
                    raise ToolError("semicolon-separated calculations must be labeled assignments")
                value = safe_eval(statement, variables)
                variables[label.group(1)] = value
            if value is None:
                raise ToolError("expression must be a non-empty string")
        else:
            label = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", source)
            value = safe_eval(source, variables)
            if label:
                variables[label.group(1)] = value
        results.append({"expression": str(expression), "value": value})
    return results[0] if single else results


def _schedule(args: dict[str, Any], today: date) -> Any:
    phases = args.get("phases")
    if not isinstance(phases, list) or not phases or len(phases) > 12:
        raise ToolError("'phases' must be a list of 1-12 {name, weeks} objects")
    start_raw = args.get("start")
    adjustment = ""
    if isinstance(start_raw, str) and start_raw.strip().lower() in {
        "yyyy-mm-dd", "yyyy-mm-dd (optional)", "optional", "tbd", "unknown"
    }:
        start_raw = None
    if start_raw:
        try:
            cursor = date.fromisoformat(str(start_raw))
        except ValueError:
            raise ToolError("'start' must be YYYY-MM-DD; omit it to use the next Monday") from None
        if cursor < today:
            requested = cursor.isoformat()
            cursor = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
            adjustment = f"Requested start {requested} was in the past; schedule starts next Monday instead."
    else:
        cursor = today + timedelta(days=(7 - today.weekday()) % 7 or 7)  # next Monday
    out, begin, total_days = [], cursor, 0
    for i, phase in enumerate(phases, 1):
        try:
            name, weeks = str(phase["name"]), float(phase["weeks"])
        except (KeyError, TypeError, ValueError):
            raise ToolError(f"phase {i}: needs 'name' and numeric 'weeks'") from None
        if not 0 < weeks <= 52:
            raise ToolError(f"phase {i}: weeks must be between 0 and 52")
        days = max(1, int(weeks * 7 + 0.5))
        end = cursor + timedelta(days=days - 1)
        out.append({"phase": name, "start": cursor.strftime("%d %b %Y"), "end": end.strftime("%d %b %Y"), "weeks": weeks})
        cursor, total_days = end + timedelta(days=1), total_days + days
    return {
        "phases": out,
        "start": begin.strftime("%d %b %Y"),
        "finish": out[-1]["end"],
        "total_weeks": round(total_days / 7, 1),
        "note": adjustment or "calendar weeks, phases run back to back; start defaults to the next Monday",
    }


def _risk_score(args: dict[str, Any], _today: date) -> Any:
    risks = args.get("risks")
    if not isinstance(risks, list) or not risks or len(risks) > 20:
        raise ToolError("'risks' must be a list of 1-20 {name, likelihood, impact} objects")
    scored = []
    for i, risk in enumerate(risks, 1):
        try:
            name, likelihood, impact = str(risk["name"]), int(risk["likelihood"]), int(risk["impact"])
        except (KeyError, TypeError, ValueError):
            raise ToolError(f"risk {i}: needs 'name', integer 'likelihood' and 'impact'") from None
        if not (1 <= likelihood <= 5 and 1 <= impact <= 5):
            raise ToolError(f"risk {i}: likelihood and impact must be between 1 and 5")
        score = likelihood * impact
        rating = "Low" if score <= 4 else "Medium" if score <= 9 else "High" if score <= 16 else "Critical"
        scored.append({"name": name, "likelihood": likelihood, "impact": impact, "score": score, "rating": rating})
    scored.sort(key=lambda r: -r["score"])
    return {"risks": scored, "scale": "score = likelihood x impact (1-25): Low<=4, Medium<=9, High<=16, Critical>16"}


TOOL_SPECS: dict[str, tuple[Callable[[dict[str, Any], date], Any], str]] = {
    "calculate": (_calculate, 'args {"expression": "4*2500 + 12%*8000"} or {"expressions": ["approach_a_cost = 0 + 12*50", "approach_b_cost = 1500 + 12*20", "variance = approach_b_cost - approach_a_cost"]}; exact arithmetic for costs, totals, percentages; labels may reference earlier results in the same list; one expression may contain semicolon-separated labeled assignments'),
    "schedule": (_schedule, 'args {"phases": [{"name": "Discovery", "weeks": 2}, ...], "start": "YYYY-MM-DD" (optional; omit unless providing a real future date)}; real future calendar dates for a phased timeline'),
    "risk_score": (_risk_score, 'args {"risks": [{"name": "...", "likelihood": 1-5, "impact": 1-5}]}; ranked risk register scores'),
}


class ToolRegistry:
    def __init__(self, today: Callable[[], date] = date.today):
        self._today = today

    @property
    def names(self) -> list[str]:
        return list(TOOL_SPECS)

    def describe(self) -> str:
        return "\n".join(f"- {name}: {spec}" for name, (_, spec) in TOOL_SPECS.items())

    def run(self, name: str, args: dict[str, Any] | None) -> ToolOutcome:
        entry = TOOL_SPECS.get(name)
        if entry is None:
            return ToolOutcome(False, error=f"unknown tool '{name}'")
        if not isinstance(args, dict):
            return ToolOutcome(False, error="tool args must be an object")
        try:
            return ToolOutcome(True, result=entry[0](args, self._today()))
        except ToolError as exc:
            return ToolOutcome(False, error=str(exc))
        except Exception as exc:  # a tool bug must never take the run down
            return ToolOutcome(False, error=f"tool crashed: {type(exc).__name__}")
