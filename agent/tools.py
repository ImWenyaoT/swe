"""Tool definitions for the AI Agent."""

from __future__ import annotations

import ast
import math
import operator
import datetime
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    """Represents a callable tool available to the agent."""

    name: str
    description: str
    func: Callable[[str], str]
    # Optional list of example usages shown to the LLM
    examples: list[str] = field(default_factory=list)

    def __call__(self, input_text: str) -> str:
        return self.func(input_text)

    def schema(self) -> str:
        """Return a human-readable schema shown to the LLM."""
        lines = [f"- {self.name}: {self.description}"]
        for ex in self.examples:
            lines.append(f"  Example: {ex}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _calculator(expression: str) -> str:
    """Evaluate a mathematical expression using a safe AST-based evaluator."""

    # Allowlist of safe unary/binary operators
    _UNARY_OPS: dict[type, Callable] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    _BIN_OPS: dict[type, Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    # Numeric constants resolvable as bare names
    _SAFE_CONSTS: dict[str, float] = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "nan": math.nan,
    }
    # Allowlist of callable math functions (no attribute access)
    _SAFE_FUNCS: dict[str, Callable] = {
        name: getattr(math, name)  # type: ignore[assignment]
        for name in dir(math)
        if not name.startswith("_") and callable(getattr(math, name))
    }
    _SAFE_FUNCS.update({
        "abs": abs,
        "round": round,
        "pow": pow,
    })

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant type: {type(node.value)}")
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _SAFE_CONSTS:
                return _SAFE_CONSTS[node.id]
            if node.id in _SAFE_FUNCS:
                raise ValueError(
                    f"'{node.id}' is a function; call it with parentheses, e.g. {node.id}(...)."
                )
            raise ValueError(f"Unknown name: '{node.id}'")
        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_fn(_eval(node.operand))
        if isinstance(node, ast.BinOp):
            op_fn = _BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed (no attribute access).")
            func = _SAFE_FUNCS.get(node.func.id)
            if func is None:
                raise ValueError(f"Unknown function: '{node.func.id}'")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported.")
            args = [_eval(a) for a in node.args]
            return func(*args)  # type: ignore[operator]
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


calculator_tool = Tool(
    name="calculator",
    description=(
        "Evaluates a mathematical expression. "
        "Input must be a valid Python math expression (e.g. '2 + 2', 'sqrt(16)', '3 ** 4')."
    ),
    func=_calculator,
    examples=[
        "Input: '2 + 2'  → Output: '4'",
        "Input: 'sqrt(144)'  → Output: '12.0'",
    ],
)


def _datetime_tool(_: str) -> str:
    """Return the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


datetime_tool = Tool(
    name="datetime",
    description="Returns the current date and time. Input is ignored.",
    func=_datetime_tool,
    examples=["Input: ''  → Output: '2024-01-15 10:30:00'"],
)


def _search(query: str) -> str:
    """
    Stub web-search tool.

    Replace the body of this function with a real search integration
    (e.g. DuckDuckGo, Bing, or a custom API) when an API key is available.
    """
    return (
        f"[Search stub] No live results for '{query}'. "
        "Integrate a real search backend (e.g. DuckDuckGo or Bing Search API) to enable live results."
    )


search_tool = Tool(
    name="search",
    description=(
        "Searches the web for up-to-date information. "
        "Input should be a concise search query."
    ),
    func=_search,
    examples=[],
)
