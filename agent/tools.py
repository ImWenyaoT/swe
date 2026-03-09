"""Tool definitions for the AI Agent."""

from __future__ import annotations

import math
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
    """Evaluate a safe mathematical expression."""
    # Allow only safe mathematical operations
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "pow": pow})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
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
    examples=["Input: 'capital of France'  → Output: 'Paris'"],
)
