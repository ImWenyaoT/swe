"""
Core AI Agent implementing the ReAct (Reasoning + Acting) loop.

ReAct loop
----------
The agent iterates between four phases until it produces a final answer:

1. **Thought**   – the LLM reasons about the current state.
2. **Action**    – the LLM picks a tool and provides input.
3. **Observation** – the tool is executed and the result is appended.
4. **Answer**    – when the LLM decides it has enough information it emits
                   ``Final Answer: <answer>``.

References
----------
Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"
https://arxiv.org/abs/2210.11610
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .tools import Tool

if TYPE_CHECKING:
    from .llm import LLMProtocol

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a helpful AI assistant that solves tasks step-by-step using tools.

You have access to the following tools:
{tool_schemas}

Use the following format EXACTLY (do not deviate):

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <input to the tool>
Observation: <tool output – filled in by the system>
... (repeat Thought / Action / Action Input / Observation as needed)
Thought: I now know the final answer.
Final Answer: <your final answer to the original question>

Begin!

Question: {question}
"""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """
    A simple ReAct agent.

    Parameters
    ----------
    llm:
        Any object with a ``complete(prompt: str) -> str`` method.
    tools:
        List of :class:`~agent.tools.Tool` instances the agent may use.
    max_iterations:
        Safety limit on the number of Thought/Action/Observation cycles.
    verbose:
        If ``True``, print each step to stdout.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, question: str) -> str:
        """
        Run the agent on *question* and return the final answer string.

        Raises
        ------
        RuntimeError
            If the agent exceeds ``max_iterations`` without producing a
            final answer.
        """
        tool_schemas = "\n".join(t.schema() for t in self.tools.values()) or "(none)"
        prompt = _SYSTEM_PROMPT.format(tool_schemas=tool_schemas, question=question)
        scratchpad = ""

        for iteration in range(self.max_iterations):
            full_prompt = prompt + scratchpad
            raw = self.llm.complete(full_prompt)

            if self.verbose:
                print(f"[iter {iteration + 1}]\n{raw}\n{'─' * 60}")

            # Check for a final answer first
            final = self._parse_final_answer(raw)
            if final is not None:
                return final

            # Parse the action the LLM wants to take
            action, action_input = self._parse_action(raw)

            if action is None:
                # The model didn't follow the format; treat the whole output
                # as the final answer rather than looping forever.
                return raw.strip()

            # Execute the tool
            observation = self._call_tool(action, action_input)

            # Append this turn to the scratchpad so the next prompt includes it
            scratchpad += (
                f"{raw.rstrip()}\n"
                f"Observation: {observation}\n"
            )

        raise RuntimeError(
            f"Agent did not produce a final answer within {self.max_iterations} iterations."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_final_answer(text: str) -> str | None:
        """Return the text after 'Final Answer:' or ``None`` if not present."""
        match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_action(text: str) -> tuple[str | None, str]:
        """
        Extract (action_name, action_input) from an LLM response.

        Returns ``(None, '')`` when no action is found.
        """
        action_match = re.search(r"Action:\s*(.+)", text)
        input_match = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
        if not action_match:
            return None, ""
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""
        return action, action_input

    def _call_tool(self, name: str, tool_input: str) -> str:
        """Execute a tool by name, returning an error string on failure."""
        tool = self.tools.get(name)
        if tool is None:
            available = ", ".join(self.tools) or "(none)"
            return f"Error: unknown tool '{name}'. Available tools: {available}."
        try:
            return tool(tool_input)
        except Exception as exc:  # noqa: BLE001
            return f"Error running tool '{name}': {exc}"
