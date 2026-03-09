"""Tests for the AI Agent."""

from __future__ import annotations

import pytest

from agent.agent import Agent
from agent.tools import Tool, calculator_tool, datetime_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM:
    """
    Deterministic mock LLM.

    Accepts a list of responses that are returned in order each time
    ``complete`` is called.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        if self._index >= len(self._responses):
            raise RuntimeError("MockLLM ran out of responses.")
        response = self._responses[self._index]
        self._index += 1
        return response


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    def test_addition(self):
        assert calculator_tool("2 + 2") == "4"

    def test_multiplication(self):
        assert calculator_tool("6 * 7") == "42"

    def test_division(self):
        assert calculator_tool("10 / 4") == "2.5"

    def test_sqrt(self):
        assert calculator_tool("sqrt(144)") == "12.0"

    def test_power(self):
        assert calculator_tool("2 ** 10") == "1024"

    def test_expression_with_spaces(self):
        assert calculator_tool("  3 + 4  ") == "7"

    def test_invalid_expression_returns_error(self):
        result = calculator_tool("import os")
        assert result.startswith("Error")

    def test_attribute_traversal_rejected(self):
        """Ensure MRO/subclass escape attempts are blocked."""
        result = calculator_tool("().__class__.__mro__[1].__subclasses__()")
        assert result.startswith("Error")

    def test_string_literal_rejected(self):
        result = calculator_tool("'hello'")
        assert result.startswith("Error")

    def test_division_by_zero_returns_error(self):
        result = calculator_tool("1 / 0")
        assert result.startswith("Error")


class TestDatetimeTool:
    def test_returns_string(self):
        result = datetime_tool("")
        assert isinstance(result, str)

    def test_format(self):
        import re

        result = datetime_tool("ignored")
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)


class TestToolSchema:
    def test_schema_contains_name(self):
        assert "calculator" in calculator_tool.schema()

    def test_schema_contains_description(self):
        assert "mathematical" in calculator_tool.schema()

    def test_schema_contains_examples(self):
        assert "Example" in calculator_tool.schema()

    def test_tool_call_delegates_to_func(self):
        spy_called_with = []

        def spy(x: str) -> str:
            spy_called_with.append(x)
            return "ok"

        t = Tool(name="spy", description="spy tool", func=spy)
        t("hello")
        assert spy_called_with == ["hello"]


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestAgentParsing:
    def test_parse_final_answer(self):
        text = "Thought: done\nFinal Answer: Paris"
        assert Agent._parse_final_answer(text) == "Paris"

    def test_parse_final_answer_multiline(self):
        text = "Final Answer: Line one\nLine two"
        assert Agent._parse_final_answer(text) == "Line one\nLine two"

    def test_parse_final_answer_missing(self):
        assert Agent._parse_final_answer("Thought: still thinking") is None

    def test_parse_action(self):
        text = "Thought: I need to calculate\nAction: calculator\nAction Input: 2 + 2"
        action, action_input = Agent._parse_action(text)
        assert action == "calculator"
        assert action_input == "2 + 2"

    def test_parse_action_case_insensitive(self):
        """LLMs sometimes emit 'action:' in lowercase."""
        text = "thought: I need to calculate\naction: calculator\naction input: 2 + 2"
        action, action_input = Agent._parse_action(text)
        assert action == "calculator"
        assert action_input == "2 + 2"

    def test_parse_action_input_stops_at_newline(self):
        """Action Input should not bleed into subsequent lines."""
        text = "Action: calculator\nAction Input: 3 + 3\nObservation: 6"
        _, action_input = Agent._parse_action(text)
        assert action_input == "3 + 3"

    def test_parse_action_missing(self):
        action, action_input = Agent._parse_action("No action here")
        assert action is None
        assert action_input == ""


class TestAgentRun:
    def test_single_step_final_answer(self):
        llm = MockLLM(["Final Answer: 42"])
        agent = Agent(llm=llm)
        result = agent.run("What is the answer?")
        assert result == "42"

    def test_uses_calculator_tool(self):
        llm = MockLLM(
            [
                "Thought: I need to calculate.\nAction: calculator\nAction Input: 6 * 7",
                "Thought: I have the answer.\nFinal Answer: 42",
            ]
        )
        agent = Agent(llm=llm, tools=[calculator_tool])
        result = agent.run("What is 6 times 7?")
        assert result == "42"

    def test_observation_included_in_next_prompt(self):
        received_prompts: list[str] = []

        class CaptureLLM:
            def complete(self, prompt: str) -> str:
                received_prompts.append(prompt)
                if len(received_prompts) == 1:
                    return "Action: calculator\nAction Input: 3 + 3"
                return "Final Answer: 6"

        agent = Agent(llm=CaptureLLM(), tools=[calculator_tool])
        agent.run("What is 3+3?")
        assert "Observation: 6" in received_prompts[1]

    def test_unknown_tool_returns_error_observation(self):
        received_prompts: list[str] = []

        class CaptureLLM:
            def complete(self, prompt: str) -> str:
                received_prompts.append(prompt)
                if len(received_prompts) == 1:
                    return "Action: nonexistent_tool\nAction Input: anything"
                return "Final Answer: I could not use the tool."

        agent = Agent(llm=CaptureLLM(), tools=[calculator_tool])
        agent.run("test")
        assert "unknown tool" in received_prompts[1]

    def test_max_iterations_raises(self):
        llm = MockLLM(
            ["Action: calculator\nAction Input: 1 + 1"] * 5
        )
        agent = Agent(llm=llm, tools=[calculator_tool], max_iterations=5)
        with pytest.raises(RuntimeError, match="did not produce a final answer"):
            agent.run("Never ending question")

    def test_no_tools_still_works(self):
        llm = MockLLM(["Final Answer: I don't need tools for this."])
        agent = Agent(llm=llm, tools=[])
        result = agent.run("Just answer directly.")
        assert result == "I don't need tools for this."

    def test_malformed_response_returned_as_answer(self):
        """If the LLM doesn't follow the format, return its output as-is."""
        llm = MockLLM(["Here is some unstructured response."])
        agent = Agent(llm=llm)
        result = agent.run("Anything")
        assert result == "Here is some unstructured response."

    def test_verbose_prints(self, capsys):
        llm = MockLLM(["Final Answer: yes"])
        agent = Agent(llm=llm, verbose=True)
        agent.run("Is verbose on?")
        captured = capsys.readouterr()
        assert "[iter 1]" in captured.out


# ---------------------------------------------------------------------------
# LLM tests
# ---------------------------------------------------------------------------


class TestOpenAILLM:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from agent.llm import OpenAILLM

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAILLM(api_key=None)

    def test_accepts_explicit_api_key(self, monkeypatch):
        """Constructor should not raise when an explicit key is provided."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from agent.llm import OpenAILLM

        try:
            llm = OpenAILLM(api_key="sk-test-key")
            assert llm.api_key == "sk-test-key"
        except ImportError:
            pytest.skip("openai package not installed")


# ---------------------------------------------------------------------------
# Search tool tests
# ---------------------------------------------------------------------------


class TestSearchTool:
    def test_no_examples_in_schema(self):
        from agent.tools import search_tool

        assert "Example" not in search_tool.schema()
