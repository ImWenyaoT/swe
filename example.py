"""
Example usage of the AI Agent.

Usage
-----
1. Set your OpenAI API key::

       export OPENAI_API_KEY="sk-..."

2. Run this script::

       python example.py

The agent will answer a question that requires using the calculator and
datetime tools, printing each reasoning step.
"""

from agent import Agent, calculator_tool, datetime_tool, search_tool
from agent.llm import OpenAILLM

def main() -> None:
    llm = OpenAILLM(model="gpt-4o-mini")

    tools = [calculator_tool, datetime_tool, search_tool]

    agent = Agent(llm=llm, tools=tools, verbose=True)

    questions = [
        "What is 17 multiplied by 38, then divided by 2?",
        "What is today's date?",
        "What is the square root of 256 plus 10?",
    ]

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print("=" * 60)
        answer = agent.run(question)
        print(f"\nFinal Answer: {answer}\n")


if __name__ == "__main__":
    main()
