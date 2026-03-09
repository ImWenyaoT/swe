# Simple AI Agent (ReAct Pattern)

A lightweight AI agent implemented in pure Python that uses the **ReAct** (Reasoning + Acting) loop to solve tasks step-by-step using tools.

> **Paper**: Yao et al., _"ReAct: Synergizing Reasoning and Acting in Language Models"_ — https://arxiv.org/abs/2210.11610

---

## How It Works

The agent repeatedly cycles through four phases until it reaches a final answer:

```
Thought:        (LLM reasons about what to do next)
Action:         (LLM picks a tool to call)
Action Input:   (LLM provides input for the tool)
Observation:    (the tool runs and returns a result)
… repeat …
Final Answer:   (LLM emits the answer and stops)
```

---

## Project Structure

```
agent/
  __init__.py   – public API
  agent.py      – Agent class implementing the ReAct loop
  tools.py      – Tool dataclass + built-in tools
  llm.py        – OpenAI LLM wrapper
tests/
  test_agent.py – pytest test suite (27 tests)
example.py      – runnable demo
requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run the example

```bash
python example.py
```

---

## Usage

```python
from agent import Agent, calculator_tool, datetime_tool, search_tool
from agent.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4o-mini")

agent = Agent(
    llm=llm,
    tools=[calculator_tool, datetime_tool, search_tool],
    verbose=True,       # print each reasoning step
    max_iterations=10,  # safety limit
)

answer = agent.run("What is the square root of 1764?")
print(answer)  # → 42.0
```

---

## Built-in Tools

| Tool         | Description                                                    |
|--------------|----------------------------------------------------------------|
| `calculator` | Evaluates Python math expressions (`sqrt`, `**`, `+`, …)      |
| `datetime`   | Returns the current date and time                              |
| `search`     | Stub web-search (replace with a real backend when needed)      |

---

## Adding Custom Tools

```python
from agent.tools import Tool

def my_func(input_text: str) -> str:
    return f"You said: {input_text}"

my_tool = Tool(
    name="echo",
    description="Echoes the input back.",
    func=my_func,
)

agent = Agent(llm=llm, tools=[my_tool])
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Using a Different LLM

Implement the `LLMProtocol` interface (one method: `complete(prompt: str) -> str`):

```python
class MyLLM:
    def complete(self, prompt: str) -> str:
        # call any LLM API here
        return "Final Answer: 42"

agent = Agent(llm=MyLLM(), tools=[...])
```
