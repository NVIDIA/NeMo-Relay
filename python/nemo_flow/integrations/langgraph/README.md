<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow LangGraph Integration

This directory contains the `nemo_flow.integrations.langgraph` package, which provides public-API LangGraph integration for NeMo Flow.

The integration builds on `nemo_flow.integrations.langchain`: `NemoFlowCallbackHandler` inherits the LangChain callback handler, and `NemoFlowMiddleware` is re-exported for LangChain agents used inside LangGraph workflows.

For an alternate approach refer to [the patch-based integration in `third_party/langchain`](../../../../third_party/README-langgraph.md).

## Setup

```bash
uv sync --all-groups --extra langgraph
just build-python
```

Installing the `langgraph` extra also installs the LangChain integration dependencies.

## Usage Example

```python
from typing_extensions import TypedDict

import nemo_flow
from langgraph.graph import END, START, StateGraph
from nemo_flow.integrations.langgraph import NemoFlowCallbackHandler


class State(TypedDict):
    value: int


def increment(state: State) -> State:
    return {"value": state["value"] + 1}


builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

graph = builder.compile()

with nemo_flow.scope.scope("langgraph-request", nemo_flow.ScopeType.Agent):
    result = graph.invoke(
        {"value": 1},
        config={"callbacks": [NemoFlowCallbackHandler()]},
    )

print(result)
```

For LangChain agents inside a LangGraph workflow, use `NemoFlowMiddleware` from this package the same way as the LangChain integration and pass the LangGraph `config` into the nested agent call:

```python
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from nemo_flow.integrations.langgraph import NemoFlowMiddleware

agent = create_agent(
    model="nvidia:nvidia/nemotron-3-nano-30b-a3b",
    tools=[],
    middleware=[NemoFlowMiddleware()],
)


def agent_node(state: dict, config: RunnableConfig) -> dict:
    return agent.invoke({"messages": state["messages"]}, config=config)
```

## Validation

```bash
uv run pytest python/tests/integrations/langgraph
```
