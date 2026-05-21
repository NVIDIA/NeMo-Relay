<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay LangGraph Integration

Use the `nemo_relay.integrations.langgraph` package to add NeMo Relay
observability to [LangGraph](https://www.langchain.com/langgraph) workflows through public LangGraph APIs.

## Setup

Install the LangGraph integration extra in your application environment.

::::{tab-set}
:sync-group: install-tool

:::{tab-item} uv
:selected:
:sync: uv

```bash
uv add "nemo-relay[langgraph]"
```
:::

:::{tab-item} pip
:sync: pip

```bash
pip install "nemo-relay[langgraph]"
```
:::

::::

Installing the `langgraph` extra also installs the LangChain integration
dependencies.

## Usage Example

```python
from typing_extensions import TypedDict

import nemo_relay
from langgraph.graph import END, START, StateGraph
from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler


class State(TypedDict):
    value: int


def increment(state: State) -> State:
    return {"value": state["value"] + 1}


builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

graph = builder.compile()

with nemo_relay.scope.scope("langgraph-request", nemo_relay.ScopeType.Agent):
    result = graph.invoke(
        {"value": 1},
        config={"callbacks": [NemoRelayCallbackHandler()]},
    )

print(result)
```

For LangChain agents inside a LangGraph workflow, use `NemoRelayMiddleware` from
this package the same way as the LangChain integration and pass the LangGraph
`config` into the nested agent call:

```python
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from nemo_relay.integrations.langgraph import NemoRelayMiddleware

agent = create_agent(
    model="nvidia:nvidia/nemotron-3-nano-30b-a3b",
    tools=[],
    middleware=[NemoRelayMiddleware()],
)


def agent_node(state: dict, config: RunnableConfig) -> dict:
    return agent.invoke({"messages": state["messages"]}, config=config)
```

Install the NVIDIA LangChain provider if you want to run the nested agent
example as written:

::::{tab-set}
:sync-group: install-tool

:::{tab-item} uv
:selected:
:sync: uv

```bash
uv add "nemo-relay[langgraph,langchain-nvidia]"
```
:::

:::{tab-item} pip
:sync: pip

```bash
pip install "nemo-relay[langgraph,langchain-nvidia]"
```
:::

::::

## Observability

Refer to [Observability](../plugins/observability/about.md) for details on exporting NeMo Relay observability data to third-party systems.
