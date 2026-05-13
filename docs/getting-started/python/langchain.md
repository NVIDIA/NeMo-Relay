<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow LangChain Integration

Use the `nemo_flow.integrations.langchain` package to add NeMo Flow
observability to [LangChain](https://www.langchain.com/langchain) agents.

## Setup

Install the LangChain integration extra in your application environment.

::::{tab-set}
:sync-group: install-tool

:::{tab-item} uv
:selected:
:sync: uv

```bash
uv add "nemo-flow[langchain]"
```
:::

:::{tab-item} pip
:sync: pip

```bash
pip install "nemo-flow[langchain]"
```
:::

::::

The example below uses the NVIDIA LangChain provider. Install that provider
extra too if you want to run the example as written:

::::{tab-set}
:sync-group: install-tool

:::{tab-item} uv
:selected:
:sync: uv

```bash
uv add "nemo-flow[langchain,langchain-nvidia]"
```
:::

:::{tab-item} pip
:sync: pip

```bash
pip install "nemo-flow[langchain,langchain-nvidia]"
```
:::

::::

## Usage Example

```python
import asyncio

import nemo_flow
from langchain.agents import create_agent
from langchain_core.tools import tool
from nemo_flow.integrations.langchain import NemoFlowCallbackHandler, NemoFlowMiddleware


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 72 degrees."


agent = create_agent(
    model="nvidia:nvidia/nemotron-3-nano-30b-a3b",
    tools=[get_weather],
    middleware=[NemoFlowMiddleware()],
    system_prompt="Use tools when they are relevant. Keep the final answer brief.",
)

input_payload = {
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in San Francisco?",
        }
    ]
}

with nemo_flow.scope.scope("langchain-request", nemo_flow.ScopeType.Agent):
    result = asyncio.run(
        agent.ainvoke(input_payload, config={"callbacks": [NemoFlowCallbackHandler()]})
    )

final_message = result["messages"][-1]
print(f"Final response: {final_message.content}")
```

## Observability

Refer to [Export Observability Data](../../export-observability-data/about.md) for details on exporting NeMo Flow observability data to third-party systems.
