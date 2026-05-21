<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay LangChain Integration

Use the `nemo_relay.integrations.langchain` package to add NeMo Relay
observability to [LangChain](https://www.langchain.com/langchain) agents.

## Setup

Install the LangChain integration extra in your application environment.

::::{tab-set}
:sync-group: install-tool

:::{tab-item} uv
:selected:
:sync: uv

```bash
uv add "nemo-relay[langchain]"
```
:::

:::{tab-item} pip
:sync: pip

```bash
pip install "nemo-relay[langchain]"
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
uv add "nemo-relay[langchain,langchain-nvidia]"
```
:::

:::{tab-item} pip
:sync: pip

```bash
pip install "nemo-relay[langchain,langchain-nvidia]"
```
:::

::::

## Usage Example

```python
import asyncio

import nemo_relay
from langchain.agents import create_agent
from langchain_core.tools import tool
from nemo_relay.integrations.langchain import NemoRelayCallbackHandler, NemoRelayMiddleware


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 72 degrees."


agent = create_agent(
    model="nvidia:nvidia/nemotron-3-nano-30b-a3b",
    tools=[get_weather],
    middleware=[NemoRelayMiddleware()],
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

with nemo_relay.scope.scope("langchain-request", nemo_relay.ScopeType.Agent):
    result = asyncio.run(
        agent.ainvoke(input_payload, config={"callbacks": [NemoRelayCallbackHandler()]})
    )

final_message = result["messages"][-1]
print(f"Final response: {final_message.content}")
```

## Observability

Refer to [Observability](../plugins/observability/about.md) for details on exporting NeMo Relay observability data to third-party systems.
