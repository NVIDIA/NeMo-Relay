<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow LangChain Integration

This directory contains the `nemo_flow.integrations.langchain` package which provides observability integration for LangChain.

The intent of this project is to enable as much NeMo Flow functionality as possible using public LangChain APIs without requiring changes to LangChain itself.

For an alternate approach refer to [the patch-based integration in `third_party/langchain`](../../../../third_party/README-langchain.md).

## Setup

```bash
uv sync --all-groups --all-extras
just build-python
```

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
    result = asyncio.run(agent.ainvoke(input_payload, config={"callbacks": [NemoFlowCallbackHandler()]}))

final_message = result["messages"][-1]
print(f"Final response: {final_message.content}")
```

## Validation

Run tests for the LangChain integration package to validate the integration:

```bash
uv run pytest python/tests/integrations/langchain
```
