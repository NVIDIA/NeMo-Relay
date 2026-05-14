<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Deep Agents Integration

This directory contains the `nemo_flow.integrations.deepagents` package, which
adds Deep Agents-specific observability on top of the LangChain and LangGraph
integrations.

Deep Agents is built on LangChain `AgentMiddleware` and the LangGraph runtime.
The integration therefore composes the existing NeMo Flow LangChain and
LangGraph hooks, then emits Deep Agents-specific marks for configured skills,
subagents, and human-in-the-loop lifecycle events.

## Setup

```bash
uv sync --all-groups --extra deepagents
just build-python
```

## Usage Example

```python
import nemo_flow
from deepagents import create_deep_agent
from nemo_flow.integrations.deepagents import (
    NemoFlowDeepAgentsCallbackHandler,
    add_nemo_flow_integration,
)

agent = create_deep_agent(
    **add_nemo_flow_integration(
        model="nvidia:nvidia/nemotron-3-nano-30b-a3b",
        tools=[],
        skills=["/skills/research/"],
        name="main-agent",
    )
)

with nemo_flow.scope.scope("deepagents-request", nemo_flow.ScopeType.Agent):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Research recent GPU news"}]},
        config={"callbacks": [NemoFlowDeepAgentsCallbackHandler()]},
    )
```

## What Is Captured

- LangChain model and tool calls through NeMo Flow managed execution.
- LangGraph run scopes through callbacks.
- Human-in-the-loop interrupt and resume marks.
- Configured skills and subagent summaries at agent-run start.
- In-process dictionary-style subagents are instrumented with the same NeMo
  Flow middleware so their model and tool calls are captured when Deep Agents
  invokes them.

Remote graphs or processes still need NeMo Flow instrumentation in that graph
or process to capture their internal model and tool calls.
