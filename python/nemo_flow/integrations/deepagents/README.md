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
LangGraph hooks, then emits stable scopes for Deep Agents tool concepts:
subagents, async subagents, filesystem tool calls, and sandbox tool calls.
Human-in-the-loop interrupts, resumes, and configured skills remain marks.
Direct backend operations are captured as NeMo Flow scopes.

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
- Deep Agents `task` delegation scopes for synchronous subagents.
- Deep Agents async subagent tool lifecycle scopes.
- Filesystem tool scopes for `ls`, `read_file`, `write_file`, `edit_file`,
  `glob`, and `grep`.
- Sandbox/local shell `execute` scopes.
- Configured skills and subagent summaries at agent-run start.
- Direct backend method call scopes when the backend is wrapped with
  `observe_backend()` or via `add_nemo_flow_integration(..., backend=...)`.

Remote async subagents still need NeMo Flow instrumentation in the remote graph
or process to capture their internal model and tool calls. Supervisor-side
observability captures the task lifecycle and returned status.
