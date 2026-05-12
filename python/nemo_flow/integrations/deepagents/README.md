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
LangGraph hooks, then emits stable mark events for Deep Agents concepts:
subagents, async subagents, human-in-the-loop interrupts and resumes, skills,
filesystem operations, and sandbox execution.

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
    with_nemo_flow_callbacks,
    with_nemo_flow_observability,
)

agent = create_deep_agent(
    **with_nemo_flow_observability(
        model="openai:gpt-5.4",
        tools=[],
        skills=["/skills/research/"],
        name="main-agent",
    )
)

with nemo_flow.scope.scope("deepagents-request", nemo_flow.ScopeType.Agent):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Research recent GPU news"}]},
        config=with_nemo_flow_callbacks(),
    )
```

## What Is Captured

- LangChain model and tool calls through NeMo Flow managed execution.
- LangGraph run scopes through callbacks.
- Human-in-the-loop interrupt and resume marks.
- Deep Agents `task` delegation marks for synchronous subagents.
- Deep Agents async subagent tool lifecycle marks.
- Filesystem tool marks for `ls`, `read_file`, `write_file`, `edit_file`,
  `glob`, and `grep`.
- Sandbox/local shell `execute` marks.
- Configured skills and subagent summaries at agent-run start.
- Direct backend method calls when the backend is wrapped with
  `observe_backend()` or via `with_nemo_flow_observability(..., backend=...)`.

Remote async subagents still need NeMo Flow instrumentation in the remote graph
or process to capture their internal model and tool calls. Supervisor-side
observability captures the task lifecycle and returned status.
