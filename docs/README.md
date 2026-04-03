<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Agent Toolkit Nexus Documentation

Nexus is a multi-language agent runtime framework providing execution scope management, lifecycle event tracking, and configurable middleware pipelines for tool and LLM calls.

## Contents

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture.md) | High-level system design, binding layers, and data flow |
| [Core Concepts](concepts.md) | Scopes, handles, events, and the middleware pipeline |
| [API Reference](api-reference.md) | Complete function signatures for all operations |
| [Middleware Pipeline](middleware-pipeline.md) | Detailed pipeline ordering for tool and LLM calls |
| [Typed Wrappers](typed-wrappers.md) | Codec-based typed APIs for Python and Node.js |
| [Context Isolation](context-isolation.md) | Multi-tenant and concurrent scope stack management |
| [ATIF Export](atif-export.md) | Agent Trajectory Interchange Format export |
| [Language Bindings](language-bindings.md) | Per-language usage guides and naming conventions |
| [Proxy Layer](proxy-layer.md) | NexusProxy configuration, DynamoIntercept, declarative and builder APIs |
| [Online Learning Engine](online-learning-engine.md) | Prediction trie, sensitivity scoring, Redis persistence, and learner pipeline |
| [Testing](testing.md) | Test commands, helpers, coverage, and conventions |
| [Integration Best Practices](integration-best-practices.md) | Patterns for integrating Nexus into agent frameworks |

See also: [Contributing Guide](../.github/CONTRIBUTING.md) for development setup, branch naming, and PR process.

## Quick Start

```python
import asyncio

import nat_nexus

async def amain():
    # Define your tool and LLM functions
    my_tool_func = lambda args: {**args, "result": "ok"}
    my_llm_func = lambda request: {**request.content, "response": "ok"}

    # Subscribe to lifecycle events
    nat_nexus.subscribers.register("logger", lambda event: print(event.name))

    # Run the agent inside a Nexus scope
    with nat_nexus.scope.scope("my_agent", nat_nexus.ScopeType.Agent) as handle:
        # Execute an LLM call through the full middleware pipeline
        request = nat_nexus.LLMRequest(
            headers={"Authorization": "Bearer ..."},
            content={"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"},
        )
        response = await nat_nexus.llm.execute("gpt-4", request, my_llm_func)

        # Execute a tool call
        result = await nat_nexus.tools.execute("search", {"query": "example"}, my_tool_func)
        print(result)


asyncio.run(amain())
```
