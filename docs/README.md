<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVMagic Documentation

NVMagic is a multi-language agent runtime framework providing execution scope management, lifecycle event tracking, and configurable middleware pipelines for tool and LLM calls.

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

## Quick Start

```python
import nvmagic
from nvmagic import LLMRequest, ScopeType

# Push an agent scope
handle = nvmagic.scope.push("my_agent", ScopeType.Agent)

# Execute an LLM call through the full middleware pipeline
request = LLMRequest(
    headers={"Authorization": "Bearer ..."},
    content={"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"},
)
response = await nvmagic.llm.execute("gpt-4", request, my_llm_func)

# Execute a tool call
result = await nvmagic.tools.execute("search", {"query": "example"}, my_tool_func)

nvmagic.scope.pop(handle)
```
