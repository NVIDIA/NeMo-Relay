<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Agent Toolkit Nexus Documentation

Nexus is a multi-language agent runtime framework providing execution scope management, lifecycle event tracking, and configurable middleware pipelines for tool and LLM calls.

This file is the canonical documentation index for the repository.

## Start Here

Choose the path that matches what you are trying to do:

### Using Nexus from Python

1. [Getting Started: Python](getting-started-python.md)
2. [Language Bindings](language-bindings.md#python)
3. [Observability with OpenTelemetry](observability-with-opentelemetry.md)
4. [Observability with OpenInference](observability-with-openinference.md)
5. [Core Concepts](concepts.md)
6. [API Reference](api-reference.md)
7. [ATIF Export](atif-export.md)
8. [Testing](testing.md)

### Using Nexus from Node.js

1. [Getting Started: Node.js](getting-started-node.md)
2. [Language Bindings](language-bindings.md#nodejs)
3. [Observability with OpenTelemetry](observability-with-opentelemetry.md)
4. [Observability with OpenInference](observability-with-openinference.md)
5. [Core Concepts](concepts.md)
6. [Middleware Pipeline](middleware-pipeline.md)
7. [Typed Wrappers](typed-wrappers.md)
8. [Testing](testing.md)

### Using Nexus from Go

1. [Getting Started: Go](getting-started-go.md)
2. [Language Bindings](language-bindings.md#go)
3. [Observability with OpenTelemetry](observability-with-opentelemetry.md)
4. [Observability with OpenInference](observability-with-openinference.md)
5. [Core Concepts](concepts.md)
6. [Context Isolation](context-isolation.md)
7. [API Reference](api-reference.md)
8. [Testing](testing.md)

### Using Nexus from WebAssembly

1. [Getting Started: WebAssembly](getting-started-wasm.md)
2. [Language Bindings](language-bindings.md#webassembly)
3. [Observability with OpenTelemetry](observability-with-opentelemetry.md)
4. [Observability with OpenInference](observability-with-openinference.md)
5. [Core Concepts](concepts.md)
6. [Middleware Pipeline](middleware-pipeline.md)
7. [Context Isolation](context-isolation.md)
8. [Testing](testing.md)

### Using the Optimizer Runtime

1. [Optimizer Layer](optimizer-layer.md)
2. [Online Learning Engine](online-learning-engine.md)
3. [Context Isolation](context-isolation.md)
4. [Testing](testing.md)

### Contributing or Extending Bindings

1. [Architecture Overview](architecture.md)
2. [Language Bindings](language-bindings.md)
3. [Integration Best Practices](integration-best-practices.md)
4. [Testing](testing.md)
5. [Contributing Guide](../.github/CONTRIBUTING.md)

## Contents

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture.md) | High-level system design, binding layers, and data flow |
| [Core Concepts](concepts.md) | Scopes, handles, events, and the middleware pipeline |
| [API Reference](api-reference.md) | Core runtime function signatures shared across bindings |
| [Middleware Pipeline](middleware-pipeline.md) | Detailed pipeline ordering for tool and LLM calls |
| [LLM Codecs](llm-codecs.md) | Structured LLM request annotation, Codec authoring, and annotated intercepts |
| [Typed Wrappers](typed-wrappers.md) | Codec-based typed APIs for Python and Node.js |
| [Typed API Reference](typed-api-reference.md) | Function signatures and codec reference for `nat_nexus.typed` |
| [Context Isolation](context-isolation.md) | Multi-tenant and concurrent scope stack management |
| [ATIF Export](atif-export.md) | Agent Trajectory Interchange Format export |
| [Observability with OpenTelemetry](observability-with-opentelemetry.md) | OTLP subscriber setup, event mapping, and per-language examples |
| [Observability with OpenInference](observability-with-openinference.md) | OpenInference semantic mapping, Phoenix-oriented OTLP setup, and per-language examples |
| [Language Bindings](language-bindings.md) | Per-language usage guides and naming conventions |
| [Getting Started: Python](getting-started-python.md) | Minimal install-to-first-tool-call path for Python |
| [Getting Started: Node.js](getting-started-node.md) | Minimal install-to-first-tool-call path for Node.js |
| [Getting Started: Go](getting-started-go.md) | Minimal install-to-first-tool-call path for Go |
| [Getting Started: WebAssembly](getting-started-wasm.md) | Minimal install-to-first-tool-call path for WASM |
| [Recipes](recipes.md) | Task-oriented patterns for logging, ATIF, optimizer setup, and context propagation |
| [Optimizer Layer](optimizer-layer.md) | Dynamic optimizer config, built-in components, hosted plugins, and runtime lifecycle |
| [Optimizer API Reference](optimizer-api-reference.md) | Public optimizer config types, diagnostics, runtime helpers, and plugin registration APIs |
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
