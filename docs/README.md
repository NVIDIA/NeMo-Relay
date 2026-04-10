<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Documentation

NeMo Flow is a multi-language agent runtime framework providing execution scope management, lifecycle event tracking, and configurable middleware pipelines for tool and LLM calls.

This file is the canonical documentation index for the repository.

## Start Here

Choose the path that matches what you are trying to do:

### Using NeMo Flow from Python

1. [Getting Started: Python](getting-started-python.md)
2. [Language Bindings](language-bindings.md#python)
3. [Core Concepts](concepts.md)
4. [Middleware Pipeline](middleware-pipeline.md)
5. [LLM Codecs](llm-codecs.md)
6. [API Reference](api-reference.md)
7. [Observability with OpenTelemetry](observability-with-opentelemetry.md)
8. [Observability with OpenInference](observability-with-openinference.md)
9. [ATIF Export](atif-export.md)
10. [Testing](testing.md)

### Using NeMo Flow from Node.js

1. [Getting Started: Node.js](getting-started-node.md)
2. [Language Bindings](language-bindings.md#nodejs)
3. [Core Concepts](concepts.md)
4. [Middleware Pipeline](middleware-pipeline.md)
5. [LLM Codecs](llm-codecs.md)
6. [Typed Wrappers](typed-wrappers.md)
7. [Observability with OpenTelemetry](observability-with-opentelemetry.md)
8. [Observability with OpenInference](observability-with-openinference.md)
9. [Testing](testing.md)

### Using NeMo Flow from Go

1. [Getting Started: Go](getting-started-go.md)
2. [Language Bindings](language-bindings.md#go)
3. [Core Concepts](concepts.md)
4. [LLM Codecs](llm-codecs.md)
5. [Context Isolation](context-isolation.md)
6. [API Reference](api-reference.md)
7. [Observability with OpenTelemetry](observability-with-opentelemetry.md)
8. [Observability with OpenInference](observability-with-openinference.md)
9. [Testing](testing.md)

### Using NeMo Flow from WebAssembly

1. [Getting Started: WebAssembly](getting-started-wasm.md)
2. [Language Bindings](language-bindings.md#webassembly)
3. [Core Concepts](concepts.md)
4. [Middleware Pipeline](middleware-pipeline.md)
5. [LLM Codecs](llm-codecs.md)
6. [Context Isolation](context-isolation.md)
7. [Observability with OpenTelemetry](observability-with-opentelemetry.md)
8. [Observability with OpenInference](observability-with-openinference.md)
9. [Testing](testing.md)

### Using Adaptive Components

1. [Adaptive Layer](adaptive-layer.md)
2. [Adaptive API Reference](adaptive-api-reference.md)
3. [Online Learning Engine](online-learning-engine.md)
4. [Context Isolation](context-isolation.md)
5. [Testing](testing.md)

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
| [LLM Codecs](llm-codecs.md) | Request and response codecs, built-in API codecs (OpenAI Chat, OpenAI Responses, Anthropic Messages), event enrichment, and custom codec authoring |
| [Typed Wrappers](typed-wrappers.md) | Codec-based typed APIs for Python and Node.js |
| [Typed API Reference](typed-api-reference.md) | Function signatures and codec reference for `nemo_flow.typed` |
| [Context Isolation](context-isolation.md) | Multi-tenant and concurrent scope stack management |
| [ATIF Export](atif-export.md) | Agent Trajectory Interchange Format export |
| [Observability with OpenTelemetry](observability-with-opentelemetry.md) | OTLP subscriber setup, event mapping, and per-language examples |
| [Observability with OpenInference](observability-with-openinference.md) | OpenInference semantic mapping, Phoenix-oriented OTLP setup, and per-language examples |
| [Language Bindings](language-bindings.md) | Per-language usage guides and naming conventions |
| [Getting Started: Python](getting-started-python.md) | Minimal install-to-first-tool-call path for Python |
| [Getting Started: Node.js](getting-started-node.md) | Minimal install-to-first-tool-call path for Node.js |
| [Getting Started: Go](getting-started-go.md) | Minimal install-to-first-tool-call path for Go |
| [Getting Started: WebAssembly](getting-started-wasm.md) | Minimal install-to-first-tool-call path for WASM |
| [Recipes](recipes.md) | Task-oriented patterns for logging, ATIF, adaptive setup, and context propagation |
| [Adaptive Layer](adaptive-layer.md) | Dynamic adaptive config, flat adaptive sections, and plugin-host integration |
| [Adaptive API Reference](adaptive-api-reference.md) | Public adaptive config types, diagnostics, runtime helpers, and plugin registration APIs |
| [Online Learning Engine](online-learning-engine.md) | Prediction trie, sensitivity scoring, Redis persistence, and learner pipeline |
| [Testing](testing.md) | Test commands, helpers, coverage, and conventions |
| [Integration Best Practices](integration-best-practices.md) | Patterns for integrating NeMo Flow into agent frameworks |

See also: [Contributing Guide](../.github/CONTRIBUTING.md) for development setup, branch naming, and PR process.

## Quick Start

```python
import asyncio

import nemo_flow

async def amain():
    # Define your tool and LLM functions
    my_tool_func = lambda args: {**args, "result": "ok"}
    my_llm_func = lambda request: {**request.content, "response": "ok"}

    # Subscribe to lifecycle events
    nemo_flow.subscribers.register("logger", lambda event: print(event.name))

    # Run the agent inside a NeMo Flow scope
    with nemo_flow.scope.scope("my_agent", nemo_flow.ScopeType.Agent) as handle:
        # Execute an LLM call through the full middleware pipeline
        request = nemo_flow.LLMRequest(
            headers={"Authorization": "Bearer ..."},
            content={"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"},
        )
        response = await nemo_flow.llm.execute("gpt-4", request, my_llm_func)

        # Execute a tool call
        result = await nemo_flow.tools.execute("search", {"query": "example"}, my_tool_func)
        print(result)


asyncio.run(amain())
```
