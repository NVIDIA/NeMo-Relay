<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started: Python

This guide takes you from install to a minimal instrumented tool call and LLM
call using the Python binding.

All examples in this guide use:

- an active Nexus scope
- the managed execution APIs (`tools.execute(...)` and `llm.execute(...)`)

This guide intentionally does not use the low-level manual lifecycle APIs such
as `tools.call(...)` / `tools.call_end(...)` or `llm.call(...)` /
`llm.call_end(...)`.

## Prerequisites

- Python 3.11+
- Rust toolchain
- `uv`

## Install and Build

From the repository root:

```bash
uv sync
```

This builds the native extension and installs the Python package into the local
virtual environment.

## Minimal Scope and Tool Execution

Save the following as a Python script and run it with `uv run python ...`:

```python
import asyncio

import nat_nexus


async def main() -> None:
    nat_nexus.subscribers.register(
        "logger",
        lambda event: print(f"[{event.kind}] {event.name}"),
    )

    async def search_tool(args: dict) -> dict:
        return {"results": [f"echo:{args['query']}"]}

    with nat_nexus.scope.scope("quickstart-agent", nat_nexus.ScopeType.Agent):
        result = await nat_nexus.tools.execute(
            "search",
            {"query": "hello"},
            search_tool,
        )
        print(result)


asyncio.run(main())
```

## Minimal LLM Execution

```python
import asyncio

import nat_nexus


async def main() -> None:
    async def llm_func(request: nat_nexus.LLMRequest) -> dict:
        return {
            "messages": request.content["messages"],
            "response": "ok",
        }

    with nat_nexus.scope.scope("quickstart-agent", nat_nexus.ScopeType.Agent):
        request = nat_nexus.LLMRequest(
            headers={},
            content={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        response = await nat_nexus.llm.execute("gpt-4", request, llm_func)
        print(response)


asyncio.run(main())
```

## Add Logging or Export

- Use `nat_nexus.subscribers.register(...)` for console logging
- Use `nat_nexus.AtifExporter(...)` when you want to export trajectories
- Use `nat_nexus.OpenTelemetryConfig()` plus `nat_nexus.OpenTelemetrySubscriber(...)`
  when you want OTLP/OpenTelemetry traces
- Use `nat_nexus.OpenInferenceConfig()` plus `nat_nexus.OpenInferenceSubscriber(...)`
  when you want OTLP export with OpenInference semantics

## Common Errors

- `ModuleNotFoundError: nat_nexus._native`
  Re-run `uv sync` so the native extension is built.
- `RuntimeError` around scope handling
  Make sure `tools.execute(...)` and `llm.execute(...)` run inside an active
  `nat_nexus.scope.scope(...)` block or after an explicit `scope.push(...)`.

## Next Docs

- [Language Bindings](language-bindings.md#python)
- [API Reference](api-reference.md)
- [Typed Wrappers](typed-wrappers.md)
- [ATIF Export](atif-export.md)
- [Observability with OpenTelemetry](observability-with-opentelemetry.md)
- [Observability with OpenInference](observability-with-openinference.md)
