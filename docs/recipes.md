<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recipes

These recipes show common integration patterns using compact examples. Unless
noted otherwise, examples use the Python binding because it is the shortest path
to demonstrate the runtime behavior.

## Add Event Logging

```python
import nemo_flow

nemo_flow.subscribers.register(
    "logger",
    lambda event: print(f"[{event.kind}] {event.name} parent={event.parent_uuid}"),
)
```

Use this when you want a quick view of scope, tool, and LLM lifecycle events
without introducing a full exporter.

## Export ATIF Trajectories

```python
import nemo_flow

exporter = nemo_flow.AtifExporter(
    session_id="session-001",
    agent_name="demo-agent",
    agent_version="1.0",
    model_name="gpt-4",
)
exporter.register("atif")

# ... run your agent workload ...

trajectory = exporter.export()
trajectory_json = exporter.export_json()
exporter.clear()
exporter.deregister("atif")
```

Read [ATIF Export](atif-export.md) when you need the schema mapping details.

## Export Traces to OpenTelemetry

```rust
use nemo_flow_otel::{OpenTelemetryConfig, OpenTelemetrySubscriber};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::http_binary("demo-agent")
            .with_endpoint("http://localhost:4318/v1/traces")
            .with_service_version("0.1.0"),
    )?;

    subscriber.register("otel")?;

    // ... run NeMo Flow-instrumented work here ...

    subscriber.deregister("otel")?;
    subscriber.force_flush()?;
    subscriber.shutdown()?;
    Ok(())
}
```

Use this when you want NeMo Flow scopes, tool calls, LLM calls, and mark events to
show up in an OTLP-compatible backend such as the OpenTelemetry Collector,
Jaeger, Tempo, or Honeycomb. For config fields, event mapping, lifecycle
guidance, and binding-specific examples, see
[Observability with OpenTelemetry](observability-with-opentelemetry.md).

If you need Phoenix or another OpenInference-oriented backend instead, use
`nemo-flow-openinference` and
[Observability with OpenInference](observability-with-openinference.md).

## Register Scope-Local Middleware

```python
import nemo_flow

handle = nemo_flow.scope.push("session", nemo_flow.ScopeType.Agent)

nemo_flow.scope_local.register_tool_conditional_execution(
    handle,
    "session-guard",
    10,
    lambda name, args: "blocked" if name == "rm" else None,
)

try:
    # tool and LLM calls here see the scope-local middleware
    ...
finally:
    nemo_flow.scope.pop(handle)
```

Use this when middleware should apply only within one request, one tenant, or
one temporary agent session.

## Use Typed Wrappers

```python
import asyncio
from dataclasses import dataclass

import nemo_flow.typed as typed


@dataclass
class SearchArgs:
    query: str


@dataclass
class SearchResult:
    results: list[str]


async def main() -> None:
    result = await typed.tool_execute(
        "search",
        SearchArgs(query="hello"),
        lambda args: SearchResult(results=[f"echo:{args.query}"]),
        typed.DataclassCodec(SearchArgs),
        typed.DataclassCodec(SearchResult),
    )
    print(result)


asyncio.run(main())
```

Use this when you want middleware to stay JSON-based while application code
works with dataclasses or Pydantic models.

## Enable Adaptive with In-Memory State

```python
import asyncio

import nemo_flow
from nemo_flow import adaptive, plugin


async def main() -> None:
    config = plugin.PluginConfig(
        components=[
            adaptive.ComponentSpec(
                adaptive.AdaptiveConfig(
                    state=adaptive.StateConfig(
                        backend=adaptive.BackendSpec.in_memory()
                    ),
                    telemetry=adaptive.TelemetryConfig(
                        learners=["latency_sensitivity"]
                    ),
                    adaptive_hints=adaptive.AdaptiveHintsConfig(),
                    tool_parallelism=adaptive.ToolParallelismConfig(),
                )
            )
        ]
    )
    await plugin.initialize(config)


asyncio.run(main())
```

Use this for local development, tests, and single-process runs.

## Enable Adaptive with Redis State

```python
import asyncio

import nemo_flow
from nemo_flow import adaptive, plugin


async def main() -> None:
    config = plugin.PluginConfig(
        components=[
            adaptive.ComponentSpec(
                adaptive.AdaptiveConfig(
                    state=adaptive.StateConfig(
                        backend=adaptive.BackendSpec.redis(
                            "redis://127.0.0.1:6379",
                            "nemo_flow:",
                        )
                    ),
                    telemetry=adaptive.TelemetryConfig(
                        learners=["latency_sensitivity"]
                    ),
                    adaptive_hints=adaptive.AdaptiveHintsConfig(),
                    tool_parallelism=adaptive.ToolParallelismConfig(),
                )
            )
        ]
    )
    await plugin.initialize(config)


asyncio.run(main())
```

This requires Redis support in the underlying build. See
[Adaptive Layer](adaptive-layer.md) and [Adaptive API Reference](adaptive-api-reference.md)
for the feature-flagged build requirements.

## Propagate Scope Context to Worker Threads

```python
from concurrent.futures import ThreadPoolExecutor

import nemo_flow


def worker(stack):
    nemo_flow.set_thread_scope_stack(stack)
    return nemo_flow.scope.get_handle().name


with nemo_flow.scope.scope("thread-demo", nemo_flow.ScopeType.Agent):
    stack = nemo_flow.propagate_scope_to_thread()
    with ThreadPoolExecutor() as pool:
        current_name = pool.submit(worker, stack).result()
        print(current_name)
```

Use this when work leaves the current asyncio task or Python thread.

## Troubleshoot a Missing Scope Stack

Symptoms:

- `RuntimeError` about the scope stack being unavailable
- middleware or subscribers not seeing the expected active scope

Checklist:

1. Ensure you pushed a scope before calling managed tool or LLM APIs.
2. In Python async code, call `nemo_flow.get_scope_stack()` if an integration
   needs to force initialization before first use.
3. For worker threads, propagate the stack explicitly with
   `propagate_scope_to_thread()` and `set_thread_scope_stack(...)`.
4. For Go integrations, use `ScopeStack.Run(...)` when a goroutine needs an
   isolated bound scope stack.

## Related Docs

- [Getting Started: Python](getting-started-python.md)
- [Typed API Reference](typed-api-reference.md)
- [Adaptive API Reference](adaptive-api-reference.md)
- [Context Isolation](context-isolation.md)
- [Observability with OpenTelemetry](observability-with-opentelemetry.md)
- [Observability with OpenInference](observability-with-openinference.md)
