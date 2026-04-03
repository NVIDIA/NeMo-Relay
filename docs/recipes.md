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
import nat_nexus

nat_nexus.subscribers.register(
    "logger",
    lambda event: print(f"[{event.event_type}] {event.name} root={event.root_uuid}"),
)
```

Use this when you want a quick view of scope, tool, and LLM lifecycle events
without introducing a full exporter.

## Export ATIF Trajectories

```python
import nat_nexus

exporter = nat_nexus.AtifExporter(
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

## Register Scope-Local Middleware

```python
import nat_nexus

handle = nat_nexus.scope.push("session", nat_nexus.ScopeType.Agent)

nat_nexus.scope_local.register_tool_conditional_execution(
    handle,
    "session-guard",
    10,
    lambda name, args: "blocked" if name == "rm" else None,
)

try:
    # tool and LLM calls here see the scope-local middleware
    ...
finally:
    nat_nexus.scope.pop(handle)
```

Use this when middleware should apply only within one request, one tenant, or
one temporary agent session.

## Use Typed Wrappers

```python
import asyncio
from dataclasses import dataclass

import nat_nexus.typed as typed


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

## Enable NexusProxy with `InMemoryBackend`

```python
import asyncio

import nat_nexus


async def main() -> None:
    nat_nexus.proxy.set_use_proxy(True)
    nat_nexus.proxy.set_proxy_backend(nat_nexus.proxy.InMemoryBackend())
    nat_nexus.proxy.set_dynamo_intercept(True)
    await nat_nexus.proxy.ensure_proxy()


asyncio.run(main())
```

Use this for local development, tests, and single-process runs.

## Enable NexusProxy with `RedisBackend`

```python
import asyncio

import nat_nexus


async def main() -> None:
    backend = await nat_nexus.proxy.RedisBackend.connect(
        "redis://127.0.0.1:6379",
        "nexus:",
    )
    nat_nexus.proxy.set_use_proxy(True)
    nat_nexus.proxy.set_proxy_backend(backend)
    nat_nexus.proxy.set_dynamo_intercept(True)
    await nat_nexus.proxy.ensure_proxy()


asyncio.run(main())
```

This requires Redis support in the underlying build. See
[Proxy Layer](proxy-layer.md) and [Proxy API Reference](proxy-api-reference.md)
for the feature-flagged build requirements.

## Propagate Scope Context to Worker Threads

```python
from concurrent.futures import ThreadPoolExecutor

import nat_nexus


def worker(stack):
    nat_nexus.set_thread_scope_stack(stack)
    return nat_nexus.scope.get_handle().name


with nat_nexus.scope.scope("thread-demo", nat_nexus.ScopeType.Agent):
    stack = nat_nexus.propagate_scope_to_thread()
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
2. In Python async code, call `nat_nexus.get_scope_stack()` if an integration
   needs to force initialization before first use.
3. For worker threads, propagate the stack explicitly with
   `propagate_scope_to_thread()` and `set_thread_scope_stack(...)`.
4. For Go integrations, use `ScopeStack.Run(...)` when a goroutine needs an
   isolated bound scope stack.

## Related Docs

- [Getting Started: Python](getting-started-python.md)
- [Typed API Reference](typed-api-reference.md)
- [Proxy API Reference](proxy-api-reference.md)
- [Context Isolation](context-isolation.md)
