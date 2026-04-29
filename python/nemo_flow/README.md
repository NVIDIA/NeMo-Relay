<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Python Package

`nemo-flow` is the NeMo Flow package for Python applications. It gives Python
code access to a portable agent runtime for execution scopes, middleware,
plugins, lifecycle events, adaptive behavior, and observability around tool and
LLM calls.

The package wraps the shared Rust runtime, so Python applications use the same
runtime semantics as the Rust and Node.js surfaces.

## Why Use It?

- 🧭 **Own execution context in Python**: Group agent, tool, and LLM work into
  one scope tree from Python application code.
- 🛡️ **Package policy around callbacks**: Use guardrails and intercepts to block
  work, sanitize observability payloads, rewrite requests, or wrap execution.
- 📡 **Emit one lifecycle stream**: Send runtime events to in-process
  subscribers, ATIF, OpenTelemetry, or OpenInference workflows.
- 🧩 **Integrate without a framework migration**: Wrap framework or provider
  callbacks while preserving the application’s orchestration model.

## What You Get

- ✅ **Scope, tool, and LLM helpers**: Managed boundaries that emit lifecycle
  events and run middleware in a consistent order.
- ✅ **Middleware APIs**: Guardrails and intercepts for tool and LLM requests,
  responses, and execution.
- ✅ **Subscribers and exporters**: Event consumers for observability and
  diagnostics.
- ✅ **Plugin and typed helpers**: Public modules for plugins, codecs, typed
  wrappers, and adaptive runtime behavior.
- ✅ **Shared Rust runtime semantics**: Python behavior aligned with the Rust
  and Node.js surfaces.

## Installation

Install the published package with `uv`:

```bash
uv add nemo-flow
```

If you are not using `uv`, install it with `pip`:

```bash
pip install nemo-flow
```

## Getting Started

Register a subscriber, create a scope, and emit a mark event:

```python
import nemo_flow


def on_event(event) -> None:
    print(f"{event.kind} {event.name}")


nemo_flow.subscribers.register("printer", on_event)

with nemo_flow.scope.scope("demo-agent", nemo_flow.ScopeType.Agent) as handle:
    nemo_flow.scope.event("initialized", handle=handle, data={"binding": "python"})

nemo_flow.subscribers.deregister("printer")
```

## Package Surface

The public package modules are:

- `nemo_flow.scope`
- `nemo_flow.tools`
- `nemo_flow.llm`
- `nemo_flow.guardrails`
- `nemo_flow.intercepts`
- `nemo_flow.subscribers`
- `nemo_flow.plugin`
- `nemo_flow.adaptive`
- `nemo_flow.typed`
- `nemo_flow.codecs`

The compiled extension is exposed as `nemo_flow._native`.

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
