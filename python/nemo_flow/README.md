<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Flow)](https://github.com/NVIDIA/NeMo-Flow/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Flow/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Flow?color=green)](https://github.com/NVIDIA/NeMo-Flow/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Flow/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Flow)
[![PyPI](https://img.shields.io/pypi/v/nemo-flow?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-flow/)
[![npm node](https://img.shields.io/npm/v/nemo-flow-node?label=nemo-flow-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-node)
[![npm wasm](https://img.shields.io/npm/v/nemo-flow-wasm?label=nemo-flow-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow?label=nemo-flow&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow-adaptive?label=nemo-flow-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow-adaptive)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Flow)

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
  wrappers, adaptive runtime behavior, and observability plugin configuration.
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

### Optional Dependencies

#### LangChain Integration

[LangChain](https://www.langchain.com/langchain) integration is available with the `langchain` extra:

```bash
# With uv
uv add "nemo-flow[langchain]"

# With pip
pip install "nemo-flow[langchain]"
```

#### LangGraph Integration
[LangGraph](https://www.langchain.com/langgraph) integration is available with the `langgraph` extra, this builds upon and includes the `langchain` extra as well.

```bash
# With uv
uv add "nemo-flow[langgraph]"

# With pip
pip install "nemo-flow[langgraph]"
```

#### LangChain NVIDIA Integration
The [LangChain NVIDIA](https://github.com/langchain-ai/langchain-nvidia) extra builds upon the `langchain` extra adding a compatible version of the `langchain-nvidia-ai-endpoints` package.

```bash
# With uv
uv add "nemo-flow[langchain-nvidia]"

# With pip
pip install "nemo-flow[langchain-nvidia]"
```

To install this along with the `langgraph` extra, use:

```bash
# With uv
uv add nemo-flow[langgraph,langchain-nvidia]
# With pip
pip install nemo-flow[langgraph,langchain-nvidia]
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
- `nemo_flow.observability`
- `nemo_flow.typed`
- `nemo_flow.codecs`

### Integrations
- `nemo_flow.integrations.langchain`
- `nemo_flow.integrations.langgraph`

The compiled extension is exposed as `nemo_flow._native`.

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
