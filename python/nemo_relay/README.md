<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay

`nemo-relay` is the NeMo Relay package for Python applications. It gives Python
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
  subscribers, Agent Trajectory Interchange Format (ATIF), OpenTelemetry, or
  OpenInference workflows.
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
uv add nemo-relay
```

If you are not using `uv`, install it with `pip`:

```bash
pip install nemo-relay
```

### Optional Dependencies

#### LangChain Integration

[LangChain](https://www.langchain.com/langchain) integration is available with the `langchain` extra:

```bash
# With uv
uv add "nemo-relay[langchain]"

# With pip
pip install "nemo-relay[langchain]"
```

#### LangGraph Integration

[LangGraph](https://www.langchain.com/langgraph) integration is available with the `langgraph` extra, this builds upon and includes the `langchain` extra as well.

```bash
# With uv
uv add "nemo-relay[langgraph]"

# With pip
pip install "nemo-relay[langgraph]"
```

#### Deep Agents Integration

[Deep Agents](https://www.langchain.com/deep-agents) integration is available
with the `deepagents` extra. This extra builds upon and includes the
`langgraph` and `langchain` extras.

```bash
# With uv
uv add "nemo-relay[deepagents]"

# With pip
pip install "nemo-relay[deepagents]"
```

#### LangChain NVIDIA Integration

The [LangChain NVIDIA](https://github.com/langchain-ai/langchain-nvidia) extra builds upon the `langchain` extra adding a compatible version of the `langchain-nvidia-ai-endpoints` package.

```bash
# With uv
uv add "nemo-relay[langchain-nvidia]"

# With pip
pip install "nemo-relay[langchain-nvidia]"
```

To install this along with the `langgraph` extra, use:

```bash
# With uv
uv add "nemo-relay[langgraph,langchain-nvidia]"
# With pip
pip install "nemo-relay[langgraph,langchain-nvidia]"
```

## Getting Started

Register a subscriber, create a scope, and emit a mark event:

```python
import nemo_relay


def on_event(event) -> None:
    print(f"{event.kind} {event.name}")


nemo_relay.subscribers.register("printer", on_event)

with nemo_relay.scope.scope("demo-agent", nemo_relay.ScopeType.Agent) as handle:
    nemo_relay.scope.event("initialized", handle=handle, data={"binding": "python"})

nemo_relay.subscribers.flush()
nemo_relay.subscribers.deregister("printer")
```

Native subscriber delivery is asynchronous, so call
`nemo_relay.subscribers.flush()` before you read subscriber output or exit.

For host integrations that need a serialized event shape, consume the
canonical JSON payload from the subscriber event object:

```python
import json
import nemo_relay


def on_event(event) -> None:
    payload = event.to_dict()
    print(payload["kind"], payload["name"])
    assert json.loads(event.to_json()) == payload


nemo_relay.subscribers.register("host-exporter", on_event)
try:
    with nemo_relay.scope.scope("demo-agent", nemo_relay.ScopeType.Agent):
        nemo_relay.scope.event("initialized", data={"binding": "python"})
finally:
    nemo_relay.subscribers.flush()
    nemo_relay.subscribers.deregister("host-exporter")
```

## Package Surface

The public package modules are:

- `nemo_relay.scope`
- `nemo_relay.tools`
- `nemo_relay.llm`
- `nemo_relay.guardrails`
- `nemo_relay.intercepts`
- `nemo_relay.subscribers`
- `nemo_relay.plugin`
- `nemo_relay.adaptive`
- `nemo_relay.observability`
- `nemo_relay.typed`
- `nemo_relay.codecs`

### Integrations

- `nemo_relay.integrations.langchain`
- `nemo_relay.integrations.langgraph`
- `nemo_relay.integrations.deepagents`

The compiled extension is exposed as `nemo_relay._native`.

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
