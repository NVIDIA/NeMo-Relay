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
[![npm wasm](https://img.shields.io/npm/v/nemo-relay-wasm?label=nemo-relay-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NVIDIA NeMo Relay

## What Is NeMo Relay?

NeMo Relay is a portable execution runtime for agent systems that already have a
framework, model provider, policy layer, or observability backend. It gives those
systems one consistent way to describe, control, and observe what happens when an
agent crosses a request, tool, or LLM boundary.

Agent applications rarely live inside one clean abstraction. A production stack
might combine NeMo Agent Toolkit, LangChain, LangGraph, provider SDKs, custom
harness code, NeMo Guardrails, tracing systems, and evaluation pipelines. NeMo
Relay sits underneath those choices as the shared runtime contract for scopes,
middleware, plugins, lifecycle events, adaptive behavior, and observability.

Built as a Rust core with primary Rust, Python, and Node.js bindings, NeMo Relay
lets applications keep their orchestration model while runtime behavior stays
consistent across frameworks and languages.

## Why Use It?

- 🧭 **Own execution context across the whole agent run**: Hierarchical scopes
  attach tools, LLM calls, middleware, subscribers, and events to the same
  parent-child execution tree.
- 🛡️ **Package policy once**: Guardrails and intercepts can block work, sanitize
  observability payloads, transform requests, or wrap execution without
  rewriting every call site.
- 📡 **Emit one lifecycle stream**: Subscribers consume canonical runtime events
  in-process or export them as [ATIF v1.6](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)
  trajectories, OpenTelemetry traces, or OpenInference-compatible traces.
- 🧩 **Integrate without a framework migration**: NeMo Relay can sit below NeMo
  ecosystem components, third-party agent frameworks, provider adapters, or
  direct application code.
- ⚙️ **Install reusable runtime behavior**: Plugins configure middleware,
  subscribers, adaptive components, observability exporters, and custom runtime
  behavior from one shared system.

## What You Get

- ✅ **Managed tool and LLM execution**: Run call boundaries through consistent
  lifecycle helpers and middleware ordering.
- ✅ **Concurrent request isolation**: Keep request-local middleware and
  subscribers attached to the scope that owns them, then clean them up when that
  scope closes.
- ✅ **Multi-language semantics**: Use the same runtime model from Rust, Python,
  and Node.js.
- ✅ **Observability-ready events**: Preserve model metadata, tool call IDs,
  inputs, outputs, scope relationships, and lifecycle timing for downstream
  analysis.
- ✅ **Built-in observability plugin**: Configure Agent Trajectory Observability
  Format (ATOF), ATIF, OpenTelemetry, and OpenInference exporters without
  registering subscribers by hand.
- ✅ **Extension points for framework authors**: Wrap stable tool and provider
  callbacks while preserving framework-owned scheduling, retries, memory, and
  result handling.

```mermaid
flowchart LR
    App[Application or Framework]

    subgraph Runtime[NeMo Relay Runtime]
        direction TB
        Scopes[Scopes]
        Middleware[Middleware]
        Plugins[Plugins]
        Events[Lifecycle Events]
    end

    Output[Subscribers and Exporters]

    App --> Scopes
    App --> Middleware
    Plugins --> Middleware
    Scopes --> Events
    Middleware --> Events
    Events --> Output
```

## Installation

Install the published package for your language:

```bash
# Rust
cargo add nemo-relay

# Python
uv add nemo-relay

# Node.js
npm install nemo-relay-node
```

The NeMo Relay CLI is offered as a separate crate:

```bash
cargo install nemo-relay-cli
```

For source builds, testing, and contribution workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

End-user documentation lives at [docs.nvidia.com/nemo/relay](https://docs.nvidia.com/nemo/relay).

The primary documentation track covers Rust, Python, and Node.js.

The Go, WebAssembly, and raw FFI surfaces are currently experimental and remain source-first under
`go/nemo_relay`, `crates/wasm`, and `crates/ffi`.

## Binding Status

The table below summarizes the support level for each binding surface.

| Binding | Status | Notes |
|---|---|---|
| Python | ✅ Fully Supported | Fully documented with Quick Start and Guides |
| Node.js | ✅ Fully Supported | Fully documented with Quick Start and Guides  |
| Rust | ✅ Fully Supported | Fully documented with Quick Start and Guides  |
| NeMo Relay CLI | 🚧 Experimental | Install with `cargo install nemo-relay-cli`. |
| Go | 🚧 Experimental | Source-first under `go/nemo_relay`. |
| WebAssembly | 🚧 Experimental | Source-first under `crates/wasm`. |
| FFI | 🚧 Experimental | Source-first under `crates/ffi`. |

## Agent Harness Support

NeMo Relay CLI offers experimental support for several agent harnesses.
Refer to the NeMo Relay CLI documentation for additional information.

Below is our support matrix for agent harnesses.

| Agent | Observability | Security | Optimization | Notes |
|:--|:--:|:--:|:--:|:--|
| Claude Code | ✅ Yes | ❌ No |  ❌ No | Observability only; no known issues |
| Codex | ✅ Yes | ❌ No |  ❌ No | Observability only; missing some necessary hooks for full features |
| Hermes Agent | ✅ Yes | ❌ No |  ❌ No | Observability only; no known issues |
| Cursor | ✅ Yes | ❌ No |  ❌ No | Observability only; not feature-rich, missing hooks under `cursor-agent` |

## Third-Party Integrations

Some framework integrations are maintained as packages in this repository. Other
sample integrations are maintained as patch sets against upstream projects.

### Public API Integrations

Some integrations can be implemented using public APIs without patching. Public
API-based integrations live under language-specific integration packages such as
`python/nemo_relay/integrations/` and `integrations/`.

Below is the support matrix for our public API integrations.

| Agent / Library | Observability | Security | Optimization | Notes |
|:--|:--:|:--:|:--:|:--|
| LangChain | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped Tool and LLM calling |
| LangGraph | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped Tool and LLM calling |
| Deep Agents | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped Tool and LLM calling |
| OpenClaw | ✅ Yes | ❌ No |  ❌ No | Observability support; missing middleware for wrapped execution |

#### LangChain

The Python `nemo-relay` package ships several extras that offer comprehensive
middleware support for the following packages:

- LangChain
- LangGraph
- Deep Agents

See the [Python package README](python/nemo_relay/README.md) for more information.

#### OpenClaw

An OpenClaw plugin is available as a Node package `nemo-relay-openclaw`.
It relies on OpenClaw public plugin hooks plus the generic NeMo Relay plugin
configuration shape to export telemetry. See the
[OpenClaw package README](integrations/openclaw/README.md) for more information.

### Patch-based Integrations

Patch-based integrations offer experimental support. Our roadmap includes switching over to first-party plugins and packages where upstream extension points allow it.

Use [third_party/README.md](third_party/README.md) for the clone, checkout, and
patch-application workflow for those integrations.

The following table summarizes maintained third-party integrations and whether each provides observability, security, and optimization support.

| Integration | Observability | Security | Optimization | Notes |
|:---|:---:|:---:|:---:|:---|
| [LangChain](third_party/README-langchain.md), [LangGraph](third_party/README-langgraph.md), [LangChain NVIDIA](third_party/README-langchain-nvidia.md) | ✅ Yes | ✅ Yes | ✅ Yes | Directly patches behavior into code requiring no middleware |
| [opencode](third_party/README-opencode.md) | ✅ Yes | ✅ Yes | ✅ Yes | Directly patches behavior into code |
| [OpenClaw](third_party/README-openclaw.md) | ✅ Yes | ✅ Yes | ✅ Yes | Adds new middleware support to OpenClaw and a built-in plugin |
| [Hermes Agent](third_party/README-hermes-agent.md) | ✅ Yes | ✅ Yes | ✅ Yes | Directly patches behavior into code |

## Roadmap

The following roadmap outlines planned features and integrations for upcoming releases.

- [ ] NemoClaw support and integration for managed tool and LLM execution flows.
- [ ] Deeper NVIDIA NeMo ecosystem integration across agent, guardrail, evaluation, and observability workflows.
- [ ] Expanded adaptive optimization capabilities for performance-aware scheduling, hints, and cache behavior.
- [ ] First-party plugins and/or packages for common agent runtimes and frameworks.

## License

NeMo Relay is licensed under the [Apache License 2.0](LICENSE).
