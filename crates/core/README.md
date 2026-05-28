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

# NeMo Relay

`nemo-relay` is the core Rust SDK for NeMo Relay, a portable execution
runtime for agent systems. Use it when a Rust application, framework adapter,
or service needs one consistent way to scope, control, and observe tool and LLM
calls.

Rust is the source of truth for NeMo Relay runtime behavior. The Python and
Node.js bindings mirror the semantics exposed by this crate.

## Why Use It?

- 🧭 **Own Rust execution context**: Hierarchical scopes preserve parent-child
  relationships across tools, LLM calls, middleware, subscribers, and events.
- 🛡️ **Put policy around real calls**: Guardrails and intercepts can block work,
  sanitize observability payloads, rewrite requests, or wrap execution.
- 📡 **Emit one lifecycle stream**: Subscribers can consume canonical runtime
  events in-process or export them to Agent Trajectory Interchange Format
  (ATIF), OpenTelemetry, and OpenInference.
- 🧩 **Integrate without changing orchestration**: Wrap framework and provider
  callbacks while leaving scheduling, retries, memory, and result handling in
  the owning application.

## What You Get

- ✅ **Managed tool and LLM execution**: Run call boundaries through consistent
  lifecycle helpers and middleware ordering.
- ✅ **Scope-local runtime behavior**: Attach middleware and subscribers to the
  scope that owns them and clean them up when that scope closes.
- ✅ **Plugin primitives**: Register reusable runtime behavior configured from
  one shared plugin system.
- ✅ **Built-in observability plugin**: Configure first-party Agent Trajectory
  Observability Format (ATOF), Agent Trajectory Interchange Format (ATIF),
  OpenTelemetry, and OpenInference exporters from the core crate.
- ✅ **Codec and typed helpers**: Normalize provider requests and responses for
  framework integrations.
- ✅ **Binding source of truth**: Use the runtime semantics mirrored by the
  Python and Node.js bindings.

## Installation

Install the published crate in a Rust application:

```bash
cargo add nemo-relay serde_json
```

To add adaptive runtime behavior, install the companion crate too:

```bash
cargo add nemo-relay-adaptive
```

When consuming a local checkout, use path dependencies:

```toml
[dependencies]
nemo-relay = { path = "../NeMo-Relay/crates/core" }
nemo-relay-adaptive = { path = "../NeMo-Relay/crates/adaptive" }
serde_json = "1"
```

## Getting Started

The smallest useful workflow is to create a scope, emit a mark event, and close
the scope:

```rust
use nemo_relay::api::scope::{
    self, EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeType,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("demo-agent")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::empty())
            .data(json!({"binding": "rust"}))
            .build(),
    )?;

    scope::event(
        EmitMarkEventParams::builder()
            .name("initialized")
            .parent(&handle)
            .data(json!({"ok": true}))
            .build(),
    )?;

    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())?;
    Ok(())
}
```

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
