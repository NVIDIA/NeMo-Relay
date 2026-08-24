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

# NeMo Relay Native Plugin SDK

`nemo-relay-plugin` is the Rust authoring SDK and stable ABI for trusted,
in-process NeMo Relay dynamic plugins. Use it to build a Rust `cdylib` that
Relay loads through the versioned native plugin interface.

Native plugins run in the Relay process and are not sandboxed. They should
depend on this crate rather than the host `nemo-relay` runtime crate, keeping
the dynamic-library boundary on the stable C-compatible ABI.

## Authoring Surface

| Surface | Role |
|---|---|
| `NativePlugin` | Defines plugin identity, configuration validation, registration, and multiple-component behavior without requiring an author to construct ABI tables. |
| `PluginContext` | Installs component-owned subscribers, guardrails, intercepts, continuations, and streams. |
| `PluginRuntime` | Emits marks and manages Relay-owned scopes and scope stacks through typed host helpers. |
| `nemo_relay_plugin!` | Exports the one versioned native entry point used by the loader. |
| Native ABI v4 | Keeps C-compatible host and plugin tables behind the safe Rust interface while the host retains frozen v3 and v2 tables for previously compiled plugins. |
| Typed async middleware | Drives guardrails, sanitizers, and intercepts on a per-component SDK-owned Tokio executor. Subscribers and raw ABI registrations remain synchronous. |
| Async continuations and streams | `ToolNext`, `LlmNext`, and `LlmStreamNext` support repeated or concurrent downstream calls. Streaming LLM continuations use a pull-based host handle. |
| Tool results | `ToolNext` returns `ToolExecutionResult`, which keeps an application result and optional annotation together. |
| Typed telemetry marks | `emit_mark_with_options` adds `DataSchema` and `LogSeverity`; `emit_metric` validates `MetricMeasurement` values before it emits the reserved Relay metric schema. |
| Runtime diagnostics | `PluginRuntime::runtime_diagnostics()` returns the active host-level `RuntimeDiagnostics` snapshot. Entries are ordered and available through `get(code)`. |

## Installation

Add the SDK to a Rust dynamic-plugin project:

```bash
cargo add nemo-relay-plugin serde_json
cargo add tokio@1 --features io-util,macros,rt,time
```

Configure the library as a dynamic library:

```toml
[lib]
crate-type = ["cdylib"]
```

## Getting Started

Implement `NativePlugin` and export a constructor symbol:

```rust
use nemo_relay_plugin::{Json, NativePlugin, PluginContext, Result};
use serde_json::Map;

struct ExamplePlugin;

impl NativePlugin for ExamplePlugin {
    fn plugin_kind(&self) -> &str {
        "example.native"
    }

    fn register(&mut self, _config: &Map<String, Json>, ctx: &mut PluginContext<'_>) -> Result<()> {
        ctx.register_subscriber("log-events", |event| {
            eprintln!("{}", event.name());
        })
    }
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, || ExamplePlugin);
```

Build the `cdylib`, describe its entry symbol and compatibility in a
`relay-plugin.toml` manifest, then register it through the Relay CLI. Refer to the
complete example for platform-specific artifact and manifest setup.

Typed async plugins require `compat.relay = ">=0.8.0,<1.0"`. Relay creates one
SDK-owned Tokio executor for each configured plugin component. It defaults to
two workers: enough for modest concurrent async I/O without broadly
oversubscribing the host. Increase the count only when measured I/O concurrency
leaves callbacks queued; lower it when the host runs many components or has a
tight CPU budget. Do not block these workers; use async I/O or
`tokio::task::spawn_blocking`.

Relay 0.8 establishes canonical tool results as the native API 1 baseline. Tool
callbacks and `ToolNext` return `ToolExecutionResult`, preserving an application result
and optional opaque annotation. Tool execution intercepts return the same pair plus
Relay-owned pending marks. The manifest contract remains `compat.native_api = "1"` and
the C host-table ABI remains v4, but plugins must rebuild and exclude pre-0.8 Relay
versions because the JSON result boundary changed.

Set a plugin-wide default in Rust, then let the component's TOML configuration
override it:

```rust
use nemo_relay_plugin::{NativeExecutorConfig, NativePlugin};

impl NativePlugin for ExamplePlugin {
    fn plugin_kind(&self) -> &str {
        "acme.example"
    }

    fn executor_config(&self) -> NativeExecutorConfig {
        NativeExecutorConfig { worker_threads: 4 }
    }

    // ... register and other trait methods ...
}
```

```toml
[[plugins.dynamic]]
manifest = "./relay-plugin.toml"

[plugins.dynamic.config.executor]
worker_threads = 4
```

The SDK validates that `worker_threads` is a positive integer. The default
`NativePlugin::executor_config_for_component` applies this override; plugins
can override that method when they need different configuration rules.

During plugin teardown, the SDK stops accepting new callbacks and drains
already accepted typed middleware before the plugin library unloads.

## Runtime Diagnostics

`PluginRuntime::runtime_diagnostics()` returns `RuntimeDiagnostic { code,
message, count }` entries for the active host configuration. Relay aggregates
repeated codes, retains the latest message, saturates counts, orders entries by
code, and caps the snapshot at 32 entries. The result is host-level and does
not attribute an entry to the plugin that emitted it. ABI v4 requires the
complete finalized host table. Plugins loaded through the frozen ABI-v3 or
ABI-v2 compatibility tables cannot use runtime diagnostics.

Relay scope context is restored around every poll of a registered middleware
future. Child tasks created with `tokio::spawn` do not automatically inherit
that scope context.
