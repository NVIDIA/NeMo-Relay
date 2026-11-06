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

`nemo-relay-plugin` is the Rust SDK for trusted plugins that run in the NeMo Relay
process. Use it to build a Rust `cdylib`. The SDK adapts your callbacks to the
checked interface defined by `nemo-relay-plugin-abi`.

Native plugins run in the Relay process and are not sandboxed. They should
depend on this crate rather than `nemo-relay-plugin-host`, keeping
the dynamic-library boundary on the checked ABI-safe interface.

Implement the root `Plugin` trait to define identity, configuration validation,
and registration. Use `PluginContext` to install callbacks owned by the component.
Relay tracks those callbacks and removes them when activation fails or the
component closes.

## Authoring Surface

| Surface | Role |
|---|---|
| `Plugin` | Defines plugin identity, configuration validation, registration, and multiple-component behavior without requiring an author to construct ABI tables. |
| `PluginContext` | Installs component-owned subscribers, guardrails, intercepts, continuations, and streams. |
| `PluginRuntime` | Emits marks and manages Relay-owned scopes and scope stacks through typed host helpers. |
| `nemo_relay_plugin!` | Exports the standard checked bootstrap header with the SDK-required native contract. |
| Native API v1.0 | Uses `abi_stable` checked layouts, typed values, and `async-ffi` foreign futures. |
| Typed async middleware | Polls guardrails, sanitizers, and intercepts directly on the caller with restored Relay scope. Tokio support is opt-in and shared by the plugin's components. |
| Async continuations and streams | `ToolNext`, `LlmNext`, and `LlmStreamNext` support repeated or concurrent downstream calls. Streaming LLM continuations use a pull-based host handle. `register_synchronous_stream_callback` accepts synchronous factories that return checked stream descriptors. |
| Tool results | `ToolNext` returns `ToolExecutionResult`, which keeps an application result and optional annotation together. |
| Typed telemetry marks | `emit_mark_with_options` adds `DataSchema` and `LogSeverity`; `emit_metric` validates `MetricMeasurement` values before it emits the reserved Relay metric schema. |
| Legacy runtime diagnostics | `PluginRuntime::runtime_diagnostics()` returns the active host-level `RuntimeDiagnostics` report. Entries are ordered and available through `get(code)`. |
| Canonical diagnostics | `PluginRuntime::diagnostics()` records internal observations and offers separately authorized snapshot and memory reads. |

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

Implement `Plugin` and export its checked bootstrap:

```rust
use nemo_relay_plugin::{Json, Plugin, PluginContext};
use serde_json::Map;

struct ExamplePlugin;

impl Plugin for ExamplePlugin {
    fn plugin_kind(&self) -> &str {
        "example.native"
    }

    fn register<'a>(
        &'a self,
        _config: &'a Map<String, Json>,
        ctx: &'a mut PluginContext<'_>,
    ) -> nemo_relay_plugin::PluginFuture<'a, ()> {
        Box::pin(async move {
            ctx.register_subscriber("log-events", |event| {
                eprintln!("{}", event.name());
            })?;
            Ok(())
        })
    }
}

nemo_relay_plugin::nemo_relay_plugin!(kind = "example.native", || ExamplePlugin);
```

Build the `cdylib`, declare `compat.native_api = "v1.0"`, `compat.relay`, and `load.library` in a
`relay-plugin.toml` manifest, then register it through the Relay CLI. Refer to the
complete example for platform-specific artifact and manifest setup.

The checked native plugin example targets the 1.0.0 SDK and declares
`compat.relay = ">=1.0.0,<2.0"`. The SDK defaults to inline middleware execution and
creates no plugin Tokio runtime. Opt into a shared library-local runtime when a
component needs Tokio timers, networking, or background tasks. Keep inline callbacks
nonblocking.

Tool continuations return `ToolExecutionResult`, preserving the application result
and optional opaque annotation. Execution intercepts also carry Relay pending marks.
LLM execution contexts preserve request/response codec identity and capabilities;
streaming execution exposes request codec operations.

Opt into a shared Tokio runtime in Rust or through the component configuration:

```rust
use nemo_relay_plugin::{NativeExecutorConfig, Plugin};

impl Plugin for ExamplePlugin {
    fn plugin_kind(&self) -> &str {
        "acme.example"
    }

    fn executor_config(&self) -> Option<NativeExecutorConfig> {
        Some(NativeExecutorConfig { worker_threads: 4 })
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

The SDK validates that `worker_threads` is a positive integer. An explicit executor
object enables Tokio support; its default worker count is two. Active Tokio-enabled
components of the same plugin must agree on the worker count and share the runtime.
`Plugin::executor_config_for_component` resolves each component's request.

Use `ctx.spawn` for component-owned background work, or capture the cloneable
`ctx.task_spawner()` in a registered callback. These helpers require Tokio support,
restore the captured Relay scope around each poll, and cancel tasks when the
component closes. Dropping a task's join handle detaches only the waiter. Direct
`tokio::spawn` calls bypass component ownership and are unsuitable for this work.

When a component closes, the SDK stops accepting callbacks, cancels its work,
closes streams, and drains tracked tasks. It releases the component's runtime
lease after cleanup. The last Tokio-enabled component stops the shared runtime.
Library code and staged dependencies stay loaded until process exit.

## Runtime Diagnostics

`PluginRuntime::runtime_diagnostics()` returns `RuntimeDiagnostic { code,
message, count }` entries for the active host configuration. Relay aggregates
repeated codes, retains the latest message, saturates counts, orders entries by
code, and caps the snapshot at 32 entries. The result is host-level and does
not attribute an entry to the plugin that emitted it. This getter is included in
the checked v1.0 runtime interface.

Relay scope context is restored around every poll of a registered middleware
future. Component-owned tasks created with `ctx.spawn` or a task spawner inherit
that scope context.

`ScopeStack::set_thread()` and `ScopeStackBinding::restore()` return `Result<()>`;
propagate failures when explicitly changing or restoring a thread's scope stack.
For operational logs, use `PluginRuntime::log` with `PluginLogLevel`, or the
`relay_error!`, `relay_warn!`, `relay_info!`, `relay_debug!`, and `relay_trace!` macros.

## Canonical Native Diagnostics

Use `PluginRuntime::diagnostics()` for plugin-internal observations. Recording
shares the host's selected context and does not require a read grant. Canonical
reads require the host's `native_diagnostics_read_context` permission, which
exposes the whole context and defaults to false. Disabled or unsupported facades
skip lazy factories; diagnostic failures must preserve the execution result.

Refer to [Native Plugin Diagnostics](../../docs/build-plugins/native/diagnostics.mdx)
for authoring, acceptance receipts, concurrency, buffer ownership, and permissions.
The [native ABI reference](../../docs/build-plugins/native/native-abi-reference.mdx)
describes the checked v1.0 interface. Diagnostics cross that boundary through
typed records and caller-owned bounded buffers. The
`runtime_diagnostics()` method provides a separate host-level report.

## Binary Compatibility and Cleanup

The loader uses the standard `abi_stable` library header to check the plugin's
interfaces. The export macro sets the minimum native contract from the build SDK.
A plugin built with SDK contract v1.1 requires a host that supports v1.1 or a
later minor in major 1. A host that supports v1.1 also supports v1.0 plugins.
The explicit native contract determines binary compatibility. The
`nemo-relay-plugin-abi` package follows Relay's release version; package version
changes do not change the contract. The plugin's `compat.relay` range separately
selects supported Relay releases.

Libraries and staged dependencies stay loaded until process exit. Closing a
component removes registrations, revokes capabilities, cancels work, drains tasks,
and shuts down executors. A failed close retains ownership for retry. Retained
runtime handles become inert. Callback panics are caught in their originating
library; destructors and foreign wakers must not panic.
