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
| Native ABI v8 | Keeps C-compatible host and plugin tables behind the safe Rust interface. Preserves v7 callbacks; adds scope inspection and a size-discovered diagnostics extension. |
| Typed async middleware | Drives guardrails, sanitizers, and intercepts on a per-component SDK-owned Tokio executor. Subscribers and raw ABI registrations remain synchronous. |
| Async continuations and streams | `ToolNext`, `LlmNext`, and `LlmStreamNext` support repeated or concurrent downstream calls. Streaming LLM continuations use a pull-based host handle. |
| Tool results | `ToolNext` returns `ToolExecutionResult`, which keeps an application result and optional annotation together. |
| Typed telemetry marks | `emit_mark_with_options` adds `DataSchema` and `LogSeverity`; `emit_metric` validates `MetricMeasurement` values before it emits the reserved Relay metric schema. |
| Legacy runtime diagnostics | `PluginRuntime::runtime_diagnostics()` returns the active host-level `RuntimeDiagnostics` report. Entries are ordered and available through `get(code)`. |
| Canonical native diagnostics | `PluginRuntime::diagnostics()` provides bounded authoring and separately authorized canonical snapshot and memory reads. |

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

Native plugins built with the 0.10 SDK require
`compat.relay = ">=0.10.0,<1.0"`. Relay creates one
SDK-owned Tokio executor for each configured plugin component. It defaults to
two workers: enough for modest concurrent async I/O without broadly
oversubscribing the host. Increase the count only when measured I/O concurrency
leaves callbacks queued; lower it when the host runs many components or has a
tight CPU budget. Do not block these workers; use async I/O or
`tokio::task::spawn_blocking`.

Relay 0.8 establishes canonical tool results as the native API 1 baseline. Tool
callbacks and `ToolNext` return `ToolExecutionResult`, preserving an application result
and optional opaque annotation. Tool execution intercepts return the same pair plus
Relay-owned pending marks. The manifest contract remains `compat.native_api = "1"`.
Relay 0.9 advances the C host-table ABI to v5 for `ToolExecutionContext`; the host keeps
the frozen v4 table for previously compiled plugins. Plugins that register a
context-aware tool execution intercept must rebuild and set
`compat.relay = ">=0.9.0,<1.0"`; the required registration is unavailable in the v4 host
table. Under Relay 0.9, typed async plugins that did not use this registration
could retain `compat.relay = ">=0.8.0,<1.0"`.

Relay 0.10 advances the internal table to ABI v7 and makes
`LlmExecutionContext` part of every non-streaming and streaming LLM execution callback.
Because this changes callback layouts, the 0.10 host rejects v2-v6 tables.
Rebuild every plugin with the 0.10 v7 SDK and set
`compat.relay = ">=0.10.0,<1.0"`. The authored manifest contract remains
`compat.native_api = "1"`.

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

## Canonical Native Diagnostics

`PluginRuntime::diagnostics()` records plugin-internal observations into the
host's selected diagnostics context. It shares core's counters, bounded circular
history, and detail pool. It does not start another service. The host resolves
runtime enablement through the existing configuration layers.

The SDK exposes three distinct diagnostic operations:

| API | Purpose | Read permission |
|---|---|---|
| `runtime_diagnostics()` | Read the existing legacy host report. | Retains its existing access contract. |
| `diagnostics().component_lazy(...)` and component `try_record_lazy(...)` | Register and record plugin-internal observations in the canonical store. | Recording does not require a context read grant. |
| `diagnostics().reader()` | Read canonical discovery, snapshots, and memory views. | Requires the host's whole-context `native_diagnostics_read_context` grant. |

The [native SDK guide](../../docs/build-plugins/native/runtime-events-and-scopes.mdx#record-internal-diagnostics)
covers authoring and reads. The [native ABI reference](../../docs/build-plugins/native/native-abi-reference.mdx#native-diagnostics-extension)
defines table discovery, buffer ownership, and compatibility behavior.

```rust
use nemo_relay_plugin::diagnostics::{
    ComponentDeclaration, DiagnosticCode, DiagnosticOutcome, FailureClass, Observation,
};
use std::num::NonZeroU64;

let diagnostics = ctx.runtime().diagnostics();
let codes = [DiagnosticCode::Extension {
    name: "lookup_failed",
    description: "An internal lookup failed",
    outcome: DiagnosticOutcome::Failure,
    allowed_classes: &[FailureClass::Unavailable],
}];
let backend = diagnostics.component_lazy(|| ComponentDeclaration {
    name: "backend",
    codes: &codes,
});
// Keep the component for the registration's lifetime. Diagnostic admission
// must not replace the operation's execution result.
if let Ok(backend) = &backend {
    let _ = backend.try_record_lazy(|| Observation::failure(
        1, // One-based position in the declared vocabulary.
        FailureClass::Unavailable,
        "Internal lookup unavailable",
        NonZeroU64::MIN,
    ));
}
```

Disabled and unsupported facades skip both factories. Enabled calls return an
acceptance receipt or a bounded adapter error. A count is an input delta: two
calls with counts 2 and 3 record five occurrences. Evidence eviction does not
undo acceptance. `delivery_stats()` reports local admission failures, explicit
host refusals, and uncertain internal failures separately; never retry an
uncertain recording automatically.

Use extensions for distinct internal operations. Common plugin built-in failure
codes are also accepted, with their declared failure classes. Relay continues to
own loading, validation, registration, and callback observations. Native-authored
observations are state-only: detail and correlation are never automatically
forwarded to operational logs, legacy reports, marks, subscribers, or exporters.
Native plugins remain trusted, in-process code, not a sandbox.

A component closes when its last clone is dropped or `close()` is called. Host
activation teardown revokes every handle and releases adapter backing; retained
counts remain subject to core retention. The service lifetime is host-owned.

### Authorized Reads

The host can separately grant whole-context canonical reads:

```toml
[plugins.policy.overrides."example.native"]
native_diagnostics_read_context = true
```

The key is the manifest plugin ID. The default is false, and the setting never
grants worker reads. A granted `diagnostics.reader()` exposes `discovery()`,
`snapshot(&query)`, `inspect_memory(&query)`, and JSON snapshot/memory methods,
using the shared diagnostics models and read reasons. This authority includes
other components' retained restricted data in the selected context. The host
checks permission and activation liveness on every read. Existing
`runtime_diagnostics()` remains the legacy host report reader.

### Bounds and Compatibility

The native adapter admits at most 32 concurrent authoring sessions per host
process, 8 live components per session, 32 codes per component, and 16 extension
codes plus 16 KiB of declaration text over a session's lifetime. Closing a
component releases its slot but does not replenish the cumulative declaration
allowance. Four process-wide data-call permits bound concurrent host work.

Registration and recording borrow fixed descriptors and UTF-8 spans. Aggregate
text is capped at 16 KiB per call; no recording JSON buffer or delivery queue is
created. Core reports detail/correlation truncation. Larger requests fail
explicitly instead of being silently shortened. Queries use core's query-byte
limit; reads allocate at most the advertised page cap before decoding. Returned
pages and arbitrary producer allocations are outside host memory budgets.

`capabilities()` reports limits, maximum session layout bytes, this activation's
current session backing, and registration parsing-array bytes. These exclude
allocator overhead, call frames, the bounded query decoder, core storage, SDK
buffers, and process RSS. Admission or allocation failure affects diagnostics,
not the existing plugin execution contract.

The host appends a size-discovered pointer to `NativeDiagnosticsApiV1` after the
frozen ABI-8 prefix. The diagnostics table has its own version and size; existing
ABI-7/8 plugins remain compatible and `compat.native_api = "1"` is unchanged.
A new SDK on a host without this extension exposes an unsupported inert facade.
Plugins that require diagnostics should declare a compatible Relay release range
that includes its availability; optional instrumentation can tolerate older hosts.
