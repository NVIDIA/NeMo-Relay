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

# NeMo Relay Worker SDK

`nemo-relay-worker` is the Rust SDK for NeMo Relay worker plugins. A worker runs
in a separate process and calls Relay through `nvidia.nemo.relay.worker.v1`.

The protocol is `nvidia.nemo.relay.worker.v1`, with contract version `1.0.0`.
Set `compat.worker_protocol = "nvidia.nemo.relay.worker.v1"` and
`compat.relay = ">=1.0.0,<2.0"` in the plugin manifest.
Crate and wheel versions follow Relay releases.

Callbacks use shared application types. The SDK converts Relay-owned values to
protobuf messages for transport. It preserves application and provider JSON,
tool annotations, pending marks, codec access, cancellation, and scope ownership.

## Authoring Surface

The SDK provides the following types and helpers:

| Surface | Role |
|---|---|
| `Plugin` | Defines plugin identity, validation, registration, and multiple-component behavior in the worker process. |
| `PluginContext` | Installs typed handlers for all 17 supported registration surfaces. |
| `PluginRuntime` and continuations | Emit marks, manage scopes, and call the remaining tool or LLM execution chain through the authenticated host service. |
| `LlmExecutionContext` | Reports the selected request and completed-response codecs and provides codec operations while the callback is active. |
| Tool results | Preserve application results and opaque annotations across tool callbacks and continuations. |
| `serve_plugin` | Starts the Tokio gRPC server from the activation identity, local endpoints, and token supplied by Relay. |

This model keeps plugin dependencies and crashes outside the Relay process, while the
SDK retains the shared runtime contract and manages authentication, cancellation, and
shutdown.

## Propagating Continuation Errors

Propagate an unchanged `ToolNext`, `LlmNext`, or `LlmStreamNext` error with `?`
to preserve its downstream origin. `WorkerSdkError::Downstream` carries the
original SDK error and the continuation identity issued by the host. Construct a new callback error only
when reporting a distinct plugin failure.

The host uses this identity to exclude forwarded errors from plugin callback
failure counts. The same contract covers stream-opening errors and error chunks.
It adds no retained error ledger. If the identity is missing or does not match, the host uses its normal error
attribution rules.

## Installation

Building a Rust worker requires `protoc` 31.1 or later and Buf 1.75.0 or later.
Install both on `PATH`, or select them with `PROTOC` and `BUF`. The first build
needs access to the Buf Schema Registry to download the locked validation schema.

Add the SDK and Tokio to the worker project:

```bash
cargo add nemo-relay-worker
cargo add tokio --features macros,rt-multi-thread
```

## Configuration

Set `[load.max_message_bytes]` in `relay-plugin.toml`. Use `default_bytes` for
the default limit and `by_name` for limits on exact callback registration names.
A callback's limit also applies to its continuations. Other callbacks and control
RPCs keep their own defaults. Host environment overrides take priority. The
default limit is 4 MiB. Restart the host and workers after changing these settings.
Refer to [execution message limits](https://docs.nvidia.com/nemo/relay/build-plugins/workers/grpc-v1-protocol#configure-execution-message-size).

## Getting Started

Implement a worker and serve it from an async entrypoint:

```rust
use nemo_relay_worker::{Json, Plugin, PluginContext, Result, serve_plugin};

struct ExampleWorker;

impl Plugin for ExampleWorker {
    fn plugin_kind(&self) -> &str {
        "example.worker"
    }
    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn register<'a>(
        &'a self,
        _config: &'a serde_json::Map<String, Json>,
        ctx: &'a mut PluginContext,
    ) -> nemo_relay_worker::PluginFuture<'a, ()> {
        Box::pin(async move {
            ctx.register_tool_request_intercept(
                "tag-request",
                0,
                false,
                |_name, mut args| async move {
                    if let Some(object) = args.as_object_mut() {
                        object.insert("checked".into(), true.into());
                    }
                    Ok(args)
                },
            );
            Ok(())
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    serve_plugin(ExampleWorker).await
}
```

Relay supplies the socket, activation ID, and authentication token through the
worker environment. Use `serve_plugin` for Relay-spawned workers; explicit
server configuration is intended for tests and custom launchers.

`PluginRuntime::emit_mark` retains its original positional contract.
`emit_mark_with_options` adds an optional `DataSchema` and `LogSeverity`, and
`emit_metric` validates typed `MetricMeasurement` values before emitting the
reserved Relay metric schema.

`PluginRuntime::runtime_diagnostics().await` returns an ordered host-level
snapshot with `RuntimeDiagnostic { code, message, count }` entries and
`get(code)` lookup. Relay aggregates repeated codes, retains the latest
message, saturates counts, and returns at most 32 entries. It does not provide
per-plugin attribution. A host that does not implement this RPC returns `UNIMPLEMENTED`, which the SDK
reports as an unsupported runtime diagnostics error.

## Concurrency and Cancellation

Unary and streaming callbacks can run concurrently. Relay sends
`CancelInvocation` when a managed caller cancels, times out, or stops reading a
stream. The SDK aborts the matching async task and waits for its cleanup.
Acceptance confirms SDK cleanup for that callback. It does not prove that
blocking work started by plugin code has stopped.
