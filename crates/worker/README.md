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

`nemo-relay-worker` is the Rust authoring SDK for out-of-process NeMo Relay
dynamic worker plugins. Use it when plugin code needs process isolation and
communicates with Relay through the versioned `grpc-v1` worker protocol.

## Why Use It?

- **Isolate plugin code**: Run custom runtime behavior outside the Relay host
  process.
- **Use typed registration APIs**: Implement `WorkerPlugin` and register
  subscribers, guardrails, or intercepts with `PluginContext`.
- **Call the host runtime**: Emit marks, manage scopes, and invoke middleware
  continuations through `PluginRuntime`.
- **Keep lifecycle managed**: Let Relay provide authenticated endpoints and
  start the worker with `serve_plugin`.

## What You Get

- **`WorkerPlugin`**: The plugin identity, validation, and registration
  contract.
- **`PluginContext`**: Typed registrations for all supported worker surfaces.
- **`PluginRuntime` and continuations**: Host-runtime callbacks and tool/LLM
  execution-chain helpers.
- **`serve_plugin`**: Tokio gRPC server startup using the Relay-provided worker
  environment.

## Installation

Add the SDK and Tokio to the worker project:

```bash
cargo add nemo-relay-worker
cargo add tokio --features macros,rt-multi-thread
```

## Getting Started

Implement a worker and serve it from an async entrypoint:

```rust
use nemo_relay_worker::{Json, PluginContext, Result, WorkerPlugin, serve_plugin};

struct ExampleWorker;

impl WorkerPlugin for ExampleWorker {
    fn plugin_id(&self) -> &str {
        "example.worker"
    }

    fn register(&self, ctx: &mut PluginContext, _config: &Json) -> Result<()> {
        ctx.register_tool_request_intercept("tag-request", 0, false, |_name, mut args| {
            if let Some(object) = args.as_object_mut() {
                object.insert("checked".into(), true.into());
            }
            Ok(args)
        });
        Ok(())
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

## Concurrency and Cancellation

Unary and streaming callbacks run concurrently. Cancellation is cooperative:
Relay sends `CancelInvocation` when a managed caller is cancelled, times out,
or stops consuming a stream, and the SDK aborts the matching async callback
task. An accepted cancellation confirms the task was found; it cannot prove
that arbitrary blocking work started by the callback has stopped.

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [Build Plugins guide](https://docs.nvidia.com/nemo/relay/build-plugins/about)
- [Python gRPC worker plugin example](../../examples/python-grpc-worker-plugin/README.md)
