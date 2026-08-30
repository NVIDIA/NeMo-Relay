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

# NeMo Relay Worker Protocol

`nemo-relay-worker-proto` provides the generated Rust bindings for the
versioned gRPC protocol used by NeMo Relay out-of-process worker plugins. It is
a protocol dependency for worker SDKs and hosts, not the usual entry point for
authoring a plugin.

Use `nemo-relay-worker` to author Rust workers. Depend on this crate directly
only when implementing another worker SDK, a custom host, or protocol-level
tooling.

Relay 0.8 establishes canonical tool results as the `grpc-v1` baseline. Workers built
for earlier releases must regenerate their bindings, rebuild, and declare
`compat.relay` beginning at `0.8.0`. `ToolNext` returns `ToolExecutionResultResponse`,
and tool execution intercepts use structural `ToolExecutionInterceptOutcome` messages.

## Protocol Surface

| Surface | Role |
|---|---|
| `WORKER_PROTOCOL_GRPC_V1` | Identifies the stable protocol accepted by Relay worker manifests. |
| `v1` module | Exposes generated `PluginWorker` and `RelayHostRuntime` Tonic clients, servers, services, and messages without regenerating protobuf in a consumer. |
| JSON envelope helpers | Serialize Relay DTOs through `json_envelope` and `decode_json_envelope`, keeping protobuf responsible for transport flow rather than runtime data modeling. |
| JSON value helpers | Serialize application-owned fields inside structural tool-result messages through `json_value` and `decode_json_value`. |
| Tool results | `ToolNext` returns `ToolExecutionResultResponse`, and `ToolExecutionInterceptResult` returns `ToolExecutionInterceptOutcome`. Both preserve the application result and optional annotation. Intercept outcomes also include ordered pending marks. These fields use lossless protobuf `JsonValue` wrappers rather than `google.protobuf.Value`. |
| Mark options | `EmitMarkRequest.data_schema` carries a `nemo.relay.DataSchema@1` envelope, `severity` carries the log severity, and `category` carries an optional semantic mark category. Omitting these fields preserves legacy behavior. |
| Runtime diagnostics | Authenticated `GetRuntimeDiagnostics` returns a bounded active-host `{ code, message, count }` snapshot. Older hosts return gRPC `UNIMPLEMENTED`. |

## Installation

Add the protocol crate when building protocol-level integrations:

```bash
cargo add nemo-relay-worker-proto
```

## Getting Started

Use the shared protocol identifier and JSON envelope helpers:

```rust
use nemo_relay_worker_proto::{WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, json_envelope};
use serde_json::{Value, json};

fn main() -> Result<(), serde_json::Error> {
    let envelope = json_envelope("example.Payload@1", &json!({"ok": true}))?;
    let payload: Value = decode_json_envelope(&envelope)?;

    assert_eq!(WORKER_PROTOCOL_GRPC_V1, "grpc-v1");
    assert_eq!(payload["ok"], true);
    Ok(())
}
```
