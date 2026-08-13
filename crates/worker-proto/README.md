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

Relay 0.8 establishes the canonical tool-result contract as the `grpc-v1`
baseline. Workers built for an earlier Relay release must be rebuilt, and their
manifests must declare a `compat.relay` range beginning at `0.8.0` or later.
The protocol identifier and protobuf package remain `grpc-v1` and
`nemo.relay.worker.v1`, respectively. However, the generated protobuf API
changes at the tool-result boundary: `ToolNext` returns
`ToolExecutionResultResponse`, and `ToolExecutionInterceptResult.outcome` is a
typed `ToolExecutionInterceptOutcome`. Rebuild every worker against the Relay
0.8 protocol definitions.

## Why Use It?

- **Share the stable transport contract**: Use the `grpc-v1` service and
  message definitions accepted by Relay worker manifests.
- **Use generated Tonic bindings**: Access versioned client and server types
  from `v1` without generating protobuf code in a consumer project.
- **Keep data ownership clear**: Use structural protobuf wrappers for tool
  results while preserving open application payloads as lossless JSON bytes.
  Other Relay DTOs continue to use JSON envelopes backed by
  `nemo-relay-types`.

## What You Get

- **`WORKER_PROTOCOL_GRPC_V1`**: The stable `grpc-v1` protocol identifier.
- **`v1` module**: Generated `PluginWorker` and `RelayHostRuntime` gRPC
  clients, servers, services, and messages.
- **JSON envelope helpers**: `json_envelope` and `decode_json_envelope` for
  serializing Relay DTOs into protocol payloads.
- **JSON value helpers**: `json_value` and `decode_json_value` for the opaque
  application values inside structural tool-result messages.

## Structural Tool Result Contract

The `grpc-v1` tool-result boundary uses these generated message types:

| Protocol Location | Protobuf Type |
| --- | --- |
| Successful `RelayHostRuntime.ToolNext` response | `ToolExecutionResultResponse.value` containing `ToolExecutionResult` |
| `ToolExecutionInterceptResult.outcome` | `ToolExecutionInterceptOutcome` |

Both messages define `result` and optional `annotation` fields. Intercept
outcomes also carry their ordered `pending_marks` as one JSON array. Arbitrary
JSON values use `JsonValue`, whose bytes contain exactly one JSON value; this
preserves JSON integers and other application data without the numeric coercion
of `google.protobuf.Value`. Hosts and SDKs reject a missing required `result`
or invalid JSON bytes. JSON null annotations normalize to absence.

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

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [Build Plugins guide](https://docs.nvidia.com/nemo/relay/build-plugins/about)
- [Rust worker SDK](https://github.com/NVIDIA/NeMo-Relay/blob/main/crates/worker/README.md)
