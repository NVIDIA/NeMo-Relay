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

## Why Use It?

- **Share the stable transport contract**: Use the `grpc-v1` service and
  message definitions accepted by Relay worker manifests.
- **Use generated Tonic bindings**: Access versioned client and server types
  from `v1` without generating protobuf code in a consumer project.
- **Keep data ownership clear**: Carry Relay DTOs in JSON envelopes backed by
  `nemo-relay-types`; protobuf owns transport control flow.

## What You Get

- **`WORKER_PROTOCOL_GRPC_V1`**: The stable `grpc-v1` protocol identifier.
- **`v1` module**: Generated `PluginWorker` and `RelayHostRuntime` gRPC
  clients, servers, services, and messages.
- **JSON envelope helpers**: `json_envelope` and `decode_json_envelope` for
  serializing Relay DTOs into protocol payloads.

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

let envelope = json_envelope("example.Payload@1", &json!({"ok": true}))?;
let payload: Value = decode_json_envelope(&envelope)?;

assert_eq!(WORKER_PROTOCOL_GRPC_V1, "grpc-v1");
assert_eq!(payload["ok"], true);
# Ok::<(), serde_json::Error>(())
```

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [Build Plugins guide](https://docs.nvidia.com/nemo/relay/build-plugins/about)
- [Rust worker SDK](../worker/README.md)
