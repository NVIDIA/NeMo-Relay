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

`nemo-relay-worker-proto` provides Rust messages, clients, and servers for worker
gRPC calls. Hosts and worker SDKs depend on this crate.

Use `nemo-relay-worker` to author Rust workers. Depend on this crate directly
only when implementing another worker SDK, a custom host, or protocol-level
tooling.

The new package name and protocol identity are `nvidia.nemo.relay.worker.v1`.
Its contract version is `1.0.0`. The host, Rust SDK, and Python SDK use this
contract. Upgrade and rebuild them together. The old `grpc-v1` schema and Rust
`v1` module have been removed. Rebuild old workers and update their manifests.

## Protocol Surface

The crate provides these modules and checks:

| Surface | Role |
| --- | --- |
| `nvidia_worker_v1` | Generated clients, servers, and messages for the worker, host-runtime, and diagnostic services. |
| `conversions` | Convert shared domain values to and from protobuf messages. Codec data uses the native SDK's JSON representation. |
| `validation` | Check schema rules, JSON syntax, timestamps, and durations before calling callbacks. |
| Message limits | Limit sent and received bytes by callback name before decoding. |
| Open payloads | `JsonValue` and `JsonObject` preserve application and provider data, including null and large integers. |
| Closed contracts | Typed events, common LLM annotations, outcomes, pending marks, diagnostics, scopes, and codec capabilities. |

Refer to [the contract](CONTRACT.md) for field presence, lifecycle, authorization,
generation, and compatibility rules. Crate and wheel versions are independent
of contract versions. The [Rust conversion guide](CONTRACT.md#rust-domain-conversions)
explains mapping modes, representation differences, and how to update adapters.

## Installation

Install the Protocol Buffers compiler (`protoc`) 31.1 or newer on PATH (or select
it with `PROTOC`). The validation dependency builds its own bindings and needs
this separate compiler. This crate uses a bundled compiler for its own schema.
CI uses version 31.1 for both.

Install [Buf](https://buf.build/docs/installation/) 1.75.0 or newer before building
this crate from source. Set `BUF` to select an executable outside PATH. The build
downloads the exact Protovalidate version recorded in `buf.lock` into Cargo's
build directory. The first build needs access to the Buf Schema Registry.

Add the protocol crate to your integration with this command:

```bash
cargo add nemo-relay-worker-proto
```

## Getting Started

Use the new protocol identity and typed converters as shown in this example:

```rust
use nemo_relay_worker_proto::{conversions::{encode, decode}, nvidia_worker_v1::{JsonValue, WORKER_PROTOCOL, CONTRACT_VERSION}};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wire: JsonValue = encode(json!({"ok": true}))?;
    let payload: Value = decode(wire)?;
    assert_eq!(WORKER_PROTOCOL, "nvidia.nemo.relay.worker.v1");
    assert_eq!(CONTRACT_VERSION, "1.0.0");
    assert_eq!(payload["ok"], true);
    Ok(())
}
```
