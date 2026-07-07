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

# NeMo Relay Types

`nemo-relay-types` provides the shared serializable data model for NeMo Relay.
Use it when a Rust integration, plugin SDK, or protocol implementation needs
the same event, scope, tool, LLM, codec, and plugin-diagnostic types as Relay.

This crate intentionally contains no runtime registries, dynamic loading,
exporters, or process-global state. Applications normally depend on
`nemo-relay`; this crate is the lower-level contract shared by the runtime and
authoring SDKs.

## Why Use It?

- **Keep wire shapes consistent**: Exchange Relay DTOs without duplicating
  serializable event, request, response, and diagnostic models.
- **Share one JSON representation**: Use `Json`, an alias for
  `serde_json::Value`, across Relay-facing payloads.
- **Build SDKs without the runtime**: Depend on data contracts without pulling
  in runtime behavior or global state.

## What You Get

- **`api` module**: Event, scope, tool, and LLM DTOs and attributes.
- **`codec` module**: Normalized LLM request and response annotations.
- **`plugin` module**: Structured plugin configuration diagnostics.
- **Optional `schema` feature**: `schemars` implementations for supported
  serializable types.

## Installation

Add the crate when implementing a Relay-adjacent SDK, protocol, or integration:

```bash
cargo add nemo-relay-types
```

Enable JSON Schema support when needed:

```bash
cargo add nemo-relay-types --features schema
```

## Getting Started

Use the shared `Json` type for a Relay-compatible payload:

```rust
use nemo_relay_types::Json;
use serde_json::json;

let payload: Json = json!({"source": "my-integration"});
assert_eq!(payload["source"], "my-integration");
```

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [NeMo Relay Rust crate](https://github.com/NVIDIA/NeMo-Relay/blob/main/crates/core/README.md)
