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

# NeMo Relay

`nemo-relay-adaptive` is the Rust companion crate for adaptive NeMo Relay
runtime behavior. Use it with `nemo-relay` when an agent runtime should learn
from observed executions, inject runtime hints, persist adaptive state, or
cache repeated LLM responses and classified tool results.

Adaptive behavior is installed through the shared NeMo Relay plugin system, so applications can enable it without changing their orchestration
framework.

## Why Use It?

- **Install adaptive behavior through plugins**: Enable adaptive runtime
  components through the same configuration path as other NeMo Relay plugins.
- **Learn from observed executions**: Derive runtime hints from scope, tool,
  and LLM events without replacing the application framework.
- **Choose local or shared state**: Use in-memory state for local runs or the
  optional Redis backend for shared persistence.
- **Keep adaptive behavior reusable**: Package telemetry, hint injection,
  tool parallelism, and cache-governor behavior behind stable component
  settings.

## What You Get

- **`AdaptiveConfig`**: A canonical config contract for the top-level
  `adaptive` plugin component.
- **Built-in component settings**: Typed config helpers for telemetry,
  adaptive hints, tool parallelism, and the Adaptive Cache Governor.
- **Storage backends**: In-memory and built-in Redis-backed storage for
  adaptive state.
- **Learning primitives**: Runtime helpers and learners built on NeMo Relay
  events.
- **Adaptive Cache Governor (ACG) module surface**: The canonical
  `nemo_relay_adaptive::acg` module for PromptIR, provider plugins, stability
  analysis, and cache telemetry normalization.

## Installation

Install the published crate alongside the Rust SDK:

```bash
cargo add nemo-relay nemo-relay-adaptive
```

Building these crates requires `protoc` 31.1 or later and Buf 1.75.0 or later.
Install both on `PATH`, or select them with `PROTOC` and `BUF`. The first build
needs access to the Buf Schema Registry to download the locked validation schema.

Select Redis-backed storage through runtime configuration when adaptive state
needs shared persistence.

The LLM response and tool-result cache is no longer part of this crate. It is
the separate `nvidia.response_cache` native plugin from NeMo Relay Plugins, and
its component configuration uses the keys of the former `response_cache`
section. An adaptive configuration that still contains `response_cache` fails
validation with `adaptive.response_cache_moved`.

For local source development:

```bash
cargo build -p nemo-relay-adaptive
cargo test -p nemo-relay-adaptive
```

## Getting Started

Create a default adaptive config and select the in-memory backend:

```rust
use nemo_relay_adaptive::{AdaptiveConfig, BackendSpec, StateConfig};

let config = AdaptiveConfig {
    state: Some(StateConfig {
        backend: BackendSpec::in_memory(),
    }),
    ..Default::default()
};
```

All SDKs and the CLI register `adaptive` automatically. Plugin configuration
controls activation. Custom embeddings using `nemo-relay-plugin-host` directly
can call this crate's component registration helper to choose their component set.

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
