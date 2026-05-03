<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Flow)](https://github.com/NVIDIA/NeMo-Flow/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Flow/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Flow?color=green)](https://github.com/NVIDIA/NeMo-Flow/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Flow/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Flow)
[![PyPI](https://img.shields.io/pypi/v/nemo-flow?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-flow/)
[![npm node](https://img.shields.io/npm/v/nemo-flow-node?label=nemo-flow-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-node)
[![npm wasm](https://img.shields.io/npm/v/nemo-flow-wasm?label=nemo-flow-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow?label=nemo-flow&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow-adaptive?label=nemo-flow-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow-adaptive)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Flow)

# nemo-flow-adaptive

`nemo-flow-adaptive` is the Rust companion crate for adaptive NeMo Flow
runtime behavior. Use it with `nemo-flow` when an agent runtime should learn
from observed executions, inject runtime hints, or persist adaptive state.

Adaptive behavior is installed through the same plugin system used by the core
runtime, so applications can enable it without changing their orchestration
framework.

## Why Use It?

- ⚙️ **Install adaptive behavior through plugins**: Enable adaptive runtime
  components through the same configuration path as other NeMo Flow plugins.
- 📈 **Learn from observed executions**: Derive runtime hints from scope, tool,
  and LLM events without replacing the application framework.
- 💾 **Choose local or shared state**: Use in-memory state for local runs or the
  optional Redis backend for shared persistence.
- 🧩 **Keep adaptive behavior reusable**: Package telemetry, hint injection,
  tool parallelism, and cache-governor behavior behind stable component
  settings.

## What You Get

- ✅ **`AdaptiveConfig`**: A canonical config contract for the top-level
  `adaptive` plugin component.
- ✅ **Built-in component settings**: Typed config helpers for telemetry,
  adaptive hints, tool parallelism, and the Adaptive Cache Governor.
- ✅ **State backends**: In-memory state by default and Redis-backed state behind
  the `redis-backend` feature.
- ✅ **Learning primitives**: Runtime helpers and learners built on NeMo Flow
  events.
- ✅ **ACG module surface**: The canonical `nemo_flow_adaptive::acg` module for
  PromptIR, provider plugins, stability analysis, and cache telemetry
  normalization.

## Installation

Install the published crate alongside the core runtime:

```bash
cargo add nemo-flow nemo-flow-adaptive
```

Enable Redis-backed state only when the application needs shared persistence:

```bash
cargo add nemo-flow-adaptive --features redis-backend
```

For local source development:

```bash
cargo build -p nemo-flow-adaptive
cargo test -p nemo-flow-adaptive
```

## Getting Started

Create a default adaptive config and select the in-memory backend:

```rust
use nemo_flow_adaptive::{AdaptiveConfig, BackendSpec, StateConfig};

let config = AdaptiveConfig {
    state: Some(StateConfig {
        backend: BackendSpec::in_memory(),
    }),
    ..Default::default()
};
```

Register the adaptive plugin component before validating or initializing plugin
configuration that includes an `adaptive` component:

```rust
nemo_flow_adaptive::plugin_component::register_adaptive_component()?;
```

## Feature Flags

- `redis-backend`: Enables the Redis-backed storage implementation.

Builds without `redis-backend` still support the in-memory backend and the rest
of the adaptive pipeline.

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
