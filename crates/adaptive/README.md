<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

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

Start with:

- The repo root `README.md`
- `docs/use-adaptive-optimization/configure.md`
- `docs/use-adaptive-optimization/adaptive-components.md`
- `docs/about/concepts/plugins.md`
