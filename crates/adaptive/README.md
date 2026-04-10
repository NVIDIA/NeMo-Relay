<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-adaptive

Adaptive layer for NeMo Flow. This crate wires NeMo Flow lifecycle
events into an online learning pipeline and can inject scheduling hints into
LLM requests for downstream inference engines.

## When To Use It

Use this crate when you need one or more of the following:

- capture multi-run agent telemetry
- build a prediction trie from observed executions
- inject `AgentHints` into LLM requests
- persist adaptive state in memory or Redis

If you only need scopes, guardrails, intercepts, and event subscribers, the
core runtime crate is sufficient and you do not need this crate directly.

## What It Provides

- `AdaptiveConfig` as the canonical config contract for the top-level `adaptive` plugin component
- typed helper configs for adaptive telemetry, hints, and tool parallelism sections
- `ComponentSpec` plus `register_adaptive_component()` for core-plugin host integration
- storage backends for in-memory and Redis-backed persistence

## Feature Flags

| Feature | Purpose |
|---------|---------|
| `redis-backend` | Enables the Redis-backed storage implementation |

The Redis backend is optional. Builds without `redis-backend` still support the
in-memory backend and the rest of the adaptive pipeline.

## Build

```bash
# Default build (in-memory backend only)
cargo build -p nemo-flow-adaptive

# Build with Redis backend support
cargo build -p nemo-flow-adaptive --features redis-backend
```

## Test

```bash
# In-memory adaptive tests
cargo test -p nemo-flow-adaptive

# Redis-backed adaptive tests
cargo test -p nemo-flow-adaptive --features redis-backend redis_tests
```

For a broader test matrix across bindings, see [docs/testing.md](../../docs/testing.md).

## Related Docs

- [docs/adaptive-layer.md](../../docs/adaptive-layer.md)
- [docs/adaptive-api-reference.md](../../docs/adaptive-api-reference.md)
- [docs/online-learning-engine.md](../../docs/online-learning-engine.md)
