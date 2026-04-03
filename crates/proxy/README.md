<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-nat-nexus-proxy

Proxy layer for NeMo Agent Toolkit Nexus. This crate wires Nexus lifecycle
events into an online learning pipeline and can inject scheduling hints into
LLM requests for downstream engines such as NVIDIA Dynamo.

## When To Use It

Use this crate when you need one or more of the following:

- capture multi-run agent telemetry
- build a prediction trie from observed executions
- inject `AgentHints` into LLM requests
- persist proxy state in memory or Redis

If you only need scopes, guardrails, intercepts, and event subscribers, the
core runtime crate is sufficient and you do not need this crate directly.

## What It Provides

- `NexusProxy<B>` and `NexusProxyBuilder<B>` for explicit proxy construction
- declarative lifecycle helpers such as `ensure_proxy()` and `teardown_proxy()`
- learner pipeline support for latency-sensitivity scoring
- storage backends for in-memory and Redis-backed persistence
- Dynamo-oriented request hint injection

## Feature Flags

| Feature | Purpose |
|---------|---------|
| `redis-backend` | Enables the Redis-backed storage implementation |

The Redis backend is optional. Builds without `redis-backend` still support the
in-memory backend and the rest of the proxy pipeline.

## Build

```bash
# Default build (in-memory backend only)
cargo build -p nvidia-nat-nexus-proxy

# Build with Redis backend support
cargo build -p nvidia-nat-nexus-proxy --features redis-backend
```

## Test

```bash
# In-memory proxy tests
cargo test -p nvidia-nat-nexus-proxy

# Redis-backed proxy tests
cargo test -p nvidia-nat-nexus-proxy --features redis-backend redis_tests
```

For a broader test matrix across bindings, see [docs/testing.md](../../docs/testing.md).

## Related Docs

- [docs/proxy-layer.md](../../docs/proxy-layer.md)
- [docs/proxy-api-reference.md](../../docs/proxy-api-reference.md)
- [docs/online-learning-engine.md](../../docs/online-learning-engine.md)
