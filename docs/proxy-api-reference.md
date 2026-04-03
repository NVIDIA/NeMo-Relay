<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Proxy API Reference

This document covers the public proxy-facing APIs exposed through
`nat_nexus.proxy` and the corresponding Rust `nvidia-nat-nexus-proxy` crate.
Use [Proxy Layer](proxy-layer.md) for architecture and lifecycle flow; use this
reference for function signatures and type summaries.

## Python Declarative API

These functions buffer proxy configuration and materialize the active proxy when
`ensure_proxy()` is awaited.

| Function | Signature | Notes |
|----------|-----------|-------|
| `set_use_proxy` | `(enabled: bool) -> None` | Buffers enable/disable intent. Setting `False` after activation tears the proxy down. |
| `set_proxy_backend` | `(backend: InMemoryBackend \| RedisBackend \| StorageBackendProtocol) -> None` | Buffers the backend selection. |
| `set_proxy_sensitivity` | `(config: SensitivityConfig) -> None` | Enables sensitivity scoring when paired with `ensure_proxy()`. |
| `set_dynamo_intercept` | `(enabled: bool) -> None` | Enables AgentHints injection for the declarative proxy. |
| `ensure_proxy` | `async () -> None` | Creates and registers the active proxy from buffered settings. Idempotent. |
| `teardown_proxy` | `() -> None` | Deregisters and removes the active proxy. Safe when inactive. |
| `proxy_active` | `() -> bool` | Returns whether the declarative proxy is currently active. |

### Declarative Example

```python
import nat_nexus

nat_nexus.proxy.set_use_proxy(True)
nat_nexus.proxy.set_proxy_backend(nat_nexus.proxy.InMemoryBackend())
nat_nexus.proxy.set_dynamo_intercept(True)

await nat_nexus.proxy.ensure_proxy()
```

## Python Builder API

`NexusProxy` gives you explicit control over proxy construction and lifecycle:

```python
proxy = nat_nexus.proxy.NexusProxy(
    agent_id="my-agent",
    backend=nat_nexus.proxy.InMemoryBackend(),
    llm_intercept_priority=100,
    tool_intercept_priority=100,
    sensitivity_config=None,
    dynamo_intercept=False,
)
await proxy.register()
proxy.deregister()
```

### `NexusProxy`

| Member | Signature | Notes |
|--------|-----------|-------|
| `__init__` | `(agent_id: str, backend: InMemoryBackend \| RedisBackend \| StorageBackendProtocol, *, llm_intercept_priority: int = 100, tool_intercept_priority: int = 100, sensitivity_config: SensitivityConfig \| None = None, dynamo_intercept: bool = False) -> None` | Creates a proxy instance bound to one agent ID. |
| `register` | `async () -> None` | Wires subscribers and intercepts into the runtime and starts the drain task. |
| `deregister` | `() -> None` | Removes proxy registrations. Safe to call repeatedly. |
| `store_plan` | `(extensions: Any \| None = None) -> None` | Seeds the hot cache with a minimal execution plan for testing. |
| `agent_id` | `str` | Read-only agent identifier for the proxy instance. |

## Backend Types

### `InMemoryBackend`

Zero-config backend for tests and single-process runs:

```python
backend = nat_nexus.proxy.InMemoryBackend()
```

### `RedisBackend`

Cross-process backend created asynchronously:

```python
backend = await nat_nexus.proxy.RedisBackend.connect(
    "redis://127.0.0.1:6379",
    "nexus:",
)
```

Notes:

- A `RedisBackend` instance is single-owner; do not reuse it across multiple
  `NexusProxy` objects.
- The Rust implementation is feature-gated behind `redis-backend`; builds that
  do not enable that feature cannot provide Redis support.

### `StorageBackendProtocol`

Custom async backend protocol accepted by both `NexusProxy(...)` and
`set_proxy_backend(...)`.

Required methods:

- `store_run(record: dict[str, Any]) -> None`
- `load_plan(agent_id: str) -> dict[str, Any] | None`
- `list_runs(agent_id: str) -> list[dict[str, Any]]`
- `store_trie(agent_id: str, envelope: dict[str, Any]) -> None`
- `load_trie(agent_id: str) -> dict[str, Any] | None`
- `store_accumulators(agent_id: str, state: dict[str, Any]) -> None`
- `load_accumulators(agent_id: str) -> dict[str, Any] | None`

## Sensitivity and Hint Types

### `SensitivityConfig`

Controls automatic latency-sensitivity scoring in the learner pipeline.

```python
config = nat_nexus.proxy.SensitivityConfig(
    sensitivity_scale=5,
    w_critical=0.5,
    w_fanout=0.3,
    w_position=0.2,
    w_parallel=0.0,
)
```

Properties:

- `sensitivity_scale: int`
- `w_critical: float`
- `w_fanout: float`
- `w_position: float`
- `w_parallel: float`

### `AgentHints`

Read-only hints injected by the proxy intercept:

- `osl: int`
- `iat: int`
- `priority: int`
- `latency_sensitivity: float`
- `prefix_id: str`
- `total_requests: int`

### `PredictionMetrics`

Aggregated metrics produced by the learner:

- `sample_count: int`
- `mean: float`
- `p50: float`
- `p90: float`
- `p95: float`

### `LlmCallPrediction`

Per-call prediction envelope:

- `remaining_calls: PredictionMetrics`
- `interarrival_ms: PredictionMetrics`
- `output_tokens: PredictionMetrics`
- `latency_sensitivity: int | None`

### `MetadataEnvelope`

Per-request metadata assembled by the proxy:

- `run_id: str`
- `agent_id: str`
- `parallel_hints: list[ParallelHint]`
- `extensions: Any`

### `ParallelHint`

Parallelism hint attached to a tool:

- `tool_name: str`
- `group_id: str`
- `explicit: bool`

## Scope-Level Latency Helper

`set_latency_sensitivity(value: int) -> None`

Applies a manual sensitivity override to the current top scope using max-merge
semantics. Raises `ValueError` if `value == 0` and `RuntimeError` when the
scope stack is unavailable.

## Rust API Surface

The `nvidia-nat-nexus-proxy` crate re-exports declarative lifecycle helpers:

| Function | Signature |
|----------|-----------|
| `set_use_proxy` | `fn(enabled: bool)` |
| `set_proxy_backend` | `fn(backend: AnyBackend)` |
| `set_proxy_sensitivity` | `fn(config: SensitivityConfig)` |
| `set_dynamo_intercept` | `fn(enabled: bool)` |
| `ensure_proxy` | `async fn() -> Result<()>` |
| `teardown_proxy` | `fn() -> Result<()>` |
| `proxy_active` | `fn() -> bool` |

Builder types:

- `proxy::NexusProxy<B>`
- `proxy::NexusProxyBuilder<B>`
- `manager::ProxyManager`
- `manager::ProxyConfig`

## Feature Flags

`redis-backend`

- Enables the Redis-backed storage implementation in the Rust proxy crate.
- Required for `redis::RedisBackend` in Rust and for any build or packaging
  flow that intends to expose Redis-backed proxy support.

## Related Docs

- [Proxy Layer](proxy-layer.md)
- [Online Learning Engine](online-learning-engine.md)
- [Testing](testing.md)
