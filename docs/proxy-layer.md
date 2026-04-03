<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NexusProxy: Proxy Layer

NexusProxy is a stateful telemetry tap that sits between your agent application and the Nexus runtime. It observes agent executions via event subscriptions, builds a prediction trie via online learning, and optionally injects scheduling hints into LLM requests for downstream inference engines like NVIDIA Dynamo.

## Overview

NexusProxy does three things:

1. **Captures run telemetry** -- An event subscriber forwards Nexus lifecycle events (LLM start/end, tool start/end, agent scope boundaries) through an async channel to a background drain task.
2. **Builds a prediction trie** -- The drain task accumulates run records and feeds them through a learner pipeline that computes latency sensitivity scores and streaming percentile statistics.
3. **Injects AgentHints** -- When enabled, a DynamoIntercept reads the prediction trie from a hot cache and injects scheduling hints into LLM request bodies at `nvext.agent_hints`.

### Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │                  NexusProxy                      │
                        │                                                  │
  Application ──────►   │  ┌────────────┐       ┌───────────────────────┐  │
     (LLM /             │  │  Event     │  tx   │     Drain Task        │  │
      Tool              │  │ Subscriber ├──────►│  ┌─────────────────┐  │  │
      calls)            │  └────────────┘  mpsc │  │ RunAccumulator  │  │  │
        │               │                       │  └────────┬────────┘  │  │
        │               │  ┌────────────┐       │           │           │  │
        │               │  │  Dynamo    │       │  ┌────────▼────────┐  │  │
        ├──────────────►│  │ Intercept  │       │  │ Learner Pipeline│  │  │
        │               │  │ (opt-in)   │       │  └────────┬────────┘  │  │
        │               │  └─────┬──────┘       │           │           │  │
        │               │        │ reads        └───────────┼───────────┘  │
        │               │  ┌─────▼──────┐                   │ writes       │
        │               │  │  Hot Cache │◄──────────────────┘              │
        │               │  │ (RwLock)   │                                  │
        │               │  └────────────┘       ┌───────────────────────┐  │
        │               │                       │   Storage Backend     │  │
        │               │                       │  (InMemory / Redis)   │  │
        │               │                       └───────────────────────┘  │
        │               └──────────────────────────────────────────────────┘
        │
  Nexus Core ◄──── Middleware Pipeline (guardrails, intercepts, subscribers)
```

The hot cache (`Arc<RwLock<HotCache>>`) holds the current execution plan, prediction trie, and pre-computed default hints. Intercepts read from the hot cache using a read lock -- they never perform I/O on the hot path.

## How It Works

### Event Flow

When your agent executes LLM and tool calls through Nexus, the proxy's event subscriber is invoked synchronously under the global context read lock. The subscriber clones each event and sends it through a `tokio::sync::mpsc::unbounded_channel` -- this operation never blocks. The background drain task receives events and:

1. Groups events by `root_uuid` using a `RunAccumulator`
2. Matches Start/End event pairs into `CallRecord` entries within a `RunRecord`
3. When an Agent scope End event arrives, finalizes the run and stores it via the storage backend
4. Invokes the learner pipeline (e.g., `LatencySensitivityLearner`) which updates accumulators, rebuilds the prediction trie, and refreshes the hot cache

### Intercept Flow

When DynamoIntercept is enabled, it runs as an LLM execution intercept in the middleware chain:

1. Extract the current scope path via `extract_scope_path()` (walks Agent/Function scopes)
2. Check for manual `@latency_sensitive` annotations via `read_manual_latency_sensitivity()`
3. Read the hot cache to look up predictions for the current scope path
4. Build `AgentHints` from the prediction (or fall back to default hints)
5. Apply manual sensitivity override with max-merge semantics (highest value wins)
6. Inject hints into the request body at `content.nvext.agent_hints` and headers at `x-nexus-proxy-agent-hints`
7. Call `next(request)` to continue the middleware chain

### Context-Local Agent ID

The agent ID is resolved from the scope stack by walking up to find the first Agent-typed scope name via `resolve_agent_id()`. This means the proxy automatically detects which agent is running when you push a scope with `ScopeType.Agent`. No explicit configuration is needed if your LangChain/LangGraph integration pushes agent scopes.

### Scope Path Extraction

`extract_scope_path()` walks the scope stack from root to top, collects names of Agent and Function scopes (skipping the implicit root scope), and returns them as a `Vec<String>`. For example, a LangGraph topology with nodes `classify -> draft -> review` would produce paths like `["classify"]` or `["draft"]` depending on which node is currently executing. These paths are the keys used for prediction trie lookup.

## Configuration

### Declarative API (Recommended)

The declarative API buffers configuration via setter functions and lazily creates the proxy when `ensure_proxy()` is called. This is the recommended approach for most users.

**Python:**

```python
import nat_nexus

# Buffer configuration
nat_nexus.set_use_proxy(True)
nat_nexus.set_proxy_backend(nat_nexus.InMemoryBackend())
nat_nexus.set_proxy_sensitivity(nat_nexus.SensitivityConfig(
    w_critical=0.35,
    w_fanout=0.15,
    w_position=0.5,
    w_parallel=0.5,
))
nat_nexus.set_dynamo_intercept(True)

# Lazily create and register the proxy
await nat_nexus.ensure_proxy()

# ... run your agent workload ...

# Clean up
nat_nexus.teardown_proxy()
```

**Rust:**

```rust
use nvidia_nat_nexus_proxy::*;

// Buffer configuration
set_use_proxy(true);
set_proxy_backend(AnyBackend::InMemory(InMemoryBackend::new()));
set_proxy_sensitivity(trie::SensitivityConfig {
    sensitivity_scale: 5,
    w_critical: 0.35,
    w_fanout: 0.15,
    w_position: 0.5,
    w_parallel: 0.5,
});
set_dynamo_intercept(true);

// Lazily create and register the proxy
ensure_proxy().await?;

// ... run agent ...

// Clean up
teardown_proxy()?;
```

**How lazy creation works:** Settings are buffered in a `ProxyConfig` stored inside a `ProxyManager` in the global context extensions map. When `ensure_proxy()` is called, it reads the buffered config, resolves the agent ID from the scope stack, releases the global context lock, then builds and registers the proxy outside the lock to avoid deadlocks. Calling `ensure_proxy()` again is a no-op if the proxy is already registered. Defaults: `InMemoryBackend`, NAT-matching sensitivity weights, DynamoIntercept disabled.

### Builder API (Advanced)

The builder API gives explicit control over the proxy construction. Use this when you need to manage multiple proxies or integrate at a lower level.

**Rust:**

```rust
use nvidia_nat_nexus_proxy::proxy::NexusProxy;
use nvidia_nat_nexus_proxy::storage::InMemoryBackend;
use nvidia_nat_nexus_proxy::learner::LatencySensitivityLearner;
use nvidia_nat_nexus_proxy::trie::SensitivityConfig;

let config = SensitivityConfig::default();
let learner = LatencySensitivityLearner::new("my-agent", config);

let mut proxy = NexusProxy::<InMemoryBackend>::builder()
    .agent_id("my-agent")
    .backend(InMemoryBackend::new())
    .dynamo_intercept(true)
    .learner(Box::new(learner))
    .llm_intercept_priority(50)
    .tool_intercept_priority(75)
    .build()?;

proxy.register().await?;
// ... run agent ...
proxy.deregister()?;
```

**Python:**

```python
import nat_nexus

proxy = nat_nexus.NexusProxy(
    "my-agent",
    nat_nexus.InMemoryBackend(),
    sensitivity_config=nat_nexus.SensitivityConfig(),
    dynamo_intercept=True,
)
await proxy.register()
# ... run agent ...
proxy.deregister()
```

The builder requires an explicit `agent_id` and `backend`. The declarative API resolves `agent_id` from the scope stack and defaults to `InMemoryBackend`.

## DynamoIntercept

DynamoIntercept is an opt-in LLM execution intercept that injects `AgentHints` into LLM request bodies for NVIDIA Dynamo inference serving.

### What It Does

DynamoIntercept reads the prediction trie from the hot cache, looks up predictions for the current scope path, and builds an `AgentHints` struct with scheduling metadata. The hints are injected at two locations:

- **Request body:** `content.nvext.agent_hints` (matching NAT's DynamoTransport injection point)
- **Request headers:** `x-nexus-proxy-agent-hints` (for backward compatibility with proxy consumers)

### AgentHints Fields

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `osl` | `u32` | `output_tokens.p90` | Output Sequence Length (tokens). Used by Dynamo to allocate output buffer. |
| `iat` | `u32` | `interarrival_ms.mean` | Inter-Arrival Time (ms). Tells the scheduler the expected gap between calls. |
| `priority` | `i32` | `scale - sensitivity` | Engine scheduler priority. Higher sensitivity = lower priority number = processed first. |
| `latency_sensitivity` | `f64` | sensitivity score | Sensitivity score as float. `1` = low, `5` = high (default scale). |
| `prefix_id` | `String` | `"{agent_id}-d{depth}"` | KV cache prefix identity for prefix sharing across calls at the same depth. |
| `total_requests` | `u32` | `remaining_calls.mean + call_index` | Expected total LLM requests in this run. Helps Dynamo plan batch scheduling. |

### When to Enable

Enable DynamoIntercept only when using NVIDIA Dynamo for inference serving. For other LLM providers (OpenAI, Anthropic, etc.), the hints are injected but have no effect -- they are safely ignored by non-Dynamo servers.

### Configuration

```python
# Declarative API
nat_nexus.set_dynamo_intercept(True)
await nat_nexus.ensure_proxy()

# Builder API
proxy = nat_nexus.NexusProxy("agent", backend, dynamo_intercept=True)
```

### Manual Latency Sensitivity

You can manually annotate critical code paths with `@latency_sensitive` to override the auto-computed sensitivity score. Manual annotations use max-merge semantics: the highest value between auto-computed and manual wins.

```python
from nat_nexus import latency_sensitive

@latency_sensitive(5)  # Maximum sensitivity: this path is critical
async def time_critical_call():
    result = await nat_nexus.llm.execute("gpt-4", request, llm_func)
    return result

# Also works as a context manager
async with latency_sensitive(4):
    result = await nat_nexus.llm.execute("gpt-4", request, llm_func)
```

The decorator pushes a Nexus scope with metadata at JSON pointer `/nexus_proxy/latency_sensitivity`. The DynamoIntercept reads this metadata when building hints.

## Python API Reference

### Declarative Proxy Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `set_use_proxy` | `(enabled: bool) -> None` | Buffer intent to enable/disable proxy. If `False` after `ensure_proxy()`, tears down immediately. |
| `set_proxy_backend` | `(backend: InMemoryBackend \| RedisBackend) -> None` | Buffer the storage backend choice. Applied at `ensure_proxy()` time. |
| `set_proxy_sensitivity` | `(config: SensitivityConfig) -> None` | Buffer the sensitivity scoring configuration. |
| `set_dynamo_intercept` | `(enabled: bool) -> None` | Buffer the DynamoIntercept opt-in flag. |
| `ensure_proxy` | `() -> None` (async) | Create and register proxy from buffered config. Idempotent. Implicitly enables if `set_use_proxy(True)` was not called. |
| `teardown_proxy` | `() -> None` | Deregister and remove the active proxy. Safe to call when no proxy is active. |
| `proxy_active` | `() -> bool` | Check whether the declarative proxy is currently active. |

### Types

**`InMemoryBackend()`** -- Zero-config in-memory backend. Good for testing and single-process deployments. Data is lost on process restart.

**`RedisBackend`** -- Cross-process persistent backend. Created asynchronously:

```python
backend = await nat_nexus.RedisBackend.connect("redis://127.0.0.1:6379", "nexus:")
```

**`SensitivityConfig`** -- Controls the 4-signal sensitivity scoring model:

```python
config = nat_nexus.SensitivityConfig(
    sensitivity_scale=5,   # Quantization scale (1..=scale)
    w_critical=0.35,       # Critical-path weight
    w_fanout=0.15,         # Fan-out degree weight
    w_position=0.5,        # Position score weight (U-shaped)
    w_parallel=0.5,        # Parallelism indicator weight
)
```

**`NexusProxy`** -- Explicit proxy constructor (builder API):

```python
proxy = nat_nexus.NexusProxy(
    agent_id="my-agent",
    backend=nat_nexus.InMemoryBackend(),
    sensitivity_config=nat_nexus.SensitivityConfig(),
    dynamo_intercept=False,
    llm_intercept_priority=100,
    tool_intercept_priority=100,
)
await proxy.register()
proxy.deregister()
```

**`MetadataEnvelope`** -- Per-request metadata (read-only properties: `run_id`, `agent_id`, `parallel_hints`, `extensions`).

**`AgentHints`** -- Scheduling hints injected by DynamoIntercept (read-only properties: `osl`, `iat`, `priority`, `latency_sensitivity`, `prefix_id`, `total_requests`).

**`ParallelHint`** -- Parallel execution hint (read-only properties: `tool_name`, `group_id`, `explicit`).

**`latency_sensitive`** -- Decorator and context manager for manual sensitivity annotation.

## Rust API Reference

### Module-Level Functions

The `nvidia_nat_nexus_proxy` crate re-exports these functions from the `manager` module:

| Function | Signature | Description |
|----------|-----------|-------------|
| `set_use_proxy` | `fn(enabled: bool)` | Buffer proxy enable/disable intent |
| `set_proxy_backend` | `fn(backend: AnyBackend)` | Buffer the storage backend |
| `set_proxy_sensitivity` | `fn(config: SensitivityConfig)` | Buffer sensitivity configuration |
| `set_dynamo_intercept` | `fn(enabled: bool)` | Buffer DynamoIntercept opt-in |
| `ensure_proxy` | `async fn() -> Result<()>` | Materialize and register proxy from buffered config |
| `teardown_proxy` | `fn() -> Result<()>` | Deregister and remove the active proxy |
| `proxy_active` | `fn() -> bool` | Check if proxy is currently registered |

### Types

| Type | Module | Description |
|------|--------|-------------|
| `NexusProxy<B>` | `proxy` | Generic proxy struct, parameterized over storage backend |
| `NexusProxyBuilder<B>` | `proxy` | Builder for `NexusProxy` with required `agent_id` and `backend` |
| `ProxyManager` | `manager` | Manages declarative proxy lifecycle (stored in global context extensions) |
| `ProxyConfig` | `manager` | Buffered proxy configuration |
| `AnyBackend` | `storage` | Enum dispatch for `InMemory` and `Redis` backends (Python monomorphization) |
| `InMemoryBackend` | `storage` | Zero-config in-memory backend |
| `RedisBackend` | `redis` | Redis-backed persistent backend (feature-gated: `redis-backend`) |
| `StorageBackend` | `storage` | RPITIT trait for backend persistence (not object-safe) |
| `StorageBackendDyn` | `storage` | Object-safe companion trait for dynamic dispatch in learner pipeline |
| `DynamoIntercept` | `dynamo_intercept` | Opt-in LLM execution intercept for AgentHints injection |
| `AgentHints` | `types` | Scheduling hints struct |
| `HotCache` | `types` | Holds plan, trie, and default hints under a single `RwLock` |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PROXY_EXTENSION_KEY` | `"proxy"` | Key for ProxyManager in global context extensions map |
| `AGENT_HINTS_HEADER_KEY` | `"x-nexus-proxy-agent-hints"` | HTTP header key for AgentHints |
| `LATENCY_SENSITIVITY_POINTER` | `"/nexus_proxy/latency_sensitivity"` | JSON pointer for manual sensitivity metadata |

## Troubleshooting

### Proxy not active

**Symptom:** `proxy_active()` returns `False` after configuration.

**Cause:** `ensure_proxy()` was not called (or not awaited in async Python).

**Fix:** Call `await nat_nexus.ensure_proxy()` after setting up configuration. In Rust, call `ensure_proxy().await?`.

### No agent_id resolved

**Symptom:** Agent ID defaults to `"default-agent"`.

**Cause:** No Agent-typed scope has been pushed to the scope stack before `ensure_proxy()` is called.

**Fix:** Push an Agent scope first, then call `ensure_proxy()`:

```python
with nat_nexus.scope.scope("my-agent", nat_nexus.ScopeType.Agent):
    await nat_nexus.ensure_proxy()
    # ...
```

Or use the builder API with an explicit `agent_id`.

### DynamoIntercept hints not appearing

**Symptom:** LLM requests do not contain `nvext.agent_hints`.

**Cause:** DynamoIntercept is disabled by default.

**Fix:** Enable it before calling `ensure_proxy()`:

```python
nat_nexus.set_dynamo_intercept(True)
await nat_nexus.ensure_proxy()
```

### Hot cache empty on first call

**Symptom:** First LLM call gets no hints (or only default hints).

**Cause:** The prediction trie has not been built yet -- no completed runs have been processed.

**Fix:** This is expected behavior. The trie is populated after the first complete agent run. Subsequent runs will have predictions from the first run. Use `@latency_sensitive` for critical paths that need hints before learning converges.

## Cross-References

- [Online Learning Engine](online-learning-engine.md) -- Prediction trie, sensitivity scoring, learner pipeline
- [Middleware Pipeline](middleware-pipeline.md) -- How intercepts and subscribers are ordered and executed
- [Context Isolation](context-isolation.md) -- Scope stacks, `resolve_agent_id()`, and multi-tenant isolation
- [Architecture Overview](architecture.md) -- Nexus system design and binding layers
