<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Online Learning Engine

The online learning engine observes agent executions in real-time, builds statistical models of each node's behavior, and produces predictions for future calls. It operates entirely within the optimizer runtime's telemetry pipeline -- no separate service or batch job is required.

The implementation lives in the `nemo-flow-optimizer` crate. Redis-backed
persistence is optional and depends on enabling the optimizer crate's
`redis-backend` feature.

## Overview

### Why It Matters

Every agent has a unique execution topology. A customer triage agent might classify requests first (latency-sensitive -- the user is waiting), fan out to four parallel specialists (low priority -- can be batched), then draft and review a response (latency-sensitive again). The online learning engine discovers these patterns automatically from observed runs and expresses them as scheduling hints that downstream inference engines can act on.

### Key Insight

The engine learns from the agent's actual execution patterns, not from static configuration. After 4-8 runs, the prediction trie converges to stable sensitivity scores that reflect the true latency profile of each node in the agent graph. No manual tuning is needed for common topologies.

## Architecture and Data Flow

```
  NeMo Flow Events
       │
       ▼
  ┌──────────┐     tx      ┌─────────────────────────────────┐
  │  Event   ├────────────►│          Drain Task              │
  │Subscriber│    mpsc     │                                  │
  └──────────┘             │  ┌────────────────────────────┐  │
                           │  │    RunAccumulator          │  │
                           │  │  (groups events by root)   │  │
                           │  └──────────┬─────────────────┘  │
                           │             │ completed RunRecord │
                           │  ┌──────────▼─────────────────┐  │
                           │  │    Learner Pipeline         │  │
                           │  │  Vec<Box<dyn Learner>>      │  │
                           │  │                            │  │
                           │  │  1. Load accumulators      │  │
                           │  │  2. Feed run               │  │
                           │  │  3. Rebuild trie           │  │
                           │  │  4. Store state            │  │
                           │  │  5. Refresh hot cache      │  │
                           │  └──────────┬─────────────────┘  │
                           │             │                     │
                           └─────────────┼─────────────────────┘
                                         │ writes
                           ┌─────────────▼──────────────┐
                           │      Storage Backend       │
                           │   (InMemory or Redis)      │
                           │                            │
                           │  accumulators, trie,       │
                           │  run records               │
                           └────────────────────────────┘
```

### Drain Task

The drain task is an async background task spawned by `OptimizerRuntime::register()` when the telemetry component is enabled. It reads events from the unbounded mpsc channel, constructs `RunRecord`s from event pairs (Start/End), and invokes the learner pipeline after each completed run. The task exits cleanly when the channel sender is dropped (optimizer shutting down).

### RunRecord

A `RunRecord` represents a completed agent run. It contains:

- `id` -- Unique UUID for this run
- `agent_id` -- Which agent executed this run
- `calls` -- Ordered sequence of `CallRecord` entries (each with `kind: Llm|Tool`, `name`, `started_at`, `ended_at`, `output_tokens`, `prompt_tokens`, `total_tokens`, `model_name`, `tool_call_count`)
- `started_at` / `ended_at` -- Run boundary timestamps

### Learner Pipeline

The drain task holds a `Vec<Box<dyn Learner>>` pipeline. Each learner processes a completed `RunRecord` and can update the storage backend and hot cache. The pipeline is pluggable -- you can add custom learners alongside the built-in `LatencySensitivityLearner`.

## The Prediction Trie

The prediction trie is a tree structure keyed by scope paths. Each node in the trie stores statistical predictions for LLM calls made at that position in the agent graph.

### Structure

```
root
├── predictions_by_call_index: {1: LlmCallPrediction, 2: ...}
├── predictions_any_index: LlmCallPrediction  (aggregated fallback)
└── children:
    ├── "classify" → PredictionTrieNode
    │   ├── predictions_by_call_index: {1: ...}
    │   └── predictions_any_index: ...
    ├── "billing_specialist" → PredictionTrieNode
    ├── "shipping_specialist" → PredictionTrieNode
    ├── "draft" → PredictionTrieNode
    └── "review" → PredictionTrieNode
```

### What Each Node Stores

Each `PredictionTrieNode` contains:

- **`predictions_by_call_index`** -- Predictions keyed by call index (1-indexed). The Nth LLM call in a scope gets index N. Useful when a node makes multiple LLM calls.
- **`predictions_any_index`** -- Aggregated predictions across all call indices. Used as a fallback when the exact index is not available.

### LlmCallPrediction

Each prediction holds four `PredictionMetrics` and an optional sensitivity score:

| Field | Type | Meaning |
|-------|------|---------|
| `remaining_calls` | `PredictionMetrics` | How many more LLM calls are expected after this one |
| `interarrival_ms` | `PredictionMetrics` | Expected time (ms) until the next LLM call |
| `output_tokens` | `PredictionMetrics` | Expected output token count for this call |
| `latency_sensitivity` | `Option<u32>` | Auto-computed sensitivity score (1 = low, scale = high) |

### PredictionMetrics

Aggregated statistics from streaming accumulators:

| Field | Type | Description |
|-------|------|-------------|
| `sample_count` | `u32` | Number of observed samples |
| `mean` | `f64` | Running mean (Welford's algorithm) |
| `p50` | `f64` | 50th percentile (median) via TDigest |
| `p90` | `f64` | 90th percentile via TDigest |
| `p95` | `f64` | 95th percentile via TDigest |

## RunningStats: Streaming Accumulators

The engine uses `RunningStats` for O(1) memory accumulation instead of storing raw samples. Each `RunningStats` instance combines:

1. **Welford's online algorithm** for running mean and variance (no batching needed)
2. **TDigest** for streaming percentile estimation (p50, p90, p95) with O(1) memory

`RunningStats` supports `merge()` for combining accumulator state from different sources, enabling incremental trie updates across process restarts.

```rust
// Welford update (inside add_sample):
self.count += 1;
let delta = value - self.mean;
self.mean += delta / self.count as f64;
let delta2 = value - self.mean;
self.m2 += delta * delta2;

// TDigest update:
self.digest = self.digest.merge_unsorted(vec![value]);
```

### NodeAccumulators

Each node in the trie has a `NodeAccumulators` struct with per-call-index stats AND aggregated stats:

- `remaining_calls`, `interarrival_ms`, `output_tokens`, `sensitivity` -- each a `HashMap<u32, RunningStats>` keyed by call index
- `all_remaining_calls`, `all_interarrival_ms`, `all_output_tokens`, `all_sensitivity` -- aggregated `RunningStats` across all call indices

### AccumulatorState

The complete accumulation state is a `HashMap<String, NodeAccumulators>` keyed by `/`-joined path strings (e.g., `"classify"`, `"draft"`). This state is persisted to the storage backend between runs and restored on startup.

## Sensitivity Scoring

The engine computes a latency sensitivity score for each node using a 4-signal model.

### The Four Signals

| Signal | Weight Key | What It Measures | Range |
|--------|-----------|------------------|-------|
| **Critical path** | `w_critical` | Fraction of total workflow duration spent in this call | 0.0 - 1.0 |
| **Fan-out degree** | `w_fanout` | How many sibling calls overlap with this one | 0.0 - 1.0 |
| **Position score** | `w_position` | U-shaped: higher for first/last calls, lower for middle | 0.0 - 1.0 |
| **Parallelism indicator** | `w_parallel` | Whether this call runs in parallel with others | 0.0 - 1.0 |

### SensitivityConfig

```python
config = nemo_flow.SensitivityConfig(
    sensitivity_scale=5,   # Output range: 1..5
    w_critical=0.35,       # Default: 0.35
    w_fanout=0.15,         # Default: 0.15
    w_position=0.5,        # Default: 0.5
    w_parallel=0.5,        # Default: 0.5
)
```

```rust
use nemo_flow_optimizer::trie::SensitivityConfig;

let config = SensitivityConfig {
    sensitivity_scale: 5,
    w_critical: 0.35,
    w_fanout: 0.15,
    w_position: 0.5,
    w_parallel: 0.5,
};
```

The weighted sum of the four signals is normalized to the `[1, sensitivity_scale]` range using min-max normalization across all nodes in a run.

### Convergence

After 4-8 observed runs, the sensitivity scores converge to stable values that reflect the true latency profile. The convergence pattern depends on the agent topology:

**Sequential pipeline** (A -> B -> C):
- A (first) = HIGH (user waiting for first response)
- B (middle) = MEDIUM
- C (last) = HIGH (final output, user waiting)

**Fan-out / fan-in** (root -> [parallel branches] -> aggregator):
- root = HIGH (critical path, first position)
- parallel branches = LOW (fan-out, can be batched)
- aggregator = HIGH (last position, critical path)

**Mixed topology** (classify -> [billing, shipping, account, product] -> draft -> review):
- classify = HIGH (first position, critical path)
- parallel branches = LOW (fan-out, parallel)
- draft = MEDIUM (middle position)
- review = HIGH (last position, critical path)

### What HIGH/MEDIUM/LOW Means

| Score | Meaning | Inference Engine Behavior |
|-------|---------|--------------------------|
| HIGH (4-5) | Latency-sensitive. User is actively waiting. | Prioritize in scheduling queue. Allocate more compute. |
| MEDIUM (2-3) | Moderate sensitivity. Important but not blocking. | Normal scheduling priority. |
| LOW (1) | Can be batched or deprioritized. Fan-out branch or background work. | Batch with other low-priority requests. |

## Trie Building and Lookup

### PredictionTrieBuilder

The builder takes accumulated stats, computes sensitivity scores, and builds the `PredictionTrieNode` tree:

```rust
use nemo_flow_optimizer::trie::PredictionTrieBuilder;

// From scratch
let mut builder = PredictionTrieBuilder::new(Some(SensitivityConfig::default()));
builder.add_run(&run1);
builder.add_run(&run2);
let trie = builder.build();

// Incremental (from stored accumulators)
let mut builder = PredictionTrieBuilder::with_accumulators(
    stored_accumulators,
    Some(config),
);
builder.add_run(&new_run);
let updated_trie = builder.build();
```

The `add_run()` method extracts LLM call contexts from the `RunRecord`, computes 4-signal sensitivity scores, and updates accumulators at every node along each call's path (root + ancestors + leaf).

### PredictionTrieLookup

The lookup provides a three-level fallback chain for fast hot-path reads:

1. **Exact path + exact call_index** -- Best match
2. **Exact path + `predictions_any_index`** -- Fallback when call index is unknown
3. **Deepest ancestor with predictions** -- Fallback when exact path is not in trie
4. Continue up to root if no ancestor has predictions
5. `None` if the trie has no predictions at all

```rust
use nemo_flow_optimizer::trie::lookup::PredictionTrieLookup;

let lookup = PredictionTrieLookup::new(&trie_root);
let path = vec!["classify".to_string()];
let prediction = lookup.find(&path, /*call_index=*/ 1);
```

### TrieEnvelope

The entire trie is serialized as a versioned JSON envelope for storage:

```json
{
  "version": "1.0",
  "generated_at": "2026-04-01T12:00:00+00:00",
  "workflow_name": "my-agent",
  "root": { ... PredictionTrieNode ... }
}
```

The wire format matches NAT's `serialization.py` for interoperability.

## Storage Backends

### InMemoryBackend

The default backend. Zero-config, single-process only.

- **Good for:** Testing, development, single-agent deployments
- **Data persistence:** Lost on process restart
- **Thread safety:** Uses `std::sync::RwLock` internally (fast for in-memory operations)

```python
state = nemo_flow.optimizer.StateConfig(
    backend=nemo_flow.optimizer.BackendSpec.in_memory()
)
```

```rust
use nemo_flow_optimizer::{BackendSpec, StateConfig};

let state = StateConfig {
    backend: BackendSpec::in_memory(),
};
```

### RedisBackend

Cross-process shared state with automatic reconnection.

- **Good for:** Production, multi-process, persistence across restarts
- **Connection:** Uses `ConnectionManager` (internally `Arc`-based, auto-reconnects on failure)
- **Persistence:** Atomic JSON blob SET/GET -- no partial updates possible
- **Feature gate:** Requires `redis-backend` Cargo feature

```python
state = nemo_flow.optimizer.StateConfig(
    backend=nemo_flow.optimizer.BackendSpec.redis("redis://localhost:6379", "nemo_flow:")
)
```

```rust
use nemo_flow_optimizer::{BackendSpec, StateConfig};

let state = StateConfig {
    backend: BackendSpec::redis("redis://localhost:6379", "nemo_flow:"),
};
```

### Redis Key Layout

All keys are prefixed with the configurable `key_prefix` (e.g., `"nemo_flow:"`):

| Kind | Key Pattern | Value |
|------|-------------|-------|
| Run record | `{prefix}runs:{agent_id}:{run_id}` | JSON `RunRecord` |
| Run index | `{prefix}runs_index:{agent_id}` | LIST of run UUIDs |
| Execution plan | `{prefix}plan:{agent_id}` | JSON `ExecutionPlan` |
| Trie envelope | `{prefix}trie:{agent_id}` | JSON `TrieEnvelope` |
| Accumulators | `{prefix}accumulators:{agent_id}` | JSON `AccumulatorState` |

Run records are stored individually and indexed via a Redis LIST for ordered retrieval. Trie envelopes and accumulator state are each stored as a single atomic JSON blob.

## Implementing a Custom Learner

The `Learner` trait is object-safe and designed for pipeline composition:

```rust
use nemo_flow_optimizer::learner::Learner;
use nemo_flow_optimizer::storage::StorageBackendDyn;
use nemo_flow_optimizer::types::{HotCache, RunRecord};
use nemo_flow_optimizer::error::Result;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

pub struct MyCustomLearner;

impl Learner for MyCustomLearner {
    fn process_run<'a>(
        &'a self,
        run: &'a RunRecord,
        backend: &'a dyn StorageBackendDyn,
        hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Process the completed run
            // Access backend for persistence
            // Update hot_cache for intercept reads
            Ok(())
        })
    }
}
```

### How LatencySensitivityLearner Works

The built-in `LatencySensitivityLearner` implements the load-merge-store cycle:

1. **Load** existing `AccumulatorState` from backend (`None` on first run is OK)
2. **Seed** a `PredictionTrieBuilder` with those accumulators
3. **Add** the new run (extracts LLM contexts, computes sensitivity, updates accumulators)
4. **Build** the prediction trie from updated accumulators
5. **Store** updated accumulators and trie envelope back to the backend
6. **Refresh** the hot cache with the new trie and default hints (no `.await` inside lock)

This cycle runs after every completed agent run. The accumulators grow incrementally -- each run adds samples to the streaming statistics. The trie is rebuilt from scratch each time (from accumulators, not raw runs), keeping the rebuild cost proportional to the number of unique scope paths, not the number of historical runs.

## Interpreting Sensitivity Reports

### Reading Trie Data

Each `LlmCallPrediction` in the trie tells you:

- **`remaining_calls`**: How many more LLM calls are expected. `mean=3.0` means the engine typically makes 3 more calls after this point.
- **`interarrival_ms`**: Gap between this call and the next. `mean=200.0` means ~200ms between calls.
- **`output_tokens`**: Expected output length. `p90=400` means 90% of responses are under 400 tokens.
- **`latency_sensitivity`**: Computed importance. `4` out of 5 means this node is highly latency-sensitive.

### Expected Convergence Patterns

For a 7-node customer triage topology (`classify -> [billing, shipping, account, product] -> draft -> review`):

| Node | Expected Sensitivity | Why |
|------|---------------------|-----|
| classify | 4-5 (HIGH) | First position, critical path, user waiting |
| billing_specialist | 1-2 (LOW) | Parallel branch, fan-out, can be batched |
| shipping_specialist | 1-2 (LOW) | Parallel branch, fan-out |
| account_specialist | 1-2 (LOW) | Parallel branch, fan-out |
| product_specialist | 1-2 (LOW) | Parallel branch, fan-out |
| draft | 2-3 (MEDIUM) | Middle position, moderate critical path |
| review | 4-5 (HIGH) | Last position, critical path, final output |

The key validation: after 6+ runs, `classify_sensitivity > mean(parallel_branch_sensitivities)`. This is the convergence check used in integration tests.

## Cross-References

- [Optimizer Layer](optimizer-layer.md) -- dynamic optimizer config, built-in components, and runtime lifecycle
- [Middleware Pipeline](middleware-pipeline.md) -- How intercepts are ordered and executed
- [Context Isolation](context-isolation.md) -- Scope stack and `resolve_agent_id()` for multi-tenant isolation
