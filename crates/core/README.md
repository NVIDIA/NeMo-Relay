<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow

Rust core runtime library for the NeMo Flow framework.

## Overview

This crate provides the foundational runtime that all language bindings build upon. It implements the complete agent execution model including scope management, middleware pipelines, event-driven lifecycle hooks, and ATIF trajectory export.

## What It Provides

- **Scope management** -- Hierarchical scope stack with UUID handles, scope-local middleware registration, and automatic cleanup on scope pop.
- **Middleware pipeline** -- Priority-sorted registries for guardrails (sanitize/gate), request intercepts (transform with optional `break_chain`), and execution intercepts (middleware chain pattern with `next`).
- **Event system** -- Observer-pattern subscriber model with typed `Event` fields (`input`, `output`, `model_name`, `tool_call_id`) populated by the runtime.
- **ATIF export** -- `AtifExporter` subscriber that collects events and exports ATIF v1.6 trajectories.
- **Stream wrapping** -- `LlmStreamWrapper` for buffering/parsing SSE events, feeding chunks to a collector, and calling a finalizer on stream end.
- **Async runtime** -- Built on Tokio with `task_local` context propagation for async scopes and thread-local for sync scopes.

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | Crate root and public re-exports |
| `src/api.rs` | Top-level API functions (scope, tool, LLM, registration) |
| `src/types.rs` | Core types (`Event`, `Json`, middleware configs) |
| `src/context.rs` | Scope stack and context propagation |
| `src/registry.rs` | `SortedRegistry<T>` for priority-based middleware |
| `src/stream.rs` | `LlmStreamWrapper` for SSE stream processing |
| `src/atif.rs` | ATIF v1.6 trajectory exporter |
| `src/error.rs` | `FlowError` enum (AlreadyExists, NotFound, etc.) |
| `src/json.rs` | JSON type alias and utilities |

## Build

```bash
cargo build -p nemo-flow
cargo test -p nemo-flow
```

## Documentation

See [docs/architecture.md](../../docs/architecture.md) for the full architecture guide.
