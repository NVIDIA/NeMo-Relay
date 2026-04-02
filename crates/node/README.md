<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-nat-nexus-node

NAPI-RS Node.js bindings for the Nexus core runtime.

## Overview

This crate compiles to a native `.node` addon using NAPI-RS, exposing the full Nexus API to Node.js. It supports both synchronous callbacks and Promise-based async operations, with automatic conversion between JavaScript values and Rust types via `serde-json`.

## What It Provides

- **Full API surface** -- Scope management, tool/LLM calls, guardrails, intercepts, subscribers, scope-local middleware, and ATIF export accessible from JavaScript.
- **Sync and Promise callbacks** -- Tool handlers and middleware can be implemented as synchronous functions or as functions returning Promises.
- **Stream support** -- LLM stream wrapping with JavaScript collector and finalizer callbacks.
- **Type conversion** -- Transparent conversion between JS objects/arrays and Rust `serde_json::Value` via NAPI's serde integration.
- **Tokio integration** -- Async Rust operations run on a Tokio runtime managed by the NAPI-RS `tokio_rt` feature.

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | NAPI module registration |
| `src/api.rs` | JavaScript-exposed API functions |
| `src/types.rs` | JS type wrappers for core types |
| `src/callable.rs` | Wrapping JS functions for Rust callbacks |
| `src/convert.rs` | JS-Rust type conversion utilities |
| `src/stream.rs` | LLM stream wrapper for Node.js |
| `src/promise_call.rs` | Promise-based callback invocation |

## Build

```bash
cd crates/node
npm install
npm run build
npm test
```

## Documentation

See [docs/language-bindings.md](../../docs/language-bindings.md) for Node.js binding details.
