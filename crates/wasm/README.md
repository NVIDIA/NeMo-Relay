<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-nat-nexus-wasm

wasm-bindgen WebAssembly bindings for the Nexus core runtime.

## Overview

This crate compiles the Nexus core API to WebAssembly via `wasm-bindgen`, making it usable in both browser and Node.js environments. It provides the same API surface as other language bindings, adapted for the single-threaded WASM execution model using `send_wrapper` for thread-safety bridging.

## What It Provides

- **Full API surface** -- Scope management, tool/LLM calls, guardrails, intercepts, subscribers, scope-local middleware, and ATIF export for JavaScript/TypeScript consumers.
- **Browser and Node.js support** -- Works in any environment that supports WebAssembly with `wasm-bindgen` glue code.
- **Async via JS Promises** -- Rust `Future`s are bridged to JavaScript Promises through `wasm-bindgen-futures`.
- **Type conversion** -- Automatic conversion between JS values and Rust types via `serde-wasm-bindgen`.
- **Stream support** -- LLM stream wrapping with JavaScript collector and finalizer callbacks.
- **TypeScript definitions** -- `wasm-pack build` generates `.d.ts` type definitions alongside the `.wasm` and `.js` output.

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | WASM module entry point |
| `src/api.rs` | `#[wasm_bindgen]`-exported API functions |
| `src/types.rs` | JS type wrappers for core types |
| `src/callable.rs` | Wrapping JS functions for Rust callbacks |
| `src/convert.rs` | JS-Rust type conversion via `serde-wasm-bindgen` |
| `src/stream.rs` | LLM stream wrapper for WASM |

## Build

```bash
wasm-pack build crates/wasm
```

This produces a `pkg/` directory containing `.wasm`, `.js`, and `.d.ts` files ready for bundling or direct import.

To run unit tests:

```bash
cargo test -p nvidia-nat-nexus-wasm
wasm-pack test --node crates/wasm
```

## Documentation

See [docs/language-bindings.md](../../docs/language-bindings.md) for WASM binding details.
