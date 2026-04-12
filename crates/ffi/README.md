<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-ffi

C FFI layer for the NeMo Flow core runtime, used by the Go bindings via CGo.

## Overview

This crate exposes the NeMo Flow core API through a C-compatible foreign function interface. It uses `cbindgen` during the Cargo build to regenerate the committed `nemo_flow.h` header, and compiles to both a dynamic library (`cdylib`) and a static library (`staticlib`). The Go bindings consume this FFI layer through CGo.

## What It Provides

- **C-compatible API** -- All exported functions use the `nemo_flow_` prefix with C-compatible types (`*const c_char`, opaque pointers, integer handles).
- **Auto-regenerated header** -- `cbindgen` refreshes `nemo_flow.h` during Cargo builds, keeping the committed header in sync with the Rust source.
- **Callback support** -- C function pointer types for tool handlers, guardrails, intercepts, and event subscribers.
- **Memory management** -- Explicit allocation/deallocation functions for strings and opaque objects returned across the FFI boundary.
- **Tokio runtime** -- Manages a multi-threaded Tokio runtime internally for async operations invoked through synchronous C calls.

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | Crate root and FFI module structure |
| `src/api.rs` | `#[no_mangle]` exported C functions |
| `src/callable.rs` | C function pointer wrappers for Rust callbacks |
| `src/types.rs` | C-compatible type definitions |
| `src/convert.rs` | C-Rust type conversion (strings, JSON, handles) |
| `src/error.rs` | Error code mapping for FFI returns |

## Build

```bash
cargo build --release -p nemo-flow-ffi
```

The shared library is output to `target/release/` (e.g., `libnemo_flow_ffi.dylib` on macOS, `libnemo_flow_ffi.so` on Linux). The regenerated committed header remains at `crates/ffi/nemo_flow.h`.

## Documentation

See [docs/architecture.md](../../docs/architecture.md) for FFI architecture details.
