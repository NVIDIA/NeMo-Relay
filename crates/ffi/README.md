<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-nat-nexus-ffi

C FFI layer for the Nexus core runtime, used by the Go bindings via CGo.

## Overview

This crate exposes the Nexus core API through a C-compatible foreign function interface. It uses `cbindgen` to automatically generate a `nat_nexus.h` header file at build time, and compiles to both a dynamic library (`cdylib`) and a static library (`staticlib`). The Go bindings consume this FFI layer through CGo.

## What It Provides

- **C-compatible API** -- All exported functions use the `nat_nexus_` prefix with C-compatible types (`*const c_char`, opaque pointers, integer handles).
- **Auto-generated header** -- `cbindgen` produces `nat_nexus.h` during the build step, keeping the header in sync with the Rust source.
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
cargo build --release -p nvidia-nat-nexus-ffi
```

The shared library is output to `target/release/` (e.g., `libnat_nexus_ffi.dylib` on macOS, `libnat_nexus_ffi.so` on Linux). The generated header `nat_nexus.h` is placed in the crate's build output directory.

## Documentation

See [docs/architecture.md](../../docs/architecture.md) for FFI architecture details.
