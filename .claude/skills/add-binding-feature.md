<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Add a new API feature across all Nexus binding layers (Python, Go, Node.js, WASM, FFI)
---

# Adding a New Feature Across All Bindings

When adding a new API function or capability to Nexus, it must be implemented
in the core Rust library and then exposed through all five binding layers.

## Implementation Order

1. **Core Rust** (`crates/core/src/api.rs`) — Define the function with full doc comment
2. **FFI/C** (`crates/ffi/src/api.rs`) — Add `extern "C"` wrapper, update `nat_nexus.h`
3. **Python** (`crates/python/src/py_api.rs`) — Add PyO3 `#[pyfunction]` wrapper
4. **Python wrapper** (`python/nat_nexus/`) — Add thin wrapper with docstring in the appropriate module
5. **Go** (`go/nat_nexus/nat_nexus.go`) — Add Go function calling FFI via CGo
6. **Node.js** (`crates/node/src/api.rs`) — Add `#[napi]` function
7. **WASM** (`crates/wasm/src/api.rs`) — Add `#[wasm_bindgen]` function

## Naming Conventions

| Layer   | Convention        | Example                              |
|---------|-------------------|--------------------------------------|
| Rust    | `snake_case`      | `nat_nexus_tool_call`                |
| C FFI   | `nat_nexus_` prefix | `nat_nexus_tool_call`              |
| Python  | `snake_case`      | `nat_nexus.tools.call`               |
| Go      | `PascalCase`      | `nat_nexus.ToolCall`                 |
| Node.js | `camelCase`       | `toolCall`                           |
| WASM    | `camelCase`       | `toolCall`                           |

## Checklist

- [ ] Core function with doc comment in `crates/core/src/api.rs`
- [ ] Types added to `crates/core/src/types.rs` if needed
- [ ] FFI wrapper in `crates/ffi/src/api.rs`
- [ ] Regenerate header: `cargo build -p nvidia-nat-nexus-ffi` (cbindgen runs automatically)
- [ ] Python native binding in `crates/python/src/py_api.rs`
- [ ] Python wrapper with docstring in `python/nat_nexus/<module>.py`
- [ ] Python type stub updated in `python/nat_nexus/__init__.pyi`
- [ ] Go wrapper in `go/nat_nexus/nat_nexus.go` with doc comment
- [ ] Go subpackage shorthand in `go/nat_nexus/<pkg>/<pkg>.go`
- [ ] Node.js binding in `crates/node/src/api.rs`
- [ ] WASM binding in `crates/wasm/src/api.rs`
- [ ] Tests in ALL languages that have tests for the feature area
- [ ] SPDX license header on any new files
- [ ] `docs/api-reference.md` updated
- [ ] `docs/language-bindings.md` updated if the feature has language-specific behavior

## Key References

- Architecture: `docs/architecture.md`
- API reference: `docs/api-reference.md`
- Language bindings guide: `docs/language-bindings.md`
- Existing patterns: Look at how `nat_nexus_tool_call_execute` is implemented across all layers
