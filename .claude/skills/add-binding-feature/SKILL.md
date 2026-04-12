<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Add or change a public NeMo Flow API surface across the core runtime and every affected binding
---

# Add a Binding Feature

Use this skill when a change affects the public runtime surface and must stay in
parity across the Rust core, FFI, and one or more bindings.

Do not use this skill for:

- internal-only core refactors with no public API change
- binding-local bug fixes that do not change shared behavior
- docs-only or example-only updates

## Implementation Order

1. **Core Rust**
   Implement the behavior first in `crates/core/src/api/` and
   `crates/core/src/types/` or related core modules.
2. **FFI / shared C surface**
   Add or update FFI wrappers in `crates/ffi/src/api/mod.rs` and ensure the
   generated `crates/ffi/nemo_flow.h` stays correct.
3. **Language-native bindings**
   Update Python, Go, Node.js, and WASM for every surface that should expose the
   capability.
4. **Language wrapper helpers**
   Update Python wrapper modules, Go shorthand packages, typed helpers, or
   adaptive/plugin helpers if the new behavior belongs there.
5. **Docs and examples**
   Update reference docs, language-binding docs, and examples when the public
   surface or expected usage changed.
6. **Validation**
   Run the validation matrix from the `validate-change` skill for the affected
   surfaces.

## Naming Conventions

| Layer   | Convention        | Example                              |
|---------|-------------------|--------------------------------------|
| Rust    | `snake_case`      | `nemo_flow_tool_call`                |
| C FFI   | `nemo_flow_` prefix | `nemo_flow_tool_call`              |
| Python  | `snake_case`      | `nemo_flow.tools.call`               |
| Go      | `PascalCase`      | `nemo_flow.ToolCall`                 |
| Node.js | `camelCase`       | `toolCall`                           |
| WASM    | `camelCase`       | `toolCall`                           |

## Parity Checklist

- [ ] Core function with doc comment in `crates/core/src/api/`
- [ ] Types added to `crates/core/src/types/` if needed
- [ ] FFI wrapper in `crates/ffi/src/api/mod.rs`
- [ ] Regenerate header with `cargo build -p nemo-flow-ffi`
- [ ] Python native binding in `crates/python/src/py_api/mod.rs`
- [ ] Python wrapper with docstring in `python/nemo_flow/<module>.py`
- [ ] Python type stub updated in `python/nemo_flow/__init__.pyi`
- [ ] Go wrapper in `go/nemo_flow/nemo_flow.go` with doc comment
- [ ] Go shorthand package updated if the capability belongs there
- [ ] Node.js binding in `crates/node/src/api/mod.rs`
- [ ] WASM binding in `crates/wasm/src/api/mod.rs`
- [ ] Typed wrapper or adaptive/plugin helper surfaces updated when applicable
- [ ] Tests added in every affected language surface
- [ ] SPDX license header on any new files
- [ ] `docs/api-reference.md` updated
- [ ] `docs/language-bindings.md` updated if behavior differs by language
- [ ] Relevant getting-started, README, or example docs updated if usage changed

## Decision Points

Lock these before implementing:

- Which bindings actually expose the new surface?
- Is the change part of the plain JSON API, typed wrappers, adaptive/plugin
  helpers, or observability helpers?
- Does the new API need manual lifecycle and managed execute variants, or only
  one of them?
- Does the new behavior change event fields, metadata, or scope expectations?
- Are docs/examples required because the intended usage changed?

## Key References

- Architecture: `docs/architecture.md`
- API reference: `docs/api-reference.md`
- Language bindings guide: `docs/language-bindings.md`
- Typed wrappers: `docs/typed-wrappers.md`
- Adaptive config/plugin host: `docs/adaptive-api-reference.md`
- Existing pattern: follow a surface already implemented across core, FFI,
  Python, Go, Node.js, and WASM rather than inventing a new shape
