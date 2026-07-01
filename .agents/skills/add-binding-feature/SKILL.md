---
name: add-binding-feature
description: Add or change a public NeMo Relay API surface across the core runtime and every affected binding
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Add a Binding Feature

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when a change affects the public runtime surface and must stay in
parity across the Rust core, FFI, and one or more bindings.

Do not use this skill for:

- Internal-only core refactors with no public API change
- Binding-local bug fixes that do not change shared behavior
- Docs-only or example-only updates

## Implementation Order

1. **Core Rust**
   Implement the behavior first in `crates/core/src/api/` and
   related core modules such as `crates/core/src/api/runtime/`,
   `crates/core/src/codec/`, or `crates/core/src/json.rs`.
2. **FFI / shared C surface**
   Add or update FFI wrappers in the relevant `crates/ffi/src/api/*.rs`
   module, re-export them through `crates/ffi/src/api/mod.rs`, and ensure the
   generated `crates/ffi/nemo_relay.h` stays correct.
3. **Language-native bindings**
   Update Python, Go, and Node.js for every surface that should expose the
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

| Layer       | Convention        | Example                              |
|-------------|-------------------|--------------------------------------|
| Rust        | `snake_case`      | `nemo_relay_tool_call`                |
| C FFI       | `nemo_relay_` prefix | `nemo_relay_tool_call`              |
| Python      | `snake_case`      | `nemo_relay.tools.call`               |
| Go          | `PascalCase`      | `nemo_relay.ToolCall`                 |
| Node.js     | `camelCase`       | `toolCall`                           |

## Parity Checklist

- [ ] Core function with doc comment in `crates/core/src/api/`
- [ ] Runtime callback/state, codec, JSON, or event/tool/LLM/scope types added
      in the relevant core module if needed
- [ ] FFI wrapper in the relevant `crates/ffi/src/api/*.rs` module and
      re-export in `crates/ffi/src/api/mod.rs`
- [ ] Regenerate the shared library/header path with `just build-go`
- [ ] Python native binding in `crates/python/src/py_api/mod.rs`
- [ ] Python wrapper with docstring in `python/nemo_relay/<module>.py`
- [ ] Python type stubs updated in the relevant `python/nemo_relay/*.pyi` modules
- [ ] Go wrapper in `go/nemo_relay/nemo_relay.go` with doc comment
- [ ] Go shorthand package updated if the capability belongs there
- [ ] Node.js binding in `crates/node/src/api/mod.rs`
- [ ] Typed wrapper or adaptive/plugin helper surfaces updated when applicable
- [ ] Tests added in every affected language surface
- [ ] SPDX license header on any new files
- [ ] Relevant pages under `docs/reference/` updated
- [ ] `README.md`, `docs/getting-started/`, or binding-level READMEs updated if behavior differs by language
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

- Architecture: `docs/about/architecture.md`
- Reference index: `docs/reference/api/index.md`
- Getting started and binding status: `README.md`,
  `docs/getting-started/quick-start.md`,
  `docs/about/release-notes/support-matrix.md`
- Typed wrappers and codecs: `docs/integrate-frameworks/using-codecs.md`,
  `docs/integrate-frameworks/provider-codecs.md`
- Adaptive config/plugins: `docs/about/concepts/plugins.md`,
  `docs/build-plugins/about.md`, `docs/plugins/adaptive/configuration.md`
- Existing pattern: follow a surface already implemented across core, FFI,
  Python, Go, and Node.js rather than inventing a new shape
