---
name: test-wasm-binding
description: Build and test the NeMo Flow WebAssembly binding; use this for crates/wasm changes or WebAssembly-facing integration checks
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Build And Test WebAssembly Binding

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when the change is primarily in `crates/wasm`, its JS wrappers,
or the WebAssembly-facing runtime surface.

## Default Path

1. Format changed WebAssembly JS/TS wrapper files with
   `npm run precommit:format --workspace=nemo-flow-node -- crates/wasm/wrappers crates/wasm/tests-js crates/wasm/scripts`.
2. Run the WebAssembly tests with `just test-wasm`.
3. If any Rust files changed as part of the WebAssembly work, also run
   `cargo fmt --all`, `just test-rust`, and
   `cargo clippy --workspace --all-targets -- -D warnings`.
4. Use `just build-wasm` when you want an explicit packaging/build pass.
5. Use `just ci=true test-wasm` when you need coverage reports.
6. Add `cargo test -p nemo-flow-wasm` when Rust-only WebAssembly helpers changed.

## Common Commands

```bash
# JS/WebAssembly integration tests
just test-wasm

# Format WebAssembly JS/TS wrapper files
npm run precommit:format --workspace=nemo-flow-node -- crates/wasm/wrappers crates/wasm/tests-js crates/wasm/scripts

# Required when the WebAssembly change also touched Rust code
cargo fmt --all
just test-rust
cargo clippy --workspace --all-targets -- -D warnings

# Build generated package
just build-wasm

# CI-style build path without coverage reports
just ci=true build-wasm

# Coverage-oriented test path
just ci=true test-wasm

# Rust-side WebAssembly crate tests when needed
cargo test -p nemo-flow-wasm
```

In the `justfile`, both `build-wasm` and `test-wasm` check the `ci` variable.
Only `just ci=true test-wasm` copies coverage output; `just ci=true build-wasm`
switches to the CI-style build path but does not generate coverage reports.

## When To Escalate

- If the change touched shared runtime semantics in `crates/core` or
  `crates/adaptive`, also use `validate-change`.
- If the issue is only packaging or generated wrapper output, prioritize the
  build plus package-prep steps before the full test sweep.

## References

- `crates/wasm/Cargo.toml`
- `crates/wasm/scripts/prepare_pkg.mjs`
- `README.md`
- `docs/getting-started/installation.md`
- `validate-change`
