---
name: test-node-binding
description: Build and test the NeMo Flow Node.js binding; use this for crates/node changes or Node-facing integration checks
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Build And Test Node Binding

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when the change is primarily in `crates/node`, the generated
Node surface, or Node-facing examples/docs.

## Default Path

1. Format changed Node files with `npm run format --workspace=nemo-flow-node`.
2. Install dependencies and build with `just build-node` when you need
   to validate packaging/build output.
3. Run `just test-node` for the normal dev/test loop.
4. If any Rust files changed as part of the Node work, also run
   `cargo fmt --all`, `just test-rust`, and
   `cargo clippy --workspace --all-targets -- -D warnings`.
5. Use `just ci=true test-node` when you want the CI-style coverage and JUnit
   path.

## Common Commands

```bash
# Explicit release build
just build-node

# Format Node files
npm run format --workspace=nemo-flow-node

# Standard test loop
just test-node

# Required when the Node change also touched Rust code
cargo fmt --all
just test-rust
cargo clippy --workspace --all-targets -- -D warnings

# CI-style coverage and JUnit test loop
just ci=true test-node
```

## Useful Extras

```bash
# Public API docstring checks when surface docs changed
npm run check:docstrings --workspace=nemo-flow-node
```

## When To Escalate

- If the change touched `crates/core`, `crates/adaptive`, or the generated Rust
  binding layer under `crates/node/src`, also use `validate-change`.
- If the change is just documentation around Node usage, keep validation targeted.

## References

- `package.json`
- `package-lock.json`
- `crates/node/package.json`
- `docs/getting-started/nodejs.md`
- `README.md`
- `docs/contribute/testing-and-docs.md`
- `validate-change`
