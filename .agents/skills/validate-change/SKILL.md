---
name: validate-change
description: Choose and run the right NeMo Flow validation matrix for a change instead of using one fixed test list
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Validate a Change

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill to choose the smallest validation set that still covers the
surfaces touched by a change.

## Mandatory Rules

- Format changed files with the language-native formatter before the final
  lint/test pass.
- If any Rust code changed, always run `just test-rust`.
- If any Rust code changed, also run `cargo fmt --all`.
- If any Rust code changed, also run `cargo clippy --workspace --all-targets -- -D warnings`.
- If `crates/core` or `crates/adaptive` changed, run the full matrix across Rust,
  Python, Go, Node.js, and WebAssembly.
- If a language surface changed, always run that language's test target even when
  Rust core did not change.
- If code changes alter APIs, bindings, commands, paths, packaging behavior,
  observability/adaptive semantics, or documented best practices, update any
  dependent maintainer or consumer skills in the same branch.
- During iteration, prefer `uv run pre-commit run --files <changed files...>`.
- Before review or handoff, run `uv run pre-commit run --all-files`.

## Start With The Change Shape

- **Core runtime or shared semantics changed**
  Use `test-rust-core`. This always includes `just test-rust`,
  `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and the full matrix across Rust, Python, Go, Node.js, and WebAssembly.
- **Python-only wrapper or binding change**
  Use `test-python-binding`.
- **Go binding change**
  Use `test-go-binding`.
- **Node.js binding change**
  Use `test-node-binding`.
- **WebAssembly binding change**
  Use `test-wasm-binding`.
- **FFI surface change**
  Use `test-ffi-surface`.
- **Third-party integration or patch change**
  Run patch validation with `./scripts/apply-patches.sh --check` and the relevant
  integration tests. Keep the root `./scripts/*.sh` wrappers for third-party
  flows; build and test entrypoints now live in the repository `justfile`.
- **Docs-only change**
  Run targeted checks only if commands, package names, or examples changed.
  Use `just docs` for docs-site builds and `just docs-linkcheck` when links
  changed. The `./scripts/build-docs.sh` wrapper remains available for
  compatibility.

## Core Validation Matrix

```bash
just test-rust
just test-python
just test-go
just test-node
just test-wasm
```

## Common Targeted Commands

```bash
# Rust only
just build-rust
just test-rust
just ci=true test-rust
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings

# Python
just build-python
just test-python
uv run ruff format python
uv run pytest -k "<pattern>"

# Go
just build-go
just test-go
cd go/nemo_flow && go fmt ./...

# Node
just build-node
just test-node
npm --prefix crates/node run format

# WebAssembly
just build-wasm
just test-wasm
npm --prefix crates/node run precommit:format -- crates/wasm/wrappers crates/wasm/tests-js crates/wasm/scripts

# Third-party patches
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check

# Docs site
just docs
just docs-linkcheck
```

## Layer-Specific Skills

- `test-rust-core`
- `test-python-binding`
- `test-go-binding`
- `test-node-binding`
- `test-wasm-binding`
- `test-ffi-surface`

## Pre-commit Semantics

Use pre-commit in two modes:

- During iteration, run `uv run pre-commit run --files <changed files...>`.
- Before review or handoff, run `uv run pre-commit run --all-files`.

Important: `--files` still triggers any matching hook whose `files` or `types`
selectors match the provided paths. Some hooks then ignore filenames and run a
whole-language or workspace-wide command because they are configured with
`pass_filenames: false`.

Examples from this repo:

- Matching Python files run Ruff on the selected files, and also trigger
  `ty check . ...` for the Python project.
- Matching Rust files trigger `cargo fmt --all --`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo check --workspace --all-targets`.
- Matching Go files trigger `gofmt` on the selected files and `go vet ./...`.
- Matching docs markdown files under `README.md`, `CONTRIBUTING.md`, or `docs/`
  trigger the docs link checker.
- Matching `Cargo.toml`, `Cargo.lock`, or `deny.toml` triggers `cargo deny check`.
- Matching `Cargo.lock`, `uv.lock`, or `crates/node/package-lock.json` triggers
  the attributions generators.
- Matching Node/WebAssembly public JS/TS surfaces can also trigger the public docstring
  checks, while matching Node/WebAssembly JS/TS files trigger the prettier wrapper.

## Hygiene Checks

Run these whenever the change is headed for review. Rust changes should still
run `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`
even if you also plan to rely on pre-commit.

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
uv run pre-commit run --all-files
```

If the change is large or public-facing, also verify:

- README and docs entry points still match current package names and paths
- Examples still run with the documented commands
- Any renamed public surfaces are reflected consistently in manifests and docs

## References

- Testing guide: `docs/contribute/testing-and-docs.md`
- Contributor guide: `CONTRIBUTING.md`
- Build and test dispatchers: `justfile`
- Patch helpers: `scripts/apply-patches.sh`, `scripts/generate-patches.sh`
- Third-party script implementations: `scripts/third-party/`
