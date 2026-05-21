<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Development Setup

This section collects the setup steps needed before building, testing, or contributing
changes.

## Package Installation

If you are consuming NeMo Relay rather than developing this repository, install
the published package for your language. Use
[Installation](../getting-started/installation.md) for package-manager commands
covering the CLI, Python, Node.js, Rust, and supported integrations.

Go, WebAssembly, and the raw FFI surface are currently experimental and remain
source-first.

## Source Development

Install these tools before you start:

- Rust stable
- Python 3.11 or newer
- `uv`
- `just`
- Go 1.21 or newer
- Node.js LTS
- `wasm-pack`
- `cargo-deny`

If you touch Go, WebAssembly, or the raw FFI surface, build and validate those
bindings from source in the same branch.

Clone the repository and build the workspace:

```bash
git clone <repo-url> && cd NeMo-Relay
uv sync
cargo install just --locked
uv run pre-commit install
just build-rust
just build-python
just build-node
```

Validate the source builds for the experimental bindings when you touch them:

```bash
cd go/nemo_relay
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" go test -v ./...
cd ../..

wasm-pack test --node crates/wasm
```

## Branch Naming

Use these prefixes:

- `feat/`
- `fix/`
- `docs/`
- `test/`
- `refactor/`

## Code Style

These style requirements keep contributions consistent across Rust, Python, Go, and
general repository files.

### Rust

Use these Rust commands and conventions when changing the core runtime or Rust-facing
API surface.

- `cargo fmt`
- `cargo clippy -- -D warnings`
- `cargo deny check`

### Python

Use these Python commands and conventions when changing the wrapper package, tests, or
docs tooling.

- Ruff linting
- Ruff formatting
- `ty` type checking

### General

These general conventions apply across files and language surfaces.

- Follow binding-appropriate naming conventions.
- Keep SPDX headers intact.
- Preserve the shared lifecycle and middleware model across bindings.
