<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Run tests for any or all Nexus binding layers
---

# Running Nexus Tests

## All Rust tests (core + binding crates)

```bash
cargo test --workspace
```

Note: Core tests share global state. Use `--test-threads=1` if you see
intermittent failures.

## Python

```bash
uv sync                    # Build native extension + install deps
uv run pytest              # Run all tests
uv run pytest -x -v        # Stop on first failure, verbose
uv run pytest python/tests/test_tools.py  # Single file
```

## Go

Requires the FFI shared library to be built first:

```bash
cargo build --release -p nvidia-nat-nexus-ffi
cd go/nat_nexus && CGO_LDFLAGS="-L../../target/release" go test -race -v ./...
```

Always use `-race` to detect data races in CGo boundary code.

## Node.js

```bash
cd crates/node && npm install && npm run build
node --test tests/*.mjs
```

## WASM

Two distinct test modes:

```bash
# Unit tests (run as native, no browser/node WASM runtime)
cargo test -p nvidia-nat-nexus-wasm

# Integration tests (run in Node.js WASM runtime via wasm-pack)
wasm-pack test --node crates/wasm
```

Both must pass.

## Pre-commit hooks

```bash
uv run pre-commit install  # One-time setup
uv run pre-commit run --all-files  # Run all hooks manually
```

Hooks check: copyright headers, trailing whitespace, ruff, ruff-format, ty,
cargo fmt, cargo clippy, cargo check, cargo deny, go fmt, go vet.
