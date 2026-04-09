<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Choose and run the right NeMo Flow validation matrix for a change instead of using one fixed test list
---

# Validate a Change

Use this skill to choose the smallest validation set that still covers the
surfaces touched by a change.

## Start With The Change Shape

- **Core runtime or shared semantics changed**
  Run the full matrix across Rust, Python, Go, Node.js, and WASM.
- **Python-only wrapper or binding change**
  Run Python tests plus any Rust-backed Python build/test step that changed.
- **Go binding change**
  Rebuild the FFI shared library, then run Go with `-race`.
- **Node.js binding change**
  Build the addon and run Node tests.
- **WASM binding change**
  Run `wasm-pack test --node crates/wasm`; add `cargo test -p nemo-flow-wasm`
  when Rust-only WASM helpers changed.
- **Third-party integration or patch change**
  Run patch validation with `./scripts/apply-patches.sh --check` and the relevant
  integration tests.
- **Docs-only change**
  Run targeted checks only if commands, package names, or examples changed.

## Core Validation Matrix

```bash
cargo test --workspace
uv run pytest
cargo build --release -p nemo-flow-ffi
cd go/nemo_flow && CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" go test -race -v ./...
cd ../../crates/node && npm install && npm test
cd ../.. && wasm-pack test --node crates/wasm
```

## Common Targeted Commands

```bash
# Rust only
cargo test -p nemo-flow-core
cargo test -p nemo-flow-optimizer

# Python
uv sync
uv run pytest -k "<pattern>"

# Go
cargo build --release -p nemo-flow-ffi
cd go/nemo_flow && CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" go test -race -v ./...

# Node
cd crates/node && npm install && npm test

# WASM
cargo test -p nemo-flow-wasm
wasm-pack test --node crates/wasm

# Third-party patches
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
```

## Hygiene Checks

Run these whenever the change is headed for review:

```bash
uv run pre-commit run --all-files
```

If the change is large or public-facing, also verify:

- README and docs entry points still match current package names and paths
- examples still run with the documented commands
- any renamed public surfaces are reflected consistently in manifests and docs

## References

- Testing guide: `docs/testing.md`
- Contributor guide: `.github/CONTRIBUTING.md`
- Patch helpers: `scripts/apply-patches.sh`, `scripts/generate-patches.sh`
