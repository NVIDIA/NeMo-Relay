<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to NeMo Agent Toolkit Nexus

Thank you for your interest in contributing to Nexus. This guide covers the development workflow, coding standards, and pull request process.

## Development Setup

### Prerequisites

Ensure you have the following installed:

- **Rust** (stable toolchain) -- install via [rustup](https://rustup.rs/)
- **Python** >= 3.11 -- with [uv](https://docs.astral.sh/uv/) for environment management
- **Go** >= 1.21
- **Node.js** (LTS)
- **wasm-pack** -- `cargo install wasm-pack`
- **cargo-deny** -- `cargo install cargo-deny`

### First Build

```bash
# Clone the repository
git clone <repo-url> && cd nexus

# Install Python dependencies and build the native extension
uv sync

# Install pre-commit hooks (required before your first commit)
uv run pre-commit install

# Build the full Rust workspace
cargo build --workspace

# Build the FFI shared library (needed for Go bindings)
cargo build --release -p nvidia-nat-nexus-ffi

# Build Node.js addon
cd crates/node && npm install && npm run build && cd ../..

# Build WASM package
wasm-pack build crates/wasm
```

Verify everything works by running the test suites (see [Testing Requirements](#testing-requirements) below).

## Branch Naming Conventions

Use the following prefixes for branch names:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features or capabilities |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `test/` | Test additions or modifications |
| `refactor/` | Code restructuring without behavior changes |

Examples: `feat/scope-context-managers`, `fix/node-wasm-silent-failures`, `docs/api-reference-update`.

## Code Style

### Rust

- **Formatting**: `cargo fmt` (rustfmt defaults)
- **Linting**: `cargo clippy -- -D warnings` -- all warnings are treated as errors
- **Dependency auditing**: `cargo deny check` -- configured in `deny.toml`

### Python

- **Linting**: [Ruff](https://docs.astral.sh/ruff/) with rule sets `E`, `F`, `W`, `I`
- **Formatting**: Ruff formatter (line length 120, double quotes)
- **Type checking**: [ty](https://github.com/astral-sh/ty)

### Go

- **Formatting**: `gofmt`
- **Static analysis**: `go vet ./...`

### General

- Use the naming conventions appropriate to each language: Rust `snake_case`, C FFI exports prefixed `nat_nexus_`, Go `PascalCase`, Node.js `camelCase`, Python `snake_case`.

## Pre-commit Hooks

Pre-commit hooks are configured in `.pre-commit-config.yaml` and run automatically on every `git commit`. Install them after cloning:

```bash
uv run pre-commit install
```

The hooks enforce:

- **General**: trailing whitespace removal, end-of-file fixup, YAML/TOML/JSON validity, merge conflict marker detection, large file check (500 KB max)
- **Python**: Ruff linting and formatting, ty type checking
- **Rust**: `cargo fmt` formatting check, `cargo clippy` lints, `cargo deny` auditing
- **Go**: `gofmt` formatting, `go vet` static analysis

To run all hooks manually against the entire codebase:

```bash
uv run pre-commit run --all-files
```

## Testing Requirements

**Run tests for every language affected by your changes.** If your change touches the core Rust crate, run tests across all bindings since they all depend on it.

```bash
# Rust
cargo test --workspace

# Python
uv run pytest

# Go (requires FFI lib built with --release)
cd go/nat_nexus && \
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" \
go test -v ./...

# Node.js (requires native addon built)
node --test crates/node/tests/*.mjs

# WASM (unit tests)
cargo test -p nvidia-nat-nexus-wasm

# WASM (integration tests)
wasm-pack test --node crates/wasm
```

When adding new functionality, include tests in the appropriate test files for each affected language binding. Tests are organized by topic: types, scope, tools, LLM, deregister, context isolation, and scope-local.

## Pull Request Process

### Before Submitting

1. Ensure all pre-commit hooks pass.
2. Run the relevant test suites and confirm they pass.
3. Verify your changes compile cleanly (`cargo build --workspace`).
4. Rebase your branch on the latest `main` to avoid merge conflicts.

### PR Description

Include the following in your pull request description:

- **What**: A concise summary of the change.
- **Why**: The motivation or issue being addressed.
- **How**: Key implementation details, especially for non-obvious design decisions.
- **Testing**: Which test suites you ran and any new tests added.
- **Breaking changes**: Note any API changes that affect existing users.

### Review Expectations

- All PRs require at least one approving review before merge.
- Reviewers may request changes for code quality, test coverage, documentation, or architectural concerns.
- Address review feedback by pushing additional commits (do not force-push during review).
- CI must pass before merging.

## Commit Message Conventions

Use the following format for commit messages:

```
type: short description of the change

Optional longer description explaining the motivation and context.
```

Valid types:

| Type | Purpose |
|------|---------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `test` | Test additions or modifications |
| `refactor` | Code restructuring without behavior changes |
| `chore` | Build, CI, or tooling changes |
| `perf` | Performance improvements |

Examples:

```
feat: add scope context managers for automatic cleanup in Go, Node.js, and WASM
fix: propagate JS callback errors instead of silent null fallback
docs: update API reference for typed wrapper methods
test: add context isolation tests for concurrent scope stacks
```

Keep the first line under 72 characters. Use the body for additional context when the change is not self-explanatory.

## SPDX License Headers

All source files must include an SPDX license header. Use the appropriate comment syntax for the file type:

**Rust / Go / JavaScript / TypeScript:**
```
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
```

**Python:**
```
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

**HTML / Markdown:**
```
<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
```

**TOML:**
```
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

The pre-commit hooks do not currently enforce SPDX headers automatically, but reviewers will check for them during PR review.

## Understanding the Architecture

Before making significant changes, read through the documentation in the [docs/](../docs/) directory, especially:

- [Architecture Overview](../docs/architecture.md) -- understand the binding layer model and data flow
- [Core Concepts](../docs/concepts.md) -- scopes, handles, events, and the middleware pipeline
- [Middleware Pipeline](../docs/middleware-pipeline.md) -- detailed execution ordering for tool and LLM calls
- [Context Isolation](../docs/context-isolation.md) -- how scope stacks are isolated across concurrent requests

The codebase follows a layered architecture: **Core (Rust)** provides the runtime, with bindings via **FFI (C, used by Go via CGo)**, **PyO3 (Python)**, **NAPI (Node.js)**, and **wasm-bindgen (WASM)**. Each binding mirrors the full API surface.
