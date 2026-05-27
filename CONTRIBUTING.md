<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to NeMo Relay

Thank you for your interest in contributing to NeMo Relay. This guide covers the development workflow, coding standards, and pull request process.

## Development Setup

This section collects the setup steps needed before building, testing, or contributing
changes.

### Package Installation

If you are consuming NeMo Relay rather than developing this repository, install
the published package for your language:

- **Rust crate** -- `cargo add nemo-relay`
- **Python package** -- `uv add nemo-relay` or `pip install nemo-relay`
- **Node.js package** -- `npm install nemo-relay-node`

Go, WebAssembly, and the raw FFI surface are currently experimental and remain
source-first.

### Source Development

Install these tools before you start:

- **Rust** (stable toolchain) -- install with [rustup](https://rustup.rs/)
- **Python** >= 3.11 -- use [uv](https://docs.astral.sh/uv/) for environment management
- **just** -- `cargo install just --locked`
- **Go** >= 1.21
- **Node.js** (LTS)
- **wasm-pack** -- `cargo install wasm-pack`
- **cargo-deny** -- `cargo install cargo-deny`

When you work in Go, WebAssembly, or the raw FFI surface, build and validate those
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
# Go binding (requires the release FFI library)
cd go/nemo_relay
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" go test -v ./...
cd ../..

# WebAssembly binding
wasm-pack test --node crates/wasm
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

## Release Tagging

Versioned release tags must use raw Rust-compatible SemVer without a leading
`v`.

- Use `0.1.0` for stable releases.
- Use `0.1.0-rc.1` for prereleases.
- Do not create tags such as `v0.1.0` or `v0.1.0-rc.1`.

This keeps release tags aligned with Cargo package versions and avoids
per-registry tag translation during packaging. CI rejects `v`-prefixed release
tags.

For the maintainer release process and release-notes policy, see
[`RELEASING.md`](RELEASING.md).

## Code Style

These style requirements keep contributions consistent across Rust, Python, Go, and
general repository files.

### Rust

Use these Rust commands and conventions when changing the core runtime or Rust-facing
API surface.

- **Formatting**: `cargo fmt` (rustfmt defaults)
- **Linting**: `cargo clippy -- -D warnings` -- all warnings are treated as errors
- **Dependency auditing**: `cargo deny check` -- configured in `deny.toml`

### Python

Use these Python commands and conventions when changing the wrapper package, tests, or
docs tooling.

- **Linting**: [Ruff](https://docs.astral.sh/ruff/) with rule sets `E`, `F`, `W`, `I`
- **Formatting**: Ruff formatter (line length 120, double quotes)
- **Type checking**: [ty](https://github.com/astral-sh/ty)

### Go

Use these Go commands and conventions when changing the experimental Go binding.

- **Formatting**: `gofmt`
- **Static analysis**: `go vet ./...`

### General

These general conventions apply across files and language surfaces.

- Use the naming conventions appropriate to each language: Rust `snake_case`, C FFI exports prefixed `nemo_relay_`, Go `PascalCase`, Node.js `camelCase`, Python `snake_case`.

## Pre-commit Hooks

Pre-commit hooks are configured in `.pre-commit-config.yaml` and run automatically on every `git commit`. Install them after cloning:

```bash
uv run pre-commit install
```

The hooks enforce:

- **General**: trailing whitespace removal, end-of-file fixup, YAML/TOML/JSON validity, merge conflict marker detection, large file check (500 KB max)
- **Docs**: Fern validation for `docs/` and `fern/`, plus external Markdown/MDX link checking for `README.md`, `CONTRIBUTING.md`, and `docs/` via `lychee`
- **Python**: Ruff linting and formatting, ty type checking
- **Rust**: FFI header sync for `crates/ffi/nemo_relay.h` through Cargo/build.rs, `cargo fmt` formatting check, `cargo clippy` lints, `cargo deny` auditing
- **Go**: `gofmt` formatting, `go vet` static analysis

To run all hooks manually against the entire codebase:

```bash
uv run pre-commit run --all-files
```

## Testing Requirements

**Run tests for every language affected by your changes.** If your change touches the core Rust crate, run tests across all bindings since they all depend on it.

Run the affected test targets directly through the repository `justfile`:

```bash
just test-rust
just ci=true test-rust
just test-python
just test-go
just test-node
just test-wasm
```

Those target recipes are the primary entrypoints for targeted reruns as well:

```bash
# Rust
just test-rust

# Python
just test-python

# Go (requires FFI lib built with --release)
just test-go

# Node.js (requires native addon built)
just test-node

# WebAssembly (unit tests)
just test-wasm
```

When adding new functionality, include tests in the appropriate test files for each affected language binding. Tests are organized by topic: types, scope, tools, LLM, deregister, context isolation, and scope-local.

## Documentation Checklist

If your change affects public behavior, bindings, examples, or workspace
structure, update the corresponding docs in the same branch.

Before opening a PR, check the following:

1. `README.md` still reflects the current workspace members and top-level docs.
2. The relevant reference docs are updated for any public API change.
3. The relevant crate or package README is updated when that surface changed.
4. Embedded documentation snippets, patch docs, and binding-support notes are updated if examples or supported bindings changed.
5. For docs site changes, run `just docs` (or `./scripts/build-docs.sh html` as a compatibility wrapper) — it regenerates ignored Fern API reference pages before validation.

For documentation-heavy changes, prefer small targeted commits so the history
shows entry-point changes, reference changes, examples, and maintenance updates
separately.

## DCO Sign-Off

Every commit in a pull request must include a Developer Certificate of Origin sign-off.

Use `git commit -s` when you create commits, or add a `Signed-off-by:` trailer manually if you are fixing an older commit before review.

## Pull Request Process

This section describes how to prepare and submit changes for review.

### Before Submitting

Complete these checks before opening or updating a pull request.

1. Ensure all pre-commit hooks pass.
2. Run the relevant test suites and confirm they pass.
3. Verify your changes compile cleanly with the relevant `./scripts/build*.sh` entrypoint.
4. Update the relevant documentation entry points and references.
5. Rebase your branch on the latest `main` to avoid merge conflicts.

### PR Description

Include the following in your pull request description:

- **What**: A concise summary of the change.
- **Why**: The motivation or issue being addressed.
- **How**: Key implementation details, especially for non-obvious design decisions.
- **Testing**: Which test suites you ran and any new tests added.
- **Breaking changes**: Note any API changes that affect existing users.

### Review Expectations

These expectations describe how review is handled before changes are merged.

- All PRs require at least one approving review before merge.
- Reviewers may request changes for code quality, test coverage, documentation, or architectural concerns.
- Address review feedback by pushing additional commits (do not force-push during review).
- CI must pass before merging.
- Scrutinize any `SONAR_IGNORE_START` / `SONAR_IGNORE_END` markers and require reviewer sign-off plus a written justification before approving them.

## Sonar Suppressions

Use `SONAR_IGNORE_START` / `SONAR_IGNORE_END` only for documented false
positives that cannot be resolved in code or by improving the analyzer
configuration. Keep the ignored block as small as possible, add a brief comment
explaining why the suppression is needed, and call it out in the PR description
so reviewers can explicitly sign off on it.

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
feat: add scope context managers for automatic cleanup in Go, Node.js, and WebAssembly
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

Before making significant changes, read through the documentation in
[`docs/`](docs/), especially:

- [Architecture Overview](docs/about-nemo-relay/architecture.mdx) -- runtime model and data flow
- [Scopes](docs/about-nemo-relay/concepts/scopes.mdx) -- scopes, handles, events, and runtime ownership
- [Middleware](docs/about-nemo-relay/concepts/middleware.mdx) -- execution ordering and middleware behavior
- [API Reference](docs/reference/api/index.mdx) -- public surfaces across Rust, Python, and Node.js

The codebase follows a layered architecture: **Core (Rust)** provides the runtime, with bindings through **FFI (C, used by Go through CGo)**, **PyO3 (Python)**, **NAPI (Node.js)**, and **wasm-bindgen (WebAssembly)**. Each binding mirrors the full API surface.
