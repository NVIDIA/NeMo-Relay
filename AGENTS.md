<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

This file provides guidance to agents, including Claude Code and OpenAI Codex, when working in this repository.

## Project Overview

NeMo Flow is a multi-language agent runtime framework for execution scopes, lifecycle events, middleware, plugins, and observability around tool and LLM calls. The core runtime is Rust. Primary supported bindings are Rust, Python, and Node.js. Go, WebAssembly, and the raw C FFI are experimental and source-first.

The shared runtime model is:

1. Scope stacks decide where work belongs and which scope-local behavior is visible.
2. Middleware registries decide what guardrails and intercepts run around managed calls.
3. Plugins install reusable runtime behavior from configuration.
4. Events record runtime behavior in ATOF form.
5. Subscribers and exporters consume events in-process or export them to ATIF, OpenTelemetry, OpenInference, or other backends.

## Repository Structure

The repository layout separates the Rust runtime, language bindings, documentation,
integration patches, and agent-facing skills.

```text
crates/
  core/       # Rust core runtime crate, published as nemo-flow
  adaptive/   # Adaptive runtime primitives and plugin components
  python/     # PyO3 native extension for the Python package
  ffi/        # Raw C ABI layer used by downstream bindings such as Go
  node/       # NAPI Node.js binding and JavaScript/TypeScript entry points
  wasm/       # wasm-bindgen WebAssembly binding and JS wrappers
python/
  nemo_flow/  # Python wrapper package: scopes, tools, LLM, middleware, typed helpers, plugins, adaptive helpers
  tests/      # Python tests
go/
  nemo_flow/  # Experimental Go CGo binding and tests
docs/         # Sphinx documentation site
integrations/ # Sample integrations using public APIs without patching
scripts/      # Stable wrappers and helper scripts; build/test/docs entry points live in justfile
third_party/  # Pinned upstream checkouts for sample integration patches
patches/      # NeMo Flow patch sets applied to third_party checkouts
skills/       # Published Codex/agent skills for NeMo Flow usage patterns
```

## Prerequisites

Install the tools needed for the surfaces you touch. For a full repository validation environment, install all of these:

| Tool | Version / Notes | Required For |
|---|---|---|
| Rust | Docs minimum is 1.86 or newer; the repo pins the active toolchain in `rust-toolchain.toml` | Rust core, native bindings, FFI, WebAssembly |
| Python | 3.11 or newer | Python package, PyO3 builds, docs tooling |
| Node.js | 20 or newer, with npm | Node.js binding, WebAssembly JS tests, generated API docs |
| Go | 1.21 or newer | Experimental Go binding |
| `uv` | Current project workflow tool | Python environments, docs dependencies, pre-commit |
| `just` | 1.40 or newer | Canonical build, test, docs, package task runner |
| `wasm-pack` | 0.14.0 or newer | WebAssembly build and integration tests |
| `cargo-deny` | Current stable | Rust dependency auditing |
| `cargo-nextest` | 0.9.111 or newer | CI-style Rust test runs |
| `cargo-llvm-cov` | 0.8.5 or newer | CI-style coverage reports |

Common setup commands:

```bash
cargo install just --locked
cargo install cargo-deny --locked
cargo install cargo-nextest --version 0.9.111 --locked
cargo install cargo-llvm-cov --version 0.8.5 --locked
cargo install wasm-pack --version 0.14.0 --locked

uv sync
uv run pre-commit install

npm install --ignore-scripts
```

`uv sync` installs Python development and test dependencies, including `maturin`, `ruff`, `ty`, and `pre-commit`. Documentation recipes sync the docs dependency group as needed, but Python, Node.js, npm, `uv`, and `just` still need to exist on PATH.

## Build, Test, And Docs Commands

Prefer the repository `just` recipes over raw tool commands. Use raw `cargo`, `pytest`, `go test`, `npm`, or `wasm-pack` commands only for focused debugging or targeted single-test reruns that do not have a `just` recipe.

Discover the current task surface with:

```bash
just --list
```

Build targets:

```bash
just build-rust
just build-python
just build-node
just build-go
just build-wasm
just build-all
```

Test targets:

```bash
just test-rust
just ci=true test-rust       # CI-style Rust test run; uses nextest and coverage tooling when available
just test-python
just test-node
just test-go
just test-wasm
just test-all
```

Documentation targets:

```bash
just docs
just docs-linkcheck
just docs-github-pages
```

Package targets:

```bash
just package-python
just package-node
just package-wasm
```

Cleanup:

```bash
just clean
```

Focused fallback commands are acceptable for narrow loops:

```bash
cargo test -p nemo-flow -- <test_name>
uv run pytest python/tests/test_scope.py
uv run pytest -k "test_name"
cd crates/node && node --test --test-name-pattern="pattern" tests/*.mjs
cd go/nemo_flow && go test -v -run TestFoo ./...
wasm-pack test --node crates/wasm
```

## Validation Expectations

Run tests for every language affected by a change. If you touch the Rust core runtime, middleware semantics, event shape, scope behavior, typed codecs, plugins, or observability, expect to validate every affected binding because the bindings share the same runtime contract.

Minimum guidance:

- Rust core or adaptive changes: `just test-rust`; add binding tests when public behavior changes.
- Python binding or wrapper changes: `just test-python`.
- Node.js binding or wrapper changes: `just test-node`.
- Go binding or raw FFI changes: `just test-go` and the relevant Rust/FFI checks.
- WebAssembly binding changes: `just test-wasm`.
- Documentation site changes: `just docs`; use `just docs-linkcheck` for link or navigation changes.
- Cross-language API changes: run the touched binding tests and update docs, package READMEs, and generated surfaces where applicable.

Before review, prefer `uv run pre-commit run --all-files` when the change crosses languages or tooling. The hooks enforce SPDX headers, file hygiene, Ruff, `ty`, docs link checks, Cargo formatting/lints/audits, Go formatting/vet, Node formatting, and public docstring checks.

## Key Conventions

These conventions keep source, documentation, and binding behavior consistent across the
repository.

- Keep SPDX headers on source, docs, scripts, and configuration files. The project is Apache-2.0.
- `SKILL.md` files are skill entrypoints and do not need SPDX headers, but they must always start with YAML frontmatter containing at least `name` and `description`.
- Follow binding naming conventions: Rust and Python `snake_case`, C FFI exports prefixed `nemo_flow_`, Go `PascalCase` for public APIs, Node.js `camelCase`.
- Preserve the shared runtime model across bindings. Do not add behavior to one primary binding without considering Rust, Python, and Node.js parity.
- Prefer documented public APIs and stable wrapper commands. Do not rely on internal helpers in examples or user-facing docs.
- Keep primary documentation focused on Rust, Python, and Node.js. Treat Go, WebAssembly, and raw FFI as experimental and source-first unless binding-support guidance changes.
- Use `Json = serde_json::Value` in Rust-facing runtime APIs where the existing code expects JSON payloads.
- Use `Result<T>` with `FlowError` in core runtime paths. Keep errors explicit and binding-appropriate at the wrapper layer.
- Keep async behavior on the existing tokio-based model. Bindings should preserve callback and future lifetimes rather than blocking or hiding async work unexpectedly.
- Do not hand-edit generated or packaged outputs unless the repository workflow expects them to be checked in. Regenerate through the documented recipe or script.

## Runtime Patterns

These runtime patterns describe the shared semantics that bindings and integrations must
preserve.

- Scope stacks are hierarchical and always have a root scope. They establish parent-child event relationships, visibility for scope-local middleware and subscribers, cleanup boundaries, and concurrent request isolation.
- Scope-local middleware and subscribers are owned by a scope and disappear when that scope closes. Global registrations stay process-wide until removed.
- Middleware is priority-ordered after merging global and visible scope-local entries.
- Intercepts change the real execution path. Request intercepts rewrite the request. Execution intercepts wrap or replace the callback. Stream execution intercepts handle streaming lifecycle behavior.
- Guardrails either block execution or sanitize emitted observability payloads. Sanitize guardrails do not rewrite the real callback arguments or return value.
- Managed execution order is conditional guardrails, request intercepts, sanitize-request guardrails for start events, execution intercepts, callback execution, then sanitize-response guardrails for end events.
- Events use ATOF `0.1` as the canonical event format. Scope events use start/end pairs; mark events record runtime checkpoints.
- LLM and tool event metadata belongs in the category profile, such as `model_name`, `tool_call_id`, and custom `subtype` fields.
- Exporters can transform runtime events to ATIF trajectories, OpenTelemetry traces, or OpenInference-compatible output. Root scope identity is used to isolate concurrent agents.

## Binding Notes

These notes summarize how each language binding relates to the Rust runtime source of
truth.

- Rust is the source of truth for runtime behavior. Binding APIs should mirror the Rust semantics unless a language-specific wrapper intentionally improves ergonomics.
- Python wrapper modules live under `python/nemo_flow/`; the native extension is built from `crates/python` with `maturin`.
- Node.js public entry points include the main runtime package plus `nemo-flow-node/typed`, `nemo-flow-node/plugin`, and `nemo-flow-node/adaptive`.
- Go uses the C FFI and requires the FFI library build before tests; `just test-go` handles the library path setup.
- WebAssembly includes Rust wasm-bindgen tests plus JS wrapper/package tests; `just test-wasm` runs both paths.

## Third-Party Integrations And Patches

### Patch-based Integrations
Sample integrations are maintained as patch sets, not as primary package source. The pinned upstream checkouts are listed in `third_party/sources.lock`, local checkouts live under `third_party/`, and NeMo Flow patches live under `patches/`.

Current integration patch sets include:

- Hermes Agent: `third_party/hermes-agent`
- LangChain: `third_party/langchain`
- LangChain NVIDIA: `third_party/langchain-nvidia`
- LangGraph: `third_party/langgraph`
- OpenClaw: `third_party/openclaw`
- opencode: `third_party/opencode`

Use the stable root-level wrappers:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh
./scripts/apply-patches.sh --check
./scripts/generate-patches.sh
```

`apply-patches.sh` expects clean third-party checkouts. After editing an integration checkout, run `./scripts/generate-patches.sh` to regenerate patch files and verify they apply to a clean detached checkout.

### Public API-based Integrations
Some integrations can be implemented using public APIs without patching. These implementations are maintained under `integrations/` with their own README files and test suites.

Current public API-based integrations include:
- LangChain: `integrations/langchain`

## Documentation And Contribution Workflow

These workflow notes keep public documentation, examples, and PR preparation aligned
with repository expectations.

- Update `README.md`, `docs/`, package READMEs, and binding-support notes when public behavior, package names, examples, or supported bindings change.
- Keep release-process details in maintainer docs such as `RELEASING.md`. Do not move release-history policy into user-facing docs or `CHANGELOG.md`.
- Keep stable public wrappers at the `scripts/` root in docs and examples. Reference namespaced helper paths only when documenting internal maintenance work.
- Use branch prefixes from the contributor docs: `feat/`, `fix/`, `docs/`, `test/`, or `refactor/`.
- Use signed-off commits for PR work: `git commit -s`.
- Before creating, opening, publishing, or editing a pull request, read `.github/pull_request_template.md` and use it as the PR body skeleton. Preserve its visible headings, checklist items, and related-issue guidance; fill the sections instead of replacing them with a generic summary.
- If repo-local PR guidance such as the `prepare-pr` skill conflicts with generic GitHub connector or plugin guidance, follow the repo-local PR guidance for PR body format and review handoff details.
- PR descriptions should include what changed, why, how it was tested, and any breaking changes within the repository template format.
