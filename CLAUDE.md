<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLAUDE.md

## Project Overview

NVAgentRT is a multi-language agent runtime framework providing execution scope management, lifecycle events, and middleware (guardrails/intercepts) for tool and LLM calls. The core is written in Rust with bindings for Python, Go, Node.js, and WebAssembly.

## Repository Structure

```
crates/
  core/           # Core runtime library (nvagentrt-core)
    src/          #   lib.rs, api.rs, atif.rs, context.rs, types.rs, registry.rs, stream.rs, error.rs, json.rs
    tests/        #   context_isolation_tests.rs, stream_tests.rs
  python/         # PyO3 Python bindings (_native C extension, abi3 stable ABI)
    src/          #   lib.rs, py_api.rs, py_types.rs, py_callable.rs, convert.rs, py_context.rs
  ffi/            # C FFI layer (used by Go, generates nvagentrt.h via cbindgen)
    src/          #   lib.rs, api.rs, callable.rs, types.rs, convert.rs, error.rs
  node/           # NAPI Node.js bindings
    src/          #   lib.rs, api.rs, callable.rs, types.rs, convert.rs, stream.rs
    tests/        #   types_tests.mjs, scope_tests.mjs, tools_tests.mjs, llm_tests.mjs,
                  #   deregister_tests.mjs, context_tests.mjs
  wasm/           # wasm-bindgen WebAssembly bindings
    src/          #   lib.rs, api.rs, callable.rs, types.rs, convert.rs, stream.rs
    tests/        #   types_tests.rs, scope_tests.rs, tools_tests.rs, llm_tests.rs,
                  #   deregister_tests.rs, context_tests.rs
python/           # Python wrapper module (nvagentrt/)
  nvagentrt/      #   __init__.py, __init__.pyi, scope.py, tools.py, llm.py,
                  #   guardrails.py, intercepts.py, subscribers.py
  tests/          #   test_types.py, test_scope.py, test_tools.py, test_llm.py,
                  #   test_subscribers.py, test_context_isolation.py
go/nvagentrt/     # Go CGo bindings
                  #   nvagentrt.go, types.go, stream.go, callbacks.go
                  #   + subpackages: scope/, tools/, llm/, guardrails/, intercepts/, subscribers/
                  #   Tests: context_test.go, scope_test.go, tools_test.go, llm_test.go,
                  #          deregister_test.go, types_test.go
```

## Build & Test Commands

```bash
# Build
cargo build --workspace
cargo build -p nvagentrt-core            # Core only
cargo build --release -p nvagentrt-ffi   # FFI shared lib (needed for Go)

# Test — Rust
cargo test --workspace                   # All Rust tests (excludes nvagentrt-python if Python < 3.11)
cargo test -p nvagentrt-core             # Core tests only
cargo test -p nvagentrt-wasm             # WASM tests (unit tests via cargo test)
wasm-pack test --node crates/wasm        # WASM integration tests (wasm-bindgen-test)

# Test — Python
uv sync                                  # Create venv, install deps, build native extension
uv run pytest                            # Runs tests in python/tests/

# Test — Go (requires FFI lib built first)
cd go/nvagentrt && CGO_LDFLAGS="-L../../target/release" go test -v ./...

# Test — Node.js (requires native addon built first)
cd crates/node && npm install && npm run build   # Build .node addon
node --test crates/node/tests/*.mjs              # Run all Node.js tests

# Build — WASM
wasm-pack build crates/wasm              # Produces pkg/ with .wasm, .js, .d.ts
```

## Key Conventions

- **Naming**: Rust snake_case, C FFI exports prefixed `nvagentrt_`, Go PascalCase, Node.js camelCase
- **Error handling**: `Result<T>` with `AgentRtError` enum (AlreadyExists, NotFound, ScopeStackEmpty, GuardrailRejected, Internal)
- **Async**: tokio runtime, `Pin<Box<dyn Future>>` for async ops
- **JSON**: `Json = serde_json::Value` type alias throughout
- **Middleware**: Priority-based `SortedRegistry<T>` with lazy re-sort; guardrails sanitize/gate, request/response intercepts transform, execution intercepts follow middleware chain pattern with `next` parameter
- **Context propagation**: `tokio::task_local` for async, thread-local for sync; Python uses `contextvars.ContextVar`; all bindings expose `create_scope_stack`/`current_scope_stack`/`set_thread_scope_stack` for per-request isolation
- **License**: Apache-2.0; SPDX headers required on all source files
- **Dependencies audited**: via `deny.toml` (cargo-deny)
- **Tests**: Split by topic (types, scope, tools, llm, deregister, context isolation) across all languages
- **Pre-commit hooks**: trailing whitespace, EOF fixup, YAML/TOML/JSON validity; Ruff + ty (Python), cargo fmt + clippy + deny (Rust), gofmt + go vet (Go)

## Architecture Patterns

- **Scope stack**: Hierarchical scopes with UUID handles; root scope always present. Each binding exposes scope stack isolation for concurrent/multi-tenant use.
- **Intercept chains**: Priority-ordered; request/response intercepts support optional `break_chain` short-circuit; execution intercepts use middleware chain pattern — each receives a `next` function to call the next intercept or the original callable
- **Stream wrapping**: `LlmStreamWrapper` buffers/parses SSE events, applies intercepts mid-stream
- **Event subscription**: Observer pattern with named subscribers
- **Event lifecycle fields**: `Event` carries typed fields (`input`, `output`, `model_name`, `tool_call_id`, `root_uuid`) populated by the runtime. `input`/`output` hold post-guardrail data; `model_name` and `tool_call_id` are set via API params on `nvagentrt_llm_call` and `nvagentrt_tool_call` respectively; `root_uuid` identifies the root scope for concurrent agent isolation.
- **ATIF trajectory export**: `AtifExporter` registers as an event subscriber, collects events, and exports ATIF v1.6 trajectories. LLM start/end events map to user/agent steps; tool start/end events map to tool_calls/observations. Filtering by `root_uuid` isolates concurrent agents. Exposed in all bindings (Python `AtifExporter`, Node.js `JsAtifExporter`, WASM `WasmAtifExporter`, FFI `nvagentrt_atif_exporter_*`, Go `AtifExporter`).
- **Binding layers**: Core (Rust) -> FFI (C, used by Go via CGo) / PyO3 (Python) / NAPI (Node.js) / wasm-bindgen (WASM). Each binding mirrors the full API surface: scopes, tools, LLM, guardrails, intercepts, subscribers, ATIF export.
