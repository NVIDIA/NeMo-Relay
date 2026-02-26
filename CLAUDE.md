# CLAUDE.md

## Project Overview

NVAgentRT is a multi-language agent runtime framework providing execution scope management, lifecycle events, and middleware (guardrails/intercepts) for tool and LLM calls. The core is written in Rust with bindings for Python, Go, Node.js, and WebAssembly.

## Repository Structure

```
crates/
  core/       # Core runtime library (nvagentrt-core)
  python/     # PyO3 Python bindings (_native C extension)
  ffi/        # C FFI layer (used by Go, generates header via cbindgen)
  node/       # NAPI Node.js bindings
  wasm/       # wasm-bindgen WebAssembly bindings
python/       # Python wrapper module (nvagentrt/)
go/           # Go CGo bindings
```

## Build & Test Commands

```bash
# Build
cargo build --workspace
cargo build -p nvagentrt-core          # Core only
cargo build --release -p nvagentrt-ffi # FFI (needed for Go)

# Test
cargo test --workspace                 # All Rust tests
cargo test -p nvagentrt-core           # Core tests only

# Python
pip install -e .                       # Dev install (uses maturin)

# Go (requires FFI lib built first)
cd go/nvagentrt && CGO_LDFLAGS="-L../../target/release" go test -v ./...
```

## Key Conventions

- **Naming**: Rust snake_case, C FFI exports prefixed `nv_agentrt_`, Go PascalCase
- **Error handling**: `Result<T>` with `AgentRtError` enum (AlreadyExists, NotFound, ScopeStackEmpty, GuardrailRejected, Internal)
- **Async**: tokio runtime, `Pin<Box<dyn Future>>` for async ops
- **JSON**: `Json = serde_json::Value` type alias throughout
- **Middleware**: Priority-based `SortedRegistry<T>` with lazy re-sort; guardrails sanitize/gate, intercepts transform/replace
- **Context propagation**: `tokio::task_local` for async, thread-local for sync
- **License**: Apache-2.0
- **Dependencies audited**: via `deny.toml` (cargo-deny)

## Architecture Patterns

- **Scope stack**: Hierarchical scopes with UUID handles; root scope always present
- **Intercept chains**: Priority-ordered, optional `break_chain` short-circuit
- **Stream wrapping**: `LlmStreamWrapper` buffers/parses SSE events, applies intercepts mid-stream
- **Event subscription**: Observer pattern with named subscribers
