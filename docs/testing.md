<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing Guide

NeMo Flow maintains test coverage across the core runtime, all language bindings,
and the optimizer layer. Every binding mirrors the same major test domains so that
behavioral parity is verified at each layer.

## Fast Local Sanity

Use this path when you want the shortest high-signal verification loop:

```bash
# Core runtime + optimizer crate
cargo test --workspace

# Python wrapper
uv run pytest

# Node binding
cd crates/node && npm install && npm test && cd ../..

# WASM binding behavior path
wasm-pack test --node crates/wasm
```

## Quick Reference

```bash
# ── Rust (core + optimizer + WASM unit tests) ──────────────────
cargo test --workspace

# Core only
cargo test -p nemo-flow-core

# Optimizer only (in-memory backend)
cargo test -p nemo-flow-optimizer

# Optimizer with Redis backend enabled
cargo test -p nemo-flow-optimizer --features redis-backend redis_tests

# WASM unit tests
cargo test -p nemo-flow-wasm

# WASM integration tests (wasm-bindgen-test)
wasm-pack test --node crates/wasm

# ── Python ─────────────────────────────────────────────────
uv sync                          # build native extension + install deps
uv run pytest                    # run all Python tests
uv run pytest -k test_typed      # run a single module

# ── Go (requires FFI shared lib) ───────────────────────────
cargo build --release -p nemo-flow-ffi
cd go/nemo_flow && \
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" \
go test -race -v ./...
cd -

# ── Node.js (requires native addon) ────────────────────────
cd crates/node && npm install && npm test
cd -

# ── Full suite ─────────────────────────────────────────────
cargo test --workspace && uv run pytest

# Optional Redis-backed optimizer validation
cargo test -p nemo-flow-optimizer --features redis-backend redis_tests

# ── Coverage artifacts ────────────────────────────────────
# Python wrapper coverage
uv run pytest --cov=nemo_flow --cov-report=xml:target/coverage/pytest_coverage_report.xml

# Node wrapper coverage
cd crates/node && npm install && npm run coverage && cd -

# Rust workspace coverage
eval "$(cargo llvm-cov show-env --sh)"
cargo llvm-cov clean --workspace
cargo nextest run --workspace \
  --exclude nemo-flow-node \
  --exclude nemo-flow-python \
  --exclude nemo-flow-wasm \
  --release --profile ci
cargo llvm-cov report --release \
  --ignore-filename-regex '(.*/crates/(node|python|wasm)/.*|.*/src/(coverage_tests|.*_tests)\.rs$)' \
  --cobertura --output-path target/coverage/rust-workspace.xml
```

## Binding Matrix

| Surface | Primary Command | Notes |
|---------|-----------------|-------|
| Core runtime | `cargo test -p nemo-flow-core` | Shared middleware, scopes, events, and ATIF behavior |
| Python binding | `uv run pytest` | Rebuild native extension if Rust-backed Python code changed |
| Go binding | `go test -race -v ./...` | Run from `go/nemo_flow` after building the FFI shared library |
| Node binding | `cd crates/node && npm install && npm test` | Runs JS integration tests against the native addon |
| WASM binding | `wasm-pack test --node crates/wasm` | Verifies the generated `wasm-bindgen` behavior path |
| Optimizer crate | `cargo test -p nemo-flow-optimizer` | Covers in-memory backend and end-to-end optimizer registration |
| Optimizer crate with Redis | `cargo test -p nemo-flow-optimizer --features redis-backend redis_tests` | Requires a local Redis instance at `redis://127.0.0.1/` |

## Test Helpers & Utilities

### Rust

- **`TEST_MUTEX`** (`context_isolation_tests.rs`, `stream_tests.rs`,
  `scope_local_tests.rs`): Static mutex that serializes tests touching global
  state, preventing interference between concurrent test threads.

> **Note:** When running core tests locally and encountering intermittent
> failures caused by shared global state, pass `--test-threads=1` to force
> serial execution:
>
> ```bash
> cargo test -p nemo-flow-core -- --test-threads=1
> ```
>
> CI already serializes via `TEST_MUTEX`, but single-threaded mode eliminates
> any residual timing sensitivity.

- **`reset_global()`**: Resets `NemoFlowContextState` to a clean default.
- **`make_llm_handle()`**: Creates an `LLMHandle` with defaults for stream tests.
- **`make_stream()`**: Builds a `tokio_stream` from `Vec<Result<Json>>`.
- **`make_collector_finalizer()`**: Returns a collector/finalizer pair backed by
  `Arc<Mutex<Vec<Json>>>` for asserting chunk reception.
- **`make_agent_info()`** (`atif.rs`): Creates a default `AtifAgentInfo` for
  ATIF tests.

### Python

- **`pytest-asyncio`** with `asyncio_mode = "auto"`: All `async def test_*`
  functions run automatically without explicit marks.
- Test classes group related tests and share setup via method-level construction.

### Node.js

- **`node:test`** built-in test runner with `describe`/`it` blocks.
- Helper functions like `envelopeCodec`, `pointCodec`, `makeNative()` defined
  inline in `typed_tests.mjs`.

### Go

- Standard `testing` package. Helper functions like `makeRequest()` defined
  inline in test files.
- Always run Go tests with the `-race` flag (`go test -race ./...`) to enable
  the race detector. The CGo callback trampolines bridge goroutines and the
  Rust runtime, so race detection is essential for catching unsafe concurrent
  access.

## Pre-Commit Hooks

The `.pre-commit-config.yaml` enforces quality gates before every commit:

| Hook | What It Checks |
|------|---------------|
| `check_copyright.py` | SPDX copyright headers on all source files |
| `trailing-whitespace` | No trailing whitespace |
| `end-of-file-fixer` | Files end with a newline |
| `check-yaml` / `check-toml` / `check-json` | Config file syntax |
| `ruff` (lint + format) | Python style and lint rules |
| `ty` | Python type checking |
| `cargo fmt` | Rust formatting |
| `cargo clippy` | Rust lints (`-D warnings`) |
| `cargo-deny` | Dependency audit via `deny.toml` |
| `gofmt` | Go formatting |
| `go vet` | Go static analysis |

## Coverage

Treat coverage as a **first-party code** metric. Exclude `third_party/`,
generated binaries, and debug symbol artifacts from any coverage target.

### CI Artifacts

Coverage artifacts are generated by the runtime-oriented GitLab jobs in
[.gitlab-ci.yml](.gitlab-ci.yml). There is
no repo-level coverage dashboard script.

The pipeline now uses five runtime test jobs plus one aggregate coverage job:

- `test:rust`: nextest JUnit plus `target/coverage/rust-workspace.xml`
  for the native Rust workspace (`core` + `ffi`)
- `test:python`: pytest JUnit plus `target/coverage/pytest_coverage_report.xml`
  as the primary GitLab coverage report, and `target/coverage/python-rust.xml`
  as an additional artifact for the Rust binding layer
- `test:node`: Node JUnit plus `target/coverage/cobertura-coverage.xml`
  as the primary GitLab coverage report, and `target/coverage/node-rust.xml`
  as an additional artifact for the Rust binding layer
- `test:go`: `go_junit_report.xml` plus `target/coverage/go_coverage_report.xml`
- `test:wasm`: native `cargo test -p nemo-flow-wasm --lib`, `wasm-pack`
  behavior tests, plus `target/coverage/wasm-rust.xml` and
  `target/coverage/wasm-js.xml`
  from a Node.js coverage harness over the generated WASM package
- `coverage:aggregate`: consumes all coverage artifacts and prints the single
  weighted pipeline coverage value GitLab should display

This keeps each runtime in a single place: one job runs the tests for that
surface and emits the coverage artifacts that belong to it.

### Python

Python tests use `pytest-cov` for coverage measurement:

```bash
uv run maturin develop
uv run pytest --cov=nemo_flow --cov-report=term-missing --cov-report=xml:target/coverage/pytest_coverage_report.xml
```

> **Important:** If you changed Rust code that backs the Python extension,
> rebuild the editable package (`uv run maturin develop`) before trusting
> Python coverage or lifecycle behavior.

Python currently targets **90%** line coverage for the wrapper module. To move
toward the high 90s, add tests for negative/error paths, codec fallbacks, and
every binding-level wrapper around the core runtime.

### Rust

For line coverage across the native Rust workspace, use `cargo-llvm-cov`:

```bash
cargo install cargo-llvm-cov
eval "$(cargo llvm-cov show-env --sh)"
cargo llvm-cov clean --workspace
cargo nextest run --workspace \
  --exclude nemo-flow-node \
  --exclude nemo-flow-python \
  --exclude nemo-flow-wasm \
  --release --profile ci
cargo llvm-cov report --release \
  --ignore-filename-regex '(.*/crates/(node|python|wasm)/.*|.*/src/(coverage_tests|.*_tests)\.rs$)' \
  --cobertura --output-path target/coverage/rust-workspace.xml
```

This workspace report intentionally excludes the Node, Python, and WASM binding
crates. Those surfaces are exercised through their own jobs and emit
binding-specific artifacts:
`target/coverage/node-rust.xml`, `target/coverage/python-rust.xml`, and the
WASM job's `target/coverage/wasm-rust.xml` and `target/coverage/wasm-js.xml`.

To measure Rust binding coverage that is only exercised via Node.js or Python,
run those external test suites under the `cargo-llvm-cov` environment:

```bash
# Node binding Rust coverage
cargo llvm-cov clean --workspace
source <(cargo llvm-cov show-env --sh)
cargo test -p nemo-flow-node --lib
cd crates/node && npm install && npm run build-debug && npm test && cd ../..
cargo llvm-cov report -p nemo-flow-node \
  --ignore-filename-regex '.*/src/(coverage_tests|.*_tests)\.rs$' \
  --cobertura --output-path target/coverage/node-rust.xml

# Python binding Rust coverage
cargo llvm-cov clean --workspace
source <(cargo llvm-cov show-env --sh)
export PYO3_PYTHON="$(uv python find)"
eval "$("$PYO3_PYTHON" - <<'PY'
import shlex
import sys
import sysconfig

def emit(name, value):
    print(f"export {name}={shlex.quote(value or '')}")

emit("PYTHONHOME", sys.base_prefix)
emit("PYTHON_STDLIB", sysconfig.get_path("stdlib"))
emit("PYTHON_PLATSTDLIB", sysconfig.get_path("platstdlib"))
emit("PYTHON_LIBDIR", sysconfig.get_config_var("LIBDIR"))
PY
)"
export PYTHONPATH="${PYTHON_STDLIB}:${PYTHON_PLATSTDLIB}"
export LD_LIBRARY_PATH="${PYTHON_LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cargo test -p nemo-flow-python --lib
uv run maturin develop
uv run pytest python/tests
cargo llvm-cov report -p nemo-flow-python \
  --ignore-filename-regex '.*/src/(coverage_tests|.*_tests)\.rs$' \
  --cobertura --output-path target/coverage/python-rust.xml
```

### Go

Go coverage should include the race detector and all packages:

```bash
cargo build --release -p nemo-flow-ffi
cd go/nemo_flow && \
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" \
go test -race -covermode=atomic -coverpkg=./... -coverprofile=coverage.out ./...
# Convert to Cobertura format for CI
go install github.com/boumenot/gocover-cobertura@latest
mkdir -p ../../target/coverage
gocover-cobertura < coverage.out > ../../target/coverage/go_coverage_report.xml
cd -
```

### Optimizer

Optimizer tests live under `crates/optimizer/tests/` and are split by backend:

- `runtime_integration_tests.rs` exercises optimizer registration, event capture, and
  hot-path intercept behavior using `InMemoryBackend`
- the same in-memory suite now covers hosted plugin registration, unknown hosted
  plugin diagnostics, and rollback on hosted plugin registration failure
- `redis_tests.rs` exercises `RedisBackend` storage round-trips and is gated by
  the `redis-backend` feature

Run the in-memory optimizer suite with:

```bash
cargo test -p nemo-flow-optimizer
```

Run the Redis-backed suite with:

```bash
cargo test -p nemo-flow-optimizer --features redis-backend redis_tests
```

Redis requirements:

- Redis must be reachable at `redis://127.0.0.1/`
- The Redis tests skip gracefully when Redis is unavailable
- The `redis-backend` Cargo feature must be enabled or the Redis test module is
  not compiled

### Node.js

Node wrapper coverage can be collected with `c8`:

```bash
cd crates/node && npm install && npm run coverage
cd -
```

This writes structured coverage reports under `crates/node/coverage/`, including
`coverage-summary.json` and `cobertura-coverage.xml`.

Node optimizer plugin coverage lives in
`crates/node/tests/optimizer_tests.mjs`.

### Python Optimizer Helpers

The external Python optimizer surface is validated through:

- `python/tests/test_optimizer.py`
- `python/tests/test_optimizer_config.py`

If the native module changes, rebuild it before trusting those tests:

```bash
uv run maturin develop
uv run pytest python/tests/test_optimizer.py python/tests/test_optimizer_config.py
```

### Go Optimizer Plugins

Go hosted optimizer plugin coverage lives in:

- `go/nemo_flow/optimizer_plugin_test.go`

Run it with the normal Go suite after building the FFI library:

```bash
cargo build --release -p nemo-flow-ffi
cd go/nemo_flow
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" go test ./...
```

### WASM Optimizer Plugins

WASM hosted optimizer plugin coverage lives in:

- `crates/wasm/tests/optimizer_tests.rs`

Run it with:

```bash
wasm-pack test --node crates/wasm
```

### WASM

WASM coverage is collected in three parts:

1. `cargo llvm-cov` + `cargo test -p nemo-flow-wasm --lib` emits
   `target/coverage/wasm-rust.xml` for the native Rust crate surface.
2. `wasm-pack test --node crates/wasm` validates the actual `wasm-bindgen`
   behavior path.
3. `wasm-pack build --target nodejs --out-dir pkg-test` plus a Node.js test
   harness under `crates/wasm/tests-js/` lets `c8` emit
   `target/coverage/wasm-js.xml` for the generated JavaScript wrapper.

This is intentionally JavaScript-side coverage for the generated package glue.
Stable Rust toolchains do not support source-level Cobertura coverage for
`wasm-bindgen-test`; that path currently requires nightly.
The `crates/wasm/tests-js/` harness should call only the public generated
bindings exported by the built package. Do not add tests that depend on private
`wasm-bindgen` helper internals, codegen structure, or VM-injected wrapper
implementation details.

## Writing New Tests

### Conventions

1. **Split by domain**: Each test file covers one domain (types, scope, tools,
   llm, scope_local, etc.). Add tests to the appropriate file.
2. **Mirror across bindings**: When adding a new feature, add tests in every
   binding that exposes it.
3. **Use descriptive names**: `test_<domain>_<behavior>_<condition>`, e.g.
   `test_exporter_merged_tool_observations`.
4. **Isolate global state**: In Rust integration tests, acquire `TEST_MUTEX`
   and call `reset_global()` before touching the default context.
5. **Async by default** (Python): Use `async def test_*` — `pytest-asyncio`
   handles the event loop.
6. **WASM has three test modes**: Unit tests run via `cargo test -p nemo-flow-wasm`
   (standard Rust test harness). Integration tests that exercise the full
   `wasm-bindgen` JavaScript interop require `wasm-pack test --node crates/wasm`,
   which compiles to WebAssembly and runs under Node.js. Coverage for the
   generated Node package comes from `crates/wasm/tests-js/*.mjs` under `c8`,
   and those tests should stay limited to public package exports. CI runs all
   three in `test:wasm`, and all three must pass.

### Adding a Rust Core Test

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_my_new_behavior() {
        // Arrange
        let exporter = AtifExporter::new("s".into(), make_agent_info());
        // Act
        let trajectory = exporter.export();
        // Assert
        assert!(trajectory.steps.is_empty());
    }
}
```

### Adding a Python Test

```python
# python/tests/test_<domain>.py
import pytest
import nemo_flow

class TestNewFeature:
    async def test_basic_behavior(self):
        handle = nemo_flow.scope.push("test", nemo_flow.ScopeType.Agent)
        assert handle.name == "test"
        nemo_flow.scope.pop(handle)
```

### Adding a Node.js Test

```javascript
// crates/node/tests/<domain>_tests.mjs
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { pushScope, popScope, ScopeType } from '../index.js';

describe('New feature', () => {
  it('basic behavior', () => {
    const handle = pushScope('test', ScopeType.Agent);
    assert.equal(handle.name, 'test');
    popScope(handle);
  });
});
```

### Adding a Go Test

```go
// go/nemo_flow/<domain>_test.go
func TestNewFeature(t *testing.T) {
    stack, err := nemo_flow.NewScopeStack()
    if err != nil {
        t.Fatalf("NewScopeStack failed: %v", err)
    }
    defer stack.Close()
    stack.Run(func() {
        handle, err := nemo_flow.PushScope("test", nemo_flow.ScopeTypeAgent)
        if err != nil {
            t.Fatalf("PushScope failed: %v", err)
        }
        nemo_flow.PopScope(handle)
    })
}
```
