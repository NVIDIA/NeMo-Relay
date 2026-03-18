<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing Guide

Nexus maintains comprehensive test coverage across all five language bindings.
Every binding mirrors the same test domains so that behavioral parity is verified
at each layer.

## Quick Reference

```bash
# ── Rust (core + WASM unit tests) ──────────────────────────
cargo test --workspace

# Core only
cargo test -p nvidia-nat-nexus-core

# WASM unit tests
cargo test -p nvidia-nat-nexus-wasm

# WASM integration tests (wasm-bindgen-test)
wasm-pack test --node crates/wasm

# ── Python ─────────────────────────────────────────────────
uv sync                        # build native extension + install deps
uv run pytest                  # run all Python tests
uv run pytest -k test_typed    # run a single module

# ── Go (requires FFI shared lib) ───────────────────────────
cargo build --release -p nvidia-nat-nexus-ffi
cd go/nat_nexus && CGO_LDFLAGS="-L../../target/release" go test -v ./...

# ── Node.js (requires native addon) ────────────────────────
cd crates/node && npm install && npm run build
node --test crates/node/tests/*.mjs

# ── Full suite ─────────────────────────────────────────────
cargo test --workspace && uv run pytest
```

## Test Helpers & Utilities

### Rust

- **`TEST_MUTEX`** (`context_isolation_tests.rs`, `stream_tests.rs`): Static
  mutex that serializes tests touching global state, preventing interference
  between concurrent test threads.
- **`reset_global()`**: Resets `NatNexusContextState` to a clean default.
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

## Writing New Tests

### Conventions

1. **Split by domain**: Each test file covers one domain (types, scope, tools,
   llm, etc.). Add tests to the appropriate file.
2. **Mirror across bindings**: When adding a new feature, add tests in every
   binding that exposes it.
3. **Use descriptive names**: `test_<domain>_<behavior>_<condition>`, e.g.
   `test_exporter_merged_tool_observations`.
4. **Isolate global state**: In Rust integration tests, acquire `TEST_MUTEX`
   and call `reset_global()` before touching the default context.
5. **Async by default** (Python): Use `async def test_*` — `pytest-asyncio`
   handles the event loop.

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
        let trajectory = exporter.export(None);
        // Assert
        assert!(trajectory.steps.is_empty());
    }
}
```

### Adding a Python Test

```python
# python/tests/test_<domain>.py
import pytest
import nat_nexus

class TestNewFeature:
    async def test_basic_behavior(self):
        handle = nat_nexus.scope.push("test", nat_nexus.ScopeType.Agent)
        assert handle.name == "test"
        nat_nexus.scope.pop(handle)
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
// go/nat_nexus/<domain>_test.go
func TestNewFeature(t *testing.T) {
    stack := nat_nexus.NewScopeStack()
    defer stack.Close()
    stack.Run(func() {
        handle, err := scope.Push("test", nat_nexus.ScopeTypeAgent, 0, nil)
        require.NoError(t, err)
        scope.Pop(handle)
    })
}
```
