<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Language Bindings

Nexus provides native bindings for Python, Node.js, Go, and WebAssembly. All bindings mirror the full API surface: scopes, tools, LLM, guardrails, intercepts, subscribers, and ATIF export.

## Architecture

```mermaid
graph TD
    subgraph "Application Code"
        PY[Python]
        JS[Node.js]
        GO[Go]
        WA[Browser/WASM]
    end

    subgraph "Binding Layers"
        PYO3["PyO3 (abi3, Python 3.11+)"]
        NAPI["NAPI-RS (Node.js addon)"]
        FFI["C FFI (cbindgen → nat_nexus.h)"]
        WASM["wasm-bindgen"]
    end

    subgraph "Core"
        CORE["nvidia-nat-nexus-core (Rust)"]
    end

    PY --> PYO3 --> CORE
    JS --> NAPI --> CORE
    GO --> FFI --> CORE
    WA --> WASM --> CORE
```

## Naming Conventions

| Aspect | Python | Go | Node.js | WASM | FFI/C |
|--------|--------|----|---------|------|-------|
| Functions | `snake_case` | `PascalCase` | `camelCase` | `camelCase` | `nat_nexus_snake_case` |
| Types | `PascalCase` | `PascalCase` | `PascalCase` | `PascalCase` | `FfiPascalCase` |
| Enums | `ScopeType.Agent` | `ScopeTypeAgent` | `ScopeType.Agent` | `ScopeType.Agent` | `NatNexusScopeTypeAgent` |
| Errors | `RuntimeError` | `error` | JS exception | JS exception | `NatNexusStatus` + `nat_nexus_last_error()` |

## Python

### Setup

```bash
uv sync        # Create venv, install deps, build native extension
uv run pytest  # Run tests
```

### Module Structure

```
python/nat_nexus/
  __init__.py       # Re-exports, ContextVar-based scope isolation
  scope.py          # Scope operations
  tools.py          # Tool lifecycle
  llm.py            # LLM lifecycle
  guardrails.py     # Guardrail registration
  intercepts.py     # Intercept registration
  subscribers.py    # Event subscriber registration
  typed.py          # Codec-based typed wrappers
```

The Python package wraps a PyO3 native extension (`_native`) built with the stable ABI (abi3), producing a single `.so` compatible with Python 3.11+.

### Usage

```python
import nat_nexus

# Guardrails
nat_nexus.guardrails.register_tool_conditional_execution(
    "block_dangerous", 1,
    lambda name, args: "blocked" if name == "rm" else None,
)

# Intercepts
nat_nexus.intercepts.register_tool_request(
    "add_context", 1, False,
    lambda name, args: {**args, "context": "injected"},
)

# Scope Context Management
with nat_nexus.scope.scope("my_agent", nat_nexus.ScopeType.Agent) as handle:
    # Inside this block, the scope "my_agent" is active
    ...

# Alternatively, manual scope push/pop:
handle = nat_nexus.scope.push("my_agent", nat_nexus.ScopeType.Agent)
nat_nexus.scope.pop(handle)

# The following examples assume you are inside an active scope context.
# Some require running inside of a coroutine (the ones that use an `await` expression).

# Tool execution
result = await nat_nexus.tools.execute("search", {"q": "test"}, search_func)

# LLM execution
request = nat_nexus.LLMRequest(
    headers={"Authorization": "Bearer ..."},
    content={"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"},
)
response = await nat_nexus.llm.execute("gpt-4", request, llm_func)
```

### Context Isolation

Python uses `contextvars.ContextVar` for async-safe per-task isolation. Each `asyncio.Task` can have its own scope stack:

```python
async def handle_request():
    # get_scope_stack() lazily creates an isolated stack per task
    nat_nexus.get_scope_stack()
    # All scope operations now use this isolated stack
```

Check whether a scope stack is active, and propagate to worker threads:

```python
if nat_nexus.scope_stack_active():
    stack = nat_nexus.propagate_scope_to_thread()
    # Pass `stack` to worker, call nat_nexus.set_thread_scope_stack(stack) there
```

## Node.js

### Setup

```bash
cd crates/node
npm install
npm run build        # Build .node addon
node --test tests/*.mjs  # Run tests
```

### Usage

```javascript
import {
    pushScope, popScope, ScopeType,
    toolCallExecute, llmCallExecute,
    registerToolRequestIntercept,
} from './index.js';

// Replace with your actual tool and LLM functions
function searchFunc() {
    return { ok: true };
}

function llmFunc(n) {
    return { response: 'hello from llm' };
}

// Intercepts
registerToolRequestIntercept("add_ctx", 1, false, (name, args) => {
    console.log("Intercepted tool call: ", name, args);
    return { ...args, context: "injected" };
});

// Scopes
const handle = pushScope("my_agent", ScopeType.Agent, null, null);

// Tool execution
const result = await toolCallExecute(
    "search", { q: "test" }, searchFunc,
    null, null, null, null,
);

// LLM execution
const request = { headers: {}, content: { messages: [{"role": "user", "content": "Hello"}], model: "gpt-4" } };
const response = await llmCallExecute(
    "gpt-4", request, llmFunc,
    null, null, null, null, "gpt-4",
);

console.log("LLM response: ", response);

popScope(handle);
process.exit(0);
```

### Typed Wrappers

Node.js provides `typed.js` with `typedToolExecute`, `typedLlmExecute`, and `typedLlmStreamExecute`:

```javascript
import { typedToolExecute } from './typed.js';

const result = await typedToolExecute(
    "search", new SearchArgs("test"),
    searchFunc, argsCodec, resultCodec,
);
```

### Stream Bridge

Node.js uses a push-based stream bridge for LLM streaming. JavaScript drives async iteration and pushes chunks back to the native layer via `pushStreamChunk()` / `endStream()`.

## Go

### Setup

```bash
# Build the FFI shared library first
cargo build --release -p nvidia-nat-nexus-ffi

# Run Go tests
cd go/nat_nexus
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" go test -v ./...
```

### Package Structure

```
go/nat_nexus/
  nat_nexus.go        # CGo declarations, core bindings
  types.go          # Type definitions (ScopeHandle, ToolHandle, etc.)
  stream.go         # LLM stream handling
  callbacks.go      # Go trampolines for Rust callbacks
  scope/            # Convenience package
  tools/            # Convenience package
  llm/              # Convenience package
  guardrails/       # Convenience package
  intercepts/       # Convenience package
  subscribers/      # Convenience package
```

### Usage

```go
import (
    "encoding/json"
    "fmt"

    "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
)

func searchFunc(args json.RawMessage) (json.RawMessage, error) {
	var input map[string]interface{}
	json.Unmarshal(args, &input)
	result, _ := json.Marshal(map[string]interface{}{"results": []string{"result for: " + input["q"].(string)}})
	return result, nil
}

func llmFunc(request json.RawMessage) (json.RawMessage, error) {
	result, _ := json.Marshal(map[string]interface{}{"response": "hello from llm"})
	return result, nil
}

func main() {
    // Scopes
    handle, _ := nat_nexus.PushScope("my_agent", nat_nexus.ScopeTypeAgent)

    // Tool execution
    result, _ := nat_nexus.ToolCallExecute("search", json.RawMessage(`{"q": "test"}`), searchFunc)

    fmt.Println("tool result:", string(result))


    // LLM execution
    request := map[string]interface{}{
        "headers": map[string]interface{}{},
        "content": map[string]interface{}{
            "messages": []interface{}{map[string]interface{}{"role": "user", "content": "Hello"}},
            "model":    "gpt-4",
        },
    }
    response, _ := nat_nexus.LlmCallExecute("gpt-4", request, llmFunc, nat_nexus.WithLLMModelName("gpt-4"))
    fmt.Println("llm response:", string(response))

    nat_nexus.PopScope(handle)
}
```

### CGo Callback Pattern

Go uses trampolines — C-compatible function pointers that bridge Rust callbacks to Go functions:

```go
// callbacks.go defines trampolines
//export goToolSanitizeTrampoline
func goToolSanitizeTrampoline(userData unsafe.Pointer, name *C.char, args *C.char) *C.char { ... }
```

Memory management requires explicit `Free()` calls on handles and scope stacks.

### Context Isolation

Go goroutines use `ScopeStack.Run()` which pins the goroutine to an OS thread:

```go
stack, _ := nat_nexus.NewScopeStack()
defer stack.Close()

go func() {
    stack.Run(func() {
        // All scope operations use this stack
        scope.Push("agent", scope.TypeAgent)

        // Check if a scope stack is explicitly bound
        if nat_nexus.ScopeStackActive() {
            // ...
        }
    })
}()
```

## WebAssembly

### Setup

```bash
wasm-pack build crates/wasm    # Produces pkg/ with .wasm, .js, .d.ts

# Unit tests
cargo test -p nvidia-nat-nexus-wasm

# Integration tests
wasm-pack test --node crates/wasm
```

### Usage

```javascript
import {
    pushScope, popScope,
} from './pkg/nvidia_nat_nexus_wasm.js';

const handle = pushScope("agent", 0 /* SCOPE_TYPE_AGENT */, null, null);
// ... operations ...
popScope(handle);
```

### Differences from Node.js

- Functions are prefixed with `natNexus` in some cases (via `#[wasm_bindgen(js_name = "...")]`)
- Single-threaded (no worker thread isolation)
- Uses `wasm_bindgen_futures::spawn_local()` for async execution
- Stream objects expose an async `next()` method

## Comparison Table

| Feature | Python | Go | Node.js | WASM |
|---------|--------|----|---------|------|
| Build tool | uv / PyO3 | CGo | napi-build | wasm-pack |
| Output | `.so` (abi3) | CGo packages | `.node` addon | `.wasm` + `.js` |
| Async | asyncio | goroutines | event loop | spawn_local |
| Context isolation | `contextvars` + `scope_stack_active()` | `ScopeStack.Run()` + `ScopeStackActive()` | `setThreadScopeStack()` + `scopeStackActive()` | `setThreadScopeStack()` + `scopeStackActive()` |
| Callback pattern | `PyAny` → closure | C trampolines | `ThreadsafeFunction` | `js_sys::Function` |
| Stream support | AsyncIterator | Channel-based | Push-based bridge | Async iterator |
| Typed wrappers | `nat_nexus.typed` | — | `typed.js` | — |
| Memory management | GC | Manual (`Free`/`Close`) | GC | GC |

## Error Handling

All bindings map core `NexusError` variants to language-appropriate errors:

| Error | Python | Go | Node.js / WASM |
|-------|--------|----|-----------------|
| `AlreadyExists` | `RuntimeError` | `error` | thrown exception |
| `NotFound` | `RuntimeError` | `error` | thrown exception |
| `GuardrailRejected` | `RuntimeError` | `error` | thrown exception |
| `ScopeStackEmpty` | `RuntimeError` | `error` | thrown exception |
| `Internal` | `RuntimeError` | `error` | thrown exception |

Go additionally provides the FFI pattern of `NatNexusStatus` return codes with `nat_nexus_last_error()` for the error message string.
