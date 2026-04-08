<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Language Bindings

Nexus provides native bindings for Python, Node.js, Go, and WebAssembly. All bindings mirror the full API surface: scopes, tools, LLM, guardrails, intercepts, subscribers, and ATIF export.

Across all bindings, subscriber callbacks run synchronously on the calling
thread after Nexus snapshots the subscriber list and releases its runtime
locks. They may call back into Nexus APIs, but they should remain lightweight
because they still execute on the request path.

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

## Rust Core Notes

Direct Rust users of `nvidia-nat-nexus-core` should note that
`EventSubscriberFn` is an `Arc<dyn Fn(&Event) + Send + Sync>`. Register
subscribers with `Arc::new(...)`, not `Box::new(...)`.

If you want OTLP export without adding exporter logic to your own callback,
use the separate `nvidia-nat-nexus-otel` crate. It turns Nexus lifecycle
events into OpenTelemetry spans and exposes a normal `EventSubscriberFn`.
If you want OTLP export with OpenInference semantic conventions, use the
separate `nvidia-nat-nexus-openinference` crate instead.

## OpenTelemetry

Every binding exposes an OpenTelemetry subscriber backed by the same Rust OTLP
exporter. The config shape follows each language's normal style:

- Rust: `OpenTelemetryConfig::{http_binary, grpc}(...)` builder-style config.
- Python: mutable `OpenTelemetryConfig()` object passed to `OpenTelemetrySubscriber(config)`.
- Node.js: plain `OpenTelemetryConfig` object passed to `new OpenTelemetrySubscriber(config)`.
- Go: `NewOpenTelemetryConfig()` returns a mutable config struct for `NewOpenTelemetrySubscriber(config)`.
- WASM: `defaultOpenTelemetryConfig()` returns a mutable JS object for `new OpenTelemetrySubscriber(config)`.

Use [Observability with OpenTelemetry](observability-with-opentelemetry.md) as
the canonical guide for event mapping, lifecycle, transport constraints, and
full per-language setup examples.

Minimal examples:

```python
import nat_nexus

config = nat_nexus.OpenTelemetryConfig()
config.endpoint = "http://localhost:4318/v1/traces"
config.service_name = "demo-agent"

subscriber = nat_nexus.OpenTelemetrySubscriber(config)
subscriber.register("otel")
```

```javascript
import { OpenTelemetrySubscriber } from "./index.js";

const config = {
  endpoint: "http://localhost:4318/v1/traces",
  serviceName: "demo-agent",
};

const subscriber = new OpenTelemetrySubscriber(config);
subscriber.register("otel");
```

## OpenInference

Every binding also exposes an OpenInference subscriber backed by the same OTLP
transport layer, but annotated with OpenInference semantic conventions for
backends such as Phoenix.

- Rust: `OpenInferenceConfig::new()` plus chained setters.
- Python: mutable `OpenInferenceConfig()` object passed to `OpenInferenceSubscriber(config)`.
- Node.js: plain `OpenInferenceConfig` object passed to `new OpenInferenceSubscriber(config)`.
- Go: `NewOpenInferenceConfig()` returns a mutable config struct for `NewOpenInferenceSubscriber(config)`.
- WASM: `defaultOpenInferenceConfig()` returns a mutable JS object for `new OpenInferenceSubscriber(config)`.

Use [Observability with OpenInference](observability-with-openinference.md) as
the canonical guide for semantic mapping, lifecycle, transport constraints, and
per-language setup examples.

Minimal examples:

```python
import nat_nexus

config = nat_nexus.OpenInferenceConfig()
config.endpoint = "http://localhost:4318/v1/traces"
config.service_name = "demo-agent"

subscriber = nat_nexus.OpenInferenceSubscriber(config)
subscriber.register("openinference")
```

```javascript
import { OpenInferenceSubscriber } from "./index.js";

const config = {
  endpoint: "http://localhost:4318/v1/traces",
  serviceName: "demo-agent",
};

const subscriber = new OpenInferenceSubscriber(config);
subscriber.register("openinference");
```

```go
config := nat_nexus.NewOpenInferenceConfig()
config.Endpoint = "http://localhost:4318/v1/traces"
config.ServiceName = "demo-agent"

subscriber, err := nat_nexus.NewOpenInferenceSubscriber(config)
```

```javascript
import init, {
  defaultOpenInferenceConfig,
  OpenInferenceSubscriber,
} from "./pkg/nvidia_nat_nexus_wasm.js";

await init();

const config = defaultOpenInferenceConfig();
config.endpoint = "http://localhost:4318/v1/traces";
config.service_name = "demo-agent";

const subscriber = new OpenInferenceSubscriber(config);
subscriber.register("openinference");
```

```go
config := nat_nexus.NewOpenTelemetryConfig()
config.Endpoint = "http://localhost:4318/v1/traces"
config.ServiceName = "demo-agent"

subscriber, err := nat_nexus.NewOpenTelemetrySubscriber(config)
```

```javascript
import init, {
  defaultOpenTelemetryConfig,
  OpenTelemetrySubscriber,
} from "./pkg/nvidia_nat_nexus_wasm.js";

await init();

const config = defaultOpenTelemetryConfig();
config.endpoint = "http://localhost:4318/v1/traces";
config.service_name = "demo-agent";

const subscriber = new OpenTelemetrySubscriber(config);
subscriber.register("otel");
```

## Callback Contracts

Nexus intentionally distinguishes fallible callback surfaces from infallible
ones:

| Surface | Contract | Failure Behavior |
|--------|----------|------------------|
| Sanitize guardrails | Infallible | Handle failures inside the callback; there is no propagated error channel |
| Conditional execution guardrails | Fallible | Callback failure aborts the originating Nexus call |
| Request intercepts | Fallible | Callback failure aborts the originating Nexus call |
| Execution intercepts | Fallible | Callback failure aborts the originating Nexus call |
| Stream collector | Fallible | Callback failure aborts the stream |
| Stream finalizer | Infallible | Handle failures inside the callback; there is no propagated error channel |
| Subscribers | Infallible | Handle failures inside the callback; there is no propagated error channel |

Language-specific error surfacing:

- Rust uses `Result<...>` for fallible conditionals, request intercepts,
  execution intercepts, and stream collectors.
- Python uses normal callback return types; raising an exception from a
  fallible callback propagates as `RuntimeError` from the originating Nexus API
  call.
- Node.js and WASM use normal callback return types; throwing from a fallible
  callback propagates as the thrown JS exception from the originating Nexus API
  call.
- Go follows the FFI callback surface, which remains the least expressive
  binding here; consult the Go API docs before assuming parity with the higher
  level bindings.

These contracts are canonical per surface. Bindings do not expose parallel
"fallible" and "infallible" variants for the same conditional guardrail or
request intercept shape; whether a callback is fallible depends on the surface
you register, not on which helper name you choose internally.

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
  scope_local.py    # Scope-local middleware registration
  optimizer.py      # Config-driven optimizer runtime helpers
  typed.py          # Codec-based typed wrappers
```

The Python package wraps a PyO3 native extension (`_native`) built with the stable ABI (abi3), producing a single `.so` compatible with Python 3.11+.

### Optimizer Runtime

Python exposes typed optimizer helpers in `nat_nexus.optimizer`:

```python
from nat_nexus.optimizer import (
    BackendSpec,
    OptimizerConfig,
    OptimizerRuntime,
    StateConfig,
    TelemetryComponent,
)

runtime = OptimizerRuntime(
    OptimizerConfig(
        state=StateConfig(backend=BackendSpec.in_memory()),
        components=[TelemetryComponent(learners=["latency_sensitivity"])],
    )
)

report = runtime.report()
await runtime.register()
runtime.deregister()
await runtime.shutdown()
```

### Optimizer Hosted Plugins

Python can register hosted optimizer plugins that the Rust optimizer runtime
calls during validation and registration.

```python
from nat_nexus import LLMRequest
from nat_nexus.optimizer import (
    ExternalComponent,
    OptimizerConfig,
    OptimizerRuntime,
    register_optimizer_plugin,
)

class HeaderPlugin:
    def validate(self, instance_id, plugin_config):
        return []

    def register(self, instance_id, plugin_config, context):
        def intercept(_name, request, annotated):
            headers = dict(request.headers)
            headers["x-plugin"] = instance_id
            return LLMRequest(headers, request.content), annotated

        context.register_llm_request_intercept(
            f"{instance_id}.header",
            25,
            False,
            intercept,
        )

register_optimizer_plugin("example.header_plugin", HeaderPlugin())

runtime = OptimizerRuntime(
    OptimizerConfig(
        components=[
            ExternalComponent(
                plugin_kind="example.header_plugin",
                instance_id="plugin-1",
            )
        ]
    )
)
```

`context` exposes:

- `register_subscriber(...)`
- `register_llm_request_intercept(...)`
- `register_llm_execution_intercept(...)`
- `register_llm_stream_execution_intercept(...)`
- `register_tool_request_intercept(...)`
- `register_tool_execution_intercept(...)`

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

### Scope-Local Middleware

```python
import nat_nexus

handle = nat_nexus.scope.push("session", nat_nexus.ScopeType.Agent)

# Register middleware bound to this scope
nat_nexus.scope_local.register_tool_conditional_execution(
    handle, "session_guard", 10,
    lambda name, args: "blocked" if name == "rm" else None,
)
nat_nexus.scope_local.register_subscriber(
    handle, "session_logger", lambda event: print(event.name),
)

# ... middleware is active while scope is on the stack ...

nat_nexus.scope.pop(handle)  # both registrations automatically removed
```

During the scope's `Start` callback, `get_handle()` sees `handle` as the active
scope. During its `End` callback, `get_handle()` sees the parent scope because
the pop has already completed.

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
npm test             # Build debug addon and run JS integration tests
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

### Scope-Local Middleware

```javascript
import {
    pushScope, popScope, ScopeType,
    scopeRegisterToolConditionalExecution,
    scopeRegisterSubscriber,
} from './index.js';

const handle = pushScope("session", ScopeType.Agent, null, null);

// Register middleware bound to this scope
scopeRegisterToolConditionalExecution(
    handle, "session_guard", 10,
    (name, args) => name === "rm" ? "blocked" : null,
);
scopeRegisterSubscriber(
    handle, "session_logger",
    (event) => console.log(event.name),
);

// ... middleware is active while scope is on the stack ...

popScope(handle);  // both registrations automatically removed
```

For infallible callback shapes that do not have an error return channel
(for example sanitize guardrails, subscribers, and stream finalizers), the
Node binding records the most recent binding-side callback failure. Read it with
`getLastCallbackError()` and clear it with `clearLastCallbackError()`.

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

### Optimizer Runtime

Node exposes optimizer helpers through `typed.js` and validation through the
generated addon:

```javascript
import { validateOptimizerConfig } from "./index.js";
import {
  OptimizerRuntime,
  defaultOptimizerConfig,
  optimizerInMemoryBackend,
  telemetryComponent,
} from "./typed.js";

const config = defaultOptimizerConfig();
config.state = { backend: optimizerInMemoryBackend() };
config.components = [telemetryComponent({ learners: ["latency_sensitivity"] })];

const validation = validateOptimizerConfig(config);
const runtime = new OptimizerRuntime(config);
const report = await runtime.report();
await runtime.register();
await runtime.deregister();
await runtime.shutdown();
```

### Optimizer Hosted Plugins

Node exposes hosted optimizer plugins through `registerOptimizerPlugin(...)`
and the `externalComponent(...)` helper in `typed.js`:

```javascript
import {
  OptimizerRuntime,
  defaultOptimizerConfig,
  externalComponent,
  registerOptimizerPlugin,
} from "./typed.js";

registerOptimizerPlugin("example.header_plugin", {
  validate(instanceId, pluginConfig) {
    return [];
  },
  register(instanceId, pluginConfig, context) {
    context.registerLlmRequestIntercept(
      `${instanceId}.header`,
      25,
      false,
      (name, request, annotated) => [
        {
          headers: { ...request.headers, "x-plugin": instanceId },
          content: request.content,
        },
        annotated,
      ],
    );
  },
});

const config = defaultOptimizerConfig();
config.components = [externalComponent("example.header_plugin", "plugin-1", {})];
const runtime = new OptimizerRuntime(config);
```

Node hosted plugin contexts expose:

- `registerSubscriber(...)`
- `registerLlmRequestIntercept(...)`
- `registerLlmExecutionIntercept(...)`
- `registerLlmStreamExecutionIntercept(...)`
- `registerToolRequestIntercept(...)`
- `registerToolExecutionIntercept(...)`

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
  optimizer.go      # Optimizer config/runtime wrapper
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

### Scope-Local Middleware

```go
import (
    "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
    "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/scope"
)

handle, _ := scope.Push("session", nat_nexus.ScopeTypeAgent, 0, nil)

// Register middleware bound to this scope
nat_nexus.ScopeRegisterToolConditionalExecution(handle, "session_guard", 10,
    func(name string, args json.RawMessage) *string {
        if name == "rm" {
            reason := "blocked"
            return &reason
        }
        return nil
    },
)
nat_nexus.ScopeRegisterSubscriber(handle, "session_logger",
    func(event json.RawMessage) { fmt.Println("event:", string(event)) },
)

// ... middleware is active while scope is on the stack ...

scope.Pop(handle)  // both registrations automatically removed
```

### CGo Callback Pattern

Go uses trampolines — C-compatible function pointers that bridge Rust callbacks to Go functions:

```go
// callbacks.go defines trampolines
//export goToolSanitizeTrampoline
func goToolSanitizeTrampoline(userData unsafe.Pointer, name *C.char, args *C.char) *C.char { ... }
```

Memory management requires explicit `Free()` calls on handles and scope stacks.

### Optimizer Runtime

Go exposes typed optimizer config builders and a synchronous runtime wrapper:

```go
config := nat_nexus.NewOptimizerConfig()
config.State = &nat_nexus.OptimizerStateConfig{
    Backend: nat_nexus.NewInMemoryOptimizerBackend(),
}
config.Components = []nat_nexus.OptimizerComponentSpec{
    nat_nexus.TelemetryComponent(nat_nexus.TelemetryComponentConfig{
        Learners: []string{"latency_sensitivity"},
    }),
}

report, err := nat_nexus.ValidateOptimizerConfig(config)
runtime, err := nat_nexus.NewOptimizerRuntime(config)
if err != nil {
    panic(err)
}
defer runtime.Close()

_ = report
_ = runtime.Register()
_ = runtime.Deregister()
_ = runtime.Shutdown()
```

### Optimizer Hosted Plugins

The Go binding exposes hosted plugin registration plus a temporary registration
context for adding subscribers and intercepts:

```go
pluginKind := "example.header_plugin"
err := nat_nexus.RegisterOptimizerPlugin(pluginKind, nat_nexus.OptimizerPluginHandlerFuncs{
    ValidateFunc: func(instanceID string, pluginConfig map[string]any) ([]nat_nexus.OptimizerConfigDiagnostic, error) {
        return nil, nil
    },
    RegisterFunc: func(instanceID string, pluginConfig map[string]any, ctx *nat_nexus.OptimizerPluginContext) error {
        return ctx.RegisterLlmRequestIntercept(
            instanceID+".header",
            25,
            false,
            func(name string, request map[string]any, annotated map[string]any) (map[string]any, map[string]any, error) {
                headers, _ := request["headers"].(map[string]any)
                if headers == nil {
                    headers = map[string]any{}
                }
                headers["x-plugin"] = instanceID
                request["headers"] = headers
                return request, annotated, nil
            },
        )
    },
})
if err != nil {
    panic(err)
}

config := nat_nexus.NewOptimizerConfig()
config.Components = []nat_nexus.OptimizerComponentSpec{
    nat_nexus.ExternalComponent(nat_nexus.ExternalComponentConfig{
        PluginKind: pluginKind,
        InstanceID: "plugin-1",
    }),
}
```

`OptimizerPluginContext` exposes:

- `RegisterSubscriber(...)`
- `RegisterLlmRequestIntercept(...)`
- `RegisterLlmExecutionIntercept(...)`
- `RegisterLlmStreamExecutionIntercept(...)`
- `RegisterToolRequestIntercept(...)`
- `RegisterToolExecutionIntercept(...)`

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
wasm-pack build crates/wasm --scope nvidia # Produces pkg/ with .wasm, .js, .d.ts

# Unit tests
cargo test -p nat-nexus-wasm

# Integration tests
wasm-pack test --node crates/wasm
```

The Cargo package remains `nat-nexus-wasm`, while the compiled WASM library
target and generated npm package are NVIDIA-branded (`nvidia_nat_nexus_wasm`
and `@nvidia/nat-nexus-wasm`).

### Build Targets

`wasm-pack` supports several output targets depending on your runtime
environment:

```bash
# Bundler (webpack, Vite, Rollup, etc.) — default
wasm-pack build crates/wasm --scope nvidia --target bundler

# Standalone web (loads via <script type="module">, no bundler needed)
wasm-pack build crates/wasm --scope nvidia --target web

# Node.js (CommonJS, for server-side or CLI usage)
wasm-pack build crates/wasm --scope nvidia --target nodejs
```

| Target | Output | Use Case |
|--------|--------|----------|
| `bundler` | ES module with `.wasm` sidecar | Bundled web apps (webpack, Vite) |
| `web` | ES module with manual `init()` | Standalone `<script type="module">` |
| `nodejs` | CommonJS with Node.js WASM loader | Server-side, CLI, or testing |

When using `--target web`, you must call the default-exported `init()` function
before invoking any other API:

```javascript
import init, { pushScope, popScope } from './pkg/nvidia_nat_nexus_wasm.js';

await init();  // loads and instantiates the .wasm binary
// Now the API is ready
```

### Usage

The following example demonstrates the full lifecycle: initializing the module,
pushing a scope, registering a tool, executing the tool through the middleware
pipeline, registering a guardrail, and popping the scope.

```javascript
import init, {
    pushScope, popScope,
    toolCallExecute,
    registerToolConditionalExecutionGuardrail,
    SCOPE_TYPE_AGENT,
} from './pkg/nvidia_nat_nexus_wasm.js';

// Required for --target web; no-op when using bundler or nodejs targets
await init();

// 1. Push a scope
const handle = pushScope("my_agent", SCOPE_TYPE_AGENT, null, null, null, null);

// 2. Register a guardrail that blocks dangerous tools
registerToolConditionalExecutionGuardrail(
    "block_dangerous", 1,
    (name, args) => name === "rm" ? "blocked: dangerous tool" : null,
);

// 3. Define a tool function
async function searchFunc(args) {
    return { results: [`result for: ${args.q}`] };
}

For infallible callback shapes that do not have an error return channel
(for example sanitize guardrails, subscribers, and stream finalizers), the
WASM binding also records the most recent binding-side callback failure. Read it with
`getLastCallbackError()` and clear it with `clearLastCallbackError()`.

// 4. Execute a tool through the full middleware pipeline
const result = await toolCallExecute(
    "search",
    { q: "test" },
    searchFunc,
    null,  // parent (uses current scope)
    null,  // attributes
    null,  // data
    null,  // metadata
);
console.log("Tool result:", result);

// 5. Pop the scope
popScope(handle);
```

### Scope-Local Middleware

```javascript
import {
    pushScope, popScope,
    scope_register_tool_conditional_execution,
    scope_register_subscriber,
} from './pkg/nvidia_nat_nexus_wasm.js';

const handle = pushScope("session", 0 /* SCOPE_TYPE_AGENT */, null, null);

scope_register_tool_conditional_execution(
    handle, "session_guard", 10,
    (name, args) => name === "rm" ? "blocked" : null,
);
scope_register_subscriber(
    handle, "session_logger",
    (event) => console.log(event),
);

// ... operations ...

popScope(handle);  // auto-cleanup
```

### Optimizer Runtime

The WASM binding uses plain JavaScript objects for optimizer config:

```javascript
import init, {
  OptimizerRuntime,
  validateOptimizerConfig,
} from "./pkg/nvidia_nat_nexus_wasm.js";

await init();

const config = {
  version: 1,
  state: { backend: { kind: "in_memory", config: {} } },
  components: [
    { kind: "telemetry", enabled: true, config: { learners: ["latency_sensitivity"] } },
  ],
};

const validation = validateOptimizerConfig(config);
const runtime = new OptimizerRuntime(config);
runtime.report();
await runtime.register();
runtime.deregister();
await runtime.shutdown();
```

### Optimizer Hosted Plugins

The WASM binding also supports hosted plugins. Register the plugin handler in
JavaScript, then activate it through `external_component` in the optimizer
config:

```javascript
import init, {
    OptimizerRuntime,
    registerOptimizerPlugin,
    deregisterOptimizerPlugin,
    validateOptimizerConfig,
} from './pkg/nvidia_nat_nexus_wasm.js';

await init();

registerOptimizerPlugin("example.header_plugin", {
    validate(instanceId, pluginConfig) {
        return [];
    },
    register(instanceId, pluginConfig, context) {
        context.registerLlmRequestIntercept(
            `${instanceId}.header`,
            25,
            false,
            (name, request, annotated) => [
                {
                    headers: { ...request.headers, "x-plugin": instanceId },
                    content: request.content,
                },
                annotated,
            ],
        );
    },
});

const config = {
    version: 1,
    components: [
        {
            kind: "external_component",
            enabled: true,
            config: {
                plugin_kind: "example.header_plugin",
                instance_id: "plugin-1",
            },
        },
    ],
};

console.log(validateOptimizerConfig(config));
const runtime = new OptimizerRuntime(config);
await runtime.register();
runtime.deregister();
await runtime.shutdown();
deregisterOptimizerPlugin("example.header_plugin");
```

WASM hosted plugin contexts expose:

- `registerSubscriber(...)`
- `registerLlmRequestIntercept(...)`
- `registerLlmExecutionIntercept(...)`
- `registerLlmStreamExecutionIntercept(...)`
- `registerToolRequestIntercept(...)`
- `registerToolExecutionIntercept(...)`

WASM stream execution note:

- `registerLlmStreamExecutionIntercept(...)` in the WASM binding produces a
  single-item stream result directly and does not delegate to downstream stream
  handlers. WASM hosted plugins therefore cannot chain stream execution
  intercepts the same way the Rust, Python, Go, and Node.js bindings can.

### Streaming LLM Example

The WASM binding supports streaming LLM responses through a collector/finalizer
pattern. The `llmStreamCallExecute` function returns a `WasmLlmStream` object
whose `next()` method yields `{ value, done }` chunks, compatible with the
JavaScript async iterator protocol.

```javascript
import init, {
    pushScope, popScope,
    llmStreamCallExecute,
    SCOPE_TYPE_AGENT,
} from './pkg/nvidia_nat_nexus_wasm.js';

await init();

const handle = pushScope("llm_agent", SCOPE_TYPE_AGENT, null, null, null, null);

// Collector: accumulates chunks as they arrive
const chunks = [];
function collector(chunk) {
    chunks.push(chunk);
}

// Finalizer: called once when the stream ends; returns the aggregated response
function finalizer() {
    return { full_response: chunks.map(c => c.text || "").join("") };
}

// LLM function that returns a streaming response (simulated here)
async function llmFunc(request) {
    return { response: "streamed content" };
}

const request = {
    headers: { "Authorization": "Bearer ..." },
    content: { messages: [{ role: "user", content: "Hello" }], model: "gpt-4" },
};

// Execute the streaming call
const stream = await llmStreamCallExecute(
    "gpt-4",
    request,
    llmFunc,
    collector,    // optional: receives each chunk
    finalizer,    // optional: produces aggregated response on stream end
    null,         // parent
    null,         // attributes
    null,         // data
    null,         // metadata
    "gpt-4",     // model_name
);

// Consume the stream
while (true) {
    const { value, done } = await stream.next();
    if (done) break;
    console.log("Chunk:", value);
}

popScope(handle);
```

### Promise-Aware `withScope`

The `withScope` helper pushes a scope, runs a callback, and automatically pops
the scope when the callback completes. If the callback returns a `Promise`,
the scope remains active until the Promise settles (resolves or rejects),
making it safe for async workflows:

```javascript
import { withScope, toolCallExecute, SCOPE_TYPE_AGENT } from './pkg/nvidia_nat_nexus_wasm.js';

// Synchronous callback — scope is popped immediately on return
const syncResult = withScope("sync_op", SCOPE_TYPE_AGENT, (handle) => {
    return { status: "done" };
});

// Async callback — scope stays active until the Promise resolves
const asyncResult = await withScope("async_op", SCOPE_TYPE_AGENT, async (handle) => {
    const result = await toolCallExecute("search", { q: "test" }, searchFunc, null, null, null, null);
    return result;
});
// Scope is automatically popped here, even if the Promise rejects
```

`withScope` also accepts optional `parent`, `attributes`, `data`, and
`metadata` arguments after the callback, mirroring `pushScope`.

### Browser CORS Requirements

When deploying WASM modules in a browser, `SharedArrayBuffer` (required by
some multi-threaded WASM configurations) is only available in
[cross-origin-isolated](https://developer.mozilla.org/en-US/docs/Web/API/crossOriginIsolated)
contexts. Your server must send the following HTTP headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without these headers, browsers will block `SharedArrayBuffer` usage and
the WASM module may fail to initialize. Note that Nexus WASM is
single-threaded by default, so `SharedArrayBuffer` is only required if you
opt into threaded builds (e.g., with `wasm-bindgen-rayon`).

### Differences from Node.js

- Functions use camelCase JS names (via `#[wasm_bindgen(js_name = "...")]`)
- Single-threaded (no worker thread isolation)
- Uses `wasm_bindgen_futures::spawn_local()` for async execution
- Stream objects expose an async `next()` method returning `{ value, done }`
- Scope type constants are exported as integer values (`SCOPE_TYPE_AGENT = 0`,
  `SCOPE_TYPE_FUNCTION = 1`, etc.) rather than an enum object

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
