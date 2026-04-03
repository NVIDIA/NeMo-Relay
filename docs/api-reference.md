<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# API Reference

This document covers the core runtime API surface shared across bindings.
Function signatures are shown in Python; other bindings mirror the same
operations with language-appropriate naming.

See also:

- [Typed API Reference](typed-api-reference.md) for `nat_nexus.typed` helper
  functions and codec types.
- [Proxy API Reference](proxy-api-reference.md) for `nat_nexus.proxy`,
  `NexusProxy`, backend types, and declarative proxy lifecycle helpers.

## Scope Operations

```python
# Get the current top-of-stack scope handle
handle = nat_nexus.scope.get_handle() -> ScopeHandle

# Push a new scope onto the stack
handle = nat_nexus.scope.push(
    name: str,
    scope_type: ScopeType,
    *,
    handle: ScopeHandle | None = None,   # parent override
    attributes: int | None = None,        # ScopeAttributes bitflags
) -> ScopeHandle

# Pop a scope from the stack
nat_nexus.scope.pop(handle: ScopeHandle) -> None

# Emit a standalone Mark event
nat_nexus.scope.event(
    name: str,
    *,
    handle: ScopeHandle | None = None,
    data: Any | None = None,
    metadata: Any | None = None,
) -> None
```

## Tool Lifecycle

### Advanced Manual Lifecycle (Low-Level)

These APIs are for advanced integrations that need to drive lifecycle start/end
manually. For normal application code and all quickstarts, prefer
`nat_nexus.tools.execute(...)` so the runtime manages the full lifecycle.

```python
# Begin a tool call — emits Start event
handle = nat_nexus.tools.call(
    name: str,
    args: dict,
    *,
    handle: ScopeHandle | None = None,    # parent override
    attributes: int | None = None,         # ToolAttributes bitflags
    data: Any | None = None,
    metadata: Any | None = None,
    tool_call_id: str | None = None,       # external correlation ID
) -> ToolHandle

# End a tool call — emits End event
nat_nexus.tools.call_end(
    handle: ToolHandle,
    result: dict,
    *,
    data: Any | None = None,
    metadata: Any | None = None,
) -> None
```

### Managed (Full Pipeline)

```python
# Execute a tool call through the full middleware pipeline
result = await nat_nexus.tools.execute(
    name: str,
    args: dict,
    func: Callable[[dict], Awaitable[dict]] | Callable[[dict], dict],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Any | None = None,
    metadata: Any | None = None,
) -> dict
```

In managed execute APIs, sanitize-request and sanitize-response guardrails are
observability-oriented: they rewrite the payload recorded on lifecycle events,
but request intercepts still control what `func(...)` receives and callers
still receive the raw execution result.

### Standalone Middleware

```python
# Run request intercept chain only
transformed_args = nat_nexus.intercepts.tool_request_intercepts(name: str, args: dict) -> dict

# Run conditional execution guardrails only (raises on rejection)
nat_nexus.tools.conditional_execution(name: str, args: dict) -> None
```

## LLM Lifecycle

### Advanced Manual Lifecycle (Low-Level)

These APIs are for advanced integrations that need to drive lifecycle start/end
manually. For normal application code and all quickstarts, prefer
`nat_nexus.llm.execute(...)` or `nat_nexus.llm.stream_execute(...)`.

```python
# Begin an LLM call — emits Start event
handle = nat_nexus.llm.call(
    name: str,
    request: LLMRequest,
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,         # LLMAttributes bitflags
    data: Any | None = None,
    metadata: Any | None = None,
    model_name: str | None = None,         # for ATIF export
) -> LLMHandle

# End an LLM call — emits End event
nat_nexus.llm.call_end(
    handle: LLMHandle,
    response: dict,
    *,
    data: Any | None = None,
    metadata: Any | None = None,
) -> None
```

### Managed (Full Pipeline)

```python
# Execute an LLM call through the full middleware pipeline
result = await nat_nexus.llm.execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], Awaitable[dict]] | Callable[[LLMRequest], dict],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Any | None = None,
    metadata: Any | None = None,
    model_name: str | None = None,
) -> dict

# Execute a streaming LLM call
stream = await nat_nexus.llm.stream_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], AsyncIterator[dict]],
    collector: Callable[[dict], None],
    finalizer: Callable[[], dict],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Any | None = None,
    metadata: Any | None = None,
    model_name: str | None = None,
) -> LlmStream
```

For managed LLM execute APIs, sanitize guardrails affect lifecycle event
payloads (`Start.input`, `End.output`) rather than the request passed into
`func(...)` or the value returned to the caller. Use request intercepts to
rewrite execution inputs.

### Standalone Middleware

```python
# Run LLM request intercept chain only
transformed = nat_nexus.llm.request_intercepts(name: str, request: LLMRequest) -> LLMRequest

# Run conditional execution guardrails only (raises on rejection)
nat_nexus.llm.conditional_execution(request: LLMRequest) -> None
```

## Guardrail Registration

All guardrails are priority-ordered (ascending — lower numbers run first). Names must be unique.

Sanitize guardrails currently affect recorded lifecycle payloads rather than
managed execution I/O. In `tools.execute(...)`, `llm.execute(...)`, and
`llm.stream_execute(...)`, they rewrite `Start.input` / `End.output` on emitted
events. They do not rewrite the arguments passed to `func(...)` or the value
returned to the caller.

### Tool Guardrails

```python
# Sanitize tool request arguments
nat_nexus.guardrails.register_tool_sanitize_request(
    name: str, priority: int,
    fn: Callable[[str, dict], dict],               # (tool_name, args) -> args
) -> None

# Sanitize tool response
nat_nexus.guardrails.register_tool_sanitize_response(
    name: str, priority: int,
    fn: Callable[[str, dict], dict],               # (tool_name, result) -> result
) -> None

# Conditionally block tool execution
nat_nexus.guardrails.register_tool_conditional_execution(
    name: str, priority: int,
    fn: Callable[[str, dict], str | None],         # (tool_name, args) -> None or reason
) -> None

# Deregister (returns True if found)
nat_nexus.guardrails.deregister_tool_sanitize_request(name: str) -> bool
nat_nexus.guardrails.deregister_tool_sanitize_response(name: str) -> bool
nat_nexus.guardrails.deregister_tool_conditional_execution(name: str) -> bool
```

### LLM Guardrails

```python
# Sanitize LLM request
nat_nexus.guardrails.register_llm_sanitize_request(
    name: str, priority: int,
    fn: Callable[[LLMRequest], LLMRequest],
) -> None

# Sanitize LLM response
nat_nexus.guardrails.register_llm_sanitize_response(
    name: str, priority: int,
    fn: Callable[[dict], dict],
) -> None

# Conditionally block LLM execution
nat_nexus.guardrails.register_llm_conditional_execution(
    name: str, priority: int,
    fn: Callable[[LLMRequest], str | None],
) -> None

# Deregister (returns True if found)
nat_nexus.guardrails.deregister_llm_sanitize_request(name: str) -> bool
nat_nexus.guardrails.deregister_llm_sanitize_response(name: str) -> bool
nat_nexus.guardrails.deregister_llm_conditional_execution(name: str) -> bool
```

## Intercept Registration

Intercepts are priority-ordered (ascending). Names must be unique.

### Tool Intercepts

```python
# Transform tool request arguments
nat_nexus.intercepts.register_tool_request(
    name: str, priority: int, break_chain: bool,
    fn: Callable[[str, dict], dict],               # (tool_name, args) -> args
) -> None

# Execution middleware chain
nat_nexus.intercepts.register_tool_execution(
    name: str, priority: int,
    fn: Callable[[str, dict, Callable], Awaitable[dict]],  # (tool_name, args, next) -> result
) -> None

# Deregister
nat_nexus.intercepts.deregister_tool_request(name: str) -> bool
nat_nexus.intercepts.deregister_tool_execution(name: str) -> bool
```

### LLM Intercepts

```python
# Transform LLM request
nat_nexus.intercepts.register_llm_request(
    name: str, priority: int, break_chain: bool,
    fn: Callable[[str, LLMRequest], LLMRequest],         # (llm_name, request) -> request
) -> None

# Execution middleware chain
nat_nexus.intercepts.register_llm_execution(
    name: str, priority: int,
    fn: Callable[[str, LLMRequest, Callable], Awaitable[dict]],  # (llm_name, request, next) -> result
) -> None

# Stream execution middleware chain
nat_nexus.intercepts.register_llm_stream_execution(
    name: str, priority: int,
    fn: Callable[[str, LLMRequest, Callable], Awaitable[AsyncIterator[dict]]],  # (llm_name, request, next) -> stream
) -> None

# Deregister
nat_nexus.intercepts.deregister_llm_request(name: str) -> bool
nat_nexus.intercepts.deregister_llm_execution(name: str) -> bool
nat_nexus.intercepts.deregister_llm_stream_execution(name: str) -> bool
```

## Event Subscribers

```python
# Register an event subscriber
nat_nexus.subscribers.register(
    name: str,
    fn: Callable[[Event], None],
) -> None

# Deregister (returns True if found)
nat_nexus.subscribers.deregister(name: str) -> bool
```

## Context Isolation

```python
# Get or create the current task/thread scope stack
stack = nat_nexus.get_scope_stack() -> ScopeStack

# Create a new isolated scope stack
stack = nat_nexus.create_scope_stack() -> ScopeStack

# Bind a scope stack to the current thread (for worker threads)
nat_nexus.set_thread_scope_stack(stack: ScopeStack) -> None

# Check if a scope stack has been explicitly initialized
nat_nexus.scope_stack_active() -> bool

# Capture current scope stack for propagation to a worker thread
# Raises RuntimeError if no scope stack is active
stack = nat_nexus.propagate_scope_to_thread() -> ScopeStack
```

## Types

### LLMRequest

```python
request = LLMRequest(
    headers: dict[str, Any],    # metadata key-value pairs
    content: Any,                # request payload (JSON-serializable)
)
request.headers  # -> dict[str, Any]
request.content  # -> Any
```

### Handle Properties

All handles expose:

| Property | Type | Description |
|----------|------|-------------|
| `uuid` | `str` | Unique identifier |
| `name` | `str` | Registered name |
| `parent_uuid` | `str \| None` | Parent scope UUID |
| `data` | `Any \| None` | Application data |
| `metadata` | `Any \| None` | Tracing metadata |
| `attributes` | `int` | Attribute bitflags |

Additionally:
- `ToolHandle.tool_call_id` — optional external correlation ID
- `LLMHandle.model_name` — optional model identifier

### Event Properties

| Property | Type | Description |
|----------|------|-------------|
| `uuid` | `str` | Event identifier |
| `parent_uuid` | `str \| None` | Parent scope UUID |
| `timestamp` | `str` | ISO 8601 UTC |
| `name` | `str \| None` | Entity name |
| `event_type` | `EventType` | Start, End, or Mark |
| `scope_type` | `ScopeType \| None` | Entity scope type |
| `data` | `Any \| None` | Data snapshot |
| `metadata` | `Any \| None` | Metadata snapshot |
| `input` | `Any \| None` | Post-guardrail request (Start events) |
| `output` | `Any \| None` | Post-guardrail response (End events) |
| `model_name` | `str \| None` | LLM model name |
| `tool_call_id` | `str \| None` | Tool correlation ID |
| `root_uuid` | `str \| None` | Root scope UUID |

## Scope-Local Registration

Scope-local middleware is bound to a specific scope and automatically cleaned up when that scope is popped. The API mirrors the global registration functions but takes an additional `handle` parameter.

Scope-local sanitize guardrails have the same managed-execution behavior as the
global ones: in `tools.execute(...)`, `llm.execute(...)`, and
`llm.stream_execute(...)` they rewrite emitted lifecycle payloads, not the
arguments passed to `func(...)` or the value returned to the caller.

### Scope-Local Guardrails

```python
# Tool guardrails (scope-local)
nat_nexus.scope_local.register_tool_sanitize_request(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, dict], dict],
) -> None

nat_nexus.scope_local.register_tool_sanitize_response(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, dict], dict],
) -> None

nat_nexus.scope_local.register_tool_conditional_execution(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, dict], str | None],
) -> None

# LLM guardrails (scope-local)
nat_nexus.scope_local.register_llm_sanitize_request(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[LLMRequest], LLMRequest],
) -> None

nat_nexus.scope_local.register_llm_sanitize_response(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[dict], dict],
) -> None

nat_nexus.scope_local.register_llm_conditional_execution(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[LLMRequest], str | None],
) -> None
```

### Scope-Local Intercepts

```python
# Tool intercepts (scope-local)
nat_nexus.scope_local.register_tool_request_intercept(
    handle: ScopeHandle, name: str, priority: int, break_chain: bool,
    fn: Callable[[str, dict], dict],
) -> None

nat_nexus.scope_local.register_tool_execution_intercept(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, dict, Callable], Awaitable[dict]],  # (tool_name, args, next) -> result
) -> None

# LLM intercepts (scope-local)
nat_nexus.scope_local.register_llm_request_intercept(
    handle: ScopeHandle, name: str, priority: int, break_chain: bool,
    fn: Callable[[str, LLMRequest], LLMRequest],           # (llm_name, request) -> request
) -> None

nat_nexus.scope_local.register_llm_execution_intercept(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, LLMRequest, Callable], Awaitable[dict]],  # (llm_name, request, next) -> result
) -> None

nat_nexus.scope_local.register_llm_stream_execution_intercept(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, LLMRequest, Callable], Awaitable[AsyncIterator[dict]]],  # (llm_name, request, next) -> stream
) -> None
```

### Scope-Local Subscribers

```python
nat_nexus.scope_local.register_subscriber(
    handle: ScopeHandle, name: str,
    fn: Callable[[Event], None],
) -> None
```

### Cross-Language Names

| Python | Go | Node.js | WASM | FFI/C |
|--------|----|---------|------|-------|
| `scope_local.register_tool_conditional_execution` | `ScopeRegisterToolConditionalExecution` | `scopeRegisterToolConditionalExecution` | `scope_register_tool_conditional_execution` | `nat_nexus_scope_register_tool_conditional_execution` |
| `scope_local.register_subscriber` | `ScopeRegisterSubscriber` | `scopeRegisterSubscriber` | `scope_register_subscriber` | `nat_nexus_scope_register_subscriber` |

All scope-local registration functions follow the same naming pattern: prefix the global registration name with `scope_register_` (FFI/WASM), `scopeRegister` (Node.js), `ScopeRegister` (Go), or place it in the `scope_local` module (Python).

## ATIF Export

```python
from nat_nexus import AtifExporter

exporter = AtifExporter()
# ... run operations ...
trajectory = exporter.export(root_uuid=None)  # dict (ATIF trajectory)
trajectory_json = exporter.export_json(root_uuid=None)  # JSON string
exporter.clear()
```
