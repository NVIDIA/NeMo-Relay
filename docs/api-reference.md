<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# API Reference

This document covers the complete API surface. Function signatures are shown in Python; other bindings mirror the same operations with language-appropriate naming.

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

### Manual (Low-Level)

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

### Standalone Middleware

```python
# Run request intercept chain only
transformed_args = nat_nexus.intercepts.tool_request_intercepts(name: str, args: dict) -> dict

# Run response intercept chain only
transformed_result = nat_nexus.intercepts.tool_response_intercepts(name: str, result: dict) -> dict

# Run conditional execution guardrails only (raises on rejection)
nat_nexus.tools.conditional_execution(name: str, args: dict) -> None
```

## LLM Lifecycle

### Manual (Low-Level)

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

### Standalone Middleware

```python
# Run LLM request intercept chain only
transformed = nat_nexus.llm.request_intercepts(request: LLMRequest) -> LLMRequest

# Run conditional execution guardrails only (raises on rejection)
nat_nexus.llm.conditional_execution(request: LLMRequest) -> None
```

## Guardrail Registration

All guardrails are priority-ordered (ascending — lower numbers run first). Names must be unique.

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

# Transform tool response
nat_nexus.intercepts.register_tool_response(
    name: str, priority: int, break_chain: bool,
    fn: Callable[[str, dict], dict],               # (tool_name, result) -> result
) -> None

# Execution middleware chain
nat_nexus.intercepts.register_tool_execution(
    name: str, priority: int,
    fn: Callable[[dict, Callable], Awaitable[dict]],  # (args, next) -> result
) -> None

# Deregister
nat_nexus.intercepts.deregister_tool_request(name: str) -> bool
nat_nexus.intercepts.deregister_tool_response(name: str) -> bool
nat_nexus.intercepts.deregister_tool_execution(name: str) -> bool
```

### LLM Intercepts

```python
# Transform LLM request
nat_nexus.intercepts.register_llm_request(
    name: str, priority: int, break_chain: bool,
    fn: Callable[[LLMRequest], LLMRequest],
) -> None

# Execution middleware chain
nat_nexus.intercepts.register_llm_execution(
    name: str, priority: int,
    fn: Callable[[LLMRequest, Callable], Awaitable[dict]],  # (request, next) -> result
) -> None

# Stream execution middleware chain
nat_nexus.intercepts.register_llm_stream_execution(
    name: str, priority: int,
    fn: Callable[[LLMRequest, Callable], Awaitable[AsyncIterator[dict]]],
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

## ATIF Export

```python
from nat_nexus import AtifExporter

exporter = AtifExporter()
# ... run operations ...
trajectory = exporter.export(root_uuid=None)  # JSON string
exporter.clear()
```
