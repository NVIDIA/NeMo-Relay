<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# API Reference

This document covers the core runtime API surface shared across bindings.
Function signatures are shown in Python; other bindings mirror the same
operations with language-appropriate naming.

See also:

- [Typed API Reference](typed-api-reference.md) for `nemo_flow.typed` helper
  functions and codec types.
- [Adaptive API Reference](adaptive-api-reference.md) for the config-driven
  adaptive component and core plugin-host helpers.

## Scope Operations

```python
# Get the current top-of-stack scope handle
handle = nemo_flow.scope.get_handle() -> ScopeHandle

# Push a new scope onto the stack
handle = nemo_flow.scope.push(
    name: str,
    scope_type: ScopeType,
    *,
    handle: ScopeHandle | None = None,   # parent override
    attributes: int | None = None,        # ScopeAttributes bitflags
) -> ScopeHandle

# Pop a scope from the stack
nemo_flow.scope.pop(handle: ScopeHandle) -> None

# Emit a standalone Mark event
nemo_flow.scope.event(
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
`nemo_flow.tools.execute(...)` so the runtime manages the full lifecycle.

```python
# Begin a tool call — emits Start event
handle = nemo_flow.tools.call(
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
nemo_flow.tools.call_end(
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
result = await nemo_flow.tools.execute(
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
transformed_args = nemo_flow.intercepts.tool_request_intercepts(name: str, args: dict) -> dict

# Run conditional execution guardrails only (raises on rejection)
nemo_flow.tools.conditional_execution(name: str, args: dict) -> None
```

## LLM Lifecycle

### Advanced Manual Lifecycle (Low-Level)

These APIs are for advanced integrations that need to drive lifecycle start/end
manually. For normal application code and all quickstarts, prefer
`nemo_flow.llm.execute(...)` or `nemo_flow.llm.stream_execute(...)`.

```python
# Begin an LLM call — emits Start event
handle = nemo_flow.llm.call(
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
nemo_flow.llm.call_end(
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
result = await nemo_flow.llm.execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], Awaitable[dict]] | Callable[[LLMRequest], dict],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Any | None = None,
    metadata: Any | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,              # request codec instance
    response_codec: LlmResponseCodec | None = None,  # response codec instance
) -> dict

# Execute a streaming LLM call
stream = await nemo_flow.llm.stream_execute(
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
    codec: LlmCodec | None = None,              # request codec instance
    response_codec: LlmResponseCodec | None = None,  # response codec instance
) -> LlmStream
```

The `codec` parameter enables structured `AnnotatedLLMRequest` for
request intercepts and populates `LLMStartEvent.annotated_request`. The
`response_codec` parameter enables structured `AnnotatedLLMResponse` on
`LLMEndEvent.annotated_response`. Both accept codec object instances (not
strings). See [LLM Codecs](llm-codecs.md) for details.

For managed LLM execute APIs, sanitize guardrails affect lifecycle event
payloads (`Start.input`, `End.output`) rather than the request passed into
`func(...)` or the value returned to the caller. Use request intercepts to
rewrite execution inputs.

### Standalone Middleware

```python
# Run LLM request intercept chain only
transformed = nemo_flow.llm.request_intercepts(name: str, request: LLMRequest) -> LLMRequest

# Run conditional execution guardrails only (raises on rejection)
nemo_flow.llm.conditional_execution(request: LLMRequest) -> None
```

## Guardrail Registration

All guardrails are priority-ordered (ascending — lower numbers run first). Names must be unique.

Sanitize guardrails currently affect recorded lifecycle payloads rather than
managed execution I/O. In `tools.execute(...)`, `llm.execute(...)`, and
`llm.stream_execute(...)`, they rewrite `Start.input` / `End.output` on emitted
events. They do not rewrite the arguments passed to `func(...)` or the value
returned to the caller.

### Callback Contracts

The callback contract is intentionally split between fallible and infallible
surfaces:

| Surface | Contract | Normal Return | Callback Failure |
|---------|----------|---------------|------------------|
| Tool/LLM sanitize guardrails | Infallible | Return sanitized value | Handle failures internally; there is no error channel |
| Tool/LLM conditional execution guardrails | Fallible | Return `None` to allow or a rejection reason to block | Raising/throwing fails the originating NeMo Flow API call |
| Tool/LLM request intercepts | Fallible | Return transformed request | Raising/throwing fails the originating NeMo Flow API call |
| Tool/LLM execution intercepts | Fallible | Return/await transformed result | Raising/throwing fails the originating NeMo Flow API call |
| Stream collector | Fallible | Return normally for each chunk | Raising/throwing aborts the stream |
| Stream finalizer | Infallible | Return the aggregated response | Handle failures internally; there is no error channel |
| Event subscribers | Infallible | Return `None` | Handle failures internally; there is no error channel |

For direct Rust users, the fallible rows above are expressed as `Result<...>`.
In Python, Node.js, and WASM, they keep natural callback return types, but
exceptions thrown from those callbacks propagate to the originating NeMo Flow API
call. There are not separate "fallible variants" of these callback surfaces:
conditional guardrails and request/execution intercepts are the canonical
fallible contract in every binding.

### Tool Guardrails

```python
# Sanitize tool request arguments
nemo_flow.guardrails.register_tool_sanitize_request(
    name: str, priority: int,
    fn: Callable[[str, dict], dict],               # (tool_name, args) -> args
) -> None

# Sanitize tool response
nemo_flow.guardrails.register_tool_sanitize_response(
    name: str, priority: int,
    fn: Callable[[str, dict], dict],               # (tool_name, result) -> result
) -> None

# Conditionally block tool execution
nemo_flow.guardrails.register_tool_conditional_execution(
    name: str, priority: int,
    fn: Callable[[str, dict], str | None],         # (tool_name, args) -> None or reason
) -> None

# Deregister (returns True if found)
nemo_flow.guardrails.deregister_tool_sanitize_request(name: str) -> bool
nemo_flow.guardrails.deregister_tool_sanitize_response(name: str) -> bool
nemo_flow.guardrails.deregister_tool_conditional_execution(name: str) -> bool
```

### LLM Guardrails

```python
# Sanitize LLM request
nemo_flow.guardrails.register_llm_sanitize_request(
    name: str, priority: int,
    fn: Callable[[LLMRequest], LLMRequest],
) -> None

# Sanitize LLM response
nemo_flow.guardrails.register_llm_sanitize_response(
    name: str, priority: int,
    fn: Callable[[dict], dict],
) -> None

# Conditionally block LLM execution
nemo_flow.guardrails.register_llm_conditional_execution(
    name: str, priority: int,
    fn: Callable[[LLMRequest], str | None],
) -> None

# Deregister (returns True if found)
nemo_flow.guardrails.deregister_llm_sanitize_request(name: str) -> bool
nemo_flow.guardrails.deregister_llm_sanitize_response(name: str) -> bool
nemo_flow.guardrails.deregister_llm_conditional_execution(name: str) -> bool
```

## Intercept Registration

Intercepts are priority-ordered (ascending). Names must be unique.

### Tool Intercepts

```python
# Transform tool request arguments
nemo_flow.intercepts.register_tool_request(
    name: str, priority: int, break_chain: bool,
    fn: Callable[[str, dict], dict],               # (tool_name, args) -> args
) -> None

# Execution middleware chain
nemo_flow.intercepts.register_tool_execution(
    name: str, priority: int,
    fn: Callable[[str, dict, Callable], Awaitable[dict]],  # (tool_name, args, next) -> result
) -> None

# Deregister
nemo_flow.intercepts.deregister_tool_request(name: str) -> bool
nemo_flow.intercepts.deregister_tool_execution(name: str) -> bool
```

### LLM Intercepts

```python
# Transform LLM request
nemo_flow.intercepts.register_llm_request(
    name: str, priority: int, break_chain: bool,
    fn: Callable[[str, LLMRequest], LLMRequest],         # (llm_name, request) -> request
) -> None

# Execution middleware chain
nemo_flow.intercepts.register_llm_execution(
    name: str, priority: int,
    fn: Callable[[str, LLMRequest, Callable], Awaitable[dict]],  # (llm_name, request, next) -> result
) -> None

# Stream execution middleware chain
nemo_flow.intercepts.register_llm_stream_execution(
    name: str, priority: int,
    fn: Callable[[str, LLMRequest, Callable], Awaitable[AsyncIterator[dict]]],  # (llm_name, request, next) -> stream
) -> None

# Deregister
nemo_flow.intercepts.deregister_llm_request(name: str) -> bool
nemo_flow.intercepts.deregister_llm_execution(name: str) -> bool
nemo_flow.intercepts.deregister_llm_stream_execution(name: str) -> bool
```

## Event Subscribers

```python
# Register an event subscriber
nemo_flow.subscribers.register(
    name: str,
    fn: Callable[[Event], None],
) -> None

# Deregister (returns True if found)
nemo_flow.subscribers.deregister(name: str) -> bool
```

Subscriber callbacks run synchronously on the calling thread, after NeMo Flow has
snapshotted the subscriber list and released its runtime locks. Subscribers may
call other NeMo Flow APIs, but should remain lightweight because they are still on
the request path. Subscribers are infallible callbacks: they do not have an
error return channel.

## Plugin Host

The plugin host is the shared configuration and registration surface used by the
adaptive component and any custom plugin kinds.

For practical guidance, start with [Plugins](hosted-plugins.md). This section is
the low-level API reference for the same host.

```python
policy = plugin.ConfigPolicy(
    unknown_component: Literal["ignore", "warn", "error"] = "warn",
    unknown_field: Literal["ignore", "warn", "error"] = "warn",
    unsupported_value: Literal["ignore", "warn", "error"] = "error",
)

component = plugin.ComponentSpec(
    kind: str,
    config: dict[str, Any] = {},
    enabled: bool = True,
)

config = plugin.PluginConfig(
    version: int = 1,
    components: list[object] = [],
    policy: ConfigPolicy = ConfigPolicy(),
)

report = plugin.validate(config: PluginConfig | dict) -> ConfigReport
report = await plugin.initialize(config: PluginConfig | dict) -> ConfigReport
plugin.clear() -> None
active = plugin.report() -> ConfigReport | None
kinds = plugin.list_kinds() -> list[str]
plugin.register(plugin_kind: str, plugin: Plugin) -> None
removed = plugin.deregister(plugin_kind: str) -> bool
```

### Config Objects

- `ConfigPolicy`
  Description: controls how the host reports unsupported component kinds, unknown fields, and unsupported values.
  Arguments: `unknown_component`, `unknown_field`, and `unsupported_value` each accept `"ignore"`, `"warn"`, or `"error"`.
  Returns: a policy object embedded in `PluginConfig`.
  Behavior: `"warn"` adds a warning diagnostic, `"error"` adds an error diagnostic that blocks `initialize(...)`, and `"ignore"` suppresses diagnostics.

- `ComponentSpec`
  Description: describes one top-level custom plugin component.
  Arguments: `kind` is the registered plugin kind string, `config` is the component-specific JSON object, and `enabled` controls whether the host should activate the component.
  Returns: a component document suitable for inclusion in `PluginConfig.components`.
  Behavior: disabled components are still validated but skipped during runtime registration. Adaptive uses its own `adaptive.ComponentSpec(...)`, but both adaptive and generic components share the same top-level `components` array.

- `PluginConfig`
  Description: the canonical plugin host configuration document.
  Arguments: `version` is the host config schema version, `components` is the ordered list of top-level components, and `policy` controls unsupported-config behavior.
  Returns: a serializable config document.
  Behavior: component order is preserved during activation, so earlier components can register middleware before later ones.

### Host Operations

- `plugin.validate(config)`
  Description: validates a plugin host config without changing runtime state.
  Arguments: a `PluginConfig` instance or an equivalent JSON object.
  Returns: `ConfigReport` with zero or more diagnostics.
  Behavior: this is pure validation. It checks host-level compatibility, unknown plugin kinds, multiplicity rules, and each registered plugin's per-component validation logic.

- `await plugin.initialize(config)`
  Description: validates and activates the full plugin configuration.
  Arguments: a `PluginConfig` instance or an equivalent JSON object.
  Returns: the successful `ConfigReport` for the activated configuration.
  Behavior: initialization replaces the current active plugin configuration. If registration fails partway through, the host rolls back partial registrations. If there was a previous active configuration, the host attempts to restore it.

- `plugin.clear()`
  Description: deregisters all middleware and subscribers installed by the active plugin configuration.
  Arguments: none.
  Returns: `None`.
  Behavior: this clears active component registrations only. It does not remove plugin kinds from the plugin registry.

- `plugin.report()`
  Description: returns the last successfully activated plugin report.
  Arguments: none.
  Returns: `ConfigReport | None`.
  Behavior: returns `None` when no plugin configuration is active.

- `plugin.list_kinds()`
  Description: lists custom plugin kinds currently registered with the plugin registry.
  Arguments: none.
  Returns: a sorted list of kind strings.
  Behavior: this reports available plugin kinds, not the currently active component set.

- `plugin.register(plugin_kind, plugin)`
  Description: registers a custom plugin implementation.
  Arguments: `plugin_kind` is the unique top-level component kind and `plugin` implements the `Plugin` contract.
  Returns: `None`.
  Behavior: registration makes the kind available to later `validate(...)` and `initialize(...)` calls. Registering the same kind twice raises an error.

- `plugin.deregister(plugin_kind)`
  Description: removes a previously registered custom plugin kind.
  Arguments: `plugin_kind` is the kind string to remove.
  Returns: `True` if a plugin was removed, otherwise `False`.
  Behavior: deregistration affects future validation and initialization. It does not retroactively clear middleware already installed by an active configuration.

### Plugin Contract

```python
class Plugin(Protocol):
    def validate(self, plugin_config: JsonObject) -> list[ConfigDiagnostic] | None: ...
    def register(self, plugin_config: JsonObject, context: PluginContext) -> None: ...
```

- `validate(plugin_config)`
  Description: validates one component's `config` object.
  Arguments: the component-local JSON config for a single `ComponentSpec`.
  Returns: a list of diagnostics or `None`.
  Behavior: validation runs during both `plugin.validate(...)` and `plugin.initialize(...)`. Returning an error-level diagnostic blocks initialization.

- `register(plugin_config, context)`
  Description: installs middleware and subscribers for one component instance.
  Arguments: the component-local JSON config and a `PluginContext`.
  Returns: `None`.
  Behavior: this runs only for enabled components during `initialize(...)`. Any exception or registration failure aborts the current initialization and triggers rollback.

### Plugin Context

`PluginContext` exposes these registration methods:

- `register_subscriber(name, callback)`
- `register_tool_sanitize_request_guardrail(name, priority, callback)`
- `register_tool_sanitize_response_guardrail(name, priority, callback)`
- `register_tool_conditional_execution_guardrail(name, priority, callback)`
- `register_llm_sanitize_request_guardrail(name, priority, callback)`
- `register_llm_sanitize_response_guardrail(name, priority, callback)`
- `register_llm_conditional_execution_guardrail(name, priority, callback)`
- `register_llm_request_intercept(name, priority, break_chain, callback)`
- `register_llm_execution_intercept(name, priority, callback)`
- `register_llm_stream_execution_intercept(name, priority, callback)`
- `register_tool_request_intercept(name, priority, break_chain, callback)`
- `register_tool_execution_intercept(name, priority, callback)`

Shared behavior:

- `name`
  Description: the plugin-local registration name.
  Behavior: names are scoped per component. The runtime namespaces them internally, so users do not provide instance ids or global registration names.

- `priority`
  Description: middleware execution order.
  Behavior: lower values run first.

- `break_chain`
  Description: request-intercept short-circuit flag.
  Behavior: when `True`, later request intercepts in that chain are skipped after the callback runs.

- `callback`
  Description: the runtime callback implementation for the subscriber or intercept.
  Behavior: the callback contracts match the normal subscriber/intercept APIs for the host language. Registration succeeds immediately, and the plugin host records a rollback action so the registration can be undone on failure or `plugin.clear()`.

## Context Isolation

```python
# Get or create the current task/thread scope stack
stack = nemo_flow.get_scope_stack() -> ScopeStack

# Create a new isolated scope stack
stack = nemo_flow.create_scope_stack() -> ScopeStack

# Bind a scope stack to the current thread (for worker threads)
nemo_flow.set_thread_scope_stack(stack: ScopeStack) -> None

# Check if a scope stack has been explicitly initialized
nemo_flow.scope_stack_active() -> bool

# Capture current scope stack for propagation to a worker thread
# Raises RuntimeError if no scope stack is active
stack = nemo_flow.propagate_scope_to_thread() -> ScopeStack
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
| `kind` | `str` | `ScopeStart`, `ScopeEnd`, `ToolStart`, `ToolEnd`, `LLMStart`, `LLMEnd`, or `Mark` |
| `scope_type` | `ScopeType \| None` | Scope category, present only on `ScopeStart` and `ScopeEnd` |
| `data` | `Any \| None` | Data snapshot |
| `metadata` | `Any \| None` | Metadata snapshot |
| `input` | `Any \| None` | Post-guardrail request (Start events) |
| `output` | `Any \| None` | Post-guardrail response (End events) |
| `model_name` | `str \| None` | LLM model name |
| `tool_call_id` | `str \| None` | Tool correlation ID |
| `annotated_request` | `AnnotatedLLMRequest \| None` | Structured request from codec decode (LLMStart events only) |
| `annotated_response` | `AnnotatedLLMResponse \| None` | Structured response from response codec decode (LLMEnd events only) |

Scope `Start` events are emitted after the scope has been pushed, and scope
`End` events are emitted after the scope has been popped. Subscribers that call
`get_handle()` therefore observe the post-mutation active scope.

## Scope-Local Registration

Scope-local middleware is bound to a specific scope and automatically cleaned up when that scope is popped. The API mirrors the global registration functions but takes an additional `handle` parameter.

Scope-local sanitize guardrails have the same managed-execution behavior as the
global ones: in `tools.execute(...)`, `llm.execute(...)`, and
`llm.stream_execute(...)` they rewrite emitted lifecycle payloads, not the
arguments passed to `func(...)` or the value returned to the caller.

### Scope-Local Guardrails

```python
# Tool guardrails (scope-local)
nemo_flow.scope_local.register_tool_sanitize_request(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, dict], dict],
) -> None

nemo_flow.scope_local.register_tool_sanitize_response(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, dict], dict],
) -> None

nemo_flow.scope_local.register_tool_conditional_execution(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, dict], str | None],
) -> None

# LLM guardrails (scope-local)
nemo_flow.scope_local.register_llm_sanitize_request(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[LLMRequest], LLMRequest],
) -> None

nemo_flow.scope_local.register_llm_sanitize_response(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[dict], dict],
) -> None

nemo_flow.scope_local.register_llm_conditional_execution(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[LLMRequest], str | None],
) -> None
```

### Scope-Local Intercepts

```python
# Tool intercepts (scope-local)
nemo_flow.scope_local.register_tool_request_intercept(
    handle: ScopeHandle, name: str, priority: int, break_chain: bool,
    fn: Callable[[str, dict], dict],
) -> None

nemo_flow.scope_local.register_tool_execution_intercept(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, dict, Callable], Awaitable[dict]],  # (tool_name, args, next) -> result
) -> None

# LLM intercepts (scope-local)
nemo_flow.scope_local.register_llm_request_intercept(
    handle: ScopeHandle, name: str, priority: int, break_chain: bool,
    fn: Callable[[str, LLMRequest], LLMRequest],           # (llm_name, request) -> request
) -> None

nemo_flow.scope_local.register_llm_execution_intercept(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, LLMRequest, Callable], Awaitable[dict]],  # (llm_name, request, next) -> result
) -> None

nemo_flow.scope_local.register_llm_stream_execution_intercept(
    handle: ScopeHandle, name: str, priority: int,
    fn: Callable[[str, LLMRequest, Callable], Awaitable[AsyncIterator[dict]]],  # (llm_name, request, next) -> stream
) -> None
```

### Scope-Local Subscribers

```python
nemo_flow.scope_local.register_subscriber(
    handle: ScopeHandle, name: str,
    fn: Callable[[Event], None],
) -> None
```

Scope-local registrations follow the same callback contract split as the global
ones: sanitize guardrails, subscribers, and stream finalizers are infallible;
conditional execution guardrails, request intercepts, execution intercepts, and
stream collectors are fallible.

### Cross-Language Names

| Python | Go | Node.js | WASM | FFI/C |
|--------|----|---------|------|-------|
| `scope_local.register_tool_conditional_execution` | `ScopeRegisterToolConditionalExecutionGuardrail` | `scopeRegisterToolConditionalExecutionGuardrail` | `scopeRegisterToolConditionalExecutionGuardrail` | `nemo_flow_scope_register_tool_conditional_execution` |
| `scope_local.register_subscriber` | `ScopeRegisterSubscriber` | `scopeRegisterSubscriber` | `scopeRegisterSubscriber` | `nemo_flow_scope_register_subscriber` |

All scope-local registration functions follow the same naming pattern: prefix the global registration name with `nemo_flow_scope_` (FFI), `scopeRegister` (Node.js and WASM), `ScopeRegister` (Go), or place it in the `scope_local` module (Python).

## Built-In Codec Types

NeMo Flow ships three built-in codecs that implement both request codec
(`LlmCodec`) and response codec (`LlmResponseCodec`). Each can be used
for both `codec=` and `response_codec=` parameters.

```python
from nemo_flow.codecs import AnthropicMessagesCodec, OpenAIChatCodec, OpenAIResponsesCodec

# Each codec instance exposes decode(), encode(), and decode_response()
codec = OpenAIChatCodec()
annotated_req = codec.decode(request)                      # -> AnnotatedLLMRequest
encoded_req = codec.encode(annotated_req, original_req)    # -> LLMRequest
annotated_resp = codec.decode_response(raw_response)       # -> AnnotatedLLMResponse
```

### AnnotatedLLMResponse Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str \| None` | Response ID from the API |
| `model` | `str \| None` | Model that served the request |
| `message` | `Any \| None` | Assistant's response content |
| `tool_calls` | `list[dict] \| None` | Tool calls with parsed JSON arguments |
| `finish_reason` | `str \| None` | Normalized: `"complete"`, `"length"`, `"tool_use"`, `"content_filter"` |
| `usage` | `dict \| None` | Token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`, `cache_read_tokens`, `cache_write_tokens`) |
| `api_specific` | `dict \| None` | Provider-specific data |
| `extra` | `dict` | Unmodeled top-level fields |

Helper methods: `response_text()` returns the text content,
`has_tool_calls()` returns `True` if tool calls are present.

See [LLM Codecs](llm-codecs.md) for the full codec system documentation.

## ATIF Export

```python
from nemo_flow import AtifExporter

exporter = AtifExporter()
# ... run operations ...
trajectory = exporter.export()  # dict (ATIF trajectory)
trajectory_json = exporter.export_json()  # JSON string
exporter.clear()
```
