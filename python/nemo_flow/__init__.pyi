# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for nemo_flow.

Provides static type information for the ``nemo_flow`` package, including
all types exported from the native Rust extension and all API functions.
"""

import contextvars
from collections.abc import Mapping, Sequence
from typing import AsyncIterator, Awaitable, Callable, Literal, Optional, TypeAlias

from nemo_flow import adaptive as adaptive
from nemo_flow import codecs as codecs
from nemo_flow import plugin as plugin
from nemo_flow import scope as scope
from nemo_flow import typed as typed
from nemo_flow.codecs import LlmCodec as LlmCodec
from nemo_flow.codecs import LlmResponseCodec as LlmResponseCodec

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Json: TypeAlias = JsonValue
"""Type alias for JSON-serializable Python objects (dicts, lists, strings, numbers, etc.)."""
UnsupportedBehavior: TypeAlias = Literal["ignore", "warn", "error"]

ToolSanitizeGuardrail: TypeAlias = Callable[[str, Json], Json]
ToolConditionalExecutionGuardrail: TypeAlias = Callable[[str, Json], Optional[str]]
LlmSanitizeRequestGuardrail: TypeAlias = Callable[["LLMRequest"], "LLMRequest"]
LlmSanitizeResponseGuardrail: TypeAlias = Callable[[JsonObject], JsonObject]
LlmConditionalExecutionGuardrail: TypeAlias = Callable[["LLMRequest"], Optional[str]]
ToolRequestIntercept: TypeAlias = Callable[[str, Json], Json]
ToolExecutionIntercept: TypeAlias = Callable[
    [str, Json, Callable[[Json], Awaitable[Json]]],
    Json | Awaitable[Json],
]
LlmRequestIntercept: TypeAlias = Callable[
    [str, "LLMRequest", "AnnotatedLLMRequest | None"],
    tuple["LLMRequest", "AnnotatedLLMRequest | None"],
]
LlmExecutionIntercept: TypeAlias = Callable[
    [str, "LLMRequest", Callable[["LLMRequest"], Awaitable[Json]]],
    Json | Awaitable[Json],
]
LlmStreamExecutionIntercept: TypeAlias = Callable[
    ["LLMRequest", Callable[["LLMRequest"], Awaitable[AsyncIterator[Json]]]],
    AsyncIterator[Json] | Awaitable[AsyncIterator[Json]],
]

# ---------------------------------------------------------------------------
# Attribute flag classes
# ---------------------------------------------------------------------------

class ScopeAttributes:
    """Bitflags describing scope properties.

    Combine with ``|``: ``ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE``.
    """

    PARALLEL: int
    """Indicates the scope may execute children in parallel."""
    RELOCATABLE: int
    """Indicates the scope may be relocated to a different execution context."""

    def __init__(self, value: int = 0) -> None:
        """Create ``ScopeAttributes`` from a raw integer bitmask."""
        ...
    @property
    def is_parallel(self) -> bool:
        """``True`` if the ``PARALLEL`` flag is set."""
        ...
    @property
    def is_relocatable(self) -> bool:
        """``True`` if the ``RELOCATABLE`` flag is set."""
        ...
    @property
    def value(self) -> int:
        """The raw integer bitmask."""
        ...
    def __or__(self, other: ScopeAttributes) -> ScopeAttributes: ...
    def __and__(self, other: ScopeAttributes) -> ScopeAttributes: ...

class ToolAttributes:
    """Bitflags describing tool call properties."""

    LOCAL: int
    """Indicates the tool runs locally (no network call)."""

    def __init__(self, value: int = 0) -> None:
        """Create ``ToolAttributes`` from a raw integer bitmask."""
        ...
    @property
    def is_local(self) -> bool:
        """``True`` if the ``LOCAL`` flag is set."""
        ...
    @property
    def value(self) -> int:
        """The raw integer bitmask."""
        ...
    def __or__(self, other: ToolAttributes) -> ToolAttributes: ...
    def __and__(self, other: ToolAttributes) -> ToolAttributes: ...

class LLMAttributes:
    """Bitflags describing LLM call properties."""

    STATELESS: int
    """Indicates the LLM call is stateless (no conversation history)."""
    STREAMING: int
    """Indicates the LLM call uses streaming SSE responses."""

    def __init__(self, value: int = 0) -> None:
        """Create ``LLMAttributes`` from a raw integer bitmask."""
        ...
    @property
    def is_stateless(self) -> bool:
        """``True`` if the ``STATELESS`` flag is set."""
        ...
    @property
    def is_streaming(self) -> bool:
        """``True`` if the ``STREAMING`` flag is set."""
        ...
    @property
    def value(self) -> int:
        """The raw integer bitmask."""
        ...
    def __or__(self, other: LLMAttributes) -> LLMAttributes: ...
    def __and__(self, other: LLMAttributes) -> LLMAttributes: ...

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScopeType:
    """Enum identifying the kind of execution scope."""

    Agent: ScopeType
    """An autonomous agent scope."""
    Function: ScopeType
    """A generic function call scope."""
    Tool: ScopeType
    """A tool invocation scope."""
    Llm: ScopeType
    """An LLM call scope."""
    Retriever: ScopeType
    """A retriever (RAG) scope."""
    Embedder: ScopeType
    """An embedding model scope."""
    Reranker: ScopeType
    """A reranker model scope."""
    Guardrail: ScopeType
    """A guardrail evaluation scope."""
    Evaluator: ScopeType
    """An evaluator/judge scope."""
    Custom: ScopeType
    """A user-defined scope type."""
    Unknown: ScopeType
    """An unknown or unspecified scope type."""

# ---------------------------------------------------------------------------
# Handle types
# ---------------------------------------------------------------------------

class ScopeHandle:
    """An active execution scope in the scope stack.

    Returned by ``scope.push()`` and ``scope.get_handle()``.
    """

    @property
    def uuid(self) -> str:
        """Unique identifier for this scope."""
        ...
    @property
    def name(self) -> str:
        """Human-readable scope name."""
        ...
    @property
    def scope_type(self) -> ScopeType:
        """The kind of scope."""
        ...
    @property
    def attributes(self) -> ScopeAttributes:
        """Bitflags describing scope properties."""
        ...
    @property
    def parent_uuid(self) -> Optional[str]:
        """UUID of the parent scope, or ``None`` for the root."""
        ...
    @property
    def data(self) -> Optional[Json]:
        """Application-specific data snapshot."""
        ...
    @property
    def metadata(self) -> Optional[Json]:
        """Metadata snapshot."""
        ...

class ToolHandle:
    """An active tool call.

    Returned by ``tools.call()``. Must be passed to ``tools.call_end()``
    when the tool completes.
    """

    @property
    def uuid(self) -> str:
        """Unique identifier for this tool call."""
        ...
    @property
    def name(self) -> str:
        """Tool name."""
        ...
    @property
    def attributes(self) -> ToolAttributes:
        """Bitflags describing tool call properties."""
        ...
    @property
    def parent_uuid(self) -> Optional[str]:
        """UUID of the parent scope."""
        ...
    @property
    def data(self) -> Optional[Json]:
        """Application-specific data snapshot."""
        ...
    @property
    def metadata(self) -> Optional[Json]:
        """Metadata snapshot."""
        ...

class LLMHandle:
    """An active LLM call.

    Returned by ``llm.call()``. Must be passed to ``llm.call_end()``
    when the LLM call completes.
    """

    @property
    def uuid(self) -> str:
        """Unique identifier for this LLM call."""
        ...
    @property
    def name(self) -> str:
        """Model/provider name."""
        ...
    @property
    def attributes(self) -> LLMAttributes:
        """Bitflags describing LLM call properties."""
        ...
    @property
    def parent_uuid(self) -> Optional[str]:
        """UUID of the parent scope."""
        ...
    @property
    def data(self) -> Optional[Json]:
        """Application-specific data snapshot."""
        ...
    @property
    def metadata(self) -> Optional[Json]:
        """Metadata snapshot."""
        ...

# ---------------------------------------------------------------------------
# LLMRequest / Event / LlmStream
# ---------------------------------------------------------------------------

class LLMRequest:
    """An LLM request carrying headers and a content payload.

    Flows through the entire middleware pipeline: guardrails, intercepts,
    and execution functions all receive and return ``LLMRequest``.

    Example:
        ```python
        request = LLMRequest(
            {"Authorization": "Bearer tok"},
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        print(request.headers)   # {"Authorization": "Bearer tok"}
        print(request.content)   # {"model": "gpt-4", ...}
        ```
    """

    def __init__(
        self,
        headers: Mapping[str, JsonValue],
        content: JsonObject,
    ) -> None:
        """Create an LLM request.

        Args:
            headers: Metadata key-value pairs.
            content: The request payload.
        """
        ...
    @property
    def headers(self) -> JsonObject:
        """Metadata key-value pairs."""
        ...
    @property
    def content(self) -> JsonObject:
        """The request payload."""
        ...

class AnnotatedLLMRequest:
    """Structured view of an LLM request produced by a Codec.

    Fields are accessed via properties. ``messages``, ``params``, ``tools``,
    ``tool_choice``, and ``extra`` return Python dicts/lists (copies).
    Modifications must go through the corresponding setter.
    """

    def __init__(
        self,
        messages: Sequence[Mapping[str, JsonValue]],
        *,
        model: Optional[str] = None,
        params: Optional[Mapping[str, JsonValue]] = None,
        tools: Optional[Sequence[Mapping[str, JsonValue]]] = None,
        tool_choice: Optional[str | Mapping[str, JsonValue]] = None,
        extra: Optional[Mapping[str, JsonValue]] = None,
    ) -> None: ...
    @property
    def messages(self) -> list[JsonObject]: ...
    @messages.setter
    def messages(self, value: Sequence[Mapping[str, JsonValue]]) -> None: ...
    @property
    def model(self) -> Optional[str]: ...
    @model.setter
    def model(self, value: Optional[str]) -> None: ...
    @property
    def params(self) -> Optional[JsonObject]: ...
    @params.setter
    def params(self, value: Optional[Mapping[str, JsonValue]]) -> None: ...
    @property
    def tools(self) -> Optional[list[JsonObject]]: ...
    @tools.setter
    def tools(self, value: Optional[Sequence[Mapping[str, JsonValue]]]) -> None: ...
    @property
    def tool_choice(self) -> Optional[str | JsonObject]: ...
    @tool_choice.setter
    def tool_choice(self, value: Optional[str | Mapping[str, JsonValue]]) -> None: ...
    @property
    def extra(self) -> JsonObject: ...
    @extra.setter
    def extra(self, value: Mapping[str, JsonValue]) -> None: ...
    def system_prompt(self) -> Optional[str]: ...
    def last_user_message(self) -> Optional[str]: ...
    def has_tool_calls(self) -> bool: ...

class AnnotatedLLMResponse:
    """Structured view of an LLM response produced by a response codec.

    Read-only: fields are accessed via properties. Complex fields
    (message, tool_calls, usage, api_specific) return Python dicts/lists.
    """

    @property
    def id(self) -> Optional[str]: ...
    @property
    def model(self) -> Optional[str]: ...
    @property
    def message(self) -> Optional[Json]: ...
    @property
    def tool_calls(self) -> Optional[list[JsonObject]]: ...
    @property
    def finish_reason(self) -> Optional[str]: ...
    @property
    def usage(self) -> Optional[JsonObject]: ...
    @property
    def api_specific(self) -> Optional[JsonObject]: ...
    @property
    def extra(self) -> JsonObject: ...
    def response_text(self) -> Optional[str]: ...
    def has_tool_calls(self) -> bool: ...

class ScopeStartEvent:
    @property
    def kind(self) -> Literal["ScopeStart"]: ...
    @property
    def parent_uuid(self) -> Optional[str]: ...
    @property
    def uuid(self) -> str: ...
    @property
    def timestamp(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def data(self) -> Optional[Json]: ...
    @property
    def metadata(self) -> Optional[Json]: ...
    @property
    def attributes(self) -> ScopeAttributes: ...
    @property
    def scope_type(self) -> ScopeType: ...

class ScopeEndEvent:
    @property
    def kind(self) -> Literal["ScopeEnd"]: ...
    @property
    def parent_uuid(self) -> Optional[str]: ...
    @property
    def uuid(self) -> str: ...
    @property
    def timestamp(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def data(self) -> Optional[Json]: ...
    @property
    def metadata(self) -> Optional[Json]: ...
    @property
    def attributes(self) -> ScopeAttributes: ...
    @property
    def scope_type(self) -> ScopeType: ...

class ToolStartEvent:
    @property
    def kind(self) -> Literal["ToolStart"]: ...
    @property
    def parent_uuid(self) -> Optional[str]: ...
    @property
    def uuid(self) -> str: ...
    @property
    def timestamp(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def data(self) -> Optional[Json]: ...
    @property
    def metadata(self) -> Optional[Json]: ...
    @property
    def attributes(self) -> ToolAttributes: ...
    @property
    def input(self) -> Optional[Json]: ...
    @property
    def tool_call_id(self) -> Optional[str]: ...

class ToolEndEvent:
    @property
    def kind(self) -> Literal["ToolEnd"]: ...
    @property
    def parent_uuid(self) -> Optional[str]: ...
    @property
    def uuid(self) -> str: ...
    @property
    def timestamp(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def data(self) -> Optional[Json]: ...
    @property
    def metadata(self) -> Optional[Json]: ...
    @property
    def attributes(self) -> ToolAttributes: ...
    @property
    def output(self) -> Optional[Json]: ...
    @property
    def tool_call_id(self) -> Optional[str]: ...

class LLMStartEvent:
    @property
    def kind(self) -> Literal["LLMStart"]: ...
    @property
    def parent_uuid(self) -> Optional[str]: ...
    @property
    def uuid(self) -> str: ...
    @property
    def timestamp(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def data(self) -> Optional[Json]: ...
    @property
    def metadata(self) -> Optional[Json]: ...
    @property
    def attributes(self) -> LLMAttributes: ...
    @property
    def input(self) -> Optional[Json]: ...
    @property
    def model_name(self) -> Optional[str]: ...
    @property
    def annotated_request(self) -> Optional[AnnotatedLLMRequest]: ...

class LLMEndEvent:
    @property
    def kind(self) -> Literal["LLMEnd"]: ...
    @property
    def parent_uuid(self) -> Optional[str]: ...
    @property
    def uuid(self) -> str: ...
    @property
    def timestamp(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def data(self) -> Optional[Json]: ...
    @property
    def metadata(self) -> Optional[Json]: ...
    @property
    def attributes(self) -> LLMAttributes: ...
    @property
    def output(self) -> Optional[Json]: ...
    @property
    def model_name(self) -> Optional[str]: ...
    @property
    def annotated_response(self) -> Optional[AnnotatedLLMResponse]: ...

class MarkEvent:
    @property
    def kind(self) -> Literal["Mark"]: ...
    @property
    def parent_uuid(self) -> Optional[str]: ...
    @property
    def uuid(self) -> str: ...
    @property
    def timestamp(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def data(self) -> Optional[Json]: ...
    @property
    def metadata(self) -> Optional[Json]: ...

Event = ScopeStartEvent | ScopeEndEvent | ToolStartEvent | ToolEndEvent | LLMStartEvent | LLMEndEvent | MarkEvent

class AtifExporter:
    """ATIF trajectory exporter that collects events and exports ATIF trajectories.

    Example:
        ```python
        exporter = AtifExporter("session-1", "my-agent", "1.0.0", model_name="gpt-4")
        exporter.register("my-exporter")
        # ... run agent workflow ...
        trajectory = exporter.export()
        exporter.deregister("my-exporter")
        ```
    """

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        agent_version: str,
        *,
        model_name: Optional[str] = None,
        tool_definitions: Optional[list[JsonObject]] = None,
        extra: Optional[Json] = None,
    ) -> None: ...
    def register(self, name: str) -> None:
        """Register this exporter as an event subscriber with the given name."""
        ...
    def deregister(self, name: str) -> bool:
        """Deregister the event subscriber. Returns ``True`` if found."""
        ...
    def export(self) -> JsonObject:
        """Export collected events as an ATIF trajectory dict."""
        ...
    def export_json(self) -> str:
        """Export collected events as a JSON string."""
        ...
    def clear(self) -> None:
        """Clear all collected events."""
        ...

class OpenTelemetryConfig:
    """Mutable configuration for ``OpenTelemetrySubscriber``."""

    transport: str
    endpoint: Optional[str]
    service_name: str
    service_namespace: Optional[str]
    service_version: Optional[str]
    instrumentation_scope: str
    timeout_millis: int

    def __init__(self) -> None: ...
    @property
    def headers(self) -> dict[str, str]:
        """Additional OTLP exporter headers/metadata."""
        ...
    @headers.setter
    def headers(self, value: dict[str, str]) -> None: ...
    @property
    def resource_attributes(self) -> dict[str, str]:
        """Additional OpenTelemetry resource attributes."""
        ...
    @resource_attributes.setter
    def resource_attributes(self, value: dict[str, str]) -> None: ...
    def set_header(self, key: str, value: str) -> None: ...
    def set_resource_attribute(self, key: str, value: str) -> None: ...

class OpenTelemetrySubscriber:
    """OpenTelemetry-backed NeMo Flow event subscriber."""

    def __init__(self, config: OpenTelemetryConfig) -> None: ...
    def register(self, name: str) -> None:
        """Register this subscriber globally with the given name."""
        ...
    def deregister(self, name: str) -> bool:
        """Deregister the subscriber. Returns ``True`` if found."""
        ...
    def force_flush(self) -> None:
        """Force a flush of finished spans through the exporter."""
        ...
    def shutdown(self) -> None:
        """Shut down the underlying tracer provider."""
        ...

class OpenInferenceConfig:
    """Mutable configuration for ``OpenInferenceSubscriber``."""

    transport: str
    endpoint: Optional[str]
    service_name: str
    service_namespace: Optional[str]
    service_version: Optional[str]
    instrumentation_scope: str
    timeout_millis: int

    def __init__(self) -> None: ...
    @property
    def headers(self) -> dict[str, str]:
        """Additional OTLP exporter headers/metadata."""
        ...
    @headers.setter
    def headers(self, value: dict[str, str]) -> None: ...
    @property
    def resource_attributes(self) -> dict[str, str]:
        """Additional OpenInference resource attributes."""
        ...
    @resource_attributes.setter
    def resource_attributes(self, value: dict[str, str]) -> None: ...
    def set_header(self, key: str, value: str) -> None: ...
    def set_resource_attribute(self, key: str, value: str) -> None: ...

class OpenInferenceSubscriber:
    """OpenInference-backed NeMo Flow event subscriber."""

    def __init__(self, config: OpenInferenceConfig) -> None: ...
    def register(self, name: str) -> None:
        """Register this subscriber globally with the given name."""
        ...
    def deregister(self, name: str) -> bool:
        """Deregister the subscriber. Returns ``True`` if found."""
        ...
    def force_flush(self) -> None:
        """Force a flush of finished spans through the exporter."""
        ...
    def shutdown(self) -> None:
        """Shut down the underlying tracer provider."""
        ...

class ScopeStack:
    """An isolated scope stack for per-request/per-task isolation.

    Example:
        ```python
        stack = create_scope_stack()
        # Use with contextvars for per-task isolation:
        nemo_flow._scope_stack_var.set(stack)
        ```
    """
    def __repr__(self) -> str: ...

class LlmStream:
    """An async iterator of Json chunks from a streaming LLM response.

    Returned by ``llm.stream_execute()``. Use with ``async for``::

        stream = await nemo_flow.llm.stream_execute("model", request, fn, collector, finalizer)
        async for chunk in stream:
            process(chunk)
    """

    def __aiter__(self) -> AsyncIterator[Json]: ...
    async def __anext__(self) -> Json: ...

# ---------------------------------------------------------------------------
# Scope stack creation
# ---------------------------------------------------------------------------

_scope_stack_var: contextvars.ContextVar[ScopeStack]

def create_scope_stack() -> ScopeStack:
    """Create a new isolated scope stack with its own root scope."""
    ...

def get_scope_stack() -> ScopeStack:
    """Get the current task's scope stack, creating one if needed."""
    ...

def set_thread_scope_stack(stack: ScopeStack) -> None:
    """Bind a ``ScopeStack`` to the current thread's thread-local storage.

    Args:
        stack: Scope stack to install for subsequent NeMo Flow API calls on the
            current thread.

    After this call, all NeMo Flow API calls on the current thread will use
    the given scope stack. Primarily used to propagate scope context into
    worker threads (e.g. ``ThreadPoolExecutor`` workers).
    """
    ...

def _native_scope_stack_active() -> bool: ...
def scope_stack_active() -> bool:
    """Return whether the current context has an explicitly-initialized scope stack.

    Returns ``True`` when the Python-side ``contextvars.ContextVar`` has been
    set (e.g. via ``get_scope_stack()``) or the Rust-side thread-local has been
    explicitly set via ``set_thread_scope_stack()``.

    Returns ``False`` when only the auto-created default is present.
    """
    ...

def propagate_scope_to_thread() -> ScopeStack:
    """Capture the current scope stack for propagation to a worker thread.

    Returns the current ``ScopeStack`` handle. Call
    ``set_thread_scope_stack()`` with the returned value inside the worker
    thread before making any NeMo Flow API calls.

    Raises:
        RuntimeError: If no scope stack has been explicitly initialized.
    """
    ...

# ---------------------------------------------------------------------------
# Scope / handle operations
# ---------------------------------------------------------------------------

def get_handle() -> Optional[ScopeHandle]:
    """Return the current scope handle from the task-local scope stack.

    Returns ``None`` if the scope stack is empty.
    """
    ...

def push_scope(
    name: str,
    scope_type: ScopeType,
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[ScopeAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> ScopeHandle:
    """Push a new child scope onto the scope stack.

    Args:
        name: Human-readable scope name.
        scope_type: The kind of scope.
        handle: Optional parent scope. Defaults to the current top of stack.
        attributes: Optional bitflags.
        data: Optional JSON-serializable application data to attach to the scope.
        metadata: Optional JSON-serializable metadata to attach to the scope.

    Returns:
        The newly created ``ScopeHandle``.

    Example:
        ```python
        handle = push_scope("my-agent", ScopeType.Agent)
        try:
            # ... do work ...
            pass
        finally:
            pop_scope(handle)
        ```
    """
    ...

def pop_scope(handle: ScopeHandle) -> None:
    """Remove a scope from the stack and emit an End event.

    Args:
        handle: The current top-of-stack scope handle returned by
            ``push_scope``.
    """
    ...

def event(
    name: str,
    *,
    handle: Optional[ScopeHandle] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """Emit a Mark event under the current or specified scope.

    Args:
        name: Event name.
        handle: Optional parent scope handle.
        data: Optional JSON-serializable application data.
        metadata: Optional JSON-serializable metadata.
    """
    ...

# ---------------------------------------------------------------------------
# Tool lifecycle
# ---------------------------------------------------------------------------

def tool_call(
    name: str,
    args: Json,
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[ToolAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
    tool_call_id: Optional[str] = None,
) -> ToolHandle:
    """Begin a tool call manually.

    Args:
        name: Tool name recorded on emitted lifecycle events.
        args: JSON-compatible tool arguments to associate with the call.
        handle: Optional parent scope handle.
        attributes: Optional ``ToolAttributes`` bitflags.
        data: Optional application data recorded on the start event.
        metadata: Optional metadata recorded on the start event.
        tool_call_id: Optional provider-specific tool call identifier.

    Returns:
        ToolHandle: Handle that must be passed to
        ``tool_call_end()``.
    """
    ...

def tool_call_end(
    handle: ToolHandle,
    result: Json,
    *,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """End a manual tool call.

    Args:
        handle: Tool handle returned by ``tool_call()``.
        result: JSON-compatible tool result to record on the end event.
        data: Optional application data recorded on the end event.
        metadata: Optional metadata recorded on the end event.
    """
    ...

def tool_call_execute(
    name: str,
    args: Json,
    func: Callable[[Json], Awaitable[Json]],
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[ToolAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> Awaitable[Json]:
    """Execute a tool call through the full middleware pipeline.

    Runs conditional-execution guardrails (on raw args) → request intercepts →
    sanitize-request guardrails (for the emitted ``Start`` event payload) →
    execution intercepts → func → response intercepts →
    sanitize-response guardrails (for the emitted ``End`` event payload). On
    rejection, only a standalone ``Mark`` event is emitted (no ``Start``/``End`` pair) and
    ``GuardrailRejected`` is raised.

    Args:
        name: Tool name.
        args: Tool arguments.
        func: Async callable ``(args) -> result`` that performs the tool work.
        handle: Optional parent scope handle.
        attributes: Optional ``ToolAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.

    Returns:
        An awaitable that resolves to the tool result after execution
        intercepts. Sanitize guardrails do not rewrite the value returned to
        the caller.

    Example:
        ```python
        async def my_tool(args):
            return {"answer": args["x"] + args["y"]}

        result = await tool_call_execute("add", {"x": 1, "y": 2}, my_tool)
        ```
    """
    ...

# ---------------------------------------------------------------------------
# LLM lifecycle
# ---------------------------------------------------------------------------

def llm_call(
    name: str,
    request: LLMRequest,
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[LLMAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
    model_name: Optional[str] = None,
) -> LLMHandle:
    """Begin an LLM call manually.

    Args:
        name: Model/provider name.
        request: An ``LLMRequest`` object with headers and content.

    Returns an ``LLMHandle`` that must be passed to ``llm_call_end``.
    Emits a Start event.
    """
    ...

def llm_call_end(
    handle: LLMHandle,
    response: Json,
    *,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """End a manual LLM call.

    Args:
        handle: LLM handle returned by ``llm_call()``.
        response: JSON-compatible response to record on the end event.
        data: Optional application data recorded on the end event.
        metadata: Optional metadata recorded on the end event.
    """
    ...

def llm_call_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], Awaitable[Json]],
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[LLMAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
    model_name: Optional[str] = None,
    codec: Optional[LlmCodec] = None,
    response_codec: Optional[LlmResponseCodec] = None,
) -> Awaitable[Json]:
    """Execute an LLM call through the full middleware pipeline.

    Runs conditional-execution guardrails → request intercepts →
    sanitize-request guardrails (for the emitted ``Start`` event payload) →
    execution intercepts → func → sanitize-response guardrails (for the
    emitted ``End`` event payload). On rejection, only a standalone
    ``Mark`` event is emitted (no ``Start``/``End`` pair) and
    ``GuardrailRejected`` is raised.

    Args:
        name: Model/provider name.
        request: An ``LLMRequest`` object with headers and content.
        func: Async callable ``(LLMRequest) -> response``.
        handle: Optional parent scope handle.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.

    Returns:
        An awaitable that resolves to the LLM response after execution
        intercepts. Sanitize guardrails do not rewrite the value returned to
        the caller.

    Example:
        ```python
        async def call_openai(req: LLMRequest) -> dict:
            # req.headers and req.content may have been modified by intercepts
            return await httpx_client.post("/chat/completions", json=req.content)

        request = LLMRequest({}, {"model": "gpt-4", "messages": [...]})
        response = await llm_call_execute("gpt-4", request, call_openai)
        ```
    """
    ...

async def llm_stream_call_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], AsyncIterator[Json]],
    collector: Callable[[Json], None],
    finalizer: Callable[[], Json],
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[LLMAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
    model_name: Optional[str] = None,
    codec: Optional[LlmCodec] = None,
    response_codec: Optional[LlmResponseCodec] = None,
) -> LlmStream:
    """Execute a streaming LLM call through the full middleware pipeline.

    Like ``llm_call_execute``, conditional-execution guardrails run
    first. On rejection, only a standalone ``Mark`` event is emitted
    (no ``Start``/``End`` pair) and ``GuardrailRejected`` is raised.

    Args:
        name: Model/provider name.
        request: An ``LLMRequest`` object with headers and content.
        func: Async callable ``(LLMRequest) -> AsyncIterator[Json]`` returning
            Json chunks.
        collector: A callable ``(chunk: Json) -> None`` invoked with each
            intercepted chunk after stream execution intercepts have been applied.
        finalizer: A callable ``() -> Json`` invoked once when the stream is
            exhausted. Its return value is the aggregated response.
        handle: Optional parent scope handle.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.

    Returns:
        An ``LlmStream`` async iterator of Json chunks.
    """
    ...

# ---------------------------------------------------------------------------
# Standalone middleware chains
# ---------------------------------------------------------------------------

def tool_request_intercepts(name: str, args: Json) -> Json:
    """Run the registered tool request intercept chain.

    Args:
        name: Tool name used when evaluating intercepts.
        args: JSON-compatible tool arguments to transform.

    Returns:
        Json: The transformed arguments.
    """
    ...

def tool_conditional_execution(name: str, args: Json) -> None:
    """Run the registered tool conditional execution guardrail chain.

    Args:
        name: Tool name used when evaluating guardrails.
        args: JSON-compatible tool arguments to validate.

    Raises ``RuntimeError`` if any guardrail rejects.
    """
    ...

def llm_request_intercepts(name: str, request: LLMRequest) -> LLMRequest:
    """Run the registered LLM request intercept chain.

    Args:
        name: Provider or logical call name used when evaluating intercepts.
        request: ``LLMRequest`` to transform.

    Returns:
        LLMRequest: The transformed request.
    """
    ...

def llm_conditional_execution(request: LLMRequest) -> None:
    """Run the registered LLM conditional execution guardrail chain.

    Args:
        request: ``LLMRequest`` to validate.

    Raises ``RuntimeError`` if any guardrail rejects.
    """
    ...

# ---------------------------------------------------------------------------
# Tool guardrails
# ---------------------------------------------------------------------------

def register_tool_sanitize_request_guardrail(name: str, priority: int, guardrail: ToolSanitizeGuardrail) -> None:
    """Register a tool sanitize-request guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callback invoked as ``(tool_name, args) ->
            sanitized_args``.

    Example:
        ```python
        def redact_keys(tool_name: str, args: dict) -> dict:
            return {k: "***" if "secret" in k else v for k, v in args.items()}

        register_tool_sanitize_request_guardrail("redact", 0, redact_keys)
        ```
    """
    ...

def deregister_tool_sanitize_request_guardrail(name: str) -> bool:
    """Remove a tool sanitize-request guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_tool_sanitize_request_guardrail()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_tool_sanitize_response_guardrail(name: str, priority: int, guardrail: ToolSanitizeGuardrail) -> None:
    """Register a tool sanitize-response guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callback invoked as ``(tool_name, result) ->
            sanitized_result``.
    """
    ...

def deregister_tool_sanitize_response_guardrail(name: str) -> bool:
    """Remove a tool sanitize-response guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_tool_sanitize_response_guardrail()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_tool_conditional_execution_guardrail(
    name: str, priority: int, guardrail: ToolConditionalExecutionGuardrail
) -> None:
    """Register a tool conditional-execution guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callback invoked as ``(tool_name, args) -> None |
            rejection_reason``.

    Example:
        ```python
        def block_dangerous(tool_name: str, args: dict) -> str | None:
            if tool_name == "rm_rf":
                return "dangerous tool blocked"
            return None  # allow

        register_tool_conditional_execution_guardrail("safety", 0, block_dangerous)
        ```
    """
    ...

def deregister_tool_conditional_execution_guardrail(name: str) -> bool:
    """Remove a tool conditional-execution guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_tool_conditional_execution_guardrail()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# Tool intercepts
# ---------------------------------------------------------------------------

def register_tool_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: ToolRequestIntercept,
) -> None:
    """Register a tool request intercept.

    Args:
        name: Unique intercept name.
        priority: Execution order for the intercept. Lower values run first.
        break_chain: Whether to stop lower-priority request intercepts after
            this one runs.
        callable: Callback invoked as ``(tool_name, args) ->
            transformed_args``.
    """
    ...

def deregister_tool_request_intercept(name: str) -> bool:
    """Remove a tool request intercept.

    Args:
        name: Intercept name previously passed to
            ``register_tool_request_intercept()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def register_tool_execution_intercept(
    name: str,
    priority: int,
    callable: ToolExecutionIntercept,
) -> None:
    """Register a tool execution intercept (middleware chain pattern).

    Args:
        name: Unique intercept name.
        priority: Execution order for the intercept. Lower values run first.
        callable: Intercept callback.

    ``callable``: ``async (tool_name, args, next) -> result`` — intercept function.
    Call ``await next(args)`` to invoke the next intercept or original
    implementation. Skip calling ``next`` to short-circuit the chain.

    Example:
        ```python
        async def cache_intercept(tool_name, args, next):
            key = json.dumps(args, sort_keys=True)
            if key in cache:
                return cache[key]
            result = await next(args)
            cache[key] = result
            return result

        register_tool_execution_intercept("cache", 0, cache_intercept)
        ```
    """
    ...

def deregister_tool_execution_intercept(name: str) -> bool:
    """Remove a tool execution intercept.

    Args:
        name: Intercept name previously passed to
            ``register_tool_execution_intercept()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# LLM guardrails
# ---------------------------------------------------------------------------

def register_llm_sanitize_request_guardrail(name: str, priority: int, guardrail: LlmSanitizeRequestGuardrail) -> None:
    """Register an LLM sanitize-request guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callback invoked as ``(request) -> sanitized_request``.
    """
    ...

def deregister_llm_sanitize_request_guardrail(name: str) -> bool:
    """Remove an LLM sanitize-request guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_llm_sanitize_request_guardrail()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_llm_sanitize_response_guardrail(name: str, priority: int, guardrail: LlmSanitizeResponseGuardrail) -> None:
    """Register an LLM sanitize-response guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callback invoked as ``(response: dict) -> dict``.
    """
    ...

def deregister_llm_sanitize_response_guardrail(name: str) -> bool:
    """Remove an LLM sanitize-response guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_llm_sanitize_response_guardrail()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_llm_conditional_execution_guardrail(
    name: str, priority: int, guardrail: LlmConditionalExecutionGuardrail
) -> None:
    """Register an LLM conditional-execution guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callback invoked as ``(request) -> None |
            rejection_reason``.
    """
    ...

def deregister_llm_conditional_execution_guardrail(name: str) -> bool:
    """Remove an LLM conditional-execution guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_llm_conditional_execution_guardrail()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# LLM intercepts
# ---------------------------------------------------------------------------

def register_llm_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: LlmRequestIntercept,
) -> None:
    """Register an LLM request intercept.

    Args:
        name: Unique intercept name.
        priority: Execution order for the intercept. Lower values run first.
        break_chain: Whether to stop lower-priority request intercepts after
            this one runs.
        callable: Callback that transforms the LLM request.
    """
    ...

def deregister_llm_request_intercept(name: str) -> bool:
    """Remove an LLM request intercept.

    Args:
        name: Intercept name previously passed to
            ``register_llm_request_intercept()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def register_llm_execution_intercept(
    name: str,
    priority: int,
    callable: LlmExecutionIntercept,
) -> None:
    """Register an LLM execution intercept (middleware chain pattern).

    Args:
        name: Unique intercept name.
        priority: Execution order for the intercept. Lower values run first.
        callable: Intercept callback.

    ``callable``: ``async (name, request, next) -> response`` — intercept function.
    Call ``await next(request)`` to invoke the next intercept or original
    implementation. Skip calling ``next`` to short-circuit the chain.

    Example:
        ```python
        async def logging_intercept(name: str, request: LLMRequest, next):
            print(f"LLM request: {request.content['model']}")
            response = await next(request)
            print(f"LLM response tokens: {len(str(response))}")
            return response

        register_llm_execution_intercept("logger", 0, logging_intercept)
        ```
    """
    ...

def deregister_llm_execution_intercept(name: str) -> bool:
    """Remove an LLM execution intercept.

    Args:
        name: Intercept name previously passed to
            ``register_llm_execution_intercept()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def register_llm_stream_execution_intercept(
    name: str,
    priority: int,
    callable: LlmStreamExecutionIntercept,
) -> None:
    """Register an LLM stream-execution intercept (middleware chain pattern).

    Args:
        name: Unique intercept name.
        priority: Execution order for the intercept. Lower values run first.
        callable: Streaming intercept callback.

    ``callable``: ``async (request, next) -> AsyncIterator[Json]`` — intercept
    function. Call ``await next(request)`` to invoke the next intercept or
    original streaming implementation. Skip calling ``next`` to short-circuit.
    """
    ...

def deregister_llm_stream_execution_intercept(name: str) -> bool:
    """Remove an LLM stream-execution intercept.

    Args:
        name: Intercept name previously passed to
            ``register_llm_stream_execution_intercept()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# Subscribers
# ---------------------------------------------------------------------------

def register_subscriber(name: str, callback: Callable[[Event], None]) -> None:
    """Register an event subscriber.

    Args:
        name: Unique subscriber name.
        callback: Callback invoked for every lifecycle event.

    Example:
        ```python
        def on_event(event: Event) -> None:
            print(f"[{event.kind}] {event.name} @ {event.timestamp}")

        register_subscriber("my-logger", on_event)
        ```
    """
    ...

def deregister_subscriber(name: str) -> bool:
    """Remove an event subscriber.

    Args:
        name: Subscriber name previously passed to
            ``register_subscriber()``.

    Returns:
        bool: ``True`` if a subscriber was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# Scope-local tool guardrails
# ---------------------------------------------------------------------------

def scope_register_tool_sanitize_request_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: ToolSanitizeGuardrail
) -> None:
    """Register a scope-local tool sanitize-request guardrail.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(tool_name, args) -> sanitized_args``. In managed
            ``tools.execute(...)`` calls, the sanitized value is used for the
            emitted ``Start`` event payload and does not replace the arguments
            passed to ``func(...)``.
    """
    ...

def scope_deregister_tool_sanitize_request_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool sanitize-request guardrail.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Guardrail name previously used during registration.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_tool_sanitize_response_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: ToolSanitizeGuardrail
) -> None:
    """Register a scope-local tool sanitize-response guardrail.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(tool_name, result) -> sanitized_result``. In managed
            ``tools.execute(...)`` calls, the sanitized value is used for the
            emitted ``End`` event payload and does not replace the value
            returned to the caller.
    """
    ...

def scope_deregister_tool_sanitize_response_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool sanitize-response guardrail.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Guardrail name previously used during registration.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_tool_conditional_execution_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: ToolConditionalExecutionGuardrail
) -> None:
    """Register a scope-local tool conditional-execution guardrail.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(tool_name, args) -> None | rejection_reason``.
    """
    ...

def scope_deregister_tool_conditional_execution_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool conditional-execution guardrail.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Guardrail name previously used during registration.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# Scope-local tool intercepts
# ---------------------------------------------------------------------------

def scope_register_tool_request_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    break_chain: bool,
    callable: ToolRequestIntercept,
) -> None:
    """Register a scope-local tool request intercept.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        break_chain: If ``True``, no lower-priority intercepts run after this one.
        callable: ``(tool_name, args) -> transformed_args``.
    """
    ...

def scope_deregister_tool_request_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool request intercept.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Intercept name previously used during registration.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def scope_register_tool_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: ToolExecutionIntercept,
) -> None:
    """Register a scope-local tool execution intercept (middleware chain pattern).

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        callable: ``async (tool_name, args, next) -> result``.
    """
    ...

def scope_deregister_tool_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool execution intercept.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Intercept name previously used during registration.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# Scope-local LLM guardrails
# ---------------------------------------------------------------------------

def scope_register_llm_sanitize_request_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: LlmSanitizeRequestGuardrail
) -> None:
    """Register a scope-local LLM sanitize-request guardrail.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(request) -> sanitized_request``. In managed
            ``llm.execute(...)`` and ``llm.stream_execute(...)`` calls, the
            sanitized value is used for the emitted ``Start`` event payload and
            does not replace the request passed to ``func(...)``.
    """
    ...

def scope_deregister_llm_sanitize_request_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM sanitize-request guardrail.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Guardrail name previously used during registration.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_sanitize_response_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: LlmSanitizeResponseGuardrail
) -> None:
    """Register a scope-local LLM sanitize-response guardrail.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(response: dict) -> dict``. In managed
            ``llm.execute(...)`` and ``llm.stream_execute(...)`` calls, the
            sanitized value is used for the emitted ``End`` event payload and
            does not replace the value returned to the caller.
    """
    ...

def scope_deregister_llm_sanitize_response_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM sanitize-response guardrail.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Guardrail name previously used during registration.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_conditional_execution_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: LlmConditionalExecutionGuardrail
) -> None:
    """Register a scope-local LLM conditional-execution guardrail.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(request) -> None | rejection_reason``.
    """
    ...

def scope_deregister_llm_conditional_execution_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM conditional-execution guardrail.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Guardrail name previously used during registration.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# Scope-local LLM intercepts
# ---------------------------------------------------------------------------

def scope_register_llm_request_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    break_chain: bool,
    callable: LlmRequestIntercept,
) -> None:
    """Register a scope-local LLM request intercept.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        break_chain: If ``True``, no lower-priority intercepts run after this one.
        callable: ``(name: str, request: LLMRequest) -> LLMRequest``.
    """
    ...

def scope_deregister_llm_request_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM request intercept.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Intercept name previously used during registration.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: LlmExecutionIntercept,
) -> None:
    """Register a scope-local LLM execution intercept (middleware chain pattern).

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        callable: ``async (name, request, next) -> response``.
    """
    ...

def scope_deregister_llm_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM execution intercept.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Intercept name previously used during registration.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_stream_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: LlmStreamExecutionIntercept,
) -> None:
    """Register a scope-local LLM stream-execution intercept (middleware chain pattern).

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        callable: ``async (request, next) -> AsyncIterator[Json]``.
    """
    ...

def scope_deregister_llm_stream_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM stream-execution intercept.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Intercept name previously used during registration.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

# ---------------------------------------------------------------------------
# Scope-local subscribers
# ---------------------------------------------------------------------------

def scope_register_subscriber(scope_uuid: str, name: str, callback: Callable[[Event], None]) -> None:
    """Register a scope-local event subscriber.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique subscriber name.
        callback: ``(event) -> None``.
    """
    ...

def scope_deregister_subscriber(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local event subscriber.

    Args:
        scope_uuid: UUID string of the scope that owns the registration.
        name: Subscriber name previously used during registration.

    Returns:
        bool: ``True`` if a subscriber was removed, otherwise ``False``.
    """
    ...
