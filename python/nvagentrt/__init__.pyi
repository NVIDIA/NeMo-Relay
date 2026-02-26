# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for nvagentrt.

Provides static type information for the ``nvagentrt`` package, including
all types exported from the native Rust extension and all API functions.
"""

import contextvars
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

Json = Any
"""Type alias for JSON-serializable Python objects (dicts, lists, strings, numbers, etc.)."""

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

class EventType:
    """Enum identifying the kind of lifecycle event."""

    Start: EventType
    """Emitted when a scope, tool call, or LLM call begins."""
    End: EventType
    """Emitted when a scope, tool call, or LLM call ends."""
    Mark: EventType
    """A user-emitted point-in-time marker event."""

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
# LLMRequest / Event / SseEvent / LlmStream
# ---------------------------------------------------------------------------

class LLMRequest:
    """Describes an HTTP request to an LLM provider.

    Passed to ``llm.call()``, ``llm.execute()``, and ``llm.stream_execute()``.
    """

    def __init__(
        self,
        method: str,
        url: str,
        headers: dict[str, Any],
        body: Json,
    ) -> None:
        """Create an LLM request.

        Args:
            method: HTTP method (e.g. ``"POST"``).
            url: Endpoint URL.
            headers: HTTP headers as a dict.
            body: JSON-serializable request body.
        """
        ...
    @property
    def method(self) -> str:
        """HTTP method."""
        ...
    @property
    def url(self) -> str:
        """Endpoint URL."""
        ...
    @property
    def headers(self) -> dict[str, Any]:
        """HTTP headers."""
        ...
    @property
    def body(self) -> Json:
        """Request body."""
        ...

class Event:
    """A lifecycle event emitted to registered subscribers.

    Events are emitted on scope push/pop, tool call start/end,
    LLM call start/end, and user-emitted marks.
    """

    @property
    def parent_uuid(self) -> Optional[str]:
        """UUID of the parent scope/handle, or ``None`` for root events."""
        ...
    @property
    def uuid(self) -> str:
        """UUID of the entity that produced this event."""
        ...
    @property
    def timestamp(self) -> str:
        """ISO 8601 UTC timestamp of when the event was created."""
        ...
    @property
    def name(self) -> Optional[str]:
        """Name of the source entity (scope, tool, LLM, or event name)."""
        ...
    @property
    def data(self) -> Optional[Json]:
        """Application-specific data snapshot at event time."""
        ...
    @property
    def metadata(self) -> Optional[Json]:
        """Metadata snapshot at event time."""
        ...
    @property
    def event_type(self) -> EventType:
        """Whether this is a Start, End, or Mark event."""
        ...
    @property
    def scope_type(self) -> Optional[ScopeType]:
        """Scope type of the source entity, if applicable."""
        ...

class SseEvent:
    """A parsed Server-Sent Events (SSE) event.

    Used by LLM stream-response intercepts to inspect or transform
    individual events within a streaming response.
    """

    def __init__(
        self,
        data: str,
        event: str | None = None,
        id: str | None = None,
        retry: int | None = None,
    ) -> None:
        """Create an SSE event.

        Args:
            data: The event data payload.
            event: Optional event type field.
            id: Optional event ID.
            retry: Optional reconnection time in milliseconds.
        """
        ...
    @property
    def event(self) -> Optional[str]:
        """The SSE event type field, or ``None``."""
        ...
    @property
    def data(self) -> str:
        """The SSE data payload."""
        ...
    @property
    def id(self) -> Optional[str]:
        """The SSE event ID, or ``None``."""
        ...
    @property
    def retry(self) -> Optional[int]:
        """Reconnection time in milliseconds, or ``None``."""
        ...

class ScopeStack:
    """An isolated scope stack for per-request/per-task isolation."""
    def __repr__(self) -> str: ...

class LlmStream:
    """An async iterator of SSE text chunks from a streaming LLM response.

    Returned by ``llm.stream_execute()``. Use with ``async for``::

        stream = await nvagentrt.llm.stream_execute("model", req, fn)
        async for chunk in stream:
            print(chunk, end="")
    """

    def __aiter__(self) -> AsyncIterator[str]: ...
    async def __anext__(self) -> str: ...

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

# ---------------------------------------------------------------------------
# Scope / handle operations
# ---------------------------------------------------------------------------

def nvagentrt_get_handle() -> Optional[ScopeHandle]:
    """Return the current scope handle from the task-local scope stack.

    Returns ``None`` if the scope stack is empty.
    """
    ...

def nvagentrt_push_scope(
    name: str,
    scope_type: ScopeType,
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[ScopeAttributes] = None,
) -> ScopeHandle:
    """Push a new child scope onto the scope stack.

    Args:
        name: Human-readable scope name.
        scope_type: The kind of scope.
        handle: Optional parent scope. Defaults to the current top of stack.
        attributes: Optional bitflags.

    Returns:
        The newly created ``ScopeHandle``.
    """
    ...

def nvagentrt_pop_scope(handle: ScopeHandle) -> None:
    """Remove a scope from the stack and emit an End event.

    Args:
        handle: The scope handle returned by ``nvagentrt_push_scope``.
    """
    ...

def nvagentrt_event(
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

def nvagentrt_tool_call(
    name: str,
    args: Json,
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[ToolAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> ToolHandle:
    """Begin a tool call manually.

    Returns a ``ToolHandle`` that must be passed to ``nvagentrt_tool_call_end``.
    Emits a Start event.
    """
    ...

def nvagentrt_tool_call_end(
    handle: ToolHandle,
    result: Json,
    *,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """End a manual tool call. Records the result and emits an End event."""
    ...

def nvagentrt_tool_call_execute(
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

    Args:
        name: Tool name.
        args: Tool arguments.
        func: Async callable ``(args) -> result`` that performs the tool work.
        handle: Optional parent scope handle.
        attributes: Optional ``ToolAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.

    Returns:
        An awaitable that resolves to the (possibly transformed) tool result.
    """
    ...

# ---------------------------------------------------------------------------
# LLM lifecycle
# ---------------------------------------------------------------------------

def nvagentrt_llm_call(
    name: str,
    request: LLMRequest,
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[LLMAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> LLMHandle:
    """Begin an LLM call manually.

    Returns an ``LLMHandle`` that must be passed to ``nvagentrt_llm_call_end``.
    Emits a Start event.
    """
    ...

def nvagentrt_llm_call_end(
    handle: LLMHandle,
    response: Json,
    *,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """End a manual LLM call. Records the response and emits an End event."""
    ...

def nvagentrt_llm_call_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], Awaitable[Json]],
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[LLMAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> Awaitable[Json]:
    """Execute an LLM call through the full middleware pipeline.

    Args:
        name: Model/provider name.
        request: The LLM request.
        func: Async callable ``(request) -> response``.
        handle: Optional parent scope handle.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.

    Returns:
        An awaitable that resolves to the (possibly transformed) LLM response.
    """
    ...

async def nvagentrt_llm_stream_call_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], AsyncIterator[str]],
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[LLMAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> LlmStream:
    """Execute a streaming LLM call through the full middleware pipeline.

    Args:
        name: Model/provider name.
        request: The LLM request.
        func: Async callable ``(request) -> AsyncIterator[str]`` returning SSE chunks.
        handle: Optional parent scope handle.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.

    Returns:
        An ``LlmStream`` async iterator of SSE text chunks.
    """
    ...

# ---------------------------------------------------------------------------
# Tool guardrails
# ---------------------------------------------------------------------------

def nvagentrt_register_tool_sanitize_request_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Json]
) -> None:
    """Register a tool sanitize-request guardrail.

    Callback: ``(tool_name, args) -> sanitized_args``.
    """
    ...

def nvagentrt_deregister_tool_sanitize_request_guardrail(name: str) -> bool:
    """Remove a tool sanitize-request guardrail. Returns ``True`` if found."""
    ...

def nvagentrt_register_tool_sanitize_response_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Json]
) -> None:
    """Register a tool sanitize-response guardrail.

    Callback: ``(tool_name, result) -> sanitized_result``.
    """
    ...

def nvagentrt_deregister_tool_sanitize_response_guardrail(name: str) -> bool:
    """Remove a tool sanitize-response guardrail. Returns ``True`` if found."""
    ...

def nvagentrt_register_tool_conditional_execution_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Optional[str]]
) -> None:
    """Register a tool conditional-execution guardrail.

    Callback: ``(tool_name, args) -> None | rejection_reason``.
    """
    ...

def nvagentrt_deregister_tool_conditional_execution_guardrail(name: str) -> bool:
    """Remove a tool conditional-execution guardrail. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Tool intercepts
# ---------------------------------------------------------------------------

def nvagentrt_register_tool_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[str, Json], Json],
) -> None:
    """Register a tool request intercept.

    Callback: ``(tool_name, args) -> transformed_args``.
    """
    ...

def nvagentrt_deregister_tool_request_intercept(name: str) -> bool:
    """Remove a tool request intercept. Returns ``True`` if found."""
    ...

def nvagentrt_register_tool_response_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[str, Json], Json],
) -> None:
    """Register a tool response intercept.

    Callback: ``(tool_name, result) -> transformed_result``.
    """
    ...

def nvagentrt_deregister_tool_response_intercept(name: str) -> bool:
    """Remove a tool response intercept. Returns ``True`` if found."""
    ...

def nvagentrt_register_tool_execution_intercept(
    name: str,
    priority: int,
    conditional: Callable[[str, Json], bool],
    callable: Callable[[Json], Awaitable[Json]],
) -> None:
    """Register a tool execution intercept.

    ``conditional``: ``(tool_name, args) -> bool`` — activates this intercept.
    ``callable``: ``async (args) -> result`` — replacement execution function.
    """
    ...

def nvagentrt_deregister_tool_execution_intercept(name: str) -> bool:
    """Remove a tool execution intercept. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# LLM guardrails
# ---------------------------------------------------------------------------

def nvagentrt_register_llm_sanitize_request_guardrail(
    name: str, priority: int, guardrail: Callable[[LLMRequest], LLMRequest]
) -> None:
    """Register an LLM sanitize-request guardrail.

    Callback: ``(request) -> sanitized_request``.
    """
    ...

def nvagentrt_deregister_llm_sanitize_request_guardrail(name: str) -> bool:
    """Remove an LLM sanitize-request guardrail. Returns ``True`` if found."""
    ...

def nvagentrt_register_llm_sanitize_response_guardrail(
    name: str, priority: int, guardrail: Callable[[Json], Json]
) -> None:
    """Register an LLM sanitize-response guardrail.

    Callback: ``(response) -> sanitized_response``.
    """
    ...

def nvagentrt_deregister_llm_sanitize_response_guardrail(name: str) -> bool:
    """Remove an LLM sanitize-response guardrail. Returns ``True`` if found."""
    ...

def nvagentrt_register_llm_conditional_execution_guardrail(
    name: str, priority: int, guardrail: Callable[[LLMRequest], Optional[str]]
) -> None:
    """Register an LLM conditional-execution guardrail.

    Callback: ``(request) -> None | rejection_reason``.
    """
    ...

def nvagentrt_deregister_llm_conditional_execution_guardrail(name: str) -> bool:
    """Remove an LLM conditional-execution guardrail. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# LLM intercepts
# ---------------------------------------------------------------------------

def nvagentrt_register_llm_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[LLMRequest], LLMRequest],
) -> None:
    """Register an LLM request intercept.

    Callback: ``(request) -> transformed_request``.
    """
    ...

def nvagentrt_deregister_llm_request_intercept(name: str) -> bool:
    """Remove an LLM request intercept. Returns ``True`` if found."""
    ...

def nvagentrt_register_llm_response_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[Json], Json],
) -> None:
    """Register an LLM response intercept.

    Callback: ``(response) -> transformed_response``.
    """
    ...

def nvagentrt_deregister_llm_response_intercept(name: str) -> bool:
    """Remove an LLM response intercept. Returns ``True`` if found."""
    ...

def nvagentrt_register_llm_stream_response_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[SseEvent], SseEvent],
) -> None:
    """Register an LLM stream-response intercept.

    Callback: ``(event) -> transformed_event`` — applied to each SSE event.
    """
    ...

def nvagentrt_deregister_llm_stream_response_intercept(name: str) -> bool:
    """Remove an LLM stream-response intercept. Returns ``True`` if found."""
    ...

def nvagentrt_register_llm_execution_intercept(
    name: str,
    priority: int,
    conditional: Callable[[LLMRequest], bool],
    callable: Callable[[LLMRequest], Awaitable[Json]],
) -> None:
    """Register an LLM execution intercept.

    ``conditional``: ``(request) -> bool`` — activates this intercept.
    ``callable``: ``async (request) -> response`` — replacement execution function.
    """
    ...

def nvagentrt_deregister_llm_execution_intercept(name: str) -> bool:
    """Remove an LLM execution intercept. Returns ``True`` if found."""
    ...

def nvagentrt_register_llm_stream_execution_intercept(
    name: str,
    priority: int,
    conditional: Callable[[LLMRequest], bool],
    callable: Callable[[LLMRequest], AsyncIterator[str]],
) -> None:
    """Register an LLM stream-execution intercept.

    ``conditional``: ``(request) -> bool`` — activates this intercept.
    ``callable``: ``async (request) -> AsyncIterator[str]`` — replacement
    streaming execution function.
    """
    ...

def nvagentrt_deregister_llm_stream_execution_intercept(name: str) -> bool:
    """Remove an LLM stream-execution intercept. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Subscribers
# ---------------------------------------------------------------------------

def nvagentrt_register_subscriber(name: str, callback: Callable[[Event], None]) -> None:
    """Register an event subscriber.

    Callback: ``(event) -> None`` — called for every lifecycle event.
    """
    ...

def nvagentrt_deregister_subscriber(name: str) -> bool:
    """Remove an event subscriber. Returns ``True`` if found."""
    ...
