# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for nvmagic.

Provides static type information for the ``nvmagic`` package, including
all types exported from the native Rust extension and all API functions.
"""

import contextvars
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from nvmagic import typed as typed

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
# LLMRequest / Event / LlmStream
# ---------------------------------------------------------------------------

class LLMRequest:
    """An LLM request carrying headers and a content payload.

    Flows through the entire middleware pipeline: guardrails, intercepts,
    and execution functions all receive and return ``LLMRequest``.
    """

    def __init__(
        self,
        headers: dict[str, Any],
        content: Json,
    ) -> None:
        """Create an LLM request.

        Args:
            headers: Metadata key-value pairs.
            content: The request payload.
        """
        ...
    @property
    def headers(self) -> dict[str, Any]:
        """Metadata key-value pairs."""
        ...
    @property
    def content(self) -> Json:
        """The request payload."""
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
    @property
    def input(self) -> Optional[Any]:
        """Post-guardrail input (tool args, LLM request)."""
        ...
    @property
    def output(self) -> Optional[Any]:
        """Post-guardrail output (tool result, LLM response)."""
        ...
    @property
    def model_name(self) -> Optional[str]:
        """LLM model identifier."""
        ...
    @property
    def tool_call_id(self) -> Optional[str]:
        """External correlation ID for tool calls."""
        ...
    @property
    def root_uuid(self) -> Optional[str]:
        """Root scope UUID for concurrent agent isolation."""
        ...

class AtifExporter:
    """ATIF trajectory exporter that collects events and exports ATIF trajectories."""

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        agent_version: str,
        *,
        model_name: Optional[str] = None,
        tool_definitions: Optional[list[Any]] = None,
        extra: Optional[Any] = None,
    ) -> None: ...
    def register(self, name: str) -> None:
        """Register this exporter as an event subscriber with the given name."""
        ...
    def deregister(self, name: str) -> bool:
        """Deregister the event subscriber. Returns ``True`` if found."""
        ...
    def export(self, root_uuid: Optional[str] = None) -> dict[str, Any]:
        """Export collected events as an ATIF trajectory dict."""
        ...
    def export_json(self, root_uuid: Optional[str] = None) -> str:
        """Export collected events as a JSON string."""
        ...
    def clear(self) -> None:
        """Clear all collected events."""
        ...

class ScopeStack:
    """An isolated scope stack for per-request/per-task isolation."""
    def __repr__(self) -> str: ...

class LlmStream:
    """An async iterator of Json chunks from a streaming LLM response.

    Returned by ``llm.stream_execute()``. Use with ``async for``::

        stream = await nvmagic.llm.stream_execute("model", request, fn, collector, finalizer)
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

# ---------------------------------------------------------------------------
# Scope / handle operations
# ---------------------------------------------------------------------------

def nvmagic_get_handle() -> Optional[ScopeHandle]:
    """Return the current scope handle from the task-local scope stack.

    Returns ``None`` if the scope stack is empty.
    """
    ...

def nvmagic_push_scope(
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

def nvmagic_pop_scope(handle: ScopeHandle) -> None:
    """Remove a scope from the stack and emit an End event.

    Args:
        handle: The scope handle returned by ``nvmagic_push_scope``.
    """
    ...

def nvmagic_event(
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

def nvmagic_tool_call(
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

    Returns a ``ToolHandle`` that must be passed to ``nvmagic_tool_call_end``.
    Emits a Start event.
    """
    ...

def nvmagic_tool_call_end(
    handle: ToolHandle,
    result: Json,
    *,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """End a manual tool call. Records the result and emits an End event."""
    ...

def nvmagic_tool_call_execute(
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
    sanitize-request guardrails → execution intercepts → func → response
    intercepts → sanitize-response guardrails. On rejection, only a standalone
    ``Mark`` event is emitted (no ``Start``/``End`` pair) and
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
        An awaitable that resolves to the (possibly transformed) tool result.
    """
    ...

# ---------------------------------------------------------------------------
# LLM lifecycle
# ---------------------------------------------------------------------------

def nvmagic_llm_call(
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

    Returns an ``LLMHandle`` that must be passed to ``nvmagic_llm_call_end``.
    Emits a Start event.
    """
    ...

def nvmagic_llm_call_end(
    handle: LLMHandle,
    response: Json,
    *,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """End a manual LLM call. Records the response and emits an End event."""
    ...

def nvmagic_llm_call_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], Awaitable[Json]],
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[LLMAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
    model_name: Optional[str] = None,
) -> Awaitable[Json]:
    """Execute an LLM call through the full middleware pipeline.

    Runs conditional-execution guardrails → request intercepts →
    sanitize-request guardrails → execution intercepts → func →
    sanitize-response guardrails. On rejection, only a standalone
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
        An awaitable that resolves to the (possibly transformed) LLM response.
    """
    ...

async def nvmagic_llm_stream_call_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], AsyncIterator[Json]],
    collector: Callable[[Json], None],
    finalizer: Callable[[], Any],
    *,
    handle: Optional[ScopeHandle] = None,
    attributes: Optional[LLMAttributes] = None,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
    model_name: Optional[str] = None,
) -> LlmStream:
    """Execute a streaming LLM call through the full middleware pipeline.

    Like ``nvmagic_llm_call_execute``, conditional-execution guardrails run
    first. On rejection, only a standalone ``Mark`` event is emitted
    (no ``Start``/``End`` pair) and ``GuardrailRejected`` is raised.

    Args:
        name: Model/provider name.
        request: An ``LLMRequest`` object with headers and content.
        func: Async callable ``(LLMRequest) -> AsyncIterator[Json]`` returning
            Json chunks.
        collector: A callable ``(chunk: Json) -> None`` invoked with each
            intercepted chunk after stream response intercepts have been applied.
        finalizer: A callable ``() -> Any`` invoked once when the stream is
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

def nvmagic_tool_request_intercepts(name: str, args: Json) -> Json:
    """Run the registered tool request intercept chain.

    Returns the transformed arguments.
    """
    ...

def nvmagic_tool_conditional_execution(name: str, args: Json) -> None:
    """Run the registered tool conditional execution guardrail chain.

    Raises ``RuntimeError`` if any guardrail rejects.
    """
    ...

def nvmagic_tool_response_intercepts(name: str, result: Json) -> Json:
    """Run the registered tool response intercept chain.

    Returns the transformed result.
    """
    ...

def nvmagic_llm_request_intercepts(request: LLMRequest) -> LLMRequest:
    """Run the registered LLM request intercept chain.

    Returns the transformed ``LLMRequest``.
    """
    ...

def nvmagic_llm_conditional_execution(request: LLMRequest) -> None:
    """Run the registered LLM conditional execution guardrail chain.

    Raises ``RuntimeError`` if any guardrail rejects.
    """
    ...

# ---------------------------------------------------------------------------
# Tool guardrails
# ---------------------------------------------------------------------------

def nvmagic_register_tool_sanitize_request_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Json]
) -> None:
    """Register a tool sanitize-request guardrail.

    Callback: ``(tool_name, args) -> sanitized_args``.
    """
    ...

def nvmagic_deregister_tool_sanitize_request_guardrail(name: str) -> bool:
    """Remove a tool sanitize-request guardrail. Returns ``True`` if found."""
    ...

def nvmagic_register_tool_sanitize_response_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Json]
) -> None:
    """Register a tool sanitize-response guardrail.

    Callback: ``(tool_name, result) -> sanitized_result``.
    """
    ...

def nvmagic_deregister_tool_sanitize_response_guardrail(name: str) -> bool:
    """Remove a tool sanitize-response guardrail. Returns ``True`` if found."""
    ...

def nvmagic_register_tool_conditional_execution_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Optional[str]]
) -> None:
    """Register a tool conditional-execution guardrail.

    Callback: ``(tool_name, args) -> None | rejection_reason``.
    """
    ...

def nvmagic_deregister_tool_conditional_execution_guardrail(name: str) -> bool:
    """Remove a tool conditional-execution guardrail. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Tool intercepts
# ---------------------------------------------------------------------------

def nvmagic_register_tool_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[str, Json], Json],
) -> None:
    """Register a tool request intercept.

    Callback: ``(tool_name, args) -> transformed_args``.
    """
    ...

def nvmagic_deregister_tool_request_intercept(name: str) -> bool:
    """Remove a tool request intercept. Returns ``True`` if found."""
    ...

def nvmagic_register_tool_response_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[str, Json], Json],
) -> None:
    """Register a tool response intercept.

    Callback: ``(tool_name, result) -> transformed_result``.
    """
    ...

def nvmagic_deregister_tool_response_intercept(name: str) -> bool:
    """Remove a tool response intercept. Returns ``True`` if found."""
    ...

def nvmagic_register_tool_execution_intercept(
    name: str,
    priority: int,
    callable: Callable[[Json, Callable[[Json], Awaitable[Json]]], Awaitable[Json]],
) -> None:
    """Register a tool execution intercept (middleware chain pattern).

    ``callable``: ``async (args, next) -> result`` — intercept function.
    Call ``await next(args)`` to invoke the next intercept or original
    implementation. Skip calling ``next`` to short-circuit the chain.
    """
    ...

def nvmagic_deregister_tool_execution_intercept(name: str) -> bool:
    """Remove a tool execution intercept. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# LLM guardrails
# ---------------------------------------------------------------------------

def nvmagic_register_llm_sanitize_request_guardrail(
    name: str, priority: int, guardrail: Callable[[LLMRequest], LLMRequest]
) -> None:
    """Register an LLM sanitize-request guardrail.

    Callback: ``(request) -> sanitized_request``.
    """
    ...

def nvmagic_deregister_llm_sanitize_request_guardrail(name: str) -> bool:
    """Remove an LLM sanitize-request guardrail. Returns ``True`` if found."""
    ...

def nvmagic_register_llm_sanitize_response_guardrail(
    name: str, priority: int, guardrail: Callable[[dict], dict]
) -> None:
    """Register an LLM sanitize-response guardrail.

    Callback: ``(response: dict) -> dict``.
    """
    ...

def nvmagic_deregister_llm_sanitize_response_guardrail(name: str) -> bool:
    """Remove an LLM sanitize-response guardrail. Returns ``True`` if found."""
    ...

def nvmagic_register_llm_conditional_execution_guardrail(
    name: str, priority: int, guardrail: Callable[[LLMRequest], Optional[str]]
) -> None:
    """Register an LLM conditional-execution guardrail.

    Callback: ``(request) -> None | rejection_reason``.
    """
    ...

def nvmagic_deregister_llm_conditional_execution_guardrail(name: str) -> bool:
    """Remove an LLM conditional-execution guardrail. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# LLM intercepts
# ---------------------------------------------------------------------------

def nvmagic_register_llm_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[LLMRequest], LLMRequest],
) -> None:
    """Register an LLM request intercept.

    Callback: ``(request: LLMRequest) -> LLMRequest`` — transforms the LLM request.
    """
    ...

def nvmagic_deregister_llm_request_intercept(name: str) -> bool:
    """Remove an LLM request intercept. Returns ``True`` if found."""
    ...

def nvmagic_register_llm_execution_intercept(
    name: str,
    priority: int,
    callable: Callable[[LLMRequest, Callable[[LLMRequest], Awaitable[Json]]], Awaitable[Json]],
) -> None:
    """Register an LLM execution intercept (middleware chain pattern).

    ``callable``: ``async (request, next) -> response`` — intercept function.
    Call ``await next(request)`` to invoke the next intercept or original
    implementation. Skip calling ``next`` to short-circuit the chain.
    """
    ...

def nvmagic_deregister_llm_execution_intercept(name: str) -> bool:
    """Remove an LLM execution intercept. Returns ``True`` if found."""
    ...

def nvmagic_register_llm_stream_execution_intercept(
    name: str,
    priority: int,
    callable: Callable[
        [LLMRequest, Callable[[LLMRequest], Awaitable[AsyncIterator[Json]]]], Awaitable[AsyncIterator[Json]]
    ],
) -> None:
    """Register an LLM stream-execution intercept (middleware chain pattern).

    ``callable``: ``async (request, next) -> AsyncIterator[Json]`` — intercept
    function. Call ``await next(request)`` to invoke the next intercept or
    original streaming implementation. Skip calling ``next`` to short-circuit.
    """
    ...

def nvmagic_deregister_llm_stream_execution_intercept(name: str) -> bool:
    """Remove an LLM stream-execution intercept. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Subscribers
# ---------------------------------------------------------------------------

def nvmagic_register_subscriber(name: str, callback: Callable[[Event], None]) -> None:
    """Register an event subscriber.

    Callback: ``(event) -> None`` — called for every lifecycle event.
    """
    ...

def nvmagic_deregister_subscriber(name: str) -> bool:
    """Remove an event subscriber. Returns ``True`` if found."""
    ...
