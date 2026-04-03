# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for nat_nexus.

Provides static type information for the ``nat_nexus`` package, including
all types exported from the native Rust extension and all API functions.
"""

import contextvars
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from nat_nexus import proxy as proxy
from nat_nexus import scope as scope
from nat_nexus import typed as typed

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

    Example::

        request = LLMRequest(
            {"Authorization": "Bearer tok"},
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        print(request.headers)   # {"Authorization": "Bearer tok"}
        print(request.content)   # {"model": "gpt-4", ...}
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
    """ATIF trajectory exporter that collects events and exports ATIF trajectories.

    Example::

        exporter = AtifExporter("session-1", "my-agent", "1.0.0", model_name="gpt-4")
        exporter.register("my-exporter")
        # ... run agent workflow ...
        trajectory = exporter.export()
        exporter.deregister("my-exporter")
    """

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
    """An isolated scope stack for per-request/per-task isolation.

    Example::

        stack = create_scope_stack()
        # Use with contextvars for per-task isolation:
        nat_nexus._scope_stack_var.set(stack)
    """
    def __repr__(self) -> str: ...

class LlmStream:
    """An async iterator of Json chunks from a streaming LLM response.

    Returned by ``llm.stream_execute()``. Use with ``async for``::

        stream = await nat_nexus.llm.stream_execute("model", request, fn, collector, finalizer)
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

    After this call, all Nexus API calls on the current thread will use
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
    thread before making any Nexus API calls.

    Raises:
        RuntimeError: If no scope stack has been explicitly initialized.
    """
    ...

# ---------------------------------------------------------------------------
# Scope / handle operations
# ---------------------------------------------------------------------------

def nat_nexus_get_handle() -> Optional[ScopeHandle]:
    """Return the current scope handle from the task-local scope stack.

    Returns ``None`` if the scope stack is empty.
    """
    ...

def nat_nexus_push_scope(
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

    Example::

        handle = nat_nexus_push_scope("my-agent", ScopeType.Agent)
        try:
            # ... do work ...
            pass
        finally:
            nat_nexus_pop_scope(handle)
    """
    ...

def nat_nexus_pop_scope(handle: ScopeHandle) -> None:
    """Remove a scope from the stack and emit an End event.

    Args:
        handle: The current top-of-stack scope handle returned by
            ``nat_nexus_push_scope``.
    """
    ...

def nat_nexus_event(
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

def nat_nexus_tool_call(
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

    Returns a ``ToolHandle`` that must be passed to ``nat_nexus_tool_call_end``.
    Emits a Start event.
    """
    ...

def nat_nexus_tool_call_end(
    handle: ToolHandle,
    result: Json,
    *,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """End a manual tool call. Records the result and emits an End event."""
    ...

def nat_nexus_tool_call_execute(
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

    Example::

        async def my_tool(args):
            return {"answer": args["x"] + args["y"]}

        result = await nat_nexus_tool_call_execute("add", {"x": 1, "y": 2}, my_tool)
    """
    ...

# ---------------------------------------------------------------------------
# LLM lifecycle
# ---------------------------------------------------------------------------

def nat_nexus_llm_call(
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

    Returns an ``LLMHandle`` that must be passed to ``nat_nexus_llm_call_end``.
    Emits a Start event.
    """
    ...

def nat_nexus_llm_call_end(
    handle: LLMHandle,
    response: Json,
    *,
    data: Optional[Json] = None,
    metadata: Optional[Json] = None,
) -> None:
    """End a manual LLM call. Records the response and emits an End event."""
    ...

def nat_nexus_llm_call_execute(
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

    Example::

        async def call_openai(req: LLMRequest) -> dict:
            # req.headers and req.content may have been modified by intercepts
            return await httpx_client.post("/chat/completions", json=req.content)

        request = LLMRequest({}, {"model": "gpt-4", "messages": [...]})
        response = await nat_nexus_llm_call_execute("gpt-4", request, call_openai)
    """
    ...

async def nat_nexus_llm_stream_call_execute(
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

    Like ``nat_nexus_llm_call_execute``, conditional-execution guardrails run
    first. On rejection, only a standalone ``Mark`` event is emitted
    (no ``Start``/``End`` pair) and ``GuardrailRejected`` is raised.

    Args:
        name: Model/provider name.
        request: An ``LLMRequest`` object with headers and content.
        func: Async callable ``(LLMRequest) -> AsyncIterator[Json]`` returning
            Json chunks.
        collector: A callable ``(chunk: Json) -> None`` invoked with each
            intercepted chunk after stream execution intercepts have been applied.
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

def nat_nexus_tool_request_intercepts(name: str, args: Json) -> Json:
    """Run the registered tool request intercept chain.

    Returns the transformed arguments.
    """
    ...

def nat_nexus_tool_conditional_execution(name: str, args: Json) -> None:
    """Run the registered tool conditional execution guardrail chain.

    Raises ``RuntimeError`` if any guardrail rejects.
    """
    ...

def nat_nexus_llm_request_intercepts(name: str, request: LLMRequest) -> LLMRequest:
    """Run the registered LLM request intercept chain.

    Returns the transformed ``LLMRequest``.
    """
    ...

def nat_nexus_llm_conditional_execution(request: LLMRequest) -> None:
    """Run the registered LLM conditional execution guardrail chain.

    Raises ``RuntimeError`` if any guardrail rejects.
    """
    ...

# ---------------------------------------------------------------------------
# Tool guardrails
# ---------------------------------------------------------------------------

def nat_nexus_register_tool_sanitize_request_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Json]
) -> None:
    """Register a tool sanitize-request guardrail.

    Callback: ``(tool_name, args) -> sanitized_args``.

    Example::

        def redact_keys(tool_name: str, args: dict) -> dict:
            return {k: "***" if "secret" in k else v for k, v in args.items()}

        nat_nexus_register_tool_sanitize_request_guardrail("redact", 0, redact_keys)
    """
    ...

def nat_nexus_deregister_tool_sanitize_request_guardrail(name: str) -> bool:
    """Remove a tool sanitize-request guardrail. Returns ``True`` if found."""
    ...

def nat_nexus_register_tool_sanitize_response_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Json]
) -> None:
    """Register a tool sanitize-response guardrail.

    Callback: ``(tool_name, result) -> sanitized_result``.
    """
    ...

def nat_nexus_deregister_tool_sanitize_response_guardrail(name: str) -> bool:
    """Remove a tool sanitize-response guardrail. Returns ``True`` if found."""
    ...

def nat_nexus_register_tool_conditional_execution_guardrail(
    name: str, priority: int, guardrail: Callable[[str, Json], Optional[str]]
) -> None:
    """Register a tool conditional-execution guardrail.

    Callback: ``(tool_name, args) -> None | rejection_reason``.

    Example::

        def block_dangerous(tool_name: str, args: dict) -> str | None:
            if tool_name == "rm_rf":
                return "dangerous tool blocked"
            return None  # allow

        nat_nexus_register_tool_conditional_execution_guardrail("safety", 0, block_dangerous)
    """
    ...

def nat_nexus_deregister_tool_conditional_execution_guardrail(name: str) -> bool:
    """Remove a tool conditional-execution guardrail. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Tool intercepts
# ---------------------------------------------------------------------------

def nat_nexus_register_tool_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[str, Json], Json],
) -> None:
    """Register a tool request intercept.

    Callback: ``(tool_name, args) -> transformed_args``.
    """
    ...

def nat_nexus_deregister_tool_request_intercept(name: str) -> bool:
    """Remove a tool request intercept. Returns ``True`` if found."""
    ...

def nat_nexus_register_tool_execution_intercept(
    name: str,
    priority: int,
    callable: Callable[[str, Json, Callable[[Json], Awaitable[Json]]], Awaitable[Json]],
) -> None:
    """Register a tool execution intercept (middleware chain pattern).

    ``callable``: ``async (tool_name, args, next) -> result`` — intercept function.
    Call ``await next(args)`` to invoke the next intercept or original
    implementation. Skip calling ``next`` to short-circuit the chain.

    Example::

        async def cache_intercept(tool_name, args, next):
            key = json.dumps(args, sort_keys=True)
            if key in cache:
                return cache[key]
            result = await next(args)
            cache[key] = result
            return result

        nat_nexus_register_tool_execution_intercept("cache", 0, cache_intercept)
    """
    ...

def nat_nexus_deregister_tool_execution_intercept(name: str) -> bool:
    """Remove a tool execution intercept. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# LLM guardrails
# ---------------------------------------------------------------------------

def nat_nexus_register_llm_sanitize_request_guardrail(
    name: str, priority: int, guardrail: Callable[[LLMRequest], LLMRequest]
) -> None:
    """Register an LLM sanitize-request guardrail.

    Callback: ``(request) -> sanitized_request``.
    """
    ...

def nat_nexus_deregister_llm_sanitize_request_guardrail(name: str) -> bool:
    """Remove an LLM sanitize-request guardrail. Returns ``True`` if found."""
    ...

def nat_nexus_register_llm_sanitize_response_guardrail(
    name: str, priority: int, guardrail: Callable[[dict], dict]
) -> None:
    """Register an LLM sanitize-response guardrail.

    Callback: ``(response: dict) -> dict``.
    """
    ...

def nat_nexus_deregister_llm_sanitize_response_guardrail(name: str) -> bool:
    """Remove an LLM sanitize-response guardrail. Returns ``True`` if found."""
    ...

def nat_nexus_register_llm_conditional_execution_guardrail(
    name: str, priority: int, guardrail: Callable[[LLMRequest], Optional[str]]
) -> None:
    """Register an LLM conditional-execution guardrail.

    Callback: ``(request) -> None | rejection_reason``.
    """
    ...

def nat_nexus_deregister_llm_conditional_execution_guardrail(name: str) -> bool:
    """Remove an LLM conditional-execution guardrail. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# LLM intercepts
# ---------------------------------------------------------------------------

def nat_nexus_register_llm_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[str, LLMRequest], LLMRequest],
) -> None:
    """Register an LLM request intercept.

    Callback: ``(name: str, request: LLMRequest) -> LLMRequest`` — transforms the LLM request.
    """
    ...

def nat_nexus_deregister_llm_request_intercept(name: str) -> bool:
    """Remove an LLM request intercept. Returns ``True`` if found."""
    ...

def nat_nexus_register_llm_execution_intercept(
    name: str,
    priority: int,
    callable: Callable[[str, LLMRequest, Callable[[LLMRequest], Awaitable[Json]]], Awaitable[Json]],
) -> None:
    """Register an LLM execution intercept (middleware chain pattern).

    ``callable``: ``async (name, request, next) -> response`` — intercept function.
    Call ``await next(request)`` to invoke the next intercept or original
    implementation. Skip calling ``next`` to short-circuit the chain.

    Example::

        async def logging_intercept(name: str, request: LLMRequest, next):
            print(f"LLM request: {request.content['model']}")
            response = await next(request)
            print(f"LLM response tokens: {len(str(response))}")
            return response

        nat_nexus_register_llm_execution_intercept("logger", 0, logging_intercept)
    """
    ...

def nat_nexus_deregister_llm_execution_intercept(name: str) -> bool:
    """Remove an LLM execution intercept. Returns ``True`` if found."""
    ...

def nat_nexus_register_llm_stream_execution_intercept(
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

def nat_nexus_deregister_llm_stream_execution_intercept(name: str) -> bool:
    """Remove an LLM stream-execution intercept. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Subscribers
# ---------------------------------------------------------------------------

def nat_nexus_register_subscriber(name: str, callback: Callable[[Event], None]) -> None:
    """Register an event subscriber.

    Callback: ``(event) -> None`` — called for every lifecycle event.

    Example::

        def on_event(event: Event) -> None:
            print(f"[{event.event_type}] {event.name} @ {event.timestamp}")

        nat_nexus_register_subscriber("my-logger", on_event)
    """
    ...

def nat_nexus_deregister_subscriber(name: str) -> bool:
    """Remove an event subscriber. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Scope-local tool guardrails
# ---------------------------------------------------------------------------

def nat_nexus_scope_register_tool_sanitize_request_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: Callable[[str, Json], Json]
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

def nat_nexus_scope_deregister_tool_sanitize_request_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool sanitize-request guardrail. Returns ``True`` if found."""
    ...

def nat_nexus_scope_register_tool_sanitize_response_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: Callable[[str, Json], Json]
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

def nat_nexus_scope_deregister_tool_sanitize_response_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool sanitize-response guardrail. Returns ``True`` if found."""
    ...

def nat_nexus_scope_register_tool_conditional_execution_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: Callable[[str, Json], Optional[str]]
) -> None:
    """Register a scope-local tool conditional-execution guardrail.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(tool_name, args) -> None | rejection_reason``.
    """
    ...

def nat_nexus_scope_deregister_tool_conditional_execution_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool conditional-execution guardrail. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Scope-local tool intercepts
# ---------------------------------------------------------------------------

def nat_nexus_scope_register_tool_request_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[str, Json], Json],
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

def nat_nexus_scope_deregister_tool_request_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool request intercept. Returns ``True`` if found."""
    ...

def nat_nexus_scope_register_tool_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: Callable[[str, Json, Callable[[Json], Awaitable[Json]]], Awaitable[Json]],
) -> None:
    """Register a scope-local tool execution intercept (middleware chain pattern).

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        callable: ``async (tool_name, args, next) -> result``.
    """
    ...

def nat_nexus_scope_deregister_tool_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool execution intercept. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Scope-local LLM guardrails
# ---------------------------------------------------------------------------

def nat_nexus_scope_register_llm_sanitize_request_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: Callable[[LLMRequest], LLMRequest]
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

def nat_nexus_scope_deregister_llm_sanitize_request_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM sanitize-request guardrail. Returns ``True`` if found."""
    ...

def nat_nexus_scope_register_llm_sanitize_response_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: Callable[[dict], dict]
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

def nat_nexus_scope_deregister_llm_sanitize_response_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM sanitize-response guardrail. Returns ``True`` if found."""
    ...

def nat_nexus_scope_register_llm_conditional_execution_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: Callable[[LLMRequest], Optional[str]]
) -> None:
    """Register a scope-local LLM conditional-execution guardrail.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(request) -> None | rejection_reason``.
    """
    ...

def nat_nexus_scope_deregister_llm_conditional_execution_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM conditional-execution guardrail. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Scope-local LLM intercepts
# ---------------------------------------------------------------------------

def nat_nexus_scope_register_llm_request_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    break_chain: bool,
    callable: Callable[[str, LLMRequest], LLMRequest],
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

def nat_nexus_scope_deregister_llm_request_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM request intercept. Returns ``True`` if found."""
    ...

def nat_nexus_scope_register_llm_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: Callable[[str, LLMRequest, Callable[[LLMRequest], Awaitable[Json]]], Awaitable[Json]],
) -> None:
    """Register a scope-local LLM execution intercept (middleware chain pattern).

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        callable: ``async (name, request, next) -> response``.
    """
    ...

def nat_nexus_scope_deregister_llm_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM execution intercept. Returns ``True`` if found."""
    ...

def nat_nexus_scope_register_llm_stream_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: Callable[
        [LLMRequest, Callable[[LLMRequest], Awaitable[AsyncIterator[Json]]]], Awaitable[AsyncIterator[Json]]
    ],
) -> None:
    """Register a scope-local LLM stream-execution intercept (middleware chain pattern).

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        callable: ``async (request, next) -> AsyncIterator[Json]``.
    """
    ...

def nat_nexus_scope_deregister_llm_stream_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM stream-execution intercept. Returns ``True`` if found."""
    ...

# ---------------------------------------------------------------------------
# Scope-local subscribers
# ---------------------------------------------------------------------------

def nat_nexus_scope_register_subscriber(scope_uuid: str, name: str, callback: Callable[[Event], None]) -> None:
    """Register a scope-local event subscriber.

    Args:
        scope_uuid: UUID string of the scope to register under.
        name: Unique subscriber name.
        callback: ``(event) -> None``.
    """
    ...

def nat_nexus_scope_deregister_subscriber(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local event subscriber. Returns ``True`` if found."""
    ...
