# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python bindings for the NeMo Relay runtime.

This package exposes the runtime's scope stack, lifecycle events, middleware
registries, typed wrappers, and adaptive helpers from Python.

The main entry points are:

- ``nemo_relay.scope`` for creating and nesting scopes
- ``nemo_relay.tools`` for tool lifecycle management
- ``nemo_relay.llm`` for non-streaming and streaming LLM lifecycle management
- ``nemo_relay.guardrails`` and ``nemo_relay.intercepts`` for global middleware
- ``nemo_relay.event_metadata`` for global event metadata injection
- ``nemo_relay.scope_local`` for middleware scoped to a specific ``ScopeHandle``
- ``nemo_relay.typed`` for codec-based typed wrappers
- ``nemo_relay.plugin`` for global plugin configuration and custom plugin registration
- ``nemo_relay.adaptive`` for adaptive component configuration helpers
- ``nemo_relay.observability`` for observability component configuration helpers
- ``nemo_relay.pii_redaction`` for PII redaction component configuration helpers
- ``nemo_relay.model_pricing`` for model pricing component configuration helpers

Top-level exports also include:

- scope stack helpers such as ``get_scope_stack()``, ``create_scope_stack()``,
  ``fork_asyncio_context()``, ``set_thread_scope_stack()``, and
  ``scope_stack_active()``
- native runtime types such as ``ScopeHandle``, ``ToolHandle``, ``LLMHandle``,
  ``LLMRequest``, ``ScopeType``, and the lifecycle event classes
- observability helpers such as ``AtifExporter``, ``AtofExporter``,
  and ``OpenTelemetrySubscriber``
- JSON and callback type aliases used by middleware, typed wrappers, and
  plugin-facing configuration helpers

Example::

    import asyncio

    import nemo_relay

    def redact_args(tool_name, args):
        return {**args, "api_key": "***"}

    def add_header(
        name: str,
        request: nemo_relay.LLMRequest,
        annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        # The request object is immutable, however we can return a new instance with updated headers.
        headers = request.headers.copy()
        headers["Authorization"] = "Bearer test-token"
        return nemo_relay.LLMRequestInterceptOutcome(
            nemo_relay.LLMRequest(headers=headers, content=request.content), annotated
        )

    async def tool_impl(args):
        return nemo_relay.ToolExecutionResult({"echo": args["query"]})

    async def llm_impl(request):
        return {"messages": request.content["messages"], "ok": True}

    async def main():
        nemo_relay.guardrails.register_tool_sanitize_request("redact", 10, redact_args)
        nemo_relay.intercepts.register_llm_request("auth", 10, False, add_header)

        with nemo_relay.scope.scope("demo-agent", nemo_relay.ScopeType.Agent):
            tool_result = await nemo_relay.tools.execute("search", {"query": "hello"}, tool_impl)
            llm_result = await nemo_relay.llm.execute(
                "demo-model",
                nemo_relay.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
                llm_impl,
            )

            print(tool_result, llm_result)

    asyncio.run(main())
"""

from __future__ import annotations

import atexit
import contextvars
import typing
from collections.abc import Callable as AbcCallable
from contextlib import contextmanager
from typing import AsyncIterator, Awaitable, Callable, Iterator, Literal, Optional, TypeAlias, TypedDict

# Native bitflag classes exported at the top level for user code.
# Native LLM request and normalized codec view types.
# Native handle types returned by scope, tool, and LLM lifecycle APIs.
# Native event classes delivered to subscribers and exporters.
# Native observability exporters and subscriber configuration types.
# Native scope stack handle and low-level synchronization functions.
from nemo_relay._native import (
    AnnotatedLLMRequest,
    AnnotatedLLMResponse,
    AtifExporter,
    AtofExporter,
    AtofExporterConfig,
    AtofExporterMode,
    AtofStreamSinkConfig,
    LLMAttributes,
    LlmCodecIdentity,
    LLMHandle,
    LLMRequest,
    LLMRequestInterceptOutcome,
    LlmSanitizeRequestCodec,
    LlmSanitizeRequestContext,
    LlmSanitizeResponseCodec,
    LlmSanitizeResponseContext,
    LogSeverity,
    MarkEvent,
    MetricKind,
    MetricMeasurement,
    MetricTemporality,
    MetricValueType,
    OpenTelemetryConfig,
    OpenTelemetryLogConfig,
    OpenTelemetryLogSubscriber,
    OpenTelemetryMetricConfig,
    OpenTelemetryMetricSubscriber,
    OpenTelemetryRuntimeDiagnostic,
    OpenTelemetryRuntimeDiagnostics,
    OpenTelemetrySubscriber,
    PendingMarkSpec,
    PropagationContext,
    ScopeAttributes,
    ScopeEvent,
    ScopeHandle,
    ScopeStack,
    ScopeType,
    ToolAttributes,
    ToolExecutionInterceptOutcome,
    ToolExecutionResult,
    ToolHandle,
    _shutdown_default_logging,
)
from nemo_relay._native import (
    capture_propagation_context as _capture_propagation_context,
)
from nemo_relay._native import (
    capture_propagation_context_with_root as _capture_propagation_context_with_root,
)
from nemo_relay._native import capture_rootless_propagation_context as _capture_rootless_propagation_context
from nemo_relay._native import capture_thread_scope_stack as _capture_thread_scope_stack
from nemo_relay._native import capture_traceparent as _capture_traceparent
from nemo_relay._native import create_scope_stack as _create_scope_stack
from nemo_relay._native import (
    create_scope_stack_from_propagation as _create_scope_stack_from_propagation,
)
from nemo_relay._native import restore_thread_scope_stack as _restore_thread_scope_stack
from nemo_relay._native import scope_stack_active as _native_scope_stack_active
from nemo_relay._native import set_thread_scope_stack as _set_thread_scope_stack
from nemo_relay._native import sync_thread_scope_stack as _sync_thread_scope_stack

atexit.register(_shutdown_default_logging)

#: Scalar JSON leaf values accepted in NeMo Relay payloads. This alias has no
#: runtime behavior; it exists to document and type JSON-compatible public API
#: arguments and return values.
JsonPrimitive: TypeAlias = str | int | float | bool | None
#: Recursive JSON-compatible value accepted by payload-carrying APIs. Lists and
#: dictionaries may nest further ``JsonValue`` instances. Invalid
#: serialization errors are raised by the downstream API that consumes a value.
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
#: Mapping-shaped JSON payload used by structured request and response helpers.
#: Keys must be strings and values must be JSON-compatible.
JsonObject: TypeAlias = dict[str, JsonValue]
#: Shorthand for any JSON-compatible payload accepted by the Python binding.
#: Functions that accept ``Json`` may still enforce a narrower object shape.
Json: TypeAlias = JsonValue
#: Policy used by config helpers when unknown fields or unsupported values are
#: encountered. Callers pass these string literals in plugin and adaptive
#: configuration dataclasses.
UnsupportedBehavior: TypeAlias = Literal["ignore", "warn", "error"]


class DataSchema(TypedDict):
    """Schema identifier attached to an event's opaque data payload."""

    name: str
    version: str


class EventSanitizeFields(TypedDict):
    """Observability fields returned by mark and scope event sanitizers."""

    data: Json | None
    category_profile: JsonObject | None
    metadata: Json | None


#: Guardrail callback that sanitizes emitted tool request or response payloads.
#: Arguments are the tool name and JSON payload. The return value is the JSON
#: payload recorded on the emitted event. Exceptions fail closed and omit the
#: observability payload, then stop the remaining sanitizer chain.
ToolSanitizeGuardrail: TypeAlias = Callable[[str, Json], Json | Awaitable[Json]]
#: Guardrail callback that sanitizes mark and scope event fields. Exceptions
#: fail closed: Relay clears all mutable observability fields and stops the
#: remaining sanitizer chain.
EventSanitizeGuardrail: TypeAlias = Callable[
    ["Event", EventSanitizeFields], EventSanitizeFields | Awaitable[EventSanitizeFields]
]
#: Primitive values accepted from event metadata injectors.
EventMetadataScalar: TypeAlias = str | int | float | bool
#: Flat event metadata value accepted from an injector. Lists must be empty or
#: contain values of one primitive type; Relay validates this at runtime.
EventMetadataValue: TypeAlias = EventMetadataScalar | list[str] | list[int] | list[float] | list[bool]
#: Flat metadata additions returned by an event metadata injector.
EventMetadata: TypeAlias = dict[str, EventMetadataValue]
#: Additive middleware callback that inspects an immutable event and returns
#: proposed metadata additions. Both synchronous and asynchronous callbacks are
#: supported.
EventMetadataInjectorCallback: TypeAlias = Callable[["Event"], EventMetadata | Awaitable[EventMetadata]]
#: Guardrail callback that can block tool execution by returning a rejection
#: message. Returning ``None`` allows execution to continue.
ToolConditionalExecutionGuardrail: TypeAlias = Callable[[str, Json], Optional[str] | Awaitable[Optional[str]]]
#: Guardrail callback that sanitizes an ``LLMRequest`` used for emitted events.
#: Callbacks receive ``(request, context)``. Returning ``None`` omits the LLM observability
#: payload and annotation without changing the caller-visible request. Exceptions
#: also fail closed and stop the remaining sanitizer chain.
LlmSanitizeRequestGuardrail: TypeAlias = Callable[
    [LLMRequest, "LlmSanitizeRequestContext"],
    Optional[LLMRequest] | Awaitable[Optional[LLMRequest]],
]
#: Guardrail callback that sanitizes an emitted JSON LLM response payload.
#: Callbacks receive ``(response, context)`` and can return ``None`` to omit
#: observability payload and annotation without changing the caller response.
#: Exceptions also fail closed and stop the remaining sanitizer chain.
LlmSanitizeResponseGuardrail: TypeAlias = Callable[
    [Json, "LlmSanitizeResponseContext"], Optional[Json] | Awaitable[Optional[Json]]
]
#: Guardrail callback that can block an LLM call by returning a rejection
#: message. Returning ``None`` allows execution to continue.
LlmConditionalExecutionGuardrail: TypeAlias = Callable[[LLMRequest], Optional[str] | Awaitable[Optional[str]]]
#: Request intercept callback that rewrites tool arguments before execution.
#: Arguments are the tool name and current JSON payload. The return value
#: becomes the payload seen by later request intercepts and tool execution.
ToolRequestIntercept: TypeAlias = AbcCallable[[str, Json], Json | Awaitable[Json]]
#: Execution intercept callback that wraps tool execution with middleware
#: behavior. The callback receives the tool name, current arguments, and the
#: next callable. It may await and return ``next(args)`` or short-circuit.
ToolExecutionIntercept: TypeAlias = Callable[
    [str, Json, Callable[[Json], Awaitable[ToolExecutionResult[Json]]]],
    ToolExecutionInterceptOutcome | Awaitable[ToolExecutionInterceptOutcome],
]
#: Request intercept callback that returns the canonical request, annotation,
#: and pending-mark outcome passed to later intercepts and managed execution.
LlmRequestIntercept: TypeAlias = Callable[
    [str, LLMRequest, AnnotatedLLMRequest | None],
    LLMRequestInterceptOutcome | Awaitable[LLMRequestInterceptOutcome],
]
#: Execution intercept callback that wraps non-streaming LLM execution. The
#: callback receives the logical LLM name, request, and next callable. It may
#: await the next callable or return a replacement JSON-compatible response.
LlmExecutionIntercept: TypeAlias = Callable[
    [str, LLMRequest, Callable[[LLMRequest], Awaitable[Json]]],
    Json | Awaitable[Json],
]
#: Execution intercept callback that wraps streaming LLM execution. The
#: callback receives the current request and a next callable that returns an
#: async iterator of chunks. It may return or await a replacement iterator.
LlmStreamExecutionIntercept: TypeAlias = Callable[
    [LLMRequest, Callable[[LLMRequest], Awaitable[AsyncIterator[Json]]]],
    AsyncIterator[Json] | Awaitable[AsyncIterator[Json]],
]

# intentionally not importing utils.py to avoid overhead of creating the ThreadPoolExecutor unless it is needed
from nemo_relay import (  # noqa: E402
    adaptive,
    codecs,
    event_metadata,
    guardrails,
    intercepts,
    llm,
    model_pricing,
    observability,
    pii_redaction,
    plugin,
    runtime_registrations,
    scope,
    scope_local,
    subscribers,
    tools,
    typed,
)

_scope_stack_var: contextvars.ContextVar[ScopeStack] = contextvars.ContextVar("scope_stack")
_propagation_parent_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "propagation_parent",
    default=None,
)
_propagation_root_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "propagation_root",
    default=None,
)


def get_scope_stack() -> ScopeStack:
    """Return the current task's active scope stack.

    If the current async context does not yet own a scope stack, this function
    creates one and synchronizes it into the Rust thread-local storage used by
    the native runtime. Most callers do not need to invoke this directly
    because higher-level helpers such as ``nemo_relay.scope.push()`` do it
    automatically.

    Returns:
        ScopeStack: The scope stack associated with the current Python context.

    Raises:
        Exception: Propagates any exception raised by native scope-stack
            creation or synchronization.

    Behavior:
        The function first checks the Python ``ContextVar``. If no stack is
        present, it creates one with the native runtime, stores it in the
        current context, and synchronizes that stack into native thread-local
        storage before returning.

    Notes:
        Calling this function synchronizes the Python ``ContextVar`` state into
        the native thread-local slot so subsequent native runtime calls observe
        the same scope hierarchy.

    Example::

        import nemo_relay

        stack = nemo_relay.get_scope_stack()
        assert stack is not None
    """
    stack = _scope_stack_var.get(None)
    if stack is None:
        stack = _create_scope_stack()
        _scope_stack_var.set(stack)
    # Keep the Rust thread-local in sync so that native calls (which read
    # from THREAD_SCOPE_STACK / TASK_SCOPE_STACK) see the same scope stack.
    # Uses sync (not set) to avoid marking this thread as explicitly active.
    _sync_thread_scope_stack(stack)
    return stack


def scope_stack_active() -> bool:
    """Report whether the current context already owns a scope stack.

    Returns:
        bool: ``True`` when the current Python context already has an active
        stack, either because it was created in this context or because a stack
        was explicitly installed for the current thread.

    Raises:
        Exception: Propagates any exception raised by the native active-stack
            status check.

    Behavior:
        The Python ``ContextVar`` is checked first. If it has no stack, the
        native runtime is asked whether the current thread has an explicitly
        active stack.

    Notes:
        This function does not create a scope stack. It is a pure status check
        used to decide whether scope propagation work is required.

    Example::

        import nemo_relay

        assert nemo_relay.scope_stack_active() is False
        nemo_relay.get_scope_stack()
        assert nemo_relay.scope_stack_active() is True
    """
    if _scope_stack_var.get(None) is not None:
        return True
    return _native_scope_stack_active()


def propagate_scope_to_thread() -> ScopeStack:
    """Capture the active scope stack for use in another thread.

    The returned stack can be passed to ``set_thread_scope_stack()`` inside a
    worker thread so that the worker emits events into the same scope hierarchy
    as the parent context.

    Returns:
        ScopeStack: The active stack from the current context.

    Raises:
        RuntimeError: If the current context does not yet have an active scope
            stack to propagate.
        Exception: Propagates any exception raised while synchronizing an
            already-active native stack into the Python context.

    Behavior:
        This function does not clone the scope hierarchy. It shares the current
        stack reference with the target thread, which is appropriate when the
        worker should contribute events to the same logical trace.

    Example::

        from concurrent.futures import ThreadPoolExecutor

        import nemo_relay

        with nemo_relay.scope.scope("parent", nemo_relay.ScopeType.Agent) as handle:
            stack = nemo_relay.propagate_scope_to_thread()

            def worker() -> None:
                nemo_relay.set_thread_scope_stack(stack)
                nemo_relay.scope.event(
                    "worker-ran",
                    handle=handle,
                    data={"source": "thread"},
                    metadata={"thread": "pool-1"},
                )

            with ThreadPoolExecutor() as pool:
                pool.submit(worker).result()
    """
    if not scope_stack_active():
        raise RuntimeError(
            "no active scope stack in current context; call nemo_relay.get_scope_stack() "
            "or nemo_relay.scope.push() first"
        )
    # Return the ContextVar value directly if available, to avoid
    # calling get_scope_stack() which would sync to the Rust thread-local.
    stack = _scope_stack_var.get(None)
    if stack is not None:
        return stack
    # Rust-side explicit flag is set. Return via get_scope_stack() which
    # will create a ContextVar entry and sync.
    return get_scope_stack()


def create_scope_stack() -> ScopeStack:
    """Create a new isolated scope stack.

    Returns:
        ScopeStack: A fresh scope stack that is not yet attached to the current
        Python context or thread.

    Raises:
        Exception: Propagates any native error raised while allocating the
            stack.

    Behavior:
        This is a direct top-level wrapper around the native stack factory. It
        does not mutate the Python ``ContextVar`` and does not install the stack
        into native thread-local storage.

    Notes:
        Use this helper when you need explicit scope isolation, such as test
        fixtures, manual context propagation, or framework-managed request
        boundaries. Most application code should prefer ``get_scope_stack()``
        so the current context is initialized lazily.

    Example::

        import nemo_relay

        stack = nemo_relay.create_scope_stack()
        nemo_relay.set_thread_scope_stack(stack)
    """
    return _create_scope_stack()


def capture_propagation_context() -> PropagationContext:
    """Capture the current Relay causal parent for application-managed transport.

    Returns:
        PropagationContext: Context carrying the current parent and root scope
        identities for propagation to another execution boundary.
    """
    get_scope_stack()
    if parent_uuid := _propagation_parent_var.get():
        return PropagationContext(parent_uuid, _propagation_root_var.get())
    return _capture_propagation_context()


def capture_rootless_propagation_context() -> PropagationContext:
    """Capture the current causal parent without a propagation root.

    Returns:
        PropagationContext: Context with no root UUID, opting out of Relay
            session propagation when it is installed in another scope stack.
    """
    get_scope_stack()
    if parent_uuid := _propagation_parent_var.get():
        return PropagationContext(parent_uuid)
    return _capture_rootless_propagation_context()


def capture_propagation_context_with_root(root_uuid: str | None) -> PropagationContext:
    """Capture the current parent with an optional stable application session root.

    Args:
        root_uuid: Root identity to include in the propagated context. Pass
            ``None`` to use Relay's current root identity.

    Returns:
        PropagationContext: Context carrying the current parent and selected
        root identities.
    """
    get_scope_stack()
    if parent_uuid := _propagation_parent_var.get():
        return PropagationContext(parent_uuid, root_uuid)
    return _capture_propagation_context_with_root(root_uuid)


def capture_traceparent() -> str:
    """Capture the current Relay context as a W3C ``traceparent`` value.

    Returns:
        str: Encoded W3C traceparent value for the current Relay context.
    """
    get_scope_stack()
    parent_uuid = _propagation_parent_var.get()
    if parent_uuid:
        return PropagationContext(parent_uuid, _propagation_root_var.get() or parent_uuid).to_traceparent()
    return _capture_traceparent()


def create_scope_stack_from_propagation(context: PropagationContext) -> ScopeStack:
    """Create an isolated stack seeded from a received propagation context.

    Args:
        context: Parent and root context received from another execution
            boundary.

    Returns:
        ScopeStack: New isolated stack whose root retains the propagated
        causal relationship.
    """
    return _create_scope_stack_from_propagation(context)


def fork_asyncio_context() -> contextvars.Context:
    """Create an asyncio child context with an isolated Relay scope stack.

    Capture the current Relay parent, then install an isolated stack seeded
    from that parent into a copy of the current Python context. Use the
    returned context when creating a child task so concurrent tasks cannot
    mutate the same scope stack.

    Returns:
        contextvars.Context: A copy of the current context with an isolated
            Relay scope stack. Other context variable values are preserved.

    Notes:
        The child stack preserves event parentage but does not transfer
        scope-local middleware or subscribers. Call this before the child task
        starts; it cannot retrofit isolation onto an already-running task. Pass
        the returned context to ``asyncio.create_task(..., context=...)`` or
        ``asyncio.TaskGroup.create_task(..., context=...)``.

    Example::

        import asyncio

        import nemo_relay

        async def worker() -> None:
            with nemo_relay.scope.scope("worker", nemo_relay.ScopeType.Function):
                await asyncio.sleep(0)

        async def main() -> None:
            with nemo_relay.scope.scope("parent", nemo_relay.ScopeType.Agent):
                task = asyncio.create_task(
                    worker(),
                    context=nemo_relay.fork_asyncio_context(),
                )
                await task
    """
    propagation = capture_propagation_context()
    stack = create_scope_stack_from_propagation(propagation)
    child_context = contextvars.copy_context()
    child_context.run(_scope_stack_var.set, stack)
    return child_context


@contextmanager
def use_scope_stack(stack: ScopeStack) -> Iterator[ScopeStack]:
    """Temporarily install ``stack`` in the current Python context.

    Args:
        stack: Scope stack to make current for the body of the ``with`` block.

    Yields:
        ScopeStack: The installed scope stack.

    Returns:
        Iterator[ScopeStack]: Context-manager iterator that restores the prior
        context after the block exits.
    """
    current_stack = _scope_stack_var.get(None)
    if current_stack is not None:
        _sync_thread_scope_stack(current_stack)
    previous_native_stack = _capture_thread_scope_stack()
    token = _scope_stack_var.set(stack)
    _sync_thread_scope_stack(stack)
    try:
        root_uuid = _capture_traceparent().split("-")[1]
    except RuntimeError:
        root_uuid = None
    root_token = _propagation_root_var.set(root_uuid)
    try:
        yield stack
    finally:
        _propagation_root_var.reset(root_token)
        _scope_stack_var.reset(token)
        _restore_thread_scope_stack(previous_native_stack)


def set_thread_scope_stack(stack: ScopeStack) -> None:
    """Install a scope stack into the current thread's native runtime context.

    Args:
        stack: Scope stack that should become active for subsequent NeMo Relay
            API calls on the current thread.

    Returns:
        None: This function returns after the native thread-local slot has been
        updated.

    Raises:
        Exception: Propagates native errors raised when installing ``stack``.

    Behavior:
        The supplied stack is installed into the native thread-local slot for
        the current OS thread. The function does not create, clone, or validate
        a Python ``ContextVar`` entry.

    Notes:
        This helper is primarily used when propagating an existing logical
        trace into worker threads. It does not create or clone a scope stack;
        it installs the supplied stack reference for the current thread.

    Example::

        from concurrent.futures import ThreadPoolExecutor

        import nemo_relay

        with nemo_relay.scope.scope("parent", nemo_relay.ScopeType.Agent):
            stack = nemo_relay.propagate_scope_to_thread()

            def worker() -> None:
                nemo_relay.set_thread_scope_stack(stack)

            with ThreadPoolExecutor() as pool:
                pool.submit(worker).result()
    """
    _set_thread_scope_stack(stack)


#: Union of every lifecycle event emitted by the Python binding. Subscriber
#: callbacks receive this union and can branch on ``event.kind`` to distinguish
#: scope lifecycle events from point-in-time mark events.
Event = typing.Union[
    ScopeEvent,
    MarkEvent,
]

__all__ = [
    # Submodules
    "scope",
    "tools",
    "llm",
    "guardrails",
    "event_metadata",
    "intercepts",
    "subscribers",
    "scope_local",
    "codecs",
    "typed",
    "plugin",
    "runtime_registrations",
    "adaptive",
    "observability",
    "pii_redaction",
    "model_pricing",
    # Scope stack isolation
    "ScopeStack",
    "PropagationContext",
    "create_scope_stack",
    "capture_propagation_context",
    "capture_rootless_propagation_context",
    "capture_propagation_context_with_root",
    "capture_traceparent",
    "create_scope_stack_from_propagation",
    "fork_asyncio_context",
    "get_scope_stack",
    "scope_stack_active",
    "propagate_scope_to_thread",
    "set_thread_scope_stack",
    "use_scope_stack",
    # Types
    "ScopeAttributes",
    "ToolAttributes",
    "LLMAttributes",
    "LogSeverity",
    "MetricKind",
    "MetricValueType",
    "MetricMeasurement",
    "MetricTemporality",
    "DataSchema",
    "ScopeType",
    "ScopeEvent",
    "MarkEvent",
    "ScopeHandle",
    "ToolHandle",
    "ToolExecutionResult",
    "ToolExecutionInterceptOutcome",
    "LLMHandle",
    "LLMRequest",
    "LLMRequestInterceptOutcome",
    "PendingMarkSpec",
    "Event",
    "AnnotatedLLMRequest",
    "AnnotatedLLMResponse",
    "AtifExporter",
    "AtofStreamSinkConfig",
    "AtofExporterMode",
    "AtofExporterConfig",
    "AtofExporter",
    "OpenTelemetryConfig",
    "OpenTelemetrySubscriber",
    "OpenTelemetryRuntimeDiagnostic",
    "OpenTelemetryRuntimeDiagnostics",
    "OpenTelemetryLogConfig",
    "OpenTelemetryLogSubscriber",
    "OpenTelemetryMetricConfig",
    "OpenTelemetryMetricSubscriber",
    "JsonPrimitive",
    "JsonValue",
    "JsonObject",
    "Json",
    "UnsupportedBehavior",
    "EventSanitizeFields",
    "EventSanitizeGuardrail",
    "EventMetadataScalar",
    "EventMetadataValue",
    "EventMetadata",
    "EventMetadataInjectorCallback",
    "ToolSanitizeGuardrail",
    "ToolConditionalExecutionGuardrail",
    "LlmSanitizeRequestGuardrail",
    "LlmSanitizeResponseGuardrail",
    "LlmCodecIdentity",
    "LlmSanitizeRequestContext",
    "LlmSanitizeResponseContext",
    "LlmSanitizeRequestCodec",
    "LlmSanitizeResponseCodec",
    "LlmConditionalExecutionGuardrail",
    "ToolRequestIntercept",
    "ToolExecutionIntercept",
    "LlmRequestIntercept",
    "LlmExecutionIntercept",
    "LlmStreamExecutionIntercept",
]
