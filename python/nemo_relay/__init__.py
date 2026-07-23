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
- ``nemo_relay.scope_local`` for middleware scoped to a specific ``ScopeHandle``
- ``nemo_relay.typed`` for codec-based typed wrappers
- ``nemo_relay.plugin`` for global plugin configuration and custom plugin registration
- ``nemo_relay.adaptive`` for adaptive component configuration helpers
- ``nemo_relay.observability`` for observability component configuration helpers
- ``nemo_relay.pii_redaction`` for PII redaction component configuration helpers
- ``nemo_relay.model_pricing`` for model pricing component configuration helpers

Top-level exports also include:

- scope stack helpers such as ``get_scope_stack()``, ``create_scope_stack()``,
  ``set_thread_scope_stack()``, and ``scope_stack_active()``
- native runtime types such as ``ScopeHandle``, ``ToolHandle``, ``LLMHandle``,
  ``LLMRequest``, ``ScopeType``, and the lifecycle event classes
- observability helpers such as ``AtifExporter``, ``AtofExporter``,
  ``OpenTelemetrySubscriber``, and ``OpenInferenceSubscriber``
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
        return {"echo": args["query"]}

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

import contextvars
import typing
from collections.abc import Callable as AbcCallable
from typing import AsyncIterator, Awaitable, Callable, Literal, Optional, TypeAlias, TypedDict

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
    LLMHandle,
    LLMRequest,
    LLMRequestInterceptOutcome,
    MarkEvent,
    OpenInferenceConfig,
    OpenInferenceSubscriber,
    OpenTelemetryConfig,
    OpenTelemetrySubscriber,
    PendingMarkSpec,
    ScopeAttributes,
    ScopeEvent,
    ScopeHandle,
    ScopeStack,
    ScopeType,
    ToolAttributes,
    ToolExecutionInterceptOutcome,
    ToolHandle,
)
from nemo_relay._native import create_scope_stack as _create_scope_stack
from nemo_relay._native import scope_stack_active as _native_scope_stack_active
from nemo_relay._native import set_thread_scope_stack as _set_thread_scope_stack
from nemo_relay._native import sync_thread_scope_stack as _sync_thread_scope_stack

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


class EventSanitizeFields(TypedDict):
    """Observability fields returned by mark and scope event sanitizers."""

    data: Json | None
    category_profile: JsonObject | None
    metadata: Json | None


#: Guardrail callback that sanitizes emitted tool request or response payloads.
#: Arguments are the tool name and JSON payload. The return value is the JSON
#: payload recorded on the emitted event. Exceptions propagate through the
#: lifecycle call that invoked the guardrail.
ToolSanitizeGuardrail: TypeAlias = Callable[[str, Json], Json]
EventSanitizeGuardrail: TypeAlias = Callable[["Event", EventSanitizeFields], EventSanitizeFields]
#: Guardrail callback that can block tool execution by returning a rejection
#: message. Returning ``None`` allows execution to continue.
ToolConditionalExecutionGuardrail: TypeAlias = Callable[[str, Json], Optional[str]]
#: Guardrail callback that sanitizes an ``LLMRequest`` used for emitted events.
#: The returned request is recorded for observability and does not replace the
#: caller-visible request value unless the managed LLM API documents otherwise.
LlmSanitizeRequestGuardrail: TypeAlias = Callable[[LLMRequest], LLMRequest]
#: Guardrail callback that sanitizes an emitted JSON LLM response payload. The
#: returned object is recorded on the event; callback exceptions propagate.
LlmSanitizeResponseGuardrail: TypeAlias = Callable[[JsonObject], JsonObject]
#: Codec identity supplied to a contextual LLM sanitizer for a managed call.
LlmSanitizeContext: TypeAlias = JsonObject
#: Codec-aware sanitizer invoked as ``callback(request, context)``. Returning
#: ``None`` omits the event payload and annotation.
ContextualLlmSanitizeRequestGuardrail: TypeAlias = Callable[[LLMRequest, LlmSanitizeContext], Optional[LLMRequest]]
#: Codec-aware sanitizer invoked as ``callback(response, context)``. Returning
#: ``None`` omits the event payload and annotation.
ContextualLlmSanitizeResponseGuardrail: TypeAlias = Callable[[JsonObject, LlmSanitizeContext], Optional[JsonObject]]
#: Guardrail callback that can block an LLM call by returning a rejection
#: message. Returning ``None`` allows execution to continue.
LlmConditionalExecutionGuardrail: TypeAlias = Callable[[LLMRequest], Optional[str]]
#: Request intercept callback that rewrites tool arguments before execution.
#: Arguments are the tool name and current JSON payload. The return value
#: becomes the payload seen by later request intercepts and tool execution.
ToolRequestIntercept: TypeAlias = AbcCallable[[str, Json], Json]
#: Execution intercept callback that wraps tool execution with middleware
#: behavior. The callback receives the tool name, current arguments, and the
#: next callable. It may await and return ``next(args)`` or short-circuit.
ToolExecutionIntercept: TypeAlias = Callable[
    [str, Json, Callable[[Json], Awaitable[Json]]],
    ToolExecutionInterceptOutcome | Awaitable[ToolExecutionInterceptOutcome],
]
#: Request intercept callback that returns the canonical request, annotation,
#: and pending-mark outcome passed to later intercepts and managed execution.
LlmRequestIntercept: TypeAlias = Callable[
    [str, LLMRequest, AnnotatedLLMRequest | None],
    LLMRequestInterceptOutcome,
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
    guardrails,
    intercepts,
    llm,
    model_pricing,
    observability,
    pii_redaction,
    plugin,
    scope,
    scope_local,
    subscribers,
    tools,
    typed,
)

_scope_stack_var: contextvars.ContextVar[ScopeStack] = contextvars.ContextVar("scope_stack")


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
    "intercepts",
    "subscribers",
    "scope_local",
    "codecs",
    "typed",
    "plugin",
    "adaptive",
    "observability",
    "pii_redaction",
    "model_pricing",
    # Scope stack isolation
    "ScopeStack",
    "create_scope_stack",
    "get_scope_stack",
    "scope_stack_active",
    "propagate_scope_to_thread",
    "set_thread_scope_stack",
    # Types
    "ScopeAttributes",
    "ToolAttributes",
    "LLMAttributes",
    "ScopeType",
    "ScopeEvent",
    "MarkEvent",
    "ScopeHandle",
    "ToolHandle",
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
    "OpenInferenceConfig",
    "OpenInferenceSubscriber",
    "OpenTelemetryConfig",
    "OpenTelemetrySubscriber",
    "JsonPrimitive",
    "JsonValue",
    "JsonObject",
    "Json",
    "UnsupportedBehavior",
    "EventSanitizeFields",
    "EventSanitizeGuardrail",
    "ToolSanitizeGuardrail",
    "ToolConditionalExecutionGuardrail",
    "LlmSanitizeRequestGuardrail",
    "LlmSanitizeResponseGuardrail",
    "LlmSanitizeContext",
    "ContextualLlmSanitizeRequestGuardrail",
    "ContextualLlmSanitizeResponseGuardrail",
    "LlmConditionalExecutionGuardrail",
    "ToolRequestIntercept",
    "ToolExecutionIntercept",
    "LlmRequestIntercept",
    "LlmExecutionIntercept",
    "LlmStreamExecutionIntercept",
]
