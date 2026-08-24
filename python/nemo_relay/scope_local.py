# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope-local middleware and subscriber registration.

These helpers mirror the global ``guardrails``, ``intercepts``, and
``subscribers`` modules, but the registrations apply only while the owning
scope is active. When that scope is popped, the registrations are removed
automatically.

Example::

    import nemo_relay

    def redact(tool_name, args):
        return {**args, "api_key": "***"}

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as handle:
        nemo_relay.scope_local.register_tool_sanitize_request(handle, "redact", 10, redact)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nemo_relay import (
    EventMetadataInjectorCallback,
    EventSanitizeGuardrail,
    LlmConditionalExecutionGuardrail,
    LlmExecutionIntercept,
    LlmRequestIntercept,
    LlmSanitizeRequestGuardrail,
    LlmSanitizeResponseGuardrail,
    LlmStreamExecutionIntercept,
    ScopeHandle,
    ToolConditionalExecutionGuardrail,
    ToolExecutionIntercept,
    ToolRequestIntercept,
    ToolSanitizeGuardrail,
)
from nemo_relay._native import (
    scope_deregister_event_metadata_injector as _deregister_event_metadata_injector,
)

if TYPE_CHECKING:
    from nemo_relay import Event
from nemo_relay._native import (
    scope_deregister_llm_conditional_execution_guardrail as _deregister_llm_conditional_execution,
)
from nemo_relay._native import (
    scope_deregister_llm_execution_intercept as _deregister_llm_execution,
)
from nemo_relay._native import (
    scope_deregister_llm_request_intercept as _deregister_llm_request,
)
from nemo_relay._native import (
    scope_deregister_llm_sanitize_request_guardrail as _deregister_llm_sanitize_request,
)
from nemo_relay._native import (
    scope_deregister_llm_sanitize_response_guardrail as _deregister_llm_sanitize_response,
)
from nemo_relay._native import (
    scope_deregister_llm_stream_execution_intercept as _deregister_llm_stream_execution,
)
from nemo_relay._native import (
    scope_deregister_mark_sanitize_guardrail as _deregister_mark_sanitize,
)
from nemo_relay._native import (
    scope_deregister_scope_sanitize_end_guardrail as _deregister_scope_sanitize_end,
)
from nemo_relay._native import (
    scope_deregister_scope_sanitize_start_guardrail as _deregister_scope_sanitize_start,
)
from nemo_relay._native import (
    scope_deregister_subscriber as _deregister_subscriber,
)
from nemo_relay._native import (
    scope_deregister_tool_conditional_execution_guardrail as _deregister_tool_conditional_execution,
)
from nemo_relay._native import (
    scope_deregister_tool_execution_intercept as _deregister_tool_execution,
)
from nemo_relay._native import (
    scope_deregister_tool_request_intercept as _deregister_tool_request,
)
from nemo_relay._native import (
    scope_deregister_tool_sanitize_request_guardrail as _deregister_tool_sanitize_request,
)
from nemo_relay._native import (
    scope_deregister_tool_sanitize_response_guardrail as _deregister_tool_sanitize_response,
)
from nemo_relay._native import (
    scope_register_event_metadata_injector as _register_event_metadata_injector,
)
from nemo_relay._native import (
    scope_register_llm_conditional_execution_guardrail as _register_llm_conditional_execution,
)
from nemo_relay._native import (
    scope_register_llm_execution_intercept as _register_llm_execution,
)
from nemo_relay._native import (
    scope_register_llm_request_intercept as _register_llm_request,
)
from nemo_relay._native import (
    scope_register_llm_sanitize_request_guardrail as _register_llm_sanitize_request,
)
from nemo_relay._native import (
    scope_register_llm_sanitize_response_guardrail as _register_llm_sanitize_response,
)
from nemo_relay._native import (
    scope_register_llm_stream_execution_intercept as _register_llm_stream_execution,
)
from nemo_relay._native import (
    scope_register_mark_sanitize_guardrail as _register_mark_sanitize,
)
from nemo_relay._native import (
    scope_register_scope_sanitize_end_guardrail as _register_scope_sanitize_end,
)
from nemo_relay._native import (
    scope_register_scope_sanitize_start_guardrail as _register_scope_sanitize_start,
)
from nemo_relay._native import (
    scope_register_subscriber as _register_subscriber,
)
from nemo_relay._native import (
    scope_register_tool_conditional_execution_guardrail as _register_tool_conditional_execution,
)
from nemo_relay._native import (
    scope_register_tool_execution_intercept as _register_tool_execution,
)
from nemo_relay._native import (
    scope_register_tool_request_intercept as _register_tool_request,
)
from nemo_relay._native import (
    scope_register_tool_sanitize_request_guardrail as _register_tool_sanitize_request,
)
from nemo_relay._native import (
    scope_register_tool_sanitize_response_guardrail as _register_tool_sanitize_response,
)

# ---------------------------------------------------------------------------
# Mark and scope event guardrails (scope-local)
# ---------------------------------------------------------------------------


def register_event_metadata_injector(
    scope_handle: ScopeHandle,
    name: str,
    priority: int,
    injector: EventMetadataInjectorCallback,
) -> None:
    """Register an event metadata injector owned by a scope.

    Args:
        scope_handle: Scope that owns the registration.
        name: Unique registration name within the scope.
        priority: Execution order; lower values run first.
        injector: Callback that returns metadata additions for emitted events.

    Returns:
        None: The injector remains active until removal or scope closure.
    """
    _register_event_metadata_injector(scope_handle.uuid, name, priority, injector)


def deregister_event_metadata_injector(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local event metadata injector.

    Args:
        scope_handle: Scope that owns the registration.
        name: Registration name to remove.

    Returns:
        bool: Whether an injector was removed.
    """
    return _deregister_event_metadata_injector(scope_handle.uuid, name)


def register_mark_sanitize(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: EventSanitizeGuardrail
) -> None:
    """Register a mark-event sanitizer owned by a scope.

    Args:
        scope_handle: Scope that owns the registration.
        name: Unique registration name within the scope.
        priority: Execution order; lower values run first.
        guardrail: Callback that sanitizes emitted mark-event fields.

    Returns:
        None: The sanitizer remains active until removal or scope closure.
    """
    return _register_mark_sanitize(scope_handle.uuid, name, priority, guardrail)


def deregister_mark_sanitize(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local mark-event sanitizer.

    Args:
        scope_handle: Scope that owns the registration.
        name: Registration name to remove.

    Returns:
        bool: Whether a sanitizer was removed.
    """
    return _deregister_mark_sanitize(scope_handle.uuid, name)


def register_scope_sanitize_start(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: EventSanitizeGuardrail
) -> None:
    """Register a scope-start event sanitizer owned by a scope.

    Args:
        scope_handle: Scope that owns the registration.
        name: Unique registration name within the scope.
        priority: Execution order; lower values run first.
        guardrail: Callback that sanitizes emitted scope-start event fields.

    Returns:
        None: The sanitizer remains active until removal or scope closure.
    """
    return _register_scope_sanitize_start(scope_handle.uuid, name, priority, guardrail)


def deregister_scope_sanitize_start(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local scope-start sanitizer.

    Args:
        scope_handle: Scope that owns the registration.
        name: Registration name to remove.

    Returns:
        bool: Whether a sanitizer was removed.
    """
    return _deregister_scope_sanitize_start(scope_handle.uuid, name)


def register_scope_sanitize_end(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: EventSanitizeGuardrail
) -> None:
    """Register a scope-end event sanitizer owned by a scope.

    Args:
        scope_handle: Scope that owns the registration.
        name: Unique registration name within the scope.
        priority: Execution order; lower values run first.
        guardrail: Callback that sanitizes emitted scope-end event fields.

    Returns:
        None: The sanitizer remains active until removal or scope closure.
    """
    return _register_scope_sanitize_end(scope_handle.uuid, name, priority, guardrail)


def deregister_scope_sanitize_end(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local scope-end sanitizer.

    Args:
        scope_handle: Scope that owns the registration.
        name: Registration name to remove.

    Returns:
        bool: Whether a sanitizer was removed.
    """
    return _deregister_scope_sanitize_end(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# Tool guardrails (scope-local)
# ---------------------------------------------------------------------------


def register_tool_sanitize_request(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: ToolSanitizeGuardrail
) -> None:
    """Register a scope-local tool sanitize-request guardrail.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique guardrail name within the owning scope.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(tool_name, args)`` that
            returns the sanitized payload recorded on emitted start events.

    Returns:
        None: This function returns after the scope-local guardrail is
        registered.

    Notes:
        As with the global variant, this sanitizes emitted event payloads only.
    """
    return _register_tool_sanitize_request(scope_handle.uuid, name, priority, guardrail)


def deregister_tool_sanitize_request(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local tool sanitize-request guardrail.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Guardrail name previously passed to
            ``register_tool_sanitize_request()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.

    Notes:
        Removing the guardrail early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_tool_sanitize_request(scope_handle.uuid, name)


def register_tool_sanitize_response(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: ToolSanitizeGuardrail
) -> None:
    """Register a scope-local tool sanitize-response guardrail.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique guardrail name within the owning scope.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(tool_name, result)`` that
            returns the sanitized payload recorded on emitted end events.

    Returns:
        None: This function returns after the scope-local guardrail is
        registered.

    Notes:
        As with the global variant, this sanitizes emitted event payloads only.
    """
    return _register_tool_sanitize_response(scope_handle.uuid, name, priority, guardrail)


def deregister_tool_sanitize_response(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local tool sanitize-response guardrail.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Guardrail name previously passed to
            ``register_tool_sanitize_response()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.

    Notes:
        Removing the guardrail early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_tool_sanitize_response(scope_handle.uuid, name)


def register_tool_conditional_execution(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: ToolConditionalExecutionGuardrail
) -> None:
    """Register a scope-local tool conditional-execution guardrail.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique guardrail name within the owning scope.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(tool_name, args)``. Return
            ``None`` to allow execution or a rejection message to block it.

    Returns:
        None: This function returns after the scope-local guardrail is
        registered.

    Notes:
        Scope-local conditional guardrails run in addition to global
        conditional guardrails for calls emitted under the owning scope.
    """
    return _register_tool_conditional_execution(scope_handle.uuid, name, priority, guardrail)


def deregister_tool_conditional_execution(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local tool conditional-execution guardrail.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Guardrail name previously passed to
            ``register_tool_conditional_execution()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.

    Notes:
        Removing the guardrail early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_tool_conditional_execution(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# Tool intercepts (scope-local)
# ---------------------------------------------------------------------------


def register_tool_request(
    scope_handle: ScopeHandle, name: str, priority: int, break_chain: bool, fn: ToolRequestIntercept
) -> None:
    """Register a scope-local tool request intercept.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique intercept name within the owning scope.
        priority: Execution order for the intercept. Lower values run first.
        break_chain: Whether to stop applying lower-priority request intercepts
            after this intercept runs.
        fn: Callable invoked as ``fn(tool_name, args)`` that returns the
            rewritten tool arguments.

    Returns:
        None: This function returns after the scope-local intercept is
        registered.

    Notes:
        Scope-local request intercepts are merged with global intercepts using
        the same priority ordering rules.
    """
    return _register_tool_request(scope_handle.uuid, name, priority, break_chain, fn)


def deregister_tool_request(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local tool request intercept.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Intercept name previously passed to ``register_tool_request()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removing the intercept early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_tool_request(scope_handle.uuid, name)


def register_tool_execution(scope_handle: ScopeHandle, name: str, priority: int, fn: ToolExecutionIntercept) -> None:
    """Register scope-local middleware around tool execution.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique intercept name within the owning scope.
        priority: Execution order for the intercept. Lower values run first.
        fn: Callable invoked as ``fn(tool_name, args, next_call)``. It may call
            ``next_call(args)`` to continue execution, modify the result, or
            short-circuit the tool call entirely. It must return
            ``ToolExecutionInterceptOutcome``.

    Returns:
        None: This function returns after the scope-local intercept is
        registered.

    Notes:
        Execution intercepts wrap only calls emitted while the owning scope
        remains active. ``next_call`` may be awaited repeatedly or concurrently
        while ``fn`` is running. Each call gets an isolated snapshot of the
        scope stack visible when it begins. Unfinished or new calls are rejected
        after ``fn`` returns or raises.
    """
    return _register_tool_execution(scope_handle.uuid, name, priority, fn)


def deregister_tool_execution(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local tool execution intercept.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Intercept name previously passed to
            ``register_tool_execution()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removing the intercept early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_tool_execution(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# LLM guardrails (scope-local)
# ---------------------------------------------------------------------------


def register_llm_sanitize_request(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: LlmSanitizeRequestGuardrail
) -> None:
    """Register a scope-local LLM sanitize-request guardrail.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique guardrail name within the owning scope.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(request)`` that returns the
            sanitized request recorded on emitted start events.

    Returns:
        None: This function returns after the scope-local guardrail is
        registered.

    Notes:
        As with the global variant, this sanitizes emitted event payloads only.
    """
    return _register_llm_sanitize_request(scope_handle.uuid, name, priority, guardrail)


def deregister_llm_sanitize_request(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local LLM sanitize-request guardrail.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Guardrail name previously passed to
            ``register_llm_sanitize_request()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.

    Notes:
        Removing the guardrail early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_llm_sanitize_request(scope_handle.uuid, name)


def register_llm_sanitize_response(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: LlmSanitizeResponseGuardrail
) -> None:
    """Register a scope-local LLM sanitize-response guardrail.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique guardrail name within the owning scope.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(response)`` that returns the
            sanitized payload recorded on emitted end events.

    Returns:
        None: This function returns after the scope-local guardrail is
        registered.

    Notes:
        As with the global variant, this sanitizes emitted event payloads only.
    """
    return _register_llm_sanitize_response(scope_handle.uuid, name, priority, guardrail)


def deregister_llm_sanitize_response(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local LLM sanitize-response guardrail.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Guardrail name previously passed to
            ``register_llm_sanitize_response()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.

    Notes:
        Removing the guardrail early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_llm_sanitize_response(scope_handle.uuid, name)


def register_llm_conditional_execution(
    scope_handle: ScopeHandle, name: str, priority: int, guardrail: LlmConditionalExecutionGuardrail
) -> None:
    """Register a scope-local LLM conditional-execution guardrail.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique guardrail name within the owning scope.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(request)``. Return ``None``
            to allow execution or a rejection message to block it.

    Returns:
        None: This function returns after the scope-local guardrail is
        registered.

    Notes:
        Scope-local conditional guardrails run in addition to global
        conditional guardrails for calls emitted under the owning scope.
    """
    return _register_llm_conditional_execution(scope_handle.uuid, name, priority, guardrail)


def deregister_llm_conditional_execution(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local LLM conditional-execution guardrail.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Guardrail name previously passed to
            ``register_llm_conditional_execution()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.

    Notes:
        Removing the guardrail early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_llm_conditional_execution(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# LLM intercepts (scope-local)
# ---------------------------------------------------------------------------


def register_llm_request(
    scope_handle: ScopeHandle, name: str, priority: int, break_chain: bool, fn: LlmRequestIntercept
) -> None:
    """Register a scope-local LLM request intercept.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique intercept name within the owning scope.
        priority: Execution order for the intercept. Lower values run first.
        break_chain: Whether to stop applying lower-priority request intercepts
            after this intercept runs.
        fn: Callable invoked as ``fn(name, request, annotated)`` that returns
            an ``LLMRequestInterceptOutcome`` or an awaitable of that outcome
            for the next stage.

    Returns:
        None: This function returns after the scope-local intercept is
        registered.

    Notes:
        Scope-local request intercepts are merged with global intercepts using
        the same priority ordering rules.
    """
    return _register_llm_request(scope_handle.uuid, name, priority, break_chain, fn)


def deregister_llm_request(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local LLM request intercept.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Intercept name previously passed to ``register_llm_request()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removing the intercept early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_llm_request(scope_handle.uuid, name)


def register_llm_execution(scope_handle: ScopeHandle, name: str, priority: int, fn: LlmExecutionIntercept) -> None:
    """Register scope-local middleware around non-streaming LLM execution.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique intercept name within the owning scope.
        priority: Execution order for the intercept. Lower values run first.
        fn: Callable invoked as ``fn(name, request, next_call)`` that may call
            ``next_call(request)`` to continue execution or short-circuit it.

    Returns:
        None: This function returns after the scope-local intercept is
        registered.

    Notes:
        Execution intercepts wrap only calls emitted while the owning scope
        remains active. ``next_call`` may be awaited repeatedly or concurrently
        while ``fn`` is running. Each call gets an isolated snapshot of the
        scope stack visible when it begins. Unfinished or new calls are rejected
        after ``fn`` returns or raises.
    """
    return _register_llm_execution(scope_handle.uuid, name, priority, fn)


def deregister_llm_execution(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local LLM execution intercept.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Intercept name previously passed to
            ``register_llm_execution()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removing the intercept early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_llm_execution(scope_handle.uuid, name)


def register_llm_stream_execution(
    scope_handle: ScopeHandle, name: str, priority: int, fn: LlmStreamExecutionIntercept
) -> None:
    """Register scope-local middleware around streaming LLM execution.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique intercept name within the owning scope.
        priority: Execution order for the intercept. Lower values run first.
        fn: Callable invoked as ``fn(request, next_call)`` that returns an
            async iterator of chunks, either by delegating to ``next_call`` or
            by replacing the stream entirely.

    Returns:
        None: This function returns after the scope-local intercept is
        registered.

    Notes:
        Streaming execution intercepts wrap chunk production only. They do not
        replace the collector or finalizer callbacks. ``next_call`` may be
        awaited repeatedly or concurrently, and each call gets an isolated
        snapshot of its visible scope stack. The returned interceptor stream
        keeps ``next_call`` active until it closes; unfinished or later calls
        are then rejected. A stream returned successfully by ``next_call``
        keeps its normal streaming lifetime.
    """
    return _register_llm_stream_execution(scope_handle.uuid, name, priority, fn)


def deregister_llm_stream_execution(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local streaming LLM execution intercept.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Intercept name previously passed to
            ``register_llm_stream_execution()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removing the intercept early affects only future work in the owning
        scope. Popping the scope would also remove it automatically.
    """
    return _deregister_llm_stream_execution(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# Subscribers (scope-local)
# ---------------------------------------------------------------------------


def register_subscriber(scope_handle: ScopeHandle, name: str, callback: "Callable[[Event], None]") -> None:
    """Register an event subscriber that is active only for ``scope_handle``.

    Args:
        scope_handle: Owning scope handle. The registration is removed when
            this scope is popped.
        name: Unique subscriber name within the owning scope.
        callback: Callable invoked as ``callback(event)`` for each lifecycle
            event emitted while the scope remains active.

    Returns:
        None: This function returns after the scope-local subscriber is
        registered.

    Notes:
        The subscriber observes only events emitted while the owning scope
        remains active.

    Example::

        import nemo_relay

        def log_event(event):
            print(event.kind, event.name)

        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as handle:
            nemo_relay.scope_local.register_subscriber(handle, "logger", log_event)
    """
    return _register_subscriber(scope_handle.uuid, name, callback)


def deregister_subscriber(scope_handle: ScopeHandle, name: str) -> bool:
    """Remove a scope-local event subscriber.

    Args:
        scope_handle: Scope handle that owns the registration.
        name: Subscriber name previously passed to ``register_subscriber()``.

    Returns:
        bool: ``True`` if a subscriber was removed, otherwise ``False``.

    Notes:
        Removing the subscriber early affects only future event delivery in the
        owning scope. Popping the scope would also remove it automatically.
    """
    return _deregister_subscriber(scope_handle.uuid, name)


__all__ = [
    # Injector registrations
    "register_event_metadata_injector",
    "deregister_event_metadata_injector",
    # Mark and scope event guardrails
    "register_mark_sanitize",
    "deregister_mark_sanitize",
    "register_scope_sanitize_start",
    "deregister_scope_sanitize_start",
    "register_scope_sanitize_end",
    "deregister_scope_sanitize_end",
    # Tool guardrails
    "register_tool_sanitize_request",
    "deregister_tool_sanitize_request",
    "register_tool_sanitize_response",
    "deregister_tool_sanitize_response",
    "register_tool_conditional_execution",
    "deregister_tool_conditional_execution",
    # Tool intercepts
    "register_tool_request",
    "deregister_tool_request",
    "register_tool_execution",
    "deregister_tool_execution",
    # LLM guardrails
    "register_llm_sanitize_request",
    "deregister_llm_sanitize_request",
    "register_llm_sanitize_response",
    "deregister_llm_sanitize_response",
    "register_llm_conditional_execution",
    "deregister_llm_conditional_execution",
    # LLM intercepts
    "register_llm_request",
    "deregister_llm_request",
    "register_llm_execution",
    "deregister_llm_execution",
    "register_llm_stream_execution",
    "deregister_llm_stream_execution",
    # Subscribers
    "register_subscriber",
    "deregister_subscriber",
]
