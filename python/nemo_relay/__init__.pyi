# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for the public ``nemo_relay`` package facade.

Summary:
    ``nemo_relay`` exposes the Python entry point for scope tracking, lifecycle
    events, middleware registration, typed helpers, plugins, adaptive
    configuration, and native observability types.

Description:
    The concrete implementations live in Python wrapper modules and in the
    compiled ``nemo_relay._native`` extension. This stub intentionally keeps
    native classes re-exported from ``_native.pyi`` so the native module remains
    the source of truth.

Exceptional flow:
    Stub declarations do not execute. Runtime exceptions are documented on the
    corresponding implementation or native declaration.
"""

from __future__ import annotations

import contextvars
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Literal, Optional, TypeAlias, TypedDict

from nemo_relay import adaptive as adaptive
from nemo_relay import codecs as codecs
from nemo_relay import event_metadata as event_metadata
from nemo_relay import guardrails as guardrails
from nemo_relay import intercepts as intercepts
from nemo_relay import llm as llm
from nemo_relay import model_pricing as model_pricing
from nemo_relay import observability as observability
from nemo_relay import pii_redaction as pii_redaction
from nemo_relay import plugin as plugin
from nemo_relay import scope as scope
from nemo_relay import scope_local as scope_local
from nemo_relay import subscribers as subscribers
from nemo_relay import tools as tools
from nemo_relay import typed as typed
from nemo_relay._native import (
    AnnotatedLLMRequest as AnnotatedLLMRequest,
)
from nemo_relay._native import (
    AnnotatedLLMResponse as AnnotatedLLMResponse,
)
from nemo_relay._native import (
    AtifExporter as AtifExporter,
)
from nemo_relay._native import (
    AtofExporter as AtofExporter,
)
from nemo_relay._native import (
    AtofExporterConfig as AtofExporterConfig,
)
from nemo_relay._native import (
    AtofExporterMode as AtofExporterMode,
)
from nemo_relay._native import (
    AtofStreamSinkConfig as AtofStreamSinkConfig,
)
from nemo_relay._native import (
    LLMAttributes as LLMAttributes,
)
from nemo_relay._native import (
    LlmCodecIdentity as LlmCodecIdentity,
)
from nemo_relay._native import (
    LLMHandle as LLMHandle,
)
from nemo_relay._native import (
    LLMRequest as LLMRequest,
)
from nemo_relay._native import (
    LLMRequestInterceptOutcome as LLMRequestInterceptOutcome,
)
from nemo_relay._native import (
    LlmSanitizeRequestCodec as LlmSanitizeRequestCodec,
)
from nemo_relay._native import (
    LlmSanitizeRequestContext as LlmSanitizeRequestContext,
)
from nemo_relay._native import (
    LlmSanitizeResponseCodec as LlmSanitizeResponseCodec,
)
from nemo_relay._native import (
    LlmSanitizeResponseContext as LlmSanitizeResponseContext,
)
from nemo_relay._native import (
    LogSeverity as LogSeverity,
)
from nemo_relay._native import (
    MarkEvent as MarkEvent,
)
from nemo_relay._native import (
    MetricKind as MetricKind,
)
from nemo_relay._native import (
    MetricMeasurement as MetricMeasurement,
)
from nemo_relay._native import (
    MetricTemporality as MetricTemporality,
)
from nemo_relay._native import (
    MetricValueType as MetricValueType,
)
from nemo_relay._native import (
    OpenTelemetryConfig as OpenTelemetryConfig,
)
from nemo_relay._native import (
    OpenTelemetryLogConfig as OpenTelemetryLogConfig,
)
from nemo_relay._native import (
    OpenTelemetryLogSubscriber as OpenTelemetryLogSubscriber,
)
from nemo_relay._native import (
    OpenTelemetryMetricConfig as OpenTelemetryMetricConfig,
)
from nemo_relay._native import (
    OpenTelemetryMetricSubscriber as OpenTelemetryMetricSubscriber,
)
from nemo_relay._native import (
    OpenTelemetryRuntimeDiagnostic as OpenTelemetryRuntimeDiagnostic,
)
from nemo_relay._native import (
    OpenTelemetryRuntimeDiagnostics as OpenTelemetryRuntimeDiagnostics,
)
from nemo_relay._native import (
    OpenTelemetrySubscriber as OpenTelemetrySubscriber,
)
from nemo_relay._native import (
    PendingMarkSpec as PendingMarkSpec,
)
from nemo_relay._native import (
    PropagationContext as PropagationContext,
)
from nemo_relay._native import (
    ScopeAttributes as ScopeAttributes,
)
from nemo_relay._native import (
    ScopeEvent as ScopeEvent,
)
from nemo_relay._native import (
    ScopeHandle as ScopeHandle,
)
from nemo_relay._native import (
    ScopeStack as ScopeStack,
)
from nemo_relay._native import (
    ScopeType as ScopeType,
)
from nemo_relay._native import (
    ToolAttributes as ToolAttributes,
)
from nemo_relay._native import (
    ToolExecutionInterceptOutcome as ToolExecutionInterceptOutcome,
)
from nemo_relay._native import (
    ToolExecutionResult as ToolExecutionResult,
)
from nemo_relay._native import (
    ToolHandle as ToolHandle,
)

JsonPrimitive: TypeAlias = str | int | float | bool | None
"""Scalar JSON leaf values accepted in NeMo Relay payloads.

Description:
    This alias documents primitive values that can appear inside public
    JSON-compatible payload arguments and return values.
"""
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
"""Recursive JSON-compatible value accepted by payload-carrying APIs.

Description:
    Lists and dictionaries may recursively contain more ``JsonValue`` objects.
    Runtime serialization or shape errors are raised by the API consuming the
    value.
"""
JsonObject: TypeAlias = dict[str, JsonValue]
"""Mapping-shaped JSON payload used by structured request and response helpers.

Description:
    Keys are strings and values are JSON-compatible. APIs may require specific
    keys for their own protocol shape.
"""
Json: TypeAlias = JsonValue
"""Shorthand for any JSON-compatible payload accepted by the Python binding."""
UnsupportedBehavior: TypeAlias = Literal["ignore", "warn", "error"]
"""Policy used by config helpers when unknown fields or values are encountered."""

class DataSchema(TypedDict):
    """Schema identifier attached to an event's opaque data payload."""

    name: str
    version: str

class EventSanitizeFields(TypedDict):
    """Observability fields returned by mark and scope event sanitizers."""

    data: Json | None
    category_profile: JsonObject | None
    metadata: Json | None

ToolSanitizeGuardrail: TypeAlias = Callable[[str, Json], Json | Awaitable[Json]]
"""Guardrail callback that sanitizes emitted tool request or response payloads.

Arguments:
    The tool name and current JSON payload.

Return:
    JSON payload recorded on the emitted lifecycle event.

Exceptional flow:
    Exceptions fail closed, omit the observability payload, and stop the
    remaining sanitizer chain.
"""
EventSanitizeGuardrail: TypeAlias = Callable[
    [Event, EventSanitizeFields],
    EventSanitizeFields | Awaitable[EventSanitizeFields],
]
"""Guardrail callback that sanitizes emitted mark or scope event fields.

Arguments:
    The immutable event snapshot and its mutable observability fields.

Return:
    Observability fields recorded on the asynchronously published event.

Exceptional flow:
    Exceptions fail closed, clear the mutable observability fields, and stop
    the remaining sanitizer chain.
"""
EventMetadataScalar: TypeAlias = str | int | float | bool
EventMetadataValue: TypeAlias = EventMetadataScalar | list[str] | list[int] | list[float] | list[bool]
EventMetadata: TypeAlias = dict[str, EventMetadataValue]
EventMetadataInjectorCallback: TypeAlias = Callable[
    [Event],
    EventMetadata | Awaitable[EventMetadata],
]
"""Additive middleware callback that returns flat event metadata additions."""
ToolConditionalExecutionGuardrail: TypeAlias = Callable[[str, Json], Optional[str] | Awaitable[Optional[str]]]
"""Guardrail callback that can block tool execution.

Arguments:
    The tool name and current JSON payload.

Return:
    ``None`` to allow execution, or a rejection message to block it.
"""
LlmSanitizeRequestGuardrail: TypeAlias = Callable[
    [LLMRequest, "LlmSanitizeRequestContext"],
    Optional[LLMRequest] | Awaitable[Optional[LLMRequest]],
]
"""Guardrail callback that sanitizes an ``LLMRequest`` used for emitted events.

Arguments:
    The current LLM request and a context object containing a tagged ``codec``
    identity. Its ``kind`` is ``none``, ``builtin``, ``runtime``, or ``opaque``;
    ``builtin`` and ``runtime`` identities include ``id``. Use
    ``context.resolve_codec()`` to access the active in-process codec.

Return:
    Request object recorded on the emitted lifecycle event, or ``None`` to omit
    the LLM observability payload and annotation.

Exceptional flow:
    Exceptions fail closed, omit the payload and annotation, and stop the
    remaining sanitizer chain.
"""
LlmSanitizeResponseGuardrail: TypeAlias = Callable[
    [Json, "LlmSanitizeResponseContext"],
    Optional[Json] | Awaitable[Optional[Json]],
]
"""Guardrail callback that sanitizes an emitted JSON LLM response payload.

Arguments:
    The response object and a context object containing a tagged ``codec``
    identity. Its ``kind`` is ``none``, ``builtin``, ``runtime``, or ``opaque``;
    ``builtin`` and ``runtime`` identities include ``id``. Use
    ``context.resolve_codec()`` to access the active in-process codec.

Return:
    Response object recorded on the emitted lifecycle event, or ``None`` to
    omit the LLM observability payload and annotation.

Exceptional flow:
    Exceptions fail closed, omit the payload and annotation, and stop the
    remaining sanitizer chain.
"""
LlmConditionalExecutionGuardrail: TypeAlias = Callable[[LLMRequest], Optional[str] | Awaitable[Optional[str]]]
"""Guardrail callback that can block an LLM call.

Arguments:
    The LLM request being evaluated.

Return:
    ``None`` to allow execution, or a rejection message to block it.
"""
ToolRequestIntercept: TypeAlias = Callable[[str, Json], Json | Awaitable[Json]]
"""Request intercept callback that rewrites tool arguments before execution.

Arguments:
    The tool name and current JSON payload.

Return:
    JSON payload passed to later request intercepts and tool execution.
"""
ToolExecutionIntercept: TypeAlias = Callable[
    [str, Json, Callable[[Json], Awaitable[ToolExecutionResult[Json]]]],
    ToolExecutionInterceptOutcome | Awaitable[ToolExecutionInterceptOutcome],
]
"""Execution intercept callback that wraps tool execution.

Arguments:
    The tool name, current JSON arguments, and next callable.

Return:
    A canonical tool execution outcome, either directly or as an awaitable.

Exceptional flow:
    The callback may short-circuit by not invoking ``next``. Exceptions
    propagate through the managed tool call.
"""
LlmRequestIntercept: TypeAlias = Callable[
    [str, LLMRequest, AnnotatedLLMRequest | None],
    LLMRequestInterceptOutcome | Awaitable[LLMRequestInterceptOutcome],
]
"""Request intercept callback that rewrites raw and annotated LLM requests.

Arguments:
    The logical LLM name, raw request, and optional annotated request view.

Return:
    The complete canonical outcome passed to later middleware.
"""
LlmExecutionIntercept: TypeAlias = Callable[
    [str, LLMRequest, Callable[[LLMRequest], Awaitable[Json]]],
    Json | Awaitable[Json],
]
"""Execution intercept callback that wraps non-streaming LLM execution.

Arguments:
    The logical LLM name, current request, and next callable.

Return:
    A JSON-compatible response, either directly or as an awaitable.
"""
LlmStreamExecutionIntercept: TypeAlias = Callable[
    [LLMRequest, Callable[[LLMRequest], Awaitable[AsyncIterator[Json]]]],
    AsyncIterator[Json] | Awaitable[AsyncIterator[Json]],
]
"""Execution intercept callback that wraps streaming LLM execution.

Arguments:
    The current request and next callable.

Return:
    An async iterator of JSON chunks, either directly or as an awaitable.
"""

Event: TypeAlias = ScopeEvent | MarkEvent
"""Union of every lifecycle event emitted by the Python binding.

Description:
    Subscribers receive this union and can inspect ``event.kind`` to
    distinguish scope lifecycle events from point-in-time mark events.
"""

_scope_stack_var: contextvars.ContextVar[ScopeStack]
_propagation_parent_var: contextvars.ContextVar[str | None]

def get_scope_stack() -> ScopeStack:
    """Return the current task's active scope stack, creating one if needed.

    Summary:
        Resolve the ``ScopeStack`` associated with the current Python context.

    Description:
        If the current context has no stack, the runtime creates one and then
        synchronizes it into native thread-local storage so later native calls
        observe the same hierarchy.

    Returns:
        The active ``ScopeStack`` for the current context.

    Exceptional flow:
        Native allocation or synchronization errors propagate unchanged.
    """
    ...

def scope_stack_active() -> bool:
    """Report whether the current context already owns a scope stack.

    Summary:
        Check for an explicitly active Python or native stack.

    Description:
        This is a status check only. It does not create a stack.

    Returns:
        ``True`` when a Python ``ContextVar`` stack exists or the native
        runtime reports an explicitly active thread-local stack.

    Exceptional flow:
        Native status-check errors propagate unchanged.
    """
    ...

def propagate_scope_to_thread() -> ScopeStack:
    """Capture the active scope stack for use in another thread.

    Summary:
        Return the current stack so worker threads can join the same trace.

    Description:
        The returned stack is shared, not cloned. Pass it to
        ``set_thread_scope_stack()`` inside the worker thread.

    Returns:
        The active ``ScopeStack``.

    Raises:
        RuntimeError: If no stack is active in the current context.
    """
    ...

def create_scope_stack() -> ScopeStack:
    """Create a new isolated scope stack.

    Summary:
        Allocate a fresh native scope stack.

    Description:
        The new stack is not installed into the Python context or native
        thread-local storage.

    Returns:
        A fresh ``ScopeStack`` handle.

    Exceptional flow:
        Native allocation errors propagate unchanged.
    """
    ...

def capture_propagation_context() -> PropagationContext: ...
def capture_rootless_propagation_context() -> PropagationContext: ...
def capture_propagation_context_with_root(root_uuid: str | None) -> PropagationContext: ...
def capture_traceparent() -> str: ...
def create_scope_stack_from_propagation(context: PropagationContext) -> ScopeStack: ...
def fork_asyncio_context() -> contextvars.Context:
    """Create a child asyncio context with an isolated Relay scope stack.

    The returned context preserves the current Relay causal parent and all
    unrelated context variables, but does not transfer scope-local middleware
    or subscribers. Pass it to ``asyncio.create_task(..., context=...)`` or
    ``asyncio.TaskGroup.create_task(..., context=...)`` before the child task
    starts.
    """
    ...

def use_scope_stack(stack: ScopeStack): ...
def set_thread_scope_stack(stack: ScopeStack) -> None:
    """Install a scope stack into the current thread's native runtime context.

    Args:
        stack: Scope stack to install for subsequent native runtime calls on
            the current OS thread.

    Returns:
        ``None``.

    Exceptional flow:
        Native installation errors propagate unchanged.
    """
    ...

def _native_scope_stack_active() -> bool: ...
def _sync_thread_scope_stack(stack: ScopeStack) -> None: ...

__all__: list[str]
