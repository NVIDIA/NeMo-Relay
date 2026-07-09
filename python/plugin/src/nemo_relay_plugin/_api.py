# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level Python API for NeMo Relay ``grpc-v1`` worker plugins.

The module exposes the authoring contract for out-of-process Python plugins.
Callbacks can be synchronous or asynchronous unless a method documents a more
specific return type. Synchronous callbacks run on the worker event-loop thread
and must not block. Asynchronous callbacks overlap only when they yield control
at an ``await``. Callback exceptions are returned to the Relay host as
structured worker errors.

Public data types:
    Json: Any JSON-serializable Python value.
    Event: A Relay event represented as a JSON object.
    LlmRequest: A Relay LLM request represented as a JSON object.
    AnnotatedLlmRequest: An annotated Relay LLM request represented as a JSON
        object.
    PendingMarkSpec: A mark Relay emits under its managed lifecycle scope.
    LlmRequestInterceptOutcome: Canonical LLM request-intercept result.
    DiagnosticLevel: Severity of a configuration diagnostic.
    ConfigDiagnostic: Structured configuration warning or error.
    ScopeType: Semantic category for a Relay execution scope.
    WorkerSdkError: SDK, host-call, or worker protocol error.

Public authoring types:
    WorkerPlugin: Base validation and registration contract for a plugin.
    PluginContext: Component-scoped callback registration context.
    PluginRuntime: Host runtime handle for event and scope operations.
    ToolNext: Continuation for a tool execution intercept.
    LlmNext: Continuation for a unary LLM execution intercept.
    LlmStreamNext: Continuation for a streaming LLM execution intercept.

Public callback aliases used in registration annotations:
    SubscriberCallback: Event subscriber callback.
    ToolSanitizeCallback: Tool request or response sanitizer callback.
    ToolConditionalCallback: Tool execution guardrail callback.
    ToolRequestCallback: Tool request intercept callback.
    ToolExecutionCallback: Tool execution intercept callback.
    LlmSanitizeRequestCallback: LLM request sanitizer callback.
    LlmSanitizeResponseCallback: LLM response sanitizer callback.
    LlmConditionalCallback: LLM execution guardrail callback.
    LlmRequestCallback: LLM request intercept callback.
    LlmExecutionCallback: Unary LLM execution intercept callback.
    LlmStreamExecutionCallback: Streaming LLM execution intercept callback.

Public functions:
    serve_plugin: Run a local ``grpc-v1`` worker until host shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import hmac
import importlib
import inspect
import ipaddress
import json
import os
import platform
import stat
import tempfile
import tomllib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict
from urllib.parse import urlsplit

grpc: Any = importlib.import_module("grpc")
pb: Any = importlib.import_module("._proto.plugin_worker_pb2", __package__)
pb_grpc: Any = importlib.import_module("._proto.plugin_worker_pb2_grpc", __package__)

#: Any JSON-serializable Python value accepted by the worker protocol.
Json: TypeAlias = Any
#: A Relay event represented as a JSON object.
Event: TypeAlias = dict[str, Any]


class EventSanitizeFields(TypedDict):
    """Observability fields returned by event sanitizer callbacks."""

    data: Json | None
    category_profile: dict[str, Any] | None
    metadata: Json | None


#: A Relay LLM request represented as a JSON object.
LlmRequest: TypeAlias = dict[str, Any]
#: An annotated Relay LLM request represented as a JSON object.
AnnotatedLlmRequest: TypeAlias = dict[str, Any]

WORKER_PROTOCOL = "grpc-v1"
JSON_SCHEMA = "nemo.relay.Json@1"
EVENT_SCHEMA = "nemo.relay.Event@1"
LLM_REQUEST_SCHEMA = "nemo.relay.LlmRequest@1"
ANNOTATED_LLM_REQUEST_SCHEMA = "nemo.relay.AnnotatedLlmRequest@1"
LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA = "nemo.relay.LlmRequestInterceptOutcome@1"
TOOL_EXECUTION_INTERCEPT_OUTCOME_SCHEMA = "nemo.relay.ToolExecutionInterceptOutcome@1"
PLUGIN_DIAGNOSTICS_SCHEMA = "nemo.relay.PluginDiagnostics@1"
_OBJECT_SCHEMAS = frozenset(
    {
        EVENT_SCHEMA,
        LLM_REQUEST_SCHEMA,
        ANNOTATED_LLM_REQUEST_SCHEMA,
        LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA,
        TOOL_EXECUTION_INTERCEPT_OUTCOME_SCHEMA,
    }
)
_UNREGISTERED = object()
_SCOPE_CONTEXT: contextvars.ContextVar[_BoundScopeContext | None] = contextvars.ContextVar(
    "nemo_relay_plugin_scope_context",
    default=None,
)


class WorkerSdkError(Exception):
    """Report a worker SDK, host-call, or protocol error to plugin code.

    The SDK raises this exception when a host runtime call fails, a host
    response is malformed, a required worker setting is missing, or a local
    operation violates the worker protocol.
    """


def _sdk_version() -> str:
    try:
        return metadata.version("nemo-relay-plugin")
    except metadata.PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        try:
            project = tomllib.loads(pyproject_path.read_text(encoding="utf-8")).get("project")
        except (OSError, tomllib.TOMLDecodeError):
            return "0+unknown"
        if isinstance(project, dict):
            version = project.get("version")
            if isinstance(version, str) and version:
                return version
        return "0+unknown"


_SDK_VERSION = _sdk_version()


class DiagnosticLevel(str, Enum):
    """Identify the severity of a plugin configuration diagnostic.

    Attributes:
        WARNING: Report a non-blocking configuration problem.
        ERROR: Report a configuration problem that blocks activation.
    """

    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class ConfigDiagnostic:
    """Describe one problem found while validating plugin configuration.

    Args:
        level: Diagnostic severity. Use :class:`DiagnosticLevel` or the
            equivalent ``"warning"`` or ``"error"`` string.
        code: Stable, plugin-defined machine-readable diagnostic code.
        message: Human-readable explanation of the problem.
        component: Optional component or plugin identifier associated with the
            problem.
        field: Optional configuration field associated with the problem.
    """

    level: DiagnosticLevel | str
    code: str
    message: str
    component: str | None = None
    field: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Convert the diagnostic to its Relay JSON representation.

        Returns:
            A JSON object containing required fields and any populated optional
            fields. Enum severity values are converted to strings.

        Raises:
            WorkerSdkError: The severity or another diagnostic field is
                invalid.
        """
        return _normalize_diagnostic(asdict(self))


@dataclass(slots=True)
class PendingMarkSpec:
    """Describe a mark Relay emits under a managed lifecycle scope."""

    name: str
    category: str | None = None
    category_profile: Json | None = None
    data: Json | None = None
    metadata: Json | None = None

    def to_json(self) -> dict[str, Json]:
        """Convert this pending mark to its canonical JSON object."""
        if not isinstance(self.name, str):
            raise WorkerSdkError("pending mark name must be a string")
        return asdict(self)


@dataclass(slots=True)
class LlmRequestInterceptOutcome:
    """Canonical result returned by a Python worker LLM request intercept."""

    request: LlmRequest
    annotated_request: AnnotatedLlmRequest | None = None
    pending_marks: list[PendingMarkSpec] = field(default_factory=list)

    def to_json(self) -> dict[str, Json]:
        """Convert this outcome to the canonical worker-envelope payload."""
        if not isinstance(self.request, dict):
            raise WorkerSdkError("LLM request intercept outcome request must be a JSON object")
        if self.annotated_request is not None and not isinstance(self.annotated_request, dict):
            raise WorkerSdkError("LLM request intercept outcome annotated_request must be a JSON object or None")
        marks = []
        for mark in self.pending_marks:
            if not isinstance(mark, PendingMarkSpec):
                raise WorkerSdkError("LLM request intercept outcome pending_marks must contain PendingMarkSpec values")
            marks.append(mark.to_json())
        return {
            "request": self.request,
            "annotated_request": self.annotated_request,
            "pending_marks": marks,
        }


@dataclass(slots=True)
class ToolExecutionInterceptOutcome:
    """Canonical result returned by a Python worker tool execution intercept."""

    result: Json
    pending_marks: list[PendingMarkSpec] = field(default_factory=list)

    def to_json(self) -> dict[str, Json]:
        """Convert this outcome to the canonical worker-envelope payload."""
        marks = []
        for mark in self.pending_marks:
            if not isinstance(mark, PendingMarkSpec):
                raise WorkerSdkError(
                    "tool execution intercept outcome pending_marks must contain PendingMarkSpec values"
                )
            marks.append(mark.to_json())
        return {
            "result": self.result,
            "pending_marks": marks,
        }


def _normalize_diagnostic(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        level = DiagnosticLevel(value.get("level")).value
    except (TypeError, ValueError) as exc:
        raise WorkerSdkError("diagnostic level must be 'warning' or 'error'") from exc

    code = value.get("code")
    if not isinstance(code, str):
        raise WorkerSdkError("diagnostic code must be a string")
    message = value.get("message")
    if not isinstance(message, str):
        raise WorkerSdkError("diagnostic message must be a string")

    normalized: dict[str, Any] = {
        "level": level,
        "code": code,
        "message": message,
    }
    for field_name in ("component", "field"):
        field_value = value.get(field_name)
        if field_value is None:
            continue
        if not isinstance(field_value, str):
            raise WorkerSdkError(f"diagnostic {field_name} must be a string")
        normalized[field_name] = field_value
    return normalized


class ScopeType(str, Enum):
    """Identify the semantic category of a Relay execution scope.

    Attributes:
        AGENT: Agent execution.
        FUNCTION: General function execution.
        TOOL: Tool execution.
        LLM: LLM execution.
        RETRIEVER: Retrieval execution.
        EMBEDDER: Embedding execution.
        RERANKER: Reranking execution.
        GUARDRAIL: Guardrail execution.
        EVALUATOR: Evaluation execution.
        CUSTOM: Plugin-defined execution.
        UNKNOWN: Execution with no known semantic category.
    """

    AGENT = "agent"
    FUNCTION = "function"
    TOOL = "tool"
    LLM = "llm"
    RETRIEVER = "retriever"
    EMBEDDER = "embedder"
    RERANKER = "reranker"
    GUARDRAIL = "guardrail"
    EVALUATOR = "evaluator"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class _BoundScopeContext:
    scope_stack_id: str
    parent_scope_id: str | None = None

    def __post_init__(self) -> None:
        if not self.scope_stack_id:
            raise WorkerSdkError("scope_stack_id must not be empty")
        if self.parent_scope_id == "":
            raise WorkerSdkError("parent_scope_id must not be empty")


class WorkerPlugin:
    """Define the validation and registration contract for a worker plugin.

    Subclass ``WorkerPlugin``, set :attr:`plugin_id`, and implement
    :meth:`register`. The Relay host calls :meth:`validate` before registration
    for each enabled component. Implementations may define either lifecycle
    method synchronously or asynchronously.

    Attributes:
        plugin_id: Stable plugin identifier. It must match ``plugin.id`` in
            ``relay-plugin.toml`` and must not be empty.
        allows_multiple_components: Whether Relay can install this activation's
            registered behavior plan for more than one namespaced component.
            One worker activation still has one configuration and one
            successful ``Register`` call.
    """

    plugin_id: str = ""
    allows_multiple_components: bool = False

    def validate(
        self,
        config: Json,
    ) -> list[ConfigDiagnostic | dict[str, Any]] | Awaitable[list[ConfigDiagnostic | dict[str, Any]]]:
        """Validate one component configuration before registration.

        Args:
            config: JSON configuration from the dynamic plugin record.

        Returns:
            Diagnostics describing configuration warnings or errors. Return an
            empty list when the configuration is valid. Error diagnostics block
            activation. Asynchronous implementations return the diagnostics
            from their coroutine.

        Raises:
            Exception: Any exception becomes a structured validation failure
                returned to the Relay host.
        """
        del config
        return []

    def register(self, ctx: PluginContext, config: Json) -> None | Awaitable[None]:
        """Register plugin callbacks for one component configuration.

        Args:
            ctx: Component-scoped context used to register subscribers,
                guardrails, and intercepts and to access the host runtime.
            config: JSON configuration from the dynamic plugin record.

        Raises:
            Exception: Any exception aborts registration. The Relay host rolls
                back registrations from the failed activation.

        Note:
            Asynchronous implementations may await setup before registering
            callbacks on ``ctx``.
        """
        del ctx, config
        raise NotImplementedError("WorkerPlugin.register must be implemented")


class _SupportsWorkerPlugin(Protocol):
    plugin_id: str
    allows_multiple_components: bool

    def validate(
        self,
        config: Json,
    ) -> list[ConfigDiagnostic | dict[str, Any]] | Awaitable[list[ConfigDiagnostic | dict[str, Any]]]: ...

    def register(self, ctx: PluginContext, config: Json) -> None | Awaitable[None]: ...


SubscriberCallback: TypeAlias = Callable[[Event], None | Awaitable[None]]
EventSanitizeCallback: TypeAlias = Callable[
    [Event, EventSanitizeFields],
    EventSanitizeFields | Awaitable[EventSanitizeFields],
]
ToolSanitizeCallback: TypeAlias = Callable[[str, Json], Json | Awaitable[Json]]
ToolConditionalCallback: TypeAlias = Callable[[str, Json], str | None | Awaitable[str | None]]
ToolRequestCallback: TypeAlias = Callable[[str, Json], Json | Awaitable[Json]]
ToolExecutionCallback: TypeAlias = Callable[
    [str, Json, "ToolNext"],
    ToolExecutionInterceptOutcome | Awaitable[ToolExecutionInterceptOutcome],
]
LlmSanitizeRequestCallback: TypeAlias = Callable[[LlmRequest], LlmRequest | Awaitable[LlmRequest]]
LlmSanitizeResponseCallback: TypeAlias = Callable[[Json], Json | Awaitable[Json]]
LlmConditionalCallback: TypeAlias = Callable[[LlmRequest], str | None | Awaitable[str | None]]
LlmRequestCallback: TypeAlias = Callable[
    [str, LlmRequest, AnnotatedLlmRequest | None],
    LlmRequestInterceptOutcome | Awaitable[LlmRequestInterceptOutcome],
]
LlmExecutionCallback: TypeAlias = Callable[[str, LlmRequest, "LlmNext"], Json | Awaitable[Json]]
LlmStreamExecutionCallback: TypeAlias = Callable[
    [str, LlmRequest, "LlmStreamNext"],
    Iterable[Json] | AsyncIterator[Json] | Awaitable[Iterable[Json] | AsyncIterator[Json]],
]


@dataclass(slots=True)
class _Handlers:
    registrations: list[Any]
    subscribers: dict[str, SubscriberCallback]
    mark_sanitizers: dict[str, EventSanitizeCallback]
    scope_start_sanitizers: dict[str, EventSanitizeCallback]
    scope_end_sanitizers: dict[str, EventSanitizeCallback]
    tool_sanitize_requests: dict[str, ToolSanitizeCallback]
    tool_sanitize_responses: dict[str, ToolSanitizeCallback]
    tool_conditionals: dict[str, ToolConditionalCallback]
    tool_requests: dict[str, ToolRequestCallback]
    tool_executions: dict[str, ToolExecutionCallback]
    llm_sanitize_requests: dict[str, LlmSanitizeRequestCallback]
    llm_sanitize_responses: dict[str, LlmSanitizeResponseCallback]
    llm_conditionals: dict[str, LlmConditionalCallback]
    llm_requests: dict[str, LlmRequestCallback]
    llm_executions: dict[str, LlmExecutionCallback]
    llm_stream_executions: dict[str, LlmStreamExecutionCallback]

    @classmethod
    def empty(cls) -> _Handlers:
        return cls(
            registrations=[],
            subscribers={},
            mark_sanitizers={},
            scope_start_sanitizers={},
            scope_end_sanitizers={},
            tool_sanitize_requests={},
            tool_sanitize_responses={},
            tool_conditionals={},
            tool_requests={},
            tool_executions={},
            llm_sanitize_requests={},
            llm_sanitize_responses={},
            llm_conditionals={},
            llm_requests={},
            llm_executions={},
            llm_stream_executions={},
        )


class PluginContext:
    """Register component-scoped callbacks and access the Relay host runtime.

    Registrations form a declarative activation plan installed and executed by
    the Relay host. Scope-local middleware lifetime, merged priority ordering,
    and ``break_chain`` behavior belong to the host runtime; pushing a scope
    through :class:`PluginRuntime` does not make later registrations scope-local.

    The SDK creates this context and passes it to :meth:`WorkerPlugin.register`.
    Plugin authors normally do not construct it directly. Registration names
    need to be unique only within a registration surface; the Relay host
    qualifies them with the plugin component identity.

    Args:
        runtime: Host runtime handle. ``None`` is supported for constructing a
            context without host access, primarily for tests.
    """

    def __init__(self, runtime: PluginRuntime | None = None) -> None:
        self._runtime = runtime
        self._handlers = _Handlers.empty()

    @property
    def runtime(self) -> PluginRuntime:
        """Return the host runtime handle for event and scope operations.

        Returns:
            The runtime associated with this plugin activation.

        Raises:
            WorkerSdkError: The context was created without a runtime handle.
        """
        if self._runtime is None:
            raise WorkerSdkError("PluginContext has no runtime handle")
        return self._runtime

    def register_subscriber(self, name: str, callback: SubscriberCallback) -> None:
        """Register a callback that receives Relay events.

        Args:
            name: Component-local registration name.
            callback: Function called with one :data:`Event`. It can return
                ``None`` or an awaitable that resolves to ``None``.

        Callback errors:
            An exception becomes a structured worker invocation error.
        """
        self._push_registration(name, pb.SUBSCRIBER, 0, False)
        self._handlers.subscribers[name] = callback

    def _register_event_sanitizer(
        self,
        name: str,
        callback: EventSanitizeCallback,
        surface: int,
        priority: int,
        handlers: dict[str, EventSanitizeCallback],
    ) -> None:
        self._push_registration(name, surface, priority, False)
        handlers[name] = callback

    def register_mark_sanitize_guardrail(
        self, name: str, callback: EventSanitizeCallback, *, priority: int = 0
    ) -> None:
        """Register a sanitizer for mark event observability fields.

        Args:
            name: Component-local registration name.
            callback: Function receiving the event and its observability fields.
                It can return sanitized fields directly or through an awaitable.
            priority: Execution order. Lower values run first.

        Callback errors:
            An exception becomes a structured worker invocation error.
        """
        self._register_event_sanitizer(
            name,
            callback,
            pb.MARK_SANITIZE_GUARDRAIL,
            priority,
            self._handlers.mark_sanitizers,
        )

    def register_scope_sanitize_start_guardrail(
        self, name: str, callback: EventSanitizeCallback, *, priority: int = 0
    ) -> None:
        """Register a sanitizer for scope start event observability fields.

        Args:
            name: Component-local registration name.
            callback: Function receiving the event and its observability fields.
                It can return sanitized fields directly or through an awaitable.
            priority: Execution order. Lower values run first.

        Callback errors:
            An exception becomes a structured worker invocation error.
        """
        self._register_event_sanitizer(
            name,
            callback,
            pb.SCOPE_SANITIZE_START_GUARDRAIL,
            priority,
            self._handlers.scope_start_sanitizers,
        )

    def register_scope_sanitize_end_guardrail(
        self, name: str, callback: EventSanitizeCallback, *, priority: int = 0
    ) -> None:
        """Register a sanitizer for scope end event observability fields.

        Args:
            name: Component-local registration name.
            callback: Function receiving the event and its observability fields.
                It can return sanitized fields directly or through an awaitable.
            priority: Execution order. Lower values run first.

        Callback errors:
            An exception becomes a structured worker invocation error.
        """
        self._register_event_sanitizer(
            name,
            callback,
            pb.SCOPE_SANITIZE_END_GUARDRAIL,
            priority,
            self._handlers.scope_end_sanitizers,
        )

    def register_tool_sanitize_request_guardrail(
        self,
        name: str,
        callback: ToolSanitizeCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a guardrail that sanitizes tool input for observability.

        Args:
            name: Component-local registration name.
            callback: Function receiving ``(tool_name, arguments)`` and
                returning the JSON recorded on the tool start event. The
                callback can return the value directly or through an awaitable.
                It does not change the arguments passed to the real tool.
            priority: Execution order. Lower values run first.
        """
        self._push_registration(name, pb.TOOL_SANITIZE_REQUEST_GUARDRAIL, priority, False)
        self._handlers.tool_sanitize_requests[name] = callback

    def register_tool_sanitize_response_guardrail(
        self,
        name: str,
        callback: ToolSanitizeCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a guardrail that sanitizes tool output for observability.

        Args:
            name: Component-local registration name.
            callback: Function receiving ``(tool_name, result)`` and returning
                the JSON recorded on the tool end event. The callback can
                return the value directly or through an awaitable. It does not
                change the real tool result.
            priority: Execution order. Lower values run first.
        """
        self._push_registration(name, pb.TOOL_SANITIZE_RESPONSE_GUARDRAIL, priority, False)
        self._handlers.tool_sanitize_responses[name] = callback

    def register_tool_conditional_execution_guardrail(
        self,
        name: str,
        callback: ToolConditionalCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a guardrail that can block tool execution.

        Args:
            name: Component-local registration name.
            callback: Function receiving ``(tool_name, arguments)``. Return
                ``None`` to allow execution or a string explaining why Relay
                must block it. The callback can return an awaitable.
            priority: Execution order. Lower values run first.
        """
        self._push_registration(name, pb.TOOL_CONDITIONAL_EXECUTION_GUARDRAIL, priority, False)
        self._handlers.tool_conditionals[name] = callback

    def register_tool_request_intercept(
        self,
        name: str,
        callback: ToolRequestCallback,
        *,
        priority: int = 0,
        break_chain: bool = False,
    ) -> None:
        """Register an intercept that rewrites tool arguments.

        Args:
            name: Component-local registration name.
            callback: Function receiving ``(tool_name, arguments)`` and
                returning replacement JSON arguments, directly or through an
                awaitable.
            priority: Execution order. Lower values run first.
            break_chain: Whether Relay skips later, lower-priority request
                intercepts after this callback runs.
        """
        self._push_registration(name, pb.TOOL_REQUEST_INTERCEPT, priority, break_chain)
        self._handlers.tool_requests[name] = callback

    def register_tool_execution_intercept(
        self,
        name: str,
        callback: ToolExecutionCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register middleware around real tool execution.

        Args:
            name: Component-local registration name.
            callback: Function receiving ``(tool_name, arguments, next_call)``
                and returning :class:`ToolExecutionInterceptOutcome`, directly
                or through an awaitable. It can call :meth:`ToolNext.call`
                zero, one, or multiple times while the invocation is active.
            priority: Execution order. Lower values run first.
        """
        self._push_registration(name, pb.TOOL_EXECUTION_INTERCEPT, priority, False)
        self._handlers.tool_executions[name] = callback

    def register_llm_sanitize_request_guardrail(
        self,
        name: str,
        callback: LlmSanitizeRequestCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a guardrail that sanitizes an LLM request for observability.

        Args:
            name: Component-local registration name.
            callback: Function receiving an :data:`LlmRequest` and returning
                the request recorded on the LLM start event, directly or
                through an awaitable. It does not change the request sent to
                the model.
            priority: Execution order. Lower values run first.
        """
        self._push_registration(name, pb.LLM_SANITIZE_REQUEST_GUARDRAIL, priority, False)
        self._handlers.llm_sanitize_requests[name] = callback

    def register_llm_sanitize_response_guardrail(
        self,
        name: str,
        callback: LlmSanitizeResponseCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a guardrail that sanitizes an LLM response for observability.

        Args:
            name: Component-local registration name.
            callback: Function receiving response JSON and returning the value
                recorded on the LLM end event, directly or through an
                awaitable. It does not change the real model response.
            priority: Execution order. Lower values run first.
        """
        self._push_registration(name, pb.LLM_SANITIZE_RESPONSE_GUARDRAIL, priority, False)
        self._handlers.llm_sanitize_responses[name] = callback

    def register_llm_conditional_execution_guardrail(
        self,
        name: str,
        callback: LlmConditionalCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a guardrail that can block LLM execution.

        Args:
            name: Component-local registration name.
            callback: Function receiving an :data:`LlmRequest`. Return ``None``
                to allow execution or a string explaining why Relay must block
                it. The callback can return an awaitable.
            priority: Execution order. Lower values run first.
        """
        self._push_registration(name, pb.LLM_CONDITIONAL_EXECUTION_GUARDRAIL, priority, False)
        self._handlers.llm_conditionals[name] = callback

    def register_llm_request_intercept(
        self,
        name: str,
        callback: LlmRequestCallback,
        *,
        priority: int = 0,
        break_chain: bool = False,
    ) -> None:
        """Register an intercept that rewrites an LLM request.

        Args:
            name: Component-local registration name.
            callback: Function receiving ``(model_name, request,
                annotated_request)``. Return
                :class:`LlmRequestInterceptOutcome`, directly or through an
                awaitable. ``annotated_request`` is ``None`` when the host did
                not provide one. When present, it is authoritative for provider
                body content and the raw request content must remain unchanged.
            priority: Execution order. Lower values run first.
            break_chain: Whether Relay skips later, lower-priority request
                intercepts after this callback runs.
        """
        self._push_registration(name, pb.LLM_REQUEST_INTERCEPT, priority, break_chain)
        self._handlers.llm_requests[name] = callback

    def register_llm_execution_intercept(
        self,
        name: str,
        callback: LlmExecutionCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register middleware around real LLM execution.

        Args:
            name: Component-local registration name.
            callback: Function receiving ``(model_name, request, next_call)``
                and returning response JSON, directly or through an awaitable.
                It can call :meth:`LlmNext.call` zero, one, or multiple times
                while the invocation is active.
            priority: Execution order. Lower values run first.
        """
        self._push_registration(name, pb.LLM_EXECUTION_INTERCEPT, priority, False)
        self._handlers.llm_executions[name] = callback

    def register_llm_stream_execution_intercept(
        self,
        name: str,
        callback: LlmStreamExecutionCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register middleware around real streaming LLM execution.

        Args:
            name: Component-local registration name.
            callback: Function receiving ``(model_name, request, next_call)``.
                Return an iterable, an async iterator, or an awaitable resolving
                to either. Every yielded item must be JSON. Strings, byte
                sequences, mappings, and scalar values are not valid streams.
                The callback can call :meth:`LlmStreamNext.call` zero, one, or
                multiple times while the invocation is active.
            priority: Execution order. Lower values run first.

        Streaming behavior:
            Relay forwards chunks in callback yield order. Callback and item
            errors terminate the replacement stream with a structured worker
            error.
        """
        self._push_registration(name, pb.LLM_STREAM_EXECUTION_INTERCEPT, priority, False)
        self._handlers.llm_stream_executions[name] = callback

    def _push_registration(self, name: str, surface: int, priority: int, break_chain: bool) -> None:
        if any(
            registration.local_name == name and registration.surface == surface
            for registration in self._handlers.registrations
        ):
            raise WorkerSdkError(f"handler {name!r} is already registered for surface {surface}")
        self._handlers.registrations.append(
            pb.Registration(
                local_name=name,
                surface=surface,
                priority=priority,
                break_chain=break_chain,
            )
        )


class PluginRuntime:
    """Call event, scope, and continuation operations on the Relay host.

    The SDK creates one runtime per worker activation. Plugin authors access it
    through :attr:`PluginContext.runtime` instead of constructing it directly.
    Scope bindings use :mod:`contextvars`, so concurrent asyncio tasks retain
    independent bindings.

    Args:
        activation_id: Opaque activation identifier supplied by the Relay host.
        auth_token: Activation token attached to every host runtime request.
        host_stub: ``RelayHostRuntime`` gRPC client used for host calls.

    Scope selection:
        An explicit ``scope_stack_id`` on a host call takes precedence over the
        locally bound stack. If only ``parent_scope_id`` is explicit, it
        overrides the parent on the bound stack. Passing a parent without an
        explicit or bound stack raises :class:`WorkerSdkError`.
    """

    def __init__(self, *, activation_id: str, auth_token: str, host_stub: Any) -> None:
        self._activation_id = activation_id
        self._auth_token = auth_token
        self._host_stub = host_stub

    async def emit_mark(
        self,
        name: str,
        data: Json | None = None,
        metadata: Json | None = None,
        *,
        scope_stack_id: str | None = None,
        parent_scope_id: str | None = None,
    ) -> None:
        """Emit a mark event through the Relay host runtime.

        Args:
            name: Mark event name.
            data: Optional JSON application payload.
            metadata: Optional JSON metadata attached to the event.
            scope_stack_id: Optional host-issued stack to correlate the event
                with. When omitted, the current local binding is used.
            parent_scope_id: Optional parent scope for the event. When omitted,
                the parent from the selected stack binding is used.

        Raises:
            WorkerSdkError: The scope selection is invalid or the host rejects
                the request.
            TypeError: A payload is not JSON-serializable.
        """
        response = await self._host_stub.EmitMark(
            pb.EmitMarkRequest(
                activation_id=self._activation_id,
                auth_token=self._auth_token,
                scope=self._scope_context(scope_stack_id, parent_scope_id),
                name=name,
                data=_optional_json_envelope(data),
                metadata=_optional_json_envelope(metadata),
            )
        )
        _ack_to_result(response)

    async def create_scope_stack(self) -> str:
        """Create an isolated, host-owned scope stack.

        Returns:
            An opaque stack identifier accepted by scope and event operations.

        Raises:
            WorkerSdkError: The host cannot create the stack.
        """
        response = await self._host_stub.CreateScopeStack(
            pb.CreateScopeStackRequest(
                activation_id=self._activation_id,
                auth_token=self._auth_token,
            )
        )
        if response.HasField("error"):
            raise _worker_error_to_sdk(response.error)
        return response.scope_stack_id

    async def drop_scope_stack(self, scope_stack_id: str) -> None:
        """Drop an isolated, host-owned scope stack.

        Args:
            scope_stack_id: Opaque identifier returned by
                :meth:`create_scope_stack`.

        Raises:
            WorkerSdkError: The host rejects the request.

        Note:
            Do not use the identifier after this call succeeds. Clear any local
            binding that still references it.
        """
        response = await self._host_stub.DropScopeStack(
            pb.DropScopeStackRequest(
                activation_id=self._activation_id,
                auth_token=self._auth_token,
                scope_stack_id=scope_stack_id,
            )
        )
        _ack_to_result(response)

    async def push_scope(
        self,
        name: str,
        *,
        scope_type: ScopeType = ScopeType.CUSTOM,
        data: Json | None = None,
        metadata: Json | None = None,
        input: Json | None = None,
        scope_stack_id: str | None = None,
        parent_scope_id: str | None = None,
    ) -> str:
        """Start a scope on a Relay host-owned stack.

        Args:
            name: Human-readable scope name.
            scope_type: Semantic category for the scope.
            data: Optional JSON application payload associated with the scope.
            metadata: Optional JSON metadata attached to the scope start event.
            input: Optional JSON semantic input attached to the start event.
            scope_stack_id: Optional host-issued target stack. When omitted,
                the current local binding is used.
            parent_scope_id: Optional parent scope. When omitted, the parent
                from the selected stack binding is used.

        Returns:
            An opaque scope handle to pass to :meth:`pop_scope`.

        Raises:
            WorkerSdkError: The scope selection is invalid or the host rejects
                the request.
            TypeError: A payload is not JSON-serializable.
            ValueError: ``scope_type`` is not a supported :class:`ScopeType`.
        """
        response = await self._host_stub.PushScope(
            pb.PushScopeRequest(
                activation_id=self._activation_id,
                auth_token=self._auth_token,
                scope=self._scope_context(scope_stack_id, parent_scope_id),
                name=name,
                scope_type=_proto_scope_type(scope_type),
                data=_optional_json_envelope(data),
                metadata=_optional_json_envelope(metadata),
                input=_optional_json_envelope(input),
            )
        )
        if response.HasField("error"):
            raise _worker_error_to_sdk(response.error)
        return response.scope_handle_id

    async def pop_scope(
        self,
        scope_handle_id: str,
        *,
        output: Json | None = None,
        metadata: Json | None = None,
    ) -> None:
        """End a host scope by its handle identifier.

        Args:
            scope_handle_id: Opaque handle returned by :meth:`push_scope`.
            output: Optional JSON semantic output attached to the scope end
                event.
            metadata: Optional JSON metadata attached to the end event.

        Raises:
            WorkerSdkError: The host rejects the request.
            TypeError: A payload is not JSON-serializable.
        """
        response = await self._host_stub.PopScope(
            pb.PopScopeRequest(
                activation_id=self._activation_id,
                auth_token=self._auth_token,
                scope_handle_id=scope_handle_id,
                output=_optional_json_envelope(output),
                metadata=_optional_json_envelope(metadata),
            )
        )
        _ack_to_result(response)

    @contextlib.contextmanager
    def bind_scope_stack(self, scope_stack_id: str | None, *, parent_scope_id: str | None = None) -> Iterator[None]:
        """Temporarily bind host calls to a worker-selected scope stack.

        Args:
            scope_stack_id: Host-issued stack identifier. Pass ``None`` to
                temporarily clear correlation.
            parent_scope_id: Optional parent scope used by mark and scope-start
                calls that do not provide an explicit parent.

        Yields:
            ``None`` while the binding is active.

        Note:
            The previous binding is restored on exit, including when the body
            raises. The binding is local to the current contextvars context.
        """
        if scope_stack_id is None:
            if parent_scope_id is not None:
                raise WorkerSdkError("parent_scope_id requires a scope_stack_id")
            scope = None
        else:
            scope = _BoundScopeContext(scope_stack_id, parent_scope_id)
        token = _SCOPE_CONTEXT.set(scope)
        try:
            yield
        finally:
            _SCOPE_CONTEXT.reset(token)

    @contextlib.contextmanager
    def clear_scope_stack(self) -> Iterator[None]:
        """Temporarily clear worker scope-stack correlation.

        Yields:
            ``None`` while no local scope stack is bound.

        Note:
            The previous binding is restored on exit.
        """
        with self.bind_scope_stack(None):
            yield

    def current_scope_stack_id(self) -> str | None:
        """Return the locally bound scope stack identifier.

        Returns:
            The current opaque stack identifier, or ``None`` when no stack is
            bound.
        """
        scope = _SCOPE_CONTEXT.get()
        return scope.scope_stack_id if scope else None

    def current_parent_scope_id(self) -> str | None:
        """Return the locally bound parent scope identifier.

        Returns:
            The current parent scope identifier, or ``None`` when the binding
            does not define one.
        """
        scope = _SCOPE_CONTEXT.get()
        return scope.parent_scope_id if scope else None

    def _scope_context(self, scope_stack_id: str | None = None, parent_scope_id: str | None = None) -> Any:
        if parent_scope_id == "":
            raise WorkerSdkError("parent_scope_id must not be empty")
        if scope_stack_id is not None:
            effective_scope = _BoundScopeContext(scope_stack_id, parent_scope_id)
        else:
            bound_scope = _SCOPE_CONTEXT.get()
            if bound_scope is not None:
                effective_scope = _BoundScopeContext(
                    bound_scope.scope_stack_id,
                    parent_scope_id if parent_scope_id is not None else bound_scope.parent_scope_id,
                )
            elif parent_scope_id is not None:
                raise WorkerSdkError("parent_scope_id requires an explicit or bound scope stack")
            else:
                effective_scope = None
        if not effective_scope:
            return None
        return pb.ScopeContext(
            scope_stack_id=effective_scope.scope_stack_id,
            parent_scope_id=effective_scope.parent_scope_id or "",
        )


class ToolNext:
    """Continue the remaining tool execution chain.

    The SDK creates this handle for a tool execution-intercept invocation. It
    remains valid only while that invocation is active.

    Args:
        runtime: Runtime used to call the Relay host.
        continuation_id: Opaque host-issued continuation identifier.
    """

    def __init__(self, runtime: PluginRuntime, continuation_id: str) -> None:
        self._runtime = runtime
        self._continuation_id = continuation_id

    async def call(self, value: Json) -> Json:
        """Call the remaining tool execution chain with replacement arguments.

        Args:
            value: JSON arguments passed to the next intercept or real tool.

        Returns:
            JSON returned by the remaining chain.

        Raises:
            WorkerSdkError: The continuation is invalid, complete, cancelled,
                or fails in the host.
            TypeError: ``value`` is not JSON-serializable.

        Note:
            An intercept can call this method zero, one, or multiple times while
            its invocation is active.
        """
        response = await self._runtime._host_stub.ToolNext(
            pb.ToolNextRequest(
                activation_id=self._runtime._activation_id,
                auth_token=self._runtime._auth_token,
                continuation_id=self._continuation_id,
                value=_json_envelope(JSON_SCHEMA, value),
            )
        )
        return _json_result_to_value(response)


class LlmNext:
    """Continue the remaining unary LLM execution chain.

    The SDK creates this handle for an LLM execution-intercept invocation. It
    remains valid only while that invocation is active.

    Args:
        runtime: Runtime used to call the Relay host.
        continuation_id: Opaque host-issued continuation identifier.
    """

    def __init__(self, runtime: PluginRuntime, continuation_id: str) -> None:
        self._runtime = runtime
        self._continuation_id = continuation_id

    async def call(self, request: LlmRequest) -> Json:
        """Call the remaining LLM execution chain with a replacement request.

        Args:
            request: LLM request passed to the next intercept or real model.

        Returns:
            JSON response returned by the remaining chain.

        Raises:
            WorkerSdkError: The continuation is invalid, complete, cancelled,
                or fails in the host.
            TypeError: ``request`` is not JSON-serializable.

        Note:
            An intercept can call this method zero, one, or multiple times while
            its invocation is active.
        """
        response = await self._runtime._host_stub.LlmNext(
            pb.LlmNextRequest(
                activation_id=self._runtime._activation_id,
                auth_token=self._runtime._auth_token,
                continuation_id=self._continuation_id,
                request=_json_envelope(LLM_REQUEST_SCHEMA, request),
            )
        )
        return _json_result_to_value(response)


class LlmStreamNext:
    """Continue the remaining streaming LLM execution chain.

    The SDK creates this handle for an LLM stream execution-intercept
    invocation. It remains valid only while that invocation is active.

    Args:
        runtime: Runtime used to call the Relay host.
        continuation_id: Opaque host-issued continuation identifier.
    """

    def __init__(self, runtime: PluginRuntime, continuation_id: str) -> None:
        self._runtime = runtime
        self._continuation_id = continuation_id

    def call(self, request: LlmRequest) -> AsyncIterator[Json]:
        """Call the remaining LLM stream chain with a replacement request.

        Args:
            request: LLM request passed to the next intercept or real model.

        Returns:
            An async iterator of ordered JSON chunks. The iterator preserves
            the scope context active when this method was called.

        Raises:
            WorkerSdkError: Iteration encounters a host stream error, an empty
                chunk, or an invalid continuation.
            TypeError: ``request`` is not JSON-serializable.

        Note:
            An intercept can call this method zero, one, or multiple times while
            its invocation is active. Consume each returned iterator before the
            invocation completes.
        """
        scope_context = _SCOPE_CONTEXT.get()
        stream = self._runtime._host_stub.LlmStreamNext(
            pb.LlmStreamNextRequest(
                activation_id=self._runtime._activation_id,
                auth_token=self._runtime._auth_token,
                continuation_id=self._continuation_id,
                request=_json_envelope(LLM_REQUEST_SCHEMA, request),
            )
        )

        async def values() -> AsyncIterator[Json]:
            token = _SCOPE_CONTEXT.set(scope_context)
            try:
                async for chunk in stream:
                    yield _stream_chunk_to_value(chunk)
            finally:
                _SCOPE_CONTEXT.reset(token)

        return values()


async def serve_plugin(plugin: _SupportsWorkerPlugin) -> None:
    """Run a local ``grpc-v1`` worker until the Relay host shuts it down.

    Args:
        plugin: Plugin implementation with a non-empty ``plugin_id`` plus
            ``validate`` and ``register`` methods. Subclassing
            :class:`WorkerPlugin` is the standard authoring path.

    Required environment variables:
        ``NEMO_RELAY_WORKER_SOCKET``: Worker listen endpoint. Use
        ``unix:///path/to/socket`` on Unix or ``tcp://host:port`` and
        ``http://host:port`` for loopback TCP. Port ``0`` requests an ephemeral
        port.
        ``NEMO_RELAY_HOST_SOCKET``: Relay host runtime endpoint in the same
        endpoint formats.
        ``NEMO_RELAY_WORKER_ID``: Opaque activation identifier.
        ``NEMO_RELAY_WORKER_TOKEN``: Activation authentication token.

    Optional environment variables:
        ``NEMO_RELAY_WORKER_ENDPOINT_FILE``: File where the SDK writes the
        resolved worker endpoint after the gRPC server is accepting requests.

    Callback concurrency:
        The gRPC AsyncIO server can keep multiple RPCs in flight. Asynchronous
        callbacks overlap only when they yield control at an ``await``.
        Synchronous callbacks and synchronous stream iterators run on the
        worker event-loop thread. Blocking I/O, ``time.sleep``, or long-running
        CPU work in those callbacks stalls all worker RPCs. Wrap blocking work
        in an asynchronous callback and offload it with
        :func:`asyncio.to_thread` or another appropriate executor.

        The SDK does not configure ``maximum_concurrent_rpcs``, so gRPC does
        not enforce an application-level RPC admission limit.

    Raises:
        WorkerSdkError: A required environment variable is empty or missing, or
            the worker endpoint cannot be bound.
        OSError: Unix socket cleanup or endpoint-file creation fails.
        grpc.RpcError: gRPC server or host-channel startup fails.
        asyncio.CancelledError: The task running the worker is cancelled.

    Note:
        The function closes the gRPC server and host channel during shutdown or
        cancellation. It returns only after the host accepts a shutdown request.
    """
    _plugin_id(plugin)
    worker_endpoint = _required_env("NEMO_RELAY_WORKER_SOCKET")
    host_endpoint = _required_env("NEMO_RELAY_HOST_SOCKET")
    activation_id = _required_env("NEMO_RELAY_WORKER_ID")
    auth_token = _required_env("NEMO_RELAY_WORKER_TOKEN")
    endpoint_file = os.environ.get("NEMO_RELAY_WORKER_ENDPOINT_FILE")

    worker_target = _grpc_target(worker_endpoint)
    await _unlink_unix_socket(worker_endpoint)
    host_channel = _open_host_channel(host_endpoint)
    runtime = PluginRuntime(
        activation_id=activation_id,
        auth_token=auth_token,
        host_stub=pb_grpc.RelayHostRuntimeStub(host_channel),
    )
    shutdown_event = asyncio.Event()
    service = _WorkerService(plugin, runtime, shutdown_event)
    server = grpc.aio.server()
    try:
        pb_grpc.add_PluginWorkerServicer_to_server(service, server)
        bound_port = server.add_insecure_port(worker_target)
        if bound_port == 0:
            raise WorkerSdkError(f"failed to bind worker endpoint {worker_endpoint}")
        await server.start()
        if endpoint_file:
            path = Path(endpoint_file)
            _write_endpoint_file(path, _announced_worker_endpoint(worker_endpoint, bound_port))
        await shutdown_event.wait()
    finally:
        await server.stop(grace=2)
        await host_channel.close()


@dataclass(slots=True)
class _ActiveInvocation:
    task: asyncio.Task[Any]
    cancel_reason: str | None = None
    cancel_requested: bool = False
    cancel_callback: Callable[[str], None] | None = None


class _WorkerService(pb_grpc.PluginWorkerServicer):
    def __init__(
        self,
        plugin: _SupportsWorkerPlugin,
        runtime: PluginRuntime,
        shutdown_event: asyncio.Event,
    ) -> None:
        self._plugin = plugin
        self._runtime = runtime
        self._shutdown_event = shutdown_event
        self._handlers = _Handlers.empty()
        self._registered_config: Json | object = _UNREGISTERED
        self._registration_lock = asyncio.Lock()
        self._active_invocations: dict[str, _ActiveInvocation] = {}

    async def Handshake(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        plugin_id = _plugin_id(self._plugin)
        return pb.HandshakeResponse(
            plugin_id=plugin_id,
            plugin_kind=plugin_id,
            allows_multiple_components=bool(getattr(self._plugin, "allows_multiple_components", False)),
            worker_protocol=WORKER_PROTOCOL,
            sdk_name="nemo-relay-plugin",
            sdk_version=_SDK_VERSION,
            runtime_name="python",
            runtime_version=platform.python_version(),
            supported_surfaces=_all_surfaces(),
        )

    async def Health(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        plugin_id = _plugin_id(self._plugin)
        return pb.HealthResponse(
            ok=True,
            message="ready",
            plugin_id=plugin_id,
            worker_protocol=WORKER_PROTOCOL,
            sdk_name="nemo-relay-plugin",
            sdk_version=_SDK_VERSION,
            runtime_name="python",
            runtime_version=platform.python_version(),
        )

    async def Validate(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        config = await _decode_optional_config_or_abort(request, context)
        try:
            diagnostics = [_diagnostic_to_json(item) for item in await _maybe_await(self._plugin.validate(config))]
            return pb.ValidateResponse(
                diagnostics=_json_envelope(PLUGIN_DIAGNOSTICS_SCHEMA, diagnostics),
            )
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            return pb.ValidateResponse(error=_sdk_error_to_worker(exc))

    async def Register(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        config = await _decode_optional_config_or_abort(request, context)
        try:
            if self._registered_config is not _UNREGISTERED:
                if config != self._registered_config:
                    raise WorkerSdkError("worker is already registered with a different component config")
                return pb.RegisterResponse(registrations=self._handlers.registrations)
            async with self._registration_lock:
                if self._registered_config is not _UNREGISTERED:
                    if config != self._registered_config:
                        raise WorkerSdkError("worker is already registered with a different component config")
                    return pb.RegisterResponse(registrations=self._handlers.registrations)
                registered_config = copy.deepcopy(config)
                ctx = PluginContext(runtime=self._runtime)
                await _maybe_await(self._plugin.register(ctx, config))
                self._handlers = ctx._handlers
                self._registered_config = registered_config
                return pb.RegisterResponse(registrations=ctx._handlers.registrations)
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            return pb.RegisterResponse(error=_sdk_error_to_worker(exc))

    async def Invoke(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        try:
            active = self._start_invocation(request.invocation_id, self._invoke_result(request))
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            return pb.InvokeResponse(error=_sdk_error_to_worker(exc))
        try:
            return await active.task
        except asyncio.CancelledError:
            if active.cancel_reason is None:
                raise
            return pb.InvokeResponse(error=_cancelled_worker_error(active.cancel_reason))
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            return pb.InvokeResponse(error=_sdk_error_to_worker(exc))
        finally:
            self._forget_invocation(request.invocation_id, active)

    async def InvokeStream(self, request: Any, context: Any) -> AsyncIterator[Any]:
        await self._authorize(request, context)
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=16)

        def cancel_stream(reason: str) -> None:
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(pb.StreamChunk(error=_cancelled_worker_error(reason)))

        async def produce() -> None:
            try:
                if request.surface != pb.LLM_STREAM_EXECUTION_INTERCEPT:
                    raise WorkerSdkError("InvokeStream only supports LLM stream execution intercepts")
                handler = self._handler(self._handlers.llm_stream_executions, request.registration_name)
                payload = _require_payload(request, "llm")
                llm_request = _decode_required_envelope(payload.request, "llm request", LLM_REQUEST_SCHEMA)
                next_call = LlmStreamNext(self._runtime, request.continuation_id)
                with _bind_invocation_scope(request):
                    stream = await _maybe_await(handler(payload.model_name, llm_request, next_call))
                    async for value in _as_async_iter(stream):
                        await queue.put(pb.StreamChunk(value=_json_envelope(JSON_SCHEMA, value)))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
                await queue.put(pb.StreamChunk(error=_sdk_error_to_worker(exc)))

        try:
            active = self._start_invocation(request.invocation_id, produce())
            active.cancel_callback = cancel_stream
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            yield pb.StreamChunk(error=_sdk_error_to_worker(exc))
            return
        try:
            while True:
                if active.task.done() and queue.empty():
                    break
                next_item = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait((next_item, active.task), return_when=asyncio.FIRST_COMPLETED)
                if next_item in done:
                    item = next_item.result()
                    if active.cancel_requested and item.error.code != "worker.cancelled":
                        continue
                    yield item
                    continue
                next_item.cancel()
                await asyncio.gather(next_item, return_exceptions=True)
                while not queue.empty():
                    yield queue.get_nowait()
                break
            if active.task.cancelled() and active.cancel_reason is None:
                yield pb.StreamChunk(error=_cancelled_worker_error("stream callback cancelled without a host request"))
        finally:
            if not active.task.done():
                active.task.cancel()
            await asyncio.gather(active.task, return_exceptions=True)
            self._forget_invocation(request.invocation_id, active)

    async def CancelInvocation(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        active = self._active_invocations.get(request.invocation_id)
        if active is None or active.cancel_requested:
            return pb.WorkerAck(accepted=False, message="invocation is not active")
        if active.task.done() and active.cancel_callback is None:
            return pb.WorkerAck(accepted=False, message="invocation is not active")
        active.cancel_requested = True
        active.cancel_reason = request.reason or "host requested cancellation"
        if active.cancel_callback is not None:
            active.cancel_callback(active.cancel_reason)
        if not active.task.done():
            active.task.cancel()
        return pb.WorkerAck(accepted=True, message=f"cancellation accepted: {active.cancel_reason}")

    async def Shutdown(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        asyncio.get_running_loop().call_soon(self._shutdown_event.set)
        return pb.WorkerAck(accepted=True, message="shutdown accepted")

    async def _invoke_result(self, request: Any) -> Any:
        with _bind_invocation_scope(request):
            if request.surface == pb.SUBSCRIBER:
                event = _decode_required_envelope(request.event, "event", EVENT_SCHEMA)
                await _maybe_await(self._handler(self._handlers.subscribers, request.registration_name)(event))
                return pb.InvokeResponse(empty=pb.EmptyResult())
            event_sanitizer_handlers = {
                pb.MARK_SANITIZE_GUARDRAIL: self._handlers.mark_sanitizers,
                pb.SCOPE_SANITIZE_START_GUARDRAIL: self._handlers.scope_start_sanitizers,
                pb.SCOPE_SANITIZE_END_GUARDRAIL: self._handlers.scope_end_sanitizers,
            }
            if request.surface in event_sanitizer_handlers:
                event = _decode_required_envelope(request.event, "event", EVENT_SCHEMA)
                fields: EventSanitizeFields = {
                    "data": event.get("data"),
                    "category_profile": event.get("category_profile"),
                    "metadata": event.get("metadata"),
                }
                handler = self._handler(event_sanitizer_handlers[request.surface], request.registration_name)
                return _json_response(await _maybe_await(handler(event, fields)))
            if request.surface == pb.TOOL_SANITIZE_REQUEST_GUARDRAIL:
                return _json_response(
                    await _maybe_await(
                        self._handler(self._handlers.tool_sanitize_requests, request.registration_name)(
                            request.tool.tool_name,
                            _decode_required_envelope(request.tool.value, "tool value"),
                        )
                    )
                )
            if request.surface == pb.TOOL_SANITIZE_RESPONSE_GUARDRAIL:
                return _json_response(
                    await _maybe_await(
                        self._handler(self._handlers.tool_sanitize_responses, request.registration_name)(
                            request.tool.tool_name,
                            _decode_required_envelope(request.tool.value, "tool value"),
                        )
                    )
                )
            if request.surface == pb.TOOL_CONDITIONAL_EXECUTION_GUARDRAIL:
                result = await _maybe_await(
                    self._handler(self._handlers.tool_conditionals, request.registration_name)(
                        request.tool.tool_name,
                        _decode_required_envelope(request.tool.value, "tool value"),
                    )
                )
                return pb.InvokeResponse(guardrail=pb.GuardrailResult(block_reason=result or ""))
            if request.surface == pb.TOOL_REQUEST_INTERCEPT:
                result = await _maybe_await(
                    self._handler(self._handlers.tool_requests, request.registration_name)(
                        request.tool.tool_name,
                        _decode_required_envelope(request.tool.value, "tool value"),
                    )
                )
                return _json_response(result)
            if request.surface == pb.TOOL_EXECUTION_INTERCEPT:
                result = await _maybe_await(
                    self._handler(self._handlers.tool_executions, request.registration_name)(
                        request.tool.tool_name,
                        _decode_required_envelope(request.tool.value, "tool value"),
                        ToolNext(self._runtime, request.continuation_id),
                    )
                )
                if not isinstance(result, ToolExecutionInterceptOutcome):
                    raise WorkerSdkError("tool execution intercept must return ToolExecutionInterceptOutcome")
                return pb.InvokeResponse(
                    tool_execution=pb.ToolExecutionInterceptResult(
                        outcome=_json_envelope(
                            TOOL_EXECUTION_INTERCEPT_OUTCOME_SCHEMA,
                            result.to_json(),
                        ),
                    )
                )
            if request.surface == pb.LLM_SANITIZE_REQUEST_GUARDRAIL:
                return _json_response(
                    await _maybe_await(
                        self._handler(self._handlers.llm_sanitize_requests, request.registration_name)(
                            _decode_required_envelope(request.llm.request, "llm request", LLM_REQUEST_SCHEMA)
                        )
                    )
                )
            if request.surface == pb.LLM_SANITIZE_RESPONSE_GUARDRAIL:
                return _json_response(
                    await _maybe_await(
                        self._handler(self._handlers.llm_sanitize_responses, request.registration_name)(
                            _decode_required_envelope(request.llm.response, "llm response")
                        )
                    )
                )
            if request.surface == pb.LLM_CONDITIONAL_EXECUTION_GUARDRAIL:
                result = await _maybe_await(
                    self._handler(self._handlers.llm_conditionals, request.registration_name)(
                        _decode_required_envelope(request.llm.request, "llm request", LLM_REQUEST_SCHEMA)
                    )
                )
                return pb.InvokeResponse(guardrail=pb.GuardrailResult(block_reason=result or ""))
            if request.surface == pb.LLM_REQUEST_INTERCEPT:
                payload = request.llm
                llm_request = _decode_required_envelope(payload.request, "llm request", LLM_REQUEST_SCHEMA)
                annotated = (
                    _decode_required_envelope(
                        payload.annotated_request,
                        "annotated llm request",
                        ANNOTATED_LLM_REQUEST_SCHEMA,
                    )
                    if payload.HasField("annotated_request")
                    else None
                )
                result = await _maybe_await(
                    self._handler(self._handlers.llm_requests, request.registration_name)(
                        payload.model_name,
                        llm_request,
                        annotated,
                    )
                )
                if not isinstance(result, LlmRequestInterceptOutcome):
                    raise WorkerSdkError("LLM request intercept must return LlmRequestInterceptOutcome")
                return pb.InvokeResponse(
                    llm_request=pb.LlmRequestInterceptResult(
                        outcome=_json_envelope(
                            LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA,
                            result.to_json(),
                        ),
                    )
                )
            if request.surface == pb.LLM_EXECUTION_INTERCEPT:
                payload = request.llm
                result = await _maybe_await(
                    self._handler(self._handlers.llm_executions, request.registration_name)(
                        payload.model_name,
                        _decode_required_envelope(payload.request, "llm request", LLM_REQUEST_SCHEMA),
                        LlmNext(self._runtime, request.continuation_id),
                    )
                )
                return _json_response(result)
            raise WorkerSdkError(f"unsupported registration surface {request.surface}")

    def _start_invocation(
        self,
        invocation_id: str,
        coroutine: Any,
    ) -> _ActiveInvocation:
        if not invocation_id:
            coroutine.close()
            raise WorkerSdkError("invocation_id must not be empty")
        current = self._active_invocations.get(invocation_id)
        if current is not None:
            coroutine.close()
            raise WorkerSdkError(f"invocation '{invocation_id}' is already active")
        task = asyncio.create_task(coroutine)
        active = _ActiveInvocation(task=task)
        self._active_invocations[invocation_id] = active
        return active

    def _forget_invocation(self, invocation_id: str, active: _ActiveInvocation) -> None:
        if self._active_invocations.get(invocation_id) is active:
            self._active_invocations.pop(invocation_id, None)

    async def _authorize(self, request: Any, context: Any) -> None:
        if not hmac.compare_digest(request.activation_id, self._runtime._activation_id):
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "invalid activation ID")
        if not hmac.compare_digest(request.auth_token, self._runtime._auth_token):
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "invalid auth token")

    def _handler(self, handlers: dict[str, Any], name: str) -> Any:
        try:
            return handlers[name]
        except KeyError as exc:
            raise WorkerSdkError(f"handler {name!r} is not registered") from exc


def _plugin_id(plugin: _SupportsWorkerPlugin) -> str:
    plugin_id = getattr(plugin, "plugin_id", "")
    if callable(plugin_id):
        plugin_id = plugin_id()
    if not isinstance(plugin_id, str) or not plugin_id:
        raise WorkerSdkError("plugin_id must be a non-empty string")
    return plugin_id


def _all_surfaces() -> list[int]:
    return [
        pb.SUBSCRIBER,
        pb.MARK_SANITIZE_GUARDRAIL,
        pb.SCOPE_SANITIZE_START_GUARDRAIL,
        pb.SCOPE_SANITIZE_END_GUARDRAIL,
        pb.TOOL_SANITIZE_REQUEST_GUARDRAIL,
        pb.TOOL_SANITIZE_RESPONSE_GUARDRAIL,
        pb.TOOL_CONDITIONAL_EXECUTION_GUARDRAIL,
        pb.TOOL_REQUEST_INTERCEPT,
        pb.TOOL_EXECUTION_INTERCEPT,
        pb.LLM_SANITIZE_REQUEST_GUARDRAIL,
        pb.LLM_SANITIZE_RESPONSE_GUARDRAIL,
        pb.LLM_CONDITIONAL_EXECUTION_GUARDRAIL,
        pb.LLM_REQUEST_INTERCEPT,
        pb.LLM_EXECUTION_INTERCEPT,
        pb.LLM_STREAM_EXECUTION_INTERCEPT,
    ]


def _diagnostic_to_json(value: ConfigDiagnostic | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, ConfigDiagnostic):
        return value.to_json()
    return _normalize_diagnostic(value)


def _json_envelope(schema: str, value: Json) -> Any:
    _validate_json_shape(schema, value)
    return pb.JsonEnvelope(
        schema=schema,
        json=json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"),
    )


def _optional_json_envelope(value: Json | None, schema: str = JSON_SCHEMA) -> Any:
    if value is None:
        return None
    return _json_envelope(schema, value)


def _decode_required_envelope(envelope: Any, field: str, expected_schema: str = JSON_SCHEMA) -> Json:
    if envelope is None or not getattr(envelope, "json", b""):
        raise WorkerSdkError(f"{field} is missing")
    if envelope.schema != expected_schema:
        raise WorkerSdkError(f"{field} has schema {envelope.schema!r}; expected {expected_schema!r}")
    try:
        value = json.loads(envelope.json.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerSdkError(f"{field} contains invalid JSON") from exc
    _validate_json_shape(expected_schema, value)
    return value


def _reject_json_constant(value: str) -> Json:
    raise WorkerSdkError(f"non-standard JSON constant {value!r} is not allowed")


def _validate_json_shape(schema: str, value: Json) -> None:
    if schema in _OBJECT_SCHEMAS and not isinstance(value, dict):
        raise WorkerSdkError(f"{schema} payload must be a JSON object")
    _validate_json_object_keys(value)


def _validate_json_object_keys(value: Json, ancestors: set[int] | None = None) -> None:
    if not isinstance(value, (dict, list, tuple)):
        return
    if ancestors is None:
        ancestors = set()
    object_id = id(value)
    if object_id in ancestors:
        raise WorkerSdkError("JSON payload must not contain circular references")
    ancestors.add(object_id)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise WorkerSdkError("JSON object keys must be strings")
                _validate_json_object_keys(item, ancestors)
        else:
            for item in value:
                _validate_json_object_keys(item, ancestors)
    finally:
        ancestors.remove(object_id)


def _decode_optional_json(message: Any, field: str, *, default: Json) -> Json:
    if hasattr(message, "HasField") and not message.HasField(field):
        return default
    return _decode_required_envelope(getattr(message, field), field)


async def _decode_optional_config_or_abort(message: Any, context: Any) -> Json:
    try:
        return _decode_optional_json(message, "config", default=None)
    except Exception as exc:  # noqa: BLE001 - malformed config is a protocol error.
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid config: {exc}")
        raise AssertionError("context.abort should not return") from exc


def _json_response(value: Json) -> Any:
    return pb.InvokeResponse(json=pb.JsonResult(value=_json_envelope(JSON_SCHEMA, value)))


def _json_result_to_value(result: Any) -> Json:
    if result.HasField("error"):
        raise _worker_error_to_sdk(result.error)
    return _decode_required_envelope(result.value, "json result")


def _stream_chunk_to_value(chunk: Any) -> Json:
    item = chunk.WhichOneof("item")
    if item == "error":
        raise _worker_error_to_sdk(chunk.error)
    if item != "value":
        raise WorkerSdkError("stream chunk is empty")
    return _decode_required_envelope(chunk.value, "stream chunk")


def _worker_error_to_sdk(error: Any) -> WorkerSdkError:
    return WorkerSdkError(f"{error.code}: {error.message}")


def _sdk_error_to_worker(error: BaseException) -> Any:
    code = "worker.error"
    if isinstance(error, WorkerSdkError):
        code = "worker.sdk_error"
    return pb.WorkerError(code=code, message=str(error), retryable=False)


def _cancelled_worker_error(reason: str) -> Any:
    return pb.WorkerError(
        code="worker.cancelled",
        message=f"worker invocation was cancelled: {reason}",
        retryable=False,
    )


def _ack_to_result(response: Any) -> None:
    if response.ok:
        return
    if response.HasField("error"):
        raise _worker_error_to_sdk(response.error)
    raise WorkerSdkError("host call failed")


def _require_payload(request: Any, payload: str) -> Any:
    if request.WhichOneof("payload") != payload:
        raise WorkerSdkError(f"expected {payload} payload")
    return getattr(request, payload)


@contextlib.contextmanager
def _bind_invocation_scope(request: Any) -> Iterator[None]:
    scope = None
    if request.HasField("scope"):
        scope_stack_id = request.scope.scope_stack_id
        parent_scope_id = request.scope.parent_scope_id or None
        if not scope_stack_id:
            if parent_scope_id is not None:
                raise WorkerSdkError("parent_scope_id requires scope_stack_id")
            raise WorkerSdkError("scope_stack_id must not be empty")
        scope = _BoundScopeContext(
            scope_stack_id=scope_stack_id,
            parent_scope_id=parent_scope_id,
        )
    token = _SCOPE_CONTEXT.set(scope)
    try:
        yield
    finally:
        _SCOPE_CONTEXT.reset(token)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _as_async_iter(value: Iterable[Json] | AsyncIterator[Json]) -> AsyncIterator[Json]:
    if isinstance(value, AsyncIterator):
        async for item in value:
            yield item
        return
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise WorkerSdkError("stream callback must return an iterable of JSON chunks, not a scalar or mapping")
    if not isinstance(value, Iterable):
        raise WorkerSdkError("stream callback must return an iterable or async iterator of JSON chunks")
    for item in value:
        yield item


def _proto_scope_type(scope_type: ScopeType | str) -> int:
    value = ScopeType(scope_type)
    mapping = {
        ScopeType.AGENT: pb.AGENT,
        ScopeType.FUNCTION: pb.FUNCTION,
        ScopeType.TOOL: pb.TOOL,
        ScopeType.LLM: pb.LLM,
        ScopeType.RETRIEVER: pb.RETRIEVER,
        ScopeType.EMBEDDER: pb.EMBEDDER,
        ScopeType.RERANKER: pb.RERANKER,
        ScopeType.GUARDRAIL: pb.GUARDRAIL,
        ScopeType.EVALUATOR: pb.EVALUATOR,
        ScopeType.CUSTOM: pb.CUSTOM,
        ScopeType.UNKNOWN: pb.UNKNOWN,
    }
    return mapping[value]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise WorkerSdkError(f"environment variable {name} is required")
    return value


def _grpc_target(endpoint: str) -> str:
    if endpoint.startswith("unix://"):
        path = endpoint.removeprefix("unix://")
        if not path:
            raise WorkerSdkError("unix endpoint path cannot be empty")
        return "unix:" + path
    if not endpoint.startswith(("tcp://", "http://")):
        raise WorkerSdkError(f"unsupported endpoint {endpoint!r}")

    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise WorkerSdkError(f"invalid endpoint {endpoint!r}: {exc}") from exc
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or port is None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise WorkerSdkError(f"invalid endpoint {endpoint!r}")

    host = parsed.hostname
    if host.lower() != "localhost":
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise WorkerSdkError(f"endpoint host {host!r} must be an explicit loopback address") from exc
        if not address.is_loopback:
            raise WorkerSdkError(f"endpoint host {host!r} is not a loopback address")
    return parsed.netloc


def _open_host_channel(endpoint: str) -> Any:
    target = _grpc_target(endpoint)
    if endpoint.startswith("unix://"):
        return grpc.aio.insecure_channel(
            target,
            options=(("grpc.default_authority", "localhost"),),
        )
    return grpc.aio.insecure_channel(target)


def _announced_worker_endpoint(worker_endpoint: str, bound_port: int) -> str:
    target = _grpc_target(worker_endpoint)
    if target.startswith("unix:"):
        return worker_endpoint
    host, _, port = target.rpartition(":")
    if port == "0":
        return f"http://{host}:{bound_port}"
    return f"http://{host}:{port}"


def _write_endpoint_file(path: Path, endpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(endpoint)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def _unlink_unix_socket(endpoint: str) -> None:
    if endpoint.startswith("unix://"):
        path = Path(endpoint.removeprefix("unix://"))
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise WorkerSdkError(f"worker socket path {path} exists and is not a socket")
        try:
            _, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), timeout=1)
        except (ConnectionRefusedError, FileNotFoundError):
            path.unlink(missing_ok=True)
        except TimeoutError as exc:
            raise WorkerSdkError(f"timed out while probing worker socket path {path}") from exc
        except OSError as exc:
            raise WorkerSdkError(f"unable to determine whether worker socket path {path} is active") from exc
        else:
            writer.close()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), timeout=1)
            raise WorkerSdkError(f"worker socket path {path} is already active")
