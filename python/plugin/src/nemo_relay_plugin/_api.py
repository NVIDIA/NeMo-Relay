# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level Python API for NeMo Relay grpc-v1 worker plugins."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import importlib
import inspect
import json
import os
import platform
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeAlias

grpc: Any = importlib.import_module("grpc")
pb: Any = importlib.import_module("._proto.plugin_worker_pb2", __package__)
pb_grpc: Any = importlib.import_module("._proto.plugin_worker_pb2_grpc", __package__)

Json: TypeAlias = Any
Event: TypeAlias = dict[str, Any]
LlmRequest: TypeAlias = dict[str, Any]
AnnotatedLlmRequest: TypeAlias = dict[str, Any]

WORKER_PROTOCOL = "grpc-v1"
JSON_SCHEMA = "nemo.relay.Json@1"
EVENT_SCHEMA = "nemo.relay.Event@1"
LLM_REQUEST_SCHEMA = "nemo.relay.LlmRequest@1"
ANNOTATED_LLM_REQUEST_SCHEMA = "nemo.relay.AnnotatedLlmRequest@1"
PLUGIN_DIAGNOSTICS_SCHEMA = "nemo.relay.PluginDiagnostics@1"
_SCOPE_CONTEXT: contextvars.ContextVar[_BoundScopeContext | None] = contextvars.ContextVar(
    "nemo_relay_plugin_scope_context",
    default=None,
)


class WorkerSdkError(Exception):
    """Error raised by the Python worker SDK."""


class DiagnosticLevel(str, Enum):
    """Plugin configuration diagnostic severity."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class ConfigDiagnostic:
    """Structured plugin configuration diagnostic."""

    level: DiagnosticLevel | str
    code: str
    message: str
    component: str | None = None
    field: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the Relay JSON representation."""
        value = asdict(self)
        if isinstance(self.level, DiagnosticLevel):
            value["level"] = self.level.value
        return {key: item for key, item in value.items() if item is not None}


class ScopeType(str, Enum):
    """Relay scope type accepted by host runtime scope calls."""

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


class WorkerPlugin:
    """Base class for Python worker plugins."""

    plugin_id: str = ""
    allows_multiple_components: bool = False

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        """Validate component config before registration."""
        del config
        return []

    def register(self, ctx: PluginContext, config: Json) -> None:
        """Register callbacks into the worker plugin context."""
        del ctx, config
        raise NotImplementedError("WorkerPlugin.register must be implemented")


class _SupportsWorkerPlugin(Protocol):
    plugin_id: str
    allows_multiple_components: bool

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]: ...

    def register(self, ctx: PluginContext, config: Json) -> None: ...


SubscriberCallback: TypeAlias = Callable[[Event], None | Awaitable[None]]
ToolSanitizeCallback: TypeAlias = Callable[[str, Json], Json | Awaitable[Json]]
ToolConditionalCallback: TypeAlias = Callable[[str, Json], str | None | Awaitable[str | None]]
ToolRequestCallback: TypeAlias = Callable[[str, Json], Json | Awaitable[Json]]
ToolExecutionCallback: TypeAlias = Callable[[str, Json, "ToolNext"], Json | Awaitable[Json]]
LlmSanitizeRequestCallback: TypeAlias = Callable[[LlmRequest], LlmRequest | Awaitable[LlmRequest]]
LlmSanitizeResponseCallback: TypeAlias = Callable[[Json], Json | Awaitable[Json]]
LlmConditionalCallback: TypeAlias = Callable[[LlmRequest], str | None | Awaitable[str | None]]
LlmRequestCallback: TypeAlias = Callable[
    [str, LlmRequest, AnnotatedLlmRequest | None],
    LlmRequest
    | tuple[LlmRequest, AnnotatedLlmRequest | None]
    | Awaitable[LlmRequest | tuple[LlmRequest, AnnotatedLlmRequest | None]],
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
    """Registration context passed to ``WorkerPlugin.register``."""

    def __init__(self, runtime: PluginRuntime | None = None) -> None:
        self._runtime = runtime
        self._handlers = _Handlers.empty()

    @property
    def runtime(self) -> PluginRuntime:
        """Return the host runtime handle for event and scope operations."""
        if self._runtime is None:
            raise WorkerSdkError("PluginContext has no runtime handle")
        return self._runtime

    def register_subscriber(self, name: str, callback: SubscriberCallback) -> None:
        """Register an event subscriber."""
        self._push_registration(name, pb.SUBSCRIBER, 0, False)
        self._handlers.subscribers[name] = callback

    def register_tool_sanitize_request_guardrail(
        self,
        name: str,
        callback: ToolSanitizeCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a tool sanitize-request guardrail."""
        self._push_registration(name, pb.TOOL_SANITIZE_REQUEST_GUARDRAIL, priority, False)
        self._handlers.tool_sanitize_requests[name] = callback

    def register_tool_sanitize_response_guardrail(
        self,
        name: str,
        callback: ToolSanitizeCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a tool sanitize-response guardrail."""
        self._push_registration(name, pb.TOOL_SANITIZE_RESPONSE_GUARDRAIL, priority, False)
        self._handlers.tool_sanitize_responses[name] = callback

    def register_tool_conditional_execution_guardrail(
        self,
        name: str,
        callback: ToolConditionalCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a tool conditional-execution guardrail."""
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
        """Register a tool request intercept."""
        self._push_registration(name, pb.TOOL_REQUEST_INTERCEPT, priority, break_chain)
        self._handlers.tool_requests[name] = callback

    def register_tool_execution_intercept(
        self,
        name: str,
        callback: ToolExecutionCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register a tool execution intercept."""
        self._push_registration(name, pb.TOOL_EXECUTION_INTERCEPT, priority, False)
        self._handlers.tool_executions[name] = callback

    def register_llm_sanitize_request_guardrail(
        self,
        name: str,
        callback: LlmSanitizeRequestCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register an LLM sanitize-request guardrail."""
        self._push_registration(name, pb.LLM_SANITIZE_REQUEST_GUARDRAIL, priority, False)
        self._handlers.llm_sanitize_requests[name] = callback

    def register_llm_sanitize_response_guardrail(
        self,
        name: str,
        callback: LlmSanitizeResponseCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register an LLM sanitize-response guardrail."""
        self._push_registration(name, pb.LLM_SANITIZE_RESPONSE_GUARDRAIL, priority, False)
        self._handlers.llm_sanitize_responses[name] = callback

    def register_llm_conditional_execution_guardrail(
        self,
        name: str,
        callback: LlmConditionalCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register an LLM conditional-execution guardrail."""
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
        """Register an LLM request intercept."""
        self._push_registration(name, pb.LLM_REQUEST_INTERCEPT, priority, break_chain)
        self._handlers.llm_requests[name] = callback

    def register_llm_execution_intercept(
        self,
        name: str,
        callback: LlmExecutionCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register an LLM execution intercept."""
        self._push_registration(name, pb.LLM_EXECUTION_INTERCEPT, priority, False)
        self._handlers.llm_executions[name] = callback

    def register_llm_stream_execution_intercept(
        self,
        name: str,
        callback: LlmStreamExecutionCallback,
        *,
        priority: int = 0,
    ) -> None:
        """Register an LLM stream execution intercept."""
        self._push_registration(name, pb.LLM_STREAM_EXECUTION_INTERCEPT, priority, False)
        self._handlers.llm_stream_executions[name] = callback

    def _push_registration(self, name: str, surface: int, priority: int, break_chain: bool) -> None:
        self._handlers.registrations.append(
            pb.Registration(
                local_name=name,
                surface=surface,
                priority=priority,
                break_chain=break_chain,
            )
        )


class PluginRuntime:
    """Handle for calling the Relay host runtime from worker callbacks."""

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
        """Emit a mark event through the host runtime."""
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
        """Create an isolated host-owned scope stack."""
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
        """Drop an isolated host-owned scope stack."""
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
        """Push a scope through the host runtime and return its handle ID."""
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
        """Pop a host scope by handle ID."""
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
        """Temporarily bind callbacks to a worker-selected scope stack."""
        scope = _BoundScopeContext(scope_stack_id, parent_scope_id) if scope_stack_id else None
        token = _SCOPE_CONTEXT.set(scope)
        try:
            yield
        finally:
            _SCOPE_CONTEXT.reset(token)

    @contextlib.contextmanager
    def clear_scope_stack(self) -> Iterator[None]:
        """Temporarily clear worker scope-stack correlation."""
        with self.bind_scope_stack(None):
            yield

    def current_scope_stack_id(self) -> str | None:
        """Return the locally bound scope stack ID, if any."""
        scope = _SCOPE_CONTEXT.get()
        return scope.scope_stack_id if scope else None

    def current_parent_scope_id(self) -> str | None:
        """Return the locally bound parent scope ID, if any."""
        scope = _SCOPE_CONTEXT.get()
        return scope.parent_scope_id if scope else None

    def _scope_context(self, scope_stack_id: str | None = None, parent_scope_id: str | None = None) -> Any:
        if scope_stack_id is not None:
            effective_scope = _BoundScopeContext(scope_stack_id, parent_scope_id)
        else:
            effective_scope = _SCOPE_CONTEXT.get()
        if not effective_scope:
            return None
        return pb.ScopeContext(
            scope_stack_id=effective_scope.scope_stack_id,
            parent_scope_id=effective_scope.parent_scope_id or "",
        )


class ToolNext:
    """Continuation handle for tool execution intercepts."""

    def __init__(self, runtime: PluginRuntime, continuation_id: str) -> None:
        self._runtime = runtime
        self._continuation_id = continuation_id

    async def call(self, value: Json) -> Json:
        """Call the remaining tool execution chain."""
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
    """Continuation handle for LLM execution intercepts."""

    def __init__(self, runtime: PluginRuntime, continuation_id: str) -> None:
        self._runtime = runtime
        self._continuation_id = continuation_id

    async def call(self, request: LlmRequest) -> Json:
        """Call the remaining LLM execution chain."""
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
    """Continuation handle for LLM stream execution intercepts."""

    def __init__(self, runtime: PluginRuntime, continuation_id: str) -> None:
        self._runtime = runtime
        self._continuation_id = continuation_id

    def call(self, request: LlmRequest) -> AsyncIterator[Json]:
        """Call the remaining LLM stream execution chain."""
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
    """Serve a worker plugin using environment variables supplied by the Relay host."""
    worker_endpoint = _required_env("NEMO_RELAY_WORKER_SOCKET")
    host_endpoint = _required_env("NEMO_RELAY_HOST_SOCKET")
    activation_id = _required_env("NEMO_RELAY_WORKER_ID")
    auth_token = _required_env("NEMO_RELAY_WORKER_TOKEN")
    endpoint_file = os.environ.get("NEMO_RELAY_WORKER_ENDPOINT_FILE")

    _unlink_unix_socket(worker_endpoint)
    host_channel = grpc.aio.insecure_channel(_grpc_target(host_endpoint))
    runtime = PluginRuntime(
        activation_id=activation_id,
        auth_token=auth_token,
        host_stub=pb_grpc.RelayHostRuntimeStub(host_channel),
    )
    shutdown_event = asyncio.Event()
    service = _WorkerService(plugin, runtime, shutdown_event)
    server = grpc.aio.server()
    pb_grpc.add_PluginWorkerServicer_to_server(service, server)
    bound_port = server.add_insecure_port(_grpc_target(worker_endpoint))
    if bound_port == 0:
        raise WorkerSdkError(f"failed to bind worker endpoint {worker_endpoint}")
    if endpoint_file:
        path = Path(endpoint_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_announced_worker_endpoint(worker_endpoint, bound_port), encoding="utf-8")

    await server.start()
    try:
        await shutdown_event.wait()
    finally:
        await server.stop(grace=2)
        await host_channel.close()


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

    async def Handshake(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        plugin_id = _plugin_id(self._plugin)
        return pb.HandshakeResponse(
            plugin_id=plugin_id,
            plugin_kind=plugin_id,
            allows_multiple_components=bool(getattr(self._plugin, "allows_multiple_components", False)),
            worker_protocol=WORKER_PROTOCOL,
            sdk_name="nemo-relay-plugin",
            sdk_version="0.5.0",
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
            sdk_version="0.5.0",
            runtime_name="python",
            runtime_version=platform.python_version(),
        )

    async def Validate(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        config = await _decode_optional_config_or_abort(request, context)
        try:
            diagnostics = [_diagnostic_to_json(item) for item in self._plugin.validate(config)]
            return pb.ValidateResponse(
                diagnostics=_json_envelope(PLUGIN_DIAGNOSTICS_SCHEMA, diagnostics),
            )
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            return pb.ValidateResponse(error=_sdk_error_to_worker(exc))

    async def Register(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        config = await _decode_optional_config_or_abort(request, context)
        try:
            ctx = PluginContext(runtime=self._runtime)
            self._plugin.register(ctx, config)
            self._handlers = ctx._handlers
            return pb.RegisterResponse(registrations=ctx._handlers.registrations)
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            return pb.RegisterResponse(error=_sdk_error_to_worker(exc))

    async def Invoke(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        try:
            return await self._invoke_result(request)
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            return pb.InvokeResponse(error=_sdk_error_to_worker(exc))

    async def InvokeStream(self, request: Any, context: Any) -> AsyncIterator[Any]:
        await self._authorize(request, context)
        try:
            if request.surface != pb.LLM_STREAM_EXECUTION_INTERCEPT:
                raise WorkerSdkError("InvokeStream only supports LLM stream execution intercepts")
            handler = self._handler(self._handlers.llm_stream_executions, request.registration_name)
            payload = _require_payload(request, "llm")
            llm_request = _decode_required_envelope(payload.request, "llm request")
            next_call = LlmStreamNext(self._runtime, request.continuation_id)
            with _bind_invocation_scope(request):
                stream = await _maybe_await(handler(payload.model_name, llm_request, next_call))
                async for value in _as_async_iter(stream):
                    yield pb.StreamChunk(value=_json_envelope(JSON_SCHEMA, value))
        except Exception as exc:  # noqa: BLE001 - callback failure is protocol data.
            yield pb.StreamChunk(error=_sdk_error_to_worker(exc))

    async def CancelInvocation(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        return pb.WorkerAck(accepted=False, message="cancel is not implemented by the Python worker SDK")

    async def Shutdown(self, request: Any, context: Any) -> Any:
        await self._authorize(request, context)
        asyncio.get_running_loop().call_soon(self._shutdown_event.set)
        return pb.WorkerAck(accepted=True, message="shutdown accepted")

    async def _invoke_result(self, request: Any) -> Any:
        with _bind_invocation_scope(request):
            if request.surface == pb.SUBSCRIBER:
                event = _decode_required_envelope(request.event, "event")
                await _maybe_await(self._handler(self._handlers.subscribers, request.registration_name)(event))
                return pb.InvokeResponse(empty=pb.EmptyResult())
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
                return _json_response(result)
            if request.surface == pb.LLM_SANITIZE_REQUEST_GUARDRAIL:
                return _json_response(
                    await _maybe_await(
                        self._handler(self._handlers.llm_sanitize_requests, request.registration_name)(
                            _decode_required_envelope(request.llm.request, "llm request")
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
                        _decode_required_envelope(request.llm.request, "llm request")
                    )
                )
                return pb.InvokeResponse(guardrail=pb.GuardrailResult(block_reason=result or ""))
            if request.surface == pb.LLM_REQUEST_INTERCEPT:
                payload = request.llm
                llm_request = _decode_required_envelope(payload.request, "llm request")
                annotated = (
                    _decode_required_envelope(payload.annotated_request, "annotated llm request")
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
                if isinstance(result, tuple):
                    llm_request, annotated = result
                else:
                    llm_request = result
                return pb.InvokeResponse(
                    llm_request=pb.LlmRequestInterceptResult(
                        request=_json_envelope(LLM_REQUEST_SCHEMA, llm_request),
                        annotated_request=_optional_json_envelope(annotated, ANNOTATED_LLM_REQUEST_SCHEMA),
                        has_annotated_request=annotated is not None,
                    )
                )
            if request.surface == pb.LLM_EXECUTION_INTERCEPT:
                payload = request.llm
                result = await _maybe_await(
                    self._handler(self._handlers.llm_executions, request.registration_name)(
                        payload.model_name,
                        _decode_required_envelope(payload.request, "llm request"),
                        LlmNext(self._runtime, request.continuation_id),
                    )
                )
                return _json_response(result)
            raise WorkerSdkError(f"unsupported registration surface {request.surface}")

    async def _authorize(self, request: Any, context: Any) -> None:
        if request.activation_id != self._runtime._activation_id:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "invalid activation ID")
        if request.auth_token != self._runtime._auth_token:
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
    return dict(value)


def _json_envelope(schema: str, value: Json) -> Any:
    return pb.JsonEnvelope(
        schema=schema,
        json=json.dumps(value, separators=(",", ":")).encode("utf-8"),
    )


def _optional_json_envelope(value: Json | None, schema: str = JSON_SCHEMA) -> Any:
    if value is None:
        return None
    return _json_envelope(schema, value)


def _decode_required_envelope(envelope: Any, field: str) -> Json:
    if envelope is None or not getattr(envelope, "json", b""):
        raise WorkerSdkError(f"{field} is missing")
    return json.loads(envelope.json.decode("utf-8"))


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
    if request.HasField("scope") and request.scope.scope_stack_id:
        scope = _BoundScopeContext(
            scope_stack_id=request.scope.scope_stack_id,
            parent_scope_id=request.scope.parent_scope_id or None,
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
        return "unix:" + endpoint.removeprefix("unix://")
    if endpoint.startswith("tcp://"):
        return endpoint.removeprefix("tcp://")
    if endpoint.startswith("http://"):
        return endpoint.removeprefix("http://")
    return endpoint


def _announced_worker_endpoint(worker_endpoint: str, bound_port: int) -> str:
    target = _grpc_target(worker_endpoint)
    if target.startswith("unix:"):
        return worker_endpoint
    host, separator, port = target.rpartition(":")
    if not separator:
        return f"http://{target}"
    if port == "0":
        return f"http://{host}:{bound_port}"
    return f"http://{host}:{port}"


def _unlink_unix_socket(endpoint: str) -> None:
    if endpoint.startswith("unix://"):
        Path(endpoint.removeprefix("unix://")).unlink(missing_ok=True)
