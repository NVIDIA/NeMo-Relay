# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python dynamic worker plugin SDK."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

if os.environ.get("NEMO_RELAY_SKIP_PYTHON_PLUGIN_TESTS") == "1":
    pytest.skip("grpcio is unavailable for Python plugin SDK tests on this runner", allow_module_level=True)

grpc = pytest.importorskip("grpc")

from nemo_relay_plugin import (  # noqa: E402
    ConfigDiagnostic,
    DiagnosticLevel,
    Json,
    PluginContext,
    PluginRuntime,
    ScopeType,
    ToolNext,
    WorkerPlugin,
    WorkerSdkError,
    serve_plugin,
)
from nemo_relay_plugin._api import (  # noqa: E402
    ANNOTATED_LLM_REQUEST_SCHEMA,
    EVENT_SCHEMA,
    JSON_SCHEMA,
    LLM_REQUEST_SCHEMA,
    WORKER_PROTOCOL,
    _announced_worker_endpoint,
    _grpc_target,
    _json_envelope,
    _required_env,
    _unlink_unix_socket,
    _WorkerService,
    pb,
    pb_grpc,
)

ACTIVATION_ID = "act"
AUTH_TOKEN = "token"


class GrpcAbort(Exception):
    def __init__(self, code: object, details: str) -> None:
        super().__init__(f"{code}: {details}")
        self.code = code
        self.details = details


class AbortContext:
    async def abort(self, code: object, details: str) -> None:
        raise GrpcAbort(code, details)


class RecordingHostStub:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.failures: dict[str, str] = {}

    async def EmitMark(self, request: Any) -> Any:
        self.requests.append(request)
        return self._host_ack("EmitMark")

    async def CreateScopeStack(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("CreateScopeStack") == "error":
            return pb.CreateScopeStackResponse(error=_worker_error("CreateScopeStack failed"))
        return pb.CreateScopeStackResponse(scope_stack_id="stack-1")

    async def DropScopeStack(self, request: Any) -> Any:
        self.requests.append(request)
        return self._host_ack("DropScopeStack")

    async def PushScope(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("PushScope") == "error":
            return pb.PushScopeResponse(error=_worker_error("PushScope failed"))
        return pb.PushScopeResponse(scope_handle_id="scope-1")

    async def PopScope(self, request: Any) -> Any:
        self.requests.append(request)
        return self._host_ack("PopScope")

    async def ToolNext(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("ToolNext") == "error":
            return pb.JsonResult(error=_worker_error("ToolNext failed"))
        value = json.loads(request.value.json.decode("utf-8"))
        return pb.JsonResult(value=_json_envelope(JSON_SCHEMA, {"next_tool": value}))

    async def LlmNext(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("LlmNext") == "error":
            return pb.JsonResult(error=_worker_error("LlmNext failed"))
        value = json.loads(request.request.json.decode("utf-8"))
        return pb.JsonResult(value=_json_envelope(JSON_SCHEMA, {"next_llm": value}))

    def LlmStreamNext(self, request: Any) -> AsyncIterator[Any]:
        self.requests.append(request)

        async def stream() -> AsyncIterator[Any]:
            failure = self.failures.get("LlmStreamNext")
            if failure == "error":
                yield pb.StreamChunk(error=_worker_error("LlmStreamNext failed"))
                return
            if failure == "empty":
                yield pb.StreamChunk()
                return
            value = json.loads(request.request.json.decode("utf-8"))
            yield pb.StreamChunk(value=_json_envelope(JSON_SCHEMA, {"next_stream": value}))

        return stream()

    def _host_ack(self, method: str) -> Any:
        failure = self.failures.get(method)
        if failure == "empty":
            return pb.HostAck(ok=False)
        if failure == "error":
            return pb.HostAck(ok=False, error=_worker_error(f"{method} failed"))
        return pb.HostAck(ok=True)


class AllSurfacesPlugin(WorkerPlugin):
    plugin_id = "tests.python_worker"

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        if isinstance(config, dict) and config.get("warn"):
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.WARNING,
                    code="tests.warn",
                    message="warning requested",
                )
            ]
        return []

    def register(self, ctx: PluginContext, config: Json) -> None:
        del config

        async def subscriber(event: Json) -> None:
            await ctx.runtime.emit_mark("tests.subscriber", event)

        def tool_sanitize(name: str, value: Json) -> Json:
            return _tag(value, f"sanitize_{name}")

        def tool_block(name: str, value: Json) -> str | None:
            del name, value
            return "tool blocked"

        async def tool_request(name: str, value: Json) -> Json:
            return _tag(value, f"request_{name}")

        async def tool_execution(name: str, value: Json, next_call: ToolNext) -> Json:
            result = await next_call.call(_tag(value, f"execute_{name}"))
            return _tag(result, "tool_execution")

        def llm_sanitize_request(request: Json) -> Json:
            return _tag_llm_request(request, "llm_sanitize_request")

        async def llm_sanitize_response(response: Json) -> Json:
            return _tag(response, "llm_sanitize_response")

        def llm_block(request: Json) -> str | None:
            del request
            return "llm blocked"

        def llm_request(name: str, request: Json, annotated: Json | None) -> tuple[Json, Json]:
            del name
            return _tag_llm_request(request, "llm_request"), _tag(annotated or {}, "annotated")

        async def llm_execution(name: str, request: Json, next_call: Any) -> Json:
            result = await next_call.call(_tag_llm_request(request, f"llm_execute_{name}"))
            return _tag(result, "llm_execution")

        async def llm_stream_execution(name: str, request: Json, next_call: Any) -> AsyncIterator[Json]:
            stream = next_call.call(_tag_llm_request(request, f"llm_stream_{name}"))
            async for chunk in stream:
                yield _tag(chunk, "llm_stream_execution")

        ctx.register_subscriber("subscriber", subscriber)
        ctx.register_tool_sanitize_request_guardrail("tool_sanitize", tool_sanitize, priority=1)
        ctx.register_tool_sanitize_response_guardrail("tool_sanitize", tool_sanitize, priority=2)
        ctx.register_tool_conditional_execution_guardrail("tool_conditional", tool_block, priority=3)
        ctx.register_tool_request_intercept("tool_request", tool_request, priority=4, break_chain=True)
        ctx.register_tool_execution_intercept("tool_execution", tool_execution, priority=5)
        ctx.register_llm_sanitize_request_guardrail("llm_sanitize_request", llm_sanitize_request, priority=6)
        ctx.register_llm_sanitize_response_guardrail("llm_sanitize_response", llm_sanitize_response, priority=7)
        ctx.register_llm_conditional_execution_guardrail("llm_conditional", llm_block, priority=8)
        ctx.register_llm_request_intercept("llm_request", llm_request, priority=9, break_chain=True)
        ctx.register_llm_execution_intercept("llm_execution", llm_execution, priority=10)
        ctx.register_llm_stream_execution_intercept("llm_stream_execution", llm_stream_execution, priority=11)


@pytest.fixture(name="host_stub")
def host_stub_fixture() -> RecordingHostStub:
    return RecordingHostStub()


@pytest.fixture(name="service")
def service_fixture(host_stub: RecordingHostStub) -> _WorkerService:
    return _service(AllSurfacesPlugin(), host_stub)


def test_generated_proto_matches_worker_contract() -> None:
    methods = {method.name for method in pb.DESCRIPTOR.services_by_name["PluginWorker"].methods}
    assert methods == {
        "Handshake",
        "Health",
        "Validate",
        "Register",
        "Invoke",
        "InvokeStream",
        "CancelInvocation",
        "Shutdown",
    }
    assert pb.InvokeRequest.DESCRIPTOR.fields_by_name["auth_token"].number == 7
    assert pb.HealthRequest.DESCRIPTOR.fields_by_name["activation_id"].number == 1
    assert pb.HealthRequest.DESCRIPTOR.fields_by_name["auth_token"].number == 2
    assert pb.SUBSCRIBER == 1
    assert pb.TOOL_SANITIZE_REQUEST_GUARDRAIL == 10
    assert pb.LLM_STREAM_EXECUTION_INTERCEPT == 25
    assert pb.CUSTOM == 10


async def test_health_handshake_validate_register_and_all_surfaces(service: _WorkerService) -> None:
    health = await service.Health(pb.HealthRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN), AbortContext())
    assert health.ok
    assert health.plugin_id == "tests.python_worker"
    assert health.worker_protocol == WORKER_PROTOCOL
    assert health.sdk_name == "nemo-relay-plugin"
    assert health.runtime_name == "python"

    handshake = await service.Handshake(_handshake_request(), AbortContext())
    assert handshake.plugin_id == "tests.python_worker"
    assert handshake.plugin_kind == "tests.python_worker"
    assert handshake.worker_protocol == WORKER_PROTOCOL
    assert set(handshake.supported_surfaces) == set(_all_expected_surfaces())

    validate = await service.Validate(
        pb.ValidateRequest(
            activation_id=ACTIVATION_ID,
            plugin_id="tests.python_worker",
            auth_token=AUTH_TOKEN,
            config=_json_envelope(JSON_SCHEMA, {"warn": True}),
        ),
        AbortContext(),
    )
    diagnostics = _envelope_value(validate.diagnostics)
    assert diagnostics == [{"level": "warning", "code": "tests.warn", "message": "warning requested"}]

    register = await _register(service)
    registrations = {
        (registration.local_name, registration.surface): registration for registration in register.registrations
    }
    assert set(registrations) == {
        ("subscriber", pb.SUBSCRIBER),
        ("tool_sanitize", pb.TOOL_SANITIZE_REQUEST_GUARDRAIL),
        ("tool_sanitize", pb.TOOL_SANITIZE_RESPONSE_GUARDRAIL),
        ("tool_conditional", pb.TOOL_CONDITIONAL_EXECUTION_GUARDRAIL),
        ("tool_request", pb.TOOL_REQUEST_INTERCEPT),
        ("tool_execution", pb.TOOL_EXECUTION_INTERCEPT),
        ("llm_sanitize_request", pb.LLM_SANITIZE_REQUEST_GUARDRAIL),
        ("llm_sanitize_response", pb.LLM_SANITIZE_RESPONSE_GUARDRAIL),
        ("llm_conditional", pb.LLM_CONDITIONAL_EXECUTION_GUARDRAIL),
        ("llm_request", pb.LLM_REQUEST_INTERCEPT),
        ("llm_execution", pb.LLM_EXECUTION_INTERCEPT),
        ("llm_stream_execution", pb.LLM_STREAM_EXECUTION_INTERCEPT),
    }
    assert registrations[("tool_request", pb.TOOL_REQUEST_INTERCEPT)].break_chain


@pytest.mark.parametrize(
    ("rpc_name", "request_factory", "streaming"),
    [
        ("Handshake", lambda: _handshake_request(), False),
        ("Health", lambda: pb.HealthRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN), False),
        (
            "Validate",
            lambda: pb.ValidateRequest(
                activation_id=ACTIVATION_ID,
                plugin_id="tests.python_worker",
                auth_token=AUTH_TOKEN,
                config=_json_envelope(JSON_SCHEMA, {}),
            ),
            False,
        ),
        (
            "Register",
            lambda: pb.RegisterRequest(
                activation_id=ACTIVATION_ID,
                plugin_id="tests.python_worker",
                auth_token=AUTH_TOKEN,
                config=_json_envelope(JSON_SCHEMA, {}),
            ),
            False,
        ),
        ("Invoke", lambda: _tool_request("missing", pb.TOOL_REQUEST_INTERCEPT, {}), False),
        (
            "InvokeStream",
            lambda: _invoke_request(
                "missing",
                pb.LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            True,
        ),
        (
            "CancelInvocation",
            lambda: pb.CancelInvocationRequest(
                activation_id=ACTIVATION_ID,
                invocation_id="invoke-1",
                auth_token=AUTH_TOKEN,
                reason="test",
            ),
            False,
        ),
        (
            "Shutdown",
            lambda: pb.ShutdownRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, reason="test"),
            False,
        ),
    ],
)
async def test_auth_and_activation_failures_for_every_rpc(
    service: _WorkerService,
    rpc_name: str,
    request_factory: Any,
    streaming: bool,
) -> None:
    for field in ("activation_id", "auth_token"):
        request = request_factory()
        setattr(request, field, "wrong")
        with pytest.raises(GrpcAbort) as exc_info:
            result = getattr(service, rpc_name)(request, AbortContext())
            if streaming:
                async for _chunk in result:
                    pass
            else:
                await result
        assert exc_info.value.code == grpc.StatusCode.PERMISSION_DENIED
        assert field.split("_")[0] in exc_info.value.details


async def test_validate_and_register_decode_errors_are_grpc_protocol_errors(service: _WorkerService) -> None:
    bad_config = _json_envelope(JSON_SCHEMA, {})
    bad_config.json = b"{"

    with pytest.raises(GrpcAbort) as validate_error:
        await service.Validate(
            pb.ValidateRequest(
                activation_id=ACTIVATION_ID,
                plugin_id="tests.python_worker",
                auth_token=AUTH_TOKEN,
                config=bad_config,
            ),
            AbortContext(),
        )
    assert validate_error.value.code == grpc.StatusCode.INVALID_ARGUMENT

    with pytest.raises(GrpcAbort) as register_error:
        await service.Register(
            pb.RegisterRequest(
                activation_id=ACTIVATION_ID,
                plugin_id="tests.python_worker",
                auth_token=AUTH_TOKEN,
                config=bad_config,
            ),
            AbortContext(),
        )
    assert register_error.value.code == grpc.StatusCode.INVALID_ARGUMENT


async def test_base_plugin_defaults_context_errors_and_plugin_id_validation() -> None:
    base = WorkerPlugin()
    assert base.validate({"unused": True}) == []
    with pytest.raises(NotImplementedError):
        base.register(PluginContext(), {})
    with pytest.raises(WorkerSdkError, match="no runtime handle"):
        _ = PluginContext().runtime

    class CallablePluginId(WorkerPlugin):
        allows_multiple_components = True

        def plugin_id(self) -> str:
            return "tests.callable_id"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config

    class InvalidPluginId(WorkerPlugin):
        plugin_id = ""

        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config

    callable_service = _service(CallablePluginId(), RecordingHostStub())
    health = await callable_service.Health(
        pb.HealthRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
        AbortContext(),
    )
    assert health.plugin_id == "tests.callable_id"

    invalid_service = _service(InvalidPluginId(), RecordingHostStub())
    with pytest.raises(WorkerSdkError, match="plugin_id"):
        await invalid_service.Health(
            pb.HealthRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            AbortContext(),
        )


async def test_validate_accepts_missing_config_and_dict_diagnostics() -> None:
    class DictDiagnosticPlugin(WorkerPlugin):
        plugin_id = "tests.dict_diagnostic"

        def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
            assert config is None
            return [{"level": "error", "code": "dict.diag", "message": "dict diagnostic"}]

        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config

    service = _service(DictDiagnosticPlugin(), RecordingHostStub())
    response = await service.Validate(
        pb.ValidateRequest(
            activation_id=ACTIVATION_ID,
            plugin_id="tests.dict_diagnostic",
            auth_token=AUTH_TOKEN,
        ),
        AbortContext(),
    )
    assert _envelope_value(response.diagnostics) == [
        {"level": "error", "code": "dict.diag", "message": "dict diagnostic"}
    ]


async def test_validate_register_and_invoke_callback_errors_are_structured() -> None:
    class FailingValidatePlugin(WorkerPlugin):
        plugin_id = "tests.failing_validate"

        def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
            del config
            raise RuntimeError("validate boom")

        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config

    class FailingRegisterPlugin(WorkerPlugin):
        plugin_id = "tests.failing_register"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config
            raise RuntimeError("register boom")

    class FailingInvokePlugin(WorkerPlugin):
        plugin_id = "tests.failing_invoke"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def fail(tool_name: str, args: Json) -> Json:
                del tool_name, args
                raise RuntimeError("invoke boom")

            ctx.register_tool_request_intercept("fail", fail)

    validate_service = _service(FailingValidatePlugin(), RecordingHostStub())
    validate = await validate_service.Validate(_validate_request(), AbortContext())
    assert validate.HasField("error")
    assert "validate boom" in validate.error.message

    register_service = _service(FailingRegisterPlugin(), RecordingHostStub())
    register = await register_service.Register(_register_request(), AbortContext())
    assert register.HasField("error")
    assert "register boom" in register.error.message

    invoke_service = _service(FailingInvokePlugin(), RecordingHostStub())
    await _register(invoke_service)
    response = await invoke_service.Invoke(_tool_request("fail", pb.TOOL_REQUEST_INTERCEPT, {}), AbortContext())
    assert response.WhichOneof("result") == "error"
    assert "invoke boom" in response.error.message


async def test_unary_invoke_success_paths(service: _WorkerService, host_stub: RecordingHostStub) -> None:
    await _register(service)

    subscriber = await service.Invoke(
        _invoke_request(
            "subscriber",
            pb.SUBSCRIBER,
            event=_json_envelope(EVENT_SCHEMA, {"name": "event"}),
            scope=pb.ScopeContext(scope_stack_id="invoke-stack", parent_scope_id="parent-scope"),
        ),
        AbortContext(),
    )
    assert subscriber.WhichOneof("result") == "empty"
    mark_request = _last_request(host_stub, pb.EmitMarkRequest)
    assert mark_request.name == "tests.subscriber"
    assert mark_request.scope.scope_stack_id == "invoke-stack"
    assert mark_request.scope.parent_scope_id == "parent-scope"

    tool_sanitize_request = await _invoke_json_async(service, "tool_sanitize", pb.TOOL_SANITIZE_REQUEST_GUARDRAIL)
    assert tool_sanitize_request["tag"] == "sanitize_lookup"
    tool_sanitize_response = await _invoke_json_async(service, "tool_sanitize", pb.TOOL_SANITIZE_RESPONSE_GUARDRAIL)
    assert tool_sanitize_response["tag"] == "sanitize_lookup"

    tool_conditional = await service.Invoke(
        _tool_request("tool_conditional", pb.TOOL_CONDITIONAL_EXECUTION_GUARDRAIL, {"query": "relay"}),
        AbortContext(),
    )
    assert tool_conditional.guardrail.block_reason == "tool blocked"

    tool_request = await _invoke_json_async(service, "tool_request", pb.TOOL_REQUEST_INTERCEPT)
    assert tool_request["tag"] == "request_lookup"
    tool_execution = await _invoke_json_async(service, "tool_execution", pb.TOOL_EXECUTION_INTERCEPT)
    assert tool_execution["tag"] == "tool_execution"
    assert tool_execution["next_tool"]["tag"] == "execute_lookup"

    llm_sanitize_request = await _invoke_json_async(
        service,
        "llm_sanitize_request",
        pb.LLM_SANITIZE_REQUEST_GUARDRAIL,
        payload=_llm_payload(request={"content": {"prompt": "hello"}}),
    )
    assert llm_sanitize_request["content"]["llm_sanitize_request"]

    llm_sanitize_response = await _invoke_json_async(
        service,
        "llm_sanitize_response",
        pb.LLM_SANITIZE_RESPONSE_GUARDRAIL,
        payload=_llm_payload(response={"answer": "hello"}),
    )
    assert llm_sanitize_response["tag"] == "llm_sanitize_response"

    llm_conditional = await service.Invoke(
        _invoke_request(
            "llm_conditional",
            pb.LLM_CONDITIONAL_EXECUTION_GUARDRAIL,
            llm=_llm_payload(request={"content": {"prompt": "hello"}}),
        ),
        AbortContext(),
    )
    assert llm_conditional.guardrail.block_reason == "llm blocked"

    llm_request = await service.Invoke(
        _invoke_request(
            "llm_request",
            pb.LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(
                request={"content": {"prompt": "hello"}},
                annotated={"messages": [], "extra": {"before": True}},
            ),
        ),
        AbortContext(),
    )
    assert _envelope_value(llm_request.llm_request.request)["content"]["llm_request"]
    assert _envelope_value(llm_request.llm_request.annotated_request)["tag"] == "annotated"
    assert llm_request.llm_request.has_annotated_request

    llm_execution = await _invoke_json_async(
        service,
        "llm_execution",
        pb.LLM_EXECUTION_INTERCEPT,
        payload=_llm_payload(model_name="gpt-test", request={"content": {"prompt": "hello"}}),
    )
    assert llm_execution["tag"] == "llm_execution"
    assert llm_execution["next_llm"]["content"]["llm_execute_gpt-test"]


async def test_unary_invoke_failure_paths(service: _WorkerService) -> None:
    await _register(service)

    invalid = _tool_request("tool_request", pb.TOOL_REQUEST_INTERCEPT, {})
    invalid.tool.value.json = b"{"
    invalid_payload = await service.Invoke(invalid, AbortContext())
    assert invalid_payload.WhichOneof("result") == "error"
    assert "Expecting" in invalid_payload.error.message

    missing_handler = await service.Invoke(_tool_request("missing", pb.TOOL_REQUEST_INTERCEPT, {}), AbortContext())
    assert missing_handler.WhichOneof("result") == "error"
    assert "not registered" in missing_handler.error.message

    unsupported = await service.Invoke(
        _tool_request("tool_request", pb.REGISTRATION_SURFACE_UNSPECIFIED, {}),
        AbortContext(),
    )
    assert unsupported.WhichOneof("result") == "error"
    assert "unsupported registration surface" in unsupported.error.message

    missing_event = await service.Invoke(_invoke_request("subscriber", pb.SUBSCRIBER), AbortContext())
    assert missing_event.WhichOneof("result") == "error"
    assert "event is missing" in missing_event.error.message


async def test_llm_request_intercept_can_return_request_without_annotation() -> None:
    class RequestOnlyPlugin(WorkerPlugin):
        plugin_id = "tests.request_only"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def llm_request(name: str, request: Json, annotated: Json | None) -> Json:
                del name, annotated
                return _tag_llm_request(request, "request_only")

            ctx.register_llm_request_intercept("request_only", llm_request)

    service = _service(RequestOnlyPlugin(), RecordingHostStub())
    await _register(service)
    response = await service.Invoke(
        _invoke_request(
            "request_only",
            pb.LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(request={"content": {"prompt": "hello"}}),
        ),
        AbortContext(),
    )
    assert _envelope_value(response.llm_request.request)["content"]["request_only"]
    assert not response.llm_request.has_annotated_request


async def test_stream_invoke_success_and_failures(service: _WorkerService, host_stub: RecordingHostStub) -> None:
    await _register(service)

    chunks = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "llm_stream_execution",
                pb.LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(model_name="gpt-test", request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert [_stream_value(chunk)["tag"] for chunk in chunks] == ["llm_stream_execution"]
    assert _stream_value(chunks[0])["next_stream"]["content"]["llm_stream_gpt-test"]

    wrong_surface = [
        chunk
        async for chunk in service.InvokeStream(
            _tool_request("tool_request", pb.TOOL_REQUEST_INTERCEPT, {}),
            AbortContext(),
        )
    ]
    assert "only supports LLM stream" in wrong_surface[0].error.message

    missing_handler = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "missing",
                pb.LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert "not registered" in missing_handler[0].error.message

    missing_payload = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request("llm_stream_execution", pb.LLM_STREAM_EXECUTION_INTERCEPT),
            AbortContext(),
        )
    ]
    assert "expected llm payload" in missing_payload[0].error.message

    host_stub.failures["LlmStreamNext"] = "error"
    host_error = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "llm_stream_execution",
                pb.LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert "LlmStreamNext failed" in host_error[0].error.message

    host_stub.failures["LlmStreamNext"] = "empty"
    empty_chunk = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "llm_stream_execution",
                pb.LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert "stream chunk is empty" in empty_chunk[0].error.message


async def test_stream_callback_exception_is_structured() -> None:
    class FailingStreamPlugin(WorkerPlugin):
        plugin_id = "tests.stream_fail"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def fail(name: str, request: Json, next_call: Any) -> AsyncIterator[Json]:
                del name, request, next_call
                raise RuntimeError("stream boom")
                yield {}

            ctx.register_llm_stream_execution_intercept("fail_stream", fail)

    service = _service(FailingStreamPlugin(), RecordingHostStub())
    await _register(service)
    chunks = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "fail_stream",
                pb.LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert "stream boom" in chunks[0].error.message


async def test_stream_callback_can_return_sync_iterable() -> None:
    class SyncStreamPlugin(WorkerPlugin):
        plugin_id = "tests.sync_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def stream(name: str, request: Json, next_call: Any) -> list[Json]:
                del name, request, next_call
                return [{"sync": True}]

            ctx.register_llm_stream_execution_intercept("sync_stream", stream)

    service = _service(SyncStreamPlugin(), RecordingHostStub())
    await _register(service)
    chunks = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "sync_stream",
                pb.LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert _stream_value(chunks[0]) == {"sync": True}


async def test_runtime_host_calls_and_scope_context(host_stub: RecordingHostStub) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    assert runtime.current_scope_stack_id() is None
    assert runtime.current_parent_scope_id() is None

    stack_id = await runtime.create_scope_stack()
    assert stack_id == "stack-1"
    with runtime.bind_scope_stack(stack_id, parent_scope_id="parent-1"):
        assert runtime.current_scope_stack_id() == stack_id
        assert runtime.current_parent_scope_id() == "parent-1"
        await runtime.emit_mark("mark", {"ok": True})
        scope_id = await runtime.push_scope("scope", scope_type=ScopeType.TOOL, input={"in": True})
        await runtime.pop_scope(scope_id, output={"out": True})
        tool_next = await ToolNext(runtime, "tool-next").call({"value": 1})
        llm_next = await _llm_next(runtime, {"content": {"prompt": "hello"}})
        stream_next = [chunk async for chunk in _llm_stream_next(runtime, {"content": {"prompt": "hello"}})]
        with runtime.clear_scope_stack():
            assert runtime.current_scope_stack_id() is None
            assert runtime.current_parent_scope_id() is None
        assert runtime.current_scope_stack_id() == stack_id
    assert runtime.current_scope_stack_id() is None
    assert tool_next["next_tool"]["value"] == 1
    assert llm_next["next_llm"]["content"]["prompt"] == "hello"
    assert stream_next[0]["next_stream"]["content"]["prompt"] == "hello"

    await runtime.emit_mark("explicit", scope_stack_id="explicit-stack", parent_scope_id="explicit-parent")
    await runtime.drop_scope_stack(stack_id)
    mark_request = _last_request(host_stub, pb.EmitMarkRequest)
    assert mark_request.scope.scope_stack_id == "explicit-stack"
    assert mark_request.scope.parent_scope_id == "explicit-parent"
    push_request = _last_request(host_stub, pb.PushScopeRequest)
    assert push_request.scope.scope_stack_id == "stack-1"
    assert push_request.scope.parent_scope_id == "parent-1"


async def test_runtime_host_call_error_paths(host_stub: RecordingHostStub) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)

    host_stub.failures["EmitMark"] = "error"
    with pytest.raises(WorkerSdkError, match="EmitMark failed"):
        await runtime.emit_mark("mark")

    host_stub.failures["EmitMark"] = "empty"
    with pytest.raises(WorkerSdkError, match="host call failed"):
        await runtime.emit_mark("mark")

    host_stub.failures["CreateScopeStack"] = "error"
    with pytest.raises(WorkerSdkError, match="CreateScopeStack failed"):
        await runtime.create_scope_stack()

    host_stub.failures["PushScope"] = "error"
    with pytest.raises(WorkerSdkError, match="PushScope failed"):
        await runtime.push_scope("scope")

    host_stub.failures["PopScope"] = "error"
    with pytest.raises(WorkerSdkError, match="PopScope failed"):
        await runtime.pop_scope("scope")

    host_stub.failures["DropScopeStack"] = "error"
    with pytest.raises(WorkerSdkError, match="DropScopeStack failed"):
        await runtime.drop_scope_stack("stack")

    host_stub.failures["ToolNext"] = "error"
    with pytest.raises(WorkerSdkError, match="ToolNext failed"):
        await ToolNext(runtime, "tool-next").call({"value": 1})

    host_stub.failures["LlmNext"] = "error"
    with pytest.raises(WorkerSdkError, match="LlmNext failed"):
        await _llm_next(runtime, {"content": {}})

    host_stub.failures["LlmStreamNext"] = "error"
    with pytest.raises(WorkerSdkError, match="LlmStreamNext failed"):
        async for _chunk in _llm_stream_next(runtime, {"content": {}}):
            pass

    host_stub.failures["LlmStreamNext"] = "empty"
    with pytest.raises(WorkerSdkError, match="stream chunk is empty"):
        async for _chunk in _llm_stream_next(runtime, {"content": {}}):
            pass


async def test_lifecycle_acks(service: _WorkerService) -> None:
    cancel = await service.CancelInvocation(
        pb.CancelInvocationRequest(
            activation_id=ACTIVATION_ID,
            invocation_id="invoke-1",
            auth_token=AUTH_TOKEN,
            reason="test",
        ),
        AbortContext(),
    )
    assert not cancel.accepted
    assert "not implemented" in cancel.message

    shutdown = await service.Shutdown(
        pb.ShutdownRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, reason="test"),
        AbortContext(),
    )
    assert shutdown.accepted
    assert "shutdown accepted" in shutdown.message


def test_required_environment_reports_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_RELAY_WORKER_SOCKET", raising=False)
    with pytest.raises(Exception, match="NEMO_RELAY_WORKER_SOCKET"):
        _required_env("NEMO_RELAY_WORKER_SOCKET")


def test_endpoint_helpers_normalize_announce_and_unlink_tcp_and_unix_targets(tmp_path: Any) -> None:
    assert _grpc_target("tcp://127.0.0.1:50051") == "127.0.0.1:50051"
    assert _grpc_target("http://127.0.0.1:50051") == "127.0.0.1:50051"
    assert _grpc_target("unix:///tmp/worker.sock") == "unix:/tmp/worker.sock"
    assert _announced_worker_endpoint("tcp://127.0.0.1:0", 43123) == "http://127.0.0.1:43123"
    assert _announced_worker_endpoint("http://127.0.0.1:50051", 43123) == "http://127.0.0.1:50051"
    assert _announced_worker_endpoint("127.0.0.1:50051", 43123) == "http://127.0.0.1:50051"
    assert _announced_worker_endpoint("unix:///tmp/worker.sock", 43123) == "unix:///tmp/worker.sock"
    assert _announced_worker_endpoint("worker-endpoint", 43123) == "http://worker-endpoint"

    socket_path = tmp_path / "worker.sock"
    socket_path.write_text("", encoding="utf-8")
    _unlink_unix_socket(f"unix://{socket_path}")
    assert not socket_path.exists()


async def test_serve_plugin_announces_tcp_endpoint_and_accepts_health_shutdown(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_file = tmp_path / "endpoint.txt"
    monkeypatch.setenv("NEMO_RELAY_WORKER_SOCKET", "tcp://127.0.0.1:0")
    monkeypatch.setenv("NEMO_RELAY_HOST_SOCKET", "http://127.0.0.1:9")
    monkeypatch.setenv("NEMO_RELAY_WORKER_ID", ACTIVATION_ID)
    monkeypatch.setenv("NEMO_RELAY_WORKER_TOKEN", AUTH_TOKEN)
    monkeypatch.setenv("NEMO_RELAY_WORKER_ENDPOINT_FILE", str(endpoint_file))

    task = asyncio.create_task(serve_plugin(AllSurfacesPlugin()))
    channel = None
    try:
        for _ in range(100):
            if endpoint_file.exists():
                break
            await asyncio.sleep(0.05)
        assert endpoint_file.exists()
        endpoint = endpoint_file.read_text(encoding="utf-8")
        assert endpoint.startswith("http://127.0.0.1:")

        channel = grpc.aio.insecure_channel(_grpc_target(endpoint))
        stub = pb_grpc.PluginWorkerStub(channel)
        health = await stub.Health(pb.HealthRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN), timeout=5)
        assert health.ok
        shutdown = await stub.Shutdown(
            pb.ShutdownRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, reason="test"),
            timeout=5,
        )
        assert shutdown.accepted
        await asyncio.wait_for(task, timeout=5)
    finally:
        if channel is not None:
            await channel.close()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def _service(plugin: WorkerPlugin, host_stub: RecordingHostStub) -> _WorkerService:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    return _WorkerService(plugin, runtime, asyncio.Event())


def _worker_error(message: str) -> Any:
    return pb.WorkerError(code="test.error", message=message, retryable=False)


def _tag(value: Json, tag: str) -> Json:
    if isinstance(value, dict):
        return {**value, "tag": tag}
    return {"value": value, "tag": tag}


def _tag_llm_request(request: Json, tag: str) -> Json:
    request = dict(request)
    content = request.get("content")
    if isinstance(content, dict):
        request["content"] = {**content, tag: True}
    else:
        request["content"] = {"value": content, tag: True}
    return request


def _handshake_request() -> Any:
    return pb.HandshakeRequest(
        activation_id=ACTIVATION_ID,
        plugin_id="tests.python_worker",
        relay_version="0.5.0",
        worker_protocol=WORKER_PROTOCOL,
        auth_token=AUTH_TOKEN,
        host_endpoint="http://127.0.0.1:9",
    )


def _validate_request(config: Json | None = None) -> Any:
    return pb.ValidateRequest(
        activation_id=ACTIVATION_ID,
        plugin_id="tests.python_worker",
        auth_token=AUTH_TOKEN,
        config=_json_envelope(JSON_SCHEMA, {} if config is None else config),
    )


def _register_request(config: Json | None = None) -> Any:
    return pb.RegisterRequest(
        activation_id=ACTIVATION_ID,
        plugin_id="tests.python_worker",
        auth_token=AUTH_TOKEN,
        config=_json_envelope(JSON_SCHEMA, {} if config is None else config),
    )


async def _register(service: _WorkerService) -> Any:
    response = await service.Register(_register_request(), AbortContext())
    assert not response.HasField("error"), response.error
    return response


def _invoke_request(registration_name: str, surface: int, **kwargs: Any) -> Any:
    return pb.InvokeRequest(
        activation_id=ACTIVATION_ID,
        invocation_id="invoke-1",
        registration_name=registration_name,
        surface=surface,
        continuation_id="next-1",
        auth_token=AUTH_TOKEN,
        **kwargs,
    )


def _tool_request(registration_name: str, surface: int, value: Json) -> Any:
    return _invoke_request(
        registration_name,
        surface,
        tool=pb.ToolInvocation(tool_name="lookup", value=_json_envelope(JSON_SCHEMA, value)),
    )


def _llm_payload(
    *,
    model_name: str = "model",
    request: Json | None = None,
    response: Json | None = None,
    annotated: Json | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "request": _json_envelope(LLM_REQUEST_SCHEMA, request or {"content": {}}),
        "response": _json_envelope(JSON_SCHEMA, response or {}),
    }
    if annotated is not None:
        kwargs["annotated_request"] = _json_envelope(ANNOTATED_LLM_REQUEST_SCHEMA, annotated)
    return pb.LlmInvocation(**kwargs)


async def _invoke_json_async(
    service: _WorkerService,
    registration_name: str,
    surface: int,
    *,
    payload: Any | None = None,
) -> Json:
    if payload is None:
        request = _tool_request(registration_name, surface, {"query": "relay"})
    else:
        request = _invoke_request(registration_name, surface, llm=payload)
    response = await service.Invoke(request, AbortContext())
    assert response.WhichOneof("result") == "json", response
    return _envelope_value(response.json.value)


def _envelope_value(envelope: Any) -> Json:
    return json.loads(envelope.json.decode("utf-8"))


def _stream_value(chunk: Any) -> Json:
    assert chunk.WhichOneof("item") == "value", chunk
    return _envelope_value(chunk.value)


def _last_request(host_stub: RecordingHostStub, request_type: Any) -> Any:
    return next(request for request in reversed(host_stub.requests) if isinstance(request, request_type))


async def _llm_next(runtime: PluginRuntime, request: Json) -> Json:
    from nemo_relay_plugin import LlmNext

    return await LlmNext(runtime, "llm-next").call(request)


def _llm_stream_next(runtime: PluginRuntime, request: Json) -> AsyncIterator[Json]:
    from nemo_relay_plugin import LlmStreamNext

    return LlmStreamNext(runtime, "llm-stream-next").call(request)


def _all_expected_surfaces() -> list[int]:
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
