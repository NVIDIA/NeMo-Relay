# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python dynamic worker plugin SDK."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, cast
from unittest import mock

import pytest

if os.environ.get("NEMO_RELAY_SKIP_PYTHON_PLUGIN_TESTS") == "1":
    pytest.skip("grpcio is unavailable for Python plugin SDK tests on this runner", allow_module_level=True)

grpc = pytest.importorskip("grpc")

from nemo_relay_plugin import (  # noqa: E402
    ConfigDiagnostic,
    DiagnosticLevel,
    Json,
    LlmRequestInterceptOutcome,
    PendingMarkSpec,
    PluginContext,
    PluginRuntime,
    ScopeType,
    ToolNext,
    WorkerPlugin,
    WorkerSdkError,
    serve_plugin,
)
from nemo_relay_plugin import _api as plugin_api  # noqa: E402
from nemo_relay_plugin._api import (  # noqa: E402
    ANNOTATED_LLM_REQUEST_SCHEMA,
    EVENT_SCHEMA,
    JSON_SCHEMA,
    LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA,
    LLM_REQUEST_SCHEMA,
    WORKER_PROTOCOL,
    _announced_worker_endpoint,
    _decode_required_envelope,
    _grpc_target,
    _json_envelope,
    _open_host_channel,
    _required_env,
    _unlink_unix_socket,
    _WorkerService,
    _write_endpoint_file,
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


class _ContendedAsyncLock(asyncio.Lock):
    def __init__(self, expected_waiters: int) -> None:
        super().__init__()
        self._expected_waiters = expected_waiters
        self._acquire_attempts = 0
        self._all_waiting = asyncio.Event()

    async def acquire(self) -> Literal[True]:
        self._acquire_attempts += 1
        if self._acquire_attempts == self._expected_waiters:
            self._all_waiting.set()
        return await super().acquire()

    async def hold(self) -> None:
        await super().acquire()

    async def wait_until_contended(self) -> None:
        await asyncio.wait_for(self._all_waiting.wait(), timeout=5)


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

        def llm_request(name: str, request: Json, annotated: Json | None) -> LlmRequestInterceptOutcome:
            del name
            return LlmRequestInterceptOutcome(
                request=_tag_llm_request(request, "llm_request"),
                annotated_request=_tag(annotated, "annotated") if annotated is not None else None,
                pending_marks=[PendingMarkSpec("worker.pending", data={"source": "python"})],
            )

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


def test_generated_proto_matches_worker_contract():
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


async def test_health_handshake_validate_register_and_all_surfaces(service: _WorkerService):
    health = await service.Health(pb.HealthRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN), AbortContext())
    assert health.ok
    assert health.plugin_id == "tests.python_worker"
    assert health.worker_protocol == WORKER_PROTOCOL
    assert health.sdk_name == "nemo-relay-plugin"
    assert health.sdk_version == plugin_api._SDK_VERSION
    assert health.runtime_name == "python"

    handshake = await service.Handshake(_handshake_request(), AbortContext())
    assert handshake.plugin_id == "tests.python_worker"
    assert handshake.plugin_kind == "tests.python_worker"
    assert handshake.worker_protocol == WORKER_PROTOCOL
    assert handshake.sdk_version == plugin_api._SDK_VERSION
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
    registrations = [
        (registration.local_name, registration.surface, registration.priority, registration.break_chain)
        for registration in register.registrations
    ]
    assert registrations == [
        ("subscriber", pb.SUBSCRIBER, 0, False),
        ("tool_sanitize", pb.TOOL_SANITIZE_REQUEST_GUARDRAIL, 1, False),
        ("tool_sanitize", pb.TOOL_SANITIZE_RESPONSE_GUARDRAIL, 2, False),
        ("tool_conditional", pb.TOOL_CONDITIONAL_EXECUTION_GUARDRAIL, 3, False),
        ("tool_request", pb.TOOL_REQUEST_INTERCEPT, 4, True),
        ("tool_execution", pb.TOOL_EXECUTION_INTERCEPT, 5, False),
        ("llm_sanitize_request", pb.LLM_SANITIZE_REQUEST_GUARDRAIL, 6, False),
        ("llm_sanitize_response", pb.LLM_SANITIZE_RESPONSE_GUARDRAIL, 7, False),
        ("llm_conditional", pb.LLM_CONDITIONAL_EXECUTION_GUARDRAIL, 8, False),
        ("llm_request", pb.LLM_REQUEST_INTERCEPT, 9, True),
        ("llm_execution", pb.LLM_EXECUTION_INTERCEPT, 10, False),
        ("llm_stream_execution", pb.LLM_STREAM_EXECUTION_INTERCEPT, 11, False),
    ]


def test_sdk_version_uses_package_metadata_and_source_tree_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
):
    monkeypatch.setattr(plugin_api.metadata, "version", lambda package: "1.2.3")
    assert plugin_api._sdk_version() == "1.2.3"

    def package_not_found(package: str) -> str:
        raise plugin_api.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(plugin_api.metadata, "version", package_not_found)
    source_file = tmp_path / "src" / "nemo_relay_plugin" / "_api.py"
    monkeypatch.setattr(plugin_api, "__file__", str(source_file))
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "2.0.0rc1"\n', encoding="utf-8")
    assert plugin_api._sdk_version() == "2.0.0rc1"

    (tmp_path / "pyproject.toml").unlink()
    assert plugin_api._sdk_version() == "0+unknown"


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        {"nested": [float("nan")]},
    ],
)
def test_json_envelope_rejects_non_finite_numbers(value: Json):
    with pytest.raises(ValueError, match="Out of range float values"):
        _json_envelope(JSON_SCHEMA, value)


@pytest.mark.parametrize(
    "value",
    [
        {1: "value"},
        {"nested": [{2: True}]},
    ],
)
def test_json_envelope_rejects_non_string_object_keys(value: Json):
    with pytest.raises(WorkerSdkError, match="JSON object keys must be strings"):
        _json_envelope(JSON_SCHEMA, value)


def test_json_envelope_rejects_cycles_and_allows_shared_subobjects():
    dict_cycle: Json = {}
    dict_cycle["self"] = dict_cycle
    list_cycle: Json = []
    list_cycle.append(list_cycle)

    for value in (dict_cycle, list_cycle):
        with pytest.raises(WorkerSdkError, match="must not contain circular references"):
            _json_envelope(JSON_SCHEMA, value)

    shared = {"value": True}
    envelope = _json_envelope(JSON_SCHEMA, {"first": shared, "second": shared})
    assert _envelope_value(envelope) == {
        "first": {"value": True},
        "second": {"value": True},
    }


@pytest.mark.parametrize("payload", [b"NaN", b"Infinity", b"-Infinity", b'{"nested":NaN}'])
def test_json_envelope_decode_rejects_non_standard_constants(payload: bytes):
    envelope = pb.JsonEnvelope(schema=JSON_SCHEMA, json=payload)
    with pytest.raises(WorkerSdkError, match="non-standard JSON constant"):
        _decode_required_envelope(envelope, "json value")


@pytest.mark.parametrize("payload", [b"{", b"\xff"])
def test_json_envelope_decode_normalizes_malformed_payload_errors(payload: bytes):
    envelope = pb.JsonEnvelope(schema=JSON_SCHEMA, json=payload)
    with pytest.raises(WorkerSdkError, match="json value contains invalid JSON"):
        _decode_required_envelope(envelope, "json value")


@pytest.mark.parametrize(
    "schema",
    [EVENT_SCHEMA, LLM_REQUEST_SCHEMA, ANNOTATED_LLM_REQUEST_SCHEMA],
)
@pytest.mark.parametrize("value", [None, [], "scalar", 42])
def test_object_schema_envelopes_reject_non_objects(schema: str, value: Json):
    with pytest.raises(WorkerSdkError, match="must be a JSON object"):
        _json_envelope(schema, value)

    envelope = pb.JsonEnvelope(schema=schema, json=json.dumps(value).encode("utf-8"))
    with pytest.raises(WorkerSdkError, match="must be a JSON object"):
        _decode_required_envelope(envelope, "typed value", schema)

    assert _envelope_value(_json_envelope(JSON_SCHEMA, value)) == value


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
):
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


async def test_validate_and_register_decode_errors_are_grpc_protocol_errors(service: _WorkerService):
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

    wrong_schema = _json_envelope(EVENT_SCHEMA, {})
    with pytest.raises(GrpcAbort) as schema_error:
        await service.Validate(
            pb.ValidateRequest(
                activation_id=ACTIVATION_ID,
                plugin_id="tests.python_worker",
                auth_token=AUTH_TOKEN,
                config=wrong_schema,
            ),
            AbortContext(),
        )
    assert schema_error.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "expected 'nemo.relay.Json@1'" in schema_error.value.details


async def test_base_plugin_defaults_context_errors_and_plugin_id_validation():
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


def test_plugin_context_rejects_duplicate_names_on_the_same_surface():
    context = PluginContext()

    def callback(tool_name: str, value: Json) -> Json:
        del tool_name
        return value

    context.register_tool_request_intercept("duplicate", callback)
    with pytest.raises(WorkerSdkError, match="already registered"):
        context.register_tool_request_intercept("duplicate", callback)

    context.register_tool_sanitize_request_guardrail("shared", callback)
    context.register_tool_sanitize_response_guardrail("shared", callback)
    registrations = [
        (registration.local_name, registration.surface) for registration in context._handlers.registrations
    ]
    assert registrations.count(("duplicate", pb.TOOL_REQUEST_INTERCEPT)) == 1
    assert ("shared", pb.TOOL_SANITIZE_REQUEST_GUARDRAIL) in registrations
    assert ("shared", pb.TOOL_SANITIZE_RESPONSE_GUARDRAIL) in registrations


async def test_validate_accepts_missing_config_and_dict_diagnostics():
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


def test_config_diagnostic_normalizes_string_level_and_optional_fields():
    diagnostic = ConfigDiagnostic(
        level="warning",
        code="tests.warning",
        message="warning",
        component="tests.plugin",
    )
    assert diagnostic.to_json() == {
        "level": "warning",
        "code": "tests.warning",
        "message": "warning",
        "component": "tests.plugin",
    }


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        ({"level": "info", "code": "tests.info", "message": "info"}, "diagnostic level"),
        ({"level": "warning", "message": "missing code"}, "diagnostic code"),
        ({"level": "warning", "code": "tests.missing_message"}, "diagnostic message"),
        (
            {"level": "warning", "code": "tests.component", "message": "component", "component": 42},
            "diagnostic component",
        ),
        (ConfigDiagnostic(level="info", code="tests.info", message="info"), "diagnostic level"),
    ],
)
async def test_validate_rejects_malformed_diagnostics(diagnostic: Any, message: str):
    class MalformedDiagnosticPlugin(WorkerPlugin):
        plugin_id = "tests.malformed_diagnostic"

        def validate(self, config: Json) -> Any:
            del config
            return [diagnostic]

        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config

    service = _service(MalformedDiagnosticPlugin(), RecordingHostStub())
    response = await service.Validate(
        _validate_request(plugin_id="tests.malformed_diagnostic"),
        AbortContext(),
    )
    assert response.HasField("error")
    assert response.error.code == "worker.sdk_error"
    assert message in response.error.message


async def test_async_validate_and_register_hooks_are_awaited():
    lifecycle: list[str] = []

    class AsyncLifecyclePlugin(WorkerPlugin):
        plugin_id = "tests.async_lifecycle"

        async def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
            await asyncio.sleep(0)
            lifecycle.append("validate")
            return [{"level": "warning", "code": "async.validate", "message": str(config)}]

        async def register(self, ctx: PluginContext, config: Json) -> None:
            await asyncio.sleep(0)
            lifecycle.append("register")

            def request(tool_name: str, args: Json) -> Json:
                return {"tool_name": tool_name, "args": args, "config": config}

            ctx.register_tool_request_intercept("async_request", request)

    service = _service(AsyncLifecyclePlugin(), RecordingHostStub())
    validate = await service.Validate(
        _validate_request(plugin_id="tests.async_lifecycle", config={"mode": "async"}),
        AbortContext(),
    )
    assert _envelope_value(validate.diagnostics) == [
        {
            "level": "warning",
            "code": "async.validate",
            "message": "{'mode': 'async'}",
        }
    ]

    register = await service.Register(
        _register_request(plugin_id="tests.async_lifecycle", config={"mode": "async"}),
        AbortContext(),
    )
    assert [(item.local_name, item.surface) for item in register.registrations] == [
        ("async_request", pb.TOOL_REQUEST_INTERCEPT)
    ]
    response = await service.Invoke(
        _tool_request("async_request", pb.TOOL_REQUEST_INTERCEPT, {"query": "relay"}),
        AbortContext(),
    )
    assert response.WhichOneof("result") == "json", response
    assert _envelope_value(response.json.value) == {
        "tool_name": "lookup",
        "args": {"query": "relay"},
        "config": {"mode": "async"},
    }
    assert lifecycle == ["validate", "register"]


async def test_validate_register_and_invoke_callback_errors_are_structured():
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
    validate = await validate_service.Validate(
        _validate_request(plugin_id="tests.failing_validate"),
        AbortContext(),
    )
    assert validate.HasField("error")
    assert "validate boom" in validate.error.message

    register_service = _service(FailingRegisterPlugin(), RecordingHostStub())
    register = await register_service.Register(
        _register_request(plugin_id="tests.failing_register"),
        AbortContext(),
    )
    assert register.HasField("error")
    assert "register boom" in register.error.message

    invoke_service = _service(FailingInvokePlugin(), RecordingHostStub())
    await _register(invoke_service)
    response = await invoke_service.Invoke(_tool_request("fail", pb.TOOL_REQUEST_INTERCEPT, {}), AbortContext())
    assert response.WhichOneof("result") == "error"
    assert "invoke boom" in response.error.message


async def test_register_is_idempotent_and_rejects_changed_component_config():
    class ConfigPlugin(WorkerPlugin):
        plugin_id = "tests.register_config"
        allows_multiple_components = True

        def __init__(self) -> None:
            self.register_calls = 0

        def register(self, ctx: PluginContext, config: Json) -> None:
            self.register_calls += 1
            assert isinstance(config, dict)
            tag = config["tag"]
            config["mutated_by_plugin"] = True

            def callback(tool_name: str, value: Json) -> Json:
                del tool_name
                assert isinstance(value, dict)
                return {**value, "tag": tag}

            ctx.register_tool_request_intercept("configured", callback)

    plugin = ConfigPlugin()
    service = _service(plugin, RecordingHostStub())

    first = await service.Register(
        _register_request({"tag": "first"}, plugin_id="tests.register_config"),
        AbortContext(),
    )
    repeated = await service.Register(
        _register_request({"tag": "first"}, plugin_id="tests.register_config"),
        AbortContext(),
    )
    changed = await service.Register(
        _register_request({"tag": "second"}, plugin_id="tests.register_config"),
        AbortContext(),
    )

    assert plugin.register_calls == 1
    assert list(repeated.registrations) == list(first.registrations)
    assert changed.HasField("error")
    assert "different component config" in changed.error.message

    result = await service.Invoke(
        _tool_request("configured", pb.TOOL_REQUEST_INTERCEPT, {"query": "relay"}),
        AbortContext(),
    )
    assert _envelope_value(result.json.value) == {"query": "relay", "tag": "first"}


async def test_concurrent_register_calls_install_handlers_once():
    class CountingPlugin(WorkerPlugin):
        plugin_id = "tests.concurrent_register"

        def __init__(self) -> None:
            self.register_calls = 0

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config
            self.register_calls += 1
            ctx.register_subscriber("subscriber", lambda event: None)

    plugin = CountingPlugin()
    service = _service(plugin, RecordingHostStub())
    registration_lock = _ContendedAsyncLock(expected_waiters=2)
    await registration_lock.hold()
    service._registration_lock = registration_lock
    requests = [
        asyncio.create_task(
            service.Register(
                _register_request(plugin_id=plugin.plugin_id),
                AbortContext(),
            )
        )
        for _ in range(2)
    ]
    await registration_lock.wait_until_contended()
    registration_lock.release()
    responses = await asyncio.gather(*requests)

    assert plugin.register_calls == 1
    assert all(not response.HasField("error") for response in responses)
    assert [registration.local_name for registration in responses[0].registrations] == ["subscriber"]
    assert list(responses[1].registrations) == list(responses[0].registrations)


async def test_concurrent_register_calls_reject_a_different_config():
    plugin = AllSurfacesPlugin()
    service = _service(plugin, RecordingHostStub())
    registration_lock = _ContendedAsyncLock(expected_waiters=2)
    await registration_lock.hold()
    service._registration_lock = registration_lock
    requests = [
        asyncio.create_task(
            service.Register(
                _register_request({"component": component}, plugin_id=plugin.plugin_id),
                AbortContext(),
            )
        )
        for component in ("first", "second")
    ]
    await registration_lock.wait_until_contended()
    registration_lock.release()
    responses = await asyncio.gather(*requests)

    assert sum(response.HasField("error") for response in responses) == 1
    error = next(response.error for response in responses if response.HasField("error"))
    assert "different component config" in error.message


async def test_unary_invoke_success_paths(service: _WorkerService, host_stub: RecordingHostStub):
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
    assert llm_request.llm_request.outcome.schema == LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA
    outcome = _envelope_value(llm_request.llm_request.outcome)
    assert outcome["request"]["content"]["llm_request"]
    assert outcome["annotated_request"]["tag"] == "annotated"
    assert outcome["pending_marks"] == [
        {
            "name": "worker.pending",
            "category": None,
            "category_profile": None,
            "data": {"source": "python"},
            "metadata": None,
        }
    ]

    request_only = await service.Invoke(
        _invoke_request(
            "llm_request",
            pb.LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(request={"content": {"prompt": "hello"}}),
        ),
        AbortContext(),
    )
    request_only_outcome = _envelope_value(request_only.llm_request.outcome)
    assert request_only_outcome["request"]["content"]["llm_request"]
    assert request_only_outcome["annotated_request"] is None
    assert request_only_outcome["pending_marks"] == outcome["pending_marks"]

    llm_execution = await _invoke_json_async(
        service,
        "llm_execution",
        pb.LLM_EXECUTION_INTERCEPT,
        payload=_llm_payload(model_name="gpt-test", request={"content": {"prompt": "hello"}}),
    )
    assert llm_execution["tag"] == "llm_execution"
    assert llm_execution["next_llm"]["content"]["llm_execute_gpt-test"]


async def test_unary_invoke_failure_paths(service: _WorkerService):
    await _register(service)

    invalid = _tool_request("tool_request", pb.TOOL_REQUEST_INTERCEPT, {})
    invalid.tool.value.json = b"{"
    invalid_payload = await service.Invoke(invalid, AbortContext())
    assert invalid_payload.WhichOneof("result") == "error"
    assert "tool value contains invalid JSON" in invalid_payload.error.message

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

    parent_only_scope = await service.Invoke(
        _invoke_request(
            "subscriber",
            pb.SUBSCRIBER,
            event=_json_envelope(EVENT_SCHEMA, {"name": "event"}),
            scope=pb.ScopeContext(parent_scope_id="parent-only"),
        ),
        AbortContext(),
    )
    assert parent_only_scope.WhichOneof("result") == "error"
    assert "parent_scope_id requires scope_stack_id" in parent_only_scope.error.message

    empty_scope = await service.Invoke(
        _invoke_request(
            "subscriber",
            pb.SUBSCRIBER,
            event=_json_envelope(EVENT_SCHEMA, {"name": "event"}),
            scope=pb.ScopeContext(),
        ),
        AbortContext(),
    )
    assert empty_scope.WhichOneof("result") == "error"
    assert "scope_stack_id must not be empty" in empty_scope.error.message


@pytest.mark.parametrize("invalid_part", ["request", "annotated_request"])
async def test_llm_request_intercept_rejects_non_object_typed_results(invalid_part: str):
    class InvalidTypedResultPlugin(WorkerPlugin):
        plugin_id = "tests.invalid_typed_result"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def invalid_result(name: str, request: Json, annotated: Json | None) -> Any:
                del name
                if invalid_part == "request":
                    return LlmRequestInterceptOutcome(request=cast(Any, []))
                return LlmRequestInterceptOutcome(request=request, annotated_request=cast(Any, []))

            ctx.register_llm_request_intercept("invalid", invalid_result)

    service = _service(InvalidTypedResultPlugin(), RecordingHostStub())
    await _register(service)
    response = await service.Invoke(
        _invoke_request(
            "invalid",
            pb.LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(
                request={"content": {"prompt": "hello"}},
                annotated={"messages": []},
            ),
        ),
        AbortContext(),
    )

    assert response.WhichOneof("result") == "error"
    assert "must be a JSON object" in response.error.message


@pytest.mark.parametrize(
    ("request_factory", "expected_message"),
    [
        (
            lambda: _invoke_request(
                "subscriber",
                pb.SUBSCRIBER,
                event=_json_envelope(JSON_SCHEMA, {"name": "event"}),
            ),
            "expected 'nemo.relay.Event@1'",
        ),
        (
            lambda: _tool_request("tool_request", pb.TOOL_REQUEST_INTERCEPT, {}),
            "expected 'nemo.relay.Json@1'",
        ),
        (
            lambda: _invoke_request(
                "llm_sanitize_request",
                pb.LLM_SANITIZE_REQUEST_GUARDRAIL,
                llm=_llm_payload(request={"content": {}}),
            ),
            "expected 'nemo.relay.LlmRequest@1'",
        ),
        (
            lambda: _invoke_request(
                "llm_conditional",
                pb.LLM_CONDITIONAL_EXECUTION_GUARDRAIL,
                llm=_llm_payload(request={"content": {}}),
            ),
            "expected 'nemo.relay.LlmRequest@1'",
        ),
        (
            lambda: _invoke_request(
                "llm_execution",
                pb.LLM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {}}),
            ),
            "expected 'nemo.relay.LlmRequest@1'",
        ),
        (
            lambda: _invoke_request(
                "llm_request",
                pb.LLM_REQUEST_INTERCEPT,
                llm=_llm_payload(request={"content": {}}, annotated={"messages": []}),
            ),
            "expected 'nemo.relay.AnnotatedLlmRequest@1'",
        ),
    ],
)
async def test_invoke_rejects_mismatched_envelope_schemas(
    service: _WorkerService,
    request_factory: Any,
    expected_message: str,
):
    await _register(service)
    request = request_factory()
    if request.surface == pb.TOOL_REQUEST_INTERCEPT:
        request.tool.value.schema = EVENT_SCHEMA
    elif request.surface in {
        pb.LLM_SANITIZE_REQUEST_GUARDRAIL,
        pb.LLM_CONDITIONAL_EXECUTION_GUARDRAIL,
        pb.LLM_EXECUTION_INTERCEPT,
    }:
        request.llm.request.schema = JSON_SCHEMA
    elif request.surface == pb.LLM_REQUEST_INTERCEPT:
        request.llm.annotated_request.schema = JSON_SCHEMA

    response = await service.Invoke(request, AbortContext())
    assert response.WhichOneof("result") == "error"
    assert expected_message in response.error.message


async def test_invoke_stream_rejects_mismatched_llm_request_schema(service: _WorkerService):
    await _register(service)
    request = _invoke_request(
        "llm_stream_execution",
        pb.LLM_STREAM_EXECUTION_INTERCEPT,
        llm=_llm_payload(request={"content": {"prompt": "hello"}}),
    )
    request.llm.request.schema = JSON_SCHEMA

    chunks = [chunk async for chunk in service.InvokeStream(request, AbortContext())]
    assert len(chunks) == 1
    assert "expected 'nemo.relay.LlmRequest@1'" in chunks[0].error.message


async def test_llm_request_intercept_can_return_request_without_annotation():
    class RequestOnlyPlugin(WorkerPlugin):
        plugin_id = "tests.request_only"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def llm_request(name: str, request: Json, annotated: Json | None) -> LlmRequestInterceptOutcome:
                del name, annotated
                return LlmRequestInterceptOutcome(request=_tag_llm_request(request, "request_only"))

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
    outcome = _envelope_value(response.llm_request.outcome)
    assert outcome["request"]["content"]["request_only"]
    assert outcome["annotated_request"] is None
    assert outcome["pending_marks"] == []


async def test_stream_invoke_success_and_failures(service: _WorkerService, host_stub: RecordingHostStub):
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


async def test_stream_callback_exception_is_structured():
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


async def test_stream_callback_can_return_sync_iterable():
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


@pytest.mark.parametrize("invalid_stream", [{"chunk": True}, "chunk", b"chunk", 42])
async def test_stream_callback_rejects_scalar_and_mapping_results(invalid_stream: Any):
    class InvalidStreamPlugin(WorkerPlugin):
        plugin_id = "tests.invalid_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def stream(name: str, request: Json, next_call: Any) -> Any:
                del name, request, next_call
                return invalid_stream

            ctx.register_llm_stream_execution_intercept("invalid_stream", stream)

    service = _service(InvalidStreamPlugin(), RecordingHostStub())
    await _register(service)
    chunks = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "invalid_stream",
                pb.LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert len(chunks) == 1
    assert "stream callback must return" in chunks[0].error.message


async def test_runtime_host_calls_and_scope_context(host_stub: RecordingHostStub):
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    assert runtime.current_scope_stack_id() is None
    assert runtime.current_parent_scope_id() is None

    stack_id = await runtime.create_scope_stack()
    assert stack_id == "stack-1"
    with runtime.bind_scope_stack(stack_id, parent_scope_id="parent-1"):
        assert runtime.current_scope_stack_id() == stack_id
        assert runtime.current_parent_scope_id() == "parent-1"
        await runtime.emit_mark("mark", {"ok": True})
        await runtime.emit_mark("override-parent", parent_scope_id="parent-2")
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

    override_mark = next(
        request
        for request in host_stub.requests
        if isinstance(request, pb.EmitMarkRequest) and request.name == "override-parent"
    )
    assert override_mark.scope.scope_stack_id == "stack-1"
    assert override_mark.scope.parent_scope_id == "parent-2"

    with pytest.raises(WorkerSdkError, match="parent_scope_id requires"):
        await runtime.emit_mark("missing-stack", parent_scope_id="parent")
    with pytest.raises(WorkerSdkError, match="scope_stack_id must not be empty"):
        with runtime.bind_scope_stack(""):
            pass
    with pytest.raises(WorkerSdkError, match="parent_scope_id must not be empty"):
        with runtime.bind_scope_stack("stack", parent_scope_id=""):
            pass
    with pytest.raises(WorkerSdkError, match="parent_scope_id requires"):
        with runtime.bind_scope_stack(None, parent_scope_id="parent"):
            pass
    with pytest.raises(WorkerSdkError, match="scope_stack_id must not be empty"):
        await runtime.emit_mark("empty-stack", scope_stack_id="")
    with pytest.raises(WorkerSdkError, match="parent_scope_id must not be empty"):
        await runtime.emit_mark("empty-parent", scope_stack_id="stack", parent_scope_id="")


async def test_invocation_scope_context_is_isolated_across_concurrent_requests(host_stub: RecordingHostStub):
    started = 0
    both_started = asyncio.Event()
    release = asyncio.Event()

    class ConcurrentScopePlugin(WorkerPlugin):
        plugin_id = "tests.concurrent_scope"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def subscriber(event: Json) -> None:
                nonlocal started
                started += 1
                if started == 2:
                    both_started.set()
                await release.wait()
                await ctx.runtime.emit_mark(event["name"])

            ctx.register_subscriber("subscriber", subscriber)

    service = _service(ConcurrentScopePlugin(), host_stub)
    await _register(service)

    async def invoke(name: str, stack: str, parent: str) -> None:
        response = await service.Invoke(
            _invoke_request(
                "subscriber",
                pb.SUBSCRIBER,
                invocation_id=f"invoke-{name}",
                continuation_id=f"next-{name}",
                event=_json_envelope(EVENT_SCHEMA, {"name": name}),
                scope=pb.ScopeContext(scope_stack_id=stack, parent_scope_id=parent),
            ),
            AbortContext(),
        )
        assert response.WhichOneof("result") == "empty"

    tasks = [
        asyncio.create_task(invoke("first", "stack-1", "parent-1")),
        asyncio.create_task(invoke("second", "stack-2", "parent-2")),
    ]
    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        await asyncio.gather(*tasks)
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    marks = [
        (request.name, request.scope.scope_stack_id, request.scope.parent_scope_id)
        for request in host_stub.requests
        if isinstance(request, pb.EmitMarkRequest)
    ]
    assert len(marks) == 2
    assert sorted(marks) == [
        ("first", "stack-1", "parent-1"),
        ("second", "stack-2", "parent-2"),
    ]


async def test_runtime_host_call_error_paths(host_stub: RecordingHostStub):
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


async def test_lifecycle_acks(service: _WorkerService):
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
    assert "not active" in cancel.message

    shutdown = await service.Shutdown(
        pb.ShutdownRequest(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, reason="test"),
        AbortContext(),
    )
    assert shutdown.accepted
    assert "shutdown accepted" in shutdown.message


async def test_cancel_invocation_stops_active_async_callback_and_is_idempotent():
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    class CancelPlugin(WorkerPlugin):
        plugin_id = "tests.cancel"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def tool_execution(tool_name: str, value: Json, next_call: ToolNext) -> Json:
                del tool_name, value, next_call
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    raise

            ctx.register_tool_execution_intercept("cancel", tool_execution)

    service = _service(CancelPlugin(), RecordingHostStub())
    await _register(service)
    request = _invoke_request(
        "cancel",
        pb.TOOL_EXECUTION_INTERCEPT,
        invocation_id="cancel-unary",
        tool=pb.ToolInvocation(tool_name="lookup", value=_json_envelope(JSON_SCHEMA, {})),
    )
    invoke_task = asyncio.create_task(service.Invoke(request, AbortContext()))
    await asyncio.wait_for(started.wait(), timeout=1)

    first = await service.CancelInvocation(
        pb.CancelInvocationRequest(
            activation_id=ACTIVATION_ID,
            invocation_id="cancel-unary",
            auth_token=AUTH_TOKEN,
            reason="test timeout",
        ),
        AbortContext(),
    )
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    second = await service.CancelInvocation(
        pb.CancelInvocationRequest(
            activation_id=ACTIVATION_ID,
            invocation_id="cancel-unary",
            auth_token=AUTH_TOKEN,
            reason="repeat",
        ),
        AbortContext(),
    )
    reused = await asyncio.wait_for(service.Invoke(request, AbortContext()), timeout=1)
    release.set()
    response = await asyncio.wait_for(invoke_task, timeout=1)

    assert first.accepted
    assert not second.accepted
    assert reused.error.code == "worker.sdk_error"
    assert "already active" in reused.error.message
    assert response.error.code == "worker.cancelled"
    assert "test timeout" in response.error.message
    assert cancelled.is_set()


async def test_cancel_invocation_stops_active_async_stream():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class CancelStreamPlugin(WorkerPlugin):
        plugin_id = "tests.cancel_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def llm_stream(model_name: str, request: Json, next_call: Any) -> AsyncIterator[Json]:
                del model_name, request, next_call
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                if False:
                    yield {}

            ctx.register_llm_stream_execution_intercept("cancel_stream", llm_stream)

    service = _service(CancelStreamPlugin(), RecordingHostStub())
    await _register(service)
    request = _invoke_request(
        "cancel_stream",
        pb.LLM_STREAM_EXECUTION_INTERCEPT,
        invocation_id="cancel-stream",
        llm=_llm_payload(),
    )

    async def consume() -> list[Any]:
        return [chunk async for chunk in service.InvokeStream(request, AbortContext())]

    consume_task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    ack = await service.CancelInvocation(
        pb.CancelInvocationRequest(
            activation_id=ACTIVATION_ID,
            invocation_id="cancel-stream",
            auth_token=AUTH_TOKEN,
            reason="stream abandoned",
        ),
        AbortContext(),
    )
    chunks = await asyncio.wait_for(consume_task, timeout=1)

    assert ack.accepted
    assert len(chunks) == 1
    assert chunks[0].error.code == "worker.cancelled"
    assert "stream abandoned" in chunks[0].error.message
    assert cancelled.is_set()


async def test_cancel_invocation_discards_buffered_chunks_after_stream_callback_finishes():
    finished = asyncio.Event()

    class BufferedStreamPlugin(WorkerPlugin):
        plugin_id = "tests.buffered_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def llm_stream(model_name: str, request: Json, next_call: Any) -> AsyncIterator[Json]:
                del model_name, request, next_call
                for index in range(4):
                    yield {"index": index}
                finished.set()

            ctx.register_llm_stream_execution_intercept("buffered_stream", llm_stream)

    service = _service(BufferedStreamPlugin(), RecordingHostStub())
    await _register(service)
    request = _invoke_request(
        "buffered_stream",
        pb.LLM_STREAM_EXECUTION_INTERCEPT,
        invocation_id="buffered-stream",
        llm=_llm_payload(),
    )
    stream = service.InvokeStream(request, AbortContext())

    first = await asyncio.wait_for(anext(stream), timeout=1)
    await asyncio.wait_for(finished.wait(), timeout=1)
    assert not first.error.code
    assert "buffered-stream" in service._active_invocations

    ack = await service.CancelInvocation(
        pb.CancelInvocationRequest(
            activation_id=ACTIVATION_ID,
            invocation_id="buffered-stream",
            auth_token=AUTH_TOKEN,
            reason="stop buffered stream",
        ),
        AbortContext(),
    )

    async def consume_remaining() -> list[Any]:
        return [chunk async for chunk in stream]

    remaining = await asyncio.wait_for(consume_remaining(), timeout=1)

    assert ack.accepted
    assert len(remaining) == 1
    assert remaining[0].error.code == "worker.cancelled"
    assert "stop buffered stream" in remaining[0].error.message
    assert "buffered-stream" not in service._active_invocations


async def test_stream_callback_cancellation_without_host_reason_is_terminal_error():
    class CancelledStreamPlugin(WorkerPlugin):
        plugin_id = "tests.cancelled_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def llm_stream(model_name: str, request: Json, next_call: Any) -> AsyncIterator[Json]:
                del model_name, request, next_call
                if False:
                    yield {}
                raise asyncio.CancelledError

            ctx.register_llm_stream_execution_intercept("cancelled_stream", llm_stream)

    service = _service(CancelledStreamPlugin(), RecordingHostStub())
    await _register(service)
    request = _invoke_request(
        "cancelled_stream",
        pb.LLM_STREAM_EXECUTION_INTERCEPT,
        invocation_id="self-cancelled-stream",
        llm=_llm_payload(),
    )

    async def consume() -> list[Any]:
        return [chunk async for chunk in service.InvokeStream(request, AbortContext())]

    chunks = await asyncio.wait_for(consume(), timeout=1)

    assert len(chunks) == 1
    assert chunks[0].error.code == "worker.cancelled"
    assert "stream callback cancelled without a host request" in chunks[0].error.message


def test_required_environment_reports_missing_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NEMO_RELAY_WORKER_SOCKET", raising=False)
    with pytest.raises(WorkerSdkError, match="NEMO_RELAY_WORKER_SOCKET"):
        _required_env("NEMO_RELAY_WORKER_SOCKET")


def test_unix_host_channel_uses_valid_authority(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def insecure_channel(
        target: str,
        *,
        options: tuple[tuple[str, str], ...] = (),
    ) -> object:
        calls.append((target, options))
        return object()

    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", insecure_channel)

    _open_host_channel("unix:///tmp/relay-host.sock")
    _open_host_channel("http://127.0.0.1:50051")

    assert calls == [
        ("unix:/tmp/relay-host.sock", (("grpc.default_authority", "localhost"),)),
        ("127.0.0.1:50051", ()),
    ]


async def test_endpoint_helpers_normalize_and_refuse_non_socket_unix_targets(tmp_path: Any):
    assert _grpc_target("tcp://127.0.0.1:50051") == "127.0.0.1:50051"
    assert _grpc_target("http://127.0.0.1:50051") == "127.0.0.1:50051"
    assert _grpc_target("tcp://localhost:50051") == "localhost:50051"
    assert _grpc_target("http://[::1]:50051") == "[::1]:50051"
    assert _grpc_target("unix:///tmp/worker.sock") == "unix:/tmp/worker.sock"
    assert _announced_worker_endpoint("tcp://127.0.0.1:0", 43123) == "http://127.0.0.1:43123"
    assert _announced_worker_endpoint("http://127.0.0.1:50051", 43123) == "http://127.0.0.1:50051"
    assert _announced_worker_endpoint("unix:///tmp/worker.sock", 43123) == "unix:///tmp/worker.sock"
    await _unlink_unix_socket("tcp://127.0.0.1:50051")
    await _unlink_unix_socket(f"unix://{tmp_path / 'missing.sock'}")

    socket_path = tmp_path / "worker.sock"
    socket_path.write_text("", encoding="utf-8")
    with pytest.raises(WorkerSdkError, match="exists and is not a socket"):
        await _unlink_unix_socket(f"unix://{socket_path}")
    assert socket_path.exists()


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://0.0.0.0:50051",
        "http://192.0.2.1:50051",
        "tcp://example.com:50051",
        "https://127.0.0.1:50051",
        "127.0.0.1:50051",
        "tcp://127.0.0.1",
        "tcp://127.0.0.1:not-a-port",
        "tcp://127.0.0.1:50051/path",
        "unix://",
    ],
)
def test_grpc_target_rejects_non_loopback_and_malformed_endpoints(endpoint: str):
    with pytest.raises(WorkerSdkError):
        _grpc_target(endpoint)


@pytest.mark.parametrize(
    ("worker_endpoint", "host_endpoint"),
    [
        ("tcp://0.0.0.0:0", "http://127.0.0.1:9"),
        ("tcp://127.0.0.1:0", "http://192.0.2.1:50051"),
    ],
)
async def test_serve_plugin_validates_endpoints_before_opening_channel(
    worker_endpoint: str,
    host_endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
):
    channel_opened = False

    def insecure_channel(target: str) -> Any:
        nonlocal channel_opened
        channel_opened = True
        raise AssertionError(f"unexpected channel for {target}")

    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", insecure_channel)
    monkeypatch.setenv("NEMO_RELAY_WORKER_SOCKET", worker_endpoint)
    monkeypatch.setenv("NEMO_RELAY_HOST_SOCKET", host_endpoint)
    monkeypatch.setenv("NEMO_RELAY_WORKER_ID", ACTIVATION_ID)
    monkeypatch.setenv("NEMO_RELAY_WORKER_TOKEN", AUTH_TOKEN)

    with pytest.raises(WorkerSdkError, match="loopback"):
        await serve_plugin(AllSurfacesPlugin())
    assert not channel_opened


async def test_serve_plugin_validates_plugin_id_before_resources(monkeypatch: pytest.MonkeyPatch):
    class InvalidPlugin(WorkerPlugin):
        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config

    channel_opened = False

    def insecure_channel(target: str) -> Any:
        nonlocal channel_opened
        channel_opened = True
        raise AssertionError(f"unexpected channel for {target}")

    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", insecure_channel)
    monkeypatch.delenv("NEMO_RELAY_WORKER_SOCKET", raising=False)

    with pytest.raises(WorkerSdkError, match="plugin_id"):
        await serve_plugin(InvalidPlugin())
    assert not channel_opened


def test_endpoint_file_is_published_atomically(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    endpoint_file = tmp_path / "endpoint.txt"
    endpoint_file.write_text("old", encoding="utf-8")
    original_replace = plugin_api.os.replace
    replacements: list[tuple[Path, Path]] = []

    def replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent
        assert source_path.read_text(encoding="utf-8") == "http://127.0.0.1:43123"
        assert destination_path.read_text(encoding="utf-8") == "old"
        replacements.append((source_path, destination_path))
        original_replace(source, destination)

    monkeypatch.setattr(plugin_api.os, "replace", replace)
    _write_endpoint_file(endpoint_file, "http://127.0.0.1:43123")

    assert len(replacements) == 1
    assert endpoint_file.read_text(encoding="utf-8") == "http://127.0.0.1:43123"
    assert not list(tmp_path.glob("*.tmp"))


def test_endpoint_file_cleans_up_temporary_file_on_replace_failure(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoint_file = tmp_path / "endpoint.txt"

    def replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(plugin_api.os, "replace", replace)
    with pytest.raises(OSError, match="replace failed"):
        _write_endpoint_file(endpoint_file, "http://127.0.0.1:43123")

    assert not endpoint_file.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
async def test_unlink_unix_socket_removes_an_existing_socket():
    with tempfile.TemporaryDirectory(prefix="nr-plugin-") as directory:
        socket_path = Path(directory) / "worker.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix_socket:
            unix_socket.bind(str(socket_path))

        await _unlink_unix_socket(f"unix://{socket_path}")
        assert not socket_path.exists()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
async def test_unlink_unix_socket_refuses_an_active_socket():
    with tempfile.TemporaryDirectory(prefix="nr-plugin-", dir="/tmp") as directory:
        socket_path = Path(directory) / "active.sock"
        server = await asyncio.start_unix_server(lambda _reader, writer: writer.close(), path=socket_path)
        try:
            with pytest.raises(WorkerSdkError, match="already active"):
                await _unlink_unix_socket(f"unix://{socket_path}")
            assert socket_path.exists()
        finally:
            server.close()
            await server.wait_closed()
            socket_path.unlink(missing_ok=True)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
async def test_unlink_unix_socket_still_refuses_active_socket_when_close_wait_times_out(
    monkeypatch: pytest.MonkeyPatch,
):
    with tempfile.TemporaryDirectory(prefix="nr-plugin-", dir="/tmp") as directory:
        socket_path = Path(directory) / "active.sock"
        writer = mock.AsyncMock(spec=asyncio.StreamWriter)
        writer.wait_closed.side_effect = TimeoutError

        async def open_unix_connection(_path: Path) -> tuple[object, mock.AsyncMock]:
            return object(), writer

        monkeypatch.setattr(plugin_api.asyncio, "open_unix_connection", open_unix_connection)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix_socket:
                unix_socket.bind(str(socket_path))
                with pytest.raises(WorkerSdkError, match="already active"):
                    await _unlink_unix_socket(f"unix://{socket_path}")
                assert socket_path.exists()
        finally:
            socket_path.unlink(missing_ok=True)


async def test_serve_plugin_announces_endpoint_only_after_server_start(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoint_file = tmp_path / "endpoint.txt"
    start_entered = asyncio.Event()
    allow_start = asyncio.Event()

    class FakeChannel:
        async def close(self) -> None:
            pass

    class FakeServer:
        def add_insecure_port(self, target: str) -> int:
            assert target == "127.0.0.1:0"
            return 43123

        async def start(self) -> None:
            start_entered.set()
            await allow_start.wait()

        async def stop(self, grace: int) -> None:
            assert grace == 2

    fake_server = FakeServer()
    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", lambda target: FakeChannel())
    monkeypatch.setattr(plugin_api.grpc.aio, "server", lambda: fake_server)
    monkeypatch.setattr(plugin_api.pb_grpc, "RelayHostRuntimeStub", lambda channel: object())
    monkeypatch.setattr(plugin_api.pb_grpc, "add_PluginWorkerServicer_to_server", lambda service, server: None)
    monkeypatch.setenv("NEMO_RELAY_WORKER_SOCKET", "tcp://127.0.0.1:0")
    monkeypatch.setenv("NEMO_RELAY_HOST_SOCKET", "http://127.0.0.1:9")
    monkeypatch.setenv("NEMO_RELAY_WORKER_ID", ACTIVATION_ID)
    monkeypatch.setenv("NEMO_RELAY_WORKER_TOKEN", AUTH_TOKEN)
    monkeypatch.setenv("NEMO_RELAY_WORKER_ENDPOINT_FILE", str(endpoint_file))

    task = asyncio.create_task(serve_plugin(AllSurfacesPlugin()))
    try:
        await asyncio.wait_for(start_entered.wait(), timeout=1)
        assert not endpoint_file.exists()
        allow_start.set()
        for _ in range(100):
            if endpoint_file.exists():
                break
            await asyncio.sleep(0.01)
        assert endpoint_file.read_text(encoding="utf-8") == "http://127.0.0.1:43123"
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_serve_plugin_closes_resources_when_server_start_fails(monkeypatch: pytest.MonkeyPatch):
    class FakeChannel:
        closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeServer:
        stopped = False

        def add_insecure_port(self, target: str) -> int:
            assert target == "127.0.0.1:0"
            return 43123

        async def start(self) -> None:
            raise RuntimeError("start failed")

        async def stop(self, grace: int) -> None:
            assert grace == 2
            self.stopped = True

    fake_channel = FakeChannel()
    fake_server = FakeServer()
    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", lambda target: fake_channel)
    monkeypatch.setattr(plugin_api.grpc.aio, "server", lambda: fake_server)
    monkeypatch.setattr(plugin_api.pb_grpc, "RelayHostRuntimeStub", lambda channel: object())
    monkeypatch.setattr(plugin_api.pb_grpc, "add_PluginWorkerServicer_to_server", lambda service, server: None)
    monkeypatch.setenv("NEMO_RELAY_WORKER_SOCKET", "tcp://127.0.0.1:0")
    monkeypatch.setenv("NEMO_RELAY_HOST_SOCKET", "http://127.0.0.1:9")
    monkeypatch.setenv("NEMO_RELAY_WORKER_ID", ACTIVATION_ID)
    monkeypatch.setenv("NEMO_RELAY_WORKER_TOKEN", AUTH_TOKEN)
    monkeypatch.delenv("NEMO_RELAY_WORKER_ENDPOINT_FILE", raising=False)

    with pytest.raises(RuntimeError, match="start failed"):
        await serve_plugin(AllSurfacesPlugin())
    assert fake_server.stopped
    assert fake_channel.closed


async def test_serve_plugin_closes_resources_when_endpoint_bind_fails(monkeypatch: pytest.MonkeyPatch):
    class FakeChannel:
        closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeServer:
        stopped = False

        def add_insecure_port(self, target: str) -> int:
            assert target == "127.0.0.1:0"
            return 0

        async def stop(self, grace: int) -> None:
            assert grace == 2
            self.stopped = True

    fake_channel = FakeChannel()
    fake_server = FakeServer()
    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", lambda target: fake_channel)
    monkeypatch.setattr(plugin_api.grpc.aio, "server", lambda: fake_server)
    monkeypatch.setattr(plugin_api.pb_grpc, "RelayHostRuntimeStub", lambda channel: object())
    monkeypatch.setattr(plugin_api.pb_grpc, "add_PluginWorkerServicer_to_server", lambda service, server: None)
    monkeypatch.setenv("NEMO_RELAY_WORKER_SOCKET", "tcp://127.0.0.1:0")
    monkeypatch.setenv("NEMO_RELAY_HOST_SOCKET", "http://127.0.0.1:9")
    monkeypatch.setenv("NEMO_RELAY_WORKER_ID", ACTIVATION_ID)
    monkeypatch.setenv("NEMO_RELAY_WORKER_TOKEN", AUTH_TOKEN)
    monkeypatch.delenv("NEMO_RELAY_WORKER_ENDPOINT_FILE", raising=False)

    with pytest.raises(WorkerSdkError, match="failed to bind worker endpoint"):
        await serve_plugin(AllSurfacesPlugin())
    assert fake_server.stopped
    assert fake_channel.closed


async def test_serve_plugin_announces_tcp_endpoint_and_accepts_health_shutdown(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
):
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


def _validate_request(
    config: Json | None = None,
    *,
    plugin_id: str = "tests.python_worker",
) -> Any:
    return pb.ValidateRequest(
        activation_id=ACTIVATION_ID,
        plugin_id=plugin_id,
        auth_token=AUTH_TOKEN,
        config=_json_envelope(JSON_SCHEMA, {} if config is None else config),
    )


def _register_request(
    config: Json | None = None,
    *,
    plugin_id: str = "tests.python_worker",
) -> Any:
    return pb.RegisterRequest(
        activation_id=ACTIVATION_ID,
        plugin_id=plugin_id,
        auth_token=AUTH_TOKEN,
        config=_json_envelope(JSON_SCHEMA, {} if config is None else config),
    )


async def _register(service: _WorkerService, *, plugin_id: str | None = None) -> Any:
    plugin_id = plugin_id or plugin_api._plugin_id(service._plugin)
    response = await service.Register(_register_request(plugin_id=plugin_id), AbortContext())
    assert not response.HasField("error"), response.error
    return response


def _invoke_request(
    registration_name: str,
    surface: int,
    *,
    invocation_id: str = "invoke-1",
    continuation_id: str = "next-1",
    **kwargs: Any,
) -> Any:
    return pb.InvokeRequest(
        activation_id=ACTIVATION_ID,
        invocation_id=invocation_id,
        registration_name=registration_name,
        surface=surface,
        continuation_id=continuation_id,
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
