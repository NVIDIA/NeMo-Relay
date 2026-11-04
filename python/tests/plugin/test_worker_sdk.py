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
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest import mock
from uuid import UUID

import pytest

if os.environ.get("NEMO_RELAY_SKIP_PYTHON_PLUGIN_TESTS") == "1":
    pytest.skip("grpcio is unavailable for Python plugin SDK tests on this runner", allow_module_level=True)

grpc = pytest.importorskip("grpc")

from nemo_relay_plugin import (  # noqa: E402
    ConditionalMiddlewareGuardrailHandle,
    ConfigDiagnostic,
    DataSchema,
    DiagnosticLevel,
    EventCategory,
    Json,
    LlmOptimizationContribution,
    LlmRequestInterceptOutcome,
    LlmSanitizeRequestContext,
    LlmSanitizeResponseContext,
    LogSeverity,
    PendingMarkSpec,
    PluginContext,
    PluginRuntime,
    RuntimeDiagnostic,
    RuntimeDiagnostics,
    RuntimeRegistrationIdentity,
    RuntimeRegistrationKind,
    RuntimeRegistrationOwnerKind,
    ScopeType,
    ToolExecutionContext,
    ToolExecutionInterceptOutcome,
    ToolExecutionResult,
    ToolNext,
    WorkerPlugin,
    WorkerSdkError,
    serve_plugin,
)
from nemo_relay_plugin import _api as plugin_api  # noqa: E402
from nemo_relay_plugin._api import (  # noqa: E402
    WORKER_PROTOCOL,
    _announced_worker_endpoint,
    _decode_json_value,
    _decode_payload,
    _grpc_target,
    _json_value,
    _open_host_channel,
    _required_env,
    _tool_execution_intercept_outcome_to_proto,
    _tool_execution_result_from_proto,
    _unlink_unix_socket,
    _WorkerService,
    _write_endpoint_file,
)
from nemo_relay_plugin._api import (  # noqa: E402
    _encode_payload as _pack_payload,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    common_pb2 as common_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    event_pb2 as event_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    llm_pb2 as llm_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    plugin_worker_service_pb2 as worker_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    plugin_worker_service_pb2_grpc as worker_grpc,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    relay_host_runtime_service_pb2 as host_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    runtime_pb2 as runtime_pb,
)


class _Fixture(SimpleNamespace):
    def HasField(self, name):
        return getattr(self, name, None) is not None

    def CopyFrom(self, value):
        self.__dict__.update(value.__dict__)

    def WhichOneof(self, name):
        for field in ("request_sanitize_context", "response_sanitize_context"):
            if self.HasField(field):
                return field
        return None


class _ExecutionContext(_Fixture):
    def __init__(self, request=None, response=None):
        super().__init__(request=request, response=response)


class _LlmPayload(_Fixture):
    def __init__(self, **kwargs):
        values = dict(
            model_name="model",
            request=None,
            response=None,
            annotated_request=None,
            request_sanitize_context=None,
            response_sanitize_context=None,
            execution_codec_context=None,
        )
        values.update(kwargs)
        super().__init__(**values)


class _ToolPayload(_Fixture):
    def __init__(self, tool_name="lookup", value=None, tool_call_id=None):
        super().__init__(tool_name=tool_name, value=value, tool_call_id=tool_call_id)


def _wire_codec(kind="none", id=None):
    return runtime_pb.LlmCodecReference(kind=kind, id=id)


def _encode_payload(schema, value):
    if schema == event_pb.Event and isinstance(value, dict):
        value = {
            "kind": "mark",
            "uuid": "0199942f-1000-7000-8000-000000000001",
            "timestamp": "2026-10-15T09:00:00Z",
            **value,
        }
        if value.get("kind") == "scope":
            value = {"scope_category": "start", "category": "custom", **value}
    return _pack_payload(schema, value)


ACTIVATION_ID = "act"
AUTH_TOKEN = "token"


@pytest.fixture(name="optimization_contribution_fixture")
def optimization_contribution_fixture_fixture() -> Json:
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "crates"
        / "types"
        / "tests"
        / "fixtures"
        / "llm_optimization_contribution_v1.json"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_tool_execution_result_and_outcome_preserve_optional_annotation() -> None:
    result = ToolExecutionResult(result={"ok": True}, annotation={"source": "worker"})
    assert result.to_json() == {
        "result": {"ok": True},
        "annotation": {"source": "worker"},
    }
    assert ToolExecutionResult.from_json(result.to_json()) == result
    null_result = ToolExecutionResult.from_json({"result": {"ok": True}, "annotation": None})
    assert null_result.annotation is None
    assert null_result.to_json() == {"result": {"ok": True}}
    assert ToolExecutionResult(result=None).to_json() == {"result": None}
    assert ToolExecutionInterceptOutcome(result={"ok": True}).to_json() == {
        "result": {"ok": True},
        "pending_marks": [],
    }
    positional_marks = [PendingMarkSpec("worker.positional-mark")]
    positional_outcome = ToolExecutionInterceptOutcome({"ok": True}, positional_marks)
    assert positional_outcome.pending_marks == positional_marks
    assert positional_outcome.annotation is None
    null_outcome = ToolExecutionInterceptOutcome(result={"ok": True}, annotation=None)
    assert null_outcome.annotation is None
    assert null_outcome.to_json() == {"result": {"ok": True}, "pending_marks": []}
    with pytest.raises(WorkerSdkError, match="result field"):
        ToolExecutionResult.from_json({"annotation": {"source": "worker"}})


@pytest.mark.parametrize(
    "value",
    [
        None,
        "relay",
        [1, "two", None, {"nested": False}],
        {"content": [{"type": "text", "text": "ok"}], "structuredContent": {"count": 2}},
    ],
)
def test_tool_execution_result_proto_preserves_arbitrary_json(value: Json) -> None:
    decoded = _tool_execution_result_from_proto(runtime_pb.ToolExecutionResult(result=_json_value(value)))

    assert decoded == ToolExecutionResult(result=value)


def test_tool_execution_result_proto_normalizes_null_annotation() -> None:
    decoded = _tool_execution_result_from_proto(
        runtime_pb.ToolExecutionResult(
            result=_json_value({"ok": True}),
            annotation=_json_value(None),
        )
    )

    assert decoded.annotation is None


@pytest.mark.parametrize(
    ("message", "expected_message"),
    [
        (runtime_pb.ToolExecutionResult(), "missing result"),
        (runtime_pb.ToolExecutionResult(result=common_pb.JsonValue()), "result contains invalid JSON"),
        (runtime_pb.ToolExecutionResult(result=common_pb.JsonValue(json=b"{")), "result contains invalid JSON"),
        (
            runtime_pb.ToolExecutionResult(
                result=_json_value({"ok": True}),
                annotation=common_pb.JsonValue(json=b"not-json"),
            ),
            "annotation contains invalid JSON",
        ),
    ],
)
def test_tool_execution_result_proto_rejects_missing_or_malformed_fields(message: Any, expected_message: str) -> None:
    with pytest.raises(WorkerSdkError, match=expected_message):
        _tool_execution_result_from_proto(message)


def test_tool_execution_outcome_proto_preserves_annotation_and_pending_marks() -> None:
    message = _tool_execution_intercept_outcome_to_proto(
        ToolExecutionInterceptOutcome(
            result=[{"text": "ok"}, None],
            annotation={"source": ["worker", 1]},
            pending_marks=[
                PendingMarkSpec(
                    "worker.mark",
                    category="tool",
                    category_profile={"subtype": "checkpoint", "nested": {"ok": True}},
                    data=False,
                    metadata=0,
                )
            ],
        )
    )

    assert _decode_json_value(message.value.result, "result") == [{"text": "ok"}, None]
    assert _decode_json_value(message.value.annotation, "annotation") == {"source": ["worker", 1]}
    assert [plugin_api._protocol.unpack(mark) for mark in message.pending_marks] == [
        {
            "name": "worker.mark",
            "category": "tool",
            "category_profile": {"subtype": "checkpoint", "nested": {"ok": True}},
            "data": False,
            "metadata": 0,
        }
    ]


def test_tool_execution_outcome_proto_omits_null_annotation_and_optional_mark_fields() -> None:
    message = _tool_execution_intercept_outcome_to_proto(
        ToolExecutionInterceptOutcome(result=None, pending_marks=[PendingMarkSpec("worker.mark")], annotation=None)
    )

    assert message.value.HasField("result")
    assert _decode_json_value(message.value.result, "result") is None
    assert not message.value.HasField("annotation")
    assert bool(message.pending_marks)
    assert [plugin_api._protocol.unpack(mark) for mark in message.pending_marks] == [
        {
            "name": "worker.mark",
        }
    ]


@pytest.mark.parametrize(
    ("outcome", "expected_message"),
    [
        (
            ToolExecutionInterceptOutcome(result={}, pending_marks=[cast(Any, {"name": "not-a-spec"})]),
            "must contain PendingMarkSpec",
        ),
        (
            ToolExecutionInterceptOutcome(result={}, pending_marks=[PendingMarkSpec(cast(Any, 1))]),
            "pending mark name must be a string",
        ),
        (
            ToolExecutionInterceptOutcome(
                result={},
                pending_marks=[PendingMarkSpec("worker.mark", category=cast(Any, 1))],
            ),
            "pending mark category must be a string or None",
        ),
    ],
)
def test_tool_execution_outcome_proto_rejects_malformed_pending_marks(
    outcome: ToolExecutionInterceptOutcome,
    expected_message: str,
) -> None:
    with pytest.raises(WorkerSdkError, match=expected_message):
        _tool_execution_intercept_outcome_to_proto(outcome)


def test_optimization_contribution_fixture_round_trips_losslessly(optimization_contribution_fixture: Json) -> None:
    fixture = optimization_contribution_fixture
    contribution = LlmOptimizationContribution.from_json(fixture)

    outcome = LlmRequestInterceptOutcome(
        request={"headers": {}, "content": {"model": "test"}},
        optimization_contributions=[contribution],
    ).to_json()

    assert outcome["optimization_contributions"] == [fixture]
    assert contribution.kind == fixture["kind"]
    assert contribution.kind not in {
        LlmOptimizationContribution.INPUT_COMPRESSION,
        LlmOptimizationContribution.MODEL_ROUTING,
    }
    assert contribution.extra["future_top_level_field"] == fixture["future_top_level_field"]
    assert (
        LlmRequestInterceptOutcome(request={"headers": {}, "content": {}}).to_json()["optimization_contributions"] == []
    )


def test_optimization_contribution_requires_schema_for_payload() -> None:
    with pytest.raises(WorkerSdkError, match="payload_schema"):
        LlmOptimizationContribution.from_json(
            {"producer": "test", "kind": "custom", "applied": True, "payload": {"value": 1}}
        )
    with pytest.raises(WorkerSdkError, match="producer must be a string"):
        LlmOptimizationContribution.from_json({"producer": 1, "kind": "custom"})


def test_optimization_contribution_omitted_applied_defaults_consistently() -> None:
    direct = LlmOptimizationContribution(producer="test", kind="custom")
    decoded = LlmOptimizationContribution.from_json({"producer": "test", "kind": "custom"})

    assert direct.applied is False
    assert decoded.applied is False
    assert direct.to_json()["applied"] is False
    assert decoded.to_json()["applied"] is False


def test_pending_mark_spec_serializes_telemetry_fields() -> None:
    mark = PendingMarkSpec(
        "worker.telemetry",
        data={"measurements": []},
        data_schema={"name": "nemo.relay.metric_measurements", "version": "1"},
        severity="warning",
    )

    assert mark.to_json() == {
        "name": "worker.telemetry",
        "category": None,
        "category_profile": None,
        "data": {"measurements": []},
        "data_schema": {
            "name": "nemo.relay.metric_measurements",
            "version": "1",
        },
        "metadata": None,
        "severity": "warn",
    }

    with pytest.raises(ValueError, match="invalid log severity"):
        PendingMarkSpec("worker.invalid", severity="fatal").to_json()


def test_optimization_contribution_preserves_future_quality_strings() -> None:
    fixture = {
        "producer": "test",
        "kind": "custom",
        "token_impact": {"quality": "provider_observed_v2"},
    }

    contribution = LlmOptimizationContribution.from_json(fixture)

    assert contribution.token_impact is not None
    assert contribution.token_impact.quality == "provider_observed_v2"
    assert contribution.to_json() == {**fixture, "applied": False}


def test_optimization_contribution_drops_known_fields_from_extra() -> None:
    contribution = LlmOptimizationContribution(
        producer="test",
        kind="custom",
        extra={
            "id": "stale-id",
            "sequence": 99,
            "model_transition": {"baseline": {"model_id": "stale"}},
            "token_impact": {"quality": "stale"},
            "payload_schema": {"name": "stale", "version": "1"},
            "payload": {"stale": True},
            "future_field": "preserved",
        },
    )

    assert contribution.to_json() == {
        "producer": "test",
        "kind": "custom",
        "applied": False,
        "future_field": "preserved",
    }


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
        plugin_api._protocol.validate(request)
        self.requests.append(request)
        return self._host_ack("EmitMark")

    async def Log(self, request: Any) -> Any:
        self.requests.append(request)
        return self._host_ack("Log")

    async def GetRuntimeDiagnostics(self, request: Any) -> Any:
        self.requests.append(request)
        return host_pb.GetRuntimeDiagnosticsResponse(
            entries=[
                runtime_pb.RuntimeDiagnostic(code="alpha", message="first", count=1),
                runtime_pb.RuntimeDiagnostic(code="zeta", message="latest", count=3),
            ]
        )

    async def ListRuntimeRegistrations(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("ListRuntimeRegistrations") == "error":
            raise WorkerSdkError("list failed")
        if self.failures.get("ListRuntimeRegistrations") == "unknown_kind":
            return host_pb.ListRuntimeRegistrationsResponse(
                registrations=[runtime_pb.RuntimeRegistrationIdentity(kind=999)]
            )
        if self.failures.get("ListRuntimeRegistrations") == "unknown_owner":
            return host_pb.ListRuntimeRegistrationsResponse(
                registrations=[
                    runtime_pb.RuntimeRegistrationIdentity(
                        kind=common_pb.REGISTRATION_SURFACE_SUBSCRIBER,
                        owner=runtime_pb.RuntimeRegistrationOwner(),
                    )
                ]
            )
        return host_pb.ListRuntimeRegistrationsResponse(
            registrations=[
                runtime_pb.RuntimeRegistrationIdentity(
                    kind=common_pb.REGISTRATION_SURFACE_SUBSCRIBER,
                    local_name="opentelemetry",
                    effective_name="nemo-relay-plugin.v1.observability:1:opentelemetry",
                    owner=runtime_pb.RuntimeRegistrationOwner(
                        plugin=runtime_pb.PluginRegistrationOwner(plugin_kind="observability", component_ordinal=1)
                    ),
                )
            ]
        )

    async def RegisterConditionalMiddlewareGuardrail(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("RegisterConditionalMiddlewareGuardrail") == "error":
            raise WorkerSdkError("register failed")
        return host_pb.RegisterConditionalMiddlewareGuardrailResponse(handle="gate-1")

    async def DeregisterConditionalMiddlewareGuardrail(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("DeregisterConditionalMiddlewareGuardrail") == "error":
            raise WorkerSdkError("deregister failed")
        return host_pb.DeregisterConditionalMiddlewareGuardrailResponse(removed=True)

    async def CreateScopeStack(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("CreateScopeStack") == "error":
            raise WorkerSdkError("CreateScopeStack failed")
        return host_pb.CreateScopeStackResponse(scope_stack_id="stack-1")

    async def DropScopeStack(self, request: Any) -> Any:
        self.requests.append(request)
        return self._host_ack("DropScopeStack")

    async def PushScope(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("PushScope") == "error":
            raise WorkerSdkError("PushScope failed")
        return host_pb.PushScopeResponse(scope_handle_id="scope-1")

    async def PopScope(self, request: Any) -> Any:
        self.requests.append(request)
        return self._host_ack("PopScope")

    async def ToolNext(self, request: Any) -> Any:
        self.requests.append(request)
        failure = self.failures.get("ToolNext")
        if failure == "error":
            return host_pb.ToolNextResponse(error=_worker_error("ToolNext failed"))
        if failure == "empty":
            return host_pb.ToolNextResponse()
        if failure == "missing_result":
            return host_pb.ToolNextResponse(
                value=runtime_pb.ToolExecutionResult(annotation=_json_value({"source": "host"}))
            )
        if failure == "malformed_result":
            return host_pb.ToolNextResponse(
                value=runtime_pb.ToolExecutionResult(result=common_pb.JsonValue(json=b"not-json"))
            )
        if failure == "malformed_annotation":
            return host_pb.ToolNextResponse(
                value=runtime_pb.ToolExecutionResult(
                    result=_json_value({"ok": True}),
                    annotation=common_pb.JsonValue(json=b"not-json"),
                )
            )
        if failure == "null_annotation":
            return host_pb.ToolNextResponse(
                value=runtime_pb.ToolExecutionResult(
                    result=_json_value({"ok": True}),
                    annotation=_json_value(None),
                )
            )
        value = json.loads(request.value.json.decode("utf-8"))
        return host_pb.ToolNextResponse(
            value=runtime_pb.ToolExecutionResult(
                result=_json_value({"next_tool": value}),
                annotation=_json_value({"source": "host"}),
            )
        )

    async def LlmNext(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("LlmNext") == "error":
            return host_pb.LlmNextResponse(error=_worker_error("LlmNext failed"))
        value = _envelope_value(request.request)
        return host_pb.LlmNextResponse(value=_encode_payload(common_pb.JsonValue, {"next_llm": value}))

    def LlmStreamNext(self, request: Any) -> AsyncIterator[Any]:
        self.requests.append(request)

        async def stream() -> AsyncIterator[Any]:
            failure = self.failures.get("LlmStreamNext")
            if failure == "error":
                yield host_pb.LlmStreamNextResponse(error=_worker_error("LlmStreamNext failed"))
                return
            if failure == "empty":
                yield host_pb.LlmStreamNextResponse()
                return
            value = _envelope_value(request.request)
            yield host_pb.LlmStreamNextResponse(value=_encode_payload(common_pb.JsonValue, {"next_stream": value}))

        return stream()

    async def DecodeLlmCodecRequest(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("DecodeLlmCodecRequest") == "error":
            return host_pb.DecodeLlmCodecRequestResponse(error=_worker_error("DecodeLlmCodecRequest failed"))
        value = _envelope_value(request.request)
        model = value.get("content", {}).get("model")
        return host_pb.DecodeLlmCodecRequestResponse(
            value=_encode_payload(
                llm_pb.AnnotatedLlmRequest,
                {"messages": [], "model": model},
            )
        )

    async def EncodeLlmCodecRequest(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("EncodeLlmCodecRequest") == "error":
            return host_pb.EncodeLlmCodecRequestResponse(error=_worker_error("EncodeLlmCodecRequest failed"))
        return host_pb.EncodeLlmCodecRequestResponse(value=request.original_request)

    async def DecodeLlmCodecResponse(self, request: Any) -> Any:
        self.requests.append(request)
        if self.failures.get("DecodeLlmCodecResponse") == "error":
            return host_pb.DecodeLlmCodecResponseResponse(error=_worker_error("DecodeLlmCodecResponse failed"))
        value = _envelope_value(request.response)
        return host_pb.DecodeLlmCodecResponseResponse(
            value=plugin_api._protocol.pack(
                llm_pb.AnnotatedLlmResponse,
                {"model": value.get("model"), "message": value.get("message")},
            )
        )

    def _host_ack(self, method: str) -> Any:
        failure = self.failures.get(method)
        if failure:
            raise WorkerSdkError(f"{method} failed" if failure == "error" else "host call failed")
        return getattr(host_pb, f"{method}Response")()


class AllSurfacesPlugin(WorkerPlugin):
    plugin_id = "tests.python_worker"

    def __init__(self) -> None:
        self.conditional_middleware_calls: list[tuple[set[RuntimeRegistrationKind], str]] = []

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

        async def event_metadata(event: Json) -> dict[str, Json]:
            return {"worker.event_name": event["name"]}

        async def mark_sanitize(event: Json, fields: Json) -> Json:
            return {**fields, "data": {"sanitized": f"mark:{event['name']}"}, "metadata": None}

        async def scope_start_sanitize(event: Json, fields: Json) -> Json:
            return {**fields, "data": {"sanitized": f"scope-start:{event['name']}"}, "metadata": None}

        async def scope_end_sanitize(event: Json, fields: Json) -> Json:
            return {**fields, "data": {"sanitized": f"scope-end:{event['name']}"}, "metadata": None}

        def tool_sanitize(name: str, value: Json) -> Json:
            return _tag(value, f"sanitize_{name}")

        def tool_block(name: str, value: Json) -> str | None:
            del name, value
            return "tool blocked"

        async def tool_request(name: str, value: Json) -> Json:
            return _tag(value, f"request_{name}")

        async def tool_execution(context: ToolExecutionContext, next_call: ToolNext) -> ToolExecutionInterceptOutcome:
            result = await next_call.call(_tag(context.args, f"execute_{context.tool_name}"))
            return ToolExecutionInterceptOutcome(
                result=_tag(result.result, "tool_execution"),
                annotation=result.annotation,
                pending_marks=[PendingMarkSpec("worker.tool.execution")],
            )

        def llm_sanitize_request(request: Json, context: LlmSanitizeRequestContext) -> Json:
            del context
            return _tag_llm_request(request, "llm_sanitize_request")

        async def llm_sanitize_response(response: Json, context: LlmSanitizeResponseContext) -> Json:
            del context
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

        async def llm_execution(name: str, request: Json, _context: Any, next_call: Any) -> Json:
            result = await next_call.call(_tag_llm_request(request, f"llm_execute_{name}"))
            return _tag(result, "llm_execution")

        async def llm_stream_execution(name: str, request: Json, _context: Any, next_call: Any) -> AsyncIterator[Json]:
            stream = next_call.call(_tag_llm_request(request, f"llm_stream_{name}"))
            async for chunk in stream:
                yield _tag(chunk, "llm_stream_execution")

        async def initial_gate(kinds: set[RuntimeRegistrationKind], name: str) -> str | None:
            await asyncio.sleep(0)
            self.conditional_middleware_calls.append((kinds, name))
            return "initial gate" if name == "missing-target" else None

        ctx.register_subscriber("subscriber", subscriber)
        ctx.register_conditional_middleware_guardrail(
            "initial_gate",
            {RuntimeRegistrationKind.SUBSCRIBER},
            "missing-target",
            initial_gate,
        )
        ctx.register_event_metadata_injector("event_metadata", event_metadata, priority=1)
        ctx.register_mark_sanitize_guardrail("event_sanitize", mark_sanitize, priority=1)
        ctx.register_scope_sanitize_start_guardrail("event_sanitize", scope_start_sanitize, priority=2)
        ctx.register_scope_sanitize_end_guardrail("scope_end_sanitize", scope_end_sanitize, priority=3)
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


async def test_register_returns_initial_conditional_middleware_guardrail(service: _WorkerService) -> None:
    response = await _register(service)
    assert len(response.plan.conditional_middleware_guardrails) == 1
    gate = response.plan.conditional_middleware_guardrails[0]
    assert gate.name == "initial_gate"
    assert list(gate.kinds) == [common_pb.REGISTRATION_SURFACE_SUBSCRIBER]
    assert gate.registration_name == "missing-target"
    assert gate.callback


async def test_register_rejects_initial_callback_name_owned_by_runtime(service: _WorkerService) -> None:
    service._runtime._conditional_middleware_callbacks["initial_gate"] = lambda _kinds, _name: "existing runtime gate"

    response = await service.Register(
        _register_request(plugin_id=plugin_api._plugin_id(service._plugin)),
        AbortContext(),
    )

    assert response.HasField("error")
    assert "already registered" in response.error.message
    assert service._handlers.registrations == []
    assert list(service._runtime._conditional_middleware_callbacks) == ["initial_gate"]
    decision = await service.Invoke(
        _invoke_request(
            "initial_gate",
            common_pb.REGISTRATION_SURFACE_CONDITIONAL_MIDDLEWARE_GUARDRAIL,
            conditional_middleware=runtime_pb.ConditionalMiddlewareInvocation(
                kinds=[common_pb.REGISTRATION_SURFACE_SUBSCRIBER],
                registration_name="target-subscriber",
            ),
        ),
        AbortContext(),
    )
    assert decision.WhichOneof("result") == "guardrail"
    assert decision.guardrail.block_reason == "existing runtime gate"


def test_generated_proto_matches_worker_contract() -> None:
    assert WORKER_PROTOCOL == "nvidia.nemo.relay.worker.v1"
    service = worker_pb.DESCRIPTOR.services_by_name["PluginWorkerService"]
    assert len(service.methods) == 8
    assert service.methods_by_name["InvokeStream"].input_type.name == "InvokeStreamRequest"
    assert worker_pb.DESCRIPTOR.message_types_by_name["InvokeRequest"].oneofs_by_name["invocation"]


async def test_health_handshake_validate_register_and_all_surfaces(service: _WorkerService) -> None:
    health = await service.Health(
        worker_pb.HealthRequest(credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN)),
        AbortContext(),
    )
    assert health.status == runtime_pb.WORKER_HEALTH_STATUS_READY
    assert health.identity.plugin_id == "tests.python_worker"
    assert health.identity.protocol.name == WORKER_PROTOCOL
    assert health.identity.sdk_name == "nemo-relay-plugin"
    assert health.identity.sdk_version == plugin_api._SDK_VERSION
    assert health.identity.runtime_name == "python"

    handshake = await service.Handshake(_handshake_request(), AbortContext())
    assert handshake.identity.plugin_id == "tests.python_worker"
    assert handshake.identity.plugin_kind == "tests.python_worker"
    assert handshake.identity.protocol.name == WORKER_PROTOCOL
    assert handshake.identity.sdk_version == plugin_api._SDK_VERSION
    assert set(handshake.supported_surfaces) == set(_all_expected_surfaces())

    validate = await service.Validate(
        worker_pb.ValidateRequest(
            plugin_id="tests.python_worker",
            config=_encode_payload(common_pb.JsonValue, {"warn": True}),
            credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
        ),
        AbortContext(),
    )
    diagnostics = [plugin_api._protocol.unpack(item) for item in validate.diagnostics.entries]
    assert diagnostics == [{"level": "warning", "code": "tests.warn", "message": "warning requested"}]

    register = await _register(service)
    registrations = [
        (registration.local_name, registration.surface, registration.priority, registration.break_chain)
        for registration in register.plan.registrations
    ]
    assert registrations == [
        ("subscriber", common_pb.REGISTRATION_SURFACE_SUBSCRIBER, 0, False),
        ("event_metadata", common_pb.REGISTRATION_SURFACE_EVENT_METADATA_INJECTOR, 1, False),
        ("event_sanitize", common_pb.REGISTRATION_SURFACE_MARK_SANITIZE_GUARDRAIL, 1, False),
        ("event_sanitize", common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_START_GUARDRAIL, 2, False),
        ("scope_end_sanitize", common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_END_GUARDRAIL, 3, False),
        ("tool_sanitize", common_pb.REGISTRATION_SURFACE_TOOL_SANITIZE_REQUEST_GUARDRAIL, 1, False),
        ("tool_sanitize", common_pb.REGISTRATION_SURFACE_TOOL_SANITIZE_RESPONSE_GUARDRAIL, 2, False),
        ("tool_conditional", common_pb.REGISTRATION_SURFACE_TOOL_CONDITIONAL_EXECUTION_GUARDRAIL, 3, False),
        ("tool_request", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT, 4, True),
        ("tool_execution", common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT, 5, False),
        ("llm_sanitize_request", common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL, 6, False),
        ("llm_sanitize_response", common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_RESPONSE_GUARDRAIL, 7, False),
        ("llm_conditional", common_pb.REGISTRATION_SURFACE_LLM_CONDITIONAL_EXECUTION_GUARDRAIL, 8, False),
        ("llm_request", common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT, 9, True),
        ("llm_execution", common_pb.REGISTRATION_SURFACE_LLM_EXECUTION_INTERCEPT, 10, False),
        ("llm_stream_execution", common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT, 11, False),
    ]


def test_sdk_version_uses_package_metadata_and_source_tree_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
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
def test_encode_payload_rejects_non_finite_numbers(value: Json) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        _encode_payload(common_pb.JsonValue, value)


@pytest.mark.parametrize(
    "value",
    [
        {1: "value"},
        {"nested": [{2: True}]},
    ],
)
def test_encode_payload_rejects_non_string_object_keys(value: Json) -> None:
    with pytest.raises(WorkerSdkError, match="JSON object keys must be strings"):
        _encode_payload(common_pb.JsonValue, value)


def test_encode_payload_rejects_cycles_and_allows_shared_subobjects() -> None:
    dict_cycle: Json = {}
    dict_cycle["self"] = dict_cycle
    list_cycle: Json = []
    list_cycle.append(list_cycle)

    for value in (dict_cycle, list_cycle):
        with pytest.raises(WorkerSdkError, match="must not contain circular references"):
            _encode_payload(common_pb.JsonValue, value)

    shared = {"value": True}
    envelope = _encode_payload(common_pb.JsonValue, {"first": shared, "second": shared})
    assert _envelope_value(envelope) == {
        "first": {"value": True},
        "second": {"value": True},
    }


@pytest.mark.parametrize("payload", [b"NaN", b"Infinity", b"-Infinity", b'{"nested":NaN}'])
def test_encode_payload_decode_rejects_non_standard_constants(payload: bytes) -> None:
    envelope = common_pb.JsonValue(json=payload)
    with pytest.raises(WorkerSdkError, match="non-standard JSON constant"):
        _decode_payload(envelope, "json value")


@pytest.mark.parametrize("payload", [b"{", b"\xff"])
def test_encode_payload_decode_normalizes_malformed_payload_errors(payload: bytes) -> None:
    envelope = common_pb.JsonValue(json=payload)
    with pytest.raises(WorkerSdkError, match="invalid json value"):
        _decode_payload(envelope, "json value")


@pytest.mark.parametrize(
    "schema",
    [event_pb.Event, llm_pb.LlmRequest, llm_pb.AnnotatedLlmRequest],
)
@pytest.mark.parametrize("value", [None, [], "scalar", 42])
def test_object_schema_envelopes_reject_non_objects(schema: str, value: Json) -> None:
    with pytest.raises(WorkerSdkError, match="must be a JSON object"):
        _encode_payload(schema, value)

    assert _envelope_value(_encode_payload(common_pb.JsonValue, value)) == value


@pytest.mark.parametrize(
    ("rpc_name", "request_factory", "streaming"),
    [
        ("Handshake", lambda: _handshake_request(), False),
        (
            "Health",
            lambda: worker_pb.HealthRequest(
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN)
            ),
            False,
        ),
        (
            "Validate",
            lambda: worker_pb.ValidateRequest(
                plugin_id="tests.python_worker",
                config=_encode_payload(common_pb.JsonValue, {}),
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            ),
            False,
        ),
        (
            "Register",
            lambda: worker_pb.RegisterRequest(
                plugin_id="tests.python_worker",
                config=_encode_payload(common_pb.JsonValue, {}),
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            ),
            False,
        ),
        ("Invoke", lambda: _tool_request("missing", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT, {}), False),
        (
            "InvokeStream",
            lambda: _invoke_request(
                "missing",
                common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            True,
        ),
        (
            "CancelInvocation",
            lambda: worker_pb.CancelInvocationRequest(
                invocation_id="invoke-1",
                reason="test",
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            ),
            False,
        ),
        (
            "Shutdown",
            lambda: worker_pb.ShutdownRequest(
                reason="test", credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN)
            ),
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
        setattr(request.credentials, field, "wrong")
        with pytest.raises(GrpcAbort) as exc_info:
            result = getattr(service, rpc_name)(request, AbortContext())
            if streaming:
                async for _chunk in result:
                    pass
            else:
                await result
        assert exc_info.value.code == grpc.StatusCode.UNAUTHENTICATED
        assert "credentials" in exc_info.value.details


async def test_validate_and_register_decode_errors_are_grpc_protocol_errors(service: _WorkerService) -> None:
    bad_config = _encode_payload(common_pb.JsonValue, {})
    bad_config.json = b"{"

    with pytest.raises(GrpcAbort) as validate_error:
        await service.Validate(
            worker_pb.ValidateRequest(
                plugin_id="tests.python_worker",
                config=bad_config,
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            ),
            AbortContext(),
        )
    assert validate_error.value.code == grpc.StatusCode.INVALID_ARGUMENT

    with pytest.raises(GrpcAbort) as register_error:
        await service.Register(
            worker_pb.RegisterRequest(
                plugin_id="tests.python_worker",
                config=bad_config,
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            ),
            AbortContext(),
        )
    assert register_error.value.code == grpc.StatusCode.INVALID_ARGUMENT

    with pytest.raises(TypeError):
        worker_pb.ValidateRequest(config=event_pb.Event())


async def test_base_plugin_defaults_context_errors_and_plugin_id_validation() -> None:
    base = WorkerPlugin()
    assert base.validate({"unused": True}) == []
    with pytest.raises(NotImplementedError):
        base.register(PluginContext(), {})
    with pytest.raises(WorkerSdkError, match="no runtime handle"):
        _ = PluginContext().runtime

    class CallablePluginId(WorkerPlugin):
        allows_multiple_components = True

        def plugin_id(self) -> str:  # ty: ignore[invalid-attribute-override]
            return "tests.callable_id"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config

    class InvalidPluginId(WorkerPlugin):
        plugin_id = ""

        def register(self, ctx: PluginContext, config: Json) -> None:
            del ctx, config

    callable_service = _service(CallablePluginId(), RecordingHostStub())
    health = await callable_service.Health(
        worker_pb.HealthRequest(credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN)),
        AbortContext(),
    )
    assert health.identity.plugin_id == "tests.callable_id"

    invalid_service = _service(InvalidPluginId(), RecordingHostStub())
    with pytest.raises(WorkerSdkError, match="plugin_id"):
        await invalid_service.Health(
            worker_pb.HealthRequest(
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN)
            ),
            AbortContext(),
        )


def test_plugin_context_rejects_duplicate_names_on_the_same_surface() -> None:
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
    assert registrations.count(("duplicate", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT)) == 1
    assert ("shared", common_pb.REGISTRATION_SURFACE_TOOL_SANITIZE_REQUEST_GUARDRAIL) in registrations
    assert ("shared", common_pb.REGISTRATION_SURFACE_TOOL_SANITIZE_RESPONSE_GUARDRAIL) in registrations


def test_plugin_context_registers_llm_sanitizers_under_standard_names() -> None:
    context = PluginContext()

    context.register_llm_sanitize_request_guardrail(
        "request",
        lambda request, codec_context: request if codec_context.codec.kind != "none" else None,
    )
    context.register_llm_sanitize_response_guardrail(
        "response",
        lambda response, codec_context: response if codec_context.codec.kind == "builtin" else None,
    )

    assert [(registration.local_name, registration.surface) for registration in context._handlers.registrations] == [
        ("request", common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL),
        ("response", common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_RESPONSE_GUARDRAIL),
    ]


async def test_execution_callback_receives_directional_codec_context() -> None:
    seen: list[plugin_api.LlmExecutionContext] = []

    class ContextualExecutionPlugin(WorkerPlugin):
        plugin_id = "tests.contextual_execution"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def execution(name: str, request: Json, context: Any, next_call: Any) -> Json:
                assert name == "model"
                assert request == {"headers": {}, "content": {"model": "gpt-test"}}
                seen.append(context)
                assert context.request_codec.codec == plugin_api.LlmCodecIdentity("builtin", "openai_chat")
                assert context.response_codec is not None
                assert context.response_codec.codec == plugin_api.LlmCodecIdentity("builtin", "openai_chat")
                request_codec = context.request_codec.resolve_codec()
                assert request_codec is not None
                await request_codec.decode(request)
                result = await next_call.call(request)
                response_codec = context.response_codec.resolve_codec()
                assert response_codec is not None
                await response_codec.decode(result)
                return result

            ctx.register_llm_execution_intercept("execution", execution)

    host = RecordingHostStub()
    service = _service(ContextualExecutionPlugin(), host)
    await _register(service)

    new_context = _ExecutionContext(
        request=runtime_pb.LlmCodecContext(
            codec=_wire_codec(kind="builtin", id="openai_chat"),
            capability_id="request-capability",
        ),
        response=runtime_pb.LlmCodecContext(
            codec=_wire_codec(kind="builtin", id="openai_chat"),
            capability_id="response-capability",
        ),
    )
    payload = _llm_payload(request={"content": {"model": "gpt-test"}})
    payload.execution_codec_context = new_context
    result = await _invoke_json_async(
        service,
        "execution",
        common_pb.REGISTRATION_SURFACE_LLM_EXECUTION_INTERCEPT,
        payload=payload,
    )
    assert "next_llm" in result

    assert len(seen) == 1
    codec_capabilities = [
        request.codec_capability_id
        for request in host.requests
        if isinstance(request, (host_pb.DecodeLlmCodecRequestRequest, host_pb.DecodeLlmCodecResponseRequest))
    ]
    assert codec_capabilities == ["request-capability", "response-capability"]


@pytest.mark.parametrize(
    ("proto_kind", "capability_prefix", "expected_kind", "resolves"),
    [
        ("none", None, "none", False),
        ("opaque", "opaque", "opaque", True),
    ],
)
def test_execution_context_distinguishes_absent_and_resolved_opaque_codecs(
    proto_kind: int,
    capability_prefix: str | None,
    expected_kind: str,
    resolves: bool,
) -> None:
    runtime = PluginRuntime(
        activation_id=ACTIVATION_ID,
        auth_token=AUTH_TOKEN,
        host_stub=RecordingHostStub(),
    )
    request = runtime_pb.LlmCodecContext(codec=_wire_codec(kind=proto_kind))
    response = runtime_pb.LlmCodecContext(codec=_wire_codec(kind=proto_kind))
    if capability_prefix is not None:
        request.capability_id = f"{capability_prefix}-request"
        response.capability_id = f"{capability_prefix}-response"
    invocation = runtime_pb.LlmExecutionInvocation(request_codec=request, response_codec=response)
    context = plugin_api._llm_execution_context(
        invocation,
        runtime,
        "invocation",
        response_required=True,
    )
    assert context.request_codec.codec == plugin_api.LlmCodecIdentity(expected_kind)
    assert (context.request_codec.resolve_codec() is not None) is resolves
    assert context.response_codec is not None
    assert context.response_codec.codec == plugin_api.LlmCodecIdentity(expected_kind)
    assert (context.response_codec.resolve_codec() is not None) is resolves


@pytest.mark.parametrize(
    ("invocation", "expected"),
    [
        (runtime_pb.LlmExecutionInvocation(), "request context is missing"),
        (
            runtime_pb.LlmExecutionInvocation(request_codec=runtime_pb.LlmCodecContext()),
            "request codec identity is missing",
        ),
        (
            runtime_pb.LlmExecutionInvocation(
                request_codec=runtime_pb.LlmCodecContext(codec=_wire_codec()),
                response_codec=runtime_pb.LlmCodecContext(),
            ),
            "response codec identity is missing",
        ),
        (
            runtime_pb.LlmExecutionInvocation(request_codec=runtime_pb.LlmCodecContext(codec=_wire_codec())),
            "response context is missing",
        ),
    ],
)
def test_execution_context_rejects_malformed_unary_contexts(
    invocation: runtime_pb.LlmExecutionInvocation,
    expected: str,
) -> None:
    runtime = PluginRuntime(
        activation_id=ACTIVATION_ID,
        auth_token=AUTH_TOKEN,
        host_stub=RecordingHostStub(),
    )

    with pytest.raises(WorkerSdkError, match=expected):
        plugin_api._llm_execution_context(
            invocation,
            runtime,
            "invocation",
            response_required=True,
        )


async def test_llm_sanitizers_receive_codec_context_and_can_omit_payloads() -> None:
    seen: list[tuple[str, LlmSanitizeRequestContext | LlmSanitizeResponseContext]] = []

    class ContextualSanitizerPlugin(WorkerPlugin):
        plugin_id = "tests.llm_sanitizer"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def request_sanitizer(request: Json, codec_context: LlmSanitizeRequestContext) -> None:
                del request
                seen.append(("request", codec_context))

            def response_sanitizer(response: Json, codec_context: LlmSanitizeResponseContext) -> None:
                del response
                seen.append(("response", codec_context))

            ctx.register_llm_sanitize_request_guardrail("request", request_sanitizer)
            ctx.register_llm_sanitize_response_guardrail("response", response_sanitizer)

    service = _service(ContextualSanitizerPlugin(), RecordingHostStub())
    await _register(service)
    for codec_kind, codec_id in [
        ("none", None),
        ("builtin", "openai_chat"),
        ("builtin", "openai_responses"),
        ("builtin", "anthropic_messages"),
        ("builtin", "gemini_generate_content"),
        ("runtime", "com.example.chat.v1"),
        ("opaque", None),
    ]:
        codec = _wire_codec(kind=codec_kind, id=codec_id)
        request_payload = _llm_payload(
            request={"content": {"prompt": "secret"}},
            response={"secret": "value"},
        )
        request_payload.request_sanitize_context = runtime_pb.LlmCodecContext(codec=codec)
        response_payload = _llm_payload(
            request={"content": {"prompt": "secret"}},
            response={"secret": "value"},
        )
        response_payload.response_sanitize_context = runtime_pb.LlmCodecContext(codec=codec)

        request_response = await service.Invoke(
            _invoke_request(
                "request",
                common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL,
                llm=request_payload,
            ),
            AbortContext(),
        )
        response_response = await service.Invoke(
            _invoke_request(
                "response",
                common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_RESPONSE_GUARDRAIL,
                llm=response_payload,
            ),
            AbortContext(),
        )
        assert request_response.WhichOneof("result") == "unchanged"
        assert response_response.WhichOneof("result") == "unchanged"

    assert seen == [
        ("request", LlmSanitizeRequestContext(plugin_api.LlmCodecIdentity("none"))),
        ("response", LlmSanitizeResponseContext(plugin_api.LlmCodecIdentity("none"))),
        ("request", LlmSanitizeRequestContext(plugin_api.LlmCodecIdentity("builtin", "openai_chat"))),
        ("response", LlmSanitizeResponseContext(plugin_api.LlmCodecIdentity("builtin", "openai_chat"))),
        ("request", LlmSanitizeRequestContext(plugin_api.LlmCodecIdentity("builtin", "openai_responses"))),
        ("response", LlmSanitizeResponseContext(plugin_api.LlmCodecIdentity("builtin", "openai_responses"))),
        (
            "request",
            LlmSanitizeRequestContext(plugin_api.LlmCodecIdentity("builtin", "anthropic_messages")),
        ),
        (
            "response",
            LlmSanitizeResponseContext(plugin_api.LlmCodecIdentity("builtin", "anthropic_messages")),
        ),
        ("request", LlmSanitizeRequestContext(plugin_api.LlmCodecIdentity("builtin", "gemini_generate_content"))),
        ("response", LlmSanitizeResponseContext(plugin_api.LlmCodecIdentity("builtin", "gemini_generate_content"))),
        ("request", LlmSanitizeRequestContext(plugin_api.LlmCodecIdentity("runtime", "com.example.chat.v1"))),
        ("response", LlmSanitizeResponseContext(plugin_api.LlmCodecIdentity("runtime", "com.example.chat.v1"))),
        ("request", LlmSanitizeRequestContext(plugin_api.LlmCodecIdentity("opaque"))),
        ("response", LlmSanitizeResponseContext(plugin_api.LlmCodecIdentity("opaque"))),
    ]


async def test_llm_sanitizers_resolve_directional_codec_proxies() -> None:
    calls: list[tuple[str, Json]] = []

    class CodecSanitizerPlugin(WorkerPlugin):
        plugin_id = "tests.llm_codec_proxy"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def request_sanitizer(
                request: Json,
                context: LlmSanitizeRequestContext,
            ) -> Json:
                codec = context.resolve_codec()
                assert codec is not None
                annotated = await codec.decode(request)
                calls.append(("request_decode", annotated))
                return await codec.encode(annotated, request)

            async def response_sanitizer(
                response: Json,
                context: LlmSanitizeResponseContext,
            ) -> Json:
                codec = context.resolve_codec()
                assert codec is not None
                calls.append(("response_decode", await codec.decode(response)))
                return response

            ctx.register_llm_sanitize_request_guardrail("request", request_sanitizer)
            ctx.register_llm_sanitize_response_guardrail("response", response_sanitizer)

    host = RecordingHostStub()
    service = _service(CodecSanitizerPlugin(), host)
    await _register(service)

    request_payload = _llm_payload(
        request={"headers": {}, "content": {"model": "runtime-model"}},
    )
    request_payload.request_sanitize_context = runtime_pb.LlmCodecContext(
        codec=_wire_codec(kind="runtime", id="com.example.runtime"), capability_id="request-capability"
    )
    request_result = await service.Invoke(
        _invoke_request(
            "request",
            common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL,
            llm=request_payload,
        ),
        AbortContext(),
    )
    assert request_result.WhichOneof("result") == "llm_request", request_result

    response_payload = _llm_payload(response={"model": "opaque-model", "message": "secret"})
    response_payload.response_sanitize_context = runtime_pb.LlmCodecContext(
        codec=_wire_codec(kind="opaque"), capability_id="response-capability"
    )
    response_result = await service.Invoke(
        _invoke_request(
            "response",
            common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_RESPONSE_GUARDRAIL,
            llm=response_payload,
        ),
        AbortContext(),
    )
    assert response_result.WhichOneof("result") == "llm_response", response_result
    assert calls == [
        ("request_decode", {"messages": [], "model": "runtime-model"}),
        (
            "response_decode",
            {"model": "opaque-model", "message": "secret"},
        ),
    ]
    assert [request.codec_capability_id for request in host.requests if hasattr(request, "codec_capability_id")] == [
        "request-capability",
        "request-capability",
        "response-capability",
    ]


@pytest.mark.parametrize(
    ("failure", "surface", "registration_name"),
    [
        ("DecodeLlmCodecRequest", common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL, "request"),
        ("EncodeLlmCodecRequest", common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL, "request"),
        ("DecodeLlmCodecResponse", common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_RESPONSE_GUARDRAIL, "response"),
    ],
)
async def test_llm_sanitizer_codec_rpc_failures_return_invocation_errors(
    failure: str,
    surface: int,
    registration_name: str,
) -> None:
    class CodecFailurePlugin(WorkerPlugin):
        plugin_id = "tests.llm_codec_failure"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def request_sanitizer(
                request: Json,
                context: LlmSanitizeRequestContext,
            ) -> Json:
                codec = context.resolve_codec()
                assert codec is not None
                annotated = await codec.decode(request)
                return await codec.encode(annotated, request)

            async def response_sanitizer(
                response: Json,
                context: LlmSanitizeResponseContext,
            ) -> Json:
                codec = context.resolve_codec()
                assert codec is not None
                await codec.decode(response)
                return response

            ctx.register_llm_sanitize_request_guardrail("request", request_sanitizer)
            ctx.register_llm_sanitize_response_guardrail("response", response_sanitizer)

    host = RecordingHostStub()
    host.failures[failure] = "error"
    service = _service(CodecFailurePlugin(), host)
    await _register(service)

    if surface == common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL:
        payload = _llm_payload(request={"headers": {}, "content": {"model": "test"}})
        payload.request_sanitize_context = runtime_pb.LlmCodecContext(
            codec=_wire_codec(kind="opaque"), capability_id="request-capability"
        )
    else:
        payload = _llm_payload(response={"model": "test", "message": "secret"})
        payload.response_sanitize_context = runtime_pb.LlmCodecContext(
            codec=_wire_codec(kind="opaque"), capability_id="response-capability"
        )

    result = await service.Invoke(
        _invoke_request(registration_name, surface, llm=payload),
        AbortContext(),
    )
    assert result.WhichOneof("result") == "error"
    assert f"{failure} failed" in result.error.message


@pytest.mark.parametrize(
    "surface",
    [
        common_pb.REGISTRATION_SURFACE_MARK_SANITIZE_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_START_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_END_GUARDRAIL,
    ],
)
async def test_event_sanitizer_surfaces_receive_context_and_return_all_fields(
    service: _WorkerService, surface: int
) -> None:
    await _register(service)
    registration = {
        common_pb.REGISTRATION_SURFACE_MARK_SANITIZE_GUARDRAIL: "event_sanitize",
        common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_START_GUARDRAIL: "event_sanitize",
        common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_END_GUARDRAIL: "scope_end_sanitize",
    }[surface]
    sanitizer = {
        common_pb.REGISTRATION_SURFACE_MARK_SANITIZE_GUARDRAIL: "mark",
        common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_START_GUARDRAIL: "scope-start",
        common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_END_GUARDRAIL: "scope-end",
    }[surface]
    response = await service.Invoke(
        _invoke_request(
            registration,
            surface,
            event=_encode_payload(
                event_pb.Event,
                {
                    "kind": "mark" if surface == common_pb.REGISTRATION_SURFACE_MARK_SANITIZE_GUARDRAIL else "scope",
                    "name": "worker-event",
                    "data": {"secret": True},
                    "category_profile": {"subtype": "test"},
                    "metadata": {"secret": True},
                },
            ),
        ),
        AbortContext(),
    )
    assert _envelope_value(response.event_sanitize) == {
        "data": {"sanitized": f"{sanitizer}:worker-event"},
        "category_profile": {"subtype": "test"},
        "metadata": None,
    }


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
        worker_pb.ValidateRequest(
            plugin_id="tests.dict_diagnostic",
            config=_json_value(None),
            credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
        ),
        AbortContext(),
    )
    assert [plugin_api._protocol.unpack(item) for item in response.diagnostics.entries] == [
        {"level": "error", "code": "dict.diag", "message": "dict diagnostic"}
    ]


def test_config_diagnostic_normalizes_string_level_and_optional_fields() -> None:
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
async def test_validate_rejects_malformed_diagnostics(diagnostic: Any, message: str) -> None:
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


async def test_async_validate_and_register_hooks_are_awaited() -> None:
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
    assert [plugin_api._protocol.unpack(item) for item in validate.diagnostics.entries] == [
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
    assert [(item.local_name, item.surface) for item in register.plan.registrations] == [
        ("async_request", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT)
    ]
    response = await service.Invoke(
        _tool_request("async_request", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT, {"query": "relay"}),
        AbortContext(),
    )
    assert response.WhichOneof("result") == "tool_value", response
    assert _envelope_value(response.tool_value) == {
        "tool_name": "lookup",
        "args": {"query": "relay"},
        "config": {"mode": "async"},
    }
    assert lifecycle == ["validate", "register"]


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
            del config
            ctx.register_conditional_middleware_guardrail(
                "failed_gate",
                {RuntimeRegistrationKind.SUBSCRIBER},
                "target-subscriber",
                lambda _kinds, _name: "must not survive",
            )
            raise RuntimeError("register boom")

    class FailingInvokePlugin(WorkerPlugin):
        plugin_id = "tests.failing_invoke"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def fail(tool_name: str, args: Json) -> Json:
                del tool_name, args
                raise RuntimeError("invoke boom")

            ctx.register_tool_request_intercept("fail", fail)

    class FailingEventMetadataPlugin(WorkerPlugin):
        plugin_id = "tests.failing_event_metadata"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def fail(event: Json) -> dict[str, Json]:
                del event
                raise RuntimeError("event metadata boom")

            ctx.register_event_metadata_injector("fail", fail)

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
    leaked_gate = await register_service.Invoke(
        _invoke_request(
            "failed_gate",
            common_pb.REGISTRATION_SURFACE_CONDITIONAL_MIDDLEWARE_GUARDRAIL,
            conditional_middleware=runtime_pb.ConditionalMiddlewareInvocation(
                kinds=[common_pb.REGISTRATION_SURFACE_SUBSCRIBER],
                registration_name="target-subscriber",
            ),
        ),
        AbortContext(),
    )
    assert leaked_gate.WhichOneof("result") == "error"
    assert "is not registered" in leaked_gate.error.message

    invoke_service = _service(FailingInvokePlugin(), RecordingHostStub())
    await _register(invoke_service)
    response = await invoke_service.Invoke(
        _tool_request("fail", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT, {}), AbortContext()
    )
    assert response.WhichOneof("result") == "error"
    assert "invoke boom" in response.error.message

    event_service = _service(FailingEventMetadataPlugin(), RecordingHostStub())
    await _register(event_service, plugin_id="tests.failing_event_metadata")
    event_response = await event_service.Invoke(
        _invoke_request(
            "fail",
            common_pb.REGISTRATION_SURFACE_EVENT_METADATA_INJECTOR,
            event=_encode_payload(event_pb.Event, {"name": "event"}),
        ),
        AbortContext(),
    )
    assert event_response.WhichOneof("result") == "error"
    assert "event metadata boom" in event_response.error.message


async def test_register_is_idempotent_and_rejects_changed_component_config() -> None:
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
    with pytest.raises(GrpcAbort) as changed:
        await service.Register(
            _register_request({"tag": "second"}, plugin_id="tests.register_config"),
            AbortContext(),
        )

    assert plugin.register_calls == 1
    assert list(repeated.plan.registrations) == list(first.plan.registrations)
    assert changed.value.code == grpc.StatusCode.FAILED_PRECONDITION

    result = await service.Invoke(
        _tool_request("configured", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT, {"query": "relay"}),
        AbortContext(),
    )
    assert _envelope_value(result.tool_value) == {"query": "relay", "tag": "first"}


async def test_concurrent_register_calls_install_handlers_once() -> None:
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
    assert [registration.local_name for registration in responses[0].plan.registrations] == ["subscriber"]
    assert list(responses[1].plan.registrations) == list(responses[0].plan.registrations)


async def test_concurrent_register_calls_reject_a_different_config() -> None:
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
    responses = await asyncio.gather(*requests, return_exceptions=True)

    assert sum(isinstance(response, GrpcAbort) for response in responses) == 1
    error = next(response for response in responses if isinstance(response, GrpcAbort))
    assert "different component config" in error.details


async def test_unary_invoke_success_paths(service: _WorkerService, host_stub: RecordingHostStub) -> None:
    await _register(service)

    subscriber = await service.Invoke(
        _invoke_request(
            "subscriber",
            common_pb.REGISTRATION_SURFACE_SUBSCRIBER,
            event=_encode_payload(event_pb.Event, {"name": "event"}),
            scope=runtime_pb.ScopeContext(scope_stack_id="invoke-stack", parent_scope_id="parent-scope"),
        ),
        AbortContext(),
    )
    assert subscriber.WhichOneof("result") == "completed"
    mark_request = _last_request(host_stub, host_pb.EmitMarkRequest)
    assert mark_request.mark.name == "tests.subscriber"
    assert mark_request.scope.scope_stack_id == "invoke-stack"
    assert mark_request.scope.parent_scope_id == "parent-scope"

    conditional_middleware = await service.Invoke(
        _invoke_request(
            "initial_gate",
            common_pb.REGISTRATION_SURFACE_CONDITIONAL_MIDDLEWARE_GUARDRAIL,
            conditional_middleware=runtime_pb.ConditionalMiddlewareInvocation(
                kinds=[common_pb.REGISTRATION_SURFACE_SUBSCRIBER],
                registration_name="missing-target",
            ),
        ),
        AbortContext(),
    )
    assert conditional_middleware.guardrail.block_reason == "initial gate"
    allowed_conditional_middleware = await service.Invoke(
        _invoke_request(
            "initial_gate",
            common_pb.REGISTRATION_SURFACE_CONDITIONAL_MIDDLEWARE_GUARDRAIL,
            conditional_middleware=runtime_pb.ConditionalMiddlewareInvocation(
                kinds=[common_pb.REGISTRATION_SURFACE_SUBSCRIBER],
                registration_name="allowed-target",
            ),
        ),
        AbortContext(),
    )
    assert allowed_conditional_middleware.guardrail.block_reason == ""
    plugin = service._plugin
    assert isinstance(plugin, AllSurfacesPlugin)
    assert plugin.conditional_middleware_calls == [
        ({RuntimeRegistrationKind.SUBSCRIBER}, "missing-target"),
        ({RuntimeRegistrationKind.SUBSCRIBER}, "allowed-target"),
    ]

    event_metadata = await service.Invoke(
        _invoke_request(
            "event_metadata",
            common_pb.REGISTRATION_SURFACE_EVENT_METADATA_INJECTOR,
            event=_encode_payload(event_pb.Event, {"name": "metadata-event"}),
        ),
        AbortContext(),
    )
    assert event_metadata.WhichOneof("result") == "metadata_additions"
    assert _envelope_value(event_metadata.metadata_additions) == {"worker.event_name": "metadata-event"}

    tool_sanitize_request = await _invoke_json_async(
        service, "tool_sanitize", common_pb.REGISTRATION_SURFACE_TOOL_SANITIZE_REQUEST_GUARDRAIL
    )
    assert tool_sanitize_request["tag"] == "sanitize_lookup"
    tool_sanitize_response = await _invoke_json_async(
        service, "tool_sanitize", common_pb.REGISTRATION_SURFACE_TOOL_SANITIZE_RESPONSE_GUARDRAIL
    )
    assert tool_sanitize_response["tag"] == "sanitize_lookup"

    tool_conditional = await service.Invoke(
        _tool_request(
            "tool_conditional", common_pb.REGISTRATION_SURFACE_TOOL_CONDITIONAL_EXECUTION_GUARDRAIL, {"query": "relay"}
        ),
        AbortContext(),
    )
    assert tool_conditional.guardrail.block_reason == "tool blocked"

    tool_request = await _invoke_json_async(
        service, "tool_request", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT
    )
    assert tool_request["tag"] == "request_lookup"
    tool_execution = await _invoke_tool_execution_async(service, "tool_execution")
    assert tool_execution["result"]["tag"] == "tool_execution"
    assert tool_execution["result"]["next_tool"]["tag"] == "execute_lookup"
    assert tool_execution["annotation"] == {"source": "host"}
    assert tool_execution["pending_marks"][0]["name"] == "worker.tool.execution"

    llm_sanitize_request = await _invoke_json_async(
        service,
        "llm_sanitize_request",
        common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL,
        payload=_llm_payload(request={"content": {"prompt": "hello"}}),
    )
    assert llm_sanitize_request["content"]["llm_sanitize_request"]

    llm_sanitize_response = await _invoke_json_async(
        service,
        "llm_sanitize_response",
        common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_RESPONSE_GUARDRAIL,
        payload=_llm_payload(response={"answer": "hello"}),
    )
    assert llm_sanitize_response["tag"] == "llm_sanitize_response"

    llm_conditional = await service.Invoke(
        _invoke_request(
            "llm_conditional",
            common_pb.REGISTRATION_SURFACE_LLM_CONDITIONAL_EXECUTION_GUARDRAIL,
            llm=_llm_payload(request={"content": {"prompt": "hello"}}),
        ),
        AbortContext(),
    )
    assert llm_conditional.guardrail.block_reason == "llm blocked"

    llm_request = await service.Invoke(
        _invoke_request(
            "llm_request",
            common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(
                request={"content": {"prompt": "hello"}},
                annotated={"messages": [], "extra": {"before": True}},
            ),
        ),
        AbortContext(),
    )
    outcome = _envelope_value(llm_request.llm_request_outcome)
    assert outcome["request"]["content"]["llm_request"]
    assert outcome["annotated_request"]["tag"] == "annotated"
    assert outcome["pending_marks"] == [{"name": "worker.pending", "data": {"source": "python"}}]

    request_only = await service.Invoke(
        _invoke_request(
            "llm_request",
            common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(request={"content": {"prompt": "hello"}}),
        ),
        AbortContext(),
    )
    request_only_outcome = _envelope_value(request_only.llm_request_outcome)
    assert request_only_outcome["request"]["content"]["llm_request"]
    assert request_only_outcome["annotated_request"] is None
    assert request_only_outcome["pending_marks"] == outcome["pending_marks"]

    llm_execution = await _invoke_json_async(
        service,
        "llm_execution",
        common_pb.REGISTRATION_SURFACE_LLM_EXECUTION_INTERCEPT,
        payload=_llm_payload(model_name="gpt-test", request={"content": {"prompt": "hello"}}),
    )
    assert llm_execution["tag"] == "llm_execution"
    assert llm_execution["next_llm"]["content"]["llm_execute_gpt-test"]


async def test_unary_invoke_failure_paths(service: _WorkerService) -> None:
    await _register(service)
    invalid = _tool_request("tool_request", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT, {})
    invalid.tool_request_intercept.value.json = b"{"
    missing = _invoke_request("subscriber", common_pb.REGISTRATION_SURFACE_SUBSCRIBER)
    bad_scope = _invoke_request(
        "subscriber",
        common_pb.REGISTRATION_SURFACE_SUBSCRIBER,
        event=_encode_payload(event_pb.Event, {"name": "event"}),
        scope=runtime_pb.ScopeContext(),
    )
    parent_only = _invoke_request(
        "subscriber",
        common_pb.REGISTRATION_SURFACE_SUBSCRIBER,
        event=_encode_payload(event_pb.Event, {"name": "event"}),
        scope=runtime_pb.ScopeContext(parent_scope_id="parent-only"),
    )
    for request in (invalid, missing, bad_scope, parent_only):
        with pytest.raises(GrpcAbort) as error:
            await service.Invoke(request, AbortContext())
        assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT
    missing_handler = await service.Invoke(
        _tool_request("missing", common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT, {}), AbortContext()
    )
    assert missing_handler.WhichOneof("result") == "error"
    assert "not registered" in missing_handler.error.message


@pytest.mark.parametrize("invalid_part", ["request", "annotated_request"])
async def test_llm_request_intercept_rejects_non_object_typed_results(invalid_part: str) -> None:
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
            common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(
                request={"content": {"prompt": "hello"}},
                annotated={"messages": []},
            ),
        ),
        AbortContext(),
    )

    assert response.WhichOneof("result") == "error"
    assert "must be a JSON object" in response.error.message


async def test_tool_execution_intercept_rejects_raw_result() -> None:
    class LegacyResultPlugin(WorkerPlugin):
        plugin_id = "tests.legacy_tool_execution_result"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def raw_result(context: ToolExecutionContext, next_call: ToolNext) -> Any:
                del context, next_call
                return {"legacy_result": True}

            ctx.register_tool_execution_intercept("legacy", raw_result)

    service = _service(LegacyResultPlugin(), RecordingHostStub())
    await _register(service)
    response = await service.Invoke(
        _tool_request("legacy", common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT, {}),
        AbortContext(),
    )

    assert response.WhichOneof("result") == "error"
    assert "must return ToolExecutionInterceptOutcome" in response.error.message


@pytest.mark.parametrize(
    "field,wrong_type",
    [
        ("subscriber", common_pb.JsonValue),
        ("tool_request_intercept", event_pb.Event),
        ("llm_sanitize_request_guardrail", common_pb.JsonValue),
        ("llm_conditional_execution_guardrail", llm_pb.LlmRequest),
        ("llm_execution_intercept", llm_pb.LlmRequest),
        ("llm_request_intercept", llm_pb.AnnotatedLlmRequest),
    ],
)
def test_invoke_rejects_mismatched_message_types(field, wrong_type):
    with pytest.raises(TypeError):
        worker_pb.InvokeRequest(**{field: wrong_type()})


async def test_invoke_stream_rejects_malformed_llm_request(service: _WorkerService) -> None:
    await _register(service)
    request = _invoke_request(
        "llm_stream_execution", common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT, llm=_llm_payload()
    )
    request.request.content.json = b"{"
    with pytest.raises(GrpcAbort) as error:
        _ = [chunk async for chunk in service.InvokeStream(request, AbortContext())]
    assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT


async def test_llm_request_intercept_can_return_request_without_annotation() -> None:
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
            common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(request={"content": {"prompt": "hello"}}),
        ),
        AbortContext(),
    )
    outcome = _envelope_value(response.llm_request_outcome)
    assert outcome["request"]["content"]["request_only"]
    assert outcome["annotated_request"] is None
    assert outcome["pending_marks"] == []


async def test_llm_request_intercept_preserves_optimization_contribution_worker_envelope(
    optimization_contribution_fixture: Json,
) -> None:
    fixture = optimization_contribution_fixture

    class OptimizationPlugin(WorkerPlugin):
        plugin_id = "tests.optimization_contribution"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def llm_request(name: str, request: Json, annotated: Json | None) -> LlmRequestInterceptOutcome:
                del name
                return LlmRequestInterceptOutcome(
                    request=request,
                    annotated_request=annotated,
                    optimization_contributions=[LlmOptimizationContribution.from_json(fixture)],
                )

            ctx.register_llm_request_intercept("optimization", llm_request)

    service = _service(OptimizationPlugin(), RecordingHostStub())
    await _register(service)
    response = await service.Invoke(
        _invoke_request(
            "optimization",
            common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(request={"content": {"prompt": "hello"}}),
        ),
        AbortContext(),
    )

    assert response.WhichOneof("result") == "llm_request_outcome"
    assert response.llm_request_outcome.DESCRIPTOR.full_name == "nvidia.nemo.relay.worker.v1.LlmRequestInterceptOutcome"
    outcome = _envelope_value(response.llm_request_outcome)
    assert outcome["optimization_contributions"] == [fixture]
    assert outcome["optimization_contributions"][0]["future_top_level_field"] == {"preserved": True}


async def test_stream_invoke_success_and_failures(service: _WorkerService, host_stub: RecordingHostStub) -> None:
    await _register(service)

    chunks = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "llm_stream_execution",
                common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(model_name="gpt-test", request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert [_stream_value(chunk)["tag"] for chunk in chunks] == ["llm_stream_execution"]
    assert _stream_value(chunks[0])["next_stream"]["content"]["llm_stream_gpt-test"]

    with pytest.raises(ValueError):
        worker_pb.InvokeStreamRequest(
            tool_request_intercept=runtime_pb.ToolValueInvocation()  # ty: ignore[unknown-argument]
        )

    missing_handler = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "missing",
                common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert "not registered" in missing_handler[0].error.message

    missing_payload = worker_pb.InvokeStreamRequest(
        credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
        invocation_id="missing",
        registration_name="llm_stream_execution",
        continuation_id="next",
    )
    with pytest.raises(GrpcAbort) as error:
        _ = [chunk async for chunk in service.InvokeStream(missing_payload, AbortContext())]
    assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT

    host_stub.failures["LlmStreamNext"] = "error"
    host_error = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "llm_stream_execution",
                common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
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
                common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert "invalid host response" in empty_chunk[0].error.message


async def test_stream_callback_exception_is_structured() -> None:
    class FailingStreamPlugin(WorkerPlugin):
        plugin_id = "tests.stream_fail"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def fail(name: str, request: Json, context: Any, next_call: Any) -> AsyncIterator[Json]:
                del name, request, context, next_call
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
                common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
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

            def stream(name: str, request: Json, context: Any, next_call: Any) -> list[Json]:
                del name, request, context, next_call
                return [{"sync": True}]

            ctx.register_llm_stream_execution_intercept("sync_stream", stream)

    service = _service(SyncStreamPlugin(), RecordingHostStub())
    await _register(service)
    chunks = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "sync_stream",
                common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert _stream_value(chunks[0]) == {"sync": True}


@pytest.mark.parametrize("invalid_stream", [{"chunk": True}, "chunk", b"chunk", 42])
async def test_stream_callback_rejects_scalar_and_mapping_results(invalid_stream: Any) -> None:
    class InvalidStreamPlugin(WorkerPlugin):
        plugin_id = "tests.invalid_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            def stream(name: str, request: Json, context: Any, next_call: Any) -> Any:
                del name, request, context, next_call
                return invalid_stream

            ctx.register_llm_stream_execution_intercept("invalid_stream", stream)

    service = _service(InvalidStreamPlugin(), RecordingHostStub())
    await _register(service)
    chunks = [
        chunk
        async for chunk in service.InvokeStream(
            _invoke_request(
                "invalid_stream",
                common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(request={"content": {"prompt": "hello"}}),
            ),
            AbortContext(),
        )
    ]
    assert len(chunks) == 1
    assert "stream callback must return" in chunks[0].error.message


@pytest.mark.parametrize("ordinal", [None, 0, 1])
async def test_runtime_plugin_registration_preserves_optional_component_ordinal(ordinal):
    class Host(RecordingHostStub):
        async def ListRuntimeRegistrations(self, request):
            response = await super().ListRuntimeRegistrations(request)
            owner = response.registrations[0].owner.plugin
            if ordinal is None:
                owner.ClearField("component_ordinal")
            else:
                owner.component_ordinal = ordinal
            return response

    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=Host())
    registrations = await runtime.list_runtime_registrations()
    assert registrations[0].owner.component_ordinal == ordinal


async def test_runtime_host_calls_and_scope_context(host_stub: RecordingHostStub) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    assert runtime.current_scope_stack_id() is None
    assert runtime.current_parent_scope_id() is None

    diagnostics = await runtime.runtime_diagnostics()
    assert isinstance(diagnostics, RuntimeDiagnostics)
    assert diagnostics.entries == (
        RuntimeDiagnostic(code="alpha", message="first", count=1),
        RuntimeDiagnostic(code="zeta", message="latest", count=3),
    )
    assert diagnostics.get("zeta") == RuntimeDiagnostic(code="zeta", message="latest", count=3)
    assert diagnostics.get("missing") is None

    registrations = await runtime.list_runtime_registrations({RuntimeRegistrationKind.SUBSCRIBER})
    assert registrations == [
        RuntimeRegistrationIdentity(
            kind=RuntimeRegistrationKind.SUBSCRIBER,
            local_name="opentelemetry",
            effective_name="nemo-relay-plugin.v1.observability:1:opentelemetry",
            owner=plugin_api.RuntimeRegistrationOwner(
                kind=RuntimeRegistrationOwnerKind.PLUGIN,
                plugin_kind="observability",
                component_ordinal=1,
            ),
        )
    ]
    list_request = _last_request(host_stub, host_pb.ListRuntimeRegistrationsRequest)
    assert list_request.credentials.activation_id == ACTIVATION_ID
    assert list_request.credentials.auth_token == AUTH_TOKEN
    assert list(list_request.kinds) == [common_pb.REGISTRATION_SURFACE_SUBSCRIBER]
    gate = await runtime.register_conditional_middleware_guardrail(
        "pause-otel",
        {RuntimeRegistrationKind.SUBSCRIBER},
        registrations[0].effective_name,
        lambda _kinds, _name: "temporarily disabled",
    )
    assert isinstance(gate, ConditionalMiddlewareGuardrailHandle)
    register_request = _last_request(host_stub, host_pb.RegisterConditionalMiddlewareGuardrailRequest)
    assert register_request.credentials.activation_id == ACTIVATION_ID
    assert register_request.credentials.auth_token == AUTH_TOKEN
    assert register_request.registration.name == "pause-otel"
    assert list(register_request.registration.kinds) == [common_pb.REGISTRATION_SURFACE_SUBSCRIBER]
    assert register_request.registration.registration_name == registrations[0].effective_name
    assert register_request.registration.callback
    assert await runtime.deregister_conditional_middleware_guardrail(gate)
    deregister_request = _last_request(host_stub, host_pb.DeregisterConditionalMiddlewareGuardrailRequest)
    assert deregister_request.credentials.activation_id == ACTIVATION_ID
    assert deregister_request.credentials.auth_token == AUTH_TOKEN
    assert deregister_request.handle == "gate-1"

    stack_id = await runtime.create_scope_stack()
    assert stack_id == "stack-1"
    with runtime.bind_scope_stack(stack_id, parent_scope_id="parent-1"):
        assert runtime.current_scope_stack_id() == stack_id
        assert runtime.current_parent_scope_id() == "parent-1"
        await runtime.emit_mark("mark", {"ok": True})
        ordinary_mark = _last_request(host_stub, host_pb.EmitMarkRequest).mark
        assert not ordinary_mark.HasField("severity")
        assert not ordinary_mark.HasField("category")
        await runtime.emit_mark("override-parent", parent_scope_id="parent-2")
        scope_id = await runtime.push_scope("scope", scope_type=ScopeType.TOOL, input={"in": True})
        ordinary_push = _last_request(host_stub, host_pb.PushScopeRequest)
        assert not ordinary_push.HasField("time")
        await runtime.pop_scope(scope_id, output={"out": True})
        ordinary_pop = _last_request(host_stub, host_pb.PopScopeRequest)
        assert not ordinary_pop.HasField("time")
        historical_scope_id = await runtime.push_scope(
            "historical-scope",
            timestamp=datetime(1969, 12, 31, 23, 59, 59, 999998, tzinfo=timezone.utc),
        )
        historical_push = _last_request(host_stub, host_pb.PushScopeRequest)
        assert historical_push.HasField("time")
        assert historical_push.time.ToMicroseconds() == -2
        await runtime.pop_scope(
            historical_scope_id,
            timestamp=datetime(1970, 1, 1, tzinfo=timezone.utc),
        )
        historical_pop = _last_request(host_stub, host_pb.PopScopeRequest)
        assert historical_pop.HasField("time")
        assert historical_pop.time.ToMicroseconds() == 0
        tool_next = await ToolNext(runtime, "tool-next").call({"value": 1})
        llm_next = await _llm_next(runtime, {"content": {"prompt": "hello"}})
        stream_next = [chunk async for chunk in _llm_stream_next(runtime, {"content": {"prompt": "hello"}})]
        with runtime.clear_scope_stack():
            assert runtime.current_scope_stack_id() is None
            assert runtime.current_parent_scope_id() is None
        assert runtime.current_scope_stack_id() == stack_id
    assert runtime.current_scope_stack_id() is None
    assert tool_next.result["next_tool"]["value"] == 1

    assert tool_next.annotation == {"source": "host"}
    assert llm_next["next_llm"]["content"]["prompt"] == "hello"
    assert stream_next[0]["next_stream"]["content"]["prompt"] == "hello"

    await runtime.emit_mark(
        "telemetry-options",
        {"measurements": []},
        data_schema=DataSchema("nemo.relay.metric_measurements", "1"),
        severity=LogSeverity.WARNING,
        category=EventCategory.CUSTOM,
    )
    await runtime.emit_metric(
        "telemetry-metric",
        [
            {
                "name": "example.tokens.saved",
                "kind": "counter",
                "value_type": "u64",
                "value": 42,
            }
        ],
        category="vendor.metric",
    )
    await runtime.emit_mark("explicit", scope_stack_id="explicit-stack", parent_scope_id="explicit-parent")
    await runtime.drop_scope_stack(stack_id)
    mark_request = _last_request(host_stub, host_pb.EmitMarkRequest)
    assert mark_request.scope.scope_stack_id == "explicit-stack"
    assert mark_request.scope.parent_scope_id == "explicit-parent"
    push_request = _last_request(host_stub, host_pb.PushScopeRequest)
    assert push_request.scope.scope_stack_id == "stack-1"
    assert push_request.scope.parent_scope_id == "parent-1"

    telemetry_mark = next(
        request
        for request in host_stub.requests
        if isinstance(request, host_pb.EmitMarkRequest) and request.mark.name == "telemetry-options"
    )

    assert plugin_api._protocol.unpack(telemetry_mark.mark.data_schema) == {
        "name": "nemo.relay.metric_measurements",
        "version": "1",
    }
    assert telemetry_mark.mark.severity == common_pb.LOG_SEVERITY_WARN
    assert telemetry_mark.mark.category == "custom"
    metric_mark = next(
        request
        for request in host_stub.requests
        if isinstance(request, host_pb.EmitMarkRequest) and request.mark.name == "telemetry-metric"
    )
    assert json.loads(metric_mark.mark.data.json)["measurements"][0]["value"] == 42
    assert plugin_api._protocol.unpack(metric_mark.mark.data_schema) == {
        "name": "nemo.relay.metric_measurements",
        "version": "1",
    }
    assert metric_mark.mark.category == "vendor.metric"

    override_mark = next(
        request
        for request in host_stub.requests
        if isinstance(request, host_pb.EmitMarkRequest) and request.mark.name == "override-parent"
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
    with pytest.raises(ValueError, match="invalid log severity"):
        await runtime.emit_mark("invalid-severity", severity="fatal")
    with pytest.raises(TypeError, match="category must be a string or None"):
        await runtime.emit_mark("invalid-category", category=cast(Any, 1))
    with pytest.raises(ValueError, match="exactly name and version"):
        await runtime.emit_mark("invalid-schema", data_schema={"name": "schema"})
    with pytest.raises(TypeError, match="measurements must contain mappings"):
        await runtime.emit_metric("invalid-metric", cast(Any, [42]))


async def test_runtime_logs_every_level_with_structured_fields(host_stub: RecordingHostStub) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    await runtime.trace("trace message", target="plugin", fields={"ordinal": 0, "nested": {"ok": True}})
    await runtime.debug("debug message", target="plugin")
    await runtime.info("info message", target="plugin")
    await runtime.warn("warn message", target="plugin")
    await runtime.error("error message", target="plugin")
    await runtime.log("warning", "warning alias", target="plugin")

    requests = [request for request in host_stub.requests if isinstance(request, host_pb.LogRequest)]
    assert [request.level for request in requests] == [
        common_pb.LOG_SEVERITY_TRACE,
        common_pb.LOG_SEVERITY_DEBUG,
        common_pb.LOG_SEVERITY_INFO,
        common_pb.LOG_SEVERITY_WARN,
        common_pb.LOG_SEVERITY_ERROR,
        common_pb.LOG_SEVERITY_WARN,
    ]
    assert all(request.credentials.activation_id == ACTIVATION_ID for request in requests)
    assert all(request.credentials.auth_token == AUTH_TOKEN for request in requests)
    assert all(request.target == "plugin" for request in requests)
    assert [request.message for request in requests] == [
        "trace message",
        "debug message",
        "info message",
        "warn message",
        "error message",
        "warning alias",
    ]
    assert plugin_api._protocol.unpack(requests[0].fields) == {"ordinal": 0, "nested": {"ok": True}}
    assert not requests[1].HasField("fields")

    with pytest.raises(TypeError, match="fields must be a mapping"):
        await runtime.info("invalid fields", fields=cast(Any, ["not", "a", "mapping"]))
    with pytest.raises(ValueError, match="invalid plugin log level"):
        await runtime.log("verbose", "invalid level")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("error", "list failed"),
        ("unknown_kind", "unknown runtime registration kind"),
        ("unknown_owner", "unknown runtime registration owner kind"),
    ],
)
async def test_runtime_registration_discovery_rejects_host_and_malformed_responses(
    host_stub: RecordingHostStub, failure: str, message: str
) -> None:
    host_stub.failures["ListRuntimeRegistrations"] = failure
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)

    with pytest.raises(WorkerSdkError, match=message):
        await runtime.list_runtime_registrations()


async def test_runtime_gate_operations_propagate_host_errors(host_stub: RecordingHostStub) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    host_stub.failures["RegisterConditionalMiddlewareGuardrail"] = "error"
    with pytest.raises(WorkerSdkError, match="register failed"):
        await runtime.register_conditional_middleware_guardrail(
            "gate",
            {RuntimeRegistrationKind.SUBSCRIBER},
            "target",
            lambda _kinds, _name: "disabled",
        )

    host_stub.failures["DeregisterConditionalMiddlewareGuardrail"] = "error"
    with pytest.raises(WorkerSdkError, match="deregister failed"):
        await runtime.deregister_conditional_middleware_guardrail(
            ConditionalMiddlewareGuardrailHandle("gate-1", "gate")
        )
    with pytest.raises(TypeError, match="ConditionalMiddlewareGuardrailHandle"):
        await runtime.deregister_conditional_middleware_guardrail(
            "gate-1"  # type: ignore[arg-type]
        )


async def test_invalid_gate_handle_rolls_back_callback_and_allows_retry() -> None:
    class Host(RecordingHostStub):
        invalid = True

        async def RegisterConditionalMiddlewareGuardrail(self, request):
            if self.invalid:
                self.invalid = False
                return host_pb.RegisterConditionalMiddlewareGuardrailResponse()
            return await super().RegisterConditionalMiddlewareGuardrail(request)

    host = Host()
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host)
    arguments = ("gate", {RuntimeRegistrationKind.SUBSCRIBER}, "target", lambda _kinds, _name: None)
    with pytest.raises(WorkerSdkError, match="invalid host response: handle"):
        await runtime.register_conditional_middleware_guardrail(*arguments)
    assert runtime._conditional_middleware_callbacks == {}
    handle = await runtime.register_conditional_middleware_guardrail(*arguments)
    assert handle._value == "gate-1"
    assert await runtime.deregister_conditional_middleware_guardrail(handle)


@pytest.mark.parametrize("field", ["local_name", "effective_name", "plugin_kind"])
async def test_runtime_registration_discovery_validates_names(field) -> None:
    class Host(RecordingHostStub):
        async def ListRuntimeRegistrations(self, request):
            response = await super().ListRuntimeRegistrations(request)
            registration = response.registrations[0]
            if field == "plugin_kind":
                registration.owner.plugin.plugin_kind = ""
            else:
                setattr(registration, field, "")
            return response

    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=Host())
    with pytest.raises(WorkerSdkError, match="invalid host response"):
        await runtime.list_runtime_registrations()


@pytest.mark.parametrize("method", ["CreateScopeStack", "PushScope"])
async def test_scope_creation_rejects_empty_host_handles_and_allows_valid_retry(method) -> None:
    class Host(RecordingHostStub):
        async def CreateScopeStack(self, request):
            if method == "CreateScopeStack":
                return host_pb.CreateScopeStackResponse()
            return await super().CreateScopeStack(request)

        async def PushScope(self, request):
            if method == "PushScope":
                return host_pb.PushScopeResponse()
            return await super().PushScope(request)

    host = Host()
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host)
    with pytest.raises(WorkerSdkError, match="invalid host response"):
        await (runtime.create_scope_stack() if method == "CreateScopeStack" else runtime.push_scope("scope"))
    assert runtime.current_scope_stack_id() is None
    assert runtime.current_parent_scope_id() is None
    runtime._host_stub = RecordingHostStub()
    assert await runtime.create_scope_stack() == "stack-1"
    assert await runtime.push_scope("scope") == "scope-1"


async def test_drop_scope_stack_rejects_empty_handle_before_host_call(host_stub: RecordingHostStub) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    with pytest.raises(WorkerSdkError, match="scope_stack_id must not be empty"):
        await runtime.drop_scope_stack("")
    assert host_stub.requests == []
    await runtime.drop_scope_stack("stack-1")


async def test_structured_log_rejects_nested_nonstring_json_keys(host_stub: RecordingHostStub) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    with pytest.raises(WorkerSdkError, match="JSON object keys must be strings"):
        await runtime.info("message", fields={"nested": {1: "first", "1": "second"}})
    assert host_stub.requests == []


@pytest.mark.parametrize("surface", ["metadata", "sanitizer", "tool_outcome", "llm_outcome"])
async def test_typed_callback_outputs_reject_nested_nonstring_json_keys(surface) -> None:
    invalid = {"nested": {1: "first", "1": "second"}}

    class InvalidPlugin(WorkerPlugin):
        plugin_id = "test.invalid-json-keys"

        def register(self, ctx, config):
            ctx.register_event_metadata_injector("metadata", lambda event: invalid)
            ctx.register_mark_sanitize_guardrail("sanitizer", lambda event, fields: {"data": invalid})
            ctx.register_tool_execution_intercept(
                "tool_outcome", lambda context, next_call: ToolExecutionInterceptOutcome(result=invalid)
            )
            ctx.register_llm_request_intercept(
                "llm_outcome",
                lambda model, request, annotation: LlmRequestInterceptOutcome(
                    request={"headers": {}, "content": invalid}
                ),
            )

    service = _service(InvalidPlugin(), RecordingHostStub())
    await _register(service)
    kind = {
        "metadata": common_pb.REGISTRATION_SURFACE_EVENT_METADATA_INJECTOR,
        "sanitizer": common_pb.REGISTRATION_SURFACE_MARK_SANITIZE_GUARDRAIL,
        "tool_outcome": common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT,
        "llm_outcome": common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT,
    }[surface]
    if surface in {"metadata", "sanitizer"}:
        payload = {"event": _encode_payload(event_pb.Event, {"name": "root"})}
    elif surface == "tool_outcome":
        payload = {"tool": _ToolPayload(value=_encode_payload(common_pb.JsonValue, {}))}
    else:
        payload = {"llm": _llm_payload(request={"headers": {}, "content": {}})}
    response = await service.Invoke(
        _invoke_request(surface, kind, **payload),
        AbortContext(),
    )
    assert response.WhichOneof("result") == "error"
    assert response.error.code == "worker.sdk_error"
    assert "JSON object keys must be strings" in response.error.message


@pytest.mark.parametrize(
    "annotated,error",
    [
        ({"messages": [], "params": {"stop": "END"}}, "StringList must be an array"),
        ({"messages": [], "params": {"stop": {"first": "END"}}}, "StringList must be an array"),
        ({"messages": [], "params": [["max_tokens", 12]]}, "GenerationParams must be an object"),
        ({"messages": [], "model": b"example"}, "model must be a string"),
        ({"messages": None}, "messages must be an array"),
    ],
)
async def test_request_interceptor_rejects_invalid_native_container_and_string_shapes(annotated, error) -> None:
    class InvalidPlugin(WorkerPlugin):
        plugin_id = "test.invalid-request-shape"

        def register(self, ctx, config):
            ctx.register_llm_request_intercept(
                "rewrite",
                lambda model, request, annotation: LlmRequestInterceptOutcome(
                    request=request, annotated_request=annotated
                ),
            )

    service = _service(InvalidPlugin(), RecordingHostStub())
    await _register(service)
    response = await service.Invoke(
        _invoke_request(
            "rewrite",
            common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT,
            llm=_llm_payload(request={"headers": {}, "content": {}}),
        ),
        AbortContext(),
    )
    assert response.WhichOneof("result") == "error"
    assert error in response.error.message


async def test_invocation_scope_context_is_isolated_across_concurrent_requests(host_stub: RecordingHostStub) -> None:
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
                common_pb.REGISTRATION_SURFACE_SUBSCRIBER,
                invocation_id=f"invoke-{name}",
                continuation_id=f"next-{name}",
                event=_encode_payload(event_pb.Event, {"name": name}),
                scope=runtime_pb.ScopeContext(scope_stack_id=stack, parent_scope_id=parent),
            ),
            AbortContext(),
        )
        assert response.WhichOneof("result") == "completed"

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
        (request.mark.name, request.scope.scope_stack_id, request.scope.parent_scope_id)
        for request in host_stub.requests
        if isinstance(request, host_pb.EmitMarkRequest)
    ]
    assert len(marks) == 2
    assert sorted(marks) == [
        ("first", "stack-1", "parent-1"),
        ("second", "stack-2", "parent-2"),
    ]


async def test_runtime_host_call_error_paths(host_stub: RecordingHostStub) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)

    host_stub.failures["EmitMark"] = "error"
    with pytest.raises(WorkerSdkError, match="EmitMark failed"):
        await runtime.emit_mark("mark")

    host_stub.failures["EmitMark"] = "empty"
    with pytest.raises(WorkerSdkError, match="host call failed"):
        await runtime.emit_mark("mark")

    class UnimplementedRuntimeDiagnosticsError(Exception):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.UNIMPLEMENTED

    class OldHostStub:
        async def GetRuntimeDiagnostics(self, request: Any) -> Any:
            del request
            raise UnimplementedRuntimeDiagnosticsError()

    with mock.patch.object(grpc.aio, "AioRpcError", UnimplementedRuntimeDiagnosticsError):
        old_host_runtime = PluginRuntime(
            activation_id=ACTIVATION_ID,
            auth_token=AUTH_TOKEN,
            host_stub=OldHostStub(),
        )
        with pytest.raises(WorkerSdkError, match="worker v1 diagnostics contract"):
            await old_host_runtime.runtime_diagnostics()

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

    for failure, expected_message in (
        ("empty", "invalid host response"),
        ("missing_result", "invalid host response"),
        ("malformed_result", "invalid host response"),
        ("malformed_annotation", "invalid host response"),
    ):
        host_stub.failures["ToolNext"] = failure
        with pytest.raises(WorkerSdkError, match=expected_message):
            await ToolNext(runtime, "tool-next").call({"value": 1})

    host_stub.failures["ToolNext"] = "null_annotation"
    assert (await ToolNext(runtime, "tool-next").call({"value": 1})).annotation is None

    host_stub.failures["LlmNext"] = "error"
    with pytest.raises(WorkerSdkError, match="LlmNext failed"):
        await _llm_next(runtime, {"content": {}})

    host_stub.failures["LlmStreamNext"] = "error"
    with pytest.raises(WorkerSdkError, match="LlmStreamNext failed"):
        async for _chunk in _llm_stream_next(runtime, {"content": {}}):
            pass

    host_stub.failures["LlmStreamNext"] = "empty"
    with pytest.raises(WorkerSdkError, match="invalid host response"):
        async for _chunk in _llm_stream_next(runtime, {"content": {}}):
            pass


@pytest.mark.parametrize(
    ("timestamp", "error_type", "message"),
    [
        (datetime(2026, 1, 1), ValueError, "timezone-aware"),
        ("2026-01-01T00:00:00Z", TypeError, "datetime.datetime"),
    ],
)
async def test_runtime_scope_timestamps_reject_invalid_values_before_host_call(
    host_stub: RecordingHostStub,
    timestamp: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host_stub)
    request_count = len(host_stub.requests)

    with pytest.raises(error_type, match=message):
        await runtime.push_scope("scope", timestamp=timestamp)
    with pytest.raises(error_type, match=message):
        await runtime.pop_scope("scope", timestamp=timestamp)

    assert len(host_stub.requests) == request_count


async def test_lifecycle_acks(service: _WorkerService) -> None:
    cancel = await service.CancelInvocation(
        worker_pb.CancelInvocationRequest(
            invocation_id="invoke-1",
            reason="test",
            credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
        ),
        AbortContext(),
    )
    assert not cancel.accepted
    assert "not active" in cancel.message

    shutdown = await service.Shutdown(
        worker_pb.ShutdownRequest(
            reason="test", credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN)
        ),
        AbortContext(),
    )
    assert shutdown == worker_pb.ShutdownResponse()


async def test_cancel_invocation_stops_active_async_callback_and_is_idempotent() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    class CancelPlugin(WorkerPlugin):
        plugin_id = "tests.cancel"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def tool_execution(
                context: ToolExecutionContext, next_call: ToolNext
            ) -> ToolExecutionInterceptOutcome:
                del context, next_call
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    raise
                raise AssertionError("unreachable")

            ctx.register_tool_execution_intercept("cancel", tool_execution)

    service = _service(CancelPlugin(), RecordingHostStub())
    await _register(service)
    request = _invoke_request(
        "cancel",
        common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT,
        invocation_id="cancel-unary",
        tool=_ToolPayload(tool_name="lookup", value=_encode_payload(common_pb.JsonValue, {})),
    )
    invoke_task = asyncio.create_task(service.Invoke(request, AbortContext()))
    await asyncio.wait_for(started.wait(), timeout=1)

    first_task = asyncio.create_task(
        service.CancelInvocation(
            worker_pb.CancelInvocationRequest(
                invocation_id="cancel-unary",
                reason="test timeout",
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            ),
            AbortContext(),
        )
    )
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    second_task = asyncio.create_task(
        service.CancelInvocation(
            worker_pb.CancelInvocationRequest(
                invocation_id="cancel-unary",
                reason="repeat",
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            ),
            AbortContext(),
        )
    )
    await asyncio.sleep(0)
    assert not first_task.done()
    assert not second_task.done()
    reused = await asyncio.wait_for(service.Invoke(request, AbortContext()), timeout=1)
    release.set()
    first, second = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=1)
    response = await asyncio.wait_for(invoke_task, timeout=1)

    assert first.accepted
    assert not second.accepted
    assert reused.error.code == "worker.sdk_error"
    assert "already active" in reused.error.message
    assert response.error.code == "worker.cancelled"
    assert "test timeout" in response.error.message
    assert cancelled.is_set()


async def test_cancel_invocation_stops_active_async_stream() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()

    class CancelStreamPlugin(WorkerPlugin):
        plugin_id = "tests.cancel_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def llm_stream(model_name: str, request: Json, context: Any, next_call: Any) -> AsyncIterator[Json]:
                del model_name, request, context, next_call
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await finish_cleanup.wait()
                    cancelled.set()
                if False:
                    yield {}

            ctx.register_llm_stream_execution_intercept("cancel_stream", llm_stream)

    service = _service(CancelStreamPlugin(), RecordingHostStub())
    await _register(service)
    request = _invoke_request(
        "cancel_stream",
        common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
        invocation_id="cancel-stream",
        llm=_llm_payload(),
    )

    async def consume() -> list[Any]:
        return [chunk async for chunk in service.InvokeStream(request, AbortContext())]

    consume_task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    cancel_task = asyncio.create_task(
        service.CancelInvocation(
            worker_pb.CancelInvocationRequest(
                invocation_id="cancel-stream",
                reason="stream abandoned",
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
            ),
            AbortContext(),
        )
    )
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    assert not cancel_task.done()
    finish_cleanup.set()
    ack = await asyncio.wait_for(cancel_task, timeout=1)
    assert cancelled.is_set()
    chunks = await asyncio.wait_for(consume_task, timeout=1)

    assert ack.accepted
    assert len(chunks) == 1
    assert chunks[0].error.code == "worker.cancelled"
    assert "stream abandoned" in chunks[0].error.message
    assert cancelled.is_set()


async def test_cancel_invocation_discards_buffered_chunks_after_stream_callback_finishes() -> None:
    finished = asyncio.Event()

    class BufferedStreamPlugin(WorkerPlugin):
        plugin_id = "tests.buffered_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def llm_stream(model_name: str, request: Json, context: Any, next_call: Any) -> AsyncIterator[Json]:
                del model_name, request, context, next_call
                for index in range(4):
                    yield {"index": index}
                finished.set()

            ctx.register_llm_stream_execution_intercept("buffered_stream", llm_stream)

    service = _service(BufferedStreamPlugin(), RecordingHostStub())
    await _register(service)
    request = _invoke_request(
        "buffered_stream",
        common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
        invocation_id="buffered-stream",
        llm=_llm_payload(),
    )
    stream = service.InvokeStream(request, AbortContext())

    first = await asyncio.wait_for(anext(stream), timeout=1)
    await asyncio.wait_for(finished.wait(), timeout=1)
    assert not first.error.code
    assert "buffered-stream" in service._active_invocations

    ack = await service.CancelInvocation(
        worker_pb.CancelInvocationRequest(
            invocation_id="buffered-stream",
            reason="stop buffered stream",
            credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
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


async def test_stream_callback_cancellation_without_host_reason_is_terminal_error() -> None:
    class CancelledStreamPlugin(WorkerPlugin):
        plugin_id = "tests.cancelled_stream"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def llm_stream(model_name: str, request: Json, context: Any, next_call: Any) -> AsyncIterator[Json]:
                del model_name, request, context, next_call
                if False:
                    yield {}
                raise asyncio.CancelledError

            ctx.register_llm_stream_execution_intercept("cancelled_stream", llm_stream)

    service = _service(CancelledStreamPlugin(), RecordingHostStub())
    await _register(service)
    request = _invoke_request(
        "cancelled_stream",
        common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
        invocation_id="self-cancelled-stream",
        llm=_llm_payload(),
    )

    async def consume() -> list[Any]:
        return [chunk async for chunk in service.InvokeStream(request, AbortContext())]

    chunks = await asyncio.wait_for(consume(), timeout=1)

    assert len(chunks) == 1
    assert chunks[0].error.code == "worker.cancelled"
    assert "stream callback cancelled without a host request" in chunks[0].error.message


def test_required_environment_reports_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_RELAY_WORKER_SOCKET", raising=False)
    with pytest.raises(WorkerSdkError, match="NEMO_RELAY_WORKER_SOCKET"):
        _required_env("NEMO_RELAY_WORKER_SOCKET")


def test_unix_host_channel_uses_valid_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain Unix authority while configuring send/receive limits for both transports."""
    calls: list[tuple[str, tuple[tuple[str, str | int], ...]]] = []

    def insecure_channel(
        target: str,
        *,
        options: tuple[tuple[str, str | int], ...] = (),
    ) -> object:
        """Capture channel options without opening a network connection."""
        calls.append((target, options))
        return object()

    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", insecure_channel)
    monkeypatch.setenv("NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES", "8388608")

    _open_host_channel("unix:///tmp/relay-host.sock")
    _open_host_channel("http://127.0.0.1:50051")

    assert calls == [
        (
            "unix:/tmp/relay-host.sock",
            (
                ("grpc.max_receive_message_length", 8388608),
                ("grpc.max_send_message_length", 8388608),
                ("grpc.default_authority", "localhost"),
            ),
        ),
        ("127.0.0.1:50051", (("grpc.max_receive_message_length", 8388608), ("grpc.max_send_message_length", 8388608))),
    ]


async def test_endpoint_helpers_normalize_and_refuse_non_socket_unix_targets(tmp_path: Any) -> None:
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
def test_grpc_target_rejects_non_loopback_and_malformed_endpoints(endpoint: str) -> None:
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
) -> None:
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


async def test_serve_plugin_validates_plugin_id_before_resources(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_endpoint_file_is_published_atomically(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
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
) -> None:
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
async def test_unlink_unix_socket_removes_an_existing_socket() -> None:
    with tempfile.TemporaryDirectory(prefix="nr-plugin-") as directory:
        socket_path = Path(directory) / "worker.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix_socket:
            unix_socket.bind(str(socket_path))

        await _unlink_unix_socket(f"unix://{socket_path}")
        assert not socket_path.exists()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
async def test_unlink_unix_socket_refuses_an_active_socket() -> None:
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
) -> None:
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
) -> None:
    """Do not publish the bound endpoint until the gRPC server accepts requests."""
    endpoint_file = tmp_path / "endpoint.txt"
    start_entered = asyncio.Event()
    allow_start = asyncio.Event()

    class FakeChannel:
        async def close(self) -> None:
            """Allow startup-cancellation cleanup without opening a real channel."""
            pass

    class FakeServer:
        def add_generic_rpc_handlers(self, *args):
            """Accept standard health service registration in this lifecycle double."""

        def add_registered_method_handlers(self, *args):
            """Accept optimized standard health method registration."""

        def add_insecure_port(self, target: str) -> int:
            """Return a resolved loopback port without publishing it yet."""
            assert target == "127.0.0.1:0"
            return 43123

        async def start(self) -> None:
            """Pause startup so the test can inspect endpoint publication ordering."""
            start_entered.set()
            await allow_start.wait()

        async def stop(self, grace: int) -> None:
            """Check the configured grace period during cancellation cleanup."""
            assert grace == 2

    fake_server = FakeServer()
    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", lambda target, **kwargs: FakeChannel())
    monkeypatch.setattr(plugin_api.grpc.aio, "server", lambda **kwargs: fake_server)
    monkeypatch.setattr(plugin_api.host_grpc, "RelayHostRuntimeServiceStub", lambda channel: object())
    monkeypatch.setattr(plugin_api.diagnostics_grpc, "RelayDiagnosticsServiceStub", lambda channel: object())
    monkeypatch.setattr(
        plugin_api.worker_grpc, "add_PluginWorkerServiceServicer_to_server", lambda service, server: None
    )
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


@pytest.mark.parametrize("failure_stage", ["registration", "start"])
async def test_serve_plugin_closes_resources_when_startup_fails(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Preserve startup errors and close channels even before health registration."""

    class FakeChannel:
        closed = False

        async def close(self) -> None:
            """Record that startup failure closed the host channel."""
            self.closed = True

    class FakeServer:
        def add_generic_rpc_handlers(self, *args):
            """Accept standard health service registration in this lifecycle double."""

        def add_registered_method_handlers(self, *args):
            """Accept optimized standard health method registration."""

        stopped = False

        def add_insecure_port(self, target: str) -> int:
            """Bind successfully so this test isolates the later startup failure."""
            assert target == "127.0.0.1:0"
            return 43123

        async def start(self) -> None:
            """Simulate a startup failure after resource allocation."""
            raise RuntimeError(f"{failure_stage} failed")

        async def stop(self, grace: int) -> None:
            """Record server cleanup while checking its grace period."""
            assert grace == 2
            self.stopped = True

    channels = [FakeChannel(), FakeChannel()]
    pending_channels = iter(channels)
    fake_server = FakeServer()

    def register_worker(service: object, server: object) -> None:
        """Fail before health setup when worker registration is the startup error."""
        if failure_stage == "registration":
            raise RuntimeError("registration failed")

    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", lambda target, **kwargs: next(pending_channels))
    monkeypatch.setattr(plugin_api.grpc.aio, "server", lambda **kwargs: fake_server)
    monkeypatch.setattr(plugin_api.host_grpc, "RelayHostRuntimeServiceStub", lambda channel: object())
    monkeypatch.setattr(plugin_api.diagnostics_grpc, "RelayDiagnosticsServiceStub", lambda channel: object())
    monkeypatch.setattr(plugin_api.worker_grpc, "add_PluginWorkerServiceServicer_to_server", register_worker)
    monkeypatch.setenv("NEMO_RELAY_WORKER_SOCKET", "tcp://127.0.0.1:0")
    monkeypatch.setenv("NEMO_RELAY_HOST_SOCKET", "http://127.0.0.1:9")
    monkeypatch.setenv("NEMO_RELAY_WORKER_ID", ACTIVATION_ID)
    monkeypatch.setenv("NEMO_RELAY_WORKER_TOKEN", AUTH_TOKEN)
    monkeypatch.delenv("NEMO_RELAY_WORKER_ENDPOINT_FILE", raising=False)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        await serve_plugin(AllSurfacesPlugin())
    assert fake_server.stopped
    assert all(channel.closed for channel in channels)


async def test_serve_plugin_closes_resources_when_endpoint_bind_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Release both channels and the server when endpoint binding returns no port."""

    class FakeChannel:
        closed = False

        async def close(self) -> None:
            """Record that bind failure closed the allocated host channel."""
            self.closed = True

    class FakeServer:
        def add_generic_rpc_handlers(self, *args):
            """Accept standard health service registration in this lifecycle double."""

        def add_registered_method_handlers(self, *args):
            """Accept optimized standard health method registration."""

        stopped = False

        def add_insecure_port(self, target: str) -> int:
            """Simulate gRPC reporting that no loopback port could be bound."""
            assert target == "127.0.0.1:0"
            return 0

        async def stop(self, grace: int) -> None:
            """Record cleanup even though the server never started listening."""
            assert grace == 2
            self.stopped = True

    channels = [FakeChannel(), FakeChannel()]
    pending_channels = iter(channels)
    fake_server = FakeServer()
    monkeypatch.setattr(plugin_api.grpc.aio, "insecure_channel", lambda target, **kwargs: next(pending_channels))
    monkeypatch.setattr(plugin_api.grpc.aio, "server", lambda **kwargs: fake_server)
    monkeypatch.setattr(plugin_api.host_grpc, "RelayHostRuntimeServiceStub", lambda channel: object())
    monkeypatch.setattr(plugin_api.diagnostics_grpc, "RelayDiagnosticsServiceStub", lambda channel: object())
    monkeypatch.setattr(
        plugin_api.worker_grpc, "add_PluginWorkerServiceServicer_to_server", lambda service, server: None
    )
    monkeypatch.setenv("NEMO_RELAY_WORKER_SOCKET", "tcp://127.0.0.1:0")
    monkeypatch.setenv("NEMO_RELAY_HOST_SOCKET", "http://127.0.0.1:9")
    monkeypatch.setenv("NEMO_RELAY_WORKER_ID", ACTIVATION_ID)
    monkeypatch.setenv("NEMO_RELAY_WORKER_TOKEN", AUTH_TOKEN)
    monkeypatch.delenv("NEMO_RELAY_WORKER_ENDPOINT_FILE", raising=False)

    with pytest.raises(WorkerSdkError, match="failed to bind worker endpoint"):
        await serve_plugin(AllSurfacesPlugin())
    assert fake_server.stopped
    assert all(channel.closed for channel in channels)


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
        stub = worker_grpc.PluginWorkerServiceStub(channel)
        health = await stub.Health(
            worker_pb.HealthRequest(
                credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN)
            ),
            timeout=5,
        )
        assert health.status == runtime_pb.WORKER_HEALTH_STATUS_READY
        shutdown = await stub.Shutdown(
            worker_pb.ShutdownRequest(
                reason="test", credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN)
            ),
            timeout=5,
        )
        assert shutdown == worker_pb.ShutdownResponse()
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
    return common_pb.WorkerError(code="test.error", message=message, retryable=False)


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
    return worker_pb.HandshakeRequest(
        plugin_id="tests.python_worker",
        relay_version="0.8.0",
        protocol=common_pb.ProtocolIdentity(name=WORKER_PROTOCOL, version="1.0.0"),
        host_endpoint="http://127.0.0.1:9",
        credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
    )


def _validate_request(
    config: Json | None = None,
    *,
    plugin_id: str = "tests.python_worker",
) -> Any:
    return worker_pb.ValidateRequest(
        plugin_id=plugin_id,
        config=_encode_payload(common_pb.JsonValue, {} if config is None else config),
        credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
    )


def _register_request(
    config: Json | None = None,
    *,
    plugin_id: str = "tests.python_worker",
) -> Any:
    return worker_pb.RegisterRequest(
        plugin_id=plugin_id,
        config=_encode_payload(common_pb.JsonValue, {} if config is None else config),
        credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
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
    surface_name = common_pb.RegistrationSurface.Name(surface).removeprefix("REGISTRATION_SURFACE_").lower()
    fields: dict[str, Any] = dict(
        invocation_id=invocation_id,
        registration_name=registration_name,
        credentials=common_pb.Credentials(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN),
    )
    if "scope" in kwargs:
        fields["scope"] = kwargs["scope"]
    if "event" in kwargs:
        fields[surface_name] = kwargs["event"]
        if surface == common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_END_GUARDRAIL:
            fields[surface_name].scope.scope_category = event_pb.SCOPE_CATEGORY_END
    elif "tool" in kwargs:
        tool = kwargs["tool"]
        if surface == common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT:
            fields[surface_name] = runtime_pb.ToolExecutionInvocation(
                tool_name=tool.tool_name,
                args=tool.value,
                tool_call_id=tool.tool_call_id,
                continuation_id=continuation_id,
            )
        else:
            fields[surface_name] = runtime_pb.ToolValueInvocation(tool_name=tool.tool_name, value=tool.value)
    elif "conditional_middleware" in kwargs:
        fields[surface_name] = kwargs["conditional_middleware"]
    elif "llm" in kwargs:
        llm = kwargs["llm"]
        context = llm.execution_codec_context or _ExecutionContext(
            request=runtime_pb.LlmCodecContext(codec=_wire_codec()),
            response=runtime_pb.LlmCodecContext(codec=_wire_codec()),
        )
        if surface == common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT:
            return worker_pb.InvokeStreamRequest(
                **fields,
                request=llm.request,
                model_name=llm.model_name,
                continuation_id=continuation_id,
                request_codec=context.request,
            )
        types = {
            common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL: (
                runtime_pb.LlmSanitizeRequestInvocation,
                dict(
                    request=llm.request,
                    context=llm.request_sanitize_context or runtime_pb.LlmCodecContext(codec=_wire_codec()),
                ),
            ),
            common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_RESPONSE_GUARDRAIL: (
                runtime_pb.LlmSanitizeResponseInvocation,
                dict(
                    response=llm.response,
                    context=llm.response_sanitize_context or runtime_pb.LlmCodecContext(codec=_wire_codec()),
                ),
            ),
            common_pb.REGISTRATION_SURFACE_LLM_CONDITIONAL_EXECUTION_GUARDRAIL: (
                runtime_pb.LlmConditionalInvocation,
                dict(request=llm.request),
            ),
            common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT: (
                runtime_pb.LlmRequestInvocation,
                dict(request=llm.request, model_name=llm.model_name, annotated_request=llm.annotated_request),
            ),
            common_pb.REGISTRATION_SURFACE_LLM_EXECUTION_INTERCEPT: (
                runtime_pb.LlmExecutionInvocation,
                dict(
                    request=llm.request,
                    model_name=llm.model_name,
                    continuation_id=continuation_id,
                    request_codec=context.request,
                    response_codec=context.response,
                ),
            ),
        }
        message_type, body = types[surface]
        fields[surface_name] = message_type(**body)
    return worker_pb.InvokeRequest(**fields)


def _tool_request(registration_name: str, surface: int, value: Json) -> Any:
    return _invoke_request(
        registration_name,
        surface,
        tool=_ToolPayload(tool_name="lookup", value=_encode_payload(common_pb.JsonValue, value)),
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
        "request": _encode_payload(llm_pb.LlmRequest, {"headers": {}, **(request or {"content": {}})}),
        "response": _encode_payload(common_pb.JsonValue, response or {}),
    }
    if annotated is not None:
        kwargs["annotated_request"] = _encode_payload(llm_pb.AnnotatedLlmRequest, annotated)
    return _LlmPayload(**kwargs)


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
    selected = response.WhichOneof("result")
    assert selected in {"tool_value", "llm_request", "llm_response", "metadata_additions", "event_sanitize"}, response
    return _envelope_value(getattr(response, selected))


async def _invoke_tool_execution_async(
    service: _WorkerService,
    registration_name: str,
) -> Json:
    response = await service.Invoke(
        _tool_request(registration_name, common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT, {"query": "relay"}),
        AbortContext(),
    )
    assert response.WhichOneof("result") == "tool_execution_outcome", response
    return plugin_api._protocol.unpack(response.tool_execution_outcome)


def _envelope_value(envelope: Any) -> Json:
    return plugin_api._protocol.unpack(envelope)


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
        common_pb.REGISTRATION_SURFACE_SUBSCRIBER,
        common_pb.REGISTRATION_SURFACE_EVENT_METADATA_INJECTOR,
        common_pb.REGISTRATION_SURFACE_MARK_SANITIZE_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_START_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_SCOPE_SANITIZE_END_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_TOOL_SANITIZE_REQUEST_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_TOOL_SANITIZE_RESPONSE_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_TOOL_CONDITIONAL_EXECUTION_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_TOOL_REQUEST_INTERCEPT,
        common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT,
        common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_REQUEST_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_LLM_SANITIZE_RESPONSE_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_LLM_CONDITIONAL_EXECUTION_GUARDRAIL,
        common_pb.REGISTRATION_SURFACE_LLM_REQUEST_INTERCEPT,
        common_pb.REGISTRATION_SURFACE_LLM_EXECUTION_INTERCEPT,
        common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
        common_pb.REGISTRATION_SURFACE_CONDITIONAL_MIDDLEWARE_GUARDRAIL,
    ]


async def test_tool_execution_intercept_receives_tool_call_id() -> None:
    seen: dict[str, Json] = {}

    class ContextPlugin(WorkerPlugin):
        plugin_id = "tests.context"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def tool_execution(context: ToolExecutionContext, next_call: ToolNext):
                seen["tool_name"] = context.tool_name
                seen["arguments"] = context.args
                seen["tool_call_id"] = context.tool_call_id
                downstream = await next_call.call(context.args)
                return ToolExecutionInterceptOutcome(result=downstream.result)

            ctx.register_tool_execution_intercept("context", tool_execution)

    service = _service(ContextPlugin(), RecordingHostStub())
    await _register(service)
    response = await service.Invoke(
        _invoke_request(
            "context",
            common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT,
            continuation_id="continuation-1",
            tool=_ToolPayload(
                tool_name="lookup",
                value=_encode_payload(common_pb.JsonValue, {"query": "relay"}),
                tool_call_id="worker-call-9",
            ),
        ),
        AbortContext(),
    )

    assert response.WhichOneof("result") == "tool_execution_outcome", response
    assert seen["tool_name"] == "lookup"
    assert seen["arguments"] == {"query": "relay"}
    assert seen["tool_call_id"] == "worker-call-9"


async def test_tool_execution_intercept_tool_call_id_is_none_when_absent() -> None:
    seen: dict[str, Json] = {}

    class ContextPlugin(WorkerPlugin):
        plugin_id = "tests.context_none"

        def register(self, ctx: PluginContext, config: Json) -> None:
            del config

            async def tool_execution(context: ToolExecutionContext, next_call: ToolNext):
                seen["tool_call_id"] = context.tool_call_id
                downstream = await next_call.call(context.args)
                return ToolExecutionInterceptOutcome(result=downstream.result)

            ctx.register_tool_execution_intercept("context_none", tool_execution)

    service = _service(ContextPlugin(), RecordingHostStub())
    await _register(service)
    await service.Invoke(
        _tool_request("context_none", common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT, {"query": "relay"}),
        AbortContext(),
    )

    assert seen["tool_call_id"] is None


@pytest.mark.parametrize("status", range(400, 600))
def test_http_error_prefix_survives_worker_and_next_round_trip(status: int) -> None:
    from nemo_relay_plugin import http_error_message
    from nemo_relay_plugin._api import _sdk_error_to_worker, _worker_error_to_sdk

    message = http_error_message(status, "provider failure")
    wire = _sdk_error_to_worker(WorkerSdkError(message))
    assert wire.message == message
    assert not wire.retryable
    assert _sdk_error_to_worker(_worker_error_to_sdk(wire)).message == message


@pytest.mark.parametrize("message", ["plain failure", "http 403: denied", "HTTP 200: success", "HTTP ４０３: denied"])
def test_unrecognized_http_error_prefix_keeps_worker_error_code(message: str) -> None:
    from nemo_relay_plugin._api import _sdk_error_to_worker, _worker_error_to_sdk

    wire = _sdk_error_to_worker(WorkerSdkError(message))
    assert str(_worker_error_to_sdk(wire)) == f"{wire.code}: {message}"


@pytest.mark.parametrize("status", [0, 200, 399, 600, 403.0, True])
def test_http_error_message_rejects_invalid_status(status: Any) -> None:
    from nemo_relay_plugin import http_error_message

    with pytest.raises(ValueError, match="between 400 and 599"):
        http_error_message(status, "invalid")


def test_worker_cancellation_keeps_code_even_with_http_prefix() -> None:
    from nemo_relay_plugin._api import _sdk_error_to_worker, _worker_error_to_sdk

    wire = _sdk_error_to_worker(WorkerSdkError("HTTP 403: denied"))
    wire.code = "worker.cancelled"
    assert str(_worker_error_to_sdk(wire)) == "worker.cancelled: HTTP 403: denied"


async def test_scope_stack_snapshot_converts_types_and_reads_each_binding_fresh() -> None:
    identifiers = [str(UUID(int=number)) for number in range(1, 12)]
    snapshot = runtime_pb.ScopeStackSnapshot(
        scopes=[
            runtime_pb.ScopeStackEntry(scope_id=identifier, scope_type=category)
            for identifier, category in zip(identifiers, list(range(1, 12)), strict=True)
        ]
    )
    snapshot_response = host_pb.GetCurrentScopeStackSnapshotResponse(snapshot=snapshot)
    host = mock.Mock()
    host.GetCurrentScopeStackSnapshot = mock.AsyncMock(
        side_effect=[
            host_pb.GetCurrentScopeStackSnapshotResponse(no_context={}),
            snapshot_response,
            host_pb.GetCurrentScopeStackSnapshotResponse(
                snapshot=runtime_pb.ScopeStackSnapshot(scopes=snapshot.scopes[:1])
            ),
            host_pb.GetCurrentScopeStackSnapshotResponse(no_context={}),
        ]
    )
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host)
    assert await runtime.current_scope_stack_snapshot() is None
    with runtime.bind_scope_stack("first", parent_scope_id=identifiers[1]):
        retained = await runtime.current_scope_stack_snapshot()
        assert retained is not None
        assert [entry.scope_id for entry in retained.scopes] == identifiers
        assert [entry.scope_type for entry in retained.scopes] == list(ScopeType)
        retained.scopes.pop()
        with runtime.bind_scope_stack("second", parent_scope_id=identifiers[0]):
            current = await runtime.current_scope_stack_snapshot()
            assert current is not None
            assert len(current.scopes) == 1
            current.scopes.clear()
        assert [entry.scope_id for entry in retained.scopes] == identifiers[:-1]
        assert [entry.scope_id for entry in snapshot_response.snapshot.scopes] == identifiers
        assert [entry.scope_id for entry in snapshot.scopes] == identifiers
    assert await runtime.current_scope_stack_snapshot() is None
    requests = [call.args[0] for call in host.GetCurrentScopeStackSnapshot.await_args_list]
    assert [request.scope.scope_stack_id for request in requests] == ["", "first", "second", ""]
    assert [request.scope.parent_scope_id for request in requests] == ["", identifiers[1], identifiers[0], ""]
    assert all(
        request.credentials.activation_id == ACTIVATION_ID and request.credentials.auth_token == AUTH_TOKEN
        for request in requests
    )


async def test_scope_stack_snapshot_distinguishes_host_error_and_unsupported_rpc() -> None:
    host = mock.Mock()
    host.GetCurrentScopeStackSnapshot = mock.AsyncMock(
        side_effect=grpc.aio.AioRpcError(
            grpc.StatusCode.FAILED_PRECONDITION, initial_metadata=(), trailing_metadata=(), details="expired binding"
        )
    )
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host)
    with pytest.raises(WorkerSdkError, match="expired binding"):
        await runtime.current_scope_stack_snapshot()
    for status, message in [
        (grpc.StatusCode.UNIMPLEMENTED, "worker v1 snapshot contract"),
        (grpc.StatusCode.PERMISSION_DENIED, "GetCurrentScopeStackSnapshot failed: denied"),
    ]:
        host.GetCurrentScopeStackSnapshot.side_effect = grpc.aio.AioRpcError(
            status, initial_metadata=(), trailing_metadata=(), details="denied"
        )
        with pytest.raises(WorkerSdkError, match=message):
            await runtime.current_scope_stack_snapshot()


async def test_scope_stack_snapshot_matches_shared_fixture() -> None:
    fixture_path = Path(__file__).resolve().parents[3] / "crates/types/tests/fixtures/scope_stack_snapshot.json"
    entries = json.loads(fixture_path.read_text())["scopes"]
    host = mock.Mock()
    host.GetCurrentScopeStackSnapshot = mock.AsyncMock(
        return_value=host_pb.GetCurrentScopeStackSnapshotResponse(
            snapshot=runtime_pb.ScopeStackSnapshot(
                scopes=[
                    runtime_pb.ScopeStackEntry(
                        scope_id=entry["scope_id"], scope_type=plugin_api._proto_scope_type(entry["scope_type"])
                    )
                    for entry in entries
                ]
            )
        )
    )
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host)
    snapshot = await runtime.current_scope_stack_snapshot()
    assert snapshot is not None
    assert [{"scope_id": entry.scope_id, "scope_type": entry.scope_type.value} for entry in snapshot.scopes] == entries


@pytest.mark.parametrize("message", ["provider failed", "HTTP 503: provider unavailable"])
def test_downstream_error_provenance_preserves_legacy_wire_semantics(message: str) -> None:
    legacy = plugin_api._sdk_error_to_worker(
        plugin_api._worker_error_to_sdk(common_pb.WorkerError(code="host.runtime_error", message=message))
    )
    wire = common_pb.WorkerError(code="host.runtime_error", message=message, downstream_continuation_id="next-owned")
    wire = common_pb.WorkerError.FromString(wire.SerializeToString())
    first = plugin_api._worker_error_to_sdk(wire)
    second = plugin_api._worker_error_to_sdk(
        common_pb.WorkerError(
            code="host.runtime_error", message="another call", downstream_continuation_id="next-other"
        )
    )
    # A caught failure followed by success or another failure does not overwrite origin.
    assert (
        plugin_api._json_result_to_value(host_pb.LlmNextResponse(value=_encode_payload(common_pb.JsonValue, None)))
        is None
    )
    for error, expected in [(second, "next-other"), (first, "next-owned")]:
        assert isinstance(error, WorkerSdkError)
        forwarded = plugin_api._sdk_error_to_worker(error)
        assert forwarded.downstream_continuation_id == expected
    forwarded = plugin_api._sdk_error_to_worker(first)
    forwarded.ClearField("downstream_continuation_id")
    assert forwarded == legacy
    assert not plugin_api._sdk_error_to_worker(WorkerSdkError(str(first))).HasField("downstream_continuation_id")


@pytest.mark.parametrize("surface", ["tool", "llm", "stream"])
async def test_continuations_validate_host_errors_before_decoding(surface: str) -> None:
    error = common_pb.WorkerError(code="host.failure", message="invalid status", http_status=399)
    with pytest.raises(WorkerSdkError, match="invalid host response"):
        if surface == "tool":
            host = mock.Mock()
            host.ToolNext = mock.AsyncMock(return_value=host_pb.ToolNextResponse(error=error))
            runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host)
            await ToolNext(runtime, "next").call({})
        elif surface == "llm":
            plugin_api._json_result_to_value(host_pb.LlmNextResponse(error=error))
        else:
            plugin_api._stream_chunk_to_value(host_pb.LlmStreamNextResponse(error=error))


class ProvenanceHost(RecordingHostStub):
    def __init__(self, surface: str) -> None:
        super().__init__()
        self.surface = surface

    def failure(self, request: Any) -> Any:
        return common_pb.WorkerError(
            code="host.runtime_error", message="provider failed", downstream_continuation_id=request.continuation_id
        )

    async def ToolNext(self, request: Any) -> Any:
        return host_pb.ToolNextResponse(error=self.failure(request))

    async def LlmNext(self, request: Any) -> Any:
        return host_pb.LlmNextResponse(error=self.failure(request))

    async def LlmStreamNext(self, request: Any) -> AsyncIterator[Any]:
        if self.surface == "stream_setup":
            raise grpc.aio.AioRpcError(
                grpc.StatusCode.INTERNAL,
                grpc.aio.Metadata(),
                grpc.aio.Metadata((plugin_api._DOWNSTREAM_CONTINUATION_METADATA, request.continuation_id)),
                "provider failed",
            )
        yield host_pb.LlmStreamNextResponse(error=self.failure(request))


class ProvenancePlugin(WorkerPlugin):
    def __init__(self, replace_error: bool) -> None:
        self.replace_error = replace_error

    plugin_id = "provenance"

    def register(self, ctx: PluginContext, config: Json) -> None:
        async def tool(context: Any, next_call: Any) -> Any:
            try:
                return await next_call.call(context.args)
            except WorkerSdkError:
                if self.replace_error:
                    raise WorkerSdkError("independent plugin failure") from None
                raise

        async def llm(name: str, request: Any, context: Any, next_call: Any) -> Any:
            try:
                return await next_call.call(request)
            except WorkerSdkError:
                if self.replace_error:
                    raise WorkerSdkError("independent plugin failure") from None
                raise

        async def stream(name: str, request: Any, context: Any, next_call: Any) -> Any:
            try:
                async for value in next_call.call(request):
                    yield value
            except (WorkerSdkError, grpc.RpcError):
                if self.replace_error:
                    raise WorkerSdkError("independent plugin failure") from None
                raise

        ctx.register_tool_execution_intercept("tool", tool)
        ctx.register_llm_execution_intercept("llm", llm)
        ctx.register_llm_stream_execution_intercept("stream", stream)


@pytest.mark.parametrize("surface", ["tool", "llm", "stream", "stream_setup"])
@pytest.mark.parametrize("replace_error", [False, True])
async def test_continuation_provenance_survives_worker_callbacks(surface: str, replace_error: bool) -> None:
    service = _service(ProvenancePlugin(replace_error), ProvenanceHost(surface))
    await _register(service)

    async def invoke(index: int) -> Any:
        if surface == "tool":
            request = _tool_request("tool", common_pb.REGISTRATION_SURFACE_TOOL_EXECUTION_INTERCEPT, {})
        else:
            request = _invoke_request(
                "llm" if surface == "llm" else "stream",
                common_pb.REGISTRATION_SURFACE_LLM_EXECUTION_INTERCEPT
                if surface == "llm"
                else common_pb.REGISTRATION_SURFACE_LLM_STREAM_EXECUTION_INTERCEPT,
                llm=_llm_payload(),
            )
        request.invocation_id = f"invoke-{index}"
        if surface.startswith("stream"):
            request.continuation_id = f"next-{index}"
        else:
            getattr(request, request.WhichOneof("invocation")).continuation_id = f"next-{index}"
        if surface.startswith("stream"):
            chunks = [chunk async for chunk in service.InvokeStream(request, AbortContext())]
            return chunks[0].error
        return (await service.Invoke(request, AbortContext())).error

    # Concurrent invocations retain independent provenance, without a shared last-error flag.
    errors = await asyncio.gather(invoke(1), invoke(2))
    for index, error in enumerate(errors, 1):
        if replace_error:
            assert not error.HasField("downstream_continuation_id")
            assert error.message == "independent plugin failure"
        else:
            assert error.downstream_continuation_id == f"next-{index}"
            assert "provider failed" in error.message


@pytest.mark.parametrize(
    "response",
    [
        host_pb.GetCurrentScopeStackSnapshotResponse(),
        host_pb.GetCurrentScopeStackSnapshotResponse(
            snapshot=runtime_pb.ScopeStackSnapshot(
                scopes=[
                    runtime_pb.ScopeStackEntry(scope_id=str(UUID(int=1)), scope_type=999),
                ]
            )
        ),
    ],
)
async def test_scope_stack_snapshot_rejects_malformed_host_response(response) -> None:
    host = mock.Mock()
    host.GetCurrentScopeStackSnapshot = mock.AsyncMock(return_value=response)
    runtime = PluginRuntime(activation_id=ACTIVATION_ID, auth_token=AUTH_TOKEN, host_stub=host)
    with pytest.raises(WorkerSdkError, match="invalid host response"):
        await runtime.current_scope_stack_snapshot()


def test_protocol_repeated_enum_round_trip() -> None:
    body = {"kinds": ["tool_execution_intercept", "llm_execution_intercept"]}
    message = plugin_api._protocol.pack(runtime_pb.ConditionalMiddlewareInvocation, body)
    assert plugin_api._protocol.unpack(message)["kinds"] == body["kinds"]
    with pytest.raises(ValueError, match="unknown value"):
        plugin_api._protocol.pack(runtime_pb.ConditionalMiddlewareInvocation, {"kinds": ["missing"]})
    with pytest.raises(ValueError, match="elements must be strings"):
        plugin_api._protocol.pack(runtime_pb.ConditionalMiddlewareInvocation, {"kinds": [1]})
    message.kinds.append(999)
    with pytest.raises(ValueError, match="unknown RegistrationSurface"):
        plugin_api._protocol.unpack(message)
