# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Relay Python type bindings."""

import http.server
import json
import threading
from typing import TypedDict, cast
from uuid import uuid4

import pytest

from nemo_relay import (
    AtifExporter,
    AtofExporter,
    AtofExporterConfig,
    AtofExporterMode,
    JsonObject,
    LLMAttributes,
    LLMRequest,
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
    OpenTelemetryRuntimeDiagnostics,
    OpenTelemetrySubscriber,
    ScopeAttributes,
    ScopeEvent,
    ScopeType,
    ToolAttributes,
    ToolExecutionResult,
    llm,
    scope,
    subscribers,
    tools,
)


class _OtelCollectorHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        server = cast("_OtelCollectorServer", self.server)
        server.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": body,
            }
        )
        server.request_event.set()
        self.send_response(server.response_status)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        return


class _CollectorRequest(TypedDict):
    path: str
    headers: dict[str, str]
    body: bytes


class _OtelCollector:
    server: "_OtelCollectorServer"

    def __init__(self, response_status: int = 200) -> None:
        self.response_status = response_status

    def __enter__(self) -> "_OtelCollector":
        self.server = _OtelCollectorServer(("127.0.0.1", 0), _OtelCollectorHandler)
        self.server.requests = []
        self.server.request_event = threading.Event()
        self.server.response_status = self.response_status
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1/traces"

    def wait_for_request(self, timeout: float = 5.0) -> _CollectorRequest:
        assert self.server.request_event.wait(timeout), "timed out waiting for OTLP request"
        return self.server.requests[0]


class _OtelCollectorServer(http.server.ThreadingHTTPServer):
    requests: list[_CollectorRequest]
    request_event: threading.Event
    response_status: int


def _encode_varint(value: int) -> bytes:
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _otlp_string_attribute(key: str, value: str) -> bytes:
    key_bytes = key.encode()
    value_bytes = value.encode()
    any_value = b"\x0a" + _encode_varint(len(value_bytes)) + value_bytes
    return b"\x0a" + _encode_varint(len(key_bytes)) + key_bytes + b"\x12" + _encode_varint(len(any_value)) + any_value


def _scope_event(events, name: str, category: str, scope_category: str) -> ScopeEvent:
    return next(
        event
        for event in events
        if event.name == name
        and isinstance(event, ScopeEvent)
        and event.category == category
        and event.scope_category == scope_category
    )


class TestScopeType:
    def test_all_variants_exist(self):
        variants = [
            ScopeType.Agent,
            ScopeType.Function,
            ScopeType.Tool,
            ScopeType.Llm,
            ScopeType.Retriever,
            ScopeType.Embedder,
            ScopeType.Reranker,
            ScopeType.Guardrail,
            ScopeType.Evaluator,
            ScopeType.Custom,
            ScopeType.Unknown,
        ]
        assert len(variants) == 11

    def test_repr(self):
        assert "Agent" in repr(ScopeType.Agent)


class TestScopeAttributes:
    def test_parallel_is_int(self):
        assert isinstance(ScopeAttributes.PARALLEL, int)
        assert ScopeAttributes.PARALLEL == 0b01

    def test_relocatable_is_int(self):
        assert isinstance(ScopeAttributes.RELOCATABLE, int)
        assert ScopeAttributes.RELOCATABLE == 0b10

    def test_construct_from_value(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL)
        assert attrs.is_parallel
        assert not attrs.is_relocatable

    def test_construct_combined(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE)
        assert attrs.is_parallel
        assert attrs.is_relocatable

    def test_or_operator(self):
        a = ScopeAttributes(ScopeAttributes.PARALLEL)
        b = ScopeAttributes(ScopeAttributes.RELOCATABLE)
        combined = a | b
        assert combined.is_parallel
        assert combined.is_relocatable

    def test_value_getter(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL)
        assert attrs.value == ScopeAttributes.PARALLEL

    def test_and_operator_and_repr(self):
        combined = ScopeAttributes(ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE)
        parallel_only = ScopeAttributes(ScopeAttributes.PARALLEL)
        intersected = combined & parallel_only
        assert intersected.is_parallel
        assert not intersected.is_relocatable
        assert "ScopeAttributes" in repr(intersected)


class TestToolAttributes:
    def test_remote_is_int(self):
        assert isinstance(ToolAttributes.REMOTE, int)
        assert ToolAttributes.REMOTE == 0b01

    def test_construct(self):
        attrs = ToolAttributes(ToolAttributes.REMOTE)
        assert attrs.is_remote

    def test_empty(self):
        attrs = ToolAttributes(0)
        assert not attrs.is_remote

    def test_or_and_and_repr(self):
        remote = ToolAttributes(ToolAttributes.REMOTE)
        empty = ToolAttributes(0)
        combined = remote | empty
        intersected = remote & empty
        assert combined.is_remote
        assert not intersected.is_remote
        assert remote.value == ToolAttributes.REMOTE
        assert "ToolAttributes" in repr(remote)


class TestLLMAttributes:
    def test_stateful_is_int(self):
        assert isinstance(LLMAttributes.STATEFUL, int)

    def test_streaming_is_int(self):
        assert isinstance(LLMAttributes.STREAMING, int)

    def test_construct_combined(self):
        attrs = LLMAttributes(LLMAttributes.STATEFUL | LLMAttributes.STREAMING)
        assert attrs.is_stateful
        assert attrs.is_streaming

    def test_or_and_and_repr(self):
        stateful = LLMAttributes(LLMAttributes.STATEFUL)
        streaming = LLMAttributes(LLMAttributes.STREAMING)
        combined = stateful | streaming
        intersected = combined & stateful
        assert combined.is_streaming
        assert intersected.is_stateful
        assert not intersected.is_streaming
        assert combined.value == LLMAttributes.STATEFUL | LLMAttributes.STREAMING
        assert "LLMAttributes" in repr(combined)


class TestLLMRequest:
    def test_constructor(self):
        req = LLMRequest({"Authorization": "Bearer token"}, {"messages": []})
        assert req.headers == {"Authorization": "Bearer token"}
        assert req.content == {"messages": []}

    def test_empty_headers(self):
        req = LLMRequest({}, {"q": "test"})
        assert req.headers == {}

    def test_repr(self):
        req = LLMRequest({}, {"model": "gpt-4"})
        r = repr(req)
        assert "LLMRequest" in r

    def test_headers_must_be_dict(self):
        with pytest.raises(TypeError, match="not an instance of 'dict'"):
            LLMRequest(cast(dict[str, str], []), {"model": "gpt-4"})


class TestHandleTypes:
    def test_scope_type_roundtrip_all_variants(self):
        variants = [
            ScopeType.Agent,
            ScopeType.Function,
            ScopeType.Tool,
            ScopeType.Llm,
            ScopeType.Retriever,
            ScopeType.Embedder,
            ScopeType.Reranker,
            ScopeType.Guardrail,
            ScopeType.Evaluator,
            ScopeType.Custom,
            ScopeType.Unknown,
        ]

        for variant in variants:
            handle = scope.push(f"scope-{variant!r}", variant)
            try:
                assert handle.scope_type == variant
            finally:
                scope.pop(handle)

    def test_scope_handle_properties_and_repr(self):
        handle = scope.push(
            "typed_scope",
            ScopeType.Agent,
            attributes=ScopeAttributes(ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE),
            data={"scope": True},
            metadata={"meta": "scope"},
        )
        try:
            assert handle.name == "typed_scope"
            assert handle.scope_type == ScopeType.Agent
            assert handle.attributes.is_parallel
            assert handle.attributes.is_relocatable
            assert handle.data == {"scope": True}
            assert handle.metadata == {"meta": "scope"}
            assert "ScopeHandle" in repr(handle)
        finally:
            scope.pop(handle)

    def test_tool_handle_properties_and_repr(self):
        parent = scope.push("typed_tool_parent", ScopeType.Agent)
        try:
            handle = tools.call(
                "typed_tool",
                {"x": 1},
                attributes=ToolAttributes(ToolAttributes.REMOTE),
                data={"tool": "data"},
                metadata={"tool": "meta"},
            )
            try:
                assert handle.name == "typed_tool"
                assert handle.attributes.is_remote
                assert handle.parent_uuid == parent.uuid
                assert handle.data == {"tool": "data"}
                assert handle.metadata == {"tool": "meta"}
                assert "ToolHandle" in repr(handle)
            finally:
                tools.call_end(handle, ToolExecutionResult({"ok": True}))
        finally:
            scope.pop(parent)

    def test_llm_handle_properties_and_repr(self):
        parent = scope.push("typed_llm_parent", ScopeType.Agent)
        request = LLMRequest({}, {"messages": [], "model": "typed-model"})
        try:
            handle = llm.call(
                "typed_llm",
                request,
                attributes=LLMAttributes(LLMAttributes.STATEFUL | LLMAttributes.STREAMING),
                data={"llm": "data"},
                metadata={"llm": "meta"},
                model_name="typed-model",
            )
            try:
                assert handle.name == "typed_llm"
                assert handle.attributes.is_stateful
                assert handle.attributes.is_streaming
                assert handle.parent_uuid == parent.uuid
                assert handle.data == {"llm": "data"}
                assert handle.metadata == {"llm": "meta"}
                assert "LLMHandle" in repr(handle)
            finally:
                llm.call_end(handle, {"ok": True})
        finally:
            scope.pop(parent)


class TestConcreteEvents:
    def test_event_properties_include_tool_and_llm_fields(self):
        events = []
        subscribers.register("py_event_types_sub", lambda event: events.append(event))
        parent = scope.push("event_root", ScopeType.Agent, data={"root": True}, metadata={"meta": "root"})
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}], "model": "event-model"})

        try:
            tool_handle = tools.call(
                "event_tool",
                {"x": 1},
                data={"tool": "start"},
                metadata={"tool_meta": True},
                tool_call_id="tool-call-123",
            )
            tools.call_end(
                tool_handle,
                ToolExecutionResult({"y": 2}),
                data={"tool": "end"},
                metadata={"tool_end": True},
            )

            llm_handle = llm.call(
                "event_llm",
                request,
                data={"llm": "start"},
                metadata={"llm_meta": True},
                model_name="event-model",
            )
            llm.call_end(llm_handle, {"message": "hello"}, data={"llm": "end"}, metadata={"llm_end": True})

            scope.event("event_mark", handle=parent, data={"mark": True}, metadata={"mark_meta": True})
        finally:
            scope.pop(parent)
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_event_types_sub")

        tool_start = _scope_event(events, "event_tool", "tool", "start")
        tool_end = _scope_event(events, "event_tool", "tool", "end")
        llm_start = _scope_event(events, "event_llm", "llm", "start")
        llm_end = _scope_event(events, "event_llm", "llm", "end")
        mark = next(event for event in events if event.name == "event_mark" and isinstance(event, MarkEvent))

        assert tool_start.data == {"x": 1}
        assert tool_start.category_profile == {"tool_call_id": "tool-call-123"}
        assert tool_end.uuid == tool_start.uuid
        assert tool_end.data == {"y": 2}
        assert tool_end.metadata == {"tool_meta": True, "tool_end": True}

        assert llm_start.data == {"headers": request.headers, "content": request.content}
        assert llm_start.category_profile == {"model_name": "event-model"}
        assert llm_end.uuid == llm_start.uuid
        assert llm_end.data == {"message": "hello"}
        assert llm_end.metadata == {"llm_meta": True, "llm_end": True, "otel.status_code": "OK"}

        assert mark.kind == "mark"
        assert mark.parent_uuid == parent.uuid
        assert mark.data == {"mark": True}
        assert mark.metadata == {"mark_meta": True}
        assert "MarkEvent" in repr(mark)
        assert "T" in mark.timestamp

    def test_scope_type_is_only_present_on_scope_events(self):
        events = []
        subscribers.register("py_scope_type_contract_sub", lambda event: events.append(event))
        parent = scope.push("scope_contract_root", ScopeType.Agent)

        try:
            child = scope.push("scope_contract_child", ScopeType.Function)
            tool_handle = tools.call("scope_contract_tool", {"x": 1})
            tools.call_end(tool_handle, ToolExecutionResult({"y": 2}))
            llm_handle = llm.call("scope_contract_llm", LLMRequest({}, {"messages": [], "model": "m"}))
            llm.call_end(llm_handle, {"done": True})
            scope.pop(child)
        finally:
            scope.pop(parent)
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_scope_type_contract_sub")

        scope_start = _scope_event(events, "scope_contract_child", "function", "start")
        tool_start = _scope_event(events, "scope_contract_tool", "tool", "start")
        llm_start = _scope_event(events, "scope_contract_llm", "llm", "start")

        assert scope_start.category == "function"
        assert tool_start.category == "tool"
        assert llm_start.category == "llm"


class TestAtifExporterType:
    def test_exporter_register_export_clear_and_repr(self):
        exporter = AtifExporter(
            "session-types",
            "py-agent",
            "1.0.0",
            model_name="typed-model",
            tool_definitions=[{"name": "typed_tool"}],
            extra={"team": "qa"},
        )
        assert "<AtifExporter>" in repr(exporter)

        exporter.register("py_atif_exporter")
        parent = scope.push("atif_root", ScopeType.Agent)
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}], "model": "typed-model"})

        try:
            handle = llm.call("atif_llm", request, model_name="typed-model")
            llm.call_end(handle, {"content": "world"})

            exported_all = exporter.export()
            exported = exporter.export()
            exported_json_all = json.loads(exporter.export_json())
            exported_json = json.loads(exporter.export_json())
            agent = cast(JsonObject, cast(JsonObject, exported)["agent"])

            assert exported_all["session_id"] == "session-types"
            assert exported["session_id"] == "session-types"
            assert cast(str, agent["name"]) == "py-agent"
            assert cast(list[JsonObject], agent["tool_definitions"]) == [{"name": "typed_tool"}]
            assert cast(JsonObject, agent["extra"]) == {"team": "qa"}
            assert exported["steps"]
            assert exported_json_all["session_id"] == "session-types"
            assert exported_json["session_id"] == "session-types"

            exporter.clear()
            assert exporter.export()["steps"] == []
        finally:
            scope.pop(parent)
            assert exporter.deregister("py_atif_exporter") is True
            assert exporter.deregister("py_atif_exporter") is False


class TestAtofExporterType:
    def test_config_defaults_mutation_and_repr(self, tmp_path):
        config = AtofExporterConfig()

        assert config.mode == AtofExporterMode.Append
        assert config.filename.startswith("nemo-relay-events-")
        assert config.filename.endswith(".jsonl")
        assert config.sink_type == "file"
        assert "AtofExporterConfig" in repr(config)

        config.output_directory = str(tmp_path)
        config.mode = AtofExporterMode.Overwrite
        config.filename = "events.jsonl"
        config.sink_type = "stream"
        config.url = "http://localhost:8080/events"
        config.transport = "http_post"
        config.headers = {"X-Test": "yes"}
        config.header_env = {"authorization": "NEMO_RELAY_ATOF_AUTH"}
        config.timeout_millis = 1000
        config.field_name_policy = "replace_dots"

        assert config.output_directory == str(tmp_path)
        assert config.mode == AtofExporterMode.Overwrite
        assert config.filename == "events.jsonl"
        assert config.url == "http://localhost:8080/events"
        assert config.transport == "http_post"
        assert config.headers == {"X-Test": "yes"}
        assert config.header_env == {"authorization": "NEMO_RELAY_ATOF_AUTH"}
        assert config.timeout_millis == 1000
        assert config.field_name_policy == "replace_dots"

    def test_stream_sink_requires_url(self):
        config = AtofExporterConfig()
        config.sink_type = "stream"

        with pytest.raises(ValueError, match="stream sink requires url"):
            AtofExporter(config)

    def test_endpoint_field_name_policy_is_validated(self):
        config = AtofExporterConfig()
        config.sink_type = "stream"
        config.url = "http://localhost:8080/events"
        config.field_name_policy = "bogus"

        with pytest.raises(ValueError, match="field_name_policy"):
            AtofExporter(config)

    def test_exporter_lifecycle_writes_raw_jsonl_events(self, tmp_path):
        config = AtofExporterConfig()
        config.output_directory = str(tmp_path)
        config.mode = AtofExporterMode.Overwrite
        config.filename = "events.jsonl"

        exporter = AtofExporter(config)
        assert "<AtofExporter>" in repr(exporter)
        assert exporter.path is not None
        assert exporter.path.endswith("events.jsonl")

        subscriber_name = f"py_atof_{uuid4().hex}"
        exporter.register(subscriber_name)
        try:
            handle = scope.push("atof_scope", ScopeType.Agent, input={"scope": True})
            try:
                scope.event("atof_mark", handle=handle, data={"step": 1})
            finally:
                scope.pop(handle, output={"done": True})
        finally:
            assert exporter.deregister(subscriber_name) is True
            assert exporter.deregister(subscriber_name) is False
            exporter.force_flush()
            exporter.shutdown()
            subscribers.deregister(subscriber_name)

        lines = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert [line["kind"] for line in lines] == ["scope", "mark", "scope"]
        assert lines[0]["name"] == "atof_scope"
        assert lines[1]["data"] == {"step": 1}
        assert lines[2]["scope_category"] == "end"

    def test_append_and_overwrite_modes(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text('{"existing": true}\n')

        append_config = AtofExporterConfig()
        append_config.output_directory = str(tmp_path)
        append_config.filename = "events.jsonl"
        append_exporter = AtofExporter(append_config)
        append_exporter.shutdown()
        assert path.read_text().startswith('{"existing": true}\n')

        overwrite_config = AtofExporterConfig()
        overwrite_config.output_directory = str(tmp_path)
        overwrite_config.mode = AtofExporterMode.Overwrite
        overwrite_config.filename = "events.jsonl"
        overwrite_exporter = AtofExporter(overwrite_config)
        overwrite_exporter.shutdown()
        assert path.read_text() == ""


class TestOpenTelemetryTypes:
    @pytest.mark.parametrize("signal", ["traces", "logs", "metrics"])
    def test_signal_subscribers_expose_runtime_diagnostics(self, signal):
        if signal == "traces":
            subscriber = OpenTelemetrySubscriber(OpenTelemetryConfig("full", "http://127.0.0.1:4318"))
        elif signal == "logs":
            subscriber = OpenTelemetryLogSubscriber(OpenTelemetryLogConfig("http://127.0.0.1:4318"))
        else:
            subscriber = OpenTelemetryMetricSubscriber(OpenTelemetryMetricConfig("http://127.0.0.1:4318"))

        subscriber_name = f"py_otel_{signal}_diagnostics_{uuid4().hex}"
        subscriber.register(subscriber_name)
        try:
            for _ in range(3):
                scope.event(
                    "invalid_metric",
                    data={"measurements": []},
                    data_schema={"name": "nemo.relay.metric_measurements", "version": "999"},
                )
            subscribers.flush()

            diagnostics = subscriber.runtime_diagnostics()
            assert isinstance(diagnostics, OpenTelemetryRuntimeDiagnostics)
            diagnostic = diagnostics.get("otel.metric_mark_invalid")
            assert diagnostic is not None
            assert diagnostic.count == 3
            assert "unsupported metric schema version" in diagnostic.message
            assert [entry.code for entry in diagnostics.entries] == ["otel.metric_mark_invalid"]
        finally:
            subscriber.deregister(subscriber_name)
            subscriber.shutdown()

    def test_signal_config_defaults_and_lifecycle(self, monkeypatch: pytest.MonkeyPatch):
        variable = f"NEMO_RELAY_PY_SIGNAL_HEADER_{uuid4().hex}"
        monkeypatch.setenv(variable, "signal-route")
        log_config = OpenTelemetryLogConfig("http://localhost:4318/v1/logs")
        assert log_config.minimum_severity == LogSeverity.Info
        assert log_config.max_queue_size == 2048
        assert log_config.max_export_batch_size == 512
        assert log_config.scheduled_delay_millis == 1000
        assert log_config.completed_span_context_ttl_millis == 60000
        assert log_config.header_env == {}
        log_config.minimum_severity = LogSeverity.Warn
        log_config.headers = {"authorization": "Bearer token"}
        log_config.set_header_from_env("x-relay-route", variable)
        log_config.resource_attributes = {"deployment.environment": "test"}
        assert log_config.header_env == {"x-relay-route": variable}

        log_subscriber = OpenTelemetryLogSubscriber(log_config)
        log_name = f"py_otel_log_{uuid4().hex}"
        log_subscriber.register(log_name)
        try:
            assert log_subscriber.deregister(log_name)
            log_subscriber.force_flush()
        finally:
            subscribers.deregister(log_name)
            log_subscriber.shutdown()

        metric_config = OpenTelemetryMetricConfig("http://localhost:4318/v1/metrics")
        assert metric_config.export_interval_millis == 60000
        assert metric_config.temporality == MetricTemporality.Cumulative
        assert metric_config.max_instruments == 256
        assert metric_config.cardinality_limit == 2000
        assert metric_config.header_env == {}
        metric_config.temporality = MetricTemporality.Delta
        metric_config.headers = {"authorization": "Bearer token"}
        metric_config.set_header_from_env("x-relay-route", variable)
        metric_config.resource_attributes = {"deployment.environment": "test"}
        assert metric_config.header_env == {"x-relay-route": variable}

        metric_subscriber = OpenTelemetryMetricSubscriber(metric_config)
        metric_name = f"py_otel_metric_{uuid4().hex}"
        metric_subscriber.register(metric_name)
        try:
            assert metric_subscriber.deregister(metric_name)
            metric_subscriber.force_flush()
        finally:
            subscribers.deregister(metric_name)
            metric_subscriber.shutdown()

    def test_signal_subscribers_validate_limits(self):
        log_config = OpenTelemetryLogConfig("http://localhost:4318/v1/logs")
        log_config.max_queue_size = 0
        with pytest.raises(RuntimeError, match="max_queue_size must be greater than 0"):
            OpenTelemetryLogSubscriber(log_config)

        log_config = OpenTelemetryLogConfig("http://localhost:4318/v1/logs")
        log_config.completed_span_context_ttl_millis = 0
        with pytest.raises(RuntimeError, match="completed_span_context_ttl must be greater than 0"):
            OpenTelemetryLogSubscriber(log_config)

        metric_config = OpenTelemetryMetricConfig("http://localhost:4318/v1/metrics")
        metric_config.cardinality_limit = 0
        with pytest.raises(RuntimeError, match="cardinality_limit must be greater than 0"):
            OpenTelemetryMetricSubscriber(metric_config)

    def test_signal_subscribers_export_to_signal_relative_paths(self, monkeypatch: pytest.MonkeyPatch):
        with _OtelCollector() as collector:
            variable = f"NEMO_RELAY_PY_LOG_HEADER_{uuid4().hex}"
            secret = "python-log-activation-route"
            monkeypatch.setenv(variable, secret)
            log_config = OpenTelemetryLogConfig(collector.endpoint)
            log_config.header_env = {"x-relay-route": variable}
            log_subscriber = OpenTelemetryLogSubscriber(log_config)
            monkeypatch.setenv(variable, "python-log-changed-route")
            log_name = f"py_otel_log_e2e_{uuid4().hex}"
            log_subscriber.register(log_name)
            try:
                scope.event("log_mark", severity=LogSeverity.Info, data={"message": "ready"})
                log_subscriber.force_flush()
                request = collector.wait_for_request()
                assert request["path"] == "/v1/logs"
                assert request["headers"]["x-relay-route"] == secret
                assert request["body"]
                assert secret.encode() not in request["body"]
                assert all(secret not in entry.message for entry in log_subscriber.runtime_diagnostics().entries)
            finally:
                log_subscriber.deregister(log_name)
                log_subscriber.shutdown()

        with _OtelCollector() as collector:
            variable = f"NEMO_RELAY_PY_METRIC_HEADER_{uuid4().hex}"
            secret = "python-metric-activation-route"
            monkeypatch.setenv(variable, secret)
            metric_config = OpenTelemetryMetricConfig(collector.endpoint)
            metric_config.export_interval_millis = 100
            metric_config.header_env = {"x-relay-route": variable}
            metric_subscriber = OpenTelemetryMetricSubscriber(metric_config)
            monkeypatch.setenv(variable, "python-metric-changed-route")
            metric_name = f"py_otel_metric_e2e_{uuid4().hex}"
            metric_subscriber.register(metric_name)
            try:
                scope.metric(
                    "metric_mark",
                    [MetricMeasurement("relay.tokens", MetricKind.Counter, MetricValueType.U64, 3)],
                )
                metric_subscriber.force_flush()
                request = collector.wait_for_request()
                assert request["path"] == "/v1/metrics"
                assert request["headers"]["x-relay-route"] == secret
                assert request["body"]
                assert secret.encode() not in request["body"]
                assert all(secret not in entry.message for entry in metric_subscriber.runtime_diagnostics().entries)
            finally:
                metric_subscriber.deregister(metric_name)
                metric_subscriber.shutdown()

    def test_direct_log_subscriber_reports_queue_drops_after_force_flush(self):
        emitted = 200
        with _OtelCollector() as collector:
            config = OpenTelemetryLogConfig(collector.endpoint)
            config.max_queue_size = 2
            config.max_export_batch_size = 1
            config.scheduled_delay_millis = 60_000
            subscriber = OpenTelemetryLogSubscriber(config)
            subscriber_name = f"py_otel_log_drops_{uuid4().hex}"
            subscriber.register(subscriber_name)
            try:
                for index in range(emitted):
                    scope.event(
                        f"log-overflow-{index}",
                        data={"index": index},
                        severity=LogSeverity.Info,
                    )
                subscribers.flush()
                subscriber.force_flush()

                arrived = len(collector.server.requests)
                assert arrived < emitted, "the fixture must overflow the log queue"
                diagnostic = subscriber.runtime_diagnostics().get("otel.logs_dropped")
                assert diagnostic is not None
                assert diagnostic.count == emitted - arrived
                assert collector.endpoint in diagnostic.message
            finally:
                subscriber.deregister(subscriber_name)
                subscriber.shutdown()

    def test_config_defaults_mutation_and_repr(self):
        config = OpenTelemetryConfig("full", "http://localhost:4318/v1/traces")

        assert config.transport == "http_binary"
        assert config.endpoint == "http://localhost:4318/v1/traces"
        assert config.service_name == "unknown_service"
        assert config.completed_span_context_ttl_millis == 60_000
        assert config.instrumentation_scope == "opentelemetry"
        assert config.timeout_millis == 3000
        assert config.headers == {}
        assert config.header_env == {}
        assert config.resource_attributes == {}
        assert config.mark_projection == "inherit"
        assert config.mark_exclude_names == ["llm.chunk"]
        assert config.attribute_mappings == []
        assert config.promote_metadata_prefixes == []
        assert config.promote_resource_metadata_prefixes == []

        config.service_name = "py-agent"
        config.service_namespace = "agents"
        config.service_version = "1.0.0"
        config.instrumentation_scope = "py-tests"
        config.timeout_millis = 1250
        config.set_header("authorization", "Bearer token")
        config.set_header_from_env("x-api-key", "NEMO_RELAY_API_KEY")
        config.set_resource_attribute("deployment.environment", "test")
        config.mark_projection = "tool"
        config.mark_exclude_names = ["custom.mark"]
        config.attribute_mappings = [{"key": "nemo_relay.model_name", "alias": "model.alias"}]
        config.promote_metadata_prefixes = ["nv."]
        config.promote_resource_metadata_prefixes = ["deployment."]

        assert config.headers == {"authorization": "Bearer token"}
        assert config.header_env == {"x-api-key": "NEMO_RELAY_API_KEY"}
        assert config.resource_attributes == {"deployment.environment": "test"}
        assert config.mark_projection == "tool"
        assert config.mark_exclude_names == ["custom.mark"]
        assert config.attribute_mappings == [{"key": "nemo_relay.model_name", "alias": "model.alias"}]
        assert config.promote_metadata_prefixes == ["nv."]
        assert config.promote_resource_metadata_prefixes == ["deployment."]
        assert "OpenTelemetryConfig" in repr(config)

    def test_config_rejects_invalid_map_values(self):
        config = OpenTelemetryConfig("full", "http://localhost:4318/v1/traces")

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.headers = cast(dict[str, str], [])

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.header_env = cast(dict[str, str], {"authorization": 1})

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.resource_attributes = cast(dict[str, str], {"env": 1})

        with pytest.raises(ValueError, match="attribute mapping key must not be blank"):
            config.attribute_mappings = [{"key": "", "alias": "x"}]

        config.attribute_mappings = []
        config.promote_metadata_prefixes = ["nv.*"]
        with pytest.raises(ValueError, match="literal prefix, not a glob"):
            OpenTelemetrySubscriber(config)

        config.promote_metadata_prefixes = []
        config.promote_resource_metadata_prefixes = ["deployment.*"]
        with pytest.raises(ValueError, match="literal prefix, not a glob"):
            OpenTelemetrySubscriber(config)

    def test_subscriber_lifecycle_and_invalid_transport(self):
        config = OpenTelemetryConfig("full", "http://localhost:4318/v1/traces")
        config.service_name = "py-agent"

        subscriber = OpenTelemetrySubscriber(config)
        assert "<OpenTelemetrySubscriber>" in repr(subscriber)

        subscriber_name = f"py_otel_subscriber_{uuid4().hex}"
        subscriber.register(subscriber_name)
        try:
            assert subscriber.deregister(subscriber_name) is True
            assert subscriber.deregister(subscriber_name) is False
            subscriber.force_flush()
            subscriber.shutdown()
        finally:
            subscribers.deregister(subscriber_name)

        bad = OpenTelemetryConfig("full", "http://localhost:4318/v1/traces")
        bad.transport = "invalid"
        with pytest.raises(ValueError, match="transport must be"):
            OpenTelemetrySubscriber(bad)

    def test_subscriber_rejects_missing_or_invalid_required_fields(self):
        with pytest.raises(TypeError):
            OpenTelemetryConfig()  # ty: ignore[missing-argument]

        invalid_type = OpenTelemetryConfig(
            "invalid",  # ty: ignore[invalid-argument-type]
            "http://localhost:4318/v1/traces",
        )
        with pytest.raises(ValueError, match="type must be"):
            OpenTelemetrySubscriber(invalid_type)

        blank_endpoint = OpenTelemetryConfig("full", " \t")
        with pytest.raises(ValueError, match="endpoint is required and must be nonblank"):
            OpenTelemetrySubscriber(blank_endpoint)

        zero_ttl = OpenTelemetryConfig("full", "http://localhost:4318/v1/traces")
        zero_ttl.completed_span_context_ttl_millis = 0
        with pytest.raises(RuntimeError, match="completed_span_context_ttl must be greater than 0"):
            OpenTelemetrySubscriber(zero_ttl)

    def test_subscriber_exports_scope_and_mark_events_end_to_end(self, monkeypatch: pytest.MonkeyPatch):
        with _OtelCollector() as collector:
            source = "python-é" * 20
            variable = f"NEMO_RELAY_PY_HEADER_{uuid4().hex}"
            secret = "Bearer python-activation-secret"
            monkeypatch.setenv(variable, secret)
            config = OpenTelemetryConfig("full", collector.endpoint)
            config.service_name = "py-agent"
            config.promote_metadata_prefixes = ["nv."]
            config.header_env = {"authorization": variable}

            subscriber = OpenTelemetrySubscriber(config)
            monkeypatch.setenv(variable, "Bearer python-changed-secret")
            subscriber_name = f"py_otel_e2e_{uuid4().hex}"
            subscriber.register(subscriber_name)

            try:
                handle = scope.push(
                    "otel_scope",
                    ScopeType.Agent,
                    data={"scope": True},
                    metadata={"nv.binding": "python"},
                )
                try:
                    scope.event(
                        "otel_mark",
                        handle=handle,
                        data={"step": 1},
                        metadata={"source": source},
                    )
                finally:
                    scope.pop(handle, metadata={"nv.binding": "python"})

                subscriber.force_flush()
                request = collector.wait_for_request()
                assert request["path"] == "/v1/traces"
                assert request["headers"]["content-type"] == "application/x-protobuf"
                assert request["headers"]["authorization"] == secret
                assert request["body"]
                assert secret.encode() not in request["body"]
                assert b"nemo_relay.mark.metadata.source" in request["body"]
                assert _otlp_string_attribute("nv.binding", "python") in request["body"]
                assert all(secret not in entry.message for entry in subscriber.runtime_diagnostics().entries)
            finally:
                subscriber.deregister(subscriber_name)
                subscriber.shutdown()

    def test_subscriber_rejects_header_env_case_collision(self, monkeypatch: pytest.MonkeyPatch):
        variable = f"NEMO_RELAY_PY_DUPLICATE_HEADER_{uuid4().hex}"
        monkeypatch.setenv(variable, "Bearer secret")
        config = OpenTelemetryConfig("full", "http://localhost:4318/v1/traces")
        config.headers = {"Authorization": "static"}
        config.header_env = {"authorization": variable}

        with pytest.raises(RuntimeError, match="unique across headers and header_env"):
            OpenTelemetrySubscriber(config)

    def test_subscriber_rejects_unset_blank_and_invalid_header_env_values(self, monkeypatch: pytest.MonkeyPatch):
        variable = f"NEMO_RELAY_PY_INVALID_HEADER_{uuid4().hex}"
        config = OpenTelemetryConfig("full", "http://localhost:4318/v1/traces")
        config.header_env = {"authorization": variable}

        monkeypatch.delenv(variable, raising=False)
        with pytest.raises(RuntimeError, match="is not set"):
            OpenTelemetrySubscriber(config)

        monkeypatch.setenv(variable, "  ")
        with pytest.raises(RuntimeError, match="nonblank value"):
            OpenTelemetrySubscriber(config)

        secret = "relay-python-secret"
        monkeypatch.setenv(variable, f"{secret}\ninvalid")
        with pytest.raises(RuntimeError, match="valid header value") as failure:
            OpenTelemetrySubscriber(config)
        assert secret not in str(failure.value)

    def test_trace_export_failure_stays_unhealthy_until_a_later_export_succeeds(self):
        with _OtelCollector(response_status=503) as collector:
            subscriber = OpenTelemetrySubscriber(OpenTelemetryConfig("full", collector.endpoint))
            subscriber_name = f"py_otel_trace_failure_{uuid4().hex}"
            subscriber.register(subscriber_name)
            try:
                failed_scope = scope.push("trace-export-failure", ScopeType.Agent)
                scope.pop(failed_scope)
                with pytest.raises(RuntimeError):
                    subscriber.force_flush()

                diagnostic = subscriber.runtime_diagnostics().get("otel.traces_export_failed")
                assert diagnostic is not None
                assert diagnostic.count == 1
                endpoint_origin = f"http://127.0.0.1:{collector.server.server_port}"
                assert endpoint_origin in diagnostic.message
                assert collector.endpoint not in diagnostic.message

                with pytest.raises(RuntimeError, match=r"otel\.traces_export_failed \(1\)"):
                    subscriber.force_flush()

                collector.server.response_status = 200
                recovered_scope = scope.push("trace-export-recovery", ScopeType.Agent)
                scope.pop(recovered_scope)
                subscriber.force_flush()
            finally:
                subscriber.deregister(subscriber_name)
                subscriber.shutdown()

    def test_gen_ai_subscriber_exports_standardized_agent_span(self):
        with _OtelCollector() as collector:
            config = OpenTelemetryConfig("gen_ai", collector.endpoint)
            subscriber = OpenTelemetrySubscriber(config)
            subscriber_name = f"py_gen_ai_e2e_{uuid4().hex}"
            subscriber.register(subscriber_name)

            try:
                handle = scope.push("research-agent", ScopeType.Agent)
                scope.pop(handle)

                subscriber.force_flush()
                request = collector.wait_for_request()
                assert request["path"] == "/v1/traces"
                assert b"invoke_agent research-agent" in request["body"]
                assert b"gen_ai.operation.name" in request["body"]
                assert b"nemo_relay." not in request["body"]
            finally:
                subscriber.deregister(subscriber_name)
                subscriber.shutdown()


class TestOpenInferenceTypes:
    def test_config_defaults_mutation_and_repr(self):
        config = OpenTelemetryConfig("openinference", "http://localhost:4318/v1/traces")

        assert config.transport == "http_binary"
        assert config.service_name == "unknown_service"
        assert config.instrumentation_scope == "opentelemetry"
        assert config.timeout_millis == 3000
        assert config.headers == {}
        assert config.resource_attributes == {}

        config.service_name = "py-agent"
        config.service_namespace = "agents"
        config.service_version = "1.0.0"
        config.instrumentation_scope = "py-tests"
        config.timeout_millis = 1250
        config.set_header("authorization", "Bearer token")
        config.set_resource_attribute("deployment.environment", "test")

        assert config.headers == {"authorization": "Bearer token"}
        assert config.resource_attributes == {"deployment.environment": "test"}
        assert "OpenTelemetryConfig" in repr(config)

    def test_config_rejects_invalid_map_values(self):
        config = OpenTelemetryConfig("openinference", "http://localhost:4318/v1/traces")

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.headers = cast(dict[str, str], [])

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.resource_attributes = cast(dict[str, str], {"env": 1})

    def test_subscriber_lifecycle_and_invalid_transport(self):
        config = OpenTelemetryConfig("openinference", "http://localhost:4318/v1/traces")
        config.service_name = "py-agent"

        subscriber = OpenTelemetrySubscriber(config)
        assert "<OpenTelemetrySubscriber>" in repr(subscriber)

        subscriber_name = f"py_openinference_subscriber_{uuid4().hex}"
        subscriber.register(subscriber_name)
        try:
            assert subscriber.deregister(subscriber_name) is True
            assert subscriber.deregister(subscriber_name) is False
            subscriber.force_flush()
            subscriber.shutdown()
        finally:
            subscribers.deregister(subscriber_name)

        grpc = OpenTelemetryConfig("openinference", "http://127.0.0.1:4317")
        grpc.transport = "grpc"
        grpc.service_name = "py-agent-grpc"
        grpc_subscriber = OpenTelemetrySubscriber(grpc)
        grpc_subscriber.shutdown()

        bad = OpenTelemetryConfig("openinference", "http://localhost:4318/v1/traces")
        bad.transport = "invalid"
        with pytest.raises(ValueError, match="transport must be"):
            OpenTelemetrySubscriber(bad)

    def test_subscriber_exports_scope_and_mark_events_end_to_end(self):
        with _OtelCollector() as collector:
            source = "python-é" * 20
            config = OpenTelemetryConfig("openinference", collector.endpoint)
            config.service_name = "py-agent"
            subscriber = OpenTelemetrySubscriber(config)
            subscriber_name = f"py_openinference_e2e_{uuid4().hex}"
            subscriber.register(subscriber_name)

            try:
                handle = scope.push("openinference_scope", ScopeType.Agent, data={"scope": True})
                try:
                    scope.event(
                        "openinference_mark",
                        handle=handle,
                        data={"step": 1},
                        metadata={"source": source},
                    )
                finally:
                    scope.pop(handle)

                subscriber.force_flush()
                request = collector.wait_for_request()
                assert request["path"] == "/v1/traces"
                assert request["headers"]["content-type"] == "application/x-protobuf"
                assert request["body"]
                assert b"openinference.span.kind" in request["body"]
                assert b"AGENT" in request["body"]
                assert b"metadata" in request["body"]
                assert b"openinference_mark" in request["body"]
            finally:
                subscriber.deregister(subscriber_name)
                subscriber.shutdown()
