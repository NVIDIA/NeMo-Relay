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
    AtofEndpointConfig,
    AtofExporter,
    AtofExporterConfig,
    AtofExporterMode,
    JsonObject,
    LLMAttributes,
    LLMRequest,
    MarkEvent,
    OpenInferenceConfig,
    OpenInferenceSubscriber,
    OpenTelemetryConfig,
    OpenTelemetrySubscriber,
    ScopeAttributes,
    ScopeEvent,
    ScopeType,
    ToolAttributes,
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
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        return


class _CollectorRequest(TypedDict):
    path: str
    headers: dict[str, str]
    body: bytes


class _OtelCollector:
    server: "_OtelCollectorServer"

    def __enter__(self) -> "_OtelCollector":
        self.server = _OtelCollectorServer(("127.0.0.1", 0), _OtelCollectorHandler)
        self.server.requests = []
        self.server.request_event = threading.Event()
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
                tools.call_end(handle, {"ok": True})
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
            tools.call_end(tool_handle, {"y": 2}, data={"tool": "end"}, metadata={"tool_end": True})

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
            tools.call_end(tool_handle, {"y": 2})
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
        assert config.endpoints == []
        assert "AtofExporterConfig" in repr(config)

        config.output_directory = str(tmp_path)
        config.mode = AtofExporterMode.Overwrite
        config.filename = "events.jsonl"
        endpoint = AtofEndpointConfig(
            "http://localhost:8080/events",
            transport="http_post",
            headers={"X-Test": "yes"},
            timeout_millis=1000,
            field_name_policy="replace_dots",
        )
        config.endpoints = [endpoint]

        assert config.output_directory == str(tmp_path)
        assert config.mode == AtofExporterMode.Overwrite
        assert config.filename == "events.jsonl"
        assert config.endpoints[0].url == "http://localhost:8080/events"
        assert config.endpoints[0].transport == "http_post"
        assert config.endpoints[0].headers == {"X-Test": "yes"}
        assert config.endpoints[0].timeout_millis == 1000
        assert config.endpoints[0].field_name_policy == "replace_dots"

    def test_endpoint_field_name_policy_is_validated(self, tmp_path):
        config = AtofExporterConfig()
        config.output_directory = str(tmp_path)
        config.endpoints = [
            AtofEndpointConfig(
                "http://localhost:8080/events",
                field_name_policy="bogus",  # type: ignore[arg-type]
            )
        ]

        with pytest.raises(ValueError, match="field_name_policy"):
            AtofExporter(config)

    def test_exporter_lifecycle_writes_raw_jsonl_events(self, tmp_path):
        config = AtofExporterConfig()
        config.output_directory = str(tmp_path)
        config.mode = AtofExporterMode.Overwrite
        config.filename = "events.jsonl"

        exporter = AtofExporter(config)
        assert "<AtofExporter>" in repr(exporter)
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
    def test_config_defaults_mutation_and_repr(self):
        config = OpenTelemetryConfig()

        assert config.transport == "http_binary"
        assert config.endpoint is None
        assert config.service_name == "nemo-relay"
        assert config.instrumentation_scope == "nemo-relay-otel"
        assert config.timeout_millis == 3000
        assert config.headers == {}
        assert config.resource_attributes == {}

        config.endpoint = "http://localhost:4318/v1/traces"
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
        config = OpenTelemetryConfig()

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.headers = cast(dict[str, str], [])

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.resource_attributes = cast(dict[str, str], {"env": 1})

    def test_subscriber_lifecycle_and_invalid_transport(self):
        config = OpenTelemetryConfig()
        config.endpoint = "http://localhost:4318/v1/traces"
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

        bad = OpenTelemetryConfig()
        bad.transport = "invalid"
        with pytest.raises(ValueError, match="transport must be"):
            OpenTelemetrySubscriber(bad)

    def test_subscriber_exports_scope_and_mark_events_end_to_end(self):
        with _OtelCollector() as collector:
            config = OpenTelemetryConfig()
            config.endpoint = collector.endpoint
            config.service_name = "py-agent"

            subscriber = OpenTelemetrySubscriber(config)
            subscriber_name = f"py_otel_e2e_{uuid4().hex}"
            subscriber.register(subscriber_name)

            try:
                handle = scope.push("otel_scope", ScopeType.Agent, data={"scope": True})
                try:
                    scope.event(
                        "otel_mark",
                        handle=handle,
                        data={"step": 1},
                        metadata={"source": "python"},
                    )
                finally:
                    scope.pop(handle)

                subscriber.force_flush()
                request = collector.wait_for_request()
                assert request["path"] == "/v1/traces"
                assert request["headers"]["content-type"] == "application/x-protobuf"
                assert request["body"]
            finally:
                subscriber.deregister(subscriber_name)
                subscriber.shutdown()


class TestOpenInferenceTypes:
    def test_config_defaults_mutation_and_repr(self):
        config = OpenInferenceConfig()

        assert config.transport == "http_binary"
        assert config.endpoint is None
        assert config.service_name == "nemo-relay"
        assert config.instrumentation_scope == "nemo-relay-openinference"
        assert config.timeout_millis == 3000
        assert config.headers == {}
        assert config.resource_attributes == {}

        config.endpoint = "http://localhost:4318/v1/traces"
        config.service_name = "py-agent"
        config.service_namespace = "agents"
        config.service_version = "1.0.0"
        config.instrumentation_scope = "py-tests"
        config.timeout_millis = 1250
        config.set_header("authorization", "Bearer token")
        config.set_resource_attribute("deployment.environment", "test")

        assert config.headers == {"authorization": "Bearer token"}
        assert config.resource_attributes == {"deployment.environment": "test"}
        assert "OpenInferenceConfig" in repr(config)

    def test_config_rejects_invalid_map_values(self):
        config = OpenInferenceConfig()

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.headers = cast(dict[str, str], [])

        with pytest.raises(ValueError, match="dict\\[str, str\\]"):
            config.resource_attributes = cast(dict[str, str], {"env": 1})

    def test_subscriber_lifecycle_and_invalid_transport(self):
        config = OpenInferenceConfig()
        config.endpoint = "http://localhost:4318/v1/traces"
        config.service_name = "py-agent"

        subscriber = OpenInferenceSubscriber(config)
        assert "<OpenInferenceSubscriber>" in repr(subscriber)

        subscriber_name = f"py_openinference_subscriber_{uuid4().hex}"
        subscriber.register(subscriber_name)
        try:
            assert subscriber.deregister(subscriber_name) is True
            assert subscriber.deregister(subscriber_name) is False
            subscriber.force_flush()
            subscriber.shutdown()
        finally:
            subscribers.deregister(subscriber_name)

        grpc = OpenInferenceConfig()
        grpc.transport = "grpc"
        grpc.endpoint = "http://127.0.0.1:4317"
        grpc.service_name = "py-agent-grpc"
        grpc_subscriber = OpenInferenceSubscriber(grpc)
        grpc_subscriber.shutdown()

        bad = OpenInferenceConfig()
        bad.transport = "invalid"
        with pytest.raises(ValueError, match="transport must be"):
            OpenInferenceSubscriber(bad)

    def test_subscriber_exports_scope_and_mark_events_end_to_end(self):
        with _OtelCollector() as collector:
            config = OpenInferenceConfig()
            config.endpoint = collector.endpoint
            config.service_name = "py-agent"

            subscriber = OpenInferenceSubscriber(config)
            subscriber_name = f"py_openinference_e2e_{uuid4().hex}"
            subscriber.register(subscriber_name)

            try:
                handle = scope.push("openinference_scope", ScopeType.Agent, data={"scope": True})
                try:
                    scope.event(
                        "openinference_mark",
                        handle=handle,
                        data={"step": 1},
                        metadata={"source": "python"},
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
