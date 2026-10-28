# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in observability plugin config helpers."""

from __future__ import annotations

import http.server
import json
import os
import threading
import time
import typing
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from plugin_host_test_helper import validate_plugin_config

from nemo_relay import (
    LLMRequest,
    PropagationContext,
    ScopeType,
    create_scope_stack_from_propagation,
    llm,
    plugin,
    scope,
    subscribers,
    use_scope_stack,
)
from nemo_relay.codecs import AnthropicMessagesCodec
from nemo_relay.observability import (
    OBSERVABILITY_PLUGIN_KIND,
    AtifConfig,
    AtofConfig,
    AtofEndpointConfig,
    AtofFileSinkConfig,
    AtofStreamSinkConfig,
    ComponentSpec,
    ConfigPolicy,
    HttpStorageConfig,
    ObservabilityConfig,
    OpenTelemetryEndpointConfig,
    OpenTelemetryFileSinkConfig,
    OpenTelemetryLogSectionConfig,
    OpenTelemetryMetricSectionConfig,
    OpenTelemetrySectionConfig,
    OpenTelemetrySignalEndpointConfig,
    S3StorageConfig,
)

if typing.TYPE_CHECKING:
    from pathlib import Path


class _AtofCaptureServer(http.server.ThreadingHTTPServer):
    requests: list[tuple[dict[str, str], bytes]]
    request_event: threading.Event
    first_response_gate: threading.Event | None
    response_status: int


class _AtofCaptureHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("content-length", "0"))
        server = typing.cast(_AtofCaptureServer, self.server)
        server.requests.append((dict(self.headers.items()), self.rfile.read(content_length)))
        server.request_event.set()
        if len(server.requests) == 1 and server.first_response_gate is not None:
            server.first_response_gate.wait(timeout=30)
        self.send_response(server.response_status)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        return


class _AtofCapture:
    server: "_AtofCaptureServer"
    thread: threading.Thread

    def __init__(self, *, block_first_response: bool = False, response_status: int = 200) -> None:
        self.first_response_gate = threading.Event() if block_first_response else None
        self.response_status = response_status

    def __enter__(self) -> _AtofCapture:
        self.server = _AtofCaptureServer(("127.0.0.1", 0), _AtofCaptureHandler)
        self.server.requests = []
        self.server.request_event = threading.Event()
        self.server.first_response_gate = self.first_response_gate
        self.server.response_status = self.response_status
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.release_first_response()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def release_first_response(self) -> None:
        if self.first_response_gate is not None:
            self.first_response_gate.set()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def wait_for_requests(self, expected: int, timeout: float = 5.0) -> list[tuple[dict[str, str], bytes]]:
        deadline = time.monotonic() + timeout
        while len(self.server.requests) < expected:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"timed out waiting for {expected} ATOF requests"
            self.server.request_event.wait(remaining)
            self.server.request_event.clear()
        return self.server.requests


class TestObservabilityConfigHelpers:
    def test_file_sink_defaults_and_strict_validation(self, tmp_path: Path) -> None:
        sink = OpenTelemetryFileSinkConfig(output_directory=str(tmp_path))
        assert sink.to_dict() == {
            "output_directory": str(tmp_path),
            "type": "full",
            "format": "json_lines",
            "mode": "overwrite",
            "mark_projection": "inherit",
            "mark_exclude_names": ["llm.chunk"],
            "attribute_mappings": [],
            "promote_metadata_prefixes": [],
            "promote_resource_metadata_prefixes": [],
            "resource_attributes": {},
            "service_name": "unknown_service",
            "instrumentation_scope": "opentelemetry",
        }
        other = OpenTelemetryFileSinkConfig(output_directory=str(tmp_path))
        sink.mark_exclude_names.append("custom.mark")
        assert other.mark_exclude_names == ["llm.chunk"]
        section = OpenTelemetrySectionConfig(enabled=True, file_sinks=[sink])
        assert OpenTelemetrySectionConfig().file_sinks == []
        config = ObservabilityConfig(opentelemetry=section, policy=ConfigPolicy(unknown_field="error"))
        report = validate_plugin_config(plugin.PluginConfig(components=[ComponentSpec(config)]))
        assert report["diagnostics"] == []

    @pytest.mark.parametrize("projection", ["full", "gen_ai", "openinference"])
    async def test_typed_file_sink_writes_spans(
        self, tmp_path: Path, projection: typing.Literal["full", "gen_ai", "openinference"]
    ) -> None:
        sink = OpenTelemetryFileSinkConfig(
            output_directory=str(tmp_path),
            type=projection,
            filename="traces.jsonl",
            mode="append",
            mark_projection="event",
            mark_exclude_names=[],
            attribute_mappings=[{"key": "nemo_relay.scope.name", "alias": "scope_name"}],
            promote_metadata_prefixes=["custom."],
            promote_resource_metadata_prefixes=["deployment."],
            resource_attributes={"environment": "test"},
            service_name="python-file-sink",
            service_namespace="tests",
            service_version="1.0",
            instrumentation_scope="python-test",
            max_queue_size=32,
            max_export_batch_size=16,
            scheduled_delay_millis=60_000,
            completed_span_context_ttl_millis=60_000,
        )
        config = ObservabilityConfig(
            opentelemetry=OpenTelemetrySectionConfig(enabled=True, file_sinks=[sink]),
            policy=ConfigPolicy(unknown_field="error"),
        )
        activation = await plugin.initialize(plugin.PluginConfig(components=[ComponentSpec(config)]))
        try:
            with scope.scope("python-file-agent", ScopeType.Agent):
                pass
        finally:
            await activation.close()
        records = [json.loads(line) for line in (tmp_path / "traces.jsonl").read_text().splitlines()]
        spans = [
            span
            for record in records
            for resource in record["resourceSpans"]
            for instrumentation in resource["scopeSpans"]
            for span in instrumentation["spans"]
        ]
        expected_name = "invoke_agent python-file-agent" if projection == "gen_ai" else "python-file-agent"
        assert any(span["name"] == expected_name for span in spans)

    async def test_otlp_queue_overflow_is_accounted_per_endpoint_after_close(self) -> None:
        burst_size = 300
        projected_spans = burst_size + 2  # The seed child, burst children, and enclosing Agent.
        with _AtofCapture(block_first_response=True) as tiny, _AtofCapture() as healthy:
            endpoints = [capture.url for capture in (tiny, healthy)]
            config = plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        ObservabilityConfig(
                            opentelemetry=OpenTelemetrySectionConfig(
                                enabled=True,
                                endpoints=[
                                    OpenTelemetryEndpointConfig(
                                        "full",
                                        endpoint,
                                        max_queue_size=capacity,
                                        max_export_batch_size=1,
                                        scheduled_delay_millis=60_000,
                                        timeout_millis=30_000,
                                    )
                                    for endpoint, capacity in zip(endpoints, (1, projected_spans), strict=True)
                                ],
                            )
                        )
                    )
                ]
            )
            activation = await plugin.initialize(config)
            try:
                with scope.scope("overflow-agent", ScopeType.Agent):
                    with scope.scope("overflow-seed", ScopeType.Function):
                        pass
                    tiny.wait_for_requests(1)
                    for index in range(burst_size):
                        with scope.scope(f"overflow-child-{index}", ScopeType.Function):
                            pass
                # Drain event delivery while the first export still holds the tiny queue.
                await subscribers.flush_async()
                tiny.release_first_response()
                with pytest.raises(RuntimeError, match=r"otel\.spans_dropped"):
                    await activation.close()

                assert not activation.is_active
                retained = activation.report["config"]["runtime_diagnostics"]
                drops = [diagnostic for diagnostic in retained if diagnostic["code"] == "otel.spans_dropped"]
                assert retained == drops
                assert len(drops) == 1
                assert drops[0]["field"] == "opentelemetry.traces[0].endpoint"
                assert endpoints[0] in drops[0]["message"]
                assert drops[0]["count"] > 0

                for capture, dropped in zip((tiny, healthy), (drops[0]["count"], 0), strict=True):
                    delivered = sum(
                        len(scope_spans.spans)
                        for _, body in capture.server.requests
                        for resource_spans in ExportTraceServiceRequest.FromString(body).resource_spans
                        for scope_spans in resource_spans.scope_spans
                    )
                    assert delivered > 0
                    assert delivered + dropped == projected_spans

                assert activation.report["config"]["runtime_diagnostics"] == retained
                assert activation.report["config"]["runtime_diagnostics"] == retained
                with pytest.raises(RuntimeError, match=r"otel\.spans_dropped"):
                    await activation.close()
                assert activation.report["config"]["runtime_diagnostics"] == retained

                replacement = await plugin.initialize(plugin.PluginConfig(components=[]))
                await replacement.close()
            finally:
                tiny.release_first_response()
                if activation.is_active:
                    await activation.close()

    def test_opentelemetry_endpoint_preserves_existing_positional_arguments(self) -> None:
        endpoint = OpenTelemetryEndpointConfig(
            "full",
            "http://localhost:4318/v1/traces",
            "event",
            ["custom.mark"],
            [{"key": "source", "alias": "destination"}],
            "grpc",
            "relay-service",
            "relay-namespace",
            "1.2.3",
            "relay-instrumentation",
            1234,
            {"x-test": "header"},
            {"authorization": "OTEL_AUTHORIZATION"},
            {"deployment.environment": "test"},
        )

        assert endpoint.headers == {"x-test": "header"}
        assert endpoint.header_env == {"authorization": "OTEL_AUTHORIZATION"}
        assert endpoint.resource_attributes == {"deployment.environment": "test"}
        assert endpoint.max_queue_size is None
        assert endpoint.max_export_batch_size is None
        assert endpoint.scheduled_delay_millis is None
        assert endpoint.completed_span_context_ttl_millis is None

    def test_header_file_serializes_for_every_remote_observability_destination(self) -> None:
        path = "/var/run/secrets/telemetry/token"
        assert AtofStreamSinkConfig(url="https://example.com/events", header_file={"authorization": path}).to_dict()[
            "header_file"
        ] == {"authorization": path}
        assert HttpStorageConfig(endpoint="https://example.com/atif", header_file={"authorization": path}).to_dict()[
            "header_file"
        ] == {"authorization": path}
        assert OpenTelemetryEndpointConfig(
            "full", "https://example.com/v1/traces", header_file={"authorization": path}
        ).to_dict()["header_file"] == {"authorization": path}
        assert OpenTelemetrySignalEndpointConfig(
            "https://example.com/v1/logs", header_file={"authorization": path}
        ).to_dict()["header_file"] == {"authorization": path}

    def test_insecure_http_hosts_serialize_at_top_level_and_pass_strict_validation(self) -> None:
        hosts = ["host.docker.internal", "collector.ns.svc.cluster.local"]
        section = OpenTelemetrySectionConfig(
            enabled=True,
            endpoints=[OpenTelemetryEndpointConfig("gen_ai", "http://host.docker.internal:8082/v1/traces")],
        )
        config = ObservabilityConfig(
            opentelemetry=section, policy=ConfigPolicy(unknown_field="error"), insecure_http_hosts=hosts
        )
        assert config.to_dict()["insecure_http_hosts"] == hosts
        assert "insecure_http_hosts" not in ObservabilityConfig().to_dict()
        assert ObservabilityConfig(insecure_http_hosts=[]).to_dict()["insecure_http_hosts"] == []

        report = validate_plugin_config(plugin.PluginConfig(components=[ComponentSpec(config)]))
        assert report["diagnostics"] == []

    async def test_insecure_http_hosts_activate_a_listed_plain_http_trace_endpoint(self) -> None:
        def config(hosts: list[str] | None) -> plugin.PluginConfig:
            section = OpenTelemetrySectionConfig(
                enabled=True,
                endpoints=[OpenTelemetryEndpointConfig("gen_ai", "http://collector.invalid:4318/v1/traces")],
            )
            return plugin.PluginConfig(
                components=[ComponentSpec(ObservabilityConfig(opentelemetry=section, insecure_http_hosts=hosts))]
            )

        with pytest.raises(ValueError, match="insecure_http_hosts"):
            await plugin.initialize(config(None))
        activation = await plugin.initialize(config(["collector.invalid"]))
        assert activation.is_active
        await activation.close()

    def test_defaults_and_component_wrapper(self) -> None:
        assert AtofConfig().to_dict() == {"enabled": False}
        assert AtifConfig().to_dict() == {
            "enabled": False,
            "agent_name": "NeMo Relay",
            "model_name": "unknown",
            "filename_template": "nemo-relay-atif-{session_id}.json",
            "include_observed_events": True,
            "include_llm_requests": True,
        }
        reduced = AtifConfig(include_observed_events=False, include_llm_requests=False).to_dict()
        assert reduced["include_observed_events"] is False
        assert reduced["include_llm_requests"] is False
        assert OpenTelemetrySectionConfig().to_dict() == {
            "enabled": False,
            "endpoints": [],
            "file_sinks": [],
        }
        assert OpenTelemetryEndpointConfig(
            "gen_ai",
            "http://localhost:4318/v1/traces",
            header_env={"authorization": "OTEL_AUTHORIZATION"},
            max_queue_size=4096,
            max_export_batch_size=256,
            scheduled_delay_millis=750,
            completed_span_context_ttl_millis=60_000,
        ).to_dict() == {
            "type": "gen_ai",
            "endpoint": "http://localhost:4318/v1/traces",
            "mark_projection": "inherit",
            "mark_exclude_names": ["llm.chunk"],
            "attribute_mappings": [],
            "promote_metadata_prefixes": [],
            "promote_resource_metadata_prefixes": [],
            "transport": "http_binary",
            "service_name": "unknown_service",
            "instrumentation_scope": "opentelemetry",
            "timeout_millis": 3000,
            "max_queue_size": 4096,
            "max_export_batch_size": 256,
            "scheduled_delay_millis": 750,
            "completed_span_context_ttl_millis": 60_000,
            "headers": {},
            "header_env": {"authorization": "OTEL_AUTHORIZATION"},
            "resource_attributes": {},
        }
        endpoint = OpenTelemetryEndpointConfig(
            "gen_ai",
            "http://localhost:4318/v1/traces",
            promote_metadata_prefixes=["nv."],
        )
        assert endpoint.to_dict()["promote_metadata_prefixes"] == ["nv."]
        endpoint.promote_resource_metadata_prefixes = ["deployment."]
        assert endpoint.to_dict()["promote_resource_metadata_prefixes"] == ["deployment."]

        wrapped = ComponentSpec(ObservabilityConfig(atof=AtofConfig())).to_dict()
        assert wrapped["kind"] == OBSERVABILITY_PLUGIN_KIND
        assert wrapped["enabled"] is True
        wrapped_config = wrapped["config"]
        assert isinstance(wrapped_config, dict)
        assert wrapped_config["version"] == 4
        assert wrapped_config["enable_full_payloads"] is False

        full_payload_config = ObservabilityConfig(enable_full_payloads=True).to_dict()
        assert full_payload_config["enable_full_payloads"] is True

    def test_opentelemetry_signal_section_defaults_and_serialization(self) -> None:
        endpoint = OpenTelemetrySignalEndpointConfig(
            "https://collector.example/custom/logs",
            headers={"x-static": "value"},
            header_env={"authorization": "OTEL_AUTHORIZATION"},
            resource_attributes={"nv.project": "observability-dev"},
            service_namespace="telemetry",
            service_version="1.0",
        )
        assert endpoint.to_dict() == {
            "endpoint": "https://collector.example/custom/logs",
            "transport": "http_binary",
            "headers": {"x-static": "value"},
            "header_env": {"authorization": "OTEL_AUTHORIZATION"},
            "resource_attributes": {"nv.project": "observability-dev"},
            "service_name": "unknown_service",
            "service_namespace": "telemetry",
            "service_version": "1.0",
            "instrumentation_scope": "opentelemetry",
            "timeout_millis": 3000,
        }

        logs = OpenTelemetryLogSectionConfig(enabled=True, endpoints=[endpoint])
        metrics = OpenTelemetryMetricSectionConfig(enabled=True, endpoints=[endpoint])
        assert OpenTelemetryLogSectionConfig().to_dict() == {
            "enabled": False,
            "minimum_severity": "info",
            "max_queue_size": 2048,
            "max_export_batch_size": 512,
            "scheduled_delay_millis": 1000,
        }
        assert OpenTelemetryMetricSectionConfig().to_dict() == {
            "enabled": False,
            "export_interval_millis": 60000,
            "temporality": "cumulative",
            "max_instruments": 256,
            "cardinality_limit": 2000,
        }

        section = OpenTelemetrySectionConfig(enabled=True, logs=logs, metrics=metrics).to_dict()
        assert section["logs"] == logs.to_dict()
        assert section["metrics"] == metrics.to_dict()
        assert typing.cast(dict[str, object], section["metrics"])["endpoints"] == [endpoint.to_dict()]

    @pytest.mark.parametrize("signal", ["logs", "metrics"])
    @pytest.mark.parametrize("prefixes", [None, [], ["deployment.", "nv.client."], ["nv.*"]])
    def test_signal_resource_promotion_serialization_and_validation(
        self, signal: str, prefixes: list[str] | None
    ) -> None:
        endpoint = OpenTelemetrySignalEndpointConfig(f"http://localhost:4318/v1/{signal}")
        if prefixes is not None:
            endpoint = OpenTelemetrySignalEndpointConfig(endpoint.endpoint, promote_resource_metadata_prefixes=prefixes)
        serialized = endpoint.to_dict()
        if prefixes:
            assert serialized["promote_resource_metadata_prefixes"] == prefixes
        else:
            assert "promote_resource_metadata_prefixes" not in serialized

        section = OpenTelemetrySectionConfig(enabled=True)
        if signal == "logs":
            section.logs = OpenTelemetryLogSectionConfig(enabled=True, endpoints=[endpoint])
        else:
            section.metrics = OpenTelemetryMetricSectionConfig(enabled=True, endpoints=[endpoint])
        report = validate_plugin_config(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        ObservabilityConfig(opentelemetry=section, policy=ConfigPolicy(unknown_field="error"))
                    )
                ]
            )
        )
        if prefixes == ["nv.*"]:
            assert any(
                diagnostic.get("component") == f"opentelemetry.{signal}"
                and diagnostic.get("field") == "endpoints[0].promote_resource_metadata_prefixes"
                for diagnostic in report["diagnostics"]
            )
        else:
            assert report["diagnostics"] == []

    def test_validation_rejects_bad_values(self) -> None:
        report = validate_plugin_config(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        {
                            "version": 4,
                            "atof": {"sinks": [{"type": "file", "mode": "bad"}]},
                            "atif": {"filename_template": "missing-placeholder"},
                        }
                    )
                ]
            )
        )
        fields = {diag.get("field") for diag in report["diagnostics"]}
        assert {"sinks[0].mode", "filename_template"} <= fields

    def test_list_kinds_includes_builtin_observability(self) -> None:
        assert OBSERVABILITY_PLUGIN_KIND in plugin.list_kinds()

    def test_s3_storage_config_serializes_credential_fields(self) -> None:
        storage = S3StorageConfig(
            bucket="my-bucket",
            key_prefix="prefix/",
            access_key_id="AKIAEXAMPLE",
            secret_access_key_var="MY_SECRET",
            session_token_var="MY_TOKEN",
            region="us-west-2",
            endpoint_url="https://s3.example.com",
            allow_http=False,
        )
        assert storage.to_dict() == {
            "type": "s3",
            "bucket": "my-bucket",
            "key_prefix": "prefix/",
            "access_key_id": "AKIAEXAMPLE",
            "secret_access_key_var": "MY_SECRET",
            "session_token_var": "MY_TOKEN",
            "region": "us-west-2",
            "endpoint_url": "https://s3.example.com",
            "allow_http": False,
        }
        atif = AtifConfig(enabled=True, storage=[storage])
        assert atif.to_dict()["storage"] == [storage.to_dict()]

    def test_atof_sink_config_serializes_streaming_fields(self) -> None:
        sink = AtofStreamSinkConfig(
            url="http://localhost:8080/events",
            name="switchyard",
            transport="http_post",
            headers={"X-Test": "yes"},
            header_env={"authorization": "NEMO_RELAY_ATOF_AUTH"},
            timeout_millis=1000,
            field_name_policy="replace_dots",
        )
        assert sink.to_dict() == {
            "type": "stream",
            "name": "switchyard",
            "url": "http://localhost:8080/events",
            "transport": "http_post",
            "headers": {"X-Test": "yes"},
            "header_env": {"authorization": "NEMO_RELAY_ATOF_AUTH"},
            "timeout_millis": 1000,
            "field_name_policy": "replace_dots",
        }
        assert AtofConfig(sinks=[sink]).to_dict()["sinks"] == [sink.to_dict()]
        assert "name" not in AtofStreamSinkConfig(url="http://localhost:8080/events").to_dict()

    def test_atof_endpoint_alias_preserves_positional_transport(self) -> None:
        endpoint = AtofEndpointConfig("http://localhost:8080/events", "websocket")
        assert endpoint.transport == "websocket"
        assert endpoint.name is None

    async def test_atof_stream_sink_snapshots_header_env(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        variable = "NEMO_RELAY_TEST_ATOF_HEADER_ENV"
        credential = "Bearer relay-499"
        monkeypatch.setenv(variable, credential)

        with _AtofCapture() as capture:
            config = ObservabilityConfig(
                atof=AtofConfig(
                    enabled=True,
                    sinks=[
                        AtofStreamSinkConfig(
                            url=capture.url,
                            transport="http_post",
                            header_env={"authorization": variable},
                        )
                    ],
                )
            )
            activation = await plugin.initialize(plugin.PluginConfig(components=[ComponentSpec(config)]))
            report = activation.report["config"]
            assert report["diagnostics"] == []
            monkeypatch.delenv(variable)

            try:
                with scope.scope("python-header-env-agent", ScopeType.Agent) as handle:
                    scope.event("python-header-env-mark", handle=handle, data={"step": 1})
            finally:
                await activation.close()
            requests = capture.wait_for_requests(3)

        assert len(requests) == 3
        payload = b"".join(body for _, body in requests).decode()
        assert '"scope_category":"start"' in payload
        assert '"name":"python-header-env-mark"' in payload
        assert '"scope_category":"end"' in payload
        for headers, _ in requests:
            authorization = next(value for name, value in headers.items() if name.lower() == "authorization")
            assert authorization == credential
        assert credential not in json.dumps(report)
        assert credential not in caplog.text

    def test_http_storage_config_serializes_headers(self) -> None:
        s3 = S3StorageConfig(bucket="archive")
        http = HttpStorageConfig(
            endpoint="https://example.com/atif",
            headers={"x-static": "value"},
            header_env={"authorization": "NEMO_RELAY_ATIF_HTTP_AUTH"},
            timeout_millis=1500,
        )
        assert http.to_dict() == {
            "type": "http",
            "endpoint": "https://example.com/atif",
            "headers": {"x-static": "value"},
            "header_env": {"authorization": "NEMO_RELAY_ATIF_HTTP_AUTH"},
            "timeout_millis": 1500,
        }
        atif = AtifConfig(enabled=True, storage=[s3, http])
        assert atif.to_dict()["storage"] == [s3.to_dict(), http.to_dict()]

    @pytest.mark.parametrize("texts", [[], ["inspect this image from the medical chatbot"], ["first", "second"]])
    @pytest.mark.parametrize("use_codec", [True, False])
    async def test_atif_inline_image_preserves_model_request(
        self, tmp_path: Path, texts: list[str], use_codec: bool
    ) -> None:
        png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        body = {
            "model": "claude-sonnet-test",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *({"type": "text", "text": text} for text in texts),
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": png_base64,
                            },
                        },
                    ],
                }
            ],
        }
        request = LLMRequest({}, body)
        received = []

        async def model_callback(model_request: LLMRequest):
            received.append(model_request.content)
            return {
                "id": "msg-image",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-test",
                "content": [{"type": "text", "text": "The image is available."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }

        config = plugin.PluginConfig(
            components=[
                ComponentSpec(
                    ObservabilityConfig(
                        atif=AtifConfig(enabled=True, output_directory=str(tmp_path)),
                    )
                )
            ]
        )
        codec = AnthropicMessagesCodec() if use_codec else None
        activation = await plugin.initialize(config)
        try:
            with scope.scope("image-turn", ScopeType.Agent) as handle:
                await llm.execute(
                    "vision-model",
                    request,
                    model_callback,
                    handle=handle,
                    codec=codec,
                    response_codec=codec,
                )
            await subscribers.flush_async()
        finally:
            await activation.close()

        assert received == [body]
        assert request.content == body
        [trajectory_file] = list(tmp_path.glob("nemo-relay-atif-*.json"))
        trajectory = json.loads(trajectory_file.read_text())
        user = next(step for step in trajectory["steps"] if step["source"] == "user")
        assert user["message"] == "\n".join(texts)
        assert png_base64 not in user["message"]
        assert user["extra"]["llm_request"] == body

    @pytest.mark.parametrize("use_context_manager", [True, False])
    async def test_atof_and_atif_file_outputs(self, tmp_path: Path, use_context_manager: bool) -> None:
        config = ObservabilityConfig(
            atof=AtofConfig(
                enabled=True,
                sinks=[
                    AtofFileSinkConfig(
                        output_directory=str(tmp_path),
                        filename="events.jsonl",
                        mode="overwrite",
                    )
                ],
            ),
            atif=AtifConfig(
                enabled=True,
                agent_name="python-agent",
                agent_version="1.2.3",
                model_name="python-model",
                tool_definitions=[{"name": "search"}],
                extra={"binding": "python"},
                output_directory=str(tmp_path),
                filename_template="trajectory-{session_id}.json",
            ),
        )

        def _inner():
            with scope.scope("python-observability-agent", ScopeType.Agent) as handle:
                scope.event("python-mark", handle=handle, data={"step": 1})

            return handle

        plugin_config = plugin.PluginConfig(components=[ComponentSpec(config)])
        if use_context_manager:
            activation = await plugin.initialize(plugin_config)
            async with activation:
                handle = _inner()
        else:
            activation = await plugin.initialize(plugin_config)
            try:
                handle = _inner()
            finally:
                await activation.close()

        lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[1])["name"] == "python-mark"

        trajectory = json.loads((tmp_path / f"trajectory-{handle.uuid}.json").read_text())
        assert trajectory["agent"]["name"] == "python-agent"
        assert trajectory["agent"]["version"] == "1.2.3"
        assert trajectory["agent"]["model_name"] == "python-model"
        assert trajectory["agent"]["tool_definitions"][0]["name"] == "search"
        assert trajectory["agent"]["extra"]["binding"] == "python"
        assert "python-observability-agent" in json.dumps(trajectory["extra"])

    async def test_atof_stream_sink_delivery_failures_reach_the_plugin_report(self, tmp_path: Path) -> None:
        with _AtofCapture(response_status=503) as rejecting:
            async with await plugin.initialize(
                plugin.PluginConfig(
                    components=[
                        ComponentSpec(
                            ObservabilityConfig(
                                atof=AtofConfig(
                                    enabled=True,
                                    sinks=[
                                        AtofFileSinkConfig(output_directory=str(tmp_path), filename="events.jsonl"),
                                        AtofStreamSinkConfig(url=f"{rejecting.url}/events", timeout_millis=5000),
                                    ],
                                )
                            )
                        )
                    ]
                )
            ) as activation:
                scope.event("python-rejected-mark")
                await subscribers.flush_async()
            diagnostics = activation.report["config"]["runtime_diagnostics"]
            assert [entry["code"] for entry in diagnostics] == ["atof.stream_delivery_failed"]
            assert diagnostics[0]["component"] == "observability"
            assert diagnostics[0].get("field") == "atof.sinks[1].url"
            assert diagnostics[0]["count"] == 1
            assert diagnostics[0]["message"] == (
                f"ATOF http_post stream to {rejecting.url} returned status 503 Service Unavailable"
            )

    @pytest.mark.parametrize(
        ("field_name_policy", "expected_data"),
        [
            (
                "preserve",
                {
                    "service.name": "relay",
                    "nested": {"deployment.region": "us-east"},
                },
            ),
            (
                "replace_dots",
                {
                    "service_name": "relay",
                    "nested": {"deployment_region": "us-east"},
                },
            ),
        ],
    )
    async def test_atof_stream_sink_dotted_fields_deliver_during_plugin_clear(
        self,
        tmp_path: Path,
        field_name_policy: typing.Literal["preserve", "replace_dots"],
        expected_data: dict[str, object],
    ) -> None:
        received: list[bytes] = []
        request_received = threading.Event()
        allow_response = threading.Event()
        teardown_started = threading.Event()

        class CaptureHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers["Content-Length"])
                received.append(self.rfile.read(content_length))
                request_received.set()
                assert allow_response.wait(timeout=5)
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: typing.Any) -> None:
                return None

        server = HTTPServer(("127.0.0.1", 0), CaptureHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        cleared = False
        activation: plugin.PluginHostActivation | None = None
        try:
            activation = await plugin.initialize(
                plugin.PluginConfig(
                    components=[
                        ComponentSpec(
                            ObservabilityConfig(
                                atof=AtofConfig(
                                    enabled=True,
                                    sinks=[
                                        AtofFileSinkConfig(
                                            output_directory=str(tmp_path),
                                            filename="events.jsonl",
                                            mode="overwrite",
                                        ),
                                        AtofStreamSinkConfig(
                                            url=f"http://127.0.0.1:{server.server_port}/events",
                                            timeout_millis=5000,
                                            field_name_policy=field_name_policy,
                                        ),
                                    ],
                                )
                            )
                        )
                    ]
                )
            )
            with scope.scope("python-stream-agent", ScopeType.Agent) as handle:
                scope.event(
                    "python-dotted-mark",
                    handle=handle,
                    data={
                        "service.name": "relay",
                        "nested": {"deployment.region": "us-east"},
                    },
                )

            def release_response() -> None:
                teardown_started.wait(timeout=5)
                request_received.wait(timeout=5)
                allow_response.set()

            response_thread = threading.Thread(target=release_response, daemon=True)
            response_thread.start()
            teardown_started.set()
            await activation.close()
            cleared = True
            assert "runtime_diagnostics" not in activation.report["config"]
            response_thread.join(timeout=2)
            assert not response_thread.is_alive()

            file_events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
            stream_events = [json.loads(body) for body in received]
            assert len(file_events) == len(stream_events) == 3
            file_mark = next(event for event in file_events if event["name"] == "python-dotted-mark")
            stream_mark = next(event for event in stream_events if event["name"] == "python-dotted-mark")
            assert file_mark["data"] == {
                "service.name": "relay",
                "nested": {"deployment.region": "us-east"},
            }
            assert stream_mark["data"] == expected_data
        finally:
            allow_response.set()
            if not cleared and activation is not None:
                await activation.close()
            server.shutdown()
            server_thread.join(timeout=5)
            server.server_close()

    async def test_atif_flushes_open_agent_on_clear(self, tmp_path) -> None:
        activation = await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(ObservabilityConfig(atif=AtifConfig(enabled=True, output_directory=str(tmp_path))))
                ]
            )
        )
        handle = scope.push("python-open-agent", ScopeType.Agent)
        try:
            await activation.close()
            assert (tmp_path / f"nemo-relay-atif-{handle.uuid}.json").exists()
        finally:
            scope.pop(handle)

    async def test_atif_can_use_propagation_root_as_run_session_id(self, tmp_path) -> None:
        request_id = "018f47a4-3af7-7d94-8e61-9f0f89b5d312"
        stack = create_scope_stack_from_propagation(PropagationContext(request_id, request_id))
        activation = await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        ObservabilityConfig(
                            atif=AtifConfig(
                                enabled=True,
                                output_directory=str(tmp_path),
                                filename_template="trajectory-{session_id}.json",
                            )
                        )
                    )
                ]
            )
        )
        try:
            with use_scope_stack(stack):
                with scope.scope("python-propagated-agent", ScopeType.Agent) as handle:
                    scope.event("python-mark", handle=handle, data={"step": 1})
        finally:
            await activation.close()

        trajectory_path = tmp_path / f"trajectory-{handle.uuid}.json"
        trajectory = json.loads(trajectory_path.read_text())
        assert trajectory["session_id"] == request_id
        assert trajectory["trajectory_id"] == handle.uuid
        assert trajectory["extra"]["nemo_relay"]["session_instance_id"] == request_id

    async def test_atif_non_string_metadata_is_reported_on_failed_close(self, tmp_path) -> None:
        activation = await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        ObservabilityConfig(
                            atif=AtifConfig(
                                enabled=True,
                                output_directory=str(tmp_path),
                                filename_template="{metadata.atif_prefix:-unassigned}/trajectory-{session_id}.json",
                            )
                        )
                    )
                ]
            )
        )

        try:
            with scope.scope("python-invalid-metadata-agent", ScopeType.Agent, metadata={"atif_prefix": 123}):
                pass
            await subscribers.flush_async()

            report = activation.report["config"]
            assert any(
                diagnostic["code"] == "atif.destination_render_failed" and "non-string" in diagnostic["message"]
                for diagnostic in report["runtime_diagnostics"]
            )
            assert not (tmp_path / "unassigned").exists()

            with pytest.raises(RuntimeError, match=r"atif\.destination_render_failed") as teardown:
                await activation.close()
            assert "could not be removed" not in str(teardown.value)
            retained = activation.report["config"]
            assert any(
                diagnostic["code"] == "atif.destination_render_failed" for diagnostic in retained["runtime_diagnostics"]
            )
        finally:
            with pytest.raises(RuntimeError, match=r"atif\.destination_render_failed"):
                await activation.close()
        assert not activation.is_active

    async def test_atif_splits_multiple_top_level_agent_scopes(self, tmp_path) -> None:
        activation = await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        ObservabilityConfig(
                            atif=AtifConfig(
                                enabled=True,
                                output_directory=str(tmp_path),
                                filename_template="trajectory-{session_id}.json",
                            )
                        )
                    )
                ]
            )
        )
        try:
            with scope.scope("python-first-agent", ScopeType.Agent) as first:
                scope.event("python-first-mark", handle=first, data={"agent": "first"})
                with scope.scope("python-nested-agent", ScopeType.Agent) as nested:
                    scope.event("python-nested-mark", handle=nested, data={"agent": "nested"})

            with scope.scope("python-second-agent", ScopeType.Agent) as second:
                scope.event("python-second-mark", handle=second, data={"agent": "second"})
        finally:
            await activation.close()

        files = sorted(tmp_path.glob("trajectory-*.json"))
        assert len(files) == 2

        first_trajectory = json.loads((tmp_path / f"trajectory-{first.uuid}.json").read_text())
        second_trajectory = json.loads((tmp_path / f"trajectory-{second.uuid}.json").read_text())
        first_payload = json.dumps(first_trajectory["extra"])
        second_payload = json.dumps(second_trajectory["extra"])

        assert "python-first-agent" in first_payload
        assert "python-nested-agent" in first_payload
        assert "python-second-agent" not in first_payload
        assert "python-second-agent" in second_payload
        assert "python-first-agent" not in second_payload
        assert "python-nested-agent" not in second_payload


def test_otlp_tls_helpers_and_direct_configs_share_schema():
    from nemo_relay import (
        OpenTelemetryConfig,
        OpenTelemetryLogConfig,
        OpenTelemetryLogSubscriber,
        OpenTelemetryMetricConfig,
        OpenTelemetryMetricSubscriber,
        OpenTelemetrySubscriber,
        OpenTelemetryTlsConfig,
    )

    tls = OpenTelemetryTlsConfig(certificate="ca.pem", client_certificate="client.pem", private_key="key.pem")
    expected = {"certificate": "ca.pem", "client_certificate": "client.pem", "private_key": "key.pem"}
    trace = OpenTelemetryEndpointConfig(type="full", endpoint="https://collector.example", tls=tls)
    signal = OpenTelemetrySignalEndpointConfig(endpoint="https://collector.example", tls=tls)
    assert trace.to_dict()["tls"] == expected
    assert signal.to_dict()["tls"] == expected
    assert "tls" not in OpenTelemetrySignalEndpointConfig(endpoint="https://collector.example").to_dict()
    for config, subscriber in [
        (OpenTelemetryConfig("full", "http://127.0.0.1:4318"), OpenTelemetrySubscriber),
        (OpenTelemetryLogConfig("http://127.0.0.1:4318"), OpenTelemetryLogSubscriber),
        (OpenTelemetryMetricConfig("http://127.0.0.1:4318"), OpenTelemetryMetricSubscriber),
    ]:
        assert config.tls is None
        config.tls = tls
        assert config.tls.private_key == "key.pem"
        with pytest.raises(Exception, match="TLS settings require an HTTPS endpoint"):
            subscriber(config)
        config.tls = OpenTelemetryTlsConfig(client_certificate="client.pem")
        with pytest.raises(Exception, match="must be supplied together"):
            subscriber(config)


class TestPlaintextPreflight:
    @pytest.fixture(autouse=True)
    def isolate_plaintext_preflight_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for variable_name in list(os.environ):
            if variable_name.startswith("OTEL_"):
                monkeypatch.delenv(variable_name)
        monkeypatch.setenv("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "1")

    @staticmethod
    def configuration(
        signal: str,
        endpoints: list[str],
        *,
        headers: dict[str, str] | None = None,
        insecure_http_hosts: list[str] | None = None,
    ) -> plugin.PluginConfig:
        observability = ObservabilityConfig(version=4, insecure_http_hosts=insecure_http_hosts)
        configured_headers = headers if headers is not None else {"authorization": "synthetic-header-secret"}
        if signal == "atof":
            observability.atof = AtofConfig(
                enabled=True,
                sinks=[AtofStreamSinkConfig(url=endpoint, headers=configured_headers) for endpoint in endpoints],
            )
        elif signal == "atif":
            observability.atif = AtifConfig(
                enabled=True,
                storage=[HttpStorageConfig(endpoint=endpoint, headers=configured_headers) for endpoint in endpoints],
            )
        else:
            section = OpenTelemetrySectionConfig(enabled=True)
            if signal == "traces":
                section.endpoints = [
                    OpenTelemetryEndpointConfig("full", endpoint, headers=configured_headers) for endpoint in endpoints
                ]
            elif signal == "logs":
                section.logs = OpenTelemetryLogSectionConfig(
                    enabled=True,
                    endpoints=[
                        OpenTelemetrySignalEndpointConfig(endpoint, headers=configured_headers)
                        for endpoint in endpoints
                    ],
                )
            else:
                assert signal == "metrics"
                section.metrics = OpenTelemetryMetricSectionConfig(
                    enabled=True,
                    endpoints=[
                        OpenTelemetrySignalEndpointConfig(endpoint, headers=configured_headers)
                        for endpoint in endpoints
                    ],
                )
            observability.opentelemetry = section
        return plugin.PluginConfig(components=[ComponentSpec(observability)])

    @pytest.mark.parametrize("signal", ["traces", "logs", "metrics", "atof", "atif"])
    async def test_plaintext_preflight_and_activation_reject_reported_cases(self, signal: str) -> None:
        configuration = self.configuration(
            signal,
            ["http://collector.example:4318/telemetry"],
            headers={} if signal == "traces" else None,
        )
        for validate in (plugin.validate, plugin.validate_exact):
            diagnostics = validate(configuration)["config"]["diagnostics"]
            assert len(diagnostics) == 1
            assert diagnostics[0]["level"] == "error"
            assert diagnostics[0]["code"] == "observability.unsafe_plaintext_destination"
        expected_exception = RuntimeError if signal in ("atof", "atif") else ValueError
        with pytest.raises(expected_exception, match="insecure_http_hosts"):
            await plugin.initialize(configuration)

    @pytest.mark.parametrize("signal", ["traces", "logs", "metrics", "atof", "atif"])
    def test_plaintext_preflight_identifies_endpoint_and_redacts_sensitive_values(self, signal: str) -> None:
        configuration = self.configuration(
            signal,
            [
                "https://collector.example:4318/safe",
                "http://synthetic-user:synthetic-password@collector.example:4318/private-path?token=synthetic-query",
            ],
        )
        component = signal if signal in ("atof", "atif") else f"opentelemetry.{signal}"
        field = {"atof": "sinks[1].url", "atif": "storage[1].endpoint"}.get(signal, "endpoints[1].endpoint")
        for validate in (plugin.validate, plugin.validate_exact):
            diagnostics = validate(configuration)["config"]["diagnostics"]
            assert len(diagnostics) == 1
            diagnostic = diagnostics[0]
            assert diagnostic["component"] == component
            assert diagnostic["field"] == field
            assert "collector.example:4318" in diagnostic["message"]
            serialized_diagnostics = json.dumps(diagnostics)
            for sensitive_value in (
                "synthetic-header-secret",
                "synthetic-user",
                "synthetic-password",
                "private-path",
                "synthetic-query",
            ):
                assert sensitive_value not in serialized_diagnostics

    @pytest.mark.parametrize("signal", ["traces", "logs", "metrics", "atof", "atif"])
    @pytest.mark.parametrize("permitted_transport", ["https", "loopback", "allowlist"])
    def test_plaintext_preflight_permits_existing_transport_controls(
        self, signal: str, permitted_transport: str
    ) -> None:
        endpoint = {
            "https": "https://collector.example:4318/telemetry",
            "loopback": "http://127.0.0.1:4318/telemetry",
            "allowlist": "http://collector.example:4318/telemetry",
        }[permitted_transport]
        configuration = self.configuration(
            signal,
            [endpoint],
            insecure_http_hosts=["collector.example"] if permitted_transport == "allowlist" else None,
        )
        for validate in (plugin.validate, plugin.validate_exact):
            assert validate(configuration)["config"]["diagnostics"] == []

    @pytest.mark.parametrize("signal", ["logs", "metrics", "atof", "atif"])
    def test_plaintext_preflight_permits_headerless_http(self, signal: str) -> None:
        configuration = self.configuration(signal, ["http://collector.example:4318/telemetry"], headers={})
        for validate in (plugin.validate, plugin.validate_exact):
            assert validate(configuration)["config"]["diagnostics"] == []

    async def test_plaintext_preflight_error_preserves_partial_trace_activation(self) -> None:
        configuration = self.configuration(
            "traces", ["http://127.0.0.1:4318/v1/traces", "http://collector.example:4318/v1/traces"], headers={}
        )
        diagnostics = plugin.validate(configuration)["config"]["diagnostics"]
        assert len(diagnostics) == 1
        assert diagnostics[0]["level"] == "error"
        assert diagnostics[0]["field"] == "endpoints[1].endpoint"
        activation = await plugin.initialize(configuration)
        try:
            assert activation.is_active
        finally:
            await activation.close()
