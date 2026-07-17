# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in observability plugin config helpers."""

from __future__ import annotations

import http.server
import json
import threading
import time
import typing

import pytest

from nemo_relay import ScopeType, plugin, scope
from nemo_relay.observability import (
    OBSERVABILITY_PLUGIN_KIND,
    AtifConfig,
    AtofConfig,
    AtofEndpointConfig,
    AtofFileSinkConfig,
    AtofStreamSinkConfig,
    ComponentSpec,
    HttpStorageConfig,
    ObservabilityConfig,
    OtlpConfig,
    S3StorageConfig,
)

if typing.TYPE_CHECKING:
    from pathlib import Path


class _AtofCaptureServer(http.server.ThreadingHTTPServer):
    requests: list[tuple[dict[str, str], bytes]]
    request_event: threading.Event


class _AtofCaptureHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("content-length", "0"))
        server = typing.cast(_AtofCaptureServer, self.server)
        server.requests.append((dict(self.headers.items()), self.rfile.read(content_length)))
        server.request_event.set()
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        return


class _AtofCapture:
    server: "_AtofCaptureServer"
    thread: threading.Thread

    def __enter__(self) -> _AtofCapture:
        self.server = _AtofCaptureServer(("127.0.0.1", 0), _AtofCaptureHandler)
        self.server.requests = []
        self.server.request_event = threading.Event()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

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
    def test_defaults_and_component_wrapper(self):
        assert AtofConfig().to_dict() == {"enabled": False}
        assert AtifConfig().to_dict() == {
            "enabled": False,
            "agent_name": "NeMo Relay",
            "model_name": "unknown",
            "filename_template": "nemo-relay-atif-{session_id}.json",
        }
        assert OtlpConfig().to_dict() == {
            "enabled": False,
            "mark_projection": "inherit",
            "mark_exclude_names": ["llm.chunk"],
            "attribute_mappings": [],
            "transport": "http_binary",
            "headers": {},
            "resource_attributes": {},
            "service_name": "nemo-relay",
            "timeout_millis": 3000,
        }
        assert OtlpConfig(mark_projection="tool").to_dict()["mark_projection"] == "tool"

        wrapped = ComponentSpec(ObservabilityConfig(atof=AtofConfig())).to_dict()
        assert wrapped["kind"] == OBSERVABILITY_PLUGIN_KIND
        assert wrapped["enabled"] is True
        wrapped_config = wrapped["config"]
        assert isinstance(wrapped_config, dict)
        assert wrapped_config["version"] == 2

    def test_validation_rejects_bad_values(self):
        report = plugin.validate(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        {
                            "version": 2,
                            "atof": {"sinks": [{"type": "file", "mode": "bad"}]},
                            "atif": {"filename_template": "missing-placeholder"},
                        }
                    )
                ]
            )
        )
        fields = {diag.get("field") for diag in report["diagnostics"]}
        assert {"sinks[0].mode", "filename_template"} <= fields

    def test_list_kinds_includes_builtin_observability(self):
        assert OBSERVABILITY_PLUGIN_KIND in plugin.list_kinds()

    def test_s3_storage_config_serializes_credential_fields(self):
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

    def test_atof_sink_config_serializes_streaming_fields(self):
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

    def test_atof_endpoint_alias_preserves_positional_transport(self):
        endpoint = AtofEndpointConfig("http://localhost:8080/events", "websocket")
        assert endpoint.transport == "websocket"
        assert endpoint.name is None

    async def test_atof_stream_sink_snapshots_header_env(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
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
            report = await plugin.initialize(plugin.PluginConfig(components=[ComponentSpec(config)]))
            assert report["diagnostics"] == []
            monkeypatch.delenv(variable)

            try:
                with scope.scope("python-header-env-agent", ScopeType.Agent) as handle:
                    scope.event("python-header-env-mark", handle=handle, data={"step": 1})
            finally:
                plugin.clear()
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

    def test_http_storage_config_serializes_headers(self):
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

    @pytest.mark.parametrize("use_context_manager", [True, False])
    async def test_atof_and_atif_file_outputs(self, tmp_path: Path, use_context_manager: bool):
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
            async with plugin.plugin(plugin_config):
                handle = _inner()
        else:
            await plugin.initialize(plugin_config)
            try:
                handle = _inner()
            finally:
                plugin.clear()

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

    async def test_atif_flushes_open_agent_on_clear(self, tmp_path):
        await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(ObservabilityConfig(atif=AtifConfig(enabled=True, output_directory=str(tmp_path))))
                ]
            )
        )
        handle = scope.push("python-open-agent", ScopeType.Agent)
        try:
            plugin.clear()
            assert (tmp_path / f"nemo-relay-atif-{handle.uuid}.json").exists()
        finally:
            scope.pop(handle)

    async def test_atif_splits_multiple_top_level_agent_scopes(self, tmp_path):
        await plugin.initialize(
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
            plugin.clear()

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
