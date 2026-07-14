# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in observability plugin config helpers."""

from __future__ import annotations

import json
import typing

import pytest

from nemo_relay import ScopeType, plugin, scope
from nemo_relay.observability import (
    OBSERVABILITY_PLUGIN_KIND,
    AtifConfig,
    AtofConfig,
    AtofEndpointConfig,
    ComponentSpec,
    HttpStorageConfig,
    ObservabilityConfig,
    OtlpConfig,
    S3StorageConfig,
)

if typing.TYPE_CHECKING:
    from pathlib import Path


class TestObservabilityConfigHelpers:
    def test_defaults_and_component_wrapper(self):
        assert AtofConfig().to_dict() == {"enabled": False, "mode": "append"}
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
        assert wrapped_config["version"] == 1

    def test_validation_rejects_bad_values(self):
        report = plugin.validate(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        {
                            "version": 1,
                            "atof": {"mode": "bad"},
                            "atif": {"filename_template": "missing-placeholder"},
                        }
                    )
                ]
            )
        )
        fields = {diag.get("field") for diag in report["diagnostics"]}
        assert {"mode", "filename_template"} <= fields

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

    def test_atof_endpoint_config_serializes_streaming_fields(self):
        endpoint = AtofEndpointConfig(
            url="http://localhost:8080/events",
            name="switchyard",
            transport="http_post",
            headers={"X-Test": "yes"},
            timeout_millis=1000,
            field_name_policy="replace_dots",
        )
        assert endpoint.to_dict() == {
            "name": "switchyard",
            "url": "http://localhost:8080/events",
            "transport": "http_post",
            "headers": {"X-Test": "yes"},
            "timeout_millis": 1000,
            "field_name_policy": "replace_dots",
        }
        assert AtofConfig(endpoints=[endpoint]).to_dict()["endpoints"] == [endpoint.to_dict()]
        assert "name" not in AtofEndpointConfig(url="http://localhost:8080/events").to_dict()

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
                output_directory=str(tmp_path),
                filename="events.jsonl",
                mode="overwrite",
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
