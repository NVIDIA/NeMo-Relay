# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic configured-service and inspection contracts, isolated from other runtimes."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from nemo_relay._diagnostics_query_contract import (
    MAX_QUERY_BYTES,
    MAX_QUERY_DEPTH,
    MAX_QUERY_FIELD_NAME_BYTES,
    MAX_QUERY_NODES,
    MAX_QUERY_OBJECT_FIELDS,
    NUMERIC_QUERY_FIELDS,
)
from nemo_relay.diagnostics import DiagnosticsError, DiagnosticsReader


def test_python_query_adapter_matches_shared_contract_fixture():
    fixture_path = Path(__file__).resolve().parents[2] / "crates/core/tests/fixtures/diagnostics_query_contract.json"
    fixture = json.loads(fixture_path.read_text())
    assert MAX_QUERY_BYTES == fixture["max_query_bytes"]
    assert MAX_QUERY_NODES == fixture["adapter_limits"]["max_query_nodes"]
    assert MAX_QUERY_DEPTH == fixture["adapter_limits"]["max_query_depth"]
    assert MAX_QUERY_OBJECT_FIELDS == fixture["adapter_limits"]["max_query_object_fields"]
    assert MAX_QUERY_FIELD_NAME_BYTES == fixture["adapter_limits"]["max_query_field_name_bytes"]
    assert NUMERIC_QUERY_FIELDS == frozenset(fixture["numeric_query_fields"])


@pytest.mark.parametrize("method", ["snapshot", "snapshot_json"])
@pytest.mark.parametrize("query", [{"\ud800": "value"}, {"component_key": "\udfff"}])
def test_reader_rejects_malformed_unicode_queries(method, query):
    # Malformed input must be rejected before the native reader is called.
    native = Mock()
    reader = DiagnosticsReader(native)
    with pytest.raises(DiagnosticsError) as error:
        getattr(reader, method)(query)
    assert error.value.reason == "invalid_query"
    assert native.mock_calls == []


class _EncodeForbidden(str):
    def encode(self, *args, **kwargs):
        pytest.fail("query input was encoded before admission")


@pytest.mark.parametrize("method", ["snapshot", "snapshot_json"])
@pytest.mark.parametrize(
    "query",
    [
        {_EncodeForbidden("k" * (MAX_QUERY_FIELD_NAME_BYTES + 1)): ""},
        {"component_key": _EncodeForbidden("v" * (MAX_QUERY_BYTES + 1))},
        {"component_key": _EncodeForbidden("v" * (MAX_QUERY_BYTES - len("component_key") + 1))},
        {"é" * (MAX_QUERY_FIELD_NAME_BYTES // 2 + 1): _EncodeForbidden("")},
        {"component_key": "é" * ((MAX_QUERY_BYTES - len("component_key")) // 2 + 1)},
        {"component_key": "\x00" * (MAX_QUERY_BYTES // 6)},
    ],
    ids=["key_characters", "value_characters", "remaining_bytes", "key_utf8", "value_utf8", "json_escaping"],
)
def test_reader_bounds_query_strings_before_native_calls(method, query):
    native = Mock()
    reader = DiagnosticsReader(native)
    with pytest.raises(DiagnosticsError) as error:
        getattr(reader, method)(query)
    assert error.value.reason == "invalid_query"
    assert native.mock_calls == []


@pytest.mark.parametrize("method", ["snapshot", "snapshot_json"])
def test_reader_preserves_multibyte_query_at_encoded_size_limit(method):
    key = "component_key"
    overhead = len(json.dumps({key: ""}, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    remaining = MAX_QUERY_BYTES - overhead
    query = {key: "é" * (remaining // 2) + "a" * (remaining % 2)}
    native = Mock()
    reader = DiagnosticsReader(native)
    getattr(reader, method)(query)
    call = getattr(native, method)
    call.assert_called_once()
    encoded = call.call_args.args[0]
    assert len(encoded.encode("utf-8")) == MAX_QUERY_BYTES
    assert json.loads(encoded) == query


def test_configured_diagnostics_outlives_plugins(tmp_path):
    script = r"""
import asyncio, json
import nemo_relay
from nemo_relay import diagnostics, plugin
try:
    diagnostics.reader()
except diagnostics.DiagnosticsError as error:
    assert error.reason == "not_initialized"
else:
    raise AssertionError("import selected a service")
async def main():
    config = plugin.PluginConfig(diagnostics={"enabled": True, "limits": {"max_events": 3}})
    plugin.validate(config)
    try:
        diagnostics.reader()
    except diagnostics.DiagnosticsError as error:
        assert error.reason == "not_initialized"
    else:
        raise AssertionError("validation started diagnostics")
    activation = await plugin.initialize(config)
    reader = diagnostics.reader()
    first = reader.snapshot()
    assert type(first["sequence"]) is int
    assert type(first["lifetime"]["totals"]["failures_total"]["value"]) is int
    assert isinstance(json.loads(reader.snapshot_json())["sequence"], str)
    assert reader.info()["limits"]["max_events"] == 3
    assert reader.info()["logging_coverage"] == "attached_since_service_start"
    await activation.close()
    second = reader.snapshot()
    assert second["source"] == first["source"]
    assert "diagnostics" not in activation.report["resolved_config"]
    for query in ({"unknown": True}, {"component_key": "x" * 4097}, {"after_sequence": 2**64}):
        try:
            reader.snapshot(query)
        except diagnostics.DiagnosticsError as error:
            assert error.reason == "invalid_query"
        else:
            raise AssertionError("invalid query accepted")
    try:
        await plugin.initialize(plugin.PluginConfig(diagnostics={"enabled": False}))
    except ValueError as error:
        assert "configuration_conflict" in str(error)
    else:
        raise AssertionError("startup-only settings changed")
    assert reader.snapshot()["source"] == first["source"]
    assert type(reader.info()["discovery"]["max_query_events"]) is int
    assert any(p["name"] == "otel" and not p["supported"] for p in reader.info()["discovery"]["producer_support"])
    for query, expected in [({"schema_version": 2}, "unsupported_schema"),
                            ({"max_events": 2**32 - 1}, "invalid_query"),
                            ({"max_response_bytes": 1}, "snapshot_capacity")]:
        try:
            reader.snapshot(query)
        except diagnostics.DiagnosticsError as error:
            assert error.reason == expected
        else:
            raise AssertionError("invalid request accepted")
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=tmp_path,
        env={**os.environ, "NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_diagnostics_disabled_is_explicit(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
        import asyncio
        from nemo_relay import diagnostics, plugin
        async def main():
            activation = await plugin.initialize(plugin.PluginConfig())
            reader = diagnostics.reader()
            assert reader.info()["binding_reserved_bytes"] == 0
            try:
                reader.snapshot()
            except diagnostics.DiagnosticsError as error:
                assert error.reason == "disabled"
            else:
                raise AssertionError("disabled returned an enabled empty snapshot")
            await activation.close()
        asyncio.run(main())
    """),
        ],
        cwd=tmp_path,
        env={**os.environ, "NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
