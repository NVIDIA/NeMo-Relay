# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic configured-service and inspection contracts, isolated from other runtimes."""

import os
import subprocess
import sys
import textwrap


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
