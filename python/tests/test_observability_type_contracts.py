# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer and signature regressions for observability constructor declarations."""

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nemo_relay import observability


def _check_consumer(consumer_path: Path) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--project",
            str(repository_root),
            "--python",
            sys.executable,
            "--extra-search-path",
            str(repository_root / "python"),
            "--python-version",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            "--output-format",
            "full",
            "--color",
            "never",
            "--no-progress",
            "--error",
            "invalid-argument-type",
            "--error",
            "missing-argument",
            str(consumer_path),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    "class_name",
    ["AtofStreamSinkConfig", "HttpStorageConfig", "OpenTelemetryEndpointConfig", "OpenTelemetrySignalEndpointConfig"],
)
def test_constructor_declaration_matches_runtime_signature(class_name: str) -> None:
    declaration_path = Path(observability.__file__).with_suffix(".pyi")
    declarations = ast.parse(declaration_path.read_text())
    class_declaration = next(
        statement
        for statement in declarations.body
        if isinstance(statement, ast.ClassDef) and statement.name == class_name
    )
    declared_parameters = {
        statement.target.id: statement.value is None
        for statement in class_declaration.body
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
    }
    runtime_parameters = inspect.signature(getattr(observability, class_name)).parameters

    assert tuple(declared_parameters) == tuple(runtime_parameters)
    assert {name for name, required in declared_parameters.items() if required} == {
        name for name, parameter in runtime_parameters.items() if parameter.default is inspect.Parameter.empty
    }


def test_valid_constructor_calls_execute_and_type_check(tmp_path: Path) -> None:
    consumer_path = tmp_path / "valid_observability_consumer.py"
    consumer_path.write_text(
        textwrap.dedent(
            """\
            from nemo_relay.observability import (
                AtofStreamSinkConfig,
                HttpStorageConfig,
                OpenTelemetryEndpointConfig,
                OpenTelemetrySignalEndpointConfig,
            )

            stream_sink = AtofStreamSinkConfig(
                "https://collector.example/events", "http_post", {}, {}, 3000,
                "preserve", "events", {"x-token": "stream-token.txt"},
            )
            http_storage = HttpStorageConfig(
                "https://collector.example/atif", {}, {}, 3000,
                {"x-token": "storage-token.txt"},
            )
            trace_endpoint = OpenTelemetryEndpointConfig(
                "full", "https://collector.example/v1/traces", "inherit", [], [],
                "http_binary", "trace-service", None, None, "trace-scope", 3000,
                {"x-inline": "trace"}, {"x-from-environment": "TRACE_TOKEN"},
                {"deployment.environment": "trace"}, 64, 16, 1000, 90000,
                ["trace."], ["deployment."], {"x-token": "trace-token.txt"},
            )
            signal_endpoint = OpenTelemetrySignalEndpointConfig(
                "https://collector.example/v1/logs", "http_binary",
                {"x-inline": "signal"}, {"x-from-environment": "SIGNAL_TOKEN"},
                {"deployment.environment": "signal"}, "signal-service", None,
                None, "signal-scope", 3000, {"x-token": "signal-token.txt"},
                ["deployment."],
            )
            keyword_configurations = (
                AtofStreamSinkConfig(url="https://collector.example/events", timeout_millis=3000),
                HttpStorageConfig(endpoint="https://collector.example/atif", timeout_millis=3000),
                OpenTelemetryEndpointConfig(
                    type="full", endpoint="https://collector.example/v1/traces", timeout_millis=3000,
                ),
                OpenTelemetrySignalEndpointConfig(
                    endpoint="https://collector.example/v1/logs", timeout_millis=3000,
                ),
            )

            assert stream_sink.timeout_millis == 3000
            assert stream_sink.header_file == {"x-token": "stream-token.txt"}
            assert http_storage.timeout_millis == 3000
            assert http_storage.header_file == {"x-token": "storage-token.txt"}
            assert trace_endpoint.headers == {"x-inline": "trace"}
            assert trace_endpoint.header_env == {"x-from-environment": "TRACE_TOKEN"}
            assert trace_endpoint.resource_attributes == {"deployment.environment": "trace"}
            assert trace_endpoint.header_file == {"x-token": "trace-token.txt"}
            assert trace_endpoint.max_queue_size == 64
            assert signal_endpoint.headers == {"x-inline": "signal"}
            assert signal_endpoint.header_env == {"x-from-environment": "SIGNAL_TOKEN"}
            assert signal_endpoint.resource_attributes == {"deployment.environment": "signal"}
            assert signal_endpoint.header_file == {"x-token": "signal-token.txt"}
            assert signal_endpoint.service_name == "signal-service"
            assert all(configuration.timeout_millis == 3000 for configuration in keyword_configurations)
            """
        )
    )

    runtime_result = subprocess.run(
        [sys.executable, str(consumer_path)], capture_output=True, text=True, timeout=60, check=False
    )
    assert runtime_result.returncode == 0, runtime_result.stdout + runtime_result.stderr
    typing_result = _check_consumer(consumer_path)
    assert typing_result.returncode == 0, typing_result.stdout + typing_result.stderr


def test_consumer_type_check_rejects_missing_stream_url(tmp_path: Path) -> None:
    consumer_path = tmp_path / "missing_url_consumer.py"
    consumer_path.write_text("from nemo_relay.observability import AtofStreamSinkConfig\nAtofStreamSinkConfig()\n")

    result = _check_consumer(consumer_path)
    diagnostics = result.stdout + result.stderr
    assert result.returncode == 1, diagnostics
    assert diagnostics.count("error[missing-argument]") == 1, diagnostics
    assert "required parameter `url`" in diagnostics
    assert "Found 1 diagnostic" in diagnostics
