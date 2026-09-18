# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys

_LOG_ENVIRONMENT = (
    "NEMO_RELAY_LOG",
    "NEMO_RELAY_LOG_STDERR",
    "NEMO_RELAY_LOG_STDERR_FORMAT",
    "NEMO_RELAY_LOG_CONFIG_PATH",
)


def _run_nemo_relay(source: str = "import nemo_relay", **logging_environment: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in _LOG_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(logging_environment)
    return subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_binding_initializes_logging_from_environment() -> None:
    completed = _run_nemo_relay(
        NEMO_RELAY_LOG="info",
        NEMO_RELAY_LOG_STDERR_FORMAT="jsonl",
    )

    assert completed.returncode == 0, completed.stderr
    assert '"event":"logging_initialized"' in completed.stderr


def test_binding_disables_stderr_logging_from_environment() -> None:
    completed = _run_nemo_relay(
        NEMO_RELAY_LOG="info",
        NEMO_RELAY_LOG_STDERR="false",
    )

    assert completed.returncode == 0, completed.stderr
    assert not completed.stderr


def test_binding_rejects_invalid_logging_environment() -> None:
    completed = _run_nemo_relay(NEMO_RELAY_LOG="")

    assert completed.returncode != 0
    assert "NEMO_RELAY_LOG must not be empty" in completed.stderr


def test_binding_flushes_file_sink_during_shutdown(tmp_path) -> None:
    config_path = tmp_path / "logging.toml"
    log_path = tmp_path / "operational.jsonl"
    config_path.write_text(
        f"""[logging]
level = "info"
stderr_format = "human"
flush_interval_millis = 0

[[logging.sinks]]
path = {json.dumps(str(log_path))}
level = "info"
format = "jsonl"
queue_capacity = 16
"""
    )

    completed = _run_nemo_relay(NEMO_RELAY_LOG_CONFIG_PATH=str(config_path))

    assert completed.returncode == 0, completed.stderr
    assert '"event":"logging_shutdown_started"' in log_path.read_text()


def test_binding_emits_every_log_level_with_structured_fields(tmp_path) -> None:
    config_path = tmp_path / "logging.toml"
    log_path = tmp_path / "operational.jsonl"
    config_path.write_text(
        f"""[logging]
level = "trace"
stderr_format = "human"
flush_interval_millis = 0

[[logging.sinks]]
path = {json.dumps(str(log_path))}
level = "trace"
format = "jsonl"
queue_capacity = 16
"""
    )
    source = """
import nemo_relay

nemo_relay.trace("trace message", target="binding", fields={"ordinal": 0})
nemo_relay.debug("debug message", target="binding", fields={"ordinal": 1})
nemo_relay.info("info message", target="binding", fields={"ordinal": 2, "nested": {"ok": True}})
nemo_relay.warn("warn message", target="binding", fields={"ordinal": 3})
nemo_relay.error("error message", target="binding", fields={"ordinal": 4})
"""

    completed = _run_nemo_relay(source, NEMO_RELAY_LOG_CONFIG_PATH=str(config_path))

    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    records_by_message = {record.get("message"): record for record in records}
    for ordinal, level in enumerate(("trace", "debug", "info", "warn", "error")):
        record = records_by_message[f"{level} message"]
        assert record["level"] == level
        assert record["target"] == "nemo_relay.python.binding"
        assert record["fields"]["ordinal"] == ordinal
    assert records_by_message["info message"]["fields"]["nested"] == {"ok": True}


def test_binding_rejects_non_object_log_fields() -> None:
    completed = _run_nemo_relay(
        "import nemo_relay; nemo_relay.info('invalid', fields=['not', 'an', 'object'])",
        NEMO_RELAY_LOG="trace",
    )

    assert completed.returncode != 0
    assert "log fields must be an object" in completed.stderr
