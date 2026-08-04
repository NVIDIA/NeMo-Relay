# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

_LOG_ENVIRONMENT = (
    "NEMO_RELAY_LOG",
    "NEMO_RELAY_LOG_STDERR_FORMAT",
    "NEMO_RELAY_LOG_CONFIG_PATH",
)


def _import_nemo_relay(**logging_environment: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in _LOG_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(logging_environment)
    return subprocess.run(
        [sys.executable, "-c", "import nemo_relay"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_binding_initializes_logging_from_environment():
    completed = _import_nemo_relay(
        NEMO_RELAY_LOG="info",
        NEMO_RELAY_LOG_STDERR_FORMAT="jsonl",
    )

    assert completed.returncode == 0, completed.stderr
    assert '"event":"logging_initialized"' in completed.stderr


def test_binding_rejects_invalid_logging_environment():
    completed = _import_nemo_relay(NEMO_RELAY_LOG="")

    assert completed.returncode != 0
    assert "NEMO_RELAY_LOG must not be empty" in completed.stderr
