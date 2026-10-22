# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the public diagnostics reader example runnable in a fresh service process."""

import os
import subprocess
import sys
from pathlib import Path


def test_diagnostics_reader_example(tmp_path):
    example = Path(__file__).resolve().parents[2] / "examples/diagnostics/python.py"
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        env={**os.environ, "NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
