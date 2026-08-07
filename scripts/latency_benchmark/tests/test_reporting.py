# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency benchmark environment reporting."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.latency_benchmark.src import reporting


class EnvironmentRecordTests(unittest.TestCase):
    def test_unknown_git_status_is_not_reported_as_dirty(self) -> None:
        version = SimpleNamespace(stdout="nemo-relay 0.1.0\n")
        with (
            mock.patch.object(reporting.subprocess, "run", return_value=version),
            mock.patch.object(reporting, "_git_output", side_effect=["unknown", "abc123"]),
            mock.patch.object(reporting.platform, "platform", return_value="test-platform"),
            mock.patch.object(reporting.platform, "machine", return_value="test-machine"),
            mock.patch.object(reporting.platform, "processor", return_value="test-processor"),
            mock.patch.object(reporting.platform, "python_version", return_value="3.13.3"),
        ):
            environment = reporting.environment_record(Path("nemo-relay"))

        self.assertEqual(environment["git_commit"], "abc123")
        self.assertFalse(environment["git_dirty"])


if __name__ == "__main__":
    unittest.main()
