# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark subprocess lifecycle helpers."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.latency_benchmark.src import processes


class TransparentRelayProcessTests(unittest.TestCase):
    def test_redirects_stdin_when_starting_transparent_relay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "nemo-relay"
            binary.touch()
            plugin_config = root / "plugins.toml"
            plugin_config.write_text("version = 1\ncomponents = []\n", encoding="utf-8")
            child = mock.Mock()
            child.poll.return_value = None
            child.wait.return_value = 0

            def launch(*_args: object, **_kwargs: object) -> mock.Mock:
                (root / "transparent-test-fixed.gateway").write_text("http://127.0.0.1:1234", encoding="utf-8")
                return child

            with (
                mock.patch.object(
                    processes.uuid,
                    "uuid4",
                    return_value=SimpleNamespace(hex="fixed"),
                ),
                mock.patch.object(processes.subprocess, "Popen", side_effect=launch) as popen,
            ):
                relay = processes.TransparentRelayProcess(
                    binary,
                    root,
                    "http://127.0.0.1:8000",
                    plugin_config,
                    "transparent-test",
                )
                relay.start()
                try:
                    self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
                finally:
                    relay.stop()


if __name__ == "__main__":
    unittest.main()
