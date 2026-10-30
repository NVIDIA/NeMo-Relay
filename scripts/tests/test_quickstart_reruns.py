# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in execution of documented Bash rerun commands against failing host fixtures."""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDES = ("hermes", "claude-code", "codex", "pi")


@unittest.skipIf(os.name == "nt", "Bash rerun regression runs on Linux/macOS")
class QuickstartRerunTests(unittest.TestCase):
    def test_failed_agent_rerun_cannot_reuse_previous_artifacts(self):
        for guide in GUIDES:
            source = (ROOT / "docs/getting-started/quick-start" / f"{guide}.mdx").read_text(encoding="utf-8")
            commands = re.finditer(
                r"\{/\* quickstart-block[ \t]+([^\n]+?)[ \t]*\*/\}\n```bash\n(.*?)\n```",
                source,
                re.DOTALL,
            )
            command = next(match[2] for match in commands if json.loads(match[1])["id"] == "run")
            with self.subTest(guide=guide), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "logs").mkdir()
                (folder / "logs/events.jsonl").write_text("old events")
                (folder / "relay-check.txt").write_text("RELAY_QUICKSTART_TOOL_OK")
                binary = folder / ("hermes" if guide == "hermes" else "nemo-relay")
                binary.write_text("#!/bin/sh\nexit 1\n")
                binary.chmod(0o755)
                environment = {**os.environ, "PATH": str(folder) + os.pathsep + os.environ["PATH"]}
                result = subprocess.run(
                    ["bash", "-c", command], cwd=folder, env=environment, capture_output=True, text=True, timeout=10
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((folder / "relay-check.txt").exists())
                self.assertFalse((folder / "logs/events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
