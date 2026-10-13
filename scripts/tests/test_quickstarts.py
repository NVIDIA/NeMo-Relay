# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for MDX quickstart extraction and evidence checks."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("quickstarts", ROOT / "scripts/check-quickstarts.py")
assert SPEC is not None and SPEC.loader is not None
QS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QS)


def example_page() -> str:
    scenario = {
        "schema": 1,
        "id": "demo",
        "mode": "local",
        "shells": ["bash"],
        "cwd": "project",
        "requires": {"python": ">=3.11"},
        "setup": [],
        "fresh": ["marker.txt", "logs/events.jsonl"],
        "checks": [
            {"kind": "file-equals", "path": "marker.txt", "value": "TOOL_OK", "trim": True},
            {"kind": "atof", "path": "logs/events.jsonl", "version": "0.1", "paired_scopes": [{"category": "agent"}]},
        ],
    }
    return (
        "{/* quickstart-scenario\n"
        + json.dumps(scenario)
        + "\n*/}\n\n"
        + '{/* quickstart-block {"id":"program","role":"file","cwd":"project","path":"demo.py"} */}\n'
        + "```python\nprint('OK')\n```\n\n"
        + '{/* quickstart-block {"id":"run","role":"run","cwd":"project","session":"main","timeout_seconds":30} */}\n'
        + "```bash\npython demo.py\n```\n\n"
        + (
            '{/* quickstart-block {"id":"expected","role":"expected",'
            '"cwd":"project","from":"run","match":"exact-lines"} */}\n'
        )
        + "```text\nOK\n```\n"
    )


def agent_events() -> str:
    identity = {
        "kind": "scope",
        "atof_version": "0.1",
        "uuid": str(uuid4()),
        "parent_uuid": None,
        "timestamp": "2026-10-13T12:00:00Z",
        "name": "demo-agent",
        "category": "agent",
    }
    return "".join(json.dumps(dict(identity, scope_category=phase)) + "\n" for phase in ("start", "end"))


class QuickstartContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.page = self.root / "demo.mdx"
        self.page.write_text(example_page(), encoding="utf-8")

    def test_extracts_commands_expected_output_and_artifacts(self):
        scenario = QS.extract(self.page, "bash")
        self.assertEqual([block["id"] for block in scenario["blocks"]], ["program", "run", "expected"])
        self.assertEqual(scenario["blocks"][1]["content"], "python demo.py\n")

        project = self.root / "project"
        (project / "logs").mkdir(parents=True)
        (project / "marker.txt").write_text("TOOL_OK\n", encoding="utf-8")
        (project / "logs/events.jsonl").write_text(agent_events(), encoding="utf-8")
        QS.verify(scenario, self.root, "OK\n", {})
        with self.assertRaisesRegex(ValueError, "stdout"):
            QS.verify(scenario, self.root, "not OK\n", {})

    def test_rejects_hidden_commands_and_unsafe_paths(self):
        source = example_page()
        for label, changed in {
            "unannotated fence": source + "\n```bash\necho hidden\n```\n",
            "unsafe output": source.replace('"path":"demo.py"', '"path":"../demo.py"'),
            "missing producer": source.replace('"from":"run"', '"from":"unknown"'),
            "unknown matcher": source.replace('"match":"exact-lines"', '"match":"contains"'),
        }.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.page.write_text(changed, encoding="utf-8")
                QS.extract(self.page)

    def test_atof_requires_matching_lifecycle_not_payload_text(self):
        check = QS.extract(self.page)["checks"][1]
        valid = agent_events()
        QS.verify_atof(valid, check)
        for invalid in ("", '{"payload":"agent"}\n', valid.splitlines(keepends=True)[0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                QS.verify_atof(invalid, check)

    def test_default_check_ignores_legacy_pages_but_rejects_partial_contracts(self):
        (self.root / "legacy.mdx").write_text("# Legacy guide\n", encoding="utf-8")
        with mock.patch.object(QS, "PAGES", self.root), mock.patch.object(sys, "argv", ["check-quickstarts", "check"]):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(QS.main(), 0)
            self.assertIn("Validated 1 quickstart contracts", output.getvalue())

            (self.root / "partial.mdx").write_text(
                '{/* quickstart-block {"id":"run","role":"run"} */}\n', encoding="utf-8"
            )
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(QS.main(), 1)


if __name__ == "__main__":
    unittest.main()
