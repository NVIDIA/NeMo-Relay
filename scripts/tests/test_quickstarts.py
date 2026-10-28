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

    def test_cli_requires_shell_and_verifies_its_expected_output(self):
        source = (
            example_page()
            .replace('"shells": ["bash"]', '"shells": ["bash", "powershell"]')
            .replace('"id":"run","role":"run"', '"id":"run","role":"run","shells":["bash"]')
            .replace('"id":"expected","role":"expected"', '"id":"expected","role":"expected","shells":["bash"]')
        )
        source += (
            '\n{/* quickstart-block {"id":"run","role":"run","shells":["powershell"],'
            '"cwd":"project","session":"main","timeout_seconds":30} */}\n'
            "```powershell\npython demo.py\n```\n\n"
            '{/* quickstart-block {"id":"expected","role":"expected","shells":["powershell"],'
            '"cwd":"project","from":"run","match":"exact-lines"} */}\n'
            "```text\nPS_OK\n```\n"
        )
        self.page.write_text(source, encoding="utf-8")
        for action in ("extract", "preflight", "verify"):
            with (
                self.subTest(action=action),
                mock.patch.object(sys, "argv", ["check-quickstarts", action, str(self.page)]),
                contextlib.redirect_stderr(io.StringIO()) as errors,
            ):
                self.assertEqual(QS.main(), 1)
                self.assertIn("select --shell", errors.getvalue())

        project = self.root / "project"
        (project / "logs").mkdir(parents=True)
        (project / "marker.txt").write_text("TOOL_OK\n", encoding="utf-8")
        (project / "logs/events.jsonl").write_text(agent_events(), encoding="utf-8")
        stdout = self.root / "stdout.txt"
        for shell, expected in (("bash", "OK"), ("powershell", "PS_OK")):
            arguments = [
                "check-quickstarts",
                "verify",
                str(self.page),
                "--shell",
                shell,
                "--workspace",
                str(self.root),
                "--stdout",
                str(stdout),
            ]
            with (
                self.subTest(shell=shell),
                mock.patch.object(sys, "argv", arguments),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                stdout.write_text(expected + "\n", encoding="utf-8")
                self.assertEqual(QS.main(), 0)
                stdout.write_text("wrong shell output\n", encoding="utf-8")
                self.assertEqual(QS.main(), 1)

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

    def test_extracts_committed_file_from_fern_code_component(self):
        source = example_page().replace(
            "```python\nprint('OK')\n```",
            '<Code src="_code/demo.py" language="python" title="demo.py" />',
        )
        snippets = self.root / "_code"
        snippets.mkdir()
        content = "# A committed sample.\nprint('OK')\n"
        (snippets / "demo.py").write_text(content, encoding="utf-8")
        self.page.write_text(source, encoding="utf-8")
        block = QS.extract(self.page, "bash")["blocks"][0]
        self.assertEqual(block["content"], content)
        self.assertEqual(block["language"], "python")
        self.assertEqual(block["path"], "demo.py")

    def test_rejects_missing_unsafe_or_untracked_code_sources(self):
        snippets = self.root / "_code"
        snippets.mkdir()
        (snippets / "demo.py").write_text("print('OK')\n", encoding="utf-8")
        component = '<Code src="_code/demo.py" language="python" title="demo.py" />'
        source = example_page().replace("```python\nprint('OK')\n```", component)
        for label, changed in {
            "missing source": source.replace("_code/demo.py", "_code/missing.py"),
            "parent traversal": source.replace("_code/demo.py", "../demo.py"),
            "remote source": source.replace("_code/demo.py", "https://example.com/demo.py"),
            "untracked component": example_page() + "\n" + component,
            "unsupported attributes": source.replace('language="python"', 'lines="1"'),
            "command component": source.replace('"role":"file"', '"role":"run"'),
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

    def test_schema_rejects_unknown_fields_and_wrong_types(self):
        source = example_page()
        cases = {
            "unknown scenario field": source.replace('"schema": 1', '"schema": 1, "typo_setting": true'),
            "boolean schema version": source.replace('"schema": 1', '"schema": true'),
            "missing prerequisite types": source.replace('">=3.11"', "false"),
            "unknown block field": source.replace('"role":"file"', '"role":"file","typo_setting":true'),
            "boolean timeout": source.replace('"timeout_seconds":30', '"timeout_seconds":true'),
            "string timeout": source.replace('"timeout_seconds":30', '"timeout_seconds":"30"'),
            "zero timeout": source.replace('"timeout_seconds":30', '"timeout_seconds":0'),
            "unknown assertion field": source.replace('"trim": true', '"trim": true, "typo_setting": true'),
            "invalid scope selector": source.replace('"category": "agent"', '"category": false'),
        }
        for label, changed in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.assertNotEqual(source, changed)
                self.page.write_text(changed, encoding="utf-8")
                QS.extract(self.page)

    def test_check_reports_multiple_schema_errors_and_failing_pages(self):
        alpha = (
            example_page()
            .replace('"id": "demo"', '"id": "alpha"')
            .replace('"schema": 1', '"schema": true')
            .replace('">=3.11"', "false")
        )
        beta = example_page().replace('"id": "demo"', '"id": "beta"').replace('"cwd":"project"', '"cwd":"../escape"')
        (self.root / "alpha.mdx").write_text(alpha, encoding="utf-8")
        (self.root / "beta.mdx").write_text(beta, encoding="utf-8")
        (self.root / "gamma.mdx").write_text("# Missing contract\n", encoding="utf-8")
        with (
            mock.patch.object(QS, "PAGES", self.root),
            mock.patch.object(sys, "argv", ["check-quickstarts", "check"]),
            contextlib.redirect_stderr(io.StringIO()) as errors,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(QS.main(), 1)
        report = errors.getvalue()
        self.assertIn("alpha.mdx: scenario.schema:", report)
        self.assertIn("alpha.mdx: scenario.requires.python:", report)
        self.assertIn("beta.mdx: unsafe block cwd", report)
        self.assertIn("gamma.mdx: expected one scenario", report)
        self.assertNotIn("gamma.mdx: gamma.mdx:", report)
        self.assertNotIn("Validated", output.getvalue())

    def test_default_check_rejects_missing_and_partial_contracts(self):
        (self.root / "legacy.mdx").write_text("# Legacy guide\n", encoding="utf-8")
        (self.root / "index.mdx").write_text("# Overview\n", encoding="utf-8")
        with mock.patch.object(QS, "PAGES", self.root), mock.patch.object(sys, "argv", ["check-quickstarts", "check"]):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(QS.main(), 1)
            (self.root / "legacy.mdx").unlink()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(QS.main(), 0)
            self.assertIn("Validated 1 quickstart contracts", output.getvalue())

            (self.root / "partial.mdx").write_text(
                '{/* quickstart-block {"id":"run","role":"run"} */}\n', encoding="utf-8"
            )
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(QS.main(), 1)

    def test_default_check_rejects_zero_guides(self):
        empty = self.root / "empty"
        empty.mkdir()
        with (
            mock.patch.object(QS, "PAGES", empty),
            mock.patch.object(sys, "argv", ["check-quickstarts", "check"]),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(QS.main(), 1)
        self.assertIn("no quickstart guides", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
