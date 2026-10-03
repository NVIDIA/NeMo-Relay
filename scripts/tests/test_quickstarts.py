# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract and negative-path tests; no credentials or model traffic required."""

import ast
import importlib.util
import json
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("quickstarts", ROOT / "scripts/check-quickstarts.py")
assert SPEC is not None and SPEC.loader is not None
QS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QS)


def lifecycle(category="agent", name="demo-agent", parent=None):
    identity = {
        "kind": "scope",
        "atof_version": "0.1",
        "uuid": str(uuid4()),
        "parent_uuid": parent,
        "timestamp": "2026-09-30T12:00:00Z",
        "name": name,
        "category": category,
    }
    return [dict(identity, scope_category=phase) for phase in ("start", "end")]


def events(host=None):
    # CLI 0.9.1 closes custom turn scopes for Claude/Codex, not agent roots.
    turn_names = {"claude-code": "claude-code-turn", "codex": "codex-turn"}
    agent = lifecycle("custom", turn_names[host]) if host in turn_names else lifecycle()
    return (
        agent + lifecycle("tool", "emit_marker", agent[0]["uuid"]) + lifecycle("llm", "demo-provider", agent[0]["uuid"])
    )


def jsonl(records):
    return "".join(json.dumps(record) + "\n" for record in records)


class QuickstartTests(unittest.TestCase):
    def test_every_page_and_shell_extracts(self):
        pages = [path for path in QS.PAGES.glob("*.mdx") if path.stem != "index"]
        self.assertEqual(len(pages), 12)
        for page in pages:
            scenario = QS.extract(page)
            for shell in scenario["shells"]:
                with self.subTest(page=page.name, shell=shell):
                    recipe = QS.extract(page, shell)
                    self.assertTrue(all(shell in block.get("shells", scenario["shells"]) for block in recipe["blocks"]))
                    for block in recipe["blocks"]:
                        if block["language"] == "python":
                            ast.parse(block["content"])
                        elif block["language"] == "toml":
                            tomllib.loads(block["content"])
                        elif block["language"] == "bash":
                            result = subprocess.run(
                                ["bash", "-n"], input=block["content"], text=True, capture_output=True
                            )
                            self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_invalid_document_contracts(self):
        original = (QS.PAGES / "python.mdx").read_text()
        changes = {
            "unlabeled": original + "\n```bash\necho hidden\n```\n",
            "unknown schema": original.replace('"schema": 1', '"schema": 2'),
            "orphan": original + '\n{/* quickstart-block {"id":"orphan"} */}\n',
            "unsafe file": original.replace('"path":"quickstart.py"', '"path":"../escape.py"'),
            "duplicate": original.replace('"id":"program"', '"id":"plugins"'),
            "missing producer": original.replace('"from":"run"', '"from":"missing"'),
            "unknown matcher": original.replace('"match":"exact-lines"', '"match":"guess"'),
        }
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "python.mdx"
            for label, text in changes.items():
                with self.subTest(label=label):
                    page.write_text(text)
                    with self.assertRaises(ValueError):
                        QS.extract(page)

    def test_atof_requires_real_paired_lifecycle(self):
        check = QS.extract(QS.PAGES / "python.mdx")["checks"][0]
        records = events()
        QS.verify_atof(jsonl(records), check)
        for label, text in {
            "empty": "",
            "malformed first line": "not json\n" + jsonl(records),
            "missing tool end": jsonl(
                [
                    record
                    for record in records
                    if not (record["category"] == "tool" and record["scope_category"] == "end")
                ]
            ),
            "unrelated JSON": '{"message":"RELAY_QUICKSTART_EVENTS_OK"}\n',
            "duplicate phase": jsonl(records + [records[0]]),
            "wrong version": jsonl(records).replace('"0.1"', '"0.2"'),
            "mismatched identity": jsonl(records[:-1] + [dict(records[-1], parent_uuid=None)]),
        }.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                QS.verify_atof(text, check)

    def test_routing_mark_cannot_be_payload_text(self):
        scenario = QS.extract(QS.PAGES / "switchyard.mdx")
        check = next(check for check in scenario["checks"] if check["kind"] == "atof")
        records = events()
        records[0]["data"] = {"prompt": "switchyard.routing.decision"}
        with self.assertRaises(ValueError):
            QS.verify_atof(jsonl(records), check)
        records.append(
            {
                "kind": "mark",
                "atof_version": "0.1",
                "uuid": str(uuid4()),
                "parent_uuid": records[0]["uuid"],
                "timestamp": "2026-09-30T12:00:00Z",
                "name": "switchyard.routing.decision",
            }
        )
        QS.verify_atof(jsonl(records), check)

    def test_stdout_is_exact_and_artifacts_are_required(self):
        scenario = QS.extract(QS.PAGES / "claude-code.mdx", "bash")
        expected = next(block["content"] for block in scenario["blocks"] if block["role"] == "expected")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "logs").mkdir()
            (workspace / "logs/events.jsonl").write_text(jsonl(events("claude-code")))
            with self.assertRaises(FileNotFoundError):
                QS.verify(scenario, workspace, expected, {})
            (workspace / "relay-check.txt").write_text("RELAY_QUICKSTART_TOOL_OK\n")
            QS.verify(scenario, workspace, expected.replace("\n", "\r\n"), {})
            with self.assertRaises(ValueError):
                QS.verify(scenario, workspace, "Echoed prompt: " + expected, {})

    @unittest.skipUnless(shutil.which("jq"), "jq is required for Unix reader verification")
    def test_actual_bash_verifiers_reject_false_success(self):
        for name in ("claude-code", "codex", "pi", "hermes", "openclaw", "switchyard"):
            scenario = QS.extract(QS.PAGES / f"{name}.mdx", "bash")
            command = next(block["content"] for block in scenario["blocks"] if block["id"] == "verify")
            expected = next(block["content"] for block in scenario["blocks"] if block["role"] == "expected")
            with self.subTest(page=name), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                (workspace / "logs").mkdir()
                command = command.replace("/absolute/path/to/relay-logs", str(workspace / "logs"))
                command = command.replace("/absolute/path/to/agent-workspace", str(workspace))
                (workspace / "relay-check.txt").write_text("RELAY_QUICKSTART_TOOL_OK")
                (workspace / "response.json").write_text(
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "RELAY_QUICKSTART_RESPONSE_OK",
                                    }
                                }
                            ]
                        }
                    )
                )
                records = events(name)
                if name == "switchyard":
                    records.append(
                        {
                            "kind": "mark",
                            "atof_version": "0.1",
                            "name": "switchyard.routing.decision",
                            "uuid": str(uuid4()),
                            "parent_uuid": records[0]["uuid"],
                            "timestamp": "2026-09-30T12:00:00Z",
                        }
                    )
                (workspace / "logs/events.jsonl").write_text(jsonl(records))
                result = subprocess.run(["bash", "-c", command], cwd=workspace, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)
                # All reader shells and the contract accept ordinary CRLF files.
                (workspace / "relay-check.txt").write_bytes(b"  RELAY_QUICKSTART_TOOL_OK\r\n")
                result = subprocess.run(["bash", "-c", command], cwd=workspace, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)
                (workspace / "relay-check.txt").write_text("RELAY_QUICKSTART_TOOL_OK")
                if name in {"claude-code", "codex"}:
                    # A paired but unrelated custom scope must not stand in for a turn.
                    unrelated = [
                        dict(record, name="unrelated") if record["category"] == "custom" else record
                        for record in records
                    ]
                    (workspace / "logs/events.jsonl").write_text(jsonl(unrelated))
                    result = subprocess.run(["bash", "-c", command], cwd=workspace, text=True, capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("RELAY_QUICKSTART_EVENTS_OK", result.stdout)
                (workspace / "logs/events.jsonl").write_text('{"payload":"switchyard.routing.decision"}\n')
                result = subprocess.run(["bash", "-c", command], cwd=workspace, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("RELAY_QUICKSTART_EVENTS_OK", result.stdout)

    def test_openclaw_requires_session_shutdown(self):
        check = next(check for check in QS.extract(QS.PAGES / "openclaw.mdx")["checks"] if check["kind"] == "atof")
        records = events()
        open_session = [
            record for record in records if not (record["category"] == "agent" and record["scope_category"] == "end")
        ]
        with self.assertRaises(ValueError):
            QS.verify_atof(jsonl(open_session), check)
        QS.verify_atof(jsonl(records), check)

    def test_switchyard_credentials_belong_to_gateway_session(self):
        for shell in ("bash", "powershell"):
            with self.subTest(shell=shell):
                blocks = {block["id"]: block for block in QS.extract(QS.PAGES / "switchyard.mdx", shell)["blocks"]}
                self.assertEqual(blocks["credentials"]["env"], "NVIDIA_API_KEY")
                self.assertEqual(blocks["credentials"]["session"], blocks["gateway"]["session"])
                self.assertNotEqual(blocks["credentials"]["session"], blocks["request"]["session"])
                self.assertIn("PATH" if shell == "bash" else "$env:Path", blocks["gateway"]["content"])

    def test_switchyard_unsupported_platform_preserves_parent_shell(self):
        blocks = QS.extract(QS.PAGES / "switchyard.mdx", "bash")["blocks"]
        install = next(block["content"] for block in blocks if block["id"] == "install")
        # Execute the documented platform/download subshell without installing Relay.
        child = install[install.index("\n(\n") + 1 :]
        command = 'uname() { printf "Unsupported"; }\n' + child + '\nstatus=$?\nprintf "PARENT_OK:%s\\n" "$status"\n'
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PARENT_OK:1\n")
        self.assertIn("No Switchyard 0.3.0 bundle", result.stderr)

    def test_codex_creates_workspace_before_using_it(self):
        for shell in ("bash", "powershell"):
            with self.subTest(shell=shell):
                blocks = {block["id"]: block for block in QS.extract(QS.PAGES / "codex.mdx", shell)["blocks"]}
                self.assertEqual(blocks["install"]["cwd"], ".")
                self.assertEqual(blocks["workspace"]["cwd"], ".")
                self.assertIn("git init", blocks["workspace"]["content"])
                for name in ("config", "plugins", "run", "verify", "expected"):
                    self.assertEqual(blocks[name]["cwd"], "relay-codex-quickstart")

    def test_openclaw_requires_explicit_paths(self):
        scenario = QS.extract(QS.PAGES / "openclaw.mdx")
        path = scenario["checks"][0]["path"]
        with self.assertRaisesRegex(ValueError, "AGENT_WORKSPACE"):
            QS.artifact_path(path, scenario, Path("/tmp/test"), {})
        result = QS.artifact_path(path, scenario, Path("/tmp/test"), {"AGENT_WORKSPACE": "/tmp/agent space"})
        self.assertEqual(result, Path("/tmp/agent space/relay-check.txt"))

    def test_preflight_rejects_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [
                "python3",
                str(ROOT / "scripts/check-quickstarts.py"),
                "preflight",
                str(QS.PAGES / "claude-code.mdx"),
                "--workspace",
                directory,
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            (Path(directory) / "relay-check.txt").write_text("RELAY_QUICKSTART_TOOL_OK")
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale artifact", result.stderr)


if __name__ == "__main__":
    unittest.main()
