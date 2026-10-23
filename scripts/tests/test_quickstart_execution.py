# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise asset identity, real shell failures, isolation, and captured ATOF evidence."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from packaging.tags import sys_tags

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("quickstart_execution", ROOT / "scripts/run-quickstarts.py")
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
ASSETS = sys.modules["quickstarts.assets"]
build = sys.modules["quickstarts.build"]
provider = sys.modules["quickstarts.provider"]
shell = sys.modules["quickstarts.shell"]


class QuickstartExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "catalog"
        self.catalog.mkdir()

    def assets(self):
        return ASSETS.Assets("0.11.0-rc.2", self.root / str(uuid4()), str(self.catalog), "NVIDIA/NeMo-Relay")

    def wheel(self, metadata_version="0.11.0rc2"):
        tag = next(sys_tags())
        path = self.catalog / f"nemo_relay-0.11.0rc2-{tag}.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "nemo_relay-0.11.0rc2.dist-info/METADATA",
                f"Metadata-Version: 2.4\nName: nemo-relay\nVersion: {metadata_version}\n",
            )
        return path

    def test_wheel_metadata_and_checksum_must_match_requested_asset(self):
        path = self.wheel()
        assets = self.assets()
        self.assertEqual(assets.python(), path.resolve())
        self.assertEqual(assets.records[path.name]["sha256"], ASSETS.sha256(path))
        self.wheel("0.10.4")
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.assets().python()
        self.wheel()
        (self.catalog / "SHA256SUMS").write_text("0" * 64 + "  " + path.name + "\n")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            self.assets().python()

    def test_wrong_version_and_ambiguous_wheels_fail_without_fallback(self):
        path = self.wheel()
        path.rename(path.with_name(path.name.replace("0.11.0rc2", "0.10.4")))
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.assets().python()
        self.wheel()
        duplicate = self.catalog / "nested"
        duplicate.mkdir()
        (duplicate / path.name).write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.assets().python()

    def test_generated_build_manifest_retains_commit_and_rejects_tampered_assets(self):
        path = self.wheel()
        manifest = {"commit": "a" * 40, "status": "built", "files": {path.name: ASSETS.sha256(path)}}
        (self.catalog / "build.json").write_text(json.dumps(manifest))
        assets = self.assets()
        self.assertEqual(assets.python(), path.resolve())
        self.assertEqual(assets.source["build"]["commit"], "a" * 40)
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            self.assets().python()

    def test_windows_cli_selection_rejects_a_foreign_platform(self):
        foreign = self.catalog / "nemo-relay-cli-x86_64-unknown-linux-gnu-0.11.0-rc.2"
        foreign.write_bytes(b"foreign")
        assets = self.assets()
        assets.platform = "windows-amd64"
        with self.assertRaisesRegex(ValueError, "found 0"):
            assets.cli()
        native = self.catalog / "nemo-relay-cli-x86_64-pc-windows-msvc-0.11.0-rc.2.exe"
        native.write_bytes(b"windows")
        assets = self.assets()
        assets.platform = "windows-amd64"
        with patch.object(ASSETS.subprocess, "check_output", return_value="nemo-relay 0.11.0-rc.2"):
            self.assertEqual(assets.cli(), native.resolve())

    def test_cleanup_failure_preserves_the_case_report(self):
        original_cleanup = tempfile.TemporaryDirectory.cleanup
        with patch.object(
            tempfile.TemporaryDirectory, "cleanup", autospec=True, side_effect=OSError("busy workspace")
        ) as cleanup:
            result = RUNNER.run_case("nodejs", self.scenario(), self.assets(), {}, self.root / "cleanup-failed")
        original_cleanup(cleanup.call_args.args[0])
        self.assertEqual(result["status"], "failed")
        self.assertIn("cleanup failed", result["error"])
        self.assertTrue((self.root / "cleanup-failed/result.json").is_file())

    def test_remote_inventory_requires_integrity_and_plain_filenames(self):
        for text in ("", "0" * 64 + "  ../escape.whl", "0" * 64 + "  a\n" + "0" * 64 + "  a"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                ASSETS.checksum_index(text)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.assets().materialize({"name": "test.whl", "url": "https://example.com/test.whl"})
        for url in ("http://example.com", "https://user:secret@example.com", "https://example.com/?token=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ASSETS.safe_url(url)

    def test_https_directory_discovers_assets_from_checksums(self):
        path = self.wheel()
        sums = f"{ASSETS.sha256(path)}  {path.name}\n".encode()
        responses = {
            "https://example.com/r2/SHA256SUMS": sums,
            "https://example.com/r2/" + path.name: path.read_bytes(),
        }
        with patch.object(ASSETS.Assets, "_read_url", side_effect=lambda url: responses[url]):
            assets = ASSETS.Assets("0.11.0-rc.2", self.root / "download", "https://example.com/r2", "NVIDIA/NeMo-Relay")
            self.assertEqual(assets.python().read_bytes(), path.read_bytes())

    def test_supplied_crate_metadata_is_verified_and_traversal_is_rejected(self):
        path = self.catalog / "nemo-relay-0.11.0-rc.2.crate"
        manifest = b'[package]\nname = "nemo-relay"\nversion = "0.11.0-rc.2"\n'
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo("nemo-relay-0.11.0-rc.2/Cargo.toml")
            member.size = len(manifest)
            archive.addfile(member, io.BytesIO(manifest))
        resolved = self.assets().rust()
        self.assertTrue((Path(resolved["nemo-relay"]) / "Cargo.toml").exists())
        with tarfile.open(path, "w:gz") as archive:
            archive.addfile(tarfile.TarInfo("../outside"), io.BytesIO(b""))
        with self.assertRaisesRegex(ValueError, "Unsafe crate"):
            self.assets().rust()

    def scenario(self):
        # These fixtures test the runner/verifier; actual local runs use the real runtime output.
        identity = {
            "kind": "scope",
            "atof_version": "0.1",
            "uuid": str(uuid4()),
            "parent_uuid": None,
            "timestamp": "2026-10-21T12:00:00Z",
            "name": "example",
            "category": "agent",
        }
        events = "".join(json.dumps(dict(identity, scope_category=phase)) + "\n" for phase in ("start", "end"))
        import shlex

        scenario = {
            "cwd": "project",
            "inputs": {},
            "fresh": ["logs/events.jsonl"],
            "checks": [
                {
                    "kind": "atof",
                    "path": "logs/events.jsonl",
                    "version": "0.1",
                    "paired_scopes": [{"category": "agent"}],
                }
            ],
            "blocks": [
                {
                    "id": "setup",
                    "role": "run",
                    "cwd": ".",
                    "session": "main",
                    "timeout_seconds": 5,
                    "content": "mkdir -p project/logs project/node_modules/nemo-relay-node\nexport MARKER=OK\n"
                    + "printf '%s' '{\"version\":\"0.11.0-rc.2\"}' > "
                    + "project/node_modules/nemo-relay-node/package.json\n"
                    + "printf %s "
                    + shlex.quote(events)
                    + " > project/logs/events.jsonl",
                },
                {
                    "id": "run",
                    "role": "run",
                    "cwd": "project",
                    "session": "main",
                    "timeout_seconds": 5,
                    "content": "printf '%s\\n' \"$MARKER\"",
                },
                {"id": "expected", "role": "expected", "cwd": "project", "from": "run", "content": "OK\n"},
            ],
        }
        if shell.WINDOWS:
            scenario["blocks"][0]["content"] = (
                "New-Item -ItemType Directory -Force "
                "-Path project/logs, project/node_modules/nemo-relay-node | Out-Null\n"
                "$env:MARKER = 'OK'\n"
                "[IO.File]::WriteAllText((Join-Path $PWD 'project/node_modules/nemo-relay-node/package.json'), "
                + shell.quote('{"version":"0.11.0-rc.2"}')
                + ")\n"
                + "[IO.File]::WriteAllText((Join-Path $PWD 'project/logs/events.jsonl'), "
                + shell.quote(events)
                + ")\n"
            )
            scenario["blocks"][1]["content"] = "Write-Output $env:MARKER"
        return scenario

    def test_execution_preserves_exports_and_copies_evidence_outside_checkout(self):
        result = RUNNER.run_case("nodejs", self.scenario(), self.assets(), {}, self.root / "pass")
        self.assertEqual(result["status"], "passed", result)
        self.assertTrue((self.root / "pass/workspace/project/logs/events.jsonl").is_file())
        self.assertFalse(list((self.root / "pass").rglob(".session-*")))

    def test_nonzero_exit_and_missing_scopes_cannot_pass_from_marker_text(self):
        for label, command in [
            ("exit", "Write-Output 'OK'; exit 7" if shell.WINDOWS else "printf 'OK\\n'; exit 7"),
            (
                "missing",
                "Write-Output 'OK'; [IO.File]::WriteAllText((Join-Path $PWD 'logs/events.jsonl'), '')"
                if shell.WINDOWS
                else "printf 'OK\\n'; : > logs/events.jsonl",
            ),
        ]:
            scenario = copy.deepcopy(self.scenario())
            scenario["blocks"][1]["content"] = command
            result = RUNNER.run_case("nodejs", scenario, self.assets(), {}, self.root / label)
            self.assertEqual(result["status"], "failed", result)
            self.assertTrue((self.root / label / "result.json").exists())

    def test_timeout_fails_and_records_the_failing_block(self):
        scenario = self.scenario()
        scenario["blocks"][1].update(content="Start-Sleep 30" if shell.WINDOWS else "sleep 30", timeout_seconds=1)
        result = RUNNER.run_case("nodejs", scenario, self.assets(), {}, self.root / "timeout")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["block"], "run")

    @unittest.skipUnless(shell.WINDOWS, "requires native PowerShell and Windows job objects")
    def test_powershell_native_failure_cannot_be_hidden_by_later_output(self):
        with self.assertRaisesRegex(ValueError, "Command exited"):
            RUNNER.execute(
                shell.command(sys.executable, "-c", "raise SystemExit(7)") + "\nWrite-Output 'false-success'",
                self.root,
                dict(os.environ),
                self.root / "out.log",
                self.root / "err.log",
                30,
            )
        self.assertNotIn("false-success", (self.root / "out.log").read_text())

    @unittest.skipUnless(shell.WINDOWS, "requires native PowerShell and Windows job objects")
    def test_powershell_session_preserves_literal_environment_values(self):
        state = self.root / "state.json"
        value = "literal '$env:PATH ` & value"
        for index, content in enumerate(("$env:ROUNDTRIP = " + shell.quote(value), "Write-Output $env:ROUNDTRIP")):
            RUNNER.execute(
                shell.session(content, self.root, state, state if index else None, False),
                self.root,
                dict(os.environ),
                self.root / f"{index}.out",
                self.root / f"{index}.err",
                30,
            )
        self.assertEqual((self.root / "1.out").read_text().strip(), value)

    @unittest.skipUnless(shell.WINDOWS, "requires native PowerShell and Windows job objects")
    def test_windows_timeout_kills_descendants(self):
        marker = self.root / "leaked-child"
        child = "import time; from pathlib import Path; time.sleep(8); Path(" + repr(str(marker)) + ").touch()"
        parent = (
            "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', " + repr(child) + "]); "
            "print('child-started', flush=True); time.sleep(30)"
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            RUNNER.execute(
                shell.command(sys.executable, "-c", parent),
                self.root,
                dict(os.environ),
                self.root / "tree.out",
                self.root / "tree.err",
                6,
            )
        self.assertIn("child-started", (self.root / "tree.out").read_text())
        time.sleep(9)
        self.assertFalse(marker.exists(), "Timed-out command left a running child process")

    def test_all_requires_adapters_for_every_discovered_case(self):
        cases = RUNNER.inventory()
        self.assertEqual(set(RUNNER.select_cases("local", cases)), {"python", "nodejs", "rust", "langgraph"})
        self.assertEqual(set(RUNNER.select_cases("all", cases)), set(cases))
        with self.assertRaisesRegex(ValueError, "Unknown Quick Starts"):
            RUNNER.select_cases("made-up", cases)

    def test_provider_requires_a_real_protocol_tool_result(self):
        fixture = provider.Fixture(self.root, self.root / "requests.jsonl")
        fixture.start()
        self.addCleanup(fixture.close)

        def post(route, payload):
            request = urllib.request.Request(
                fixture.url + route, json.dumps(payload).encode(), {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.read().decode()

        diagnostic = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Health check"}]})
        self.assertNotIn(provider.RESPONSE_MARKER, diagnostic)
        self.assertEqual(fixture.diagnostics, 1)
        with self.assertRaises(urllib.error.HTTPError) as error:
            post("/api/show", {"model": "relay-quickstart"})
        self.assertEqual(error.exception.code, 404)
        self.assertEqual(fixture.errors, [])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            fixture.verify()

        prompt = {
            "messages": [{"role": "user", "content": provider.TOOL_MARKER + provider.RESPONSE_MARKER}],
            "tools": [{"type": "function", "function": {"name": "emit_marker"}}],
        }
        first = json.loads(post("/v1/chat/completions", prompt))
        self.assertEqual(first["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "emit_marker")
        self.assertNotIn(provider.RESPONSE_MARKER, json.dumps(first))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            fixture.verify()
        prompt["messages"].append({"role": "tool", "content": provider.TOOL_MARKER, "tool_call_id": "call_quickstart"})
        self.assertIn(provider.RESPONSE_MARKER, post("/v1/chat/completions", prompt))
        fixture.verify()
        self.assertFalse((self.root / "relay-check.txt").exists())
        self.assertFalse((self.root / "logs/events.jsonl").exists())

    def test_provider_streams_real_write_tool_protocols(self):
        for route, tool, expected in (
            ("/v1/responses", "exec_command", "response.function_call_arguments.delta"),
            ("/v1/messages", "Write", "input_json_delta"),
            ("/v1/chat/completions", "write", "tool_calls"),
        ):
            with self.subTest(route=route):
                fixture = provider.Fixture(self.root, self.root / "requests.jsonl")
                fixture.start()
                try:
                    payload = {"stream": True, "tools": [{"name": tool, "type": "function"}], "messages": []}
                    request = urllib.request.Request(
                        fixture.url + route, json.dumps(payload).encode(), {"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        frames = response.read().decode()
                    self.assertIn(expected, frames)
                    self.assertIn(provider.TOOL_MARKER, frames)
                    self.assertNotIn("require_escalated", frames)
                    self.assertFalse((self.root / "relay-check.txt").exists())
                finally:
                    fixture.close()

    def test_gemini_requires_successful_function_response_before_completion(self):
        fixture = provider.Fixture(self.root, self.root / "requests.jsonl")
        fixture.start()
        self.addCleanup(fixture.close)

        def post(payload, streaming=False):
            route = "streamGenerateContent?alt=sse" if streaming else "generateContent"
            request = urllib.request.Request(
                fixture.url + "/v1beta/models/gemini-3.5-flash:" + route,
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.read().decode()

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": provider.RESPONSE_MARKER}]}],
            "tools": [{"functionDeclarations": [{"name": "emit_marker", "parameters": {"type": "OBJECT"}}]}],
        }
        first = json.loads(post(payload))
        self.assertEqual(
            first["candidates"][0]["content"]["parts"][0]["functionCall"], {"name": "emit_marker", "args": {}}
        )
        self.assertNotIn(provider.RESPONSE_MARKER, json.dumps(first))
        function_response: dict[str, Any] = {"name": "emit_marker", "response": {"error": "failed"}}
        payload["contents"].append({"role": "user", "parts": [{"functionResponse": function_response}]})
        self.assertNotIn(provider.RESPONSE_MARKER, post(payload))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            fixture.verify()
        function_response["response"] = {"result": provider.TOOL_MARKER}
        self.assertIn(provider.RESPONSE_MARKER, post(payload))
        streamed = post(payload, streaming=True)
        self.assertIn(provider.RESPONSE_MARKER, streamed)
        self.assertNotIn("[DONE]", streamed)
        fixture.verify()
        self.assertFalse((self.root / "logs/events.jsonl").exists())

    def test_source_build_pins_commit_and_ignores_dirty_work(self):
        import subprocess

        root = self.root / "repo"
        root.mkdir()
        for command in (
            ["git", "init", str(root)],
            ["git", "-C", str(root), "config", "user.email", "test@example.com"],
            ["git", "-C", str(root), "config", "user.name", "Test"],
        ):
            subprocess.run(command, check=True, capture_output=True)
        manifest = root / "Cargo.toml"
        manifest.write_text('[workspace.package]\nversion = "0.11.0"\n')
        subprocess.run(["git", "-C", str(root), "add", "Cargo.toml"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True, capture_output=True)
        manifest.write_text("uncommitted work")
        source, version, record = build.checkout(root, "HEAD", self.root)
        build.prepare(root, source, self.root, set(), RUNNER.execute, record)
        self.assertEqual(version, "0.11.0")
        self.assertEqual(len(record["commit"]), 40)
        self.assertIn("[workspace.package]", (source / "Cargo.toml").read_text())
        self.assertEqual(manifest.read_text(), "uncommitted work")


if __name__ == "__main__":
    unittest.main()
