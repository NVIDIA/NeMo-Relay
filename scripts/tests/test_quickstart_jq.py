# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the documented jq expressions without running agents or inference."""

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDES = ("hermes", "claude-code", "codex", "pi")


def events_for(guide: str) -> list[dict]:
    """Build paired scopes with the turn category and name required by a guide."""
    custom_turn = guide in {"claude-code", "codex"}
    category = "custom" if custom_turn else "agent"
    name = f"{guide}-turn" if custom_turn else "demo-agent"
    events = []
    for index, (scope_category, scope_name) in enumerate(
        ((category, name), ("tool", "emit_marker"), ("llm", "demo-provider")), start=1
    ):
        for phase in ("start", "end"):
            events.append(
                {
                    "atof_version": "0.1",
                    "kind": "scope",
                    "scope_category": phase,
                    "category": scope_category,
                    "name": scope_name,
                    "uuid": f"00000000-0000-4000-8000-{index:012d}",
                    "parent_uuid": None if index == 1 else "00000000-0000-4000-8000-000000000001",
                }
            )
    return events


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


class QuickstartJqTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("QUICKSTART_JQ_BINARIES")
        if configured is not None:
            cls.binaries = configured.split(os.pathsep)
            if not all(cls.binaries):
                raise ValueError("QUICKSTART_JQ_BINARIES must contain executable paths")
        else:
            binary = shutil.which("jq")
            if binary is None:
                raise unittest.SkipTest("Install jq to run the documented verification-expression tests")
            cls.binaries = [binary]

    def expressions(self, guide):
        """Read the copyable expressions rather than maintain a second verifier."""
        page = ROOT / "docs/getting-started/quick-start" / f"{guide}.mdx"
        source = page.read_text(encoding="utf-8")
        markers = re.findall(r"jq -eRs '([^']*)'", source)
        events = re.findall(r"jq -se '([^']*)'", source)
        self.assertEqual(len(markers), 1, page)
        self.assertEqual(len(events), 1, page)
        return markers[0], events[0]

    def run_jq(self, binary, flags, expression, content):
        return subprocess.run(
            [binary, flags, expression], input=content, capture_output=True, text=True, timeout=10, check=False
        )

    def test_event_verifiers_accept_complete_scopes_and_reject_bad_evidence(self):
        for guide in GUIDES:
            _, expression = self.expressions(guide)
            valid = events_for(guide)
            cases = {"complete": (valid, 0), "empty": ([], 1)}
            for category in {event["category"] for event in valid}:
                cases[f"missing-{category}-end"] = (
                    [
                        event
                        for event in valid
                        if not (event["category"] == category and event["scope_category"] == "end")
                    ],
                    1,
                )
            for key in ("uuid", "name", "parent_uuid"):
                broken = copy.deepcopy(valid)
                broken[-1][key] = "mismatch"
                cases[f"mismatched-{key}"] = (broken, 1)
            for key in ("atof_version", "kind"):
                broken = copy.deepcopy(valid)
                broken[0][key] = "invalid"
                cases[f"invalid-{key}"] = (broken, 1)
            if guide in {"claude-code", "codex"}:
                broken = copy.deepcopy(valid)
                for event in broken[:2]:
                    event["name"] = "unrelated-turn"
                cases["wrong-turn-name"] = (broken, 1)

            for binary in self.binaries:
                for label, (fixture, expected) in cases.items():
                    with self.subTest(guide=guide, jq=binary, fixture=label):
                        content = "".join(json.dumps(event) + "\n" for event in fixture)
                        result = self.run_jq(binary, "-se", expression, content)
                        self.assertEqual(result.returncode, expected, result.stderr)
                        self.assertEqual(result.stdout.strip(), "true" if expected == 0 else "false")
                with self.subTest(guide=guide, jq=binary, fixture="malformed-json"):
                    result = self.run_jq(binary, "-se", expression, "{broken json")
                    self.assertNotEqual(result.returncode, 0)

    def test_marker_verifiers_require_the_exact_tool_marker(self):
        for guide in GUIDES:
            expression, _ = self.expressions(guide)
            for binary in self.binaries:
                for content, expected in (
                    ("RELAY_QUICKSTART_TOOL_OK", 0),
                    ("  RELAY_QUICKSTART_TOOL_OK\n", 0),
                    ("", 1),
                    ("wrong marker", 1),
                    ("RELAY_QUICKSTART_TOOL_OK extra", 1),
                ):
                    with self.subTest(guide=guide, jq=binary, content=content):
                        result = self.run_jq(binary, "-eRs", expression, content)
                        self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
