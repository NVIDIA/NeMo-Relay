# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the examples' real verification code without invoking models or frameworks."""

from __future__ import annotations

import ast
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "docs/getting-started/quick-start/_code"


def program(guide: str) -> ast.Module:
    return ast.parse((CODE / guide / "quickstart.py").read_text(encoding="utf-8"))


def paired_scopes(*identities: tuple[str, str]) -> list[dict]:
    events = []
    for index, (category, name) in enumerate(identities):
        for phase in ("start", "end"):
            events.append(
                {
                    "atof_version": "0.1",
                    "kind": "scope",
                    "scope_category": phase,
                    "category": category,
                    "name": name,
                    "uuid": str(index),
                    "parent_uuid": None if index == 0 else "0",
                }
            )
    return events


class QuickstartProgramTests(unittest.TestCase):
    def verify_events(self, guide: str, events: list[dict]) -> None:
        """Execute the checked-in post-run verifier, preserving its actual assertions."""
        main = next(node for node in program(guide).body if isinstance(node, ast.AsyncFunctionDef))
        start = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "events" for t in node.targets)
        )
        verifier = ast.Module(body=main.body[start:], type_ignores=[])
        content = "".join(json.dumps(event) + "\n" for event in events)
        output = Mock()
        namespace = {"json": json, "Path": lambda _: SimpleNamespace(read_text=lambda: content), "print": output}
        exec(compile(verifier, str(CODE / guide / "quickstart.py"), "exec"), namespace)
        output.assert_called_once_with("RELAY_QUICKSTART_EVENTS_OK")

    def test_named_scopes_are_required(self):
        cases = {
            "langgraph": (("agent", "langgraph-request"), ("agent", "emit_marker")),
            "deep-agents": (("agent", "deepagents-request"), ("tool", "emit_marker"), ("llm", "provider")),
        }
        for guide, identities in cases.items():
            valid = paired_scopes(*identities)
            with self.subTest(guide=guide, evidence="complete"):
                self.verify_events(guide, valid)
            for category, name in identities:
                missing = [event for event in valid if event["name"] != name]
                with self.subTest(guide=guide, evidence=f"missing-{name}"), self.assertRaises(RuntimeError):
                    self.verify_events(guide, missing)
                if category != "llm":
                    unrelated = copy.deepcopy(valid)
                    for event in unrelated:
                        if event["name"] == name:
                            event["name"] = "unrelated-scope"
                    with self.subTest(guide=guide, evidence=f"unrelated-{name}"), self.assertRaises(RuntimeError):
                        self.verify_events(guide, unrelated)
            broken = copy.deepcopy(valid)
            broken[-1]["parent_uuid"] = "different-parent"
            with self.subTest(guide=guide, evidence="mismatched-pair"), self.assertRaises(RuntimeError):
                self.verify_events(guide, broken)

    def test_langchain_marker_result_is_exactly_once(self):
        check = next(
            node
            for node in ast.walk(program("langchain"))
            if isinstance(node, ast.If) and "tool_outputs" in ast.unparse(node.test)
        )
        verifier = compile(ast.Module(body=[check], type_ignores=[]), "langchain-marker-check", "exec")
        marker = "RELAY_QUICKSTART_TOOL_OK"
        exec(verifier, {"tool_outputs": [marker]})
        for outputs in ([], ["wrong"], [marker, marker]):
            with self.subTest(outputs=outputs), self.assertRaises(RuntimeError):
                exec(verifier, {"tool_outputs": outputs})


if __name__ == "__main__":
    unittest.main()
