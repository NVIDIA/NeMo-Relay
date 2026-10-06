#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract quickstart contracts from MDX and verify their captured artifacts.

This deliberately does not execute documentation or provision authenticated hosts.
Commands, fixtures, and assertions belong in the page; this tool only interprets
the small shared annotation format. Requires Python 3.11+, no third-party packages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "docs/getting-started/quick-start"
SCENARIO = re.compile(r"\{/\* quickstart-scenario\s+(.*?)\*/\}", re.DOTALL)
BLOCK = re.compile(r"\{/\* quickstart-block\s+(.*?)\*/\}\s*$", re.DOTALL)
FENCE = re.compile(r"^```([\w+-]+)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
FENCE_MARKER = re.compile(r"^[ \t]*(?:`{3,}|~{3,})", re.MULTILINE)
ROLES = {"file", "run", "expected", "input", "manual"}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative_path(value: str) -> bool:
    """Contracts use portable, workspace-relative paths, not shell expansion."""
    return (
        bool(value)
        and not value.startswith("/")
        and not any(part == ".." or "\\" in part or ":" in part for part in value.split("/"))
    )


def extract(page: Path, shell: str | None = None) -> dict[str, Any]:
    text = page.read_text(encoding="utf-8")
    scenarios = SCENARIO.findall(text)
    require(len(scenarios) == 1, f"{page.name}: expected one scenario")
    scenario = json.loads(scenarios[0])
    require(scenario.get("schema") == 1, "unsupported quickstart schema")
    require(scenario.get("id") == page.stem, "scenario id must match the page filename")
    require(scenario.get("mode") in {"local", "live"}, "declare local or live execution")
    shells = scenario.get("shells", [])
    require(shells and len(shells) == len(set(shells)), "declare unique shell variants")
    require(set(shells) <= {"bash", "powershell"}, "unknown shell variant")
    require(shell is None or shell in shells, f"shell {shell!r} is not supported by this page")
    require(relative_path(scenario.get("cwd", "")), "unsafe scenario cwd")
    require(isinstance(scenario.get("requires"), dict), "declare prerequisite versions")
    require(isinstance(scenario.get("setup"), list), "declare external setup")
    require(isinstance(scenario.get("fresh"), list) and scenario["fresh"], "declare fresh artifacts")
    inputs = scenario.get("inputs", {})
    require(isinstance(inputs, dict), "inputs must be an object")
    for name, spec in inputs.items():
        require(re.fullmatch(r"[A-Z][A-Z0-9_]*", name), "invalid input name")
        require(spec.get("kind") in {"env", "path", "argument"}, "unknown input kind")
        if spec["kind"] != "env":
            require(bool(spec.get("token")), "non-environment inputs require a literal token")

    blocks = []
    previous = 0
    for fence in FENCE.finditer(text):
        # Search only between fences: code content can never become an annotation.
        prefix = text[previous : fence.start()]
        annotations = list(re.finditer(r"\{/\* quickstart-block\s+", prefix))
        require(len(annotations) == 1, f"{page.name}: each fence needs exactly one block annotation")
        annotation = BLOCK.fullmatch(prefix[annotations[0].start() :].rstrip())
        if annotation is None:
            raise ValueError("block annotation must immediately precede its fence")
        block = json.loads(annotation.group(1))
        require(block.get("role") in ROLES, "unknown block role")
        require(re.fullmatch(r"[a-z][a-z0-9-]*", block.get("id", "")), "invalid block id")
        require(relative_path(block.get("cwd", "")), "unsafe block cwd")
        variants = block.get("shells", shells)
        require(
            variants and len(variants) == len(set(variants)) and set(variants) <= set(shells),
            "invalid block shell variants",
        )
        role = block["role"]
        if role == "file":
            require(relative_path(block.get("path", "")), "unsafe artifact path")
        elif role == "run":
            require(
                isinstance(block.get("timeout_seconds"), int) and block["timeout_seconds"] > 0,
                "commands need a bounded timeout",
            )
            require(bool(block.get("session")), "commands need an explicit shell session")
            require(fence.group(1) in {"bash", "powershell"}, "commands must be shell blocks")
            if "background" in block:
                process = block["background"]
                require(
                    process.get("stop") == "interrupt" and process.get("shutdown_timeout_seconds", 0) > 0,
                    "background processes need bounded graceful cleanup",
                )
                require(
                    process.get("ready", {}).get("tcp") and process["ready"].get("timeout_seconds", 0) > 0,
                    "background processes need bounded readiness",
                )
        elif role == "input":
            require(inputs.get(block.get("env"), {}).get("kind") == "env", "undeclared environment input")
        elif role == "manual":
            require(bool(block.get("reason")), "manual setup needs an explicit reason")
        else:
            require(block.get("match") == "exact-lines" and fence.group(1) == "text", "unknown output matcher")
            require(bool(block.get("from")), "expected output must name its producing command")
        block.update(language=fence.group(1), content=fence.group(2), line=text[: fence.start()].count("\n") + 1)
        blocks.append(block)
        previous = fence.end()
    # Reject fences that the extractor cannot pair with a contract, including
    # indented, tilde, or longer-backtick variants that Markdown still renders.
    require(len(FENCE_MARKER.findall(text)) == len(blocks) * 2, "unsupported or unclosed code fence")
    require("quickstart-block" not in text[previous:], "orphan block annotation")
    require(bool(blocks), "no annotated code blocks")
    for variant in shells:
        selected = [block for block in blocks if variant in block.get("shells", shells)]
        ids = [block["id"] for block in selected]
        require(len(ids) == len(set(ids)), f"duplicate block id for {variant}")
        expected = [block for block in selected if block["role"] == "expected"]
        require(len(expected) == 1, f"declare one expected-output block for {variant}")
        commands = {block["id"] for block in selected if block["role"] == "run"}
        require(expected[0]["from"] in commands, "expected output refers to a missing command")
        for block in selected:
            if "background" in block:
                require(block["background"].get("stop_before") in commands, "missing shutdown boundary")
    checks = scenario.get("checks", [])
    require(checks and any(check.get("kind") == "atof" for check in checks), "declare ATOF success criteria")
    for check in checks:
        require(check.get("kind") in {"file-equals", "atof", "json-response"}, "unknown artifact check")
        validate_artifact_path(check.get("path", ""), inputs)
        if check["kind"] == "atof":
            require(check.get("version") == "0.1", "unsupported ATOF version")
            require(check.get("paired_scopes") or check.get("marks"), "ATOF check has no lifecycle assertions")
            for selector in check.get("paired_scopes", []):
                require(selector and set(selector) <= {"category", "name"}, "invalid scope selector")
        elif check["kind"] == "file-equals":
            require(isinstance(check.get("value"), str) and check["value"], "empty file assertion")
        else:
            require(check.get("role") == "assistant" and check.get("contains"), "declare assistant response assertion")
    for path in scenario["fresh"]:
        validate_artifact_path(path, inputs)
    scenario["blocks"] = [block for block in blocks if shell is None or shell in block.get("shells", shells)]
    return scenario


def validate_artifact_path(path: str, inputs: dict[str, Any]) -> None:
    tokens = [spec["token"] for spec in inputs.values() if spec["kind"] == "path"]
    require(
        relative_path(path)
        or any(path.startswith(token + "/") and relative_path(path[len(token) + 1 :]) for token in tokens),
        "unsafe or undeclared artifact path",
    )


def artifact_path(path: str, scenario: dict[str, Any], workspace: Path, values: dict[str, str]) -> Path:
    for name, spec in scenario["inputs"].items():
        if spec["kind"] == "path" and path.startswith(spec["token"] + "/"):
            require(name in values, f"supply --input {name}=<absolute directory>")
            base = Path(values[name])
            require(base.is_absolute(), f"{name} must be an absolute directory")
            return base / path[len(spec["token"]) + 1 :]
    return workspace / scenario["cwd"] / path


def verify_atof(text: str, check: dict[str, Any]) -> None:
    """Require real lifecycle fields; marker text inside arbitrary payloads is not evidence."""
    events = [json.loads(line) for line in text.splitlines()]
    require(events, "event file is empty")
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    for event in events:
        require(isinstance(event, dict), "event must be a JSON object")
        require(event.get("atof_version") == check["version"], "unexpected ATOF version")
        require(event.get("kind") in {"scope", "mark"}, "unexpected event kind")
        require(isinstance(event.get("uuid"), str), "missing event UUID")
        UUID(event["uuid"])
        require("parent_uuid" in event, "missing parent UUID field")
        if event["parent_uuid"] is not None:
            UUID(event["parent_uuid"])
        require(isinstance(event.get("name"), str) and event["name"], "missing event name")
        require(event.get("timestamp") is not None, "missing event timestamp")
        if event["kind"] == "scope":
            require(event.get("scope_category") in {"start", "end"}, "invalid scope phase")
            require(isinstance(event.get("category"), str), "missing scope category")
            target = starts if event["scope_category"] == "start" else ends
            require(event["uuid"] not in target, "duplicate scope phase")
            target[event["uuid"]] = event
    closed = []
    for uuid, end in ends.items():
        start = starts.get(uuid)
        if start:
            require(
                all(start.get(key) == end.get(key) for key in ("name", "category", "parent_uuid")),
                "inconsistent scope start/end identity",
            )
            closed.append(end)
    for selector in check.get("paired_scopes", []):
        require(
            any(all(event.get(key) == value for key, value in selector.items()) for event in closed),
            f"missing paired scope: {selector}",
        )
    for name in check.get("marks", []):
        require(any(event["kind"] == "mark" and event["name"] == name for event in events), f"missing mark: {name}")


def verify(scenario: dict[str, Any], workspace: Path, stdout: str, values: dict[str, str]) -> None:
    expected = next(block for block in scenario["blocks"] if block["role"] == "expected")
    require(
        stdout.splitlines() == expected["content"].splitlines(),
        f"stdout from {expected['from']} does not match the expected block",
    )
    for check in scenario["checks"]:
        path = artifact_path(check["path"], scenario, workspace, values)
        # Windows PowerShell can write a UTF-8 BOM. It is not meaningful JSON content.
        text = path.read_text(encoding="utf-8-sig")
        if check["kind"] == "atof":
            verify_atof(text, check)
        elif check["kind"] == "file-equals":
            actual = text.strip() if check.get("trim") else text
            require(actual == check["value"], f"incorrect tool artifact: {path}")
        else:
            response = json.loads(text)
            message = response["choices"][0]["message"]
            require(
                message.get("role") == check["role"]
                and isinstance(message.get("content"), str)
                and check["contains"] in message["content"],
                "incorrect assistant response",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "extract", "preflight", "verify"])
    parser.add_argument("page", type=Path, nargs="?")
    parser.add_argument("--shell", choices=["bash", "powershell"])
    parser.add_argument("--workspace", type=Path, help="fresh workspace containing the documented project directory")
    parser.add_argument("--stdout", type=Path, help="stdout only from the command named by the expected-output block")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH", help="non-secret path inputs")
    args = parser.parse_args()
    try:
        pages = [args.page] if args.page else sorted(path for path in PAGES.glob("*.mdx") if path.stem != "index")
        require(args.action == "check" or args.page is not None, "this action requires one page")
        scenarios = [extract(page, args.shell) for page in pages]
        if args.action == "check":
            print(f"Validated {len(scenarios)} quickstart contracts (no commands executed)")
            return 0
        scenario = scenarios[0]
        if args.action == "extract":
            require(args.shell, "select --shell to extract one executable branch")
            print(json.dumps(scenario, indent=2))
            return 0
        require(args.workspace, "supply --workspace")
        values = dict(value.split("=", 1) for value in args.input)
        require(
            all(scenario["inputs"].get(key, {}).get("kind") == "path" for key in values),
            "--input accepts only declared non-secret path inputs",
        )
        if args.action == "preflight":
            for path in scenario["fresh"]:
                require(not artifact_path(path, scenario, args.workspace, values).exists(), f"stale artifact: {path}")
            print("Fresh-artifact preflight passed; authentication and prerequisites still require provisioning")
        else:
            require(args.stdout, "supply --stdout from a successful command (exit code 0)")
            verify(scenario, args.workspace, args.stdout.read_text(encoding="utf-8-sig"), values)
            print("Quickstart artifacts and expected output verified")
        return 0
    except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
        print(f"Quickstart validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
