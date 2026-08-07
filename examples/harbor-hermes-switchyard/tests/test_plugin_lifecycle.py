# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = EXAMPLE_ROOT / "agents" / "harbor_hermes_agent.py"


def bridge_class() -> ast.ClassDef:
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HarborHermesAgent")


def method(name: str) -> ast.AsyncFunctionDef:
    node = next(item for item in bridge_class().body if isinstance(item, ast.AsyncFunctionDef) and item.name == name)
    return node


def super_calls(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == attribute
        and isinstance(item.func.value, ast.Call)
        and isinstance(item.func.value.func, ast.Name)
        and item.func.value.func.id == "super"
    ]


def test_bridge_delegates_exactly_once_to_each_inherited_lifecycle_phase() -> None:
    assert len(super_calls(method("setup"), "setup")) == 1
    assert len(super_calls(method("run"), "run")) == 1


def test_run_frames_artifacts_in_finally_after_inherited_run() -> None:
    run = method("run")
    tries = [item for item in run.body if isinstance(item, ast.Try)]
    assert len(tries) == 1
    lifecycle = tries[0]
    assert super_calls(lifecycle, "run")
    assert any(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "exec_as_agent"
        for final in lifecycle.finalbody
        for item in ast.walk(final)
    )


def test_install_verifies_detached_commit_and_relay_release() -> None:
    source = ast.unparse(method("install"))
    assert "checkout --detach" in source
    assert "rev-parse HEAD" in source
    assert "/tmp/hermes-install-path/ffmpeg" in source
    assert "uv sync --frozen --extra all" in source
    assert "m.version('nemo-relay').split('.')" in source


def test_setup_uploads_finalizer_with_its_relay_version_dependency() -> None:
    uploads = {
        (ast.unparse(call.args[0]), ast.literal_eval(call.args[1]))
        for call in ast.walk(method("setup"))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "upload_file"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Constant)
    }
    assert ("self._finalizer_path", "/installed-agent/finalize_artifacts.py") in uploads
    assert ("self._relay_version_path", "/installed-agent/relay_version.py") in uploads
