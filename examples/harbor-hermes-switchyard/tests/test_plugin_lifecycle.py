# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import yaml

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = EXAMPLE_ROOT / "agents" / "harbor_hermes_agent.py"


def bridge_class() -> ast.ClassDef:
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HarborHermesAgent")


def method(name: str) -> ast.AsyncFunctionDef:
    node = next(item for item in bridge_class().body if isinstance(item, ast.AsyncFunctionDef) and item.name == name)
    return node


def load_bridge():
    spec = importlib.util.spec_from_file_location("harbor_hermes_agent_config_test", BRIDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_bridge_disables_background_persistence_maintenance() -> None:
    bridge = load_bridge()
    inherited = yaml.safe_load(bridge.Hermes._build_config_yaml("openai/relay-stub"))
    config = yaml.safe_load(bridge.HarborHermesAgent._build_config_yaml("openai/relay-stub"))
    inherited.setdefault("skills", {})["creation_nudge_interval"] = 0
    inherited.setdefault("curator", {})["enabled"] = False
    assert config == inherited
    assert config["agent"]["max_turns"] == 90
    assert config["skills"]["creation_nudge_interval"] == 0
    assert config["curator"]["enabled"] is False


def test_direct_slash_qualified_baselines_use_custom_openai_compatible_provider() -> None:
    bridge = load_bridge()
    for model in (
        "openai/aws/anthropic/bedrock-claude-opus-4-8",
        "openai/openai/openai/gpt-5.6-sol",
    ):
        config = yaml.safe_load(bridge.HarborHermesAgent._build_config_yaml(model))
        assert config["provider"] == "custom"


def test_direct_opus_baseline_exports_host_scoped_nvidia_credential() -> None:
    source = (EXAMPLE_ROOT / "agents" / "harbor_hermes_agent.py").read_text(encoding="utf-8")
    assert 'export NVIDIA_API_KEY="$OPENAI_API_KEY"' in source


def test_direct_baseline_normalizes_the_configured_harbor_model_for_custom_provider() -> None:
    source = (EXAMPLE_ROOT / "agents" / "harbor_hermes_agent.py").read_text(encoding="utf-8")
    assert 'harbor_model = f"--model openai/{self.direct_model}"' in source
    assert 'provider_model = f"--model {self.direct_model} --provider custom"' in source


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
    assert '"$hermes_uv" sync --frozen --extra all' in source
    assert 'find /tmp/hermes/tools -mindepth 2 -maxdepth 2 -type f -name uv' in source
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
