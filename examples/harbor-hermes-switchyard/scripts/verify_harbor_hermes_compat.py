# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the temporary agent is a narrow Harbor Hermes compatibility bridge."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import tempfile
from pathlib import Path

from harbor.agents.installed.hermes import Hermes


def load_bridge(path: Path):
    spec = importlib.util.spec_from_file_location("harbor_hermes_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_mixed_mode_rejection(module, valid_config: Path) -> str:
    text = valid_config.read_text(encoding="utf-8")
    text += """

[[dynamic_plugins]]
plugin_id = "invalid.worker"
kind = "worker"
manifest_ref = "worker/relay-plugin.toml"
environment_ref = "worker-env"
"""
    with tempfile.TemporaryDirectory(prefix="harbor-hermes-mixed-") as directory:
        path = Path(directory) / "plugins.toml"
        path.write_text(text, encoding="utf-8")
        try:
            module._validate_relay_config(path)
        except ValueError as error:
            return str(error)
    raise AssertionError("mixed standard and worker plugin modes were accepted")


def verify_run_wrapper(source: str) -> None:
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    bridge = next(node for node in classes if node.name == "HarborHermesAgent")
    run_method = next(
        node for node in bridge.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    )
    super_run_calls = [
        node
        for node in ast.walk(run_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    ]
    if len(super_run_calls) != 1:
        raise AssertionError("bridge run() must delegate exactly once to super().run()")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--relay-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = load_bridge(args.bridge.resolve())
    bridge = module.HarborHermesAgent
    if not issubclass(bridge, Hermes):
        raise AssertionError("HarborHermesAgent must subclass Harbor's built-in Hermes")
    if bridge._build_config_yaml is not Hermes._build_config_yaml:
        raise AssertionError("bridge must inherit Harbor's Hermes config.yaml behavior")
    if bridge.populate_context_post_run is not Hermes.populate_context_post_run:
        raise AssertionError("bridge must inherit Harbor's ATIF conversion behavior")
    verify_run_wrapper(args.bridge.read_text(encoding="utf-8"))
    module._validate_relay_config(args.relay_config.resolve())
    mixed_error = run_mixed_mode_rejection(module, args.relay_config.resolve())

    result = {
        "schema_version": "harbor-hermes-switchyard.compatibility.v1",
        "status": "passed",
        "harbor_version": importlib.metadata.version("harbor"),
        "bridge_base": "harbor.agents.installed.hermes.Hermes",
        "inherited_contracts": [
            "task lifecycle",
            "provider environment",
            "prompt rendering",
            "Hermes session export",
            "Harbor ATIF conversion",
        ],
        "overrides": ["installation", "configuration staging", "artifact framing"],
        "mixed_mode_rejection": mixed_error,
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
