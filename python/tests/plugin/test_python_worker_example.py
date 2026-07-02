# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python gRPC worker plugin example."""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

if os.environ.get("NEMO_RELAY_SKIP_PYTHON_PLUGIN_TESTS") == "1":
    pytest.skip("grpcio is unavailable for Python plugin SDK tests on this runner", allow_module_level=True)

pytest.importorskip("grpc")

from nemo_relay_plugin import PluginContext, PluginRuntime  # noqa: E402


def test_manifest_integrity_matches_artifact_bytes():
    example_root = Path(__file__).parents[3] / "examples/python-grpc-worker-plugin"
    manifest = tomllib.loads((example_root / "relay-plugin.toml").read_text(encoding="utf-8"))
    artifact = example_root / manifest["source"]["artifact"]

    actual = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    assert actual == manifest["integrity"]["sha256"]


@pytest.fixture(name="example", scope="module")
def example_fixture(tmp_path_factory: pytest.TempPathFactory) -> Any:
    example_root = Path(__file__).parents[3] / "examples/python-grpc-worker-plugin"
    manifest = tomllib.loads((example_root / "relay-plugin.toml").read_text(encoding="utf-8"))
    entrypoint = manifest["load"]["entrypoint"]
    module_name, separator, function_name = entrypoint.partition(":")
    assert separator == ":"
    assert function_name == "main"

    build_root = tmp_path_factory.mktemp("python-worker-example")
    project_root = build_root / "project"
    wheel_dir = build_root / "wheel"
    shutil.copytree(
        example_root,
        project_root,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", ".venv", "__pycache__", "*.py[cod]"),
    )
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir), str(project_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    package_name = module_name.partition(".")[0]

    def purge_example_modules() -> None:
        for loaded_name in tuple(sys.modules):
            if loaded_name == package_name or loaded_name.startswith(f"{package_name}."):
                sys.modules.pop(loaded_name, None)

    sys.path.insert(0, str(wheel))
    importlib.invalidate_caches()
    purge_example_modules()
    try:
        module = importlib.import_module(module_name)
        assert getattr(module, function_name) is module.main
        module_file = module.__file__
        assert module_file is not None
        assert Path(module_file).is_relative_to(wheel)
        yield module
    finally:
        purge_example_modules()
        sys.path.remove(str(wheel))


def test_example_validates_tag_configuration(example: Any):
    plugin = example.ExamplePythonWorker()

    assert plugin.validate({"tag": "demo"}) == []
    diagnostics = plugin.validate(None)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "examples.python_grpc_worker.invalid_config"
    diagnostics = plugin.validate({"reject": True})
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "examples.python_grpc_worker.rejected"
    with pytest.raises(ValueError, match="Python gRPC worker rejection requested"):
        plugin.register(None, {"reject": True})
    diagnostics = plugin.validate({"tag": 42})
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "examples.python_grpc_worker.invalid_tag"
    with pytest.raises(TypeError, match="plugin config must be a JSON object"):
        plugin.register(None, None)
    with pytest.raises(TypeError, match="tag must be a string"):
        plugin.register(None, {"tag": 42})


async def test_manifest_entrypoint_serves_example_plugin(
    example: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    served: list[Any] = []

    async def capture(plugin: Any) -> None:
        served.append(plugin)

    monkeypatch.setattr(example, "serve_plugin", capture)
    await example.main()

    assert len(served) == 1
    assert isinstance(served[0], example.ExamplePythonWorker)


async def test_example_register_propagates_configured_tag(example: Any):
    runtime = MagicMock(spec=PluginRuntime)
    runtime.emit_mark = AsyncMock()
    context = MagicMock(spec=PluginContext)
    context.runtime = runtime
    plugin = example.ExamplePythonWorker()
    plugin.register(context, {"tag": "demo"})

    context.register_tool_request_intercept.assert_called_once()
    name, callback = context.register_tool_request_intercept.call_args.args
    assert name == "tag_tool_request"

    assert await callback("lookup", {"query": "relay"}) == {
        "query": "relay",
        "_nemo_relay_plugin": {"tag": "demo"},
    }
    assert await callback("search", {"query": "plugins"}) == {
        "query": "plugins",
        "_nemo_relay_plugin": {"tag": "demo"},
    }
    assert await callback("collision", {"demo": False}) == {
        "demo": False,
        "_nemo_relay_plugin": {"tag": "demo"},
    }
    assert await callback("existing_metadata", {"_nemo_relay_plugin": {"existing": True}}) == {
        "_nemo_relay_plugin": {"existing": True, "tag": "demo"},
    }
    assert await callback("scalar", ["not", "an", "object"]) == ["not", "an", "object"]
    assert await callback("primitive", "relay") == "relay"
    scalar_metadata = {"_nemo_relay_plugin": "owned-by-caller"}
    array_metadata = {"_nemo_relay_plugin": ["owned", "by", "caller"]}
    assert await callback("scalar_metadata", scalar_metadata) is scalar_metadata
    assert await callback("array_metadata", array_metadata) is array_metadata
    assert runtime.emit_mark.await_args_list == [
        call(
            "examples.python_grpc_worker.tool_request",
            {"tool_name": tool_name, "source": "python-grpc-worker", "tag": "demo"},
        )
        for tool_name in (
            "lookup",
            "search",
            "collision",
            "existing_metadata",
            "scalar",
            "primitive",
            "scalar_metadata",
            "array_metadata",
        )
    ]
