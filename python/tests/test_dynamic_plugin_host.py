# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Python-owned dynamic plugin activation."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import os
import subprocess
import sys
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from nemo_relay import Json, plugin, tools


@dataclass(frozen=True, slots=True)
class _BuiltPlugin:
    plugin_id: str
    kind: plugin.DynamicPluginKind
    manifest: Path

    def spec(self, **config: Json) -> plugin.DynamicPluginActivationSpec:
        return plugin.DynamicPluginActivationSpec(
            plugin_id=self.plugin_id,
            kind=self.kind,
            manifest_ref=str(self.manifest),
            config=config,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relay_version() -> str:
    with (_repo_root() / "Cargo.toml").open("rb") as file:
        return str(tomllib.load(file)["workspace"]["package"]["version"])


def _native_library_name() -> str:
    if sys.platform == "win32":
        return "nemo_relay_plugin_fixture.dll"
    if sys.platform == "darwin":
        return "libnemo_relay_plugin_fixture.dylib"
    return "libnemo_relay_plugin_fixture.so"


@pytest.fixture(scope="session")
def native_dynamic_plugin(tmp_path_factory: pytest.TempPathFactory) -> _BuiltPlugin:
    root = _repo_root()
    target = tmp_path_factory.mktemp("native-plugin-target")
    manifest_dir = tmp_path_factory.mktemp("native-plugin-manifest")
    subprocess.run(
        [
            os.environ.get("CARGO", "cargo"),
            "build",
            "--quiet",
            "--manifest-path",
            str(root / "crates/core/tests/fixtures/native_plugin/Cargo.toml"),
            "--target-dir",
            str(target),
        ],
        cwd=root,
        check=True,
    )
    library = target / "debug" / _native_library_name()
    assert library.is_file()
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    manifest = manifest_dir / "relay-plugin.toml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            manifest_version = 1

            [plugin]
            id = "fixture_native"
            kind = "rust_dynamic"

            [compat]
            relay = "={_relay_version()}"
            native_api = "1"

            [defaults]
            enabled = false

            [capabilities]
            items = ["plugin_native"]

            [integrity]
            sha256 = "sha256:{digest}"

            [load]
            library = {str(library)!r}
            symbol = "nemo_relay_fixture_native_plugin"
            """
        )
    )
    return _BuiltPlugin("fixture_native", "rust_dynamic", manifest)


@pytest.fixture(scope="session")
def worker_dynamic_plugin(tmp_path_factory: pytest.TempPathFactory) -> _BuiltPlugin:
    root = _repo_root()
    target = tmp_path_factory.mktemp("worker-plugin-target")
    manifest_dir = tmp_path_factory.mktemp("worker-plugin-manifest")
    subprocess.run(
        [
            os.environ.get("CARGO", "cargo"),
            "build",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(root / "crates/core/tests/fixtures/worker_plugin/Cargo.toml"),
            "--target-dir",
            str(target),
        ],
        cwd=root,
        check=True,
    )
    executable = target / "debug" / ("nemo-relay-worker-plugin-fixture" + (".exe" if sys.platform == "win32" else ""))
    assert executable.is_file()
    manifest = manifest_dir / "relay-plugin.toml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            manifest_version = 1

            [plugin]
            id = "fixture_worker"
            kind = "worker"

            [compat]
            relay = "={_relay_version()}"
            worker_protocol = "grpc-v1"

            [defaults]
            enabled = false

            [capabilities]
            items = ["plugin_worker"]

            [load]
            runtime = "rust"
            entrypoint = {str(executable)!r}
            """
        )
    )
    return _BuiltPlugin("fixture_worker", "worker", manifest)


def test_dynamic_plugin_activation_spec_serializes_canonical_shape():
    spec = plugin.DynamicPluginActivationSpec(
        plugin_id="example.plugin",
        kind="worker",
        manifest_ref="/plugins/example/relay-plugin.toml",
        environment_ref="/plugins/example/.venv",
        config={"enabled": True},
    )

    assert spec.to_dict() == {
        "plugin_id": "example.plugin",
        "kind": "worker",
        "manifest_ref": "/plugins/example/relay-plugin.toml",
        "environment_ref": "/plugins/example/.venv",
        "config": {"enabled": True},
    }


async def test_native_activation_context_owns_callbacks_and_close_is_idempotent(
    native_dynamic_plugin: _BuiltPlugin,
):
    activation = await plugin.activate_dynamic_plugins(plugin.PluginConfig(), [native_dynamic_plugin.spec()])
    assert activation.is_active
    assert activation.report == {"diagnostics": []}

    async with activation as active:
        result = await tools.execute("python-native-fixture", {"input": True}, lambda args: {"args": args})
        assert active is activation
        assert result["native_plugin_tool_execution"] is True
        assert result["args"]["native_plugin_tool_execution_request"] is True

    assert not activation.is_active
    await activation.close()
    result = await tools.execute("python-native-after-close", {"input": True}, lambda args: {"args": args})
    assert "native_plugin_tool_execution" not in result
    assert result == {"args": {"input": True}}


async def test_activation_reports_conflicts_and_rolls_back_partial_loads(
    native_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
):
    activation = await plugin.activate_dynamic_plugins({}, [native_dynamic_plugin.spec()])
    try:
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.activate_dynamic_plugins({}, [native_dynamic_plugin.spec()])
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize({})
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            plugin.clear()
    finally:
        await activation.close()

    missing = plugin.DynamicPluginActivationSpec(
        plugin_id="missing_native",
        kind="rust_dynamic",
        manifest_ref=str(tmp_path / "missing-relay-plugin.toml"),
    )
    with pytest.raises(FileNotFoundError, match="missing-relay-plugin.toml"):
        await plugin.activate_dynamic_plugins({}, [native_dynamic_plugin.spec(), missing])

    assert "fixture_native" not in plugin.list_kinds()
    retry = await plugin.activate_dynamic_plugins({}, [native_dynamic_plugin.spec()])
    await retry.close()


async def test_invalid_dynamic_inputs_raise_normal_python_exceptions(native_dynamic_plugin: _BuiltPlugin):
    with pytest.raises(ValueError, match="unknown variant"):
        await plugin.activate_dynamic_plugins(
            {},
            [
                {
                    "plugin_id": "invalid",
                    "kind": "invalid",
                    "manifest_ref": str(native_dynamic_plugin.manifest),
                }
            ],
        )

    with pytest.raises(ValueError, match="fixture rejection requested"):
        await plugin.activate_dynamic_plugins({}, [native_dynamic_plugin.spec(reject=True)])

    assert "fixture_native" not in plugin.list_kinds()


async def test_native_activation_finalizer_releases_callbacks(native_dynamic_plugin: _BuiltPlugin):
    activation = await plugin.activate_dynamic_plugins({}, [native_dynamic_plugin.spec()])
    assert "fixture_native" in plugin.list_kinds()

    del activation
    # The asyncio Future returned by the native binding retains its completed
    # result until the event loop processes the completion callback.
    await asyncio.sleep(0)
    gc.collect()

    assert "fixture_native" not in plugin.list_kinds()
    result = await tools.execute("python-native-after-finalize", {"input": True}, lambda args: args)
    assert result == {"input": True}


async def test_worker_activation_executes_and_releases_callbacks(worker_dynamic_plugin: _BuiltPlugin):
    activation = await plugin.activate_dynamic_plugins({}, [worker_dynamic_plugin.spec()])
    try:
        result = await tools.execute("python-worker-fixture", {"input": True}, lambda args: {"args": args})
        assert result["worker_plugin_tool_execution"] is True
        assert result["args"]["worker_plugin_tool_execution_request"] is True
    finally:
        await activation.close()

    assert not activation.is_active
    result = await tools.execute("python-worker-after-close", {"input": True}, lambda args: {"args": args})
    assert result == {"args": {"input": True}}
