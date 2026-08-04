# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Python-owned dynamic plugin activation."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_relay
from nemo_relay import Json, LLMRequest, llm, plugin, scope, tools


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


def _enable_hydrated_plugin(plugins_toml: Path, plugin_id: str) -> None:
    state_path = plugins_toml.with_name(".dynamic-plugins.json")
    state = json.loads(state_path.read_text())
    record = next(record for record in state["records"] if record["metadata"]["id"] == plugin_id)
    record["spec"]["enabled"] = True
    state_path.write_text(json.dumps(state, indent=2) + "\n")


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
    built_library = target / "debug" / _native_library_name()
    assert built_library.is_file()
    library = manifest_dir / built_library.name
    shutil.copy2(built_library, library)
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

            [source]
            artifact = {library.name!r}

            [integrity]
            sha256 = "sha256:{digest}"

            [load]
            library = {library.name!r}
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
    built_executable = (
        target / "debug" / ("nemo-relay-worker-plugin-fixture" + (".exe" if sys.platform == "win32" else ""))
    )
    assert built_executable.is_file()
    executable = manifest_dir / built_executable.name
    shutil.copy2(built_executable, executable)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
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

            [source]
            artifact = {executable.name!r}

            [integrity]
            sha256 = "sha256:{digest}"

            [load]
            runtime = "rust"
            entrypoint = {executable.name!r}
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


def test_dynamic_plugin_activation_spec_preserves_nested_json_nulls():
    spec = plugin.DynamicPluginActivationSpec(
        plugin_id="example.plugin",
        kind="worker",
        manifest_ref="/plugins/example/relay-plugin.toml",
        config={
            "top_level": None,
            "nested": {"value": None},
            "items": [None, {"value": None}],
        },
    )

    assert spec.to_dict()["config"] == {
        "top_level": None,
        "nested": {"value": None},
        "items": [None, {"value": None}],
    }


def test_validate_omits_raw_plugin_config_nulls_but_preserves_component_config_nulls(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[object] = []

    def validate(config: object) -> object:
        captured.append(config)
        return {"diagnostics": []}

    monkeypatch.setattr(plugin, "_validate_plugin_config", validate)

    assert plugin.validate(
        {
            "version": None,
            "policy": {"unknown_component": None},
            "components": [
                {
                    "kind": "example",
                    "enabled": None,
                    "config": {"top_level": None, "nested": {"value": None}},
                }
            ],
        }
    ) == {"diagnostics": []}
    assert captured == [
        {
            "policy": {},
            "components": [
                {
                    "kind": "example",
                    "config": {"top_level": None, "nested": {"value": None}},
                }
            ],
        }
    ]


def test_validate_raw_plugin_config_nulls_as_omitted_fields():
    assert plugin.validate({"version": None, "components": None, "policy": None}) == {"diagnostics": []}


async def test_initialize_omits_raw_plugin_config_nulls_but_preserves_component_config_nulls(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[object] = []

    async def initialize(config: object) -> object:
        captured.append(config)
        return {"diagnostics": []}

    monkeypatch.setattr(plugin, "_initialize_plugins", initialize)

    assert await plugin.initialize(
        {
            "version": None,
            "components": [
                {
                    "kind": "example",
                    "config": {"top_level": None, "items": [None, {"value": None}]},
                }
            ],
        }
    ) == {"diagnostics": []}
    assert captured == [
        {
            "components": [
                {
                    "kind": "example",
                    "config": {"top_level": None, "items": [None, {"value": None}]},
                }
            ],
        }
    ]


async def test_initialize_from_plugins_toml_normalizes_path_and_owns_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured: list[tuple[object | None, str | None]] = []
    mock_native = MagicMock()
    mock_native.report = {"diagnostics": []}
    mock_native.is_active = True

    async def close() -> None:
        mock_native.is_active = False

    mock_native.close = AsyncMock(side_effect=close)

    async def initialize(
        config: object | None,
        *,
        plugin_config_path: str | None,
    ) -> MagicMock:
        captured.append((config, plugin_config_path))
        return mock_native

    monkeypatch.setattr(plugin, "_initialize_from_plugins_toml", initialize)
    path = tmp_path / "plugins.toml"

    activation = await plugin.initialize_from_plugins_toml(
        plugin.PluginConfig(),
        plugin_config_path=path,
    )

    assert captured == [(plugin.PluginConfig().to_dict(), str(path))]
    assert activation.report == {"diagnostics": []}
    assert activation.is_active
    async with activation as active:
        assert active is activation
    assert not activation.is_active
    await activation.close()


async def test_initialize_from_plugins_toml_preserves_absent_config_and_path(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[tuple[object | None, str | None]] = []
    mock_native = MagicMock()
    mock_native.report = {"diagnostics": []}
    mock_native.is_active = False
    mock_native.close = AsyncMock(return_value=None)

    async def initialize(
        config: object | None,
        *,
        plugin_config_path: str | None,
    ) -> MagicMock:
        captured.append((config, plugin_config_path))
        return mock_native

    monkeypatch.setattr(plugin, "_initialize_from_plugins_toml", initialize)

    activation = await plugin.initialize_from_plugins_toml()

    assert captured == [(None, None)]
    assert not activation.is_active
    await activation.close()


async def test_empty_dynamic_specs_preserve_static_initialization_path():
    with pytest.raises(ValueError, match="at least one dynamic plugin"):
        await plugin.initialize_with_dynamic_plugins(plugin.PluginConfig(), [])

    assert plugin.report() is None
    report = await plugin.initialize(plugin.PluginConfig())
    assert report == {"diagnostics": []}
    await plugin.clear_async()


def test_file_initializer_is_exported_only_from_plugin_module():
    assert plugin.initialize_from_plugins_toml is not None
    assert plugin.PluginFileActivation is not None
    assert not hasattr(nemo_relay, "initialize_from_plugins_toml")
    assert not hasattr(nemo_relay, "PluginFileActivation")


async def test_file_activation_without_input_is_inactive_and_user_scope_skips_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    static_kind = "python.fixture.user-scope-project"

    class ProjectPlugin:
        def validate(self, _plugin_config):
            return None

        def register(self, _plugin_config, context):
            context.register_tool_request_intercept(
                "mark-project",
                0,
                False,
                lambda _name, args: {**args, "project_loaded": True},
            )

    project_config = tmp_path / ".nemo-relay"
    project_config.mkdir()
    (project_config / "plugins.toml").write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[components]]
            kind = {static_kind!r}
            enabled = true
            """
        )
    )
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")

    plugin.register(static_kind, cast(plugin.Plugin, ProjectPlugin()))
    try:
        activation = await plugin.initialize_from_plugins_toml()
        assert not activation.is_active
        assert activation.report == {"diagnostics": []}
        result = await tools.execute("python-file-user-scope", {"input": True}, lambda args: args)
        assert result == {"input": True}
        await activation.close()
        await activation.close()
        assert not activation.is_active
    finally:
        plugin.deregister(static_kind)


async def test_explicit_empty_config_acquires_file_activation_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")

    activation = await plugin.initialize_from_plugins_toml(
        plugin.PluginConfig(),
        plugin_config_path=tmp_path / "missing-plugins.toml",
    )
    try:
        assert activation.is_active
        assert activation.report == {"diagnostics": []}
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize_from_plugins_toml(plugin.PluginConfig())
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize(plugin.PluginConfig())
        missing = plugin.DynamicPluginActivationSpec(
            plugin_id="python.fixture.blocked-explicit-host",
            kind="rust_dynamic",
            manifest_ref=str(tmp_path / "missing-relay-plugin.toml"),
        )
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize_with_dynamic_plugins(plugin.PluginConfig(), [missing])
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await asyncio.to_thread(plugin.clear)
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.clear_async()
    finally:
        await activation.close()
    assert not activation.is_active


async def test_existing_empty_file_acquires_owned_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text("")
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")

    activation = await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
    try:
        assert activation.is_active
        assert activation.report["diagnostics"][0]["code"] == "plugin.configuration_inherited"
    finally:
        await activation.close()


async def test_missing_declared_manifest_maps_to_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text('[[plugins.dynamic]]\nmanifest = "missing/relay-plugin.toml"\n')
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")

    with pytest.raises(FileNotFoundError, match="missing/relay-plugin.toml"):
        await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
    assert not plugins_toml.with_name(".dynamic-plugins.json").exists()


async def test_selected_static_only_file_is_owned_until_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    static_kind = "python.fixture.file-static-owned"

    class FileStaticPlugin:
        def validate(self, _plugin_config):
            return None

        def register(self, _plugin_config, context):
            context.register_tool_request_intercept(
                "mark-file-static-owned",
                0,
                False,
                lambda _name, args: {**args, "file_static_owned": True},
            )

    plugins_toml = tmp_path / "selected" / "plugins.toml"
    plugins_toml.parent.mkdir()
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[components]]
            kind = {static_kind!r}
            enabled = true
            """
        )
    )
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.delenv("NEMO_RELAY_CONFIG_SCOPE", raising=False)

    plugin.register(static_kind, cast(plugin.Plugin, FileStaticPlugin()))
    activation = None
    try:
        activation = await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
        assert activation.is_active
        assert activation.report["diagnostics"][0]["code"] == "plugin.configuration_inherited"
        result = await tools.execute("python-file-static-owned", {"input": True}, lambda args: args)
        assert result == {"input": True, "file_static_owned": True}
    finally:
        if activation is not None:
            await activation.close()
        plugin.deregister(static_kind)

    result = await tools.execute("python-file-static-after-close", {"input": True}, lambda args: args)
    assert result == {"input": True}


async def test_file_activation_finalizer_releases_static_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    static_kind = "python.fixture.file-static-finalizer"

    class FileStaticPlugin:
        def validate(self, _plugin_config):
            return None

        def register(self, _plugin_config, context):
            context.register_tool_request_intercept(
                "mark-file-static-finalizer",
                0,
                False,
                lambda _name, args: {**args, "file_static_finalizer": True},
            )

    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[components]]
            kind = {static_kind!r}
            enabled = true
            """
        )
    )
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")

    plugin.register(static_kind, cast(plugin.Plugin, FileStaticPlugin()))
    try:
        activation = await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
        result = await tools.execute("python-file-finalizer-active", {"input": True}, lambda args: args)
        assert result["file_static_finalizer"] is True

        del activation
        await asyncio.sleep(0)
        gc.collect()
        for _ in range(100):
            result = await tools.execute("python-file-finalizer-poll", {"input": True}, lambda args: args)
            if "file_static_finalizer" not in result:
                break
            await asyncio.sleep(0.01)
        assert result == {"input": True}
    finally:
        plugin.deregister(static_kind)


async def test_file_activation_hydrates_disabled_then_loads_enabled_native_plugin(
    native_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    project_config = tmp_path / ".nemo-relay"
    project_config.mkdir()
    plugins_toml = project_config / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[plugins.dynamic]]
            manifest = {native_dynamic_plugin.manifest.as_posix()!r}
            """
        )
    )
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.delenv("NEMO_RELAY_CONFIG_SCOPE", raising=False)

    hydrated = await plugin.initialize_from_plugins_toml()
    assert hydrated.is_active
    assert hydrated.report == {
        "diagnostics": [
            {
                "level": "warning",
                "code": "plugin.configuration_inherited",
                "message": f"inherited plugin configuration from discovered file: {plugins_toml.resolve()}",
            }
        ]
    }
    result = await tools.execute("python-file-disabled", {"input": True}, lambda args: args)
    assert result == {"input": True}
    await hydrated.close()

    _enable_hydrated_plugin(plugins_toml, native_dynamic_plugin.plugin_id)

    activation = await plugin.initialize_from_plugins_toml()
    try:
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize(plugin.PluginConfig())
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize_with_dynamic_plugins(
                plugin.PluginConfig(),
                [native_dynamic_plugin.spec()],
            )
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await asyncio.to_thread(plugin.clear)
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.clear_async()

        result = await tools.execute("python-file-enabled", {"input": True}, lambda args: {"args": args})
        assert result["native_plugin_tool_execution"] is True
    finally:
        await activation.close()


async def test_file_activation_loads_enabled_grpc_worker(
    worker_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    project_config = tmp_path / ".nemo-relay"
    project_config.mkdir()
    plugins_toml = project_config / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[plugins.dynamic]]
            manifest = {worker_dynamic_plugin.manifest.as_posix()!r}
            """
        )
    )
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.delenv("NEMO_RELAY_CONFIG_SCOPE", raising=False)

    hydrated = await plugin.initialize_from_plugins_toml()
    await hydrated.close()
    _enable_hydrated_plugin(plugins_toml, worker_dynamic_plugin.plugin_id)

    activation = await plugin.initialize_from_plugins_toml()
    try:
        result = await tools.execute("python-file-worker", {"input": True}, lambda args: {"args": args})
        assert result["worker_plugin_tool_execution"] is True
        assert result["args"]["worker_plugin_tool_execution_request"] is True
    finally:
        await activation.close()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX lifecycle file locking")
async def test_file_activation_cancellation_before_activation_enqueue_never_loads_code(
    native_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import fcntl

    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[plugins.dynamic]]
            manifest = {native_dynamic_plugin.manifest.as_posix()!r}
            """
        )
    )
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")

    hydrated = await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
    await hydrated.close()
    _enable_hydrated_plugin(plugins_toml, native_dynamic_plugin.plugin_id)

    lock_path = plugins_toml.with_name(".dynamic-plugins.lock")
    native_submitted = asyncio.Event()
    initialize_native = getattr(plugin, "_initialize_from_plugins_toml")

    async def observe_native_submission(
        config: object | None,
        *,
        plugin_config_path: str | None,
    ) -> object:
        pending = initialize_native(config, plugin_config_path=plugin_config_path)
        native_submitted.set()
        return await pending

    monkeypatch.setattr(plugin, "_initialize_from_plugins_toml", observe_native_submission)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            activation_task = asyncio.create_task(plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml))
            await asyncio.wait_for(native_submitted.wait(), timeout=5)
            activation_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await activation_task

            # Lifecycle reconciliation cannot complete while this lock is held,
            # so cancellation here is necessarily before the core activation
            # plan is queued and no dynamic code can have loaded.
            assert native_dynamic_plugin.plugin_id not in plugin.list_kinds()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    retry = await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
    try:
        result = await tools.execute("python-file-cancel-before-enqueue", {"input": True}, lambda args: args)
        assert result["native_plugin_tool_execution"] is True
    finally:
        await retry.close()


async def test_file_activation_cancellation_after_enqueue_cleans_undelivered_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    static_kind = "python.fixture.file-cancel-after-enqueue"
    register_started = threading.Event()
    register_release = threading.Event()
    register_finished = threading.Event()

    class BlockingFilePlugin:
        def validate(self, _plugin_config):
            return None

        def register(self, _plugin_config, context):
            register_started.set()
            assert register_release.wait(timeout=5)
            context.register_tool_request_intercept(
                "mark-file-cancel-after-enqueue",
                0,
                False,
                lambda _name, args: {**args, "file_cancel_after_enqueue": True},
            )
            register_finished.set()

    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[components]]
            kind = {static_kind!r}
            enabled = true
            """
        )
    )
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")

    plugin.register(static_kind, cast(plugin.Plugin, BlockingFilePlugin()))
    activation_task = asyncio.create_task(plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml))
    recovered = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(register_started.wait, 5), timeout=6)
        activation_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await activation_task

        register_release.set()
        assert await asyncio.wait_for(asyncio.to_thread(register_finished.wait, 5), timeout=6)

        async with asyncio.timeout(5):
            while recovered is None:
                try:
                    recovered = await plugin.initialize_from_plugins_toml(
                        plugin.PluginConfig(),
                        plugin_config_path=tmp_path / "missing-plugins.toml",
                    )
                except RuntimeError as error:
                    assert "active dynamic plugin host" in str(error)
                    await asyncio.sleep(0)

        result = await tools.execute("python-file-cancel-after-enqueue", {"input": True}, lambda args: args)
        assert result == {"input": True}
    finally:
        register_release.set()
        await asyncio.gather(activation_task, return_exceptions=True)
        if recovered is not None:
            await recovered.close()
        plugin.deregister(static_kind)


async def test_file_activation_mixed_static_native_and_command_worker_rolls_back_and_retries(
    native_dynamic_plugin: _BuiltPlugin,
    worker_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    static_kind = "python.fixture.file-mixed-host"

    class MixedFilePlugin:
        def validate(self, _plugin_config):
            return None

        def register(self, _plugin_config, context):
            context.register_tool_request_intercept(
                "mark-file-mixed-host",
                0,
                False,
                lambda _name, args: {**args, "file_mixed_static": True},
            )

    def write_plugins_toml(*, worker_register_error: bool) -> None:
        worker_config = "config = { register_error = true }" if worker_register_error else ""
        plugins_toml.write_text(
            textwrap.dedent(
                f"""
                version = 1

                [[components]]
                kind = {static_kind!r}
                enabled = true

                [[plugins.dynamic]]
                manifest = {native_dynamic_plugin.manifest.as_posix()!r}

                [[plugins.dynamic]]
                manifest = {worker_dynamic_plugin.manifest.as_posix()!r}
                {worker_config}
                """
            )
        )

    plugins_toml = tmp_path / "plugins.toml"
    write_plugins_toml(worker_register_error=True)
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")

    plugin.register(static_kind, cast(plugin.Plugin, MixedFilePlugin()))
    activation = None
    try:
        hydrated = await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
        await hydrated.close()
        _enable_hydrated_plugin(plugins_toml, native_dynamic_plugin.plugin_id)
        _enable_hydrated_plugin(plugins_toml, worker_dynamic_plugin.plugin_id)

        with pytest.raises(RuntimeError, match="fixture registration error requested"):
            await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)

        assert native_dynamic_plugin.plugin_id not in plugin.list_kinds()
        assert worker_dynamic_plugin.plugin_id not in plugin.list_kinds()
        result = await tools.execute("python-file-mixed-after-rollback", {"input": True}, lambda args: args)
        assert result == {"input": True}

        write_plugins_toml(worker_register_error=False)
        activation = await plugin.initialize_from_plugins_toml(plugin_config_path=plugins_toml)
        result = await tools.execute("python-file-mixed-success", {"input": True}, lambda args: {"args": args})
        assert result["native_plugin_tool_execution"] is True
        assert result["worker_plugin_tool_execution"] is True
        assert result["args"]["native_plugin_tool_execution_request"] is True
        assert result["args"]["worker_plugin_tool_execution_request"] is True
        assert result["args"]["file_mixed_static"] is True
    finally:
        if activation is not None:
            await activation.close()
        plugin.deregister(static_kind)

    assert native_dynamic_plugin.plugin_id not in plugin.list_kinds()
    assert worker_dynamic_plugin.plugin_id not in plugin.list_kinds()
    result = await tools.execute("python-file-mixed-after-close", {"input": True}, lambda args: args)
    assert result == {"input": True}


async def test_native_activation_context_owns_callbacks_and_close_is_idempotent(
    native_dynamic_plugin: _BuiltPlugin,
):
    activation = await plugin.initialize_with_dynamic_plugins(plugin.PluginConfig(), [native_dynamic_plugin.spec()])
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


async def test_dynamic_activation_layers_plugins_toml_static_components(
    native_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    static_kind = "python.fixture.file-static-base"

    class FileStaticPlugin:
        def validate(self, _plugin_config):
            return None

        def register(self, _plugin_config, context):
            context.register_tool_request_intercept(
                "mark-file-static-base",
                0,
                False,
                lambda _name, args: {**args, "file_static_base": True},
            )

    project_config = tmp_path / ".nemo-relay"
    project_config.mkdir()
    plugins_toml = project_config / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[components]]
            kind = {static_kind!r}
            enabled = true
            """
        )
    )
    isolated_user_config = tmp_path / "xdg"
    isolated_user_config.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))

    plugin.register(static_kind, cast(plugin.Plugin, FileStaticPlugin()))
    activation = None
    try:
        activation = await plugin.initialize_with_dynamic_plugins(plugin.PluginConfig(), [native_dynamic_plugin.spec()])
        assert activation.report == {
            "diagnostics": [
                {
                    "level": "warning",
                    "code": "plugin.configuration_inherited",
                    "message": f"inherited plugin configuration from discovered file: {plugins_toml.resolve()}",
                }
            ]
        }
        result = await tools.execute("python-file-static-base", {"input": True}, lambda args: args)
        assert result["file_static_base"] is True
        assert result["native_plugin_tool_execution"] is True
    finally:
        if activation is not None:
            await activation.close()
        plugin.deregister(static_kind)


async def test_concurrent_close_waiters_share_cancellation_resistant_teardown(
    native_dynamic_plugin: _BuiltPlugin,
):
    started = threading.Event()
    release = threading.Event()
    plugin_kind = "python.dynamic_close_waiter"

    class BlockingSubscriberPlugin:
        def validate(self, _plugin_config):
            return None

        def register(self, _plugin_config, context):
            def block(_event):
                started.set()
                assert release.wait(timeout=5)

            context.register_subscriber("block_teardown", block)

    plugin.register(plugin_kind, cast(plugin.Plugin, BlockingSubscriberPlugin()))
    activation = await plugin.initialize_with_dynamic_plugins(
        {
            "version": 1,
            "components": [{"kind": plugin_kind, "config": {}}],
        },
        [native_dynamic_plugin.spec()],
    )
    second_close: asyncio.Task[None] | None = None
    try:
        scope.event("python-dynamic-close-blocker")
        assert await asyncio.to_thread(started.wait, 2)

        first_close = asyncio.create_task(activation.close())
        while activation.is_active:
            await asyncio.sleep(0)
        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close

        second_close = asyncio.create_task(activation.close())
        await asyncio.sleep(0.05)
        assert not second_close.done()

        release.set()
        await second_close
        await activation.close()
        assert not activation.is_active
    finally:
        release.set()
        if second_close is not None:
            await asyncio.gather(second_close, return_exceptions=True)
        await activation.close()
        plugin.deregister(plugin_kind)


async def test_activation_reports_conflicts_and_rolls_back_partial_loads(
    native_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
):
    activation = await plugin.initialize_with_dynamic_plugins({}, [native_dynamic_plugin.spec()])
    try:
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize_with_dynamic_plugins({}, [native_dynamic_plugin.spec()])
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.initialize({})
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await plugin.clear_async()
    finally:
        await activation.close()

    missing = plugin.DynamicPluginActivationSpec(
        plugin_id="missing_native",
        kind="rust_dynamic",
        manifest_ref=str(tmp_path / "missing-relay-plugin.toml"),
    )
    with pytest.raises(FileNotFoundError, match="missing-relay-plugin.toml"):
        await plugin.initialize_with_dynamic_plugins({}, [native_dynamic_plugin.spec(), missing])

    assert "fixture_native" not in plugin.list_kinds()
    retry = await plugin.initialize_with_dynamic_plugins({}, [native_dynamic_plugin.spec()])
    await retry.close()


async def test_invalid_dynamic_inputs_raise_normal_python_exceptions(native_dynamic_plugin: _BuiltPlugin):
    with pytest.raises(ValueError, match="unknown variant"):
        await plugin.initialize_with_dynamic_plugins(
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
        await plugin.initialize_with_dynamic_plugins({}, [native_dynamic_plugin.spec(reject=True)])

    assert "fixture_native" not in plugin.list_kinds()


async def test_native_activation_finalizer_releases_callbacks(native_dynamic_plugin: _BuiltPlugin):
    activation = await plugin.initialize_with_dynamic_plugins({}, [native_dynamic_plugin.spec()])
    assert "fixture_native" in plugin.list_kinds()

    del activation
    # The asyncio Future returned by the native binding retains its completed
    # result until the event loop processes the completion callback.
    await asyncio.sleep(0)
    gc.collect()

    for _ in range(100):
        if "fixture_native" not in plugin.list_kinds():
            break
        await asyncio.sleep(0.01)
    assert "fixture_native" not in plugin.list_kinds()
    result = await tools.execute("python-native-after-finalize", {"input": True}, lambda args: args)
    assert result == {"input": True}


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX worker stop/continue signals")
async def test_worker_activation_finalizer_never_waits_on_python_thread(
    worker_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
):
    with worker_dynamic_plugin.manifest.open("rb") as file:
        worker_entrypoint_value = str(tomllib.load(file)["load"]["entrypoint"])
    worker_entrypoint = Path(worker_entrypoint_value)
    if not worker_entrypoint.is_absolute():
        worker_entrypoint = worker_dynamic_plugin.manifest.parent / worker_entrypoint

    pid_file = tmp_path / "worker.pid"
    wrapper = tmp_path / "worker-wrapper.sh"
    wrapper.write_text(f"#!/bin/sh\nprintf '%s' \"$$\" > {str(pid_file)!r}\nexec {str(worker_entrypoint)!r}\n")
    wrapper.chmod(0o755)
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text(
        worker_dynamic_plugin.manifest.read_text().replace(
            f"entrypoint = {worker_entrypoint_value!r}",
            f"entrypoint = {str(wrapper)!r}",
        )
    )

    activation = await plugin.initialize_with_dynamic_plugins(
        {},
        [_BuiltPlugin("fixture_worker", "worker", manifest).spec()],
    )
    native_activation = getattr(activation, "_native")
    del activation
    await asyncio.sleep(0)
    gc.collect()

    worker_pid = int(pid_file.read_text())
    os.kill(worker_pid, signal.SIGSTOP)
    resumer = subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            'sleep 0.8; kill -CONT "$1"',
            "resume-worker",
            str(worker_pid),
        ]
    )
    started_at = time.perf_counter()
    try:
        del native_activation
        gc.collect()
        elapsed = time.perf_counter() - started_at
    finally:
        try:
            os.kill(worker_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        resumer.wait(timeout=5)

    assert elapsed < 0.4
    for _ in range(500):
        if "fixture_worker" not in plugin.list_kinds():
            break
        await asyncio.sleep(0.01)
    assert "fixture_worker" not in plugin.list_kinds()


async def test_worker_activation_executes_and_releases_callbacks(worker_dynamic_plugin: _BuiltPlugin):
    activation = await plugin.initialize_with_dynamic_plugins({}, [worker_dynamic_plugin.spec()])
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    async def tool_provider(args: Json) -> Json:
        assert asyncio.get_running_loop() is loop
        assert threading.get_ident() == loop_thread
        await asyncio.sleep(0)
        return {"args": args}

    async def llm_provider(request: LLMRequest) -> Json:
        assert asyncio.get_running_loop() is loop
        assert threading.get_ident() == loop_thread
        await asyncio.sleep(0)
        return {"request": request.content}

    async def stream_provider(request: LLMRequest):
        assert asyncio.get_running_loop() is loop
        assert threading.get_ident() == loop_thread
        await asyncio.sleep(0)
        yield {"request": request.content}

    try:
        result = await tools.execute("python-worker-fixture", {"input": True}, tool_provider)
        assert result["worker_plugin_tool_execution"] is True
        assert result["args"]["worker_plugin_tool_execution_request"] is True

        llm_result = await llm.execute(
            "python-worker-llm",
            LLMRequest({}, {"model": "worker"}),
            llm_provider,
        )
        assert llm_result["worker_plugin_llm_execution"] is True
        assert llm_result["request"]["worker_plugin_llm_execution_request"] is True

        stream = await llm.stream_execute(
            "python-worker-stream",
            LLMRequest({}, {"model": "worker"}),
            stream_provider,
            lambda _chunk: None,
            lambda: {},
        )
        chunks = [chunk async for chunk in stream]
        assert chunks
        chunk = chunks[0]
        assert isinstance(chunk, dict)
        assert chunk["worker_plugin_llm_stream_execution"] is True
        request = chunk["request"]
        assert isinstance(request, dict)
        assert request["worker_plugin_llm_stream_execution_request"] is True
    finally:
        await activation.close()

    assert not activation.is_active
    result = await tools.execute("python-worker-after-close", {"input": True}, lambda args: {"args": args})
    assert result == {"args": {"input": True}}
