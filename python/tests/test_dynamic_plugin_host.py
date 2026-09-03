# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the unified Python plugin-host API."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from nemo_relay import Json, LLMRequest, ToolExecutionResult, llm, plugin, scope, tools


@dataclass(frozen=True, slots=True)
class _BuiltPlugin:
    plugin_id: str
    kind: plugin.DynamicPluginKind
    manifest: Path


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


def _prepared_plugin_fixture(environment: str) -> Path:
    value = os.environ.get(environment)
    if value is not None:
        path = Path(value)
    else:
        filename = (
            _native_library_name()
            if environment == "NEMO_RELAY_TEST_NATIVE_PLUGIN"
            else "nemo-relay-worker-plugin-fixture" + (".exe" if sys.platform == "win32" else "")
        )
        path = _repo_root() / "target/test-plugin-fixtures/debug" / filename
    if not path.is_file():
        raise RuntimeError(f"missing plugin test fixture; run `just build-test-plugin-fixtures`: {path}")
    return path


@pytest.fixture(scope="session")
def native_dynamic_plugin(tmp_path_factory: pytest.TempPathFactory) -> _BuiltPlugin:
    manifest_dir = tmp_path_factory.mktemp("native-plugin-manifest")
    library = _prepared_plugin_fixture("NEMO_RELAY_TEST_NATIVE_PLUGIN")
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
            artifact = {json.dumps(str(library))}

            [integrity]
            sha256 = "sha256:{digest}"

            [load]
            library = {json.dumps(str(library))}
            symbol = "nemo_relay_fixture_native_plugin"
            """
        )
    )
    return _BuiltPlugin("fixture_native", "rust_dynamic", manifest)


@pytest.fixture(scope="session")
def worker_dynamic_plugin(tmp_path_factory: pytest.TempPathFactory) -> _BuiltPlugin:
    manifest_dir = tmp_path_factory.mktemp("worker-plugin-manifest")
    executable = _prepared_plugin_fixture("NEMO_RELAY_TEST_WORKER_PLUGIN")
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
            artifact = {json.dumps(str(executable))}

            [integrity]
            sha256 = "sha256:{digest}"

            [load]
            runtime = "rust"
            entrypoint = {json.dumps(str(executable))}
            """
        )
    )
    return _BuiltPlugin("fixture_worker", "worker", manifest)


def _write_plugins_toml(
    directory: Path,
    declarations: list[tuple[Path, dict[str, Json]]],
) -> Path:
    path = directory / "plugins.toml"
    sections = ['version = 1\n\n[plugins.policy.defaults]\nstartup = "required"\nattestation = "integrity_only"\n']
    for manifest, config in declarations:
        sections.append(f"\n[[plugins.dynamic]]\nmanifest = {json.dumps(str(manifest))}\n")
        if config:
            entries = ", ".join(f"{key} = {json.dumps(value)}" for key, value in config.items())
            sections.append(f"config = {{ {entries} }}\n")
    path.write_text("".join(sections))
    return path


async def _initialize(
    config: plugin.PluginConfig | dict[str, Json] | None = None,
    *declarations: tuple[_BuiltPlugin | Path, dict[str, Json]],
) -> plugin.PluginHostActivation:
    with tempfile.TemporaryDirectory(prefix="nemo-relay-plugin-host-") as directory:
        plugins_toml = _write_plugins_toml(
            Path(directory),
            [
                (item.manifest if isinstance(item, _BuiltPlugin) else item, declaration_config)
                for item, declaration_config in declarations
            ],
        )
        return await plugin.initialize(config or plugin.PluginConfig(), plugins_toml)


def test_removed_plugin_host_entry_points_are_not_exported():
    retired = {
        "load_dynamic_plugin_activation_specs",
        "validate_plugin_host",
        "validate_dynamic_plugins",
        "initialize_plugin_host",
        "initialize_with_dynamic_plugins",
        "clear",
        "configuration_report",
    }
    assert all(not hasattr(plugin, name) for name in retired)


def test_validate_initialize_and_activate_accept_the_same_arguments():
    initialize_parameters = tuple(inspect.signature(plugin.initialize).parameters.values())
    validate_parameters = tuple(inspect.signature(plugin.validate).parameters.values())
    activate_parameters = tuple(inspect.signature(plugin.activate).parameters.values())
    assert initialize_parameters == validate_parameters
    assert initialize_parameters == activate_parameters


def test_validate_uses_core_report(
    native_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
):
    plugins_toml = _write_plugins_toml(tmp_path, [(native_dynamic_plugin.manifest, {})])
    report = plugin.validate(plugin.PluginConfig(), plugins_toml)
    assert not any(item["level"] == "error" for item in report["config"]["diagnostics"])
    assert report["dynamic_plugins"][0]["plugin_id"] == native_dynamic_plugin.plugin_id
    assert report["dynamic_plugins"][0]["selected"] is True


async def test_native_host_owns_callbacks_and_close_is_idempotent(
    native_dynamic_plugin: _BuiltPlugin,
):
    activation = await _initialize(None, (native_dynamic_plugin, {}))
    assert activation.is_active
    assert activation.report["dynamic_plugins"][0]["plugin_id"] == "fixture_native"
    async with activation as active:
        result = await tools.execute(
            "python-native-fixture",
            {"input": True},
            lambda args: ToolExecutionResult({"args": args}),
        )
        assert active is activation
        assert result.result["native_plugin_tool_execution"] is True
        assert result.result["args"]["native_plugin_tool_execution_request"] is True
    assert not activation.is_active
    await activation.close()
    result = await tools.execute(
        "python-native-after-close",
        {"input": True},
        lambda args: ToolExecutionResult({"args": args}),
    )
    assert result.result == {"args": {"input": True}}


async def test_activate_closes_host_when_context_raises(
    native_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
):
    plugins_toml = _write_plugins_toml(tmp_path, [(native_dynamic_plugin.manifest, {})])
    activation = None

    with pytest.raises(RuntimeError, match="application failed"):
        async with plugin.activate(plugin.PluginConfig(), plugins_toml) as active:
            activation = active
            assert active.is_active
            raise RuntimeError("application failed")

    assert activation is not None
    assert not activation.is_active


async def test_discovered_configuration_layers_last(
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
                "mark-file-static-base", 0, False, lambda _name, args: {**args, "file_static_base": True}
            )

    user_config = tmp_path / "xdg" / "nemo-relay"
    user_config.mkdir(parents=True)
    plugins_toml = user_config / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[components]]
            kind = {json.dumps(static_kind)}
            enabled = true

            [plugins.policy.defaults]
            attestation = "integrity_only"

            [[plugins.dynamic]]
            manifest = {json.dumps(str(native_dynamic_plugin.manifest))}
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    plugin.register(static_kind, cast(plugin.Plugin, FileStaticPlugin()))
    activation = None
    try:
        activation = await plugin.initialize(plugin.PluginConfig())
        assert activation.report["config"]["diagnostics"][0]["code"] == "plugin.configuration_inherited"
        result = await tools.execute("python-file-static-base", {"input": True}, lambda args: ToolExecutionResult(args))
        assert result.result["file_static_base"] is True
        assert result.result["native_plugin_tool_execution"] is True
    finally:
        if activation is not None:
            await activation.close()
        plugin.deregister(static_kind)


async def test_concurrent_close_waiters_share_teardown(native_dynamic_plugin: _BuiltPlugin):
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
    activation = await _initialize(
        plugin.PluginConfig(components=[plugin.ComponentSpec(kind=plugin_kind)]),
        (native_dynamic_plugin, {}),
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
    finally:
        release.set()
        if second_close is not None:
            await asyncio.gather(second_close, return_exceptions=True)
        await activation.close()
        plugin.deregister(plugin_kind)


async def test_host_conflict_and_failed_preflight_leave_no_partial_activation(
    native_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
):
    activation = await _initialize(None, (native_dynamic_plugin, {}))
    try:
        with pytest.raises(RuntimeError, match="active dynamic plugin host"):
            await _initialize(None, (native_dynamic_plugin, {}))
    finally:
        await activation.close()

    missing = tmp_path / "missing-relay-plugin.toml"
    with pytest.raises(FileNotFoundError, match="missing-relay-plugin.toml"):
        await _initialize(None, (native_dynamic_plugin, {}), (missing, {}))
    assert "fixture_native" not in plugin.list_kinds()
    retry = await _initialize(None, (native_dynamic_plugin, {}))
    await retry.close()


async def test_invalid_dynamic_config_fails_closed(native_dynamic_plugin: _BuiltPlugin):
    with pytest.raises(ValueError, match="fixture rejection requested"):
        await _initialize(None, (native_dynamic_plugin, {"reject": True}))
    assert "fixture_native" not in plugin.list_kinds()


async def test_native_finalizer_releases_callbacks(native_dynamic_plugin: _BuiltPlugin):
    activation = await _initialize(None, (native_dynamic_plugin, {}))
    assert "fixture_native" in plugin.list_kinds()
    del activation
    await asyncio.sleep(0)
    gc.collect()
    for _ in range(100):
        if "fixture_native" not in plugin.list_kinds():
            break
        await asyncio.sleep(0.01)
    assert "fixture_native" not in plugin.list_kinds()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX worker stop/continue signals")
async def test_worker_finalizer_never_waits_on_python_thread(
    worker_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
):
    with worker_dynamic_plugin.manifest.open("rb") as file:
        worker_entrypoint = Path(tomllib.load(file)["load"]["entrypoint"])
    pid_file = tmp_path / "worker.pid"
    wrapper = tmp_path / "worker-wrapper.sh"
    wrapper.write_text(f"#!/bin/sh\nprintf '%s' \"$$\" > {str(pid_file)!r}\nexec {str(worker_entrypoint)!r}\n")
    wrapper.chmod(0o755)
    digest = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    original = worker_dynamic_plugin.manifest.read_text()
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text(
        original.replace(
            f"entrypoint = {json.dumps(str(worker_entrypoint))}",
            f"entrypoint = {json.dumps(str(wrapper))}",
        )
        .replace(
            f"artifact = {json.dumps(str(worker_entrypoint))}",
            f"artifact = {json.dumps(str(wrapper))}",
        )
        .replace(
            next(line for line in original.splitlines() if line.startswith("sha256 =")),
            f'sha256 = "sha256:{digest}"',
        )
    )

    activation = await _initialize(None, (manifest, {}))
    native_activation = getattr(activation, "_native")
    del activation
    await asyncio.sleep(0)
    gc.collect()
    worker_pid = int(pid_file.read_text())
    os.kill(worker_pid, signal.SIGSTOP)
    resumer = subprocess.Popen(["/bin/sh", "-c", 'sleep 0.8; kill -CONT "$1"', "resume-worker", str(worker_pid)])
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


async def test_worker_host_executes_and_releases_callbacks(worker_dynamic_plugin: _BuiltPlugin):
    activation = await _initialize(None, (worker_dynamic_plugin, {}))
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    async def tool_provider(args: Json) -> ToolExecutionResult[dict[str, Json]]:
        assert asyncio.get_running_loop() is loop
        assert threading.get_ident() == loop_thread
        return ToolExecutionResult({"args": args})

    async def llm_provider(request: LLMRequest) -> Json:
        assert asyncio.get_running_loop() is loop
        assert threading.get_ident() == loop_thread
        return {"request": request.content}

    async def stream_provider(request: LLMRequest):
        assert asyncio.get_running_loop() is loop
        assert threading.get_ident() == loop_thread
        yield {"request": request.content}

    try:
        result = await tools.execute("python-worker-fixture", {"input": True}, tool_provider)
        assert result.result["worker_plugin_tool_execution"] is True
        llm_result = await llm.execute("python-worker-llm", LLMRequest({}, {"model": "worker"}), llm_provider)
        assert llm_result["worker_plugin_llm_execution"] is True
        stream = await llm.stream_execute(
            "python-worker-stream",
            LLMRequest({}, {"model": "worker"}),
            stream_provider,
            lambda _chunk: None,
            lambda: {},
        )
        chunks = [chunk async for chunk in stream]
        assert chunks[0]["worker_plugin_llm_stream_execution"] is True
    finally:
        await activation.close()

    result = await tools.execute(
        "python-worker-after-close",
        {"input": True},
        lambda args: ToolExecutionResult({"args": args}),
    )
    assert result.result == {"args": {"input": True}}
