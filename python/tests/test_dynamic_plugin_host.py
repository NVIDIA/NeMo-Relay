# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Python-owned dynamic plugin activation."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import os
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

import pytest

from nemo_relay import Json, LLMRequest, ToolExecutionResult, llm, plugin, scope, tools


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


def _native_library_name() -> str:
    if sys.platform == "win32":
        return "nemo_relay_plugin_fixture.dll"
    if sys.platform == "darwin":
        return "libnemo_relay_plugin_fixture.dylib"
    return "libnemo_relay_plugin_fixture.so"


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

            [integrity]
            sha256 = "sha256:{digest}"

            [load]
            library = {library.as_posix()!r}
            symbol = "nemo_relay_fixture_native_plugin"
            """
        )
    )
    return _BuiltPlugin("fixture_native", "rust_dynamic", manifest)


@pytest.fixture(scope="session")
def worker_dynamic_plugin(tmp_path_factory: pytest.TempPathFactory) -> _BuiltPlugin:
    manifest_dir = tmp_path_factory.mktemp("worker-plugin-manifest")
    executable = _prepared_plugin_fixture("NEMO_RELAY_TEST_WORKER_PLUGIN")
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
            entrypoint = {executable.as_posix()!r}
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


def test_load_dynamic_plugin_activation_specs_resolves_standard_toml(tmp_path: Path):
    manifests = tmp_path / "plugins"
    manifests.mkdir()
    native_manifest = manifests / "native.toml"
    native_manifest.write_text(
        textwrap.dedent(
            """
            manifest_version = 1

            [plugin]
            id = "fixture.native"
            kind = "rust_dynamic"
            """
        )
    )
    worker_manifest = manifests / "worker.toml"
    worker_manifest.write_text(
        textwrap.dedent(
            """
            manifest_version = 1

            [plugin]
            id = "fixture.worker"
            kind = "worker"
            """
        )
    )
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[plugins.dynamic]]
            manifest = "plugins/native.toml"

            [plugins.dynamic.config]
            mode = "strict"
            nested = {{ enabled = true }}

            [[plugins.dynamic]]
            manifest = {str(worker_manifest)!r}
            """
        )
    )

    specs = plugin.load_dynamic_plugin_activation_specs(plugins_toml)

    assert [spec.to_dict() for spec in specs] == [
        {
            "plugin_id": "fixture.native",
            "kind": "rust_dynamic",
            "manifest_ref": str(native_manifest.resolve()),
            "config": {"mode": "strict", "nested": {"enabled": True}},
        },
        {
            "plugin_id": "fixture.worker",
            "kind": "worker",
            "manifest_ref": str(worker_manifest.resolve()),
            "config": {},
        },
    ]


@pytest.mark.parametrize(
    ("toml_version", "version"),
    [("true", True), ("0", 0), ("2", 2), ('"1"', "1")],
)
def test_load_dynamic_plugin_activation_specs_rejects_unsupported_version(
    tmp_path: Path, toml_version: str, version: object
):
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(f"version = {toml_version}\n")

    with pytest.raises(ValueError) as error:
        plugin.load_dynamic_plugin_activation_specs(plugins_toml)
    assert str(error.value) == (
        f"plugin config version {version!r} in {plugins_toml.resolve()} is unsupported; expected 1"
    )


def test_load_dynamic_plugin_activation_specs_rejects_duplicate_ids(tmp_path: Path):
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text("[plugin]\nid = 'duplicate'\nkind = 'rust_dynamic'\n")
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        "[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\n[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\n"
    )

    with pytest.raises(ValueError, match="duplicate dynamic plugin id 'duplicate'"):
        plugin.load_dynamic_plugin_activation_specs(plugins_toml)


@pytest.mark.parametrize(
    ("plugins_toml", "manifest", "message"),
    [
        ("plugins = []\n", None, "'plugins' must be a table"),
        ("[plugins]\ndynamic = 'invalid'\n", None, "'plugins.dynamic' must be an array of tables"),
        ("[plugins]\ndynamic = ['invalid']\n", None, r"plugins.dynamic\[0\] must be a table"),
        (
            "[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\nunsupported = true\n",
            "[plugin]\nid = 'fixture'\nkind = 'rust_dynamic'\n",
            "has unknown fields: unsupported",
        ),
        ("[[plugins.dynamic]]\nmanifest = ''\n", None, "manifest must be a non-empty string"),
        (
            "[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\n",
            "plugin = 'invalid'\n",
            "'plugin' must be a table",
        ),
        (
            "[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\n",
            "[plugin]\nkind = 'rust_dynamic'\n",
            "'plugin.id' must be a non-empty string",
        ),
        (
            "[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\n",
            "[plugin]\nid = 'fixture'\nkind = 'invalid'\n",
            "'plugin.kind' must be 'rust_dynamic' or 'worker'",
        ),
        (
            "[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\nconfig = 'invalid'\n",
            "[plugin]\nid = 'fixture'\nkind = 'rust_dynamic'\n",
            "config must be a table",
        ),
        (
            "[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\n"
            "[plugins.dynamic.config]\nwhen = 1979-05-27T07:32:00Z\n",
            "[plugin]\nid = 'fixture'\nkind = 'rust_dynamic'\n",
            "config must contain JSON values",
        ),
    ],
)
def test_load_dynamic_plugin_activation_specs_rejects_invalid_records(
    tmp_path: Path,
    plugins_toml: str,
    manifest: str | None,
    message: str,
):
    config_path = tmp_path / "plugins.toml"
    config_path.write_text(plugins_toml)
    if manifest is not None:
        (tmp_path / "relay-plugin.toml").write_text(manifest)

    with pytest.raises(ValueError, match=message):
        plugin.load_dynamic_plugin_activation_specs(config_path)


@pytest.mark.parametrize("filename", ["plugins.toml", "relay-plugin.toml"])
def test_load_dynamic_plugin_activation_specs_rejects_invalid_toml(tmp_path: Path, filename: str):
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text("[[plugins.dynamic]]\nmanifest = 'relay-plugin.toml'\n")
    (tmp_path / "relay-plugin.toml").write_text("[plugin]\nid = 'fixture'\nkind = 'rust_dynamic'\n")
    (tmp_path / filename).write_text("invalid = [\n")

    with pytest.raises(ValueError, match="invalid .* in"):
        plugin.load_dynamic_plugin_activation_specs(plugins_toml)


def test_load_dynamic_plugin_activation_specs_reports_missing_manifest(tmp_path: Path):
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text("[[plugins.dynamic]]\nmanifest = 'missing/relay-plugin.toml'\n")
    missing_manifest = (tmp_path / "missing" / "relay-plugin.toml").resolve()

    with pytest.raises(FileNotFoundError) as error:
        plugin.load_dynamic_plugin_activation_specs(plugins_toml)
    assert Path(error.value.filename) == missing_manifest


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


async def test_empty_dynamic_specs_preserve_static_initialization_path():
    with pytest.raises(ValueError, match="at least one dynamic plugin"):
        await plugin.initialize_with_dynamic_plugins(plugin.PluginConfig(), [])

    assert plugin.report() is None
    report = await plugin.initialize(plugin.PluginConfig())
    assert report == {"diagnostics": []}
    await plugin.clear_async()


async def test_native_activation_context_owns_callbacks_and_close_is_idempotent(
    native_dynamic_plugin: _BuiltPlugin,
):
    activation = await plugin.initialize_with_dynamic_plugins(plugin.PluginConfig(), [native_dynamic_plugin.spec()])
    assert activation.is_active
    assert activation.report == {"diagnostics": []}

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
    assert "native_plugin_tool_execution" not in result.result
    assert result.result == {"args": {"input": True}}


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

    isolated_user_config = tmp_path / "xdg"
    user_config = isolated_user_config / "nemo-relay"
    user_config.mkdir(parents=True)
    plugins_toml = user_config / "plugins.toml"
    plugins_toml.write_text(
        textwrap.dedent(
            f"""
            version = 1

            [[components]]
            kind = {static_kind!r}
            enabled = true

            [[plugins.dynamic]]
            manifest = {str(native_dynamic_plugin.manifest)!r}
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_user_config))

    plugin.register(static_kind, cast(plugin.Plugin, FileStaticPlugin()))
    activation = None
    try:
        dynamic_plugins = plugin.load_dynamic_plugin_activation_specs(plugins_toml)
        activation = await plugin.initialize_with_dynamic_plugins(plugin.PluginConfig(), tuple(dynamic_plugins))
        assert activation.report == {
            "diagnostics": [
                {
                    "level": "warning",
                    "code": "plugin.configuration_inherited",
                    "message": f"inherited plugin configuration from discovered file: {plugins_toml.resolve()}",
                }
            ]
        }
        result = await tools.execute(
            "python-file-static-base",
            {"input": True},
            lambda args: ToolExecutionResult(args),
        )
        assert result.result["file_static_base"] is True
        assert result.result["native_plugin_tool_execution"] is True
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
    result = await tools.execute(
        "python-native-after-finalize",
        {"input": True},
        lambda args: ToolExecutionResult(args),
    )
    assert result.result == {"input": True}


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX worker stop/continue signals")
async def test_worker_activation_finalizer_never_waits_on_python_thread(
    worker_dynamic_plugin: _BuiltPlugin,
    tmp_path: Path,
):
    with worker_dynamic_plugin.manifest.open("rb") as file:
        worker_entrypoint = Path(tomllib.load(file)["load"]["entrypoint"])

    pid_file = tmp_path / "worker.pid"
    wrapper = tmp_path / "worker-wrapper.sh"
    wrapper.write_text(f"#!/bin/sh\nprintf '%s' \"$$\" > {str(pid_file)!r}\nexec {str(worker_entrypoint)!r}\n")
    wrapper.chmod(0o755)
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text(
        worker_dynamic_plugin.manifest.read_text().replace(
            f"entrypoint = {str(worker_entrypoint)!r}",
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

    async def tool_provider(args: Json) -> ToolExecutionResult[dict[str, Json]]:
        assert asyncio.get_running_loop() is loop
        assert threading.get_ident() == loop_thread
        await asyncio.sleep(0)
        return ToolExecutionResult({"args": args})

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
        assert result.result["worker_plugin_tool_execution"] is True
        assert isinstance(result.result["args"], dict)
        assert result.result["args"]["worker_plugin_tool_execution_request"] is True

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
    result = await tools.execute(
        "python-worker-after-close",
        {"input": True},
        lambda args: ToolExecutionResult({"args": args}),
    )
    assert result.result == {"args": {"input": True}}
