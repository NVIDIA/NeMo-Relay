# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic contract tests for the standalone Python grpc-v1 worker example."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

if os.environ.get("NEMO_RELAY_SKIP_PYTHON_PLUGIN_TESTS") == "1":
    pytest.skip("grpcio is unavailable for Python plugin SDK tests on this runner", allow_module_level=True)

pytest.importorskip("grpc")

from nemo_relay_plugin import PluginContext, PluginRuntime, ToolExecutionResult  # noqa: E402

EXAMPLE_ROOT = Path(__file__).parents[1]
MODULE_NAME = "nemo_relay_python_grpc_worker_example.worker"
PACKAGE_NAME = MODULE_NAME.partition(".")[0]


def purge_example_modules() -> None:
    for loaded_name in tuple(sys.modules):
        if loaded_name == PACKAGE_NAME or loaded_name.startswith(f"{PACKAGE_NAME}."):
            sys.modules.pop(loaded_name, None)


@pytest.fixture(name="example")
def example_fixture() -> Iterator[Any]:
    """Import a fresh copy of the example directly from its own project root."""

    sys.path.insert(0, str(EXAMPLE_ROOT))
    importlib.invalidate_caches()
    purge_example_modules()
    try:
        yield importlib.import_module(MODULE_NAME)
    finally:
        purge_example_modules()
        sys.path.remove(str(EXAMPLE_ROOT))


def read_manifest() -> dict[str, Any]:
    return tomllib.loads((EXAMPLE_ROOT / "relay-plugin.toml").read_text(encoding="utf-8"))


def configured_context() -> tuple[MagicMock, MagicMock]:
    runtime = MagicMock(spec=PluginRuntime)
    runtime.emit_mark = AsyncMock()
    runtime.push_scope = AsyncMock(return_value="scope-handle")
    runtime.pop_scope = AsyncMock()
    runtime.create_scope_stack = AsyncMock(return_value="isolated-stack")
    runtime.drop_scope_stack = AsyncMock()
    runtime.bind_scope_stack.side_effect = lambda _stack: contextlib.nullcontext()
    context = MagicMock(spec=PluginContext)
    context.runtime = runtime
    return context, runtime


def register_example(example: Any) -> tuple[MagicMock, MagicMock]:
    context, runtime = configured_context()
    example.ExamplePythonWorker().register(context, example.DEFAULT_CONFIG)
    return context, runtime


def callback(context: MagicMock, method: str, name: str | None = None) -> Any:
    calls = getattr(context, method).call_args_list
    if name is not None:
        for call in calls:
            if call.args[0] == name:
                return call.args[1]
        raise AssertionError(f"{method} did not register {name!r}")
    assert len(calls) == 1, f"{method} has multiple callbacks; select one by name"
    return calls[0].args[1]


def test_manifest_digest_matches_worker_source() -> None:
    manifest = read_manifest()
    artifact = EXAMPLE_ROOT / manifest["source"]["artifact"]
    actual = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"

    assert actual == manifest["integrity"]["sha256"]


def test_manifest_declares_current_worker_protocol() -> None:
    manifest = read_manifest()

    assert manifest["compat"] == {"relay": ">=0.8.0,<1.0", "worker_protocol": "grpc-v1"}


def test_schema_declares_only_supported_groups() -> None:
    manifest = read_manifest()
    schema = json.loads((EXAMPLE_ROOT / manifest["config_schema"]["path"]).read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "tag",
        "observe",
        "requests",
        "execution",
        "runtime",
        "registration_control",
    }


def test_project_builds_an_importable_wheel(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    wheel_dir = tmp_path / "wheel"
    shutil.copytree(
        EXAMPLE_ROOT,
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
    sys.path.insert(0, str(wheel))
    importlib.invalidate_caches()
    purge_example_modules()
    try:
        module = importlib.import_module(MODULE_NAME)
        assert module.__file__ is not None
        assert Path(module.__file__).is_relative_to(wheel)
    finally:
        purge_example_modules()
        sys.path.remove(str(wheel))


def test_default_configuration_is_valid(example: Any) -> None:
    assert example.ExamplePythonWorker().validate(example.DEFAULT_CONFIG) == []
    assert example.DEFAULT_CONFIG["registration_control"]["enabled"] is False
    context, _runtime = configured_context()

    example.ExamplePythonWorker().register(context, example.DEFAULT_CONFIG)

    context.register_conditional_middleware_guardrail.assert_not_called()


def test_non_object_configuration_is_rejected(example: Any) -> None:
    diagnostic = example.ExamplePythonWorker().validate(None)[0]

    assert diagnostic.code == "examples.python_grpc_worker.invalid_config"


def test_unsupported_mode_is_rejected(example: Any) -> None:
    diagnostics = example.ExamplePythonWorker().validate({"requests": {"mode": "sometimes"}})

    assert any(item.code == "examples.python_grpc_worker.unsupported_mode" for item in diagnostics)


def test_wrong_type_is_rejected(example: Any) -> None:
    diagnostics = example.ExamplePythonWorker().validate({"requests": {"priority": "high"}})

    assert any(item.code == "examples.python_grpc_worker.invalid_type" for item in diagnostics)


@pytest.mark.parametrize(
    ("config", "field"),
    [
        ({"registration_control": {"kinds": []}}, "registration_control.kinds"),
        ({"registration_control": {"registration_name": ""}}, "registration_control.registration_name"),
        ({"registration_control": {"reason": ""}}, "registration_control.reason"),
    ],
)
def test_invalid_registration_control_is_rejected(example: Any, config: dict[str, Any], field: str):
    diagnostics = example.ExamplePythonWorker().validate(config)

    assert any(item.code == "examples.python_grpc_worker.invalid_type" and item.field == field for item in diagnostics)


def test_unknown_field_is_rejected(example: Any) -> None:
    diagnostics = example.ExamplePythonWorker().validate({"requests": {"unknown": True}})

    assert any(
        item.code == "examples.python_grpc_worker.unknown_field" and item.level.name == "ERROR" for item in diagnostics
    )


def test_register_rejects_invalid_configuration(example: Any) -> None:
    with pytest.raises(ValueError, match="requests.priority must be an integer"):
        example.ExamplePythonWorker().register(
            MagicMock(spec=PluginContext),
            {"requests": {"priority": "high"}},
        )


async def test_manifest_entrypoint_serves_worker(example: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[Any] = []

    async def capture(plugin: Any) -> None:
        served.append(plugin)

    monkeypatch.setattr(example, "serve_plugin", capture)
    await example.main()

    assert len(served) == 1
    assert isinstance(served[0], example.ExamplePythonWorker)


def test_register_installs_all_protocol_surfaces(example: Any) -> None:
    context, _runtime = register_example(example)
    registration_methods = {
        "register_event_metadata_injector",
        "register_subscriber",
        "register_mark_sanitize_guardrail",
        "register_scope_sanitize_start_guardrail",
        "register_scope_sanitize_end_guardrail",
        "register_tool_sanitize_request_guardrail",
        "register_tool_sanitize_response_guardrail",
        "register_tool_conditional_execution_guardrail",
        "register_tool_request_intercept",
        "register_tool_execution_intercept",
        "register_llm_sanitize_request_guardrail",
        "register_llm_sanitize_response_guardrail",
        "register_llm_conditional_execution_guardrail",
        "register_llm_request_intercept",
        "register_llm_execution_intercept",
        "register_llm_stream_execution_intercept",
    }

    assert all(getattr(context, method).call_count >= 1 for method in registration_methods)


def test_enabled_registration_control_registers_expected_gate(example: Any):
    context, _runtime = configured_context()
    config = deepcopy(example.DEFAULT_CONFIG)
    config["registration_control"]["enabled"] = True

    example.ExamplePythonWorker().register(context, config)

    args = context.register_conditional_middleware_guardrail.call_args.args
    assert args[:3] == (
        "documentation_registration_control",
        {example.RuntimeRegistrationKind.SUBSCRIBER},
        "documentation-controlled-subscriber",
    )
    assert args[3]({example.RuntimeRegistrationKind.SUBSCRIBER}, args[2]) == ("disabled by documentation plugin")
    assert args[3]({example.RuntimeRegistrationKind.SUBSCRIBER}, "documentation-observed-subscriber") is None


async def test_event_metadata_injector_adds_transport_metadata(example: Any) -> None:
    context, _runtime = register_example(example)
    injector = callback(context, "register_event_metadata_injector")

    assert await injector({"name": "external-plugin-event-metadata-injection-example"}) == {
        "external.injector.transport": "python_grpc_worker"
    }
    assert await injector({"name": "unrelated"}) == {}


def test_runtime_registration_does_not_depend_on_request_configuration(example: Any) -> None:
    context, _runtime = configured_context()
    config = deepcopy(example.DEFAULT_CONFIG)
    config["requests"]["enabled"] = False

    example.ExamplePythonWorker().register(context, config)

    assert context.register_tool_request_intercept.call_count == 0
    callback(context, "register_tool_execution_intercept", "documentation_runtime_events")


async def test_subscriber_emits_observation_mark(example: Any) -> None:
    context, runtime = register_example(example)
    subscriber = callback(context, "register_subscriber")

    await subscriber({"name": "tool.start"})

    runtime.emit_mark.assert_awaited_once()


def test_event_sanitizer_redacts_and_tags_fields(example: Any) -> None:
    context, _runtime = register_example(example)
    sanitize = callback(context, "register_mark_sanitize_guardrail")

    fields = sanitize(
        {"name": "mark"},
        {"data": {"secret": "value"}, "category_profile": {}, "metadata": {}},
    )

    assert fields["data"] == {"secret": "[REDACTED]"}
    assert fields["metadata"]["plugin_tag"] == "documentation"


def test_tool_request_sanitizer_redacts_observability_value(example: Any) -> None:
    context, _runtime = register_example(example)
    sanitize = callback(context, "register_tool_sanitize_request_guardrail")

    assert sanitize("safe_tool", {"secret": "value"}) == {"secret": "[REDACTED]"}


def test_tool_response_sanitizer_redacts_observability_value(example: Any) -> None:
    context, _runtime = register_example(example)
    sanitize = callback(context, "register_tool_sanitize_response_guardrail")

    assert sanitize("safe_tool", {"secret": "value"}) == {"secret": "[REDACTED]"}


async def test_llm_request_sanitizer_uses_codec(example: Any) -> None:
    context, _runtime = register_example(example)
    sanitize = callback(context, "register_llm_sanitize_request_guardrail")
    codec = MagicMock()
    codec.decode = AsyncMock(return_value={"secret": "value"})
    codec.encode = AsyncMock(return_value={"headers": {}, "content": {"secret": "encoded"}})
    codec_context = MagicMock()
    codec_context.resolve_codec.return_value = codec

    result = await sanitize({"headers": {}, "content": {"secret": "raw"}}, codec_context)

    codec.decode.assert_awaited_once()
    codec.encode.assert_awaited_once_with({"secret": "[REDACTED]"}, {"headers": {}, "content": {"secret": "raw"}})
    assert result["content"] == {"secret": "[REDACTED]"}


async def test_llm_response_sanitizer_uses_codec(example: Any) -> None:
    context, _runtime = register_example(example)
    sanitize = callback(context, "register_llm_sanitize_response_guardrail")
    codec = MagicMock()
    codec.decode = AsyncMock(return_value={"message": "decoded"})
    codec_context = MagicMock()
    codec_context.resolve_codec.return_value = codec

    result = await sanitize({"secret": "value"}, codec_context)

    codec.decode.assert_awaited_once_with({"secret": "value"})
    assert result == {"secret": "[REDACTED]"}


def test_tool_policy_blocks_configured_tool(example: Any) -> None:
    context, _runtime = register_example(example)
    policy = callback(context, "register_tool_conditional_execution_guardrail")

    assert policy("dangerous_tool", {}) == "tool 'dangerous_tool' is blocked by documentation policy"


def test_llm_policy_blocks_configured_model(example: Any) -> None:
    context, _runtime = register_example(example)
    policy = callback(context, "register_llm_conditional_execution_guardrail")

    assert policy({"headers": {}, "content": {"model": "restricted-model"}}) == (
        "model 'restricted-model' is blocked by documentation policy"
    )


async def test_tool_request_intercept_tags_real_request(example: Any) -> None:
    context, _runtime = register_example(example)
    intercept = callback(context, "register_tool_request_intercept")

    result = await intercept("safe_tool", {"value": 1})

    assert result == {
        "value": 1,
        "_nemo_relay_plugin": {"tag": "documentation", "tool": "safe_tool"},
        "plugin_tag": "documentation",
        "plugin_tool": "safe_tool",
    }


async def test_runtime_helpers_clean_up_successful_request(example: Any) -> None:
    context, runtime = register_example(example)
    intercept = callback(context, "register_tool_execution_intercept", "documentation_runtime_events")
    next_call = MagicMock()
    next_call.call = AsyncMock(return_value=ToolExecutionResult({"ok": True}))

    await intercept("safe_tool", {"value": 1}, next_call)

    runtime.push_scope.assert_awaited_once()
    runtime.pop_scope.assert_awaited_once_with("scope-handle", output={"done": True})
    runtime.drop_scope_stack.assert_awaited_once_with("isolated-stack")


async def test_runtime_helpers_close_failed_request(example: Any) -> None:
    context, runtime = register_example(example)
    runtime.emit_mark.side_effect = RuntimeError("mark failed")
    intercept = callback(context, "register_tool_execution_intercept", "documentation_runtime_events")
    next_call = MagicMock()
    next_call.call = AsyncMock(return_value=ToolExecutionResult({"ok": True}))

    with pytest.raises(RuntimeError, match="mark failed"):
        await intercept("safe_tool", {"value": 1}, next_call)

    runtime.pop_scope.assert_awaited_once_with("scope-handle", metadata={"failed": True})
    runtime.create_scope_stack.assert_not_awaited()


async def test_runtime_cleanup_preserves_the_callback_error(example: Any) -> None:
    context, runtime = register_example(example)
    runtime.emit_mark.side_effect = RuntimeError("mark failed")
    runtime.pop_scope.side_effect = RuntimeError("cleanup failed")
    intercept = callback(context, "register_tool_execution_intercept", "documentation_runtime_events")
    next_call = MagicMock()
    next_call.call = AsyncMock(return_value=ToolExecutionResult({"ok": True}))

    with pytest.raises(RuntimeError, match="mark failed"):
        await intercept("safe_tool", {"value": 1}, next_call)

    runtime.pop_scope.assert_awaited_once_with("scope-handle", metadata={"failed": True})


def test_llm_request_intercept_preserves_outcome_fields(example: Any) -> None:
    context, _runtime = register_example(example)
    intercept = callback(context, "register_llm_request_intercept")
    annotated = {"messages": [{"role": "user", "content": "hello"}]}

    outcome = intercept("allowed-model", {"headers": {}, "content": {}}, annotated)

    assert outcome.request["headers"]["x-nemo-relay-plugin"] == "documentation"
    assert outcome.annotated_request is annotated
    assert len(outcome.pending_marks) == 1
    assert len(outcome.optimization_contributions) == 1


async def test_tool_execution_returns_pending_mark(example: Any) -> None:
    context, _runtime = register_example(example)
    intercept = callback(context, "register_tool_execution_intercept", "documentation_tool_execution")
    next_call = MagicMock()
    next_call.call = AsyncMock(return_value=ToolExecutionResult({"ok": True}, annotation={"source": "application"}))

    outcome = await intercept("safe_tool", {"value": 1}, next_call)

    assert outcome.result == {"ok": True}
    assert outcome.annotation == {
        "upstream": {"source": "application"},
        "worker": {"tool_name": "safe_tool", "tag": "documentation"},
    }
    assert len(outcome.pending_marks) == 1


async def test_llm_execution_can_repeat_continuation(example: Any) -> None:
    context, _runtime = register_example(example)
    intercept = callback(context, "register_llm_execution_intercept")
    next_call = MagicMock()
    next_call.call = AsyncMock(side_effect=[{"choice": 1}, {"choice": 2}])

    result = await intercept(
        "allowed-model",
        {"headers": {}, "content": {"repeat_downstream": True}},
        next_call,
    )

    assert result == {"choice": 1}
    assert next_call.call.await_count == 2


async def test_repeated_llm_continuation_ignores_the_second_failure(example: Any) -> None:
    context, _runtime = register_example(example)
    intercept = callback(context, "register_llm_execution_intercept")
    next_call = MagicMock()
    next_call.call = AsyncMock(side_effect=[{"choice": 1}, RuntimeError("second call failed")])

    result = await intercept(
        "allowed-model",
        {"headers": {}, "content": {"repeat_downstream": True}},
        next_call,
    )

    assert result == {"choice": 1}
    assert next_call.call.await_count == 2


async def test_stream_execution_transforms_chunks_lazily(example: Any) -> None:
    context, _runtime = register_example(example)
    intercept = callback(context, "register_llm_stream_execution_intercept")

    async def downstream() -> Any:
        yield {"chunk": 1}
        yield {"chunk": 2}

    next_call = MagicMock()
    next_call.call.return_value = downstream()

    stream = intercept("allowed-model", {"headers": {}, "content": {}}, next_call)

    assert [item async for item in stream] == [
        {"chunk": 1, "plugin_stream": True},
        {"chunk": 2, "plugin_stream": True},
    ]
