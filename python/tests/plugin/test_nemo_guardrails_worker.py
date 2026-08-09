# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the first-party NeMo Guardrails dynamic worker."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shlex
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_relay import LLMRequest, llm, plugin
from nemo_relay.codecs import (
    AnthropicMessagesCodec,
    GeminiGenerateContentCodec,
    OpenAIChatCodec,
    OpenAIResponsesCodec,
)
from nemo_relay_plugin import LlmFinalInputPolicyOutcome, PluginContext, PolicyFailureMode

PLUGIN_ID = "nvidia.nemo_guardrails"


def _plugin_root() -> Path:
    return Path(__file__).parents[3] / "plugins/nemo-guardrails"


@pytest.fixture(name="worker_module", scope="module")
def worker_module_fixture() -> Any:
    plugin_root = _plugin_root()
    package_name = "nvidia_nemo_guardrails_worker"
    sys.path.insert(0, str(plugin_root))
    importlib.invalidate_caches()
    for loaded_name in tuple(sys.modules):
        if loaded_name == package_name or loaded_name.startswith(f"{package_name}."):
            sys.modules.pop(loaded_name, None)
    try:
        yield importlib.import_module(f"{package_name}.worker")
    finally:
        for loaded_name in tuple(sys.modules):
            if loaded_name == package_name or loaded_name.startswith(f"{package_name}."):
                sys.modules.pop(loaded_name, None)
        sys.path.remove(str(plugin_root))


@pytest.fixture(name="managed_worker_environment")
def managed_worker_environment_fixture(tmp_path: Path) -> Path:
    """Expose the test interpreter through Relay's managed-environment shape."""
    if os.name == "nt":
        pytest.skip("test interpreter launcher fixture requires a POSIX shell")
    environment = tmp_path / ".dynamic-plugin-environments" / hashlib.sha256(PLUGIN_ID.encode()).hexdigest()
    scripts = environment / "bin"
    scripts.mkdir(parents=True)
    launcher = scripts / "python"
    launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    return environment


def _activation_spec(environment: Path, config: dict[str, Any]) -> plugin.DynamicPluginActivationSpec:
    return plugin.DynamicPluginActivationSpec(
        plugin_id=PLUGIN_ID,
        kind="worker",
        manifest_ref=str(_plugin_root() / "relay-plugin.toml"),
        environment_ref=str(environment),
        config=config,
    )


class RecordingEngine:
    """Return a configured Guardrails-like result and record input."""

    def __init__(self, status: str, *, content: str = "", rail: str | None = None) -> None:
        self.result = SimpleNamespace(status=status, content=content, rail=rail)
        self.calls: list[tuple[list[dict[str, Any]], list[Any]]] = []

    async def check_async(self, messages: list[dict[str, Any]], rail_types: list[Any]) -> Any:
        self.calls.append((messages, rail_types))
        return self.result


def _config(**overrides: Any) -> dict[str, Any]:
    return {
        "config_yaml": "rails: {}",
        **overrides,
    }


def _request(content: str = "hello") -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        "headers": {"x-test": "preserved"},
        "content": {"provider_owned": True},
    }
    annotated = {
        "instructions": "Follow policy.",
        "messages": [
            {"role": "developer", "content": "Use concise answers."},
            {"role": "user", "content": content},
        ],
        "model": "test-model",
    }
    return request, annotated


def _registered_callback(worker_module: Any, engine: RecordingEngine, **config: Any) -> Any:
    worker = worker_module.NemoGuardrailsWorker(engine_factory=lambda _config: engine)
    raw_config = _config(**config)
    assert worker.validate(raw_config) == []
    context = MagicMock(spec=PluginContext)
    worker.register(context, raw_config)
    context.register_llm_final_input_policy.assert_called_once()
    name, callback = context.register_llm_final_input_policy.call_args.args
    assert name == "input"
    return callback, context


def test_manifest_integrity_and_package_contract():
    plugin_root = Path(__file__).parents[3] / "plugins/nemo-guardrails"
    manifest = tomllib.loads((plugin_root / "relay-plugin.toml").read_text(encoding="utf-8"))
    artifact = plugin_root / manifest["source"]["artifact"]
    pyproject = tomllib.loads((plugin_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert manifest["plugin"] == {"id": "nvidia.nemo_guardrails", "kind": "worker"}
    assert manifest["compat"] == {"relay": ">=0.8,<1.0", "worker_protocol": "grpc-v1"}
    assert manifest["load"]["entrypoint"] == "nvidia_nemo_guardrails_worker.worker:main"
    assert manifest["integrity"]["sha256"] == f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    assert "nemoguardrails==0.23.0" in pyproject["project"]["dependencies"]
    assert pyproject["project"]["requires-python"] == ">=3.11,<3.14"


def test_config_schema_is_strict_and_matches_runtime_defaults(worker_module: Any):
    plugin_root = Path(__file__).parents[3] / "plugins/nemo-guardrails"
    schema = json.loads((plugin_root / "config.schema.json").read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert schema["properties"]["timeout_ms"]["default"] == 30_000
    assert schema["properties"]["failure_mode"]["default"] == "fail_closed"
    assert schema["properties"]["max_concurrency"]["default"] == 16
    settings = worker_module._parse_config(_config())
    assert settings.timeout_ms == 30_000
    assert settings.failure_mode is PolicyFailureMode.FAIL_CLOSED
    assert settings.max_concurrency == 16


@pytest.mark.parametrize(
    ("config", "field"),
    [
        ({}, "config_path"),
        ({"config_path": "rails", "config_yaml": "rails: {}"}, "config_path"),
        ({"config_yaml": "rails: {}", "colang_content": 42}, "colang_content"),
        ({"config_yaml": "rails: {}", "timeout_ms": 0}, "timeout_ms"),
        ({"config_yaml": "rails: {}", "failure_mode": "sometimes"}, "failure_mode"),
        ({"config_yaml": "rails: {}", "surprise": True}, "surprise"),
    ],
)
def test_validation_reports_configuration_field(worker_module: Any, config: dict[str, Any], field: str):
    worker = worker_module.NemoGuardrailsWorker(engine_factory=lambda _config: RecordingEngine("passed"))
    diagnostics = worker.validate(config)

    assert len(diagnostics) == 1
    assert diagnostics[0].level is worker_module.DiagnosticLevel.ERROR
    assert diagnostics[0].code == "nvidia.nemo_guardrails.invalid_config"
    assert diagnostics[0].field == field


async def test_passed_input_allows_and_preserves_redacted_evidence(worker_module: Any):
    engine = RecordingEngine("passed", content="hello")
    callback, context = _registered_callback(
        worker_module,
        engine,
        priority=7,
        timeout_ms=1_250,
        failure_mode="fail_open",
        max_concurrency=2,
    )
    request, annotated = _request()

    outcome = await callback("chat", request, annotated)

    assert isinstance(outcome, LlmFinalInputPolicyOutcome)
    assert outcome.decision == "allow"
    assert outcome.evidence == {
        "policy": "nvidia.nemo_guardrails",
        "engine": "LLMRails",
        "library_version": "0.23.0",
        "rail_type": "input",
        "status": "passed",
    }
    assert engine.calls[0][0] == [
        {"role": "system", "content": "Follow policy."},
        {"role": "system", "content": "Use concise answers."},
        {"role": "user", "content": "hello"},
    ]
    assert engine.calls[0][1] == [worker_module.RailType.INPUT]
    assert context.register_llm_final_input_policy.call_args.kwargs == {
        "priority": 7,
        "timeout_ms": 1_250,
        "failure_mode": PolicyFailureMode.FAIL_OPEN,
    }


async def test_modified_input_replaces_only_last_user_content(worker_module: Any):
    engine = RecordingEngine("modified", content="masked input")
    callback, _context = _registered_callback(worker_module, engine)
    request, annotated = _request("original input")
    original_request = json.loads(json.dumps(request))
    original_annotated = json.loads(json.dumps(annotated))

    outcome = await callback("chat", request, annotated)

    assert outcome.decision == "transform"
    assert outcome.request == original_request
    assert outcome.request is not request
    assert outcome.annotated_request["messages"][-1]["content"] == "masked input"
    assert request == original_request
    assert annotated == original_annotated


async def test_blocked_input_rejects_without_returning_guardrails_content(worker_module: Any):
    engine = RecordingEngine("blocked", content="internal refusal", rail="regex check input")
    callback, _context = _registered_callback(worker_module, engine, blocked_message="Safe rejection.")
    request, annotated = _request("blocked phrase")

    outcome = await callback("chat", request, annotated)

    assert outcome.decision == "reject"
    assert outcome.reason_code == "nemo_guardrails.input_blocked"
    assert outcome.safe_message == "Safe rejection."
    assert outcome.evidence["rail"] == "regex check input"
    assert "internal refusal" not in json.dumps(outcome.to_json())


@pytest.mark.parametrize(
    "annotated",
    [
        None,
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
        {"messages": [{"role": "tool", "content": "result", "tool_call_id": "call-1"}]},
        {"messages": [{"role": "assistant", "content": None, "tool_calls": []}]},
    ],
)
async def test_unsupported_input_is_an_explicit_terminal_rejection(worker_module: Any, annotated: Any):
    engine = RecordingEngine("passed")
    callback, _context = _registered_callback(worker_module, engine, failure_mode="fail_open")
    request, _ = _request()

    outcome = await callback("chat", request, annotated)

    assert outcome.decision == "reject"
    assert outcome.reason_code == "nemo_guardrails.unsupported_input"
    assert outcome.evidence["status"] == "unsupported"
    assert engine.calls == []


async def test_main_serves_first_party_worker(worker_module: Any, monkeypatch: pytest.MonkeyPatch):
    served: list[Any] = []

    async def capture(plugin: Any) -> None:
        served.append(plugin)

    monkeypatch.setattr(worker_module, "serve_plugin", capture)
    await worker_module.main()

    assert len(served) == 1
    assert isinstance(served[0], worker_module.NemoGuardrailsWorker)


async def test_real_guardrails_023_regex_rail_allows_and_blocks(worker_module: Any):
    config_yaml = """
rails:
  config:
    regex_detection:
      input:
        patterns:
          - blocked phrase
        case_insensitive: true
  input:
    flows:
      - regex check input
"""
    worker = worker_module.NemoGuardrailsWorker()
    config = {"config_yaml": config_yaml}
    assert worker.validate(config) == []
    context = MagicMock(spec=PluginContext)
    worker.register(context, config)
    callback = context.register_llm_final_input_policy.call_args.args[1]
    request, allowed = _request("ordinary request")
    _, blocked = _request("contains BLOCKED PHRASE here")

    allowed_outcome = await callback("chat", request, allowed)
    blocked_outcome = await callback("chat", request, blocked)

    assert allowed_outcome.decision == "allow"
    assert blocked_outcome.decision == "reject"
    assert blocked_outcome.evidence["rail"] == "regex check input"


async def test_real_guardrails_023_modification_updates_provider_annotation(worker_module: Any):
    worker = worker_module.NemoGuardrailsWorker()
    config = {
        "config_yaml": """
rails:
  input:
    flows:
      - rewrite input
""",
        "colang_content": """
define subflow rewrite input
  $user_message = "rewritten by Guardrails"
""",
    }
    assert worker.validate(config) == []
    context = MagicMock(spec=PluginContext)
    worker.register(context, config)
    callback = context.register_llm_final_input_policy.call_args.args[1]
    request, annotated = _request("original input")

    outcome = await callback("chat", request, annotated)

    assert outcome.decision == "transform"
    assert outcome.annotated_request["messages"][-1]["content"] == "rewritten by Guardrails"
    assert outcome.evidence["status"] == "modified"
    assert request["content"] == {"provider_owned": True}
    assert annotated["messages"][-1]["content"] == "original input"


@pytest.mark.parametrize(
    ("codec", "request_content", "expected_content"),
    [
        (
            OpenAIChatCodec(),
            {"model": "test", "messages": [{"role": "user", "content": "original input"}]},
            {"model": "test", "messages": [{"role": "user", "content": "rewritten by Guardrails"}]},
        ),
        (
            OpenAIResponsesCodec(),
            {"model": "test", "input": [{"role": "user", "content": "original input"}]},
            {"model": "test", "input": [{"role": "user", "content": "rewritten by Guardrails"}]},
        ),
        (
            AnthropicMessagesCodec(),
            {
                "model": "test",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "original input"}],
            },
            {
                "model": "test",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "rewritten by Guardrails"}],
            },
        ),
        (
            GeminiGenerateContentCodec(),
            {
                "model": "test",
                "contents": [{"role": "user", "parts": [{"text": "original input"}]}],
            },
            {
                "model": "test",
                "contents": [{"role": "user", "parts": [{"text": "rewritten by Guardrails"}]}],
            },
        ),
    ],
    ids=["openai-chat", "openai-responses", "anthropic-messages", "gemini-generate-content"],
)
async def test_dynamic_host_rewrites_all_builtin_provider_shapes(
    managed_worker_environment: Path,
    codec: Any,
    request_content: dict[str, Any],
    expected_content: dict[str, Any],
):
    config = {
        "config_yaml": """
rails:
  input:
    flows:
      - rewrite input
""",
        "colang_content": """
define subflow rewrite input
  $user_message = "rewritten by Guardrails"
""",
    }
    activation = await plugin.initialize_with_dynamic_plugins(
        {},
        [_activation_spec(managed_worker_environment, config)],
    )
    provider_requests: list[LLMRequest] = []

    async def provider(request: LLMRequest) -> dict[str, bool]:
        provider_requests.append(request)
        return {"ok": True}

    try:
        result = await llm.execute(
            "guardrails-host-rewrite",
            LLMRequest({"x-preserved": "true"}, request_content),
            provider,
            codec=codec,
        )
    finally:
        await activation.close()

    assert result == {"ok": True}
    assert len(provider_requests) == 1
    assert provider_requests[0].headers == {"x-preserved": "true"}
    assert provider_requests[0].content == expected_content


async def test_dynamic_host_allows_and_terminally_blocks_before_provider(
    managed_worker_environment: Path,
):
    config = {
        "config_yaml": """
rails:
  config:
    regex_detection:
      input:
        patterns:
          - blocked phrase
        case_insensitive: true
  input:
    flows:
      - regex check input
"""
    }
    activation = await plugin.initialize_with_dynamic_plugins(
        {},
        [_activation_spec(managed_worker_environment, config)],
    )
    provider_requests: list[LLMRequest] = []

    async def provider(request: LLMRequest) -> dict[str, bool]:
        provider_requests.append(request)
        return {"ok": True}

    try:
        allowed = await llm.execute(
            "guardrails-host-allow",
            LLMRequest({}, {"model": "test", "messages": [{"role": "user", "content": "ordinary request"}]}),
            provider,
            codec=OpenAIChatCodec(),
        )
        with pytest.raises(RuntimeError, match="Request blocked by NeMo Guardrails input policy"):
            await llm.execute(
                "guardrails-host-block",
                LLMRequest(
                    {},
                    {"model": "test", "messages": [{"role": "user", "content": "contains BLOCKED PHRASE"}]},
                ),
                provider,
                codec=OpenAIChatCodec(),
            )
    finally:
        await activation.close()

    assert allowed == {"ok": True}
    assert len(provider_requests) == 1


async def test_dynamic_host_blocks_stream_before_provider_opens(
    managed_worker_environment: Path,
):
    config = {
        "config_yaml": """
rails:
  config:
    regex_detection:
      input:
        patterns:
          - blocked phrase
  input:
    flows:
      - regex check input
"""
    }
    activation = await plugin.initialize_with_dynamic_plugins(
        {},
        [_activation_spec(managed_worker_environment, config)],
    )
    provider_opened = False

    async def stream_provider(_request: LLMRequest):
        nonlocal provider_opened
        provider_opened = True
        yield {"text": "must not run"}

    try:
        with pytest.raises(RuntimeError, match="Request blocked by NeMo Guardrails input policy"):
            await llm.stream_execute(
                "guardrails-host-stream-block",
                LLMRequest({}, {"model": "test", "messages": [{"role": "user", "content": "blocked phrase"}]}),
                stream_provider,
                lambda _chunk: None,
                lambda: {},
                codec=OpenAIChatCodec(),
            )
    finally:
        await activation.close()

    assert not provider_opened
