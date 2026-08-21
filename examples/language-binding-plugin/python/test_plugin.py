# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic tests for the Python language-binding plugin example."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio
from main import DocumentationPlugin, component

import nemo_relay
from nemo_relay import llm, plugin, runtime_registrations, scope, subscribers, tools
from nemo_relay.runtime_registrations import RuntimeRegistrationKind


@dataclass
class ActivatedExample:
    implementation: DocumentationPlugin
    report: dict[str, Any]


class RecordingContext:
    """Records one registration per context method without touching global state."""

    def __init__(self) -> None:
        self.registrations: list[str] = []

    def __getattr__(self, name: str):
        if not name.startswith("register_"):
            raise AttributeError(name)

        def register(_registration_name: str, *_args: Any) -> None:
            self.registrations.append(name)

        return register


@pytest_asyncio.fixture
async def active_plugin(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ActivatedExample]:
    """Activate a fresh component and remove every owned registration afterward."""

    implementation = DocumentationPlugin()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    plugin.register("documentation-plugin", implementation)
    try:
        report = await plugin.initialize(component("enforce"))
        yield ActivatedExample(implementation, report)
    finally:
        await plugin.clear_async()
        plugin.deregister("documentation-plugin")


def test_validation_accepts_supported_mode() -> None:
    assert DocumentationPlugin().validate({"requests": {"mode": "enforce"}}) == []


def test_default_registration_control_is_disabled_and_valid() -> None:
    configuration = component("enforce").components[0].config

    assert configuration["registration_control"]["enabled"] is False
    assert DocumentationPlugin().validate(configuration) == []


def test_validation_rejects_unsupported_mode() -> None:
    diagnostics = DocumentationPlugin().validate({"requests": {"mode": "invalid"}})

    assert diagnostics[0]["code"] == "documentation-plugin.unsupported_mode"


def test_validation_rejects_wrong_type() -> None:
    diagnostics = DocumentationPlugin().validate({"requests": {"priority": "high"}})

    assert diagnostics[0]["code"] == "documentation-plugin.invalid_config"


def test_registration_rejects_a_duplicate_kind_and_missing_deregistration_is_false() -> None:
    plugin.register("documentation-plugin", DocumentationPlugin())
    try:
        with pytest.raises(RuntimeError):
            plugin.register("documentation-plugin", DocumentationPlugin())
        assert plugin.deregister("missing-documentation-plugin") is False
    finally:
        plugin.deregister("documentation-plugin")


@pytest.mark.parametrize(
    ("config", "field", "code"),
    [
        ({"tag": ""}, "tag", "documentation-plugin.invalid_tag"),
        ({"requests": {"header_name": ""}}, "requests.header_name", "documentation-plugin.invalid_header"),
        ({"requests": {"header_value": ""}}, "requests.header_value", "documentation-plugin.invalid_header"),
        ({"registration_control": {"kinds": []}}, "registration_control.kinds", "documentation-plugin.invalid_config"),
        (
            {"registration_control": {"registration_name": ""}},
            "registration_control.registration_name",
            "documentation-plugin.invalid_config",
        ),
        (
            {"registration_control": {"reason": ""}},
            "registration_control.reason",
            "documentation-plugin.invalid_config",
        ),
    ],
)
def test_validation_rejects_empty_required_strings(config: dict[str, Any], field: str, code: str) -> None:
    diagnostics = DocumentationPlugin().validate(config)

    assert any(item["code"] == code and item["field"] == field for item in diagnostics)


def test_validation_warns_about_unknown_field() -> None:
    diagnostics = DocumentationPlugin().validate({"unexpected": True})

    assert diagnostics[0]["level"] == "warning"
    assert diagnostics[0]["field"] == "unexpected"


def test_disabled_invalid_component_is_still_validated() -> None:
    plugin.register("documentation-plugin", DocumentationPlugin())
    try:
        report = plugin.validate(component("invalid", enabled=False))
        assert report["diagnostics"][0]["code"] == "documentation-plugin.unsupported_mode"
    finally:
        plugin.deregister("documentation-plugin")


def test_registers_each_safe_plugin_surface() -> None:
    context = RecordingContext()

    DocumentationPlugin().register(component("enforce").components[0].config, context)  # type: ignore[arg-type]

    assert set(context.registrations) == {
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

    configuration = component("enforce").components[0].config
    configuration["registration_control"]["enabled"] = True
    context = RecordingContext()
    DocumentationPlugin().register(configuration, context)  # type: ignore[arg-type]
    assert "register_conditional_middleware_guardrail" in context.registrations


async def test_registration_control_is_owned_by_activation(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    target = "documentation-controlled-subscriber"
    observed: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    subscribers.register(target, lambda event: observed.append(event.name))
    configuration = component("enforce")
    configuration.components[0].config["registration_control"]["enabled"] = True
    plugin.register("documentation-plugin", DocumentationPlugin())
    try:
        before = runtime_registrations.list_runtime_registrations({RuntimeRegistrationKind.SUBSCRIBER})
        assert any(item.effective_name == target for item in before)
        await plugin.initialize(configuration)
        baseline = len(observed)
        scope.event("registration-control-active")
        await subscribers.flush_async()
        assert len(observed) == baseline
        await plugin.clear_async()
        scope.event("registration-control-cleared")
        await subscribers.flush_async()
        assert observed[-1] == "registration-control-cleared"
    finally:
        await plugin.clear_async()
        plugin.deregister("documentation-plugin")
        subscribers.deregister(target)


async def test_activation_reports_no_diagnostics(active_plugin: ActivatedExample) -> None:
    assert active_plugin.report["diagnostics"] == []


async def test_tool_request_is_rewritten(active_plugin: ActivatedExample) -> None:
    result = await tools.execute(
        "safe_tool",
        {"value": 1},
        lambda args: nemo_relay.ToolExecutionResult(args, {"source": "application"}),
    )

    assert result.result == {"value": 1, "plugin_tag": "documentation"}
    assert result.annotation == {"source": "application"}


async def test_tool_policy_blocks_configured_tool(active_plugin: ActivatedExample) -> None:
    with pytest.raises(RuntimeError, match="guardrail rejected"):
        await tools.execute("dangerous_tool", {"value": 1}, lambda _args: pytest.fail("provider must not run"))


async def test_llm_request_is_rewritten(active_plugin: ActivatedExample) -> None:
    request = nemo_relay.LLMRequest({}, {"model": "allowed-model"})

    result = await llm.execute("allowed-model", request, lambda rewritten: {"headers": rewritten.headers})

    assert result["headers"]["x-nemo-relay-plugin"] == "documentation"


async def test_llm_policy_blocks_configured_model(active_plugin: ActivatedExample) -> None:
    request = nemo_relay.LLMRequest({}, {"model": "restricted-model"})

    with pytest.raises(RuntimeError, match="guardrail rejected"):
        await llm.execute("restricted-model", request, lambda _request: pytest.fail("provider must not run"))


async def test_llm_stream_is_transformed(active_plugin: ActivatedExample) -> None:
    request = nemo_relay.LLMRequest({}, {"model": "allowed-model"})

    async def provider(_request: Any) -> AsyncIterator[dict[str, int]]:
        yield {"chunk": 1}
        yield {"chunk": 2}

    stream = await llm.stream_execute("allowed-model", request, provider, lambda _chunk: None, lambda: {"done": True})
    chunks = [chunk async for chunk in stream]

    assert chunks == [
        {"chunk": 1, "plugin_stream": True},
        {"chunk": 2, "plugin_stream": True},
    ]


async def test_subscriber_observes_managed_call(active_plugin: ActivatedExample) -> None:
    await tools.execute("safe_tool", {"value": 1}, nemo_relay.ToolExecutionResult)
    await subscribers.flush_async()

    assert active_plugin.implementation.events


async def test_runtime_events_do_not_depend_on_request_rewriting(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    implementation = DocumentationPlugin()
    configuration = component("enforce")
    configuration.components[0].config["requests"]["enabled"] = False
    plugin.register("documentation-plugin", implementation)
    try:
        await plugin.initialize(configuration)
        await tools.execute("safe_tool", {"value": 1}, nemo_relay.ToolExecutionResult)
        await subscribers.flush_async()

        assert "documentation-plugin.request" in implementation.events
    finally:
        await plugin.clear_async()
        plugin.deregister("documentation-plugin")


async def test_teardown_removes_plugin_kind(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    plugin.register("documentation-plugin", DocumentationPlugin())
    try:
        await plugin.initialize(component("enforce"))
        await plugin.clear_async()
        assert plugin.deregister("documentation-plugin") is True
        assert "documentation-plugin" not in plugin.list_kinds()
    finally:
        await plugin.clear_async()
        plugin.deregister("documentation-plugin")
