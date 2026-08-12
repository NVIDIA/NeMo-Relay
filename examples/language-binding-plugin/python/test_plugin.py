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
from nemo_relay import llm, plugin, subscribers, tools


@dataclass
class ActivatedExample:
    implementation: DocumentationPlugin
    report: dict[str, Any]


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


def test_validation_rejects_unsupported_mode() -> None:
    diagnostics = DocumentationPlugin().validate({"requests": {"mode": "invalid"}})

    assert diagnostics[0]["code"] == "documentation-plugin.unsupported_mode"


def test_validation_rejects_wrong_type() -> None:
    diagnostics = DocumentationPlugin().validate({"requests": {"priority": "high"}})

    assert diagnostics[0]["code"] == "documentation-plugin.invalid_config"


def test_validation_warns_about_unknown_field() -> None:
    diagnostics = DocumentationPlugin().validate({"unexpected": True})

    assert diagnostics[0]["level"] == "warning"
    assert diagnostics[0]["field"] == "unexpected"


async def test_activation_reports_no_diagnostics(active_plugin: ActivatedExample) -> None:
    assert active_plugin.report["diagnostics"] == []


async def test_tool_request_is_rewritten(active_plugin: ActivatedExample) -> None:
    result = await tools.execute("safe_tool", {"value": 1}, lambda args: args)

    assert result == {"value": 1, "plugin_tag": "documentation"}


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
    await tools.execute("safe_tool", {"value": 1}, lambda args: args)
    await subscribers.flush_async()

    assert active_plugin.implementation.events


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
