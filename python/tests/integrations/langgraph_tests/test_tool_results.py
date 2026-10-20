# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check native ToolNode results and the content recorded by Relay."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import nemo_relay


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("result_kind", ["message", "command", "list"])
@pytest.mark.parametrize("status", ["success", "error"])
def test_tool_wrappers_preserve_results_and_record_content(subscribed_events, use_async, result_kind, status):
    from langchain_core.messages import ToolMessage
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command, Send

    from nemo_relay.integrations.langgraph.tool_node import awrap_tool_call, wrap_tool_call

    artifact = object()
    message = ToolMessage(content="tool answer", status=status, tool_call_id="target", artifact=artifact)
    command = Command(update={"messages": [message], "state": 42}, goto=Send("next", {"value": 1}))
    unrelated = Command(update={"messages": [ToolMessage(content="other", tool_call_id="other")]})
    result = message if result_kind == "message" else command if result_kind == "command" else [unrelated, command]
    request = ToolCallRequest(
        tool_call={"name": "lookup", "args": {}, "id": "target"},
        tool=MagicMock(),
        state={},
        runtime=MagicMock(),
    )

    def handler(_request):
        return result

    async def ahandler(_request):
        return result

    exporter = nemo_relay.AtifExporter("tool-node", "agent", "1")
    exporter.register("tool-node-results")
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            restored = (
                asyncio.run(awrap_tool_call(request, ahandler)) if use_async else wrap_tool_call(request, handler)
            )
        nemo_relay.subscribers.flush()
        trajectory = cast(dict[str, Any], exporter.export())
    finally:
        exporter.deregister("tool-node-results")
    if result_kind == "list":
        assert isinstance(restored, list)
        assert restored[0] == unrelated
        restored = restored[1]
    if result_kind != "message":
        assert isinstance(restored, Command)
        assert restored.goto == command.goto
        assert restored.update["state"] == 42
        restored = restored.update["messages"][0]
    assert isinstance(restored, ToolMessage)
    assert restored.content == "tool answer"
    assert restored.status == status
    assert restored.artifact is artifact
    event = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent) and event.name == "lookup" and event.scope_category == "end"
    )
    assert event.data["content"] == "tool answer"
    assert event.data["is_error"] == (status == "error")
    assert event.metadata["otel.status_code"] == ("ERROR" if status == "error" else "OK")
    observations = [result for step in trajectory["steps"] for result in step.get("observation", {}).get("results", [])]
    assert len(observations) == 1
    assert observations[0]["content"] == "tool answer"
    assert observations[0]["extra"].get("is_error") == (True if status == "error" else None)
