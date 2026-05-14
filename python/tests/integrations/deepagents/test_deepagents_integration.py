# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Deep Agents NeMo Flow integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.agents.middleware import ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.callbacks import GraphInterruptEvent, GraphResumeEvent
from langgraph.types import Interrupt

import nemo_flow
from nemo_flow.integrations.deepagents import (
    NemoFlowDeepAgentsCallbackHandler,
    NemoFlowDeepAgentsMiddleware,
    add_nemo_flow_integration,
)


@pytest.fixture(name="mock_tool_execute")
def mock_tool_execute_fixture(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    async def execute_side_effect(*, func: Any, args: Any, **kwargs: Any) -> Any:
        result = func(args)
        if hasattr(result, "__await__"):
            return await result
        return result

    mock_execute = AsyncMock(side_effect=execute_side_effect)
    monkeypatch.setattr(nemo_flow.scope, "get_handle", lambda: MagicMock(name="mock_handle"))
    monkeypatch.setattr(nemo_flow.typed, "tool_execute", mock_execute)
    return mock_execute


class _MockDeepAgentsChatModel(FakeMessagesListChatModel):
    model: str = "mock-model"

    def bind_tools(self, _tools: Any, *_args: Any, **_kwargs: Any) -> _MockDeepAgentsChatModel:
        return self


@pytest.fixture(name="middleware")
def middleware_fixture() -> NemoFlowDeepAgentsMiddleware:
    return NemoFlowDeepAgentsMiddleware(agent_name="main-agent")


def _filter_mark_events(events: list[nemo_flow.Event]) -> list[nemo_flow.MarkEvent]:
    return [event for event in events if isinstance(event, nemo_flow.MarkEvent)]


def _filter_deepagents_scope_events(events: list[nemo_flow.Event]) -> list[nemo_flow.ScopeEvent]:
    return [
        event
        for event in events
        if isinstance(event, nemo_flow.ScopeEvent)
        and isinstance(event.metadata, dict)
        and event.metadata.get("integration") == "deepagents"
    ]


def _mark_data(mark: nemo_flow.MarkEvent) -> dict[str, Any]:
    assert isinstance(mark.data, dict)
    return cast(dict[str, Any], mark.data)


def _mark_metadata(mark: nemo_flow.MarkEvent) -> dict[str, Any]:
    assert isinstance(mark.metadata, dict)
    return cast(dict[str, Any], mark.metadata)


def _scope_data(event: nemo_flow.ScopeEvent) -> dict[str, Any]:
    assert isinstance(event.data, dict)
    return cast(dict[str, Any], event.data)


def _scope_metadata(event: nemo_flow.ScopeEvent) -> dict[str, Any]:
    assert isinstance(event.metadata, dict)
    return cast(dict[str, Any], event.metadata)


def _mk_tool_request(tool_name: str, args: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": args, "id": "call-1"},
        tool=None,
        state={},
        runtime=MagicMock(name="mock_runtime"),
    )


@pytest.mark.parametrize(
    ("tool_name", "args", "expected_kind", "expected_scope"),
    [
        ("task", {"name": "researcher", "task": "research GPUs"}, "subagent", "DeepAgents Subagent"),
        (
            "start_async_task",
            {"agent_name": "researcher", "task": "research GPUs"},
            "async_subagent",
            "DeepAgents Async Subagent",
        ),
        ("read_file", {"path": "/workspace/notes.md"}, "filesystem", "DeepAgents Filesystem"),
        ("execute", {"command": "python main.py"}, "sandbox", "DeepAgents Sandbox"),
    ],
)
def test_wrap_tool_call_emits_deepagents_scopes(
    tool_name: str,
    args: dict[str, Any],
    expected_kind: str,
    expected_scope: str,
    middleware: NemoFlowDeepAgentsMiddleware,
    mock_tool_execute: AsyncMock,
    subscribed_events: list[nemo_flow.Event],
):
    def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="done", tool_call_id=request.tool_call["id"])

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        result = middleware.wrap_tool_call(_mk_tool_request(tool_name, args), handler)

    assert result.content == "done"
    mock_tool_execute.assert_awaited_once()

    scopes = _filter_deepagents_scope_events(subscribed_events)
    assert [(event.name, event.scope_category) for event in scopes] == [
        (expected_scope, "start"),
        (expected_scope, "end"),
    ]
    assert _scope_metadata(scopes[0])["deepagents_kind"] == expected_kind
    assert _scope_data(scopes[0])["tool_name"] == tool_name
    assert scopes[1].data is None


async def test_awrap_tool_call_emits_deepagents_scopes(
    middleware: NemoFlowDeepAgentsMiddleware,
    mock_tool_execute: AsyncMock,
    subscribed_events: list[nemo_flow.Event],
):
    async def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="started", tool_call_id=request.tool_call["id"])

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        result = await middleware.awrap_tool_call(
            _mk_tool_request("check_async_task", {"task_id": "task-1"}),
            handler,
        )

    assert result.content == "started"
    mock_tool_execute.assert_awaited_once()

    scopes = _filter_deepagents_scope_events(subscribed_events)
    assert [(event.name, event.scope_category) for event in scopes] == [
        ("DeepAgents Async Subagent", "start"),
        ("DeepAgents Async Subagent", "end"),
    ]
    assert _scope_data(scopes[0])["task_id"] == "task-1"


def test_wrap_tool_call_emits_error_scope(
    middleware: NemoFlowDeepAgentsMiddleware,
    mock_tool_execute: AsyncMock,
    subscribed_events: list[nemo_flow.Event],
):
    def handler(request: ToolCallRequest) -> ToolMessage:
        raise RuntimeError("approval failed")

    with pytest.raises(RuntimeError, match="approval failed"):
        with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
            middleware.wrap_tool_call(_mk_tool_request("task", {"name": "researcher"}), handler)

    mock_tool_execute.assert_awaited_once()
    scopes = _filter_deepagents_scope_events(subscribed_events)
    assert [(event.name, event.scope_category) for event in scopes] == [
        ("DeepAgents Subagent", "start"),
        ("DeepAgents Subagent", "end"),
    ]
    assert scopes[1].data is None


def test_before_agent_emits_configuration_mark(subscribed_events: list[nemo_flow.Event]):
    middleware = NemoFlowDeepAgentsMiddleware(
        agent_name="main-agent",
        skills=["/skills/research/"],
        subagents=[{"name": "researcher"}],
        backend_name="StateBackend",
    )

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        middleware.before_agent(MagicMock(name="mock_state"), MagicMock(name="mock_runtime"))

    marks = _filter_mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["DeepAgents Skills Configured"]
    assert _mark_metadata(marks[0])["deepagents_kind"] == "skill"
    assert _mark_data(marks[0])["skills"] == ["/skills/research/"]
    assert _mark_data(marks[0])["subagents"] == [{"name": "researcher"}]
    assert _mark_data(marks[0])["backend"] == "StateBackend"


def test_callback_handler_emits_human_in_the_loop_marks(subscribed_events: list[nemo_flow.Event]):
    handler = NemoFlowDeepAgentsCallbackHandler()
    run_id = uuid4()
    hitl_request = {
        "action_requests": [
            {
                "name": "edit_file",
                "args": {"file_path": "/workspace/notes.md"},
                "description": "Tool execution requires approval",
            }
        ],
        "review_configs": [{"action_name": "edit_file", "allowed_decisions": ["approve", "reject"]}],
    }

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        handler.on_interrupt(
            GraphInterruptEvent(
                run_id=run_id,
                status="interrupt_after",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
                interrupts=(Interrupt(hitl_request, id="interrupt-1"),),
            )
        )
        handler.on_resume(
            GraphResumeEvent(
                run_id=run_id,
                status="pending",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
            )
        )

    marks = _filter_mark_events(subscribed_events)
    assert [mark.name for mark in marks] == [
        "DeepAgents Human In The Loop Interrupt",
        "DeepAgents Human In The Loop Resume",
    ]
    assert _mark_metadata(marks[0])["deepagents_kind"] == "human_in_the_loop"
    assert _mark_data(marks[0])["interrupts"] == [{"id": "interrupt-1", "value": hitl_request}]
    assert _mark_metadata(marks[1])["phase"] == "resume"


def test_callback_handler_falls_back_for_non_hitl_interrupt(subscribed_events: list[nemo_flow.Event]):
    handler = NemoFlowDeepAgentsCallbackHandler()
    run_id = uuid4()

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        handler.on_interrupt(
            GraphInterruptEvent(
                run_id=run_id,
                status="interrupt_after",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
                interrupts=(Interrupt("custom pause", id="interrupt-1"),),
            )
        )
        handler.on_resume(
            GraphResumeEvent(
                run_id=run_id,
                status="pending",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
            )
        )

    marks = _filter_mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["Graph Interrupt", "Graph Resume"]
    assert _mark_metadata(marks[0])["integration"] == "langgraph"
    assert "deepagents_kind" not in _mark_metadata(marks[0])


def test_add_nemo_flow_integration_preserves_backend():
    mock_backend = MagicMock(name="mock_backend")
    mock_compiled_subagent = MagicMock(name="mock_compiled_subagent")
    kwargs = add_nemo_flow_integration(
        model="mock-model",
        name="main-agent",
        skills=["/skills/main/"],
        backend=mock_backend,
        middleware=[MagicMock(name="mock_middleware")],
        subagents=[
            {"name": "researcher", "description": "Research", "skills": ["/skills/research/"]},
            mock_compiled_subagent,
        ],
    )

    assert kwargs["backend"] is mock_backend
    assert any(isinstance(item, NemoFlowDeepAgentsMiddleware) for item in kwargs["middleware"])
    assert any(isinstance(item, NemoFlowDeepAgentsMiddleware) for item in kwargs["subagents"][0]["middleware"])
    assert kwargs["subagents"][1] is mock_compiled_subagent


def test_e2e_agent(
    tmp_path: Path,
    subscribed_events: list[nemo_flow.Event],
):
    model = _MockDeepAgentsChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/turtle", "content": "shell"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="created turtle"),
        ]
    )
    kwargs = add_nemo_flow_integration(
        model=model,
        tools=[],
        name="main-agent",
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True),
    )
    agent = create_deep_agent(**kwargs)

    with nemo_flow.scope.scope("deepagents-request", nemo_flow.ScopeType.Agent):
        result = agent.invoke({"messages": [{"role": "user", "content": "Create a file named turtle."}]})

    assert (tmp_path / "turtle").read_text() == "shell"
    assert result["messages"][-1].content == "created turtle"
    found_write_file_message = False
    for message in result["messages"]:
        if (
            isinstance(message, ToolMessage)
            and message.name == "write_file"
            and message.content == "Updated file /turtle"
        ):
            found_write_file_message = True
            break

    assert found_write_file_message

    scopes = _filter_deepagents_scope_events(subscribed_events)
    assert any(
        _scope_data(event).get("tool_name") == "write_file" for event in scopes if event.scope_category == "start"
    )
