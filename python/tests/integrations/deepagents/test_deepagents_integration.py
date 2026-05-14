# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Deep Agents NeMo Flow integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
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


class _MockDeepAgentsChatModel(FakeMessagesListChatModel):
    model: str = "mock-model"

    def bind_tools(self, _tools: Any, *_args: Any, **_kwargs: Any) -> _MockDeepAgentsChatModel:
        return self


def _filter_mark_events(events: list[nemo_flow.Event]) -> list[nemo_flow.MarkEvent]:
    return [event for event in events if isinstance(event, nemo_flow.MarkEvent)]


def _mark_data(mark: nemo_flow.MarkEvent) -> dict[str, Any]:
    assert isinstance(mark.data, dict)
    return cast(dict[str, Any], mark.data)


def _mark_metadata(mark: nemo_flow.MarkEvent) -> dict[str, Any]:
    assert isinstance(mark.metadata, dict)
    return cast(dict[str, Any], mark.metadata)


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
    reviewer_description = "Reviews filesystem work performed by the main agent."
    reviewer_model = _MockDeepAgentsChatModel(
        responses=[
            AIMessage(content="reviewer verified turtle"),
        ]
    )
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
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Review the file creation result and report whether turtle was created.",
                            "subagent_type": "reviewer",
                        },
                        "id": "call-2",
                    }
                ],
            ),
            AIMessage(content="created turtle after reviewer verified turtle"),
        ]
    )
    kwargs = add_nemo_flow_integration(
        model=model,
        tools=[],
        name="main-agent",
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True),
        subagents=[
            {
                "name": "reviewer",
                "description": reviewer_description,
                "system_prompt": "Review the delegated task and return one concise verification sentence.",
                "model": reviewer_model,
                "tools": [],
            }
        ],
    )
    agent = create_deep_agent(**kwargs)

    with nemo_flow.scope.scope("deepagents-request", nemo_flow.ScopeType.Agent):
        result = agent.invoke({"messages": [{"role": "user", "content": "Create a file named turtle."}]})

    assert (tmp_path / "turtle").read_text() == "shell"
    assert result["messages"][-1].content == "created turtle after reviewer verified turtle"
    found_write_file_message = False
    found_subagent_message = False
    for message in result["messages"]:
        if (
            isinstance(message, ToolMessage)
            and message.name == "write_file"
            and message.content == "Updated file /turtle"
        ):
            found_write_file_message = True
        if isinstance(message, ToolMessage) and message.content == "reviewer verified turtle":
            found_subagent_message = True

    assert found_write_file_message
    assert found_subagent_message

    marks = _filter_mark_events(subscribed_events)
    configured_subagents = [
        subagent
        for mark in marks
        if mark.name == "DeepAgents Skills Configured"
        for subagent in _mark_data(mark).get("subagents", [])
        if isinstance(subagent, dict)
    ]
    assert any(
        subagent.get("name") == "reviewer" and subagent.get("description") == reviewer_description
        for subagent in configured_subagents
    )
