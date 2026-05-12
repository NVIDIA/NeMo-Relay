# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Deep Agents NeMo Flow integration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from deepagents.backends.protocol import ExecuteResponse, LsResult, ReadResult, SandboxBackendProtocol
from deepagents.middleware.filesystem import supports_execution
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.callbacks import GraphInterruptEvent, GraphResumeEvent
from langgraph.types import Interrupt

import nemo_flow
from nemo_flow.integrations.deepagents import (
    NemoFlowDeepAgentsBackend,
    NemoFlowDeepAgentsCallbackHandler,
    NemoFlowDeepAgentsMiddleware,
    add_nemo_flow_integration,
    observe_backend,
)


@pytest.fixture(name="subscribed_events")
def subscribed_events_fixture() -> Iterator[list[nemo_flow.Event]]:
    events: list[nemo_flow.Event] = []

    def event_recorder(event: nemo_flow.Event) -> None:
        events.append(event)

    subscriber_name = f"deepagents-test-{uuid4()}"
    nemo_flow.subscribers.register(subscriber_name, event_recorder)
    yield events
    nemo_flow.subscribers.deregister(subscriber_name)


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


@pytest.fixture(name="middleware")
def middleware_fixture() -> NemoFlowDeepAgentsMiddleware:
    return NemoFlowDeepAgentsMiddleware(agent_name="main-agent")


def mark_events(events: list[nemo_flow.Event]) -> list[nemo_flow.MarkEvent]:
    return [event for event in events if isinstance(event, nemo_flow.MarkEvent)]


def tool_request(tool_name: str, args: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": args, "id": "call-1"},
        tool=None,
        state={},
        runtime=MagicMock(name="mock_runtime"),
    )


@pytest.mark.parametrize(
    ("tool_name", "args", "expected_kind", "expected_mark"),
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
def test_wrap_tool_call_emits_deepagents_marks(
    tool_name: str,
    args: dict[str, Any],
    expected_kind: str,
    expected_mark: str,
    middleware: NemoFlowDeepAgentsMiddleware,
    mock_tool_execute: AsyncMock,
    subscribed_events: list[nemo_flow.Event],
) -> None:
    def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="done", tool_call_id=request.tool_call["id"])

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        result = middleware.wrap_tool_call(tool_request(tool_name, args), handler)

    assert result.content == "done"
    mock_tool_execute.assert_awaited_once()

    marks = mark_events(subscribed_events)
    assert [mark.name for mark in marks] == [f"{expected_mark} Start", f"{expected_mark} End"]
    assert marks[0].metadata["deepagents_kind"] == expected_kind
    assert marks[0].metadata["phase"] == "start"
    assert marks[0].data["tool_name"] == tool_name
    assert marks[1].metadata["phase"] == "end"


async def test_awrap_tool_call_emits_deepagents_marks(
    middleware: NemoFlowDeepAgentsMiddleware,
    mock_tool_execute: AsyncMock,
    subscribed_events: list[nemo_flow.Event],
) -> None:
    async def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="started", tool_call_id=request.tool_call["id"])

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        result = await middleware.awrap_tool_call(
            tool_request("check_async_task", {"task_id": "task-1"}),
            handler,
        )

    assert result.content == "started"
    mock_tool_execute.assert_awaited_once()

    marks = mark_events(subscribed_events)
    assert [mark.name for mark in marks] == [
        "DeepAgents Async Subagent Start",
        "DeepAgents Async Subagent End",
    ]
    assert marks[0].data["task_id"] == "task-1"


def test_wrap_tool_call_emits_error_mark(
    middleware: NemoFlowDeepAgentsMiddleware,
    mock_tool_execute: AsyncMock,
    subscribed_events: list[nemo_flow.Event],
) -> None:
    def handler(request: ToolCallRequest) -> ToolMessage:
        raise RuntimeError("approval failed")

    with pytest.raises(RuntimeError, match="approval failed"):
        with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
            middleware.wrap_tool_call(tool_request("task", {"name": "researcher"}), handler)

    mock_tool_execute.assert_awaited_once()
    marks = mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["DeepAgents Subagent Start", "DeepAgents Subagent Error"]
    assert "RuntimeError" in marks[1].data["error"]


def test_before_agent_emits_configuration_mark(subscribed_events: list[nemo_flow.Event]) -> None:
    middleware = NemoFlowDeepAgentsMiddleware(
        agent_name="main-agent",
        skills=["/skills/research/"],
        subagents=[{"name": "researcher"}],
        backend_name="StateBackend",
    )

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        middleware.before_agent(MagicMock(name="mock_state"), MagicMock(name="mock_runtime"))

    marks = mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["DeepAgents Skills Configured"]
    assert marks[0].metadata["deepagents_kind"] == "skill"
    assert marks[0].data["skills"] == ["/skills/research/"]
    assert marks[0].data["subagents"] == [{"name": "researcher"}]
    assert marks[0].data["backend"] == "StateBackend"


def test_callback_handler_emits_human_in_the_loop_marks(subscribed_events: list[nemo_flow.Event]) -> None:
    handler = NemoFlowDeepAgentsCallbackHandler()
    run_id = uuid4()

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        handler.on_interrupt(
            GraphInterruptEvent(
                run_id=run_id,
                status="interrupt_after",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
                interrupts=(Interrupt("needs approval", id="interrupt-1"),),
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

    marks = mark_events(subscribed_events)
    assert [mark.name for mark in marks] == [
        "DeepAgents Human In The Loop Interrupt",
        "DeepAgents Human In The Loop Resume",
    ]
    assert marks[0].metadata["deepagents_kind"] == "human_in_the_loop"
    assert marks[0].data["interrupts"] == [{"id": "interrupt-1", "value": "needs approval"}]
    assert marks[1].metadata["phase"] == "resume"


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs", "expected_call_kwargs", "expected_kind"),
    [
        ("read", ("/workspace/notes.md",), {}, {"offset": 0, "limit": 2000}, "filesystem"),
        ("grep", ("TODO",), {"path": "/workspace"}, {"path": "/workspace", "glob": None}, "filesystem"),
    ],
)
def test_observe_backend_emits_sync_marks(
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected_call_kwargs: dict[str, Any],
    expected_kind: str,
    subscribed_events: list[nemo_flow.Event],
) -> None:
    mock_backend = MagicMock(name="mock_backend")
    getattr(mock_backend, method_name).return_value = {"ok": True}
    backend = observe_backend(mock_backend, name="MockBackend")

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        result = getattr(backend, method_name)(*args, **kwargs)

    assert result == {"ok": True}
    getattr(mock_backend, method_name).assert_called_once_with(*args, **expected_call_kwargs)

    marks = mark_events(subscribed_events)
    assert marks[0].metadata["deepagents_kind"] == expected_kind
    assert marks[0].data["backend"] == "MockBackend"
    assert marks[0].data["method"] == method_name
    assert marks[1].metadata["phase"] == "end"


async def test_observe_backend_emits_async_marks(subscribed_events: list[nemo_flow.Event]) -> None:
    mock_backend = MagicMock(name="mock_backend")
    mock_backend.aread = AsyncMock(return_value=ReadResult(file_data={"content": "contents", "encoding": "utf-8"}))
    backend = observe_backend(mock_backend, name="MockBackend")

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        result = await backend.aread("/workspace/notes.md")

    assert result == ReadResult(file_data={"content": "contents", "encoding": "utf-8"})
    mock_backend.aread.assert_awaited_once_with("/workspace/notes.md", offset=0, limit=2000)

    marks = mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["DeepAgents Filesystem Start", "DeepAgents Filesystem End"]
    assert "ReadResult" in marks[1].data["result"]


def test_observe_backend_preserves_sandbox_protocol(subscribed_events: list[nemo_flow.Event]) -> None:
    class MockSandboxBackend(SandboxBackendProtocol):
        @property
        def id(self) -> str:
            return "sandbox-1"

        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            return ExecuteResponse(output=f"{command}:{timeout}", exit_code=0)

        def ls(self, path: str) -> LsResult:
            return LsResult(entries=[{"path": path}])

    plain_backend = observe_backend(MagicMock(name="plain_backend"))
    sandbox_backend = observe_backend(MockSandboxBackend(), name="MockSandbox")

    assert not supports_execution(plain_backend)
    assert supports_execution(sandbox_backend)
    assert sandbox_backend.id == "sandbox-1"

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        result = sandbox_backend.execute("python main.py", timeout=10)

    assert result == ExecuteResponse(output="python main.py:10", exit_code=0)

    marks = mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["DeepAgents Sandbox Start", "DeepAgents Sandbox End"]
    assert marks[0].data["method"] == "execute"


def test_add_nemo_flow_integration_instruments_kwargs() -> None:
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

    assert isinstance(kwargs["backend"], NemoFlowDeepAgentsBackend)
    assert any(isinstance(item, NemoFlowDeepAgentsMiddleware) for item in kwargs["middleware"])
    assert any(isinstance(item, NemoFlowDeepAgentsMiddleware) for item in kwargs["subagents"][0]["middleware"])
    assert kwargs["subagents"][1] is mock_compiled_subagent
