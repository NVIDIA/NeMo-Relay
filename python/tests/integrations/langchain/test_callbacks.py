# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Flow callback handler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from nemo_flow.integrations.langchain import callbacks as callbacks_module
from nemo_flow.integrations.langchain.callbacks import NemoFlowCallbackHandler


def _make_mock_nemo_flow() -> MagicMock:
    """Build a minimal mock of the ``nemo_flow`` module."""
    mock_nemo_flow = MagicMock(name="nemo_flow")
    mock_nemo_flow.ScopeType = SimpleNamespace(Agent="Agent")

    scope = SimpleNamespace()
    scope.push = MagicMock(
        side_effect=lambda name, scope_type, **kwargs: SimpleNamespace(
            uuid=str(uuid4()),
            name=name,
            scope_type=scope_type,
            kwargs=kwargs,
        )
    )
    scope.pop = MagicMock()
    mock_nemo_flow.scope = scope
    return mock_nemo_flow


@pytest.fixture()
def mock_nemo_flow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_nemo_flow = _make_mock_nemo_flow()
    monkeypatch.setattr(callbacks_module, "nemo_flow", mock_nemo_flow)
    return mock_nemo_flow


@pytest.fixture()
def handler(mock_nemo_flow: MagicMock) -> NemoFlowCallbackHandler:
    return NemoFlowCallbackHandler()


class TestScopeLifecycle:
    """Verify that chain start/end/error map to scope push/pop."""

    def test_handler_runs_inline_for_async_callback_managers(self, handler: NemoFlowCallbackHandler):
        assert handler.run_inline is True

    def test_on_chain_start_pushes_scope(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        run_id = uuid4()

        handler.on_chain_start(
            {"name": "MyChain"},
            {"input": "test"},
            run_id=run_id,
            metadata={"source": "unit-test"},
        )

        mock_nemo_flow.scope.push.assert_called_once()
        args, kwargs = mock_nemo_flow.scope.push.call_args
        assert args == ("MyChain", mock_nemo_flow.ScopeType.Agent)
        assert kwargs["input"] == {"input": "test"}
        assert kwargs["metadata"] == {
            "langchain_run_id": str(run_id),
            "source": "unit-test",
        }
        assert run_id in handler._scope_handles

    def test_on_chain_start_uses_callback_name(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        run_id = uuid4()

        handler.on_chain_start(
            {},
            {"input": "test"},
            run_id=run_id,
            name="LangGraph",
        )

        assert mock_nemo_flow.scope.push.call_args.args[0] == "LangGraph"

    def test_on_chain_end_pops_scope(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        run_id = uuid4()
        handler.on_chain_start(
            {"name": "MyChain"},
            {"input": "test"},
            run_id=run_id,
        )
        handle = handler._scope_handles[run_id]

        handler.on_chain_end(
            {"output": "result"},
            run_id=run_id,
        )

        mock_nemo_flow.scope.pop.assert_called_once_with(handle, output={"output": "result"})
        assert run_id not in handler._scope_handles

    def test_on_chain_error_pops_scope(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        run_id = uuid4()
        handler.on_chain_start(
            {"name": "MyChain"},
            {"input": "test"},
            run_id=run_id,
        )
        handle = handler._scope_handles[run_id]

        handler.on_chain_error(
            RuntimeError("boom"),
            run_id=run_id,
        )

        mock_nemo_flow.scope.pop.assert_called_once_with(handle, output={"error": "RuntimeError('boom')"})
        assert run_id not in handler._scope_handles

    def test_on_chain_end_prepares_command_outputs(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        run_id = uuid4()
        handler.on_chain_start(
            {"name": "MyChain"},
            {"input": "test"},
            run_id=run_id,
        )
        handle = handler._scope_handles[run_id]

        handler.on_chain_end(
            {
                "result": Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content="done",
                                tool_call_id="call-1",
                                name="task",
                            )
                        ]
                    }
                )
            },
            run_id=run_id,
        )

        mock_nemo_flow.scope.pop.assert_called_once_with(
            handle,
            output={
                "result": {
                    "type": "command",
                    "command": {
                        "graph": None,
                        "update": {
                            "messages": [
                                {
                                    "type": "tool_message",
                                    "tool_call": {
                                        "name": "task",
                                        "id": None,
                                        "tool_call_id": "call-1",
                                        "content": "done",
                                    },
                                }
                            ]
                        },
                        "resume": None,
                        "goto": [],
                    },
                }
            },
        )

    def test_parent_scope_passed_to_push(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        parent_id = uuid4()
        child_id = uuid4()
        handler.on_chain_start(
            {"name": "Parent"},
            {},
            run_id=parent_id,
        )
        parent_handle = handler._scope_handles[parent_id]

        handler.on_chain_start(
            {"name": "Child"},
            {},
            run_id=child_id,
            parent_run_id=parent_id,
        )

        child_call = mock_nemo_flow.scope.push.call_args_list[1]
        assert child_call.kwargs["handle"] is parent_handle

    def test_chain_end_without_start_is_noop(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        handler.on_chain_end(
            {"output": "result"},
            run_id=uuid4(),
        )

        mock_nemo_flow.scope.pop.assert_not_called()

    def test_name_fallback_to_id(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        run_id = uuid4()

        handler.on_chain_start(
            {"id": ["langchain", "core", "RunnableSequence"]},
            {},
            run_id=run_id,
        )

        assert mock_nemo_flow.scope.push.call_args.args[0] == "RunnableSequence"


class TestGracefulNoOp:
    """Verify callbacks are silent if the module-level runtime is unavailable."""

    def test_no_nemo_flow_on_chain_start(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(callbacks_module, "nemo_flow", None)
        handler = NemoFlowCallbackHandler()

        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())

    def test_no_nemo_flow_on_chain_end(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(callbacks_module, "nemo_flow", None)
        handler = NemoFlowCallbackHandler()

        handler.on_chain_end({}, run_id=uuid4())

    def test_no_nemo_flow_on_chain_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(callbacks_module, "nemo_flow", None)
        handler = NemoFlowCallbackHandler()

        handler.on_chain_error(RuntimeError("e"), run_id=uuid4())


class TestErrorSwallowing:
    """Ensure NeMo Flow errors never propagate."""

    def test_scope_push_error_swallowed(self, mock_nemo_flow: MagicMock):
        mock_nemo_flow.scope.push.side_effect = RuntimeError("nemo flow failure")
        handler = NemoFlowCallbackHandler()

        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())

    def test_scope_pop_error_swallowed(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: MagicMock):
        run_id = uuid4()
        handler.on_chain_start({"name": "x"}, {}, run_id=run_id)
        mock_nemo_flow.scope.pop.side_effect = RuntimeError("nemo flow failure")

        handler.on_chain_end({}, run_id=run_id)
