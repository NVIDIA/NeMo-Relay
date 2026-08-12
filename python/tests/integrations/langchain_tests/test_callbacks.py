# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Relay callback handler."""

from __future__ import annotations

import types
import typing
from unittest.mock import ANY, MagicMock
from uuid import uuid4

import pytest

if typing.TYPE_CHECKING:
    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler


def _make_mock_nemo_relay() -> MagicMock:
    """Build a minimal mock of the ``nemo_relay`` module.

    The mock models the runtime's LIFO scope stack rather than accepting any call in any
    order. The handler closes whichever scope is on top, so a mock that cannot answer
    "what is on top?" would report no closes at all.
    """
    mock_nemo_relay = MagicMock(name="nemo_relay")
    mock_nemo_relay.ScopeType = types.SimpleNamespace(Agent="Agent")

    stack = [types.SimpleNamespace(uuid=str(uuid4()), name="root")]

    def push(name, scope_type, **kwargs):
        handle = types.SimpleNamespace(
            uuid=str(uuid4()),
            name=name,
            scope_type=scope_type,
            kwargs=kwargs,
        )
        stack.append(handle)
        return handle

    def pop(handle, **kwargs):
        # Closing anything but the top is what the real stack rejects, so the mock has to
        # reject it too; otherwise a handler that closed the wrong scope would pass.
        assert handle is stack[-1], "closed a scope that was not on top"
        stack.pop()

    scope = types.SimpleNamespace()
    scope.push = MagicMock(side_effect=push)
    scope.pop = MagicMock(side_effect=pop)
    scope.get_handle = MagicMock(side_effect=lambda: stack[-1])
    mock_nemo_relay.scope = scope
    # A stable object so the handler can tell one stack from another by identity.
    mock_nemo_relay.get_scope_stack = MagicMock(side_effect=lambda: stack)
    return mock_nemo_relay


@pytest.fixture(name="callbacks_module", scope="session")
def callbacks_module_fixture() -> types.ModuleType:
    """Fixture to provide the callbacks module."""
    import nemo_relay.integrations.langchain.callbacks as callbacks_module

    return callbacks_module


@pytest.fixture()
def mock_nemo_relay(monkeypatch: pytest.MonkeyPatch, callbacks_module: types.ModuleType) -> MagicMock:
    mock_nemo_relay = _make_mock_nemo_relay()
    monkeypatch.setattr(callbacks_module, "nemo_relay", mock_nemo_relay)
    return mock_nemo_relay


@pytest.fixture()
def handler(mock_nemo_relay: MagicMock) -> NemoRelayCallbackHandler:
    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

    return NemoRelayCallbackHandler()


class TestScopeLifecycle:
    """Verify that chain start/end/error map to scope push/pop."""

    def test_handler_runs_inline_for_async_callback_managers(self, handler: NemoRelayCallbackHandler):
        assert handler.run_inline is True

    def test_on_chain_start_pushes_scope(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
        run_id = uuid4()

        handler.on_chain_start(
            {"name": "MyChain"},
            {"input": "test"},
            run_id=run_id,
            metadata={"source": "unit-test"},
        )

        mock_nemo_relay.scope.push.assert_called_once()
        args, kwargs = mock_nemo_relay.scope.push.call_args
        assert args == ("MyChain", mock_nemo_relay.ScopeType.Agent)
        assert kwargs["input"] == {"input": "test"}
        assert kwargs["metadata"] == {
            "langchain_run_id": str(run_id),
            "source": "unit-test",
        }
        assert run_id in handler._scope_handles

    def test_on_chain_start_uses_callback_name(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
        run_id = uuid4()

        handler.on_chain_start(
            {},
            {"input": "test"},
            run_id=run_id,
            name="LangGraph",
        )

        assert mock_nemo_relay.scope.push.call_args.args[0] == "LangGraph"

    def test_on_chain_end_pops_scope(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
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

        mock_nemo_relay.scope.pop.assert_called_once_with(
            handle,
            output={"output": "result"},
            metadata={"otel.status_code": "OK"},
            timestamp=ANY,
        )
        assert run_id not in handler._scope_handles

    def test_on_chain_error_pops_scope(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
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

        mock_nemo_relay.scope.pop.assert_called_once_with(
            handle,
            output={"error": "RuntimeError('boom')"},
            metadata={"otel.status_code": "ERROR", "otel.status_description": "boom"},
            timestamp=ANY,
        )
        assert run_id not in handler._scope_handles

    def test_on_chain_end_prepares_command_outputs(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

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

        mock_nemo_relay.scope.pop.assert_called_once_with(
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
            metadata={"otel.status_code": "OK"},
            timestamp=ANY,
        )

    def test_parent_scope_passed_to_push(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
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

        child_call = mock_nemo_relay.scope.push.call_args_list[1]
        assert child_call.kwargs["handle"] is parent_handle

    def test_chain_end_without_start_is_noop(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
        handler.on_chain_end(
            {"output": "result"},
            run_id=uuid4(),
        )

        mock_nemo_relay.scope.pop.assert_not_called()

    def test_name_fallback_to_id(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
        run_id = uuid4()

        handler.on_chain_start(
            {"id": ["langchain", "core", "RunnableSequence"]},
            {},
            run_id=run_id,
        )

        assert mock_nemo_relay.scope.push.call_args.args[0] == "RunnableSequence"


class TestGracefulNoOp:
    """Verify callbacks are silent if the module-level runtime is unavailable."""

    def test_no_nemo_relay_on_chain_start(self, monkeypatch: pytest.MonkeyPatch, callbacks_module: types.ModuleType):
        monkeypatch.setattr(callbacks_module, "nemo_relay", None)
        from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

        handler = NemoRelayCallbackHandler()

        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())

    def test_no_nemo_relay_on_chain_end(self, monkeypatch: pytest.MonkeyPatch, callbacks_module: types.ModuleType):
        monkeypatch.setattr(callbacks_module, "nemo_relay", None)
        from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

        handler = NemoRelayCallbackHandler()

        handler.on_chain_end({}, run_id=uuid4())

    def test_no_nemo_relay_on_chain_error(self, monkeypatch: pytest.MonkeyPatch, callbacks_module: types.ModuleType):
        monkeypatch.setattr(callbacks_module, "nemo_relay", None)
        from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

        handler = NemoRelayCallbackHandler()

        handler.on_chain_error(RuntimeError("e"), run_id=uuid4())


class TestErrorSwallowing:
    """Ensure NeMo Relay errors never propagate."""

    def test_scope_push_error_swallowed(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
        mock_nemo_relay.scope.push.side_effect = RuntimeError("nemo relay failure")

        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())

    def test_scope_pop_error_swallowed(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
        run_id = uuid4()
        handler.on_chain_start({"name": "x"}, {}, run_id=run_id)
        mock_nemo_relay.scope.pop.side_effect = RuntimeError("nemo relay failure")

        handler.on_chain_end({}, run_id=run_id)
