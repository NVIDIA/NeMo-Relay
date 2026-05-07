# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Flow callback handler."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from langchain_nemo_flow import callbacks as callbacks_module
from langchain_nemo_flow.callbacks import NemoFlowCallbackHandler


def _make_mock_nemo_flow() -> ModuleType:
    """Build a minimal mock of the ``nemo_flow`` module."""
    mock_nemo_flow = ModuleType("nemo_flow")
    mock_nemo_flow.ScopeType = SimpleNamespace(Agent="Agent")  # ty: ignore[unresolved-attribute]

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
    mock_nemo_flow.scope = scope  # ty: ignore[unresolved-attribute]
    return mock_nemo_flow


@pytest.fixture()
def mock_nemo_flow(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    mock_nemo_flow = _make_mock_nemo_flow()
    monkeypatch.setattr(callbacks_module, "nemo_flow", mock_nemo_flow)
    return mock_nemo_flow


@pytest.fixture()
def handler(mock_nemo_flow: ModuleType) -> NemoFlowCallbackHandler:
    return NemoFlowCallbackHandler()


class TestScopeLifecycle:
    """Verify that chain start/end/error map to scope push/pop."""

    def test_on_chain_start_pushes_scope(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: ModuleType) -> None:
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

    def test_on_chain_end_pops_scope(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: ModuleType) -> None:
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

    def test_on_chain_error_pops_scope(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: ModuleType) -> None:
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

    def test_parent_scope_passed_to_push(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: ModuleType) -> None:
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

    def test_chain_end_without_start_is_noop(
        self, handler: NemoFlowCallbackHandler, mock_nemo_flow: ModuleType
    ) -> None:
        handler.on_chain_end(
            {"output": "result"},
            run_id=uuid4(),
        )

        mock_nemo_flow.scope.pop.assert_not_called()

    def test_name_fallback_to_id(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: ModuleType) -> None:
        run_id = uuid4()

        handler.on_chain_start(
            {"id": ["langchain", "core", "RunnableSequence"]},
            {},
            run_id=run_id,
        )

        assert mock_nemo_flow.scope.push.call_args.args[0] == "RunnableSequence"


class TestGracefulNoOp:
    """Verify callbacks are silent if the module-level runtime is unavailable."""

    def test_no_nemo_flow_on_chain_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(callbacks_module, "nemo_flow", None)
        handler = NemoFlowCallbackHandler()

        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())

    def test_no_nemo_flow_on_chain_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(callbacks_module, "nemo_flow", None)
        handler = NemoFlowCallbackHandler()

        handler.on_chain_end({}, run_id=uuid4())

    def test_no_nemo_flow_on_chain_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(callbacks_module, "nemo_flow", None)
        handler = NemoFlowCallbackHandler()

        handler.on_chain_error(RuntimeError("e"), run_id=uuid4())


class TestErrorSwallowing:
    """Ensure NeMo Flow errors never propagate."""

    def test_scope_push_error_swallowed(self, mock_nemo_flow: ModuleType) -> None:
        mock_nemo_flow.scope.push.side_effect = RuntimeError("nemo flow failure")
        handler = NemoFlowCallbackHandler()

        handler.on_chain_start({"name": "x"}, {}, run_id=uuid4())

    def test_scope_pop_error_swallowed(self, handler: NemoFlowCallbackHandler, mock_nemo_flow: ModuleType) -> None:
        run_id = uuid4()
        handler.on_chain_start({"name": "x"}, {}, run_id=run_id)
        mock_nemo_flow.scope.pop.side_effect = RuntimeError("nemo flow failure")

        handler.on_chain_end({}, run_id=run_id)
