# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangGraph NeMo Flow callback integration."""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import CallbackManager
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent
from langgraph.graph import END, START, StateGraph
from langgraph.types import Interrupt
from typing_extensions import TypedDict

import nemo_flow
from nemo_flow.integrations.langchain import NemoFlowMiddleware
from nemo_flow.integrations.langchain.callbacks import NemoFlowCallbackHandler as LangChainCallbackHandler
from nemo_flow.integrations.langgraph import (
    NemoFlowCallbackHandler,
    instrument_graph,
    with_nemo_flow_callbacks,
)


class State(TypedDict):
    value: int


def _build_graph() -> Any:
    def increment(state: State) -> State:
        return {"value": state["value"] + 1}

    builder = StateGraph(State)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile()


def _build_async_graph() -> Any:
    async def increment(state: State) -> State:
        await asyncio.sleep(0)
        return {"value": state["value"] + 1}

    builder = StateGraph(State)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile()


def _record_events() -> tuple[list[Any], str]:
    events: list[Any] = []
    subscriber_name = f"langgraph-test-{uuid4()}"
    nemo_flow.subscribers.register(subscriber_name, events.append)
    return events, subscriber_name


def test_langgraph_handler_builds_on_langchain_handler() -> None:
    handler = NemoFlowCallbackHandler()

    assert isinstance(handler, LangChainCallbackHandler)
    assert isinstance(handler, GraphCallbackHandler)
    assert handler.run_inline is True


def test_with_nemo_flow_callbacks_preserves_existing_callbacks() -> None:
    class ExistingHandler(GraphCallbackHandler):
        pass

    existing = ExistingHandler()
    config = with_nemo_flow_callbacks({"callbacks": [existing]})

    callbacks = config["callbacks"]
    assert callbacks[0] is existing
    assert isinstance(callbacks[1], NemoFlowCallbackHandler)


def test_with_nemo_flow_callbacks_handles_callback_managers() -> None:
    manager = CallbackManager([])
    config = with_nemo_flow_callbacks({"callbacks": manager})

    callbacks = config["callbacks"]
    assert callbacks is not manager
    assert any(isinstance(handler, NemoFlowCallbackHandler) for handler in callbacks.handlers)


def test_instrumented_graph_invoke_emits_named_graph_and_node_scopes() -> None:
    graph = instrument_graph(_build_graph())
    events, subscriber_name = _record_events()

    try:
        with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
            result = graph.invoke({"value": 1})
    finally:
        nemo_flow.subscribers.deregister(subscriber_name)

    assert result == {"value": 2}
    scope_names = [event.name for event in events if event.kind == "scope" and event.scope_category == "start"]
    assert scope_names == ["request", "LangGraph", "increment"]


def test_instrumented_graph_ainvoke_completes_with_inline_callbacks() -> None:
    graph = instrument_graph(_build_async_graph())

    async def run_graph() -> dict[str, int]:
        with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
            return await graph.ainvoke({"value": 1})

    assert asyncio.run(run_graph()) == {"value": 2}


def test_graph_lifecycle_callbacks_emit_marks() -> None:
    handler = NemoFlowCallbackHandler()
    events, subscriber_name = _record_events()
    run_id = uuid4()

    try:
        with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
            handler.on_resume(
                GraphResumeEvent(
                    run_id=run_id,
                    status="pending",
                    checkpoint_id="checkpoint-1",
                    checkpoint_ns=("parent", "child"),
                )
            )
            handler.on_interrupt(
                GraphInterruptEvent(
                    run_id=run_id,
                    status="interrupt_after",
                    checkpoint_id="checkpoint-2",
                    checkpoint_ns=("parent",),
                    interrupts=(Interrupt("needs approval", id="interrupt-1"),),
                )
            )
    finally:
        nemo_flow.subscribers.deregister(subscriber_name)

    marks = [event for event in events if event.kind == "mark"]
    assert [event.name for event in marks] == ["Graph Resume", "Graph Interrupt"]
    assert marks[0].data["checkpoint_ns"] == ["parent", "child"]
    assert marks[1].data["interrupts"] == [{"id": "interrupt-1", "value": "needs approval"}]
    assert marks[1].metadata == {"integration": "langgraph"}


def test_instrumented_graph_stream_emits_public_task_marks() -> None:
    graph = instrument_graph(_build_graph())
    events, subscriber_name = _record_events()

    try:
        with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
            chunks = list(graph.stream({"value": 1}, stream_mode="tasks"))
    finally:
        nemo_flow.subscribers.deregister(subscriber_name)

    assert len(chunks) == 2
    marks = [event.name for event in events if event.kind == "mark"]
    assert marks == ["Graph Task Start", "Graph Task End"]


def test_langgraph_extra_includes_langchain_integration_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extra = pyproject["project"]["optional-dependencies"]["langgraph"]

    assert "langchain-core" in extra
    assert any(requirement.startswith("langchain>=") for requirement in extra)
    assert any(requirement.startswith("langgraph>=") for requirement in extra)
    assert NemoFlowMiddleware is not None
