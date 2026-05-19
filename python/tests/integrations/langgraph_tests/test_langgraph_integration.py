# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangGraph NeMo Flow callback integration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import pytest
from typing_extensions import TypedDict

import nemo_flow

if TYPE_CHECKING:
    from langgraph.graph import CompiledStateGraph

    from nemo_flow.integrations.langgraph import NemoFlowCallbackHandler


class State(TypedDict):
    value: int


def increment(state: State) -> State:
    return {"value": state["value"] + 1}


async def aincrement(state: State) -> State:
    await asyncio.sleep(0)
    return {"value": state["value"] + 1}


def _build_graph(use_async: bool = False) -> CompiledStateGraph:
    from langgraph.graph import END, START, StateGraph

    # The cast here avoids a ty linting error
    builder = StateGraph(cast(Any, State))
    if use_async:
        builder.add_node("increment", aincrement)
    else:
        builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile()


@pytest.fixture(name="sync_graph")
def graph_fixture() -> CompiledStateGraph:
    return _build_graph(use_async=False)


@pytest.fixture(name="async_graph")
def async_graph_fixture() -> CompiledStateGraph:
    return _build_graph(use_async=True)


@pytest.fixture(name="callback_handler")
def callback_handler_fixture() -> NemoFlowCallbackHandler:
    from nemo_flow.integrations.langgraph import NemoFlowCallbackHandler

    return NemoFlowCallbackHandler()


def _events_to_strings(events: list[nemo_flow.Event]) -> list[str]:
    event_strings: list[str] = []

    for event in events:
        if isinstance(event, nemo_flow.ScopeEvent):
            event_strings.append(f"{event.kind}.{event.scope_category}.{event.name}")
        else:
            event_strings.append(f"{event.kind}.{event.name}")

    return event_strings


def test_handler_type(callback_handler: NemoFlowCallbackHandler):
    from langgraph.callbacks import GraphCallbackHandler

    from nemo_flow.integrations.langchain.callbacks import NemoFlowCallbackHandler as LangChainCallbackHandler

    assert isinstance(callback_handler, LangChainCallbackHandler)
    assert isinstance(callback_handler, GraphCallbackHandler)


class TestGraphCallbacks:
    _expected_events = [
        "scope.start.request",
        "scope.start.LangGraph",
        "scope.start.increment",
        "scope.end.increment",
        "scope.end.LangGraph",
        "scope.end.request",
    ]

    def test_sync(
        self,
        sync_graph: CompiledStateGraph,
        subscribed_events: list[nemo_flow.Event],
        callback_handler: NemoFlowCallbackHandler,
    ):
        with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
            result = sync_graph.invoke({"value": 1}, config={"callbacks": [callback_handler]})

        assert result == {"value": 2}
        assert _events_to_strings(subscribed_events) == self._expected_events

    async def test_async(
        self,
        async_graph: CompiledStateGraph,
        subscribed_events: list[nemo_flow.Event],
        callback_handler: NemoFlowCallbackHandler,
    ):
        with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
            result = await async_graph.ainvoke({"value": 1}, config={"callbacks": [callback_handler]})

        assert result == {"value": 2}
        assert _events_to_strings(subscribed_events) == self._expected_events


def test_graph_lifecycle_callbacks_emit_marks(
    subscribed_events: list[nemo_flow.Event],
    callback_handler: NemoFlowCallbackHandler,
):
    from langgraph.callbacks import GraphInterruptEvent, GraphResumeEvent
    from langgraph.types import Interrupt

    run_id = uuid4()

    expected_event_strings = [
        "scope.start.request",
        "mark.Graph Interrupt",
        "mark.Graph Resume",
        "scope.end.request",
    ]

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        callback_handler.on_interrupt(
            GraphInterruptEvent(
                run_id=run_id,
                status="interrupt_after",
                checkpoint_id="checkpoint-2",
                checkpoint_ns=("parent",),
                interrupts=(Interrupt("needs approval", id="interrupt-1"),),
            )
        )

        callback_handler.on_resume(
            GraphResumeEvent(
                run_id=run_id,
                status="pending",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent", "child"),
            )
        )

    assert _events_to_strings(subscribed_events) == expected_event_strings

    interrupt_event = subscribed_events[1]
    assert isinstance(interrupt_event, nemo_flow.MarkEvent)
    interrupt_data = cast(dict[str, Any], interrupt_event.data)
    assert interrupt_data["interrupts"] == [{"id": "interrupt-1", "value": "needs approval"}]

    resume_event = subscribed_events[2]
    assert isinstance(resume_event, nemo_flow.MarkEvent)
    resume_data = cast(dict[str, Any], resume_event.data)
    assert resume_data["checkpoint_ns"] == ["parent", "child"]
    assert resume_event.metadata == {"integration": "langgraph"}
