# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangGraph NeMo Flow callback integration."""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent
from langgraph.graph import END, START, StateGraph
from langgraph.types import Interrupt
from typing_extensions import TypedDict

import pytest

import nemo_flow
from nemo_flow.integrations.langchain.callbacks import NemoFlowCallbackHandler as LangChainCallbackHandler
from nemo_flow.integrations.langgraph import NemoFlowCallbackHandler


if TYPE_CHECKING:
    from langgraph.graph import CompiledStateGraph

class State(TypedDict):
    value: int

def increment(state: State) -> State:
    return {"value": state["value"] + 1}

async def aincrement(state: State) -> State:
    await asyncio.sleep(0)
    return {"value": state["value"] + 1}

def _build_graph(use_async: bool = False) -> CompiledStateGraph:
    builder = StateGraph(State)
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

@pytest.fixture(name="subscribed_events")
def subscribed_events_fixture() -> list[nemo_flow.Event]:
    events: list[nemo_flow.Event] = []

    def event_recorder(event: nemo_flow.Event) -> None:
        events.append(event)

    subscriber_name = f"langgraph-test-{uuid4()}"
    nemo_flow.subscribers.register(subscriber_name, event_recorder)
    yield events
    nemo_flow.subscribers.deregister(subscriber_name)

def events_to_strings(events: list[nemo_flow.Event]) -> list[str]:
    event_strings: list[str] = []

    for event in events:
        if event.kind == "scope":
            event_strings.append(f"{event.kind}.{event.scope_category}.{event.name}")
        else:
            event_strings.append(f"{event.kind}.{event.name}")

    return event_strings


def test_handler_type():
    handler = NemoFlowCallbackHandler()
    assert isinstance(handler, LangChainCallbackHandler)
    assert isinstance(handler, GraphCallbackHandler)


@pytest.mark.parametrize("use_async", [False, True])
def test_graph_callbacks(use_async: bool,
                         sync_graph: CompiledStateGraph,
                         async_graph: CompiledStateGraph,
                         subscribed_events: list[nemo_flow.Event]):
    graph = async_graph if use_async else sync_graph
    expected_events = [
        "scope.start.request",
        "scope.start.LangGraph",
        "scope.start.increment",
        "scope.end.increment",
        "scope.end.LangGraph",
        "scope.end.request",
    ]

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        if use_async:
            result = asyncio.run(graph.ainvoke({"value": 1}, config={"callbacks": [NemoFlowCallbackHandler()]}))
        else:
            result = graph.invoke({"value": 1}, config={"callbacks": [NemoFlowCallbackHandler()]})

    assert result == {"value": 2}
    assert events_to_strings(subscribed_events) == expected_events


def test_graph_lifecycle_callbacks_emit_marks(subscribed_events: list[nemo_flow.Event]):
    handler = NemoFlowCallbackHandler()
    run_id = uuid4()

    expected_event_strings = [
        'scope.start.request', 'mark.Graph Interrupt', 'mark.Graph Resume', 'scope.end.request',
    ]

    with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
        handler.on_interrupt(
            GraphInterruptEvent(
                run_id=run_id,
                status="interrupt_after",
                checkpoint_id="checkpoint-2",
                checkpoint_ns=("parent",),
                interrupts=(Interrupt("needs approval", id="interrupt-1"),),
            )
        )

        handler.on_resume(
            GraphResumeEvent(
                run_id=run_id,
                status="pending",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent", "child"),
            )
        )


    assert events_to_strings(subscribed_events) == expected_event_strings


    interupt_event = subscribed_events[1]
    assert interupt_event.data["interrupts"] == [{"id": "interrupt-1", "value": "needs approval"}]

    resume_event = subscribed_events[2]
    assert resume_event.data["checkpoint_ns"] == ["parent", "child"]
    assert resume_event.metadata == {"integration": "langgraph"}
