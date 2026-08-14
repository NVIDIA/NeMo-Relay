# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangGraph NeMo Relay callback integration."""

from __future__ import annotations

import asyncio
import operator
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import uuid4

import pytest
from typing_extensions import TypedDict

import nemo_relay

if TYPE_CHECKING:
    from langgraph.graph import CompiledStateGraph

    from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler


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
def callback_handler_fixture() -> NemoRelayCallbackHandler:
    from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler

    return NemoRelayCallbackHandler()


def _events_to_strings(events: list[nemo_relay.Event]) -> list[str]:
    event_strings: list[str] = []

    for event in events:
        if isinstance(event, nemo_relay.ScopeEvent):
            event_strings.append(f"{event.kind}.{event.scope_category}.{event.name}")
        else:
            event_strings.append(f"{event.kind}.{event.name}")

    return event_strings


def test_handler_type(callback_handler: NemoRelayCallbackHandler):
    from langgraph.callbacks import GraphCallbackHandler

    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler as LangChainCallbackHandler

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
        subscribed_events: list[nemo_relay.Event],
        callback_handler: NemoRelayCallbackHandler,
    ):
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            result = sync_graph.invoke({"value": 1}, config={"callbacks": [callback_handler]})

        nemo_relay.subscribers.flush()

        assert result == {"value": 2}
        assert _events_to_strings(subscribed_events) == self._expected_events

    async def test_async(
        self,
        async_graph: CompiledStateGraph,
        subscribed_events: list[nemo_relay.Event],
        callback_handler: NemoRelayCallbackHandler,
    ):
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            result = await async_graph.ainvoke({"value": 1}, config={"callbacks": [callback_handler]})

        await nemo_relay.subscribers.flush_async()

        assert result == {"value": 2}
        assert _events_to_strings(subscribed_events) == self._expected_events


def test_complete_skill_read_inside_langgraph_emits_mark(
    subscribed_events: list[nemo_relay.Event],
    callback_handler: NemoRelayCallbackHandler,
):
    from langgraph.graph import END, START, StateGraph

    def load_skill(state: State) -> State:
        handle = nemo_relay.tools.call("read_file", {"path": "/skills/review/SKILL.md"})
        nemo_relay.tools.call_end(handle, nemo_relay.ToolExecutionResult({"loaded": True}))
        return state

    builder = StateGraph(cast(Any, State))
    builder.add_node("load_skill", load_skill)
    builder.add_edge(START, "load_skill")
    builder.add_edge("load_skill", END)
    graph = builder.compile()

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        result = graph.invoke({"value": 1}, config={"callbacks": [callback_handler]})

    nemo_relay.subscribers.flush()
    assert result == {"value": 1}
    mark = next(
        event for event in subscribed_events if isinstance(event, nemo_relay.MarkEvent) and event.name == "skill.load"
    )
    tool_start = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent) and event.name == "read_file" and event.scope_category == "start"
    )
    assert mark.parent_uuid == tool_start.uuid
    assert mark.data == {"skill_name": "review"}


def test_graph_lifecycle_callbacks_emit_marks(
    subscribed_events: list[nemo_relay.Event],
    callback_handler: NemoRelayCallbackHandler,
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

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
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

    nemo_relay.subscribers.flush()
    assert _events_to_strings(subscribed_events) == expected_event_strings

    interrupt_event = subscribed_events[1]
    assert isinstance(interrupt_event, nemo_relay.MarkEvent)
    interrupt_data = cast(dict[str, Any], interrupt_event.data)
    assert interrupt_data["interrupts"] == [{"id": "interrupt-1", "value": "needs approval"}]

    resume_event = subscribed_events[2]
    assert isinstance(resume_event, nemo_relay.MarkEvent)
    resume_data = cast(dict[str, Any], resume_event.data)
    assert resume_data["checkpoint_ns"] == ["parent", "child"]
    assert resume_event.metadata == {"integration": "langgraph"}


class FanOutState(TypedDict):
    branches: Annotated[list[str], operator.add]


def _build_fan_out_graph() -> CompiledStateGraph:
    """A graph whose two branches finish in a different order than they started.

    LangGraph runs the branches as concurrent tasks sharing one scope stack, so the
    slower branch's scope is still open when the faster one closes. That is the ordering
    Relay's stack rejects, and reproducing it here needs no stubbing at all.
    """
    from langgraph.graph import END, START, StateGraph

    async def slow(state: FanOutState) -> FanOutState:
        await asyncio.sleep(0.05)
        return {"branches": ["slow"]}

    async def fast(state: FanOutState) -> FanOutState:
        await asyncio.sleep(0)
        return {"branches": ["fast"]}

    builder = StateGraph(cast(Any, FanOutState))
    builder.add_node("slow", slow)
    builder.add_node("fast", fast)
    builder.add_edge(START, "slow")
    builder.add_edge(START, "fast")
    builder.add_edge("slow", END)
    builder.add_edge("fast", END)
    return builder.compile()


async def test_parallel_fan_out_leaves_the_enclosing_scope_closable(
    callback_handler: NemoRelayCallbackHandler,
    subscribed_events: list[nemo_relay.Event],
):
    """The reported failure, driven by a real graph rather than synthesized callbacks.

    Before the ordering fix, the branch that finished first was abandoned on the stack
    and the enclosing ``request`` scope raised on exit, turning a graph that ran to
    completion into a reported failure.
    """

    graph = _build_fan_out_graph()

    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        baseline = nemo_relay.scope.get_handle()
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            result = await graph.ainvoke({"branches": []}, config={"callbacks": [callback_handler]})

        # The graph really did fan out, and the stack is back where it started.
        assert sorted(result["branches"]) == ["fast", "slow"]
        assert nemo_relay.scope.get_handle().uuid == baseline.uuid

    # ``on_chain_start`` swallows its exceptions, so a handler that pushed nothing would
    # satisfy everything above. Pin that both branches actually opened and closed scopes.
    await nemo_relay.subscribers.flush_async()
    emitted = _events_to_strings(subscribed_events)
    for branch in ("slow", "fast"):
        assert f"scope.start.{branch}" in emitted
        assert f"scope.end.{branch}" in emitted


def _build_nested_fan_out_graph() -> CompiledStateGraph:
    """Three branches of differing durations, one of which fans out again.

    A two-branch graph only ever has one completion waiting. This leaves several
    waiting at once, at more than one depth, so a drain has to close a run of them in
    the right order rather than one scope at a time.
    """
    from langgraph.graph import END, START, StateGraph

    async def slowest(state: FanOutState) -> FanOutState:
        await asyncio.sleep(0.06)
        return {"branches": ["slowest"]}

    async def middle(state: FanOutState) -> FanOutState:
        await asyncio.sleep(0.03)
        return {"branches": ["middle"]}

    async def quickest(state: FanOutState) -> FanOutState:
        await asyncio.sleep(0)
        return {"branches": ["quickest"]}

    inner = StateGraph(cast(Any, FanOutState))
    inner.add_node("middle", middle)
    inner.add_node("quickest", quickest)
    inner.add_edge(START, "middle")
    inner.add_edge(START, "quickest")
    inner.add_edge("middle", END)
    inner.add_edge("quickest", END)

    builder = StateGraph(cast(Any, FanOutState))
    builder.add_node("slowest", slowest)
    builder.add_node("inner", inner.compile())
    builder.add_edge(START, "slowest")
    builder.add_edge(START, "inner")
    builder.add_edge("slowest", END)
    builder.add_edge("inner", END)
    return builder.compile()


async def test_nested_fan_out_closes_every_scope(
    callback_handler: NemoRelayCallbackHandler,
    subscribed_events: list[nemo_relay.Event],
):
    """Several completions waiting at once, at two depths, must all close in order."""

    graph = _build_nested_fan_out_graph()

    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        baseline = nemo_relay.scope.get_handle()
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            result = await graph.ainvoke({"branches": []}, config={"callbacks": [callback_handler]})

        assert sorted(result["branches"]) == ["middle", "quickest", "slowest"]
        assert nemo_relay.scope.get_handle().uuid == baseline.uuid
        assert callback_handler._completed == {}

    await nemo_relay.subscribers.flush_async()
    emitted = _events_to_strings(subscribed_events)
    for node in ("slowest", "middle", "quickest"):
        assert f"scope.start.{node}" in emitted
        assert f"scope.end.{node}" in emitted


async def test_a_failing_node_still_closes_its_siblings(
    callback_handler: NemoRelayCallbackHandler,
    subscribed_events: list[nemo_relay.Event],
):
    """A node raising mid-fan-out completes through ``on_chain_error``.

    The failed run's scope has to be closed like any other, or it strands its siblings
    and the enclosing scope underneath it.
    """
    from langgraph.graph import END, START, StateGraph

    async def boom(state: FanOutState) -> FanOutState:
        await asyncio.sleep(0)
        raise ValueError("node failed")

    async def survivor(state: FanOutState) -> FanOutState:
        await asyncio.sleep(0.05)
        return {"branches": ["survivor"]}

    builder = StateGraph(cast(Any, FanOutState))
    builder.add_node("boom", boom)
    builder.add_node("survivor", survivor)
    builder.add_edge(START, "boom")
    builder.add_edge(START, "survivor")
    builder.add_edge("boom", END)
    builder.add_edge("survivor", END)
    graph = builder.compile()

    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        baseline = nemo_relay.scope.get_handle()
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            with pytest.raises(ValueError, match="node failed"):
                await graph.ainvoke({"branches": []}, config={"callbacks": [callback_handler]})

        # The graph failed, but the telemetry did not leave the stack dirty.
        assert nemo_relay.scope.get_handle().uuid == baseline.uuid
        assert callback_handler._completed == {}

    # ``on_chain_start`` swallows its exceptions, so a handler that opened no branch
    # scopes would satisfy everything above. Pin that both branches ran and closed.
    await nemo_relay.subscribers.flush_async()
    emitted = _events_to_strings(subscribed_events)
    for node in ("boom", "survivor"):
        assert f"scope.start.{node}" in emitted
        assert f"scope.end.{node}" in emitted
