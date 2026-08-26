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


class ToolNodeState(TypedDict):
    messages: Annotated[list[Any], operator.add]
    offset: int


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


@pytest.mark.parametrize("use_async", [False, True])
def test_create_tool_node_routes_standalone_tool_calls_through_relay(
    use_async: bool,
    subscribed_events: list[nemo_relay.Event],
):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import InjectedState

    from nemo_relay.integrations.langgraph import create_tool_node

    def add_offset(value: int, state: Any) -> str:
        """Add the graph state's offset to a model-provided value."""
        return str(value + state["offset"])

    add_offset.__annotations__["state"] = Annotated[dict[str, Any], InjectedState]
    add_offset_tool = tool(add_offset)

    async def rewrite_tool_args(_name: str, args: nemo_relay.Json, next_call: Any) -> Any:
        downstream = await next_call({**cast(dict[str, Any], args), "value": 4})
        return nemo_relay.ToolExecutionInterceptOutcome(
            downstream.result,
            annotation=downstream.annotation,
        )

    node = create_tool_node([add_offset_tool], name="managed-tools")
    builder = StateGraph(ToolNodeState)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()
    nemo_relay.intercepts.register_tool_execution("langgraph-rewrite-tool-args", 1, rewrite_tool_args)
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as request:
            input_state = {
                "offset": 3,
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "add_offset", "args": {"value": 1}, "id": "call-1"},
                            {"name": "add_offset", "args": {"value": 2}, "id": "call-3"},
                        ],
                    )
                ],
            }
            if use_async:
                result = asyncio.run(graph.ainvoke(input_state))
            else:
                result = graph.invoke(input_state)
    finally:
        nemo_relay.intercepts.deregister_tool_execution("langgraph-rewrite-tool-args")

    nemo_relay.subscribers.flush()
    assert [message.content for message in result["messages"][-2:]] == ["7", "7"]
    tool_events = [
        event for event in subscribed_events if isinstance(event, nemo_relay.ScopeEvent) and event.name == "add_offset"
    ]
    assert len(tool_events) == 4
    assert {event.scope_category for event in tool_events} == {"start", "end"}
    assert all(event.parent_uuid == request.uuid for event in tool_events)
    assert {event.category_profile["tool_call_id"] for event in tool_events if event.category_profile} == {
        "call-1",
        "call-3",
    }


def test_exported_tool_node_wrappers_support_direct_tool_node_construction(
    subscribed_events: list[nemo_relay.Event],
):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    from nemo_relay.integrations.langgraph import awrap_tool_call, wrap_tool_call

    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([echo], wrap_tool_call=wrap_tool_call, awrap_tool_call=awrap_tool_call))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        result = graph.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "echo", "args": {"value": "managed"}, "id": "call-2"}],
                    )
                ]
            }
        )

    nemo_relay.subscribers.flush()
    assert result["messages"][-1].content == "managed"
    assert any(
        isinstance(event, nemo_relay.ScopeEvent)
        and event.name == "echo"
        and event.category_profile == {"tool_call_id": "call-2"}
        for event in subscribed_events
    )


def test_create_tool_node_preserves_command_and_error_handling(
    subscribed_events: list[nemo_relay.Event],
):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from nemo_relay.integrations.langgraph import create_tool_node

    @tool
    def update_offset():
        """Update graph state through a LangGraph command."""
        return Command(
            update={
                "offset": 9,
                "messages": [ToolMessage(content="updated", tool_call_id="call-command")],
            }
        )

    @tool
    def fail() -> str:
        """Raise a tool failure handled by ToolNode."""
        raise ValueError("expected failure")

    command_builder = StateGraph(ToolNodeState)
    command_builder.add_node("command", create_tool_node([update_offset]))
    command_builder.add_edge(START, "command")
    command_builder.add_edge("command", END)
    command_graph = command_builder.compile()

    failure_builder = StateGraph(ToolNodeState)
    failure_builder.add_node("failure", create_tool_node([fail], handle_tool_errors=True))
    failure_builder.add_edge(START, "failure")
    failure_builder.add_edge("failure", END)
    failure_graph = failure_builder.compile()

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        command_result = command_graph.invoke(
            {
                "offset": 0,
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "update_offset", "args": {}, "id": "call-command"}],
                    )
                ],
            }
        )
        failure_result = failure_graph.invoke(
            {
                "offset": 0,
                "messages": [AIMessage(content="", tool_calls=[{"name": "fail", "args": {}, "id": "call-failure"}])],
            }
        )

    nemo_relay.subscribers.flush()
    assert command_result["offset"] == 9
    assert command_result["messages"][-1].content == "updated"
    assert failure_result["messages"][-1].status == "error"
    assert {event.name for event in subscribed_events if isinstance(event, nemo_relay.ScopeEvent)} == {
        "request",
        "update_offset",
        "fail",
    }


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("policy", [ValueError, (ValueError,)])
def test_create_tool_node_preserves_selective_error_handling(
    use_async: bool,
    policy: type[ValueError] | tuple[type[ValueError], ...],
):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph

    from nemo_relay.integrations.langgraph import create_tool_node

    @tool
    def value_error() -> str:
        """Raise an error selected by the ToolNode policy."""
        raise ValueError("handled")

    @tool
    def runtime_error() -> str:
        """Raise an error outside the ToolNode policy."""
        raise RuntimeError("unhandled")

    def build_graph(tool: Any):
        builder = StateGraph(ToolNodeState)
        builder.add_node("tools", create_tool_node([tool], handle_tool_errors=policy))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        return builder.compile()

    handled_graph = build_graph(value_error)
    unhandled_graph = build_graph(runtime_error)
    handled_input = {
        "offset": 0,
        "messages": [AIMessage(content="", tool_calls=[{"name": "value_error", "args": {}, "id": "call-value"}])],
    }
    unhandled_input = {
        "offset": 0,
        "messages": [AIMessage(content="", tool_calls=[{"name": "runtime_error", "args": {}, "id": "call-runtime"}])],
    }

    if use_async:

        async def invoke_handled() -> ToolNodeState:
            return await handled_graph.ainvoke(handled_input)

        async def invoke_unhandled() -> ToolNodeState:
            return await unhandled_graph.ainvoke(unhandled_input)

        handled_result = asyncio.run(invoke_handled())
        with pytest.raises(RuntimeError, match="internal error"):
            asyncio.run(invoke_unhandled())
    else:
        handled_result = handled_graph.invoke(handled_input)
        with pytest.raises(RuntimeError, match="internal error"):
            unhandled_graph.invoke(unhandled_input)

    assert handled_result["messages"][-1].status == "error"


@pytest.mark.parametrize("use_async", [False, True])
def test_create_tool_node_preserves_list_command_results(use_async: bool):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from nemo_relay.integrations.langgraph import create_tool_node

    @tool
    def update_offset():
        """Return a list containing a graph state update command."""
        return [
            Command(
                update={
                    "offset": 9,
                    "messages": [ToolMessage(content="updated", tool_call_id="call-list")],
                }
            )
        ]

    builder = StateGraph(ToolNodeState)
    builder.add_node("tools", create_tool_node([update_offset]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()
    input_state = {
        "offset": 0,
        "messages": [AIMessage(content="", tool_calls=[{"name": "update_offset", "args": {}, "id": "call-list"}])],
    }

    if use_async:
        result = asyncio.run(graph.ainvoke(input_state))
    else:
        result = graph.invoke(input_state)

    assert result["offset"] == 9
    assert result["messages"][-1].content == "updated"


@pytest.mark.parametrize("use_async", [False, True])
def test_create_tool_node_propagates_graph_interrupts(use_async: bool):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    from nemo_relay.integrations.langgraph import create_tool_node

    @tool
    def await_approval() -> str:
        """Pause the graph until approval is supplied."""
        interrupt("approval required")
        return "approved"

    builder = StateGraph(ToolNodeState)
    builder.add_node("tools", create_tool_node([await_approval]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}}
    input_state = {
        "offset": 0,
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "await_approval", "args": {}, "id": "call-interrupt"}])
        ],
    }

    if use_async:

        async def collect_updates() -> list[dict[str, Any]]:
            return [update async for update in graph.astream(input_state, config, stream_mode="updates")]

        updates = asyncio.run(collect_updates())
    else:
        updates = list(graph.stream(input_state, config, stream_mode="updates"))

    assert any("__interrupt__" in update for update in updates)


@pytest.mark.parametrize("wrapper_name", ["wrap_tool_call", "awrap_tool_call"])
def test_create_tool_node_rejects_custom_tool_wrappers(wrapper_name: str):
    from nemo_relay.integrations.langgraph import create_tool_node

    with pytest.raises(ValueError, match="construct ToolNode directly"):
        create_tool_node([], **{wrapper_name: lambda *_args: None})


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
