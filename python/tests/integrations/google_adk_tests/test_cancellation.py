# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test how the Google ADK NeMo Relay plugin closes scopes when a run is cancelled, times out, or closes early."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Literal, TypeAlias

import pytest
from _helpers import (
    HANG_TIMEOUT,
    ScriptedLlmFactory,
    as_object,
    assert_balanced,
    hang,
    invocation_start,
    invocation_starts,
    make_agent,
    make_runner,
    metadata_of,
    named,
    named_end,
    new_session,
    run_turn,
    status_of,
    tool_call,
    weather_call,
)
from typing_extensions import override

import nemo_relay

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine

    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    from google.adk.runners import Runner
    from google.genai import types

_Interruption: TypeAlias = Literal["task cancel", "timeout"]
_Root: TypeAlias = Literal["LlmAgent", "custom BaseAgent"]


def _hanging_weather(reached: asyncio.Event) -> Callable[[str], Awaitable[dict[str, str]]]:
    async def get_weather(location: str) -> dict[str, str]:
        """Get the weather, and block each call until the run is cancelled."""
        reached.set()
        await hang()
        return {"location": location, "forecast": "sunny"}

    return get_weather


def _root(kind: _Root, agent: LlmAgent) -> BaseAgent:
    """Return the agent, or a custom `BaseAgent` that runs it, which ADK drives outside its node runtime."""
    from google.adk.agents import BaseAgent

    class Delegate(BaseAgent):
        @override
        async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
            async with contextlib.aclosing(agent.run_async(ctx)) as events:
                async for event in events:
                    yield event

    return agent if kind == "LlmAgent" else Delegate(name="delegate", sub_agents=[agent])


async def _drained(events: AsyncGenerator[Event, None]) -> None:
    async with contextlib.aclosing(events) as stream:
        async for _ in stream:
            pass


async def _cancelled(run: Coroutine[object, object, object], reached: asyncio.Event) -> None:
    """Run the coroutine in its own task, and cancel the task once `reached` is set."""
    task = asyncio.ensure_future(run)
    async with asyncio.timeout(HANG_TIMEOUT):
        await reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _timed_out(events: AsyncGenerator[Event, None], reached: asyncio.Event) -> None:
    """Drain the events, and let a timeout expire once `reached` is set."""
    async with asyncio.timeout(HANG_TIMEOUT) as deadline:
        waiter = asyncio.ensure_future(reached.wait())
        waiter.add_done_callback(lambda _: deadline.reschedule(asyncio.get_running_loop().time()))
        try:
            await _drained(events)
        finally:
            waiter.cancel()


async def _interrupted(runner: Runner, reached: asyncio.Event, how: _Interruption) -> None:
    """Run a turn, and cancel its task or let its timeout expire once `reached` is set."""
    from google.genai import types

    events = runner.run_async(user_id="user", session_id=await new_session(runner), new_message=types.UserContent("hi"))
    if how == "task cancel":
        await _cancelled(_drained(events), reached)
        return
    with pytest.raises(TimeoutError):
        await _timed_out(events, reached)


def _assert_cancelled(event: nemo_relay.ScopeEvent) -> None:
    assert status_of(event) == "ERROR"
    assert event.data == {"status": "cancelled"}


@pytest.mark.parametrize("how", ["task cancel", "timeout"])
@pytest.mark.parametrize("kind", ["LlmAgent", "custom BaseAgent"])
async def test_an_interrupted_run_closes_its_scopes_as_cancelled_and_leaves_the_caller_stack_clean(
    kind: _Root, how: _Interruption, scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from nemo_relay.integrations.google_adk._context import parent_handle

    reached = asyncio.Event()
    agent = make_agent(scripted_llm, "weather_agent", tools=[_hanging_weather(reached)], call=weather_call())
    runner = make_runner(_root(kind, agent))
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as request:
            await _interrupted(runner, reached, how)
            assert parent_handle().uuid == request.uuid
        await nemo_relay.subscribers.flush_async()

        assert_balanced(subscribed_events)
        invocation = invocation_start(subscribed_events)
        assert invocation.parent_uuid == request.uuid
        invocation_end = named_end(subscribed_events, invocation.name, "agent")
        _assert_cancelled(invocation_end)
        assert metadata_of(invocation_end)["otel.status_description"] == "invocation cancelled"
        _assert_cancelled(named_end(subscribed_events, "weather_agent", "agent"))
        tool_end = named_end(subscribed_events, "get_weather", "tool")
        assert metadata_of(tool_end)["otel.status_description"] == "tool execution cancelled"
    finally:
        await runner.close()


@pytest.mark.parametrize("how", ["task cancel", "timeout"])
@pytest.mark.parametrize("kind", ["LlmAgent", "custom BaseAgent"])
async def test_a_run_interrupted_while_another_plugin_handles_its_event_closes_as_cancelled(
    kind: _Root, how: _Interruption, scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    """Verify the cancellation while ADK's root node waits for the runner to finish handling its event."""
    from google.adk.plugins import BasePlugin

    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    reached = asyncio.Event()

    class Slow(BasePlugin):
        @override
        async def on_event_callback(self, *, invocation_context: InvocationContext, event: Event) -> Event | None:
            if event.get_function_calls():
                reached.set()
                await hang()
            return None

    agent = make_agent(scripted_llm, "weather_agent", tools=[_hanging_weather(asyncio.Event())], call=weather_call())
    runner = make_runner(_root(kind, agent), plugins=[NemoRelayPlugin(), Slow("slow")])
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            await _interrupted(runner, reached, how)
        await nemo_relay.subscribers.flush_async()

        assert_balanced(subscribed_events)
        _assert_cancelled(named_end(subscribed_events, invocation_start(subscribed_events).name, "agent"))
    finally:
        await runner.close()


async def test_an_aborted_run_closes_its_invocation_as_cancelled(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.genai import types

    reached, abort = asyncio.Event(), asyncio.Event()
    agent = make_agent(scripted_llm, "weather_agent", tools=[_hanging_weather(reached)], call=weather_call())
    runner = make_runner(agent)
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            events = runner.run_async(
                user_id="user",
                session_id=await new_session(runner),
                new_message=types.UserContent("hi"),
                abort_signal=abort,
            )
            run = asyncio.ensure_future(_drained(events))
            async with asyncio.timeout(HANG_TIMEOUT):
                await reached.wait()
                abort.set()
                await run
        await nemo_relay.subscribers.flush_async()

        assert_balanced(subscribed_events)
        invocation_end = named_end(subscribed_events, invocation_start(subscribed_events).name, "agent")
        assert as_object(invocation_end.data)["status"] == "cancelled"
        assert metadata_of(invocation_end)["otel.status_description"] == "invocation cancelled"
    finally:
        await runner.close()


@pytest.mark.parametrize("kind", ["LlmAgent", "custom BaseAgent"])
async def test_a_run_its_caller_stops_consuming_closes_as_adk_reports_it(
    kind: _Root, scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    """Verify that a run closed early by its caller ends through `after_run_callback`, which ADK calls a completion."""
    from google.genai import types

    agent = make_agent(scripted_llm, "weather_agent", tools=[_hanging_weather(asyncio.Event())], call=weather_call())
    runner = make_runner(_root(kind, agent))
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            events = runner.run_async(
                user_id="user", session_id=await new_session(runner), new_message=types.UserContent("hi")
            )
            async with contextlib.aclosing(events) as stream:
                async for event in stream:
                    if event.get_function_calls():
                        break
        await nemo_relay.subscribers.flush_async()

        assert_balanced(subscribed_events)
        invocation_end = named_end(subscribed_events, invocation_start(subscribed_events).name, "agent")
        assert invocation_end.data == {"status": "ok"}
        assert status_of(invocation_end) == "OK"
    finally:
        await runner.close()


async def test_a_run_cancelled_before_its_root_starts_records_no_invocation_scope(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.plugins import BasePlugin

    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    reached = asyncio.Event()

    class Gate(BasePlugin):
        @override
        async def before_run_callback(self, *, invocation_context: InvocationContext) -> types.Content | None:
            reached.set()
            await hang()
            return None

    runner = make_runner(make_agent(scripted_llm, "weather_agent"), plugins=[NemoRelayPlugin(), Gate("gate")])
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            await _cancelled(run_turn(runner), reached)
    finally:
        await runner.close()
    await nemo_relay.subscribers.flush_async()

    assert invocation_starts(subscribed_events) == []
    assert_balanced(subscribed_events)


async def test_a_run_whose_task_ends_without_adk_run_hooks_closes_its_scope_as_cancelled(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    """Verify the close when the task ends while a plugin ahead of NeMo Relay holds `after_run_callback`."""
    from google.adk.plugins import BasePlugin

    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    reached = asyncio.Event()

    class Gate(BasePlugin):
        @override
        async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
            reached.set()
            await hang()

    runner = make_runner(make_agent(scripted_llm, "weather_agent"), plugins=[Gate("gate"), NemoRelayPlugin()])
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            await _cancelled(run_turn(runner), reached)
        await nemo_relay.subscribers.flush_async()

        assert_balanced(subscribed_events)
        invocation_end = named_end(subscribed_events, invocation_start(subscribed_events).name, "agent")
        assert status_of(invocation_end) == "ERROR"
        assert invocation_end.data == {
            "status": "cancelled",
            "response": {"parts": [{"text": "Answer from weather_agent to 'hi'."}], "role": "model"},
        }
    finally:
        await runner.close()


async def test_a_cancelled_wrapped_workflow_run_closes_its_scopes_as_cancelled(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.workflow import Workflow

    from nemo_relay.integrations.google_adk import wrap_node

    reached = asyncio.Event()

    async def slow(node_input: str) -> str:
        reached.set()
        await hang()
        return node_input

    runner = make_runner(wrap_node(Workflow(name="pipeline", edges=[("START", slow)])))
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            await _cancelled(run_turn(runner), reached)
        await nemo_relay.subscribers.flush_async()

        assert_balanced(subscribed_events)
        _assert_cancelled(named_end(subscribed_events, invocation_start(subscribed_events).name, "agent"))
        _assert_cancelled(named_end(subscribed_events, "pipeline", "agent"))
        _assert_cancelled(named_end(subscribed_events, "slow", "function"))
    finally:
        await runner.close()


async def test_a_wrapped_workflow_run_its_caller_stops_consuming_closes_its_running_node_as_cancelled(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    """Verify that ADK's early close cancels the running node, while the invocation ends as ADK reports it."""
    from google.adk.workflow import Workflow, node
    from google.genai import types

    from nemo_relay.integrations.google_adk import wrap_node

    def first(node_input: str) -> str:
        return node_input

    @node
    async def second(node_input: str) -> AsyncGenerator[str, None]:
        yield "working"
        await hang()

    runner = make_runner(wrap_node(Workflow(name="pipeline", edges=[("START", first, second)])))
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            events = runner.run_async(
                user_id="user", session_id=await new_session(runner), new_message=types.UserContent("hi")
            )
            async with contextlib.aclosing(events) as stream:
                async for event in stream:
                    if event.output == "working":
                        break
        await nemo_relay.subscribers.flush_async()

        assert_balanced(subscribed_events)
        assert metadata_of(named_end(subscribed_events, "second", "function"))["otel.status_description"] == (
            "node cancelled"
        )
        assert named_end(subscribed_events, invocation_start(subscribed_events).name, "agent").data == {"status": "ok"}
    finally:
        await runner.close()


async def test_an_unwrapped_workflow_run_cancelled_before_it_needs_a_scope_records_none(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.workflow import Workflow

    reached = asyncio.Event()

    async def slow(node_input: str) -> str:
        reached.set()
        await hang()
        return node_input

    runner = make_runner(Workflow(name="pipeline", edges=[("START", slow)]))
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            await _cancelled(run_turn(runner), reached)
        await nemo_relay.subscribers.flush_async()

        assert invocation_starts(subscribed_events) == []
        assert_balanced(subscribed_events)
    finally:
        await runner.close()


async def test_a_run_cancelled_during_a_transfer_hop_closes_its_invocation_as_cancelled(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    reached = asyncio.Event()
    specialist = make_agent(scripted_llm, "specialist", tools=[_hanging_weather(reached)], call=weather_call())
    router = make_agent(
        scripted_llm, "router", sub_agents=[specialist], call=tool_call("transfer_to_agent", agent_name="specialist")
    )
    runner = make_runner(router)
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            await _cancelled(run_turn(runner), reached)
        await nemo_relay.subscribers.flush_async()

        assert_balanced(subscribed_events)
        invocation = invocation_start(subscribed_events)
        assert named(subscribed_events, "specialist", "agent").parent_uuid == invocation.uuid
        assert named_end(subscribed_events, "router", "agent").data == {"transferred_to": "specialist"}
        _assert_cancelled(named_end(subscribed_events, "specialist", "agent"))
        _assert_cancelled(named_end(subscribed_events, invocation.name, "agent"))
    finally:
        await runner.close()
