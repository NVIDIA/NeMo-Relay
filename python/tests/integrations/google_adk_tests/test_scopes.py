# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the scopes, marks, and lifecycle of the Google ADK NeMo Relay plugin."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NoReturn, get_args

import pytest
from _helpers import (
    HANG_TIMEOUT,
    ScriptedLlmFactory,
    assert_balanced,
    data_of,
    ends,
    event_signal,
    final_text,
    get_weather,
    hang,
    invocation_context,
    invocation_start,
    invocation_starts,
    make_agent,
    make_runner,
    marks,
    metadata_of,
    model_answer,
    named,
    named_end,
    new_session,
    parent_of,
    present,
    request_of,
    response_of,
    run_app,
    run_recorded,
    run_turn,
    run_until_interrupted,
    scope_events,
    starts,
    status_of,
    tool_call,
    toolset_of,
    weather_call,
    weather_result,
)
from typing_extensions import override

import nemo_relay

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator, Mapping

    from google.adk.agents.callback_context import CallbackContext
    from google.adk.events.event import Event
    from google.adk.models import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.tools.tool_context import ToolContext

_PLUGIN_LOGGER = "nemo_relay.integrations.google_adk.plugin"


async def _weather_run(scripted_llm: ScriptedLlmFactory) -> list[Event]:
    weather_agent = make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
    weather_agent.description = "Answers weather questions."
    return await run_recorded(weather_agent)


async def test_single_agent_records_invocation_and_agent_scopes_closing_with_the_final_response(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    adk_events = await _weather_run(scripted_llm)

    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    assert_balanced(subscribed_events)
    invocation = invocation_start(subscribed_events)
    assert parent_of(subscribed_events, invocation).name == "request"
    assert data_of(invocation)["root_agent"] == "weather_agent"
    agent = named(subscribed_events, "weather_agent", "agent")
    assert agent.parent_uuid == invocation.uuid
    assert data_of(agent)["agent_type"] == "LlmAgent"
    assert data_of(agent)["description"] == "Answers weather questions."
    response = {"parts": [{"text": "Answer from weather_agent to 'hi'."}], "role": "model"}
    assert named_end(subscribed_events, "weather_agent", "agent").data == {"status": "ok", "response": response}
    assert named_end(subscribed_events, invocation.name, "agent").data == {"status": "ok", "response": response}


async def test_single_agent_records_an_llm_scope_per_model_call_under_the_agent(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    await _weather_run(scripted_llm)

    agent = named(subscribed_events, "weather_agent", "agent")
    llm_calls = starts(subscribed_events, "llm")
    assert [event.name for event in llm_calls] == ["scripted-weather_agent"] * 2
    assert all(event.parent_uuid == agent.uuid for event in llm_calls)
    assert present(llm_calls[0].category_profile)["model_name"] == "scripted-weather_agent"
    assert metadata_of(llm_calls[0])["adk_model_type"] == "ScriptedLlm"
    assert request_of(llm_calls[0]).model == "scripted-weather_agent"
    assert request_of(llm_calls[0]).messages[-1]["content"] == "hi"
    llm_ends = ends(subscribed_events, "llm")
    assert all(status_of(event) == "OK" for event in llm_ends)
    assert response_of(llm_ends[0]).finish_reason == "tool_use"
    assert response_of(llm_ends[-1]).message == "Answer from weather_agent to 'hi'."
    assert response_of(llm_ends[-1]).usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


async def test_single_agent_records_the_tool_call_under_the_agent(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    await _weather_run(scripted_llm)

    agent = named(subscribed_events, "weather_agent", "agent")
    (tool,) = starts(subscribed_events, "tool")
    assert tool.name == "get_weather"
    assert tool.parent_uuid == agent.uuid
    assert tool.data == {"location": "San Francisco"}
    assert metadata_of(tool)["adk_tool_type"] == "FunctionTool"
    assert present(tool.category_profile)["tool_call_id"]
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == weather_result()


async def test_streaming_records_one_llm_scope_per_model_call_with_chunk_marks(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents.run_config import RunConfig, StreamingMode

    adk_events = await run_recorded(
        make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call()),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    assert [event.partial for event in adk_events if event.partial] == [True, True]
    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    llm_starts = starts(subscribed_events, "llm")
    assert len(llm_starts) == 2
    assert all(start.attributes == ["streaming"] for start in llm_starts)
    llm_ends = ends(subscribed_events, "llm")
    assert len(llm_ends) == 2
    assert response_of(llm_ends[-1]).message == "Answer from weather_agent to 'hi'."
    chunks = marks(subscribed_events, "llm.chunk")
    assert [chunk.parent_uuid for chunk in chunks] == [llm_starts[0].uuid] + [llm_starts[1].uuid] * 3
    assert [data_of(chunk)["chunk_index"] for chunk in chunks if isinstance(chunk.data, dict)] == [0, 0, 1, 2]
    assert_balanced(subscribed_events)


@pytest.mark.filterwarnings("ignore:ParallelAgent is deprecated:DeprecationWarning")
async def test_parallel_agents_keep_branch_parentage(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import ParallelAgent  # ty: ignore[deprecated]

    root = ParallelAgent(  # ty: ignore[deprecated]
        name="par", sub_agents=[make_agent(scripted_llm, "branch_a"), make_agent(scripted_llm, "branch_b")]
    )
    await run_recorded(root)

    assert_balanced(subscribed_events)
    agents = {event.name: event for event in starts(subscribed_events, "agent")}
    assert agents["branch_a"].parent_uuid == agents["par"].uuid
    assert agents["branch_b"].parent_uuid == agents["par"].uuid
    llm_parents = {parent_of(subscribed_events, event).name for event in starts(subscribed_events, "llm")}
    assert llm_parents == {"branch_a", "branch_b"}


@pytest.mark.filterwarnings("ignore:The `plugins` argument is deprecated:DeprecationWarning")
async def test_agent_tool_runs_the_nested_invocation_under_the_calling_agent(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.tools import AgentTool

    inner = make_agent(scripted_llm, "inner_agent")
    outer = make_agent(
        scripted_llm, "outer", tools=[AgentTool(agent=inner)], call=tool_call("inner_agent", request="hello inner")
    )
    adk_events = await run_recorded(outer)

    assert final_text(adk_events) == "Answer from outer to 'hi'."
    assert_balanced(subscribed_events)
    outer_scope = named(subscribed_events, "outer", "agent")
    (tool,) = starts(subscribed_events, "tool")
    assert tool.name == "inner_agent"
    assert metadata_of(tool)["adk_tool_type"] == "AgentTool"
    assert tool.parent_uuid == outer_scope.uuid
    invocations = invocation_starts(subscribed_events)
    assert len(invocations) == 2
    (nested,) = [event for event in invocations if event.parent_uuid == outer_scope.uuid]
    inner_agent = named(subscribed_events, "inner_agent", "agent")
    assert inner_agent.parent_uuid == nested.uuid
    inner_call = named(subscribed_events, "scripted-inner_agent", "llm")
    assert parent_of(subscribed_events, inner_call).uuid == inner_agent.uuid
    ordered = [(event.scope_category, event.name) for event in scope_events(subscribed_events)]
    assert ordered.index(("end", "inner_agent")) < ordered.index(("end", "outer"))


async def test_transfer_closes_transferring_agent_and_records_marks(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    router = make_agent(
        scripted_llm,
        "router",
        sub_agents=[make_agent(scripted_llm, "specialist")],
        call=tool_call("transfer_to_agent", agent_name="specialist"),
    )
    adk_events = await run_recorded(router)

    assert [event.author for event in adk_events] == ["router", "router", "specialist"]
    assert_balanced(subscribed_events)
    ordered = [
        (event.scope_category, event.name) for event in scope_events(subscribed_events) if event.category == "agent"
    ]
    assert ordered.index(("end", "router")) < ordered.index(("start", "specialist"))
    transfer_tool = named(subscribed_events, "transfer_to_agent", "tool")
    assert metadata_of(transfer_tool)["adk_tool_type"] == "TransferToAgentTool"
    assert transfer_tool.data == {"agent_name": "specialist"}
    (mark,) = marks(subscribed_events, "adk.agent_transfer")
    assert mark.data == {"from": "router", "to": "specialist", "reason": None}


async def test_escalation_is_recorded_as_a_mark(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    def escalate(tool_context: ToolContext) -> dict[str, str]:
        tool_context.actions.escalate = True
        return {"status": "escalated"}

    await run_recorded(make_agent(scripted_llm, "worker", tools=[escalate], call=tool_call("escalate")))

    (mark,) = marks(subscribed_events, "adk.escalate")
    assert mark.data == {"author": "worker"}
    assert mark.parent_uuid == invocation_start(subscribed_events).uuid
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(("prefix", "expected_name"), [(None, "get_weather"), ("geo", "geo_get_weather")])
async def test_toolset_tools_run_through_relay(
    prefix: str | None, expected_name: str, scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.tools import FunctionTool

    toolset = toolset_of([FunctionTool(get_weather)], prefix=prefix)
    call = tool_call(expected_name, location="San Francisco")
    adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent", tools=[toolset], call=call))

    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    (tool,) = starts(subscribed_events, "tool")
    assert tool.name == expected_name
    assert metadata_of(tool)["adk_tool_type"] == "FunctionTool"
    assert_balanced(subscribed_events)


async def test_tools_keep_relay_parentage_under_adk_tool_thread_pool(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig

    agent = make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
    adk_events = await run_recorded(agent, run_config=RunConfig(tool_thread_pool_config=ToolThreadPoolConfig()))

    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    assert_balanced(subscribed_events)
    agent_scope = named(subscribed_events, "weather_agent", "agent")
    (tool,) = starts(subscribed_events, "tool")
    assert tool.parent_uuid == agent_scope.uuid
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == weather_result()


async def test_agent_short_circuited_by_before_agent_callback_still_closes_its_scope(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.genai import types

    def answer_directly(callback_context: CallbackContext) -> types.Content:
        return types.ModelContent("Handled without running the agent.")

    agent = make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
    agent.before_agent_callback = answer_directly
    adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Handled without running the agent."
    assert starts(subscribed_events, "llm") == []
    assert_balanced(subscribed_events)
    agent_end = named_end(subscribed_events, "weather_agent", "agent")
    assert agent_end.data == {
        "status": "ok",
        "response": {"parts": [{"text": "Handled without running the agent."}], "role": "model"},
    }
    assert status_of(agent_end) == "OK"


def test_sync_runner_keeps_the_caller_scope_as_parent(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.genai import types

    runner = make_runner(make_agent(scripted_llm, "weather_agent"))
    session_id = asyncio.run(new_session(runner))
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as request:
            adk_events = list(runner.run(user_id="user", session_id=session_id, new_message=types.UserContent("hi")))
    finally:
        asyncio.run(runner.close())
    nemo_relay.subscribers.flush()

    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    assert_balanced(subscribed_events)
    assert invocation_start(subscribed_events).parent_uuid == request.uuid
    assert len(starts(subscribed_events, "llm")) == 1


async def test_run_config_custom_metadata_recorded_on_the_invocation_scope(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents.run_config import RunConfig

    run_config = RunConfig(custom_metadata={"tenant": "t1"})
    await run_recorded(make_agent(scripted_llm, "weather_agent"), run_config=run_config)

    invocation = invocation_start(subscribed_events)
    assert metadata_of(invocation)["adk_custom_metadata"] == {"tenant": "t1"}
    assert metadata_of(invocation)["integration"] == "google-adk"
    assert_balanced(subscribed_events)


async def test_model_call_answered_by_a_later_plugin_records_no_llm_scope(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins import BasePlugin
    from google.genai import types

    class CannedAnswer(BasePlugin):
        @override
        async def before_model_callback(
            self, *, callback_context: CallbackContext, llm_request: LlmRequest
        ) -> LlmResponse | None:
            return LlmResponse(content=types.ModelContent("Canned."))

    adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"), extra_plugins=[CannedAnswer("canned")])

    assert final_text(adk_events) == "Canned."
    assert scripted_llm.seen_requests == []
    assert starts(subscribed_events, "llm") == []
    assert_balanced(subscribed_events)


async def test_long_running_tool_records_its_scope_and_mark(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.tools import LongRunningFunctionTool

    def slow_tool() -> dict[str, str]:
        """Start a slow job and report it as pending."""
        return {"status": "pending"}

    agent = make_agent(
        scripted_llm, "weather_agent", tools=[LongRunningFunctionTool(slow_tool)], call=tool_call("slow_tool")
    )
    adk_events = await run_recorded(agent)

    (tool,) = starts(subscribed_events, "tool")
    assert tool.name == "slow_tool"
    assert metadata_of(tool)["adk_long_running"] is True
    (mark,) = marks(subscribed_events, "adk.long_running_tools")
    assert data_of(mark)["function_call_ids"]
    assert any(event.long_running_tool_ids for event in adk_events)
    assert_balanced(subscribed_events)


async def test_close_pops_invocation_scope_left_open_by_cancellation(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    plugin = NemoRelayPlugin()
    context = invocation_context(make_agent(scripted_llm, "cancelled_agent"), "e-cancelled")
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        await plugin.before_run_callback(invocation_context=context)
        await plugin.close()
    await nemo_relay.subscribers.flush_async()

    (invocation_end,) = [
        event for event in ends(subscribed_events, "agent") if event.name.startswith("adk-invocation:")
    ]
    assert metadata_of(invocation_end)["otel.status_description"] == "invocation cancelled"
    assert_balanced(subscribed_events)


async def test_close_from_another_context_closes_the_invocation_on_its_own_stack(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    plugin = NemoRelayPlugin()
    context = invocation_context(make_agent(scripted_llm, "cancelled_agent"), "e-elsewhere")
    with (
        nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()),
        nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent),
    ):
        await plugin.before_run_callback(invocation_context=context)
        await asyncio.create_task(plugin.close(), context=contextvars.Context())
    await nemo_relay.subscribers.flush_async()

    assert_balanced(subscribed_events)
    (invocation_end,) = [
        event for event in ends(subscribed_events, "agent") if event.name.startswith("adk-invocation:")
    ]
    assert metadata_of(invocation_end)["otel.status_description"] == "invocation cancelled"


async def test_concurrent_invocations_on_one_runner_keep_their_own_scopes(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    first_tool_running, second_finished = asyncio.Event(), asyncio.Event()

    async def get_weather(location: str) -> dict[str, str]:
        """Get the weather, and hold the first call until the second invocation finishes."""
        if not first_tool_running.is_set():
            first_tool_running.set()
            await second_finished.wait()
        return {"location": location, "forecast": "sunny"}

    runner = make_runner(make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call()))

    async def invoke(request_name: str) -> str:
        with (
            nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()),
            nemo_relay.scope.scope(request_name, nemo_relay.ScopeType.Agent) as request,
        ):
            await run_turn(runner)
        return request.uuid

    async def second() -> str:
        await first_tool_running.wait()
        try:
            return await invoke("request-2")
        finally:
            second_finished.set()

    try:
        async with asyncio.timeout(HANG_TIMEOUT), asyncio.TaskGroup() as group:
            first_task = group.create_task(invoke("request-1"))
            second_task = group.create_task(second())
    finally:
        await runner.close()
    await nemo_relay.subscribers.flush_async()
    first_uuid, second_uuid = first_task.result(), second_task.result()

    assert_balanced(subscribed_events)

    def request_of(event: nemo_relay.ScopeEvent) -> str:
        while not event.name.startswith("request-"):
            event = parent_of(subscribed_events, event)
        return event.uuid

    invocations = invocation_starts(subscribed_events)
    assert Counter(request_of(event) for event in invocations) == {first_uuid: 1, second_uuid: 1}
    assert Counter(request_of(event) for event in starts(subscribed_events, "tool")) == {first_uuid: 1, second_uuid: 1}
    assert Counter(request_of(event) for event in starts(subscribed_events, "llm")) == {first_uuid: 2, second_uuid: 2}


async def test_one_plugin_serving_two_runners_warns_once(
    scripted_llm: ScriptedLlmFactory, caplog: pytest.LogCaptureFixture
) -> None:
    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    held_running, release = asyncio.Event(), asyncio.Event()

    async def get_weather(location: str) -> dict[str, str]:
        held_running.set()
        await release.wait()
        return {"location": location, "forecast": "sunny"}

    plugin = NemoRelayPlugin()
    held = make_runner(
        make_agent(scripted_llm, "held_agent", tools=[get_weather], call=weather_call()), plugins=[plugin]
    )
    other = make_runner(make_agent(scripted_llm, "other_agent"), plugins=[plugin])
    caplog.set_level(logging.WARNING, logger=_PLUGIN_LOGGER)
    try:
        async with asyncio.timeout(HANG_TIMEOUT), asyncio.TaskGroup() as group:
            held_task = group.create_task(_run_on_own_stack(held))
            await held_running.wait()
            await _run_on_own_stack(other)
            await _run_on_own_stack(other)
            release.set()
    finally:
        await held.close()
        await other.close()

    assert final_text(held_task.result()) == "Answer from held_agent to 'hi'."
    warnings = [
        record
        for record in caplog.records
        if record.name == _PLUGIN_LOGGER and "serves more than one Runner" in record.getMessage()
    ]
    assert len(warnings) == 1


async def _run_on_own_stack(runner: Runner) -> list[Event]:
    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        return await run_turn(runner)


async def test_a_failing_plugin_hook_is_logged_and_the_adk_run_continues(
    scripted_llm: ScriptedLlmFactory, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unavailable(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("scope stack unavailable")

    monkeypatch.setattr(nemo_relay.scope, "push", unavailable)
    with caplog.at_level(logging.ERROR, logger=_PLUGIN_LOGGER):
        adk_events = await run_app(make_agent(scripted_llm, "weather_agent"))

    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    failures = [record.getMessage() for record in caplog.records if record.name == _PLUGIN_LOGGER and record.exc_info]
    assert failures == ["NeMo Relay: before_run_callback failed"]


def test_plugin_implements_every_adk_plugin_hook_it_needs() -> None:
    """Fail when the hooks that `NemoRelayPlugin` leaves unimplemented differ from this test's list."""
    from google.adk.plugins.plugin_manager import PluginCallbackName

    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    adk_hooks = set(get_args(PluginCallbackName))
    implemented = {name for name in adk_hooks if name in vars(NemoRelayPlugin)}
    assert adk_hooks - implemented == {
        "on_user_message_callback",
        "after_model_callback",
        "on_model_error_callback",
        "after_tool_callback",
        "on_tool_error_callback",
    }


def test_wrapping_keeps_the_identity_and_type_of_a_model(scripted_llm: ScriptedLlmFactory) -> None:
    from nemo_relay.integrations.google_adk import _models
    from nemo_relay.integrations.google_adk._swap import is_swapped

    model = scripted_llm(model="identity")
    original_type = type(model)
    _models.wrap_model(model)
    _models.wrap_model(model)

    assert is_swapped(model)
    assert isinstance(model, original_type)
    assert type(model).__name__ == original_type.__name__
    assert type(model).__module__ == original_type.__module__
    assert type(model).__mro__[1] is original_type
    assert model.model == "identity"


def test_wrapping_keeps_the_identity_and_type_of_a_tool(scripted_llm: ScriptedLlmFactory) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.tools import AgentTool, FunctionTool

    from nemo_relay.integrations.google_adk import _tools
    from nemo_relay.integrations.google_adk._swap import is_swapped

    tool = FunctionTool(get_weather)
    _tools.wrap_tool(tool)
    _tools.wrap_tool(tool)
    agent_tool = AgentTool(agent=LlmAgent(name="sub", model=scripted_llm()))
    _tools.wrap_tool(agent_tool)
    other = FunctionTool(get_weather)
    _tools.wrap_tool(other)

    assert is_swapped(tool)
    assert isinstance(tool, FunctionTool)
    assert type(tool).__name__ == "FunctionTool"
    assert tool.func is get_weather
    assert isinstance(agent_tool, AgentTool)
    assert type(other) is type(tool)


def test_wrapping_keeps_the_identity_and_type_of_a_workflow_and_its_nodes() -> None:
    from google.adk.workflow import START, Workflow

    from nemo_relay.integrations.google_adk import wrap_node
    from nemo_relay.integrations.google_adk._swap import is_swapped

    workflow = Workflow(name="wf", edges=[("START", get_weather)])

    assert wrap_node(workflow) is workflow
    assert wrap_node(workflow) is workflow
    assert is_swapped(workflow)
    assert isinstance(workflow, Workflow)
    assert type(workflow).__name__ == "Workflow"
    (function_node,) = [child for child in present(workflow.graph).nodes if child is not START]
    assert is_swapped(function_node)
    assert type(function_node).__name__ == "FunctionNode"
    assert not is_swapped(START)


@pytest.fixture(name="default_llm_class")
def default_llm_class_fixture(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[BaseLlm]]:
    from google.adk.agents import LlmAgent
    from google.adk.models import BaseLlm, registry

    class DefaultLlm(BaseLlm):
        model: str = "relay-default-model"

        @override
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield model_answer("Default.")

    monkeypatch.setattr(LlmAgent, "_default_model", LlmAgent._default_model)
    monkeypatch.setitem(registry._llm_registry_dict, "relay-default-model", DefaultLlm)
    registry.LLMRegistry.resolve.cache_clear()
    yield DefaultLlm
    registry.LLMRegistry.resolve.cache_clear()


async def test_agent_relying_on_the_default_model_name_is_pinned_for_the_run_and_recorded(
    default_llm_class: type[BaseLlm], subscribed_events: list[nemo_relay.Event], caplog: pytest.LogCaptureFixture
) -> None:
    from google.adk.agents import LlmAgent

    LlmAgent.set_default_model("relay-default-model")
    agent = LlmAgent(name="default_agent", instruction="Be brief.")
    assert agent.canonical_model is not agent.canonical_model
    with caplog.at_level(logging.WARNING):
        adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Default."
    assert agent.model == ""
    (llm_start,) = starts(subscribed_events, "llm")
    assert llm_start.name == "relay-default-model"
    assert [record.message for record in caplog.records if "NeMo Relay" in record.message] == []
    assert_balanced(subscribed_events)


async def test_agent_relying_on_a_default_model_instance_is_recorded(
    default_llm_class: type[BaseLlm], subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent

    from nemo_relay.integrations.google_adk._swap import is_swapped

    default = default_llm_class(model="relay-default-model")
    LlmAgent.set_default_model(default)
    agent = LlmAgent(name="default_agent", instruction="Be brief.")
    adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Default."
    assert is_swapped(default)
    assert agent.model == ""
    (llm_start,) = starts(subscribed_events, "llm")
    assert present(llm_start.category_profile)["model_name"] == "relay-default-model"
    assert_balanced(subscribed_events)


async def test_cancelled_invocation_does_not_nest_the_next_one(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    reached = asyncio.Event()

    async def get_weather(location: str) -> dict[str, str]:
        """Get the weather, and block the first call until the run is cancelled."""
        if not reached.is_set():
            reached.set()
            await hang()
        return {"location": location, "forecast": "sunny"}

    runner = make_runner(make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call()))
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as request:
        try:
            await run_until_interrupted(runner, await new_session(runner), reached)
            second = await run_turn(runner)
        finally:
            await runner.close()
    await nemo_relay.subscribers.flush_async()

    assert final_text(second) == "Answer from weather_agent to 'hi'."
    invocations = invocation_starts(subscribed_events)
    assert len(invocations) == 2
    assert all(invocation.parent_uuid == request.uuid for invocation in invocations)
    (cancelled_end,) = [end for end in ends(subscribed_events, "agent") if end.uuid == invocations[0].uuid]
    assert metadata_of(cancelled_end)["otel.status_description"] == "invocation cancelled"
    assert_balanced(subscribed_events)


async def test_stream_closed_before_its_final_response_records_the_partials(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.models import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    class HangingLlm(BaseLlm):
        model: str = "hanging-model"

        @override
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Partial ")]), partial=True)
            await hang()

    partial_emitted = asyncio.Event()
    runner = make_runner(
        LlmAgent(name="weather_agent", model=HangingLlm()),
        plugins=[NemoRelayPlugin(), event_signal(partial_emitted, lambda event: bool(event.partial))],
    )
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        try:
            await run_until_interrupted(
                runner,
                await new_session(runner),
                partial_emitted,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        finally:
            await runner.close()
    await nemo_relay.subscribers.flush_async()

    (llm_end,) = ends(subscribed_events, "llm")
    assert response_of(llm_end).message == "Partial "
    assert status_of(llm_end) == "ERROR"
    assert_balanced(subscribed_events)


def test_racing_first_swaps_share_one_subclass() -> None:
    from nemo_relay.integrations.google_adk._swap import swap_class

    class Fresh:
        pass

    building = threading.Event()
    second_asked = threading.Event()
    built: list[type[Fresh]] = []

    def methods(cls: type[Fresh]) -> Mapping[str, object]:
        built.append(cls)
        building.set()
        assert second_asked.wait(HANG_TIMEOUT)
        return {"origin": cls}

    first, second = Fresh(), Fresh()

    def swap_second() -> None:
        second_asked.set()
        swap_class(second, methods)

    with ThreadPoolExecutor(max_workers=2) as pool:
        racing = pool.submit(swap_class, first, methods)
        assert building.wait(HANG_TIMEOUT)
        pool.submit(swap_second).result(HANG_TIMEOUT)
        racing.result(HANG_TIMEOUT)

    assert built == [Fresh]
    assert type(first) is type(second)
    assert type(first).__mro__[1] is Fresh


async def test_live_run_leaves_the_model_and_its_connection_unwrapped(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.live import LiveRequestQueue
    from google.adk.models import BaseLlm
    from google.adk.models.base_llm_connection import BaseLlmConnection
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from nemo_relay.integrations.google_adk._swap import is_swapped

    class EchoConnection(BaseLlmConnection):
        def __init__(self) -> None:
            self.replies: asyncio.Queue[LlmResponse | None] = asyncio.Queue()
            self.closed = False

        @override
        async def send_history(self, history: list[types.Content]) -> None:
            pass

        @override
        async def send_content(self, content: types.Content) -> None:
            text = content.parts[0].text if content.parts else None
            self.replies.put_nowait(LlmResponse(content=types.ModelContent(f"Echo {text}"), turn_complete=True))

        @override
        async def send_realtime(self, blob: types.Blob) -> None:
            pass

        @override
        async def receive(self) -> AsyncGenerator[LlmResponse, None]:
            if self.closed:
                return
            while (reply := await self.replies.get()) is not None:
                yield reply

        @override
        async def close(self) -> None:
            self.closed = True
            self.replies.put_nowait(None)

    connections: list[EchoConnection] = []

    class EchoLlm(BaseLlm):
        model: str = "gemini-live-2.5-flash"

        @override
        @contextlib.asynccontextmanager
        async def connect(self, llm_request: LlmRequest) -> AsyncGenerator[BaseLlmConnection, None]:
            connections.append(EchoConnection())
            yield connections[-1]

        @override
        def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            raise AssertionError("a live run calls connect only")

    model = EchoLlm()
    runner = make_runner(LlmAgent(name="live_agent", model=model))
    queue = LiveRequestQueue()
    queue.send_content(types.UserContent("hi"))
    echoed: list[Event] = []
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        try:
            async with asyncio.timeout(HANG_TIMEOUT):
                async for event in runner.run_live(
                    user_id="user",
                    session_id=await new_session(runner),
                    live_request_queue=queue,
                    run_config=RunConfig(streaming_mode=StreamingMode.BIDI, response_modalities=[types.Modality.TEXT]),
                ):
                    if final_text([event]) == "Echo hi":
                        echoed.append(event)
                        await connections[0].close()
                        queue.close()
        finally:
            await runner.close()
    await nemo_relay.subscribers.flush_async()

    assert len(echoed) == 1
    assert not is_swapped(model)
    assert starts(subscribed_events, "llm") == []
    assert named(subscribed_events, "live_agent", "agent").parent_uuid == invocation_start(subscribed_events).uuid
    assert_balanced(subscribed_events)


async def run_confirmed(scripted_llm: ScriptedLlmFactory, *, confirmed: bool) -> list[Event]:
    """Run an agent whose tool needs confirmation, and then resume the run with the confirmation answer."""
    from google.adk.tools import FunctionTool
    from google.genai import types

    tool = FunctionTool(get_weather, require_confirmation=True)
    runner = make_runner(make_agent(scripted_llm, "weather_agent", tools=[tool], call=weather_call()))
    try:
        session_id = await new_session(runner)
        first = await run_turn(runner, session_id=session_id)
        (request,) = [
            call for event in first for call in event.get_function_calls() if call.name == "adk_request_confirmation"
        ]
        answer = types.FunctionResponse(
            id=request.id, name="adk_request_confirmation", response={"confirmed": confirmed}
        )
        resume = types.Content(role="user", parts=[types.Part(function_response=answer)])
        return await run_turn(runner, session_id=session_id, message=resume)
    finally:
        await runner.close()


async def test_tool_confirmation_records_the_pause_as_a_mark_and_the_confirmed_call_as_a_scope(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        resumed_turn = await run_confirmed(scripted_llm, confirmed=True)
    await nemo_relay.subscribers.flush_async()

    assert final_text(resumed_turn) == "Answer from weather_agent to 'hi'."
    (tool,) = starts(subscribed_events, "tool")
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool.name == "get_weather"
    assert tool_end.uuid == tool.uuid
    assert tool_end.data == weather_result()
    (requested,) = marks(subscribed_events, "adk.tool_confirmation_requested")
    (resumed,) = marks(subscribed_events, "adk.tool_confirmation_resumed")
    assert resumed.data == {"function_call_id": data_of(requested)["function_call_ids"][0], "confirmed": True}
    assert_balanced(subscribed_events)


async def test_tool_confirmation_rejected_runs_no_tool(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        resumed_turn = await run_confirmed(scripted_llm, confirmed=False)
    await nemo_relay.subscribers.flush_async()

    assert final_text(resumed_turn) == "Answer from weather_agent to 'hi'."
    (rejected,) = [
        response
        for event in resumed_turn
        for response in event.get_function_responses()
        if response.name == "get_weather"
    ]
    assert rejected.response == {"error": "This tool call is rejected."}
    assert starts(subscribed_events, "tool") == []
    (resumed,) = marks(subscribed_events, "adk.tool_confirmation_resumed")
    assert data_of(resumed)["confirmed"] is False
    assert_balanced(subscribed_events)


async def test_hooks_outside_an_observed_invocation_still_wrap_without_scopes(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    """Verify that agent hooks without `before_run_callback` record no scope and still wrap the agent's model."""
    from google.adk.agents.callback_context import CallbackContext

    from nemo_relay.integrations.google_adk import NemoRelayPlugin
    from nemo_relay.integrations.google_adk._swap import is_swapped

    plugin = NemoRelayPlugin()
    agent = make_agent(scripted_llm, "orphan")
    context = invocation_context(agent, "e-orphan")
    assert await plugin.before_agent_callback(agent=agent, callback_context=CallbackContext(context)) is None
    assert await plugin.after_agent_callback(agent=agent, callback_context=CallbackContext(context)) is None
    await nemo_relay.subscribers.flush_async()

    assert is_swapped(agent.canonical_model)
    assert starts(subscribed_events) == []


async def test_tool_hooks_wrap_tools_before_adk_runs_them(scripted_llm: ScriptedLlmFactory) -> None:
    """Verify that `before_tool_callback` wraps a tool that arrives without a model request.

    A workflow tool node reaches ADK's tool hooks in this way.
    """
    from google.adk.tools import FunctionTool
    from google.adk.tools.tool_context import ToolContext

    from nemo_relay.integrations.google_adk import NemoRelayPlugin
    from nemo_relay.integrations.google_adk._swap import is_swapped

    plugin = NemoRelayPlugin()
    tool = FunctionTool(get_weather)
    context = ToolContext(invocation_context(make_agent(scripted_llm, "a"), "e-tool"))
    assert await plugin.before_tool_callback(tool=tool, tool_args={"location": "Paris"}, tool_context=context) is None
    assert is_swapped(tool)


async def test_custom_tool_class_under_adk_tool_thread_pool_is_recorded(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    """Verify that a custom tool class runs on the event loop and is recorded under ADK's tool thread pool.

    The pool offloads only the sync callables of function tools.
    """
    from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig
    from google.adk.tools import BaseTool
    from google.genai import types

    class CustomWeather(BaseTool):
        def __init__(self) -> None:
            super().__init__(name="get_weather", description="Get the weather.")

        @override
        def _get_declaration(self) -> types.FunctionDeclaration:
            return types.FunctionDeclaration(
                name="get_weather",
                description="Get the weather.",
                parameters_json_schema={"type": "object", "properties": {"location": {"type": "string"}}},
            )

        @override
        async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
            return {"location": args["location"], "forecast": "sunny"}

    agent = make_agent(scripted_llm, "weather_agent", tools=[CustomWeather()], call=weather_call())
    adk_events = await run_recorded(agent, run_config=RunConfig(tool_thread_pool_config=ToolThreadPoolConfig()))

    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    agent_scope = named(subscribed_events, "weather_agent", "agent")
    (tool,) = starts(subscribed_events, "tool")
    assert tool.parent_uuid == agent_scope.uuid
    assert metadata_of(tool)["adk_tool_type"] == "CustomWeather"
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == weather_result()
    assert_balanced(subscribed_events)


async def test_cancelled_invocation_closed_elsewhere_does_not_nest_a_run_on_another_plugin(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    """Verify that a second plugin's invocation nests under the request scope.

    The context still holds the variables of an invocation that another context cancelled by closing its plugin.
    """
    reached = asyncio.Event()

    async def get_weather(location: str) -> dict[str, str]:
        """Get the weather, and block each call until the run is cancelled."""
        reached.set()
        await hang()
        return {"location": location, "forecast": "sunny"}

    first = make_runner(make_agent(scripted_llm, "hanging_agent", tools=[get_weather], call=weather_call()))
    second = make_runner(make_agent(scripted_llm, "weather_agent"))
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as request:
        await run_until_interrupted(first, await new_session(first), reached)
        await asyncio.create_task(first.close(), context=contextvars.Context())
        try:
            events = await run_turn(second)
        finally:
            await second.close()
    await nemo_relay.subscribers.flush_async()

    assert final_text(events) == "Answer from weather_agent to 'hi'."
    invocations = invocation_starts(subscribed_events)
    assert len(invocations) == 2
    assert all(invocation.parent_uuid == request.uuid for invocation in invocations)
    assert_balanced(subscribed_events)


def test_closer_drain_closes_pending_scopes_in_stack_order_and_cancels_owned_scopes_left_open(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from nemo_relay.integrations.google_adk._context import OK_STATUS, DeferredCloser, ScopeContext

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        closer = DeferredCloser(ScopeContext.capture(), "e-drain")
        outer = nemo_relay.scope.push("outer", nemo_relay.ScopeType.Agent)
        inner = nemo_relay.scope.push("inner", nemo_relay.ScopeType.Function)
        closer.own(outer.uuid)
        closer.own(inner.uuid)
        closer.close(outer, output={"status": "ok"}, metadata=OK_STATUS)
        closer.drain()
    nemo_relay.subscribers.flush()

    assert [(event.name, event.data) for event in ends(subscribed_events)] == [
        ("inner", {"status": "cancelled"}),
        ("outer", {"status": "ok"}),
        ("request", None),
    ]
    inner_end = named_end(subscribed_events, "inner")
    assert metadata_of(inner_end)["otel.status_description"] == "invocation cancelled"
    assert_balanced(subscribed_events)
