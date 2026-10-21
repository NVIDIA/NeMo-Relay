# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test workflow nodes with the Google ADK NeMo Relay integration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from _helpers import (
    HANG_TIMEOUT,
    ScriptedLlmFactory,
    assert_balanced,
    data_of,
    ends,
    final_text,
    get_weather,
    hang,
    invocation_start,
    make_agent,
    make_runner,
    marks,
    metadata_of,
    named,
    named_end,
    run_app,
    run_recorded,
    run_turn,
    starts,
    status_of,
    tool_call,
    weather_result,
)

import nemo_relay

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.context import Context
    from google.adk.events import RequestInput


async def test_workflow_nodes_record_scopes_under_the_workflow(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.workflow import Workflow

    from nemo_relay.integrations.google_adk import wrap_node

    def shout(node_input: str) -> str:
        return node_input.upper()

    workflow = wrap_node(Workflow(name="pipeline", edges=[("START", make_agent(scripted_llm, "greeter"), shout)]))
    adk_events = await run_recorded(workflow)

    assert any(event.output == "ANSWER FROM GREETER TO 'HI'." for event in adk_events)
    assert_balanced(subscribed_events)
    invocation = invocation_start(subscribed_events)
    pipeline = named(subscribed_events, "pipeline", "agent")
    assert pipeline.parent_uuid == invocation.uuid
    greeter = named(subscribed_events, "greeter", "agent")
    assert greeter.parent_uuid == pipeline.uuid
    (llm_call,) = starts(subscribed_events, "llm")
    assert llm_call.parent_uuid == greeter.uuid
    shout_node = named(subscribed_events, "shout")
    assert shout_node.parent_uuid == pipeline.uuid
    assert data_of(shout_node)["node_type"] == "FunctionNode"
    assert data_of(shout_node)["node_input"] == "Answer from greeter to 'hi'."
    assert named_end(subscribed_events, "shout").data == {"status": "ok", "output": "ANSWER FROM GREETER TO 'HI'."}
    assert marks(subscribed_events, "adk.node_completed") == []


async def test_tool_node_runs_its_tool_through_relay_under_the_node(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.tools import FunctionTool
    from google.adk.workflow import Workflow

    from nemo_relay.integrations.google_adk import wrap_node

    def pick_city(node_input: object) -> dict[str, str]:
        return {"location": "Paris"}

    workflow = wrap_node(Workflow(name="pipeline", edges=[("START", pick_city, FunctionTool(get_weather))]))
    adk_events = await run_recorded(workflow)

    assert any(event.output == {"location": "Paris", "forecast": "sunny"} for event in adk_events)
    assert_balanced(subscribed_events)
    pipeline = named(subscribed_events, "pipeline", "agent")
    picker = named(subscribed_events, "pick_city", "function")
    assert picker.parent_uuid == pipeline.uuid
    tool_node = named(subscribed_events, "get_weather", "function")
    assert tool_node.parent_uuid == pipeline.uuid
    (tool,) = starts(subscribed_events, "tool")
    assert tool.parent_uuid == tool_node.uuid
    assert tool.data == {"location": "Paris"}
    assert metadata_of(tool)["adk_tool_type"] == "FunctionTool"
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == weather_result("Paris")
    assert named_end(subscribed_events, "get_weather", "function").data == {
        "status": "ok",
        "output": {"location": "Paris", "forecast": "sunny"},
    }


async def test_workflow_agent_node_that_ends_the_invocation_closes_cleanly(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.workflow import Workflow
    from google.genai import types

    from nemo_relay.integrations.google_adk import wrap_node

    def answer_directly(callback_context: CallbackContext) -> types.Content:
        return types.ModelContent("Handled without running the agent.")

    agent = make_agent(scripted_llm, "greeter")
    agent.before_agent_callback = answer_directly
    workflow = wrap_node(Workflow(name="pipeline", edges=[("START", agent)]))
    await run_recorded(workflow)

    assert_balanced(subscribed_events)
    agent_end = named_end(subscribed_events, "greeter", "agent")
    assert agent_end.data == {
        "status": "ok",
        "response": {"parts": [{"text": "Handled without running the agent."}], "role": "model"},
    }
    assert status_of(agent_end) == "OK"


async def test_wrapped_nodes_run_dynamically_from_an_orchestrator_record_scopes(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.workflow import Workflow, node

    from nemo_relay.integrations.google_adk import wrap_node

    def shout(node_input: str) -> str:
        return node_input.upper()

    child = wrap_node(node(shout))

    @node(rerun_on_resume=True)
    async def orchestrate(ctx: Context, node_input: object) -> str:
        return str(await ctx.run_node(child, "dynamic"))

    workflow = wrap_node(Workflow(name="pipeline", edges=[("START", orchestrate)]))
    adk_events = await run_recorded(workflow)

    assert any(event.output == "DYNAMIC" for event in adk_events)
    assert_balanced(subscribed_events)
    orchestrator = named(subscribed_events, "orchestrate", "function")
    dynamic = named(subscribed_events, "shout", "function")
    assert dynamic.parent_uuid == orchestrator.uuid
    assert data_of(dynamic)["node_input"] == "dynamic"


async def test_unwrapped_dynamic_nodes_are_recorded_as_marks(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.events.event import NodeInfo
    from google.adk.workflow import Workflow, node

    from nemo_relay.integrations.google_adk import wrap_node

    @node
    def shout(node_input: str) -> str:
        return node_input.upper()

    @node
    def explode(node_input: str) -> str:
        raise ValueError("boom")

    @node(rerun_on_resume=True)
    async def orchestrate(ctx: Context, node_input: object) -> str:
        shouted = str(await ctx.run_node(shout, "dynamic"))
        await ctx.run_node(explode, shouted)
        return shouted

    workflow = wrap_node(Workflow(name="pipeline", edges=[("START", orchestrate)]))
    with pytest.raises(ValueError, match="boom"):
        await run_recorded(workflow)

    assert_balanced(subscribed_events)
    (completed,) = marks(subscribed_events, "adk.node_completed")
    assert NodeInfo(path=data_of(completed)["path"]).name == "shout"
    assert data_of(completed)["output"] == "DYNAMIC"
    (failed,) = marks(subscribed_events, "adk.node_failed")
    assert NodeInfo(path=data_of(failed)["path"]).name == "explode"
    assert data_of(failed)["error_message"] == "boom"
    assert {event.name for event in starts(subscribed_events, "function")} == {"orchestrate"}


async def test_node_given_to_an_agent_as_a_tool_records_the_tool_and_the_node(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.workflow import node

    def geo_get_weather(location: str) -> dict[str, str]:
        return {"location": location, "forecast": "sunny"}

    agent = make_agent(
        scripted_llm,
        "weather_agent",
        tools=[node(geo_get_weather)],
        call=tool_call("geo_get_weather", location="San Francisco"),
    )
    adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    assert_balanced(subscribed_events)
    weather_agent = named(subscribed_events, "weather_agent", "agent")
    (tool,) = starts(subscribed_events, "tool")
    assert tool.name == "geo_get_weather"
    assert tool.parent_uuid == weather_agent.uuid
    assert metadata_of(tool)["adk_tool_type"] == "NodeTool"
    node_run = named(subscribed_events, "geo_get_weather", "function")
    assert node_run.parent_uuid == weather_agent.uuid
    assert named_end(subscribed_events, "geo_get_weather", "function").data == {
        "status": "ok",
        "output": {"location": "San Francisco", "forecast": "sunny"},
    }


async def test_scope_local_middleware_applies_inside_workflow_nodes(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.tools import FunctionTool
    from google.adk.workflow import Workflow

    from nemo_relay.integrations.google_adk import wrap_node

    def pick_city(node_input: object) -> dict[str, str]:
        return {"location": "Paris"}

    workflow = wrap_node(Workflow(name="pipeline", edges=[("START", pick_city, FunctionTool(get_weather))]))
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as request:
        nemo_relay.scope_local.register_tool_conditional_execution(
            request, f"adk-local-block-{uuid4()}", 1, lambda name, args: "blocked by a request-local guardrail"
        )
        with pytest.raises(RuntimeError, match="blocked by a request-local guardrail"):
            await run_app(workflow)
    await nemo_relay.subscribers.flush_async()

    assert_balanced(subscribed_events)
    assert starts(subscribed_events, "tool") == []
    pipeline_end = named_end(subscribed_events, "pipeline", "agent")
    assert status_of(pipeline_end) == "ERROR"
    assert "blocked by a request-local guardrail" in str(data_of(pipeline_end)["error"])


async def test_failing_workflow_node_closes_its_scope_with_the_error(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.workflow import Workflow

    from nemo_relay.integrations.google_adk import wrap_node

    def explode(node_input: str) -> str:
        raise ValueError("boom")

    workflow = wrap_node(Workflow(name="pipeline", edges=[("START", explode)]))
    with pytest.raises(ValueError, match="boom"):
        await run_recorded(workflow)

    assert_balanced(subscribed_events)
    node_end = named_end(subscribed_events, "explode")
    assert status_of(node_end) == "ERROR"
    assert node_end.data == {"error": "ValueError('boom')"}
    pipeline_end = named_end(subscribed_events, "pipeline", "agent")
    assert status_of(pipeline_end) == "ERROR"
    assert "explode" in str(data_of(pipeline_end)["error_node_path"])


async def test_join_node_records_the_aggregated_inputs_as_its_output(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.workflow import JoinNode, Workflow, node

    from nemo_relay.integrations.google_adk import wrap_node

    def left(node_input: object) -> str:
        return "L"

    def right(node_input: object) -> str:
        return "R"

    left_node, right_node, join = node(left), node(right), JoinNode(name="join")
    workflow = wrap_node(
        Workflow(name="fan_in", edges=[("START", (left_node, right_node)), (left_node, join), (right_node, join)])
    )
    adk_events = await run_recorded(workflow)

    assert any(event.output == {"left": "L", "right": "R"} for event in adk_events)
    assert_balanced(subscribed_events)
    fan_in = named(subscribed_events, "fan_in", "agent")
    functions = {event.name: event for event in starts(subscribed_events, "function")}
    assert set(functions) == {"left", "right", "join"}
    assert all(event.parent_uuid == fan_in.uuid for event in functions.values())
    assert data_of(functions["join"])["node_type"] == "JoinNode"
    assert named_end(subscribed_events, "join", "function").data == {
        "status": "ok",
        "output": {"left": "L", "right": "R"},
    }


async def test_node_timeout_closes_the_node_and_workflow_scopes(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.workflow import NodeTimeoutError, Workflow, node

    from nemo_relay.integrations.google_adk import wrap_node

    async def slow(node_input: str) -> str:
        await hang()
        return "late"

    workflow = wrap_node(Workflow(name="pipeline", edges=[("START", node(slow, timeout=0.05))]))
    with pytest.raises(NodeTimeoutError):
        async with asyncio.timeout(HANG_TIMEOUT):
            await run_recorded(workflow)

    assert_balanced(subscribed_events)
    node_end = named_end(subscribed_events, "slow")
    assert status_of(node_end) == "ERROR"
    pipeline_end = named_end(subscribed_events, "pipeline", "agent")
    assert status_of(pipeline_end) == "ERROR"


async def test_interrupt_request_is_not_recorded_as_the_node_output(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.events import RequestInput
    from google.adk.workflow import Workflow, node

    from nemo_relay.integrations.google_adk import wrap_node

    @node
    async def approve(node_input: object) -> AsyncGenerator[RequestInput, None]:
        yield RequestInput(message="Approve?", payload={"city": "Paris"})

    workflow = wrap_node(Workflow(name="approval", edges=[("START", approve)]))
    adk_events = await run_recorded(workflow)

    (request,) = [call for event in adk_events for call in event.get_function_calls()]
    assert request.name == "adk_request_input"
    assert_balanced(subscribed_events)
    assert named_end(subscribed_events, "approve", "function").data == {"status": "ok"}


async def test_wrapped_workflow_without_the_plugin_runs_unrecorded(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.workflow import Workflow

    from nemo_relay.integrations.google_adk import wrap_node

    def shout(node_input: str) -> str:
        return node_input.upper()

    runner = make_runner(wrap_node(Workflow(name="pipeline", edges=[("START", shout)])), plugins=[])
    try:
        adk_events = await run_turn(runner)
    finally:
        await runner.close()
    await nemo_relay.subscribers.flush_async()

    assert [event.output for event in adk_events if event.output is not None] == ["HI"]
    assert starts(subscribed_events) == []
