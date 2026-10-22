# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test request intercepts with the Google ADK NeMo Relay integration."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest
from _helpers import (
    LlmNext,
    ScriptedLlmFactory,
    ToolCallRecord,
    as_instance,
    as_object,
    assert_balanced,
    data_of,
    ends,
    final_text,
    function_responses,
    get_weather,
    llm_execution_intercept,
    llm_request_intercept,
    make_agent,
    model_answer,
    present,
    responding_llm,
    run_recorded,
    starts,
    status_of,
    tool_call,
    tool_request_intercept,
    tool_witness,
    weather_call,
)

import nemo_relay

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext
    from google.genai import types


async def test_tool_request_intercept_rewrites_the_arguments_the_tool_receives(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    seen: list[str] = []
    records: list[ToolCallRecord] = []

    def forecast(location: str) -> dict[str, str]:
        """Record the location that the tool receives."""
        seen.append(location)
        return {"location": location}

    def relocate(name: str, args: nemo_relay.Json) -> nemo_relay.Json:
        return {**as_object(args), "location": "Intercepted City"}

    with tool_request_intercept(
        relocate,
    ):
        adk_events = await run_recorded(
            make_agent(
                scripted_llm,
                "weather_agent",
                tools=[forecast],
                call=tool_call("forecast", location="San Francisco"),
            ),
            extra_plugins=[tool_witness(records)],
        )

    assert seen == ["Intercepted City"]
    assert [args for args, _ in records] == [{"location": "Intercepted City"}]
    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"location": "Intercepted City"}
    (tool,) = starts(subscribed_events, "tool")
    assert tool.data == {"location": "Intercepted City"}
    assert_balanced(subscribed_events)


async def test_tool_request_intercept_edit_keeps_the_arguments_it_left_unchanged(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    seen: list[tuple[str, bytes]] = []

    def forecast(location: str, image: bytes = b"") -> dict[str, str]:
        """Record the location and image that the tool receives."""
        seen.append((location, image))
        return {"location": location}

    def add_image(tool: BaseTool, args: dict[str, object], tool_context: ToolContext) -> None:
        args["image"] = b"\x89PNG"

    def relocate(name: str, args: nemo_relay.Json) -> nemo_relay.Json:
        return {**as_object(args), "location": "Intercepted City"}

    agent = make_agent(
        scripted_llm, "weather_agent", tools=[forecast], call=tool_call("forecast", location="San Francisco")
    )
    agent.before_tool_callback = add_image
    with tool_request_intercept(relocate):
        await run_recorded(agent)

    assert seen == [("Intercepted City", b"\x89PNG")]
    (tool,) = starts(subscribed_events, "tool")
    assert tool.data == {"location": "Intercepted City", "image": "iVBORw=="}
    assert_balanced(subscribed_events)


async def test_tool_request_intercept_returning_a_non_object_fails_the_call(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    with (
        tool_request_intercept(
            lambda name, args: ["not", "an", "object"],
        ),
        pytest.raises(TypeError, match="must be a JSON object"),
    ):
        await run_recorded(make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call()))

    assert_balanced(subscribed_events)
    (tool_end,) = ends(subscribed_events, "tool")
    assert status_of(tool_end) == "ERROR"


async def test_parallel_calls_to_one_tool_without_ids_run_through_relay(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    def call_twice_then_answer(request: LlmRequest) -> LlmResponse:
        if any(part.function_response for content in request.contents for part in content.parts or []):
            return model_answer("Both sunny.")
        parts = [
            types.Part(function_call=types.FunctionCall(name="get_weather", args={"location": city}))
            for city in ("Paris", "Rome")
        ]
        return LlmResponse(content=types.Content(role="model", parts=parts), finish_reason=types.FinishReason.STOP)

    model = responding_llm(call_twice_then_answer, model="parallel-model")
    adk_events = await run_recorded(
        LlmAgent(name="weather_agent", model=model, instruction="Be brief.", tools=[get_weather])
    )

    assert final_text(adk_events) == "Both sunny."
    assert_balanced(subscribed_events)
    assert [event.name for event in starts(subscribed_events, "llm")] == ["RespondingLlm"] * 2
    tools = starts(subscribed_events, "tool")
    assert sorted(str(data_of(tool)["location"]) for tool in tools if isinstance(tool.data, dict)) == ["Paris", "Rome"]
    adk_ids = [
        part.function_call.id
        for event in adk_events
        if event.content
        for part in event.content.parts or []
        if part.function_call is not None
    ]
    assert len(adk_ids) == 2
    assert len(set(adk_ids)) == 2
    assert None not in adk_ids
    assert {str(present(tool.category_profile)["tool_call_id"]) for tool in tools if tool.category_profile} == set(
        adk_ids
    )


async def test_llm_request_intercept_write_back_reaches_the_model(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    def rewrite_last_user_message(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        annotated = present(annotated)
        messages = list(annotated.messages)
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                messages[index] = {**messages[index], "content": "REWRITTEN BY RELAY"}
                break
        annotated.messages = messages
        return nemo_relay.LLMRequestInterceptOutcome(request, annotated)

    with llm_request_intercept(
        rewrite_last_user_message,
    ):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"))

    assert final_text(adk_events) == "Answer from weather_agent to 'REWRITTEN BY RELAY'."
    seen = scripted_llm.seen_requests[-1]
    assert present(seen.contents[-1].parts)[0].text == "REWRITTEN BY RELAY"
    assert present(seen.config.labels)["adk_agent_name"] == "weather_agent"
    assert_balanced(subscribed_events)


async def test_structured_output_schema_reaches_intercepts_and_an_edit_reaches_the_model(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from pydantic import BaseModel

    class Forecast(BaseModel):
        location: str
        forecast: str

    schemas: list[types.SchemaUnion | None] = []
    mime_types: list[str | None] = []
    seen_annotated: list[nemo_relay.Json] = []

    def answer_with_json(request: LlmRequest) -> LlmResponse:
        schemas.append(request.config.response_schema)
        mime_types.append(request.config.response_mime_type)
        return model_answer('{"location": "San Francisco", "forecast": "sunny"}')

    def read_schema(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        annotated = present(annotated)
        seen_annotated.append(present(annotated.api_specific)["data"])
        return nemo_relay.LLMRequestInterceptOutcome(request, annotated)

    async def describe_schema(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmNext
    ) -> nemo_relay.Json:
        content = copy.deepcopy(request.content)
        content["config"]["response_schema"]["description"] = "A forecast for one location."
        return await next_call(nemo_relay.LLMRequest(request.headers, content))

    model = responding_llm(answer_with_json, model="json-model")
    with (
        llm_request_intercept(
            read_schema,
        ),
        llm_execution_intercept(
            describe_schema,
        ),
    ):
        adk_events = await run_recorded(
            LlmAgent(name="weather_agent", model=model, instruction="Be brief.", output_schema=Forecast)
        )

    (data,) = seen_annotated
    config = as_object(as_object(data)["config"])
    assert config["response_schema"] == Forecast.model_json_schema()
    assert config["response_mime_type"] == "application/json"
    assert schemas == [{**Forecast.model_json_schema(), "description": "A forecast for one location."}]
    assert mime_types == ["application/json"]
    assert final_text(adk_events) == '{"location": "San Francisco", "forecast": "sunny"}'
    assert_balanced(subscribed_events)


async def test_llm_request_intercept_edits_the_tool_declaration_the_model_sees(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.genai import types

    def describe_tool(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        annotated = present(annotated)
        tools = list(annotated.tools or [])
        weather = as_object(tools[0])
        function = as_object(weather["function"])
        tools[0] = {**weather, "function": {**function, "description": "Edited by Relay."}}
        annotated.tools = tools
        return nemo_relay.LLMRequestInterceptOutcome(request, annotated)

    with llm_request_intercept(
        describe_tool,
    ):
        adk_events = await run_recorded(
            make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
        )

    seen = scripted_llm.seen_requests[0]
    (tool,) = seen.config.tools or []
    assert present(as_instance(tool, types.Tool).function_declarations)[0].description == "Edited by Relay."
    assert "get_weather" in seen.tools_dict
    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"location": "San Francisco", "forecast": "sunny"}
    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    assert_balanced(subscribed_events)


async def test_adk_request_headers_reach_intercepts_and_a_removal_reaches_the_model(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.genai import types

    seen_by_intercept: list[nemo_relay.JsonObject] = []
    seen_by_model: list[dict[str, str]] = []

    def record_headers(request: LlmRequest) -> LlmResponse:
        options = request.config.http_options
        seen_by_model.append(dict((options and options.headers) or {}))
        return model_answer("ok")

    def drop_secret(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        seen_by_intercept.append(dict(request.headers))
        headers = {key: value for key, value in request.headers.items() if key != "x-secret"}
        return nemo_relay.LLMRequestInterceptOutcome(nemo_relay.LLMRequest(headers, request.content), annotated)

    config = types.GenerateContentConfig(http_options=types.HttpOptions(headers={"x-secret": "s", "x-keep": "k"}))
    agent = LlmAgent(
        name="weather_agent",
        model=responding_llm(record_headers, model="header-model"),
        instruction="Be brief.",
        generate_content_config=config,
    )
    with llm_request_intercept(drop_secret):
        adk_events = await run_recorded(agent)

    assert seen_by_intercept == [{"x-secret": "s", "x-keep": "k"}]
    (sent,) = seen_by_model
    assert sent["x-keep"] == "k"
    assert "x-secret" not in sent
    assert final_text(adk_events) == "ok"
    assert_balanced(subscribed_events)


async def test_relay_request_headers_reach_the_model_on_every_call(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    seen_headers: list[dict[str, str]] = []

    def record_headers(request: LlmRequest) -> LlmResponse:
        options = request.config.http_options
        seen_headers.append(dict((options and options.headers) or {}))
        return model_answer("ok")

    def add_lineage(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        headers = {**request.headers, "x-dynamo-session": "session-1"}
        return nemo_relay.LLMRequestInterceptOutcome(nemo_relay.LLMRequest(headers, request.content), annotated)

    async def retry_with_attempt(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmNext
    ) -> nemo_relay.Json:
        first = {**request.headers, "x-attempt": "1", "x-first-try": "yes"}
        await next_call(nemo_relay.LLMRequest(first, request.content))
        return await next_call(nemo_relay.LLMRequest({**request.headers, "x-attempt": "2"}, request.content))

    model = responding_llm(record_headers, model="header-model")
    with (
        llm_request_intercept(
            add_lineage,
        ),
        llm_execution_intercept(
            retry_with_attempt,
        ),
    ):
        adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert [headers.get("x-dynamo-session") for headers in seen_headers] == ["session-1", "session-1"]
    assert [headers.get("x-attempt") for headers in seen_headers] == ["1", "2"]
    assert [headers.get("x-first-try") for headers in seen_headers] == ["yes", None]
    assert final_text(adk_events) == "ok"
    assert_balanced(subscribed_events)
