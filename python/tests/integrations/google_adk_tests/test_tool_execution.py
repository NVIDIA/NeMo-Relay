# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test tool execution intercepts with the Google ADK NeMo Relay integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from _helpers import (
    ScriptedLlmFactory,
    ToolCallRecord,
    ToolNext,
    as_object,
    assert_balanced,
    data_of,
    ends,
    function_responses,
    get_weather,
    json_tool_result,
    make_agent,
    marks,
    metadata_of,
    present,
    run_recorded,
    starts,
    status_of,
    tool_call,
    tool_execution_intercept,
    tool_witness,
    weather_call,
    weather_result,
)

import nemo_relay

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FunctionType

    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext


async def test_tool_execution_intercept_rewrites_the_arguments_and_the_result(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    checked = json_tool_result({"location": "San Francisco, CA", "forecast": "sunny", "checked": True})

    async def rewrite(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        result = await next_call({**as_object(context.args), "location": "San Francisco, CA"})
        assert result.result == weather_result("San Francisco, CA")
        return nemo_relay.ToolExecutionInterceptOutcome(checked)

    with tool_execution_intercept(rewrite):
        adk_events = await run_recorded(
            make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
        )

    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"location": "San Francisco, CA", "forecast": "sunny", "checked": True}
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == checked
    assert_balanced(subscribed_events)


async def test_tool_execution_intercept_that_retries_runs_the_tool_again(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    calls: list[str] = []

    def get_weather(location: str) -> dict[str, str]:
        """Get the weather, and record each requested location."""
        calls.append(location)
        return {"location": location, "forecast": "sunny"}

    async def retry(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        await next_call(context.args)
        second = await next_call({**as_object(context.args), "location": "Oakland"})
        return nemo_relay.ToolExecutionInterceptOutcome(second.result)

    with tool_execution_intercept(retry):
        adk_events = await run_recorded(
            make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
        )

    assert calls == ["San Francisco", "Oakland"]
    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"location": "Oakland", "forecast": "sunny"}
    assert len(starts(subscribed_events, "tool")) == 1
    assert_balanced(subscribed_events)


async def test_tool_execution_intercept_that_retries_with_the_original_arguments_restores_them(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    calls: list[str] = []
    records: list[ToolCallRecord] = []

    def get_weather(location: str) -> dict[str, str]:
        """Get the weather, and record each requested location."""
        calls.append(location)
        return {"location": location, "forecast": "sunny"}

    async def retry(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        await next_call({**as_object(context.args), "location": "Oakland"})
        return nemo_relay.ToolExecutionInterceptOutcome((await next_call(context.args)).result)

    with tool_execution_intercept(retry):
        adk_events = await run_recorded(
            make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call()),
            extra_plugins=[tool_witness(records)],
        )

    assert calls == ["Oakland", "San Francisco"]
    assert [args for args, _ in records] == [{"location": "San Francisco"}]
    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"location": "San Francisco", "forecast": "sunny"}
    assert_balanced(subscribed_events)


async def test_tool_error_retried_by_an_execution_intercept(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    attempts = 0

    def get_weather(location: str) -> dict[str, str]:
        """Get the weather, and raise a timeout on the first call."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("slow upstream")
        return {"location": location, "forecast": "sunny"}

    async def retry_once(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        try:
            result = await next_call(context.args)
        except TimeoutError:
            result = await next_call(context.args)
        return nemo_relay.ToolExecutionInterceptOutcome(result.result)

    with tool_execution_intercept(retry_once):
        adk_events = await run_recorded(
            make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
        )

    assert attempts == 2
    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"location": "San Francisco", "forecast": "sunny"}
    (tool_end,) = ends(subscribed_events, "tool")
    assert status_of(tool_end) == "OK"
    assert_balanced(subscribed_events)


async def test_tool_execution_intercept_that_answers_skips_the_tool(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    calls: list[str] = []

    def get_weather(location: str) -> dict[str, str]:
        """Get the weather, and record each requested location."""
        calls.append(location)
        return {"location": location, "forecast": "sunny"}

    with tool_execution_intercept(
        lambda context, next_call: nemo_relay.ToolExecutionInterceptOutcome(json_tool_result({"forecast": "cached"})),
    ):
        adk_events = await run_recorded(
            make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
        )

    assert calls == []
    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"forecast": "cached"}
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == json_tool_result({"forecast": "cached"})
    assert_balanced(subscribed_events)


async def test_tool_error_recovered_by_an_execution_intercept(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    def get_weather(location: str) -> dict[str, str]:
        """Raise a connection error from a weather service that is down."""
        raise ConnectionError("weather service down")

    async def recover(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        try:
            result = await next_call(context.args)
        except ConnectionError as error:
            return nemo_relay.ToolExecutionInterceptOutcome(json_tool_result({"recovered": True, "reason": str(error)}))
        return nemo_relay.ToolExecutionInterceptOutcome(result.result)

    with tool_execution_intercept(recover):
        adk_events = await run_recorded(
            make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
        )

    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"recovered": True, "reason": "weather service down"}
    (tool_end,) = ends(subscribed_events, "tool")
    assert status_of(tool_end) == "OK"
    assert_balanced(subscribed_events)


async def test_tool_error_without_recovery_reaches_adk_error_hooks(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    def get_weather(location: str) -> dict[str, str]:
        """Raise a connection error from a weather service that is down."""
        raise ConnectionError("weather service down")

    def on_tool_error(
        tool: BaseTool, args: dict[str, object], tool_context: ToolContext, error: Exception
    ) -> dict[str, object] | None:
        return {"error": f"ADK handled {error}"}

    agent = make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
    agent.on_tool_error_callback = on_tool_error
    adk_events = await run_recorded(agent)

    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"error": "ADK handled weather service down"}
    (tool_end,) = ends(subscribed_events, "tool")
    assert status_of(tool_end) == "ERROR"
    assert_balanced(subscribed_events)


class _Session:
    pass


async def test_a_tool_result_with_no_json_form_reaches_adk_callbacks_unchanged(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    session = _Session()
    seen: list[object] = []

    def open_session(location: str) -> dict[str, object]:
        """Open a session and return it with its status."""
        return {"status": "open", "session": session}

    def drop_session(
        tool: BaseTool, args: dict[str, object], tool_context: ToolContext, tool_response: dict[str, object]
    ) -> dict[str, object]:
        seen.append(tool_response["session"])
        return {"status": tool_response["status"]}

    agent = make_agent(
        scripted_llm, "weather_agent", tools=[open_session], call=tool_call("open_session", location="Paris")
    )
    agent.after_tool_callback = drop_session
    adk_events = await run_recorded(agent)

    assert len(seen) == 1
    assert seen[0] is session
    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"status": "open"}
    (tool_end,) = ends(subscribed_events, "tool")
    name = f"{_Session.__module__}.{_Session.__qualname__}"
    assert as_object(as_object(data_of(tool_end)["result"])["value"])["session"] == {
        "kind": "unserializable",
        "type": name,
        "ref": 0,
        "value": f"<unserializable {name}>",
    }
    assert_balanced(subscribed_events)


async def test_a_generator_a_tool_returns_reaches_the_next_model_call_unconsumed(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    def counting() -> Iterator[int]:
        yield 0
        yield 1
        yield 2

    generator = counting()

    def list_items(location: str) -> dict[str, object]:
        """Return the items as a generator."""
        return {"items": generator}

    await run_recorded(
        make_agent(scripted_llm, "weather_agent", tools=[list_items], call=tool_call("list_items", location="Paris"))
    )

    seen = scripted_llm.seen_requests[-1].contents[-1]
    response = present(present(present(seen.parts)[0].function_response).response)
    assert response["items"] is generator
    assert [*generator] == [0, 1, 2]
    (tool_end,) = ends(subscribed_events, "tool")
    recorded = as_object(as_object(data_of(tool_end)["result"])["value"])["items"]
    assert recorded == {
        "kind": "unserializable",
        "type": "builtins.generator",
        "ref": 0,
        "value": "<unserializable iterator>",
    }
    assert_balanced(subscribed_events)


async def test_bytes_and_a_set_a_tool_returns_reach_the_next_model_call_as_adk_holds_them(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    raw, ids = b"\x89PNG", {1, 9}

    def fetch(location: str) -> dict[str, object]:
        """Return binary data and a set of ids."""
        return {"raw": raw, "ids": ids}

    await run_recorded(
        make_agent(scripted_llm, "weather_agent", tools=[fetch], call=tool_call("fetch", location="Paris"))
    )

    seen = scripted_llm.seen_requests[-1].contents[-1]
    response = present(present(present(seen.parts)[0].function_response).response)
    assert response["raw"] is raw
    assert response["ids"] is ids
    assert "iVBORw==" in json.dumps(data_of(starts(subscribed_events, "llm")[-1]))
    assert_balanced(subscribed_events)


async def test_tool_result_objects_reach_adk_unchanged_and_are_recorded_in_full(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.genai import types

    returned: list[types.Part] = []
    records: list[ToolCallRecord] = []

    def get_weather(location: str) -> types.Part:
        """Return a PNG image part."""
        returned.append(types.Part.from_bytes(data=b"\x89PNG", mime_type="image/png"))
        return returned[-1]

    adk_events = await run_recorded(
        make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call()),
        extra_plugins=[tool_witness(records)],
    )

    ((_, result),) = records
    assert result is returned[0]
    (function_response,) = function_responses(adk_events)
    assert present(present(function_response.parts)[0].inline_data).data == b"\x89PNG"
    (tool_end,) = ends(subscribed_events, "tool")
    assert data_of(tool_end)["content"] == '{"inlineData": {"data": "iVBORw==", "mimeType": "image/png"}}'
    recorded = as_object(data_of(tool_end)["result"])
    assert recorded["kind"] == "typed"
    assert recorded["type"] == "google.genai.types.Part"
    assert as_object(recorded["value"])["inlineData"] == {
        "data": "iVBORw==",
        "displayName": None,
        "mimeType": "image/png",
    }
    assert_balanced(subscribed_events)


def weather_text(location: str) -> str:
    """Get the weather as text."""
    return f"Sunny in {location}."


async def test_text_tool_result_records_its_content_and_takes_an_intercept_edit(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    async def edit(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        result = as_object((await next_call(context.args)).result)
        assert result["content"] == "Sunny in San Francisco."
        return nemo_relay.ToolExecutionInterceptOutcome({**result, "content": "Foggy in San Francisco."})

    with tool_execution_intercept(edit):
        adk_events = await run_recorded(
            make_agent(
                scripted_llm,
                "weather_agent",
                tools=[weather_text],
                call=tool_call("weather_text", location="San Francisco"),
            )
        )

    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"result": "Foggy in San Francisco."}
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == {
        "schema": "nemo.relay.tool_result@1",
        "content": "Foggy in San Francisco.",
        "is_error": False,
        "result": {"kind": "json", "value": "Sunny in San Francisco."},
    }
    assert status_of(tool_end) == "OK"
    assert_balanced(subscribed_events)


def weather_down(location: str) -> dict[str, str]:
    """Return an error result that reports the weather service as down."""
    return {"error": f"No weather for {location}."}


async def test_a_tool_result_with_an_error_records_a_failed_call_as_adk_reads_it(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    adk_events = await run_recorded(
        make_agent(
            scripted_llm, "weather_agent", tools=[weather_down], call=tool_call("weather_down", location="Paris")
        )
    )

    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"error": "No weather for Paris."}
    (tool_end,) = ends(subscribed_events, "tool")
    assert data_of(tool_end)["content"] == "No weather for Paris."
    assert data_of(tool_end)["is_error"] is True
    assert status_of(tool_end) == "ERROR"
    assert_balanced(subscribed_events)


async def test_an_intercept_fails_an_object_result_by_writing_its_json_text(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    content = '{"error": "Stale forecast."}'

    async def fail(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        result = as_object((await next_call(context.args)).result)
        return nemo_relay.ToolExecutionInterceptOutcome({**result, "content": content, "is_error": True})

    with tool_execution_intercept(fail):
        adk_events = await run_recorded(
            make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
        )

    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"error": "Stale forecast."}
    (tool_end,) = ends(subscribed_events, "tool")
    assert data_of(tool_end)["content"] == content
    assert data_of(tool_end)["is_error"] is True
    assert status_of(tool_end) == "ERROR"
    assert_balanced(subscribed_events)


async def test_an_intercept_fails_a_text_result_through_adk_error_convention(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    async def fail(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        result = as_object((await next_call(context.args)).result)
        return nemo_relay.ToolExecutionInterceptOutcome({**result, "is_error": True})

    with tool_execution_intercept(fail):
        adk_events = await run_recorded(
            make_agent(
                scripted_llm,
                "weather_agent",
                tools=[weather_text],
                call=tool_call("weather_text", location="San Francisco"),
            )
        )

    (function_response,) = function_responses(adk_events)
    assert function_response.response == {"error": "Sunny in San Francisco."}
    (tool_end,) = ends(subscribed_events, "tool")
    assert data_of(tool_end)["content"] == "Sunny in San Francisco."
    assert data_of(tool_end)["is_error"] is True
    assert status_of(tool_end) == "ERROR"
    assert_balanced(subscribed_events)


async def test_arguments_relay_cannot_project_fail_the_call_under_a_rejection_mark(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from pydantic import BaseModel

    class Upload(BaseModel):
        data: bytes

    def attach_upload(tool: BaseTool, args: dict[str, object], tool_context: ToolContext) -> None:
        args["upload"] = Upload(data=b"\x89PNG")

    agent = make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
    agent.before_tool_callback = attach_upload
    with pytest.raises(UnicodeDecodeError) as raised:
        await run_recorded(agent)

    assert starts(subscribed_events, "tool") == []
    (rejected,) = marks(subscribed_events, "adk.tool_arguments_rejected")
    (agent_start,) = [event for event in starts(subscribed_events, "agent") if event.name == "weather_agent"]
    assert rejected.parent_uuid == agent_start.uuid
    data = data_of(rejected)
    assert str(data["tool_call_id"]).startswith("adk-")
    assert {key: data[key] for key in ("tool", "error_type", "error_message")} == {
        "tool": "get_weather",
        "error_type": "UnicodeDecodeError",
        "error_message": str(raised.value),
    }
    assert metadata_of(rejected)["nemo_relay.log.severity"] == "error"
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(
    ("tool", "edit", "message"),
    [
        pytest.param(
            weather_text,
            {"content": None, "is_error": True},
            "A failed ADK tool result needs error text",
            id="text_failed",
        ),
        pytest.param(get_weather, {"content": "Foggy."}, "is its JSON text", id="object_content"),
    ],
)
async def test_tool_result_edits_adk_cannot_hold_fail_the_run(
    tool: FunctionType,
    edit: nemo_relay.JsonObject,
    message: str,
    scripted_llm: ScriptedLlmFactory,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    async def apply(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        result = as_object((await next_call(context.args)).result)
        return nemo_relay.ToolExecutionInterceptOutcome({**result, **edit})

    call = tool_call(tool.__name__, location="San Francisco")
    with (
        tool_execution_intercept(apply),
        pytest.raises(ValueError, match=message) as raised,
    ):
        await run_recorded(make_agent(scripted_llm, "weather_agent", tools=[tool], call=call))

    (tool_start,) = starts(subscribed_events, "tool")
    (rejected,) = marks(subscribed_events, "adk.tool_result_rejected")
    assert rejected.parent_uuid == tool_start.parent_uuid
    assert data_of(rejected) == {
        "tool": tool.__name__,
        "tool_call_id": present(tool_start.category_profile)["tool_call_id"],
        "error_type": "ValueError",
        "error_message": str(raised.value),
    }
    assert metadata_of(rejected)["nemo_relay.log.severity"] == "error"
    assert_balanced(subscribed_events)
