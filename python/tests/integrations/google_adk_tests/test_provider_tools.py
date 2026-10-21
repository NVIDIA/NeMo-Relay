# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test code executors and provider-run tools with the Google ADK NeMo Relay integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from _helpers import (
    ToolNext,
    as_instance,
    as_object,
    assert_balanced,
    data_of,
    ends,
    final_text,
    marks,
    metadata_of,
    named,
    present,
    responding_llm,
    run_recorded,
    starts,
    status_of,
    tool_execution_intercept,
    tool_request_intercept,
    tool_sanitize_request_guardrail,
    tool_sanitize_response_guardrail,
    toolset_of,
)

import nemo_relay

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from google.adk.agents import LlmAgent
    from google.adk.events.event import Event
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools import BaseTool
    from google.genai import types


def _grounded(grounding: types.GroundingMetadata) -> Callable[[LlmRequest], LlmResponse]:
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    def respond(request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Sunny.")]),
            grounding_metadata=grounding,
            finish_reason=types.FinishReason.STOP,
        )

    return respond


def _grounding_tool(name: str) -> tuple[BaseTool, types.GroundingMetadata]:
    from google.adk.tools import google_search
    from google.adk.tools.enterprise_search_tool import EnterpriseWebSearchTool
    from google.adk.tools.google_maps_grounding_tool import google_maps_grounding
    from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
    from google.genai import types

    chunks = [types.GroundingChunk(web=types.GroundingChunkWeb(uri="https://weather.example", title="Weather"))]
    searched = types.GroundingMetadata(web_search_queries=["weather"], grounding_chunks=chunks)
    if name == "google_search":
        return google_search, searched
    if name == "enterprise_web_search":
        return EnterpriseWebSearchTool(), searched
    if name == "vertex_ai_search":
        tool = VertexAiSearchTool(data_store_id="projects/p/locations/l/collections/c/dataStores/weather")
        return tool, types.GroundingMetadata(retrieval_queries=["weather"], grounding_chunks=chunks)
    return google_maps_grounding, types.GroundingMetadata(
        google_maps_widget_context_token="token", grounding_chunks=chunks
    )


@pytest.mark.parametrize(
    ("name", "tool_type", "data"),
    [
        ("google_search", "GoogleSearchTool", {"queries": ["weather"]}),
        ("enterprise_web_search", "EnterpriseWebSearchTool", {"queries": ["weather"]}),
        ("vertex_ai_search", "VertexAiSearchTool", {"queries": ["weather"]}),
        ("google_maps", "GoogleMapsGroundingTool", {}),
    ],
)
async def test_provider_grounding_is_recorded_as_a_tool_scope(
    name: str, tool_type: str, data: nemo_relay.Json, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent

    tool, grounding = _grounding_tool(name)
    model = responding_llm(_grounded(grounding), model="gemini-2.5-flash")
    adk_events = await run_recorded(LlmAgent(name="searcher", model=model, tools=[tool]))

    assert final_text(adk_events) == "Sunny."
    assert_balanced(subscribed_events)
    searcher = named(subscribed_events, "searcher", "agent")
    (recorded,) = starts(subscribed_events, "tool")
    assert recorded.name == name
    assert recorded.parent_uuid == searcher.uuid
    assert recorded.data == data
    assert metadata_of(recorded)["adk_tool_type"] == tool_type
    assert metadata_of(recorded)["adk_server_side"] is True
    (tool_end,) = ends(subscribed_events, "tool")
    web = {"domain": None, "title": "Weather", "uri": "https://weather.example"}
    chunk = {"image": None, "maps": None, "retrievedContext": None, "web": web}
    content = {"chunks": [{"web": {"title": "Weather", "uri": "https://weather.example"}}]}
    assert tool_end.data == {
        "schema": "nemo.relay.tool_result@1",
        "content": json.dumps(content, ensure_ascii=False, sort_keys=True),
        "is_error": False,
        "result": {
            "kind": "dict",
            "value": {
                "chunks": {
                    "kind": "list",
                    "value": [{"kind": "typed", "type": "google.genai.types.GroundingChunk", "value": chunk}],
                }
            },
        },
    }
    assert status_of(tool_end) == "OK"


async def test_provider_tool_from_a_toolset_is_recorded_under_its_own_name(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    tool, grounding = _grounding_tool("enterprise_web_search")
    model = responding_llm(_grounded(grounding), model="gemini-2.5-flash")
    await run_recorded(LlmAgent(name="searcher", model=model, tools=[toolset_of([tool])]))

    (recorded,) = starts(subscribed_events, "tool")
    assert recorded.name == "enterprise_web_search"
    assert metadata_of(recorded)["adk_tool_type"] == "EnterpriseWebSearchTool"
    assert_balanced(subscribed_events)


async def test_tool_sanitize_guardrails_apply_to_provider_run_tools(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.tools import google_search
    from google.genai import types

    def redact(name: str, payload: nemo_relay.Json) -> nemo_relay.Json:
        return "[REDACTED]" if name == "google_search" else payload

    grounding = types.GroundingMetadata(web_search_queries=["weather at 1 Main St"])
    model = responding_llm(_grounded(grounding), model="gemini-2.5-flash")
    with (
        tool_sanitize_request_guardrail(
            redact,
        ),
        tool_sanitize_response_guardrail(
            redact,
        ),
    ):
        await run_recorded(LlmAgent(name="searcher", model=model, tools=[google_search]))

    (tool,) = starts(subscribed_events, "tool")
    assert tool.data == "[REDACTED]"
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == "[REDACTED]"


@pytest.mark.parametrize(
    ("outcome", "output", "is_error"),
    [
        pytest.param("OUTCOME_OK", "1\n", False, id="ok"),
        pytest.param("OUTCOME_FAILED", "NameError: name 'x' is not defined", True, id="failed"),
        pytest.param("OUTCOME_DEADLINE_EXCEEDED", "", True, id="deadline_exceeded"),
    ],
)
async def test_built_in_code_execution_is_recorded_as_a_tool_scope(
    outcome: str, output: str, is_error: bool, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.code_executors import BuiltInCodeExecutor
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    def executed(request: LlmRequest) -> LlmResponse:
        result = types.CodeExecutionResult(outcome=types.Outcome(outcome), output=output)
        parts = [
            types.Part(executable_code=types.ExecutableCode(code="print(1)", language=types.Language.PYTHON)),
            types.Part(code_execution_result=result),
            types.Part(text="It prints 1."),
        ]
        return LlmResponse(content=types.Content(role="model", parts=parts), finish_reason=types.FinishReason.STOP)

    model = responding_llm(executed, model="gemini-2.5-flash")
    adk_events = await run_recorded(LlmAgent(name="coder", model=model, code_executor=BuiltInCodeExecutor()))

    assert final_text(adk_events) == "It prints 1."
    assert_balanced(subscribed_events)
    (tool,) = starts(subscribed_events, "tool")
    assert tool.name == "code_execution"
    assert tool.parent_uuid == named(subscribed_events, "coder", "agent").uuid
    assert tool.data == {"code": "print(1)", "language": "PYTHON"}
    assert metadata_of(tool)["adk_tool_type"] == "BuiltInCodeExecutor"
    assert metadata_of(tool)["adk_server_side"] is True
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == {
        "schema": "nemo.relay.tool_result@1",
        "content": output,
        "is_error": is_error,
        "result": {
            "kind": "typed",
            "type": "google.genai.types.CodeExecutionResult",
            "value": {"id": None, "outcome": outcome, "output": output},
        },
    }
    assert status_of(tool_end) == ("ERROR" if is_error else "OK")


async def test_built_in_code_executions_pair_with_their_results_by_id(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.code_executors import BuiltInCodeExecutor
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    def executed(request: LlmRequest) -> LlmResponse:
        ok = types.Outcome.OUTCOME_OK
        parts = [
            types.Part(executable_code=types.ExecutableCode(id="c1", code="print(1)", language=types.Language.PYTHON)),
            types.Part(executable_code=types.ExecutableCode(id="c2", code="print(2)", language=types.Language.PYTHON)),
            types.Part(code_execution_result=types.CodeExecutionResult(id="c2", outcome=ok, output="2\n")),
            types.Part(code_execution_result=types.CodeExecutionResult(id="c1", outcome=ok, output="1\n")),
            types.Part(text="Done."),
        ]
        return LlmResponse(content=types.Content(role="model", parts=parts), finish_reason=types.FinishReason.STOP)

    model = responding_llm(executed, model="gemini-2.5-flash")
    await run_recorded(LlmAgent(name="coder", model=model, code_executor=BuiltInCodeExecutor()))

    assert_balanced(subscribed_events)
    recorded = {
        present(start.category_profile)["tool_call_id"]: (data_of(start)["code"], data_of(end)["content"])
        for start, end in zip(starts(subscribed_events, "tool"), ends(subscribed_events, "tool"), strict=True)
    }
    assert recorded == {"c1": ("print(1)", "1\n"), "c2": ("print(2)", "2\n")}


async def test_server_side_tool_calls_are_recorded_with_their_ids_in_place_of_the_grounding(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools import google_search, url_context
    from google.genai import types

    def respond(request: LlmRequest) -> LlmResponse:
        parts = [
            types.Part(tool_call=types.ToolCall(id="s1", tool_type=types.ToolType.GOOGLE_SEARCH_WEB, args={"q": "x"})),
            types.Part(tool_call=types.ToolCall(id="u1", tool_type=types.ToolType.URL_CONTEXT, args={"url": "u"})),
            types.Part(tool_response=types.ToolResponse(id="u1", response={"text": "page"})),
            types.Part(tool_response=types.ToolResponse(id="s1", response={"results": ["Sunny"]})),
            types.Part(tool_call=types.ToolCall(id="f1", tool_type=types.ToolType.FILE_SEARCH, args={})),
            types.Part(tool_response=types.ToolResponse(id="f1", response={})),
            types.Part(tool_call=types.ToolCall(id="n1", args={})),
            types.Part(tool_response=types.ToolResponse(id="n1", response={})),
            types.Part(text="Sunny."),
        ]
        grounding = types.GroundingMetadata(web_search_queries=["x"])
        return LlmResponse(
            content=types.Content(role="model", parts=parts),
            grounding_metadata=grounding,
            finish_reason=types.FinishReason.STOP,
        )

    model = responding_llm(respond, model="gemini-2.5-flash")
    adk_events = await run_recorded(LlmAgent(name="searcher", model=model, tools=[google_search, url_context]))

    assert final_text(adk_events) == "Sunny."
    assert_balanced(subscribed_events)
    recorded = [
        (start.name, present(start.category_profile)["tool_call_id"], metadata_of(start).get("adk_tool_type"))
        for start in starts(subscribed_events, "tool")
    ]
    assert recorded == [
        ("url_context", "u1", "UrlContextTool"),
        ("google_search", "s1", "GoogleSearchTool"),
        ("file_search", "f1", None),
        ("tool_type_unspecified", "n1", None),
    ]
    search_start, search_end = starts(subscribed_events, "tool")[1], ends(subscribed_events, "tool")[1]
    assert data_of(search_start) == {"args": {"q": "x"}, "id": "s1", "toolType": "GOOGLE_SEARCH_WEB"}
    assert json.loads(str(data_of(search_end)["content"])) == {"id": "s1", "response": {"results": ["Sunny"]}}


async def test_a_provider_tool_result_that_cannot_be_recorded_opens_no_tool_scope(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent

    from nemo_relay.integrations.google_adk import _server_tools

    def unrecordable(value: object) -> nemo_relay.JsonObject:
        raise ValueError("no record")

    monkeypatch.setattr(_server_tools, "tool_result_json", unrecordable)
    tool, grounding = _grounding_tool("google_search")
    model = responding_llm(_grounded(grounding), model="gemini-2.5-flash")
    adk_events = await run_recorded(LlmAgent(name="searcher", model=model, tools=[tool]))

    assert final_text(adk_events) == "Sunny."
    assert starts(subscribed_events, "tool") == []
    assert_balanced(subscribed_events)


def _coding_agent(code: str = 'print("hi")') -> LlmAgent:
    from google.adk.agents import LlmAgent
    from google.adk.code_executors import UnsafeLocalCodeExecutor
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    def respond(request: LlmRequest) -> LlmResponse:
        parts = [part for content in request.contents for part in content.parts or []]
        ran = any("tool_output" in (part.text or "") for part in parts)
        text = "Done." if ran else f"Running it:\n```python\n{code}\n```"
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason=types.FinishReason.STOP
        )

    return LlmAgent(
        name="coder", model=responding_llm(respond, model="coding-model"), code_executor=UnsafeLocalCodeExecutor()
    )


def _execution_outputs(adk_events: list[Event]) -> list[str | None]:
    return [
        part.code_execution_result.output
        for event in adk_events
        if event.content
        for part in event.content.parts or []
        if part.code_execution_result
    ]


async def test_local_code_executor_runs_its_code_through_relay(subscribed_events: list[nemo_relay.Event]) -> None:
    def shout(name: str, args: nemo_relay.Json) -> nemo_relay.Json:
        if name == "code_execution" and isinstance(args, dict):
            return {**args, "code": 'print("HI")'}
        return args

    with tool_request_intercept(
        shout,
    ):
        adk_events = await run_recorded(_coding_agent())

    assert final_text(adk_events) == "Done."
    assert_balanced(subscribed_events)
    (tool,) = starts(subscribed_events, "tool")
    assert tool.name == "code_execution"
    assert tool.parent_uuid == named(subscribed_events, "coder", "agent").uuid
    assert data_of(tool)["input_files"] == []
    assert metadata_of(tool)["adk_code_executor"] == "UnsafeLocalCodeExecutor"
    (tool_end,) = ends(subscribed_events, "tool")
    assert data_of(tool_end)["content"] == "HI\n"
    assert data_of(tool_end)["is_error"] is False
    assert data_of(tool_end)["result"]["type"] == "google.adk.code_executors.code_execution_utils.CodeExecutionResult"
    assert data_of(tool_end)["result"]["value"]["exit_code"] == {"kind": "json", "value": 0}
    assert status_of(tool_end) == "OK"
    assert _execution_outputs(adk_events) == ["Code execution result:\nHI\n\n"]


async def test_local_code_executor_stderr_is_a_failed_tool_call(subscribed_events: list[nemo_relay.Event]) -> None:
    adk_events = await run_recorded(_coding_agent('raise ValueError("boom")'))

    assert_balanced(subscribed_events)
    tool_end = ends(subscribed_events, "tool")[0]
    content = as_instance(data_of(tool_end)["content"], str)
    assert content.rstrip().endswith("ValueError: boom")
    assert data_of(tool_end)["is_error"] is True
    assert status_of(tool_end) == "ERROR"
    assert _execution_outputs(adk_events)[0] == content


async def test_code_execution_intercept_edits_the_result_adk_receives(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    async def edit_content(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        result = as_object((await next_call(context.args)).result)
        return nemo_relay.ToolExecutionInterceptOutcome({**result, "content": "edited\n"})

    with tool_execution_intercept(edit_content):
        adk_events = await run_recorded(_coding_agent())

    assert _execution_outputs(adk_events) == ["Code execution result:\nedited\n\n"]
    assert_balanced(subscribed_events)


async def test_code_execution_intercept_can_fail_the_code_execution(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    async def deny(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        result = as_object((await next_call(context.args)).result)
        return nemo_relay.ToolExecutionInterceptOutcome({**result, "content": "denied", "is_error": True})

    with tool_execution_intercept(deny):
        adk_events = await run_recorded(_coding_agent())

    assert _execution_outputs(adk_events)[0] == "denied"
    tool_end = ends(subscribed_events, "tool")[0]
    assert status_of(tool_end) == "ERROR"
    assert_balanced(subscribed_events)


async def test_code_execution_intercept_that_answers_skips_running_the_code(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    calls: list[nemo_relay.Json] = []

    from google.adk.code_executors.code_execution_utils import CodeExecutionResult

    from nemo_relay.integrations.google_adk._tool_codec import tool_result_json

    cached = tool_result_json(CodeExecutionResult(stdout="cached\n"))

    async def answer(
        context: nemo_relay.ToolExecutionContext, next_call: ToolNext
    ) -> nemo_relay.ToolExecutionInterceptOutcome:
        calls.append(context.args)
        return nemo_relay.ToolExecutionInterceptOutcome(cached)

    with tool_execution_intercept(answer):
        adk_events = await run_recorded(_coding_agent())

    assert len(calls) == 1
    assert _execution_outputs(adk_events) == ["Code execution result:\ncached\n\n"]
    (tool_end,) = ends(subscribed_events, "tool")
    assert tool_end.data == cached
    assert_balanced(subscribed_events)


async def _send_code_as_text(
    context: nemo_relay.ToolExecutionContext, next_call: ToolNext
) -> nemo_relay.ToolExecutionInterceptOutcome:
    return nemo_relay.ToolExecutionInterceptOutcome((await next_call('print("hi")')).result)


async def _add_input_file(
    context: nemo_relay.ToolExecutionContext, next_call: ToolNext
) -> nemo_relay.ToolExecutionInterceptOutcome:
    edited = {**as_object(context.args), "input_files": [{"name": "data.csv", "content": "a,b"}]}
    return nemo_relay.ToolExecutionInterceptOutcome((await next_call(edited)).result)


async def _answer_with_text(
    context: nemo_relay.ToolExecutionContext, next_call: ToolNext
) -> nemo_relay.ToolExecutionInterceptOutcome:
    await next_call(context.args)
    return nemo_relay.ToolExecutionInterceptOutcome("hi")


async def _replace_the_encoded_result(
    context: nemo_relay.ToolExecutionContext, next_call: ToolNext
) -> nemo_relay.ToolExecutionInterceptOutcome:
    result = as_object((await next_call(context.args)).result)
    return nemo_relay.ToolExecutionInterceptOutcome({**result, "result": {"stdout": "edited\n"}})


async def _fail_without_text(
    context: nemo_relay.ToolExecutionContext, next_call: ToolNext
) -> nemo_relay.ToolExecutionInterceptOutcome:
    result = as_object((await next_call(context.args)).result)
    return nemo_relay.ToolExecutionInterceptOutcome({**result, "content": "", "is_error": True})


async def _answer_with_an_object(
    context: nemo_relay.ToolExecutionContext, next_call: ToolNext
) -> nemo_relay.ToolExecutionInterceptOutcome:
    await next_call(context.args)
    return nemo_relay.ToolExecutionInterceptOutcome(
        {
            "schema": "nemo.relay.tool_result@1",
            "content": "1",
            "is_error": False,
            "result": {"kind": "json", "value": 1},
        }
    )


@pytest.mark.parametrize(
    ("intercept", "error", "message", "result_rejected"),
    [
        pytest.param(_send_code_as_text, TypeError, "only the code", False, id="code_input_not_an_object"),
        pytest.param(_add_input_file, ValueError, "only the code", False, id="code_input_field_other_than_code"),
        pytest.param(_answer_with_text, ValueError, "versioned Relay tool result", True, id="result_not_a_tool_result"),
        pytest.param(
            _replace_the_encoded_result, ValueError, "encoded ADK tool result value", True, id="result_not_encoded"
        ),
        pytest.param(_fail_without_text, ValueError, "needs error text", True, id="failure_without_text"),
        pytest.param(_answer_with_an_object, TypeError, "expects a CodeExecutionResult", True, id="result_not_code"),
    ],
)
async def test_code_execution_intercept_edits_adk_cannot_hold_fail_the_run(
    intercept: Callable[
        [nemo_relay.ToolExecutionContext, ToolNext], Awaitable[nemo_relay.ToolExecutionInterceptOutcome]
    ],
    error: type[Exception],
    message: str,
    result_rejected: bool,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    with (
        tool_execution_intercept(intercept),
        pytest.raises(error, match=message) as raised,
    ):
        await run_recorded(_coding_agent())

    (tool_end,) = ends(subscribed_events, "tool")
    rejections = marks(subscribed_events, "adk.tool_result_rejected")
    if result_rejected:
        (rejected,) = rejections
        assert rejected.parent_uuid == tool_end.parent_uuid
        assert data_of(rejected) == {
            "tool": "code_execution",
            "tool_call_id": None,
            "error_type": error.__name__,
            "error_message": str(raised.value),
        }
    else:
        assert rejections == []
        assert status_of(tool_end) == "ERROR"
    assert_balanced(subscribed_events)
