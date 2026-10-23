# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Google ADK serialization codec and payloads."""

from __future__ import annotations

import dataclasses
import datetime
import json
from typing import TYPE_CHECKING, Final

import pydantic
import pytest
from _helpers import as_instance, as_list, as_object, get_weather, present

pytest.importorskip(
    "google.adk",
    reason="google-adk must be installed to run Google ADK based tests",
    exc_type=ModuleNotFoundError,
)

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import FunctionTool
from google.genai import types

import nemo_relay
from nemo_relay.integrations.google_adk._backends._anthropic import ANTHROPIC_CODEC, CACHE_WRITE_KEY
from nemo_relay.integrations.google_adk._backends._chat_completions import CHAT_CODEC
from nemo_relay.integrations.google_adk._backends._gemini import GEMINI_CODEC, URL_CONTEXT_KEY
from nemo_relay.integrations.google_adk._backends._openai import RESPONSES_CODEC
from nemo_relay.integrations.google_adk._serialization import (
    RECORDED_ID_KEY,
    RECORDED_KEY,
    RECORDED_USAGE_KEY,
    AdkCodec,
    ProducedResponses,
    aggregate_partials,
    final_response,
    header_text,
    http_headers,
    json_safe,
    request_payload,
    response_from_payload,
    response_payload,
    response_schema_payload,
    write_back_request,
    write_headers,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class Forecast(pydantic.BaseModel):
    forecast: str


def _calls(*locations: str) -> list[types.Part]:
    return [
        types.Part(function_call=types.FunctionCall(name="get_weather", args={"location": location}))
        for location in locations
    ]


def _responses(*locations: str) -> list[types.Part]:
    return [
        types.Part(function_response=types.FunctionResponse(name="get_weather", response={"location": location}))
        for location in locations
    ]


def _request() -> LlmRequest:
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="hi")]),
            types.Content(role="model", parts=[types.Part(text="Checking."), *_calls("Paris", "Rome")]),
            types.Content(role="user", parts=_responses("Paris", "Rome")),
        ],
        config=types.GenerateContentConfig(
            system_instruction="Be brief.",
            temperature=0.2,
            top_k=5,
            max_output_tokens=50,
            stop_sequences=["END"],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=["get_weather"]
                )
            ),
            http_options=types.HttpOptions(timeout=5),
            response_schema=Forecast,
            response_mime_type="application/json",
        ),
    )
    request.append_tools([FunctionTool(get_weather)])
    return request


def _raw(request: LlmRequest) -> nemo_relay.LLMRequest:
    return nemo_relay.LLMRequest({}, request_payload(request))


def test_request_payload_writes_a_value_with_no_json_form_as_its_stand_in() -> None:
    request = LlmRequest(model="gemini-2.5-flash", config=types.GenerateContentConfig(tools=[get_weather]))

    assert as_object(request_payload(request)["config"])["tools"] == ["<unserializable builtins.function>"]


def _counting() -> Iterator[int]:
    yield 0
    yield 1
    yield 2


def _with_items(items: object) -> LlmRequest:
    response = types.FunctionResponse(id="call-1", name="items", response={"items": items, "count": 3})
    return LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="list them")]),
            types.Content(
                role="model", parts=[types.Part(function_call=types.FunctionCall(id="call-1", name="items"))]
            ),
            types.Content(role="user", parts=[types.Part(function_response=response)]),
        ],
    )


def test_request_payload_stands_in_for_an_iterator_without_consuming_it() -> None:
    generator = _counting()
    request = _with_items(generator)

    payload = request_payload(request)

    contents = payload["contents"]
    assert isinstance(contents, list)
    response = as_object(
        as_object(as_object(as_list(as_object(contents[2])["parts"])[0])["function_response"])["response"]
    )
    assert response == {"items": "<unserializable iterator>", "count": 3}
    assert [*generator] == [0, 1, 2]


def test_request_payload_writes_bytes_as_genai_sends_them_and_sets_in_one_order() -> None:
    forward, backward = {1, 9}, {9, 1}
    assert list(forward) != list(backward)

    payloads = [request_payload(_with_items({"raw": b"\x89PNG", "ids": ids})) for ids in (forward, backward)]

    assert payloads[0] == payloads[1]
    contents = payloads[0]["contents"]
    assert isinstance(contents, list)
    response = as_object(
        as_object(as_object(as_list(as_object(contents[2])["parts"])[0])["function_response"])["response"]
    )
    assert response["items"] == {"raw": "iVBORw==", "ids": [1, 9]}


def test_write_back_keeps_a_function_responses_own_values_when_an_edit_changes_another() -> None:
    ids, raw = {1, 9}, b"\x89PNG"
    request = _with_items({"raw": raw, "ids": ids})
    raw_request = _raw(request)
    annotated = GEMINI_CODEC.decode(raw_request)
    messages = annotated.messages
    tool_message = messages[2]
    content = as_object(json.loads(str(tool_message["content"])))
    messages[2] = {**tool_message, "content": json.dumps({**content, "count": 4})}
    annotated.messages = messages

    payload = GEMINI_CODEC.encode(annotated, raw_request).content
    assert write_back_request(request, payload=payload, original_payload=raw_request.content)

    response = present(present(present(request.contents[2].parts)[0].function_response).response)
    assert response["count"] == 4
    items = response["items"]
    assert isinstance(items, dict)
    assert items["raw"] is raw
    assert items["ids"] is ids


def test_write_back_keeps_adks_own_objects_for_every_part_an_edit_left_unchanged() -> None:
    generator = _counting()
    request = _with_items(generator)
    originals = list(request.contents)
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    messages = annotated.messages
    messages[0] = {**messages[0], "content": "list them all"}
    annotated.messages = messages

    assert write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )

    assert present(present(request.contents[0].parts)[0].text) == "list them all"
    assert request.contents[1] is originals[1]
    assert request.contents[2] is originals[2]
    assert [*generator] == [0, 1, 2]


def test_an_edited_response_keeps_the_parts_the_edit_left_unchanged() -> None:
    marker, ids, raw = object(), {1, 9}, b"\x89PNG"
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="hello")]),
        custom_metadata={"trace": marker, "ids": ids, "raw": raw},
    )
    url_context = types.UrlContextMetadata(url_metadata=[types.UrlMetadata(retrieved_url="https://weather.example")])
    produced = ProducedResponses()
    payload = produced.add(
        response,
        recorded={URL_CONTEXT_KEY: url_context.model_dump(mode="json", exclude_none=True), RECORDED_ID_KEY: "resp-1"},
        response_model=None,
    )
    assert payload["custom_metadata"] == {"trace": "<unserializable builtins.object>", "ids": [1, 9], "raw": "iVBORw=="}
    edited: nemo_relay.JsonObject = {
        **payload,
        "content": {"role": "model", "parts": [{"text": "edited"}]},
        RECORDED_KEY: {URL_CONTEXT_KEY: {"url_metadata": []}},
    }

    rebuilt = produced.restore_result(json.loads(json.dumps(edited)))

    assert rebuilt.content == types.Content(role="model", parts=[types.Part(text="edited")])
    metadata = present(rebuilt.custom_metadata)
    assert metadata["trace"] is marker
    assert metadata["ids"] is ids
    assert metadata["raw"] is raw


def test_request_payload_keeps_transport_fields_off_the_payload() -> None:
    payload = request_payload(_request())

    config = as_object(payload["config"])
    assert "http_options" not in config
    assert config["response_schema"] == Forecast.model_json_schema()
    assert payload["model"] == "gemini-2.5-flash"
    assert "tools_dict" not in payload


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param(Forecast, Forecast.model_json_schema(), id="model"),
        pytest.param(list[Forecast], pydantic.TypeAdapter(list[Forecast]).json_schema(), id="list_of_models"),
        pytest.param(
            {"type": "object", "properties": {"forecast": {"type": "string"}}},
            {"type": "object", "properties": {"forecast": {"type": "string"}}},
            id="dict",
        ),
        pytest.param(
            types.Schema(type=types.Type.OBJECT, properties={"forecast": types.Schema(type=types.Type.STRING)}),
            {"type": "OBJECT", "properties": {"forecast": {"type": "STRING"}}},
            id="genai_schema",
        ),
    ],
)
def test_response_schema_payload_renders_every_schema_form_adk_accepts(
    schema: types.SchemaUnion | None, expected: nemo_relay.Json
) -> None:
    assert response_schema_payload(schema) == expected


def test_the_pipeline_headers_replace_those_on_adk_http_options_and_keep_the_other_options() -> None:
    request = _request()
    own = types.HttpOptions(timeout=5, headers={"x-existing": "kept", "X-Shared": "adk"})
    request.config.http_options = own
    assert http_headers(request) == {"x-existing": "kept", "X-Shared": "adk"}

    write_headers(request, http_headers(request))
    assert request.config.http_options is own

    write_headers(request, header_text({"x-shared": "relay", "x-dynamo-lineage": ["a", "b"]}))
    options = present(request.config.http_options)
    assert options.timeout == 5
    assert options.headers == {"x-shared": "relay", "x-dynamo-lineage": '["a","b"]'}
    assert own.headers == {"x-existing": "kept", "X-Shared": "adk"}

    bare = LlmRequest(model="gemini-2.5-flash")
    assert http_headers(bare) == {}
    write_headers(bare, {})
    assert bare.config.http_options is None
    write_headers(bare, {"x-relay": "value"})
    assert present(bare.config.http_options).headers == {"x-relay": "value"}


def test_the_extra_request_body_is_projected_and_written_back_keeping_the_other_http_options() -> None:
    request = _request()
    raw = {"n": 1}
    own = types.HttpOptions(timeout=5, headers={"x-existing": "kept"}, extra_body={"labels": {"team": "a"}, "raw": raw})
    request.config.http_options = own
    original = request_payload(request)
    assert as_object(original["config"])["http_options"] == {"extra_body": {"labels": {"team": "a"}, "raw": {"n": 1}}}
    assert "headers" not in json.dumps(original)

    edited = json.loads(json.dumps(original))
    edited["config"]["http_options"]["extra_body"]["labels"] = {"team": "b"}
    assert write_back_request(request, payload=edited, original_payload=original)
    options = present(request.config.http_options)
    assert options.extra_body == {"labels": {"team": "b"}, "raw": raw}
    assert present(options.extra_body)["raw"] is raw
    assert (options.timeout, options.headers) == (5, {"x-existing": "kept"})

    removed = json.loads(json.dumps(edited))
    del removed["config"]["http_options"]
    assert write_back_request(request, payload=removed, original_payload=edited)
    options = present(request.config.http_options)
    assert (options.extra_body, options.timeout, options.headers) == (None, 5, {"x-existing": "kept"})

    bare = LlmRequest(model="gemini-2.5-flash")
    projected = request_payload(bare)
    added = {**projected, "config": {**as_object(projected["config"]), "http_options": {"extra_body": {"x": 1}}}}
    assert write_back_request(bare, payload=added, original_payload=projected)
    assert present(bare.config.http_options).extra_body == {"x": 1}


@pytest.mark.parametrize(
    ("http_options", "error", "message"),
    [
        pytest.param({"headers": {"x-relay": "1"}}, ValueError, "only extra_body", id="headers"),
        pytest.param({"extra_body": {}, "timeout": 5}, ValueError, "only extra_body", id="timeout"),
        pytest.param({"extra_body": [1]}, TypeError, "extra_body must be a JSON object", id="extra_body_list"),
    ],
)
def test_http_options_writes_beyond_the_extra_request_body_fail(
    http_options: nemo_relay.JsonObject, error: type[Exception], message: str
) -> None:
    request = _request()
    original = request_payload(request)
    edited = {**original, "config": {**as_object(original["config"]), "http_options": http_options}}

    with pytest.raises(error, match=message):
        write_back_request(request, payload=edited, original_payload=original)


def test_write_back_applies_an_edited_response_schema_and_keeps_an_unchanged_one() -> None:
    request = _request()
    original = request_payload(request)
    edited = json.loads(json.dumps(original))
    edited["config"]["response_schema"]["description"] = "edited by an intercept"
    assert write_back_request(request, payload=edited, original_payload=original)
    assert request.config.response_schema == {**Forecast.model_json_schema(), "description": "edited by an intercept"}
    assert request.config.response_mime_type == "application/json"

    request = _request()
    untouched = json.loads(json.dumps(original))
    untouched["config"]["temperature"] = 0.9
    assert write_back_request(request, payload=untouched, original_payload=original)
    assert request.config.response_schema is Forecast

    request = _request()
    removed = json.loads(json.dumps(original))
    del removed["config"]["response_schema"]
    assert write_back_request(request, payload=removed, original_payload=original)
    assert request.config.response_schema is None


def test_write_back_applies_an_edit_between_equal_numbers_and_booleans() -> None:
    call = types.FunctionCall(id="c1", name="count", args={"n": 1})
    request = LlmRequest(
        model="gemini-2.5-flash", contents=[types.Content(role="model", parts=[types.Part(function_call=call)])]
    )
    original = request_payload(request)
    edited = json.loads(json.dumps(original))
    edited["contents"][0]["parts"][0]["function_call"]["args"]["n"] = True

    assert write_back_request(request, payload=edited, original_payload=original)
    written = present(present(present(request.contents[0].parts)[0].function_call).args)
    assert json.dumps(written, sort_keys=True) == json.dumps({"n": True}, sort_keys=True)


def test_encode_writes_back_a_config_edit_between_equal_numbers_and_booleans() -> None:
    config = types.GenerateContentConfig(response_json_schema={"type": "integer", "default": 1})
    request = LlmRequest(model="gemini-2.5-flash", config=config)
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    api_specific = as_object(annotated.api_specific)
    data = as_object(api_specific["data"])
    edited_config = {**as_object(data["config"]), "response_json_schema": {"type": "integer", "default": True}}
    annotated.api_specific = {**api_specific, "data": {**data, "config": edited_config}}

    encoded = GEMINI_CODEC.encode(annotated, raw).content
    assert write_back_request(request, payload=encoded, original_payload=raw.content)
    assert json.dumps(request.config.response_json_schema, sort_keys=True) == json.dumps(
        {"type": "integer", "default": True}, sort_keys=True
    )


def test_encode_writes_back_a_tool_edit_between_equal_numbers_and_booleans() -> None:
    def schema(*, default: int | bool) -> nemo_relay.JsonObject:
        return {"type": "object", "properties": {"n": {"type": "integer", "default": default}}}

    declaration = types.FunctionDeclaration(name="count", parameters_json_schema=schema(default=1))
    tools: list[types.Tool | Callable[..., object]] = [types.Tool(function_declarations=[declaration])]
    request = LlmRequest(model="gemini-2.5-flash", config=types.GenerateContentConfig(tools=tools))
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    (function,) = annotated.tools or []
    annotated.tools = [
        {**function, "function": {**as_object(function["function"]), "parameters": schema(default=True)}}
    ]

    encoded = GEMINI_CODEC.encode(annotated, raw).content
    assert write_back_request(request, payload=encoded, original_payload=raw.content)
    (tool,) = request.config.tools or []
    (written,) = present(as_instance(tool, types.Tool).function_declarations)
    assert json.dumps(written.parameters_json_schema, sort_keys=True) == json.dumps(
        schema(default=True), sort_keys=True
    )


def test_decode_maps_messages_instructions_params_tools_and_tool_choice() -> None:
    annotated = GEMINI_CODEC.decode(_raw(_request()))

    assert annotated.model == "gemini-2.5-flash"
    assert annotated.instructions == "Be brief."
    assert annotated.params == {"temperature": 0.2, "max_tokens": 50, "stop": ["END"]}
    user, assistant, paris, rome = annotated.messages
    assert user == {"role": "user", "content": "hi"}
    assert assistant["content"] == "Checking."
    tool_calls = as_list(assistant["tool_calls"])
    ids = [call["id"] for call in tool_calls]
    assert ids == ["get_weather#0", "get_weather#1"]
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"location": "Rome"}
    assert [message["tool_call_id"] for message in (paris, rome)] == ids
    assert paris["content"] == '{"location":"Paris"}'
    (tool,) = annotated.tools or []
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["parameters"]["properties"]["location"]["type"] == "string"
    assert annotated.tool_choice == {"type": "function", "function": {"name": "get_weather"}}
    api_specific = present(annotated.api_specific)
    assert api_specific["api"] == "custom"
    assert api_specific["api_name"] == "google_adk"
    data = as_object(api_specific["data"])
    assert set(data) == {"config", "live_connect_config"}
    assert data["config"] == {
        "top_k": 5,
        "response_mime_type": "application/json",
        "response_schema": Forecast.model_json_schema(),
    }
    assert annotated.extra == {}


def test_decode_keeps_a_single_id_less_call_on_its_name_and_provider_ids_unchanged() -> None:
    parts = [*_calls("Paris"), types.Part(function_call=types.FunctionCall(id="call-1", name="get_time", args={}))]

    (assistant,) = GEMINI_CODEC.decode(_raw(LlmRequest(contents=[types.Content(role="model", parts=parts)]))).messages

    tool_calls = as_list(assistant["tool_calls"])
    assert [call["id"] for call in tool_calls] == ["get_weather", "call-1"]


def test_decode_keeps_thoughts_and_binary_parts_as_provider_native() -> None:
    content = types.Content(
        role="model",
        parts=[
            types.Part(text="why", thought=True),
            types.Part(inline_data=types.Blob(mime_type="image/png", data=b"\x89PNG")),
            types.Part(text="done"),
        ],
    )

    (message,) = GEMINI_CODEC.decode(_raw(LlmRequest(contents=[content]))).messages

    thought, image, text = as_list(message["content"])
    assert thought["type"] == "provider_native"
    assert thought["kind"] == "thought"
    assert thought["value"]["thought"] is True
    assert image["type"] == "provider_native"
    assert image["kind"] == "inline_data"
    assert text == {"type": "text", "text": "done"}


def test_encode_writes_back_function_tool_edits_and_keeps_native_tools() -> None:
    request = _request()
    request.config.tools = [*(request.config.tools or []), types.Tool(google_search=types.GoogleSearch())]
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    function, native = annotated.tools or []
    assert native == {
        "type": "provider_native",
        "provider": "google_adk",
        "kind": "tool",
        "value": {"google_search": {}},
    }
    definition = as_object(function["function"])
    parameters = as_object(definition["parameters"])
    location = as_object(parameters["properties"]["location"])
    edited_parameters: nemo_relay.JsonObject = {
        **parameters,
        "properties": {"location": {**location, "description": "A city name."}},
    }
    edited_definition: nemo_relay.JsonObject = {
        **definition,
        "description": "Fetch the forecast.",
        "parameters": edited_parameters,
    }
    annotated.tools = [{**function, "function": edited_definition}, native]

    encoded = GEMINI_CODEC.encode(annotated, raw).content
    assert write_back_request(request, payload=encoded, original_payload=raw.content)

    weather, search = request.config.tools or []
    (declaration,) = present(as_instance(weather, types.Tool).function_declarations)
    assert declaration.name == "get_weather"
    assert declaration.description == "Fetch the forecast."
    assert declaration.parameters_json_schema["properties"]["location"]["description"] == "A city name."
    assert as_instance(search, types.Tool).google_search is not None
    assert "get_weather" in request.tools_dict


def _renamed(tool: nemo_relay.JsonObject) -> nemo_relay.JsonObject:
    return {**tool, "function": {**as_object(tool["function"]), "name": "other"}}


def _calling_request(mode: str, names: list[str] | None = None) -> LlmRequest:
    calling = types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode(mode), allowed_function_names=names)
    return LlmRequest(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(tool_config=types.ToolConfig(function_calling_config=calling)),
    )


@pytest.mark.parametrize(
    ("mode", "names", "expected"),
    [
        pytest.param("NONE", None, "none", id="none"),
        pytest.param("AUTO", None, "auto", id="auto"),
        pytest.param("ANY", None, "required", id="required"),
        pytest.param(
            "ANY", ["get_weather"], {"type": "function", "function": {"name": "get_weather"}}, id="one_function"
        ),
        pytest.param(
            "ANY",
            ["get_weather", "get_time"],
            {
                "provider": "google_adk",
                "kind": "function_calling_config",
                "value": {"mode": "ANY", "allowed_function_names": ["get_weather", "get_time"]},
            },
            id="several_functions",
        ),
        pytest.param(
            "VALIDATED",
            None,
            {"provider": "google_adk", "kind": "function_calling_config", "value": {"mode": "VALIDATED"}},
            id="validated",
        ),
    ],
)
def test_decode_maps_function_calling_config_to_tool_choice(
    mode: str, names: list[str] | None, expected: nemo_relay.Json
) -> None:
    assert GEMINI_CODEC.decode(_raw(_calling_request(mode, names))).tool_choice == expected


@pytest.mark.parametrize(
    ("choice", "mode", "names"),
    [
        pytest.param("none", "NONE", None, id="none"),
        pytest.param("required", "ANY", None, id="required"),
        pytest.param(
            {"type": "function", "function": {"name": "get_weather"}}, "ANY", ["get_weather"], id="one_function"
        ),
        pytest.param(
            {"provider": "google_adk", "kind": "function_calling_config", "value": {"mode": "VALIDATED"}},
            "VALIDATED",
            None,
            id="provider_native",
        ),
    ],
)
def test_encode_writes_a_tool_choice_back_as_function_calling_config(
    choice: str | nemo_relay.JsonObject, mode: str, names: list[str] | None
) -> None:
    request = _calling_request("AUTO")
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    annotated.tool_choice = choice

    assert write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )

    calling = present(present(request.config.tool_config).function_calling_config)
    assert calling.mode == types.FunctionCallingConfigMode(mode)
    assert calling.allowed_function_names == names


def test_encode_clears_a_removed_tool_choice() -> None:
    request = _calling_request("ANY", ["get_weather"])
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    annotated.tool_choice = None

    assert write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )

    assert request.config.tool_config is None


@pytest.mark.parametrize(
    "system_instruction",
    [
        pytest.param(
            types.Content(role="system", parts=[types.Part(text="Be brief."), types.Part(text="Use metric.")]),
            id="content",
        ),
        pytest.param(["Be brief.", "Use metric."], id="list"),
    ],
)
def test_instruction_parts_decode_to_blocks_and_an_edit_writes_back_as_content(
    system_instruction: types.ContentUnion,
) -> None:
    request = LlmRequest(
        model="gemini-2.5-flash", config=types.GenerateContentConfig(system_instruction=system_instruction)
    )
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)

    assert annotated.instructions == [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "Use metric."}]

    annotated.instructions = [{"type": "text", "text": "Be terse."}, {"type": "text", "text": "Use imperial."}]
    assert write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )

    assert request.config.system_instruction == types.Content(
        parts=[types.Part(text="Be terse."), types.Part(text="Use imperial.")]
    )


def test_a_part_instruction_decodes_as_text_and_an_edit_writes_back() -> None:
    request = LlmRequest(
        model="gemini-2.5-flash", config=types.GenerateContentConfig(system_instruction=types.Part(text="Be brief."))
    )
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)

    assert annotated.instructions == "Be brief."

    annotated.instructions = "Be terse."
    assert write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )

    assert request.config.system_instruction == "Be terse."


def test_a_file_instruction_decodes_as_a_native_block_and_round_trips() -> None:
    file = types.File(name="files/x", uri="gs://bucket/x", mime_type="text/plain")
    request = LlmRequest(model="gemini-2.5-flash", config=types.GenerateContentConfig(system_instruction=file))
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)

    (block,) = as_list(annotated.instructions)
    assert as_object(block)["type"] == "provider_native"
    assert as_object(block)["value"] == _dumped_model(file)
    assert not write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )


@pytest.mark.parametrize(
    "edit",
    [
        lambda tools: tools[:0],
        lambda tools: [*tools, tools[0]],
        lambda tools: [_renamed(tools[0])],
        lambda tools: [{"type": "provider_native", "provider": "google_adk", "kind": "tool", "value": {"x": {}}}],
    ],
    ids=["removed", "added", "renamed", "replaced"],
)
def test_encode_rejects_tool_edits_that_change_the_tool_set(
    edit: Callable[[list[nemo_relay.JsonObject]], list[nemo_relay.JsonObject]],
) -> None:
    raw = _raw(_request())
    annotated = GEMINI_CODEC.decode(raw)
    annotated.tools = edit(list(annotated.tools or []))

    with pytest.raises(ValueError, match="cannot add, remove, rename, or reorder"):
        GEMINI_CODEC.encode(annotated, raw)


def test_encode_returns_the_original_payload_when_nothing_changed() -> None:
    raw = _raw(_request())

    assert GEMINI_CODEC.encode(GEMINI_CODEC.decode(raw), raw).content == raw.content


def test_encode_writes_back_edits_and_keeps_derived_ids_off_the_request() -> None:
    request = _request()
    http_options = request.config.http_options
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    messages = annotated.messages
    messages[0] = {**messages[0], "content": "REWRITTEN"}
    messages.append({"role": "user", "content": "and Berlin?"})
    annotated.messages = messages
    annotated.instructions = "Be terse."
    annotated.model = "gemini-2.5-pro"
    annotated.params = {**(annotated.params or {}), "temperature": 0.9}
    annotated.tool_choice = "auto"

    encoded = GEMINI_CODEC.encode(annotated, raw).content
    assert write_back_request(request, payload=encoded, original_payload=raw.content)

    assert request.model == "gemini-2.5-pro"
    assert len(request.contents) == 4
    assert request.contents[0].parts[0].text == "REWRITTEN"
    assert request.contents[3].parts[0].text == "and Berlin?"
    calls = [part.function_call for part in request.contents[1].parts or [] if part.function_call]
    assert [(call.name, call.id, call.args) for call in calls] == [
        ("get_weather", None, {"location": "Paris"}),
        ("get_weather", None, {"location": "Rome"}),
    ]
    responses = [part.function_response for part in request.contents[2].parts or [] if part.function_response]
    assert [(response.name, response.id, response.response) for response in responses] == [
        ("get_weather", None, {"location": "Paris"}),
        ("get_weather", None, {"location": "Rome"}),
    ]
    assert request.config.system_instruction == "Be terse."
    assert request.config.temperature == 0.9
    assert request.config.top_k == 5
    calling = present(present(request.config.tool_config).function_calling_config)
    assert calling.mode == "AUTO"
    assert calling.allowed_function_names is None
    assert request.config.http_options is http_options
    assert request.config.response_schema is Forecast
    assert "get_weather" in request.tools_dict
    (tool,) = request.config.tools or []
    assert present(as_instance(tool, types.Tool).function_declarations)[0].name == "get_weather"


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        pytest.param(
            {"contents": ["hi"]}, TypeError, "ADK contents entries must be objects", id="content_not_an_object"
        ),
        pytest.param(
            {"contents": [{"role": "user", "parts": {}}]}, TypeError, "ADK content parts must be a list", id="parts"
        ),
        pytest.param(
            {"contents": [{"role": "model", "parts": [{"function_call": "x"}]}]},
            TypeError,
            "ADK function_call must be an object",
            id="call_not_an_object",
        ),
        pytest.param(
            {"contents": [{"role": "model", "parts": [{"function_call": {"args": {}}}]}]},
            ValueError,
            "ADK function must have a name",
            id="call_without_a_name",
        ),
        pytest.param(
            {"config": {"system_instruction": 1}}, TypeError, "system_instruction must be text", id="instruction"
        ),
        pytest.param({"config": {"tools": {}}}, TypeError, "ADK config tools must be a list", id="tools_not_a_list"),
        pytest.param(
            {"config": {"tool_config": {"function_calling_config": "ANY"}}},
            TypeError,
            "function_calling_config must be an object",
            id="calling_config",
        ),
        pytest.param({"model": 1}, TypeError, "ADK model must be text", id="model_not_text"),
        pytest.param(
            {"contents": [{"role": "user", "parts": []}]},
            ValueError,
            "user content must have parts",
            id="user_without_parts",
        ),
        pytest.param(
            {"contents": [{"role": "user", "parts": [{"function_response": {"name": "f"}}]}]},
            ValueError,
            "function_response must have a response",
            id="response_without_a_response",
        ),
    ],
)
def test_decode_rejects_request_fields_of_the_wrong_shape(
    payload: nemo_relay.JsonObject, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        GEMINI_CODEC.decode(nemo_relay.LLMRequest({}, payload))


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        pytest.param({"finish_reason": 1}, TypeError, "ADK finish_reason must be text", id="finish_reason"),
        pytest.param({"model_version": 1}, TypeError, "ADK model_version must be text", id="model_version"),
        pytest.param(
            {"usage_metadata": {"prompt_token_count": "10"}},
            TypeError,
            "prompt_token_count must be an integer",
            id="token_count",
        ),
        pytest.param({"avg_logprobs": "x"}, TypeError, "ADK avg_logprobs must be a number", id="avg_logprobs"),
        pytest.param({"content": {"parts": [{}]}}, ValueError, "An ADK part must have a field", id="empty_part"),
    ],
)
def test_decode_response_rejects_fields_of_the_wrong_shape(
    payload: nemo_relay.JsonObject, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        GEMINI_CODEC.decode_response(payload)


def test_encode_writes_an_emptied_turn_with_no_parts() -> None:
    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[1] = {"role": "assistant", "content": None}

    assert _edited(_history(), edit).contents[1].parts == []


def test_encode_rejects_content_it_cannot_write_back() -> None:
    raw = _raw(_request())
    annotated = GEMINI_CODEC.decode(raw)
    annotated.messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]

    with pytest.raises(ValueError, match="cannot write"):
        GEMINI_CODEC.encode(annotated, raw)


def test_write_back_is_a_no_op_for_an_unchanged_payload() -> None:
    request = _request()
    payload = request_payload(request)

    assert not write_back_request(request, payload=payload, original_payload=payload)


def test_decode_response_maps_message_tool_calls_finish_reason_usage_and_model() -> None:
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="hello"), *_calls("Rome")]),
        finish_reason=types.FinishReason.STOP,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            cached_content_token_count=2,
            thoughts_token_count=3,
            tool_use_prompt_token_count=4,
        ),
        citation_metadata=types.CitationMetadata(citations=[types.Citation(uri="https://weather.example")]),
        avg_logprobs=-0.5,
        model_version="gemini-2.5-flash-001",
        partial=False,
        error_code="UNAVAILABLE",
        error_message="try later",
    )

    annotated = GEMINI_CODEC.decode_response(response_payload(response))

    assert annotated.message == "hello"
    assert annotated.tool_calls == [{"id": "get_weather", "name": "get_weather", "arguments": {"location": "Rome"}}]
    assert annotated.finish_reason == "tool_use"
    assert annotated.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_tokens": 2,
        "uncached_input_tokens": 8,
    }
    assert annotated.model == "gemini-2.5-flash-001"
    assert annotated.api_specific == {
        "api": "gemini_generate_content",
        "thoughts_tokens": 3,
        "tool_use_prompt_tokens": 4,
        "citation_metadata": {"citations": [{"uri": "https://weather.example"}]},
        "avgLogprobs": -0.5,
    }
    assert annotated.extra == {"error_code": "UNAVAILABLE", "error_message": "try later", "partial": False}


def test_decode_response_writes_candidate_metadata_in_the_gemini_wire_shape() -> None:
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="hello")]),
        grounding_metadata=types.GroundingMetadata(
            web_search_queries=["weather in Rome"],
            grounding_chunks=[
                types.GroundingChunk(
                    retrieved_context=types.GroundingChunkRetrievedContext(file_search_store="fileSearchStores/1")
                )
            ],
        ),
        logprobs_result=types.LogprobsResult(
            chosen_candidates=[types.LogprobsResultCandidate(token="hello", log_probability=-0.1)]
        ),
    )
    payload = response_payload(response)
    payload[RECORDED_KEY] = {URL_CONTEXT_KEY: {"url_metadata": [{"retrieved_url": "https://weather.example"}]}}

    annotated = GEMINI_CODEC.decode_response(payload)

    assert annotated.api_specific == {
        "api": "gemini_generate_content",
        "grounding_metadata": {
            "webSearchQueries": ["weather in Rome"],
            "groundingChunks": [{"retrievedContext": {"fileSearchStore": "fileSearchStores/1"}}],
        },
        "logprobsResult": {"chosenCandidates": [{"token": "hello", "logProbability": -0.1}]},
        "urlContextMetadata": {"urlMetadata": [{"retrievedUrl": "https://weather.example"}]},
    }
    assert annotated.extra == {}


def test_url_context_rides_on_the_payload_and_is_dropped_from_a_rebuilt_response() -> None:
    produced = ProducedResponses()
    response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="hello")]))
    url_context = types.UrlContextMetadata(url_metadata=[types.UrlMetadata(retrieved_url="https://weather.example")])

    payload = produced.add(
        response,
        recorded={URL_CONTEXT_KEY: url_context.model_dump(mode="json", exclude_none=True), RECORDED_ID_KEY: "resp-1"},
        response_model=None,
    )

    assert payload[RECORDED_KEY] == {
        URL_CONTEXT_KEY: {"url_metadata": [{"retrieved_url": "https://weather.example"}]},
        RECORDED_ID_KEY: "resp-1",
    }
    aggregated = aggregate_partials([{**payload, "partial": True}])
    assert aggregated[RECORDED_KEY] == payload[RECORDED_KEY]
    assert produced.restore_result(payload) is response
    edited: nemo_relay.JsonObject = {**payload, "content": {"role": "model", "parts": [{"text": "edited"}]}}
    assert response_from_payload(edited).content == types.Content(role="model", parts=[types.Part(text="edited")])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STOP", "complete"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("LANGUAGE", "content_filter"),
        ("FINISH_REASON_UNSPECIFIED", None),
        ("OTHER", "OTHER"),
    ],
)
def test_decode_response_maps_finish_reasons(raw: str, expected: str | None) -> None:
    annotated = GEMINI_CODEC.decode_response(
        {"content": {"role": "model", "parts": [{"text": "x"}]}, "finish_reason": raw}
    )

    assert annotated.finish_reason == expected


def test_decode_response_maps_an_unspecified_finish_reason_with_calls_to_tool_use() -> None:
    parts = [{"function_call": {"name": "get_weather", "args": {}}}]
    annotated = GEMINI_CODEC.decode_response(
        {"content": {"role": "model", "parts": parts}, "finish_reason": "FINISH_REASON_UNSPECIFIED"}
    )

    assert annotated.finish_reason == "tool_use"


def test_response_from_payload_validates_a_dump_and_rejects_anything_else() -> None:
    rebuilt = response_from_payload({"content": {"role": "model", "parts": [{"text": "edited"}]}, "partial": False})
    assert present(present(rebuilt.content).parts)[0].text == "edited"
    assert rebuilt.partial is False
    with pytest.raises(TypeError):
        response_from_payload("nope")


def test_json_safe_counts_binary_anywhere_and_serializes_everything_else() -> None:
    blob = types.Blob(mime_type="image/png", data=b"\x89PNG" * 25, display_name="pic")
    counted: nemo_relay.JsonObject = {
        "inlineData": {"mimeType": "image/png", "displayName": "pic", "data": {"bytes": 100}}
    }
    content = types.Content(role="user", parts=[types.Part(text="look"), types.Part(inline_data=blob)])
    nested = types.Content(
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="f",
                    response={},
                    parts=[
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(mime_type="image/png", data=b"\x00" * 3)
                        )
                    ],
                )
            )
        ]
    )

    assert json_safe(content) == {"role": "user", "parts": [{"text": "look"}, counted]}
    assert json_safe(types.Part(inline_data=blob)) == counted
    assert json_safe([content])[0] == json_safe(content)
    assert json_safe(nested) == {
        "parts": [
            {
                "functionResponse": {
                    "name": "f",
                    "response": {},
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": {"bytes": 3}}}],
                }
            }
        ]
    }

    class Opaque:
        pass

    class Holder(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
        thing: Opaque
        note: str | None = None

    class Stamped(pydantic.BaseModel):
        when: datetime.datetime
        secret: bytes

        @pydantic.field_serializer("when")
        def _serialize_when(self, when: datetime.datetime) -> str:
            return f"at {when.year}"

    @dataclasses.dataclass
    class Packet:
        payload: bytes
        note: str | None = None

    opaque = Opaque()
    unserializable = f"<unserializable {Opaque.__module__}.{Opaque.__qualname__}>"
    assert json_safe(
        {"when": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC), "raw": b"\xff", "obj": opaque, 3: None, "set": {1}}
    ) == {"when": "2026-01-01T00:00:00Z", "raw": {"bytes": 1}, "obj": unserializable, "3": None, "set": [1]}
    assert json_safe(Holder(thing=opaque)) == {"thing": unserializable}
    assert json_safe(Stamped(when=datetime.datetime(2026, 1, 1), secret=b"\x01\x02")) == {
        "when": "at 2026",
        "secret": {"bytes": 2},
    }
    assert json_safe(Packet(b"\x00\x00")) == {"payload": {"bytes": 2}, "note": None}


def test_json_safe_writes_equal_sets_alike_whatever_order_they_iterate_in() -> None:
    forward, backward = {1, 9}, {9, 1}
    assert list(forward) != list(backward)

    assert json_safe({"ids": forward}) == json_safe({"ids": backward}) == {"ids": [1, 9]}
    assert json_safe({frozenset({3, 2}), frozenset({1})}) == [[1], [2, 3]]
    assert json_safe(frozenset({(2, b"xy"), (1, b"x")})) == [[1, {"bytes": 1}], [2, {"bytes": 2}]]


def test_json_safe_writes_an_iterator_as_its_stand_in_without_consuming_it() -> None:
    def counting() -> Iterator[int]:
        yield 1
        yield 2

    generator = counting()

    assert json_safe({"items": generator, "rest": [iter([3])]}) == {
        "items": "<unserializable iterator>",
        "rest": ["<unserializable iterator>"],
    }
    assert [*generator] == [1, 2]


def test_aggregate_partials_merges_streamed_call_arguments_into_one_call() -> None:
    chunks: list[nemo_relay.Json] = [
        {"content": {"role": "model", "parts": [{"text": "Checking "}]}, "partial": True},
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "thought_signature": "sig",
                        "function_call": {
                            "name": "get_weather",
                            "id": "adk-1",
                            "will_continue": True,
                            "partial_args": [{"json_path": "$.city", "string_value": "Pa", "will_continue": True}],
                        },
                    }
                ],
            },
            "partial": True,
        },
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "adk-1",
                            "will_continue": False,
                            "partial_args": [
                                {"json_path": "$.city", "string_value": "ris"},
                                {"json_path": "$.days", "number_value": 3},
                                {"json_path": "$.units['temp']", "string_value": "C"},
                                {"json_path": "$.tags[1]", "bool_value": True},
                                {"json_path": "$.note", "null_value": "NULL_VALUE"},
                            ],
                        }
                    }
                ],
            },
            "partial": True,
        },
        {"content": {"role": "model", "parts": [{"text": "now."}]}, "partial": True, "finish_reason": "STOP"},
    ]

    aggregate = aggregate_partials(chunks)

    args: nemo_relay.JsonObject = {
        "city": "Paris",
        "days": 3,
        "units": {"temp": "C"},
        "tags": [None, True],
        "note": None,
    }
    assert aggregate["content"] == {
        "role": "model",
        "parts": [
            {"text": "Checking "},
            {"thought_signature": "sig", "function_call": {"name": "get_weather", "id": "adk-1", "args": args}},
            {"text": "now."},
        ],
    }
    response = GEMINI_CODEC.decode_response(aggregate)
    assert response.tool_calls == [{"id": "adk-1", "name": "get_weather", "arguments": args}]
    assert response.finish_reason == "tool_use"
    assert present(response_from_payload(aggregate).content).parts is not None


def test_aggregate_partials_merges_fragments_that_spell_one_argument_path_differently() -> None:
    chunks: list[nemo_relay.Json] = [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "id": "adk-1",
                            "will_continue": True,
                            "partial_args": [{"json_path": "$.city", "string_value": "Pa"}],
                        }
                    }
                ],
            },
            "partial": True,
        },
        {
            "content": {
                "role": "model",
                "parts": [
                    {"function_call": {"id": "adk-1", "partial_args": [{"json_path": "city", "string_value": "ris"}]}}
                ],
            },
            "partial": True,
        },
    ]

    assert aggregate_partials(chunks)["content"] == {
        "role": "model",
        "parts": [{"function_call": {"name": "get_weather", "id": "adk-1", "args": {"city": "Paris"}}}],
    }


def test_aggregate_partials_keeps_whole_calls_and_drops_a_streamed_call_that_never_names_itself() -> None:
    chunks: list[nemo_relay.Json] = [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "first",
                            "will_continue": True,
                            "partial_args": [{"json_path": "$.a", "string_value": "x"}],
                        }
                    }
                ],
            },
            "partial": True,
        },
        {
            "content": {"role": "model", "parts": [{"function_call": {"name": "second", "args": {"b": 1}}}]},
            "partial": True,
        },
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "will_continue": True,
                            "partial_args": [{"json_path": "$.c", "string_value": "y"}],
                        }
                    }
                ],
            },
            "partial": True,
        },
        {"content": {"role": "model", "parts": [{"text": "done"}]}, "partial": True},
    ]

    assert aggregate_partials(chunks)["content"] == {
        "role": "model",
        "parts": [
            {"function_call": {"name": "first", "args": {"a": "x"}}},
            {"function_call": {"name": "second", "args": {"b": 1}}},
            {"text": "done"},
        ],
    }


def _call_chunk(function: nemo_relay.JsonObject) -> nemo_relay.Json:
    return {"content": {"role": "model", "parts": [{"function_call": function}]}, "partial": True}


def test_aggregate_partials_merges_fragment_args_and_clears_chunks_under_a_written_path() -> None:
    chunks = [
        _call_chunk(
            {
                "name": "f",
                "will_continue": True,
                "args": {"city": "Pa", "days": 2},
                "partial_args": [
                    {"json_path": "$.city", "string_value": "ris"},
                    {"json_path": "$.address.city", "string_value": "Paris"},
                    {"json_path": "$.tags[0].name", "string_value": "x"},
                    {"json_path": "$.tags[0]", "null_value": "NULL_VALUE"},
                ],
            }
        ),
        _call_chunk({"args": {"days": 3, "address": {"city": "London"}}}),
    ]

    assert aggregate_partials(chunks)["content"] == {
        "role": "model",
        "parts": [
            {
                "function_call": {
                    "name": "f",
                    "args": {"city": "Paris", "days": 3, "address": {"city": "London"}, "tags": [None]},
                }
            }
        ],
    }


def test_aggregate_partials_starts_a_new_call_when_a_fragment_carries_a_different_id() -> None:
    chunks = [
        _call_chunk(
            {
                "name": "get_weather",
                "id": "c1",
                "will_continue": True,
                "partial_args": [{"json_path": "$.city", "string_value": "Paris"}],
            }
        ),
        _call_chunk(
            {
                "name": "get_weather",
                "id": "c2",
                "will_continue": True,
                "partial_args": [{"json_path": "$.city", "string_value": "Rome"}],
            }
        ),
        _call_chunk({"will_continue": False, "partial_args": [{"json_path": "$.days", "number_value": 2}]}),
    ]

    assert aggregate_partials(chunks)["content"] == {
        "role": "model",
        "parts": [
            {"function_call": {"name": "get_weather", "id": "c1", "args": {"city": "Paris"}}},
            {"function_call": {"name": "get_weather", "id": "c2", "args": {"city": "Rome", "days": 2}}},
        ],
    }


def test_aggregate_partials_treats_empty_partial_args_as_a_whole_call_and_empty_args_as_a_fragment() -> None:
    chunks = [
        _call_chunk(
            {"name": "first", "will_continue": True, "partial_args": [{"json_path": "$.q", "string_value": "x"}]}
        ),
        _call_chunk({"name": "first", "args": {}}),
        _call_chunk({"name": "second", "args": {"k": 1}, "partial_args": []}),
    ]

    assert aggregate_partials(chunks)["content"] == {
        "role": "model",
        "parts": [
            {"function_call": {"name": "first", "args": {"q": "x"}}},
            {"function_call": {"name": "second", "args": {"k": 1}, "partial_args": []}},
        ],
    }


def test_aggregate_partials_reads_argument_paths_like_adk() -> None:
    chunk = _call_chunk(
        {
            "name": "paths",
            "partial_args": [
                {"json_path": "$['a.b']", "string_value": "dotted"},
                {"json_path": '$["q\\u0041"]', "number_value": 1},
                {"json_path": "$['\\u12']", "number_value": 2},
                {"json_path": "$.x\\.y", "bool_value": True},
                {"json_path": "$.list[ 1 ]", "string_value": "one"},
                {"json_path": "$.k\\", "string_value": "trail"},
                {"json_path": "$['\\uD83D\\uDE00']", "string_value": "pair"},
                {"json_path": "$['\\uD83Dx']", "string_value": "lone"},
            ],
        }
    )

    parts = as_list(as_object(aggregate_partials([chunk])["content"])["parts"])
    assert as_object(as_object(parts[0])["function_call"])["args"] == {
        "a.b": "dotted",
        "qA": 1,
        "u12": 2,
        "x.y": True,
        "list": [None, "one"],
        "k": "trail",
        "\U0001f600": "pair",
        "\ufffdx": "lone",
    }


@pytest.mark.parametrize(
    ("partial_args", "error", "message"),
    [
        pytest.param("nope", TypeError, "partial_args must be a list", id="not_a_list"),
        pytest.param(
            [{"json_path": "$.x", "null_value": "null"}],
            ValueError,
            "null_value must be NULL_VALUE",
            id="null_value",
        ),
        pytest.param(
            [{"json_path": "$", "number_value": 1}], ValueError, "does not address an argument", id="root_path"
        ),
        pytest.param(
            [{"json_path": "$.big[10001]", "string_value": "x"}], ValueError, "indexes past 10000", id="index_cap"
        ),
        pytest.param(
            [{"json_path": "$." + ".".join(["d"] * 129), "number_value": 1}],
            ValueError,
            "nests deeper than 128 levels",
            id="depth_cap",
        ),
    ],
)
def test_aggregate_partials_rejects_a_malformed_fragment(
    partial_args: nemo_relay.Json, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        aggregate_partials([_call_chunk({"name": "g", "partial_args": partial_args})])


def test_aggregate_partials_merges_text_runs_and_keeps_calls_and_the_last_metadata() -> None:
    chunks: list[nemo_relay.Json] = [
        {"content": {"role": "model", "parts": [{"text": "Thinking", "thought": True}]}, "partial": True},
        {"content": {"role": "model", "parts": [{"text": "Hel"}]}, "partial": True, "model_version": "v1"},
        {
            "content": {"role": "model", "parts": [{"text": "lo", "thought_signature": "c2ln"}, {"text": " there"}]},
            "partial": True,
        },
        {
            "content": {
                "role": "model",
                "parts": [{"function_call": {"name": "get_weather", "args": {"location": "Paris"}}}],
            },
            "partial": True,
        },
        {"content": {"role": "model", "parts": [{"text": "!"}]}, "finish_reason": "STOP", "partial": True},
        {"usage_metadata": {"total_token_count": 5}, "finish_reason": "STOP", "partial": True},
    ]

    assert aggregate_partials(chunks) == {
        "content": {
            "role": "model",
            "parts": [
                {"text": "Thinking", "thought": True},
                {"text": "Hello there", "thought_signature": "c2ln"},
                {"function_call": {"name": "get_weather", "args": {"location": "Paris"}}},
                {"text": "!"},
            ],
        },
        "model_version": "v1",
        "finish_reason": "STOP",
        "usage_metadata": {"total_token_count": 5},
        "partial": False,
    }
    assert aggregate_partials([]) == {"partial": False}
    assert aggregate_partials(chunks[1:3]) == {
        "content": {"role": "model", "parts": [{"text": "Hello there", "thought_signature": "c2ln"}]},
        "model_version": "v1",
        "partial": False,
    }
    with pytest.raises(TypeError, match="LlmResponse dumps"):
        aggregate_partials(["not a dump"])


def test_final_response_keeps_the_last_whole_chunk_and_the_fields_later_chunks_report() -> None:
    whole: nemo_relay.JsonObject = {
        "content": {"role": "model", "parts": [{"text": "Sunny."}]},
        "finish_reason": "STOP",
        "model_version": "v1",
        "partial": False,
    }
    chunks: list[nemo_relay.Json] = [
        {"content": {"role": "model", "parts": [{"text": "Sun"}]}, "partial": True},
        {"content": {"role": "model", "parts": [{"text": "ny."}]}, "partial": True},
        whole,
        {"usage_metadata": {"total_token_count": 5}, "model_version": "v2", "partial": True},
        {RECORDED_KEY: {RECORDED_USAGE_KEY: {"prompt_tokens": 5, "cost": 0.1}}, "partial": True},
    ]

    assert final_response(chunks) == {
        **whole,
        "usage_metadata": {"total_token_count": 5},
        "model_version": "v2",
        RECORDED_KEY: {RECORDED_USAGE_KEY: {"prompt_tokens": 5, "cost": 0.1}},
    }
    assert final_response(chunks[:2]) == aggregate_partials(chunks[:2])
    with pytest.raises(TypeError, match="LlmResponse dumps"):
        final_response(["not a dump"])


def _signed_call(name: str, location: str, signature: bytes) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(name=name, args={"location": location}), thought_signature=signature
    )


def _history() -> LlmRequest:
    """Return request whose model turn holds a function call with a thought signature.

    Calls and responses carry no ids, since ADK strips them before a request.
    """
    return LlmRequest(
        model="gemini-3-pro-preview",
        contents=[
            types.Content(role="user", parts=[types.Part(text="weather?")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(text="Checking."),
                    _signed_call("get_weather", "Paris", b"sig-paris"),
                    types.Part(text="One moment."),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="get_weather", response={"forecast": "sunny"}, will_continue=False
                        )
                    )
                ],
            ),
            types.Content(role="user", parts=[types.Part(text="and tomorrow?")]),
        ],
    )


def _edited(request: LlmRequest, edit: Callable[[list[nemo_relay.JsonObject]], None]) -> LlmRequest:
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    messages = list(annotated.messages)
    edit(messages)
    annotated.messages = messages
    assert write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )
    return request


def _dumped(content: types.Content) -> nemo_relay.Json:
    return content.model_dump(mode="json", exclude_none=True)


def _dumped_model(model: pydantic.BaseModel) -> nemo_relay.Json:
    return model.model_dump(mode="json", exclude_none=True)


def test_encode_keeps_unchanged_contents_whole_when_another_message_is_edited() -> None:
    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[-1] = {**messages[-1], "content": "and the day after?"}

    edited = _edited(_history(), edit)

    assert [_dumped(content) for content in edited.contents[:3]] == [
        _dumped(content) for content in _history().contents[:3]
    ]
    assert present(edited.contents[3].parts)[0].text == "and the day after?"


def test_encode_keeps_the_slots_and_part_fields_of_an_edited_turn() -> None:
    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        blocks = as_list(messages[1]["content"])
        first = as_object(blocks[0])
        messages[1] = {**messages[1], "content": [{**first, "text": "Looking."}, blocks[1]]}

    parts = _edited(_history(), edit).contents[1].parts or []

    assert [part.text for part in parts] == ["Looking.", None, "One moment."]
    assert present(parts[1].function_call).args == {"location": "Paris"}
    assert parts[1].thought_signature == b"sig-paris"


def test_encode_keeps_a_leading_image_when_a_string_edit_changes_the_text() -> None:
    image = types.Part(inline_data=types.Blob(mime_type="image/png", data=b"\x89PNG"))
    request = LlmRequest(
        model="gemini-2.5-pro",
        contents=[types.Content(role="user", parts=[image, types.Part(text="what is in this image?")])],
    )

    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[0] = {**messages[0], "content": "Please describe this image in detail."}

    parts = present(_edited(request, edit).contents[0].parts)

    assert [part.text for part in parts] == [None, "Please describe this image in detail."]
    assert present(parts[0].inline_data).data == b"\x89PNG"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("Edited.", [types.Part(text="Edited.")], id="string_edit"),
        pytest.param(None, [], id="cleared"),
    ],
)
def test_encode_applies_a_string_edit_over_a_signed_last_text_part(
    content: str | None, expected: list[types.Part]
) -> None:
    request = LlmRequest(
        model="gemini-3-pro-preview",
        contents=[
            types.Content(role="user", parts=[types.Part(text="hi")]),
            types.Content(
                role="model",
                parts=[types.Part(text="One."), types.Part(text="Two.", thought_signature=b"sig-last")],
            ),
        ],
    )

    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[1] = {**messages[1], "content": content}

    assert _edited(request, edit).contents[1].parts == expected


@pytest.mark.parametrize(
    ("calls", "expected"),
    [
        pytest.param(
            [_signed_call("get_weather", "Paris", b"sig-weather"), _signed_call("get_time", "Paris", b"sig-time")],
            [("get_time", b"sig-time"), ("get_weather", b"sig-weather")],
            id="named_calls",
        ),
        pytest.param(
            [_signed_call("get_weather", "Paris", b"sig-paris"), _signed_call("get_weather", "Rome", b"sig-rome")],
            [("get_weather", b"sig-rome"), ("get_weather", b"sig-paris")],
            id="id_less_calls_to_one_tool",
        ),
    ],
)
def test_encode_moves_each_reordered_call_with_its_signature(
    calls: list[types.Part], expected: list[tuple[str, bytes]]
) -> None:
    request = LlmRequest(model="gemini-3-pro-preview", contents=[types.Content(role="model", parts=calls)])

    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        tool_calls = as_list(messages[0]["tool_calls"])
        messages[0] = {**messages[0], "tool_calls": tool_calls[::-1]}

    parts = _edited(request, edit).contents[0].parts or []

    assert [(part.function_call.name, part.thought_signature) for part in parts if part.function_call] == expected


def test_encode_keeps_the_signature_of_a_call_moved_into_a_new_turn() -> None:
    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[1:2] = [
            {"role": "assistant", "content": "Let me check."},
            {"role": "assistant", "content": None, "tool_calls": messages[1]["tool_calls"]},
        ]

    edited = _edited(_history(), edit)

    assert [content.role for content in edited.contents] == ["user", "model", "model", "user", "user"]
    assert present(edited.contents[1].parts)[0].text == "Let me check."
    (call,) = edited.contents[2].parts or []
    assert present(call.function_call).name == "get_weather"
    assert call.thought_signature == b"sig-paris"


@pytest.mark.parametrize("text_first", [True, False], ids=["text_first", "response_first"])
def test_text_beside_a_function_response_is_a_user_message_and_edits_keep_the_slots(*, text_first: bool) -> None:
    note = types.Part(text="Note: be brief.")
    response = types.Part(function_response=types.FunctionResponse(name="get_weather", response={"forecast": "sunny"}))
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="weather?")]),
            types.Content(role="model", parts=_calls("Paris")),
            types.Content(role="user", parts=[note, response] if text_first else [response, note]),
        ],
    )
    raw = _raw(request)

    assert GEMINI_CODEC.decode(raw).messages[2:] == [
        {"role": "user", "content": "Note: be brief."},
        {"role": "tool", "tool_call_id": "get_weather", "content": '{"forecast":"sunny"}'},
    ]

    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[2] = {**messages[2], "content": "Note: be terse."}
        messages[3] = {**messages[3], "content": '{"forecast":"rain"}'}

    edited = types.Part(text="Note: be terse.")
    rained = types.Part(function_response=types.FunctionResponse(name="get_weather", response={"forecast": "rain"}))
    assert _edited(request, edit).contents[2].parts == ([edited, rained] if text_first else [rained, edited])


def test_an_assistant_replacement_beside_a_function_response_is_rejected() -> None:
    response = types.Part(function_response=types.FunctionResponse(name="get_weather", response={"forecast": "sunny"}))
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="weather?")]),
            types.Content(role="model", parts=_calls("Paris")),
            types.Content(role="user", parts=[types.Part(text="Note: be brief."), response]),
        ],
    )

    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[2] = {
            "role": "assistant",
            "content": "redo",
            "tool_calls": [{"id": "c9", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}],
        }

    with pytest.raises(ValueError, match="only a user message can share the turn"):
        _edited(request, edit)


def test_removing_the_user_message_beside_a_function_response_drops_its_parts() -> None:
    response = types.Part(function_response=types.FunctionResponse(name="get_weather", response={"forecast": "sunny"}))
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="weather?")]),
            types.Content(role="model", parts=_calls("Paris")),
            types.Content(role="user", parts=[types.Part(text="Note: be brief."), response]),
        ],
    )

    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        del messages[2]

    assert _edited(request, edit).contents[2].parts == [response]


@pytest.mark.parametrize("image_first", [True, False])
def test_tool_output_edit_keeps_unchanged_visible_parts_in_their_slots(*, image_first: bool) -> None:
    image = types.Part(inline_data=types.Blob(mime_type="image/png", data=b"hi"))
    response = types.Part(function_response=types.FunctionResponse(name="get_weather", response={"forecast": "sunny"}))
    parts = [image if image_first else types.Part(text="one"), response, types.Part(text="two")]
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="weather?")]),
            types.Content(role="model", parts=_calls("Paris")),
            types.Content(role="user", parts=parts),
        ],
    )

    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[3] = {**messages[3], "content": '{"forecast":"rain"}'}

    rained = types.Part(function_response=types.FunctionResponse(name="get_weather", response={"forecast": "rain"}))
    assert _edited(request, edit).contents[2].parts == [parts[0], rained, parts[2]]


def test_decode_keeps_tool_arguments_and_results_as_utf8_json() -> None:
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name="get_weather", args={"city": "café"}))],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(function_response=types.FunctionResponse(name="get_weather", response={"note": "café"}))
                ],
            ),
        ],
    )

    messages = GEMINI_CODEC.decode(_raw(request)).messages

    assert as_object(as_object(as_list(messages[0]["tool_calls"])[0])["function"])["arguments"] == '{"city":"café"}'
    assert messages[1]["content"] == '{"note":"café"}'


def test_encode_keeps_the_other_fields_of_an_edited_function_response() -> None:
    def edit(messages: list[nemo_relay.JsonObject]) -> None:
        messages[2] = {**messages[2], "content": '{"forecast":"rain"}'}

    (part,) = _edited(_history(), edit).contents[2].parts or []

    function_response = present(part.function_response)
    assert function_response.response == {"forecast": "rain"}
    assert function_response.will_continue is False


def test_the_extra_request_body_reaches_api_specific_and_an_edit_there_comes_back() -> None:
    request = _request()
    request.config.http_options = types.HttpOptions(headers={"x-adk": "own"}, extra_body={"labels": {"team": "a"}})
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    api_specific = present(annotated.api_specific)
    data = as_object(api_specific["data"])
    config = as_object(data["config"])
    assert config["http_options"] == {"extra_body": {"labels": {"team": "a"}}}
    edited_config = {**config, "http_options": {"extra_body": {"labels": {"team": "b"}}}}
    annotated.api_specific = {**api_specific, "data": {**data, "config": edited_config}}

    assert write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )

    options = present(request.config.http_options)
    assert options.extra_body == {"labels": {"team": "b"}}
    assert options.headers == {"x-adk": "own"}


def test_calls_sharing_an_id_keep_their_own_part_fields_in_order_when_their_turn_is_edited() -> None:
    calls = [
        types.Part(
            function_call=types.FunctionCall(id="dup", name="get_weather", args={"location": city}),
            thought_signature=signature,
        )
        for city, signature in (("Paris", b"sig-paris"), ("Rome", b"sig-rome"))
    ]
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="hi")]),
            types.Content(role="model", parts=[types.Part(text="Checking."), *calls]),
        ],
    )
    raw = _raw(request)
    original_parts = as_list(as_object(as_list(raw.content["contents"])[1])["parts"])
    signatures = [as_object(part)["thought_signature"] for part in original_parts if "function_call" in as_object(part)]
    annotated = GEMINI_CODEC.decode(raw)
    messages = list(annotated.messages)
    messages[1] = {**messages[1], "content": "Checking twice."}
    annotated.messages = messages

    encoded = GEMINI_CODEC.encode(annotated, raw).content

    parts = as_list(as_object(as_list(encoded["contents"])[1])["parts"])
    kept = [as_object(part)["thought_signature"] for part in parts if "function_call" in as_object(part)]
    assert kept == signatures


def test_encode_writes_api_specific_edits_back_to_the_request() -> None:
    request = _request()
    raw = _raw(request)
    annotated = GEMINI_CODEC.decode(raw)
    api_specific = present(annotated.api_specific)
    data = as_object(api_specific["data"])
    config = as_object(data["config"])
    annotated.api_specific = {
        **api_specific,
        "data": {
            **data,
            "previous_interaction_id": "interaction-1",
            "config": {**config, "top_k": 9, "tool_config": {"retrieval_config": {"language_code": "fr"}}},
        },
    }

    assert write_back_request(
        request, payload=GEMINI_CODEC.encode(annotated, raw).content, original_payload=raw.content
    )

    assert request.previous_interaction_id == "interaction-1"
    assert request.config.top_k == 9
    assert request.config.response_schema is Forecast
    tool_config = present(request.config.tool_config)
    assert present(tool_config.retrieval_config).language_code == "fr"
    assert present(tool_config.function_calling_config).allowed_function_names == ["get_weather"]


def _with_api_data(annotated: nemo_relay.AnnotatedLLMRequest, **changes: nemo_relay.Json) -> None:
    api_specific = present(annotated.api_specific)
    data = as_object(api_specific["data"])
    annotated.api_specific = {**api_specific, "data": {**data, **changes}}


@pytest.mark.parametrize(
    ("edit", "error"),
    [
        pytest.param(
            lambda annotated: setattr(annotated, "api_specific", {"api": "custom", "api_name": "other", "data": {}}),
            "'other' api_specific",
            id="foreign_api_specific",
        ),
        pytest.param(
            lambda annotated: _with_api_data(annotated, contents=[]), "cannot write 'contents'", id="modeled_field"
        ),
        pytest.param(
            lambda annotated: _with_api_data(annotated, config={"temperature": 1}),
            "through the normalized request",
            id="modeled_config_field",
        ),
        pytest.param(
            lambda annotated: _with_api_data(annotated, omega=1, alpha=2),
            "cannot write 'alpha'",
            id="first_unwritable_field_by_name",
        ),
        pytest.param(
            lambda annotated: _with_api_data(annotated, config={"top_p": 0.5, "temperature": 1}),
            "config field 'temperature'",
            id="first_modeled_config_field_by_name",
        ),
        pytest.param(
            lambda annotated: setattr(annotated, "extra", {"unknown": 1}),
            "no fields outside LlmRequest",
            id="extra_field",
        ),
        pytest.param(
            lambda annotated: setattr(
                annotated,
                "messages",
                [
                    *annotated.messages,
                    {"role": "provider_native", "provider": "openai_responses", "kind": "reasoning", "value": {}},
                ],
            ),
            "'openai_responses' 'reasoning' provider-native message",
            id="foreign_native_message",
        ),
        pytest.param(
            lambda annotated: setattr(
                annotated,
                "messages",
                [
                    *annotated.messages,
                    {
                        "role": "user",
                        "content": [
                            {"type": "provider_native", "provider": "anthropic_messages", "kind": "image", "value": {}}
                        ],
                    },
                ],
            ),
            "'anthropic_messages' provider-native part",
            id="foreign_native_part",
        ),
        pytest.param(
            lambda annotated: setattr(
                annotated, "tool_choice", {"provider": "openai_chat", "kind": "function_calling_config", "value": {}}
            ),
            "'openai_chat' tool choice",
            id="foreign_native_tool_choice",
        ),
        pytest.param(
            lambda annotated: setattr(
                annotated, "tool_choice", {"provider": "google_adk", "kind": "function_calling_config", "value": "ANY"}
            ),
            "cannot write tool choice",
            id="native_tool_choice_without_an_object",
        ),
        pytest.param(
            lambda annotated: setattr(
                annotated,
                "messages",
                [*annotated.messages, {"role": "tool", "tool_call_id": "call-9", "content": "{}"}],
            ),
            "unknown call 'call-9'",
            id="tool_message_for_an_unknown_call",
        ),
        pytest.param(
            lambda annotated: setattr(
                annotated,
                "messages",
                [
                    *annotated.messages,
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call-9", "type": "function", "function": {"name": "get_time", "arguments": "no"}}
                        ],
                    },
                ],
            ),
            "JSON object text as tool call arguments",
            id="arguments_not_json_object_text",
        ),
    ],
)
def test_encode_rejects_edits_an_adk_request_cannot_hold(
    edit: Callable[[nemo_relay.AnnotatedLLMRequest], None], error: str
) -> None:
    raw = _raw(_request())
    annotated = GEMINI_CODEC.decode(raw)
    edit(annotated)

    with pytest.raises(ValueError, match=error):
        GEMINI_CODEC.encode(annotated, raw)


def test_encode_rejects_a_tool_message_whose_blocks_do_not_start_with_the_response_text() -> None:
    raw = _raw(_request())
    annotated = GEMINI_CODEC.decode(raw)
    annotated.messages = [
        *annotated.messages[:2],
        {
            **annotated.messages[2],
            "content": [{"type": "provider_native", "provider": "google_adk", "kind": "inline_data", "value": {}}],
        },
        *annotated.messages[3:],
    ]

    with pytest.raises(TypeError, match="first block is the response text"):
        GEMINI_CODEC.encode(annotated, raw)


_USAGE_DUMP: Final[nemo_relay.JsonObject] = {
    "prompt_token_count": 100,
    "candidates_token_count": 20,
    "total_token_count": 120,
    "cached_content_token_count": 40,
    "thoughts_token_count": 5,
    CACHE_WRITE_KEY: 10,
}


@pytest.mark.parametrize(
    ("codec", "usage", "api_specific", "unmodeled"),
    [
        pytest.param(
            GEMINI_CODEC,
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cache_read_tokens": 40,
                "uncached_input_tokens": 60,
            },
            {"api": "gemini_generate_content", "thoughts_tokens": 5},
            {CACHE_WRITE_KEY: 10},
            id="gemini",
        ),
        pytest.param(
            ANTHROPIC_CODEC,
            {
                "prompt_tokens": 50,
                "uncached_input_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75,
                "cache_read_tokens": 40,
                "cache_write_tokens": 10,
            },
            {"api": "anthropic_messages"},
            {"thoughts_token_count": 5},
            id="anthropic",
        ),
        pytest.param(
            CHAT_CODEC,
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cache_read_tokens": 40,
                "uncached_input_tokens": 60,
            },
            {"api": "openai_chat"},
            {"thoughts_token_count": 5, CACHE_WRITE_KEY: 10},
            id="openai_chat",
        ),
        pytest.param(
            RESPONSES_CODEC,
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cache_read_tokens": 40,
                "uncached_input_tokens": 60,
            },
            {
                "api": "openai_responses",
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
            {CACHE_WRITE_KEY: 10},
            id="openai_responses",
        ),
    ],
)
def test_decode_response_reads_usage_by_provider_api(
    codec: AdkCodec, usage: nemo_relay.JsonObject, api_specific: nemo_relay.JsonObject, unmodeled: nemo_relay.JsonObject
) -> None:
    annotated = codec.decode_response(
        {"content": {"role": "model", "parts": [{"text": "x"}]}, "usage_metadata": _USAGE_DUMP}
    )

    assert annotated.usage == usage
    assert annotated.api_specific == api_specific
    assert annotated.extra == {"usage_metadata": unmodeled}


def test_decode_response_takes_the_recorded_response_id_over_the_interaction_id() -> None:
    assert GEMINI_CODEC.decode_response({"interaction_id": "int-1", RECORDED_KEY: {RECORDED_ID_KEY: "resp-1"}}).id == (
        "resp-1"
    )
    assert GEMINI_CODEC.decode_response({"interaction_id": "int-1"}).id == "int-1"


def test_decode_response_keeps_recorded_fields_no_variant_reads_in_extra() -> None:
    url_context: nemo_relay.JsonObject = {"url_metadata": [{"retrieved_url": "https://weather.example"}]}
    usage: nemo_relay.JsonObject = {"prompt_tokens": 5}
    recorded: nemo_relay.JsonObject = {
        RECORDED_ID_KEY: "chatcmpl-1",
        URL_CONTEXT_KEY: url_context,
        RECORDED_USAGE_KEY: usage,
    }

    annotated = CHAT_CODEC.decode_response({RECORDED_KEY: recorded})

    assert annotated.id == "chatcmpl-1"
    assert annotated.extra == {RECORDED_KEY: {URL_CONTEXT_KEY: url_context}}
    assert GEMINI_CODEC.decode_response({RECORDED_KEY: recorded}).extra == {RECORDED_KEY: {RECORDED_USAGE_KEY: usage}}


@pytest.mark.parametrize(
    ("codec", "usage"),
    [
        pytest.param(ANTHROPIC_CODEC, {"input_tokens": 5, "cost_usd": 0.1}, id="anthropic"),
        pytest.param(CHAT_CODEC, {"prompt_tokens": 5, "cost": 0.1}, id="openai_chat"),
        pytest.param(RESPONSES_CODEC, {"input_tokens": 5, "cost_usd": 0.1}, id="openai_responses"),
    ],
)
def test_decode_response_reads_the_cost_of_the_recorded_usage(codec: AdkCodec, usage: nemo_relay.JsonObject) -> None:
    annotated = codec.decode_response(
        {"usage_metadata": {"prompt_token_count": 5}, RECORDED_KEY: {RECORDED_USAGE_KEY: usage}}
    )

    assert present(annotated.usage)["cost"] == {"currency": "USD", "source": "provider_reported", "total": 0.1}
    assert annotated.extra == {}


def test_anthropic_usage_rejects_cache_tokens_beyond_the_prompt() -> None:
    with pytest.raises(ValueError, match="must include"):
        ANTHROPIC_CODEC.decode_response(
            {"usage_metadata": {"prompt_token_count": 10, "cached_content_token_count": 20}}
        )


def test_cache_write_rides_on_the_usage_dump_and_returns_to_a_rebuilt_response() -> None:
    usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=10)
    object.__setattr__(usage, CACHE_WRITE_KEY, 3)

    payload = response_payload(LlmResponse(usage_metadata=usage))

    assert payload["usage_metadata"] == {"prompt_token_count": 10, CACHE_WRITE_KEY: 3}
    rebuilt = response_from_payload(payload)
    assert present(rebuilt.usage_metadata).prompt_token_count == 10
    assert getattr(rebuilt.usage_metadata, CACHE_WRITE_KEY) == 3


def test_a_camel_case_usage_key_rebuilds_as_its_field() -> None:
    rebuilt = response_from_payload({"usage_metadata": {"promptTokenCount": 5, CACHE_WRITE_KEY: 3}})

    usage = present(rebuilt.usage_metadata)
    assert usage.prompt_token_count == 5
    assert "promptTokenCount" not in vars(usage)
    assert getattr(usage, CACHE_WRITE_KEY) == 3
