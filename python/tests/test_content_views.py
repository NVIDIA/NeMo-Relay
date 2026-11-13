# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python selection, typed traversal, and snapshot ownership regressions."""

import gc

import pytest

from nemo_relay import AnnotatedLLMRequest, AnnotatedLLMResponse
from nemo_relay.codecs import ContentEntry, MessageView, OpenAIResponsesCodec


def test_selection_instructions_indices_and_no_backward_search() -> None:
    request = AnnotatedLLMRequest([])
    assert request.last_message() is None
    assert request.instructions_view() is None
    for index in (0, -1):
        with pytest.raises(IndexError):
            request.message(index)
    request.messages = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": [{"type": "image", "image": {"url": "image"}}]},
    ]
    request.instructions = "policy"
    selected = request.last_message()
    assert selected is not None
    assert selected.text() is None
    assert request.message(0).text() == "earlier"
    assert request.message(-2).text() == "earlier"
    instruction = request.instructions_view()
    assert instruction is not None
    assert instruction.text() == "policy"
    entry = next(selected.parts())
    assert (entry.location.root, entry.location.message_index, entry.location.field, entry.location.index) == (
        "request_message",
        1,
        "content",
        0,
    )
    assert next(instruction.parts()).location.root == "content"
    for index in (2, -3, -(2**63)):
        with pytest.raises(IndexError):
            request.message(index)


@pytest.mark.parametrize(
    ("content", "expected_parts", "expected"),
    [
        ([], [], None),
        ("", [""], ""),
        ("  ", ["  "], "  "),
        (
            [{"type": "text", "text": " a "}, {"type": "refusal", "refusal": ""}, {"type": "text", "text": "z\n"}],
            [" a ", "", "z\n"],
            " a \n\nz\n",
        ),
        ([{"type": "tool_result", "tool_use_id": "c", "content": "hidden"}], [], None),
    ],
)
def test_text_extraction_matches_helpers(content, expected_parts, expected) -> None:
    view = MessageView.from_content(content)
    assert list(view.text_parts()) == expected_parts
    assert view.text() == expected
    response = AnnotatedLLMResponse(message=content)
    assert response.output().text() == response.response_text() == expected
    request = AnnotatedLLMRequest([{"role": "user", "content": content}])
    selected = request.last_message()
    assert selected is not None
    assert selected.text() == request.last_user_message() == expected


def test_mixed_parts_and_tool_calls_keep_typed_payloads_and_order() -> None:
    request = AnnotatedLLMRequest(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "inline", "name": "lookup", "input": {"q": " a "}},
                    {"type": "tool_result", "tool_use_id": "inline", "content": "result", "is_error": True},
                    {"type": "refusal", "refusal": "no"},
                ],
                "tool_calls": [
                    {"id": "separate", "type": "function", "function": {"name": "search", "arguments": "{not JSON}"}}
                ],
            }
        ]
    )
    view = request.last_message()
    assert view is not None
    entries = list(view.parts())
    assert all(isinstance(entry, ContentEntry) for entry in entries)
    assert [entry.kind for entry in entries] == ["text", "tool_call", "tool_result", "refusal", "tool_call"]
    assert view.text() == "answer\nno"
    call = entries[1].tool_call
    assert call is not None
    assert (call.id, call.call_id, call.name, call.arguments.kind, call.arguments.value) == (
        "inline",
        None,
        "lookup",
        "json",
        {"q": " a "},
    )
    result = entries[2].tool_result
    assert result is not None
    assert (result.id, result.call_id, result.name, result.is_error) == (None, "inline", None, True)
    assert (result.payload.kind, result.payload.value) == ("json", "result")
    separate = entries[-1].tool_call
    assert separate is not None
    assert (separate.arguments.kind, separate.arguments.value) == ("string", "{not JSON}")
    assert (entries[-1].location.field, entries[-1].location.index) == ("tool_calls", 0)
    assert entries[0].tool_call is None
    assert entries[0].tool_result is None
    assert entries[0].value is None
    assert entries[0].provider is None
    assert entries[0].native_kind is None
    assert entries[2].text is None


@pytest.mark.parametrize(
    ("message", "kind", "payload"),
    [
        ({"role": "tool", "tool_call_id": "c", "content": "hidden"}, "content", "hidden"),
        ({"role": "function", "name": "f", "content": "hidden"}, "string", "hidden"),
        ({"role": "function", "name": "f", "content": None}, "null", None),
        ({"role": "tool_result", "call_id": "c", "output": {"text": "hidden"}}, "json", {"text": "hidden"}),
    ],
)
def test_whole_tool_results_are_not_recursively_extracted(message, kind, payload) -> None:
    view = MessageView.from_message(message)
    result = next(view.parts()).tool_result
    assert result is not None
    assert (result.payload.kind, result.payload.value) == (kind, payload)
    assert view.text() is None
    assert list(view.text_parts()) == []


def test_media_native_and_shallow_fallbacks() -> None:
    view = MessageView.from_content(
        [
            {"type": "image_url", "image_url": {"url": "https://example.com/i.png"}},
            {"type": "image", "image": {"text": "hidden"}},
            {"type": "audio", "audio": {"data": "sound"}},
            {"type": "file", "file": {"id": "file"}},
            {"type": "provider_native", "provider": "custom", "kind": "extension", "value": {"text": "native"}},
            {"type": "provider_native", "provider": "custom", "kind": "thinking", "value": {"text": "hidden"}},
            {
                "type": "provider_native",
                "provider": "custom",
                "kind": "extension",
                "value": {"nested": {"text": "hidden"}},
            },
        ]
    )
    entries = list(view.parts())
    assert [entry.kind for entry in entries[:4]] == ["image_url", "image", "audio", "file"]
    assert entries[0].value == {"url": "https://example.com/i.png"}
    assert [entry.value for entry in entries[1:4]] == [{"text": "hidden"}, {"data": "sound"}, {"id": "file"}]
    assert (entries[4].provider, entries[4].native_kind, entries[4].value) == (
        "custom",
        "extension",
        {"text": "native"},
    )
    assert view.text() == "native"
    native = MessageView.from_message(
        {
            "role": "provider_native",
            "provider": "openai_responses",
            "kind": "message",
            "value": {"role": "user", "content": [{"type": "input_text", "text": "known"}]},
        }
    )
    assert native.text() == "known"
    assert next(native.parts()).location.field == "native_content"


def test_snapshot_lifetime_and_python_value_mutation_isolation() -> None:
    request = AnnotatedLLMRequest([{"role": "user", "content": [{"type": "file", "file": {"nested": [1]}}]}])
    before = request.messages
    view = request.last_message()
    assert view is not None
    iterator = view.parts()
    request.messages = [{"role": "user", "content": "new"}]
    del request, view
    gc.collect()
    entry = next(iterator)
    del iterator
    gc.collect()
    value = entry.value
    assert isinstance(value, dict)
    value["nested"] = [2]
    assert entry.value == before[0]["content"][0]["file"] == {"nested": [1]}
    with pytest.raises(AttributeError):
        setattr(entry, "kind", "text")
    with pytest.raises(AttributeError):
        setattr(entry.location, "index", 7)


def test_tool_payload_survives_owner_and_returns_independent_values() -> None:
    response = AnnotatedLLMResponse(tool_calls=[{"id": "c", "name": "f", "arguments": {"q": [1]}}])
    view = response.output()
    call = next(view.parts()).tool_call
    assert call is not None
    payload = call.arguments
    del response, view, call
    gc.collect()
    value = payload.value
    assert isinstance(value, dict)
    value["q"] = [2]
    assert (payload.kind, payload.value) == ("json", {"q": [1]})


def test_response_refusals_locations_and_snapshot_are_consistent() -> None:
    response = OpenAIResponsesCodec().decode_response(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "yes"}, {"type": "refusal", "refusal": "no"}],
                }
            ]
        }
    )
    before = response.message
    view = response.output()
    assert view.text() == "yes\nno"
    assert response.message == before
    entries = list(view.parts())
    assert [entry.kind for entry in entries] == ["text", "refusal"]
    assert (entries[1].location.root, entries[1].location.field, entries[1].location.index) == (
        "response",
        "message",
        1,
    )
    assert isinstance(before, list)
    before.clear()
    assert response.output().text() == view.text() == "yes\nno"
    assert list(AnnotatedLLMResponse().output().parts()) == []


@pytest.mark.parametrize("factory,value", [(MessageView.from_message, 42), (MessageView.from_content, {})])
def test_invalid_standalone_input_is_rejected(factory, value) -> None:
    with pytest.raises(ValueError):
        factory(value)


def test_standalone_tool_ids_and_json_string_representation() -> None:
    view = MessageView.from_message(
        {"role": "tool_call", "id": "item", "call_id": "correlation", "name": "lookup", "arguments": "{invalid}"}
    )
    entry = next(view.parts())
    call = entry.tool_call
    assert call is not None
    assert (entry.location.root, entry.location.field, entry.location.index) == ("message", "tool_call_item", None)
    assert (call.id, call.call_id, call.name, call.arguments.kind, call.arguments.value) == (
        "item",
        "correlation",
        "lookup",
        "json",
        "{invalid}",
    )
    assert view.text() is None
    response = AnnotatedLLMResponse(tool_calls=[{"id": "c", "name": "lookup", "arguments": "{invalid}"}])
    output = response.output()
    assert output.text() is None
    call = next(output.parts()).tool_call
    assert call is not None
    assert (call.arguments.kind, call.arguments.value) == ("json", "{invalid}")


def test_iterators_are_independent_and_retain_the_snapshot() -> None:
    view = MessageView.from_content([{"type": "text", "text": "first"}, {"type": "refusal", "refusal": "second"}])
    text = view.text_parts()
    first = view.parts()
    second = view.parts()
    assert iter(first) is first
    assert iter(text) is text
    assert next(first).text == next(second).text == "first"
    del view
    gc.collect()
    assert list(text) == ["first", "second"]
    assert next(first).text == next(second).text == "second"
    for iterator in (first, second, text):
        with pytest.raises(StopIteration):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)


def test_empty_selected_items_remain_independent_snapshots() -> None:
    request = AnnotatedLLMRequest(
        [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": []}],
        instructions=[],
    )
    selected = request.last_message()
    instructions = request.instructions_view()
    assert selected is not None and instructions is not None
    assert list(selected.parts()) == list(instructions.parts()) == []
    assert selected.text() is None
    assert request.system_prompt() is None
    request.messages = [{"role": "user", "content": "later"}]
    request.instructions = "later policy"
    assert list(selected.parts()) == list(instructions.parts()) == []
