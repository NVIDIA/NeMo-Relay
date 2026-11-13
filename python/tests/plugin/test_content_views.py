# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone worker snapshots and differential parity with Rust-backed views."""

from __future__ import annotations

import gc
import inspect
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from nemo_relay_plugin import codecs

CONTENT = [
    {"type": "text", "text": " a "},
    {"type": "refusal", "refusal": ""},
    {"type": "tool_use", "id": "inline", "name": "lookup", "input": {"q": [1]}},
    {"type": "tool_result", "tool_use_id": "inline", "content": {"text": "hidden"}, "is_error": True},
    {"type": "image_url", "image_url": {"url": "https://example.com/i.png", "detail": "low", "ignored": True}},
    {"type": "image", "image": {"text": "hidden"}},
    {"type": "audio", "audio": {"data": "sound"}},
    {"type": "file", "file": {"id": "file"}},
    {"type": "provider_native", "provider": "custom", "kind": "extension", "value": {"refusal": "fallback"}},
    {"type": "provider_native", "provider": "custom", "kind": "thinking", "value": {"text": "hidden"}},
    {"type": "provider_native", "provider": "custom", "kind": "extension", "value": {"nested": {"text": "hidden"}}},
    {"type": "text", "text": "z\n"},
]

MESSAGES = [
    {"role": "system", "content": "policy"},
    {"role": "developer", "content": []},
    {"role": "user", "content": CONTENT},
    {"role": "assistant"},
    {
        "role": "assistant",
        "content": CONTENT,
        "tool_calls": [{"id": "separate", "type": "function", "function": {"name": "f", "arguments": "{invalid}"}}],
    },
    {"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "hidden"}]},
    {"role": "function", "name": "f", "content": "hidden"},
    {"role": "function", "name": "f", "content": None},
    {"role": "tool_call", "id": "item", "call_id": "c", "name": "f", "arguments": "{invalid}"},
    {"role": "tool_result", "id": "item", "call_id": "c", "output": {"text": "hidden"}},
    {
        "role": "provider_native",
        "provider": "openai_responses",
        "kind": "message",
        "value": {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "known"},
                {"type": "refusal", "refusal": "no"},
                {"type": "reasoning", "text": "hidden"},
                {"type": "unknown", "nested": {"text": "hidden"}},
            ],
        },
    },
    {
        "role": "provider_native",
        "provider": "openai_chat",
        "kind": "user",
        "value": {"role": "user", "text": "not message content"},
    },
    {
        "role": "provider_native",
        "provider": "anthropic_messages",
        "kind": "user",
        "value": {"role": "user", "content": "known"},
    },
    {"role": "provider_native", "provider": "custom", "kind": "extension", "value": {"text": "fallback"}},
]


def _describe(view: Any) -> dict[str, Any]:
    def payload(value):
        return None if value is None else {"kind": value.kind, "value": value.value}

    entries = []
    for entry in view.parts():
        call, result, location = entry.tool_call, entry.tool_result, entry.location
        entries.append(
            {
                "kind": entry.kind,
                "text": entry.text,
                "value": entry.value,
                "provider": entry.provider,
                "native_kind": entry.native_kind,
                "location": (location.root, location.field, location.index, location.message_index),
                "call": None if call is None else (call.id, call.call_id, call.name, payload(call.arguments)),
                "result": None
                if result is None
                else (result.id, result.call_id, result.name, result.is_error, payload(result.payload)),
            }
        )
    return {"text": view.text(), "text_parts": list(view.text_parts()), "entries": entries}


@pytest.mark.parametrize("item", MESSAGES)
def test_worker_message_parity_with_rust(item) -> None:
    native = pytest.importorskip("nemo_relay.codecs")
    assert _describe(codecs.MessageView.from_message(item)) == _describe(native.MessageView.from_message(item))


@pytest.mark.parametrize("content", ["", " ", [], CONTENT])
def test_worker_standalone_content_parity_with_rust(content) -> None:
    native = pytest.importorskip("nemo_relay.codecs")
    assert _describe(codecs.MessageView.from_content(content)) == _describe(native.MessageView.from_content(content))


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"message": ""},
        {"message": CONTENT},
        {"tool_calls": [{"id": "c", "name": "lookup", "arguments": "{invalid}"}]},
        {"message": CONTENT, "tool_calls": [{"id": "c", "name": "lookup", "arguments": {"q": [1]}}]},
    ],
)
def test_worker_response_parity_with_rust(response) -> None:
    native = pytest.importorskip("nemo_relay")
    assert _describe(codecs.output(response)) == _describe(native.AnnotatedLLMResponse(**response).output())
    assert codecs.response_text(response) == native.AnnotatedLLMResponse(**response).response_text()


@pytest.mark.parametrize(
    "annotation",
    [
        {"messages": []},
        {"messages": MESSAGES},
        {"messages": MESSAGES, "instructions": ""},
        {"messages": [{"role": "system", "content": []}, {"role": "system", "content": "later"}]},
        {"messages": [{"role": "user", "content": "earlier"}, MESSAGES[-3]]},
    ],
)
def test_worker_selection_and_helpers_match_rust(annotation) -> None:
    native = pytest.importorskip("nemo_relay").AnnotatedLLMRequest(**annotation)
    worker_last, native_last = codecs.last_message(annotation), native.last_message()
    assert (None if worker_last is None else _describe(worker_last)) == (
        None if native_last is None else _describe(native_last)
    )
    worker_instructions, native_instructions = codecs.instructions_view(annotation), native.instructions_view()
    assert (None if worker_instructions is None else _describe(worker_instructions)) == (
        None if native_instructions is None else _describe(native_instructions)
    )
    assert codecs.system_prompt(annotation) == native.system_prompt()
    assert codecs.last_user_message(annotation) == native.last_user_message()
    for index in range(-len(annotation["messages"]), len(annotation["messages"])):
        assert _describe(codecs.message(annotation, index)) == _describe(native.message(index))


def test_snapshots_do_not_retain_annotations_or_mutable_payloads() -> None:
    source = [{"type": "tool_use", "id": "c", "name": "f", "input": {"q": [1]}}]
    selected_message: dict[str, Any] = {"role": "assistant", "content": source}
    request = {"messages": [object(), selected_message], "extra": object()}
    view = codecs.last_message(request)
    assert view is not None
    iterator = view.parts()
    selected_message["content"] = "replacement"
    source[0]["input"]["q"].append(2)
    del request, source, view, selected_message
    gc.collect()
    entry = next(iterator)
    call = entry.tool_call
    assert call is not None
    assert call.arguments.value == {"q": [1]}
    assert entry.location.message_index == 1
    value = call.arguments.value
    value["q"].append(99)
    assert call.arguments.value == {"q": [1]}
    with pytest.raises(FrozenInstanceError):
        setattr(entry, "kind", "text")
    with pytest.raises(TypeError):
        call.arguments._value["q"] = (7,)
    with pytest.raises(FrozenInstanceError):
        setattr(entry.location, "index", 9)
    retained = call.arguments
    del entry, call, iterator
    assert retained.value == {"q": [1]}


def test_output_ignores_metadata_and_returns_independent_values() -> None:
    response = {"message": [{"type": "file", "file": {"ids": [1]}}], "extra": object(), "usage": object()}
    view = codecs.output(response)
    response["message"].clear()
    entry = next(view.parts())
    value = entry.value
    value["ids"].append(2)
    assert entry.value == {"ids": [1]}
    assert list(codecs.output({}).parts()) == []
    assert codecs.response_text({"tool_calls": [{"id": "c", "name": "f", "arguments": {"text": "hidden"}}]}) is None


def test_explicit_text_rules_and_iterators() -> None:
    view = codecs.MessageView.from_content(CONTENT)
    assert view.text() == " a \n\nfallback\nz\n"
    assert list(view.text_parts()) == [" a ", "", "fallback", "z\n"]
    first, second, text = view.parts(), view.parts(), view.text_parts()
    assert iter(first) is first
    assert iter(text) is text
    assert next(first).text == next(second).text == " a "
    del view
    gc.collect()
    assert list(text) == [" a ", "", "fallback", "z\n"]
    list(first)
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(first)
    assert codecs.MessageView.from_content("").text() == ""
    assert codecs.MessageView.from_content([]).text() is None


@pytest.mark.parametrize(
    "provider,kind,role",
    [
        ("openai_chat", "user", "user"),
        ("anthropic_messages", "user", "user"),
        ("openai_responses", "message", "user"),
        ("openai_chat", "system", "system"),
        ("openai_responses", "message", "developer"),
    ],
)
def test_native_selection_has_no_backward_fallback(provider, kind, role) -> None:
    request = {
        "messages": [
            {"role": role, "content": "earlier"},
            {"role": "provider_native", "provider": provider, "kind": kind, "value": {"role": role, "content": []}},
        ]
    }
    if role == "user":
        assert codecs.last_user_message(request) is None
    else:
        request["messages"].reverse()
        assert codecs.system_prompt(request) is None
    request["instructions"] = []
    assert codecs.system_prompt(request) is None


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        ("extension", {"text": "", "refusal": "ignored"}, ""),
        ("extension", {"refusal": "fallback"}, "fallback"),
        ("extension", {"text": "hidden", "thought": True}, None),
        ("extension", {"type": "tool_use", "text": "hidden"}, None),
        ("extension", {"nested": {"text": "hidden"}}, None),
        ("tool_result", {"text": "hidden"}, None),
    ],
)
def test_native_fallback_is_shallow_and_excludes_non_text(kind, value, expected) -> None:
    assert (
        codecs.MessageView.from_message(
            {"role": "provider_native", "provider": "custom", "kind": kind, "value": value}
        ).text()
        == expected
    )


def test_indices_and_selection_boundaries() -> None:
    request = {"messages": [{"role": "user", "content": "old"}, {"role": "assistant"}]}
    selected = codecs.last_message(request)
    assert selected is not None and selected.text() is None
    assert codecs.message(request, -2).text() == "old"
    assert codecs.last_message({}) is None
    assert codecs.instructions_view({}) is None
    assert codecs.last_user_message({}) is None
    assert codecs.system_prompt({}) is None
    for index in (2, -3, 2**100, -(2**100)):
        with pytest.raises(IndexError):
            codecs.message(request, index)
    with pytest.raises(TypeError):
        codecs.message(request, 1.5)


@pytest.mark.parametrize(
    "factory,value",
    [
        (codecs.MessageView.from_message, 42),
        (codecs.MessageView.from_message, {"role": "unknown"}),
        (codecs.MessageView.from_message, {"role": "function", "name": "f", "content": []}),
        (codecs.MessageView.from_content, {}),
        (codecs.MessageView.from_content, [{"type": "text", "text": 1}]),
        (codecs.MessageView.from_content, [{"type": "unknown"}]),
        (codecs.MessageView.from_content, [{"type": "image"}]),
        (codecs.MessageView.from_content, [{"type": "tool_result", "tool_use_id": "c", "is_error": "yes"}]),
        (codecs.MessageView.from_content, [{"type": "image_url", "image_url": "url"}]),
        (codecs.MessageView.from_content, [{"type": "image", "image": object()}]),
        (codecs.MessageView.from_content, [{"type": "image", "image": float("nan")}]),
        (codecs.MessageView.from_content, [{"type": "image", "image": {1: "bad"}}]),
        (codecs.output, {"tool_calls": {}}),
        (codecs.output, {"tool_calls": [{"name": "f"}]}),
        (codecs.last_message, {"messages": {}}),
    ],
)
def test_invalid_normalized_inputs_fail_at_selection(factory, value) -> None:
    with pytest.raises(ValueError):
        factory(value)
    with pytest.raises(TypeError):
        codecs.MessageView()


def test_public_codec_members_have_documentation() -> None:
    for name in codecs.__all__:
        value = getattr(codecs, name)
        assert inspect.getdoc(value), name
        if inspect.isclass(value):
            for member_name, member in inspect.getmembers_static(value):
                if member_name.startswith("_"):
                    continue
                if isinstance(member, property):
                    member = member.fget
                if inspect.isfunction(member):
                    assert inspect.getdoc(member), f"{name}.{member_name}"


def test_worker_content_helpers_do_not_import_main_binding() -> None:
    code = """
import importlib.abc
import sys
class NoMainBinding(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "nemo_relay" or fullname.startswith("nemo_relay."):
            raise AssertionError("standalone SDK imported the main runtime binding")
sys.meta_path.insert(0, NoMainBinding())
from nemo_relay_plugin.codecs import last_message
assert last_message({"messages": [{"role": "user", "content": "standalone"}]}).text() == "standalone"
"""
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True, capture_output=True, text=True)
