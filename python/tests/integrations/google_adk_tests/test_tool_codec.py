# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Google ADK tool codec round trips, projections, edits, and result records."""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

import pytest
from _helpers import as_object

pytest.importorskip(
    "google.adk",
    reason="google-adk must be installed to run Google ADK based tests",
    exc_type=ModuleNotFoundError,
)

from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult, File
from google.genai import types
from pydantic import BaseModel, ConfigDict

from nemo_relay.integrations.google_adk._tool_codec import (
    AdkArgumentsCodec,
    AdkCodeInputCodec,
    ProducedResults,
    tool_result_json,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import nemo_relay


class _Weather(BaseModel):
    location: str
    when: datetime.date


@dataclasses.dataclass
class _Report:
    weather: _Weather
    tags: set[str]
    chart: bytes


class _Opaque:
    pass


def _report(location: str) -> _Report:
    return _Report(_Weather(location=location, when=datetime.date(2026, 10, 21)), {"warm"}, b"\x89PNG")


def _over_the_wire(data: nemo_relay.Json) -> nemo_relay.JsonObject:
    return as_object(json.loads(json.dumps(data)))


def _name(cls: type[object]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _at(data: nemo_relay.Json, *keys: str) -> nemo_relay.JsonObject:
    for key in keys:
        data = as_object(data)[key]
    return as_object(data)


@pytest.mark.parametrize(
    ("value", "content", "is_error"),
    [
        pytest.param(
            {"location": "Paris", "tags": ("warm", "dry"), "rain": None},
            '{"location": "Paris", "rain": null, "tags": ["warm", "dry"]}',
            False,
            id="object",
        ),
        pytest.param(
            {"tags": {"warm"}, "ids": frozenset({1, 2}), "raw": b"\x89PNG", "buffer": bytearray(b"\x89")},
            '{"buffer": "iQ==", "ids": [1, 2], "raw": "iVBORw==", "tags": ["warm"]}',
            False,
            id="standard_types",
        ),
        pytest.param({"city": "Zürich", "temp": 21}, '{"city": "Zürich", "temp": 21}', False, id="object_text_kept"),
        pytest.param([1, {"b": 2, "a": 1}], '[1, {"a": 1, "b": 2}]', False, id="list"),
        pytest.param("Sunny in Paris.", "Sunny in Paris.", False, id="text"),
        pytest.param(None, "null", False, id="none"),
        pytest.param(
            CodeExecutionResult(stdout="1\n", output_files=[File(name="out.csv", content="a,b")], exit_code=0),
            "1\n",
            False,
            id="code_execution",
        ),
        pytest.param(CodeExecutionResult(stdout="partial", stderr="boom", exit_code=1), "boom", True, id="code_failed"),
        pytest.param(
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_OK, output="1\n"), "1\n", False, id="gemini_ok"
        ),
        pytest.param(
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_FAILED, output="trace"),
            "trace",
            True,
            id="gemini_failed",
        ),
        pytest.param(
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_DEADLINE_EXCEEDED),
            None,
            True,
            id="gemini_deadline_exceeded",
        ),
        pytest.param(
            types.FunctionResponse(response={"forecast": "sunny"}, scheduling=types.FunctionResponseScheduling.SILENT),
            '{"forecast": "sunny"}',
            False,
            id="function_response",
        ),
        pytest.param({"error": "boom", "code": 7}, "boom", True, id="object_failed"),
        pytest.param({"error": {"code": 7}}, '{"error": {"code": 7}}', True, id="object_failed_without_text"),
        pytest.param({"error": ""}, '{"error": ""}', False, id="object_empty_error"),
        pytest.param({"result": "Sunny.", "unit": "C"}, "Sunny.", False, id="object_text"),
        pytest.param(
            {"content": [{"type": "text", "text": "hi"}], "isError": False},
            [{"type": "text", "text": "hi"}],
            False,
            id="mcp",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "down"}], "isError": True, "structuredContent": {"a": 1}},
            [{"type": "text", "text": "down"}],
            True,
            id="mcp_failed",
        ),
        pytest.param(types.FunctionResponse(response={"error": "boom"}), "boom", True, id="function_response_failed"),
        pytest.param(
            {"content": [{"type": "text", "text": "hi"}]},
            [{"type": "text", "text": "hi"}],
            False,
            id="mcp_without_flag",
        ),
        pytest.param({"error": "boom", "result": "Sunny."}, "boom", True, id="object_error_over_result"),
        pytest.param(
            types.FunctionResponse(response={"content": [{"type": "text", "text": "hi"}], "isError": True}),
            [{"type": "text", "text": "hi"}],
            True,
            id="function_response_mcp",
        ),
    ],
)
def test_portable_results_round_trip_through_a_new_record_with_their_projection(
    value: object, content: nemo_relay.Json, is_error: bool
) -> None:
    encoded = tool_result_json(value)

    assert encoded["schema"] == "nemo.relay.tool_result@1"
    assert encoded["content"] == content
    assert encoded["is_error"] is is_error
    restored = ProducedResults().restore(_over_the_wire(encoded), object)
    assert restored == value
    assert type(restored) is type(value)


def test_the_encoding_names_each_kind_and_type() -> None:
    value = {"result": CodeExecutionResult(stdout="1\n"), "pair": (1, True), "when": datetime.date(2026, 10, 21)}

    encoded = tool_result_json(value)

    assert encoded["result"] == {
        "kind": "dict",
        "value": {
            "result": {
                "kind": "dataclass",
                "type": _name(CodeExecutionResult),
                "value": {
                    "stdout": {"kind": "json", "value": "1\n"},
                    "stderr": {"kind": "json", "value": ""},
                    "output_files": {"kind": "list", "value": []},
                    "exit_code": {"kind": "json", "value": None},
                },
            },
            "pair": {"kind": "tuple", "value": [{"kind": "json", "value": 1}, {"kind": "json", "value": True}]},
            "when": {"kind": "typed", "type": "datetime.date", "value": "2026-10-21"},
        },
    }


def test_the_same_result_always_encodes_to_the_same_json() -> None:
    report = _report("Paris")
    produced = ProducedResults()

    encoded = tool_result_json(report)

    assert tool_result_json(report) == encoded
    assert produced.add(report) == encoded
    assert produced.add(report) == encoded
    assert encoded["result"] == {
        "kind": "dataclass",
        "type": _name(_Report),
        "value": {
            "weather": {"kind": "typed", "type": _name(_Weather), "value": {"location": "Paris", "when": "2026-10-21"}},
            "tags": {"kind": "set", "value": [{"kind": "json", "value": "warm"}]},
            "chart": {"kind": "bytes", "value": "iVBORw=="},
        },
    }


def test_equal_sets_encode_alike_whatever_order_they_iterate_in() -> None:
    forward, backward = {1, 9}, {9, 1}
    assert list(forward) != list(backward)

    assert tool_result_json({"ids": forward}) == tool_result_json({"ids": backward})


def test_a_result_passed_through_is_the_object_the_tool_returned() -> None:
    part = types.Part.from_bytes(data=b"\x89PNG", mime_type="image/png")
    value = {"chart": part, "report": _report("Paris")}
    produced = ProducedResults()

    restored = produced.restore(_over_the_wire(produced.add(value)), object)

    assert restored is value


def test_an_edited_result_keeps_the_parts_the_edit_left_unchanged() -> None:
    part = types.Part.from_bytes(data=b"\x89PNG", mime_type="image/png")
    first, second = (
        _Weather(location="Paris", when=datetime.date(2026, 10, 21)),
        _Weather(location="Paris", when=datetime.date(2026, 10, 21)),
    )
    value = {"chart": part, "report": _report("Paris"), "days": [first, second]}
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add(value))

    _at(encoded, "result", "value", "report", "value", "weather", "value")["location"] = "Rome"
    restored = as_object(produced.restore(encoded, object))

    assert restored["chart"] is part
    assert restored["report"] == _report("Rome")
    assert type(restored["report"]) is _Report
    days = restored["days"]
    assert isinstance(days, list)
    assert days[0] is first
    assert days[1] is second


def test_an_object_moved_elsewhere_is_rebuilt() -> None:
    weather = _Weather(location="Paris", when=datetime.date(2026, 10, 21))
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add({"today": weather}))

    fields = _at(encoded, "result", "value")
    fields["tomorrow"] = fields.pop("today")
    restored = as_object(produced.restore(encoded, object))

    assert restored["tomorrow"] == weather
    assert restored["tomorrow"] is not weather


def test_a_retried_call_hands_back_the_latest_of_equal_results() -> None:
    first, latest = _report("Paris"), _report("Paris")
    produced = ProducedResults()
    produced.add(first)

    restored = produced.restore(_over_the_wire(produced.add(latest)), object)

    assert restored is latest


def test_objects_of_other_calls_are_rebuilt_only_by_known_types() -> None:
    encoded = _over_the_wire(tool_result_json(_report("Paris")))

    with pytest.raises(ValueError, match="Unknown type"):
        ProducedResults().restore(encoded, object)


def test_genai_objects_are_recorded_in_their_wire_form_and_restored_by_type() -> None:
    part = types.Part.from_bytes(data=b"\x89PNG", mime_type="image/png")

    encoded = _over_the_wire(tool_result_json(part))

    result = _at(encoded, "result")
    assert result["kind"] == "typed"
    assert result["type"] == "google.genai.types.Part"
    assert _at(result, "value")["inlineData"] == {"data": "iVBORw==", "displayName": None, "mimeType": "image/png"}
    restored = ProducedResults().restore(encoded, object)
    assert restored == part
    assert type(restored) is types.Part


class _Session(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    handle: _Opaque


def test_an_object_with_no_json_form_is_recorded_by_class_and_handed_back_as_it_was() -> None:
    handle = _Opaque()
    session = _Session(name="weather", handle=handle)
    value = {"forecast": "sunny", "handle": handle, "session": session}
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add(value))
    stand_in = f"<unserializable {_name(_Opaque)}>"

    fields = _at(encoded, "result", "value")
    assert fields["handle"] == {"kind": "unserializable", "type": _name(_Opaque), "ref": 0, "value": stand_in}
    assert fields["session"] == {
        "kind": "unserializable",
        "type": _name(_Session),
        "ref": 1,
        "value": {"name": "weather", "handle": stand_in},
    }
    assert produced.restore(encoded, object) is value
    fields["forecast"] = {"kind": "json", "value": "rain"}
    restored = as_object(produced.restore(encoded, object))
    assert restored["forecast"] == "rain"
    assert restored["handle"] is handle
    assert restored["session"] is session


@pytest.mark.parametrize(
    ("key", "field", "edit"),
    [
        pytest.param("handle", "value", "edited", id="value"),
        pytest.param("session", "value", {"name": "edited"}, id="holder_value"),
        pytest.param("handle", "ref", 7, id="unknown_ref"),
        pytest.param("handle", "ref", 1, id="ref_of_another_object"),
    ],
)
def test_an_edit_to_an_object_with_no_json_form_is_rejected(key: str, field: str, edit: nemo_relay.Json) -> None:
    handle = _Opaque()
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add({"handle": handle, "session": _Session(name="weather", handle=handle)}))

    _at(encoded, "result", "value", key)[field] = edit
    with pytest.raises(ValueError, match="no JSON form to rebuild from"):
        produced.restore(encoded, object)


def test_an_object_with_no_json_form_is_handed_back_by_its_ref_wherever_an_edit_places_it() -> None:
    earlier, latest = _Opaque(), _Opaque()
    produced = ProducedResults()
    first = _over_the_wire(produced.add({"handle": earlier}))
    encoded = _over_the_wire(produced.add({"handle": latest}))

    fields = _at(encoded, "result", "value")
    fields["copy"] = fields["handle"]
    fields["earlier"] = _at(first, "result", "value", "handle")
    fields["moved"] = fields.pop("handle")
    restored = as_object(produced.restore(encoded, object))

    assert restored["moved"] is latest
    assert restored["copy"] is latest
    assert restored["earlier"] is earlier


def test_an_object_with_no_json_form_keeps_one_ref_however_often_the_call_returns_it() -> None:
    handle = _Opaque()
    produced = ProducedResults()

    first = produced.add({"a": handle, "b": handle})
    second = produced.add([handle])

    reference = _at(first, "result", "value", "a")
    assert reference["ref"] == 0
    assert _at(first, "result", "value", "b") == reference
    assert _at(second, "result")["value"] == [reference]


def _counting() -> Iterator[int]:
    yield 1
    yield 2


class _Batch(BaseModel):
    name: str
    items: Iterable[int]


def test_iterators_are_recorded_without_consuming_them_and_handed_back_as_they_were() -> None:
    generator, iterator, batch = _counting(), iter([1, 2]), _Batch(name="weather", items=_counting())
    value = {"generator": generator, "iterator": iterator, "batch": batch}
    produced = ProducedResults()

    encoded = _over_the_wire(produced.add(value))

    fields = _at(encoded, "result", "value")
    assert fields["generator"] == {
        "kind": "unserializable",
        "type": "builtins.generator",
        "ref": 0,
        "value": "<unserializable iterator>",
    }
    assert as_object(fields["batch"])["value"] == {"name": "weather", "items": "<unserializable iterator>"}
    assert produced.restore(encoded, object) is value
    assert [*generator] == [1, 2]
    assert [*iterator] == [1, 2]
    assert [*batch.items] == [1, 2]


def test_iterator_arguments_are_projected_without_consuming_them() -> None:
    generator = _counting()
    arguments = {"query": "weather", "items": generator}
    codec = AdkArgumentsCodec(arguments)

    projected = _over_the_wire(codec.to_json(arguments))

    assert projected == {"query": "weather", "items": "<unserializable iterator>"}
    assert codec.from_json({**projected, "query": "forecast"})["items"] is generator
    assert [*generator] == [1, 2]


def test_an_insertion_keeps_the_objects_after_it_as_the_tool_returned_them() -> None:
    first, second = _Opaque(), _Opaque()
    rows = [{"id": 1}, {"id": 2}]
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add({"handles": [first, second], "rows": rows, "status": "ok"}))
    handles = _at(encoded, "result", "value", "handles")["value"]
    assert isinstance(handles, list)

    handles.insert(0, {"kind": "json", "value": "first"})
    _at(encoded, "result", "value")["status"] = {"kind": "json", "value": "done"}
    restored = as_object(produced.restore(encoded, object))

    restored_handles = restored["handles"]
    assert isinstance(restored_handles, list)
    assert restored_handles[0] == "first"
    assert restored_handles[1] is first
    assert restored_handles[2] is second
    assert restored["rows"] is rows
    assert restored["status"] == "done"


def test_an_object_whose_own_serializer_rejects_its_data_fails_at_encoding() -> None:
    class Upload(BaseModel):
        data: bytes

    with pytest.raises(UnicodeDecodeError):
        tool_result_json({"upload": Upload(data=b"\x89PNG")})
    with pytest.raises(UnicodeDecodeError):
        AdkArgumentsCodec({"upload": Upload(data=b"\x89PNG")})


def test_an_edit_naming_distinct_classes_of_one_name_is_rejected() -> None:
    def weather() -> type[BaseModel]:
        class Weather(BaseModel):
            location: str

        return Weather

    days = [weather().model_validate({"location": "Paris"}), weather().model_validate({"location": "Rome"})]
    value = {"result": "Sunny.", "days": days}
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add(value))
    assert produced.restore(encoded, object) is value

    entries = _at(encoded, "result", "value", "days")["value"]
    assert isinstance(entries, list)
    _at(entries[0], "value")["location"] = "Oslo"
    with pytest.raises(ValueError, match="distinct classes named"):
        produced.restore(encoded, object)


@pytest.mark.parametrize(
    ("value", "content", "is_error", "expected"),
    [
        pytest.param("Sunny.", "Foggy.", False, "Foggy.", id="text"),
        pytest.param("Sunny.", None, False, "", id="text_cleared"),
        pytest.param(
            CodeExecutionResult(stdout="1\n", stderr="", exit_code=0),
            "2\n",
            False,
            CodeExecutionResult(stdout="2\n", stderr="", exit_code=0),
            id="code_output",
        ),
        pytest.param(
            CodeExecutionResult(stdout="1\n", exit_code=0),
            "denied",
            True,
            CodeExecutionResult(stdout="1\n", stderr="denied", exit_code=0),
            id="code_failed",
        ),
        pytest.param(
            CodeExecutionResult(stdout="partial", stderr="boom", exit_code=1),
            "recovered\n",
            False,
            CodeExecutionResult(stdout="recovered\n", stderr="", exit_code=1),
            id="code_recovered",
        ),
        pytest.param(
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_OK, output="1\n"),
            "denied",
            True,
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_FAILED, output="denied"),
            id="gemini_failed",
        ),
        pytest.param(
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_DEADLINE_EXCEEDED, output="slow"),
            "too slow",
            True,
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_DEADLINE_EXCEEDED, output="too slow"),
            id="gemini_keeps_its_failure_outcome",
        ),
        pytest.param(
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_FAILED, output="trace"),
            None,
            False,
            types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_OK),
            id="gemini_recovered",
        ),
        pytest.param("Sunny.", "boom", True, {"error": "boom"}, id="text_failed"),
        pytest.param("Sunny.", "Sunny.", True, {"error": "Sunny."}, id="text_failed_as_is"),
        pytest.param(
            {"forecast": "sunny"},
            '{"error": "boom", "forecast": "sunny"}',
            True,
            {"error": "boom", "forecast": "sunny"},
            id="object_failed",
        ),
        pytest.param({"rows": ["secret"]}, '{"error": "blocked"}', True, {"error": "blocked"}, id="object_replaced"),
        pytest.param(
            {"error": {"code": 500, "detail": "x"}},
            '{"error": {"code": 500, "detail": "[redacted]"}}',
            True,
            {"error": {"code": 500, "detail": "[redacted]"}},
            id="failed_object_edited",
        ),
        pytest.param({"error": {"code": 500}}, '{"code": 500}', False, {"code": 500}, id="failed_object_recovered"),
        pytest.param({"error": "boom", "code": 7}, "worse", True, {"code": 7, "error": "worse"}, id="error_text"),
        pytest.param({"error": "boom", "code": 7}, None, True, {"code": 7, "error": "boom"}, id="error_kept"),
        pytest.param({"error": "boom", "code": 7}, "fine", False, {"code": 7, "result": "fine"}, id="error_recovered"),
        pytest.param({"error": "boom", "code": 7}, None, False, {"code": 7}, id="error_dropped"),
        pytest.param(
            {"result": "Sunny.", "unit": "C"}, "Foggy.", False, {"result": "Foggy.", "unit": "C"}, id="text_slot"
        ),
        pytest.param(
            {"result": "Sunny.", "unit": "C"}, "boom", True, {"unit": "C", "error": "boom"}, id="text_slot_failed"
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "hi"}], "isError": False},
            "HI",
            False,
            {"content": [{"type": "text", "text": "HI"}], "isError": False},
            id="mcp_text",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "hi"}], "isError": False, "structuredContent": {"a": 1}},
            ["down", {"type": "image", "data": "AA==", "mimeType": "image/png"}],
            True,
            {
                "content": [
                    {"type": "text", "text": "down"},
                    {"type": "image", "data": "AA==", "mimeType": "image/png"},
                ],
                "isError": True,
                "structuredContent": {"a": 1},
            },
            id="mcp_blocks_failed",
        ),
        pytest.param(
            types.FunctionResponse(response={"error": "boom"}, scheduling=types.FunctionResponseScheduling.SILENT),
            "fine",
            False,
            types.FunctionResponse(response={"result": "fine"}, scheduling=types.FunctionResponseScheduling.SILENT),
            id="function_response_recovered",
        ),
        pytest.param(
            types.FunctionResponse(response={"a": 1}),
            '{"a": 1, "error": "boom"}',
            True,
            types.FunctionResponse(response={"a": 1, "error": "boom"}),
            id="function_response_failed",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "hi"}], "isError": False},
            None,
            False,
            {"content": [], "isError": False},
            id="mcp_cleared",
        ),
    ],
)
def test_content_and_failure_edits_are_written_back_into_the_result(
    value: object, content: nemo_relay.Json, is_error: bool, expected: object
) -> None:
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add(value))

    restored = produced.restore({**encoded, "content": content, "is_error": is_error}, object)

    assert restored == expected
    assert type(restored) is type(expected)


@pytest.mark.parametrize(
    ("value", "edit", "error", "message"),
    [
        pytest.param("", {"is_error": True}, ValueError, "needs error text", id="text_failed_without_text"),
        pytest.param("Sunny.", {"content": ["Foggy."]}, TypeError, "only text content", id="text_parts"),
        pytest.param({"forecast": "sunny"}, {"content": "Foggy."}, ValueError, "is its JSON text", id="object_content"),
        pytest.param({"forecast": "sunny"}, {"content": None}, TypeError, "is its JSON text", id="object_null"),
        pytest.param(
            {"forecast": "sunny"},
            {"content": "blocked", "is_error": True},
            ValueError,
            "is its JSON text",
            id="object_failed_with_text",
        ),
        pytest.param(
            {"forecast": "sunny"},
            {"is_error": True},
            ValueError,
            "exactly when it is written as a failure",
            id="object_failed_as_sent",
        ),
        pytest.param(
            {"forecast": "sunny"},
            {"content": '{"forecast": "sunny", "error": "down"}'},
            ValueError,
            "exactly when it is written as a failure",
            id="object_recovered_with_an_error",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "hi"}], "isError": False},
            {"content": [1]},
            TypeError,
            "only text or content blocks",
            id="mcp_blocks",
        ),
        pytest.param({"error": "boom"}, {"content": ["x"]}, TypeError, "only text content", id="error_parts"),
        pytest.param(
            {"result": "Sunny."},
            {"content": None, "is_error": True},
            ValueError,
            "needs error text",
            id="text_slot_failed_without_text",
        ),
        pytest.param(types.Part(text="hi"), {"content": "Foggy."}, ValueError, "is its JSON text", id="part_content"),
        pytest.param(
            types.FunctionResponse(response=None),
            {"is_error": True},
            ValueError,
            "exactly when it is written as a failure",
            id="response_none",
        ),
        pytest.param(
            CodeExecutionResult(stdout="1\n"),
            {"content": "", "is_error": True},
            ValueError,
            "needs error text",
            id="code",
        ),
        pytest.param({}, {"schema": "nemo.relay.tool_result@2"}, ValueError, "versioned Relay", id="schema"),
        pytest.param({}, {"content": 1}, ValueError, "versioned Relay tool result", id="content_type"),
        pytest.param({}, {"is_error": "no"}, ValueError, "versioned Relay tool result", id="status_type"),
        pytest.param({}, {"result": {"kind": "json"}}, ValueError, "encoded ADK tool result value", id="no_value"),
        pytest.param({}, {"result": {"kind": "deque", "value": []}}, ValueError, "kind: 'deque'", id="kind"),
        pytest.param({}, {"result": {"kind": "json", "value": {}}}, ValueError, "kind: 'json'", id="json_value"),
        pytest.param({}, {"result": {"value": 1}}, TypeError, "needs a kind", id="no_kind"),
        pytest.param({}, {"result": {"kind": "bytes", "value": "***"}}, ValueError, "base64", id="binary"),
        pytest.param(
            {},
            {"result": {"kind": "typed", "type": "google.genai.types.Client", "value": {}}},
            ValueError,
            "Unknown type",
            id="unknown_type",
        ),
        pytest.param(
            {},
            {"result": {"kind": "dataclass", "type": _name(CodeExecutionInput), "value": {}}},
            ValueError,
            "Unknown type",
            id="unknown_dataclass",
        ),
        pytest.param(
            {},
            {"result": {"kind": "dataclass", "type": "google.genai.types.Part", "value": {}}},
            ValueError,
            "is not a dataclass",
            id="not_a_dataclass",
        ),
        pytest.param(
            {},
            {"result": {"kind": "typed", "type": "google.genai.types.Part", "value": 1}},
            ValueError,
            "valid dictionary",
            id="typed_value",
        ),
        pytest.param(
            {},
            {
                "result": {
                    "kind": "dataclass",
                    "type": _name(CodeExecutionResult),
                    "value": {"stdout": {"kind": "json", "value": "1\n"}},
                }
            },
            ValueError,
            "needs the fields",
            id="dataclass_fields",
        ),
    ],
)
def test_results_adk_cannot_hold_are_rejected(
    value: object, edit: nemo_relay.JsonObject, error: type[Exception], message: str
) -> None:
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add(value))

    with pytest.raises(error, match=message):
        produced.restore({**encoded, **edit}, object)


def test_a_content_edit_between_equal_numbers_and_booleans_is_written_back() -> None:
    value = {"content": [{"type": "text", "text": "hi", "annotations": {"priority": 1}}], "isError": False}
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add(value))
    edited: nemo_relay.Json = [{"type": "text", "text": "hi", "annotations": {"priority": True}}]

    restored = produced.restore({**encoded, "content": edited}, object)

    assert json.dumps(restored) == json.dumps({"content": edited, "isError": False})


def test_content_written_as_json_text_keeps_the_parts_it_left_unchanged() -> None:
    handle, rows = _Opaque(), [{"id": 1}, {"id": 2}]
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add({"status": "running", "handle": handle, "rows": rows}))
    content = json.loads(str(encoded["content"]))
    assert content == {"handle": f"<unserializable {_name(_Opaque)}>", "rows": rows, "status": "running"}

    restored = produced.restore({**encoded, "content": json.dumps({**content, "status": "done"})}, object)

    assert as_object(restored)["status"] == "done"
    assert as_object(restored)["handle"] is handle
    assert as_object(restored)["rows"] is rows


def test_an_edit_to_the_result_record_stands_when_the_content_is_left_as_sent() -> None:
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add({"result": "Sunny.", "unit": "C"}))

    _at(encoded, "result", "value")["result"] = {"kind": "json", "value": "Foggy."}

    assert produced.restore(encoded, object) == {"result": "Foggy.", "unit": "C"}


def test_json_null_content_clears_a_result_without_text_of_its_own() -> None:
    produced = ProducedResults()
    encoded = _over_the_wire(produced.add({"forecast": "sunny"}))

    assert produced.restore({**encoded, "content": "null"}, object) is None


def test_a_result_of_the_wrong_type_is_rejected() -> None:
    encoded = tool_result_json("hi")

    with pytest.raises(TypeError, match="restored a str where ADK expects a CodeExecutionResult"):
        ProducedResults().restore(encoded, CodeExecutionResult)


def test_arguments_come_back_as_adk_passed_them_unless_edited() -> None:
    arguments = {"location": "Paris", "when": object()}
    codec = AdkArgumentsCodec(arguments)
    projected = as_object(codec.to_json(arguments))

    assert codec.from_json(projected) is arguments
    edited = codec.from_json({**projected, "location": "Rome"})
    assert edited == {"location": "Rome", "when": arguments["when"]}
    assert edited["when"] is arguments["when"]
    with pytest.raises(TypeError, match="must be a JSON object"):
        codec.from_json(["Rome"])


class _Document(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, ser_json_bytes="base64")
    title: str
    data: bytes
    handle: _Opaque


def test_an_argument_edit_keeps_the_arguments_it_left_unchanged_as_adk_passed_them() -> None:
    arguments = {
        "query": "weather",
        "image": b"\x89PNG",
        "weather": _Weather(location="Paris", when=datetime.date(2026, 10, 22)),
        "report": _report("Paris"),
        "pair": (1, 2),
        "tags": {"warm", "dry"},
        "handle": _Opaque(),
    }
    codec = AdkArgumentsCodec(arguments)
    projected = _over_the_wire(codec.to_json(arguments))

    edited = codec.from_json({**projected, "query": "forecast"})

    assert edited["query"] == "forecast"
    for key in ("image", "weather", "report", "pair", "tags", "handle"):
        assert edited[key] is arguments[key]


def test_an_edit_inside_an_argument_keeps_the_parts_it_left_unchanged() -> None:
    handle = _Opaque()
    arguments = {"document": _Document(title="Draft", data=b"\x89PNG", handle=handle), "files": {(1, b"a"), (2, b"b")}}
    codec = AdkArgumentsCodec(arguments)
    projected = _over_the_wire(codec.to_json(arguments))
    document = {**as_object(projected["document"]), "title": "Final"}
    files = projected["files"]
    assert isinstance(files, list)

    edited = codec.from_json({**projected, "document": document, "files": [*files, [3, "c"]]})

    restored_document = as_object(edited["document"])
    assert restored_document["title"] == "Final"
    assert restored_document["data"] == b"\x89PNG"
    assert restored_document["handle"] is handle
    restored_files = edited["files"]
    assert isinstance(restored_files, list)
    assert sorted(restored_files[:2]) == [(1, b"a"), (2, b"b")]
    assert restored_files[2] == [3, "c"]


def test_an_edit_to_an_argument_list_keeps_each_item_it_left_unchanged_in_order() -> None:
    arguments = {"images": [b"aaaa", b"bbbb"], "files": [{"name": "a", "data": b"12"}, {"name": "b", "data": b"34"}]}
    codec = AdkArgumentsCodec(arguments)
    projected = _over_the_wire(codec.to_json(arguments))
    images, files = projected["images"], projected["files"]
    assert isinstance(images, list)
    assert isinstance(files, list)

    edited = codec.from_json(
        {"images": ["caption", *images], "files": [files[0], {**as_object(files[1]), "name": "renamed"}]}
    )

    assert edited["images"] == ["caption", b"aaaa", b"bbbb"]
    assert edited["files"] == [{"name": "a", "data": b"12"}, {"name": "renamed", "data": b"34"}]


def test_an_edit_to_an_argument_whose_keys_share_a_json_name_is_rejected() -> None:
    arguments = {"query": "weather", "counts": {1: "one", "1": "uno"}}
    codec = AdkArgumentsCodec(arguments)
    projected = _over_the_wire(codec.to_json(arguments))

    assert codec.from_json({**projected, "query": "forecast"})["counts"] is arguments["counts"]
    with pytest.raises(ValueError, match="keys share a JSON name"):
        codec.from_json({**projected, "counts": {"1": "eins"}})


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param({"limit": True, "exact": True}, id="number_to_boolean"),
        pytest.param({"limit": 1.0, "exact": True}, id="integer_to_float"),
        pytest.param({"limit": 1, "exact": 1}, id="boolean_to_number"),
    ],
)
def test_argument_edits_between_equal_numbers_and_booleans_reach_the_tool(edit: nemo_relay.JsonObject) -> None:
    arguments = {"limit": 1, "exact": True}
    codec = AdkArgumentsCodec(arguments)

    restored = codec.from_json(_over_the_wire(edit))

    assert restored is not arguments
    assert json.dumps(restored) == json.dumps(edit)


def test_arguments_holding_equal_sets_project_alike_and_pass_through() -> None:
    forward, backward = {1, 9}, {9, 1}
    assert list(forward) != list(backward)
    arguments = {"ids": forward}
    codec = AdkArgumentsCodec(arguments)

    projected = codec.to_json({"ids": backward})

    assert projected == codec.to_json(arguments)
    assert codec.from_json(_over_the_wire(projected)) is arguments


def test_code_input_edits_can_change_only_the_code() -> None:
    chart = File(name="chart.png", content=b"\x89PNG", mime_type="image/png")
    code_input = CodeExecutionInput(code="print(1)", input_files=[chart])
    codec = AdkCodeInputCodec(code_input)
    projected = as_object(codec.to_json(code_input))

    assert projected["input_files"] == [{"name": "chart.png", "content": "iVBORw==", "mime_type": "image/png"}]
    assert codec.from_json(projected) is code_input
    edited = codec.from_json({**projected, "code": "print(2)"})
    assert edited == CodeExecutionInput(code="print(2)", input_files=code_input.input_files)
    with pytest.raises(TypeError, match="only the code"):
        codec.from_json({**projected, "code": 2})
    with pytest.raises(ValueError, match="only the code"):
        codec.from_json({**projected, "code": "print(2)", "execution_id": "run-2"})


@pytest.mark.parametrize("code", ["print(1)", "print(2)"])
def test_code_input_edits_outside_the_code_are_rejected_between_equal_numbers_and_booleans(code: str) -> None:
    code_input = CodeExecutionInput(code="print(1)", input_files=[File(name="data.bin", content=b"x")])
    codec = AdkCodeInputCodec(code_input)
    projected = _over_the_wire(codec.to_json(code_input))
    files = projected["input_files"]
    assert isinstance(files, list)
    (file,) = files

    edited_file = {**as_object(file), "content": {"bytes": True}}
    with pytest.raises(ValueError, match="only the code"):
        codec.from_json({**projected, "code": code, "input_files": [edited_file]})
