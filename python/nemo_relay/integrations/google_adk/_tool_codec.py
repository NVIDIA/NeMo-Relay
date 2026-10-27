# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry ADK tool calls through managed tool pipeline in Relay's shared tool-result format."""

from __future__ import annotations

import base64
import dataclasses
import functools
import json
from collections.abc import Hashable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Final, NamedTuple, TypeVar

from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult, File
from google.genai import types
from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python
from typing_extensions import TypeIs, override

from nemo_relay._tool_result import TOOL_RESULT_SCHEMA, projected_tool_result
from nemo_relay.integrations._json_values import canonical, paired_items, same_json
from nemo_relay.integrations.google_adk._serialization import (
    by_json_key,
    holds_iterator,
    is_ordered,
    is_plain_dataclass,
    python_dump,
    sorted_members,
    unserializable,
    wire_json,
)
from nemo_relay.typed import Codec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from nemo_relay import Json, JsonObject

_T = TypeVar("_T")
_K = TypeVar("_K", bound=Hashable)
_CODE_ONLY: Final = "NeMo Relay intercepts can change only the code of an ADK code execution"
_JSON_CONTENT: Final = "The content of an ADK tool result without text of its own is its JSON text"
_JSON_STATUS: Final = (
    "The JSON text of an ADK tool result holds a truthy error, or an MCP isError, "
    "exactly when it is written as a failure"
)
_FAILED_OUTCOMES: Final = frozenset({types.Outcome.OUTCOME_FAILED, types.Outcome.OUTCOME_DEADLINE_EXCEEDED})
_PRIMITIVES: Final = (str, int, float, bool)
_CONTAINERS: Final[Mapping[str, Callable[[Iterable[object]], object]]] = {
    "list": list,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
}
_PYTHON_VALUES: Final = TypeAdapter(object)
_BINARY: Final[Mapping[str, Callable[[bytes], object]]] = {"bytes": bytes, "bytearray": bytearray}


def _type_name(cls: type[object]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


_KNOWN_TYPES: Final[
    Annotated[
        Mapping[str, type[object]],
        "Genai model and enum types, plus ADK's code execution result types, by qualified name.",
    ]
] = {
    _type_name(cls): cls
    for cls in (
        *(
            member
            for member in vars(types).values()
            if isinstance(member, type) and member.__module__ == types.__name__ and issubclass(member, BaseModel | Enum)
        ),
        CodeExecutionResult,
        File,
    )
}


def _is_json_object(value: object) -> TypeIs[JsonObject]:
    return isinstance(value, dict)


def _is_str_dict(value: object) -> TypeIs[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _is_primitive(value: object) -> TypeIs[str | int | float | bool | None]:
    return value is None or type(value) in _PRIMITIVES


def _is_plain_dict(value: object) -> TypeIs[dict[str, object]]:
    return type(value) is dict and _is_str_dict(value)


def _is_container(value: object) -> TypeIs[list[object] | tuple[object, ...] | set[object] | frozenset[object]]:
    return _CONTAINERS.get(type(value).__name__) is type(value)


def _is_binary(value: object) -> TypeIs[bytes | bytearray]:
    return _BINARY.get(type(value).__name__) is type(value)


def _round_trip_dump(value: object) -> object:
    return _PYTHON_VALUES.dump_python(value, mode="python", round_trip=True)


class _IteratorScan:
    """Whether a result holds an iterator anywhere.

    Scans result only when an object in it first needs the answer.
    """

    def __init__(self, result: object) -> None:
        self._result = result

    @functools.cached_property
    def found(self) -> bool:
        return holds_iterator(_round_trip_dump(self._result))


def _is_block(value: object) -> TypeIs[JsonObject]:
    return _is_json_object(value) and isinstance(value.get("type"), str)


def _is_block_list(value: object) -> TypeIs[list[JsonObject]]:
    return isinstance(value, list) and all(_is_block(item) for item in value)


def _mcp_result(value: dict[str, object]) -> tuple[list[JsonObject], bool] | None:
    """Return content blocks and failure flag of a result shaped like an ADK MCP tool dump.

    None for any other result.
    """
    content, is_error = value.get("content"), value.get("isError", False)
    if _is_block_list(content) and isinstance(is_error, bool):
        return content, is_error
    return None


def _has_own_text(value: dict[str, object]) -> bool:
    """Return whether a result object holds its own text.

    True when its `error` is non-empty text, or its `result` is text and its `error` is falsy.
    """
    error = value.get("error")
    return (isinstance(error, str) and bool(error)) or (not error and isinstance(value.get("result"), str))


def _fails(value: object) -> bool:
    """Return whether a result is a failure.

    MCP result fails when its `isError` is true. Any other object fails when its `error` is truthy. A non-object never
    fails.
    """
    if not _is_str_dict(value):
        return False
    mcp = _mcp_result(value)
    return mcp[1] if mcp is not None else bool(value.get("error"))


def _content_json(value: object) -> str:
    """Return JSON text the model reads for a result without its own text."""
    return json.dumps(wire_json(value), ensure_ascii=False, sort_keys=True)


def _projection(value: object) -> tuple[Json, bool]:
    """Return content the model reads, and whether result is a failure.

    Follow ADK's convention for each result type.
    """
    if isinstance(value, CodeExecutionResult):
        return value.stderr or value.stdout, bool(value.stderr)
    if isinstance(value, types.CodeExecutionResult):
        return value.output, value.outcome in _FAILED_OUTCOMES
    if isinstance(value, types.FunctionResponse):
        return _projection(value.response)
    if isinstance(value, str):
        return value, False
    if _is_str_dict(value):
        if (mcp := _mcp_result(value)) is not None:
            blocks, failed = mcp
            return [*blocks], failed
        error, result = value.get("error"), value.get("result")
        if isinstance(error, str) and error:
            return error, True
        if not error and isinstance(result, str):
            return result, False
    return _content_json(value), _fails(value)


def _content_text(content: Json) -> str:
    if content is None:
        return ""
    if not isinstance(content, str):
        raise TypeError("NeMo Relay can write only text content back to an ADK tool result")
    return content


def _error_text(content: Json, existing: object = None) -> str:
    """Return intercept's `content` as failure text, or `existing` error text when `content` is None.

    Raise `TypeError` for non-text content, `ValueError` when no error text exists.
    """
    if isinstance(content, str) and content:
        return content
    if content is None and isinstance(existing, str) and existing:
        return existing
    if content is not None and not isinstance(content, str):
        raise TypeError("NeMo Relay can write only text content back to an ADK tool result")
    raise ValueError("A failed ADK tool result needs error text")


def _blocks(content: Json) -> list[JsonObject]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise TypeError("NeMo Relay can write only text or content blocks back to an MCP tool result")
    blocks: list[JsonObject] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif _is_block(item):
            blocks.append(item)
        else:
            raise TypeError("NeMo Relay can write only text or content blocks back to an MCP tool result")
    return blocks


def _projected_object(result: dict[str, object], content: Json, is_error: bool) -> dict[str, object]:
    """Return result object with content and failure status written in per ADK's conventions."""
    if _mcp_result(result) is not None:
        return {**result, "content": _blocks(content), "isError": is_error}
    rest = {key: item for key, item in result.items() if key not in ("error", "result")}
    if is_error:
        return {**rest, "error": _error_text(content, result.get("error"))}
    if content is None:
        return rest
    return {**rest, "result": _content_text(content)}


def _projected_value(result: object, content: Json, is_error: bool) -> object:
    """Rebuild a result without its own text from content holding its new JSON text.

    Raise `TypeError` when content is not text. Raise `ValueError` when text is not JSON, or rebuilt result's failure
    state differs from `is_error`.
    """
    if not isinstance(content, str):
        raise TypeError(_JSON_CONTENT)
    try:
        written: Json = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(_JSON_CONTENT) from error
    restored = _restored(written, wire_json(result), result, wire_json)
    if _fails(restored) != is_error:
        raise ValueError(_JSON_STATUS)
    return restored


def _projected(result: object, content: Json, is_error: bool) -> object:
    """Return result with intercept's content and failure status written back in."""
    if isinstance(result, CodeExecutionResult):
        text = _content_text(content)
        if not is_error:
            return dataclasses.replace(result, stdout=text, stderr="")
        if not text:
            raise ValueError("A failed ADK code execution needs error text")
        return dataclasses.replace(result, stderr=text)
    if isinstance(result, types.CodeExecutionResult):
        outcome = result.outcome
        if (outcome in _FAILED_OUTCOMES) != is_error:
            outcome = types.Outcome.OUTCOME_FAILED if is_error else types.Outcome.OUTCOME_OK
        output = None if content is None else _content_text(content)
        return result.model_copy(update={"output": output, "outcome": outcome})
    if isinstance(result, types.FunctionResponse):
        response = _projected(result.response, content, is_error)
        if not (response is None or _is_str_dict(response)):
            raise ValueError("The response of an ADK function response is a JSON object")
        return result.model_copy(update={"response": response})
    if isinstance(result, str):
        if is_error:
            return {"error": _error_text(content)}
        return _content_text(content)
    if _is_str_dict(result) and (_mcp_result(result) is not None or _has_own_text(result)):
        return _projected_object(result, content, is_error)
    return _projected_value(result, content, is_error)


class _Recorded(NamedTuple):
    """One part of a recorded result.

    Holds tool's own value, its encoding, and its parts other than JSON values.
    """

    value: object
    encoded: JsonObject
    fields: Mapping[str, _Recorded]
    items: Sequence[tuple[JsonObject, _Recorded | None]]


def _recorded_leaf(value: object, encoded: JsonObject) -> _Recorded:
    return _Recorded(value, encoded, {}, ())


class _Produced(NamedTuple):
    """One result the tool produced.

    Holds result's record, its encoding key, and content and status sent with it.
    """

    key: str
    recorded: _Recorded
    content: Json
    is_error: bool


class ProducedResults:
    """Results of one ADK tool call, kept for that call alone.

    Hand back tool's own objects where pipeline left them unchanged.
    """

    def __init__(self) -> None:
        self._classes: dict[str, set[type[object]]] = {}
        self._produced: list[_Produced] = []
        self._refs: dict[int, int] = {}
        self._unserializable: dict[str, object] = {}

    def add(self, result: object) -> JsonObject:
        """Return result in Relay's shared tool-result format, and remember each part it holds."""
        content, is_error = _projection(result)
        recorded = self._recorded(result, _IteratorScan(result))
        self._produced.append(_Produced(canonical(recorded.encoded), recorded, content, is_error))
        return {"schema": TOOL_RESULT_SCHEMA, "content": content, "is_error": is_error, "result": recorded.encoded}

    def restore(self, data: Json, result_type: type[_T]) -> _T:
        """Return result ADK receives for pipeline's tool result.

        Start from latest produced result with same encoding, or rebuild result from its encoding over latest produced
        one. Write tool result's content and failure status in when they differ from what call sent. Raise `ValueError`
        for malformed tool result, `TypeError` for a value of wrong type in it or a restored value that is not a
        `result_type`.
        """
        projected = projected_tool_result(data)
        if not _is_json_object(data) or projected is None:
            raise ValueError(
                "Expected a versioned Relay tool result with text or block content and a boolean error status"
            )
        content, is_error = projected
        encoded = data["result"]
        key = canonical(encoded)
        matched = next((produced for produced in reversed(self._produced) if produced.key == key), None)
        if matched is not None:
            result, sent = matched.recorded.value, (matched.content, matched.is_error)
        elif self._produced:
            latest = self._produced[-1]
            result, sent = self._rebuilt(encoded, latest.recorded), (latest.content, latest.is_error)
        else:
            result = self._rebuilt(encoded, None)
            sent = _projection(result)
        unchanged = sent[1] == is_error and same_json(sent[0], content)
        restored = result if unchanged else _projected(result, content, is_error)
        if not isinstance(restored, result_type):
            raise TypeError(
                f"NeMo Relay restored a {type(restored).__name__} where ADK expects a {result_type.__name__}"
            )
        return restored

    def _part(self, value: object, scan: _IteratorScan) -> tuple[JsonObject, _Recorded | None]:
        """Return part's encoding and record. Record is None for a JSON scalar."""
        if _is_primitive(value):
            return {"kind": "json", "value": value}, None
        recorded = self._recorded(value, scan)
        return recorded.encoded, recorded

    def _recorded(self, value: object, scan: _IteratorScan) -> _Recorded:
        """Record a value by walking its standard data types and dataclasses.

        Encode any other object as pydantic dumps it for rebuild.
        """
        if _is_primitive(value):
            return _recorded_leaf(value, {"kind": "json", "value": value})
        if _is_plain_dict(value):
            return self._keyed(value, {"kind": "dict"}, value.items(), scan)
        if _is_container(value):
            items = [self._part(item, scan) for item in value]
            if isinstance(value, set | frozenset):
                items.sort(key=lambda part: canonical(part[0]))
            return _Recorded(value, {"kind": type(value).__name__, "value": [part[0] for part in items]}, {}, items)
        if _is_binary(value):
            return _recorded_leaf(
                value, {"kind": type(value).__name__, "value": base64.b64encode(value).decode("ascii")}
            )
        cls = type(value)
        name = _type_name(cls)
        self._classes.setdefault(name, set()).add(cls)
        if is_plain_dataclass(cls):
            fields = ((field.name, getattr(value, field.name)) for field in dataclasses.fields(cls))
            return self._keyed(value, {"kind": "dataclass", "type": name}, fields, scan)
        return _recorded_leaf(value, self._object(value, name, scan))

    def _object(self, value: object, name: str, scan: _IteratorScan) -> JsonObject:
        """Return pydantic round-trip JSON for an object.

        Return per-call reference to object instead when part of it has no JSON form or holds an iterator.
        """
        if not (scan.found and holds_iterator(_round_trip_dump(value))):
            unknown: list[object] = []

            def stand_in(part: object) -> str:
                unknown.append(part)
                return unserializable(part)

            dumped: Json = to_jsonable_python(value, round_trip=True, fallback=stand_in)
            if not unknown:
                return {"kind": "typed", "type": name, "value": dumped}
        ref = self._refs.setdefault(id(value), len(self._refs))
        encoded: JsonObject = {"kind": "unserializable", "type": name, "ref": ref, "value": wire_json(value)}
        self._unserializable[canonical(encoded)] = value
        return encoded

    def _keyed(
        self, value: object, header: JsonObject, entries: Iterable[tuple[str, object]], scan: _IteratorScan
    ) -> _Recorded:
        encodings: JsonObject = {}
        fields: dict[str, _Recorded] = {}
        for key, item in entries:
            encodings[key], part = self._part(item, scan)
            if part is not None:
                fields[key] = part
        return _Recorded(value, {**header, "value": encodings}, fields, ())

    def _decode(self, data: Json, recorded: _Recorded | None) -> object:
        """Return tool's own value when pipeline left its encoding unchanged, else rebuild value from encoding."""
        if recorded is not None and same_json(data, recorded.encoded):
            return recorded.value
        return self._rebuilt(data, recorded)

    def _rebuilt(self, data: Json, recorded: _Recorded | None) -> object:
        """Rebuild a changed value from its encoding.

        Keep each part pipeline left unchanged as tool's own object. Raise `TypeError` for non-text kind, `ValueError`
        for unknown or malformed encoding.
        """
        if not _is_json_object(data) or "value" not in data:
            raise ValueError("Expected an encoded ADK tool result value")
        kind, value = data.get("kind"), data["value"]
        if not isinstance(kind, str):
            raise TypeError("An encoded ADK tool result value needs a kind")
        if kind == "json" and _is_primitive(value):
            return value
        fields = recorded.fields if recorded is not None else {}
        if kind == "dict" and _is_json_object(value):
            return {key: self._decode(item, fields.get(key)) for key, item in value.items()}
        if (container := _CONTAINERS.get(kind)) is not None and isinstance(value, list):
            items = recorded.items if recorded is not None else ()
            positions = paired_items(value, [encoding for encoding, _ in items])
            return container(
                self._decode(item, None if position is None else items[position][1])
                for item, position in zip(value, positions, strict=True)
            )
        if (binary := _BINARY.get(kind)) is not None and isinstance(value, str):
            return binary(base64.b64decode(value, validate=True))
        if kind == "typed":
            return TypeAdapter(self._class(data.get("type"))).validate_python(value)
        if kind == "dataclass" and _is_json_object(value):
            return self._dataclass(self._class(data.get("type")), value, fields)
        if kind == "unserializable":
            try:
                return self._unserializable[canonical(data)]
            except KeyError as error:
                raise ValueError(
                    f"An ADK tool result {data.get('type')!r} has no JSON form to rebuild from, "
                    "so it must be the unchanged record of an object the call's results hold"
                ) from error
        raise ValueError(f"Unsupported ADK tool result kind: {kind!r}")

    def _class(self, name: Json) -> type[object]:
        """Return the one class an encoded object names.

        Look name up among genai types, ADK types, and classes the call recorded. Raise `TypeError` when name is not
        text, `ValueError` when it matches no class or several.
        """
        if not isinstance(name, str):
            raise TypeError(f"An encoded ADK tool result names its type with {name!r}, which is not text")
        found = set(self._classes.get(name, ()))
        if (known := _KNOWN_TYPES.get(name)) is not None:
            found.add(known)
        if len(found) > 1:
            raise ValueError(f"An ADK tool result holds distinct classes named {name}")
        if not found:
            raise ValueError(f"Unknown type in an ADK tool result: {name!r}")
        (cls,) = found
        return cls

    def _dataclass(self, cls: type[object], fields: JsonObject, recorded: Mapping[str, _Recorded]) -> object:
        """Rebuild a dataclass from recorded fields without running its constructor.

        Raise `ValueError` when recorded fields differ from dataclass fields.
        """
        if not is_plain_dataclass(cls):
            raise ValueError(f"{_type_name(cls)} is not a dataclass")
        names = {field.name for field in dataclasses.fields(cls)}
        if fields.keys() != names:
            raise ValueError(f"A {cls.__qualname__} in an ADK tool result needs the fields {sorted(names)}")
        instance = object.__new__(cls)
        for name, item in fields.items():
            object.__setattr__(instance, name, self._decode(item, recorded.get(name)))
        return instance


def tool_result_json(value: object) -> JsonObject:
    """Return result in Relay's shared tool-result format."""
    return ProducedResults().add(value)


def _arguments_json(arguments: Mapping[str, object]) -> JsonObject:
    return {key: wire_json(value) for key, value in arguments.items()}


def _restored_object(
    edited: JsonObject, projected: JsonObject, source: Mapping[_K, object], json_of: Callable[[object], Json]
) -> dict[str, object]:
    parts = by_json_key(source)
    return {
        key: _restored(item, projected[key], parts[key], json_of) if key in parts else item
        for key, item in edited.items()
    }


def _restored(edited: Json, projected: Json, original: object, json_of: Callable[[object], Json]) -> object:
    """Return `original` when `edited` leaves its JSON unchanged.

    Otherwise return `edited` with each unchanged part restored from `original`.
    """
    if same_json(edited, projected):
        return original
    source = original if isinstance(original, Mapping) or is_ordered(original) else python_dump(original)
    if _is_json_object(edited) and _is_json_object(projected) and isinstance(source, Mapping):
        return _restored_object(edited, projected, source, json_of)
    if isinstance(edited, list) and isinstance(projected, list) and is_ordered(source):
        originals = sorted_members(source, json_of) if isinstance(source, AbstractSet) else [*source]
        return [
            item if position is None else _restored(item, projected[position], originals[position], json_of)
            for item, position in zip(edited, paired_items(edited, projected), strict=True)
        ]
    return edited


class AdkArgumentsCodec(Codec[dict[str, Any]]):
    """Encode tool arguments as JSON.

    Pass each part an intercept leaves unchanged to the tool as ADK passed it.
    """

    def __init__(self, arguments: dict[str, Any]) -> None:
        self._arguments = arguments
        self._projected = _arguments_json(arguments)

    @override
    def to_json(self, value: dict[str, Any]) -> Json:
        return _arguments_json(value)

    @override
    def from_json(self, data: Json) -> dict[str, Any]:
        if not _is_json_object(data):
            raise TypeError("ADK tool arguments must be a JSON object")
        if same_json(data, self._projected):
            return self._arguments
        return _restored_object(data, self._projected, self._arguments, wire_json)


class AdkCodeInputCodec(Codec[CodeExecutionInput]):
    """Encode code execution input as JSON. Accept intercept edits to code only."""

    def __init__(self, code_input: CodeExecutionInput) -> None:
        self._code_input = code_input
        self._projected = wire_json(code_input)

    @override
    def to_json(self, value: CodeExecutionInput) -> Json:
        return wire_json(value)

    @override
    def from_json(self, data: Json) -> CodeExecutionInput:
        if same_json(data, self._projected):
            return self._code_input
        code = data.get("code") if _is_json_object(data) else None
        if not isinstance(code, str):
            raise TypeError(_CODE_ONLY)
        edited = dataclasses.replace(self._code_input, code=code)
        if not same_json(wire_json(edited), data):
            raise ValueError(_CODE_ONLY)
        return edited
