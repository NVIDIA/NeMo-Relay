# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encode LangChain tool results, content, and failure status."""

from __future__ import annotations

import base64
import copy
import dataclasses
import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Annotated, Any, Final, NamedTuple, TypeAlias, TypeGuard, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    RemoveMessage,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
    messages_from_dict,
)
from langgraph.types import Command, Overwrite, Send, TimeoutPolicy
from pydantic import BaseModel, RootModel, TypeAdapter
from pydantic_core import to_jsonable_python

from nemo_relay._tool_result import TOOL_RESULT_SCHEMA, projected_tool_result
from nemo_relay.integrations._json_values import (
    SCALAR_KINDS,
    canonical,
    is_json_key,
    paired_items,
    relay_scalar,
    same_json,
    scalar_json,
    scalar_value,
    typed_json,
)
from nemo_relay.typed import Codec

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from nemo_relay import Json, JsonObject

ToolResultValue: TypeAlias = "ToolMessage | Command[Any] | list[ToolResultValue]"
_Visit: TypeAlias = Callable[[object], object]
_PYTHON_VALUES: Final = TypeAdapter(object)


def _type_name(cls: type[object]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


_PORTABLE_TYPES: Final[Annotated[Mapping[str, type[object]], "Classes a new call can rebuild from JSON alone."]] = {
    _type_name(cls): cls
    for cls in (
        AIMessage,
        AIMessageChunk,
        ChatMessage,
        ChatMessageChunk,
        FunctionMessage,
        FunctionMessageChunk,
        HumanMessage,
        HumanMessageChunk,
        RemoveMessage,
        SystemMessage,
        SystemMessageChunk,
        ToolMessage,
        ToolMessageChunk,
        Command,
        Overwrite,
        TimeoutPolicy,
    )
}
_CONTAINERS: Final[Mapping[str, Callable[[Iterable[object]], object]]] = {
    "list": list,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
}


def _is_json_object(value: object) -> TypeIs[JsonObject]:
    return isinstance(value, dict)


def _is_json_list(value: object) -> TypeIs[list[Json]]:
    return isinstance(value, list)


def _is_names(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(name, str) for name in cast(list[object], value))


def _is_tool_message_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Recognize LangChain message dictionaries and LangGraph role shorthand."""
    if not isinstance(value, dict):
        return False
    fields = cast(dict[str, object], value)
    return fields.get("role", fields.get("type")) == "tool"


def _is_update_pair(item: object) -> TypeGuard[tuple[str, object]]:
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)


def _stand_in(value: object) -> str:
    return f"<{_type_name(type(value))}>"


def _display(value: object) -> Json:
    """Return readable JSON. Parts with no JSON form show their type name."""
    path: set[int] = set()

    def shown(part: object) -> Json:
        if part is None or isinstance(part, bool):
            return part
        if isinstance(part, int):
            return relay_scalar(int(part))
        if isinstance(part, float):
            return relay_scalar(float(part))
        if isinstance(part, str):
            return relay_scalar(str.__str__(part))
        if isinstance(part, bytes | bytearray):
            return base64.b64encode(part).decode("ascii")
        if isinstance(part, Iterator) or id(part) in path:
            return _stand_in(part)
        path.add(id(part))
        try:
            return nested(part)
        finally:
            path.discard(id(part))

    def key_text(key: object) -> str:
        text = shown(key)
        return text if isinstance(text, str) else canonical(text)

    def nested(part: object) -> Json:
        if isinstance(part, Mapping):
            entries = cast(Mapping[object, object], part).items()
            return {key_text(key): shown(item) for key, item in entries}
        if isinstance(part, AbstractSet):
            return sorted((shown(member) for member in cast(AbstractSet[object], part)), key=canonical)
        if isinstance(part, Sequence):
            return [shown(item) for item in cast(Sequence[object], part)]
        if isinstance(part, Send):
            return {"node": shown(part.node), "arg": shown(part.arg), "timeout": shown(part.timeout)}
        if isinstance(part, BaseModel):
            return {key_text(key): shown(item) for key, item in BaseModel.__iter__(part)}
        if dataclasses.is_dataclass(part) and not isinstance(part, type):
            return {field.name: shown(getattr(part, field.name)) for field in dataclasses.fields(part)}
        # First dump leaves iterators alone. Only a value it returns unchanged gets a JSON dump.
        dumped: object = _PYTHON_VALUES.dump_python(part, mode="python", fallback=_stand_in)
        if dumped is part:
            return shown(to_jsonable_python(part, fallback=_stand_in))
        return shown(dumped)

    return shown(value)


def _declares_messages(cls: type[object]) -> bool:
    return any("messages" in inspect.get_annotations(base) for base in cls.__mro__)


def _map_write(write: object, visit: _Visit) -> object:
    """Return `messages` write with `visit` applied to each message."""
    if isinstance(write, Overwrite):
        inner = _map_write(write.value, visit)
        return write if inner is write.value else dataclasses.replace(write, value=inner)
    if isinstance(write, list | tuple):
        messages = cast(Sequence[object], write)
        items = [visit(item) for item in messages]
        if all(new is old for new, old in zip(items, messages, strict=True)):
            return write
        return items if isinstance(write, list) else tuple(items)
    return visit(write)


def _map_update(update: object, visit: _Visit) -> object:
    """Return state update with `visit` applied to its `messages` write."""
    if isinstance(update, dict):
        fields = cast(dict[object, object], update)
        if "messages" not in fields:
            return update
        write = _map_write(fields["messages"], visit)
        return update if write is fields["messages"] else {**fields, "messages": write}
    if isinstance(update, list | tuple):
        writes = cast(Sequence[object], update)
        pairs = [pair for pair in writes if _is_update_pair(pair)]
        if len(pairs) != len(writes):
            return _map_write(update, visit)
        mapped = [(key, _map_write(item, visit) if key == "messages" else item) for key, item in pairs]
        if all(new[1] is old[1] for new, old in zip(mapped, pairs, strict=True)):
            return update
        return mapped if isinstance(update, list) else tuple(mapped)
    if update is None or not _declares_messages(type(update)):
        return update
    write = getattr(update, "messages", None)
    mapped_write = _map_write(write, visit)
    if mapped_write is write:
        return update
    if isinstance(update, BaseModel):
        return update.model_copy(update={"messages": mapped_write})
    if dataclasses.is_dataclass(update) and not isinstance(update, type):
        return dataclasses.replace(update, messages=mapped_write)
    updated = copy.copy(update)
    object.__setattr__(updated, "messages", mapped_write)
    return updated


def _map_messages(result: ToolResultValue, visit: _Visit) -> ToolResultValue:
    """Return result with `visit` applied to each message. Unchanged parts stay same objects."""
    if isinstance(result, list):
        items = [_map_messages(item, visit) for item in result]
        return result if all(new is old for new, old in zip(items, result, strict=True)) else items
    if isinstance(result, Command):
        update = _map_update(result.update, visit)
        return result if update is result.update else dataclasses.replace(result, update=update)
    return cast(ToolMessage, visit(result))


def _message(result: ToolResultValue, tool_call_id: str | None) -> tuple[object, ToolMessage] | None:
    """Return tool message for call ID, with item it came from. With no ID, return the only tool message."""
    candidates: list[tuple[object, ToolMessage]] = []

    def collect(item: object) -> object:
        if isinstance(item, ToolMessage):
            candidates.append((item, item))
        elif _is_tool_message_dict(item):
            candidates.append((item, ToolMessage.model_validate(item)))
        return item

    _map_messages(result, collect)
    if tool_call_id is not None:
        return next((found for found in candidates if found[1].tool_call_id == tool_call_id), None)
    return candidates[0] if len(candidates) == 1 else None


def _content(message: ToolMessage) -> Json:
    """Return message content as JSON, with text parts as text blocks."""
    content = message.content
    if isinstance(content, list):
        return _display([{"type": "text", "text": part} if isinstance(part, str) else part for part in content])
    return _display(content)


def _is_content(content: Json) -> bool:
    return not isinstance(content, list) or all(isinstance(part, str | dict) for part in content)


def _rebind_tool_messages(value: object, source_id: str | None, tool_call_id: str, *, messages: bool = False) -> object:
    """Give replayed framework messages the current invocation's identity."""
    if isinstance(value, ToolMessage):
        if value.tool_call_id != source_id:
            return value
        return value.model_copy(update={"tool_call_id": tool_call_id, "id": None})
    if isinstance(value, Command):
        return dataclasses.replace(
            value,
            update=_rebind_tool_messages(value.update, source_id, tool_call_id, messages=True),
            resume=_rebind_tool_messages(value.resume, source_id, tool_call_id),
            goto=_rebind_tool_messages(value.goto, source_id, tool_call_id),
        )
    if isinstance(value, Send):
        return Send(value.node, _rebind_tool_messages(value.arg, source_id, tool_call_id), timeout=value.timeout)
    if isinstance(value, Overwrite):
        inner = _rebind_tool_messages(value.value, source_id, tool_call_id, messages=messages)
        return dataclasses.replace(value, value=inner)
    if isinstance(value, dict):
        fields = cast(dict[object, object], value)
        if messages and _is_tool_message_dict(fields) and fields.get("tool_call_id") == source_id:
            return {**fields, "tool_call_id": tool_call_id, "id": None}
        return {
            key: _rebind_tool_messages(item, source_id, tool_call_id, messages=key == "messages")
            for key, item in fields.items()
        }
    if isinstance(value, list | tuple):
        items = [
            _rebind_tool_messages(item, source_id, tool_call_id, messages=messages)
            for item in cast(Sequence[object], value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _validated(value: object) -> ToolResultValue:
    if isinstance(value, list):
        for item in cast(list[object], value):
            _validated(item)
        return cast(list[ToolResultValue], value)
    if not isinstance(value, ToolMessage | Command):
        raise TypeError("Expected a ToolMessage, Command, or a list of those values")
    return cast(ToolMessage | Command[Any], value)


def _object(value: Json, kind: str) -> JsonObject:
    if not _is_json_object(value):
        raise ValueError(f"Invalid encoded {kind} value")
    return value


def _items(value: Json, kind: str) -> list[Json]:
    if not _is_json_list(value):
        raise ValueError(f"Invalid encoded {kind} value")
    return value


class _Recorded(NamedTuple):
    """A result part with its JSON and records of its own parts."""

    value: object
    encoded: Json
    parts: Mapping[str, _Recorded]
    items: Sequence[_Recorded]


@dataclasses.dataclass(slots=True)
class _Encoding:
    """State of one result encoding."""

    path: Annotated[set[int], "Objects that hold current part."] = dataclasses.field(default_factory=set)
    replayable: Annotated[bool, "Whether a new call can rebuild result from JSON alone."] = True


class LangChainToolResultCodec(Codec[ToolResultValue]):
    """Encode tool results without losing messages or graph updates.

    Use one codec per tool call. With a call ID, only a matching `ToolMessage` provides content and status. Values with
    no JSON form stay with the call that made them.
    """

    def __init__(self, tool_call_id: str | None) -> None:
        self._tool_call_id = tool_call_id
        self._classes: dict[str, set[type[object]]] = {}
        self._refs: dict[int, int] = {}
        self._locals: dict[str, object] = {}
        self._produced: list[_Recorded] = []

    def to_json(self, value: ToolResultValue) -> JsonObject:
        """Encode the result with display content, failure status, and cache replay metadata."""
        found = _message(_validated(value), self._tool_call_id)
        encoding = _Encoding()
        recorded = self._encoded(value, encoding)
        self._produced.append(recorded)
        return {
            "schema": TOOL_RESULT_SCHEMA,
            "content": None if found is None else _content(found[1]),
            "is_error": found is not None and found[1].status == "error",
            "result": recorded.encoded,
            "cache": {
                "replayable": encoding.replayable and self._tool_call_id is not None,
                "tool_call_id": self._tool_call_id,
            },
        }

    def from_json(self, data: Json) -> ToolResultValue:
        """Restore a result, apply projection edits, and rebind messages to the current call."""
        if not _is_json_object(data) or data.get("schema") != TOOL_RESULT_SCHEMA:
            raise ValueError("Expected a versioned Relay tool result")
        if "result" not in data:
            raise ValueError("Expected an encoded LangChain tool result value")
        projected = projected_tool_result(data)
        if projected is None or not _is_content(projected[0]):
            raise ValueError("Invalid tool result content or error status")
        content, is_error = projected
        result = _validated(self._restored(data["result"]))
        source_id = self._tool_call_id
        cache = data.get("cache")
        if _is_json_object(cache) and "tool_call_id" in cache:
            cached_id = cache["tool_call_id"]
            if cached_id is not None and not isinstance(cached_id, str):
                raise ValueError("Invalid cached tool call ID")
            source_id = cached_id
        found = _message(result, source_id)
        if found is None:
            if content is not None or is_error:
                raise ValueError("Cannot project content or status without a matching ToolMessage")
        else:
            source, message = found
            changes: dict[str, object] = {}
            if not same_json(_content(message), content):
                changes["content"] = content if content is not None else ""
            if (message.status == "error") != is_error:
                changes["status"] = "error" if is_error else "success"
            if changes:
                updated = message.model_copy(update=changes)
                result = _map_messages(result, lambda item: updated if item is source else item)
        if self._tool_call_id is not None and source_id != self._tool_call_id:
            return _validated(_rebind_tool_messages(result, source_id, self._tool_call_id))
        return result

    def _encoded(self, value: object, encoding: _Encoding) -> _Recorded:
        scalar = scalar_json(value)
        if scalar is not None:
            return _Recorded(value, scalar, {}, ())
        if id(value) in encoding.path:
            return self._local(value, encoding)
        encoding.path.add(id(value))
        try:
            return self._structured(value, encoding)
        finally:
            encoding.path.discard(id(value))

    def _structured(self, value: object, encoding: _Encoding) -> _Recorded:
        """Record value by walking containers, models, and dataclasses."""
        cls = type(value)
        if cls is dict:
            fields = cast(dict[object, object], value)
            if all(is_json_key(key) for key in fields):
                return self._keyed(value, {"kind": "dict"}, cast(dict[str, object], fields).items(), encoding)
            pairs = [self._pair(key, item, encoding) for key, item in fields.items()]
            return _Recorded(value, {"kind": "dict_items", "value": [pair.encoded for pair in pairs]}, {}, pairs)
        if isinstance(value, Mapping):
            entries = cast(Mapping[object, object], value).items()
            pairs = [self._pair(key, item, encoding) for key, item in entries]
            header = {"kind": "mapping", "type": self._class_name(cls, encoding)}
            return _Recorded(value, {**header, "value": [pair.encoded for pair in pairs]}, {}, pairs)
        if _CONTAINERS.get(cls.__name__) is cls:
            items = [self._encoded(item, encoding) for item in cast(Iterable[object], value)]
            if isinstance(value, AbstractSet):
                items.sort(key=lambda part: canonical(part.encoded))
            return _Recorded(value, {"kind": cls.__name__, "value": [part.encoded for part in items]}, {}, items)
        if isinstance(value, Send) and cls is Send:
            arguments = {"node": value.node, "arg": value.arg, "timeout": value.timeout}
            inner = self._keyed(arguments, {"kind": "dict"}, arguments.items(), encoding)
            return _Recorded(value, {"kind": "send", "value": inner.encoded}, {"value": inner}, ())
        if isinstance(value, BaseModel) and not isinstance(value, RootModel):
            entries = list(BaseModel.__iter__(value))
            if all(is_json_key(key) for key, _ in entries):
                fields_set: list[Json] = [*sorted(value.model_fields_set)]
                header: JsonObject = {
                    "kind": "model",
                    "type": self._class_name(cls, encoding),
                    "fields_set": fields_set,
                }
                return self._keyed(value, header, entries, encoding)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = [(field.name, getattr(value, field.name)) for field in dataclasses.fields(value)]
            return self._keyed(value, {"kind": "dataclass", "type": self._class_name(cls, encoding)}, fields, encoding)
        dumped = typed_json(value)
        if dumped is None:
            return self._local(value, encoding)
        encoding.replayable = False
        return _Recorded(value, {"kind": "typed", "type": self._class_name(cls, encoding), "value": dumped}, {}, ())

    def _keyed(
        self, value: object, header: JsonObject, entries: Iterable[tuple[str, object]], encoding: _Encoding
    ) -> _Recorded:
        parts = {key: self._encoded(item, encoding) for key, item in entries}
        encoded: JsonObject = {**header, "value": {key: part.encoded for key, part in parts.items()}}
        return _Recorded(value, encoded, parts, ())

    def _pair(self, key: object, item: object, encoding: _Encoding) -> _Recorded:
        parts = (self._encoded(key, encoding), self._encoded(item, encoding))
        return _Recorded((key, item), [part.encoded for part in parts], {}, parts)

    def _class_name(self, cls: type[object], encoding: _Encoding) -> str:
        name = _type_name(cls)
        self._classes.setdefault(name, set()).add(cls)
        if _PORTABLE_TYPES.get(name) is not cls:
            encoding.replayable = False
        return name

    def _local(self, value: object, encoding: _Encoding) -> _Recorded:
        """Record a value with no JSON form and keep it for this call."""
        encoding.replayable = False
        ref = self._refs.setdefault(id(value), len(self._refs))
        encoded: JsonObject = {"kind": "local", "type": _type_name(type(value)), "ref": ref, "value": _display(value)}
        self._locals[canonical(encoded)] = value
        return _Recorded(value, encoded, {}, ())

    def _restored(self, encoded: Json) -> object:
        """Return produced result with this JSON, or rebuild from latest produced result."""
        key = canonical(encoded)
        for recorded in reversed(self._produced):
            if canonical(recorded.encoded) == key:
                return recorded.value
        return self._rebuilt(encoded, self._produced[-1] if self._produced else None)

    def _decoded(self, data: Json, recorded: _Recorded | None) -> object:
        if recorded is not None and same_json(data, recorded.encoded):
            return recorded.value
        return self._rebuilt(data, recorded)

    def _rebuilt(self, data: Json, recorded: _Recorded | None) -> object:
        """Rebuild a changed value. Unchanged parts stay the tool's own objects."""
        if not _is_json_object(data) or "value" not in data:
            raise ValueError("Expected an encoded LangChain tool result value")
        kind, value = data.get("kind"), data["value"]
        if not isinstance(kind, str):
            raise TypeError("An encoded LangChain tool result value needs a kind")
        if kind in SCALAR_KINDS:
            return scalar_value(kind, value)
        parts = recorded.parts if recorded is not None else {}
        items = recorded.items if recorded is not None else ()
        if (container := _CONTAINERS.get(kind)) is not None:
            return container(self._decoded_items(_items(value, kind), items))
        if kind == "dict":
            return {key: self._decoded(item, parts.get(key)) for key, item in _object(value, kind).items()}
        if kind == "dict_items":
            return dict(self._entries(_items(value, kind), items))
        if kind == "mapping":
            return self._mapping(self._class(data.get("type")), _items(value, kind), recorded)
        if kind == "send":
            return Send(**self._arguments(value, parts.get("value")))
        # Results cached by 0.11.0 release candidates use `command` and `message` kinds.
        if kind == "command":
            return Command(**self._arguments(value, None))
        if kind == "message":
            message = self._arguments(value, None)
            if message.keys() != {"type", "data"}:
                raise ValueError("A cached LangChain message needs `type` and `data`")
            return messages_from_dict([message])[0]
        if kind == "dataclass":
            return self._dataclass(self._class(data.get("type")), _object(value, kind), parts)
        if kind == "model":
            return self._model(self._class(data.get("type")), _object(value, kind), data.get("fields_set"), parts)
        if kind == "typed":
            return TypeAdapter(self._class(data.get("type"))).validate_python(value)
        if kind == "local":
            local = self._locals.get(canonical(data))
            if local is None:
                raise ValueError(f"{data.get('type')} in a LangChain tool result has no JSON form. Keep it unchanged.")
            return local
        raise ValueError(f"Unsupported LangChain tool result kind: {kind!r}")

    def _decoded_items(self, value: list[Json], items: Sequence[_Recorded]) -> list[object]:
        positions = paired_items(value, [item.encoded for item in items])
        return [
            self._decoded(item, None if position is None else items[position])
            for item, position in zip(value, positions, strict=True)
        ]

    def _entries(self, value: list[Json], items: Sequence[_Recorded]) -> list[tuple[object, object]]:
        entries: list[tuple[object, object]] = []
        positions = paired_items(value, [item.encoded for item in items])
        for entry, position in zip(value, positions, strict=True):
            recorded = None if position is None else items[position]
            if recorded is not None and same_json(entry, recorded.encoded):
                entries.append(cast(tuple[object, object], recorded.value))
                continue
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError("Each entry of a LangChain tool result dictionary needs a key and a value")
            key_record, item_record = recorded.items if recorded is not None else (None, None)
            entries.append((self._decoded(entry[0], key_record), self._decoded(entry[1], item_record)))
        if len({key for key, _ in entries}) != len(entries):
            raise ValueError("A LangChain tool result dictionary repeats a key")
        return entries

    def _mapping(self, cls: type[object], value: list[Json], recorded: _Recorded | None) -> object:
        name = cls.__qualname__
        if recorded is None or type(recorded.value) is not cls:
            raise ValueError(f"A {name} in a LangChain tool result is rebuilt only from the mapping it replaces")
        if not isinstance(recorded.value, dict):
            raise ValueError(f"A {name} in a LangChain tool result cannot take edited entries. Keep it unchanged.")
        entries = self._entries(value, recorded.items)
        rebuilt = copy.copy(cast(dict[object, object], recorded.value))
        rebuilt.clear()
        for key, item in entries:
            rebuilt[key] = item
        if list(rebuilt.items()) != entries:
            raise ValueError(f"A {name} in a LangChain tool result did not take the edited entries")
        return rebuilt

    def _arguments(self, data: Json, recorded: _Recorded | None) -> dict[str, Any]:
        """Decode constructor keyword arguments."""
        arguments = self._decoded(data, recorded)
        if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
            raise ValueError("Expected encoded keyword arguments in a LangChain tool result")
        return cast(dict[str, Any], arguments)

    def _dataclass(self, cls: type[object], value: JsonObject, parts: Mapping[str, _Recorded]) -> object:
        """Rebuild a dataclass from its fields without running `__init__`."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{_type_name(cls)} is not a dataclass")
        names = {field.name for field in dataclasses.fields(cls)}
        if value.keys() != names:
            raise ValueError(f"A {cls.__qualname__} in a LangChain tool result needs the fields {sorted(names)}")
        instance = object.__new__(cls)
        for name, item in value.items():
            object.__setattr__(instance, name, self._decoded(item, parts.get(name)))
        return instance

    def _model(
        self, cls: type[object], value: JsonObject, fields_set: Json, parts: Mapping[str, _Recorded]
    ) -> BaseModel:
        """Validate a model from its fields and restore its set fields."""
        if not issubclass(cls, BaseModel):
            raise TypeError(f"{_type_name(cls)} is not a pydantic model")
        if not _is_names(fields_set) or not set(fields_set) <= value.keys():
            raise ValueError(f"A {cls.__qualname__} in a LangChain tool result needs a list of its set fields")
        fields = {key: self._decoded(item, parts.get(key)) for key, item in value.items()}
        model = cls.model_validate(fields, by_alias=False, by_name=True)
        object.__setattr__(model, "__pydantic_fields_set__", set(fields_set))
        return model

    def _class(self, name: Json) -> type[object]:
        """Return class with this name from portable types or this call's results."""
        if not isinstance(name, str):
            raise TypeError(f"LangChain tool result type name must be text, not {name!r}")
        found = set(self._classes.get(name, ()))
        if (portable := _PORTABLE_TYPES.get(name)) is not None:
            found.add(portable)
        if len(found) > 1:
            raise ValueError(f"A LangChain tool result holds distinct classes named {name}")
        if not found:
            raise ValueError(f"Only the call that returned {name} can rebuild it")
        (cls,) = found
        return cls
