# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only content snapshots for normalized worker annotation dictionaries.

Selection copies only the selected message/content/output into immutable local
storage. Entries and tool views share that storage; Python payload values are
independent copies on access. No host RPC or main ``nemo_relay`` extension is used.
Locations describe the normalized snapshot, not edit handles or provider offsets.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

ContentKind: TypeAlias = Literal[
    "text", "refusal", "tool_call", "tool_result", "image_url", "image", "audio", "file", "provider_native"
]
ContentRoot: TypeAlias = Literal["request_message", "message", "response", "content"]
ContentField: TypeAlias = Literal[
    "content", "message", "tool_calls", "tool_call_item", "output", "value", "native_content"
]
PayloadKind: TypeAlias = Literal["null", "string", "json", "content"]

_NON_TEXT = frozenset(
    {
        "reasoning",
        "thinking",
        "redacted_thinking",
        "tool",
        "function",
        "tool_use",
        "mcp_tool_use",
        "server_tool_use",
        "tool_result",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "image",
        "image_url",
        "input_image",
        "audio",
        "input_audio",
        "file",
        "input_file",
        "document",
    }
)


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    raise ValueError("normalized content must contain JSON values and string object keys")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a normalized object")
    return value


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a normalized string")
    return value


def _optional_string(value: Any) -> str | None:
    return None if value is None else _string(value)


def _native_role(message: Mapping[str, Any]) -> str | None:
    if message.get("role") != "provider_native":
        return None
    value = message.get("value")
    if not isinstance(value, Mapping) or ("type" in value and value["type"] != "message"):
        return None
    role, provider, kind = value.get("role"), message.get("provider"), message.get("kind")
    if not isinstance(role, str):
        return None
    recognized = (
        (provider == "openai_chat" and kind == role and role in {"system", "developer", "user", "assistant"})
        or (provider == "anthropic_messages" and kind == role and role in {"system", "user", "assistant"})
        or (
            provider == "openai_responses"
            and kind == "message"
            and role in {"system", "developer", "user", "assistant"}
        )
    )
    return role if recognized else None


def _native_text(kind: str | None, value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    native_type = value.get("type")
    if value.get("thought") is True or kind in _NON_TEXT or (isinstance(native_type, str) and native_type in _NON_TEXT):
        return None
    text = value.get("text")
    return text if isinstance(text, str) else value.get("refusal") if isinstance(value.get("refusal"), str) else None


@dataclass(frozen=True, slots=True)
class ContentLocation:
    """Read-time normalized location; message_index is set for request selections."""

    root: ContentRoot
    field: ContentField
    index: int | None = None
    message_index: int | None = None


@dataclass(frozen=True, slots=True)
class PayloadView:
    """Original tool payload representation, without JSON parsing or rendering."""

    kind: PayloadKind
    _value: Any = field(repr=False)

    @property
    def value(self) -> Any:
        """Return an independent Python value; JSON strings retain kind json."""
        return _thaw(self._value)


@dataclass(frozen=True, slots=True)
class ToolCallView:
    """Read-only invocation, preserving item and correlation identifiers."""

    id: str | None
    call_id: str | None
    name: str
    arguments: PayloadView


@dataclass(frozen=True, slots=True)
class ToolResultView:
    """Read-only result whose complete payload is excluded from default text."""

    id: str | None
    call_id: str | None
    name: str | None
    is_error: bool | None
    payload: PayloadView


@dataclass(frozen=True, slots=True)
class ContentEntry:
    """Typed content entry sharing immutable snapshot data."""

    kind: ContentKind
    location: ContentLocation
    text: str | None = None
    tool_call: ToolCallView | None = None
    tool_result: ToolResultView | None = None
    provider: str | None = None
    native_kind: str | None = None
    _value: Any = field(default=None, repr=False)

    @property
    def value(self) -> Any:
        """Return independent media/native data, or None for text and tools."""
        return _thaw(self._value)


def _part(part: Any, location: ContentLocation) -> ContentEntry:
    part = _object(part)
    kind = _string(part.get("type"))
    if kind in {"text", "refusal"}:
        return ContentEntry(kind, location, text=_string(part.get(kind)))
    if kind == "tool_use":
        return ContentEntry(
            "tool_call",
            location,
            tool_call=ToolCallView(
                _string(part.get("id")), None, _string(part.get("name")), PayloadView("json", part.get("input"))
            ),
        )
    if kind == "tool_result":
        is_error = part.get("is_error")
        if is_error is not None and not isinstance(is_error, bool):
            raise ValueError("tool result is_error must be a boolean or None")
        return ContentEntry(
            "tool_result",
            location,
            tool_result=ToolResultView(
                None, _string(part.get("tool_use_id")), None, is_error, PayloadView("json", part.get("content"))
            ),
        )
    if kind in {"image_url", "image", "audio", "file"}:
        if kind not in part:
            raise ValueError(f"missing normalized {kind} payload")
        if kind == "image_url":
            image = _object(part[kind])
            value = {"url": _string(image.get("url"))}
            if image.get("detail") is not None:
                value["detail"] = _string(image["detail"])
            return ContentEntry(kind, location, _value=MappingProxyType(value))
        return ContentEntry(kind, location, _value=part[kind])
    if kind == "provider_native":
        return ContentEntry(
            "provider_native",
            location,
            provider=_string(part.get("provider")),
            native_kind=_string(part.get("kind")),
            _value=part.get("value"),
        )
    raise ValueError(f"unknown normalized content type: {kind!r}")


def _content_parts(
    content: Any, root: ContentRoot, field_name: ContentField, message_index: int | None
) -> Iterator[ContentEntry]:
    if isinstance(content, str):
        yield ContentEntry("text", ContentLocation(root, field_name, message_index=message_index), text=content)
    elif isinstance(content, tuple):
        for index, part in enumerate(content):
            yield _part(part, ContentLocation(root, field_name, index, message_index))
    else:
        raise ValueError("normalized content must be a string or content-part list")


@dataclass(frozen=True, slots=True, init=False)
class MessageView:
    """Read-only snapshot selected with module helpers or from_message/from_content."""

    _source: Any = field(repr=False)
    _mode: Literal["message", "response", "content"]
    _message_index: int | None

    def __init__(self) -> None:
        raise TypeError("use selection helpers or MessageView.from_message/from_content")

    @classmethod
    def _snapshot(
        cls, value: Any, mode: Literal["message", "response", "content"], message_index: int | None = None
    ) -> MessageView:
        view = object.__new__(cls)
        object.__setattr__(view, "_source", _freeze(value))
        object.__setattr__(view, "_mode", mode)
        object.__setattr__(view, "_message_index", message_index)
        # Reject invalid standalone input now, rather than during later traversal.
        for _ in view.parts():
            pass
        return view

    @classmethod
    def from_message(cls, message: Mapping[str, Any]) -> MessageView:
        """Snapshot a normalized history item, preserving role/tool semantics."""
        return cls._snapshot(_object(message), "message")

    @classmethod
    def from_content(cls, content: str | list[Any]) -> MessageView:
        """Snapshot standalone normalized content, including instructions."""
        return cls._snapshot(content, "content")

    def parts(self) -> Iterator[ContentEntry]:
        """Iterate normalized content order, then separate assistant/response calls."""
        if self._mode == "content":
            yield from _content_parts(self._source, "content", "content", None)
            return
        source = _object(self._source)
        root: ContentRoot = (
            "response"
            if self._mode == "response"
            else ("request_message" if self._message_index is not None else "message")
        )
        if self._mode == "response":
            if source.get("message") is not None:
                yield from _content_parts(source["message"], root, "message", None)
        else:
            yield from self._message_parts(source, root)
        if self._mode == "response" or source.get("role") == "assistant":
            calls = source.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, tuple):
                    raise ValueError("tool_calls must be a list or None")
                for index, call in enumerate(calls):
                    call = _object(call)
                    function = call if self._mode == "response" else _object(call.get("function"))
                    arguments = function.get("arguments")
                    payload = (
                        PayloadView("json", arguments)
                        if self._mode == "response"
                        else (PayloadView("string", _string(arguments)))
                    )
                    yield ContentEntry(
                        "tool_call",
                        ContentLocation(root, "tool_calls", index, self._message_index),
                        tool_call=ToolCallView(_string(call.get("id")), None, _string(function.get("name")), payload),
                    )

    def _message_parts(self, source: Mapping[str, Any], root: ContentRoot) -> Iterator[ContentEntry]:
        role = _string(source.get("role"))
        location = ContentLocation(root, "content", message_index=self._message_index)
        if role in {"system", "developer", "user", "assistant"}:
            content = source.get("content")
            if role != "assistant" or content is not None:
                yield from _content_parts(content, root, "content", self._message_index)
        elif role in {"tool", "function", "tool_result"}:
            if role == "tool_result":
                location = ContentLocation(root, "output", message_index=self._message_index)
            content = source.get("output" if role == "tool_result" else "content")
            payload_kind: PayloadKind = (
                "json"
                if role == "tool_result"
                else "content"
                if role == "tool"
                else ("null" if content is None else "string")
            )
            if role == "tool":
                for _ in _content_parts(content, root, "content", self._message_index):
                    pass
            elif role == "function":
                _optional_string(content)
            yield ContentEntry(
                "tool_result",
                location,
                tool_result=ToolResultView(
                    _optional_string(source.get("id")) if role == "tool_result" else None,
                    _string(source.get("tool_call_id" if role == "tool" else "call_id"))
                    if role != "function"
                    else None,
                    _string(source.get("name")) if role == "function" else None,
                    None,
                    PayloadView(payload_kind, content),
                ),
            )
        elif role == "tool_call":
            yield ContentEntry(
                "tool_call",
                ContentLocation(root, "tool_call_item", message_index=self._message_index),
                tool_call=ToolCallView(
                    _optional_string(source.get("id")),
                    _string(source.get("call_id")),
                    _string(source.get("name")),
                    PayloadView("json", source.get("arguments")),
                ),
            )
        elif role == "provider_native":
            provider, kind = _string(source.get("provider")), _string(source.get("kind"))
            value = source.get("value")
            if _native_role(source) is not None:
                content = _object(value).get("content")
                if isinstance(content, str):
                    yield ContentEntry(
                        "text", ContentLocation(root, "native_content", message_index=self._message_index), text=content
                    )
                    return
                if isinstance(content, tuple):
                    for index, part in enumerate(content):
                        part_kind = part.get("type") if isinstance(part, Mapping) else None
                        yield ContentEntry(
                            "provider_native",
                            ContentLocation(root, "native_content", index, self._message_index),
                            provider=provider,
                            native_kind=part_kind if isinstance(part_kind, str) else kind,
                            _value=part,
                        )
                    return
            yield ContentEntry(
                "provider_native",
                ContentLocation(root, "value", message_index=self._message_index),
                provider=provider,
                native_kind=kind,
                _value=value,
            )
        else:
            raise ValueError(f"unknown normalized message role: {role!r}")

    def text_parts(self) -> Iterator[str]:
        """Iterate eligible text/refusals and shallow native fallbacks; exclude tool payloads."""
        for entry in self.parts():
            if entry.kind in {"text", "refusal"}:
                # Explicit empty strings remain present.
                assert entry.text is not None
                yield entry.text
            elif entry.kind == "provider_native":
                if entry.location.field == "value" and _native_role(self._source) is not None:
                    continue
                text = _native_text(entry.native_kind, entry._value)
                if text is not None:
                    yield text

    def text(self) -> str | None:
        """Return None for absent text, otherwise newline-join without trimming empty parts."""
        parts = self.text_parts()
        first = next(parts, None)
        if first is None:
            return None
        second = next(parts, None)
        return first if second is None else "\n".join(chain((first, second), parts))


def _history(request: Mapping[str, Any]) -> list[Any]:
    messages = _object(request).get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("normalized request messages must be a list")
    return messages


def message(request: Mapping[str, Any], index: int) -> MessageView:
    """Snapshot a history item by Python index; raise IndexError for out-of-range indices."""
    messages = _history(request)
    index = operator.index(index)
    if index < 0:
        index += len(messages)
    if index < 0 or index >= len(messages):
        raise IndexError("message index out of range")
    return MessageView._snapshot(_object(messages[index]), "message", index)


def last_message(request: Mapping[str, Any]) -> MessageView | None:
    """Snapshot exactly the final supplied request item, or None for empty history."""
    messages = _history(request)
    return message(request, len(messages) - 1) if messages else None


def instructions_view(request: Mapping[str, Any]) -> MessageView | None:
    """Snapshot top-level instructions without falling back to history."""
    content = _object(request).get("instructions")
    return None if content is None else MessageView.from_content(content)


def output(response: Mapping[str, Any]) -> MessageView:
    """Snapshot current normalized message and separate calls, excluding response metadata."""
    response = _object(response)
    return MessageView._snapshot(
        {"message": response.get("message"), "tool_calls": response.get("tool_calls")}, "response"
    )


def system_prompt(request: Mapping[str, Any]) -> str | None:
    """Extract instructions, otherwise the first system/developer item; do not skip textless selections."""
    instructions = instructions_view(request)
    if instructions is not None:
        return instructions.text()
    for index, item in enumerate(_history(request)):
        item = _object(item)
        if item.get("role") in {"system", "developer"} or _native_role(item) in {"system", "developer"}:
            return message(request, index).text()
    return None


def last_user_message(request: Mapping[str, Any]) -> str | None:
    """Extract the last portable/recognized-native user item without searching past textless content."""
    messages = _history(request)
    for index in range(len(messages) - 1, -1, -1):
        item = _object(messages[index])
        if item.get("role") == "user" or _native_role(item) == "user":
            return message(request, index).text()
    return None


def response_text(response: Mapping[str, Any]) -> str | None:
    """Extract current response text/refusals, excluding tools, media, and reasoning."""
    return output(response).text()


__all__ = [
    "ContentEntry",
    "ContentLocation",
    "MessageView",
    "PayloadView",
    "ToolCallView",
    "ToolResultView",
    "instructions_view",
    "last_message",
    "last_user_message",
    "message",
    "output",
    "response_text",
    "system_prompt",
]
