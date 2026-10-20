# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encode LangChain tool results, content, and failure status."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

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
from langgraph.types import Command, Send

from nemo_relay import Json
from nemo_relay._tool_result import TOOL_RESULT_SCHEMA
from nemo_relay.typed import Codec

# messages_from_dict restores these exact classes. Custom subclasses stay local.
_PORTABLE_MESSAGE_TYPES = frozenset(
    {
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
    }
)


def _encode(value: Any, opaque: dict[str, Any]) -> Json:
    """Encode known framework objects without importing types or using pickle."""
    if type(value) in _PORTABLE_MESSAGE_TYPES:
        # model_dump() can coerce application-owned artifacts (for example,
        # dataclasses) to dictionaries. Retain their native values instead.
        fields = {name: getattr(value, name) for name in type(value).model_fields}
        fields.update(value.model_extra or {})
        return {"kind": "message", "value": _encode({"type": value.type, "data": fields}, opaque)}
    if isinstance(value, Command):
        return {
            "kind": "command",
            "value": _encode(
                {"graph": value.graph, "update": value.update, "resume": value.resume, "goto": value.goto}, opaque
            ),
        }
    if isinstance(value, Send):
        return {"kind": "send", "value": _encode({"node": value.node, "arg": value.arg}, opaque)}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {"kind": "dict", "value": {key: _encode(item, opaque) for key, item in value.items()}}
    if isinstance(value, list | tuple):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "value": [_encode(item, opaque) for item in value],
        }
    if value is None or isinstance(value, str | int | float | bool):
        return {"kind": "json", "value": value}
    # Keep custom artifacts and graph state in this call's memory.
    # Do not restore unknown Python classes from event data.
    token = str(uuid4())
    opaque[token] = value
    return {"kind": "opaque", "value": token}


def _decode(data: Any, opaque: dict[str, Any]) -> Any:
    """Restore known framework types, JSON values, and saved object references."""
    if not isinstance(data, dict) or "value" not in data:
        raise ValueError("Expected a serialized LangChain tool result value")
    kind, value = data.get("kind"), data["value"]
    if kind == "json":
        if value is None or isinstance(value, str | int | float | bool):
            return value
    elif kind == "dict" and isinstance(value, dict):
        return {key: _decode(item, opaque) for key, item in value.items()}
    elif kind in ("list", "tuple") and isinstance(value, list):
        items = [_decode(item, opaque) for item in value]
        return tuple(items) if kind == "tuple" else items
    elif kind == "message":
        return messages_from_dict([_decode(value, opaque)])[0]
    elif kind == "command":
        return Command(**_decode(value, opaque))
    elif kind == "send":
        return Send(**_decode(value, opaque))
    elif kind == "opaque" and isinstance(value, str):
        if value not in opaque:
            raise ValueError("Opaque LangChain tool result values belong to their original invocation")
        return opaque[value]
    raise ValueError(f"Unsupported LangChain tool result kind: {kind!r}")


ToolResultValue = ToolMessage | Command[Any] | list[Any]


class LangChainToolResultCodec(Codec[ToolResultValue]):
    """Encode tool content and status without losing messages or graph updates.

    Use one codec for each tool call. When a call ID is provided, only a
    matching ToolMessage provides content and status. The codec keeps other
    Command updates so they can be restored later. It also supports lists of
    messages and commands returned by LangGraph tools.
    """

    def __init__(self, tool_call_id: str | None) -> None:
        self._tool_call_id = tool_call_id
        self._opaque: dict[str, Any] = {}

    def _candidates(self, value: ToolResultValue) -> list[ToolMessage]:
        if isinstance(value, list):
            return [message for item in value for message in self._candidates(item)]
        if isinstance(value, ToolMessage):
            return [value]
        else:
            update = value.update
            messages = update.get("messages", []) if isinstance(update, dict) else update
            candidates = []
            if isinstance(messages, list | tuple):
                for message in messages:
                    if isinstance(message, dict) and message.get("type") == "tool":
                        message = ToolMessage.model_validate(message)
                    if isinstance(message, ToolMessage):
                        candidates.append(message)
        return candidates

    def _message(self, value: ToolResultValue) -> ToolMessage | None:
        candidates = self._candidates(value)
        if self._tool_call_id is not None:
            return next((message for message in candidates if message.tool_call_id == self._tool_call_id), None)
        return candidates[0] if len(candidates) == 1 else None

    def _validate(self, value: object) -> None:
        if isinstance(value, list):
            for item in value:
                self._validate(item)
        elif not isinstance(value, ToolMessage | Command):
            raise TypeError("Expected a ToolMessage, Command, or a list of those values")

    def to_json(self, value: ToolResultValue) -> dict[str, Json]:
        self._validate(value)
        message = self._message(value)
        content = message.content if message is not None else None
        if isinstance(content, list):
            content = [{"type": "text", "text": part} if isinstance(part, str) else part for part in content]
        return {
            "schema": TOOL_RESULT_SCHEMA,
            "content": cast(Json, content),
            "is_error": message is not None and message.status == "error",
            "result": _encode(value, self._opaque),
        }

    def from_json(self, data: Json) -> ToolResultValue:
        if not isinstance(data, dict) or data.get("schema") != TOOL_RESULT_SCHEMA:
            raise ValueError("Expected a versioned Relay tool result")
        content, is_error = data.get("content"), data.get("is_error")
        if (
            "content" not in data
            or not (content is None or isinstance(content, str | list))
            or not isinstance(is_error, bool)
            or (isinstance(content, list) and not all(isinstance(part, str | dict) for part in content))
        ):
            raise ValueError("Invalid tool result content or error status")
        result = _decode(data.get("result"), self._opaque)
        self._validate(result)
        message = self._message(result)
        if message is None:
            if content is not None or is_error:
                raise ValueError("Cannot project content or status without a matching ToolMessage")
            return result
        # Keep native content blocks (including string parts) on an unchanged round trip.
        original_content = message.content
        if isinstance(original_content, list):
            original_content = [
                {"type": "text", "text": part} if isinstance(part, str) else part for part in original_content
            ]
        if original_content == content and (message.status == "error") == is_error:
            return result
        updated = message.model_copy(
            update={"content": content if content is not None else "", "status": "error" if is_error else "success"}
        )
        return self._replace_message(result, message, updated)

    def _replace_message(self, result: ToolResultValue, message: ToolMessage, updated: ToolMessage) -> ToolResultValue:
        if isinstance(result, list):
            return [self._replace_message(item, message, updated) for item in result]
        if isinstance(result, ToolMessage):
            return updated if result is message else result
        match = self._message(result)
        if match is None or match.tool_call_id != message.tool_call_id:
            return result
        update = result.update
        messages = update["messages"] if isinstance(update, dict) else update
        assert isinstance(messages, list | tuple)
        replacements = [
            updated
            if item is message
            or (
                isinstance(item, dict)
                and item.get("type") == "tool"
                and item.get("tool_call_id") == message.tool_call_id
            )
            else item
            for item in messages
        ]
        if isinstance(messages, tuple):
            replacements = tuple(replacements)
        return replace(
            result, update={**update, "messages": replacements} if isinstance(update, dict) else replacements
        )
