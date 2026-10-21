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


def _rebind_tool_messages(value: Any, source_id: str | None, tool_call_id: str, *, messages: bool = False) -> Any:
    """Give replayed framework messages the current invocation's identity."""
    if isinstance(value, ToolMessage) and value.tool_call_id == source_id:
        return value.model_copy(update={"tool_call_id": tool_call_id, "id": None})
    if isinstance(value, Command):
        return replace(
            value,
            update=_rebind_tool_messages(value.update, source_id, tool_call_id, messages=True),
            resume=_rebind_tool_messages(value.resume, source_id, tool_call_id),
            goto=_rebind_tool_messages(value.goto, source_id, tool_call_id),
        )
    if isinstance(value, Send):
        return Send(value.node, _rebind_tool_messages(value.arg, source_id, tool_call_id))
    if isinstance(value, dict):
        if messages and value.get("type") == "tool" and value.get("tool_call_id") == source_id:
            return {**value, "tool_call_id": tool_call_id, "id": None}
        return {
            key: _rebind_tool_messages(item, source_id, tool_call_id, messages=key == "messages")
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        items = [_rebind_tool_messages(item, source_id, tool_call_id, messages=messages) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


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

    def _message(self, value: ToolResultValue, tool_call_id: str | None) -> ToolMessage | None:
        candidates = self._candidates(value)
        if tool_call_id is not None:
            return next((message for message in candidates if message.tool_call_id == tool_call_id), None)
        return candidates[0] if len(candidates) == 1 else None

    def _validate(self, value: object) -> None:
        if isinstance(value, list):
            for item in value:
                self._validate(item)
        elif not isinstance(value, ToolMessage | Command):
            raise TypeError("Expected a ToolMessage, Command, or a list of those values")

    def to_json(self, value: ToolResultValue) -> dict[str, Json]:
        self._validate(value)
        message = self._message(value, self._tool_call_id)
        content = message.content if message is not None else None
        if isinstance(content, list):
            content = [{"type": "text", "text": part} if isinstance(part, str) else part for part in content]
        opaque: dict[str, Any] = {}
        result = _encode(value, opaque)
        self._opaque.update(opaque)
        return {
            "schema": TOOL_RESULT_SCHEMA,
            "content": cast(Json, content),
            "is_error": message is not None and message.status == "error",
            "result": result,
            "cache": {
                "replayable": not opaque and self._tool_call_id is not None,
                "tool_call_id": self._tool_call_id,
            },
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
        source_id = self._tool_call_id
        cache = data.get("cache")
        if isinstance(cache, dict) and "tool_call_id" in cache:
            cached_id = cache["tool_call_id"]
            if cached_id is not None and not isinstance(cached_id, str):
                raise ValueError("Invalid cached tool call ID")
            source_id = cached_id
        replayed = self._tool_call_id is not None and source_id != self._tool_call_id
        message = self._message(result, source_id)
        if message is None:
            if content is not None or is_error:
                raise ValueError("Cannot project content or status without a matching ToolMessage")
            if replayed and self._tool_call_id is not None:
                return _rebind_tool_messages(result, source_id, self._tool_call_id)
            return result
        # Keep native content blocks (including string parts) on an unchanged round trip.
        original_content = message.content
        if isinstance(original_content, list):
            original_content = [
                {"type": "text", "text": part} if isinstance(part, str) else part for part in original_content
            ]
        if original_content == content and (message.status == "error") == is_error and not replayed:
            return result
        changes: dict[str, Any] = {
            "content": content if content is not None else "",
            "status": "error" if is_error else "success",
        }
        if replayed:
            # A graph message ID belongs to the old invocation too. Let the graph
            # assign a new one instead of replacing the previous tool message.
            changes.update(tool_call_id=self._tool_call_id, id=None)
        updated = message.model_copy(update=changes)
        result = self._replace_message(result, message, updated)
        if replayed and self._tool_call_id is not None:
            return _rebind_tool_messages(result, source_id, self._tool_call_id)
        return result

    def _replace_message(self, result: ToolResultValue, message: ToolMessage, updated: ToolMessage) -> ToolResultValue:
        if isinstance(result, list):
            return [self._replace_message(item, message, updated) for item in result]
        if isinstance(result, ToolMessage):
            return updated if result is message else result
        match = self._message(result, message.tool_call_id)
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
