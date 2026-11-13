# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed adapters for host-projected normalized content.

Selection and extraction execute in Rust through one authenticated gRPC call.
Returned snapshots are independent and read-only; all subsequent inspection is
local. This module contains no provider recognition or text eligibility policy.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from . import _protocol
from ._proto.nvidia.nemo.relay.worker.v1 import content_pb2 as content_pb

if TYPE_CHECKING:
    from ._api import PluginRuntime

ContentKind: TypeAlias = Literal[
    "text", "refusal", "tool_call", "tool_result", "image_url", "image", "audio", "file", "provider_native"
]
ContentRoot: TypeAlias = Literal["request_message", "message", "response", "content"]
ContentField: TypeAlias = Literal[
    "content", "message", "tool_calls", "tool_call_item", "output", "value", "native_content"
]
PayloadKind: TypeAlias = Literal["null", "string", "json", "content"]


def _freeze(value: Any) -> Any:
    # This is ownership conversion only; JSON validation belongs to the protocol boundary.
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


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


@dataclass(frozen=True, slots=True, init=False)
class MessageView:
    """Independent typed host snapshot; obtain it through an async selection helper."""

    _parts: tuple[ContentEntry, ...] = field(repr=False)
    _text_parts: tuple[str, ...] = field(repr=False)

    def __init__(self) -> None:
        raise TypeError("use async content selection helpers")

    @classmethod
    async def from_message(cls, runtime: PluginRuntime, message: Mapping[str, Any]) -> MessageView:
        """Ask the host to project one normalized message, including tool-result semantics."""
        return _required(await runtime.inspect_llm_content("message", message, "all"))

    @classmethod
    async def from_content(cls, runtime: PluginRuntime, content: str | list[Any]) -> MessageView:
        """Ask the host to project standalone normalized content or instructions."""
        return _required(await runtime.inspect_llm_content("content", content, "all"))

    def parts(self) -> Iterator[ContentEntry]:
        """Iterate the host-projected typed entries locally, without further RPCs."""
        return iter(self._parts)

    def text_parts(self) -> Iterator[str]:
        """Iterate authoritative text strings chosen by Rust; no local reclassification occurs."""
        return iter(self._text_parts)

    def text(self) -> str | None:
        """Join host-selected strings, preserving absent versus empty text and whitespace."""
        return "\n".join(self._text_parts) if self._text_parts else None


def _required(view: MessageView | None) -> MessageView:
    if view is None:
        from ._api import WorkerSdkError

        raise WorkerSdkError("host returned no selection for required content")
    return view


def _optional(value: Any, field_name: str) -> Any:
    return getattr(value, field_name) if value.HasField(field_name) else None


def _payload(value: Any) -> PayloadView:
    kind = value.WhichOneof("kind")
    if kind == "null":
        data = None
    elif kind == "string":
        data = value.string
    else:
        data = _protocol.unpack(getattr(value, kind))
    return PayloadView(cast(PayloadKind, kind), _freeze(data))


def _entry(value: Any) -> ContentEntry:
    position = value.location
    location = ContentLocation(
        cast(ContentRoot, content_pb.LlmContentRoot.Name(position.root).removeprefix("LLM_CONTENT_ROOT_").lower()),
        cast(ContentField, content_pb.LlmContentField.Name(position.field).removeprefix("LLM_CONTENT_FIELD_").lower()),
        _optional(position, "index"),
        _optional(position, "message_index"),
    )
    kind = value.WhichOneof("kind")
    if kind in {"text", "refusal"}:
        return ContentEntry(kind, location, text=getattr(value, kind))
    if kind == "tool_call":
        call = value.tool_call
        return ContentEntry(
            kind,
            location,
            tool_call=ToolCallView(
                _optional(call, "id"), _optional(call, "call_id"), call.name, _payload(call.arguments)
            ),
        )
    if kind == "tool_result":
        result = value.tool_result
        return ContentEntry(
            kind,
            location,
            tool_result=ToolResultView(
                _optional(result, "id"),
                _optional(result, "call_id"),
                _optional(result, "name"),
                _optional(result, "is_error"),
                _payload(result.payload),
            ),
        )
    if kind == "provider_native":
        native = value.provider_native
        return ContentEntry(
            kind,
            location,
            provider=native.provider,
            native_kind=native.kind,
            _value=_freeze(_protocol.unpack(native.value)),
        )
    return ContentEntry(cast(ContentKind, kind), location, _value=_freeze(_protocol.unpack(getattr(value, kind))))


def _from_proto(snapshot: Any) -> MessageView:
    view = object.__new__(MessageView)
    object.__setattr__(view, "_parts", tuple(_entry(part) for part in snapshot.parts))
    object.__setattr__(view, "_text_parts", tuple(snapshot.text_parts))
    return view


async def message(runtime: PluginRuntime, request: Mapping[str, Any], index: int) -> MessageView:
    """Ask the host to select a history item; negative indices count from the end."""
    index = operator.index(index)
    if not -(2**63) <= index < 2**63:
        raise IndexError("message index out of range")
    return _required(await runtime.inspect_llm_content("request", request, "message_index", message_index=index))


async def last_message(runtime: PluginRuntime, request: Mapping[str, Any]) -> MessageView | None:
    """Ask the host to select exactly the final supplied request item, or return None."""
    return await runtime.inspect_llm_content("request", request, "last_message")


async def instructions_view(runtime: PluginRuntime, request: Mapping[str, Any]) -> MessageView | None:
    """Ask the host to select top-level instructions without a history fallback."""
    return await runtime.inspect_llm_content("request", request, "instructions")


async def output(runtime: PluginRuntime, response: Mapping[str, Any]) -> MessageView:
    """Ask the host to project the current message and separate response tool calls."""
    return _required(await runtime.inspect_llm_content("response", response, "output"))


async def system_prompt(runtime: PluginRuntime, request: Mapping[str, Any]) -> str | None:
    """Extract instructions or the first system/developer item through the shared host selection."""
    view = await runtime.inspect_llm_content("request", request, "system_prompt")
    return None if view is None else view.text()


async def last_user_message(runtime: PluginRuntime, request: Mapping[str, Any]) -> str | None:
    """Extract the last portable/native user item through the host, without backward text fallback."""
    view = await runtime.inspect_llm_content("request", request, "last_user_message")
    return None if view is None else view.text()


async def response_text(runtime: PluginRuntime, response: Mapping[str, Any]) -> str | None:
    """Extract current response text/refusals through Rust, excluding tools and non-text data."""
    return (await output(runtime, response)).text()


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
