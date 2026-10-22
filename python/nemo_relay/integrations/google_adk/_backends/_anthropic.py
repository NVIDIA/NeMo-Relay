# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages backend for ADK's Anthropic classes. Swaps their client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, TypeAlias, cast

from anthropic import NOT_GIVEN, AsyncAnthropicFoundry, NotGiven
from anthropic.lib.vertex import AsyncAnthropicVertex
from anthropic.types import Message, RawMessageDeltaEvent, RawMessageStartEvent
from google.genai import types

from nemo_relay.codecs import AnthropicMessagesCodec
from nemo_relay.integrations.google_adk._backends._attempt import (
    CURRENT_ATTEMPT,
    Attempt,
    Backend,
    record_cost,
    usage_fields,
    with_extra_headers,
)
from nemo_relay.integrations.google_adk._serialization import (
    BASE_USAGE,
    BASE_USAGE_KEYS,
    GENAI_USAGE,
    RECORDED_ID_KEY,
    AdkCodec,
    Normalized,
    token_counts,
)
from nemo_relay.integrations.google_adk._swap import swap_class

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Iterable, Mapping

    from anthropic import AsyncAnthropic, AsyncStream
    from anthropic.resources.messages import AsyncMessages
    from anthropic.types import RawMessageStreamEvent, ToolChoiceParam, ToolUnionParam
    from google.adk.models.anthropic_llm import AnthropicLlm

    from nemo_relay import JsonObject

_Provider: TypeAlias = Literal["vertex_ai", "azure_ai", "anthropic"]
_Stream: TypeAlias = "AsyncStream[RawMessageStreamEvent]"
_API: Final = "anthropic_messages"
CACHE_WRITE_KEY: Final[Annotated[str, "Usage attribute ADK sets for cache writes outside GenAI schema."]] = (
    "cache_creation_input_tokens"
)
_CACHE_WRITE: Final = ((CACHE_WRITE_KEY, "cache_write_tokens"),)
_USAGE_KEYS: Final = BASE_USAGE_KEYS | {CACHE_WRITE_KEY}
_NATIVE_CODEC: Final = AnthropicMessagesCodec()
_ANY: Final = types.FunctionCallingConfigMode.ANY
_FORCED: Final[Mapping[types.FunctionCallingConfigMode, ToolChoiceParam]] = {
    types.FunctionCallingConfigMode.AUTO: {"type": "auto"},
    _ANY: {"type": "any"},
    types.FunctionCallingConfigMode.NONE: {"type": "none"},
}


def _read(response: JsonObject, recorded: JsonObject, usage_metadata: JsonObject) -> Normalized:
    """Unfold Anthropic counts from GenAI usage shape ADK fits them into.

    Raise `ValueError` when prompt count is below cache tokens ADK folds into it.
    """
    counts = token_counts(usage_metadata, BASE_USAGE)
    separate = token_counts(usage_metadata, GENAI_USAGE)
    writes = token_counts(usage_metadata, _CACHE_WRITE)
    usage: dict[str, int] = {}
    cached, written = counts.get("cache_read_tokens"), writes.get("cache_write_tokens")
    if (prompt := counts.get("prompt_tokens")) is not None:
        folded = (cached or 0) + (written or 0)
        if folded > prompt:
            raise ValueError("ADK prompt_token_count must include the Anthropic cache tokens it reports")
        usage["prompt_tokens"] = usage["uncached_input_tokens"] = prompt - folded
    if (candidates := counts.get("completion_tokens")) is not None:
        usage["completion_tokens"] = candidates + separate.get("thoughts_tokens", 0)
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if cached is not None:
        usage["cache_read_tokens"] = cached
    if written is not None:
        usage["cache_write_tokens"] = written
    return Normalized(usage, {"api": _API}, _USAGE_KEYS, frozenset(), frozenset())


ANTHROPIC_CODEC: Final = AdkCodec(_read)


async def anthropic_backend(model: AnthropicLlm) -> Backend:
    """Return Anthropic Messages backend for model's client, and swap client's messages resource.

    Provider: `vertex_ai` for Vertex AI client, `azure_ai` for Microsoft Foundry client, `anthropic` for any other,
    including Anthropic API's AWS and Google Cloud gateways.
    """
    client = await model._get_anthropic_client()
    swap_class(client.messages, _messages_methods)
    return Backend(_provider(client), ANTHROPIC_CODEC, client_headers=True)


def _provider(client: AsyncAnthropic | AsyncAnthropicVertex) -> _Provider:
    if isinstance(client, AsyncAnthropicVertex):
        return "vertex_ai"
    if isinstance(client, AsyncAnthropicFoundry):
        return "azure_ai"
    return "anthropic"


class _Create(Protocol):
    def __call__(self, resource: AsyncMessages, /, **kwargs: object) -> Awaitable[Message | _Stream]: ...


def _messages_methods(cls: type[AsyncMessages]) -> Mapping[str, object]:
    """Return `AsyncMessages` methods that apply attempt's tool choice edit and request headers.

    ADK builds Anthropic request without function calling config. Methods also record message id and cost on attempt.
    `LlmResponse` drops both.
    """
    original = cast("_Create", cls.create)

    async def create(self: AsyncMessages, **kwargs: object) -> Message | AsyncIterator[RawMessageStreamEvent]:
        attempt = CURRENT_ATTEMPT.get()
        if attempt is not None and attempt.tool_choice is not None:
            tools = cast("Iterable[ToolUnionParam] | NotGiven", kwargs["tools"])
            choice = cast("ToolChoiceParam | NotGiven", kwargs["tool_choice"])
            kwargs["tools"], kwargs["tool_choice"] = _tool_choice(attempt.tool_choice, tools, choice)
        result = await original(self, **with_extra_headers(kwargs))
        if attempt is None:
            return result
        if isinstance(result, Message):
            attempt.recorded[RECORDED_ID_KEY] = result.id
            record_cost(attempt, usage_fields(result.usage), codec=_NATIVE_CODEC)
            return result
        return _recorded_messages(result, attempt)

    return {"create": create}


async def _recorded_messages(stream: _Stream, attempt: Attempt) -> AsyncIterator[RawMessageStreamEvent]:
    try:
        async for event in stream:
            if isinstance(event, RawMessageStartEvent):
                attempt.recorded[RECORDED_ID_KEY] = event.message.id
                record_cost(attempt, usage_fields(event.message.usage), codec=_NATIVE_CODEC)
            elif isinstance(event, RawMessageDeltaEvent):
                record_cost(attempt, usage_fields(event.usage), codec=_NATIVE_CODEC)
            yield event
    finally:
        await stream.close()


def _tool_choice(
    config: types.FunctionCallingConfig,
    tools: Iterable[ToolUnionParam] | NotGiven,
    choice: ToolChoiceParam | NotGiven,
) -> tuple[list[ToolUnionParam] | NotGiven, ToolChoiceParam | NotGiven]:
    """Return Anthropic tools and tool choice for function calling config.

    Keep only tools listed in `allowed_function_names`. Force named tool when mode is `ANY` with one allowed name. Keep
    request's tool choice when mode is unset or `MODE_UNSPECIFIED`. Return `NOT_GIVEN` for both when no tools remain.
    Raise `ValueError` for undeclared allowed name, mode with no Anthropic equivalent, or mode requiring tool call when
    no tools remain.
    """
    declared: list[ToolUnionParam] = [] if isinstance(tools, NotGiven) else list(tools)
    names = config.allowed_function_names or []
    if names:
        known = {tool["name"] for tool in declared}
        if missing := [name for name in names if name not in known]:
            raise ValueError(f"ADK allowed_function_names {missing} are not declared on the request")
        declared = [tool for tool in declared if tool["name"] in names]
    mode = config.mode
    forced: ToolChoiceParam | None = None
    if mode is not None and mode is not types.FunctionCallingConfigMode.MODE_UNSPECIFIED:
        if mode not in _FORCED:
            raise ValueError(f"Anthropic has no equivalent of the {mode.value} function calling mode")
        forced = {"type": "tool", "name": names[0]} if mode is _ANY and len(names) == 1 else _FORCED[mode]
    if not declared:
        if forced is not None and forced["type"] == "any":
            raise ValueError("the function calling config requires a tool call but the request declares no tools")
        return NOT_GIVEN, NOT_GIVEN
    return declared, choice if forced is None else forced
