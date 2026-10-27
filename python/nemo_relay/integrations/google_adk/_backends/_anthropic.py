# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages backend for ADK's Anthropic classes. Swaps their client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias, cast

from anthropic import NOT_GIVEN, AsyncAnthropicFoundry, NotGiven
from anthropic.lib.vertex import AsyncAnthropicVertex
from anthropic.types import Message, RawMessageDeltaEvent, RawMessageStartEvent
from google.genai import types

from nemo_relay.codecs import AnthropicMessagesCodec
from nemo_relay.integrations.google_adk._backends._attempt import (
    CURRENT_ATTEMPT,
    Attempt,
    Backend,
    ToolChoiceApi,
    provider_tool_choice,
    record_usage,
    usage_fields,
    with_extra_headers,
)
from nemo_relay.integrations.google_adk._serialization import (
    BASE_USAGE,
    BASE_USAGE_KEYS,
    CACHE_WRITE_KEY,
    GENAI_USAGE,
    RECORDED_ID_KEY,
    RECORDED_USAGE_KEY,
    AdkCodec,
    Normalized,
    reported_cost,
    token_counts,
    without_cache,
)
from nemo_relay.integrations.google_adk._swap import Methods, swap_class

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
_CACHE_WRITE: Final = ((CACHE_WRITE_KEY, "cache_write_tokens"),)
_USAGE_KEYS: Final = BASE_USAGE_KEYS | {CACHE_WRITE_KEY}
_RECORDED_KEYS: Final = frozenset({RECORDED_USAGE_KEY})
_NATIVE_CODEC: Final = AnthropicMessagesCodec()
_CHOICES: Final[Mapping[types.FunctionCallingConfigMode, ToolChoiceParam]] = {
    types.FunctionCallingConfigMode.AUTO: {"type": "auto"},
    types.FunctionCallingConfigMode.ANY: {"type": "any"},
    types.FunctionCallingConfigMode.NONE: {"type": "none"},
}


def _named_tool(name: str) -> ToolChoiceParam:
    return {"type": "tool", "name": name}


def _tool_name(tool: ToolUnionParam) -> str | None:
    """Return function name of `tool`, or None for a built-in tool, which carries a versioned `type`."""
    return tool["name"] if tool.get("type") in (None, "custom") else None


_TOOL_CHOICE: Final[ToolChoiceApi[ToolUnionParam, ToolChoiceParam | NotGiven]] = ToolChoiceApi(
    name="Anthropic", choices=_CHOICES, named=_named_tool, tool_name=_tool_name
)


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
        usage["prompt_tokens"] = usage["uncached_input_tokens"] = without_cache(
            prompt, cached, written, folded="the Anthropic cache tokens it reports"
        )
    if (candidates := counts.get("completion_tokens")) is not None:
        usage["completion_tokens"] = candidates + separate.get("thoughts_tokens", 0)
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if cached is not None:
        usage["cache_read_tokens"] = cached
    if written is not None:
        usage["cache_write_tokens"] = written
    return Normalized(
        usage=usage,
        cost=reported_cost(recorded, codec=_NATIVE_CODEC),
        api_specific={"api": _API},
        usage_keys=_USAGE_KEYS,
        response_keys=frozenset(),
        recorded_keys=_RECORDED_KEYS,
    )


ANTHROPIC_CODEC: Final = AdkCodec(_read)


async def anthropic_backend(model: AnthropicLlm, request_model: str | None) -> Backend:
    """Return Anthropic Messages backend for model's client, and swap client's messages resource.

    Provider: `vertex_ai` for Vertex AI client, `azure_ai` for Microsoft Foundry client, `anthropic` for any other,
    including Anthropic API's AWS and Google Cloud gateways.
    """
    client = await model._get_anthropic_client()
    swap_class(client.messages, _messages_methods)
    return Backend(_provider(client), ANTHROPIC_CODEC, client_headers=True, model=request_model or model.model)


def _provider(client: AsyncAnthropic | AsyncAnthropicVertex) -> _Provider:
    if isinstance(client, AsyncAnthropicVertex):
        return "vertex_ai"
    if isinstance(client, AsyncAnthropicFoundry):
        return "azure_ai"
    return "anthropic"


class _Create(Protocol):
    def __call__(self, resource: AsyncMessages, /, **kwargs: object) -> Awaitable[Message | _Stream]: ...


def _messages_methods(cls: type[AsyncMessages]) -> Methods[AsyncMessages]:
    """Return `AsyncMessages` methods that apply attempt's tool choice edit and request headers.

    ADK builds Anthropic request without function calling config. Methods also record message id and usage on attempt.
    `LlmResponse` drops id and usage cost.
    """
    original = cast("_Create", cls.create)

    async def create(self: AsyncMessages, **kwargs: object) -> Message | AsyncIterator[RawMessageStreamEvent]:
        attempt = CURRENT_ATTEMPT.get()
        if attempt is not None and attempt.tool_choice is not None:
            tools = cast("Iterable[ToolUnionParam] | NotGiven", kwargs["tools"])
            choice = cast("ToolChoiceParam | NotGiven", kwargs["tool_choice"])
            declared: list[ToolUnionParam] = [] if isinstance(tools, NotGiven) else list(tools)
            chosen = provider_tool_choice(_TOOL_CHOICE, attempt.tool_choice, declared, choice)
            kwargs["tools"], kwargs["tool_choice"] = (NOT_GIVEN, NOT_GIVEN) if chosen is None else chosen
        result = await original(self, **with_extra_headers(kwargs))
        if attempt is None:
            return result
        if isinstance(result, Message):
            attempt.recorded[RECORDED_ID_KEY] = result.id
            record_usage(attempt, result.usage)
            return result
        return _recorded_messages(result, attempt)

    return {"create": create}


async def _recorded_messages(stream: _Stream, attempt: Attempt) -> AsyncIterator[RawMessageStreamEvent]:
    usage: JsonObject = {}
    try:
        async for event in stream:
            if isinstance(event, RawMessageStartEvent):
                attempt.recorded[RECORDED_ID_KEY] = event.message.id
                usage = usage_fields(event.message.usage)
                attempt.recorded[RECORDED_USAGE_KEY] = usage
            elif isinstance(event, RawMessageDeltaEvent):
                usage = {**usage, **usage_fields(event.usage)}
                attempt.recorded[RECORDED_USAGE_KEY] = usage
            yield event
    finally:
        await stream.close()
