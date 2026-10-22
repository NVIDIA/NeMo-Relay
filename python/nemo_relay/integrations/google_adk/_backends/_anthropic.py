# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages backend for ADK's Anthropic classes. Swaps their client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias, cast

from anthropic import NOT_GIVEN, AsyncAnthropicFoundry, AsyncAnthropicVertex, NotGiven
from anthropic.types import Message, RawMessageDeltaEvent, RawMessageStartEvent
from google.genai import types

from nemo_relay.integrations.google_adk._backends._attempt import (
    CURRENT_ATTEMPT,
    Attempt,
    Backend,
    record_cost,
    usage_fields,
    with_headers,
)
from nemo_relay.integrations.google_adk._serialization import CODECS, Api
from nemo_relay.integrations.google_adk._swap import swap_class

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Iterable, Mapping

    from anthropic import AsyncAnthropic, AsyncStream
    from anthropic.resources.messages import AsyncMessages
    from anthropic.types import RawMessageStreamEvent, ToolChoiceParam, ToolUnionParam
    from google.adk.models.anthropic_llm import AnthropicLlm

_Provider: TypeAlias = Literal["vertex_ai", "azure_ai", "anthropic"]
_Stream: TypeAlias = "AsyncStream[RawMessageStreamEvent]"
_ANY: Final = types.FunctionCallingConfigMode.ANY
_FORCED: Final[Mapping[types.FunctionCallingConfigMode, ToolChoiceParam]] = {
    types.FunctionCallingConfigMode.AUTO: {"type": "auto"},
    _ANY: {"type": "any"},
    types.FunctionCallingConfigMode.NONE: {"type": "none"},
}


async def anthropic_backend(model: AnthropicLlm) -> Backend:
    """Return the Anthropic Messages backend for the model's client and swap its messages resource.

    Provider: `vertex_ai` for a Vertex AI client, `azure_ai` for a Microsoft Foundry client, `anthropic` for any other,
    including AWS and Google Cloud gateways of the Anthropic API.
    """
    client = await model._get_anthropic_client()
    swap_class(client.messages, _messages_methods)
    return Backend(_provider(client), CODECS[Api.ANTHROPIC_MESSAGES], client_headers=True)


def _provider(client: AsyncAnthropic | AsyncAnthropicVertex) -> _Provider:
    if isinstance(client, AsyncAnthropicVertex):
        return "vertex_ai"
    if isinstance(client, AsyncAnthropicFoundry):
        return "azure_ai"
    return "anthropic"


class _Create(Protocol):
    def __call__(self, resource: AsyncMessages, /, **kwargs: object) -> Awaitable[Message | _Stream]: ...


def _messages_methods(cls: type[AsyncMessages]) -> Mapping[str, object]:
    """Return `AsyncMessages` methods that apply the attempt's tool choice edit and request headers.

    ADK builds the Anthropic request without the function calling config. Methods also record on the attempt the message
    id and cost `LlmResponse` drops.
    """
    original = cast("_Create", cls.create)

    async def create(self: AsyncMessages, **kwargs: object) -> Message | AsyncIterator[RawMessageStreamEvent]:
        attempt = CURRENT_ATTEMPT.get()
        if attempt is not None and attempt.tool_choice is not None:
            tools = cast("Iterable[ToolUnionParam] | NotGiven", kwargs["tools"])
            choice = cast("ToolChoiceParam | NotGiven", kwargs["tool_choice"])
            kwargs["tools"], kwargs["tool_choice"] = _tool_choice(attempt.tool_choice, tools, choice)
        result = await original(self, **with_headers(kwargs))
        if attempt is None:
            return result
        if isinstance(result, Message):
            attempt.response_id = result.id
            record_cost(attempt, usage_fields(result.usage))
            return result
        return _recorded_messages(result, attempt)

    return {"create": create}


async def _recorded_messages(stream: _Stream, attempt: Attempt) -> AsyncIterator[RawMessageStreamEvent]:
    try:
        async for event in stream:
            if isinstance(event, RawMessageStartEvent):
                attempt.response_id = event.message.id
                record_cost(attempt, usage_fields(event.message.usage))
            elif isinstance(event, RawMessageDeltaEvent):
                record_cost(attempt, usage_fields(event.usage))
            yield event
    finally:
        await stream.close()


def _tool_choice(
    config: types.FunctionCallingConfig,
    tools: Iterable[ToolUnionParam] | NotGiven,
    choice: ToolChoiceParam | NotGiven,
) -> tuple[list[ToolUnionParam] | NotGiven, ToolChoiceParam | NotGiven]:
    """Return Anthropic tools and tool choice for a function calling config.

    Keep only tools `allowed_function_names` lists. Force the named tool when mode is `ANY` with one allowed name. Keep
    the request's own tool choice when mode is unset. Return `NOT_GIVEN` for both when no tools remain. Raise
    `ValueError` for an undeclared allowed name, a mode with no Anthropic equivalent, or a mode that requires a tool
    call when no tools remain.
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
