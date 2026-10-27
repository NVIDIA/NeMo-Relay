# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat Completions and Responses backends for ADK's OpenAI model classes. Swaps their clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast

import openai
from google.adk.integrations.openai import AzureOpenAIResponsesLlm
from openai.types.chat import ChatCompletion
from openai.types.responses import Response, ResponseCompletedEvent, ResponseIncompleteEvent

from nemo_relay.integrations.google_adk._backends._attempt import (
    CURRENT_ATTEMPT,
    Attempt,
    Backend,
    record_usage,
    with_extra_headers,
)
from nemo_relay.integrations.google_adk._backends._chat_completions import CHAT_CODEC
from nemo_relay.integrations.google_adk._backends._responses import RESPONSES_CODEC
from nemo_relay.integrations.google_adk._serialization import (
    RECORDED_ID_KEY,
)
from nemo_relay.integrations.google_adk._swap import Methods, swap_class

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

    from google.adk.integrations.openai import OpenAILlm, OpenAIResponsesLlm
    from openai import AsyncOpenAI, AsyncStream
    from openai.resources.chat.completions import AsyncCompletions
    from openai.resources.responses import AsyncResponses
    from openai.types.chat import ChatCompletionChunk
    from openai.types.responses import ResponseStreamEvent


_Provider: TypeAlias = Literal["azure", "bedrock", "openai"]
_ChatStream: TypeAlias = "AsyncStream[ChatCompletionChunk]"
_ResponsesStream: TypeAlias = "AsyncStream[ResponseStreamEvent]"


def openai_chat_backend(model: OpenAILlm) -> Backend:
    """Return Chat Completions backend for model's client, and swap client's completions resource."""
    client = model._openai_client
    swap_class(client.chat.completions, _completions_methods)
    return Backend(_provider(client), CHAT_CODEC, client_headers=True, model=model.model)


def openai_responses_backend(model: OpenAIResponsesLlm, request_model: str | None) -> Backend:
    """Return Responses backend for model's client, and swap client's responses resource.

    Provider: `azure` for `AzureOpenAIResponsesLlm`, otherwise client's provider.
    """
    client = model._openai_client
    swap_class(client.responses, _responses_methods)
    provider = "azure" if isinstance(model, AzureOpenAIResponsesLlm) else _provider(client)
    return Backend(provider, RESPONSES_CODEC, client_headers=True, model=request_model or model.model)


def _provider(client: AsyncOpenAI) -> _Provider:
    """Return `azure` for Azure OpenAI client, `bedrock` for Bedrock client, `openai` for any other.

    Bedrock only when installed SDK defines `AsyncBedrockOpenAI`.
    """
    if isinstance(client, openai.AsyncAzureOpenAI):
        return "azure"
    bedrock: type[AsyncOpenAI] | None = getattr(openai, "AsyncBedrockOpenAI", None)
    if bedrock is not None and isinstance(client, bedrock):
        return "bedrock"
    return "openai"


class _Complete(Protocol):
    def __call__(self, resource: AsyncCompletions, /, **kwargs: object) -> Awaitable[ChatCompletion | _ChatStream]: ...


def _completions_methods(cls: type[AsyncCompletions]) -> Methods[AsyncCompletions]:
    """Return `AsyncCompletions` methods that send attempt's request headers.

    Methods also record completion id, model, and usage on attempt. `LlmResponse` drops id, model, and usage cost.
    """
    original = cast("_Complete", cls.create)

    async def create(self: AsyncCompletions, **kwargs: object) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        result = await original(self, **with_extra_headers(kwargs))
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return result
        if isinstance(result, ChatCompletion):
            attempt.recorded[RECORDED_ID_KEY] = result.id
            attempt.response_model = result.model
            record_usage(attempt, result.usage)
            return result
        return _recorded_completions(result, attempt)

    return {"create": create}


async def _recorded_completions(stream: _ChatStream, attempt: Attempt) -> AsyncIterator[ChatCompletionChunk]:
    try:
        async for chunk in stream:
            attempt.recorded[RECORDED_ID_KEY] = chunk.id
            attempt.response_model = chunk.model
            record_usage(attempt, chunk.usage)
            yield chunk
    finally:
        await stream.close()


class _Respond(Protocol):
    def __call__(self, resource: AsyncResponses, /, **kwargs: object) -> Awaitable[Response | _ResponsesStream]: ...


def _responses_methods(cls: type[AsyncResponses]) -> Methods[AsyncResponses]:
    """Return `AsyncResponses` methods that send attempt's request headers.

    Methods also record usage on attempt. `LlmResponse` drops usage cost.
    """
    original = cast("_Respond", cls.create)

    async def create(self: AsyncResponses, **kwargs: object) -> Response | AsyncIterator[ResponseStreamEvent]:
        result = await original(self, **with_extra_headers(kwargs))
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return result
        if isinstance(result, Response):
            record_usage(attempt, result.usage)
            return result
        return _recorded_responses(result, attempt)

    return {"create": create}


async def _recorded_responses(stream: _ResponsesStream, attempt: Attempt) -> AsyncIterator[ResponseStreamEvent]:
    try:
        async for event in stream:
            if isinstance(event, ResponseCompletedEvent | ResponseIncompleteEvent):
                record_usage(attempt, event.response.usage)
            yield event
    finally:
        await stream.close()
