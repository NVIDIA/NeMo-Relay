# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat Completions and Responses backends for ADK's OpenAI model classes. Swaps their clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast

import openai
from google.adk.integrations.openai import AzureOpenAIResponsesLlm
from openai.types.chat import ChatCompletion
from openai.types.responses import Response, ResponseCompletedEvent

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
    from collections.abc import AsyncIterator, Awaitable, Mapping

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
    """Return the Chat Completions backend for the model's client and swap its completions resource."""
    client = model._openai_client
    swap_class(client.chat.completions, _completions_methods)
    return Backend(_provider(client), CODECS[Api.OPENAI_CHAT], client_headers=True)


def openai_responses_backend(model: OpenAIResponsesLlm) -> Backend:
    """Return the Responses backend for the model's client and swap its responses resource.

    Provider: `azure` for an `AzureOpenAIResponsesLlm`, otherwise the client's.
    """
    client = model._openai_client
    swap_class(client.responses, _responses_methods)
    provider = "azure" if isinstance(model, AzureOpenAIResponsesLlm) else _provider(client)
    return Backend(provider, CODECS[Api.OPENAI_RESPONSES], client_headers=True)


def _provider(client: AsyncOpenAI) -> _Provider:
    """Return `azure` for an Azure OpenAI client, `openai` for any other.

    `bedrock` for a Bedrock client when the installed SDK defines `AsyncBedrockOpenAI`.
    """
    if isinstance(client, openai.AsyncAzureOpenAI):
        return "azure"
    bedrock: type[AsyncOpenAI] | None = getattr(openai, "AsyncBedrockOpenAI", None)
    if bedrock is not None and isinstance(client, bedrock):
        return "bedrock"
    return "openai"


class _Complete(Protocol):
    def __call__(self, resource: AsyncCompletions, /, **kwargs: object) -> Awaitable[ChatCompletion | _ChatStream]: ...


def _completions_methods(cls: type[AsyncCompletions]) -> Mapping[str, object]:
    """Return `AsyncCompletions` methods that send the attempt's request headers.

    Methods also record on the attempt the completion id, model, and cost `LlmResponse` drops.
    """
    original = cast("_Complete", cls.create)

    async def create(self: AsyncCompletions, **kwargs: object) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        result = await original(self, **with_headers(kwargs))
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return result
        if isinstance(result, ChatCompletion):
            attempt.response_id = result.id
            attempt.response_model = result.model
            record_cost(attempt, usage_fields(result.usage))
            return result
        return _recorded_completions(result, attempt)

    return {"create": create}


async def _recorded_completions(stream: _ChatStream, attempt: Attempt) -> AsyncIterator[ChatCompletionChunk]:
    try:
        async for chunk in stream:
            attempt.response_id = chunk.id
            attempt.response_model = chunk.model
            record_cost(attempt, usage_fields(chunk.usage))
            yield chunk
    finally:
        await stream.close()


class _Respond(Protocol):
    def __call__(self, resource: AsyncResponses, /, **kwargs: object) -> Awaitable[Response | _ResponsesStream]: ...


def _responses_methods(cls: type[AsyncResponses]) -> Mapping[str, object]:
    """Return `AsyncResponses` methods that send the attempt's request headers.

    Methods also record on the attempt the cost `LlmResponse` drops.
    """
    original = cast("_Respond", cls.create)

    async def create(self: AsyncResponses, **kwargs: object) -> Response | AsyncIterator[ResponseStreamEvent]:
        result = await original(self, **with_headers(kwargs))
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return result
        if isinstance(result, Response):
            record_cost(attempt, usage_fields(result.usage))
            return result
        return _recorded_responses(result, attempt)

    return {"create": create}


async def _recorded_responses(stream: _ResponsesStream, attempt: Attempt) -> AsyncIterator[ResponseStreamEvent]:
    try:
        async for event in stream:
            if isinstance(event, ResponseCompletedEvent):
                record_cost(attempt, usage_fields(event.response.usage))
            yield event
    finally:
        await stream.close()
