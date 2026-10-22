# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini backend for ADK's `Gemini` family. Swaps the client's models resource."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, ParamSpec, TypeAlias, cast

from google.genai import types

from nemo_relay.integrations.google_adk._backends._attempt import CURRENT_ATTEMPT, Backend
from nemo_relay.integrations.google_adk._serialization import CODECS, Api
from nemo_relay.integrations.google_adk._swap import swap_class

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

    from google.adk.models.google_llm import Gemini
    from google.genai.models import AsyncModels

_P = ParamSpec("_P")
_Stream: TypeAlias = AsyncGenerator[types.GenerateContentResponse, None]


def gemini_backend(model: Gemini) -> Backend:
    """Return the backend for the model's client: `vertex_ai` for a Vertex AI client, `gemini` otherwise.

    Swap the client's models resource to record each response, unless the model uses the interactions API.
    """
    client = model.api_client
    if not model.use_interactions_api:
        swap_class(client.aio.models, _models_methods)
    provider = "vertex_ai" if client.vertexai else "gemini"
    return Backend(provider, CODECS[Api.GEMINI_GENERATE_CONTENT], client_headers=False)


def _models_methods(cls: type[AsyncModels]) -> Mapping[str, object]:
    """Return `AsyncModels` methods that record response id and URL context on the current attempt.

    `LlmResponse` drops `response_id` and the first candidate's `url_context_metadata`.
    """
    return {
        "generate_content": _recorded(cls.generate_content),
        "generate_content_stream": _recorded_stream(cls.generate_content_stream),
    }


def _record(response: types.GenerateContentResponse) -> None:
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None:
        return
    if response.response_id is not None:
        attempt.response_id = response.response_id
    candidate = next((candidate for candidate in response.candidates or [] if not candidate.index), None)
    if candidate is not None and candidate.url_context_metadata is not None:
        attempt.urls.append(candidate.url_context_metadata)


def _recorded(
    generate: Callable[_P, Awaitable[types.GenerateContentResponse]],
) -> Callable[_P, Awaitable[types.GenerateContentResponse]]:
    async def generate_content(*args: _P.args, **kwargs: _P.kwargs) -> types.GenerateContentResponse:
        response = await generate(*args, **kwargs)
        _record(response)
        return response

    return generate_content


def _recorded_stream(
    generate: Callable[_P, Awaitable[AsyncIterator[types.GenerateContentResponse]]],
) -> Callable[_P, Awaitable[AsyncIterator[types.GenerateContentResponse]]]:
    async def generate_content_stream(
        *args: _P.args, **kwargs: _P.kwargs
    ) -> AsyncIterator[types.GenerateContentResponse]:
        return _recording(cast("_Stream", await generate(*args, **kwargs)))

    return generate_content_stream


async def _recording(responses: _Stream) -> _Stream:
    async with contextlib.aclosing(responses) as stream:
        async for response in stream:
            _record(response)
            yield response
