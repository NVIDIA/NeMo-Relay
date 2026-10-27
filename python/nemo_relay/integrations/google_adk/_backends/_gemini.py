# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini backend for ADK's `Gemini` family. Swaps client's models resource."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated, Final, ParamSpec, TypeAlias, cast

from google.genai import types

from nemo_relay.integrations.google_adk._backends._attempt import CURRENT_ATTEMPT, Backend
from nemo_relay.integrations.google_adk._serialization import (
    BASE_USAGE,
    BASE_USAGE_KEYS,
    GENAI_USAGE,
    RECORDED_ID_KEY,
    AdkCodec,
    Normalized,
    token_counts,
    with_uncached,
)
from nemo_relay.integrations.google_adk._swap import Methods, swap_class

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from google.adk.models.google_llm import Gemini
    from google.genai.models import AsyncModels
    from pydantic import BaseModel

    from nemo_relay import Json, JsonObject

_P = ParamSpec("_P")
_Stream: TypeAlias = AsyncGenerator[types.GenerateContentResponse, None]
_API: Final = "gemini_generate_content"
URL_CONTEXT_KEY: Final[Annotated[str, "Recorded key for first candidate's URL context metadata."]] = (
    "url_context_metadata"
)
_WIRE_METADATA: Final[tuple[tuple[str, str, type[BaseModel]], ...]] = (
    ("grounding_metadata", "grounding_metadata", types.GroundingMetadata),
    ("citation_metadata", "citation_metadata", types.CitationMetadata),
    ("logprobs_result", "logprobsResult", types.LogprobsResult),
)
_RESPONSE_KEYS: Final = frozenset({"avg_logprobs", *(key for key, _, _ in _WIRE_METADATA)})
_RECORDED_KEYS: Final = frozenset({URL_CONTEXT_KEY})
_USAGE_KEYS: Final = BASE_USAGE_KEYS | frozenset(source for source, _ in GENAI_USAGE)


def _read(response: JsonObject, recorded: JsonObject, usage_metadata: JsonObject) -> Normalized:
    """Read GenAI usage.

    Thought and tool-use prompt counts go to `api_specific`, beside candidate metadata in Gemini wire shape. No
    `api_specific` when it holds only API name.
    """
    counts = token_counts(usage_metadata, BASE_USAGE)
    api_specific: JsonObject = {"api": _API, **token_counts(usage_metadata, GENAI_USAGE)}
    api_specific.update(
        (target, _wire(model, value))
        for source, target, model in _WIRE_METADATA
        if (value := response.get(source)) is not None
    )
    if (url_context := recorded.get(URL_CONTEXT_KEY)) is not None:
        api_specific["urlContextMetadata"] = _wire(types.UrlContextMetadata, url_context)
    avg_logprobs = response.get("avg_logprobs")
    if avg_logprobs is not None:
        if not isinstance(avg_logprobs, int | float):
            raise TypeError("ADK avg_logprobs must be a number")
        api_specific["avgLogprobs"] = avg_logprobs
    return Normalized(
        usage=with_uncached(counts),
        cost=None,
        api_specific=api_specific if len(api_specific) > 1 else {},
        usage_keys=_USAGE_KEYS,
        response_keys=_RESPONSE_KEYS,
        recorded_keys=_RECORDED_KEYS,
    )


def _wire(model: type[BaseModel], value: Json) -> Json:
    return model.model_validate(value).model_dump(mode="json", by_alias=True, exclude_none=True)


GEMINI_CODEC: Final = AdkCodec(_read)


def gemini_backend(model: Gemini, request_model: str | None) -> Backend:
    """Return backend for model's client: `vertex_ai` for Vertex AI client, `gemini` otherwise.

    Swap client's models resource to record each response, unless model uses interactions API.
    """
    client = model.api_client
    if not model.use_interactions_api:
        swap_class(client.aio.models, _models_methods)
    provider = "vertex_ai" if client.vertexai else "gemini"
    return Backend(provider, GEMINI_CODEC, client_headers=False, model=request_model)


def _models_methods(cls: type[AsyncModels]) -> Methods[AsyncModels]:
    """Return `AsyncModels` methods that record response id and URL context on current attempt.

    `LlmResponse` drops `response_id` and first candidate's `url_context_metadata`.
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
        attempt.recorded[RECORDED_ID_KEY] = response.response_id
    candidate = next((candidate for candidate in response.candidates or [] if not candidate.index), None)
    if candidate is not None and candidate.url_context_metadata is not None:
        attempt.recorded[URL_CONTEXT_KEY] = _merged(
            attempt.recorded.get(URL_CONTEXT_KEY), candidate.url_context_metadata
        )


def _merged(recorded: Json, metadata: types.UrlContextMetadata) -> JsonObject:
    """Return recorded URL context dump plus `metadata` URLs not already in it."""
    urls: list[types.UrlMetadata] = []
    if recorded is not None:
        urls.extend(types.UrlContextMetadata.model_validate(recorded).url_metadata or [])
    for url in metadata.url_metadata or []:
        if url not in urls:
            urls.append(url)
    return types.UrlContextMetadata(url_metadata=urls).model_dump(mode="json", exclude_none=True)


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
