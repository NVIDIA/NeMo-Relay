# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ADK's model calls through the managed LLM pipeline with swapped `BaseLlm` subclasses."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final, ParamSpec, Protocol, TypeAlias, cast

from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import nemo_relay
from nemo_relay import Json, JsonObject, LLMAttributes, LLMRequest
from nemo_relay.integrations.google_adk._context import integration_metadata, parent_handle, rejection_marked
from nemo_relay.integrations.google_adk._serialization import (
    CODEC,
    ProducedResponses,
    aggregate_partials,
    header_text,
    http_headers,
    request_payload,
    resolve_provider,
    write_back_request,
    write_headers,
)
from nemo_relay.integrations.google_adk._swap import swap_class

if TYPE_CHECKING:
    from google.adk.models import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.genai.models import AsyncModels

_Responses: TypeAlias = AsyncGenerator[LlmResponse, None]
_Stream: TypeAlias = AsyncGenerator[types.GenerateContentResponse, None]
_P = ParamSpec("_P")
_URL_CONTEXT: Final[ContextVar[list[types.UrlContextMetadata] | None]] = ContextVar(
    "nemo_relay_adk_url_context", default=None
)


class _Inner(Protocol):
    def __call__(self, request: LlmRequest, *, stream: bool) -> _Responses: ...


def wrap_model(model: BaseLlm) -> None:
    """Swap the model's class for a subclass whose calls run through Relay."""
    swap_class(model, _methods)


def _methods(cls: type[BaseLlm]) -> Mapping[str, object]:
    original = cls.generate_content_async

    async def generate_content_async(self: BaseLlm, llm_request: LlmRequest, stream: bool = False) -> _Responses:
        if isinstance(self, Gemini) and not self.use_interactions_api:
            swap_class(self.api_client.aio.models, _client_methods)

        def inner(request: LlmRequest, *, stream: bool) -> _Responses:
            return original(self, request, stream)

        async with contextlib.aclosing(
            _run(llm_request, inner, model_name=self.model, model_type=cls.__name__, stream=stream)
        ) as responses:
            async for response in responses:
                yield response

    return {"generate_content_async": generate_content_async}


def _client_methods(cls: type[AsyncModels]) -> Mapping[str, object]:
    """Return `AsyncModels` methods that record the first candidate's `url_context_metadata`.

    ADK's `LlmResponse` drops that metadata. The methods append it to the current URL context collector when one is set.
    """
    return {
        "generate_content": _recorded(cls.generate_content),
        "generate_content_stream": _recorded_stream(cls.generate_content_stream),
    }


def _record(response: types.GenerateContentResponse) -> None:
    urls = _URL_CONTEXT.get()
    candidate = next((candidate for candidate in response.candidates or [] if not candidate.index), None)
    if urls is not None and candidate is not None and candidate.url_context_metadata is not None:
        urls.append(candidate.url_context_metadata)


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


async def _with_url_context(responses: _Responses, urls: list[types.UrlContextMetadata]) -> _Responses:
    """Yield ADK's responses, and set `urls` as the current URL context collector while each response is awaited."""
    try:
        while True:
            token = _URL_CONTEXT.set(urls)
            try:
                response = await anext(responses)
            except StopAsyncIteration:
                return
            finally:
                _URL_CONTEXT.reset(token)
            yield response
    finally:
        await responses.aclose()


def _merged(urls: list[types.UrlContextMetadata]) -> types.UrlContextMetadata | None:
    if not urls:
        return None
    merged: list[types.UrlMetadata] = []
    for url in (url for metadata in urls for url in metadata.url_metadata or []):
        if url not in merged:
            merged.append(url)
    return types.UrlContextMetadata(url_metadata=merged)


def _is_final(chunk: Json) -> bool:
    return isinstance(chunk, dict) and not chunk.get("partial")


async def _run(llm_request: LlmRequest, inner: _Inner, *, model_name: str, model_type: str, stream: bool) -> _Responses:
    """Run one ADK model call as one managed LLM call.

    A non-streaming call yields one restored response, and a streaming call yields one restored response for each chunk.
    """
    name = resolve_provider(model_name)
    parent = parent_handle()
    with rejection_marked("adk.llm_request_rejected", parent, provider=name, model=model_name):
        payload = request_payload(llm_request)
    current = payload
    produced = ProducedResponses()
    in_flight = 0

    def target(request: LLMRequest) -> LlmRequest:
        """Return ADK's request with the pipeline's edits.

        Edit ADK's request in place when no other call is in flight, and edit a copy otherwise. Write the request
        headers into the request's HTTP options.
        """
        nonlocal current
        content = request.content
        if in_flight == 0:
            edited = llm_request
            if write_back_request(edited, payload=content, original_payload=current):
                current = content
        else:
            edited = llm_request.model_copy()
            write_back_request(edited, payload=content, original_payload=current)
        write_headers(edited, header_text(request.headers))
        return edited

    async def call(request: LLMRequest) -> Json:
        nonlocal in_flight
        edited = target(request)
        in_flight += 1
        last: JsonObject | None = None
        urls: list[types.UrlContextMetadata] = []
        try:
            async with contextlib.aclosing(_with_url_context(inner(edited, stream=False), urls)) as responses:
                async for response in responses:
                    last = produced.add(response, url_context=_merged(urls))
        finally:
            in_flight -= 1
        return last

    async def produce(request: LLMRequest) -> AsyncIterator[Json]:
        nonlocal in_flight
        edited = target(request)
        in_flight += 1
        urls: list[types.UrlContextMetadata] = []
        try:
            async with contextlib.aclosing(_with_url_context(inner(edited, stream=True), urls)) as responses:
                async for response in responses:
                    yield produced.add(response, url_context=_merged(urls))
        finally:
            in_flight -= 1

    collected: list[Json] = []

    def finalize() -> Json:
        """Return the last whole chunk of the stream.

        Aggregate the partial chunks when the stream closed before a whole chunk arrived.
        """
        whole = next((chunk for chunk in reversed(collected) if _is_final(chunk)), None)
        return whole if whole is not None else aggregate_partials(collected)

    request = LLMRequest(http_headers(llm_request), payload)
    metadata = integration_metadata(adk_model_type=model_type)
    if not stream:
        result = await nemo_relay.llm.execute(
            name,
            request,
            call,
            handle=parent,
            metadata=metadata,
            model_name=model_name,
            codec=CODEC,
            response_codec=CODEC,
        )
        with rejection_marked("adk.llm_response_rejected", parent, provider=name, model=model_name):
            response = produced.restore_result(result)
        yield response
        return
    llm_stream = await nemo_relay.llm.stream_execute(
        name,
        request,
        produce,
        collected.append,
        finalize,
        handle=parent,
        attributes=LLMAttributes(LLMAttributes.STREAMING),
        metadata=metadata,
        model_name=model_name,
        codec=CODEC,
        response_codec=CODEC,
    )
    try:
        async for chunk in llm_stream:
            with rejection_marked("adk.llm_response_rejected", parent, provider=name, model=model_name):
                response = produced.restore_chunk(chunk)
            yield response
    finally:
        await llm_stream.aclose()
