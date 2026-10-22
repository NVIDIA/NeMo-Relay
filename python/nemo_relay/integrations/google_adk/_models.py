# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ADK's model calls through the managed LLM pipeline with swapped `BaseLlm` subclasses."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import TYPE_CHECKING, Protocol, TypeAlias

from google.adk.models import FallbackModel, LLMRegistry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import nemo_relay
from nemo_relay import Json, JsonObject, LLMAttributes, LLMRequest
from nemo_relay.integrations.google_adk._backends import CURRENT_ATTEMPT, Attempt, Backend, backend_of
from nemo_relay.integrations.google_adk._context import integration_metadata, parent_handle, rejection_marked
from nemo_relay.integrations.google_adk._serialization import (
    ProducedResponses,
    final_response,
    header_text,
    http_headers,
    request_payload,
    write_back_request,
    write_headers,
)
from nemo_relay.integrations.google_adk._swap import swap_class

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.models.llm_request import LlmRequest

_Responses: TypeAlias = AsyncGenerator[LlmResponse, None]


class _Inner(Protocol):
    def __call__(self, request: LlmRequest, *, stream: bool) -> _Responses: ...


def wrap_agent_models(agent: LlmAgent) -> None:
    """Swap the agent's class for a subclass that wraps each model the agent selects for a call.

    ADK selects the model of each call through `canonical_model_async`, which a subclass can override to choose another
    model per invocation. The wrap leaves the model of a compositional function calling run unwrapped, as its live
    connection bypasses the managed pipeline.
    """
    swap_class(agent, _agent_methods)


def _agent_methods(cls: type[LlmAgent]) -> Mapping[str, object]:
    original = cls.canonical_model_async

    async def canonical_model_async(self: LlmAgent, ctx: ReadonlyContext) -> BaseLlm:
        model = await original(self, ctx)
        run_config = ctx.run_config
        if run_config is None or not run_config.support_cfc:
            wrap_model(model)
        return model

    return {"canonical_model_async": canonical_model_async}


def wrap_model(model: BaseLlm) -> None:
    """Swap the model's class for a subclass whose calls run through Relay.

    For a `FallbackModel`, wrap each delegate instead, and replace each delegate given by name with the model that
    `LLMRegistry` builds for it.
    """
    if isinstance(model, FallbackModel):
        for index, entry in enumerate(model.models):
            delegate = entry if isinstance(entry, BaseLlm) else LLMRegistry.new_llm(entry)
            model.models[index] = delegate
            wrap_model(delegate)
        return
    swap_class(model, _methods)


def _methods(cls: type[BaseLlm]) -> Mapping[str, object]:
    original = cls.generate_content_async

    async def generate_content_async(self: BaseLlm, llm_request: LlmRequest, stream: bool = False) -> _Responses:
        backend = await backend_of(self)

        def inner(request: LlmRequest, *, stream: bool) -> _Responses:
            return original(self, request, stream)

        async with contextlib.aclosing(
            _run(llm_request, inner, backend=backend, model_name=self.model, model_type=cls.__name__, stream=stream)
        ) as responses:
            async for response in responses:
                yield response

    return {"generate_content_async": generate_content_async}


async def _attempted(responses: _Responses, attempt: Attempt) -> _Responses:
    """Yield ADK's responses, and set `attempt` as the current attempt while each response is awaited."""
    try:
        while True:
            token = CURRENT_ATTEMPT.set(attempt)
            try:
                response = await anext(responses)
            except StopAsyncIteration:
                return
            finally:
                CURRENT_ATTEMPT.reset(token)
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


def _function_calling_config(request: LlmRequest) -> types.FunctionCallingConfig | None:
    tool_config = request.config.tool_config
    return None if tool_config is None else tool_config.function_calling_config


def _tool_choice_edit(
    edited: LlmRequest, original: types.FunctionCallingConfig | None
) -> types.FunctionCallingConfig | None:
    """Return the edited function calling config when that config differs from ADK's original, or None otherwise."""
    config = _function_calling_config(edited)
    return None if config is None or config == original else config


async def _run(
    llm_request: LlmRequest, inner: _Inner, *, backend: Backend, model_name: str, model_type: str, stream: bool
) -> _Responses:
    """Run one ADK model call as one managed LLM call.

    A non-streaming call yields one restored response, and a streaming call yields one restored response for each chunk.
    """
    parent = parent_handle()
    with rejection_marked("adk.llm_request_rejected", parent, provider=backend.provider, model=model_name):
        payload = request_payload(llm_request)
    adk_tool_choice = _function_calling_config(llm_request)
    current = payload
    produced = ProducedResponses()
    in_flight = 0

    def target(request: LLMRequest) -> tuple[LlmRequest, Attempt]:
        """Return ADK's request with the pipeline's edits, and a new attempt.

        Edit ADK's request in place when no other attempt is in flight, and edit a copy otherwise. Put the request
        headers on the attempt for a backend with client headers, and write them into the request's HTTP options
        otherwise.
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
        headers = header_text(request.headers)
        tool_choice = _tool_choice_edit(edited, adk_tool_choice)
        if backend.client_headers:
            return edited, Attempt(tool_choice=tool_choice, headers=headers)
        write_headers(edited, headers)
        return edited, Attempt(tool_choice=tool_choice, headers={})

    def added(response: LlmResponse, attempt: Attempt) -> JsonObject:
        return produced.add(
            response,
            url_context=_merged(attempt.urls),
            response_id=attempt.response_id,
            response_model=attempt.response_model,
            provider_cost=attempt.provider_cost,
        )

    async def call(request: LLMRequest) -> Json:
        nonlocal in_flight
        edited, attempt = target(request)
        in_flight += 1
        last: JsonObject | None = None
        try:
            async with contextlib.aclosing(_attempted(inner(edited, stream=False), attempt)) as responses:
                async for response in responses:
                    last = added(response, attempt)
        finally:
            in_flight -= 1
        return last

    async def produce(request: LLMRequest) -> AsyncIterator[Json]:
        nonlocal in_flight
        edited, attempt = target(request)
        in_flight += 1
        try:
            async with contextlib.aclosing(_attempted(inner(edited, stream=True), attempt)) as responses:
                async for response in responses:
                    yield added(response, attempt)
        finally:
            in_flight -= 1

    collected: list[Json] = []

    def finalize() -> Json:
        return final_response(collected)

    request = LLMRequest({} if backend.client_headers else http_headers(llm_request), payload)
    metadata = integration_metadata(adk_model_type=model_type)
    if not stream:
        result = await nemo_relay.llm.execute(
            backend.provider,
            request,
            call,
            handle=parent,
            metadata=metadata,
            model_name=model_name,
            codec=backend.codec,
            response_codec=backend.codec,
        )
        with rejection_marked("adk.llm_response_rejected", parent, provider=backend.provider, model=model_name):
            response = produced.restore_result(result)
        yield response
        return
    llm_stream = await nemo_relay.llm.stream_execute(
        backend.provider,
        request,
        produce,
        collected.append,
        finalize,
        handle=parent,
        attributes=LLMAttributes(LLMAttributes.STREAMING),
        metadata=metadata,
        model_name=model_name,
        codec=backend.codec,
        response_codec=backend.codec,
    )
    try:
        async for chunk in llm_stream:
            with rejection_marked("adk.llm_response_rejected", parent, provider=backend.provider, model=model_name):
                response = produced.restore_chunk(chunk)
            yield response
    finally:
        await llm_stream.aclose()
