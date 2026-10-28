# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ADK model calls through managed LLM pipeline via swapped `BaseLlm` subclasses."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Protocol, TypeAlias

from google.adk.models import FallbackModel, LLMRegistry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse

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
from nemo_relay.integrations.google_adk._swap import Methods, swap_class

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

_Responses: TypeAlias = AsyncGenerator[LlmResponse, None]


class _Inner(Protocol):
    def __call__(self, request: LlmRequest, *, stream: bool) -> _Responses: ...


def wrap_agent_models(agent: LlmAgent) -> None:
    """Swap agent's class for a subclass that wraps each model the agent selects for a call.

    ADK selects each call's model through `canonical_model_async`, which a subclass can override to pick a model per
    invocation. Compositional function calling runs keep their model unwrapped, since their live connection bypasses
    managed pipeline.
    """
    swap_class(agent, _agent_methods)


def _agent_methods(cls: type[LlmAgent]) -> Methods[LlmAgent]:
    original = cls.canonical_model_async

    async def canonical_model_async(self: LlmAgent, ctx: ReadonlyContext) -> BaseLlm:
        model = await original(self, ctx)
        run_config = ctx.run_config
        if run_config is None or not run_config.support_cfc:
            wrap_model(model)
        return model

    return {"canonical_model_async": canonical_model_async}


def wrap_model(model: BaseLlm) -> None:
    """Swap model's class for a subclass whose calls run through Relay.

    For `FallbackModel`, wrap each delegate instead. Replace each delegate given by name with the model `LLMRegistry`
    builds for it.
    """
    if isinstance(model, FallbackModel):
        for index, entry in enumerate(model.models):
            delegate = entry if isinstance(entry, BaseLlm) else LLMRegistry.new_llm(entry)
            model.models[index] = delegate
            wrap_model(delegate)
        return
    swap_class(model, _methods)


def _methods(cls: type[BaseLlm]) -> Methods[BaseLlm]:
    original = cls.generate_content_async

    async def generate_content_async(self: BaseLlm, llm_request: LlmRequest, stream: bool = False) -> _Responses:
        backend = await backend_of(self, llm_request.model)

        def inner(request: LlmRequest, *, stream: bool) -> _Responses:
            return original(self, request, stream)

        async with contextlib.aclosing(
            _run(llm_request, inner, backend=backend, model_type=cls.__name__, stream=stream)
        ) as responses:
            async for response in responses:
                yield response

    return {"generate_content_async": generate_content_async}


async def _attempted(responses: _Responses, attempt: Attempt) -> _Responses:
    """Yield ADK's responses, with `attempt` as current attempt while each is awaited."""
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
        async with contextlib.AsyncExitStack() as closing:
            for stream in attempt.streams:
                closing.push_async_callback(stream.aclose)
            await responses.aclose()


def _function_calling_config(request: LlmRequest) -> types.FunctionCallingConfig | None:
    tool_config = request.config.tool_config
    return None if tool_config is None else tool_config.function_calling_config


def _tool_choice_edit(
    edited: LlmRequest, original: types.FunctionCallingConfig | None
) -> types.FunctionCallingConfig | None:
    """Return edited function calling config when it differs from `original`, else None."""
    config = _function_calling_config(edited)
    return None if config is None or config == original else config


async def _run(
    llm_request: LlmRequest, inner: _Inner, *, backend: Backend, model_type: str, stream: bool
) -> _Responses:
    """Run one ADK model call as one managed LLM call.

    Non-streaming call yields one restored response. Streaming call yields one restored response per chunk.
    """
    parent = parent_handle()
    with rejection_marked("adk.llm_request_rejected", parent, provider=backend.provider, model=backend.model):
        payload = request_payload(llm_request)
    adk_tool_choice = _function_calling_config(llm_request)
    current = payload
    produced = ProducedResponses()
    in_flight = 0

    def target(request: LLMRequest) -> tuple[LlmRequest, Attempt]:
        """Return ADK's request with pipeline's edits, plus a new attempt.

        Edit ADK's request in place when no other attempt is in flight, else edit a copy. For a backend with client
        headers, put request headers on attempt. Otherwise write them into request's HTTP options.
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
            return edited, Attempt(tool_choice=tool_choice, headers=headers, provider=backend.provider)
        write_headers(edited, headers)
        return edited, Attempt(tool_choice=tool_choice, headers={}, provider=backend.provider)

    def added(response: LlmResponse, attempt: Attempt) -> JsonObject:
        return produced.add(response, recorded=attempt.recorded, response_model=attempt.response_model)

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
            model_name=backend.model,
            codec=backend.codec,
            response_codec=backend.codec,
        )
        with rejection_marked("adk.llm_response_rejected", parent, provider=backend.provider, model=backend.model):
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
        model_name=backend.model,
        codec=backend.codec,
        response_codec=backend.codec,
    )
    try:
        async for chunk in llm_stream:
            with rejection_marked("adk.llm_response_rejected", parent, provider=backend.provider, model=backend.model):
                response = produced.restore_chunk(chunk)
            yield response
    finally:
        await llm_stream.aclose()
