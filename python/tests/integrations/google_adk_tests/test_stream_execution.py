# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test stream execution intercepts with the Google ADK NeMo Relay integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _helpers import (
    LlmStreamNext,
    ScriptedLlmFactory,
    as_object,
    assert_balanced,
    chunk_indexes,
    data_of,
    ends,
    final_text,
    llm_stream_execution_intercept,
    make_agent,
    marks,
    metadata_of,
    partial_texts,
    present,
    response_of,
    run_recorded,
    status_of,
    stream_intercept,
    text_response,
    uppercase_chunk,
    with_last_text,
)
from pydantic import ValidationError
from typing_extensions import override

import nemo_relay

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.run_config import RunConfig
    from google.adk.models import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse


@pytest.fixture(name="sse")
def sse_fixture() -> RunConfig:
    from google.adk.agents.run_config import RunConfig, StreamingMode

    return RunConfig(streaming_mode=StreamingMode.SSE)


def _stream_cut_after_one_chunk() -> BaseLlm:
    from google.adk.models import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class CutStreamLlm(BaseLlm):
        model: str = "cut-stream"

        @override
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Partial ")]), partial=True)
            raise ConnectionError("cut")

    return CutStreamLlm()


async def test_stream_execution_intercept_rewrites_the_partials_and_the_final_response(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event], sse: RunConfig
) -> None:
    with llm_stream_execution_intercept(
        stream_intercept(uppercase_chunk),
    ):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"), run_config=sse)

    assert partial_texts(adk_events) == ["ANSWER", " FROM WEATHER_AGENT TO 'HI'."]
    assert final_text(adk_events) == "ANSWER FROM WEATHER_AGENT TO 'HI'."
    final = next(event for event in reversed(adk_events) if event.content and not event.partial)
    assert present(final.usage_metadata).total_token_count == 15
    assert_balanced(subscribed_events)
    (llm_end,) = ends(subscribed_events, "llm")
    assert status_of(llm_end) == "OK"
    assert response_of(llm_end).message == "ANSWER FROM WEATHER_AGENT TO 'HI'."
    assert chunk_indexes(subscribed_events) == [0, 1, 2]


async def test_stream_execution_intercept_that_adds_chunks_delivers_each_as_its_own_partial(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event], sse: RunConfig
) -> None:
    def split(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmStreamNext
    ) -> AsyncIterator[nemo_relay.Json]:
        async def gen() -> AsyncIterator[nemo_relay.Json]:
            async for chunk in await next_call(request):
                yield chunk
                if isinstance(chunk, dict) and chunk.get("partial"):
                    yield {"content": {"role": "model", "parts": [{"text": "+"}]}, "partial": True}

        return gen()

    with llm_stream_execution_intercept(
        split,
    ):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"), run_config=sse)

    assert partial_texts(adk_events) == ["Answer", "+", " from weather_agent to 'hi'.", "+"]
    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    assert_balanced(subscribed_events)
    assert chunk_indexes(subscribed_events) == [0, 1, 2, 3, 4]


async def test_stream_execution_intercept_that_drops_a_chunk_suppresses_its_partial(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event], sse: RunConfig
) -> None:
    def drop_first(chunk: nemo_relay.Json) -> nemo_relay.Json | None:
        content = as_object(as_object(chunk)["content"])
        return None if content["parts"][0]["text"] == "Answer" else chunk

    with llm_stream_execution_intercept(
        stream_intercept(drop_first),
    ):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"), run_config=sse)

    assert partial_texts(adk_events) == [" from weather_agent to 'hi'."]
    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    assert_balanced(subscribed_events)
    assert chunk_indexes(subscribed_events) == [0, 1]


async def test_stream_execution_intercept_that_answers_skips_the_model(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event], sse: RunConfig
) -> None:
    def answer(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmStreamNext
    ) -> AsyncIterator[nemo_relay.Json]:
        async def gen() -> AsyncIterator[nemo_relay.Json]:
            yield {"content": {"role": "model", "parts": [{"text": "Relay "}]}, "partial": True}
            yield {"content": {"role": "model", "parts": [{"text": "answered."}]}, "partial": True}
            yield {**text_response("Relay answered."), "partial": False}

        return gen()

    with llm_stream_execution_intercept(
        answer,
    ):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"), run_config=sse)

    assert scripted_llm.seen_requests == []
    assert partial_texts(adk_events) == ["Relay ", "answered."]
    assert final_text(adk_events) == "Relay answered."
    assert_balanced(subscribed_events)


async def test_a_chunk_adk_cannot_take_fails_the_stream_under_a_rejection_mark(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event], sse: RunConfig
) -> None:
    def answer(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmStreamNext
    ) -> AsyncIterator[nemo_relay.Json]:
        async def gen() -> AsyncIterator[nemo_relay.Json]:
            yield {"partial": "nope"}

        return gen()

    with (
        llm_stream_execution_intercept(answer),
        pytest.raises(ValidationError, match="partial") as raised,
    ):
        await run_recorded(make_agent(scripted_llm, "weather_agent"), run_config=sse)

    (llm_end,) = ends(subscribed_events, "llm")
    (rejected,) = marks(subscribed_events, "adk.llm_response_rejected")
    assert rejected.parent_uuid == llm_end.parent_uuid
    assert data_of(rejected) == {
        "provider": llm_end.name,
        "model": "scripted-weather_agent",
        "error_type": "ValidationError",
        "error_message": str(raised.value),
    }
    assert_balanced(subscribed_events)


async def test_stream_execution_intercept_that_retries_gets_a_second_stream(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event], sse: RunConfig
) -> None:
    def retry(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmStreamNext
    ) -> AsyncIterator[nemo_relay.Json]:
        async def gen() -> AsyncIterator[nemo_relay.Json]:
            async for _ in await next_call(request):
                pass
            content = with_last_text(request.content, "second try")
            async for chunk in await next_call(nemo_relay.LLMRequest(request.headers, content)):
                yield chunk

        return gen()

    with llm_stream_execution_intercept(
        retry,
    ):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"), run_config=sse)

    assert len(scripted_llm.seen_requests) == 2
    assert partial_texts(adk_events) == ["Answer", " from weather_agent to 'second try'."]
    assert final_text(adk_events) == "Answer from weather_agent to 'second try'."
    assert_balanced(subscribed_events)
    (llm_end,) = ends(subscribed_events, "llm")
    assert response_of(llm_end).message == "Answer from weather_agent to 'second try'."


async def test_model_stream_error_recovered_by_a_stream_execution_intercept(
    subscribed_events: list[nemo_relay.Event], sse: RunConfig
) -> None:
    from google.adk.agents import LlmAgent

    def recover(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmStreamNext
    ) -> AsyncIterator[nemo_relay.Json]:
        async def gen() -> AsyncIterator[nemo_relay.Json]:
            try:
                async for chunk in await next_call(request):
                    yield chunk
            except ConnectionError:
                yield {**text_response("Partial recovered."), "partial": False}

        return gen()

    with llm_stream_execution_intercept(
        recover,
    ):
        agent = LlmAgent(name="weather_agent", model=_stream_cut_after_one_chunk(), instruction="Be brief.")
        adk_events = await run_recorded(agent, run_config=sse)

    assert partial_texts(adk_events) == ["Partial "]
    assert final_text(adk_events) == "Partial recovered."
    (llm_end,) = ends(subscribed_events, "llm")
    assert status_of(llm_end) == "OK"
    assert_balanced(subscribed_events)


async def test_model_stream_error_without_recovery_reaches_adk_error_hooks(
    subscribed_events: list[nemo_relay.Event], sse: RunConfig
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    def on_model_error(
        callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
    ) -> LlmResponse | None:
        return LlmResponse(content=types.ModelContent(f"ADK handled {error}"))

    agent = LlmAgent(
        name="weather_agent",
        model=_stream_cut_after_one_chunk(),
        instruction="Be brief.",
        on_model_error_callback=on_model_error,
    )
    adk_events = await run_recorded(agent, run_config=sse)

    assert partial_texts(adk_events) == ["Partial "]
    assert final_text(adk_events) == "ADK handled cut"
    (llm_end,) = ends(subscribed_events, "llm")
    assert status_of(llm_end) == "ERROR"
    assert metadata_of(llm_end)["exception.type"] == "ConnectionError"
    assert_balanced(subscribed_events)
