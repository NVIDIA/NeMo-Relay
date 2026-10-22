# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test LLM execution intercepts with the Google ADK NeMo Relay integration."""

from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING

import pytest
from _helpers import (
    HANG_TIMEOUT,
    LlmNext,
    ScriptedLlmFactory,
    assert_balanced,
    data_of,
    ends,
    final_text,
    llm_execution_intercept,
    make_agent,
    marks,
    metadata_of,
    model_answer,
    present,
    responding_llm,
    response_of,
    run_recorded,
    starts,
    status_of,
    text_response,
)
from pydantic import ValidationError
from typing_extensions import override

import nemo_relay

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse


def _connection_down(request: LlmRequest) -> LlmResponse:
    raise ConnectionError("down")


async def test_llm_execution_intercept_rewrites_the_request_and_the_response(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    async def rewrite(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmNext
    ) -> nemo_relay.Json:
        content = copy.deepcopy(request.content)
        content["contents"][-1]["parts"][0]["text"] = "REWRITTEN"
        response = await next_call(nemo_relay.LLMRequest(request.headers, content))
        response["content"]["parts"][0]["text"] += " [stamped]"
        return response

    with llm_execution_intercept(rewrite):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"))

    assert present(scripted_llm.seen_requests[-1].contents[-1].parts)[0].text == "REWRITTEN"
    assert final_text(adk_events) == "Answer from weather_agent to 'REWRITTEN'. [stamped]"
    assert_balanced(subscribed_events)
    (llm_end,) = ends(subscribed_events, "llm")
    assert status_of(llm_end) == "OK"
    assert "[stamped]" in str(response_of(llm_end).message)


async def test_llm_execution_intercept_that_answers_skips_the_model(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    with llm_execution_intercept(
        lambda name, request, context, next_call: text_response("Answered by Relay."),
    ):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"))

    assert final_text(adk_events) == "Answered by Relay."
    assert scripted_llm.seen_requests == []
    assert_balanced(subscribed_events)
    (llm_start,) = starts(subscribed_events, "llm")
    (llm_end,) = ends(subscribed_events, "llm")
    assert llm_end.uuid == llm_start.uuid
    assert status_of(llm_end) == "OK"


async def test_a_request_relay_cannot_project_fails_the_call_under_a_rejection_mark(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    class Opaque:
        pass

    agent = make_agent(scripted_llm, "weather_agent")
    agent.output_schema = list[Opaque]
    with pytest.raises(TypeError, match="cannot render the structured output schema") as raised:
        await run_recorded(agent)

    assert scripted_llm.seen_requests == []
    assert starts(subscribed_events, "llm") == []
    (rejected,) = marks(subscribed_events, "adk.llm_request_rejected")
    (agent_start,) = [event for event in starts(subscribed_events, "agent") if event.name == "weather_agent"]
    assert rejected.parent_uuid == agent_start.uuid
    assert data_of(rejected) == {
        "provider": "ScriptedLlm",
        "model": "scripted-weather_agent",
        "error_type": "TypeError",
        "error_message": str(raised.value),
    }
    assert_balanced(subscribed_events)


async def test_a_response_adk_cannot_take_fails_the_call_under_a_rejection_mark(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    with (
        llm_execution_intercept(lambda name, request, context, next_call: {"partial": "nope"}),
        pytest.raises(ValidationError, match="partial") as raised,
    ):
        await run_recorded(make_agent(scripted_llm, "weather_agent"))

    (llm_end,) = ends(subscribed_events, "llm")
    (rejected,) = marks(subscribed_events, "adk.llm_response_rejected")
    assert rejected.parent_uuid == llm_end.parent_uuid
    assert data_of(rejected) == {
        "provider": llm_end.name,
        "model": "scripted-weather_agent",
        "error_type": "ValidationError",
        "error_message": str(raised.value),
    }
    assert metadata_of(rejected)["nemo_relay.log.severity"] == "error"
    assert_balanced(subscribed_events)


async def test_llm_execution_intercept_that_retries_calls_the_model_again(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    async def retry(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmNext
    ) -> nemo_relay.Json:
        await next_call(request)
        content = copy.deepcopy(request.content)
        content["contents"][-1]["parts"][0]["text"] = "second try"
        return await next_call(nemo_relay.LLMRequest(request.headers, content))

    with llm_execution_intercept(retry):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"))

    yielded_texts = [str(response.content.parts[0].text) for response in scripted_llm.yielded if response.content]
    assert yielded_texts == ["Answer from weather_agent to 'hi'.", "Answer from weather_agent to 'second try'."]
    assert final_text(adk_events) == "Answer from weather_agent to 'second try'."
    assert len([event for event in adk_events if event.content and not event.partial]) == 1
    assert_balanced(subscribed_events)
    (llm_end,) = ends(subscribed_events, "llm")
    assert response_of(llm_end).message == "Answer from weather_agent to 'second try'."


async def test_model_error_retried_by_an_execution_intercept(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent

    calls = 0

    def flaky(request: LlmRequest) -> LlmResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("transient")
        return model_answer("Recovered.")

    async def retry_once(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmNext
    ) -> nemo_relay.Json:
        try:
            return await next_call(request)
        except ConnectionError:
            return await next_call(request)

    agent = LlmAgent(name="weather_agent", model=responding_llm(flaky, model="flaky-model"), instruction="Be brief.")
    with llm_execution_intercept(retry_once):
        adk_events = await run_recorded(agent)

    assert calls == 2
    assert final_text(adk_events) == "Recovered."
    (llm_end,) = ends(subscribed_events, "llm")
    assert status_of(llm_end) == "OK"
    assert_balanced(subscribed_events)


async def test_model_error_recovered_by_an_execution_intercept(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent

    async def recover(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmNext
    ) -> nemo_relay.Json:
        try:
            return await next_call(request)
        except ConnectionError:
            return text_response("Fallback answer.")

    agent = LlmAgent(
        name="weather_agent", model=responding_llm(_connection_down, model="broken-model"), instruction="Be brief."
    )
    with llm_execution_intercept(recover):
        adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Fallback answer."
    (llm_end,) = ends(subscribed_events, "llm")
    assert status_of(llm_end) == "OK"
    assert_balanced(subscribed_events)


async def test_model_error_without_recovery_reaches_adk_error_hooks(
    subscribed_events: list[nemo_relay.Event],
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
        model=responding_llm(_connection_down, model="broken-model"),
        instruction="Be brief.",
        on_model_error_callback=on_model_error,
    )
    adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "ADK handled down"
    (llm_end,) = ends(subscribed_events, "llm")
    assert status_of(llm_end) == "ERROR"
    assert metadata_of(llm_end)["exception.type"] == "ConnectionError"
    assert_balanced(subscribed_events)


async def test_a_hedge_makes_its_calls_at_the_same_time(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models import BaseLlm

    both_in_flight = asyncio.Barrier(2)
    texts: list[str] = []

    class RendezvousLlm(BaseLlm):
        model: str = "rendezvous-model"

        @override
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            parts = llm_request.contents[-1].parts or []
            text = str(parts[0].text)
            texts.append(text)
            await both_in_flight.wait()
            yield model_answer(f"echo {text}")

    async def hedge(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmNext
    ) -> nemo_relay.Json:
        alternative = copy.deepcopy(request.content)
        alternative["contents"][-1]["parts"][0]["text"] = "hedged"
        _primary, secondary = await asyncio.gather(
            next_call(request), next_call(nemo_relay.LLMRequest(request.headers, alternative))
        )
        return secondary

    with llm_execution_intercept(hedge):
        async with asyncio.timeout(HANG_TIMEOUT):
            adk_events = await run_recorded(
                LlmAgent(name="weather_agent", model=RendezvousLlm(), instruction="Be brief.")
            )

    assert sorted(texts) == ["hedged", "hi"]
    assert final_text(adk_events) == "echo hedged"
    assert_balanced(subscribed_events)
