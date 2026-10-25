# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test guardrails with the Google ADK NeMo Relay integration."""

from __future__ import annotations

from uuid import uuid4

import pytest
from _helpers import (
    ScriptedLlmFactory,
    assert_balanced,
    final_text,
    get_weather,
    llm_conditional_execution_guardrail,
    llm_sanitize_request_guardrail,
    make_agent,
    marks,
    named_end,
    request_of,
    run_app,
    run_recorded,
    starts,
    status_of,
    tool_conditional_execution_guardrail,
    weather_call,
    with_last_text,
)

import nemo_relay


async def test_tool_guardrail_rejection_propagates_through_adk_error_chain(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    def block_san_francisco(name: str, args: nemo_relay.Json) -> str | None:
        if name == "get_weather" and isinstance(args, dict) and args.get("location") == "San Francisco":
            return "San Francisco is not allowed"
        return None

    with (
        tool_conditional_execution_guardrail(
            block_san_francisco,
        ),
        pytest.raises(RuntimeError, match="guardrail rejected"),
    ):
        await run_recorded(make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call()))

    assert_balanced(subscribed_events)
    assert starts(subscribed_events, "tool") == []
    (rejection,) = marks(subscribed_events, "get_weather")
    assert rejection.data == {"rejected": True, "rejection_reason": "San Francisco is not allowed"}
    agent_end = named_end(subscribed_events, "weather_agent", "agent")
    assert status_of(agent_end) == "ERROR"


async def test_model_guardrail_rejection_propagates_by_default(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    with (
        llm_conditional_execution_guardrail(
            lambda request: "no model calls",
        ),
        pytest.raises(RuntimeError, match="guardrail rejected: no model calls"),
    ):
        await run_recorded(make_agent(scripted_llm, "weather_agent"))

    assert scripted_llm.seen_requests == []
    assert starts(subscribed_events, "llm") == []
    (rejection,) = marks(subscribed_events, "ScriptedLlm")
    assert rejection.data == {"rejected": True, "rejection_reason": "no model calls"}
    assert_balanced(subscribed_events)


async def test_guardrail_error_fails_closed(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    def broken(name: str, args: nemo_relay.Json) -> str | None:
        raise ValueError("guardrail backend unavailable")

    with (
        tool_conditional_execution_guardrail(
            broken,
        ),
        pytest.raises(ValueError, match="guardrail backend unavailable"),
    ):
        await run_recorded(make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call()))

    assert starts(subscribed_events, "tool") == []
    assert_balanced(subscribed_events)


async def test_scope_local_middleware_applies_to_adk_calls(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent) as request:
        nemo_relay.scope_local.register_tool_conditional_execution(
            request, f"adk-local-block-{uuid4()}", 1, lambda name, args: "blocked by a request-local guardrail"
        )
        agent = make_agent(scripted_llm, "weather_agent", tools=[get_weather], call=weather_call())
        with pytest.raises(RuntimeError, match="blocked by a request-local guardrail"):
            await run_app(agent)
    await nemo_relay.subscribers.flush_async()

    assert starts(subscribed_events, "tool") == []
    assert_balanced(subscribed_events)


async def test_sanitize_request_guardrail_rewrites_the_recorded_annotation(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    def redact_user_text(
        request: nemo_relay.LLMRequest, context: nemo_relay.LlmSanitizeRequestContext
    ) -> nemo_relay.LLMRequest:
        content = with_last_text(request.content, "[REDACTED]")
        return nemo_relay.LLMRequest(request.headers, content)

    with llm_sanitize_request_guardrail(
        redact_user_text,
    ):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"))

    assert final_text(adk_events) == "Answer from weather_agent to 'hi'."
    (llm_start,) = starts(subscribed_events, "llm")
    assert request_of(llm_start).messages[-1]["content"] == "[REDACTED]"
    assert_balanced(subscribed_events)
