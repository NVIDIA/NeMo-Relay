# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test built-in plugins with the Google ADK NeMo Relay integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from _helpers import (
    ScriptedLlmFactory,
    as_object,
    assert_balanced,
    data_of,
    ends,
    final_text,
    make_agent,
    marks,
    model_answer,
    present,
    request_of,
    responding_llm,
    response_of,
    run_recorded,
    starts,
    status_of,
)

from nemo_relay.model_pricing import InlineSource, ModelPricing, PricingCatalog, PricingConfig, TokenPricingRates

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from google.adk.models.google_llm import Gemini
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    import nemo_relay
    from nemo_relay import Json, JsonObject


async def test_pii_redaction_redacts_the_request_through_the_codec_and_omits_the_response(
    scripted_llm: ScriptedLlmFactory, subscribed_events: list[nemo_relay.Event]
) -> None:
    from nemo_relay import plugin
    from nemo_relay.pii_redaction import BuiltinConfig, ComponentSpec, PiiRedactionConfig

    redaction = PiiRedactionConfig(
        tool_input=False,
        tool_output=False,
        builtin=BuiltinConfig(
            action="regex_replace",
            pattern="San Francisco",
            replacement="[CITY]",
            target_path_globs=["/messages/*/content"],
        ),
    )
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(redaction)])):
        adk_events = await run_recorded(make_agent(scripted_llm, "weather_agent"), message="Weather in San Francisco?")

    assert present(scripted_llm.seen_requests[-1].contents[-1].parts)[0].text == "Weather in San Francisco?"
    assert final_text(adk_events) == "Answer from weather_agent to 'Weather in San Francisco?'."
    (llm_start,) = starts(subscribed_events, "llm")
    assert request_of(llm_start).messages[-1]["content"] == "Weather in [CITY]?"
    assert data_of(llm_start)["content"]["contents"][-1]["parts"][0]["text"] == "Weather in [CITY]?"
    (llm_end,) = ends(subscribed_events, "llm")
    assert llm_end.annotated_response is None
    assert llm_end.data is None
    assert status_of(llm_end) == "OK"
    assert_balanced(subscribed_events)


async def test_model_pricing_bills_thinking_tokens_at_the_output_rate(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from nemo_relay import plugin
    from nemo_relay.model_pricing import ComponentSpec

    def priced(request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Priced.")]),
            finish_reason=types.FinishReason.STOP,
            model_version="gemini-2.5-flash",
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1_000_000,
                candidates_token_count=300_000,
                thoughts_token_count=200_000,
                total_token_count=1_500_000,
            ),
        )

    pricing = _flash_pricing()
    model = responding_llm(priced, model="gemini-2.5-flash")
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(pricing)])):
        adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Priced."
    (llm_start,) = starts(subscribed_events, "llm")
    assert llm_start.name == "RespondingLlm"
    (llm_end,) = ends(subscribed_events, "llm")
    assert present(response_of(llm_end).usage)["completion_tokens"] == 300_000
    assert response_of(llm_end).api_specific == {"api": "gemini_generate_content", "thoughts_tokens": 200_000}
    cost = as_object(present(response_of(llm_end).usage)["cost"])
    assert cost["currency"] == "USD"
    assert cost["total"] == pytest.approx(0.30 + 1.25)
    assert cost["source"] == "model_pricing"
    assert_balanced(subscribed_events)


def _flash_pricing() -> PricingConfig:
    rates = TokenPricingRates(input_per_million=0.30, output_per_million=2.50)
    entry = ModelPricing(
        provider="gemini",
        model_id="gemini-2.5-flash",
        pricing_as_of="2026-01-01",
        pricing_source="test",
        rates=rates,
    )
    return PricingConfig(sources=[InlineSource(PricingCatalog(entries=[entry]))])


_URL_CONTEXT: Final[JsonObject] = {"url_metadata": [{"retrieved_url": "https://weather.example"}]}
_SEARCH: Final[JsonObject] = {"web_search_queries": ["weather in Rome"]}
_CODE_EXECUTION: Final[list[Json]] = [
    {"executable_code": {"language": "PYTHON", "code": "print(1)"}},
    {"code_execution_result": {"outcome": "OUTCOME_OK", "output": "1"}},
]


def _canned_gemini(response: types.GenerateContentResponse, monkeypatch: pytest.MonkeyPatch) -> Gemini:
    from google import genai
    from google.adk.models.google_llm import Gemini
    from google.genai.models import AsyncModels

    class CannedModels(AsyncModels):
        def __init__(self, canned: types.GenerateContentResponse) -> None:
            self.canned = canned

        async def generate_content(
            self,
            *,
            model: str,
            contents: types.ContentListUnion | types.ContentListUnionDict,
            config: types.GenerateContentConfigOrDict | None = None,
        ) -> types.GenerateContentResponse:
            return self.canned

        async def generate_content_stream(
            self,
            *,
            model: str,
            contents: types.ContentListUnion | types.ContentListUnionDict,
            config: types.GenerateContentConfigOrDict | None = None,
        ) -> AsyncIterator[types.GenerateContentResponse]:
            canned = self.canned

            async def responses() -> AsyncIterator[types.GenerateContentResponse]:
                yield canned

            return responses()

    client = genai.Client(api_key="test-key")
    monkeypatch.setattr(client.aio, "_models", CannedModels(response))
    return Gemini(model="gemini-2.5-flash", client=client)


@pytest.mark.parametrize(
    ("parts", "candidate", "streaming", "expected_total"),
    [
        pytest.param([], {"url_context_metadata": _URL_CONTEXT}, False, 0.36 + 0.75, id="url_context"),
        pytest.param([], {"url_context_metadata": _URL_CONTEXT}, True, 0.36 + 0.75, id="url_context_streamed"),
        pytest.param([], {"grounding_metadata": _SEARCH}, False, 0.30 + 0.75, id="search"),
        pytest.param(
            [],
            {"grounding_metadata": _SEARCH, "url_context_metadata": _URL_CONTEXT},
            True,
            None,
            id="search_and_url_context_streamed",
        ),
        pytest.param(_CODE_EXECUTION, {"grounding_metadata": _SEARCH}, False, None, id="search_and_code_execution"),
    ],
)
async def test_model_pricing_bills_gemini_tool_use_prompt_tokens_by_tool_evidence(
    parts: list[Json],
    candidate: JsonObject,
    streaming: bool,
    expected_total: float | None,
    monkeypatch: pytest.MonkeyPatch,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.genai import types

    from nemo_relay import plugin
    from nemo_relay.model_pricing import ComponentSpec

    response = types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [*parts, {"text": "Priced."}]},
                    "finish_reason": "STOP",
                    **candidate,
                }
            ],
            "model_version": "gemini-2.5-flash",
            "usage_metadata": {
                "prompt_token_count": 1_000_000,
                "candidates_token_count": 300_000,
                "tool_use_prompt_token_count": 200_000,
                "total_token_count": 1_500_000,
            },
        }
    )
    pricing = _flash_pricing()
    agent = LlmAgent(name="weather_agent", model=_canned_gemini(response, monkeypatch), instruction="Be brief.")
    run_config = RunConfig(streaming_mode=StreamingMode.SSE) if streaming else None
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(pricing)])):
        await run_recorded(agent, run_config=run_config)

    (llm_end,) = ends(subscribed_events, "llm")
    cost = present(response_of(llm_end).usage).get("cost")
    if expected_total is None:
        assert cost is None
    else:
        assert as_object(cost)["total"] == pytest.approx(expected_total)
    assert_balanced(subscribed_events)


async def test_response_cache_bypasses_adk_requests_as_nondeterministic(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.genai import types

    from nemo_relay import plugin
    from nemo_relay.adaptive import AdaptiveConfig, BackendSpec, ComponentSpec, ResponseCacheConfig, StateConfig

    calls = 0

    def count(request: LlmRequest) -> LlmResponse:
        nonlocal calls
        calls += 1
        return model_answer(f"Call {calls}.")

    adaptive = AdaptiveConfig(
        state=StateConfig(backend=BackendSpec.in_memory()),
        response_cache=ResponseCacheConfig(namespace="adk-tests", backend=BackendSpec.in_memory()),
    )
    agent = LlmAgent(
        name="weather_agent",
        model=responding_llm(count, model="gemini-2.5-flash"),
        instruction="Be brief.",
        generate_content_config=types.GenerateContentConfig(temperature=0),
    )
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(adaptive)])):
        first = await run_recorded(agent)
        second = await run_recorded(agent)

    assert calls == 2
    assert final_text(first) == "Call 1."
    assert final_text(second) == "Call 2."
    cache_marks = marks(subscribed_events, "response_cache")
    statuses = [data_of(mark)["status"] for mark in cache_marks if isinstance(mark.data, dict)]
    assert statuses == ["bypass", "bypass"]
    assert_balanced(subscribed_events)
