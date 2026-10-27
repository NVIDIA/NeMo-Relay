# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test ADK's provider model classes with the Google ADK NeMo Relay integration."""

from __future__ import annotations

import functools
import json
import re
from typing import TYPE_CHECKING, Final
from unittest import mock

import httpx
import pytest
from _helpers import (
    as_list,
    as_object,
    assert_balanced,
    data_of,
    ends,
    final_text,
    get_weather,
    llm_request_intercept,
    marks,
    metadata_of,
    model_answer,
    partial_texts,
    present,
    response_of,
    run_recorded,
    starts,
    status_of,
)
from typing_extensions import override

import nemo_relay
from nemo_relay.model_pricing import (
    ComponentSpec,
    InlineSource,
    ModelPricing,
    PricingCatalog,
    PricingConfig,
    PromptCachePricing,
    TokenPricingRates,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from anthropic import AsyncAnthropic
    from google.adk.models import BaseLlm
    from google.adk.models.google_llm import Gemini
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from openai import AsyncOpenAI

_CLAUDE: Final = "claude-sonnet-4-5-20250929"
_GPT: Final = "gpt-4o-2024-08-06"
_WEATHER_CALL: Final[nemo_relay.JsonObject] = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "get_weather",
    "input": {"location": "Paris"},
}
_ANTHROPIC_USAGE: Final[nemo_relay.JsonObject] = {
    "input_tokens": 100,
    "output_tokens": 20,
    "cache_read_input_tokens": 40,
    "cache_creation_input_tokens": 10,
    "output_tokens_details": {"thinking_tokens": 5},
}
_OPENAI_USAGE: Final[nemo_relay.JsonObject] = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
    "prompt_tokens_details": {"cached_tokens": 40},
    "completion_tokens_details": {"reasoning_tokens": 5},
}
_STREAM_USAGE: Final[nemo_relay.JsonObject] = {"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57}
_SSE: Final = {"content-type": "text/event-stream"}


class _Canned:
    """Serve HTTP responses in order, and keep the JSON body and headers of each request that the client sends."""

    def __init__(self, responses: Sequence[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[nemo_relay.JsonObject] = []
        self.headers: list[dict[str, str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body: nemo_relay.JsonObject = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        return self.responses.pop(0)

    def http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


def _text(text: str) -> nemo_relay.JsonObject:
    return {"type": "text", "text": text}


def _message(content: list[nemo_relay.JsonObject], stop_reason: str, *, message_id: str = "msg_1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": _CLAUDE,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": _ANTHROPIC_USAGE,
        },
    )


def _events(events: Sequence[tuple[str, nemo_relay.JsonObject]]) -> httpx.Response:
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx.Response(200, content=body.encode(), headers=_SSE)


def _message_stream(pieces: Sequence[str], *, message_id: str, usage: nemo_relay.JsonObject) -> httpx.Response:
    started: nemo_relay.JsonObject = {
        "type": "message",
        "id": message_id,
        "role": "assistant",
        "model": _CLAUDE,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 1},
    }
    events: list[tuple[str, nemo_relay.JsonObject]] = [
        ("message_start", {"type": "message_start", "message": started}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": _text("")}),
    ]
    events.extend(
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
        )
        for piece in pieces
    )
    events.extend(
        [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": usage,
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    return _events(events)


def _anthropic_client(canned: _Canned, *, max_retries: int = 2) -> AsyncAnthropic:
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key="test-key", http_client=canned.http_client(), max_retries=max_retries)


def _anthropic_pricing() -> PricingConfig:
    rates = TokenPricingRates(
        input_per_million=3.0, output_per_million=15.0, cache_read_per_million=0.30, cache_write_per_million=3.75
    )
    entry = ModelPricing(
        provider="anthropic",
        model_id=_CLAUDE,
        pricing_as_of="2026-01-01",
        pricing_source="test",
        rates=rates,
        prompt_cache=PromptCachePricing(read_accounting="separate"),
    )
    return PricingConfig(sources=[InlineSource(PricingCatalog(entries=[entry]))])


def _completion(text: str, *, completion_id: str = "chatcmpl-1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": completion_id,
            "object": "chat.completion",
            "created": 1,
            "model": _GPT,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": _OPENAI_USAGE,
        },
    )


def _completion_stream(pieces: Sequence[str], *, completion_id: str, usage: nemo_relay.JsonObject) -> httpx.Response:
    def chunk(choices: list[nemo_relay.Json], **fields: nemo_relay.Json) -> nemo_relay.JsonObject:
        body: nemo_relay.JsonObject = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": _GPT,
            "choices": choices,
        }
        body.update(fields)
        return body

    chunks = [
        chunk([{"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": None}])
        for piece in pieces
    ]
    chunks.extend([chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}]), chunk([], usage=usage)])
    body = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers=_SSE)


def _responses_reply(text: str, *, response_id: str = "resp_1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": response_id,
            "object": "response",
            "created_at": 1,
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        },
    )


def _openai_client(canned: _Canned) -> AsyncOpenAI:
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key="test-key", http_client=canned.http_client())


def _openai_pricing(model_id: str) -> PricingConfig:
    rates = TokenPricingRates(input_per_million=2.5, output_per_million=10.0, cache_read_per_million=1.25)
    entry = ModelPricing(
        provider="openai", model_id=model_id, pricing_as_of="2026-01-01", pricing_source="test", rates=rates
    )
    return PricingConfig(sources=[InlineSource(PricingCatalog(entries=[entry]))])


def _gemini(response: types.GenerateContentResponse, monkeypatch: pytest.MonkeyPatch, *, model: str) -> Gemini:
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

    client = genai.Client(api_key="test-key")
    monkeypatch.setattr(client.aio, "_models", CannedModels(response))
    return Gemini(model=model, client=client)


def _without_cost(usage: nemo_relay.JsonObject) -> nemo_relay.JsonObject:
    return {key: value for key, value in usage.items() if key != "cost"}


async def test_anthropic_model_is_read_as_anthropic_messages(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.anthropic_llm import AnthropicLlm

    from nemo_relay import plugin

    canned = _Canned(
        [
            _message([_text("Checking."), _WEATHER_CALL], "tool_use"),
            _message([_text("Sunny in Paris.")], "end_turn", message_id="msg_2"),
        ]
    )
    model = AnthropicLlm(model=_CLAUDE, client=_anthropic_client(canned))
    agent = LlmAgent(name="weather_agent", model=model, instruction="Be brief.", tools=[get_weather])
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(_anthropic_pricing())])):
        adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Sunny in Paris."
    assert isinstance(model, AnthropicLlm)
    assert type(model).__name__ == "AnthropicLlm"
    assert canned.requests[0]["tool_choice"] == {"type": "auto"}
    first, second = starts(subscribed_events, "llm")
    assert (first.name, second.name) == ("anthropic", "anthropic")
    assert metadata_of(first)["adk_model_type"] == "AnthropicLlm"
    first_end, second_end = ends(subscribed_events, "llm")
    response = response_of(first_end)
    assert (response.id, response_of(second_end).id) == ("msg_1", "msg_2")
    assert response.finish_reason == "tool_use"
    assert response.api_specific == {"api": "anthropic_messages"}
    assert response.extra == {"usage_metadata": {"thoughts_token_count": 5}}
    usage = present(response.usage)
    assert _without_cost(usage) == {
        "prompt_tokens": 100,
        "uncached_input_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 40,
        "cache_write_tokens": 10,
    }
    cost = as_object(usage["cost"])
    assert cost["input"] == pytest.approx(100 * 3.0 / 1e6)
    assert cost["output"] == pytest.approx(20 * 15.0 / 1e6)
    assert cost["cache_read"] == pytest.approx(40 * 0.30 / 1e6)
    assert cost["cache_write"] == pytest.approx(10 * 3.75 / 1e6)
    assert [start.name for start in starts(subscribed_events, "tool")] == ["get_weather"]
    assert_balanced(subscribed_events)


async def test_claude_on_vertex_is_routed_as_vertex_ai(subscribed_events: list[nemo_relay.Event]) -> None:
    from anthropic import AsyncAnthropicVertex
    from google.adk.agents import LlmAgent
    from google.adk.models.anthropic_llm import Claude

    canned = _Canned([_message([_text("Sunny.")], "end_turn")])
    client = AsyncAnthropicVertex(project_id="p", region="r", access_token="t", http_client=canned.http_client())
    model = Claude(model="claude-sonnet-4-5@20250929", client=client)

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "vertex_ai"
    assert metadata_of(start)["adk_model_type"] == "Claude"
    (end,) = ends(subscribed_events, "llm")
    assert response_of(end).api_specific == {"api": "anthropic_messages"}
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(
    ("choice", "sent"),
    [
        pytest.param("required", {"type": "any"}, id="required"),
        pytest.param("none", {"type": "none"}, id="none"),
        pytest.param(
            {
                "provider": "google_adk",
                "kind": "function_calling_config",
                "value": {"mode": "ANY", "allowed_function_names": ["get_weather"]},
            },
            {"type": "tool", "name": "get_weather"},
            id="one_allowed_function",
        ),
    ],
)
async def test_tool_choice_edits_reach_anthropic(
    choice: str | nemo_relay.JsonObject, sent: nemo_relay.JsonObject, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.anthropic_llm import AnthropicLlm

    canned = _Canned([_message([_text("Sunny.")], "end_turn")])
    model = AnthropicLlm(model=_CLAUDE, client=_anthropic_client(canned))

    def choose(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        edited = present(annotated)
        edited.tool_choice = choice
        return nemo_relay.LLMRequestInterceptOutcome(request, edited)

    with llm_request_intercept(choose):
        adk_events = await run_recorded(
            LlmAgent(name="weather_agent", model=model, instruction="Be brief.", tools=[get_weather])
        )

    assert final_text(adk_events) == "Sunny."
    assert canned.requests[0]["tool_choice"] == sent
    assert [tool["name"] for tool in as_list(canned.requests[0]["tools"])] == ["get_weather"]
    assert_balanced(subscribed_events)


async def test_tool_choice_edits_anthropic_cannot_carry_fail_the_call(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.anthropic_llm import AnthropicLlm

    canned = _Canned([_message([_text("Sunny.")], "end_turn")])
    model = AnthropicLlm(model=_CLAUDE, client=_anthropic_client(canned))

    def choose(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        edited = present(annotated)
        edited.tool_choice = {
            "provider": "google_adk",
            "kind": "function_calling_config",
            "value": {"mode": "ANY", "allowed_function_names": ["get_forecast"]},
        }
        return nemo_relay.LLMRequestInterceptOutcome(request, edited)

    with llm_request_intercept(choose), pytest.raises(ValueError, match="get_forecast"):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief.", tools=[get_weather]))

    assert canned.requests == []
    assert status_of(ends(subscribed_events, "llm")[0]) == "ERROR"
    assert_balanced(subscribed_events)


async def test_anthropic_stream_records_the_message_id(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode
    from google.adk.models.anthropic_llm import AnthropicLlm

    canned = _Canned([_message_stream(["Sunny ", "streamed."], message_id="msg_s", usage={"output_tokens": 7})])
    model = AnthropicLlm(model=_CLAUDE, client=_anthropic_client(canned))

    adk_events = await run_recorded(
        LlmAgent(name="stream_agent", model=model, instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    assert partial_texts(adk_events) == ["Sunny ", "streamed."]
    assert final_text(adk_events) == "Sunny streamed."
    (end,) = ends(subscribed_events, "llm")
    response = response_of(end)
    assert response.id == "msg_s"
    assert response.usage == {
        "prompt_tokens": 50,
        "uncached_input_tokens": 50,
        "completion_tokens": 7,
        "total_tokens": 57,
    }
    assert response.api_specific == {"api": "anthropic_messages"}
    assert len(marks(subscribed_events, "llm.chunk")) == 3
    assert_balanced(subscribed_events)


async def test_openai_chat_model_is_read_as_openai_chat(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.integrations.openai import OpenAILlm

    from nemo_relay import plugin

    canned = _Canned([_completion("Sunny in Paris.")])
    model = OpenAILlm(model="gpt-4o", client=_openai_client(canned))
    agent = LlmAgent(name="weather_agent", model=model, instruction="Be brief.", tools=[get_weather])
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(_openai_pricing(_GPT))])):
        adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Sunny in Paris."
    assert canned.requests[0]["tool_choice"] == "auto"
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "openai"
    assert metadata_of(start)["adk_model_type"] == "OpenAILlm"
    (end,) = ends(subscribed_events, "llm")
    response = response_of(end)
    assert response.id == "chatcmpl-1"
    assert response.model == _GPT
    assert response.api_specific == {"api": "openai_chat"}
    assert response.extra == {"usage_metadata": {"thoughts_token_count": 5}}
    usage = present(response.usage)
    assert _without_cost(usage) == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 40,
        "uncached_input_tokens": 60,
    }
    cost = as_object(usage["cost"])
    assert cost["input"] == pytest.approx(60 * 2.5 / 1e6)
    assert cost["output"] == pytest.approx(20 * 10.0 / 1e6)
    assert cost["cache_read"] == pytest.approx(40 * 1.25 / 1e6)
    assert_balanced(subscribed_events)


async def test_openai_chat_stream_records_the_completion_id(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode
    from google.adk.integrations.openai import OpenAILlm

    canned = _Canned([_completion_stream(["Sunny ", "streamed."], completion_id="chatcmpl-s", usage=_STREAM_USAGE)])
    model = OpenAILlm(model="gpt-4o", client=_openai_client(canned))

    adk_events = await run_recorded(
        LlmAgent(name="stream_agent", model=model, instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    assert partial_texts(adk_events) == ["Sunny ", "streamed."]
    assert final_text(adk_events) == "Sunny streamed."
    (end,) = ends(subscribed_events, "llm")
    response = response_of(end)
    assert response.id == "chatcmpl-s"
    assert response.usage == {"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57}
    assert response.api_specific == {"api": "openai_chat"}
    assert_balanced(subscribed_events)


async def test_openai_responses_model_is_read_as_openai_responses(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.integrations.openai import OpenAIResponsesLlm

    canned = _Canned([_responses_reply("Sunny in Paris.")])
    model = OpenAIResponsesLlm(model="gpt-5", client=_openai_client(canned))

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny in Paris."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "openai"
    assert metadata_of(start)["adk_model_type"] == "OpenAIResponsesLlm"
    (end,) = ends(subscribed_events, "llm")
    response = response_of(end)
    assert response.id == "resp_1"
    assert response.api_specific == {
        "api": "openai_responses",
        "input_tokens_details": {"cached_tokens": 40},
        "output_tokens_details": {"reasoning_tokens": 5},
    }
    assert response.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 40,
        "uncached_input_tokens": 60,
    }
    assert "usage_metadata" not in (response.extra or {})
    assert_balanced(subscribed_events)


def _future_llm(module: str) -> BaseLlm:
    from google.adk.models import BaseLlm

    class FutureLlm(BaseLlm):
        __module__ = module
        __qualname__ = "FutureLlm"
        model: str = "future"

        @override
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield model_answer("From the future.")

    return FutureLlm()


@pytest.mark.parametrize(
    "module",
    [
        "google.adk.integrations.openai._future_llm",
        "google.adk.models.anthropic_llm",
        "google.adk.models._future_llm",
        "google.adk.integrations.future",
    ],
    ids=["openai_integration", "anthropic_module", "models_package", "integrations_package"],
)
async def test_a_class_from_an_adk_provider_package_relay_does_not_model_fails_the_call(
    module: str, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent

    with pytest.raises(TypeError, match=re.escape(f"{module}.FutureLlm")):
        await run_recorded(LlmAgent(name="weather_agent", model=_future_llm(module), instruction="Be brief."))

    assert starts(subscribed_events, "llm") == []
    assert_balanced(subscribed_events)


async def test_an_adk_class_outside_its_provider_packages_is_named_after_its_class(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    model = _future_llm("google.adk.evaluation.simulation._future_llm")

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "From the future."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "FutureLlm"
    assert_balanced(subscribed_events)


async def test_fallback_model_records_each_model_it_tries(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models import BaseLlm, FallbackModel, LLMRegistry
    from google.adk.models import registry as model_registry
    from google.adk.models.anthropic_llm import AnthropicLlm

    class Backup(BaseLlm):
        model: str = "fallback-backup"

        @classmethod
        @override
        def supported_models(cls) -> list[str]:
            return [r"fallback-backup"]

        @override
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield model_answer("From the backup.")

    monkeypatch.setattr(model_registry, "_llm_registry_dict", dict(model_registry._llm_registry_dict))
    request.addfinalizer(LLMRegistry.resolve.cache_clear)
    LLMRegistry.register(Backup)
    refused = _Canned(
        [httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "later"}})]
    )
    primary = AnthropicLlm(model=_CLAUDE, client=_anthropic_client(refused, max_retries=0))
    fallback = FallbackModel(models=[primary, "fallback-backup"])

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=fallback, instruction="Be brief."))

    assert final_text(adk_events) == "From the backup."
    assert type(fallback) is FallbackModel
    assert isinstance(fallback.models[1], Backup)
    first, second = starts(subscribed_events, "llm")
    assert (first.name, second.name) == ("anthropic", "Backup")
    assert (metadata_of(first)["adk_model_type"], metadata_of(second)["adk_model_type"]) == ("AnthropicLlm", "Backup")
    first_end, second_end = ends(subscribed_events, "llm")
    assert (status_of(first_end), status_of(second_end)) == ("ERROR", "OK")
    assert_balanced(subscribed_events)


async def test_gemini_model_records_the_response_id(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.genai import types

    response = types.GenerateContentResponse(
        response_id="resp-g",
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="Sunny.")]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
    )

    adk_events = await run_recorded(
        LlmAgent(
            name="weather_agent",
            model=_gemini(response, monkeypatch, model="gemini-2.5-flash"),
            instruction="Be brief.",
        )
    )

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "gemini"
    (end,) = ends(subscribed_events, "llm")
    assert response_of(end).id == "resp-g"
    assert_balanced(subscribed_events)


def _anthropic_model(canned: _Canned) -> BaseLlm:
    from google.adk.models.anthropic_llm import AnthropicLlm

    return AnthropicLlm(model=_CLAUDE, client=_anthropic_client(canned))


def _foundry_model(canned: _Canned) -> BaseLlm:
    from anthropic import AsyncAnthropicFoundry
    from google.adk.models.anthropic_llm import AnthropicLlm

    client = AsyncAnthropicFoundry(resource="relay", api_key="test-key", http_client=canned.http_client())
    return AnthropicLlm(model=_CLAUDE, client=client)


def _openai_chat_model(canned: _Canned) -> BaseLlm:
    from google.adk.integrations.openai import OpenAILlm

    return OpenAILlm(model="gpt-4o", client=_openai_client(canned))


def _azure_chat_model(canned: _Canned) -> BaseLlm:
    from google.adk.integrations.openai import OpenAILlm
    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(
        azure_endpoint="https://relay.openai.azure.com",
        api_version="2024-10-21",
        api_key="test-key",
        http_client=canned.http_client(),
    )
    return OpenAILlm(model="gpt-4o", client=client)


def _openai_responses_model(canned: _Canned) -> BaseLlm:
    from google.adk.integrations.openai import OpenAIResponsesLlm

    return OpenAIResponsesLlm(model="gpt-5", client=_openai_client(canned))


def _azure_responses_model(canned: _Canned) -> BaseLlm:
    from google.adk.integrations.openai import AzureOpenAIResponsesLlm

    return AzureOpenAIResponsesLlm(model="gpt-5", client=_openai_client(canned))


def _apigee_chat_model(canned: _Canned) -> BaseLlm:
    from google.adk.models.apigee_llm import ApigeeLlm

    model = ApigeeLlm(model="apigee/openai/gpt-4o", proxy_url="https://proxy.example")
    canned_client = functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(canned.handle))
    with mock.patch.object(httpx, "AsyncClient", canned_client):
        _ = model._completions_http_client._client
    return model


def _claude_reply() -> httpx.Response:
    return _message([_text("Sunny.")], "end_turn")


def _chat_reply() -> httpx.Response:
    return _completion("Sunny.")


def _responses_text_reply() -> httpx.Response:
    return _responses_reply("Sunny.")


@pytest.mark.parametrize(
    ("build", "reply", "provider", "api"),
    [
        pytest.param(_anthropic_model, _claude_reply, "anthropic", "anthropic_messages", id="anthropic"),
        pytest.param(_openai_chat_model, _chat_reply, "openai", "openai_chat", id="openai_chat"),
        pytest.param(
            _openai_responses_model,
            _responses_text_reply,
            "openai",
            "openai_responses",
            id="responses",
        ),
        pytest.param(_apigee_chat_model, _chat_reply, "openai", "openai_chat", id="apigee_chat"),
    ],
)
async def test_relay_request_headers_reach_models_whose_client_takes_them(
    build: Callable[[_Canned], BaseLlm],
    reply: Callable[[], httpx.Response],
    provider: str,
    api: str,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    seen: list[nemo_relay.JsonObject] = []

    def stamp(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        seen.append(dict(request.headers))
        headers = {**request.headers, "x-relay": "1"}
        return nemo_relay.LLMRequestInterceptOutcome(nemo_relay.LLMRequest(headers, request.content), annotated)

    canned = _Canned([reply()])
    with llm_request_intercept(stamp):
        adk_events = await run_recorded(LlmAgent(name="weather_agent", model=build(canned), instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    assert seen == [{}]
    assert canned.headers[0]["x-relay"] == "1"
    (start,) = starts(subscribed_events, "llm")
    assert start.name == provider
    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).api_specific)["api"] == api
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(
    ("build", "reply", "provider"),
    [
        pytest.param(_foundry_model, _claude_reply, "azure_ai", id="anthropic_on_foundry"),
        pytest.param(_azure_chat_model, _chat_reply, "azure", id="openai_on_azure"),
        pytest.param(_azure_responses_model, _responses_text_reply, "azure", id="azure_responses"),
    ],
)
async def test_the_llm_scope_is_named_after_the_provider_the_client_reaches(
    build: Callable[[_Canned], BaseLlm],
    reply: Callable[[], httpx.Response],
    provider: str,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    adk_events = await run_recorded(
        LlmAgent(name="weather_agent", model=build(_Canned([reply()])), instruction="Be brief.")
    )

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == provider
    assert_balanced(subscribed_events)


async def test_a_tuned_gemini_model_is_named_after_the_gemini_api(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.genai import types

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="Sunny.")]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
    )
    model = _gemini(response, monkeypatch, model="tunedModels/weather-123")

    await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    (start,) = starts(subscribed_events, "llm")
    assert start.name == "gemini"
    assert present(start.category_profile)["model_name"] == "tunedModels/weather-123"


def _usd(total: float) -> nemo_relay.JsonObject:
    return {"currency": "USD", "source": "provider_reported", "total": total}


def _claude_reply_reporting_cost() -> httpx.Response:
    message: nemo_relay.JsonObject = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": _CLAUDE,
        "content": [_text("Sunny.")],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {**_ANTHROPIC_USAGE, "cost_usd": 0.0021},
    }
    return httpx.Response(200, json=message)


def _chat_reply_reporting_cost() -> httpx.Response:
    completion: nemo_relay.JsonObject = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": _GPT,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Sunny."}}],
        "usage": {**_OPENAI_USAGE, "cost": 0.0042},
    }
    return httpx.Response(200, json=completion)


def _responses_reply_reporting_cost() -> httpx.Response:
    reply = _responses_reply("Sunny.")
    body = as_object(reply.json())
    body["usage"] = {**as_object(body["usage"]), "cost": {"total": 0.5, "currency": "EUR"}}
    return httpx.Response(200, json=body)


@pytest.mark.parametrize(
    ("build", "reply", "cost"),
    [
        pytest.param(_anthropic_model, _claude_reply_reporting_cost, _usd(0.0021), id="anthropic_cost_usd"),
        pytest.param(_openai_chat_model, _chat_reply_reporting_cost, _usd(0.0042), id="openai_chat_cost"),
        pytest.param(
            _openai_responses_model,
            _responses_reply_reporting_cost,
            {"currency": "EUR", "source": "provider_reported", "total": 0.5},
            id="responses_structured_cost",
        ),
    ],
)
async def test_a_cost_the_provider_reports_reaches_the_annotation(
    build: Callable[[_Canned], BaseLlm],
    reply: Callable[[], httpx.Response],
    cost: nemo_relay.JsonObject,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    adk_events = await run_recorded(
        LlmAgent(name="weather_agent", model=build(_Canned([reply()])), instruction="Be brief.")
    )

    assert final_text(adk_events) == "Sunny."
    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).usage)["cost"] == cost
    assert_balanced(subscribed_events)


async def test_a_cost_the_provider_reports_is_kept_over_a_pricing_estimate(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    from nemo_relay import plugin

    model = _openai_chat_model(_Canned([_chat_reply_reporting_cost()]))
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(_openai_pricing(_GPT))])):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).usage)["cost"] == _usd(0.0042)


async def test_a_cost_reported_at_the_end_of_a_chat_completions_stream_reaches_the_annotation(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode

    chat = _Canned([_completion_stream(["Sunny."], completion_id="c", usage={**_STREAM_USAGE, "cost": 0.0042})])

    await run_recorded(
        LlmAgent(name="stream_agent", model=_openai_chat_model(chat), instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).usage)["cost"] == _usd(0.0042)


async def test_a_cost_reported_at_the_end_of_an_anthropic_stream_reaches_the_annotation(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode

    claude = _Canned([_message_stream(["Sunny."], message_id="m", usage={"output_tokens": 7, "cost_usd": 0.0021})])

    await run_recorded(
        LlmAgent(name="stream_agent", model=_anthropic_model(claude), instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).usage)["cost"] == _usd(0.0021)
    assert data_of(end)["provider_reported"] == {
        "id": "m",
        "usage": {"input_tokens": 50, "output_tokens": 7, "cost_usd": 0.0021},
    }


def _responses_stream(pieces: Sequence[str], *, finish: str, usage: nemo_relay.JsonObject) -> httpx.Response:
    response: nemo_relay.JsonObject = {
        "id": "resp_s",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5",
        "status": "in_progress",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    events: list[tuple[str, nemo_relay.JsonObject]] = [
        ("response.created", {"type": "response.created", "sequence_number": 0, "response": response})
    ]
    events.extend(
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": number,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": piece,
                "logprobs": [],
            },
        )
        for number, piece in enumerate(pieces, start=1)
    )
    message: nemo_relay.JsonObject = {
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "".join(pieces), "annotations": []}],
    }
    done: nemo_relay.JsonObject = {
        **response,
        "status": finish.removeprefix("response."),
        "output": [message],
        "usage": usage,
    }
    events.append((finish, {"type": finish, "sequence_number": len(events), "response": done}))
    return _events(events)


@pytest.mark.parametrize("finish", ["response.completed", "response.incomplete"])
async def test_a_cost_a_finished_responses_stream_reports_reaches_the_annotation(
    finish: str, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode

    usage: nemo_relay.JsonObject = {
        "input_tokens": 50,
        "output_tokens": 7,
        "total_tokens": 57,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
        "cost_usd": 0.0042,
    }
    model = _openai_responses_model(_Canned([_responses_stream(["Sunny ", "streamed."], finish=finish, usage=usage)]))

    adk_events = await run_recorded(
        LlmAgent(name="stream_agent", model=model, instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    assert final_text(adk_events) == "Sunny streamed."
    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).usage)["cost"] == _usd(0.0042)
    assert_balanced(subscribed_events)


async def test_a_usage_the_native_codec_cannot_read_leaves_the_call_unannotated(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    completion: nemo_relay.JsonObject = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": _GPT,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Sunny."}}],
        "usage": {**_OPENAI_USAGE, "cost": "0.0042"},
    }
    model = _openai_chat_model(_Canned([httpx.Response(200, json=completion)]))

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (end,) = ends(subscribed_events, "llm")
    assert status_of(end) == "OK"
    assert end.annotated_response is None
    assert_balanced(subscribed_events)


async def test_apigee_chat_records_the_completion_id_and_cost(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent

    model = _apigee_chat_model(_Canned([_chat_reply_reporting_cost()]))

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (end,) = ends(subscribed_events, "llm")
    response = response_of(end)
    assert response.id == "chatcmpl-1"
    assert response.model == _GPT
    assert present(response.usage)["cost"] == _usd(0.0042)
    assert_balanced(subscribed_events)


async def test_apigee_chat_stream_records_the_completion_id_and_cost(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode

    usage: nemo_relay.JsonObject = {**_STREAM_USAGE, "cost": 0.0042}
    canned = _Canned([_completion_stream(["Sunny ", "streamed."], completion_id="chatcmpl-a", usage=usage)])
    model = _apigee_chat_model(canned)

    adk_events = await run_recorded(
        LlmAgent(name="stream_agent", model=model, instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    assert partial_texts(adk_events) == ["Sunny ", "streamed."]
    assert final_text(adk_events) == "Sunny streamed."
    (end,) = ends(subscribed_events, "llm")
    response = response_of(end)
    assert response.id == "chatcmpl-a"
    assert response.usage == {"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57, "cost": _usd(0.0042)}
    assert_balanced(subscribed_events)
