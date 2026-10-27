# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test ADK's provider model classes with the Google ADK NeMo Relay integration."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final
from unittest import mock

import httpx
import pytest
from _helpers import (
    HANG_TIMEOUT,
    LlmNext,
    as_instance,
    as_list,
    as_object,
    assert_balanced,
    data_of,
    ends,
    final_text,
    get_weather,
    llm_execution_intercept,
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
    text_response,
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
    from collections.abc import AsyncIterator, Callable, Sequence

    from anthropic import AsyncAnthropic
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models import BaseLlm
    from google.adk.models.google_llm import Gemini
    from google.adk.models.lite_llm import LiteLLMClient
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from litellm import CustomStreamWrapper
    from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.integrations.custom_prompt_management import CustomPromptManagement
    from litellm.integrations.prompt_management_base import PromptManagementClient
    from litellm.types.prompts.init_prompts import PromptSpec
    from litellm.types.router import GenericLiteLLMParams
    from litellm.types.utils import ModelResponse, StandardCallbackDynamicParams
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
_LITE_LLM_USAGE: Final[nemo_relay.JsonObject] = {**_OPENAI_USAGE, "cache_creation_input_tokens": 10}
_LITELLM_IMPORT_PROBE: Final = """
import asyncio
import sys

from google.adk.models.lite_llm import LiteLlm

from nemo_relay.integrations.google_adk._backends import backend_of

assert "litellm" not in sys.modules
model = LiteLlm(model="openai/gpt-4o")
asyncio.run(backend_of(model, model.model))
import litellm

print(litellm.litellm_mode)
"""
_ABSENT_SDK_PROBE: Final = """
import asyncio
import sys
from collections.abc import AsyncGenerator

absent, module = sys.argv[1:]
sys.modules[absent] = None

from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from nemo_relay.integrations.google_adk._backends import backend_of


class FutureLlm(BaseLlm):
    __module__ = module
    __qualname__ = "FutureLlm"
    model: str = "future"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse()


model = FutureLlm()
try:
    asyncio.run(backend_of(model, model.model))
except TypeError as error:
    print(error)
"""
_SSE: Final = {"content-type": "text/event-stream"}
_ADK_USAGE_DUMP_WARNING: Final[
    Annotated[str, "Filter for the Pydantic warning ADK causes when it dumps LiteLLM usage with a cost breakdown."]
] = "ignore:Pydantic serializer warnings:UserWarning"


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


class _StreamBody(httpx.AsyncByteStream):
    """Stream body that sends `chunks` as server-sent events, then raises `error` if set.

    Attributes:
        closed: True once the body closed.
    """

    def __init__(self, chunks: Sequence[nemo_relay.JsonObject], *, error: Exception | None) -> None:
        self.chunks = chunks
        self.error = error
        self.closed = False

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        if self.error is not None:
            raise self.error

    @override
    async def aclose(self) -> None:
        self.closed = True


class _HeldCloseBody(_StreamBody):
    """Stream body whose close waits for `release`.

    Attributes:
        closing: Set when close starts.
        release: Set by the test to let close finish.
        closed_event: Set when close finished.
    """

    def __init__(self, chunks: Sequence[nemo_relay.JsonObject]) -> None:
        super().__init__(chunks, error=None)
        self.closing = asyncio.Event()
        self.release = asyncio.Event()
        self.closed_event = asyncio.Event()

    @override
    async def aclose(self) -> None:
        self.closing.set()
        await self.release.wait()
        await super().aclose()
        self.closed_event.set()


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


def _chunk(choices: list[nemo_relay.Json], *, completion_id: str, **fields: nemo_relay.Json) -> nemo_relay.JsonObject:
    chunk: nemo_relay.JsonObject = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _GPT,
        "choices": choices,
    }
    chunk.update(fields)
    return chunk


def _content_chunk(text: str, *, completion_id: str) -> nemo_relay.JsonObject:
    return _chunk(
        [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        completion_id=completion_id,
    )


def _stop_chunk(*, completion_id: str, **fields: nemo_relay.Json) -> nemo_relay.JsonObject:
    return _chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}], completion_id=completion_id, **fields)


def _completion_stream(pieces: Sequence[str], *, completion_id: str, usage: nemo_relay.JsonObject) -> httpx.Response:
    chunks = [
        *(_content_chunk(piece, completion_id=completion_id) for piece in pieces),
        _stop_chunk(completion_id=completion_id),
        _chunk([], completion_id=completion_id, usage=usage),
    ]
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


def _import_litellm() -> None:
    from google.adk.models.lite_llm import _ensure_litellm_imported

    _ensure_litellm_imported()


def _lite_llm_reply(text: str, *, response_id: str, model: str, usage: nemo_relay.JsonObject) -> ModelResponse:
    _import_litellm()
    from litellm.types.utils import ModelResponse

    return ModelResponse(
        id=response_id,
        model=model,
        object="chat.completion",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        usage=usage,
    )


def _lite_llm_tool_call(*, response_id: str) -> ModelResponse:
    _import_litellm()
    from litellm.types.utils import ModelResponse

    call: nemo_relay.JsonObject = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"location": "Paris"})},
    }
    message: nemo_relay.JsonObject = {"role": "assistant", "content": None, "tool_calls": [call]}
    return ModelResponse(
        id=response_id,
        model=_GPT,
        object="chat.completion",
        choices=[{"index": 0, "finish_reason": "tool_calls", "message": message}],
        usage=_OPENAI_USAGE,
    )


def _answering_client(replies: Sequence[ModelResponse], *, calls: list[dict[str, object]]) -> LiteLLMClient:
    from google.adk.models.lite_llm import LiteLLMClient

    pending = list(replies)

    class Answering(LiteLLMClient):
        @override
        async def acompletion(
            self, model: object, messages: object, tools: object, **kwargs: object
        ) -> ModelResponse | CustomStreamWrapper:
            calls.append({"model": model, "tools": tools, **kwargs})
            return pending.pop(0)

    return Answering()


def get_time(location: str) -> dict[str, str]:
    """Get the time for a location."""
    return {"location": location, "time": "noon"}


def _probe_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key != "LITELLM_MODE"}
    environment["PYTHONPATH"] = str(Path(nemo_relay.__file__).resolve().parents[1])
    return environment


def _gemini(
    response: types.GenerateContentResponse, monkeypatch: pytest.MonkeyPatch, *, model: str
) -> tuple[Gemini, list[str]]:
    """Return Gemini model with a canned client, plus the model name each call sends."""
    from google import genai
    from google.adk.models.google_llm import Gemini
    from google.genai.models import AsyncModels

    sent: list[str] = []

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
            sent.append(model)
            return self.canned

    client = genai.Client(api_key="test-key")
    monkeypatch.setattr(client.aio, "_models", CannedModels(response))
    return Gemini(model=model, client=client), sent


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
        "google.adk.models.lite_llm",
        "google.adk.models._future_llm",
        "google.adk.integrations.future",
    ],
    ids=["openai_integration", "anthropic_module", "lite_llm_module", "models_package", "integrations_package"],
)
async def test_a_class_from_an_adk_provider_package_relay_does_not_model_fails_the_call(
    module: str, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent

    with pytest.raises(TypeError, match=re.escape(f"{module}.FutureLlm")):
        await run_recorded(LlmAgent(name="weather_agent", model=_future_llm(module), instruction="Be brief."))

    assert starts(subscribed_events, "llm") == []
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(
    ("module", "absent"),
    [
        pytest.param("google.adk.integrations.openai._future_llm", "openai", id="openai_integration"),
        pytest.param("google.adk.models._future_llm", "anthropic", id="anthropic"),
        pytest.param("google.adk.models._future_llm", "litellm", id="litellm"),
    ],
)
def test_a_class_relay_does_not_model_fails_without_its_provider_sdk(module: str, absent: str) -> None:
    probe = subprocess.run(
        [sys.executable, "-c", _ABSENT_SDK_PROBE, absent, module],
        env=_probe_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.rstrip().endswith(f"{module}.FutureLlm")


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


async def test_lite_llm_model_is_read_as_openai_chat(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_CLAUDE, usage=_LITE_LLM_USAGE)
    model = LiteLlm(model=f"anthropic/{_CLAUDE}", mock_response=reply)

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "anthropic"
    assert metadata_of(start)["adk_model_type"] == "LiteLlm"
    (end,) = ends(subscribed_events, "llm")
    response = response_of(end)
    assert response.id == "chatcmpl-lite"
    assert response.model == _CLAUDE
    assert response.api_specific == {"api": "openai_chat"}
    assert response.extra == {"partial": False, "usage_metadata": {"thoughts_token_count": 5}}
    assert response.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 40,
        "cache_write_tokens": 10,
        "uncached_input_tokens": 50,
    }
    assert_balanced(subscribed_events)


async def test_a_lite_llm_anthropic_route_is_priced_like_the_anthropic_model(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    from nemo_relay import plugin

    _import_litellm()
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    client = AsyncHTTPHandler(transport=httpx.MockTransport(_Canned([_message([_text("Sunny.")], "end_turn")]).handle))
    model = LiteLlm(model=f"anthropic/{_CLAUDE}", client=client, api_key="test-key")
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(_anthropic_pricing())])):
        adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (end,) = ends(subscribed_events, "llm")
    recorded = present(response_of(end).usage)
    assert _without_cost(recorded) == {
        "prompt_tokens": 150,
        "uncached_input_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 170,
        "cache_read_tokens": 40,
        "cache_write_tokens": 10,
    }
    cost = as_object(recorded["cost"])
    assert cost["input"] == pytest.approx(100 * 3.0 / 1e6)
    assert cost["output"] == pytest.approx(20 * 15.0 / 1e6)
    assert cost["cache_read"] == pytest.approx(40 * 0.30 / 1e6)
    assert cost["cache_write"] == pytest.approx(10 * 3.75 / 1e6)
    assert_balanced(subscribed_events)


async def test_a_lite_llm_cache_write_without_a_cache_read_is_billed_once(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    from nemo_relay import plugin

    usage: nemo_relay.JsonObject = {
        "prompt_tokens": 110,
        "completion_tokens": 20,
        "total_tokens": 130,
        "prompt_tokens_details": {"cache_write_tokens": 10},
    }
    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_CLAUDE, usage=usage)
    model = LiteLlm(model=f"anthropic/{_CLAUDE}", mock_response=reply)
    async with plugin.activate(plugin.PluginConfig(components=[ComponentSpec(_anthropic_pricing())])):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    (end,) = ends(subscribed_events, "llm")
    recorded = present(response_of(end).usage)
    assert _without_cost(recorded) == {
        "prompt_tokens": 110,
        "uncached_input_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 130,
        "cache_write_tokens": 10,
    }
    cost = as_object(recorded["cost"])
    assert cost["input"] == pytest.approx(100 * 3.0 / 1e6)
    assert cost["cache_write"] == pytest.approx(10 * 3.75 / 1e6)
    assert_balanced(subscribed_events)


def _editing_response(edit: Callable[[nemo_relay.JsonObject], None]) -> nemo_relay.LlmExecutionIntercept:
    async def apply(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmNext
    ) -> nemo_relay.Json:
        response = as_object(await next_call(request))
        edit(response)
        return response

    return apply


async def _run_edited_lite_llm_call(edit: Callable[[nemo_relay.JsonObject], None]) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_CLAUDE, usage=_LITE_LLM_USAGE)
    model = LiteLlm(model=f"anthropic/{_CLAUDE}", mock_response=reply)
    with llm_execution_intercept(_editing_response(edit)):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))


async def test_an_intercept_cache_edit_reaches_the_lite_llm_annotation(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    await _run_edited_lite_llm_call(
        lambda response: as_object(response["usage_metadata"]).update(cached_content_token_count=0)
    )

    (end,) = ends(subscribed_events, "llm")
    assert response_of(end).usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 0,
        "cache_write_tokens": 10,
        "uncached_input_tokens": 90,
    }
    assert_balanced(subscribed_events)


async def test_an_intercept_that_removes_the_lite_llm_cache_write_removes_it_from_the_annotation(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    await _run_edited_lite_llm_call(
        lambda response: as_object(response["usage_metadata"]).pop("cache_creation_input_tokens")
    )

    (end,) = ends(subscribed_events, "llm")
    assert response_of(end).usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 40,
        "uncached_input_tokens": 60,
    }
    assert_balanced(subscribed_events)


async def test_a_recorded_lite_llm_cache_write_that_is_not_an_integer_drops_the_annotation(
    subscribed_events: list[nemo_relay.Event], capfd: pytest.CaptureFixture[str]
) -> None:
    def corrupt(response: nemo_relay.JsonObject) -> None:
        usage_metadata = as_object(response["usage_metadata"])
        del usage_metadata["cache_creation_input_tokens"]
        usage = as_object(as_object(response["provider_reported"])["usage"])
        del usage["cache_creation_input_tokens"]
        as_object(usage["prompt_tokens_details"])["cache_write_tokens"] = "ten"

    await _run_edited_lite_llm_call(corrupt)

    (end,) = ends(subscribed_events, "llm")
    assert end.annotated_response is None
    assert "LiteLLM usage prompt_tokens_details cache_write_tokens must be an integer" in capfd.readouterr().err
    assert_balanced(subscribed_events)


async def test_a_lite_llm_response_an_intercept_answers_keeps_its_cache_read(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    usage_metadata: nemo_relay.JsonObject = {
        "prompt_token_count": 100,
        "candidates_token_count": 20,
        "total_token_count": 120,
        "cached_content_token_count": 40,
    }
    answer = {**text_response("Sunny."), "usage_metadata": usage_metadata}
    model = LiteLlm(model=f"anthropic/{_CLAUDE}", mock_response="Unused.")
    with llm_execution_intercept(lambda name, request, context, next_call: answer):
        adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (end,) = ends(subscribed_events, "llm")
    assert response_of(end).usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 40,
        "uncached_input_tokens": 60,
    }
    assert_balanced(subscribed_events)


async def test_lite_llm_stream_records_the_response_id_and_usage(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode

    usage: nemo_relay.JsonObject = {**_STREAM_USAGE, "cost": 0.0042}
    canned = _Canned([_completion_stream(["Sunny ", "streamed."], completion_id="chatcmpl-lite-s", usage=usage)])

    adk_events = await run_recorded(
        LlmAgent(name="stream_agent", model=_lite_llm_model(canned), instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    assert partial_texts(adk_events) == ["Sunny ", "streamed."]
    assert final_text(adk_events) == "Sunny streamed."
    (end,) = ends(subscribed_events, "llm")
    response = response_of(end)
    assert response.id == "chatcmpl-lite-s"
    assert response.api_specific == {"api": "openai_chat"}
    assert response.usage == {
        "prompt_tokens": 50,
        "completion_tokens": 7,
        "total_tokens": 57,
        "cost": _usd(0.0042),
    }
    reported = as_object(data_of(end)["provider_reported"])
    assert reported["id"] == "chatcmpl-lite-s"
    assert as_object(reported["usage"]).items() >= usage.items()
    assert_balanced(subscribed_events)


async def test_a_lite_llm_stream_reports_no_cost_its_provider_does_not_report(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode

    canned = _Canned([_completion_stream(["Sunny."], completion_id="chatcmpl-lite-s", usage=_STREAM_USAGE)])

    await run_recorded(
        LlmAgent(name="stream_agent", model=_lite_llm_model(canned), instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    (end,) = ends(subscribed_events, "llm")
    assert "cost" not in present(response_of(end).usage)
    assert "cost" not in as_object(as_object(data_of(end)["provider_reported"])["usage"])
    assert_balanced(subscribed_events)


async def _run_lite_llm_stream(chunks: Sequence[nemo_relay.JsonObject]) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode

    canned = _Canned([httpx.Response(200, headers=_SSE, stream=_StreamBody(chunks, error=None))])
    await run_recorded(
        LlmAgent(name="stream_agent", model=_lite_llm_model(canned), instruction="Be brief."),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )


async def test_a_lite_llm_stream_keeps_a_cost_reported_before_its_last_usage_chunk(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    await _run_lite_llm_stream(
        [
            _content_chunk("Sunny.", completion_id="chatcmpl-lite-s"),
            _stop_chunk(completion_id="chatcmpl-lite-s", usage={**_STREAM_USAGE, "cost": 0.0042}),
            _chunk([], completion_id="chatcmpl-lite-s", usage=_STREAM_USAGE),
        ]
    )

    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).usage)["cost"] == _usd(0.0042)
    assert_balanced(subscribed_events)


@pytest.mark.filterwarnings(_ADK_USAGE_DUMP_WARNING)
async def test_a_lite_llm_stream_reports_the_total_of_a_cost_breakdown(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    usage: nemo_relay.JsonObject = {**_STREAM_USAGE, "cost": {"request_cost": 0.005, "total_cost": 0.0042}}

    await _run_lite_llm_stream(
        [
            _content_chunk("Sunny.", completion_id="chatcmpl-lite-s"),
            _chunk([], completion_id="chatcmpl-lite-s", usage=usage),
        ]
    )

    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).usage)["cost"] == _usd(0.0042)
    assert_balanced(subscribed_events)


async def test_a_lite_llm_stream_reports_the_cost_of_its_last_usage_chunk_with_a_cost(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    await _run_lite_llm_stream(
        [
            _content_chunk("Sunny.", completion_id="chatcmpl-lite-s"),
            _stop_chunk(completion_id="chatcmpl-lite-s", usage={**_STREAM_USAGE, "cost": 0.001}),
            _chunk([], completion_id="chatcmpl-lite-s", usage={**_STREAM_USAGE, "cost": 0.0042}),
        ]
    )

    (end,) = ends(subscribed_events, "llm")
    assert present(response_of(end).usage)["cost"] == _usd(0.0042)
    assert_balanced(subscribed_events)


async def test_a_lite_llm_usage_with_more_cache_tokens_than_prompt_tokens_leaves_the_call_unannotated(
    subscribed_events: list[nemo_relay.Event], capfd: pytest.CaptureFixture[str]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    usage: nemo_relay.JsonObject = {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
        "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 10},
    }
    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_GPT, usage=usage)
    model = LiteLlm(model="openai/gpt-4o", mock_response=reply)

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (end,) = ends(subscribed_events, "llm")
    assert status_of(end) == "OK"
    assert end.annotated_response is None
    assert "ADK prompt_token_count must include the cache tokens LiteLLM reports" in capfd.readouterr().err
    assert_balanced(subscribed_events)


async def test_a_lite_llm_stream_records_its_first_response_id(subscribed_events: list[nemo_relay.Event]) -> None:
    await _run_lite_llm_stream(
        [
            _content_chunk("Sunny ", completion_id=""),
            _content_chunk("streamed.", completion_id="chatcmpl-first"),
            _chunk([], completion_id="chatcmpl-later", usage=_STREAM_USAGE),
        ]
    )

    (end,) = ends(subscribed_events, "llm")
    assert response_of(end).id == "chatcmpl-first"
    assert_balanced(subscribed_events)


async def test_a_lite_llm_stream_that_fails_records_the_error_and_the_response_id(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode
    from openai import APIError

    body = _StreamBody([_content_chunk("Sunny ", completion_id="chatcmpl-lite-s")], error=httpx.ReadError("dropped"))
    canned = _Canned([httpx.Response(200, headers=_SSE, stream=body)])

    with pytest.raises(APIError):
        await run_recorded(
            LlmAgent(name="stream_agent", model=_lite_llm_model(canned), instruction="Be brief."),
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )

    (end,) = ends(subscribed_events, "llm")
    assert status_of(end) == "ERROR"
    assert response_of(end).id == "chatcmpl-lite-s"
    assert_balanced(subscribed_events)


async def test_a_lite_llm_stream_reads_the_cache_tokens_litellm_reports(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    usage: nemo_relay.JsonObject = {
        **_STREAM_USAGE,
        "prompt_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
    }

    await _run_lite_llm_stream(
        [
            _content_chunk("Sunny.", completion_id="chatcmpl-lite-s"),
            _chunk([], completion_id="chatcmpl-lite-s", usage=usage),
        ]
    )

    (end,) = ends(subscribed_events, "llm")
    assert _without_cost(present(response_of(end).usage)) == {
        "prompt_tokens": 50,
        "completion_tokens": 7,
        "total_tokens": 57,
        "cache_read_tokens": 20,
        "cache_write_tokens": 10,
        "uncached_input_tokens": 20,
    }
    assert_balanced(subscribed_events)


async def test_closing_a_recorded_lite_llm_stream_early_closes_the_provider_stream() -> None:
    from google.adk.models.lite_llm import LiteLlm

    from nemo_relay.integrations.google_adk._backends import CURRENT_ATTEMPT, Attempt, backend_of

    _import_litellm()
    from litellm.types.utils import ModelResponseStream

    chunks = [_content_chunk(text, completion_id="chatcmpl-lite-s") for text in ("Sunny ", "streamed.")]
    body = _StreamBody(chunks, error=None)
    client = _openai_client(_Canned([httpx.Response(200, headers=_SSE, stream=body)]))
    model = LiteLlm(model="openai/gpt-4o", client=client)
    await backend_of(model, model.model)
    attempt = Attempt(tool_choice=None, headers={}, provider="openai")
    token = CURRENT_ATTEMPT.set(attempt)
    try:
        stream = await model.llm_client.acompletion(
            model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi."}], tools=None, stream=True, client=client
        )
    finally:
        CURRENT_ATTEMPT.reset(token)

    async with contextlib.aclosing(as_instance(stream, AsyncGenerator)) as recorded:
        assert as_instance(await anext(recorded), ModelResponseStream).id == "chatcmpl-lite-s"

    assert body.closed
    assert attempt.recorded == {"id": "chatcmpl-lite-s"}


async def test_cancelling_an_early_closed_lite_llm_stream_during_provider_close_finishes_the_close() -> None:
    from google.adk.models.lite_llm import LiteLlm

    from nemo_relay.integrations.google_adk._backends import CURRENT_ATTEMPT, Attempt, backend_of

    chunks = [_content_chunk(text, completion_id="chatcmpl-lite-s") for text in ("Sunny ", "streamed.")]
    body = _HeldCloseBody(chunks)
    client = _openai_client(_Canned([httpx.Response(200, headers=_SSE, stream=body)]))
    model = LiteLlm(model="openai/gpt-4o", client=client)
    await backend_of(model, model.model)
    token = CURRENT_ATTEMPT.set(Attempt(tool_choice=None, headers={}, provider="openai"))
    try:
        stream = await model.llm_client.acompletion(
            model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi."}], tools=None, stream=True, client=client
        )
    finally:
        CURRENT_ATTEMPT.reset(token)

    async def read_one_chunk() -> None:
        async with contextlib.aclosing(as_instance(stream, AsyncGenerator)) as recorded:
            await anext(recorded)

    task = asyncio.create_task(read_one_chunk())
    async with asyncio.timeout(HANG_TIMEOUT):
        await body.closing.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    body.release.set()
    async with asyncio.timeout(HANG_TIMEOUT):
        await body.closed_event.wait()

    assert body.closed


@pytest.mark.parametrize(
    ("model_name", "arguments", "aliases", "provider"),
    [
        pytest.param("openai/gpt-4o", {}, {}, "openai", id="openai"),
        pytest.param(f"anthropic/{_CLAUDE}", {}, {}, "anthropic", id="anthropic"),
        pytest.param("relay-model", {"custom_llm_provider": "openai"}, {}, "openai", id="custom_llm_provider"),
        pytest.param("gpt-4o", {"api_base": "https://api.deepseek.com/v1"}, {}, "deepseek", id="api_base"),
        pytest.param(
            "gpt-4o",
            {"api_base": "https://api.deepseek.com/v1", "base_url": "https://api.perplexity.ai"},
            {},
            "perplexity",
            id="base_url_over_api_base",
        ),
        pytest.param("gpt-4o", {}, {"gpt-4o": f"anthropic/{_CLAUDE}"}, "anthropic", id="model_alias"),
        pytest.param("openai/gpt-4o", {"azure": True}, {}, "azure", id="azure_flag"),
        pytest.param("openai/gpt-4o", {"deployment_id": "relay-deployment"}, {}, "azure", id="deployment_id"),
        pytest.param(
            "prometheus-eval/prometheus-7b-v2.0",
            {"custom_llm_provider": "hosted_vllm"},
            {},
            "hosted_vllm",
            id="model_named_like_a_litellm_logger",
        ),
    ],
)
async def test_a_lite_llm_model_is_named_after_the_provider_litellm_routes_it_to(
    model_name: str,
    arguments: dict[str, object],
    aliases: dict[str, str],
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm

    monkeypatch.setattr(litellm, "model_alias_map", aliases)
    model = LiteLlm(model=model_name, mock_response="Sunny.", **arguments)

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == provider


async def test_a_lite_llm_model_is_named_after_the_provider_its_litellm_credential_routes_it_to(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm
    from litellm.types.utils import CredentialItem

    credential = CredentialItem(
        credential_name="relay-credential",
        credential_values={"api_base": "https://api.deepseek.com/v1"},
        credential_info={},
    )
    monkeypatch.setattr(litellm, "credential_list", [credential])
    model = LiteLlm(model="gpt-4o", mock_response="Sunny.", litellm_credential_name="relay-credential")

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "deepseek"


async def test_a_lite_llm_model_is_named_after_the_provider_its_provider_arguments_reach(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    monkeypatch.setenv("BEDROCK_MANTLE_REGION", "US_EAST_1")
    model = LiteLlm(
        model="openai.gpt-oss-120b",
        mock_response="Sunny.",
        custom_llm_provider="bedrock_mantle",
        aws_region_name="us-east-1",
        api_key="relay-token",
    )

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "bedrock_mantle"


async def test_a_lite_llm_call_is_named_after_the_provider_of_the_model_adk_requests(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    def to_claude(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
        llm_request.model = f"anthropic/{_CLAUDE}"
        return None

    model = LiteLlm(model="openai/gpt-4o", mock_response="Sunny.")
    agent = LlmAgent(name="weather_agent", model=model, instruction="Be brief.", before_model_callback=to_claude)

    adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "anthropic"
    assert present(start.category_profile)["model_name"] == f"anthropic/{_CLAUDE}"
    assert_balanced(subscribed_events)


def _renaming(model: str) -> nemo_relay.LlmRequestIntercept:
    def rename(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        edited = present(annotated)
        edited.model = model
        return nemo_relay.LLMRequestInterceptOutcome(request, edited)

    return rename


async def test_a_model_edit_litellm_routes_to_the_recorded_provider_reaches_lite_llm(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    calls: list[dict[str, object]] = []
    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_GPT, usage=_OPENAI_USAGE)
    model = LiteLlm(model="openai/gpt-4o", llm_client=_answering_client([reply], calls=calls))

    with llm_request_intercept(_renaming("openai/gpt-4o-mini")):
        adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    assert [call["model"] for call in calls] == ["openai/gpt-4o-mini"]
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "openai"
    assert_balanced(subscribed_events)


async def test_a_model_edit_litellm_routes_to_another_provider_fails_the_call(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    calls: list[dict[str, object]] = []
    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_GPT, usage=_OPENAI_USAGE)
    model = LiteLlm(model="openai/gpt-4o", llm_client=_answering_client([reply], calls=calls))

    with llm_request_intercept(_renaming(f"anthropic/{_CLAUDE}")), pytest.raises(ValueError, match="to anthropic"):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert calls == []
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "openai"
    assert status_of(ends(subscribed_events, "llm")[0]) == "ERROR"
    assert_balanced(subscribed_events)


async def test_a_model_edit_onto_a_context_window_fallback_key_fails_the_call(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    calls: list[dict[str, object]] = []
    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_GPT, usage=_OPENAI_USAGE)
    model = LiteLlm(
        model="openai/gpt-4o",
        llm_client=_answering_client([reply], calls=calls),
        context_window_fallback_dict={"openai/gpt-4o-mini": f"anthropic/{_CLAUDE}"},
    )

    with (
        llm_request_intercept(_renaming("openai/gpt-4o-mini")),
        pytest.raises(TypeError, match="context_window_fallback_dict"),
    ):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert calls == []
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "openai"
    assert status_of(ends(subscribed_events, "llm")[0]) == "ERROR"
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(
    ("model_name", "arguments", "feature"),
    [
        pytest.param("openai/gpt-4o", {"fallbacks": [f"anthropic/{_CLAUDE}"]}, "fallbacks", id="fallbacks"),
        pytest.param(
            "openai/gpt-4o",
            {"model_list": [{"model_name": "openai/gpt-4o", "litellm_params": {"model": f"anthropic/{_CLAUDE}"}}]},
            "model_list",
            id="model_list",
        ),
        pytest.param(
            "openai/gpt-4o",
            {"context_window_fallback_dict": {"openai/gpt-4o": f"anthropic/{_CLAUDE}"}},
            "context_window_fallback_dict",
            id="context_window_fallback_dict",
        ),
        pytest.param("openai/gpt-4o", {"prompt_id": "weather"}, "prompt_id", id="prompt_id"),
        pytest.param("litellm_agent/openai/gpt-4o", {}, "LiteLLM agent model", id="litellm_agent_model"),
    ],
)
async def test_litellm_options_that_can_move_a_call_to_another_provider_fail_lite_llm_calls(
    model_name: str, arguments: dict[str, object], feature: str, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    model = LiteLlm(model=model_name, mock_response="Sunny.", **arguments)

    with pytest.raises(TypeError, match=re.escape(feature)):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert starts(subscribed_events, "llm") == []
    assert_balanced(subscribed_events)


def _hooked_callbacks(hook: str) -> list[CustomLogger]:
    _import_litellm()
    from litellm.integrations.custom_logger import CustomLogger

    async def hooked(self: CustomLogger, *args: object, **kwargs: object) -> None:
        return None

    return [type("Hooked", (CustomLogger,), {hook: hooked})()]


def _instance_hooked_callbacks(hook: str) -> list[CustomLogger]:
    _import_litellm()
    from litellm.integrations.custom_logger import CustomLogger

    async def hooked(*args: object, **kwargs: object) -> None:
        return None

    callback = CustomLogger()
    setattr(callback, hook, hooked)
    return [callback]


def _prompt_manager() -> CustomPromptManagement:
    _import_litellm()
    from litellm.integrations.custom_prompt_management import CustomPromptManagement

    return CustomPromptManagement()


def _failing_prompt_manager() -> CustomPromptManagement:
    _import_litellm()
    from litellm.integrations.custom_prompt_management import CustomPromptManagement

    class Failing(CustomPromptManagement):
        @override
        def should_run_prompt_management(
            self,
            prompt_id: str | None,
            prompt_spec: PromptSpec | None,
            dynamic_callback_params: StandardCallbackDynamicParams,
        ) -> bool:
            raise RuntimeError("prompt store down")

    return Failing()


def _prompt_id_manager() -> AnthropicCacheControlHook:
    _import_litellm()
    from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook

    return AnthropicCacheControlHook()


def _prompt_management_logger() -> CustomLogger:
    _import_litellm()
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.integrations.prompt_management_base import PromptManagementBase

    class Managed(PromptManagementBase, CustomLogger):
        @property
        @override
        def integration_name(self) -> str:
            return "relay"

        @override
        def should_run_prompt_management(
            self,
            prompt_id: str | None,
            prompt_spec: PromptSpec | None,
            dynamic_callback_params: StandardCallbackDynamicParams,
        ) -> bool:
            return True

        @override
        def _compile_prompt_helper(
            self,
            prompt_id: str | None,
            prompt_spec: PromptSpec | None,
            prompt_variables: dict[str, object] | None,
            dynamic_callback_params: StandardCallbackDynamicParams,
            prompt_label: str | None = None,
            prompt_version: int | None = None,
        ) -> PromptManagementClient:
            raise AssertionError("LiteLLM selects no prompt manager for this call")

        @override
        async def async_compile_prompt_helper(
            self,
            prompt_id: str | None,
            prompt_variables: dict[str, object] | None,
            dynamic_callback_params: StandardCallbackDynamicParams,
            prompt_spec: PromptSpec | None = None,
            prompt_label: str | None = None,
            prompt_version: int | None = None,
        ) -> PromptManagementClient:
            raise AssertionError("LiteLLM selects no prompt manager for this call")

    return Managed()


@pytest.mark.parametrize(
    ("attribute", "value", "feature"),
    [
        pytest.param("model_fallbacks", lambda: [f"anthropic/{_CLAUDE}"], "LiteLLM fallbacks", id="model_fallbacks"),
        *(
            pytest.param(
                name, lambda: [_prompt_manager()], "callback CustomPromptManagement", id=f"{name}_prompt_manager"
            )
            for name in (
                "callbacks",
                "success_callback",
                "failure_callback",
                "_async_success_callback",
                "_async_failure_callback",
            )
        ),
        *(
            pytest.param("callbacks", functools.partial(_hooked_callbacks, hook), "callback Hooked", id=hook)
            for hook in (
                "async_pre_call_deployment_hook",
                "async_post_call_success_deployment_hook",
                "async_post_call_streaming_deployment_hook",
                "async_should_run_agentic_loop",
                "async_should_run_chat_completion_agentic_loop",
            )
        ),
        *(
            pytest.param(
                "callbacks",
                functools.partial(_instance_hooked_callbacks, hook),
                "callback CustomLogger",
                id=f"instance_{hook}",
            )
            for hook in ("async_pre_call_deployment_hook", "async_should_run_chat_completion_agentic_loop")
        ),
    ],
)
async def test_litellm_settings_that_can_move_a_call_to_another_provider_fail_lite_llm_calls(
    attribute: str,
    value: Callable[[], object],
    feature: str,
    monkeypatch: pytest.MonkeyPatch,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm

    monkeypatch.setattr(litellm, attribute, value())
    model = LiteLlm(model="openai/gpt-4o", mock_response="Sunny.")

    with pytest.raises(TypeError, match=re.escape(feature)):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert starts(subscribed_events, "llm") == []
    assert_balanced(subscribed_events)


_CALLBACK_LISTS: Final = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        pytest.param(
            "success_callback",
            functools.partial(_hooked_callbacks, "async_pre_call_deployment_hook"),
            id="deployment_hook_outside_litellm_callbacks",
        ),
        pytest.param(
            "callbacks",
            functools.partial(_hooked_callbacks, "async_build_agentic_loop_plan"),
            id="agentic_loop_plan_without_gate",
        ),
        pytest.param("success_callback", lambda: [_prompt_id_manager()], id="prompt_manager_needing_prompt_id"),
        pytest.param("callbacks", lambda: [_failing_prompt_manager()], id="prompt_manager_whose_check_raises"),
        pytest.param("success_callback", lambda: [_prompt_management_logger()], id="prompt_management_logger"),
        pytest.param(
            "callbacks",
            functools.partial(_instance_hooked_callbacks, "async_should_run_agentic_loop"),
            id="agentic_gate_on_an_instance",
        ),
    ],
)
async def test_litellm_settings_that_cannot_move_a_call_reach_lite_llm_calls(
    attribute: str,
    value: Callable[[], object],
    monkeypatch: pytest.MonkeyPatch,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm

    for name in _CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, attribute, value())
    model = LiteLlm(model="openai/gpt-4o", mock_response="Sunny.")

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "openai"
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(
    ("argument", "value", "feature"),
    [
        pytest.param("callbacks", lambda: [_prompt_manager()], "callback CustomPromptManagement", id="prompt_manager"),
        pytest.param(
            "success_callback",
            functools.partial(_hooked_callbacks, "async_should_run_agentic_loop"),
            "callback Hooked",
            id="agentic_loop",
        ),
        pytest.param(
            "success_callback",
            functools.partial(_hooked_callbacks, "async_should_run_chat_completion_agentic_loop"),
            "callback Hooked",
            id="chat_completion_agentic_loop",
        ),
    ],
)
async def test_lite_llm_callback_arguments_that_can_move_a_call_fail_lite_llm_calls(
    argument: str,
    value: Callable[[], object],
    feature: str,
    monkeypatch: pytest.MonkeyPatch,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm

    for name in _CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])
    model = LiteLlm(model="openai/gpt-4o", mock_response="Sunny.", **{argument: value()})

    with pytest.raises(TypeError, match=re.escape(feature)):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert starts(subscribed_events, "llm") == []
    assert_balanced(subscribed_events)


async def test_a_deployment_hook_in_the_lite_llm_callbacks_argument_reaches_lite_llm_calls(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm

    for name in _CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])
    model = LiteLlm(
        model="openai/gpt-4o",
        mock_response="Sunny.",
        callbacks=_hooked_callbacks("async_pre_call_deployment_hook"),
    )

    for _ in range(2):
        adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))
        assert final_text(adk_events) == "Sunny."

    assert [start.name for start in starts(subscribed_events, "llm")] == ["openai", "openai"]
    assert_balanced(subscribed_events)


async def test_a_lite_llm_credential_that_carries_a_model_list_fails_lite_llm_calls(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm
    from litellm.types.utils import CredentialItem

    deployments = [{"model_name": "openai/gpt-4o", "litellm_params": {"model": f"anthropic/{_CLAUDE}"}}]
    credential = CredentialItem(
        credential_name="relay-credential", credential_values={"model_list": deployments}, credential_info={}
    )
    monkeypatch.setattr(litellm, "credential_list", [credential])
    model = LiteLlm(model="openai/gpt-4o", mock_response="Sunny.", litellm_credential_name="relay-credential")

    with pytest.raises(TypeError, match="model_list"):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert starts(subscribed_events, "llm") == []
    assert_balanced(subscribed_events)


async def test_callbacks_in_a_lite_llm_credential_reach_lite_llm_calls(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm
    from litellm.types.utils import CredentialItem

    for name in _CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])
    credential = CredentialItem(
        credential_name="relay-credential",
        credential_values={
            "callbacks": [_prompt_manager()],
            "success_callback": _hooked_callbacks("async_should_run_agentic_loop"),
        },
        credential_info={},
    )
    monkeypatch.setattr(litellm, "credential_list", [credential])
    model = LiteLlm(model="openai/gpt-4o", mock_response="Sunny.", litellm_credential_name="relay-credential")

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "openai"
    assert_balanced(subscribed_events)


async def test_a_lite_llm_model_litellm_names_no_provider_for_is_named_after_its_class(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model="relay-model", usage=_OPENAI_USAGE)
    model = LiteLlm(model="relay-model", llm_client=_answering_client([reply], calls=[]))

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "LiteLlm"
    (end,) = ends(subscribed_events, "llm")
    assert response_of(end).id == "chatcmpl-lite"
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(
    ("model_name", "arguments", "aliases"),
    [
        pytest.param("relay-model", {}, {}, id="unknown_name"),
        pytest.param("relay-alias", {}, {"relay-alias": f"anthropic/{_CLAUDE}"}, id="alias_of_an_unknown_name"),
        pytest.param("relay-deployment", {"azure": True}, {}, id="azure_flag_on_an_unknown_name"),
        pytest.param("relay-model", {"deployment_id": "relay-deployment"}, {}, id="deployment_id_on_an_unknown_name"),
        pytest.param(
            "relay-model",
            {"api_base": "https://relay.invalid/v1", "base_url": "https://api.deepseek.com/v1"},
            {},
            id="api_base_litellm_does_not_know",
        ),
    ],
)
async def test_a_lite_llm_call_litellm_cannot_route_fails_inside_its_llm_scope(
    model_name: str,
    arguments: dict[str, object],
    aliases: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    import litellm
    from litellm.exceptions import BadRequestError

    monkeypatch.setattr(litellm, "model_alias_map", aliases)
    model = LiteLlm(model=model_name, mock_response="Sunny.", **arguments)

    with pytest.raises(BadRequestError, match="LLM Provider NOT provided"):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    (start,) = starts(subscribed_events, "llm")
    assert start.name == "LiteLlm"
    (end,) = ends(subscribed_events, "llm")
    assert status_of(end) == "ERROR"
    assert_balanced(subscribed_events)


async def test_a_lite_llm_call_with_a_provider_argument_litellm_rejects_fails_inside_its_llm_scope(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    _import_litellm()
    from litellm.exceptions import APIConnectionError

    model = LiteLlm(model="openai/gpt-4o", mock_response="Sunny.", tpm="relay")

    with pytest.raises(APIConnectionError, match="validation error for GenericLiteLLMParams"):
        await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    (start,) = starts(subscribed_events, "llm")
    assert start.name == "LiteLlm"
    (end,) = ends(subscribed_events, "llm")
    assert status_of(end) == "ERROR"
    assert_balanced(subscribed_events)


def test_resolving_a_lite_llm_provider_imports_litellm_in_production_mode() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", _LITELLM_IMPORT_PROBE],
        env=_probe_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines()[-1] == "PRODUCTION"


async def test_reading_an_anthropic_model_leaves_adk_lite_llm_unimported(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent

    model = _anthropic_model(_Canned([_claude_reply()]))
    monkeypatch.delitem(sys.modules, "google.adk.models.lite_llm", raising=False)

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    assert final_text(adk_events) == "Sunny."
    (start,) = starts(subscribed_events, "llm")
    assert start.name == "anthropic"
    assert "google.adk.models.lite_llm" not in sys.modules
    assert_balanced(subscribed_events)


async def test_a_lite_llm_provider_is_resolved_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.adk.models.lite_llm import LiteLlm

    from nemo_relay.integrations.google_adk._backends import backend_of

    _import_litellm()
    import litellm

    resolve = litellm.get_llm_provider
    threads: list[threading.Thread] = []

    def get_llm_provider(
        model: str,
        custom_llm_provider: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> tuple[str, str, str | None, str | None]:
        threads.append(threading.current_thread())
        return resolve(model, custom_llm_provider, api_base, api_key, litellm_params)

    monkeypatch.setattr(litellm, "get_llm_provider", get_llm_provider)

    model = LiteLlm(model="openai/gpt-4o")
    backend = await backend_of(model, model.model)

    assert backend.provider == "openai"
    assert threads
    assert threading.current_thread() not in threads


async def test_a_lite_llm_model_sharing_a_relay_client_runs_outside_relay(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    client = LiteLLMClient()
    model = LiteLlm(model="openai/gpt-4o", mock_response="Sunny.", llm_client=client)
    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))
    other = LiteLlm(model="openai/gpt-4o", mock_response="Cloudy.", llm_client=client)
    request = LlmRequest(model="openai/gpt-4o", contents=[types.UserContent("Hi.")])

    (response,) = [response async for response in other.generate_content_async(request)]

    assert final_text(adk_events) == "Sunny."
    assert present(present(response.content).parts)[0].text == "Cloudy."
    assert len(starts(subscribed_events, "llm")) == 1


async def test_a_lite_llm_tool_call_is_recorded_and_runs_its_tool(subscribed_events: list[nemo_relay.Event]) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    replies = [
        _lite_llm_tool_call(response_id="chatcmpl-call"),
        _lite_llm_reply("Sunny in Paris.", response_id="chatcmpl-2", model=_GPT, usage=_OPENAI_USAGE),
    ]
    model = LiteLlm(model="openai/gpt-4o", llm_client=_answering_client(replies, calls=[]))
    agent = LlmAgent(name="weather_agent", model=model, instruction="Be brief.", tools=[get_weather])

    adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Sunny in Paris."
    first, second = ends(subscribed_events, "llm")
    assert (response_of(first).id, response_of(second).id) == ("chatcmpl-call", "chatcmpl-2")
    assert response_of(first).tool_calls == [
        {"id": "call_1", "name": "get_weather", "arguments": {"location": "Paris"}}
    ]
    assert [start.name for start in starts(subscribed_events, "tool")] == ["get_weather"]
    assert_balanced(subscribed_events)


def _calling_config(value: nemo_relay.JsonObject) -> nemo_relay.JsonObject:
    return {"provider": "google_adk", "kind": "function_calling_config", "value": value}


def _choosing(choice: str | nemo_relay.JsonObject) -> nemo_relay.LlmRequestIntercept:
    def choose(
        name: str, request: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        edited = present(annotated)
        edited.tool_choice = choice
        return nemo_relay.LLMRequestInterceptOutcome(request, edited)

    return choose


@pytest.mark.parametrize(
    ("choice", "model_args", "sent", "declared"),
    [
        pytest.param("required", {}, "required", ["get_weather", "get_time"], id="required"),
        pytest.param("none", {}, "none", ["get_weather", "get_time"], id="none"),
        pytest.param(
            _calling_config({"mode": "ANY", "allowed_function_names": ["get_time"]}),
            {},
            {"type": "function", "function": {"name": "get_time"}},
            ["get_time"],
            id="one_allowed_function",
        ),
        pytest.param(
            _calling_config({"mode": "AUTO"}), {"tool_choice": "required"}, None, ["get_weather", "get_time"], id="auto"
        ),
        pytest.param(
            _calling_config({"allowed_function_names": ["get_time"]}),
            {"tool_choice": "required"},
            "required",
            ["get_time"],
            id="unset_mode",
        ),
    ],
)
async def test_tool_choice_edits_reach_lite_llm(
    choice: str | nemo_relay.JsonObject,
    model_args: dict[str, object],
    sent: str | nemo_relay.JsonObject | None,
    declared: list[str],
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    calls: list[dict[str, object]] = []
    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_GPT, usage=_OPENAI_USAGE)
    model = LiteLlm(model="openai/gpt-4o", llm_client=_answering_client([reply], calls=calls), **model_args)
    agent = LlmAgent(name="weather_agent", model=model, instruction="Be brief.", tools=[get_weather, get_time])

    with llm_request_intercept(_choosing(choice)):
        adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Sunny."
    (call,) = calls
    assert call.get("tool_choice") == sent
    assert [as_object(as_object(tool)["function"])["name"] for tool in as_list(call["tools"])] == declared
    assert_balanced(subscribed_events)


@pytest.mark.parametrize(
    ("value", "tools", "message"),
    [
        pytest.param(
            {"mode": "ANY", "allowed_function_names": ["get_forecast"]}, [get_weather], "get_forecast", id="undeclared"
        ),
        pytest.param({"mode": "VALIDATED"}, [get_weather], "VALIDATED", id="validated_mode"),
        pytest.param({"mode": "ANY"}, [], "requires a tool call", id="required_without_tools"),
    ],
)
async def test_tool_choice_edits_lite_llm_cannot_carry_fail_the_call(
    value: nemo_relay.JsonObject,
    tools: list[Callable[[str], dict[str, str]]],
    message: str,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    canned = _Canned([_chat_reply()])
    agent = LlmAgent(name="weather_agent", model=_lite_llm_model(canned), instruction="Be brief.", tools=[*tools])

    with llm_request_intercept(_choosing(_calling_config(value))), pytest.raises(ValueError, match=message):
        await run_recorded(agent)

    assert canned.requests == []
    assert status_of(ends(subscribed_events, "llm")[0]) == "ERROR"
    assert_balanced(subscribed_events)


async def test_a_none_tool_choice_edit_on_a_lite_llm_call_without_tools_sends_no_tool_choice(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    canned = _Canned([_chat_reply()])
    agent = LlmAgent(name="weather_agent", model=_lite_llm_model(canned), instruction="Be brief.")

    with llm_request_intercept(_choosing("none")):
        adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Sunny."
    assert "tool_choice" not in canned.requests[0]
    assert "tools" not in canned.requests[0]
    assert_balanced(subscribed_events)


async def test_a_tool_choice_edit_keeps_the_built_in_tools_lite_llm_sends(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.tools import google_search

    monkeypatch.setenv("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")
    calls: list[dict[str, object]] = []
    reply = _lite_llm_reply("Sunny.", response_id="chatcmpl-lite", model=_GPT, usage=_OPENAI_USAGE)
    model = LiteLlm(model="gemini/gemini-2.5-flash", llm_client=_answering_client([reply], calls=calls))
    agent = LlmAgent(
        name="weather_agent", model=model, instruction="Be brief.", tools=[get_weather, get_time, google_search]
    )
    choice = _calling_config({"mode": "ANY", "allowed_function_names": ["get_time"]})

    with llm_request_intercept(_choosing(choice)):
        adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Sunny."
    (call,) = calls
    assert call["tool_choice"] == {"type": "function", "function": {"name": "get_time"}}
    function, search = as_list(call["tools"])
    assert as_object(as_object(function)["function"])["name"] == "get_time"
    assert search == {"googleSearch": {}}
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

    model, _ = _gemini(response, monkeypatch, model="gemini-2.5-flash")

    adk_events = await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

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


def _lite_llm_model(canned: _Canned) -> BaseLlm:
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model="openai/gpt-4o", client=_openai_client(canned))


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
        pytest.param(_openai_responses_model, _responses_text_reply, "openai", "openai_responses", id="responses"),
        pytest.param(_apigee_chat_model, _chat_reply, "openai", "openai_chat", id="apigee_chat"),
        pytest.param(_lite_llm_model, _chat_reply, "openai", "openai_chat", id="lite_llm"),
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
    ("build", "reply", "requested", "recorded", "sent"),
    [
        pytest.param(
            _anthropic_model,
            _claude_reply,
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20251001",
            id="anthropic",
        ),
        pytest.param(_openai_chat_model, _chat_reply, "gpt-4o-mini", "gpt-4o", "gpt-4o", id="openai_chat"),
        pytest.param(
            _openai_responses_model, _responses_text_reply, "gpt-5-mini", "gpt-5-mini", "gpt-5-mini", id="responses"
        ),
        pytest.param(
            _apigee_chat_model,
            _chat_reply,
            "apigee/openai/gpt-4o-mini",
            "apigee/openai/gpt-4o-mini",
            "gpt-4o-mini",
            id="apigee_chat",
        ),
        pytest.param(
            _lite_llm_model, _chat_reply, "openai/gpt-4o-mini", "openai/gpt-4o-mini", "gpt-4o-mini", id="lite_llm"
        ),
    ],
)
async def test_a_call_is_recorded_with_the_model_its_class_reads_after_a_callback_requests_another(
    build: Callable[[_Canned], BaseLlm],
    reply: Callable[[], httpx.Response],
    requested: str,
    recorded: str,
    sent: str,
    subscribed_events: list[nemo_relay.Event],
) -> None:
    from google.adk.agents import LlmAgent

    def request(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
        llm_request.model = requested
        return None

    canned = _Canned([reply()])
    agent = LlmAgent(name="weather_agent", model=build(canned), instruction="Be brief.", before_model_callback=request)

    adk_events = await run_recorded(agent)

    assert final_text(adk_events) == "Sunny."
    assert canned.requests[0]["model"] == sent
    (start,) = starts(subscribed_events, "llm")
    assert present(start.category_profile)["model_name"] == recorded
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
    model, _ = _gemini(response, monkeypatch, model="tunedModels/weather-123")

    await run_recorded(LlmAgent(name="weather_agent", model=model, instruction="Be brief."))

    (start,) = starts(subscribed_events, "llm")
    assert start.name == "gemini"
    assert present(start.category_profile)["model_name"] == "tunedModels/weather-123"


async def test_a_gemini_call_is_recorded_with_the_model_a_callback_requests(
    monkeypatch: pytest.MonkeyPatch, subscribed_events: list[nemo_relay.Event]
) -> None:
    from google.adk.agents import LlmAgent
    from google.genai import types

    def to_pro(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
        llm_request.model = "gemini-2.5-pro"
        return None

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="Sunny.")]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
    )
    model, sent = _gemini(response, monkeypatch, model="gemini-2.5-flash")
    agent = LlmAgent(name="weather_agent", model=model, instruction="Be brief.", before_model_callback=to_pro)

    await run_recorded(agent)

    assert sent == ["gemini-2.5-pro"]
    (start,) = starts(subscribed_events, "llm")
    assert present(start.category_profile)["model_name"] == "gemini-2.5-pro"
    assert_balanced(subscribed_events)


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


def _chat_reply_reporting_cost_breakdown() -> httpx.Response:
    body = as_object(_chat_reply_reporting_cost().json())
    body["usage"] = {**as_object(body["usage"]), "cost": {"request_cost": 0.005, "total_cost": 0.0042}}
    return httpx.Response(200, json=body)


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
        pytest.param(_lite_llm_model, _chat_reply_reporting_cost, _usd(0.0042), id="lite_llm_cost"),
        pytest.param(
            _lite_llm_model,
            _chat_reply_reporting_cost_breakdown,
            _usd(0.0042),
            id="lite_llm_cost_breakdown",
            marks=pytest.mark.filterwarnings(_ADK_USAGE_DUMP_WARNING),
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
