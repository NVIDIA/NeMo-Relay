# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Flow middleware."""

from __future__ import annotations

import asyncio
from typing import Any

import nemo_flow
import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from nemo_flow.codecs import AnthropicMessagesCodec, OpenAIChatCodec, OpenAIResponsesCodec

from langchain_nemo_flow import _serialization
from langchain_nemo_flow.middleware import NemoFlowMiddleware


class FakeModel:
    model = "fake-model"


class RecordingMiddleware(NemoFlowMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def llm_execute(
        self,
        model_name: str,
        request: nemo_flow.LLMRequest,
        codec: Any,
        response_codec: Any,
        func: Any,
    ) -> Any:
        self.calls.append(
            {
                "model_name": model_name,
                "request": request,
                "codec": codec,
                "response_codec": response_codec,
            }
        )
        intercepted = nemo_flow.LLMRequest(
            request.headers,
            {
                **request.content,
                "model_settings": {"temperature": 0.25},
            },
        )
        return await func(intercepted)


def _model_request() -> ModelRequest[Any]:
    return ModelRequest(
        model=FakeModel(),
        messages=[HumanMessage(content="hello")],
        model_settings={"temperature": 1.0},
    )


def test_wrap_model_call_routes_through_llm_execute() -> None:
    middleware = RecordingMiddleware()
    seen_request: ModelRequest[Any] | None = None

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal seen_request
        seen_request = request
        return ModelResponse(result=[AIMessage(content="done")])

    response = middleware.wrap_model_call(_model_request(), handler)

    assert response.result[0].content == "done"
    assert seen_request is not None
    assert seen_request.model_settings == {"temperature": 0.25}
    assert middleware.calls[0]["model_name"] == "fake-model"
    assert middleware.calls[0]["request"].content["model"] == "fake-model"
    assert middleware.calls[0]["codec"] is None
    assert middleware.calls[0]["response_codec"] is None


def test_awrap_model_call_routes_through_llm_execute() -> None:
    middleware = RecordingMiddleware()
    seen_request: ModelRequest[Any] | None = None

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal seen_request
        seen_request = request
        return ModelResponse(result=[AIMessage(content="done")])

    response = asyncio.run(middleware.awrap_model_call(_model_request(), handler))

    assert response.result[0].content == "done"
    assert seen_request is not None
    assert seen_request.model_settings == {"temperature": 0.25}
    assert middleware.calls[0]["model_name"] == "fake-model"
    assert middleware.calls[0]["request"].content["model"] == "fake-model"
    assert middleware.calls[0]["codec"] is None
    assert middleware.calls[0]["response_codec"] is None


def test_infer_codec_from_supported_model_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeChatAnthropic:
        pass

    class FakeChatOpenAI:
        def __init__(self, *, use_responses_api: bool = False) -> None:
            self.use_responses_api = use_responses_api

    monkeypatch.setattr(_serialization, "ChatAnthropic", FakeChatAnthropic)
    monkeypatch.setattr(_serialization, "ChatOpenAI", FakeChatOpenAI)

    assert isinstance(_serialization.infer_codec_from_model(FakeChatAnthropic()), AnthropicMessagesCodec)
    assert isinstance(_serialization.infer_codec_from_model(FakeChatOpenAI()), OpenAIChatCodec)
    assert isinstance(
        _serialization.infer_codec_from_model(FakeChatOpenAI(use_responses_api=True)),
        OpenAIResponsesCodec,
    )
    assert _serialization.infer_codec_from_model(object()) is None
