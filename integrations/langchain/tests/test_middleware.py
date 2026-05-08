# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Flow middleware."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import nemo_flow
import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from nemo_flow.codecs import AnthropicMessagesCodec, OpenAIChatCodec, OpenAIResponsesCodec

from langchain_nemo_flow import _serialization
from langchain_nemo_flow.middleware import NemoFlowMiddleware


class RecordingMiddleware(NemoFlowMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def _llm_execute(
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
    fake_model = MagicMock(spec=BaseChatModel)
    fake_model.model = "fake-model"

    return ModelRequest(
        model=fake_model,
        messages=[HumanMessage(content="hello")],
        model_settings={"temperature": 1.0},
    )


def _tool_call_request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": "lookup", "args": {"query": "original"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=MagicMock(),
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


def test_wrap_tool_call_routes_through_tool_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = NemoFlowMiddleware()
    parent_handle = MagicMock()
    seen_request: ToolCallRequest | None = None

    async def execute_side_effect(
        *,
        func: Any,
        **kwargs: Any
    ) -> ToolMessage:
        return func({"query": "intercepted"})

    mock_tool_execute = AsyncMock(side_effect=execute_side_effect)

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal seen_request
        seen_request = request
        return ToolMessage(content="done", tool_call_id=request.tool_call["id"])

    monkeypatch.setattr(nemo_flow.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_flow.typed, "tool_execute", mock_tool_execute)

    response = middleware.wrap_tool_call(_tool_call_request(), handler)

    assert response.content == "done"
    assert seen_request is not None
    assert seen_request.tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle
    assert isinstance(kwargs["args_codec"], nemo_flow.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_flow.typed.BestEffortAnyCodec)


def test_awrap_tool_call_routes_through_tool_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = NemoFlowMiddleware()
    parent_handle = MagicMock()
    seen_request: ToolCallRequest | None = None

    async def execute_side_effect(
        *,
        func: Any,
        **kwargs: Any
    ) -> ToolMessage:
        return await func({"query": "intercepted"})

    mock_tool_execute = AsyncMock(side_effect=execute_side_effect)

    async def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal seen_request
        seen_request = request
        return ToolMessage(content="done", tool_call_id=request.tool_call["id"])

    monkeypatch.setattr(nemo_flow.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_flow.typed, "tool_execute", mock_tool_execute)

    response = asyncio.run(middleware.awrap_tool_call(_tool_call_request(), handler))

    assert response.content == "done"
    assert seen_request is not None
    assert seen_request.tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle
    assert isinstance(kwargs["args_codec"], nemo_flow.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_flow.typed.BestEffortAnyCodec)


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
