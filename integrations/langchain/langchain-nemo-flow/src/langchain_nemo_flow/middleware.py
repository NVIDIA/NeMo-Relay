# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain AgentMiddleware implementation for NeMo Flow."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from nemo_flow._native import LLMRequest

from langchain_nemo_flow._nemo_flow import run_sync
from langchain_nemo_flow._serialization import (
    ModelRequestHeaders,
    ModelRequestToPayload,
    ModelResponseFromJson,
    ModelResponseToJson,
    PayloadToModelRequest,
    best_effort_model_response_from_json,
    best_effort_model_response_to_json,
    default_model_request_payload,
    default_payload_to_model_request,
    get_model_name,
    get_model_provider,
)

import nemo_flow

_logger = logging.getLogger(__name__)
ProviderName = str | Callable[[ModelRequest[Any]], str]

if TYPE_CHECKING:
    from nemo_flow.codecs import LlmCodec, LlmResponseCodec


class NemoFlowMiddleware(AgentMiddleware):
    """Route LangChain agent model and tool calls through NeMo Flow.

    This uses LangChain's public ``AgentMiddleware`` hooks. It applies to agents
    built with ``langchain.agents.create_agent(..., middleware=[...])``.
    """

    def __init__(
        self,
        *,
        name: str = "NemoFlowMiddleware",
        codec: LlmCodec,
        model_request_to_payload: ModelRequestToPayload = default_model_request_payload,
        model_request_headers: ModelRequestHeaders | None = None,
        payload_to_model_request: PayloadToModelRequest = default_payload_to_model_request,
        model_response_to_json: ModelResponseToJson = best_effort_model_response_to_json,
        model_response_from_json: ModelResponseFromJson = best_effort_model_response_from_json,
    ) -> None:
        super().__init__()
        self._name = name
        self._codec = codec
        self._model_request_to_payload = model_request_to_payload
        self._model_request_headers = model_request_headers
        self._payload_to_model_request = payload_to_model_request
        self._model_response_to_json = model_response_to_json
        self._model_response_from_json = model_response_from_json

    @property
    def name(self) -> str:
        """Middleware name used by LangChain graph nodes and traces."""
        return self._name

    async def llm_execute(
        self,
        model_name: str,
        request: "LLMRequest",
        codec: LlmCodec,
        response_codec: LlmResponseCodec,
        func: Callable[..., Any],
    ) -> Any:
        """Execute a non-streaming LLM call through the NeMo Flow pipeline."""
        # TODO: just import this
        return await nemo_flow.llm.execute(
            model_name,
            request,
            func,
            model_name=model_name,
            codec=codec,
            response_codec=response_codec,
        )

    async def llm_stream_execute(
        self,
        model_name: str,
        request: "LLMRequest",
        func: Callable[..., Any],
        collector: Callable[[Any], None],
        finalizer: Callable[[], Any],
        codec: LlmCodec,
        response_codec: LlmResponseCodec,
    ) -> Any:
        """Execute a streaming LLM call through the NeMo Flow pipeline."""
        return await nemo_flow.llm.stream_execute(
            model_name,
            request,
            func,
            collector,
            finalizer,
            model_name=model_name,
            codec=codec,
            response_codec=response_codec,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Wrap a sync LangChain agent model call in NeMo Flow LLM execution."""

        llm_request = nemo_flow.LLMRequest(self._headers_for(request), self._model_request_to_payload(request))
        model_name = get_model_name(request.model)

        async def _call(req: Any) -> Any:
            response = handler(self._payload_to_model_request(request, req.content))
            return self._model_response_to_json(response, self._codec)

        result = run_sync(
            self.llm_execute(
                model_name=model_name,
                request=llm_request,
                func=_call,
                codec=self._codec,
                response_codec=self._codec,
            )
        )
        return self._model_response_from_json(result, self._codec)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Wrap an async LangChain agent model call in NeMo Flow LLM execution."""
        llm_request = nemo_flow.LLMRequest(self._headers_for(request), self._model_request_to_payload(request))
        model_name = get_model_name(request.model)

        async def _call(req: Any) -> Any:
            response = await handler(self._payload_to_model_request(request, req.content))
            return self._model_response_to_json(response, self._codec)

        result = await self.llm_execute(
            model_name=model_name,
            request=llm_request,
            func=_call,
            codec=self._codec,
            response_codec=self._codec,
        )
        return self._model_response_from_json(result, self._codec)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Wrap a sync LangChain agent tool call in NeMo Flow tool execution."""

        codec = nemo_flow.typed.BestEffortAnyCodec()
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args") or {}

        def _call(args: Any) -> ToolMessage | Command[Any]:
            return handler(request.override(tool_call={**request.tool_call, "args": args}))

        return run_sync(
            nemo_flow.typed.tool_execute(tool_name, tool_args, _call, codec, codec)
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap an async LangChain agent tool call in NeMo Flow tool execution."""

        codec = nemo_flow.typed.BestEffortAnyCodec()
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args") or {}

        async def _call(args: Any) -> ToolMessage | Command[Any]:
            return await handler(request.override(tool_call={**request.tool_call, "args": args}))

        return await nemo_flow.typed.tool_execute(tool_name, tool_args, _call, codec, codec)

    def _headers_for(self, request: ModelRequest[Any]) -> dict[str, str]:
        return self._model_request_headers(request) if self._model_request_headers else {}
