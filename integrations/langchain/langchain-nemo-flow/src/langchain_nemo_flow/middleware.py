# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain AgentMiddleware implementation for NeMo Flow."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from nemo_flow._native import LLMRequest

from langchain_nemo_flow._nemo_flow import get_nemo_flow, run_sync
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
    model_name,
    model_provider,
)

import nemo_flow

_logger = logging.getLogger(__name__)
ProviderName = str | Callable[[ModelRequest[Any]], str]


class NemoFlowMiddleware(AgentMiddleware):
    """Route LangChain agent model and tool calls through NeMo Flow.

    This uses LangChain's public ``AgentMiddleware`` hooks. It applies to agents
    built with ``langchain.agents.create_agent(..., middleware=[...])``.
    """

    def __init__(
        self,
        *,
        name: str = "NemoFlowMiddleware",
        provider_name: ProviderName | None = None,
        require_active_scope: bool = False,
        model_request_to_payload: ModelRequestToPayload = default_model_request_payload,
        model_request_headers: ModelRequestHeaders | None = None,
        payload_to_model_request: PayloadToModelRequest = default_payload_to_model_request,
        model_response_to_json: ModelResponseToJson = best_effort_model_response_to_json,
        model_response_from_json: ModelResponseFromJson = best_effort_model_response_from_json,
        request_codec_factory: Callable[[Any], Any] | None = None,
        response_codec_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._provider_name = provider_name
        self._require_active_scope = require_active_scope
        self._model_request_to_payload = model_request_to_payload
        self._model_request_headers = model_request_headers
        self._payload_to_model_request = payload_to_model_request
        self._model_response_to_json = model_response_to_json
        self._model_response_from_json = model_response_from_json
        self._request_codec_factory = request_codec_factory
        self._response_codec_factory = response_codec_factory

    @property
    def name(self) -> str:
        """Middleware name used by LangChain graph nodes and traces."""
        return self._name

    async def llm_execute(
        self,
        model_name: str,
        request: "LLMRequest",
        codec: Any, # TODO: add proper type
        response_codec: Any, # TODO: add proper type
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
        codec: Any, # TODO: add proper type
        response_codec: Any, # TODO: add proper type
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

        codec = nemo_flow.typed.BestEffortAnyCodec()
        llm_request = nemo_flow.LLMRequest(self._headers_for(request), self._model_request_to_payload(request))
        provider = self._provider_for(request)
        model = model_name(request.model)

        async def _call(req: Any) -> Any:
            response = handler(self._payload_to_model_request(request, req.content))
            return self._model_response_to_json(response, codec)

        result = run_sync(
            self.llm_execute(
                model_name=model,
                request=llm_request,
                func=_call,
                codec=self._make_request_codec(),
                response_codec=self._make_response_codec(),
            )
        )
        return self._model_response_from_json(result, codec)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Wrap an async LangChain agent model call in NeMo Flow LLM execution."""
        codec = nemo_flow.typed.BestEffortAnyCodec()
        llm_request = nemo_flow.LLMRequest(self._headers_for(request), self._model_request_to_payload(request))
        provider = self._provider_for(request)
        model = model_name(request.model)

        async def _call(req: Any) -> Any:
            response = await handler(self._payload_to_model_request(request, req.content))
            return self._model_response_to_json(response, codec)

        result = await self.llm_execute(
            model_name=model,
            request=llm_request,
            func=_call,
            codec=self._make_request_codec(),
            response_codec=self._make_response_codec(),
        )
        return self._model_response_from_json(result, codec)

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


    def _make_request_codec(self) -> Any | None:
        return self._request_codec_factory(nemo_flow) if self._request_codec_factory else None

    def _make_response_codec(self) -> Any | None:
        return self._response_codec_factory(nemo_flow) if self._response_codec_factory else None

    def _headers_for(self, request: ModelRequest[Any]) -> dict[str, str]:
        return self._model_request_headers(request) if self._model_request_headers else {}

    def _provider_for(self, request: ModelRequest[Any]) -> str:
        if callable(self._provider_name):
            return self._provider_name(request)
        return self._provider_name or model_provider(request.model)
