# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend for ADK's `ApigeeLlm` by API mode. Swaps its chat completions client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.models.apigee_llm import ApigeeLlm

from nemo_relay.integrations.google_adk._backends._attempt import CURRENT_ATTEMPT, Backend, sent_headers
from nemo_relay.integrations.google_adk._backends._chat_completions import CHAT_CODEC
from nemo_relay.integrations.google_adk._backends._gemini import gemini_backend
from nemo_relay.integrations.google_adk._serialization import RECORDED_ID_KEY, RECORDED_USAGE_KEY
from nemo_relay.integrations.google_adk._swap import Methods, swap_class

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Mapping

    import httpx
    from google.adk.models.apigee_llm import ChatCompletionsResponseHandler, CompletionsHTTPClient
    from google.adk.models.llm_response import LlmResponse

    from nemo_relay import Json


def apigee_backend(model: ApigeeLlm, request_model: str | None) -> Backend:
    """Return Gemini backend in GenAI mode, `openai` Chat Completions backend in chat completions mode.

    In chat completions mode, swap proxy client's class to send attempt's request headers and record each parsed
    response.
    """
    if model._api_type is not ApigeeLlm.ApiType.CHAT_COMPLETIONS:
        return gemini_backend(model, request_model)
    swap_class(model._completions_http_client, _completions_methods)
    return Backend("openai", CHAT_CODEC, client_headers=True, model=request_model)


def _completions_methods(cls: type[CompletionsHTTPClient]) -> Methods[CompletionsHTTPClient]:
    """Return `CompletionsHTTPClient` methods that send attempt's request headers and record each parsed body.

    Attempt header replaces proxy header of same name, ignoring case. Parsed body carries completion id and usage.
    `LlmResponse` drops id and usage cost.
    """
    post = cls._httpx_post_with_retry
    stream = cls._handle_streaming
    parse = cls._parse_response
    parse_line = cls._parse_streaming_line

    async def _httpx_post_with_retry(
        self: CompletionsHTTPClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        timeout: float | None,
    ) -> httpx.Response:
        return await post(self, url, payload, sent_headers(headers), timeout=timeout)

    def _handle_streaming(
        self: CompletionsHTTPClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        timeout: float | None,
    ) -> AsyncGenerator[LlmResponse, None]:
        return stream(self, url, payload, sent_headers(headers), timeout=timeout)

    def _parse_response(self: CompletionsHTTPClient, response: dict[str, Any]) -> LlmResponse:
        _record(response)
        return parse(self, response)

    def _parse_streaming_line(
        self: CompletionsHTTPClient, chunk: dict[str, Any], accumulator: ChatCompletionsResponseHandler
    ) -> Generator[LlmResponse, None, None]:
        _record(chunk)
        return parse_line(self, chunk, accumulator)

    return {
        "_httpx_post_with_retry": _httpx_post_with_retry,
        "_handle_streaming": _handle_streaming,
        "_parse_response": _parse_response,
        "_parse_streaming_line": _parse_streaming_line,
    }


def _record(completion: Mapping[str, Json]) -> None:
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None:
        return
    if isinstance(identifier := completion.get("id"), str):
        attempt.recorded[RECORDED_ID_KEY] = identifier
    if isinstance(usage := completion.get("usage"), dict):
        attempt.recorded[RECORDED_USAGE_KEY] = usage
