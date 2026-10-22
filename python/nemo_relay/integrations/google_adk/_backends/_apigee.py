# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend for ADK's `ApigeeLlm` by API mode. Swaps its chat completions client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.models.apigee_llm import ApigeeLlm

from nemo_relay.integrations.google_adk._backends._attempt import CURRENT_ATTEMPT, Backend, record_cost, sent_headers
from nemo_relay.integrations.google_adk._backends._gemini import gemini_backend
from nemo_relay.integrations.google_adk._serialization import CODECS, Api
from nemo_relay.integrations.google_adk._swap import swap_class

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Mapping

    import httpx
    from google.adk.models.apigee_llm import ChatCompletionsResponseHandler, CompletionsHTTPClient
    from google.adk.models.llm_response import LlmResponse

    from nemo_relay import Json


def apigee_backend(model: ApigeeLlm) -> Backend:
    """Return the Gemini backend in GenAI mode, the `openai` Chat Completions backend in chat mode.

    In chat mode, swap the proxy client's class to send the attempt's request headers and record each parsed response.
    """
    if model._api_type is not ApigeeLlm.ApiType.CHAT_COMPLETIONS:
        return gemini_backend(model)
    swap_class(model._completions_http_client, _completions_methods)
    return Backend("openai", CODECS[Api.OPENAI_CHAT], client_headers=True)


def _completions_methods(cls: type[CompletionsHTTPClient]) -> Mapping[str, object]:
    """Return `CompletionsHTTPClient` methods that send the attempt's request headers and record each parsed body.

    An attempt header replaces the proxy's header of the same name. The parsed body carries the completion id and cost
    `LlmResponse` drops.
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
        attempt.response_id = identifier
    if isinstance(usage := completion.get("usage"), dict):
        record_cost(attempt, usage)
