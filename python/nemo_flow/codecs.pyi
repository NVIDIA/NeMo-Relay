# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from nemo_flow import AnnotatedLLMRequest, AnnotatedLLMResponse, Json, LLMRequest

@runtime_checkable
class LlmCodec(Protocol):
    """Protocol for request codecs used by annotated LLM intercepts."""

    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode a raw provider request into ``AnnotatedLLMRequest``.

        Args:
            request: Provider-specific request payload to normalize.

        Returns:
            AnnotatedLLMRequest: Normalized request consumed by annotated
            intercepts.
        """
        ...

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Merge annotated edits back into the original raw request.

        Args:
            annotated: Normalized request after intercept edits.
            original: Original provider-specific request payload.

        Returns:
            LLMRequest: Provider-specific request to pass downstream.
        """
        ...

@runtime_checkable
class LlmResponseCodec(Protocol):
    """Protocol for codecs that normalize raw LLM responses."""

    def decode_response(self, response: Json) -> AnnotatedLLMResponse:
        """Decode a raw provider response into ``AnnotatedLLMResponse``.

        Args:
            response: Raw JSON-compatible response payload.

        Returns:
            AnnotatedLLMResponse: Normalized response attached to ``LLMEnd``
            events.
        """
        ...

class OpenAIChatCodec:
    """Built-in codec for OpenAI Chat Completions requests and responses."""

    def __init__(self) -> None: ...
    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode an OpenAI Chat Completions request.

        Args:
            request: Raw OpenAI Chat Completions request payload.

        Returns:
            AnnotatedLLMRequest: Normalized request representation.
        """
        ...

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Encode a normalized request back into Chat Completions format.

        Args:
            annotated: Normalized request after intercept edits.
            original: Original Chat Completions request.

        Returns:
            LLMRequest: Updated Chat Completions request payload.
        """
        ...

    def decode_response(self, response: Json) -> AnnotatedLLMResponse:
        """Decode an OpenAI Chat Completions response.

        Args:
            response: Raw Chat Completions response payload.

        Returns:
            AnnotatedLLMResponse: Normalized response representation.
        """
        ...

class OpenAIResponsesCodec:
    """Built-in codec for OpenAI Responses requests and responses."""

    def __init__(self) -> None: ...
    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode an OpenAI Responses request.

        Args:
            request: Raw OpenAI Responses request payload.

        Returns:
            AnnotatedLLMRequest: Normalized request representation.
        """
        ...

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Encode a normalized request back into Responses format.

        Args:
            annotated: Normalized request after intercept edits.
            original: Original Responses request.

        Returns:
            LLMRequest: Updated Responses request payload.
        """
        ...

    def decode_response(self, response: Json) -> AnnotatedLLMResponse:
        """Decode an OpenAI Responses response.

        Args:
            response: Raw Responses API payload.

        Returns:
            AnnotatedLLMResponse: Normalized response representation.
        """
        ...

class AnthropicMessagesCodec:
    """Built-in codec for Anthropic Messages requests and responses."""

    def __init__(self) -> None: ...
    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode an Anthropic Messages request.

        Args:
            request: Raw Anthropic Messages request payload.

        Returns:
            AnnotatedLLMRequest: Normalized request representation.
        """
        ...

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Encode a normalized request back into Messages format.

        Args:
            annotated: Normalized request after intercept edits.
            original: Original Messages request.

        Returns:
            LLMRequest: Updated Messages request payload.
        """
        ...

    def decode_response(self, response: Json) -> AnnotatedLLMResponse:
        """Decode an Anthropic Messages response.

        Args:
            response: Raw Anthropic response payload.

        Returns:
            AnnotatedLLMResponse: Normalized response representation.
        """
        ...

__all__ = [
    "AnnotatedLLMRequest",
    "AnthropicMessagesCodec",
    "LlmCodec",
    "LlmResponseCodec",
    "OpenAIChatCodec",
    "OpenAIResponsesCodec",
]
