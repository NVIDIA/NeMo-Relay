# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM Codec protocol for bidirectional request translation.

Codecs translate opaque ``LLMRequest`` payloads into structured
``AnnotatedLLMRequest`` objects, enabling typed intercept development.

Pass a codec instance directly to ``llm.execute()`` or
``llm.stream_execute()`` via the ``codec=`` parameter.

.. note::
    This module is distinct from ``nat_nexus.typed.Codec``, which provides
    generic JSON serialization for typed tool execute wrappers. ``LlmCodec``
    here is specifically for bidirectional LLM request translation.

Classes:
    LlmCodec
        Protocol for LLM request codecs. Implement ``decode()`` and
        ``encode()`` to satisfy the protocol.
    LlmResponseCodec
        Protocol for LLM response codecs. Implement ``decode_response()``
        to satisfy the protocol.

Example::

    from nat_nexus.codecs import LlmCodec
    from nat_nexus import LLMRequest, AnnotatedLLMRequest, llm

    class MyCodec(LlmCodec):
        def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
            content = request.content
            return AnnotatedLLMRequest(
                content.get("messages", []),
                model=content.get("model"),
            )

        def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
            content = {**original.content, "messages": annotated.messages}
            if annotated.model is not None:
                content["model"] = annotated.model
            return LLMRequest(original.headers, content)

    # Pass codec instance directly to execute:
    result = await llm.execute("gpt-4", request, my_fn, codec=MyCodec())
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nat_nexus._native import AnnotatedLLMRequest, LLMRequest

if TYPE_CHECKING:
    from nat_nexus._native import AnnotatedLLMResponse


@runtime_checkable
class LlmCodec(Protocol):
    """Protocol for LLM request codecs.

    Implement ``decode()`` and ``encode()`` to provide bidirectional
    translation between opaque ``LLMRequest`` payloads and structured
    ``AnnotatedLLMRequest`` objects.

    ``decode()`` parses the opaque request content into typed fields.
    ``encode()`` merges structured changes back into the opaque request
    using merge-not-replace semantics (overlay changes, preserve unmodeled fields).
    """

    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Parse an opaque LLMRequest into a structured AnnotatedLLMRequest.

        Args:
            request: The opaque LLM request to decode.

        Returns:
            A structured AnnotatedLLMRequest with typed fields.
        """
        ...

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Merge structured changes back into the opaque request.

        Must use merge-not-replace semantics: overlay structured changes
        onto the original content, preserving unmodeled fields.

        Args:
            annotated: The structured request (potentially modified by intercepts).
            original: The pre-intercept opaque request (for preserving unmodeled fields).

        Returns:
            A new LLMRequest with the merged content.
        """
        ...


@runtime_checkable
class LlmResponseCodec(Protocol):
    """Protocol for LLM response codecs.

    Implement ``decode_response()`` to provide response introspection.
    Built-in codecs (``OpenAIChatCodec``, ``OpenAIResponsesCodec``,
    ``AnthropicMessagesCodec``) implement both ``LlmCodec`` and
    ``LlmResponseCodec``.
    """

    def decode_response(self, response: object) -> "AnnotatedLLMResponse":
        """Parse a raw JSON response into a structured AnnotatedLLMResponse.

        Args:
            response: The raw LLM response as a JSON-serializable object.

        Returns:
            A structured AnnotatedLLMResponse with normalized fields.
        """
        ...


__all__ = [
    "AnnotatedLLMRequest",
    "LlmCodec",
    "LlmResponseCodec",
]
