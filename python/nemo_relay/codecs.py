# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol definitions for request and response codecs used by ``nemo_relay.llm``.

``LlmCodec`` is used for request-side translation. It lets intercepts work
against ``AnnotatedLLMRequest`` instead of provider-specific raw payloads.

``LlmResponseCodec`` is used for response-side translation. It lets NeMo Relay attach
an ``AnnotatedLLMResponse`` to emitted ``LLMEnd`` events without changing the
return value of ``llm.execute()`` or ``llm.stream_execute()``.

Example::

    from nemo_relay import AnnotatedLLMRequest, LLMRequest, llm
    from nemo_relay.codecs import LlmCodec, OpenAIChatCodec

    class DemoCodec(LlmCodec):
        def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
            return AnnotatedLLMRequest(
                request.content.get("messages", []),
                model=request.content.get("model"),
            )

        def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
            content = {**original.content, "messages": annotated.messages}
            if annotated.model is not None:
                content["model"] = annotated.model
            return LLMRequest(original.headers, content)

    async def impl(request: LLMRequest):
        return {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    # Request-side codec for intercepts, response-side codec for event annotation.
    result = await llm.execute(
        "demo-model",
        LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]}),
        impl,
        codec=DemoCodec(),
        response_codec=OpenAIChatCodec(),
    )
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nemo_relay import Json
from nemo_relay._native import (
    AnnotatedLLMRequest,
    AnthropicMessagesCodec,
    LLMRequest,
    OpenAIChatCodec,
    OpenAIResponsesCodec,
)

if TYPE_CHECKING:
    from nemo_relay._native import AnnotatedLLMResponse


@runtime_checkable
class LlmCodec(Protocol):
    """Protocol for request codecs used by annotated LLM intercepts.

    ``decode()`` converts a provider-specific ``LLMRequest`` into an
    ``AnnotatedLLMRequest`` so request intercepts can work with a normalized
    structure. ``encode()`` merges any annotated edits back into the original
    raw payload before the provider callback is invoked.

    Notes:
        ``encode()`` should preserve unknown provider-specific fields whenever
        possible instead of rebuilding the payload from scratch. That keeps
        transport-specific settings intact even when an intercept edits the
        normalized representation.

    Example::

        from nemo_relay import AnnotatedLLMRequest, LLMRequest
        from nemo_relay.codecs import LlmCodec

        class DemoCodec(LlmCodec):
            def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
                return AnnotatedLLMRequest(
                    request.content.get("messages", []),
                    model=request.content.get("model"),
                )

            def encode(
                self,
                annotated: AnnotatedLLMRequest,
                original: LLMRequest,
            ) -> LLMRequest:
                content = {**original.content, "messages": annotated.messages}
                if annotated.model is not None:
                    content["model"] = annotated.model
                return LLMRequest(original.headers, content)
    """

    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode a raw provider request into ``AnnotatedLLMRequest``.

        Args:
            request: The provider-specific request payload received by
                ``nemo_relay.llm.execute()`` or ``nemo_relay.llm.stream_execute()``.

        Returns:
            AnnotatedLLMRequest: The normalized request consumed by annotated
            intercepts.
        """
        ...

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Merge annotated edits back into the original raw request.

        Args:
            annotated: The normalized request after intercepts have applied any
                edits.
            original: The original provider-specific request passed into the
                runtime before normalization.

        Returns:
            LLMRequest: The provider-specific request that should be forwarded
            to the provider callback.
        """
        ...


@runtime_checkable
class LlmResponseCodec(Protocol):
    """Protocol for codecs that normalize raw LLM responses.

    A response codec is used only for observability. The value returned from
    ``llm.execute()`` or ``llm.stream_execute()`` stays unchanged; the decoded
    response is attached to the emitted ``LLMEnd`` event as
    ``annotated_response``.

    Example::

        import nemo_relay

        result = await nemo_relay.llm.execute(
            "demo-provider",
            nemo_relay.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
            impl,
            response_codec=nemo_relay.codecs.OpenAIChatCodec(),
        )
    """

    def decode_response(self, response: Json) -> "AnnotatedLLMResponse":
        """Decode a raw provider response into ``AnnotatedLLMResponse``.

        Args:
            response: The raw JSON-compatible value returned by the provider
                callback.

        Returns:
            AnnotatedLLMResponse: The normalized response attached to the
            ``LLMEnd`` event for downstream subscribers.
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
