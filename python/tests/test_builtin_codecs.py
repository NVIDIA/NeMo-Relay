# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for built-in codec Python classes and LlmResponseCodec protocol.

Covers:
- Built-in codec construction (OpenAIChatCodec, OpenAIResponsesCodec, AnthropicMessagesCodec)
- Built-in codec decode/encode/decode_response methods
- LlmResponseCodec protocol
- response_codec parameter accepts object (not string)
"""

from typing import cast

import nemo_flow
from nemo_flow import (
    AnnotatedLLMRequest,
    AnnotatedLLMResponse,
    JsonObject,
    LLMRequest,
    llm,
    subscribers,
)
from nemo_flow.codecs import AnthropicMessagesCodec, OpenAIChatCodec, OpenAIResponsesCodec

# ---------------------------------------------------------------------------
# 1. Built-in codec construction
# ---------------------------------------------------------------------------


class TestBuiltinCodecConstruction:
    def test_openai_chat_codec_constructable(self):
        """OpenAIChatCodec() is constructable."""
        codec = OpenAIChatCodec()
        assert codec is not None

    def test_openai_responses_codec_constructable(self):
        """OpenAIResponsesCodec() is constructable."""
        codec = OpenAIResponsesCodec()
        assert codec is not None

    def test_anthropic_messages_codec_constructable(self):
        """AnthropicMessagesCodec() is constructable."""
        codec = AnthropicMessagesCodec()
        assert codec is not None

    def test_openai_chat_codec_has_methods(self):
        """OpenAIChatCodec has decode, encode, decode_response methods."""
        codec = OpenAIChatCodec()
        assert hasattr(codec, "decode")
        assert hasattr(codec, "encode")
        assert hasattr(codec, "decode_response")

    def test_openai_responses_codec_has_methods(self):
        """OpenAIResponsesCodec has decode, encode, decode_response methods."""
        codec = OpenAIResponsesCodec()
        assert hasattr(codec, "decode")
        assert hasattr(codec, "encode")
        assert hasattr(codec, "decode_response")

    def test_anthropic_messages_codec_has_methods(self):
        """AnthropicMessagesCodec has decode, encode, decode_response methods."""
        codec = AnthropicMessagesCodec()
        assert hasattr(codec, "decode")
        assert hasattr(codec, "encode")
        assert hasattr(codec, "decode_response")


# ---------------------------------------------------------------------------
# 2. Built-in codec decode/encode round-trip
# ---------------------------------------------------------------------------


class TestBuiltinCodecDecodeEncode:
    def test_openai_chat_decode(self):
        """OpenAIChatCodec.decode() returns AnnotatedLLMRequest."""
        codec = OpenAIChatCodec()
        request = LLMRequest(
            {},
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
            },
        )
        annotated = codec.decode(request)
        assert isinstance(annotated, AnnotatedLLMRequest)
        assert annotated.model == "gpt-4"
        assert annotated.messages == [{"role": "user", "content": "hi"}]

    def test_openai_chat_encode(self):
        """OpenAIChatCodec.encode() returns LLMRequest preserving unmodeled fields."""
        codec = OpenAIChatCodec()
        original = LLMRequest(
            {"Authorization": "Bearer test"},
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
            },
        )
        annotated = codec.decode(original)
        # Modify the annotated request
        annotated.messages = [
            *annotated.messages,
            {"role": "assistant", "content": "hello"},
        ]
        encoded = codec.encode(annotated, original)
        encoded_content = cast(JsonObject, encoded.content)
        assert isinstance(encoded, LLMRequest)
        assert encoded.headers == {"Authorization": "Bearer test"}
        assert cast(float, encoded_content["temperature"]) == 0.7
        assert len(cast(list[JsonObject], encoded_content["messages"])) == 2


# ---------------------------------------------------------------------------
# 3. Built-in codec decode_response
# ---------------------------------------------------------------------------


class TestBuiltinCodecDecodeResponse:
    def test_openai_chat_decode_response(self):
        """OpenAIChatCodec.decode_response() returns AnnotatedLLMResponse."""
        codec = OpenAIChatCodec()
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        annotated = codec.decode_response(response)
        assert isinstance(annotated, AnnotatedLLMResponse)
        assert annotated.model == "gpt-4"
        assert annotated.response_text() == "Hello!"
        assert annotated.has_tool_calls() is False

    def test_anthropic_messages_decode_response(self):
        """AnthropicMessagesCodec.decode_response() returns AnnotatedLLMResponse."""
        codec = AnthropicMessagesCodec()
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        annotated = codec.decode_response(response)
        assert isinstance(annotated, AnnotatedLLMResponse)
        assert annotated.model == "claude-3-sonnet-20240229"
        assert annotated.response_text() == "Hello!"


# ---------------------------------------------------------------------------
# 4. LlmResponseCodec protocol
# ---------------------------------------------------------------------------


class TestLlmResponseCodecProtocol:
    def test_protocol_importable(self):
        """LlmResponseCodec protocol is importable from codecs module."""
        from nemo_flow.codecs import LlmResponseCodec

        assert LlmResponseCodec is not None

    def test_builtin_codecs_satisfy_protocol(self):
        """Built-in codecs satisfy LlmResponseCodec protocol."""
        from nemo_flow.codecs import LlmResponseCodec

        assert isinstance(OpenAIChatCodec(), LlmResponseCodec)
        assert isinstance(OpenAIResponsesCodec(), LlmResponseCodec)
        assert isinstance(AnthropicMessagesCodec(), LlmResponseCodec)


# ---------------------------------------------------------------------------
# 5. response_codec parameter accepts object
# ---------------------------------------------------------------------------


class TestResponseCodecObjectParam:
    async def test_response_codec_accepts_builtin_object(self):
        """response_codec= accepts a built-in codec object, not a string."""
        captured_events = []

        def capture(event):
            captured_events.append(event)

        subscribers.register("test-builtin-codec-obj", capture)

        try:
            codec = OpenAIChatCodec()
            request = LLMRequest(
                {},
                {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

            # Mock LLM function that returns an OpenAI-like response
            async def mock_llm(req):
                return {
                    "id": "chatcmpl-test",
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }

            await llm.execute(
                "gpt-4",
                request,
                mock_llm,
                response_codec=codec,
            )

            # Find LLMEnd event
            end_events = [e for e in captured_events if e.kind == "LLMEnd"]
            assert len(end_events) == 1

            annotated = end_events[0].annotated_response
            assert annotated is not None, "annotated_response should be populated"
            assert isinstance(annotated, AnnotatedLLMResponse)
            assert annotated.response_text() == "Hello!"
            assert annotated.model == "gpt-4"

        finally:
            subscribers.deregister("test-builtin-codec-obj")

    async def test_response_codec_none_gives_no_annotation(self):
        """response_codec=None still works (backward compat)."""
        captured_events = []

        def capture(event):
            captured_events.append(event)

        subscribers.register("test-no-codec-obj", capture)

        try:
            request = LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]})

            async def mock_llm(req):
                return {"result": "ok"}

            await llm.execute("test-llm", request, mock_llm)

            end_events = [e for e in captured_events if e.kind == "LLMEnd"]
            assert len(end_events) == 1
            assert end_events[0].annotated_response is None

        finally:
            subscribers.deregister("test-no-codec-obj")


# ---------------------------------------------------------------------------
# 6. BUILTIN_CODECS removed from codecs module
# ---------------------------------------------------------------------------


class TestBuiltinCodecsTupleRemoved:
    def test_no_builtin_codecs_tuple(self):
        """BUILTIN_CODECS tuple is no longer in codecs module."""
        from nemo_flow import codecs as codecs_mod

        assert not hasattr(codecs_mod, "BUILTIN_CODECS")


# ---------------------------------------------------------------------------
# 7. Module imports
# ---------------------------------------------------------------------------


class TestBuiltinCodecImports:
    def test_importable_from_codecs_module(self):
        """Built-in codecs are importable from nemo_flow.codecs."""
        from nemo_flow.codecs import AnthropicMessagesCodec, OpenAIChatCodec, OpenAIResponsesCodec

        assert OpenAIChatCodec is not None
        assert OpenAIResponsesCodec is not None
        assert AnthropicMessagesCodec is not None

    def test_not_reexported_from_top_level(self):
        """Built-in codecs are not re-exported from nemo_flow."""
        assert not hasattr(nemo_flow, "OpenAIChatCodec")
        assert not hasattr(nemo_flow, "OpenAIResponsesCodec")
        assert not hasattr(nemo_flow, "AnthropicMessagesCodec")
