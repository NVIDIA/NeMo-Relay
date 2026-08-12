# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for built-in codec Python classes and LlmResponseCodec protocol.

Covers:
- Built-in codec construction for OpenAIChatCodec, OpenAIResponsesCodec,
  AnthropicMessagesCodec, OCIGenAIChatCodec, and GeminiGenerateContentCodec
- Built-in codec decode/encode/decode_response methods for all five providers
- LlmResponseCodec protocol
- response_codec parameter accepts object (not string)
"""

from typing import cast

import nemo_relay
from nemo_relay import (
    AnnotatedLLMRequest,
    AnnotatedLLMResponse,
    JsonObject,
    LLMRequest,
    guardrails,
    llm,
    subscribers,
)
from nemo_relay.codecs import (
    AnthropicMessagesCodec,
    GeminiGenerateContentCodec,
    OCIGenAIChatCodec,
    OpenAIChatCodec,
    OpenAIResponsesCodec,
)

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

    def test_oci_genai_chat_codec_constructable(self):
        """OCIGenAIChatCodec() is constructable."""
        codec = OCIGenAIChatCodec()
        assert codec is not None

    def test_oci_genai_chat_codec_has_methods(self):
        """OCIGenAIChatCodec has decode, encode, decode_response methods."""
        codec = OCIGenAIChatCodec()
        assert hasattr(codec, "decode")
        assert hasattr(codec, "encode")
        assert hasattr(codec, "decode_response")

    def test_gemini_codec_constructable(self):
        """GeminiGenerateContentCodec() is constructable."""
        codec = GeminiGenerateContentCodec()
        assert codec is not None

    def test_gemini_codec_has_methods(self):
        """GeminiGenerateContentCodec has decode, encode, decode_response methods."""
        codec = GeminiGenerateContentCodec()
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

    def test_anthropic_issue_501_roundtrip_and_annotated_edit(self):
        """Anthropic cache blocks survive unchanged and edited Python annotations."""
        codec = AnthropicMessagesCodec()
        original = LLMRequest(
            {},
            {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 128,
                "system": [
                    {
                        "type": "text",
                        "text": "Keep this cached.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "lookup",
                                "input": {"q": "x"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "done",
                            }
                        ],
                    },
                ],
                "future_field": None,
            },
        )

        annotated = codec.decode(original)
        assert annotated.instructions == [
            {
                "type": "text",
                "text": "Keep this cached.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert codec.encode(annotated, original).content == original.content

        instructions = cast(list[JsonObject], annotated.instructions)
        instructions[0]["text"] = "Edited safely."
        annotated.instructions = instructions
        encoded = codec.encode(annotated, original)
        expected = dict(cast(JsonObject, original.content))
        expected["system"] = [
            {
                "type": "text",
                "text": "Edited safely.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert encoded.content == expected


# ---------------------------------------------------------------------------
# 3. Built-in codec decode_response
# ---------------------------------------------------------------------------


class TestBuiltinCodecDecodeResponse:
    def test_annotated_response_constructable_for_custom_codecs(self):
        """AnnotatedLLMResponse() lets Python response codecs return normalized responses."""
        annotated = AnnotatedLLMResponse(
            id="langchain-response-1",
            model="mock-model",
            message="I will search docs.",
            tool_calls=[
                {
                    "id": "call-search-docs",
                    "name": "search_docs",
                    "arguments": {"query": "Deep Agents"},
                }
            ],
            finish_reason="tool_use",
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            api_specific={"api": "custom", "api_name": "provider", "data": {"id": "raw"}},
            extra={"framework": "langchain"},
        )

        assert annotated.id == "langchain-response-1"
        assert annotated.model == "mock-model"
        assert annotated.response_text() == "I will search docs."
        assert annotated.tool_calls == [
            {
                "id": "call-search-docs",
                "name": "search_docs",
                "arguments": {"query": "Deep Agents"},
            }
        ]
        assert annotated.finish_reason == "tool_use"
        assert annotated.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        assert annotated.api_specific == {"api": "custom", "api_name": "provider", "data": {"id": "raw"}}
        assert annotated.extra == {"framework": "langchain"}

    def test_annotated_response_exposes_unknown_finish_reason(self):
        """Unknown native finish reasons are still visible to Python callers."""
        annotated = AnnotatedLLMResponse(
            message="done",
            finish_reason="provider_custom_stop",
        )
        annotated_from_native_shape = AnnotatedLLMResponse(
            message="done",
            finish_reason={"unknown": "provider_custom_stop"},
        )

        assert annotated.finish_reason == "provider_custom_stop"
        assert annotated_from_native_shape.finish_reason == "provider_custom_stop"

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

    def test_oci_genai_request_decode_encode_round_trip(self):
        """OCIGenAIChatCodec decodes and re-encodes an OCI ChatDetails request."""
        codec = OCIGenAIChatCodec()
        original = LLMRequest(
            {},
            {
                "compartmentId": "ocid1.compartment.oc1..example",
                "servingMode": {"servingType": "ON_DEMAND", "modelId": "meta.llama-3.3-70b-instruct"},
                "chatRequest": {
                    "apiFormat": "GENERIC",
                    "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "My SSN is 111-22-3333."}]}],
                    "maxTokens": 600,
                },
            },
        )
        annotated = codec.decode(original)
        assert isinstance(annotated, AnnotatedLLMRequest)
        assert annotated.model == "meta.llama-3.3-70b-instruct"

        # Identity: an unedited annotation re-encodes byte-identically.
        identical = codec.encode(annotated, original)
        assert identical.content == original.content

        annotated.messages = [
            {"role": "user", "content": "My SSN is [REDACTED]."},
        ]
        encoded = codec.encode(annotated, original)
        encoded_content = cast(JsonObject, encoded.content)
        chat_request = cast(JsonObject, encoded_content["chatRequest"])
        messages = cast(list[JsonObject], chat_request["messages"])
        assert messages[0]["content"] == [{"type": "TEXT", "text": "My SSN is [REDACTED]."}]
        assert cast(int, chat_request["maxTokens"]) == 600

    def test_oci_genai_decode_response(self):
        """OCIGenAIChatCodec.decode_response() returns AnnotatedLLMResponse."""
        codec = OCIGenAIChatCodec()
        response = {
            "modelId": "meta.llama-3.3-70b-instruct",
            "chatResponse": {
                "apiFormat": "GENERIC",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "Hello!"}],
                        },
                        "finishReason": "stop",
                    }
                ],
                "usage": {"promptTokens": 10, "completionTokens": 5, "totalTokens": 15},
            },
        }
        annotated = codec.decode_response(response)
        assert isinstance(annotated, AnnotatedLLMResponse)
        assert annotated.model == "meta.llama-3.3-70b-instruct"
        assert annotated.response_text() == "Hello!"
        assert annotated.finish_reason == "complete"

    def test_gemini_codec_decode(self):
        """GeminiGenerateContentCodec.decode() returns AnnotatedLLMRequest with messages and params."""
        codec = GeminiGenerateContentCodec()
        request = LLMRequest(
            {},
            {
                "model": "gemini-2.0-flash",
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "generationConfig": {"temperature": 0.5, "maxOutputTokens": 256},
                "systemInstruction": {"parts": [{"text": "Be concise."}]},
            },
        )
        annotated = codec.decode(request)
        assert isinstance(annotated, AnnotatedLLMRequest)
        assert annotated.model == "gemini-2.0-flash"
        # System message comes first, then the user message.
        assert len(annotated.messages) == 2
        assert annotated.messages[0]["role"] == "system"
        assert annotated.messages[1]["role"] == "user"
        assert annotated.params is not None

    def test_gemini_codec_encode_round_trip(self):
        """GeminiGenerateContentCodec.encode(decode(req), req) is idempotent when nothing changes."""
        codec = GeminiGenerateContentCodec()
        original = LLMRequest(
            {},
            {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "generationConfig": {"temperature": 0.7},
                "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
            },
        )
        annotated = codec.decode(original)
        re_encoded = codec.encode(annotated, original)
        assert re_encoded.content == original.content, (
            "encode(decode(req), req) must be idempotent when nothing changes"
        )

    def test_gemini_codec_decode_response_text(self):
        """GeminiGenerateContentCodec.decode_response() extracts text and usage from a generateContent response."""
        codec = GeminiGenerateContentCodec()
        response = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "Hello from Gemini!"}]},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4,
                "totalTokenCount": 12,
            },
            "modelVersion": "gemini-2.0-flash",
        }
        annotated = codec.decode_response(response)
        assert isinstance(annotated, AnnotatedLLMResponse)
        assert annotated.response_text() == "Hello from Gemini!"
        assert annotated.finish_reason == "complete"
        assert annotated.model == "gemini-2.0-flash"
        assert annotated.usage is not None
        assert annotated.usage["prompt_tokens"] == 8

    def test_gemini_codec_decode_response_safety_finish_reason(self):
        """GeminiGenerateContentCodec maps SAFETY finish reason to 'content_filter', not 'unknown'."""
        codec = GeminiGenerateContentCodec()
        response = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "SAFETY",
                    "index": 0,
                }
            ],
            "usageMetadata": {"promptTokenCount": 5},
        }
        annotated = codec.decode_response(response)
        assert annotated.finish_reason == "content_filter", (
            "SAFETY finish reason must map to content_filter, not unknown"
        )

    def test_gemini_codec_decode_response_function_call(self):
        """GeminiGenerateContentCodec.decode_response() extracts functionCall parts as tool_calls."""
        codec = GeminiGenerateContentCodec()
        response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "id": "call_abc",
                                    "args": {"location": "NYC"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10},
        }
        annotated = codec.decode_response(response)
        assert annotated.has_tool_calls() is True
        tool_calls = annotated.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_abc"
        assert tool_calls[0]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# 4. LlmResponseCodec protocol
# ---------------------------------------------------------------------------


class TestLlmResponseCodecProtocol:
    def test_protocol_importable(self):
        """LlmResponseCodec protocol is importable from codecs module."""
        from nemo_relay.codecs import LlmResponseCodec

        assert LlmResponseCodec.__name__ == "LlmResponseCodec"

    def test_builtin_codecs_satisfy_protocol(self):
        """Built-in codecs satisfy LlmResponseCodec protocol."""
        from nemo_relay.codecs import LlmResponseCodec

        assert isinstance(OpenAIChatCodec(), LlmResponseCodec)
        assert isinstance(OpenAIResponsesCodec(), LlmResponseCodec)
        assert isinstance(AnthropicMessagesCodec(), LlmResponseCodec)
        assert isinstance(OCIGenAIChatCodec(), LlmResponseCodec)

        assert isinstance(GeminiGenerateContentCodec(), LlmResponseCodec)


# ---------------------------------------------------------------------------
# 5. response_codec parameter accepts object
# ---------------------------------------------------------------------------


class TestResponseCodecObjectParam:
    def test_manual_call_end_response_codec_attaches_annotation(self):
        """manual llm.call_end() accepts response_codec for end-event annotations."""
        captured_events = []

        def capture(event):
            captured_events.append(event)

        subscribers.register("test-manual-call-end-response-codec", capture)

        try:
            handle = llm.call(
                "manual-codec-llm",
                LLMRequest(
                    {},
                    {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                ),
            )
            llm.call_end(
                handle,
                {
                    "id": "chatcmpl-manual",
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
                },
                response_codec=OpenAIChatCodec(),
            )
            subscribers.flush()

            end_events = [
                e for e in captured_events if e.kind == "scope" and e.category == "llm" and e.scope_category == "end"
            ]
            assert len(end_events) == 1

            annotated = end_events[0].annotated_response
            assert annotated is not None
            assert annotated.usage == {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
            assert annotated.response_text() == "Hello!"

        finally:
            subscribers.deregister("test-manual-call-end-response-codec")

    def test_manual_call_end_accepts_annotated_response_mapping(self):
        """manual llm.call_end() accepts an explicit JSON annotation mapping."""
        captured_events = []

        def capture(event):
            captured_events.append(event)

        subscribers.register("test-manual-call-end-annotated-response", capture)

        try:
            handle = llm.call(
                "manual-annotated-llm",
                LLMRequest({}, {"model": "gpt-4", "messages": []}),
            )
            llm.call_end(
                handle,
                {"status": "ok"},
                annotated_response={
                    "model": "gpt-4",
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
            subscribers.flush()

            end_events = [
                e for e in captured_events if e.kind == "scope" and e.category == "llm" and e.scope_category == "end"
            ]
            assert len(end_events) == 1

            annotated = end_events[0].annotated_response
            assert annotated is not None
            assert annotated.model == "gpt-4"
            assert annotated.usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

        finally:
            subscribers.deregister("test-manual-call-end-annotated-response")

    def test_manual_call_end_response_codec_uses_sanitized_payload(self):
        """manual llm.call_end() decodes response annotations from sanitized event data."""
        captured_events = []

        def capture(event):
            captured_events.append(event)

        def sanitize_response(response, context):
            del context
            return {
                "id": "chatcmpl-sanitized",
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Sanitized"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

        guardrails.register_llm_sanitize_response("test-call-end-codec-sanitizer", 1, sanitize_response)
        subscribers.register("test-manual-call-end-sanitized-response-codec", capture)

        try:
            handle = llm.call(
                "manual-codec-sanitized-llm",
                LLMRequest({}, {"model": "gpt-4", "messages": []}),
            )
            llm.call_end(handle, "raw response", response_codec=OpenAIChatCodec())
            subscribers.flush()

            end_events = [
                e for e in captured_events if e.kind == "scope" and e.category == "llm" and e.scope_category == "end"
            ]
            assert len(end_events) == 1
            assert end_events[0].data["id"] == "chatcmpl-sanitized"

            annotated = end_events[0].annotated_response
            assert annotated is not None
            assert annotated.response_text() == "Sanitized"

        finally:
            subscribers.deregister("test-manual-call-end-sanitized-response-codec")
            guardrails.deregister_llm_sanitize_response("test-call-end-codec-sanitizer")

    def test_manual_call_end_response_codec_failure_defers_without_raising(self):
        """manual llm.call_end() records deferred response codec failures without blocking."""
        captured_events = []

        def capture(event):
            captured_events.append(event)

        subscribers.register("test-manual-call-end-response-codec-error", capture)

        try:
            handle = llm.call(
                "manual-codec-error-llm",
                LLMRequest({}, {"model": "gpt-4", "messages": []}),
            )
            llm.call_end(handle, "malformed response", response_codec=OpenAIChatCodec())
            subscribers.flush()

            end_events = [
                e for e in captured_events if e.kind == "scope" and e.category == "llm" and e.scope_category == "end"
            ]
            assert len(end_events) == 1
            assert end_events[0].annotated_response is None

        finally:
            subscribers.deregister("test-manual-call-end-response-codec-error")

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
            await subscribers.flush_async()

            # Find LLMEnd event
            end_events = [
                e for e in captured_events if e.kind == "scope" and e.category == "llm" and e.scope_category == "end"
            ]
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
            await subscribers.flush_async()

            end_events = [
                e for e in captured_events if e.kind == "scope" and e.category == "llm" and e.scope_category == "end"
            ]
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
        from nemo_relay import codecs as codecs_mod

        assert not hasattr(codecs_mod, "BUILTIN_CODECS")


# ---------------------------------------------------------------------------
# 7. Module imports
# ---------------------------------------------------------------------------


class TestBuiltinCodecImports:
    def test_importable_from_codecs_module(self):
        """Built-in codecs are importable from nemo_relay.codecs."""
        from nemo_relay.codecs import (
            AnthropicMessagesCodec,
            GeminiGenerateContentCodec,
            OpenAIChatCodec,
            OpenAIResponsesCodec,
        )

        assert OpenAIChatCodec is not None
        assert OpenAIResponsesCodec is not None
        assert AnthropicMessagesCodec is not None
        assert GeminiGenerateContentCodec is not None

    def test_not_reexported_from_top_level(self):
        """Built-in codecs are not re-exported from nemo_relay."""
        assert not hasattr(nemo_relay, "OpenAIChatCodec")
        assert not hasattr(nemo_relay, "OpenAIResponsesCodec")
        assert not hasattr(nemo_relay, "AnthropicMessagesCodec")
        assert not hasattr(nemo_relay, "GeminiGenerateContentCodec")
