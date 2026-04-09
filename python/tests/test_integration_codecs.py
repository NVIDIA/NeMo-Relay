# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for OpenAI and NIM LLM Codecs.

- OpenAI Codec decode/encode with full round-trip fidelity
- NIM Codec inheritance from OpenAI with NIM-specific field handling
- LangGraph delegation model via explicit codec= parameter (direct instance passing)

The OpenAICodec and NIMCodec classes are defined inline here (same logic
as in the LangChain and LangChain-NVIDIA patch files).
"""

from typing import cast

from nemo_flow import AnnotatedLLMRequest, JsonObject, LLMRequest, ScopeType, llm, scope
from nemo_flow.codecs import (
    LlmCodec,
)

# ---------------------------------------------------------------------------
# Codec implementations (inline -- same code that goes into patches)
# ---------------------------------------------------------------------------


class OpenAICodec(LlmCodec):
    """Codec for OpenAI Chat Completions API payloads.

    Decodes flat dict payloads (messages, model, temperature, etc.) into
    AnnotatedLLMRequest. Unrecognized keys go to extra for lossless
    round-trip.
    """

    _PARAM_KEYS = {"temperature", "max_tokens", "max_completion_tokens", "top_p", "stop"}
    _MODELED_KEYS = {"messages", "model", "tools", "tool_choice"} | _PARAM_KEYS

    def decode(self, request):
        c = request.content
        messages = c.get("messages", [])
        model = c.get("model")
        params = {}
        for k in self._PARAM_KEYS:
            if k in c and c[k] is not None:
                # Normalize max_completion_tokens -> max_tokens
                key = "max_tokens" if k == "max_completion_tokens" else k
                params[key] = c[k]
        extra = {k: v for k, v in c.items() if k not in self._MODELED_KEYS}
        return AnnotatedLLMRequest(
            messages,
            model=model,
            params=params or None,
            tools=c.get("tools"),
            tool_choice=c.get("tool_choice"),
            extra=extra or None,
        )

    def encode(self, annotated, original):
        content = dict(original.content)
        content["messages"] = annotated.messages
        if annotated.model is not None:
            content["model"] = annotated.model
        if annotated.params:
            for k, v in annotated.params.items():
                # Write back to max_completion_tokens when original used that key
                if k == "max_tokens" and "max_completion_tokens" in original.content:
                    content["max_completion_tokens"] = v
                else:
                    content[k] = v
        if annotated.tools is not None:
            content["tools"] = annotated.tools
        if annotated.tool_choice is not None:
            content["tool_choice"] = annotated.tool_choice
        if annotated.extra:
            for k, v in annotated.extra.items():
                content[k] = v
        return LLMRequest(original.headers, content)


class NIMCodec(OpenAICodec):
    """Codec for NVIDIA NIM payloads.

    NIM is a strict OpenAI superset. NIM-specific fields
    (guided_json, guided_choice, guided_regex, nvext, stream_options)
    automatically go to extra via parent's decode logic.
    """

    pass


# ---------------------------------------------------------------------------
# Test fixtures -- representative payloads
# ---------------------------------------------------------------------------

SIMPLE_CHAT = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "model": "gpt-4",
    "temperature": 0.7,
    "stream": True,
}

TOOL_CALLING = {
    "messages": [
        {"role": "user", "content": "What is the weather in NYC?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                }
            ],
        },
        {"role": "tool", "content": "72F, sunny", "tool_call_id": "call_abc123"},
    ],
    "model": "gpt-4",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ],
    "tool_choice": "auto",
    "temperature": 0.0,
}

MULTIMODAL_CONTENT = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "text", "text": "Please describe the main elements."},
            ],
        }
    ],
    "model": "gpt-4-vision-preview",
    "max_tokens": 300,
}

MAX_COMPLETION_TOKENS = {
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "model": "o1-preview",
    "max_completion_tokens": 4096,
    "reasoning_effort": "high",
}

NIM_STRUCTURED_OUTPUT = {
    "messages": [{"role": "user", "content": "Extract name and age"}],
    "model": "meta/llama-3.1-70b-instruct",
    "temperature": 0.0,
    "guided_json": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
}

NIM_WITH_NVEXT = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "meta/llama-3.1-8b-instruct",
    "temperature": 0.5,
    "nvext": {"guardrails": {"enabled": True, "config_id": "safety-v1"}},
    "stream_options": {"include_usage": True},
}


# ---------------------------------------------------------------------------
# Round-trip assertion helper
# ---------------------------------------------------------------------------


def _normalize_message(msg):
    """Normalize a message dict for semantic comparison.

    The Rust AnnotatedLLMRequest type round-trips messages through serde,
    which may reorder keys and drop ``None`` values (e.g., ``content: None``
    on assistant messages with tool_calls). This helper normalizes both
    sides for semantic equivalence.
    """
    normalized = {k: v for k, v in msg.items() if v is not None}
    return normalized


def assert_round_trip(codec, payload):
    """Verify that decode(encode(x)) preserves all fields in the original payload.

    Messages are compared semantically (key order and None values are normalized).
    All other keys are compared by exact equality.
    """
    original = LLMRequest({}, cast(JsonObject, payload))
    annotated = codec.decode(original)
    result = codec.encode(annotated, original)
    for k, v in payload.items():
        if k == "messages":
            # Semantic comparison: normalize both sides
            orig_msgs = [_normalize_message(m) for m in v]
            result_msgs = [_normalize_message(m) for m in result.content[k]]
            assert result_msgs == orig_msgs, f"Messages differ:\n  orig: {orig_msgs}\n  got:  {result_msgs}"
        else:
            assert result.content[k] == v, f"Key {k!r} changed: {v!r} -> {result.content.get(k)!r}"


# ---------------------------------------------------------------------------
# Test: OpenAI Codec decode
# ---------------------------------------------------------------------------


class TestOpenAICodecDecode:
    def test_openai_decode_simple_chat(self):
        """Verify messages, model, params extracted; stream goes to extra."""
        codec = OpenAICodec()
        request = LLMRequest({}, cast(JsonObject, SIMPLE_CHAT))
        annotated = codec.decode(request)

        assert annotated.messages == SIMPLE_CHAT["messages"]
        assert annotated.model == "gpt-4"
        assert annotated.params == {"temperature": 0.7}
        assert annotated.extra is not None
        assert annotated.extra["stream"] is True

    def test_openai_decode_tool_calling(self):
        """Verify tools, tool_choice extracted; messages with tool_calls preserved as-is."""
        codec = OpenAICodec()
        request = LLMRequest({}, cast(JsonObject, TOOL_CALLING))
        annotated = codec.decode(request)

        assert annotated.tools == TOOL_CALLING["tools"]
        assert annotated.tool_choice == "auto"
        assert len(annotated.messages) == 3
        # Verify assistant message with tool_calls is preserved
        assert annotated.messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert annotated.params == {"temperature": 0.0}

    def test_openai_decode_multimodal(self):
        """Content-as-array messages decoded correctly."""
        codec = OpenAICodec()
        request = LLMRequest({}, cast(JsonObject, MULTIMODAL_CONTENT))
        annotated = codec.decode(request)

        assert len(annotated.messages) == 1
        assert isinstance(annotated.messages[0]["content"], list)
        assert annotated.messages[0]["content"][0]["type"] == "text"
        assert annotated.messages[0]["content"][0]["text"] == "What is in this image?"
        assert annotated.messages[0]["content"][1]["type"] == "text"
        assert annotated.messages[0]["content"][1]["text"] == "Please describe the main elements."
        assert annotated.params == {"max_tokens": 300}

    def test_openai_decode_max_completion_tokens(self):
        """max_completion_tokens normalized to max_tokens in params."""
        codec = OpenAICodec()
        request = LLMRequest({}, cast(JsonObject, MAX_COMPLETION_TOKENS))
        annotated = codec.decode(request)

        assert annotated.params is not None
        assert annotated.params["max_tokens"] == 4096
        assert "max_completion_tokens" not in annotated.params
        # reasoning_effort is unmodeled -> goes to extra
        assert annotated.extra is not None
        assert annotated.extra["reasoning_effort"] == "high"

    def test_openai_decode_no_params(self):
        """When no param keys present, params is None."""
        codec = OpenAICodec()
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }
        request = LLMRequest({}, cast(JsonObject, payload))
        annotated = codec.decode(request)

        assert annotated.params is None


# ---------------------------------------------------------------------------
# Test: OpenAI Codec encode
# ---------------------------------------------------------------------------


class TestOpenAICodecEncode:
    def test_openai_encode_preserves_unmodeled(self):
        """stream, response_format, reasoning_effort survive round-trip."""
        codec = OpenAICodec()
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
            "temperature": 0.5,
            "stream": True,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "medium",
        }
        original = LLMRequest({}, cast(JsonObject, payload))
        annotated = codec.decode(original)
        result = codec.encode(annotated, original)

        assert result.content["stream"] is True
        assert result.content["response_format"] == {"type": "json_object"}
        assert result.content["reasoning_effort"] == "medium"

    def test_openai_encode_max_completion_tokens_key(self):
        """When original had max_completion_tokens, encode writes back to that key."""
        codec = OpenAICodec()
        original = LLMRequest({}, cast(JsonObject, MAX_COMPLETION_TOKENS))
        annotated = codec.decode(original)
        result = codec.encode(annotated, original)

        assert result.content["max_completion_tokens"] == 4096
        # max_tokens should not be injected as a separate key
        assert "max_tokens" not in result.content or result.content.get("max_tokens") is None

    def test_openai_encode_overlay_messages(self):
        """Modified messages in annotated reflected in output."""
        codec = OpenAICodec()
        original = LLMRequest({}, cast(JsonObject, SIMPLE_CHAT))
        annotated = codec.decode(original)

        # Modify messages
        new_msg = {"role": "assistant", "content": "Hello! How can I help?"}
        annotated.messages = [*annotated.messages, new_msg]

        result = codec.encode(annotated, original)
        assert len(result.content["messages"]) == 3
        assert result.content["messages"][-1]["content"] == "Hello! How can I help?"

    def test_openai_encode_overlay_model(self):
        """Changed model name reflected in output."""
        codec = OpenAICodec()
        original = LLMRequest({}, cast(JsonObject, SIMPLE_CHAT))
        annotated = codec.decode(original)

        annotated.model = "gpt-4-turbo"
        result = codec.encode(annotated, original)
        assert result.content["model"] == "gpt-4-turbo"

    def test_openai_encode_extra_round_trip(self):
        """Extra fields from decode survive encode."""
        codec = OpenAICodec()
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
            "custom_provider_field": {"nested": True},
            "another_field": 42,
        }
        original = LLMRequest({}, cast(JsonObject, payload))
        annotated = codec.decode(original)

        assert annotated.extra is not None
        assert annotated.extra["custom_provider_field"] == {"nested": True}
        assert annotated.extra["another_field"] == 42

        result = codec.encode(annotated, original)
        assert result.content["custom_provider_field"] == {"nested": True}
        assert result.content["another_field"] == 42


# ---------------------------------------------------------------------------
# Test: NIM Codec decode
# ---------------------------------------------------------------------------


class TestNIMCodecDecode:
    def test_nim_decode_guided_json_to_extra(self):
        """guided_json goes to extra, not lost."""
        codec = NIMCodec()
        request = LLMRequest({}, cast(JsonObject, NIM_STRUCTURED_OUTPUT))
        annotated = codec.decode(request)

        assert annotated.extra is not None
        assert annotated.extra["guided_json"] == NIM_STRUCTURED_OUTPUT["guided_json"]
        assert annotated.model == "meta/llama-3.1-70b-instruct"
        assert annotated.params == {"temperature": 0.0}

    def test_nim_decode_nvext_to_extra(self):
        """nvext goes to extra."""
        codec = NIMCodec()
        request = LLMRequest({}, cast(JsonObject, NIM_WITH_NVEXT))
        annotated = codec.decode(request)

        assert annotated.extra is not None
        assert annotated.extra["nvext"] == NIM_WITH_NVEXT["nvext"]
        assert annotated.extra["stream_options"] == NIM_WITH_NVEXT["stream_options"]

    def test_nim_inherits_openai(self):
        """NIMCodec is a subclass of OpenAICodec."""
        assert issubclass(NIMCodec, OpenAICodec)
        assert issubclass(NIMCodec, LlmCodec)

        # Instance check
        codec = NIMCodec()
        assert isinstance(codec, OpenAICodec)
        assert isinstance(codec, LlmCodec)


# ---------------------------------------------------------------------------
# Test: Round-trip (decode then encode preserves original)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_roundtrip_simple_chat(self):
        """decode then encode == original content for simple chat."""
        assert_round_trip(OpenAICodec(), SIMPLE_CHAT)

    def test_roundtrip_tool_calling(self):
        """Full tool conversation round-trips."""
        assert_round_trip(OpenAICodec(), TOOL_CALLING)

    def test_roundtrip_multimodal(self):
        """Content arrays preserved."""
        assert_round_trip(OpenAICodec(), MULTIMODAL_CONTENT)

    def test_roundtrip_max_completion_tokens(self):
        """max_completion_tokens key preserved in output."""
        assert_round_trip(OpenAICodec(), MAX_COMPLETION_TOKENS)

    def test_roundtrip_nim_structured(self):
        """NIM guided_json preserved."""
        assert_round_trip(NIMCodec(), NIM_STRUCTURED_OUTPUT)

    def test_roundtrip_nim_nvext(self):
        """NIM nvext preserved."""
        assert_round_trip(NIMCodec(), NIM_WITH_NVEXT)


# ---------------------------------------------------------------------------
# Test: LangGraph delegation model
# ---------------------------------------------------------------------------


class TestLangGraphDelegation:
    async def test_explicit_codec_delegates_to_provider(self):
        """Pass OpenAI codec instance directly, verify it resolves.

        LangGraph does not need its own Codec -- it passes an explicit
        codec= instance to each LLM call that delegates to the provider's Codec.
        """
        codec = OpenAICodec()

        intercept_data = {}

        from nemo_flow import intercepts

        def annotated_intercept(name, request, annotated):
            if annotated is not None:
                intercept_data["model"] = annotated.model
                intercept_data["messages"] = annotated.messages
            return (request, annotated)

        intercepts.register_llm_request("delegation-test-intercept", 1, False, annotated_intercept)

        try:

            def func(request):
                return {"ok": True}

            request = LLMRequest(
                {},
                {
                    "messages": [{"role": "user", "content": "test delegation"}],
                    "model": "gpt-4",
                    "temperature": 0.5,
                },
            )

            # Pass codec instance directly per LLM call
            with scope.scope("langgraph-scope", ScopeType.Agent) as _handle:
                await llm.execute("delegation-llm", request, func, codec=codec)

            # Verify the annotated intercept received decoded data
            assert intercept_data.get("model") == "gpt-4"
            assert intercept_data.get("messages") == [{"role": "user", "content": "test delegation"}]
        finally:
            intercepts.deregister_llm_request("delegation-test-intercept")
