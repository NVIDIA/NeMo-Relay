# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Relay LLM Codec system.

Covers:
- AnnotatedLLMRequest construction, field access, setters, helpers
- LlmCodec subclassing with decode/encode round-trip
- codec parameter on llm.execute (pass codec instance directly)
- Module import structure
"""

from typing import cast

from nemo_relay import (
    AnnotatedLLMRequest,
    JsonObject,
    LLMRequest,
    LLMRequestInterceptOutcome,
    intercepts,
    llm,
)
from nemo_relay.codecs import (
    LlmCodec,
)

# ---------------------------------------------------------------------------
# Shared test codec
# ---------------------------------------------------------------------------


class SimpleCodec(LlmCodec):
    """Test codec that treats LLMRequest.content as an OpenAI-like payload."""

    def decode(self, request):
        content = request.content
        messages = content.get("messages", [])
        return AnnotatedLLMRequest(
            messages,
            model=content.get("model"),
            params={"temperature": content.get("temperature")} if "temperature" in content else None,
        )

    def encode(self, annotated, original):
        content = {**original.content, "messages": annotated.messages}
        if annotated.model is not None:
            content["model"] = annotated.model
        return LLMRequest(original.headers, content)


class AlternateCodec(LlmCodec):
    """Second codec for testing codec parameter selection."""

    def decode(self, request):
        content = request.content
        messages = content.get("messages", [])
        return AnnotatedLLMRequest(
            messages,
            model=content.get("model"),
            extra={"codec_used": "alternate"},
        )

    def encode(self, annotated, original):
        content = {**original.content, "messages": annotated.messages}
        if annotated.extra and "codec_used" in annotated.extra:
            content["codec_used"] = annotated.extra["codec_used"]
        return LLMRequest(original.headers, content)


def make_request():
    return LLMRequest(
        {"Authorization": "Bearer test"},
        {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4"},
    )


# ---------------------------------------------------------------------------
# 1. AnnotatedLLMRequest construction
# ---------------------------------------------------------------------------


class TestAnnotatedLLMRequestConstruction:
    def test_annotated_llm_request_construction(self):
        """Construct AnnotatedLLMRequest with messages and verify all fields."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        annotated = AnnotatedLLMRequest(messages)

        assert annotated.messages == messages
        assert annotated.model is None
        assert annotated.params is None
        assert annotated.tools is None
        assert annotated.tool_choice is None
        # extra defaults to an empty dict (not None)
        assert annotated.extra == {} or annotated.extra is None

    def test_annotated_llm_request_setter_roundtrip(self):
        """Construct, set messages to new value via setter, verify getter returns new value."""
        original_messages = [{"role": "user", "content": "first"}]
        annotated = AnnotatedLLMRequest(original_messages)
        assert annotated.messages == original_messages

        new_messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "second"},
        ]
        annotated.messages = new_messages
        assert annotated.messages == new_messages

        # Also test model setter
        assert annotated.model is None
        annotated.model = "gpt-4-turbo"
        assert annotated.model == "gpt-4-turbo"

    def test_annotated_llm_request_helpers(self):
        """Test system_prompt(), last_user_message(), has_tool_calls()."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Thanks!"},
        ]
        annotated = AnnotatedLLMRequest(messages)

        assert annotated.system_prompt() == "You are helpful."
        assert annotated.last_user_message() == "Thanks!"
        assert annotated.has_tool_calls() is False

    def test_annotated_llm_request_helpers_no_system(self):
        """system_prompt() returns None when there is no system message."""
        messages = [{"role": "user", "content": "hi"}]
        annotated = AnnotatedLLMRequest(messages)
        assert annotated.system_prompt() is None
        assert annotated.last_user_message() == "hi"

    def test_annotated_llm_request_has_tool_calls(self):
        """has_tool_calls() returns True when messages contain tool_calls."""
        messages = [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            },
        ]
        annotated = AnnotatedLLMRequest(messages)
        assert annotated.has_tool_calls() is True

    def test_annotated_llm_request_all_fields(self):
        """Construct with all optional fields populated and verify round-trip."""
        messages = [{"role": "user", "content": "hello"}]
        tools_list = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
        params = {"temperature": 0.7, "max_tokens": 100}
        extra = {"custom_field": "custom_value", "nested": {"key": "val"}}

        annotated = AnnotatedLLMRequest(
            messages,
            model="gpt-4-turbo",
            params=params,
            tools=tools_list,
            tool_choice="auto",
            extra=extra,
        )

        assert annotated.messages == messages
        assert annotated.model == "gpt-4-turbo"
        assert annotated.params == params
        assert annotated.tools == tools_list
        assert annotated.tool_choice == "auto"
        assert annotated.extra == extra


# ---------------------------------------------------------------------------
# 2. LlmCodec subclass decode/encode round-trip
# ---------------------------------------------------------------------------


class TestCodecDecodeEncode:
    def test_codec_decode_encode_roundtrip(self):
        """Decode an LLMRequest into AnnotatedLLMRequest, then encode back."""
        codec = SimpleCodec()
        request = LLMRequest(
            {"Authorization": "Bearer xyz"},
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "gpt-4",
                "temperature": 0.5,
            },
        )

        # Decode
        annotated = codec.decode(request)
        assert annotated.messages == [{"role": "user", "content": "hi"}]
        assert annotated.model == "gpt-4"
        assert annotated.params == {"temperature": 0.5}

        # Modify the annotated request
        annotated.messages = [
            *annotated.messages,
            {"role": "assistant", "content": "hello!"},
        ]

        # Encode back
        encoded = codec.encode(annotated, request)
        encoded_content = cast(JsonObject, encoded.content)
        assert encoded.headers == {"Authorization": "Bearer xyz"}
        assert cast(list[JsonObject], encoded_content["messages"]) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
        ]
        assert cast(str, encoded_content["model"]) == "gpt-4"
        # Original fields are preserved
        assert cast(float, encoded_content["temperature"]) == 0.5


# ---------------------------------------------------------------------------
# 3. LlmCodec instantiation (codec base class is still usable)
# ---------------------------------------------------------------------------


class TestCodecInstantiation:
    def test_llm_codec_subclass_creation(self):
        """LlmCodec subclass can be instantiated and has decode/encode methods."""
        codec = SimpleCodec()
        assert isinstance(codec, LlmCodec)
        assert hasattr(codec, "decode")
        assert hasattr(codec, "encode")

    def test_llm_codec_is_runtime_checkable_protocol(self):
        """LlmCodec is a runtime-checkable Protocol."""
        codec = SimpleCodec()
        assert isinstance(codec, LlmCodec)

        # An object without decode/encode does not satisfy the protocol
        assert not isinstance(object(), LlmCodec)


# ---------------------------------------------------------------------------
# 4. Pipeline integration tests (codec= param)
# ---------------------------------------------------------------------------


class TestCodecPipeline:
    async def test_pipeline_with_codec(self):
        """Full pipeline: pass codec instance directly + annotated intercept, execute LLM call."""
        codec = SimpleCodec()

        intercept_called = []

        def annotated_intercept(name, request, annotated):
            intercept_called.append(True)
            assert annotated is not None
            assert isinstance(annotated, AnnotatedLLMRequest)
            # Add a system message via the annotated request
            new_messages = [
                {"role": "system", "content": "injected by intercept"},
                *annotated.messages,
            ]
            annotated.messages = new_messages
            return LLMRequestInterceptOutcome(request, annotated)

        intercepts.register_llm_request("test-annot-intercept-pipeline", 1, False, annotated_intercept)

        try:

            def func(request):
                return {"messages": request.content.get("messages", []), "model": request.content.get("model")}

            request = make_request()
            result = await llm.execute("pipeline-llm", request, func, codec=codec)

            assert len(intercept_called) == 1
            # The encode step should have written modified messages back
            assert result["messages"][0]["role"] == "system"
            assert result["messages"][0]["content"] == "injected by intercept"
        finally:
            intercepts.deregister_llm_request("test-annot-intercept-pipeline")

    async def test_codec_parameter(self):
        """codec parameter passes the specified codec instance directly."""
        alternate = AlternateCodec()

        intercept_data = {}

        def annotated_intercept(name, request, annotated):
            if annotated is not None:
                intercept_data["extra"] = annotated.extra
            return LLMRequestInterceptOutcome(request, annotated)

        intercepts.register_llm_request("test-annot-intercept-cn", 1, False, annotated_intercept)

        try:

            def func(request):
                return {"ok": True}

            request = make_request()
            await llm.execute("cn-llm", request, func, codec=alternate)

            # The alternate codec sets extra={"codec_used": "alternate"}
            assert intercept_data.get("extra") is not None
            assert intercept_data["extra"].get("codec_used") == "alternate"
        finally:
            intercepts.deregister_llm_request("test-annot-intercept-cn")

    async def test_annotated_request_intercept_receives_typed(self):
        """Annotated intercept receives an AnnotatedLLMRequest instance when codec is active."""
        codec = SimpleCodec()

        intercept_called = []

        def annotated_intercept(name, request, annotated):
            intercept_called.append(True)
            assert annotated is not None
            assert isinstance(annotated, AnnotatedLLMRequest)
            assert isinstance(request, LLMRequest)
            return LLMRequestInterceptOutcome(request, annotated)

        intercepts.register_llm_request("test-annot-typed", 1, False, annotated_intercept)

        try:

            def func(request):
                return {"ok": True}

            await llm.execute("typed-llm", make_request(), func, codec=codec)
            assert len(intercept_called) == 1
        finally:
            intercepts.deregister_llm_request("test-annot-typed")


# ---------------------------------------------------------------------------
# 5. Module import structure
# ---------------------------------------------------------------------------


class TestCodecsModuleImport:
    def test_codecs_module_import(self):
        """Verify module import structure works correctly."""
        from nemo_relay import codecs as codecs_mod

        # LlmCodec is accessible from the module
        assert hasattr(codecs_mod, "LlmCodec")
        assert codecs_mod.LlmCodec is LlmCodec

        # AnnotatedLLMRequest at top level matches _native
        from nemo_relay import AnnotatedLLMRequest as top_level_alr
        from nemo_relay._native import AnnotatedLLMRequest as native_alr

        assert top_level_alr is native_alr

        # AnnotatedLLMRequest is also re-exported from codecs
        assert codecs_mod.AnnotatedLLMRequest is native_alr
