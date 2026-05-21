# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for codec parameter on typed.llm_execute and typed.llm_stream_execute.

Validates that the typed API properly forwards codec= to the underlying
llm.execute() and llm.stream_execute() functions.
"""

from unittest.mock import AsyncMock, patch

from nemo_relay import AnnotatedLLMRequest, LLMRequest
from nemo_relay.codecs import LlmCodec
from nemo_relay.typed import JsonPassthrough, llm_execute, llm_stream_execute


class SimpleCodec(LlmCodec):
    """Test codec for validating codec forwarding."""

    def decode(self, request):
        content = request.content
        return AnnotatedLLMRequest(content.get("messages", []), model=content.get("model"))

    def encode(self, annotated, original):
        content = {**original.content, "messages": annotated.messages}
        if annotated.model is not None:
            content["model"] = annotated.model
        return LLMRequest(original.headers, content)


class TestLlmExecuteCodec:
    async def test_llm_execute_accepts_codec(self):
        """typed.llm_execute accepts codec kwarg without error."""
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"})

        def func(req):
            return {"ok": True}

        result = await llm_execute(
            "test-model",
            request,
            func,
            JsonPassthrough(),
            codec=SimpleCodec(),
        )
        assert result == {"ok": True}

    async def test_llm_execute_codec_none_default(self):
        """typed.llm_execute works with codec=None (default behavior)."""
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"})

        def func(req):
            return {"ok": True}

        # codec defaults to None -- should work as before
        result = await llm_execute(
            "test-model",
            request,
            func,
            JsonPassthrough(),
        )
        assert result == {"ok": True}

    async def test_llm_execute_forwards_codec(self):
        """typed.llm_execute forwards codec to llm.execute."""
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"})

        def func(req):
            return {"ok": True}

        codec_instance = SimpleCodec()

        # Patch llm.execute to capture the codec argument
        with patch("nemo_relay.typed.llm.execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"ok": True}
            await llm_execute(
                "test-model",
                request,
                func,
                JsonPassthrough(),
                codec=codec_instance,
            )
            # Verify codec was forwarded
            _, kwargs = mock_execute.call_args
            assert kwargs.get("codec") is codec_instance


class TestLlmStreamExecuteCodec:
    async def test_llm_stream_execute_accepts_codec(self):
        """typed.llm_stream_execute accepts codec kwarg without error."""
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"})
        chunks = []

        async def func(req):
            yield {"chunk": 1}

        def collector(chunk):
            chunks.append(chunk)

        def finalizer():
            return {"done": True}

        stream = await llm_stream_execute(
            "test-model",
            request,
            func,
            collector,
            finalizer,
            JsonPassthrough(),
            JsonPassthrough(),
            codec=SimpleCodec(),
        )
        assert [chunk async for chunk in stream] == [{"chunk": 1}]

    async def test_llm_stream_execute_forwards_codec(self):
        """typed.llm_stream_execute forwards codec to llm.stream_execute."""
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"})

        async def func(req):
            yield {"chunk": 1}

        def collector(chunk):
            del chunk

        def finalizer():
            return {"done": True}

        codec_instance = SimpleCodec()

        with patch("nemo_relay.typed.llm.stream_execute", new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value = AsyncMock()
            await llm_stream_execute(
                "test-model",
                request,
                func,
                collector,
                finalizer,
                JsonPassthrough(),
                JsonPassthrough(),
                codec=codec_instance,
            )
            _, kwargs = mock_stream.call_args
            assert kwargs.get("codec") is codec_instance
