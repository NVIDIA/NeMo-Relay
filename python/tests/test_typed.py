# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVMagic typed wrappers with explicit Codec protocol."""

import dataclasses

from nvmagic import intercepts, typed
from nvmagic.typed import Codec, DataclassCodec, JsonPassthrough

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SearchArgs:
    query: str
    limit: int = 10


@dataclasses.dataclass
class SearchResult:
    items: list[str]
    total: int


@dataclasses.dataclass
class DcArgs:
    x: int
    y: int = 0


@dataclasses.dataclass
class DcResult:
    value: int


@dataclasses.dataclass
class StreamChunk:
    token: str


@dataclasses.dataclass
class StreamResponse:
    chunks: list[str]


# Codec instances
search_args_codec = DataclassCodec(SearchArgs)
search_result_codec = DataclassCodec(SearchResult)
dc_args_codec = DataclassCodec(DcArgs)
dc_result_codec = DataclassCodec(DcResult)
stream_chunk_codec = DataclassCodec(StreamChunk)
stream_response_codec = DataclassCodec(StreamResponse)
passthrough = JsonPassthrough()


class PrefixCodec(Codec[str]):
    """Custom codec that wraps a plain string in an envelope dict."""

    def to_json(self, value):
        return {"text": f"pfx:{value}"}

    def from_json(self, data):
        return data["text"].removeprefix("pfx:")


class SumCodec(Codec[int]):
    """Custom codec that stores an int under a 'total' key."""

    def to_json(self, value):
        return {"total": value}

    def from_json(self, data):
        return data["total"]


prefix_codec = PrefixCodec()
sum_codec = SumCodec()


# ---------------------------------------------------------------------------
# Codec unit tests
# ---------------------------------------------------------------------------


class TestJsonPassthrough:
    def test_to_json_identity(self):
        p = JsonPassthrough()
        obj = {"a": 1}
        assert p.to_json(obj) is obj

    def test_from_json_identity(self):
        p = JsonPassthrough()
        obj = {"b": 2}
        assert p.from_json(obj) is obj

    def test_primitive_passthrough(self):
        p = JsonPassthrough()
        assert p.to_json(42) == 42
        assert p.from_json("hello") == "hello"


class TestDataclassCodec:
    def test_to_json(self):
        result = dc_args_codec.to_json(DcArgs(x=1, y=2))
        assert result == {"x": 1, "y": 2}

    def test_from_json(self):
        obj = dc_args_codec.from_json({"x": 1, "y": 2})
        assert isinstance(obj, DcArgs)
        assert obj.x == 1

    def test_roundtrip(self):
        original = DcResult(value=42)
        restored = dc_result_codec.from_json(dc_result_codec.to_json(original))
        assert restored == original


class TestCustomCodec:
    def test_custom_codec(self):
        class EnvelopeCodec(Codec[int]):
            def to_json(self, value):
                return {"value": value}

            def from_json(self, data):
                return data["value"]

        codec = EnvelopeCodec()
        assert codec.to_json(42) == {"value": 42}
        assert codec.from_json({"value": 99}) == 99


# ---------------------------------------------------------------------------
# tool_execute tests
# ---------------------------------------------------------------------------


class TestTypedToolExecute:
    async def test_dataclass_roundtrip(self):
        async def search(args: SearchArgs) -> SearchResult:
            return SearchResult(items=[args.query], total=1)

        result = await typed.tool_execute(
            "search",
            SearchArgs(query="hello", limit=5),
            search,
            search_args_codec,
            search_result_codec,
        )
        assert isinstance(result, SearchResult)
        assert result.items == ["hello"]
        assert result.total == 1

    async def test_dataclass_add(self):
        async def add(args: DcArgs) -> DcResult:
            return DcResult(value=args.x + args.y)

        result = await typed.tool_execute(
            "add",
            DcArgs(x=3, y=7),
            add,
            dc_args_codec,
            dc_result_codec,
        )
        assert isinstance(result, DcResult)
        assert result.value == 10

    async def test_passthrough(self):
        """With JsonPassthrough codecs, dicts pass through unchanged."""

        async def echo(args):
            return {"echoed": args}

        result = await typed.tool_execute(
            "echo",
            {"key": "value"},
            echo,
            passthrough,
            passthrough,
        )
        assert result == {"echoed": {"key": "value"}}

    async def test_sync_func(self):
        def double(args: SearchArgs) -> SearchResult:
            return SearchResult(items=[args.query, args.query], total=2)

        result = await typed.tool_execute(
            "sync_search",
            SearchArgs(query="hi"),
            double,
            search_args_codec,
            search_result_codec,
        )
        assert isinstance(result, SearchResult)
        assert result.total == 2

    async def test_intercepts_see_json(self):
        """Request intercepts operate on JSON dicts, not typed objects."""
        seen_args = []

        def intercept_fn(name, args):
            seen_args.append(args)
            args["limit"] = 99
            return args

        intercepts.register_tool_request("typed_req_int", 1, False, intercept_fn)

        async def search(args: SearchArgs) -> SearchResult:
            assert args.limit == 99
            return SearchResult(items=[], total=0)

        result = await typed.tool_execute(
            "intercepted_search",
            SearchArgs(query="test", limit=5),
            search,
            search_args_codec,
            search_result_codec,
        )
        assert isinstance(result, SearchResult)
        assert len(seen_args) == 1
        assert isinstance(seen_args[0], dict)

        intercepts.deregister_tool_request("typed_req_int")

    async def test_response_intercepts_see_json(self):
        """Response intercepts operate on JSON dicts, not typed objects."""

        def intercept_fn(name, result):
            result["total"] = 42
            return result

        intercepts.register_tool_response("typed_resp_int", 1, False, intercept_fn)

        async def search(args: SearchArgs) -> SearchResult:
            return SearchResult(items=["a"], total=1)

        result = await typed.tool_execute(
            "resp_intercepted",
            SearchArgs(query="x"),
            search,
            search_args_codec,
            search_result_codec,
        )
        assert isinstance(result, SearchResult)
        assert result.total == 42

        intercepts.deregister_tool_response("typed_resp_int")

    async def test_mixed_codecs(self):
        """Use different codec types for args and result."""

        async def convert(args: SearchArgs) -> DcResult:
            return DcResult(value=len(args.query))

        result = await typed.tool_execute(
            "mixed",
            SearchArgs(query="hello"),
            convert,
            search_args_codec,
            dc_result_codec,
        )
        assert isinstance(result, DcResult)
        assert result.value == 5


# ---------------------------------------------------------------------------
# llm_execute tests
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LLMResponse:
    text: str
    tokens: int


@dataclasses.dataclass
class DcLLMResponse:
    content: str


llm_response_codec = DataclassCodec(LLMResponse)
dc_llm_response_codec = DataclassCodec(DcLLMResponse)


def make_native():
    return {"messages": [], "model": "test-model"}


class TestTypedLlmExecute:
    async def test_dataclass_response(self):
        async def call_llm(native) -> LLMResponse:
            return LLMResponse(text="hello", tokens=5)

        result = await typed.llm_execute(
            "gpt-4",
            make_native(),
            call_llm,
            llm_response_codec,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "hello"
        assert result.tokens == 5

    async def test_alternate_dataclass_response(self):
        async def call_llm(native) -> DcLLMResponse:
            return DcLLMResponse(content="world")

        result = await typed.llm_execute(
            "model",
            make_native(),
            call_llm,
            dc_llm_response_codec,
        )
        assert isinstance(result, DcLLMResponse)
        assert result.content == "world"

    async def test_passthrough(self):
        """With JsonPassthrough codec, dicts pass through."""

        async def call_llm(native) -> dict:
            return {"response": "ok"}

        result = await typed.llm_execute(
            "model",
            make_native(),
            call_llm,
            passthrough,
        )
        assert result == {"response": "ok"}

    async def test_sync_func(self):
        def call_llm(native) -> LLMResponse:
            return LLMResponse(text="sync", tokens=1)

        result = await typed.llm_execute(
            "sync_model",
            make_native(),
            call_llm,
            llm_response_codec,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "sync"

    async def test_with_model_name(self):
        async def call_llm(native) -> LLMResponse:
            return LLMResponse(text="named", tokens=2)

        result = await typed.llm_execute(
            "provider",
            make_native(),
            call_llm,
            llm_response_codec,
            model_name="gpt-4-turbo",
        )
        assert isinstance(result, LLMResponse)


# ---------------------------------------------------------------------------
# llm_stream_execute tests
# ---------------------------------------------------------------------------


class TestTypedLlmStreamExecute:
    async def test_stream_passthrough(self):
        def stream_func(native):
            async def gen():
                yield {"token": "hello"}
                yield {"token": "world"}

            return gen()

        collected = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return {"chunks": collected}

        native = make_native()
        stream = await typed.llm_stream_execute(
            "stream_model",
            native,
            stream_func,
            collector,
            finalizer,
            passthrough,
            passthrough,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert len(collected) == len(chunks)

    async def test_stream_dataclass_codec(self):
        """Streaming with DataclassCodec produces typed dataclass instances."""

        def stream_func(native):
            async def gen():
                yield StreamChunk(token="hello")
                yield StreamChunk(token="world")

            return gen()

        collected: list[StreamChunk] = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return StreamResponse(chunks=[c.token for c in collected])

        native = make_native()
        stream = await typed.llm_stream_execute(
            "dc_stream",
            native,
            stream_func,
            collector,
            finalizer,
            stream_chunk_codec,
            stream_response_codec,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert len(collected) == len(chunks)
        # Collector must receive typed StreamChunk instances, not raw dicts
        for c in collected:
            assert isinstance(c, StreamChunk)
            assert isinstance(c.token, str)
        assert collected[0].token == "hello"
        assert collected[1].token == "world"

    async def test_stream_custom_codec(self):
        """Streaming with a custom Codec subclass encodes/decodes correctly."""

        def stream_func(native):
            async def gen():
                yield "alpha"
                yield "beta"

            return gen()

        collected: list[str] = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return len(collected)

        native = make_native()
        stream = await typed.llm_stream_execute(
            "custom_stream",
            native,
            stream_func,
            collector,
            finalizer,
            prefix_codec,
            sum_codec,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert len(collected) == len(chunks)
        # Verify the custom codec round-tripped: collector gets decoded strings
        assert collected[0] == "alpha"
        assert collected[1] == "beta"


# ---------------------------------------------------------------------------
# Additional sync-function tests with custom codecs
# ---------------------------------------------------------------------------


class TestTypedToolExecuteCustomCodec:
    async def test_sync_func_custom_codec(self):
        """Sync tool function with a fully custom Codec subclass."""

        def repeat(value: str) -> int:
            return len(value)

        result = await typed.tool_execute(
            "repeat_tool",
            "hello",
            repeat,
            prefix_codec,
            sum_codec,
        )
        assert isinstance(result, int)
        assert result == 5


class TestTypedLlmExecuteCustomCodec:
    async def test_sync_func_custom_codec(self):
        """Sync LLM function with a fully custom Codec subclass."""

        def call_llm(native) -> int:
            return 42

        result = await typed.llm_execute(
            "custom_llm",
            make_native(),
            call_llm,
            sum_codec,
        )
        assert isinstance(result, int)
        assert result == 42
