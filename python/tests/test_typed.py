# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Agent Toolkit Nexus typed wrappers with explicit Codec protocol."""

import dataclasses
from typing import cast

import pytest
from nat_nexus import JsonObject, LLMRequest, intercepts, typed
from nat_nexus.typed import BestEffortAnyCodec, Codec, DataclassCodec, JsonPassthrough

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


class BrokenValidatedModel:
    @classmethod
    def model_validate(cls, data):
        raise ValueError("broken validation")


class FaultyDumpValue:
    def model_dump(self, mode=None):
        raise RuntimeError("broken dump")


class UnpickleableValue:
    def __reduce_ex__(self, protocol):
        raise TypeError("cannot pickle")

    def __str__(self):
        return "unpickleable"


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

    def test_base_codec_methods_raise(self):
        codec = Codec()
        with pytest.raises(NotImplementedError):
            codec.to_json("value")
        with pytest.raises(NotImplementedError):
            codec.from_json("value")


class TestPydanticCodec:
    def test_direct_roundtrip(self):
        import pydantic

        class Point(pydantic.BaseModel):
            x: int
            y: int

        codec = typed.PydanticCodec(Point)
        encoded = codec.to_json(Point(x=2, y=3))
        restored = codec.from_json({"x": 5, "y": 8})

        assert encoded == {"x": 2, "y": 3}
        assert restored == Point(x=5, y=8)


class TestTypedHelpers:
    def test_register_and_resolve_runtime_type(self):
        token = typed._register_runtime_type(SearchArgs)
        assert typed._resolve_runtime_type(token) is SearchArgs

    def test_resolve_runtime_type_non_string(self):
        assert typed._resolve_runtime_type(123) is None

    def test_resolve_importable_type_success(self):
        path = f"{SearchArgs.__module__}.{SearchArgs.__qualname__}"
        assert typed._resolve_importable_type(path) is SearchArgs

    def test_resolve_importable_type_invalid_inputs(self):
        assert typed._resolve_importable_type("") is None
        assert typed._resolve_importable_type(123) is None
        assert typed._resolve_importable_type("does.not.exist.Type") is None

    def test_resolve_importable_type_skips_local_classes(self):
        class LocalType:
            pass

        path = f"{LocalType.__module__}.{LocalType.__qualname__}"
        assert typed._resolve_importable_type(path) is None


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


def make_request():
    return LLMRequest({}, {"messages": [], "model": "test-model"})


class TestTypedLlmExecute:
    async def test_dataclass_response(self):
        async def call_llm(request) -> LLMResponse:
            return LLMResponse(text="hello", tokens=5)

        result = await typed.llm_execute(
            "gpt-4",
            make_request(),
            call_llm,
            llm_response_codec,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "hello"
        assert result.tokens == 5

    async def test_alternate_dataclass_response(self):
        async def call_llm(request) -> DcLLMResponse:
            return DcLLMResponse(content="world")

        result = await typed.llm_execute(
            "model",
            make_request(),
            call_llm,
            dc_llm_response_codec,
        )
        assert isinstance(result, DcLLMResponse)
        assert result.content == "world"

    async def test_passthrough(self):
        """With JsonPassthrough codec, dicts pass through."""

        async def call_llm(request) -> dict:
            return {"response": "ok"}

        result = await typed.llm_execute(
            "model",
            make_request(),
            call_llm,
            passthrough,
        )
        assert result == {"response": "ok"}

    async def test_sync_func(self):
        def call_llm(request) -> LLMResponse:
            return LLMResponse(text="sync", tokens=1)

        result = await typed.llm_execute(
            "sync_model",
            make_request(),
            call_llm,
            llm_response_codec,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "sync"

    async def test_with_model_name(self):
        async def call_llm(request) -> LLMResponse:
            return LLMResponse(text="named", tokens=2)

        result = await typed.llm_execute(
            "provider",
            make_request(),
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
        def stream_func(request):
            async def gen():
                yield {"token": "hello"}
                yield {"token": "world"}

            return gen()

        collected = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return {"chunks": collected}

        request = make_request()
        stream = await typed.llm_stream_execute(
            "stream_model",
            request,
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

        def stream_func(request):
            async def gen():
                yield StreamChunk(token="hello")
                yield StreamChunk(token="world")

            return gen()

        collected: list[StreamChunk] = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return StreamResponse(chunks=[c.token for c in collected])

        request = make_request()
        stream = await typed.llm_stream_execute(
            "dc_stream",
            request,
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

        def stream_func(request):
            async def gen():
                yield "alpha"
                yield "beta"

            return gen()

        collected: list[str] = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return len(collected)

        request = make_request()
        stream = await typed.llm_stream_execute(
            "custom_stream",
            request,
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

    async def test_stream_wrapper_closures_are_executed(self, monkeypatch):
        collected: list[StreamChunk] = []

        async def fake_stream_execute(name, request, func, collector, finalizer, **kwargs):
            json_chunks = []
            async for chunk in func(request):
                json_chunks.append(chunk)
                collector(chunk)
            return {"chunks": json_chunks, "final": finalizer(), "kwargs": kwargs}

        async def stream_func(request):
            yield StreamChunk(token="hello")
            yield StreamChunk(token="world")

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return StreamResponse(chunks=[chunk.token for chunk in collected])

        monkeypatch.setattr(typed.llm, "stream_execute", fake_stream_execute)

        result = await typed.llm_stream_execute(
            "wrapped_stream",
            make_request(),
            stream_func,
            collector,
            finalizer,
            stream_chunk_codec,
            stream_response_codec,
        )

        assert result["chunks"] == [{"token": "hello"}, {"token": "world"}]
        assert result["final"] == {"chunks": ["hello", "world"]}
        assert [chunk.token for chunk in collected] == ["hello", "world"]


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

        def call_llm(request) -> int:
            return 42

        result = await typed.llm_execute(
            "custom_llm",
            make_request(),
            call_llm,
            sum_codec,
        )
        assert isinstance(result, int)
        assert result == 42


# ---------------------------------------------------------------------------
# BestEffortAnyCodec tests
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BEPoint:
    x: int
    y: int


class TestBestEffortAnyCodec:
    """Tests for BestEffortAnyCodec round-trip and from_json edge cases."""

    def setup_method(self):
        self.codec = BestEffortAnyCodec()

    # -- Round-trip: JSON-native types --

    def test_roundtrip_int(self):
        assert self.codec.from_json(self.codec.to_json(42)) == 42

    def test_roundtrip_zero(self):
        assert self.codec.from_json(self.codec.to_json(0)) == 0

    def test_roundtrip_float(self):
        assert self.codec.from_json(self.codec.to_json(3.14)) == 3.14

    def test_roundtrip_string(self):
        assert self.codec.from_json(self.codec.to_json("hello")) == "hello"

    def test_roundtrip_empty_string(self):
        assert self.codec.from_json(self.codec.to_json("")) == ""

    def test_roundtrip_string_containing_data(self):
        """String containing 'data' should not trigger tag dispatch."""
        assert self.codec.from_json(self.codec.to_json("some data here")) == "some data here"

    def test_roundtrip_bool_true(self):
        assert self.codec.from_json(self.codec.to_json(True)) is True

    def test_roundtrip_bool_false(self):
        assert self.codec.from_json(self.codec.to_json(False)) is False

    def test_roundtrip_none(self):
        assert self.codec.from_json(self.codec.to_json(None)) is None

    def test_roundtrip_list(self):
        assert self.codec.from_json(self.codec.to_json([1, 2, 3])) == [1, 2, 3]

    def test_roundtrip_empty_list(self):
        assert self.codec.from_json(self.codec.to_json([])) == []

    def test_roundtrip_dict(self):
        assert self.codec.from_json(self.codec.to_json({"a": 1})) == {"a": 1}

    def test_roundtrip_empty_dict(self):
        assert self.codec.from_json(self.codec.to_json({})) == {}

    def test_roundtrip_nested(self):
        val = {"items": [1, "two", None, {"nested": True}]}
        assert self.codec.from_json(self.codec.to_json(val)) == val

    # -- Round-trip: dataclass --

    def test_roundtrip_dataclass(self):
        pt = BEPoint(x=1, y=2)
        encoded = self.codec.to_json(pt)
        assert isinstance(encoded, dict)
        assert "__nv_dataclass__" in encoded
        restored = self.codec.from_json(encoded)
        assert isinstance(restored, BEPoint)
        assert restored == pt

    def test_roundtrip_function_local_dataclass(self):
        @dataclasses.dataclass
        class LocalPoint:
            x: int
            y: int

        pt = LocalPoint(x=7, y=8)
        encoded = self.codec.to_json(pt)
        assert isinstance(encoded, dict)
        assert "__nv_dataclass__" in encoded
        restored = self.codec.from_json(encoded)
        assert isinstance(restored, LocalPoint)
        assert restored == pt

    # -- Round-trip: pydantic (if available) --

    def test_roundtrip_pydantic(self):
        import pydantic

        class PydPoint(pydantic.BaseModel):
            x: int
            y: int

        pt = PydPoint(x=3, y=4)
        encoded = self.codec.to_json(pt)
        assert isinstance(encoded, dict)
        assert "__nv_pydantic__" in encoded
        assert encoded["data"] == {"x": 3, "y": 4}
        restored = self.codec.from_json(encoded)
        assert isinstance(restored, PydPoint)
        assert restored.x == 3
        assert restored.y == 4

    # -- Round-trip: pickle fallback (BestEffortAnyCodec uses pickle
    #    internally for non-JSON-serializable types; this tests that
    #    existing code path) --

    def test_roundtrip_frozenset(self):
        """Non-JSON-serializable objects use the pickle fallback path."""
        val = frozenset([1, 2, 3])
        encoded = self.codec.to_json(val)
        assert isinstance(encoded, dict)
        assert "__nv_pickle__" in encoded
        restored = self.codec.from_json(encoded)
        assert restored == val

    def test_faulty_model_dump_falls_back_to_pickle(self):
        encoded = self.codec.to_json(FaultyDumpValue())
        assert "__nv_pickle__" in cast(JsonObject, encoded)

    def test_unpickleable_value_falls_back_to_string(self):
        encoded = cast(JsonObject, self.codec.to_json(UnpickleableValue()))
        assert cast(str, encoded["__nv_fallback_str__"]).endswith(".UnpickleableValue")
        assert cast(str, encoded["data"]) == "unpickleable"

    # -- from_json: non-dict inputs (the original bug) --

    @pytest.mark.parametrize(
        "value",
        [42, 0, -1, 3.14, "hello", "data", "", [1, 2], [], [1, "data", 3]],
        ids=[
            "int",
            "zero",
            "negative",
            "float",
            "string",
            "string_data",
            "empty_string",
            "list",
            "empty_list",
            "list_with_data",
        ],
    )
    def test_from_json_non_dict_passthrough(self, value):
        """from_json must return non-dict values unchanged without raising."""
        assert self.codec.from_json(value) == value

    @pytest.mark.parametrize("value", [None, True, False])
    def test_from_json_singleton_passthrough(self, value):
        """from_json must return singletons by identity."""
        assert self.codec.from_json(value) is value

    # -- from_json: untagged dicts pass through --

    def test_from_json_untagged_dict(self):
        val = {"key": "value", "data": 123}
        assert self.codec.from_json(val) == val

    def test_from_json_empty_dict(self):
        assert self.codec.from_json({}) == {}

    def test_from_json_pydantic_validation_failure_returns_raw_dict(self):
        data = {
            "__nv_pydantic__": f"{BrokenValidatedModel.__module__}.{BrokenValidatedModel.__qualname__}",
            "data": {"x": 1},
        }
        assert self.codec.from_json(data) == data

    def test_from_json_dataclass_reconstruction_failure_returns_raw_dict(self):
        data = {
            "__nv_dataclass__": f"{BEPoint.__module__}.{BEPoint.__qualname__}",
            "data": {"x": 1},
        }
        assert self.codec.from_json(data) == data

    def test_from_json_invalid_pickle_returns_raw_dict(self):
        data = {"__nv_pickle__": "broken.Type", "data": "not-base64"}
        assert self.codec.from_json(data) == data

    def test_from_json_fallback_string_returns_string(self):
        data = {"__nv_fallback_str__": "broken.Type", "data": "fallback-value"}
        assert self.codec.from_json(data) == "fallback-value"

    # -- to_json: tagging --

    def test_to_json_dataclass_tagged(self):
        encoded = cast(JsonObject, self.codec.to_json(BEPoint(x=0, y=0)))
        assert "__nv_dataclass__" in encoded
        assert cast(JsonObject, encoded["data"]) == {"x": 0, "y": 0}

    def test_to_json_native_types_untagged(self):
        """JSON-native types should pass through without tags."""
        for val in [42, "text", 3.14, True, None, [1], {"k": "v"}]:
            encoded = self.codec.to_json(val)
            if isinstance(encoded, dict):
                assert not any(k.startswith("__nv_") for k in encoded)
