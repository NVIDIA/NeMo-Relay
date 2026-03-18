# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Agent Toolkit Nexus Python type bindings."""

from nat_nexus import (
    EventType,
    LLMAttributes,
    LLMRequest,
    ScopeAttributes,
    ScopeType,
    ToolAttributes,
)


class TestScopeType:
    def test_all_variants_exist(self):
        variants = [
            ScopeType.Agent,
            ScopeType.Function,
            ScopeType.Tool,
            ScopeType.Llm,
            ScopeType.Retriever,
            ScopeType.Embedder,
            ScopeType.Reranker,
            ScopeType.Guardrail,
            ScopeType.Evaluator,
            ScopeType.Custom,
            ScopeType.Unknown,
        ]
        assert len(variants) == 11

    def test_repr(self):
        assert "Agent" in repr(ScopeType.Agent)


class TestEventType:
    def test_all_variants(self):
        variants = [EventType.Start, EventType.End, EventType.Mark]
        assert len(variants) == 3


class TestScopeAttributes:
    def test_parallel_is_int(self):
        assert isinstance(ScopeAttributes.PARALLEL, int)
        assert ScopeAttributes.PARALLEL == 0b01

    def test_relocatable_is_int(self):
        assert isinstance(ScopeAttributes.RELOCATABLE, int)
        assert ScopeAttributes.RELOCATABLE == 0b10

    def test_construct_from_value(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL)
        assert attrs.is_parallel
        assert not attrs.is_relocatable

    def test_construct_combined(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE)
        assert attrs.is_parallel
        assert attrs.is_relocatable

    def test_or_operator(self):
        a = ScopeAttributes(ScopeAttributes.PARALLEL)
        b = ScopeAttributes(ScopeAttributes.RELOCATABLE)
        combined = a | b
        assert combined.is_parallel
        assert combined.is_relocatable

    def test_value_getter(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL)
        assert attrs.value == ScopeAttributes.PARALLEL


class TestToolAttributes:
    def test_local_is_int(self):
        assert isinstance(ToolAttributes.LOCAL, int)
        assert ToolAttributes.LOCAL == 0b01

    def test_construct(self):
        attrs = ToolAttributes(ToolAttributes.LOCAL)
        assert attrs.is_local

    def test_empty(self):
        attrs = ToolAttributes(0)
        assert not attrs.is_local


class TestLLMAttributes:
    def test_stateless_is_int(self):
        assert isinstance(LLMAttributes.STATELESS, int)

    def test_streaming_is_int(self):
        assert isinstance(LLMAttributes.STREAMING, int)

    def test_construct_combined(self):
        attrs = LLMAttributes(LLMAttributes.STATELESS | LLMAttributes.STREAMING)
        assert attrs.is_stateless
        assert attrs.is_streaming


class TestLLMRequest:
    def test_constructor(self):
        req = LLMRequest({"Authorization": "Bearer token"}, {"messages": []})
        assert req.headers == {"Authorization": "Bearer token"}
        assert req.content == {"messages": []}

    def test_empty_headers(self):
        req = LLMRequest({}, {"q": "test"})
        assert req.headers == {}

    def test_repr(self):
        req = LLMRequest({}, {"model": "gpt-4"})
        r = repr(req)
        assert "LLMRequest" in r
