# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Agent Toolkit Nexus Python type bindings."""

import json
from typing import Any, cast

import pytest
from nat_nexus import (
    AtifExporter,
    EventType,
    LLMAttributes,
    LLMRequest,
    ScopeAttributes,
    ScopeType,
    ToolAttributes,
    llm,
    scope,
    subscribers,
    tools,
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

    def test_and_operator_and_repr(self):
        combined = ScopeAttributes(ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE)
        parallel_only = ScopeAttributes(ScopeAttributes.PARALLEL)
        intersected = combined & parallel_only
        assert intersected.is_parallel
        assert not intersected.is_relocatable
        assert "ScopeAttributes" in repr(intersected)


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

    def test_or_and_and_repr(self):
        local = ToolAttributes(ToolAttributes.LOCAL)
        empty = ToolAttributes(0)
        combined = local | empty
        intersected = local & empty
        assert combined.is_local
        assert not intersected.is_local
        assert local.value == ToolAttributes.LOCAL
        assert "ToolAttributes" in repr(local)


class TestLLMAttributes:
    def test_stateless_is_int(self):
        assert isinstance(LLMAttributes.STATELESS, int)

    def test_streaming_is_int(self):
        assert isinstance(LLMAttributes.STREAMING, int)

    def test_construct_combined(self):
        attrs = LLMAttributes(LLMAttributes.STATELESS | LLMAttributes.STREAMING)
        assert attrs.is_stateless
        assert attrs.is_streaming

    def test_or_and_and_repr(self):
        stateless = LLMAttributes(LLMAttributes.STATELESS)
        streaming = LLMAttributes(LLMAttributes.STREAMING)
        combined = stateless | streaming
        intersected = combined & stateless
        assert combined.is_streaming
        assert intersected.is_stateless
        assert not intersected.is_streaming
        assert combined.value == LLMAttributes.STATELESS | LLMAttributes.STREAMING
        assert "LLMAttributes" in repr(combined)


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

    def test_headers_must_be_dict(self):
        with pytest.raises(TypeError, match="not an instance of 'dict'"):
            LLMRequest(cast(Any, []), {"model": "gpt-4"})


class TestHandleTypes:
    def test_scope_type_roundtrip_all_variants(self):
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

        for variant in variants:
            handle = scope.push(f"scope-{variant!r}", variant)
            try:
                assert handle.scope_type == variant
            finally:
                scope.pop(handle)

    def test_scope_handle_properties_and_repr(self):
        handle = scope.push(
            "typed_scope",
            ScopeType.Agent,
            attributes=ScopeAttributes(ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE),
            data={"scope": True},
            metadata={"meta": "scope"},
        )
        try:
            assert handle.name == "typed_scope"
            assert handle.scope_type == ScopeType.Agent
            assert handle.attributes.is_parallel
            assert handle.attributes.is_relocatable
            assert handle.data == {"scope": True}
            assert handle.metadata == {"meta": "scope"}
            assert "ScopeHandle" in repr(handle)
        finally:
            scope.pop(handle)

    def test_tool_handle_properties_and_repr(self):
        parent = scope.push("typed_tool_parent", ScopeType.Agent)
        try:
            handle = tools.call(
                "typed_tool",
                {"x": 1},
                attributes=ToolAttributes(ToolAttributes.LOCAL),
                data={"tool": "data"},
                metadata={"tool": "meta"},
            )
            try:
                assert handle.name == "typed_tool"
                assert handle.attributes.is_local
                assert handle.parent_uuid == parent.uuid
                assert handle.data == {"tool": "data"}
                assert handle.metadata == {"tool": "meta"}
                assert "ToolHandle" in repr(handle)
            finally:
                tools.call_end(handle, {"ok": True})
        finally:
            scope.pop(parent)

    def test_llm_handle_properties_and_repr(self):
        parent = scope.push("typed_llm_parent", ScopeType.Agent)
        request = LLMRequest({}, {"messages": [], "model": "typed-model"})
        try:
            handle = llm.call(
                "typed_llm",
                request,
                attributes=LLMAttributes(LLMAttributes.STATELESS | LLMAttributes.STREAMING),
                data={"llm": "data"},
                metadata={"llm": "meta"},
                model_name="typed-model",
            )
            try:
                assert handle.name == "typed_llm"
                assert handle.attributes.is_stateless
                assert handle.attributes.is_streaming
                assert handle.parent_uuid == parent.uuid
                assert handle.data == {"llm": "data"}
                assert handle.metadata == {"llm": "meta"}
                assert "LLMHandle" in repr(handle)
            finally:
                llm.call_end(handle, {"ok": True})
        finally:
            scope.pop(parent)


class TestEventTypes:
    def test_event_properties_include_tool_and_llm_fields(self):
        events = []
        subscribers.register("py_event_types_sub", lambda event: events.append(event))
        parent = scope.push("event_root", ScopeType.Agent, data={"root": True}, metadata={"meta": "root"})
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}], "model": "event-model"})

        try:
            tool_handle = tools.call(
                "event_tool",
                {"x": 1},
                data={"tool": "start"},
                metadata={"tool_meta": True},
                tool_call_id="tool-call-123",
            )
            tools.call_end(tool_handle, {"y": 2}, data={"tool": "end"}, metadata={"tool_end": True})

            llm_handle = llm.call(
                "event_llm",
                request,
                data={"llm": "start"},
                metadata={"llm_meta": True},
                model_name="event-model",
            )
            llm.call_end(llm_handle, {"message": "hello"}, data={"llm": "end"}, metadata={"llm_end": True})

            scope.event("event_mark", handle=parent, data={"mark": True}, metadata={"mark_meta": True})
        finally:
            scope.pop(parent)
            subscribers.deregister("py_event_types_sub")

        tool_start = next(
            event for event in events if event.name == "event_tool" and event.event_type == EventType.Start
        )
        tool_end = next(event for event in events if event.name == "event_tool" and event.event_type == EventType.End)
        llm_start = next(event for event in events if event.name == "event_llm" and event.event_type == EventType.Start)
        llm_end = next(event for event in events if event.name == "event_llm" and event.event_type == EventType.End)
        mark = next(event for event in events if event.name == "event_mark")
        root_uuid = tool_start.root_uuid

        assert tool_start.input == {"x": 1}
        assert tool_start.tool_call_id == "tool-call-123"
        assert root_uuid is not None
        assert tool_end.root_uuid == root_uuid
        assert tool_end.output == {"y": 2}
        assert tool_end.metadata == {"tool_meta": True, "tool_end": True}

        assert llm_start.input == {"headers": request.headers, "content": request.content}
        assert llm_start.model_name == "event-model"
        assert llm_start.root_uuid == root_uuid
        assert llm_end.root_uuid == root_uuid
        assert llm_end.output == {"message": "hello"}
        assert llm_end.metadata == {"llm_meta": True, "llm_end": True}

        assert mark.event_type == EventType.Mark
        assert mark.scope_type is None
        assert mark.root_uuid == root_uuid
        assert mark.data == {"mark": True}
        assert mark.metadata == {"mark_meta": True}
        assert "Event(" in repr(mark)
        assert "T" in mark.timestamp


class TestAtifExporterType:
    def test_exporter_register_export_clear_and_repr(self):
        exporter = AtifExporter(
            "session-types",
            "py-agent",
            "1.0.0",
            model_name="typed-model",
            tool_definitions=[{"name": "typed_tool"}],
            extra={"team": "qa"},
        )
        assert "<AtifExporter>" in repr(exporter)

        exporter.register("py_atif_exporter")
        parent = scope.push("atif_root", ScopeType.Agent)
        request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}], "model": "typed-model"})

        try:
            handle = llm.call("atif_llm", request, model_name="typed-model")
            llm.call_end(handle, {"content": "world"})

            exported_all = exporter.export()
            exported = exporter.export(parent.uuid)
            exported_json_all = json.loads(exporter.export_json())
            exported_json = json.loads(exporter.export_json(parent.uuid))

            assert exported_all["session_id"] == "session-types"
            assert exported["session_id"] == "session-types"
            assert exported["agent"]["name"] == "py-agent"
            assert exported["agent"]["tool_definitions"] == [{"name": "typed_tool"}]
            assert exported["agent"]["extra"] == {"team": "qa"}
            assert exported["steps"]
            assert exported_json_all["session_id"] == "session-types"
            assert exported_json["session_id"] == "session-types"

            exporter.clear()
            assert exporter.export(parent.uuid)["steps"] == []
        finally:
            scope.pop(parent)
            assert exporter.deregister("py_atif_exporter") is True
            assert exporter.deregister("py_atif_exporter") is False

    def test_exporter_invalid_root_uuid_raises(self):
        exporter = AtifExporter("session-types-invalid", "py-agent", "1.0.0")
        with pytest.raises(ValueError, match="Invalid UUID"):
            exporter.export("not-a-uuid")
        with pytest.raises(ValueError, match="Invalid UUID"):
            exporter.export_json("not-a-uuid")
