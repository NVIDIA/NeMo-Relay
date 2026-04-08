# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Agent Toolkit Nexus subscriber and event handling."""

import pytest
from nat_nexus import (
    LLMEndEvent,
    LLMRequest,
    LLMStartEvent,
    MarkEvent,
    ScopeEndEvent,
    ScopeStartEvent,
    ScopeType,
    ToolEndEvent,
    ToolStartEvent,
    llm,
    scope,
    subscribers,
    tools,
)

EVENT_VARIANTS = (
    ScopeStartEvent,
    ScopeEndEvent,
    ToolStartEvent,
    ToolEndEvent,
    LLMStartEvent,
    LLMEndEvent,
    MarkEvent,
)


def make_request():
    return LLMRequest({}, {"messages": [], "model": "test-model"})


class TestSubscribers:
    def test_register_and_deregister(self):
        events = []
        subscribers.register("py_test_sub", lambda e: events.append(e))
        handle = scope.push("sub_test", ScopeType.Function)
        scope.pop(handle)
        assert subscribers.deregister("py_test_sub")
        assert len(events) >= 2

    def test_subscriber_receives_event_objects(self):
        events = []
        subscribers.register("py_evt_sub", lambda e: events.append(e))
        handle = scope.push("evt_obj_test", ScopeType.Agent)
        scope.pop(handle)
        subscribers.deregister("py_evt_sub")

        assert len(events) >= 2
        for e in events:
            assert isinstance(e, EVENT_VARIANTS)
            assert e.uuid is not None
            assert e.kind is not None

    def test_duplicate_subscriber_raises(self):
        subscribers.register("py_dup_sub", lambda e: None)
        with pytest.raises(RuntimeError):
            subscribers.register("py_dup_sub", lambda e: None)
        subscribers.deregister("py_dup_sub")

    def test_deregister_nonexistent(self):
        assert not subscribers.deregister("nonexistent_sub")


class TestSubscriberEventDetails:
    def test_scope_events_have_correct_types(self):
        events = []
        subscribers.register("py_detail_sub", lambda e: events.append(e))
        handle = scope.push("detail_test", ScopeType.Evaluator)
        scope.pop(handle)
        subscribers.deregister("py_detail_sub")

        assert len(events) >= 2
        assert isinstance(events[0], ScopeStartEvent)
        assert isinstance(events[1], ScopeEndEvent)

    def test_tool_events(self):
        events = []
        subscribers.register("py_tool_evt", lambda e: events.append(e))
        handle = tools.call("evt_tool", {"x": 1})
        tools.call_end(handle, {"y": 2})
        subscribers.deregister("py_tool_evt")

        start_events = [e for e in events if isinstance(e, ToolStartEvent)]
        end_events = [e for e in events if isinstance(e, ToolEndEvent)]
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    def test_llm_events(self):
        events = []
        subscribers.register("py_llm_evt", lambda e: events.append(e))
        request = make_request()
        handle = llm.call("evt_llm", request)
        llm.call_end(handle, {"done": True})
        subscribers.deregister("py_llm_evt")

        start_events = [e for e in events if isinstance(e, LLMStartEvent)]
        end_events = [e for e in events if isinstance(e, LLMEndEvent)]
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    def test_mark_event(self):
        events = []
        subscribers.register("py_mark_evt", lambda e: events.append(e))
        scope.event("test_mark", data={"info": "test"})
        subscribers.deregister("py_mark_evt")

        mark_events = [e for e in events if isinstance(e, MarkEvent)]
        assert len(mark_events) >= 1


class TestHandleProperties:
    def test_scope_handle_all_properties(self):
        handle = scope.push("prop_test", ScopeType.Embedder)
        assert isinstance(handle.uuid, str)
        assert len(handle.uuid) > 0
        assert handle.name == "prop_test"
        assert handle.scope_type == ScopeType.Embedder
        assert handle.parent_uuid is not None  # root is parent
        # data and metadata are None by default for scope handles
        scope.pop(handle)

    def test_tool_handle_all_properties(self):
        handle = tools.call("prop_tool", {"x": 1}, data={"d": "v"}, metadata={"m": "v"})
        assert isinstance(handle.uuid, str)
        assert handle.name == "prop_tool"
        # data includes sanitized_args from the call
        assert handle.data is not None
        tools.call_end(handle, {})

    def test_llm_handle_all_properties(self):
        request = make_request()
        handle = llm.call("prop_llm", request, data={"d": 1}, metadata={"m": 2})
        assert isinstance(handle.uuid, str)
        assert handle.name == "prop_llm"
        assert handle.data is not None
        llm.call_end(handle, {})

    def test_event_all_properties(self):
        events = []
        subscribers.register("py_prop_evt", lambda e: events.append(e))
        scope.event("prop_mark", data={"key": "val"}, metadata={"meta": "data"})
        subscribers.deregister("py_prop_evt")

        assert len(events) >= 1
        e = events[0]
        assert isinstance(e, MarkEvent)
        assert isinstance(e.uuid, str)
        assert e.name == "prop_mark"
        assert e.kind == "Mark"
        assert e.timestamp is not None
        assert isinstance(e.timestamp, str)
