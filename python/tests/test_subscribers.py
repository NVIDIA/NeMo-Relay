# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Flow subscriber and event handling."""

from datetime import datetime, timezone
from typing import Any, cast

import pytest

from nemo_flow import (
    LLMRequest,
    MarkEvent,
    ScopeEvent,
    ScopeType,
    llm,
    scope,
    subscribers,
    tools,
)

EVENT_VARIANTS = (
    ScopeEvent,
    MarkEvent,
)


def make_request():
    return LLMRequest({}, {"messages": [], "model": "test-model"})


def parse_event_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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
        assert isinstance(events[0], ScopeEvent)
        assert isinstance(events[1], ScopeEvent)
        assert events[0].scope_category == "start"
        assert events[1].scope_category == "end"
        assert events[0].category == "evaluator"
        assert events[1].category == "evaluator"

    def test_tool_events(self):
        events = []
        subscribers.register("py_tool_evt", lambda e: events.append(e))
        handle = tools.call("evt_tool", {"x": 1})
        tools.call_end(handle, {"y": 2})
        subscribers.deregister("py_tool_evt")

        start_events = [
            e for e in events if isinstance(e, ScopeEvent) and e.category == "tool" and e.scope_category == "start"
        ]
        end_events = [
            e for e in events if isinstance(e, ScopeEvent) and e.category == "tool" and e.scope_category == "end"
        ]
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    def test_llm_events(self):
        events = []
        subscribers.register("py_llm_evt", lambda e: events.append(e))
        request = make_request()
        handle = llm.call("evt_llm", request)
        llm.call_end(handle, {"done": True})
        subscribers.deregister("py_llm_evt")

        start_events = [
            e for e in events if isinstance(e, ScopeEvent) and e.category == "llm" and e.scope_category == "start"
        ]
        end_events = [
            e for e in events if isinstance(e, ScopeEvent) and e.category == "llm" and e.scope_category == "end"
        ]
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    def test_mark_event(self):
        events = []
        subscribers.register("py_mark_evt", lambda e: events.append(e))
        scope.event("test_mark", data={"info": "test"})
        subscribers.deregister("py_mark_evt")

        mark_events = [e for e in events if isinstance(e, MarkEvent)]
        assert len(mark_events) >= 1

    def test_manual_lifecycle_timestamps_accept_datetime(self):
        events = []
        subscribers.register("py_timestamp_evt", lambda e: events.append(e))
        timestamps = [
            datetime(2026, 1, 1, 0, 0, second, 123456 + (second * 1000), tzinfo=timezone.utc) for second in range(7)
        ]
        scope_handle = scope.push("py_ts_scope", ScopeType.Agent, timestamp=timestamps[0])
        scope.event("py_ts_mark", handle=scope_handle, timestamp=timestamps[1])
        tool_handle = tools.call("py_ts_tool", {"x": 1}, timestamp=timestamps[2])
        tools.call_end(tool_handle, {"ok": True}, timestamp=timestamps[3])
        llm_handle = llm.call("py_ts_llm", make_request(), timestamp=timestamps[4])
        llm.call_end(llm_handle, {"ok": True}, timestamp=timestamps[5])
        scope.pop(scope_handle, timestamp=timestamps[6])
        subscribers.deregister("py_timestamp_evt")

        observed = [
            (event.name, parse_event_timestamp(event.timestamp)) for event in events if event.name.startswith("py_ts_")
        ]
        assert observed == [
            ("py_ts_scope", timestamps[0]),
            ("py_ts_mark", timestamps[1]),
            ("py_ts_tool", timestamps[2]),
            ("py_ts_tool", timestamps[3]),
            ("py_ts_llm", timestamps[4]),
            ("py_ts_llm", timestamps[5]),
            ("py_ts_scope", timestamps[6]),
        ]

    @pytest.mark.parametrize(
        ("bad_timestamp", "error_type", "message"),
        [
            (cast(Any, "2026-01-01T00:00:00Z"), TypeError, "datetime.datetime"),
            (datetime(2026, 1, 1), ValueError, "timezone-aware"),
        ],
    )
    def test_manual_lifecycle_timestamps_reject_invalid_datetime_values(self, bad_timestamp, error_type, message):
        with pytest.raises(error_type, match=message):
            scope.push("py_bad_ts_scope_start", ScopeType.Agent, timestamp=bad_timestamp)

        scope_handle = scope.push("py_bad_ts_scope", ScopeType.Agent)
        try:
            with pytest.raises(error_type, match=message):
                scope.event("py_bad_ts_mark", handle=scope_handle, timestamp=bad_timestamp)

            with pytest.raises(error_type, match=message):
                tools.call("py_bad_ts_tool_start", {"x": 1}, timestamp=bad_timestamp)

            tool_handle = tools.call("py_bad_ts_tool", {"x": 1})
            try:
                with pytest.raises(error_type, match=message):
                    tools.call_end(tool_handle, {"ok": True}, timestamp=bad_timestamp)
            finally:
                tools.call_end(tool_handle, {"ok": True})

            with pytest.raises(error_type, match=message):
                llm.call("py_bad_ts_llm_start", make_request(), timestamp=bad_timestamp)

            llm_handle = llm.call("py_bad_ts_llm", make_request())
            try:
                with pytest.raises(error_type, match=message):
                    llm.call_end(llm_handle, {"ok": True}, timestamp=bad_timestamp)
            finally:
                llm.call_end(llm_handle, {"ok": True})

            with pytest.raises(error_type, match=message):
                scope.pop(scope_handle, timestamp=bad_timestamp)
        finally:
            scope.pop(scope_handle)

    @pytest.mark.parametrize(
        ("bad_timestamp", "error_type", "message"),
        [
            (cast(Any, "2026-01-01T00:00:00Z"), TypeError, "datetime.datetime"),
            (datetime(2026, 1, 1), ValueError, "timezone-aware"),
        ],
    )
    def test_scope_context_manager_timestamps_reject_invalid_datetime_values(self, bad_timestamp, error_type, message):
        with pytest.raises(error_type, match=message):
            with scope.scope("py_bad_ts_context_start", ScopeType.Agent, timestamp=bad_timestamp):
                raise AssertionError("invalid start timestamp should fail before entering the body")

        pushed_handle = None
        with pytest.raises(error_type, match=message):
            with scope.scope("py_bad_ts_context_end", ScopeType.Agent, end_timestamp=bad_timestamp) as handle:
                pushed_handle = handle
        if pushed_handle is not None:
            scope.pop(pushed_handle)


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
        assert e.kind == "mark"
        assert e.timestamp is not None
        assert isinstance(e.timestamp, str)
