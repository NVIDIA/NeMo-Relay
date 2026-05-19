# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stable telemetry-v1 facade."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

import pytest

from nemo_flow import LLMRequest, ScopeType, llm, scope, subscribers, telemetry_v1, tools


def _subscriber_name(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


def _make_request() -> LLMRequest:
    return LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}], "model": "test-model"})


def test_event_to_dict_serializes_stable_scope_and_mark_shapes() -> None:
    events = []
    name = _subscriber_name("telemetry-v1-serialize")
    subscribers.register(name, events.append)
    try:
        root = scope.push(
            "telemetry-v1-agent",
            ScopeType.Agent,
            data={"session_id": "s1"},
            metadata={"source": "test"},
        )
        scope.event("telemetry-v1-mark", handle=root, data={"checkpoint": 1})
        tool = tools.call("telemetry-v1-tool", {"query": "hello"}, handle=root, tool_call_id="tool-1")
        tools.call_end(tool, {"result": "ok"})
        llm_handle = llm.call(
            "telemetry-v1-llm",
            _make_request(),
            handle=root,
            model_name="test-model",
            metadata={"api_request_id": "api-1"},
        )
        llm.call_end(llm_handle, {"assistant_message": {"content": "hello"}})
        scope.pop(root, output={"done": True})
    finally:
        subscribers.deregister(name)

    payloads = [telemetry_v1.event_to_dict(event) for event in events]
    assert all(payload["schema_version"] == telemetry_v1.EVENT_SCHEMA_VERSION for payload in payloads)
    assert {payload["kind"] for payload in payloads} == {"scope", "mark"}

    scope_payload = next(payload for payload in payloads if payload["name"] == "telemetry-v1-agent")
    assert scope_payload["scope_category"] == "start"
    assert scope_payload["category"] == "agent"
    assert scope_payload["data"] is None
    assert scope_payload["metadata"] == {"source": "test"}
    assert scope_payload["uuid"]
    assert scope_payload["parent_uuid"]
    assert scope_payload["timestamp"]

    scope_end_payload = next(
        payload
        for payload in payloads
        if payload["name"] == "telemetry-v1-agent" and payload["scope_category"] == "end"
    )
    assert scope_end_payload["data"] == {"done": True}
    assert scope_end_payload["metadata"] == {"source": "test"}

    mark_payload = next(payload for payload in payloads if payload["name"] == "telemetry-v1-mark")
    assert mark_payload["scope_category"] is None
    assert mark_payload["data"] == {"checkpoint": 1}
    assert mark_payload["annotated_request"] is None
    assert mark_payload["annotated_response"] is None

    llm_payload = next(
        payload
        for payload in payloads
        if payload["name"] == "telemetry-v1-llm" and payload["scope_category"] == "start"
    )
    assert llm_payload["category"] == "llm"
    assert llm_payload["category_profile"] == {"model_name": "test-model"}
    assert llm_payload["data"]["content"]["messages"][0]["content"] == "hi"
    assert llm_payload["metadata"] == {"api_request_id": "api-1"}


def test_event_to_json_round_trips_stable_payload() -> None:
    events = []
    name = _subscriber_name("telemetry-v1-json")
    subscribers.register(name, events.append)
    try:
        scope.event("telemetry-v1-json-mark", data={"ok": True})
    finally:
        subscribers.deregister(name)

    payload = json.loads(telemetry_v1.event_to_json(events[-1]))
    assert payload["schema_version"] == telemetry_v1.EVENT_SCHEMA_VERSION
    assert payload["kind"] == "mark"
    assert payload["name"] == "telemetry-v1-json-mark"
    assert payload["data"] == {"ok": True}


def test_observer_context_manager_receives_serialized_events_and_deregisters() -> None:
    events = []
    name = _subscriber_name("telemetry-v1-observer")

    with telemetry_v1.observer(name, events.append):
        scope.event("telemetry-v1-observed", data={"step": 1})

    scope.event("telemetry-v1-not-observed", data={"step": 2})

    assert [event["name"] for event in events] == ["telemetry-v1-observed"]
    assert events[0]["schema_version"] == telemetry_v1.EVENT_SCHEMA_VERSION


def test_observer_log_policy_isolates_callback_errors(caplog: pytest.LogCaptureFixture) -> None:
    name = _subscriber_name("telemetry-v1-log")

    def boom(_event) -> None:
        raise RuntimeError("subscriber failed")

    with caplog.at_level(logging.ERROR):
        with telemetry_v1.observer(name, boom, error_policy="log"):
            scope.event("telemetry-v1-log-policy")

    assert "NeMo Flow telemetry observer" in caplog.text
    assert "subscriber failed" in caplog.text


def test_register_observer_rejects_unknown_error_policy() -> None:
    with pytest.raises(ValueError, match="error_policy"):
        telemetry_v1.register_observer(
            _subscriber_name("telemetry-v1-bad-policy"),
            lambda _event: None,
            error_policy="bad",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="error_policy"):
        telemetry_v1.register_observer(
            _subscriber_name("telemetry-v1-raise-policy"),
            lambda _event: None,
            error_policy="raise",  # type: ignore[arg-type]
        )
