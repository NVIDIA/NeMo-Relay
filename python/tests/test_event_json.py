# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for canonical subscriber event JSON helpers."""

from __future__ import annotations

import json
from uuid import uuid4

from nemo_relay import MarkEvent, ScopeEvent, ScopeType, scope, subscribers


def _subscriber_name(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


def test_subscriber_events_expose_canonical_json_helpers():
    events = []
    name = _subscriber_name("py-event-json")
    subscribers.register(name, events.append)
    try:
        with scope.scope(
            "json-scope",
            ScopeType.Agent,
            input={"input": True},
            metadata={"trace": "abc"},
        ):
            scope.event("json-mark", data={"mark": True}, metadata={"source": "test"})
    finally:
        subscribers.flush()
        subscribers.deregister(name)

    scope_event = next(
        event
        for event in events
        if isinstance(event, ScopeEvent) and event.name == "json-scope" and event.scope_category == "start"
    )
    scope_payload = scope_event.to_dict()
    assert scope_payload["kind"] == "scope"
    assert scope_payload["scope_category"] == "start"
    assert scope_payload["category"] == "agent"
    assert scope_payload["name"] == "json-scope"
    assert scope_payload["data"] == {"input": True}
    assert scope_payload["metadata"] == {"trace": "abc"}
    assert json.loads(scope_event.to_json()) == scope_payload

    mark_event = next(event for event in events if isinstance(event, MarkEvent) and event.name == "json-mark")
    mark_payload = mark_event.to_dict()
    assert mark_payload["kind"] == "mark"
    assert mark_payload["name"] == "json-mark"
    assert mark_payload["data"] == {"mark": True}
    assert mark_payload["metadata"] == {"source": "test"}
    assert json.loads(mark_event.to_json()) == mark_payload
