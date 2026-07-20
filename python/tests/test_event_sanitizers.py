# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest

import nemo_relay
from nemo_relay import EventSanitizeFields, guardrails, plugin, scope, scope_local, subscribers


@pytest.fixture(name="capture_events")
def capture_events_fixture() -> Iterator[tuple[str, list[nemo_relay.Event]]]:
    events: list[nemo_relay.Event] = []
    name = "test-event-sanitizer-capture"
    subscribers.register(name, events.append)
    yield name, events
    subscribers.deregister(name)


def test_global_mark_sanitizers_order_convert_fields_and_remove_values(capture_events):
    _capture_name, events = capture_events
    calls: list[tuple[str, object]] = []

    def first(event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        calls.append((event.name, fields["data"]))
        return {
            "data": {"stage": "first"},
            "category_profile": fields["category_profile"],
            "metadata": None,
        }

    def second(event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        calls.append((event.kind, fields["data"]))
        return {
            "data": {"stage": "second"},
            "category_profile": fields["category_profile"],
            "metadata": fields["metadata"],
        }

    guardrails.register_mark_sanitize("python-mark-second", 20, second)
    guardrails.register_mark_sanitize("python-mark-first", 10, first)
    try:
        scope.event("checkpoint", data={"secret": "raw"}, metadata={"secret": "raw"})
        subscribers.flush()
    finally:
        guardrails.deregister_mark_sanitize("python-mark-first")
        guardrails.deregister_mark_sanitize("python-mark-second")

    mark = events[-1]
    assert mark.data == {"stage": "second"}
    assert mark.metadata is None
    assert calls == [("checkpoint", {"secret": "raw"}), ("mark", {"stage": "first"})]


def test_mark_sanitizer_exception_clears_observability_fields(capture_events):
    _capture_name, events = capture_events

    def raises(_event: nemo_relay.Event, _fields: EventSanitizeFields) -> EventSanitizeFields:
        raise RuntimeError("sanitize boom")

    guardrails.register_mark_sanitize("python-mark-raises", 0, raises)
    try:
        scope.event("checkpoint", data={"kept": True})
        subscribers.flush()
    finally:
        guardrails.deregister_mark_sanitize("python-mark-raises")

    assert events[-1].data is None
    assert events[-1].metadata is None


def test_scope_start_and_end_sanitizers_cover_category_profile(capture_events):
    _capture_name, events = capture_events

    def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        profile = dict(fields["category_profile"] or {})
        profile["subtype"] = "sanitized"
        return {"data": None, "category_profile": profile, "metadata": {"safe": True}}

    guardrails.register_scope_sanitize_start("python-scope-start", 0, sanitize)
    guardrails.register_scope_sanitize_end("python-scope-end", 0, sanitize)
    try:
        handle = scope.push(
            "generic",
            nemo_relay.ScopeType.Custom,
            data={"secret": "start"},
            metadata={"secret": "start"},
            input={"secret": "input"},
        )
        scope.pop(handle, output={"secret": "output"}, metadata={"secret": "end"})
        subscribers.flush()
    finally:
        guardrails.deregister_scope_sanitize_start("python-scope-start")
        guardrails.deregister_scope_sanitize_end("python-scope-end")

    lifecycle = [event for event in events if event.name == "generic"]
    assert len(lifecycle) == 2
    assert all(event.data is None for event in lifecycle)
    assert all(event.metadata == {"safe": True} for event in lifecycle)
    assert all(event.category_profile["subtype"] == "sanitized" for event in lifecycle)


def test_scope_local_event_sanitizers_are_inherited_and_cleaned_up(capture_events):
    _capture_name, events = capture_events

    def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        return {
            "data": {"scope_local": True},
            "category_profile": fields["category_profile"],
            "metadata": fields["metadata"],
        }

    owner = scope.push("owner", nemo_relay.ScopeType.Agent)
    try:
        scope_local.register_mark_sanitize(owner, "python-local-mark", 0, sanitize)
        scope.event("inside", data={"raw": True})
        child = scope.push("child", nemo_relay.ScopeType.Function)
        try:
            scope.event("inherited", data={"raw": True})
        finally:
            scope.pop(child)
    finally:
        scope.pop(owner)
    scope.event("outside", data={"raw": True})
    subscribers.flush()

    marks = {event.name: event for event in events if event.kind == "mark"}
    assert marks["inside"].data == {"scope_local": True}
    assert marks["inherited"].data == {"scope_local": True}
    assert marks["outside"].data == {"raw": True}


async def test_in_process_plugin_event_sanitizers_are_removed_on_clear(capture_events):
    class EventPlugin:
        def validate(self, _config):
            return None

        def register(self, _config, context):
            def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
                return {
                    "data": {"plugin": True},
                    "category_profile": fields["category_profile"],
                    "metadata": fields["metadata"],
                }

            context.register_mark_sanitize_guardrail("mark", 0, sanitize)

    kind = "python.test_event_sanitizer"
    _capture_name, events = capture_events
    plugin.register(kind, cast(plugin.Plugin, EventPlugin()))
    try:
        await plugin.initialize(plugin.PluginConfig(components=[plugin.ComponentSpec(kind=kind)]))
        scope.event("configured", data={"raw": True})
        subscribers.flush()
        plugin.clear()
        scope.event("cleared", data={"raw": True})
        subscribers.flush()
    finally:
        plugin.clear()
        plugin.deregister(kind)

    marks = {event.name: event for event in events if event.kind == "mark"}
    assert marks["configured"].data == {"plugin": True}
    assert marks["cleared"].data == {"raw": True}


async def test_in_process_plugin_rolls_back_event_sanitizer_when_registration_fails(capture_events):
    class FailingPlugin:
        def validate(self, _config):
            return None

        def register(self, _config, context):
            def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
                return {
                    "data": {"leaked": True},
                    "category_profile": fields["category_profile"],
                    "metadata": fields["metadata"],
                }

            context.register_mark_sanitize_guardrail("mark", 0, sanitize)
            raise RuntimeError("registration failed")

    kind = "python.test_event_sanitizer_rollback"
    plugin.register(kind, cast(plugin.Plugin, FailingPlugin()))
    _capture_name, events = capture_events
    try:
        with pytest.raises(RuntimeError, match="registration failed"):
            await plugin.initialize(plugin.PluginConfig(components=[plugin.ComponentSpec(kind=kind)]))
        scope.event("after-failure", data={"raw": True})
        subscribers.flush()
        assert events[-1].data == {"raw": True}
    finally:
        plugin.clear()
        plugin.deregister(kind)
