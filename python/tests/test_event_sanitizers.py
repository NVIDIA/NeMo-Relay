# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
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


def test_mark_sanitizer_exception_preserves_observability_fields(capture_events):
    _capture_name, events = capture_events

    def raises(_event: nemo_relay.Event, _fields: EventSanitizeFields) -> EventSanitizeFields:
        raise RuntimeError("sanitize boom")

    guardrails.register_mark_sanitize("python-mark-raises", 0, raises)
    try:
        scope.event("checkpoint", data={"kept": True})
        subscribers.flush()
    finally:
        guardrails.deregister_mark_sanitize("python-mark-raises")

    assert events[-1].data == {"kept": True}
    assert events[-1].metadata is None


async def test_async_mark_sanitizer_runs_on_originating_loop(capture_events):
    _capture_name, events = capture_events
    originating_loop = asyncio.get_running_loop()

    async def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        await asyncio.sleep(0)
        assert asyncio.get_running_loop() is originating_loop
        return {
            "data": {"async": True},
            "category_profile": fields["category_profile"],
            "metadata": fields["metadata"],
        }

    guardrails.register_mark_sanitize("python-async-mark", 0, sanitize)
    try:
        scope.event("async-checkpoint", data={"raw": True})
        await subscribers.flush_async()
    finally:
        guardrails.deregister_mark_sanitize("python-async-mark")

    assert events[-1].data == {"async": True}


async def test_nested_async_sanitizer_event_precedes_already_queued_event(capture_events):
    _capture_name, events = capture_events
    entered = asyncio.Event()
    release = asyncio.Event()

    async def sanitize(event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        if event.name == "python-outer-event":
            entered.set()
            await release.wait()
            with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
                scope.event("python-nested-event")
        return fields

    guardrails.register_mark_sanitize("python-nested-event-order", 0, sanitize)
    try:
        scope.event("python-outer-event")
        await entered.wait()
        scope.event("python-later-event")
        release.set()
        await subscribers.flush_async()
    finally:
        guardrails.deregister_mark_sanitize("python-nested-event-order")

    assert [event.name for event in events] == [
        "python-outer-event",
        "python-nested-event",
        "python-later-event",
    ]


async def test_scope_start_sanitizer_uses_started_scope_context(capture_events):
    _capture_name, events = capture_events
    observed_scope_uuids: list[str] = []

    async def sanitize(event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        if event.name == "python-start-context":
            await asyncio.sleep(0)
            observed_scope_uuids.append(scope.get_handle().uuid)
            scope.event("python-start-context-nested")
        return fields

    guardrails.register_scope_sanitize_start("python-start-context", 0, sanitize)
    handle = scope.push("python-start-context", nemo_relay.ScopeType.Agent)
    try:
        await subscribers.flush_async()
    finally:
        scope.pop(handle)
        await subscribers.flush_async()
        guardrails.deregister_scope_sanitize_start("python-start-context")

    nested = next(event for event in events if event.name == "python-start-context-nested")
    assert observed_scope_uuids == [handle.uuid]
    assert nested.parent_uuid == handle.uuid


async def test_async_mark_sanitizer_uses_each_emitter_context(capture_events):
    request_id = contextvars.ContextVar("request_id", default="registration")
    observed: dict[str, str] = {}

    async def sanitize(event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        await asyncio.sleep(0)
        observed[event.name] = request_id.get()
        return fields

    async def emit(name: str) -> None:
        token = request_id.set(name)
        try:
            scope.event(name)
        finally:
            request_id.reset(token)

    guardrails.register_mark_sanitize("python-emitter-context", 0, sanitize)
    try:
        await asyncio.gather(emit("request-a"), emit("request-b"))
        await subscribers.flush_async()
    finally:
        guardrails.deregister_mark_sanitize("python-emitter-context")

    assert observed == {"request-a": "request-a", "request-b": "request-b"}


async def test_async_mark_sanitizer_uses_cross_thread_emitter_context(capture_events):
    request_id = contextvars.ContextVar("cross_thread_request_id", default="registration")
    observed: list[str] = []

    async def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        observed.append(request_id.get())
        await asyncio.sleep(0)
        observed.append(request_id.get())
        return fields

    guardrails.register_mark_sanitize("python-cross-thread-emitter-context", 0, sanitize)
    token = request_id.set("emission")
    try:
        await asyncio.to_thread(scope.event, "cross-thread-emitter-context")
        await subscribers.flush_async()
    finally:
        request_id.reset(token)
        guardrails.deregister_mark_sanitize("python-cross-thread-emitter-context")

    assert observed == ["emission", "emission"]


def test_sync_mark_sanitizer_uses_emitter_context(capture_events):
    request_id = contextvars.ContextVar("request_id", default="registration")
    observed: list[str] = []

    def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        observed.append(request_id.get())
        return fields

    guardrails.register_mark_sanitize("python-sync-emitter-context", 0, sanitize)
    try:
        token = request_id.set("emission")
        try:
            scope.event("sync-emitter-context")
        finally:
            request_id.reset(token)
        subscribers.flush()
    finally:
        guardrails.deregister_mark_sanitize("python-sync-emitter-context")

    assert observed == ["emission"]


async def test_async_flush_keeps_originating_sanitizer_loop_running(capture_events):
    _capture_name, events = capture_events

    async def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        await asyncio.sleep(0)
        return {
            "data": {"async_flush": True},
            "category_profile": fields["category_profile"],
            "metadata": fields["metadata"],
        }

    guardrails.register_mark_sanitize("python-async-flush", 0, sanitize)
    try:
        scope.event("async-flush-checkpoint", data={"raw": True})
        with pytest.raises(RuntimeError, match=r"await subscribers\.flush_async"):
            subscribers.flush()
        await asyncio.wait_for(subscribers.flush_async(), timeout=2)
    finally:
        guardrails.deregister_mark_sanitize("python-async-flush")

    assert events[-1].data == {"async_flush": True}


async def test_queued_sanitizer_keeps_emission_scope_after_pop(capture_events):
    _capture_name, events = capture_events
    handle = scope.push("python-emission-scope", nemo_relay.ScopeType.Agent)
    entered = asyncio.Event()
    release = asyncio.Event()
    observed: list[str] = []

    async def sanitize(event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        if event.name != "python-scope-snapshot":
            return fields
        entered.set()
        await release.wait()
        observed.append(scope.get_handle().uuid)
        scope.event("python-scope-snapshot-nested")
        return fields

    guardrails.register_mark_sanitize("python-scope-snapshot", 0, sanitize)
    try:
        scope.event("python-scope-snapshot")
        await asyncio.wait_for(entered.wait(), timeout=1)
        scope.pop(handle)
        release.set()
        await asyncio.wait_for(subscribers.flush_async(), timeout=2)
        await asyncio.wait_for(subscribers.flush_async(), timeout=2)
    finally:
        release.set()
        guardrails.deregister_mark_sanitize("python-scope-snapshot")

    assert observed == [handle.uuid]
    nested = next(event for event in events if event.name == "python-scope-snapshot-nested")
    assert nested.parent_uuid == handle.uuid


async def test_async_flush_does_not_consume_default_executor(capture_events):
    _capture_name, events = capture_events
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

    async def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        await asyncio.to_thread(lambda: None)
        return {
            "data": {"default_executor": True},
            "category_profile": fields["category_profile"],
            "metadata": fields["metadata"],
        }

    guardrails.register_mark_sanitize("python-async-flush-executor", 0, sanitize)
    try:
        scope.event("async-flush-executor-checkpoint", data={"raw": True})
        await asyncio.wait_for(subscribers.flush_async(), timeout=2)
    finally:
        guardrails.deregister_mark_sanitize("python-async-flush-executor")

    assert events[-1].data == {"default_executor": True}


def test_async_sanitizer_registered_on_closed_loop_uses_fallback(capture_events):
    _capture_name, events = capture_events
    request_id = contextvars.ContextVar("fallback_request_id", default="registration")
    observed: list[str] = []

    async def sanitize(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        observed.append(request_id.get())
        await asyncio.sleep(0)
        observed.append(request_id.get())
        return {
            "data": {"fresh_loop": True},
            "category_profile": fields["category_profile"],
            "metadata": fields["metadata"],
        }

    async def register() -> None:
        guardrails.register_mark_sanitize("python-closed-loop-fallback", 0, sanitize)

    asyncio.run(register())
    token = request_id.set("emission")
    try:
        scope.event("closed-loop-checkpoint", data={"raw": True})
        subscribers.flush()
    finally:
        request_id.reset(token)
        guardrails.deregister_mark_sanitize("python-closed-loop-fallback")

    assert events[-1].data == {"fresh_loop": True}
    assert observed == ["emission", "emission"]


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
        await subscribers.flush_async()
        plugin.clear()
        scope.event("cleared", data={"raw": True})
        await subscribers.flush_async()
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
        await subscribers.flush_async()
        assert events[-1].data == {"raw": True}
    finally:
        plugin.clear()
        plugin.deregister(kind)
