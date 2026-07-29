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


async def test_sanitizer_descendants_lose_reentrant_flush_after_settlement(capture_events):
    blocker_entered = asyncio.Event()
    release_blocker = asyncio.Event()
    descendant_finished = asyncio.Event()
    descendant_task: asyncio.Task[None] | None = None

    async def descendant() -> None:
        await blocker_entered.wait()
        await subscribers.flush_async()
        descendant_finished.set()

    async def sanitize(event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        nonlocal descendant_task
        if event.name == "descendant-origin":
            descendant_task = asyncio.create_task(descendant())
        elif event.name == "descendant-blocked":
            blocker_entered.set()
            await release_blocker.wait()
        return fields

    guardrails.register_mark_sanitize("python-descendant-flush-liveness", 0, sanitize)
    try:
        scope.event("descendant-origin")
        scope.event("descendant-blocked")
        await asyncio.wait_for(blocker_entered.wait(), timeout=1)
        await asyncio.sleep(0.05)
        assert not descendant_finished.is_set()
        release_blocker.set()
        await asyncio.wait_for(subscribers.flush_async(), timeout=1)
        assert descendant_task is not None
        await asyncio.wait_for(descendant_task, timeout=1)
    finally:
        release_blocker.set()
        guardrails.deregister_mark_sanitize("python-descendant-flush-liveness")


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


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_event_sanitizer_flush_is_reentrant(capture_events, asynchronous):
    _capture_name, events = capture_events
    flush_returned = False

    def sanitize_sync(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        nonlocal flush_returned
        subscribers.flush()
        flush_returned = True
        return fields

    async def sanitize_async(_event: nemo_relay.Event, fields: EventSanitizeFields) -> EventSanitizeFields:
        await asyncio.sleep(0)
        return sanitize_sync(_event, fields)

    guardrails.register_mark_sanitize(
        "python-reentrant-mark",
        0,
        sanitize_async if asynchronous else sanitize_sync,
    )
    try:
        scope.event("reentrant-checkpoint", data={"raw": True})
        await subscribers.flush_async()
    finally:
        guardrails.deregister_mark_sanitize("python-reentrant-mark")

    assert flush_returned is True
    assert events[-1].data == {"raw": True}


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
