# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the callback handler against a real NeMo Relay scope stack.

``test_callbacks.py`` drives the handler with a ``MagicMock`` of ``nemo_relay``. That is
the right tool for asserting which calls the handler makes, but a mocked ``scope.pop``
accepts any handle in any order, so it cannot observe the stack's LIFO rule at all. The
failure covered here is exactly that rule being violated, so these tests use the real
stack.

The shape under test is two sibling chain runs that finish out of LIFO order: A starts,
B starts, A ends, B ends. That ordering is what LangGraph produces when it schedules
sibling nodes as concurrent asyncio tasks sharing one scope stack. These tests drive the
handler's callbacks directly rather than through a callback manager, so they reproduce
the stack-level consequence deterministically; they do not attempt to prove how LangGraph
dispatches callbacks, which belongs in the langgraph integration tests.

Two things make a test here easy to get wrong, so each test guards against them:

* ``on_chain_start`` logs and swallows every exception, so a handler that pushed nothing
  would satisfy "no scope was stranded" and "the stack is unchanged" while exercising
  nothing. Each test asserts the scopes were genuinely open before reading the end state.
* ``scope.push`` parents a new scope to the current top of stack unless an explicit
  parent handle is given, so callbacks that omit ``parent_run_id`` produce a nested
  chain, not siblings. The sibling runs here declare a common tracked parent run.
"""

from __future__ import annotations

import asyncio
import datetime
import typing
from uuid import UUID, uuid4

import pytest

import nemo_relay

if typing.TYPE_CHECKING:
    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

# Handles the handler tracks while both siblings are open: the parent run plus A and B.
_OPEN_AT_OVERLAP = 3


@pytest.fixture(autouse=True)
def isolated_scope_stack():
    """Run each test on its own stack so a stranded scope cannot leak into the next."""

    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        yield


@pytest.fixture(name="handler")
def handler_fixture() -> NemoRelayCallbackHandler:
    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

    return NemoRelayCallbackHandler()


def _start(
    handler: NemoRelayCallbackHandler,
    run_id: UUID,
    name: str,
    parent_run_id: UUID | None = None,
) -> None:
    handler.on_chain_start({}, {"task": name}, run_id=run_id, parent_run_id=parent_run_id, name=name)


def _end(handler: NemoRelayCallbackHandler, run_id: UUID, name: str) -> None:
    handler.on_chain_end({"done": name}, run_id=run_id)


async def _overlapping_sibling_runs(handler: NemoRelayCallbackHandler) -> int:
    """Interleave two sibling chain runs so they close out of LIFO order.

    A and B both declare ``parent`` as their parent run, so they are siblings rather than
    a nested chain. They then close as A, B — valid for the graph, rejected by the stack.

    Returns the number of scopes the handler had open while both siblings were running,
    so a caller can tell a real overlap from a handler that pushed nothing.
    """

    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    a_started, b_started, allow_b_end = (asyncio.Event() for _ in range(3))
    open_at_overlap = 0

    _start(handler, parent_run, "parent")

    async def drive_a() -> None:
        _start(handler, run_a, "A", parent_run_id=parent_run)
        a_started.set()
        await b_started.wait()
        _end(handler, run_a, "A")
        allow_b_end.set()

    async def drive_b() -> None:
        nonlocal open_at_overlap
        await a_started.wait()
        _start(handler, run_b, "B", parent_run_id=parent_run)
        open_at_overlap = len(handler._scope_handles)
        b_started.set()
        await allow_b_end.wait()
        _end(handler, run_b, "B")

    await asyncio.gather(asyncio.create_task(drive_a()), asyncio.create_task(drive_b()))
    _end(handler, parent_run, "parent")
    return open_at_overlap


async def test_strictly_nested_runs_close_cleanly(handler: NemoRelayCallbackHandler):
    """Control: the harness itself is sound when the runs nest properly."""

    baseline = nemo_relay.scope.get_handle()
    run_a, run_b = uuid4(), uuid4()

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        _start(handler, run_a, "A")
        _start(handler, run_b, "B", parent_run_id=run_a)
        assert len(handler._scope_handles) == 2, "the handler did not open both scopes"
        _end(handler, run_b, "B")
        _end(handler, run_a, "A")

    assert nemo_relay.scope.get_handle().uuid == baseline.uuid
    assert handler._scope_handles == {}


async def test_sibling_runs_are_parented_to_a_common_run(
    handler: NemoRelayCallbackHandler,
    subscribed_events: list[nemo_relay.Event],
):
    """Pin the topology the regression tests below depend on.

    ``scope.push`` parents to the current top of stack when no explicit handle is given,
    so callbacks that omit ``parent_run_id`` yield a nested chain that reaches the stack
    the same way but is not the sibling shape LangGraph produces. Without this, a fix
    that only handled true siblings would still look covered.
    """

    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    await _overlapping_sibling_runs(handler)
    try:
        nemo_relay.scope.pop(request)
    except RuntimeError:
        pass
    await nemo_relay.subscribers.flush_async()

    scopes = {}
    for event in subscribed_events:
        payload = event.to_dict()
        if payload.get("kind") == "scope":
            scopes[payload["name"]] = payload

    assert {"parent", "A", "B"} <= scopes.keys(), "the sibling runs did not open scopes"
    assert scopes["A"]["parent_uuid"] == scopes["parent"]["uuid"]
    assert scopes["B"]["parent_uuid"] == scopes["parent"]["uuid"]
    assert scopes["B"]["parent_uuid"] != scopes["A"]["uuid"], "B is nested under A"


async def test_a_deferred_close_records_when_the_run_actually_ended(
    handler: NemoRelayCallbackHandler,
    subscribed_events: list[nemo_relay.Event],
):
    """A close held back for ordering must not be dated to when it was replayed.

    A ends before B but can only be closed after it, so recording the replay time would
    inflate A's duration by however long B ran.

    Driven step by step so the moment A ended can be marked. Comparing A's close against
    B's would depend on the two callbacks landing in different clock ticks, which is not
    true on every platform.
    """

    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, parent_run, "parent")
    _start(handler, run_a, "A", parent_run_id=parent_run)
    _start(handler, run_b, "B", parent_run_id=parent_run)

    _end(handler, run_a, "A")
    assert len(handler._pending_pops) == 1, "A should be waiting on B"
    a_ended_by = datetime.datetime.now(datetime.timezone.utc)

    _end(handler, run_b, "B")
    _end(handler, parent_run, "parent")
    nemo_relay.scope.pop(request)
    await nemo_relay.subscribers.flush_async()

    stamps: dict[str, list[str]] = {}
    for event in subscribed_events:
        payload = event.to_dict()
        if payload.get("kind") == "scope":
            stamps.setdefault(str(payload["name"]), []).append(str(payload["timestamp"]))

    # Two events per scope, start then end; the second is the close.
    assert len(stamps.get("A", [])) == 2, "A did not open and close"
    assert len(stamps.get("B", [])) == 2, "B did not open and close"

    a_closed_at = datetime.datetime.fromisoformat(stamps["A"][1])
    b_closed_at = datetime.datetime.fromisoformat(stamps["B"][1])
    assert a_closed_at <= a_ended_by, "A's close was dated to its replay, not its end"
    assert a_closed_at <= b_closed_at, "A was recorded as outliving the run it waited on"


async def test_overlapping_sibling_runs_leave_the_outer_scope_closable(
    handler: NemoRelayCallbackHandler,
):
    """A sibling closing out of order must not break the enclosing scope.

    The out-of-order pop is the handler's problem to absorb. Today it is swallowed and
    the scope stays on the stack, so the caller's own scope raises on exit and an
    operation that fully succeeded is reported as failed.
    """

    # Pushed and popped explicitly so only the close is guarded: an error raised by the
    # scenario itself must fail the test rather than be mistaken for a teardown failure.
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    open_at_overlap = await _overlapping_sibling_runs(handler)

    teardown_error: BaseException | None = None
    try:
        nemo_relay.scope.pop(request)
    except RuntimeError as exc:
        teardown_error = exc

    assert open_at_overlap == _OPEN_AT_OVERLAP, "the sibling runs never actually overlapped"
    assert teardown_error is None


async def test_overlapping_sibling_runs_do_not_strand_a_scope(
    handler: NemoRelayCallbackHandler,
):
    """Every scope the handler opened must be closed by the time its runs have ended.

    ``_pop_scope`` drops the handle from ``_scope_handles`` before attempting the pop, so
    a rejected pop leaves a scope that is live on the stack and untracked by the handler:
    nothing can close it afterwards, and it stays current for everything that follows.
    """

    baseline = nemo_relay.scope.get_handle()

    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    open_at_overlap = await _overlapping_sibling_runs(handler)
    try:
        nemo_relay.scope.pop(request)
    except RuntimeError:
        # Closing the enclosing scope is asserted by the test above; this one is about
        # the stack that was left behind, which is why only the close is tolerated here.
        pass

    assert open_at_overlap == _OPEN_AT_OVERLAP, "the sibling runs never actually overlapped"
    assert handler._scope_handles == {}
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid


async def test_relay_still_reports_out_of_order_closes_the_way_the_handler_detects(
    handler: NemoRelayCallbackHandler,
):
    """Pin the message the deferral depends on.

    The runtime raises a plain ``RuntimeError`` for every pop failure, so the handler
    tells a retryable ordering rejection from a terminal one by text. If that text ever
    changes, deferral silently becomes "give up", so fail here instead.
    """

    from nemo_relay.integrations.langchain.callbacks import _OUT_OF_ORDER_CLOSE

    outer = nemo_relay.scope.push("outer", nemo_relay.ScopeType.Agent)
    inner = nemo_relay.scope.push("inner", nemo_relay.ScopeType.Agent)
    try:
        with pytest.raises(RuntimeError) as rejected:
            nemo_relay.scope.pop(outer)
        assert _OUT_OF_ORDER_CLOSE in str(rejected.value)
    finally:
        nemo_relay.scope.pop(inner)
        nemo_relay.scope.pop(outer)


async def test_a_close_is_queued_only_until_the_scopes_above_it_go(
    handler: NemoRelayCallbackHandler,
):
    """Observe the queue itself, rather than inferring it from the end state.

    Driven step by step so the state between callbacks is visible: an implementation that
    simply delayed every close would satisfy the end-state assertions elsewhere.
    """

    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, parent_run, "parent")
    _start(handler, run_a, "A", parent_run_id=parent_run)
    _start(handler, run_b, "B", parent_run_id=parent_run)

    _end(handler, run_a, "A")
    # A cannot close while B sits above it, so it must be held rather than abandoned.
    assert len(handler._pending_pops) == 1

    _end(handler, run_b, "B")
    # Closing B exposes A, so both leave the queue on this one callback.
    assert handler._pending_pops == []

    _end(handler, parent_run, "parent")
    nemo_relay.scope.pop(request)
    assert handler._scope_handles == {}


async def test_a_close_that_can_never_succeed_is_not_queued_forever(
    handler: NemoRelayCallbackHandler,
    caplog: pytest.LogCaptureFixture,
):
    """A terminal failure must be reported and dropped, not retried on every callback.

    Retrying it forever would grow the queue without bound and keep reporting a failure
    that cannot be fixed.
    """

    run_id = uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    _start(handler, run_id, "A")

    # Close A behind the handler's back, so its own close finds nothing to pop.
    nemo_relay.scope.pop(handler._scope_handles[run_id])

    with caplog.at_level("DEBUG"):
        _end(handler, run_id, "A")

    assert handler._pending_pops == [], "a hopeless close was queued for retry"
    assert any(record.levelname == "ERROR" for record in caplog.records), "a terminal close failure was not reported"
    nemo_relay.scope.pop(request)


async def test_a_queued_close_that_turns_terminal_leaves_the_queue(
    handler: NemoRelayCallbackHandler,
):
    """A queued close that can no longer succeed must be dropped on the retry path too.

    Scopes are per-context, so a replay attempted under a different stack can never find
    the handle. Keeping it queued would retry it on every later callback and grow the
    queue without bound.
    """

    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, parent_run, "parent")
    _start(handler, run_a, "A", parent_run_id=parent_run)
    _start(handler, run_b, "B", parent_run_id=parent_run)
    _end(handler, run_a, "A")
    assert len(handler._pending_pops) == 1, "A should be waiting on B"

    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        handler._drain_pending_pops()

    assert handler._pending_pops == [], "a close that can never succeed stayed queued"
