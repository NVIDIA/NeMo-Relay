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
import contextlib
import datetime
import threading
import typing
from uuid import UUID, uuid4

import pytest

import nemo_relay

if typing.TYPE_CHECKING:
    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

# Handles the handler tracks while both siblings are open: the parent run plus A and B.
_OPEN_AT_OVERLAP = 3


@pytest.fixture(name="isolated_scope_stack", autouse=True)
def isolated_scope_stack_fixture():
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
    # This test is about parentage, not the close, which is asserted elsewhere.
    with contextlib.suppress(RuntimeError):
        nemo_relay.scope.pop(request)
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
    assert len(handler._completed) == 1, "A should be waiting on B"
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
    # Closing the enclosing scope is asserted by the test above; this one is about the
    # stack that was left behind, which is why only the close is tolerated here.
    with contextlib.suppress(RuntimeError):
        nemo_relay.scope.pop(request)

    assert open_at_overlap == _OPEN_AT_OVERLAP, "the sibling runs never actually overlapped"
    assert handler._scope_handles == {}
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid


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
    assert len(handler._completed) == 1

    _end(handler, run_b, "B")
    # Closing B exposes A, so both leave the queue on this one callback.
    assert handler._completed == {}

    _end(handler, parent_run, "parent")
    nemo_relay.scope.pop(request)
    assert handler._scope_handles == {}


async def test_a_scope_closed_out_of_band_does_not_block_other_closes(
    handler: NemoRelayCallbackHandler,
):
    """A completion whose scope is gone waits forever, but holds nothing else up.

    Closing only what is on top means there is no failed attempt to classify, and so no
    way to notice that a scope was closed by someone else and can never come back to the
    top. Its completion is retained. That is the trade for never guessing at an error
    message: the entry is inert, and everything around it still closes normally.

    Retention is therefore bounded by open runs plus any scope closed out of band, and
    an out-of-band close is not something the handler can cause on its own.
    """

    run_a, run_b = uuid4(), uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, run_a, "A")
    # Close A behind the handler's back, so its scope leaves the stack.
    nemo_relay.scope.pop(handler._scope_handles[run_a])
    _end(handler, run_a, "A")
    assert len(handler._completed) == 1, "A's completion should be retained"

    # A later run on the same stack is unaffected.
    _start(handler, run_b, "B")
    _end(handler, run_b, "B")

    # And the enclosing scope still closes, which is what actually matters.
    nemo_relay.scope.pop(request)
    assert handler._scope_handles == {}


async def test_a_queued_close_is_not_discarded_by_another_stacks_drain(
    handler: NemoRelayCallbackHandler,
):
    """A completion is only closed when its scope is the top of the active stack.

    That is what confines a shared handler to one stack at a time: draining while another
    stack is active cannot match this completion's handle, so it is left alone rather
    than consumed or discarded.
    """

    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, parent_run, "parent")
    _start(handler, run_a, "A", parent_run_id=parent_run)
    _start(handler, run_b, "B", parent_run_id=parent_run)
    _end(handler, run_a, "A")
    assert len(handler._completed) == 1, "A should be waiting on B"

    # A drain driven by an unrelated stack must leave A alone.
    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        handler._close_completed_scopes()
    assert len(handler._completed) == 1, "another stack's drain discarded A"

    # Back on its own stack, A still closes normally.
    _end(handler, run_b, "B")
    assert handler._completed == {}
    _end(handler, parent_run, "parent")
    nemo_relay.scope.pop(request)


async def test_one_handler_shared_across_two_scope_stacks(
    handler: NemoRelayCallbackHandler,
):
    """The reported failure: two concurrent invocations sharing one handler.

    Invocation 1 queues A behind B. Invocation 2 then closes a run of its own, and its
    drain must not touch A. If it does, A is dropped as terminal and invocation 1's
    request scope cannot close.
    """

    first_parent, first_a, first_b = uuid4(), uuid4(), uuid4()
    second_run = uuid4()
    first_stack = nemo_relay.create_scope_stack()
    second_stack = nemo_relay.create_scope_stack()

    with nemo_relay.use_scope_stack(first_stack):
        first_request = nemo_relay.scope.push("request-1", nemo_relay.ScopeType.Agent)
        _start(handler, first_parent, "parent-1")
        _start(handler, first_a, "A", parent_run_id=first_parent)
        _start(handler, first_b, "B", parent_run_id=first_parent)
        _end(handler, first_a, "A")
        assert len(handler._completed) == 1

    with nemo_relay.use_scope_stack(second_stack):
        second_request = nemo_relay.scope.push("request-2", nemo_relay.ScopeType.Agent)
        _start(handler, second_run, "X")
        _end(handler, second_run, "X")
        nemo_relay.scope.pop(second_request)

    with nemo_relay.use_scope_stack(first_stack):
        _end(handler, first_b, "B")
        _end(handler, first_parent, "parent-1")
        # The whole point: invocation 1 can still close its own request scope.
        nemo_relay.scope.pop(first_request)

    assert handler._completed == {}
    assert handler._scope_handles == {}


async def test_the_output_is_snapshotted_when_the_callback_fires(
    handler: NemoRelayCallbackHandler,
    subscribed_events: list[nemo_relay.Event],
):
    """A deferred close must report the output as it was when the run ended.

    The caller owns the mapping passed to the callback and may reuse or mutate it, so
    serializing at replay time would let telemetry depend on when the queue happened to
    drain.
    """

    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, parent_run, "parent")
    _start(handler, run_a, "A", parent_run_id=parent_run)
    _start(handler, run_b, "B", parent_run_id=parent_run)

    outputs = {"done": "A-at-callback"}
    handler.on_chain_end(outputs, run_id=run_a)
    assert len(handler._completed) == 1, "A should be waiting on B"

    outputs["done"] = "A-mutated-after-callback"

    _end(handler, run_b, "B")
    _end(handler, parent_run, "parent")
    nemo_relay.scope.pop(request)
    await nemo_relay.subscribers.flush_async()

    payloads = [event.to_dict() for event in subscribed_events]
    a_events = [p for p in payloads if p.get("name") == "A" and p.get("data")]
    assert a_events, "A emitted no event carrying output data"
    rendered = str(a_events[-1]["data"])
    assert "A-at-callback" in rendered
    assert "A-mutated-after-callback" not in rendered


def test_a_run_completed_on_a_worker_thread_still_closes(
    handler: NemoRelayCallbackHandler,
):
    """A handler reused by worker-thread callbacks must still close its scopes.

    Propagating a stack to a thread hands back a different ``ScopeStack`` wrapper for the
    same stack, and the two cannot be correlated through the public API. Anything that
    identified the owning stack by object would stop draining here and strand every
    scope. Ownership is established by handle uuid instead, which crosses threads.
    """

    run_id = uuid4()
    stack = nemo_relay.get_scope_stack()
    baseline = nemo_relay.scope.get_handle()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    # Start on this context, finish on a worker thread bound to the same stack.
    _start(handler, run_id, "A")

    failures: list[BaseException] = []

    def finish_on_thread() -> None:
        try:
            nemo_relay.set_thread_scope_stack(stack)
            _end(handler, run_id, "A")
        except BaseException as exc:  # noqa: BLE001 - surfaced through ``failures``
            failures.append(exc)

    worker = threading.Thread(target=finish_on_thread)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "the worker thread did not finish"
    assert failures == []
    assert handler._completed == {}, "the run's scope was never closed"
    nemo_relay.scope.pop(request)
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid


def test_completing_a_run_without_a_stack_does_not_create_one(
    handler: NemoRelayCallbackHandler,
):
    """Reading the top scope must not leave a stack behind in a context that had none.

    ``get_handle`` creates one on demand, so an unguarded read would materialise an empty
    stack in every context that happens to complete a run it does not own.
    """

    run_id = uuid4()
    _start(handler, run_id, "A")

    created: list[bool] = []

    def finish_without_a_stack() -> None:
        # A worker thread with no stack bound: nothing here can be closed.
        _end(handler, run_id, "A")
        created.append(nemo_relay.scope_stack_active())

    worker = threading.Thread(target=finish_without_a_stack)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "the worker thread did not finish"
    assert created == [False], "completing a run created a scope stack out of nothing"


async def test_two_handlers_on_one_stack_only_close_their_own_scopes(
    handler: NemoRelayCallbackHandler,
):
    """Ownership is per handler, not per stack.

    Applications can attach more than one callback handler. Each closes only the runs it
    opened; a scope on top that belongs to the other handler must stop the drain rather
    than be closed by whoever happens to be draining.
    """

    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

    other = NemoRelayCallbackHandler()
    mine, theirs = uuid4(), uuid4()
    baseline = nemo_relay.scope.get_handle()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, mine, "mine")
    _start(other, theirs, "theirs")

    # ``mine`` sits under ``theirs``; my drain must not close a scope I do not own.
    _end(handler, mine, "mine")
    assert len(handler._completed) == 1, "my completion should be waiting"
    assert other._completed == {}
    assert nemo_relay.scope.get_handle().name == "theirs", "someone closed another's scope"

    _end(other, theirs, "theirs")
    assert other._completed == {}
    # Closing theirs exposes mine, but only my own handler can close it.
    handler._close_completed_scopes()
    assert handler._completed == {}

    nemo_relay.scope.pop(request)
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid


def test_the_handler_lock_is_reentrant(handler: NemoRelayCallbackHandler):
    """Closing a scope can re-enter the handler, so the lock has to be reentrant.

    A subscriber or middleware reacting to a scope end may issue another callback on the
    same thread while the lock is held. Probed with a timed acquire rather than by
    actually re-entering, so a non-reentrant lock fails the test instead of deadlocking
    the suite.
    """

    with handler._lock:
        reacquired = handler._lock.acquire(timeout=1)
        if reacquired:
            handler._lock.release()

    assert reacquired, "handler lock is not reentrant; a re-entrant close would deadlock"


async def test_a_transient_pop_failure_keeps_the_completion_for_a_later_close(
    handler: NemoRelayCallbackHandler,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """The runtime can refuse a close before it mutates the stack.

    The completion is the only record able to close that scope, so dropping it on the
    first refusal would strand the scope exactly as the original bug did.
    """

    run_id = uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    _start(handler, run_id, "A")

    real_pop = nemo_relay.scope.pop
    attempts = {"count": 0}

    def flaky_pop(handle: nemo_relay.ScopeHandle, **kwargs: typing.Any) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("runtime busy")
        real_pop(handle, **kwargs)

    monkeypatch.setattr(nemo_relay.scope, "pop", flaky_pop)

    with caplog.at_level("ERROR"):
        _end(handler, run_id, "A")

    assert len(handler._completed) == 1, "a refused close discarded its only record"
    assert any(record.levelname == "ERROR" for record in caplog.records)

    # The next drain closes it, and the enclosing scope is closable again.
    handler._close_completed_scopes()
    assert handler._completed == {}
    nemo_relay.scope.pop(request)


async def test_a_pop_value_error_keeps_the_original_completion(
    handler: NemoRelayCallbackHandler,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """A non-output validation failure must not trigger an output-less retry."""
    run_id = uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    _start(handler, run_id, "A")

    real_pop = nemo_relay.scope.pop
    attempted_outputs: list[nemo_relay.Json | None] = []

    def reject_pop(handle: nemo_relay.ScopeHandle, **kwargs: typing.Any) -> None:
        attempted_outputs.append(kwargs.get("output"))
        raise ValueError("timestamp datetime must be timezone-aware")

    monkeypatch.setattr(nemo_relay.scope, "pop", reject_pop)

    with caplog.at_level("ERROR"):
        _end(handler, run_id, "A")

    assert attempted_outputs == [{"done": "A"}]
    assert len(handler._completed) == 1
    assert next(iter(handler._completed.values())).output == {"done": "A"}

    monkeypatch.setattr(nemo_relay.scope, "pop", real_pop)
    handler._close_completed_scopes()
    assert handler._completed == {}
    nemo_relay.scope.pop(request)


async def test_an_unserializable_output_does_not_strand_the_scope(
    handler: NemoRelayCallbackHandler,
    caplog: pytest.LogCaptureFixture,
):
    """Serializing the output walks caller data and can fail on its own.

    A cyclic output raises ``RecursionError``. If that escaped after the run had been
    dropped from the handler's maps, nothing would be left able to close the scope.
    """

    run_id = uuid4()
    baseline = nemo_relay.scope.get_handle()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    _start(handler, run_id, "A")

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    with caplog.at_level("ERROR"):
        handler.on_chain_end(cyclic, run_id=run_id)

    # The scope closed; only the output payload was lost.
    assert handler._completed == {}
    assert any(record.levelname == "ERROR" for record in caplog.records)
    nemo_relay.scope.pop(request)
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid


async def test_invalid_output_does_not_block_the_completion_queue(
    handler: NemoRelayCallbackHandler,
    caplog: pytest.LogCaptureFixture,
):
    """A permanently invalid output cannot strand every completed scope below it."""
    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, parent_run, "parent")
    _start(handler, run_a, "A", parent_run_id=parent_run)
    _start(handler, run_b, "B", parent_run_id=parent_run)

    _end(handler, run_a, "A")
    assert len(handler._completed) == 1

    with caplog.at_level("ERROR"):
        handler.on_chain_end({"unsupported": object()}, run_id=run_b)

    assert handler._completed == {}
    assert nemo_relay.scope.get_handle().name == "parent"
    assert any(record.levelname == "ERROR" for record in caplog.records)

    _end(handler, parent_run, "parent")
    assert handler._completed == {}
    nemo_relay.scope.pop(request)


async def test_pydantic_output_is_serialized_before_scope_close(
    handler: NemoRelayCallbackHandler,
    subscribed_events: list[nemo_relay.Event],
):
    """Structured outputs retain their fields without blocking scope closure."""
    from pydantic import BaseModel

    class StructuredResult(BaseModel):
        answer: str
        confidence: float
        request_id: UUID

    run_id = uuid4()
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    _start(handler, run_id, "A")

    handler.on_chain_end(
        {
            "structured_response": StructuredResult(
                answer="complete",
                confidence=0.95,
                request_id=request_id,
            )
        },
        run_id=run_id,
    )

    await nemo_relay.subscribers.flush_async()
    scope_end = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent) and event.scope_category == "end" and event.name == "A"
    )
    assert scope_end.data == {
        "structured_response": {
            "answer": "complete",
            "confidence": 0.95,
            "request_id": str(request_id),
        }
    }
    assert handler._completed == {}
    nemo_relay.scope.pop(request)


async def test_a_failed_run_completes_its_scope_the_same_way(
    handler: NemoRelayCallbackHandler,
):
    """``on_chain_error`` completes a run exactly as ``on_chain_end`` does.

    A failed sibling that finished out of order strands its scope just as readily as a
    successful one, so the error path needs the same treatment.
    """

    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    baseline = nemo_relay.scope.get_handle()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, parent_run, "parent")
    _start(handler, run_a, "A", parent_run_id=parent_run)
    _start(handler, run_b, "B", parent_run_id=parent_run)

    handler.on_chain_error(RuntimeError("boom"), run_id=run_a)
    assert len(handler._completed) == 1, "a failed run should wait for B like any other"

    _end(handler, run_b, "B")
    _end(handler, parent_run, "parent")
    nemo_relay.scope.pop(request)

    assert handler._completed == {}
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid


async def test_two_concurrent_invocations_share_one_handler(
    handler: NemoRelayCallbackHandler,
):
    """Two asyncio tasks, each on its own stack, interleaved through one handler.

    Invocation 2 is active while invocation 1 has a completion waiting, so anything that
    drained without regard to which scope is on top would consume it.
    """

    one_parked, two_done = asyncio.Event(), asyncio.Event()
    outcomes: dict[str, BaseException | None] = {}

    async def invocation_one() -> None:
        parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            request = nemo_relay.scope.push("request-1", nemo_relay.ScopeType.Agent)
            _start(handler, parent_run, "parent-1")
            _start(handler, run_a, "A", parent_run_id=parent_run)
            _start(handler, run_b, "B", parent_run_id=parent_run)
            _end(handler, run_a, "A")
            one_parked.set()
            await two_done.wait()
            _end(handler, run_b, "B")
            _end(handler, parent_run, "parent-1")
            try:
                nemo_relay.scope.pop(request)
            except RuntimeError as exc:
                outcomes["one"] = exc
            else:
                outcomes["one"] = None

    async def invocation_two() -> None:
        run_x = uuid4()
        await one_parked.wait()
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            request = nemo_relay.scope.push("request-2", nemo_relay.ScopeType.Agent)
            _start(handler, run_x, "X")
            _end(handler, run_x, "X")
            try:
                nemo_relay.scope.pop(request)
            except RuntimeError as exc:
                outcomes["two"] = exc
            else:
                outcomes["two"] = None
        two_done.set()

    await asyncio.gather(asyncio.create_task(invocation_one()), asyncio.create_task(invocation_two()))

    assert outcomes == {"one": None, "two": None}
    assert handler._completed == {}
    assert handler._scope_handles == {}


async def test_a_propagated_stack_does_not_consume_another_stacks_completion(
    handler: NemoRelayCallbackHandler,
):
    """A propagated stack seeds a scope carrying an existing scope's uuid.

    ``create_scope_stack_from_propagation`` rebuilds a parent scope from a captured
    context, so the same uuid legitimately appears on a second stack. Matching a
    completion on uuid alone closes that stand-in, discards the record, and strands the
    real scope — the original defect, reached through propagation instead of ordering.
    """

    import uuid as uuid_module

    parent_run, run_a, run_b = uuid4(), uuid4(), uuid4()
    baseline = nemo_relay.scope.get_handle()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)

    _start(handler, parent_run, "parent")
    _start(handler, run_a, "A", parent_run_id=parent_run)
    # Captured while A is on top, so the propagated stack's parent carries A's uuid.
    context = nemo_relay.capture_propagation_context_with_root(str(uuid_module.uuid4()))
    _start(handler, run_b, "B", parent_run_id=parent_run)

    _end(handler, run_a, "A")
    assert len(handler._completed) == 1, "A should be waiting on B"

    propagated = nemo_relay.create_scope_stack_from_propagation(context)
    with nemo_relay.use_scope_stack(propagated):
        assert nemo_relay.scope.get_handle().uuid == context.parent_uuid
        handler._close_completed_scopes()

    assert len(handler._completed) == 1, "a propagated stack consumed another stack's completion"

    # The real scope is still closable, and the enclosing scope with it.
    _end(handler, run_b, "B")
    _end(handler, parent_run, "parent")
    assert handler._completed == {}
    nemo_relay.scope.pop(request)
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid


async def test_propagated_stand_ins_keep_their_reserved_names(
    handler: NemoRelayCallbackHandler,
):
    """Pin the naming the uuid+name check depends on.

    A propagated stack rebuilds scopes carrying an existing uuid, so the name is what
    separates a stand-in from the scope it represents. Those names are fixed in the
    runtime (``propagated-parent``/``propagated-root`` in
    ``crates/core/src/api/runtime/scope_stack.rs``); if either changes, matching silently
    starts closing the wrong scope, so fail here instead.

    Propagating an already-propagated stack yields stand-ins identical in both uuid and
    name, which still cannot be confused with a run this handler opened.
    """

    import uuid as uuid_module

    run_id = uuid4()
    request = nemo_relay.scope.push("request", nemo_relay.ScopeType.Agent)
    _start(handler, run_id, "A")
    opened = handler._scope_handles[run_id]

    with_root = nemo_relay.capture_propagation_context_with_root(str(uuid_module.uuid4()))
    first = nemo_relay.create_scope_stack_from_propagation(with_root)
    with nemo_relay.use_scope_stack(first):
        stand_in = nemo_relay.scope.get_handle()
        assert stand_in.uuid == opened.uuid, "the stand-in should carry the same uuid"
        assert stand_in.name == "propagated-parent"
        again = nemo_relay.capture_propagation_context_with_root(str(uuid_module.uuid4()))

    # Propagating the propagation: identical uuid and name, still not our scope.
    second = nemo_relay.create_scope_stack_from_propagation(again)
    with nemo_relay.use_scope_stack(second):
        repeated = nemo_relay.scope.get_handle()
        assert (repeated.uuid, repeated.name) == (stand_in.uuid, stand_in.name)
        assert (repeated.uuid, repeated.name) != (opened.uuid, opened.name)

    # And the branch where the propagated parent is the root.
    as_root = nemo_relay.create_scope_stack_from_propagation(nemo_relay.PropagationContext(opened.uuid, opened.uuid))
    with nemo_relay.use_scope_stack(as_root):
        root_stand_in = nemo_relay.scope.get_handle()
        assert root_stand_in.name == "propagated-root"
        assert (root_stand_in.uuid, root_stand_in.name) != (opened.uuid, opened.name)

    _end(handler, run_id, "A")
    nemo_relay.scope.pop(request)
