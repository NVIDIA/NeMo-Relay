# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-request scope stack isolation via ContextVar."""

import asyncio
import contextvars
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from async_helpers import resolve_async_result

import nemo_relay
from nemo_relay._native import capture_thread_scope_stack, restore_thread_scope_stack


@pytest.fixture
def restore_native_scope_stack():
    """Restore the test thread's native scope binding after each test."""
    binding = capture_thread_scope_stack()
    try:
        yield
    finally:
        restore_thread_scope_stack(binding)


def test_create_scope_stack_returns_scope_stack() -> None:
    """create_scope_stack returns a ScopeStack instance."""
    stack = nemo_relay.create_scope_stack()
    assert isinstance(stack, nemo_relay.ScopeStack)
    assert repr(stack) == "<ScopeStack>"


def test_propagation_context_installs_and_restores_a_scoped_stack() -> None:
    original = nemo_relay.get_scope_stack()
    root_uuid = str(uuid.uuid4())
    parent_uuid = str(uuid.uuid4())
    context = nemo_relay.PropagationContext(parent_uuid, root_uuid)
    stack = nemo_relay.create_scope_stack_from_propagation(context)

    with nemo_relay.use_scope_stack(stack):
        assert nemo_relay.get_scope_stack() is stack
        assert nemo_relay.scope.get_handle().uuid == parent_uuid

    assert nemo_relay.get_scope_stack() is original


def test_propagation_context_capture_and_constructor_validation() -> None:
    root_uuid = str(uuid.uuid4())

    with nemo_relay.scope.scope("sender", nemo_relay.ScopeType.Agent) as sender:
        rooted_default = nemo_relay.capture_propagation_context()
        rootless = nemo_relay.capture_rootless_propagation_context()
        rooted = nemo_relay.capture_propagation_context_with_root(root_uuid)

    assert rooted_default.version == 1
    assert rooted_default.root_uuid == sender.uuid
    assert rooted_default.parent_uuid == sender.uuid
    assert rootless.version == 1
    assert rootless.root_uuid is None
    assert rootless.parent_uuid == sender.uuid
    assert rootless.traceparent is None
    assert rootless.tracestate is None
    assert rooted.version == 1
    assert rooted.root_uuid == root_uuid
    assert rooted.parent_uuid == sender.uuid
    assert rooted.traceparent is None
    assert rooted.tracestate is None
    assert rooted.to_traceparent().split("-")[1] == root_uuid.replace("-", "")

    with pytest.raises(ValueError, match="invalid character"):
        nemo_relay.PropagationContext("not-a-uuid")
    with pytest.raises(ValueError, match="invalid character"):
        nemo_relay.PropagationContext(str(uuid.uuid4()), "not-a-uuid")
    with pytest.raises(ValueError, match="unsupported propagation context version 2; expected 1"):
        nemo_relay.PropagationContext(str(uuid.uuid4()), version=2)
    with pytest.raises(ValueError, match="invalid character"):
        nemo_relay.capture_propagation_context_with_root("not-a-uuid")


def test_propagation_context_json_round_trip_and_validation() -> None:
    context = nemo_relay.PropagationContext(str(uuid.uuid4()), str(uuid.uuid4()))

    encoded = context.to_json()
    decoded = nemo_relay.PropagationContext.from_json(encoded)

    assert decoded.version == context.version
    assert decoded.root_uuid == context.root_uuid
    assert decoded.parent_uuid == context.parent_uuid
    with pytest.raises(ValueError, match="invalid propagation context JSON"):
        nemo_relay.PropagationContext.from_json("not JSON")
    with pytest.raises(ValueError, match="unsupported propagation context version 2; expected 1"):
        nemo_relay.PropagationContext.from_json(f'{{"version":2,"parent_uuid":"{uuid.uuid4()}"}}')


def test_propagation_context_preserves_w3c_trace_context() -> None:
    context = nemo_relay.PropagationContext(
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        tracestate="vendor=value",
    )
    decoded = nemo_relay.PropagationContext.from_json(context.to_json())
    assert decoded.traceparent == context.traceparent
    assert decoded.tracestate == context.tracestate

    invalid = nemo_relay.PropagationContext.from_json(
        f'{{"version":1,"parent_uuid":"{uuid.uuid4()}","traceparent":"invalid","tracestate":"vendor=value"}}'
    )
    assert invalid.traceparent is None
    assert invalid.tracestate is None

    constructed_invalid = nemo_relay.PropagationContext(
        str(uuid.uuid4()),
        traceparent="invalid",
        tracestate="vendor=value",
    )
    assert constructed_invalid.traceparent is None
    assert constructed_invalid.tracestate is None


@pytest.mark.parametrize(
    "tracestate",
    [
        "vendor=bad\r\nX-Injected: yes",
        "vendor=one,vendor=two",
        ",".join(f"v{i}=x" for i in range(33)),
        "vendor=" + "z" * 300,
    ],
)
def test_invalid_tracestate_keeps_valid_traceparent(tracestate: str) -> None:
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    parent_uuid, root_uuid = str(uuid.uuid4()), str(uuid.uuid4())
    context = nemo_relay.PropagationContext(parent_uuid, root_uuid, traceparent=traceparent, tracestate=tracestate)
    decoded = nemo_relay.PropagationContext.from_json(context.to_json())
    for imported in (context, decoded):
        assert imported.traceparent == traceparent
        assert imported.tracestate is None
        assert imported.parent_uuid == parent_uuid
        assert imported.root_uuid == root_uuid
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack_from_propagation(imported)):
            assert nemo_relay.capture_traceparent() == traceparent


def test_rootless_and_root_parent_propagation_contexts_install_current_handle() -> None:
    parent_uuid = str(uuid.uuid4())
    rootless_stack = nemo_relay.create_scope_stack_from_propagation(nemo_relay.PropagationContext(parent_uuid))

    with nemo_relay.use_scope_stack(rootless_stack):
        assert nemo_relay.scope.get_handle().uuid == parent_uuid

    root_stack = nemo_relay.create_scope_stack_from_propagation(nemo_relay.PropagationContext(parent_uuid, parent_uuid))
    with nemo_relay.use_scope_stack(root_stack):
        assert nemo_relay.scope.get_handle().uuid == parent_uuid


async def test_fork_asyncio_context_isolates_siblings_and_preserves_parentage() -> None:
    marker = contextvars.ContextVar("fork-marker", default="missing")
    marker.set("parent")

    async def child(name, delay):
        assert marker.get() == "parent"
        with nemo_relay.scope.scope(name, nemo_relay.ScopeType.Function) as handle:
            await asyncio.sleep(delay)
            result = await nemo_relay.tools.execute(
                "forked-tool",
                {"name": name},
                lambda args: nemo_relay.ToolExecutionResult(args),
            )
            return handle, result, nemo_relay.get_scope_stack()

    with nemo_relay.scope.scope("fork-parent", nemo_relay.ScopeType.Agent) as parent:
        parent_stack = nemo_relay.get_scope_stack()
        nemo_relay.scope_local.register_tool_request(
            parent,
            "fork-parent-local",
            1,
            False,
            lambda name, args: {**args, "scope_local": True},
        )
        first = asyncio.create_task(child("first", 0.0), context=nemo_relay.fork_asyncio_context())
        second = asyncio.create_task(child("second", 0.01), context=nemo_relay.fork_asyncio_context())
        (first_handle, first_result, first_stack), (second_handle, second_result, second_stack) = await asyncio.gather(
            first, second
        )

        assert nemo_relay.get_scope_stack() is parent_stack
        assert nemo_relay.scope.get_handle().uuid == parent.uuid

    assert first_handle.parent_uuid == parent.uuid
    assert second_handle.parent_uuid == parent.uuid
    assert first_stack is not second_stack
    assert first_stack is not parent_stack
    assert second_stack is not parent_stack
    assert first_result.result == {"name": "first"}
    assert second_result.result == {"name": "second"}


async def test_fork_asyncio_context_works_from_implicit_root():
    parent_stack = nemo_relay.get_scope_stack()
    child_context = nemo_relay.fork_asyncio_context()

    async def child() -> nemo_relay.ScopeStack:
        return nemo_relay.get_scope_stack()

    child_stack = await asyncio.create_task(child(), context=child_context)
    assert child_stack is not parent_stack
    assert nemo_relay.get_scope_stack() is parent_stack


async def test_fork_asyncio_context_preserves_parent_trace_for_llm(subscribed_events):
    captured_traceparents = []

    async def provider(request):
        captured_traceparents.append(request.headers["traceparent"])
        return {"ok": True}

    async def child():
        return await nemo_relay.llm.execute(
            "forked-context-llm",
            nemo_relay.LLMRequest({}, {"prompt": "hello"}),
            provider,
        )

    with nemo_relay.scope.scope("forked-llm-turn", nemo_relay.ScopeType.Custom):
        parent_trace_id = nemo_relay.capture_traceparent().split("-")[1]
        task = asyncio.create_task(
            child(),
            context=nemo_relay.fork_asyncio_context(),
        )
        assert await task == {"ok": True}

    await nemo_relay.subscribers.flush_async()
    start = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent)
        and event.name == "forked-context-llm"
        and event.scope_category == "start"
    )
    expected = f"00-{parent_trace_id}-{start.uuid.replace('-', '')[-16:]}-01"
    assert captured_traceparents == [expected]


async def test_fork_asyncio_context_inside_managed_callback_preserves_event_parent(subscribed_events):
    async def child() -> tuple[str, str]:
        with nemo_relay.scope.scope("callback-fork-child", nemo_relay.ScopeType.Function) as handle:
            await asyncio.sleep(0)
            return handle.uuid, nemo_relay.capture_propagation_context().parent_uuid

    async def provider(_request):
        task = asyncio.create_task(child(), context=nemo_relay.fork_asyncio_context())
        child_uuid, captured_parent_uuid = await task
        return {"child_uuid": child_uuid, "captured_parent_uuid": captured_parent_uuid}

    with nemo_relay.scope.scope("callback-fork-outer", nemo_relay.ScopeType.Custom):
        result = await nemo_relay.llm.execute(
            "callback-fork-llm",
            nemo_relay.LLMRequest({}, {"prompt": "hello"}),
            provider,
        )

    await nemo_relay.subscribers.flush_async()
    llm_start = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent)
        and event.name == "callback-fork-llm"
        and event.scope_category == "start"
    )
    child_start = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent)
        and event.name == "callback-fork-child"
        and event.scope_category == "start"
    )
    assert child_start.parent_uuid == llm_start.uuid
    assert result == {
        "child_uuid": child_start.uuid,
        "captured_parent_uuid": child_start.uuid,
    }


def test_use_scope_stack_restores_a_previously_bound_native_stack(restore_native_scope_stack) -> None:
    previous = nemo_relay.create_scope_stack()
    replacement = nemo_relay.create_scope_stack()
    nemo_relay.set_thread_scope_stack(previous)
    previous_uuid = nemo_relay.scope.get_handle().uuid
    assert nemo_relay.scope_stack_active()

    with nemo_relay.use_scope_stack(replacement):
        assert nemo_relay.scope.get_handle().uuid != previous_uuid

    assert nemo_relay.scope.get_handle().uuid == previous_uuid
    assert nemo_relay.scope_stack_active()


def test_use_scope_stack_restores_nested_and_failing_contexts(restore_native_scope_stack) -> None:
    previous = nemo_relay.create_scope_stack()
    outer = nemo_relay.create_scope_stack()
    inner = nemo_relay.create_scope_stack()
    nemo_relay.set_thread_scope_stack(previous)
    previous_uuid = nemo_relay.scope.get_handle().uuid

    with nemo_relay.use_scope_stack(outer):
        outer_uuid = nemo_relay.scope.get_handle().uuid
        with pytest.raises(RuntimeError, match="expected failure"):
            with nemo_relay.use_scope_stack(inner):
                assert nemo_relay.scope.get_handle().uuid != outer_uuid
                raise RuntimeError("expected failure")
        assert nemo_relay.scope.get_handle().uuid == outer_uuid

    assert nemo_relay.scope.get_handle().uuid == previous_uuid


def test_get_scope_stack_returns_same_in_same_context() -> None:
    """get_scope_stack returns the same instance within the same context."""
    s1 = nemo_relay.get_scope_stack()
    s2 = nemo_relay.get_scope_stack()
    assert s1 is s2


def test_get_scope_stack_different_across_tasks() -> None:
    """Two asyncio tasks get different scope stacks."""
    results = {}

    async def task(name) -> None:
        # Each task gets its own context (asyncio.create_task copies ContextVar)
        # But since the ContextVar hasn't been set yet at fork time,
        # each task creates its own when get_scope_stack is first called.
        # We need to reset the ContextVar in each task to test isolation.
        nemo_relay._scope_stack_var.set(nemo_relay.create_scope_stack())
        stack = nemo_relay.get_scope_stack()
        results[name] = id(stack)

    async def main() -> None:
        t1 = asyncio.create_task(task("a"))
        t2 = asyncio.create_task(task("b"))
        await t1
        await t2

    asyncio.run(main())
    assert results["a"] != results["b"], "Tasks should have different scope stacks"


def test_scope_context_manager_closes_on_its_own_task_stack() -> None:
    """Concurrent scope exits restore the stack owned by the exiting task."""
    completed = []

    async def run_scope(name) -> None:
        token = nemo_relay._scope_stack_var.set(nemo_relay.create_scope_stack())
        try:
            with nemo_relay.scope.scope(name, nemo_relay.ScopeType.Agent):
                await asyncio.sleep(0)
            completed.append(name)
        finally:
            nemo_relay._scope_stack_var.reset(token)

    async def main() -> None:
        await asyncio.gather(run_scope("agent-a"), run_scope("agent-b"))

    asyncio.run(main())
    assert completed == ["agent-a", "agent-b"]


def test_concurrent_tool_lifecycle_uses_owning_task_stack(subscribed_events) -> None:
    """Tool lifecycle helpers preserve task-local middleware and event ancestry."""

    async def run_tool(owner):
        token = nemo_relay._scope_stack_var.set(nemo_relay.create_scope_stack())
        try:
            with nemo_relay.scope.scope(f"tool-scope-{owner}", nemo_relay.ScopeType.Agent) as handle:
                scope_local = nemo_relay.scope_local
                scope_local.register_tool_request(
                    handle,
                    f"tool-request-{owner}",
                    1,
                    False,
                    lambda name, args: {**args, "intercepted_by": owner},
                )
                scope_local.register_tool_conditional_execution(
                    handle,
                    f"tool-condition-{owner}",
                    1,
                    lambda name, args: None if args["owner"] == owner else "wrong task scope",
                )

                await asyncio.sleep(0)
                args = await resolve_async_result(nemo_relay.tools.request_intercepts("task-tool", {"owner": owner}))
                await resolve_async_result(nemo_relay.tools.conditional_execution("task-tool", args))

                manual_handle = nemo_relay.tools.call(f"manual-tool-{owner}", args)
                await asyncio.sleep(0)
                nemo_relay.tools.call_end(manual_handle, nemo_relay.ToolExecutionResult({"owner": owner}))

                result = await nemo_relay.tools.execute(
                    f"managed-tool-{owner}",
                    {"owner": owner},
                    lambda managed_args: nemo_relay.ToolExecutionResult(managed_args),
                )
                return handle.uuid, result
        finally:
            nemo_relay._scope_stack_var.reset(token)

    async def main():
        return await asyncio.gather(run_tool("agent-a"), run_tool("agent-b"))

    results = asyncio.run(main())
    nemo_relay.subscribers.flush()

    for owner, (scope_uuid, result) in zip(("agent-a", "agent-b"), results, strict=True):
        assert result.result == {"owner": owner, "intercepted_by": owner}
        for tool_name in (f"manual-tool-{owner}", f"managed-tool-{owner}"):
            tool_events = [
                event
                for event in subscribed_events
                if isinstance(event, nemo_relay.ScopeEvent) and event.category == "tool" and event.name == tool_name
            ]
            assert {event.scope_category for event in tool_events} == {"start", "end"}
            assert {event.parent_uuid for event in tool_events} == {scope_uuid}


def test_concurrent_llm_lifecycle_uses_owning_task_stack(subscribed_events) -> None:
    """LLM lifecycle helpers preserve task-local middleware and event ancestry."""

    async def run_llm(owner):
        token = nemo_relay._scope_stack_var.set(nemo_relay.create_scope_stack())
        try:
            with nemo_relay.scope.scope(f"llm-scope-{owner}", nemo_relay.ScopeType.Agent) as handle:
                scope_local = nemo_relay.scope_local

                def intercept(name, request, annotated):
                    return nemo_relay.LLMRequestInterceptOutcome(
                        nemo_relay.LLMRequest(request.headers, {**request.content, "intercepted_by": owner}),
                        annotated,
                    )

                scope_local.register_llm_request(handle, f"llm-request-{owner}", 1, False, intercept)
                scope_local.register_llm_conditional_execution(
                    handle,
                    f"llm-condition-{owner}",
                    1,
                    lambda request: None if request.content["owner"] == owner else "wrong task scope",
                )

                request = nemo_relay.LLMRequest({}, {"messages": [], "owner": owner})
                await asyncio.sleep(0)
                intercepted = await resolve_async_result(nemo_relay.llm.request_intercepts("task-llm", request))
                assert intercepted.request.content["intercepted_by"] == owner
                await resolve_async_result(nemo_relay.llm.conditional_execution(request))

                manual_handle = nemo_relay.llm.call(f"manual-llm-{owner}", request)
                await asyncio.sleep(0)
                nemo_relay.llm.call_end(manual_handle, {"owner": owner})

                result = await nemo_relay.llm.execute(
                    f"managed-llm-{owner}",
                    request,
                    lambda managed_request: managed_request.content,
                )

                collected = []

                async def stream_func(stream_request):
                    yield {
                        "owner": stream_request.content["owner"],
                        "intercepted_by": stream_request.content["intercepted_by"],
                    }

                stream = await nemo_relay.llm.stream_execute(
                    f"stream-llm-{owner}",
                    request,
                    stream_func,
                    collected.append,
                    lambda: {"count": len(collected)},
                )
                chunks = [chunk async for chunk in stream]
                return handle.uuid, result, chunks
        finally:
            nemo_relay._scope_stack_var.reset(token)

    async def main():
        return await asyncio.gather(run_llm("agent-a"), run_llm("agent-b"))

    results = asyncio.run(main())
    nemo_relay.subscribers.flush()

    for owner, (scope_uuid, result, chunks) in zip(("agent-a", "agent-b"), results, strict=True):
        expected = {"messages": [], "owner": owner, "intercepted_by": owner}
        assert result == expected
        assert chunks == [{"owner": owner, "intercepted_by": owner}]
        for llm_name in (f"manual-llm-{owner}", f"managed-llm-{owner}", f"stream-llm-{owner}"):
            llm_events = [
                event
                for event in subscribed_events
                if isinstance(event, nemo_relay.ScopeEvent) and event.category == "llm" and event.name == llm_name
            ]
            assert {event.scope_category for event in llm_events} == {"start", "end"}
            assert {event.parent_uuid for event in llm_events} == {scope_uuid}


def test_scope_stack_repr() -> None:
    """ScopeStack has a meaningful repr."""
    stack = nemo_relay.create_scope_stack()
    assert "<ScopeStack>" in repr(stack)


def test_scope_stack_active_false_by_default() -> None:
    """scope_stack_active returns False before any scope stack is initialized."""
    import threading

    result = {}

    def worker() -> None:
        # Fresh thread, no ContextVar set
        result["active"] = nemo_relay.scope_stack_active()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["active"] is False


def test_scope_stack_active_true_after_get_scope_stack() -> None:
    """scope_stack_active returns True after get_scope_stack is called (ContextVar path)."""
    import threading

    result = {}

    def worker() -> None:
        nemo_relay.get_scope_stack()
        result["active"] = nemo_relay.scope_stack_active()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["active"] is True


def test_scope_stack_active_true_after_set_thread() -> None:
    """scope_stack_active returns True after set_thread_scope_stack on a fresh thread."""
    import threading

    result = {}
    stack = nemo_relay.create_scope_stack()

    def worker() -> None:
        nemo_relay.set_thread_scope_stack(stack)
        result["active"] = nemo_relay.scope_stack_active()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["active"] is True


def test_propagate_scope_to_thread_fails_when_inactive() -> None:
    """propagate_scope_to_thread raises RuntimeError when no scope is active."""
    import threading

    result = {}

    def worker() -> None:
        try:
            nemo_relay.propagate_scope_to_thread()
            result["raised"] = False
        except RuntimeError:
            result["raised"] = True

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["raised"] is True


def test_propagate_scope_to_thread_returns_scope_stack() -> None:
    """propagate_scope_to_thread returns the current ScopeStack."""
    nemo_relay.get_scope_stack()
    stack = nemo_relay.propagate_scope_to_thread()
    assert isinstance(stack, nemo_relay.ScopeStack)


def test_propagate_scope_to_thread_cross_thread() -> None:
    """Propagated scope stack works on a worker thread."""
    import threading

    # Initialize scope stack and push a scope
    nemo_relay.get_scope_stack()
    handle = nemo_relay.scope.push("parent_scope", nemo_relay.ScopeType.Agent)

    propagated = nemo_relay.propagate_scope_to_thread()
    result = {}

    def worker() -> None:
        nemo_relay.set_thread_scope_stack(propagated)
        h = nemo_relay.scope.get_handle()
        result["name"] = h.name

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert result["name"] == "parent_scope"
    nemo_relay.scope.pop(handle)


def test_propagate_scope_to_thread_uses_native_active_stack_without_contextvar() -> None:
    """Verify propagate_scope_to_thread uses current_scope_stack().

    This covers the case where set_thread_scope_stack() initializes only the
    Rust thread-local and the Python ContextVar is not initialized, so
    propagate_scope_to_thread does not need get_scope_stack() in that case.
    """
    import threading

    result = {}
    stack = nemo_relay.create_scope_stack()

    def worker() -> None:
        nemo_relay.set_thread_scope_stack(stack)
        propagated = nemo_relay.propagate_scope_to_thread()
        result["active"] = nemo_relay.scope_stack_active()
        result["repr"] = repr(propagated)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert result["active"] is True
    assert result["repr"] == "<ScopeStack>"


def test_scope_stack_snapshot_absence_and_contextvar_only_root() -> None:
    """Inspection does not activate a stack and honors Python-only bindings."""

    def inspect_fresh_thread() -> None:
        assert nemo_relay.current_scope_stack_snapshot() is None
        assert nemo_relay.current_scope_stack_snapshot() is None
        assert not nemo_relay.scope_stack_active()
        stack = nemo_relay.create_scope_stack()
        token = nemo_relay._scope_stack_var.set(stack)
        try:
            snapshot = nemo_relay.current_scope_stack_snapshot()
            assert snapshot is not None
            assert len(snapshot.scopes) == 1
            assert snapshot.scopes[0].scope_type == nemo_relay.ScopeType.Agent
            uuid.UUID(snapshot.scopes[0].scope_id)
            from nemo_relay._native import scope_stack_active

            assert not scope_stack_active()
        finally:
            nemo_relay._scope_stack_var.reset(token)
        assert nemo_relay.current_scope_stack_snapshot() is None

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(inspect_fresh_thread).result()


def test_scope_stack_snapshot_orders_actual_handles_and_retains_values(restore_native_scope_stack) -> None:
    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        root = nemo_relay.current_scope_stack_snapshot()
        assert root is not None
        with nemo_relay.scope.scope("agent", nemo_relay.ScopeType.Agent) as agent:
            with nemo_relay.scope.scope("function", nemo_relay.ScopeType.Function) as function:
                with nemo_relay.scope.scope("tool", nemo_relay.ScopeType.Tool) as tool:
                    retained = nemo_relay.current_scope_stack_snapshot()
                    assert retained is not None
                    expected = [
                        (root.scopes[0].scope_id, nemo_relay.ScopeType.Agent),
                        (agent.uuid, nemo_relay.ScopeType.Agent),
                        (function.uuid, nemo_relay.ScopeType.Function),
                        (tool.uuid, nemo_relay.ScopeType.Tool),
                    ]
                    assert [(entry.scope_id, entry.scope_type) for entry in retained.scopes] == expected
                    repeated = nemo_relay.current_scope_stack_snapshot()
                    assert repeated is not None
                    assert [(entry.scope_id, entry.scope_type) for entry in repeated.scopes] == expected
                after_pop = nemo_relay.current_scope_stack_snapshot()
                assert after_pop is not None
                assert [entry.scope_id for entry in after_pop.scopes] == [entry[0] for entry in expected[:-1]]
                with nemo_relay.scope.scope("replacement", nemo_relay.ScopeType.Llm):
                    assert [(entry.scope_id, entry.scope_type) for entry in retained.scopes] == expected
                    returned_entries = retained.scopes
                    returned_entries.clear()
                    assert [(entry.scope_id, entry.scope_type) for entry in retained.scopes] == expected


async def test_scope_stack_snapshot_isolates_awaited_siblings_and_nested_bindings(restore_native_scope_stack) -> None:
    async def inspect_sibling(name: str) -> str:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            with nemo_relay.scope.scope(name, nemo_relay.ScopeType.Agent) as agent:
                await asyncio.sleep(0)
                before = nemo_relay.current_scope_stack_snapshot()
                assert before is not None
                assert before.scopes[-1].scope_id == agent.uuid
                with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
                    await asyncio.sleep(0)
                    nested = nemo_relay.current_scope_stack_snapshot()
                    assert nested is not None
                    assert len(nested.scopes) == 1
                    assert nested.scopes[0].scope_id != before.scopes[0].scope_id
                await asyncio.sleep(0)
                restored = nemo_relay.current_scope_stack_snapshot()
                assert restored is not None
                assert [entry.scope_id for entry in restored.scopes] == [entry.scope_id for entry in before.scopes]
                return agent.uuid

    first, second = await asyncio.gather(inspect_sibling("first"), inspect_sibling("second"))
    assert first != second


def test_scope_stack_snapshot_preserves_fixture_propagation_uuids(restore_native_scope_stack) -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "crates/types/tests/fixtures/scope_stack_snapshot.json"
    entries = json.loads(fixture_path.read_text())["scopes"]
    root_identifier, parent_identifier = entries[0]["scope_id"], entries[3]["scope_id"]
    stack = nemo_relay.create_scope_stack_from_propagation(
        nemo_relay.PropagationContext(parent_identifier, root_identifier)
    )
    with nemo_relay.use_scope_stack(stack):
        snapshot = nemo_relay.current_scope_stack_snapshot()
        assert snapshot is not None
        assert [(entry.scope_id, entry.scope_type) for entry in snapshot.scopes] == [
            (root_identifier, nemo_relay.ScopeType.Agent),
            (parent_identifier, nemo_relay.ScopeType.Unknown),
        ]


async def test_scope_stack_snapshot_preserves_parent_after_child_cancellation(restore_native_scope_stack) -> None:
    ready = asyncio.Event()

    async def child() -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            with nemo_relay.scope.scope("cancelled-agent", nemo_relay.ScopeType.Agent):
                ready.set()
                await asyncio.Future()

    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        before = nemo_relay.current_scope_stack_snapshot()
        assert before is not None
        task = asyncio.create_task(child())
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        after = nemo_relay.current_scope_stack_snapshot()
        assert after is not None
        assert [(entry.scope_id, entry.scope_type) for entry in after.scopes] == [
            (entry.scope_id, entry.scope_type) for entry in before.scopes
        ]


async def test_scope_stack_snapshot_correlates_managed_callbacks_with_reused_tool_call_ids(
    restore_native_scope_stack,
) -> None:
    """Each callback sees its enclosing Agent before and after awaited continuations."""

    async def run_agent(name: str) -> tuple[str, list[tuple[str, str | None, str]]]:
        observations: list[tuple[str, str | None, str]] = []
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            with nemo_relay.scope.scope(name, nemo_relay.ScopeType.Agent) as agent:

                def observe(boundary: str, tool_call_id: str | None = None) -> None:
                    snapshot = nemo_relay.current_scope_stack_snapshot()
                    assert snapshot is not None
                    nearest_agent = next(
                        entry for entry in reversed(snapshot.scopes) if entry.scope_type == nemo_relay.ScopeType.Agent
                    )
                    assert nearest_agent.scope_id == agent.uuid
                    observations.append((boundary, tool_call_id, nearest_agent.scope_id))

                async def llm_intercept(_name, request, _context, next_call):
                    observe("llm-before")
                    await asyncio.sleep(0)
                    response = await next_call(request)
                    observe("llm-after")
                    return response

                async def tool_intercept(context, next_call):
                    observe("tool-before", context.tool_call_id)
                    await asyncio.sleep(0)
                    response = await next_call(context.args)
                    observe("tool-after", context.tool_call_id)
                    return nemo_relay.ToolExecutionInterceptOutcome(response.result)

                nemo_relay.scope_local.register_llm_execution(agent, f"snapshot-llm-{name}", 1, llm_intercept)
                nemo_relay.scope_local.register_tool_execution(agent, f"snapshot-tool-{name}", 1, tool_intercept)
                with nemo_relay.scope.scope(f"workflow-{name}", nemo_relay.ScopeType.Function):
                    response = await nemo_relay.llm.execute(
                        "snapshot-llm",
                        nemo_relay.LLMRequest({}, {"messages": []}),
                        lambda request: {"tool_call_id": "call_1"},
                    )
                    for attempt in range(2):
                        result = await nemo_relay.tools.execute(
                            "snapshot-tool",
                            {"attempt": attempt},
                            lambda arguments: nemo_relay.ToolExecutionResult(arguments),
                            tool_call_id=response["tool_call_id"],
                        )
                        assert result.result == {"attempt": attempt}
                return agent.uuid, observations

    first, second = await asyncio.gather(run_agent("first"), run_agent("second"))
    assert first[0] != second[0]
    for identifier, observations in (first, second):
        assert observations == [
            ("llm-before", None, identifier),
            ("llm-after", None, identifier),
            ("tool-before", "call_1", identifier),
            ("tool-after", "call_1", identifier),
            ("tool-before", "call_1", identifier),
            ("tool-after", "call_1", identifier),
        ]


async def test_scope_stack_snapshot_preserves_agent_while_streaming_and_after_failure(
    restore_native_scope_stack,
) -> None:
    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
        with nemo_relay.scope.scope("stream-agent", nemo_relay.ScopeType.Agent) as agent:
            retained = nemo_relay.current_scope_stack_snapshot()
            assert retained is not None
            observed_identifiers: list[str] = []

            async def stream_response(request):
                for number in range(2):
                    await asyncio.sleep(0)
                    snapshot = nemo_relay.current_scope_stack_snapshot()
                    assert snapshot is not None
                    observed_identifiers.append(snapshot.scopes[-1].scope_id)
                    yield {"number": number}
                raise ValueError("expected streaming failure")

            collected = []
            received_chunks = []
            response_stream = await nemo_relay.llm.stream_execute(
                "snapshot-stream",
                nemo_relay.LLMRequest({}, {"messages": []}),
                stream_response,
                collected.append,
                lambda: {"count": len(collected)},
            )
            with pytest.raises(ValueError, match="expected streaming failure"):
                async for chunk in response_stream:
                    snapshot = nemo_relay.current_scope_stack_snapshot()
                    assert snapshot is not None
                    assert snapshot.scopes[-1].scope_id == agent.uuid
                    received_chunks.append(chunk)
            expected_chunks = [{"number": 0}, {"number": 1}]
            assert received_chunks == expected_chunks
            assert collected == expected_chunks
            assert observed_identifiers == [agent.uuid, agent.uuid]
            after = nemo_relay.current_scope_stack_snapshot()
            assert after is not None
            assert [(entry.scope_id, entry.scope_type) for entry in after.scopes] == [
                (entry.scope_id, entry.scope_type) for entry in retained.scopes
            ]
