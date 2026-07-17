# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-request scope stack isolation via ContextVar."""

import asyncio

import nemo_relay


def test_create_scope_stack_returns_scope_stack():
    """create_scope_stack returns a ScopeStack instance."""
    stack = nemo_relay.create_scope_stack()
    assert isinstance(stack, nemo_relay.ScopeStack)
    assert repr(stack) == "<ScopeStack>"


def test_get_scope_stack_returns_same_in_same_context():
    """get_scope_stack returns the same instance within the same context."""
    s1 = nemo_relay.get_scope_stack()
    s2 = nemo_relay.get_scope_stack()
    assert s1 is s2


def test_get_scope_stack_different_across_tasks():
    """Two asyncio tasks get different scope stacks."""
    results = {}

    async def task(name):
        # Each task gets its own context (asyncio.create_task copies ContextVar)
        # But since the ContextVar hasn't been set yet at fork time,
        # each task creates its own when get_scope_stack is first called.
        # We need to reset the ContextVar in each task to test isolation.
        nemo_relay._scope_stack_var.set(nemo_relay.create_scope_stack())
        stack = nemo_relay.get_scope_stack()
        results[name] = id(stack)

    async def main():
        t1 = asyncio.create_task(task("a"))
        t2 = asyncio.create_task(task("b"))
        await t1
        await t2

    asyncio.run(main())
    assert results["a"] != results["b"], "Tasks should have different scope stacks"


def test_scope_context_manager_closes_on_its_own_task_stack():
    """Concurrent scope exits restore the stack owned by the exiting task."""
    completed = []

    async def run_scope(name):
        token = nemo_relay._scope_stack_var.set(nemo_relay.create_scope_stack())
        try:
            with nemo_relay.scope.scope(name, nemo_relay.ScopeType.Agent):
                await asyncio.sleep(0)
            completed.append(name)
        finally:
            nemo_relay._scope_stack_var.reset(token)

    async def main():
        await asyncio.gather(run_scope("agent-a"), run_scope("agent-b"))

    asyncio.run(main())
    assert completed == ["agent-a", "agent-b"]


def test_concurrent_tool_lifecycle_uses_owning_task_stack(subscribed_events):
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
                args = nemo_relay.tools.request_intercepts("task-tool", {"owner": owner})
                nemo_relay.tools.conditional_execution("task-tool", args)

                manual_handle = nemo_relay.tools.call(f"manual-tool-{owner}", args)
                await asyncio.sleep(0)
                nemo_relay.tools.call_end(manual_handle, {"owner": owner})

                result = await nemo_relay.tools.execute(
                    f"managed-tool-{owner}",
                    {"owner": owner},
                    lambda managed_args: managed_args,
                )
                return handle.uuid, result
        finally:
            nemo_relay._scope_stack_var.reset(token)

    async def main():
        return await asyncio.gather(run_tool("agent-a"), run_tool("agent-b"))

    results = asyncio.run(main())
    nemo_relay.subscribers.flush()

    for owner, (scope_uuid, result) in zip(("agent-a", "agent-b"), results, strict=True):
        assert result == {"owner": owner, "intercepted_by": owner}
        for tool_name in (f"manual-tool-{owner}", f"managed-tool-{owner}"):
            tool_events = [
                event
                for event in subscribed_events
                if isinstance(event, nemo_relay.ScopeEvent) and event.category == "tool" and event.name == tool_name
            ]
            assert {event.scope_category for event in tool_events} == {"start", "end"}
            assert {event.parent_uuid for event in tool_events} == {scope_uuid}


def test_concurrent_llm_lifecycle_uses_owning_task_stack(subscribed_events):
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
                intercepted = nemo_relay.llm.request_intercepts("task-llm", request)
                assert intercepted.request.content["intercepted_by"] == owner
                nemo_relay.llm.conditional_execution(request)

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


def test_scope_stack_repr():
    """ScopeStack has a meaningful repr."""
    stack = nemo_relay.create_scope_stack()
    assert "<ScopeStack>" in repr(stack)


def test_scope_stack_active_false_by_default():
    """scope_stack_active returns False before any scope stack is initialized."""
    import threading

    result = {}

    def worker():
        # Fresh thread, no ContextVar set
        result["active"] = nemo_relay.scope_stack_active()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["active"] is False


def test_scope_stack_active_true_after_get_scope_stack():
    """scope_stack_active returns True after get_scope_stack is called (ContextVar path)."""
    import threading

    result = {}

    def worker():
        nemo_relay.get_scope_stack()
        result["active"] = nemo_relay.scope_stack_active()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["active"] is True


def test_scope_stack_active_true_after_set_thread():
    """scope_stack_active returns True after set_thread_scope_stack on a fresh thread."""
    import threading

    result = {}
    stack = nemo_relay.create_scope_stack()

    def worker():
        nemo_relay.set_thread_scope_stack(stack)
        result["active"] = nemo_relay.scope_stack_active()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["active"] is True


def test_propagate_scope_to_thread_fails_when_inactive():
    """propagate_scope_to_thread raises RuntimeError when no scope is active."""
    import threading

    result = {}

    def worker():
        try:
            nemo_relay.propagate_scope_to_thread()
            result["raised"] = False
        except RuntimeError:
            result["raised"] = True

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["raised"] is True


def test_propagate_scope_to_thread_returns_scope_stack():
    """propagate_scope_to_thread returns the current ScopeStack."""
    nemo_relay.get_scope_stack()
    stack = nemo_relay.propagate_scope_to_thread()
    assert isinstance(stack, nemo_relay.ScopeStack)


def test_propagate_scope_to_thread_cross_thread():
    """Propagated scope stack works on a worker thread."""
    import threading

    # Initialize scope stack and push a scope
    nemo_relay.get_scope_stack()
    handle = nemo_relay.scope.push("parent_scope", nemo_relay.ScopeType.Agent)

    propagated = nemo_relay.propagate_scope_to_thread()
    result = {}

    def worker():
        nemo_relay.set_thread_scope_stack(propagated)
        h = nemo_relay.scope.get_handle()
        result["name"] = h.name

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert result["name"] == "parent_scope"
    nemo_relay.scope.pop(handle)


def test_propagate_scope_to_thread_uses_native_active_stack_without_contextvar():
    """Verify propagate_scope_to_thread uses current_scope_stack().

    This covers the case where set_thread_scope_stack() initializes only the
    Rust thread-local and the Python ContextVar is not initialized, so
    propagate_scope_to_thread does not need get_scope_stack() in that case.
    """
    import threading

    result = {}
    stack = nemo_relay.create_scope_stack()

    def worker():
        nemo_relay.set_thread_scope_stack(stack)
        propagated = nemo_relay.propagate_scope_to_thread()
        result["active"] = nemo_relay.scope_stack_active()
        result["repr"] = repr(propagated)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert result["active"] is True
    assert result["repr"] == "<ScopeStack>"
