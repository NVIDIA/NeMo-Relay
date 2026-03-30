# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-request scope stack isolation via ContextVar."""

import asyncio

import nat_nexus


def test_create_scope_stack_returns_scope_stack():
    """create_scope_stack returns a ScopeStack instance."""
    stack = nat_nexus.create_scope_stack()
    assert isinstance(stack, nat_nexus.ScopeStack)
    assert repr(stack) == "<ScopeStack>"


def test_get_scope_stack_returns_same_in_same_context():
    """get_scope_stack returns the same instance within the same context."""
    s1 = nat_nexus.get_scope_stack()
    s2 = nat_nexus.get_scope_stack()
    assert s1 is s2


def test_get_scope_stack_different_across_tasks():
    """Two asyncio tasks get different scope stacks."""
    results = {}

    async def task(name):
        # Each task gets its own context (asyncio.create_task copies ContextVar)
        # But since the ContextVar hasn't been set yet at fork time,
        # each task creates its own when get_scope_stack is first called.
        # We need to reset the ContextVar in each task to test isolation.
        nat_nexus._scope_stack_var.set(nat_nexus.create_scope_stack())
        stack = nat_nexus.get_scope_stack()
        results[name] = id(stack)

    async def main():
        t1 = asyncio.create_task(task("a"))
        t2 = asyncio.create_task(task("b"))
        await t1
        await t2

    asyncio.run(main())
    assert results["a"] != results["b"], "Tasks should have different scope stacks"


def test_scope_stack_repr():
    """ScopeStack has a meaningful repr."""
    stack = nat_nexus.create_scope_stack()
    assert "<ScopeStack>" in repr(stack)


def test_scope_stack_active_false_by_default():
    """scope_stack_active returns False before any scope stack is initialized."""
    import threading

    result = {}

    def worker():
        # Fresh thread, no ContextVar set
        result["active"] = nat_nexus.scope_stack_active()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["active"] is False


def test_scope_stack_active_true_after_get_scope_stack():
    """scope_stack_active returns True after get_scope_stack is called (ContextVar path)."""
    import threading

    result = {}

    def worker():
        nat_nexus.get_scope_stack()
        result["active"] = nat_nexus.scope_stack_active()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["active"] is True


def test_scope_stack_active_true_after_set_thread():
    """scope_stack_active returns True after set_thread_scope_stack on a fresh thread."""
    import threading

    result = {}
    stack = nat_nexus.create_scope_stack()

    def worker():
        nat_nexus.set_thread_scope_stack(stack)
        result["active"] = nat_nexus.scope_stack_active()

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
            nat_nexus.propagate_scope_to_thread()
            result["raised"] = False
        except RuntimeError:
            result["raised"] = True

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["raised"] is True


def test_propagate_scope_to_thread_returns_scope_stack():
    """propagate_scope_to_thread returns the current ScopeStack."""
    nat_nexus.get_scope_stack()
    stack = nat_nexus.propagate_scope_to_thread()
    assert isinstance(stack, nat_nexus.ScopeStack)


def test_propagate_scope_to_thread_cross_thread():
    """Propagated scope stack works on a worker thread."""
    import threading

    # Initialize scope stack and push a scope
    nat_nexus.get_scope_stack()
    handle = nat_nexus.scope.push("parent_scope", nat_nexus.ScopeType.Agent)

    propagated = nat_nexus.propagate_scope_to_thread()
    result = {}

    def worker():
        nat_nexus.set_thread_scope_stack(propagated)
        h = nat_nexus.scope.get_handle()
        result["name"] = h.name

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert result["name"] == "parent_scope"
    nat_nexus.scope.pop(handle)
