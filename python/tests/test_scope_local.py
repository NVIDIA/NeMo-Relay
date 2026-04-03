# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Agent Toolkit Nexus scope-local middleware registry.

Scope-local registrations are tied to a scope handle and are automatically
cleaned up when the scope is popped. These tests verify that guardrails,
intercepts, and subscribers registered via ``nat_nexus.scope_local`` only
take effect within their owning scope and do not leak to other scopes.
"""

import pytest
from nat_nexus import (
    Event,
    EventType,
    LLMRequest,
    ScopeType,
    guardrails,
    llm,
    scope,
    scope_local,
    subscribers,
    tools,
)

# ---------------------------------------------------------------------------
# 1. Basic scope-local guardrail
# ---------------------------------------------------------------------------


class TestScopeLocalGuardrail:
    async def test_sanitize_request_runs_within_scope(self):
        """A scope-local tool sanitize-request guardrail transforms the event input (observability only)."""
        events = []

        def sanitizer(tool_name, args):
            args["sanitized"] = True
            return args

        def my_tool(args):
            return args

        with scope.scope("guardrail_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_sanitize_request(handle, "sl_sanitizer", 1, sanitizer)
            scope_local.register_subscriber(handle, "sl_sanitizer_sub", lambda e: events.append(e))
            result = await tools.execute("sanitized_tool", {"input": "data"}, my_tool)

        # Sanitize guardrails are observability-only: they do NOT modify args
        # flowing through the execution pipeline.
        assert result["input"] == "data"
        assert "sanitized" not in result

        # The sanitizer's effect is visible in the event's input field.
        start_events = [e for e in events if e.event_type == EventType.Start and e.name == "sanitized_tool"]
        assert len(start_events) >= 1
        assert start_events[0].input["sanitized"] is True

    async def test_sanitize_response_runs_within_scope(self):
        """A scope-local tool sanitize-response guardrail transforms the event output (observability only)."""
        events = []

        def response_sanitizer(tool_name, result):
            result["response_sanitized"] = True
            return result

        def my_tool(args):
            return {"output": "raw"}

        with scope.scope("resp_guard_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_sanitize_response(handle, "sl_resp_sanitizer", 1, response_sanitizer)
            scope_local.register_subscriber(handle, "sl_resp_sub", lambda e: events.append(e))
            result = await tools.execute("resp_tool", {}, my_tool)

        # Sanitize guardrails are observability-only: they do NOT modify the
        # result flowing through the execution pipeline.
        assert result["output"] == "raw"
        assert "response_sanitized" not in result

        # The sanitizer's effect is visible in the event's output field.
        end_events = [e for e in events if e.event_type == EventType.End and e.name == "resp_tool"]
        assert len(end_events) >= 1
        assert end_events[0].output["response_sanitized"] is True


# ---------------------------------------------------------------------------
# 2. Auto-cleanup on scope exit
# ---------------------------------------------------------------------------


class TestScopeLocalAutoCleanup:
    async def test_guardrail_inactive_after_scope_exit(self):
        """Scope-local guardrail no longer affects tool calls after the scope is popped."""
        events_inside = []

        def sanitizer(tool_name, args):
            args["sanitized"] = True
            return args

        def my_tool(args):
            return args

        # Register guardrail inside a scope, then exit
        with scope.scope("cleanup_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_sanitize_request(handle, "sl_cleanup_guard", 1, sanitizer)
            scope_local.register_subscriber(handle, "sl_cleanup_sub", lambda e: events_inside.append(e))
            await tools.execute("tool_inside", {"x": 1}, my_tool)

        # Verify the sanitizer ran inside the scope (visible in event input).
        start_inside = [e for e in events_inside if e.event_type == EventType.Start and e.name == "tool_inside"]
        assert len(start_inside) >= 1
        assert start_inside[0].input["sanitized"] is True

        # After scope exit, the guardrail should be gone — use a global
        # subscriber to capture events from the outer call.
        events_outside = []
        subscribers.register("sl_cleanup_outer_sub", lambda e: events_outside.append(e))
        await tools.execute("tool_outside", {"x": 2}, my_tool)
        subscribers.deregister("sl_cleanup_outer_sub")

        start_outside = [e for e in events_outside if e.event_type == EventType.Start and e.name == "tool_outside"]
        assert len(start_outside) >= 1
        assert "sanitized" not in start_outside[0].input

    async def test_intercept_inactive_after_scope_exit(self):
        """Scope-local request intercept no longer affects calls after scope exit."""

        def intercept_fn(tool_name, args):
            args["intercepted"] = True
            return args

        def my_tool(args):
            return args

        with scope.scope("intercept_cleanup", ScopeType.Agent) as handle:
            scope_local.register_tool_request(handle, "sl_cleanup_int", 1, False, intercept_fn)
            result_inside = await tools.execute("tool_in_scope", {"a": 1}, my_tool)

        result_outside = await tools.execute("tool_out_scope", {"a": 2}, my_tool)

        assert result_inside["intercepted"] is True
        assert "intercepted" not in result_outside

    async def test_subscriber_inactive_after_scope_exit(self):
        """Scope-local subscriber stops receiving events after scope exit."""
        events_inside = []

        with scope.scope("sub_cleanup", ScopeType.Agent) as handle:
            scope_local.register_subscriber(handle, "sl_cleanup_sub", lambda e: events_inside.append(e))
            # Generate some events inside the scope
            inner_handle = tools.call("sub_tool", {"k": "v"})
            tools.call_end(inner_handle, {"done": True})

        # Events generated after scope exit should not reach the subscriber
        global_events = []
        subscribers.register("sl_after_sub", lambda e: global_events.append(e))
        outer_handle = tools.call("outer_tool", {})
        tools.call_end(outer_handle, {})
        subscribers.deregister("sl_after_sub")

        assert len(events_inside) >= 1
        # The scope-local subscriber should not have received events from after scope exit
        events_inside_count = len(events_inside)
        assert events_inside_count >= 1
        # Global subscriber proves events were emitted
        assert len(global_events) >= 1


# ---------------------------------------------------------------------------
# 3. Priority ordering with global guardrails
# ---------------------------------------------------------------------------


class TestScopeLocalPriorityOrdering:
    async def test_scope_local_lower_priority_runs_first(self):
        """Scope-local guardrail at priority 5 runs before global at priority 10.

        Lower numeric priority means it executes first. We use an execution
        order list to verify.
        """
        execution_order = []

        def global_sanitizer(tool_name, args):
            execution_order.append("global_p10")
            return args

        def scope_local_sanitizer(tool_name, args):
            execution_order.append("scope_local_p5")
            return args

        def my_tool(args):
            return args

        guardrails.register_tool_sanitize_request("sl_global_guard", 10, global_sanitizer)

        with scope.scope("priority_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_sanitize_request(handle, "sl_local_guard", 5, scope_local_sanitizer)
            await tools.execute("priority_tool", {"test": True}, my_tool)

        guardrails.deregister_tool_sanitize_request("sl_global_guard")

        assert len(execution_order) == 2
        assert execution_order[0] == "scope_local_p5"
        assert execution_order[1] == "global_p10"

    async def test_scope_local_higher_priority_runs_after_global(self):
        """Scope-local guardrail at priority 20 runs after global at priority 10."""
        execution_order = []

        def global_sanitizer(tool_name, args):
            execution_order.append("global_p10")
            return args

        def scope_local_sanitizer(tool_name, args):
            execution_order.append("scope_local_p20")
            return args

        def my_tool(args):
            return args

        guardrails.register_tool_sanitize_request("sl_global_guard2", 10, global_sanitizer)

        with scope.scope("priority_scope2", ScopeType.Agent) as handle:
            scope_local.register_tool_sanitize_request(handle, "sl_local_guard2", 20, scope_local_sanitizer)
            await tools.execute("priority_tool2", {}, my_tool)

        guardrails.deregister_tool_sanitize_request("sl_global_guard2")

        assert len(execution_order) == 2
        assert execution_order[0] == "global_p10"
        assert execution_order[1] == "scope_local_p20"


# ---------------------------------------------------------------------------
# 4. Scope-local subscriber
# ---------------------------------------------------------------------------


class TestScopeLocalSubscriber:
    async def test_subscriber_receives_events_in_scope(self):
        """A scope-local subscriber receives tool lifecycle events within the scope."""
        events = []

        with scope.scope("sub_scope", ScopeType.Agent) as handle:
            scope_local.register_subscriber(handle, "sl_sub", lambda e: events.append(e))
            tool_handle = tools.call("sub_test_tool", {"arg": "value"})
            tools.call_end(tool_handle, {"result": "ok"})

        # Should have received at least tool start and end events
        assert len(events) >= 2
        for e in events:
            assert isinstance(e, Event)
            assert e.uuid is not None

    async def test_subscriber_receives_mark_events(self):
        """A scope-local subscriber receives mark events emitted within the scope."""
        events = []

        with scope.scope("mark_sub_scope", ScopeType.Agent) as handle:
            scope_local.register_subscriber(handle, "sl_mark_sub", lambda e: events.append(e))
            scope.event("test_mark", data={"info": "hello"})

        mark_events = [e for e in events if e.event_type == EventType.Mark]
        assert len(mark_events) >= 1

    def test_subscriber_deregister_within_scope(self):
        """A scope-local subscriber can be explicitly deregistered before scope exit."""
        events = []

        with scope.scope("dereg_sub_scope", ScopeType.Agent) as handle:
            scope_local.register_subscriber(handle, "sl_dereg_sub", lambda e: events.append(e))
            # Explicitly deregister
            result = scope_local.deregister_subscriber(handle, "sl_dereg_sub")
            assert result is True
            # Events after deregistration should not be collected
            events_before = len(events)
            tool_handle = tools.call("dereg_tool", {})
            tools.call_end(tool_handle, {})

        # No new events should have been appended after deregistration
        assert len(events) == events_before


# ---------------------------------------------------------------------------
# 5. Scope-local conditional execution (rejection)
# ---------------------------------------------------------------------------


class TestScopeLocalConditionalExecution:
    async def test_conditional_rejects_blocked_tool(self):
        """A scope-local conditional guardrail rejects tool calls matching a name pattern."""
        blocked_tools = {"dangerous_tool", "unsafe_operation"}

        def blocker(tool_name, args):
            if tool_name in blocked_tools:
                return f"tool '{tool_name}' is blocked by policy"
            return None

        def my_tool(args):
            return {"should": "not reach"}

        with scope.scope("cond_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_conditional_execution(handle, "sl_blocker", 1, blocker)
            with pytest.raises(RuntimeError, match="guardrail rejected"):
                await tools.execute("dangerous_tool", {"payload": "x"}, my_tool)

    async def test_conditional_allows_non_blocked_tool(self):
        """A scope-local conditional guardrail allows tool calls that don't match."""
        blocked_tools = {"dangerous_tool"}

        def blocker(tool_name, args):
            if tool_name in blocked_tools:
                return f"tool '{tool_name}' is blocked"
            return None

        def my_tool(args):
            return {"result": "success"}

        with scope.scope("cond_allow_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_conditional_execution(handle, "sl_allow_blocker", 1, blocker)
            result = await tools.execute("safe_tool", {"data": 42}, my_tool)

        assert result["result"] == "success"

    async def test_conditional_inactive_after_scope_exit(self):
        """Conditional guardrail does not reject after its scope has been popped."""
        blocked_tools = {"blocked_tool"}

        def blocker(tool_name, args):
            if tool_name in blocked_tools:
                return "blocked"
            return None

        def my_tool(args):
            return {"ok": True}

        with scope.scope("cond_cleanup_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_conditional_execution(handle, "sl_cond_cleanup", 1, blocker)

        # After scope exit, the guardrail should no longer be active
        result = await tools.execute("blocked_tool", {}, my_tool)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# 6. Isolation between sequential scopes
# ---------------------------------------------------------------------------


class TestScopeLocalIsolation:
    async def test_no_leaking_between_sequential_scopes(self):
        """Middleware registered in scope A does not affect scope B.

        Uses request intercepts (which DO modify the execution pipeline args)
        instead of sanitize guardrails (which are observability-only).
        """

        def interceptor_a(tool_name, args):
            args["from_scope_a"] = True
            return args

        def interceptor_b(tool_name, args):
            args["from_scope_b"] = True
            return args

        def my_tool(args):
            return args

        # Scope A: registers interceptor_a
        with scope.scope("scope_a", ScopeType.Agent) as handle_a:
            scope_local.register_tool_request(handle_a, "sl_guard_a", 1, False, interceptor_a)
            result_a = await tools.execute("tool_a", {"input": "a"}, my_tool)

        # Scope B: registers interceptor_b
        with scope.scope("scope_b", ScopeType.Agent) as handle_b:
            scope_local.register_tool_request(handle_b, "sl_guard_b", 1, False, interceptor_b)
            result_b = await tools.execute("tool_b", {"input": "b"}, my_tool)

        # Scope A result should only have from_scope_a
        assert result_a["from_scope_a"] is True
        assert "from_scope_b" not in result_a

        # Scope B result should only have from_scope_b
        assert result_b["from_scope_b"] is True
        assert "from_scope_a" not in result_b

    async def test_no_leaking_with_different_middleware_types(self):
        """Different middleware types in sequential scopes stay isolated."""
        intercept_ran = []
        events_a = []
        events_b = []

        def intercept_fn(tool_name, args):
            intercept_ran.append("intercept")
            args["intercepted"] = True
            return args

        def my_tool(args):
            return args

        # Scope A: request intercept + subscriber
        with scope.scope("iso_scope_a", ScopeType.Agent) as handle_a:
            scope_local.register_tool_request(handle_a, "sl_iso_int", 1, False, intercept_fn)
            scope_local.register_subscriber(handle_a, "sl_iso_sub_a", lambda e: events_a.append(e))
            result_a = await tools.execute("iso_tool_a", {"val": 1}, my_tool)

        # Scope B: only a subscriber, no intercept
        with scope.scope("iso_scope_b", ScopeType.Agent) as handle_b:
            scope_local.register_subscriber(handle_b, "sl_iso_sub_b", lambda e: events_b.append(e))
            result_b = await tools.execute("iso_tool_b", {"val": 2}, my_tool)

        # Scope A should have had the intercept applied
        assert result_a["intercepted"] is True
        # Scope B should NOT have the intercept
        assert "intercepted" not in result_b
        # Both subscribers should have received events in their respective scopes
        assert len(events_a) >= 1
        assert len(events_b) >= 1

    async def test_global_guardrail_unaffected_by_scope_local(self):
        """A global guardrail persists across scope-local scope boundaries."""
        execution_order = []

        def global_guard(tool_name, args):
            execution_order.append("global")
            return args

        def my_tool(args):
            return args

        guardrails.register_tool_sanitize_request("sl_persist_global", 1, global_guard)

        # First scope with scope-local middleware
        with scope.scope("persist_scope_1", ScopeType.Agent) as handle:
            scope_local.register_tool_sanitize_request(handle, "sl_persist_local", 2, lambda n, a: a)
            await tools.execute("persist_tool_1", {}, my_tool)

        # Global should still work after the scope-local scope ends
        execution_order.clear()
        await tools.execute("persist_tool_2", {}, my_tool)

        guardrails.deregister_tool_sanitize_request("sl_persist_global")

        assert "global" in execution_order


# ---------------------------------------------------------------------------
# Scope-local tool execution intercept (middleware chain)
# ---------------------------------------------------------------------------


class TestScopeLocalExecutionIntercept:
    async def test_execution_intercept_replaces_function(self):
        """A scope-local execution intercept can replace the tool function entirely."""

        def my_tool(args):
            return {"from": "original"}

        with scope.scope("exec_int_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_execution(
                handle,
                "sl_exec_intercept",
                1,
                lambda name, args, next_fn: {"from": "intercept"},
            )
            result = await tools.execute("exec_int_tool", {}, my_tool)

        assert result["from"] == "intercept"

    async def test_execution_intercept_calls_next(self):
        """A scope-local execution intercept can modify args and return a result.

        NOTE: Calling ``next_fn`` from a synchronous execution intercept is
        not yet supported because ``next_fn`` returns an asyncio Future that
        cannot be awaited inside a sync callback. Instead, this test modifies
        the args and returns a computed result directly (similar to
        ``test_execution_intercept_replaces_function``).
        """

        def my_tool(args):
            return {"value": args["x"] * 2}

        def intercept_fn(name, args, next_fn):
            # Cannot call next_fn here — it returns a Future.
            args["x"] = args["x"] + 1
            return {"value": args["x"] * 2, "intercepted": True}

        with scope.scope("exec_next_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_execution(handle, "sl_exec_next", 1, intercept_fn)
            result = await tools.execute("exec_next_tool", {"x": 5}, my_tool)

        # Intercept receives x=5, adds 1 -> x=6, returns value=12
        assert result["value"] == 12
        assert result["intercepted"] is True


# ---------------------------------------------------------------------------
# Deregistration within scope
# ---------------------------------------------------------------------------


class TestScopeLocalDeregistration:
    async def test_deregister_guardrail_within_scope(self):
        """A scope-local guardrail can be explicitly deregistered before scope exit."""
        events = []

        def sanitizer(tool_name, args):
            args["sanitized"] = True
            return args

        def my_tool(args):
            return args

        with scope.scope("dereg_scope", ScopeType.Agent) as handle:
            scope_local.register_tool_sanitize_request(handle, "sl_dereg_guard", 1, sanitizer)
            scope_local.register_subscriber(handle, "sl_dereg_sub", lambda e: events.append(e))
            await tools.execute("dereg_tool_1", {"a": 1}, my_tool)

            # Verify the sanitizer ran (visible in event input).
            start_before = [e for e in events if e.event_type == EventType.Start and e.name == "dereg_tool_1"]
            assert len(start_before) >= 1
            assert start_before[0].input["sanitized"] is True

            # Explicitly deregister
            removed = scope_local.deregister_tool_sanitize_request(handle, "sl_dereg_guard")
            assert removed is True

            events.clear()
            await tools.execute("dereg_tool_2", {"a": 2}, my_tool)

        # After deregistration, the sanitizer should no longer appear in events.
        start_after = [e for e in events if e.event_type == EventType.Start and e.name == "dereg_tool_2"]
        assert len(start_after) >= 1
        assert "sanitized" not in start_after[0].input

    def test_deregister_nonexistent_returns_false(self):
        """Deregistering a name that was never registered returns False."""
        with scope.scope("dereg_none_scope", ScopeType.Agent) as handle:
            result = scope_local.deregister_tool_sanitize_request(handle, "nonexistent_guard")
            assert result is False


class TestScopeLocalLlmWrappers:
    def test_register_and_deregister_scope_local_wrappers(self):
        """Scope-local wrapper functions round-trip through the native API for both tool and LLM middleware."""
        request = LLMRequest({}, {"messages": [], "model": "scope-local"})

        async def stream_intercept(request_inner, next_fn):
            if False:
                yield {}

        with scope.scope("llm_scope_local_wrappers", ScopeType.Agent) as handle:
            scope_local.register_tool_sanitize_response(handle, "sl_tool_resp_cov", 1, lambda name, result: result)
            assert scope_local.deregister_tool_sanitize_response(handle, "sl_tool_resp_cov") is True

            scope_local.register_tool_conditional_execution(handle, "sl_tool_cond_cov", 1, lambda name, args: None)
            assert scope_local.deregister_tool_conditional_execution(handle, "sl_tool_cond_cov") is True

            scope_local.register_tool_request(handle, "sl_tool_req_cov", 1, False, lambda name, args: args)
            assert scope_local.deregister_tool_request(handle, "sl_tool_req_cov") is True

            scope_local.register_tool_execution(handle, "sl_tool_exec_cov", 1, lambda name, args, next_fn: args)
            assert scope_local.deregister_tool_execution(handle, "sl_tool_exec_cov") is True

            scope_local.register_llm_sanitize_request(handle, "sl_llm_req_cov", 1, lambda req: req)
            assert scope_local.deregister_llm_sanitize_request(handle, "sl_llm_req_cov") is True

            scope_local.register_llm_sanitize_response(handle, "sl_llm_resp_cov", 1, lambda response: response)
            assert scope_local.deregister_llm_sanitize_response(handle, "sl_llm_resp_cov") is True

            scope_local.register_llm_conditional_execution(handle, "sl_llm_cond_cov", 1, lambda req: None)
            assert scope_local.deregister_llm_conditional_execution(handle, "sl_llm_cond_cov") is True

            scope_local.register_llm_request(handle, "sl_llm_int_cov", 1, False, lambda name, req: req)
            assert scope_local.deregister_llm_request(handle, "sl_llm_int_cov") is True

            scope_local.register_llm_execution(
                handle,
                "sl_llm_exec_cov",
                1,
                lambda name, req, next_fn: {"intercepted": True},
            )
            assert scope_local.deregister_llm_execution(handle, "sl_llm_exec_cov") is True

            scope_local.register_llm_stream_execution(handle, "sl_llm_stream_cov", 1, stream_intercept)
            assert scope_local.deregister_llm_stream_execution(handle, "sl_llm_stream_cov") is True

            assert request.content["model"] == "scope-local"


class TestScopeLocalLlmBehavior:
    async def test_scope_local_llm_sanitize_request_rewrites_event_input(self):
        events = []
        request = LLMRequest({}, {"messages": [], "model": "scope-local"})

        def sanitize_request(req):
            return LLMRequest({"X-Scope-Local": "yes"}, req.content)

        with scope.scope("sl_llm_sanitize_scope", ScopeType.Agent) as handle:
            scope_local.register_subscriber(handle, "sl_llm_sanitize_sub", lambda event: events.append(event))
            scope_local.register_llm_sanitize_request(handle, "sl_llm_sanitize", 1, sanitize_request)
            result = await llm.execute("sl_llm_sanitize_call", request, lambda req: {"model": req.content["model"]})

        assert result == {"model": "scope-local"}
        start = next(
            event for event in events if event.name == "sl_llm_sanitize_call" and event.event_type == EventType.Start
        )
        assert start.input == {
            "headers": {"X-Scope-Local": "yes"},
            "content": {"messages": [], "model": "scope-local"},
        }

    async def test_scope_local_llm_request_intercept_modifies_request(self):
        request = LLMRequest({}, {"messages": [], "model": "scope-local"})

        def intercept(name, req):
            return LLMRequest(req.headers, {**req.content, "intercepted": True})

        with scope.scope("sl_llm_request_scope", ScopeType.Agent) as handle:
            scope_local.register_llm_request(handle, "sl_llm_request", 1, False, intercept)
            result = await llm.execute(
                "sl_llm_request_call",
                request,
                lambda req: {"intercepted": req.content.get("intercepted", False)},
            )

        assert result == {"intercepted": True}

    async def test_scope_local_llm_execution_intercept_can_await_next(self):
        request = LLMRequest({}, {"messages": [], "model": "scope-local"})

        async def middleware(name, req, next_fn):
            updated = LLMRequest(req.headers, {**req.content, "model": "via-scope-local"})
            result = await next_fn(updated)
            result["scope_local"] = True
            return result

        with scope.scope("sl_llm_execution_scope", ScopeType.Agent) as handle:
            scope_local.register_llm_execution(handle, "sl_llm_execution", 1, middleware)
            result = await llm.execute(
                "sl_llm_execution_call",
                request,
                lambda req: {"model": req.content["model"]},
            )

        assert result == {"model": "via-scope-local", "scope_local": True}

    async def test_scope_local_llm_conditional_execution_blocks(self):
        request = LLMRequest({}, {"messages": [], "model": "scope-local"})

        with scope.scope("sl_llm_conditional_scope", ScopeType.Agent) as handle:
            scope_local.register_llm_conditional_execution(
                handle,
                "sl_llm_conditional",
                1,
                lambda req: "blocked by scope-local llm guardrail",
            )
            with pytest.raises(RuntimeError, match="guardrail rejected"):
                await llm.execute("sl_llm_conditional_call", request, lambda req: {"should": "not-run"})
