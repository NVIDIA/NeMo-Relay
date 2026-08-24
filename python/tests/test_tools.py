# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Relay tool lifecycle, guardrails, and intercepts."""

import asyncio
import contextvars
import gc
import warnings
from collections import UserDict, UserList
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast

import pytest
from async_helpers import resolve_async_result

from nemo_relay import (
    Event,
    MarkEvent,
    PendingMarkSpec,
    ScopeEvent,
    ScopeType,
    ToolAttributes,
    ToolExecutionInterceptOutcome,
    ToolExecutionResult,
    ToolHandle,
    create_scope_stack,
    guardrails,
    intercepts,
    scope,
    subscribers,
    tools,
    use_scope_stack,
)


def raise_runtime_error(message: str):
    raise RuntimeError(message)


def _tool_event(events, name: str, scope_category: str) -> ScopeEvent:
    return next(
        event
        for event in events
        if event.name == name
        and isinstance(event, ScopeEvent)
        and event.category == "tool"
        and event.scope_category == scope_category
    )


class TestTools:
    def test_call_and_call_end(self):
        handle = tools.call("my_tool", {"input": "data"})
        assert isinstance(handle, ToolHandle)
        assert handle.name == "my_tool"
        tools.call_end(handle, ToolExecutionResult({"output": "result"}))

    def test_call_end_preserves_result_annotation(self, subscribed_events: list[Event]):
        handle = tools.call("manual_annotated_tool", {"input": "data"})
        tools.call_end(
            handle,
            ToolExecutionResult(
                {"output": "result"},
                {"provider": "manual"},
            ),
        )
        subscribers.flush()

        end = _tool_event(subscribed_events, "manual_annotated_tool", "end")
        assert end.data == {"output": "result"}
        assert end.category_profile == {
            "tool_result_annotation": {"provider": "manual"},
        }

    def test_call_with_attributes(self):
        attrs = ToolAttributes(ToolAttributes.REMOTE)
        handle = tools.call("local_tool", {"x": 1}, attributes=attrs)
        assert handle.name == "local_tool"
        tools.call_end(handle, ToolExecutionResult({"y": 2}))

    def test_call_with_data_metadata(self):
        handle = tools.call(
            "tool_dm",
            {"arg": 1},
            data={"custom": "info"},
            metadata={"trace_id": "abc123"},
        )
        tools.call_end(handle, ToolExecutionResult("ok"), data={"end_data": True}, metadata={"end_meta": True})

    def test_call_with_parent_handle(self):
        parent = scope.push("tool_parent", ScopeType.Agent)
        handle = tools.call("child_tool", {}, handle=parent)
        assert handle.parent_uuid == parent.uuid
        tools.call_end(handle, ToolExecutionResult({}))
        scope.pop(parent)

    def test_complete_skill_read_emits_minimal_eager_mark(self, subscribed_events: list[Event]):
        handle = tools.call("read_file", {"path": "/skills/review/SKILL.md"})
        tools.call_end(handle, ToolExecutionResult({"ok": True}))
        subscribers.flush()

        start = _tool_event(subscribed_events, "read_file", "start")
        mark = next(event for event in subscribed_events if isinstance(event, MarkEvent) and event.name == "skill.load")
        end = _tool_event(subscribed_events, "read_file", "end")
        assert subscribed_events.index(start) < subscribed_events.index(mark) < subscribed_events.index(end)
        assert mark.parent_uuid == start.uuid
        assert mark.data == {"skill_name": "review"}
        assert mark.metadata == {"skill_load_source": "structured_read", "tool_name": "read_file"}


class TestToolsAsync:
    async def test_execute_basic(self):
        # tools.execute wraps a Python callable; use sync func
        def my_func(args):
            return ToolExecutionResult({"result": args["x"] * 2})

        result = await tools.execute("double", {"x": 5}, my_func)
        assert result.result == {"result": 10}

    async def test_execute_rejects_legacy_raw_result(self):
        with pytest.raises(RuntimeError, match="must return ToolExecutionResult") as error:
            await tools.execute(
                "legacy_raw_result",
                {},
                lambda _args: {"legacy": True},  # ty: ignore[invalid-argument-type]
            )
        assert "ToolExecutionResult(payload), not a raw dict" in str(error.value)
        assert "internal error" not in str(error.value)

    async def test_execute_rejects_cyclic_results_and_remains_usable(self):
        def cyclic_result(_args):
            result = {}
            result["self"] = result
            return ToolExecutionResult(result)

        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("cyclic_result", {}, cyclic_result)

        async def async_cyclic_result(_args):
            return cyclic_result(_args)

        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("async_cyclic_result", {}, async_cyclic_result)

        cyclic_mapping = UserDict()
        cyclic_mapping["self"] = cyclic_mapping
        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("cyclic_mapping", {}, lambda _args: ToolExecutionResult(cyclic_mapping))

        cyclic_sequence = UserList()
        cyclic_sequence.append(cyclic_sequence)
        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("cyclic_sequence", {}, lambda _args: ToolExecutionResult(cyclic_sequence))

        @dataclass
        class CyclicDataclass:
            child: object | None = None

        cyclic_dataclass = CyclicDataclass()
        cyclic_dataclass.child = cyclic_dataclass
        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("cyclic_dataclass", {}, lambda _args: ToolExecutionResult(cyclic_dataclass))

        result = await tools.execute(
            "post_cycle_result",
            {},
            lambda _args: ToolExecutionResult({"status": "ok"}),
        )
        assert result.result == {"status": "ok"}

    async def test_execute_allows_shared_non_cyclic_results(self):
        shared = {"value": True}

        result = await tools.execute(
            "shared_result",
            {},
            lambda _args: ToolExecutionResult({"first": shared, "second": shared}),
        )

        assert result.result == {
            "first": {"value": True},
            "second": {"value": True},
        }

    async def test_execute_returns_string(self):
        def func(args):
            return ToolExecutionResult("hello")

        result = await tools.execute("str_tool", {}, func)
        assert result.result == "hello"

    async def test_execute_with_attributes(self):
        def func(args):
            return ToolExecutionResult(args)

        attrs = ToolAttributes(ToolAttributes.REMOTE)
        result = await tools.execute(
            "attr_tool",
            {"test": True},
            func,
            attributes=attrs,
        )
        assert result.result["test"] is True

    async def test_execute_async_func(self):
        """tools.execute should accept async functions."""

        async def my_async_func(args):
            return ToolExecutionResult({"result": args["x"] + 1})

        result = await tools.execute("async_tool", {"x": 10}, my_async_func)
        assert result.result == {"result": 11}

    async def test_execute_async_func_returns_string(self):
        async def func(args):
            return ToolExecutionResult("async_hello")

        result = await tools.execute("async_str_tool", {}, func)
        assert result.result == "async_hello"

    async def test_execute_async_func_with_attributes(self):
        async def func(args):
            return ToolExecutionResult(args)

        attrs = ToolAttributes(ToolAttributes.REMOTE)
        result = await tools.execute(
            "async_attr_tool",
            {"key": "value"},
            func,
            attributes=attrs,
        )
        assert result.result["key"] == "value"

    async def test_execute_propagates_tool_call_id_to_lifecycle(self, subscribed_events: list[Event]):
        result = await tools.execute(
            "managed_tool_call_id",
            {"value": 1},
            lambda args: ToolExecutionResult(args),
            tool_call_id="managed-call-123",
        )
        assert result.result == {"value": 1}

        await subscribers.flush_async()
        lifecycle = [
            event
            for event in subscribed_events
            if isinstance(event, ScopeEvent) and event.name == "managed_tool_call_id"
        ]
        assert [event.scope_category for event in lifecycle] == ["start", "end"]
        assert all(
            event.category_profile is not None and event.category_profile.get("tool_call_id") == "managed-call-123"
            for event in lifecycle
        )

    async def test_execute_failure_emits_end_event(self):
        events = []
        subscribers.register("py_tool_exec_failure_sub", lambda e: events.append(e))

        def failing(args):
            raise ValueError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await tools.execute(
                "failing_tool",
                {"x": 1},
                failing,
                tool_call_id="managed-failure-123",
            )

        try:
            await subscribers.flush_async()
        finally:
            subscribers.deregister("py_tool_exec_failure_sub")

        assert [e.kind for e in events] == ["scope", "scope"]
        assert all(isinstance(event, ScopeEvent) for event in events)
        assert [e.scope_category for e in events] == ["start", "end"]
        assert all(e.category == "tool" for e in events)
        assert all(
            event.category_profile is not None and event.category_profile.get("tool_call_id") == "managed-failure-123"
            for event in events
        )
        assert events[0].uuid == events[1].uuid
        assert events[1].data is None
        assert events[1].metadata["error.type"] == "internal_error"
        assert events[1].metadata["exception.type"] == "ValueError"


class TestToolGuardrails:
    def test_sanitize_request_guardrail(self):
        def sanitizer(name, args):
            args["sanitized"] = True
            return args

        guardrails.register_tool_sanitize_request("py_san_req", 1, sanitizer)

        events = []
        subscribers.register("py_san_req_sub", lambda e: events.append(e))
        handle = tools.call("guarded_tool", {"input": "data"})
        tools.call_end(handle, ToolExecutionResult({}))
        subscribers.flush()
        subscribers.deregister("py_san_req_sub")
        guardrails.deregister_tool_sanitize_request("py_san_req")

        start_events = [e for e in events if isinstance(e, ScopeEvent) and e.category == "tool"]
        assert len(start_events) >= 1

    def test_sanitize_response_guardrail(self):
        def resp_sanitizer(name, result):
            result["cleaned"] = True
            return result

        guardrails.register_tool_sanitize_response("py_san_resp", 1, resp_sanitizer)
        handle = tools.call("tool", {})
        tools.call_end(handle, ToolExecutionResult({"output": "raw"}))
        guardrails.deregister_tool_sanitize_response("py_san_resp")

    def test_conditional_execution_guardrail(self):
        def blocker(name, args):
            if args.get("blocked"):
                return "execution blocked"
            return None

        guardrails.register_tool_conditional_execution("py_cond", 1, blocker)
        guardrails.deregister_tool_conditional_execution("py_cond")

    def test_conditional_execution_direct(self):
        guardrails.register_tool_conditional_execution("py_cond_direct", 1, lambda name, args: "blocked directly")
        with pytest.raises(RuntimeError, match="guardrail rejected"):
            tools.conditional_execution("direct_tool", {})
        guardrails.deregister_tool_conditional_execution("py_cond_direct")

    def test_duplicate_guardrail_raises(self):
        guardrails.register_tool_sanitize_request("py_dup_guard", 1, lambda n, a: a)
        with pytest.raises(RuntimeError):
            guardrails.register_tool_sanitize_request("py_dup_guard", 1, lambda n, a: a)
        guardrails.deregister_tool_sanitize_request("py_dup_guard")

    def test_sanitize_request_failure_omits_observability_input(self):
        events = []
        subscribers.register("py_tool_sanitize_req_sub", lambda event: events.append(event))
        guardrails.register_tool_sanitize_request(
            "py_tool_sanitize_req_fail",
            1,
            lambda name, args: raise_runtime_error("boom"),
        )
        try:
            handle = tools.call("tool_sanitize_req_fail", {"value": 1})
            tools.call_end(handle, ToolExecutionResult({"ok": True}))
        finally:
            guardrails.deregister_tool_sanitize_request("py_tool_sanitize_req_fail")
            subscribers.flush()
            subscribers.deregister("py_tool_sanitize_req_sub")

        start = _tool_event(events, "tool_sanitize_req_fail", "start")
        assert start.data is None

    def test_sanitize_response_invalid_return_omits_observability_output(self):
        events = []
        subscribers.register("py_tool_sanitize_resp_sub", lambda event: events.append(event))
        guardrails.register_tool_sanitize_response(
            "py_tool_sanitize_resp_bad",
            1,
            cast(guardrails.ToolSanitizeGuardrail, lambda name, result: object()),
        )
        try:
            handle = tools.call("tool_sanitize_resp_bad", {"value": 1})
            tools.call_end(handle, ToolExecutionResult({"ok": True}))
        finally:
            guardrails.deregister_tool_sanitize_response("py_tool_sanitize_resp_bad")
            subscribers.flush()
            subscribers.deregister("py_tool_sanitize_resp_sub")

        end = _tool_event(events, "tool_sanitize_resp_bad", "end")
        assert end.data is None

    def test_deregister_nonexistent(self):
        assert not guardrails.deregister_tool_sanitize_request("nonexistent")
        assert not guardrails.deregister_tool_sanitize_response("nonexistent")
        assert not guardrails.deregister_tool_conditional_execution("nonexistent")


class TestToolGuardrailsAsync:
    async def test_async_conditional_runs_on_originating_loop(self):
        originating_loop = asyncio.get_running_loop()

        async def allow(_name, _args):
            await asyncio.sleep(0)
            assert asyncio.get_running_loop() is originating_loop
            return None

        guardrails.register_tool_conditional_execution("py_async_conditional_loop", 1, allow)
        try:
            result = await tools.execute("allowed_tool", {}, lambda args: ToolExecutionResult(args))
        finally:
            guardrails.deregister_tool_conditional_execution("py_async_conditional_loop")

        assert result.result == {}

    async def test_manual_async_sanitizers_publish_transformed_payloads_and_can_flush(self):
        events = []
        request_flushed = False
        response_flushed = False

        async def sanitize_request(name, args):
            nonlocal request_flushed
            await asyncio.sleep(0)
            subscribers.flush()
            request_flushed = True
            return {**args, "request_sanitized": True}

        async def sanitize_response(name, response):
            nonlocal response_flushed
            await asyncio.sleep(0)
            subscribers.flush()
            response_flushed = True
            return {**response, "response_sanitized": True}

        subscribers.register("py_manual_tool_flush_subscriber", events.append)
        guardrails.register_tool_sanitize_request("py_manual_tool_flush_request", 1, sanitize_request)
        guardrails.register_tool_sanitize_response("py_manual_tool_flush_response", 1, sanitize_response)
        try:
            handle = tools.call("py_manual_tool_flush", {"original": True})
            tools.call_end(handle, ToolExecutionResult({"ok": True}))
            await asyncio.wait_for(subscribers.flush_async(), timeout=2)
        finally:
            guardrails.deregister_tool_sanitize_request("py_manual_tool_flush_request")
            guardrails.deregister_tool_sanitize_response("py_manual_tool_flush_response")
            subscribers.deregister("py_manual_tool_flush_subscriber")

        assert request_flushed
        assert response_flushed
        assert _tool_event(events, "py_manual_tool_flush", "start").data == {
            "original": True,
            "request_sanitized": True,
        }
        assert _tool_event(events, "py_manual_tool_flush", "end").data == {
            "ok": True,
            "response_sanitized": True,
        }

    async def test_conditional_blocks_execution(self):
        guardrails.register_tool_conditional_execution("py_async_blocker", 1, lambda name, args: "blocked by policy")

        def func(args):
            return {"should": "not reach"}

        with pytest.raises(RuntimeError, match="guardrail rejected"):
            await tools.execute("blocked_tool", {}, func)

        guardrails.deregister_tool_conditional_execution("py_async_blocker")


class TestToolIntercepts:
    def test_request_intercept_register_deregister(self):
        intercepts.register_tool_request("py_req_int", 1, False, lambda n, a: a)
        assert intercepts.deregister_tool_request("py_req_int")
        assert not intercepts.deregister_tool_request("py_req_int")

    def test_request_intercepts_direct(self):
        def intercept_fn(name, args):
            args["direct"] = True
            return args

        intercepts.register_tool_request("py_req_int_direct", 1, False, intercept_fn)
        transformed = tools.request_intercepts("direct_tool", {"input": True})
        intercepts.deregister_tool_request("py_req_int_direct")

        assert not isinstance(transformed, Awaitable)
        assert transformed == {"input": True, "direct": True}

    def test_execution_intercept_register_deregister(self):
        intercepts.register_tool_execution(
            "py_exec_int",
            1,
            lambda name, args, next: ToolExecutionInterceptOutcome({"intercepted": True}),
        )
        assert intercepts.deregister_tool_execution("py_exec_int")

    def test_duplicate_intercept_raises(self):
        intercepts.register_tool_request("py_dup_int", 1, False, lambda n, a: a)
        with pytest.raises(RuntimeError):
            intercepts.register_tool_request("py_dup_int", 1, False, lambda n, a: a)
        intercepts.deregister_tool_request("py_dup_int")

    def test_request_intercept_raises_on_exception(self):
        intercepts.register_tool_request("py_req_raise", 1, False, lambda n, a: raise_runtime_error("boom"))
        try:
            with pytest.raises(RuntimeError, match="RuntimeError: boom"):
                tools.request_intercepts("raise_tool", {"value": 1})
        finally:
            intercepts.deregister_tool_request("py_req_raise")

    def test_request_intercept_raises_on_unserializable_return(self):
        intercepts.register_tool_request(
            "py_req_bad_return",
            1,
            False,
            cast(intercepts.ToolRequestIntercept, lambda n, a: object()),
        )
        try:
            with pytest.raises(RuntimeError, match="unsupported type object"):
                tools.request_intercepts("bad_return_tool", {"value": 1})
        finally:
            intercepts.deregister_tool_request("py_req_bad_return")


class TestToolInterceptsAsync:
    def test_loop_shutdown_cancels_pending_middleware_without_unraisable_errors(self, capsys):
        async def request_intercept(_name, args):
            await asyncio.Event().wait()
            return args

        async def scenario():
            execution = asyncio.ensure_future(
                tools.execute("shutdown_tool", {}, lambda args: ToolExecutionResult(args))
            )
            await asyncio.sleep(0.01)
            assert not execution.done()

        intercepts.register_tool_request("py_tool_shutdown_request", 1, False, request_intercept)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                asyncio.run(scenario())
                gc.collect()
        finally:
            intercepts.deregister_tool_request("py_tool_shutdown_request")

        diagnostics = capsys.readouterr().err + "\n".join(str(item.message) for item in caught)
        assert "Event loop is closed" not in diagnostics
        assert "Task was destroyed but it is pending" not in diagnostics
        assert "was never awaited" not in diagnostics

    async def test_cancelling_conditional_guardrail_closes_guardrail_scope(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        events: list[Event] = []

        async def conditional(_name, _args):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        guardrails.register_tool_conditional_execution("py_tool_cancel_conditional", 1, conditional)
        subscribers.register("py_tool_cancel_conditional_events", events.append)
        try:
            execution = asyncio.ensure_future(
                tools.execute("cancel_conditional_tool", {}, lambda x: ToolExecutionResult(x))
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await execution
            await asyncio.wait_for(cancelled.wait(), timeout=1)
            await subscribers.flush_async()
        finally:
            guardrails.deregister_tool_conditional_execution("py_tool_cancel_conditional")
            subscribers.deregister("py_tool_cancel_conditional_events")

        guardrail_lifecycle = [
            event.scope_category
            for event in events
            if isinstance(event, ScopeEvent) and event.name == "py_tool_cancel_conditional"
        ]
        assert guardrail_lifecycle == ["start", "end"]

    async def test_cancelling_execute_cancels_pending_request_intercept(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        provider_calls: list[dict] = []

        async def request_intercept(_name, args):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return args

        def provider(args):
            provider_calls.append(args)
            return ToolExecutionResult(args)

        intercepts.register_tool_request("py_tool_cancel_request", 1, False, request_intercept)
        try:
            execution = asyncio.ensure_future(tools.execute("cancel_request_tool", {}, provider))
            await asyncio.wait_for(started.wait(), timeout=1)
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await execution
            await asyncio.wait_for(cancelled.wait(), timeout=1)
        finally:
            intercepts.deregister_tool_request("py_tool_cancel_request")

        assert provider_calls == []

    async def test_cancelling_execute_cancels_pending_execution_intercept(self):
        started = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()
        provider_calls: list[dict] = []
        events: list[Event] = []

        async def middleware(_name, args, next):
            started.set()
            try:
                await release.wait()
                downstream = await next(args)
                return ToolExecutionInterceptOutcome(
                    downstream.result,
                    annotation=downstream.annotation,
                )
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def provider(args):
            provider_calls.append(args)
            return ToolExecutionResult(args)

        intercepts.register_tool_execution("py_tool_cancel_intercept", 1, middleware)
        subscribers.register("py_tool_cancel_events", events.append)
        try:
            execution = asyncio.ensure_future(
                tools.execute(
                    "cancel_tool",
                    {"ok": True},
                    provider,
                    tool_call_id="managed-cancel-123",
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await execution
            await asyncio.wait_for(cancelled.wait(), timeout=1)
            await subscribers.flush_async()
        finally:
            release.set()
            intercepts.deregister_tool_execution("py_tool_cancel_intercept")
            subscribers.deregister("py_tool_cancel_events")

        assert provider_calls == []
        lifecycle = [event for event in events if isinstance(event, ScopeEvent) and event.name == "cancel_tool"]
        assert [event.scope_category for event in lifecycle] == ["start", "end"]
        assert all(
            event.category_profile is not None and event.category_profile.get("tool_call_id") == "managed-cancel-123"
            for event in lifecycle
        )

    async def test_sync_middleware_preserves_async_caller_context(self):
        request_id = contextvars.ContextVar("tool_middleware_request_id", default="registration")
        observed: list[tuple[str, str]] = []

        def conditional(_name, _args):
            observed.append(("conditional", request_id.get()))
            return None

        def request_intercept(_name, args):
            observed.append(("request", request_id.get()))
            return args

        def execution_intercept(_name, args, _next):
            observed.append(("execution", request_id.get()))
            return ToolExecutionInterceptOutcome(args)

        guardrails.register_tool_conditional_execution("py_tool_context_conditional", 1, conditional)
        intercepts.register_tool_request("py_tool_context_request", 1, False, request_intercept)
        intercepts.register_tool_execution("py_tool_context_execution", 1, execution_intercept)
        token = request_id.set("emitter")
        try:
            result = await tools.execute("context_tool", {"ok": True}, lambda args: ToolExecutionResult(args))
            assert result.result == {"ok": True}
            await resolve_async_result(tools.conditional_execution("context_tool_standalone", {}))
            assert await resolve_async_result(tools.request_intercepts("context_tool_standalone", {"ok": True})) == {
                "ok": True
            }
        finally:
            request_id.reset(token)
            intercepts.deregister_tool_execution("py_tool_context_execution")
            intercepts.deregister_tool_request("py_tool_context_request")
            guardrails.deregister_tool_conditional_execution("py_tool_context_conditional")

        assert observed == [
            ("conditional", "emitter"),
            ("request", "emitter"),
            ("execution", "emitter"),
            ("conditional", "emitter"),
            ("request", "emitter"),
        ]

    async def test_async_request_intercept_runs_on_originating_loop(self):
        originating_loop = asyncio.get_running_loop()

        async def intercept_fn(_name, args):
            await asyncio.sleep(0)
            assert asyncio.get_running_loop() is originating_loop
            return {**args, "intercepted": True}

        intercepts.register_tool_request("py_async_request_loop", 1, False, intercept_fn)
        try:
            result = await tools.execute("intercepted_tool", {}, lambda args: ToolExecutionResult(args))
        finally:
            intercepts.deregister_tool_request("py_async_request_loop")

        assert result.result == {"intercepted": True}

    async def test_request_intercept_modifies_args(self):
        def intercept_fn(name, args):
            args["intercepted"] = True
            return args

        intercepts.register_tool_request("py_req_mod", 1, False, intercept_fn)

        def func(args):
            return ToolExecutionResult(args)

        result = await tools.execute("intercepted_tool", {"original": True}, func)
        assert result.result["original"] is True
        assert result.result["intercepted"] is True

        intercepts.deregister_tool_request("py_req_mod")

    async def test_execution_intercept_replaces_func(self):
        intercepts.register_tool_execution(
            "py_exec_replace",
            1,
            lambda name, args, next: ToolExecutionInterceptOutcome({"from_intercept": True}),
        )

        def original_func(args):
            return ToolExecutionResult({"from_original": True})

        result = await tools.execute("replaced_tool", {}, original_func)
        assert result.result["from_intercept"] is True
        assert "from_original" not in result.result

        intercepts.deregister_tool_execution("py_exec_replace")

    async def test_execution_intercept_can_await_next(self):
        events = []

        async def middleware(name, args, next):
            downstream = await next({"value": args["value"] + 1})
            result = dict(downstream.result)
            result["from_intercept"] = True
            return ToolExecutionInterceptOutcome(
                result,
                [PendingMarkSpec("python.tool.execution")],
                annotation=downstream.annotation,
            )

        intercepts.register_tool_execution("py_exec_next", 1, middleware)
        subscribers.register("py_exec_mark_sub", lambda event: events.append(event))

        def original(args):
            return ToolExecutionResult({"value": args["value"] * 2}, {"source": "provider"})

        try:
            result = await tools.execute("next_tool", {"value": 2}, original)
            assert result.result == {"value": 6, "from_intercept": True}
            assert result.annotation == {"source": "provider"}
            await subscribers.flush_async()
            start = _tool_event(events, "next_tool", "start")
            end = _tool_event(events, "next_tool", "end")
            assert end.category_profile == {
                "tool_result_annotation": {"source": "provider"},
            }
            mark = next(
                event for event in events if isinstance(event, MarkEvent) and event.name == "python.tool.execution"
            )
            assert mark.parent_uuid == start.uuid
            assert events.index(end) < events.index(mark)
        finally:
            intercepts.deregister_tool_execution("py_exec_next")
            subscribers.deregister("py_exec_mark_sub")

    async def test_execution_intercept_rejects_detached_next_after_settlement(self):
        release_late_next = asyncio.Event()
        late_task: asyncio.Task[dict] | None = None
        provider_calls: list[dict] = []

        async def middleware(_name, args, next):
            nonlocal late_task

            async def invoke_late():
                await release_late_next.wait()
                return await next(args)

            late_task = asyncio.create_task(invoke_late())
            return ToolExecutionInterceptOutcome({"source": "intercept"})

        def provider(args):
            provider_calls.append(args)
            return ToolExecutionResult(args)

        intercepts.register_tool_execution("py_exec_late_next", 1, middleware)
        try:
            result = await tools.execute("late_next_tool", {"value": 1}, provider)
            assert result.result == {"source": "intercept"}
            assert late_task is not None
            release_late_next.set()
            with pytest.raises(RuntimeError, match="execution continuation is no longer active"):
                await late_task
        finally:
            release_late_next.set()
            if late_task is not None and not late_task.done():
                late_task.cancel()
            intercepts.deregister_tool_execution("py_exec_late_next")

        assert provider_calls == []

    async def test_execution_intercept_isolates_concurrent_next_scope_branches(self):
        both_pushed = asyncio.Event()
        pushed = 0

        async def middleware(_name, _args, next):
            first, second = await asyncio.gather(
                next({"branch": "first"}),
                next({"branch": "second"}),
            )
            return ToolExecutionInterceptOutcome([first.result, second.result])

        async def provider(args):
            nonlocal pushed
            handle = scope.push(f"python-next-{args['branch']}", ScopeType.Custom)
            try:
                pushed += 1
                if pushed == 2:
                    both_pushed.set()
                await both_pushed.wait()
                if args["branch"] == "first":
                    await asyncio.sleep(0)
                assert scope.get_handle().uuid == handle.uuid
                return ToolExecutionResult(args)
            finally:
                scope.pop(handle)

        intercepts.register_tool_execution("py_exec_concurrent_next_scopes", 1, middleware)
        try:
            result = await tools.execute("concurrent_next_tool", {}, provider)
            assert result.result == [{"branch": "first"}, {"branch": "second"}]
        finally:
            intercepts.deregister_tool_execution("py_exec_concurrent_next_scopes")

    async def test_execution_next_honors_concurrent_scope_stack_replacements(self):
        first_stack = create_scope_stack()
        second_stack = create_scope_stack()
        both_entered = asyncio.Event()
        entered = 0
        with use_scope_stack(first_stack):
            first_scope = scope.get_handle().uuid
        with use_scope_stack(second_stack):
            second_scope = scope.get_handle().uuid

        async def middleware(_name, _args, next):
            async def invoke(stack, branch):
                nonlocal entered
                with use_scope_stack(stack):
                    entered += 1
                    if entered == 2:
                        both_entered.set()
                    await both_entered.wait()
                    await asyncio.sleep(0)
                    return await next({"branch": branch})

            first, second = await asyncio.gather(
                invoke(first_stack, "first"),
                invoke(second_stack, "second"),
            )
            return ToolExecutionInterceptOutcome([first.result, second.result])

        async def provider(args):
            await asyncio.sleep(0)
            return ToolExecutionResult({"branch": args["branch"], "scope": scope.get_handle().uuid})

        intercepts.register_tool_execution("py_exec_replaced_next_scopes", 1, middleware)
        try:
            result = await tools.execute("replaced_next_scopes", {}, provider)
            assert result.result == [
                {"branch": "first", "scope": first_scope},
                {"branch": "second", "scope": second_scope},
            ]
        finally:
            intercepts.deregister_tool_execution("py_exec_replaced_next_scopes")

    async def test_plain_next_uses_each_concurrent_middleware_invocation_scope(self):
        first_stack = create_scope_stack()
        second_stack = create_scope_stack()
        both_entered = asyncio.Event()
        entered = 0

        async def middleware(_name, args, next):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()
            await asyncio.sleep(0)
            downstream = await next(args)
            return ToolExecutionInterceptOutcome(
                downstream.result,
                annotation=downstream.annotation,
            )

        async def provider(args):
            await asyncio.sleep(0)
            return ToolExecutionResult({"branch": args["branch"], "scope": scope.get_handle().uuid})

        async def invoke(stack, branch):
            with use_scope_stack(stack):
                expected_scope = scope.get_handle().uuid
                result = await tools.execute(
                    f"plain_next_{branch}",
                    {"branch": branch},
                    provider,
                )
                return result.result, expected_scope

        intercepts.register_tool_execution("py_exec_plain_next_scopes", 1, middleware)
        try:
            (first, first_scope), (second, second_scope) = await asyncio.gather(
                invoke(first_stack, "first"),
                invoke(second_stack, "second"),
            )
            assert first == {"branch": "first", "scope": first_scope}
            assert second == {"branch": "second", "scope": second_scope}
        finally:
            intercepts.deregister_tool_execution("py_exec_plain_next_scopes")

    async def test_execution_intercept_rejects_legacy_raw_result(self):
        intercepts.register_tool_execution(
            "py_exec_legacy",
            1,
            lambda name, args, next: {"legacy_result": True},  # type: ignore[arg-type] # ty: ignore[invalid-argument-type]
        )
        try:
            with pytest.raises(RuntimeError, match="must return ToolExecutionInterceptOutcome") as error:
                await tools.execute("legacy_tool", {}, lambda args: ToolExecutionResult(args))
            assert "after await next(args)" in str(error.value)
            assert "internal error" not in str(error.value)
        finally:
            intercepts.deregister_tool_execution("py_exec_legacy")

    async def test_execution_intercept_rejects_downstream_result_without_unwrapping(self):
        async def leftover(_name, args, next):
            downstream = await next(args)
            return ToolExecutionInterceptOutcome(downstream)  # type: ignore[arg-type]

        intercepts.register_tool_execution("py_exec_leftover", 1, leftover)
        try:
            with pytest.raises(RuntimeError, match="must return ToolExecutionInterceptOutcome") as error:
                await tools.execute("leftover_tool", {}, lambda args: ToolExecutionResult(args))
            assert "result=downstream.result, annotation=downstream.annotation" in str(error.value)
            assert "internal error" not in str(error.value)
        finally:
            intercepts.deregister_tool_execution("py_exec_leftover")

    async def test_execution_intercept_rejects_downstream_result_as_its_outcome(self):
        async def leftover_return(_name, args, next):
            return await next(args)  # type: ignore[return-value]

        intercepts.register_tool_execution("py_exec_leftover_return", 1, leftover_return)
        try:
            with pytest.raises(RuntimeError, match="must return ToolExecutionInterceptOutcome") as error:
                await tools.execute("leftover_return_tool", {}, lambda args: ToolExecutionResult(args))
            assert "result=downstream.result, annotation=downstream.annotation" in str(error.value)
            assert "internal error" not in str(error.value)
        finally:
            intercepts.deregister_tool_execution("py_exec_leftover_return")

    async def test_request_intercept_break_chain(self):
        def first_fn(name, args):
            args["from_first"] = True
            return args

        def second_fn(name, args):
            args["from_second"] = True
            return args

        intercepts.register_tool_request("py_chain1", 1, True, first_fn)
        intercepts.register_tool_request("py_chain2", 2, False, second_fn)

        def func(args):
            return ToolExecutionResult(args)

        result = await tools.execute("chain_tool", {}, func)
        assert result.result["from_first"] is True
        assert "from_second" not in result.result

        intercepts.deregister_tool_request("py_chain1")
        intercepts.deregister_tool_request("py_chain2")


class TestToolGuardrailsEdgeCases:
    def test_conditional_execution_invalid_return_type_raises(self):
        guardrails.register_tool_conditional_execution(
            "py_cond_bad_type",
            1,
            cast(guardrails.ToolConditionalExecutionGuardrail, lambda name, args: 123),
        )
        try:
            with pytest.raises(RuntimeError, match="unexpected type"):
                tools.conditional_execution("bad_type_tool", {})
        finally:
            guardrails.deregister_tool_conditional_execution("py_cond_bad_type")

    def test_conditional_execution_callable_error_raises(self):
        guardrails.register_tool_conditional_execution(
            "py_cond_error",
            1,
            lambda name, args: raise_runtime_error("boom"),
        )
        try:
            with pytest.raises(RuntimeError, match="RuntimeError: boom"):
                tools.conditional_execution("error_tool", {})
        finally:
            guardrails.deregister_tool_conditional_execution("py_cond_error")
