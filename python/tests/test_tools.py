# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Relay tool lifecycle, guardrails, and intercepts."""

from collections import UserDict, UserList
from dataclasses import dataclass
from typing import cast

import pytest

from nemo_relay import (
    Event,
    MarkEvent,
    PendingMarkSpec,
    ScopeEvent,
    ScopeType,
    ToolAttributes,
    ToolExecutionInterceptOutcome,
    ToolHandle,
    guardrails,
    intercepts,
    scope,
    subscribers,
    tools,
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
        tools.call_end(handle, {"output": "result"})

    def test_call_with_attributes(self):
        attrs = ToolAttributes(ToolAttributes.REMOTE)
        handle = tools.call("local_tool", {"x": 1}, attributes=attrs)
        assert handle.name == "local_tool"
        tools.call_end(handle, {"y": 2})

    def test_call_with_data_metadata(self):
        handle = tools.call(
            "tool_dm",
            {"arg": 1},
            data={"custom": "info"},
            metadata={"trace_id": "abc123"},
        )
        tools.call_end(handle, "ok", data={"end_data": True}, metadata={"end_meta": True})

    def test_call_with_parent_handle(self):
        parent = scope.push("tool_parent", ScopeType.Agent)
        handle = tools.call("child_tool", {}, handle=parent)
        assert handle.parent_uuid == parent.uuid
        tools.call_end(handle, {})
        scope.pop(parent)

    def test_complete_skill_read_emits_minimal_eager_mark(self, subscribed_events: list[Event]):
        handle = tools.call("read_file", {"path": "/skills/review/SKILL.md"})
        tools.call_end(handle, {"ok": True})
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
            return {"result": args["x"] * 2}

        result = await tools.execute("double", {"x": 5}, my_func)
        assert result == {"result": 10}

    async def test_execute_rejects_cyclic_results_and_remains_usable(self):
        def cyclic_result(_args):
            result = {}
            result["self"] = result
            return result

        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("cyclic_result", {}, cyclic_result)

        async def async_cyclic_result(_args):
            return cyclic_result(_args)

        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("async_cyclic_result", {}, async_cyclic_result)

        cyclic_mapping = UserDict()
        cyclic_mapping["self"] = cyclic_mapping
        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("cyclic_mapping", {}, lambda _args: cyclic_mapping)

        cyclic_sequence = UserList()
        cyclic_sequence.append(cyclic_sequence)
        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("cyclic_sequence", {}, lambda _args: cyclic_sequence)

        @dataclass
        class CyclicDataclass:
            child: object | None = None

        cyclic_dataclass = CyclicDataclass()
        cyclic_dataclass.child = cyclic_dataclass
        with pytest.raises(RuntimeError, match="circular reference detected"):
            await tools.execute("cyclic_dataclass", {}, lambda _args: cyclic_dataclass)

        result = await tools.execute(
            "post_cycle_result",
            {},
            lambda _args: {"status": "ok"},
        )
        assert result == {"status": "ok"}

    async def test_execute_allows_shared_non_cyclic_results(self):
        shared = {"value": True}

        result = await tools.execute(
            "shared_result",
            {},
            lambda _args: {"first": shared, "second": shared},
        )

        assert result == {
            "first": {"value": True},
            "second": {"value": True},
        }

    async def test_execute_returns_string(self):
        def func(args):
            return "hello"

        result = await tools.execute("str_tool", {}, func)
        assert result == "hello"

    async def test_execute_with_attributes(self):
        def func(args):
            return args

        attrs = ToolAttributes(ToolAttributes.REMOTE)
        result = await tools.execute(
            "attr_tool",
            {"test": True},
            func,
            attributes=attrs,
        )
        assert result["test"] is True

    async def test_execute_async_func(self):
        """tools.execute should accept async functions."""

        async def my_async_func(args):
            return {"result": args["x"] + 1}

        result = await tools.execute("async_tool", {"x": 10}, my_async_func)
        assert result == {"result": 11}

    async def test_execute_async_func_returns_string(self):
        async def func(args):
            return "async_hello"

        result = await tools.execute("async_str_tool", {}, func)
        assert result == "async_hello"

    async def test_execute_async_func_with_attributes(self):
        async def func(args):
            return args

        attrs = ToolAttributes(ToolAttributes.REMOTE)
        result = await tools.execute(
            "async_attr_tool",
            {"key": "value"},
            func,
            attributes=attrs,
        )
        assert result["key"] == "value"

    async def test_execute_failure_emits_end_event(self):
        events = []
        subscribers.register("py_tool_exec_failure_sub", lambda e: events.append(e))

        def failing(args):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await tools.execute("failing_tool", {"x": 1}, failing)

        try:
            subscribers.flush()
        finally:
            subscribers.deregister("py_tool_exec_failure_sub")

        assert [e.kind for e in events] == ["scope", "scope"]
        assert all(isinstance(event, ScopeEvent) for event in events)
        assert [e.scope_category for e in events] == ["start", "end"]
        assert all(e.category == "tool" for e in events)
        assert events[0].uuid == events[1].uuid
        assert events[1].data is None


class TestToolGuardrails:
    def test_sanitize_request_guardrail(self):
        def sanitizer(name, args):
            args["sanitized"] = True
            return args

        guardrails.register_tool_sanitize_request("py_san_req", 1, sanitizer)

        events = []
        subscribers.register("py_san_req_sub", lambda e: events.append(e))
        handle = tools.call("guarded_tool", {"input": "data"})
        tools.call_end(handle, {})
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
        tools.call_end(handle, {"output": "raw"})
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

    def test_sanitize_request_failure_falls_back_to_original_input(self):
        events = []
        subscribers.register("py_tool_sanitize_req_sub", lambda event: events.append(event))
        guardrails.register_tool_sanitize_request(
            "py_tool_sanitize_req_fail",
            1,
            lambda name, args: raise_runtime_error("boom"),
        )
        try:
            handle = tools.call("tool_sanitize_req_fail", {"value": 1})
            tools.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_tool_sanitize_request("py_tool_sanitize_req_fail")
            subscribers.flush()
            subscribers.deregister("py_tool_sanitize_req_sub")

        start = _tool_event(events, "tool_sanitize_req_fail", "start")
        assert start.data == {"value": 1}

    def test_sanitize_response_invalid_return_falls_back_to_original_output(self):
        events = []
        subscribers.register("py_tool_sanitize_resp_sub", lambda event: events.append(event))
        guardrails.register_tool_sanitize_response(
            "py_tool_sanitize_resp_bad",
            1,
            cast(guardrails.ToolSanitizeGuardrail, lambda name, result: object()),
        )
        try:
            handle = tools.call("tool_sanitize_resp_bad", {"value": 1})
            tools.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_tool_sanitize_response("py_tool_sanitize_resp_bad")
            subscribers.flush()
            subscribers.deregister("py_tool_sanitize_resp_sub")

        end = _tool_event(events, "tool_sanitize_resp_bad", "end")
        assert end.data == {"ok": True}

    def test_deregister_nonexistent(self):
        assert not guardrails.deregister_tool_sanitize_request("nonexistent")
        assert not guardrails.deregister_tool_sanitize_response("nonexistent")
        assert not guardrails.deregister_tool_conditional_execution("nonexistent")


class TestToolGuardrailsAsync:
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

        assert transformed["direct"] is True

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
            with pytest.raises(RuntimeError, match="callable failed"):
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
            with pytest.raises(RuntimeError, match="py_to_json failed"):
                tools.request_intercepts("bad_return_tool", {"value": 1})
        finally:
            intercepts.deregister_tool_request("py_req_bad_return")


class TestToolInterceptsAsync:
    async def test_request_intercept_modifies_args(self):
        def intercept_fn(name, args):
            args["intercepted"] = True
            return args

        intercepts.register_tool_request("py_req_mod", 1, False, intercept_fn)

        def func(args):
            return args

        result = await tools.execute("intercepted_tool", {"original": True}, func)
        assert result["original"] is True
        assert result["intercepted"] is True

        intercepts.deregister_tool_request("py_req_mod")

    async def test_execution_intercept_replaces_func(self):
        intercepts.register_tool_execution(
            "py_exec_replace",
            1,
            lambda name, args, next: ToolExecutionInterceptOutcome({"from_intercept": True}),
        )

        def original_func(args):
            return {"from_original": True}

        result = await tools.execute("replaced_tool", {}, original_func)
        assert result["from_intercept"] is True
        assert "from_original" not in result

        intercepts.deregister_tool_execution("py_exec_replace")

    async def test_execution_intercept_can_await_next(self):
        events = []

        async def middleware(name, args, next):
            result = await next({"value": args["value"] + 1})
            result["from_intercept"] = True
            return ToolExecutionInterceptOutcome(result, [PendingMarkSpec("python.tool.execution")])

        intercepts.register_tool_execution("py_exec_next", 1, middleware)
        subscribers.register("py_exec_mark_sub", lambda event: events.append(event))

        def original(args):
            return {"value": args["value"] * 2}

        try:
            result = await tools.execute("next_tool", {"value": 2}, original)
            assert result == {"value": 6, "from_intercept": True}
            subscribers.flush()
            start = _tool_event(events, "next_tool", "start")
            end = _tool_event(events, "next_tool", "end")
            mark = next(
                event for event in events if isinstance(event, MarkEvent) and event.name == "python.tool.execution"
            )
            assert mark.parent_uuid == start.uuid
            assert events.index(end) < events.index(mark)
        finally:
            intercepts.deregister_tool_execution("py_exec_next")
            subscribers.deregister("py_exec_mark_sub")

    async def test_execution_intercept_rejects_legacy_raw_result(self):
        intercepts.register_tool_execution(
            "py_exec_legacy",
            1,
            lambda name, args, next: {"legacy_result": True},  # type: ignore[arg-type] # ty: ignore[invalid-argument-type]
        )
        try:
            with pytest.raises(RuntimeError, match="must return ToolExecutionInterceptOutcome"):
                await tools.execute("legacy_tool", {}, lambda args: args)
        finally:
            intercepts.deregister_tool_execution("py_exec_legacy")

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
            return args

        result = await tools.execute("chain_tool", {}, func)
        assert result["from_first"] is True
        assert "from_second" not in result

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
            with pytest.raises(RuntimeError, match="expected str or None"):
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
            with pytest.raises(RuntimeError, match="callable failed"):
                tools.conditional_execution("error_tool", {})
        finally:
            guardrails.deregister_tool_conditional_execution("py_cond_error")
