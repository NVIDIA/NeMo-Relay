# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-invocation execution stage controls across public and typed wrappers."""

import asyncio
from itertools import product

import pytest

import nemo_relay as relay
from nemo_relay import ExecutionOptions, ToolExecutionResult, guardrails, intercepts, llm, tools, typed


@pytest.mark.parametrize("conditional,requests,execution", product((False, True), repeat=3))
@pytest.mark.parametrize("kind", ["tool", "llm", "stream", "typed_tool", "typed_llm", "typed_stream"])
async def test_execution_stage_combinations(conditional, requests, execution, kind, subscribed_events):
    calls = []
    is_tool = kind.endswith("tool")
    name = "stage-options"

    def check(*args):
        calls.append("conditional")
        return None

    def rewrite(*args):
        calls.append("request")
        if is_tool:
            return {"rewritten": True}
        return relay.LLMRequestInterceptOutcome(relay.LLMRequest({}, {"rewritten": True}), None)

    async def wrap(*args):
        calls.append("execution")
        if is_tool:
            context, next_call = args
            result = await next_call(context.args)
            return relay.ToolExecutionInterceptOutcome(result.result, annotation=result.annotation)
        _, request_arg, _, next_call = args
        return await next_call(request_arg)

    conditional_register = (
        guardrails.register_tool_conditional_execution if is_tool else guardrails.register_llm_conditional_execution
    )
    conditional_remove = (
        guardrails.deregister_tool_conditional_execution if is_tool else guardrails.deregister_llm_conditional_execution
    )
    request_register = intercepts.register_tool_request if is_tool else intercepts.register_llm_request
    request_remove = intercepts.deregister_tool_request if is_tool else intercepts.deregister_llm_request
    if is_tool:
        execution_register, execution_remove = intercepts.register_tool_execution, intercepts.deregister_tool_execution
    elif kind.endswith("stream"):
        execution_register, execution_remove = (
            intercepts.register_llm_stream_execution,
            intercepts.deregister_llm_stream_execution,
        )
    else:
        execution_register, execution_remove = intercepts.register_llm_execution, intercepts.deregister_llm_execution
    conditional_register(name, 0, check)
    request_register(name, 0, False, rewrite)
    execution_register(name, 0, wrap)
    try:
        opts = ExecutionOptions(
            conditional_execution=conditional, request_intercepts=requests, execution_intercepts=execution
        )
        expected = {"rewritten": True} if requests else {"original": True}

        async def tool(arg: relay.Json) -> ToolExecutionResult[relay.Json]:
            calls.append("callback")
            assert arg == expected
            return ToolExecutionResult(arg, {"opaque": True})

        async def provider(req):
            calls.append("callback")
            assert req.content == expected
            return req.content

        async def stream(req):
            calls.append("callback")
            assert req.content == expected
            yield req.content

        original: relay.Json = {"original": True}
        codec = typed.JsonPassthrough()
        request_arg = relay.LLMRequest({}, {"original": True})
        chunks = []
        if kind == "tool":
            result = await tools.execute(name, {"original": True}, tool, execution_options=opts)
        elif kind == "typed_tool":
            result = await typed.tool_execute(name, original, tool, codec, codec, execution_options=opts)
        elif kind == "llm":
            result = await llm.execute(name, request_arg, provider, execution_options=opts)
        elif kind == "typed_llm":
            result = await typed.llm_execute(name, request_arg, provider, codec, execution_options=opts)
        else:
            if kind == "stream":
                result = await llm.stream_execute(
                    name, request_arg, stream, chunks.append, lambda: chunks, execution_options=opts
                )
            else:
                result = await typed.llm_stream_execute(
                    name, request_arg, stream, chunks.append, lambda: chunks, codec, codec, execution_options=opts
                )
            assert [chunk async for chunk in result] == [expected]
        if is_tool:
            assert isinstance(result, ToolExecutionResult)
            assert result.result == expected
            assert result.annotation == {"opaque": True}
        elif not kind.endswith("stream"):
            assert result == expected
        assert calls == (["conditional"] if conditional else []) + (["request"] if requests else []) + (
            ["execution"] if execution else []
        ) + ["callback"]
        await relay.subscribers.flush_async()
        lifecycle = [
            e.scope_category
            for e in subscribed_events
            if isinstance(e, relay.ScopeEvent) and e.name == name and e.category in ("tool", "llm")
        ]
        assert lifecycle == ["start", "end"]
    finally:
        conditional_remove(name)
        request_remove(name)
        execution_remove(name)


async def test_options_do_not_inherit_into_nested_or_concurrent_calls():
    intercepts.register_tool_request("options-isolation", 0, False, lambda name, args: {"intercepted": True})
    try:

        async def echo(args):
            await asyncio.sleep(0)
            return ToolExecutionResult(args)

        async def outer(args):
            assert args == {}
            return await tools.execute("nested", {}, echo)

        skipped = ExecutionOptions(request_intercepts=False)
        outer_result, skipped_result, default_result = await asyncio.gather(
            tools.execute("outer", {}, outer, execution_options=skipped),
            tools.execute("skipped", {}, echo, execution_options=skipped),
            tools.execute("default", {}, echo),
        )
        assert outer_result.result == default_result.result == {"intercepted": True}
        assert skipped_result.result == {}
    finally:
        intercepts.deregister_tool_request("options-isolation")


async def test_conditional_bypass_is_explicit_and_options_are_immutable():
    options = ExecutionOptions(conditional_execution=False)
    with pytest.raises(AttributeError):
        setattr(options, "conditional_execution", True)
    guardrails.register_tool_conditional_execution("options-rejection", 0, lambda name, args: "blocked")
    try:
        with pytest.raises(RuntimeError, match="blocked"):
            await tools.execute("rejected", {}, lambda args: ToolExecutionResult(args))
        result = await tools.execute("allowed", {}, lambda args: ToolExecutionResult(args), execution_options=options)
        assert result.result == {}
    finally:
        guardrails.deregister_tool_conditional_execution("options-rejection")


async def test_skipping_stages_preserves_sanitization_and_failure_completion(subscribed_events):
    options = ExecutionOptions(conditional_execution=False, request_intercepts=False, execution_intercepts=False)
    guardrails.register_tool_sanitize_request("options-sanitize", 0, lambda name, args: {"redacted": True})
    try:

        async def fail(args):
            assert args == {"secret": True}
            raise ValueError("provider failed")

        with pytest.raises(ValueError, match="provider failed"):
            await tools.execute("options-failure", {"secret": True}, fail, execution_options=options)
        await relay.subscribers.flush_async()
        events = [e for e in subscribed_events if isinstance(e, relay.ScopeEvent) and e.name == "options-failure"]
        assert [e.scope_category for e in events] == ["start", "end"]
        assert events[0].data == {"redacted": True}
    finally:
        guardrails.deregister_tool_sanitize_request("options-sanitize")


async def test_skipping_stream_intercepts_preserves_early_close(subscribed_events):
    closed = asyncio.Event()
    chunks = []

    async def provider(req):
        try:
            yield {"first": True}
            await asyncio.Event().wait()
        finally:
            closed.set()

    stream = await llm.stream_execute(
        "options-close",
        relay.LLMRequest({}, {}),
        provider,
        chunks.append,
        lambda: chunks,
        execution_options=ExecutionOptions(execution_intercepts=False),
    )
    assert await anext(stream) == {"first": True}
    await stream.aclose()
    assert closed.is_set()
    await relay.subscribers.flush_async()
    events = [e for e in subscribed_events if isinstance(e, relay.ScopeEvent) and e.name == "options-close"]
    assert [e.scope_category for e in events] == ["start", "end"]
