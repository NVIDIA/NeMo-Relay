# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Agent Toolkit Nexus LLM lifecycle, guardrails, intercepts, and streaming."""

import pytest
from nat_nexus import (
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    ScopeType,
    guardrails,
    intercepts,
    llm,
    scope,
)


def make_request():
    return LLMRequest({}, {"messages": [], "model": "test-model"})


class TestLLM:
    def test_call_and_call_end(self):
        request = make_request()
        handle = llm.call("my_llm", request)
        assert isinstance(handle, LLMHandle)
        assert handle.name == "my_llm"
        llm.call_end(handle, {"response": "ok"})

    def test_call_with_attributes(self):
        request = make_request()
        attrs = LLMAttributes(LLMAttributes.STREAMING)
        handle = llm.call("streaming_llm", request, attributes=attrs)
        llm.call_end(handle, {})

    def test_call_with_data_metadata(self):
        request = make_request()
        handle = llm.call(
            "llm_dm",
            request,
            data={"custom": "data"},
            metadata={"trace": "xyz"},
        )
        llm.call_end(handle, {"result": "ok"}, data={"end": True})

    def test_call_with_parent(self):
        parent = scope.push("llm_parent", ScopeType.Agent)
        request = make_request()
        handle = llm.call("child_llm", request, handle=parent)
        assert handle.parent_uuid == parent.uuid
        llm.call_end(handle, {})
        scope.pop(parent)


class TestLLMAsync:
    async def test_execute_basic(self):
        # LLM execute receives an LLMRequest object
        def func(request):
            return {"model": request.content["model"]}

        request = make_request()
        result = await llm.execute("exec_llm", request, func)
        assert result["model"] == "test-model"

    async def test_execute_with_sync_func(self):
        def func(request):
            return {"echoed_messages": request.content["messages"]}

        request = make_request()
        result = await llm.execute("sync_llm", request, func)
        assert result["echoed_messages"] == []

    async def test_execute_async_func(self):
        """llm.execute should accept async functions."""

        async def func(request):
            return {"model": request.content["model"], "async": True}

        request = make_request()
        result = await llm.execute("async_exec_llm", request, func)
        assert result["model"] == "test-model"
        assert result["async"] is True

    async def test_execute_async_func_with_messages(self):
        async def func(request):
            return {"messages": request.content["messages"]}

        request = make_request()
        result = await llm.execute("async_method_llm", request, func)
        assert result["messages"] == []


class TestLLMGuardrails:
    def test_sanitize_request_guardrail(self):
        def sanitizer(request):
            # request is an LLMRequest object; must return a new LLMRequest
            headers = request.headers
            headers["X-Sanitized"] = "true"
            return LLMRequest(headers, request.content)

        guardrails.register_llm_sanitize_request("py_llm_san_req", 1, sanitizer)
        guardrails.deregister_llm_sanitize_request("py_llm_san_req")

    def test_sanitize_response_guardrail(self):
        def sanitizer(response):
            # response is a plain dict
            response["cleaned"] = True
            return response

        guardrails.register_llm_sanitize_response("py_llm_san_resp", 1, sanitizer)
        guardrails.deregister_llm_sanitize_response("py_llm_san_resp")

    def test_conditional_execution_guardrail(self):
        def checker(request):
            return None

        guardrails.register_llm_conditional_execution("py_llm_cond", 1, checker)
        guardrails.deregister_llm_conditional_execution("py_llm_cond")

    def test_duplicate_raises(self):
        guardrails.register_llm_sanitize_request("py_llm_dup", 1, lambda r: r)
        with pytest.raises(RuntimeError):
            guardrails.register_llm_sanitize_request("py_llm_dup", 1, lambda r: r)
        guardrails.deregister_llm_sanitize_request("py_llm_dup")

    def test_deregister_nonexistent(self):
        assert not guardrails.deregister_llm_sanitize_request("nope")
        assert not guardrails.deregister_llm_sanitize_response("nope")
        assert not guardrails.deregister_llm_conditional_execution("nope")


class TestLLMGuardrailsAsync:
    async def test_conditional_blocks_execution(self):
        guardrails.register_llm_conditional_execution("py_llm_blocker", 1, lambda req: "LLM blocked")

        def func(request):
            return {"should": "not reach"}

        request = make_request()
        with pytest.raises(RuntimeError, match="guardrail rejected"):
            await llm.execute("blocked_llm", request, func)

        guardrails.deregister_llm_conditional_execution("py_llm_blocker")


class TestLLMIntercepts:
    def test_request_intercept(self):
        # Request intercepts now operate on LLMRequest
        intercepts.register_llm_request("py_llm_req", 1, False, lambda request: request)
        assert intercepts.deregister_llm_request("py_llm_req")

    def test_execution_intercept(self):
        # Execution intercepts now take LLMRequest
        intercepts.register_llm_execution(
            "py_llm_exec",
            1,
            lambda request, next: {"intercepted": True},
        )
        assert intercepts.deregister_llm_execution("py_llm_exec")

    def test_stream_execution_intercept(self):
        def stream_fn(request, next):
            async def gen():
                yield {"token": "test"}

            return gen()

        intercepts.register_llm_stream_execution(
            "py_llm_sexec",
            1,
            stream_fn,
        )
        assert intercepts.deregister_llm_stream_execution("py_llm_sexec")

    def test_deregister_nonexistent(self):
        assert not intercepts.deregister_llm_request("nope")
        assert not intercepts.deregister_llm_execution("nope")
        assert not intercepts.deregister_llm_stream_execution("nope")
        assert not intercepts.deregister_tool_request("nope")
        assert not intercepts.deregister_tool_response("nope")
        assert not intercepts.deregister_tool_execution("nope")


class TestLLMInterceptsAsync:
    async def test_request_intercept_modifies(self):
        def intercept_fn(request):
            # Request intercepts now operate on LLMRequest
            content = request.content
            content["intercepted"] = True
            return LLMRequest(request.headers, content)

        intercepts.register_llm_request("py_llm_req_mod", 1, False, intercept_fn)

        def func(request):
            return {"saw_intercepted": request.content.get("intercepted", False)}

        request = make_request()
        result = await llm.execute("int_llm", request, func)
        assert result["saw_intercepted"] is True

        intercepts.deregister_llm_request("py_llm_req_mod")

    async def test_execution_intercept_replaces(self):
        intercepts.register_llm_execution(
            "py_llm_exec_rep",
            1,
            lambda request, next: {"from_intercept": True},
        )

        def original_func(request):
            return {"from_original": True}

        request = make_request()
        result = await llm.execute("exec_llm", request, original_func)
        assert result["from_intercept"] is True
        assert "from_original" not in result

        intercepts.deregister_llm_execution("py_llm_exec_rep")


class TestLLMStreaming:
    async def test_stream_execute(self):
        # Stream functions now take LLMRequest and return async iterator of Json
        def stream_func(request):
            async def gen():
                yield {"token": "hello"}
                yield {"token": "world"}

            return gen()

        collected = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return {"chunks": collected}

        request = make_request()
        stream = await llm.stream_execute("stream_llm", request, stream_func, collector, finalizer)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 2
        # Collector should have received all chunks
        assert len(collected) == len(chunks)
