# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVAgentRT LLM lifecycle, guardrails, intercepts, and streaming."""

import pytest
from nvagentrt import (
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    ScopeType,
    guardrails,
    intercepts,
    llm,
    scope,
)


def make_request(url="https://api.example.com"):
    return LLMRequest("POST", url, {}, {"messages": []})


class TestLLM:
    def test_call_and_call_end(self):
        req = make_request()
        handle = llm.call("my_llm", req)
        assert isinstance(handle, LLMHandle)
        assert handle.name == "my_llm"
        llm.call_end(handle, {"response": "ok"})

    def test_call_with_attributes(self):
        req = make_request()
        attrs = LLMAttributes(LLMAttributes.STREAMING)
        handle = llm.call("streaming_llm", req, attributes=attrs)
        llm.call_end(handle, {})

    def test_call_with_data_metadata(self):
        req = make_request()
        handle = llm.call(
            "llm_dm",
            req,
            data={"custom": "data"},
            metadata={"trace": "xyz"},
        )
        llm.call_end(handle, {"result": "ok"}, data={"end": True})

    def test_call_with_parent(self):
        parent = scope.push("llm_parent", ScopeType.Agent)
        req = make_request()
        handle = llm.call("child_llm", req, handle=parent)
        assert handle.parent_uuid == parent.uuid
        llm.call_end(handle, {})
        scope.pop(parent)


class TestLLMAsync:
    async def test_execute_basic(self):
        # LLM execute receives LLMRequest object, not a dict
        def func(request):
            return {"model": request.url}

        req = make_request()
        result = await llm.execute("exec_llm", req, func)
        assert result["model"] == "https://api.example.com"

    async def test_execute_with_sync_func(self):
        def func(request):
            return {"echoed_method": request.method}

        req = make_request()
        result = await llm.execute("sync_llm", req, func)
        assert result["echoed_method"] == "POST"

    async def test_execute_async_func(self):
        """llm.execute should accept async functions."""

        async def func(request):
            return {"model": request.url, "async": True}

        req = make_request()
        result = await llm.execute("async_exec_llm", req, func)
        assert result["model"] == "https://api.example.com"
        assert result["async"] is True

    async def test_execute_async_func_with_method(self):
        async def func(request):
            return {"method": request.method}

        req = make_request()
        result = await llm.execute("async_method_llm", req, func)
        assert result["method"] == "POST"


class TestLLMGuardrails:
    def test_sanitize_request_guardrail(self):
        def sanitizer(request):
            # request is an LLMRequest object; must return a new LLMRequest
            headers = request.headers
            headers["X-Sanitized"] = "true"
            return LLMRequest(request.method, request.url, headers, request.body)

        guardrails.register_llm_sanitize_request("py_llm_san_req", 1, sanitizer)
        guardrails.deregister_llm_sanitize_request("py_llm_san_req")

    def test_sanitize_response_guardrail(self):
        def sanitizer(response):
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

        req = make_request()
        with pytest.raises(RuntimeError, match="guardrail rejected"):
            await llm.execute("blocked_llm", req, func)

        guardrails.deregister_llm_conditional_execution("py_llm_blocker")


class TestLLMIntercepts:
    def test_request_intercept(self):
        intercepts.register_llm_request("py_llm_req", 1, False, lambda r: r)
        assert intercepts.deregister_llm_request("py_llm_req")

    def test_response_intercept(self):
        intercepts.register_llm_response("py_llm_resp", 1, False, lambda r: r)
        assert intercepts.deregister_llm_response("py_llm_resp")

    def test_stream_response_intercept(self):
        intercepts.register_llm_stream_response("py_llm_sr", 1, False, lambda e: e)
        assert intercepts.deregister_llm_stream_response("py_llm_sr")

    def test_execution_intercept(self):
        intercepts.register_llm_execution(
            "py_llm_exec",
            1,
            lambda req: False,
            lambda req: {"intercepted": True},
        )
        assert intercepts.deregister_llm_execution("py_llm_exec")

    def test_stream_execution_intercept(self):
        def stream_fn(req):
            async def gen():
                yield "data: test\n\n"

            return gen()

        intercepts.register_llm_stream_execution(
            "py_llm_sexec",
            1,
            lambda req: False,
            stream_fn,
        )
        assert intercepts.deregister_llm_stream_execution("py_llm_sexec")

    def test_deregister_nonexistent(self):
        assert not intercepts.deregister_llm_request("nope")
        assert not intercepts.deregister_llm_response("nope")
        assert not intercepts.deregister_llm_stream_response("nope")
        assert not intercepts.deregister_llm_execution("nope")
        assert not intercepts.deregister_llm_stream_execution("nope")
        assert not intercepts.deregister_tool_request("nope")
        assert not intercepts.deregister_tool_response("nope")
        assert not intercepts.deregister_tool_execution("nope")


class TestLLMInterceptsAsync:
    async def test_request_intercept_modifies(self):
        def intercept_fn(request):
            # request is an LLMRequest object; must return a new LLMRequest
            return LLMRequest(request.method, "https://intercepted.com", request.headers, request.body)

        intercepts.register_llm_request("py_llm_req_mod", 1, False, intercept_fn)

        def func(request):
            return {"called_url": request.url}

        req = make_request()
        result = await llm.execute("int_llm", req, func)
        assert result["called_url"] == "https://intercepted.com"

        intercepts.deregister_llm_request("py_llm_req_mod")

    async def test_response_intercept_modifies(self):
        def intercept_fn(response):
            response["modified"] = True
            return response

        intercepts.register_llm_response("py_llm_resp_mod", 1, False, intercept_fn)

        def func(request):
            return {"original": True}

        req = make_request()
        result = await llm.execute("resp_llm", req, func)
        assert result["original"] is True
        assert result["modified"] is True

        intercepts.deregister_llm_response("py_llm_resp_mod")

    async def test_execution_intercept_replaces(self):
        intercepts.register_llm_execution(
            "py_llm_exec_rep",
            1,
            lambda req: True,
            lambda req: {"from_intercept": True},
        )

        def original_func(request):
            return {"from_original": True}

        req = make_request()
        result = await llm.execute("exec_llm", req, original_func)
        assert result["from_intercept"] is True
        assert "from_original" not in result

        intercepts.deregister_llm_execution("py_llm_exec_rep")


class TestLLMStreaming:
    async def test_stream_execute(self):
        # Stream functions should be sync, returning an async generator
        def stream_func(request):
            async def gen():
                yield 'data: {"token": "hello"}\n\n'
                yield 'data: {"token": "world"}\n\n'

            return gen()

        collected = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return {"chunks": collected}

        req = make_request()
        stream = await llm.stream_execute("stream_llm", req, stream_func, collector, finalizer)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 2
        # Collector should have received all chunks
        assert len(collected) == len(chunks)
