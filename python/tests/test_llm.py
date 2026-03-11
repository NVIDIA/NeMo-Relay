# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVMagic LLM lifecycle, guardrails, intercepts, and streaming."""

import pytest
from nvmagic import (
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    LLMResponse,
    ScopeType,
    guardrails,
    intercepts,
    llm,
    scope,
)


def make_native():
    return {"messages": [], "model": "test-model"}


class TestLLM:
    def test_call_and_call_end(self):
        native = make_native()
        handle = llm.call("my_llm", native)
        assert isinstance(handle, LLMHandle)
        assert handle.name == "my_llm"
        llm.call_end(handle, {"response": "ok"})

    def test_call_with_attributes(self):
        native = make_native()
        attrs = LLMAttributes(LLMAttributes.STREAMING)
        handle = llm.call("streaming_llm", native, attributes=attrs)
        llm.call_end(handle, {})

    def test_call_with_data_metadata(self):
        native = make_native()
        handle = llm.call(
            "llm_dm",
            native,
            data={"custom": "data"},
            metadata={"trace": "xyz"},
        )
        llm.call_end(handle, {"result": "ok"}, data={"end": True})

    def test_call_with_parent(self):
        parent = scope.push("llm_parent", ScopeType.Agent)
        native = make_native()
        handle = llm.call("child_llm", native, handle=parent)
        assert handle.parent_uuid == parent.uuid
        llm.call_end(handle, {})
        scope.pop(parent)


class TestLLMAsync:
    async def test_execute_basic(self):
        # LLM execute receives native Json dict, not an LLMRequest object
        def func(native):
            return {"model": native["model"]}

        native = make_native()
        result = await llm.execute("exec_llm", native, func)
        assert result["model"] == "test-model"

    async def test_execute_with_sync_func(self):
        def func(native):
            return {"echoed_messages": native["messages"]}

        native = make_native()
        result = await llm.execute("sync_llm", native, func)
        assert result["echoed_messages"] == []

    async def test_execute_async_func(self):
        """llm.execute should accept async functions."""

        async def func(native):
            return {"model": native["model"], "async": True}

        native = make_native()
        result = await llm.execute("async_exec_llm", native, func)
        assert result["model"] == "test-model"
        assert result["async"] is True

    async def test_execute_async_func_with_messages(self):
        async def func(native):
            return {"messages": native["messages"]}

        native = make_native()
        result = await llm.execute("async_method_llm", native, func)
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
            # response is an LLMResponse object
            data = response.data
            data["cleaned"] = True
            return LLMResponse(data)

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

        def func(native):
            return {"should": "not reach"}

        native = make_native()
        with pytest.raises(RuntimeError, match="guardrail rejected"):
            await llm.execute("blocked_llm", native, func)

        guardrails.deregister_llm_conditional_execution("py_llm_blocker")


class TestLLMIntercepts:
    def test_request_intercept(self):
        # Request intercepts now operate on opaque Json (dicts), not LLMRequest
        intercepts.register_llm_request("py_llm_req", 1, False, lambda native: native)
        assert intercepts.deregister_llm_request("py_llm_req")

    def test_response_intercept(self):
        # Response intercepts now operate on LLMResponse objects
        intercepts.register_llm_response("py_llm_resp", 1, False, lambda r: r)
        assert intercepts.deregister_llm_response("py_llm_resp")

    def test_stream_response_intercept(self):
        # Stream response intercepts now operate on Json chunks
        intercepts.register_llm_stream_response("py_llm_sr", 1, False, lambda e: e)
        assert intercepts.deregister_llm_stream_response("py_llm_sr")

    def test_execution_intercept(self):
        # Execution intercepts now take native Json, not LLMRequest
        intercepts.register_llm_execution(
            "py_llm_exec",
            1,
            lambda native: False,
            lambda native, next: {"intercepted": True},
        )
        assert intercepts.deregister_llm_execution("py_llm_exec")

    def test_stream_execution_intercept(self):
        def stream_fn(native, next):
            async def gen():
                yield {"token": "test"}

            return gen()

        intercepts.register_llm_stream_execution(
            "py_llm_sexec",
            1,
            lambda native: False,
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
        def intercept_fn(native):
            # Request intercepts now operate on opaque Json
            native["intercepted"] = True
            return native

        intercepts.register_llm_request("py_llm_req_mod", 1, False, intercept_fn)

        def func(native):
            return {"saw_intercepted": native.get("intercepted", False)}

        native = make_native()
        result = await llm.execute("int_llm", native, func)
        assert result["saw_intercepted"] is True

        intercepts.deregister_llm_request("py_llm_req_mod")

    async def test_response_intercept_modifies(self):
        def intercept_fn(response):
            # Response intercepts now take LLMResponse objects
            data = response.data
            data["modified"] = True
            return LLMResponse(data)

        intercepts.register_llm_response("py_llm_resp_mod", 1, False, intercept_fn)

        def func(native):
            return {"original": True}

        native = make_native()
        result = await llm.execute("resp_llm", native, func)
        assert result["original"] is True
        assert result["modified"] is True

        intercepts.deregister_llm_response("py_llm_resp_mod")

    async def test_execution_intercept_replaces(self):
        intercepts.register_llm_execution(
            "py_llm_exec_rep",
            1,
            lambda native: True,
            lambda native, next: {"from_intercept": True},
        )

        def original_func(native):
            return {"from_original": True}

        native = make_native()
        result = await llm.execute("exec_llm", native, original_func)
        assert result["from_intercept"] is True
        assert "from_original" not in result

        intercepts.deregister_llm_execution("py_llm_exec_rep")


class TestLLMStreaming:
    async def test_stream_execute(self):
        # Stream functions now take native Json and return async iterator of Json
        def stream_func(native):
            async def gen():
                yield {"token": "hello"}
                yield {"token": "world"}

            return gen()

        collected = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return {"chunks": collected}

        native = make_native()
        stream = await llm.stream_execute("stream_llm", native, stream_func, collector, finalizer)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 2
        # Collector should have received all chunks
        assert len(collected) == len(chunks)
