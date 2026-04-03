# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Agent Toolkit Nexus LLM lifecycle, guardrails, intercepts, and streaming."""

import pytest
from nat_nexus import (
    EventType,
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    ScopeType,
    guardrails,
    intercepts,
    llm,
    scope,
    subscribers,
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

    def test_conditional_execution_direct(self):
        guardrails.register_llm_conditional_execution("py_llm_cond_direct", 1, lambda request: "blocked directly")
        with pytest.raises(RuntimeError, match="guardrail rejected"):
            llm.conditional_execution(make_request())
        guardrails.deregister_llm_conditional_execution("py_llm_cond_direct")

    def test_duplicate_raises(self):
        guardrails.register_llm_sanitize_request("py_llm_dup", 1, lambda r: r)
        with pytest.raises(RuntimeError):
            guardrails.register_llm_sanitize_request("py_llm_dup", 1, lambda r: r)
        guardrails.deregister_llm_sanitize_request("py_llm_dup")

    def test_sanitize_request_callable_error_falls_back_to_original_input(self):
        events = []
        subscribers.register("py_llm_sanitize_req_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_request(
            "py_llm_sanitize_req_fail",
            1,
            lambda request: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        try:
            request = make_request()
            handle = llm.call("llm_sanitize_req_fail", request)
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_sanitize_req_fail")
            subscribers.deregister("py_llm_sanitize_req_sub")

        start = next(
            event for event in events if event.name == "llm_sanitize_req_fail" and event.event_type == EventType.Start
        )
        request = make_request()
        assert start.input == {"headers": request.headers, "content": request.content}

    def test_sanitize_request_invalid_return_falls_back_to_original_input(self):
        events = []
        subscribers.register("py_llm_sanitize_req_bad_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_request("py_llm_sanitize_req_bad", 1, lambda request: object())
        try:
            request = make_request()
            handle = llm.call("llm_sanitize_req_bad", request)
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_sanitize_req_bad")
            subscribers.deregister("py_llm_sanitize_req_bad_sub")

        start = next(
            event for event in events if event.name == "llm_sanitize_req_bad" and event.event_type == EventType.Start
        )
        request = make_request()
        assert start.input == {"headers": request.headers, "content": request.content}

    def test_sanitize_response_callable_error_falls_back_to_original_output(self):
        events = []
        subscribers.register("py_llm_sanitize_resp_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_response(
            "py_llm_sanitize_resp_fail",
            1,
            lambda response: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        try:
            handle = llm.call("llm_sanitize_resp_fail", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_response("py_llm_sanitize_resp_fail")
            subscribers.deregister("py_llm_sanitize_resp_sub")

        end = next(
            event for event in events if event.name == "llm_sanitize_resp_fail" and event.event_type == EventType.End
        )
        assert end.output == {"ok": True}

    def test_sanitize_response_invalid_return_falls_back_to_original_output(self):
        events = []
        subscribers.register("py_llm_sanitize_resp_bad_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_response("py_llm_sanitize_resp_bad", 1, lambda response: object())
        try:
            handle = llm.call("llm_sanitize_resp_bad", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_response("py_llm_sanitize_resp_bad")
            subscribers.deregister("py_llm_sanitize_resp_bad_sub")

        end = next(
            event for event in events if event.name == "llm_sanitize_resp_bad" and event.event_type == EventType.End
        )
        assert end.output == {"ok": True}

    def test_deregister_nonexistent(self):
        assert not guardrails.deregister_llm_sanitize_request("nope")
        assert not guardrails.deregister_llm_sanitize_response("nope")
        assert not guardrails.deregister_llm_conditional_execution("nope")

    def test_conditional_execution_invalid_return_type_raises(self):
        guardrails.register_llm_conditional_execution("py_llm_cond_bad_type", 1, lambda request: 123)
        try:
            with pytest.raises(RuntimeError, match="expected str or None"):
                llm.conditional_execution(make_request())
        finally:
            guardrails.deregister_llm_conditional_execution("py_llm_cond_bad_type")

    def test_conditional_execution_callable_error_raises(self):
        guardrails.register_llm_conditional_execution(
            "py_llm_cond_error",
            1,
            lambda request: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        try:
            with pytest.raises(RuntimeError, match="callable failed"):
                llm.conditional_execution(make_request())
        finally:
            guardrails.deregister_llm_conditional_execution("py_llm_cond_error")


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
        intercepts.register_llm_request("py_llm_req", 1, False, lambda name, request: request)
        assert intercepts.deregister_llm_request("py_llm_req")

    def test_request_intercepts_direct(self):
        def intercept_fn(name, request):
            content = request.content
            content["direct"] = True
            return LLMRequest(request.headers, content)

        intercepts.register_llm_request("py_llm_req_direct", 1, False, intercept_fn)
        transformed = llm.request_intercepts("direct_llm", make_request())
        intercepts.deregister_llm_request("py_llm_req_direct")

        assert transformed.content["direct"] is True

    def test_request_intercept_falls_back_on_exception(self):
        intercepts.register_llm_request(
            "py_llm_req_raise",
            1,
            False,
            lambda name, request: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        try:
            transformed = llm.request_intercepts("raise_llm", make_request())
            assert transformed.content["model"] == "test-model"
        finally:
            intercepts.deregister_llm_request("py_llm_req_raise")

    def test_request_intercept_falls_back_on_invalid_return(self):
        intercepts.register_llm_request("py_llm_req_bad_return", 1, False, lambda name, request: object())
        try:
            transformed = llm.request_intercepts("bad_return_llm", make_request())
            assert transformed.content["model"] == "test-model"
        finally:
            intercepts.deregister_llm_request("py_llm_req_bad_return")

    def test_execution_intercept(self):
        # Execution intercepts now take LLMRequest
        intercepts.register_llm_execution(
            "py_llm_exec",
            1,
            lambda name, request, next: {"intercepted": True},
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
        assert not intercepts.deregister_tool_execution("nope")


class TestLLMInterceptsAsync:
    async def test_request_intercept_modifies(self):
        def intercept_fn(name, request):
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
            lambda name, request, next: {"from_intercept": True},
        )

        def original_func(request):
            return {"from_original": True}

        request = make_request()
        result = await llm.execute("exec_llm", request, original_func)
        assert result["from_intercept"] is True
        assert "from_original" not in result

        intercepts.deregister_llm_execution("py_llm_exec_rep")

    async def test_execution_intercept_can_await_next(self):
        async def middleware(name, request, next):
            updated = LLMRequest(request.headers, {**request.content, "model": "via-next"})
            result = await next(updated)
            result["from_intercept"] = True
            return result

        intercepts.register_llm_execution("py_llm_exec_next", 1, middleware)

        def original_func(request):
            return {"model": request.content["model"]}

        try:
            result = await llm.execute("exec_llm_next", make_request(), original_func)
            assert result == {"model": "via-next", "from_intercept": True}
        finally:
            intercepts.deregister_llm_execution("py_llm_exec_next")

    async def test_stream_execution_intercept_can_await_next(self):
        def middleware(request, next):
            async def gen():
                updated = LLMRequest(request.headers, {**request.content, "prefix": "wrapped"})
                stream = await next(updated)
                async for chunk in stream:
                    yield {"token": f"{updated.content['prefix']}:{chunk['token']}"}

            return gen()

        def stream_func(request):
            async def gen():
                yield {"token": request.content["model"]}
                yield {"token": "done"}

            return gen()

        intercepts.register_llm_stream_execution("py_llm_stream_next", 1, middleware)
        try:
            stream = await llm.stream_execute(
                "stream_next_llm", make_request(), stream_func, lambda chunk: None, lambda: {}
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert chunks == [{"token": "wrapped:test-model"}, {"token": "wrapped:done"}]
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_next")

    async def test_stream_execution_intercept_async_function_is_supported(self):
        def middleware(request, next):
            updated = LLMRequest(request.headers, {**request.content, "prefix": "async"})

            async def gen():
                upstream = await next(updated)
                async for chunk in upstream:
                    yield {"token": f"{updated.content['prefix']}:{chunk['token']}"}

            return gen()

        def stream_func(request):
            async def gen():
                yield {"token": request.content["model"]}
                yield {"token": "done"}

            return gen()

        intercepts.register_llm_stream_execution("py_llm_stream_async", 1, middleware)
        try:
            stream = await llm.stream_execute(
                "stream_async_llm", make_request(), stream_func, lambda chunk: None, lambda: {}
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert chunks == [{"token": "async:test-model"}, {"token": "async:done"}]
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_async")


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

    async def test_stream_execute_propagates_generator_error(self):
        def stream_func(request):
            async def gen():
                yield {"token": "hello"}
                raise RuntimeError("stream boom")

            return gen()

        stream = await llm.stream_execute(
            "stream_error_llm", make_request(), stream_func, lambda chunk: None, lambda: {}
        )
        assert await anext(stream) == {"token": "hello"}
        with pytest.raises(RuntimeError, match="stream boom"):
            await anext(stream)

    async def test_stream_execute_rejects_invalid_iterator(self):
        stream = await llm.stream_execute(
            "stream_invalid_iter_llm", make_request(), lambda request: object(), lambda chunk: None, lambda: {}
        )
        with pytest.raises(RuntimeError, match="__anext__"):
            await anext(stream)

    async def test_stream_execute_handles_iterator_that_stops_in___anext__(self):
        stream = await llm.stream_execute(
            "stream_direct_stop_llm",
            make_request(),
            lambda request: _ImmediateStopAsyncIter(),
            lambda chunk: None,
            lambda: {},
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks == []

    async def test_stream_execute_propagates_direct___anext__error(self):
        stream = await llm.stream_execute(
            "stream_direct_error_llm",
            make_request(),
            lambda request: _BrokenAsyncIter(),
            lambda chunk: None,
            lambda: {},
        )
        with pytest.raises(RuntimeError, match="direct __anext__ boom"):
            await anext(stream)

    async def test_stream_execution_intercept_rejects_invalid_iterator(self):
        intercepts.register_llm_stream_execution("py_llm_stream_bad_iter", 1, lambda request, next: object())
        try:
            stream = await llm.stream_execute(
                "stream_intercept_invalid_iter_llm",
                make_request(),
                lambda request: _single_chunk_stream(),
                lambda chunk: None,
                lambda: {},
            )
            with pytest.raises(RuntimeError, match="__anext__"):
                await anext(stream)
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_bad_iter")

    async def test_stream_execution_intercept_handles_iterator_that_stops_in___anext__(self):
        intercepts.register_llm_stream_execution(
            "py_llm_stream_direct_stop",
            1,
            lambda request, next: _ImmediateStopAsyncIter(),
        )
        try:
            stream = await llm.stream_execute(
                "stream_intercept_direct_stop_llm",
                make_request(),
                lambda request: _single_chunk_stream(),
                lambda chunk: None,
                lambda: {},
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks == []
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_direct_stop")

    async def test_stream_execution_intercept_propagates_direct___anext__error(self):
        intercepts.register_llm_stream_execution(
            "py_llm_stream_direct_error",
            1,
            lambda request, next: _BrokenAsyncIter(),
        )
        try:
            stream = await llm.stream_execute(
                "stream_intercept_direct_error_llm",
                make_request(),
                lambda request: _single_chunk_stream(),
                lambda chunk: None,
                lambda: {},
            )
            with pytest.raises(RuntimeError, match="direct __anext__ boom"):
                await anext(stream)
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_direct_error")

    async def test_stream_execute_collector_failure_raises(self):
        def stream_func(request):
            async def gen():
                yield {"token": "hello"}

            return gen()

        stream = await llm.stream_execute(
            "stream_collector_fail_llm",
            make_request(),
            stream_func,
            lambda chunk: (_ for _ in ()).throw(RuntimeError("collector boom")),
            lambda: {},
        )
        with pytest.raises(RuntimeError, match="collector boom"):
            await anext(stream)

    async def test_stream_execute_finalizer_failure_records_null_output(self):
        events = []
        subscribers.register("py_llm_finalizer_fail_sub", lambda event: events.append(event))

        def stream_func(request):
            async def gen():
                yield {"token": "hello"}

            return gen()

        try:
            stream = await llm.stream_execute(
                "stream_finalizer_fail_llm",
                make_request(),
                stream_func,
                lambda chunk: None,
                lambda: object(),
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks == [{"token": "hello"}]
        finally:
            subscribers.deregister("py_llm_finalizer_fail_sub")

        end = next(
            event for event in events if event.name == "stream_finalizer_fail_llm" and event.event_type == EventType.End
        )
        assert end.output is None

    async def test_stream_execute_finalizer_callable_error_records_null_output(self):
        events = []
        subscribers.register("py_llm_finalizer_callable_fail_sub", lambda event: events.append(event))

        def stream_func(request):
            async def gen():
                yield {"token": "hello"}

            return gen()

        try:
            stream = await llm.stream_execute(
                "stream_finalizer_callable_fail_llm",
                make_request(),
                stream_func,
                lambda chunk: None,
                lambda: (_ for _ in ()).throw(RuntimeError("finalizer boom")),
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks == [{"token": "hello"}]
        finally:
            subscribers.deregister("py_llm_finalizer_callable_fail_sub")

        end = next(
            event
            for event in events
            if event.name == "stream_finalizer_callable_fail_llm" and event.event_type == EventType.End
        )
        assert end.output is None

    async def test_subscriber_exception_does_not_break_streaming(self):
        seen = []
        subscribers.register("py_llm_bad_sub", lambda event: (_ for _ in ()).throw(RuntimeError("subscriber boom")))
        subscribers.register("py_llm_good_sub", lambda event: seen.append(event.event_type))
        try:
            handle = llm.call("llm_subscriber_error", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            subscribers.deregister("py_llm_bad_sub")
            subscribers.deregister("py_llm_good_sub")

        assert seen == [EventType.Start, EventType.End]


async def _single_chunk_stream():
    yield {"token": "downstream"}


class _ImmediateStopAsyncIter:
    def __aiter__(self):
        return self

    def __anext__(self):
        raise StopAsyncIteration


class _BrokenAsyncIter:
    def __aiter__(self):
        return self

    def __anext__(self):
        raise RuntimeError("direct __anext__ boom")
