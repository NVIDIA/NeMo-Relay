# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Relay LLM lifecycle, guardrails, intercepts, and streaming."""

import asyncio
import contextvars
import threading
from collections.abc import AsyncIterator, Awaitable
from typing import NoReturn, cast

import pytest
from async_helpers import resolve_async_result

from nemo_relay import (
    Event,
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    LLMRequestInterceptOutcome,
    LogSeverity,
    PendingMarkSpec,
    PropagationContext,
    ScopeEvent,
    ScopeType,
    capture_propagation_context,
    capture_traceparent,
    create_scope_stack_from_propagation,
    guardrails,
    intercepts,
    llm,
    scope,
    subscribers,
    use_scope_stack,
)
from nemo_relay.codecs import OpenAIChatCodec


def make_request():
    return LLMRequest({}, {"messages": [], "model": "test-model"})


def raise_runtime_error(message: str) -> NoReturn:
    raise RuntimeError(message)


def _llm_event(events, name: str, scope_category: str) -> ScopeEvent:
    return next(
        event
        for event in events
        if event.name == name
        and isinstance(event, ScopeEvent)
        and event.category == "llm"
        and event.scope_category == scope_category
    )


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

    async def test_event_and_response_sanitizers_use_independent_context_snapshots(self):
        events = []
        start_entered = threading.Event()
        release_start = threading.Event()
        response_called = False

        def sanitize_start(event, fields):
            if event.name == "context_snapshot_llm":
                start_entered.set()
                assert release_start.wait(timeout=2)
            return fields

        def sanitize_response(response, context):
            nonlocal response_called
            del response, context
            response_called = True
            return {"sanitized": True}

        async def provider(request):
            del request
            assert await asyncio.to_thread(start_entered.wait, 2)
            return {"raw": True}

        subscribers.register("py_llm_context_snapshot_sub", events.append)
        guardrails.register_scope_sanitize_start("py_llm_context_snapshot_start", 0, sanitize_start)
        guardrails.register_llm_sanitize_response(
            "py_llm_context_snapshot_response",
            0,
            sanitize_response,
        )
        try:
            assert await llm.execute("context_snapshot_llm", make_request(), provider) == {"raw": True}
        finally:
            release_start.set()
            guardrails.deregister_scope_sanitize_start("py_llm_context_snapshot_start")
            guardrails.deregister_llm_sanitize_response("py_llm_context_snapshot_response")
            try:
                await subscribers.flush_async()
            finally:
                subscribers.deregister("py_llm_context_snapshot_sub")

        assert response_called
        assert _llm_event(events, "context_snapshot_llm", "end").data == {"sanitized": True}


class TestLLMGuardrails:
    @pytest.mark.parametrize(
        ("register", "callback"),
        [
            (guardrails.register_llm_sanitize_request, lambda request: request),
            (guardrails.register_llm_sanitize_response, lambda response: response),
            (guardrails.register_llm_sanitize_request, object()),
            (guardrails.register_llm_sanitize_response, object()),
        ],
    )
    def test_sanitizer_registration_rejects_legacy_or_uninspectable_callbacks(self, register, callback):
        with pytest.raises(TypeError, match="payload, context"):
            register("py_llm_invalid_signature", 1, callback)

    def test_sanitize_request_guardrail(self):
        def sanitizer(request, context):
            del context
            # request is an LLMRequest object; must return a new LLMRequest
            headers = request.headers
            headers["X-Sanitized"] = "true"
            return LLMRequest(headers, request.content)

        guardrails.register_llm_sanitize_request("py_llm_san_req", 1, sanitizer)
        guardrails.deregister_llm_sanitize_request("py_llm_san_req")

    def test_sanitize_response_guardrail(self):
        def sanitizer(response, context):
            del context
            # response is a plain dict
            response["cleaned"] = True
            return response

        guardrails.register_llm_sanitize_response("py_llm_san_resp", 1, sanitizer)
        guardrails.deregister_llm_sanitize_response("py_llm_san_resp")

    def test_sanitizers_receive_a_structured_codec_context(self):
        request_contexts = []
        response_contexts = []

        def sanitize_request(request, context):
            request_contexts.append(context)
            return request

        def sanitize_response(response, context):
            response_contexts.append(context)
            return response

        guardrails.register_llm_sanitize_request("py_llm_structured_context_request", 1, sanitize_request)
        guardrails.register_llm_sanitize_response("py_llm_structured_context_response", 1, sanitize_response)
        try:
            handle = llm.call("py_llm_structured_context", make_request())
            llm.call_end(handle, {"response": "ok"})
            subscribers.flush()
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_structured_context_request")
            guardrails.deregister_llm_sanitize_response("py_llm_structured_context_response")

        assert len(request_contexts) == 1
        assert len(response_contexts) == 1
        for context in [*request_contexts, *response_contexts]:
            assert context.codec.kind == "none"
            assert context.codec.id is None

    async def test_manual_async_sanitizers_can_flush_subscribers(self):
        request_flushed = False
        response_flushed = False

        async def sanitize_request(request, context) -> LLMRequest:
            nonlocal request_flushed
            del context
            await asyncio.sleep(0)
            subscribers.flush()
            request_flushed = True
            return request

        async def sanitize_response(response, context) -> dict:
            nonlocal response_flushed
            del context
            await asyncio.sleep(0)
            subscribers.flush()
            response_flushed = True
            return response

        guardrails.register_llm_sanitize_request("py_manual_flush_request", 1, sanitize_request)
        guardrails.register_llm_sanitize_response("py_manual_flush_response", 1, sanitize_response)
        subscribers.register("py_manual_flush_subscriber", lambda _event: None)
        try:
            handle = llm.call("py_manual_flush", make_request())
            llm.call_end(handle, {"response": "ok"})
            await asyncio.wait_for(subscribers.flush_async(), timeout=2)
        finally:
            guardrails.deregister_llm_sanitize_request("py_manual_flush_request")
            guardrails.deregister_llm_sanitize_response("py_manual_flush_response")
            subscribers.deregister("py_manual_flush_subscriber")

        assert request_flushed
        assert response_flushed

    async def test_sanitizers_resolve_active_builtin_codecs(self):
        request_codec_used = False
        response_codec_used = False

        def sanitize_request(request, context):
            nonlocal request_codec_used
            assert context.codec.kind == "builtin"
            assert context.codec.id == "openai_chat"
            codec = context.resolve_codec()
            assert codec is not None
            annotated = codec.decode(request)
            request_codec_used = True
            return codec.encode(annotated, request)

        def sanitize_response(response, context):
            nonlocal response_codec_used
            assert context.codec.kind == "builtin"
            assert context.codec.id == "openai_chat"
            codec = context.resolve_codec()
            assert codec is not None
            assert codec.decode_response(response).model == "test-model"
            response_codec_used = True
            return response

        codec = OpenAIChatCodec()
        response = {
            "id": "chatcmpl-python",
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        }
        guardrails.register_llm_sanitize_request("py_llm_builtin_context_request", 1, sanitize_request)
        guardrails.register_llm_sanitize_response("py_llm_builtin_context_response", 1, sanitize_response)
        try:
            result = await llm.execute(
                "py_llm_builtin_context",
                make_request(),
                lambda request: response,
                codec=codec,
                response_codec=codec,
            )
            await subscribers.flush_async()
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_builtin_context_request")
            guardrails.deregister_llm_sanitize_response("py_llm_builtin_context_response")

        assert result == response
        assert request_codec_used is True
        assert response_codec_used is True

    def test_none_omits_payload_and_short_circuits_later_sanitizers(self):
        events = []
        later_called = False

        def omit(request, context):
            return None

        def later(request, context):
            nonlocal later_called
            later_called = True
            return request

        subscribers.register("py_llm_none_sanitize_sub", events.append)
        guardrails.register_llm_sanitize_request("py_llm_none_sanitize_first", 1, omit)
        guardrails.register_llm_sanitize_request("py_llm_none_sanitize_later", 2, later)
        try:
            handle = llm.call("py_llm_none_sanitize", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_none_sanitize_first")
            guardrails.deregister_llm_sanitize_request("py_llm_none_sanitize_later")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_none_sanitize_sub")

        start = _llm_event(events, "py_llm_none_sanitize", "start")
        assert start.data is None
        assert start.annotated_request is None
        assert later_called is False

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
        guardrails.register_llm_sanitize_request("py_llm_dup", 1, lambda r, context: r)
        with pytest.raises(RuntimeError):
            guardrails.register_llm_sanitize_request("py_llm_dup", 1, lambda r, context: r)
        guardrails.deregister_llm_sanitize_request("py_llm_dup")

    def test_sanitize_request_callable_error_omits_observability_input(self):
        events = []
        subscribers.register("py_llm_sanitize_req_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_request(
            "py_llm_sanitize_req_fail",
            1,
            lambda request, context: raise_runtime_error("boom"),
        )
        try:
            request = LLMRequest(
                {"authorization": "secret", "x-request-id": "safe"},
                make_request().content,
            )
            handle = llm.call("llm_sanitize_req_fail", request)
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_sanitize_req_fail")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_req_sub")

        start = _llm_event(events, "llm_sanitize_req_fail", "start")
        assert start.data is None
        assert start.annotated_request is None

    def test_sanitize_request_invalid_return_omits_observability_input(self):
        events = []
        subscribers.register("py_llm_sanitize_req_bad_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_request(
            "py_llm_sanitize_req_bad",
            1,
            cast(guardrails.LlmSanitizeRequestGuardrail, lambda request, context: object()),
        )
        try:
            request = LLMRequest(
                {"authorization": "secret", "x-request-id": "safe"},
                make_request().content,
            )
            handle = llm.call("llm_sanitize_req_bad", request)
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_sanitize_req_bad")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_req_bad_sub")

        start = _llm_event(events, "llm_sanitize_req_bad", "start")
        assert start.data is None
        assert start.annotated_request is None

    def test_sanitize_response_callable_error_omits_observability_output(self):
        events = []
        subscribers.register("py_llm_sanitize_resp_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_response(
            "py_llm_sanitize_resp_fail",
            1,
            lambda response, context: raise_runtime_error("boom"),
        )
        try:
            handle = llm.call("llm_sanitize_resp_fail", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_response("py_llm_sanitize_resp_fail")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_resp_sub")

        end = _llm_event(events, "llm_sanitize_resp_fail", "end")
        assert end.data is None
        assert end.annotated_response is None

    def test_sanitize_response_invalid_return_omits_observability_output(self):
        events = []
        subscribers.register("py_llm_sanitize_resp_bad_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_response(
            "py_llm_sanitize_resp_bad",
            1,
            cast(guardrails.LlmSanitizeResponseGuardrail, lambda response, context: object()),
        )
        try:
            handle = llm.call("llm_sanitize_resp_bad", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_response("py_llm_sanitize_resp_bad")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_resp_bad_sub")

        end = _llm_event(events, "llm_sanitize_resp_bad", "end")
        assert end.data is None
        assert end.annotated_response is None

    def test_sanitize_response_guardrail_accepts_scalar_json_payloads(self):
        events = []
        subscribers.register("py_llm_sanitize_scalar_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_response(
            "py_llm_sanitize_scalar",
            1,
            lambda response, context: f"sanitized:{response}",
        )
        try:
            handle = llm.call("llm_sanitize_scalar", make_request())
            llm.call_end(handle, "raw-response")
        finally:
            guardrails.deregister_llm_sanitize_response("py_llm_sanitize_scalar")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_scalar_sub")

        end = _llm_event(events, "llm_sanitize_scalar", "end")
        assert end.data == "sanitized:raw-response"

    def test_deregister_nonexistent(self):
        assert not guardrails.deregister_llm_sanitize_request("nope")
        assert not guardrails.deregister_llm_sanitize_response("nope")
        assert not guardrails.deregister_llm_conditional_execution("nope")

    def test_conditional_execution_invalid_return_type_raises(self):
        guardrails.register_llm_conditional_execution(
            "py_llm_cond_bad_type",
            1,
            cast(guardrails.LlmConditionalExecutionGuardrail, lambda request: 123),
        )
        try:
            with pytest.raises(RuntimeError, match="unexpected type"):
                llm.conditional_execution(make_request())
        finally:
            guardrails.deregister_llm_conditional_execution("py_llm_cond_bad_type")

    def test_conditional_execution_callable_error_raises(self):
        guardrails.register_llm_conditional_execution(
            "py_llm_cond_error",
            1,
            lambda request: raise_runtime_error("boom"),
        )
        try:
            with pytest.raises(RuntimeError, match="RuntimeError: boom"):
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
        intercepts.register_llm_request(
            "py_llm_req",
            1,
            False,
            lambda name, request, annotated: LLMRequestInterceptOutcome(request, annotated),
        )
        assert intercepts.deregister_llm_request("py_llm_req")

    def test_request_intercepts_direct(self):
        pending_mark = PendingMarkSpec(
            "request.direct",
            data={"source": "python"},
            data_schema={"name": "example.pending", "version": "1"},
            severity=LogSeverity.Info,
        )

        def intercept_fn(name, request, annotated):
            content = request.content
            content["direct"] = True
            return LLMRequestInterceptOutcome(
                LLMRequest(request.headers, content),
                annotated,
                [pending_mark],
            )

        intercepts.register_llm_request("py_llm_req_direct", 1, False, intercept_fn)
        transformed = llm.request_intercepts("direct_llm", make_request())
        intercepts.deregister_llm_request("py_llm_req_direct")

        assert not isinstance(transformed, Awaitable)
        assert transformed.request.content["direct"] is True
        assert len(transformed.pending_marks) == 1
        assert transformed.pending_marks[0].name == pending_mark.name
        assert transformed.pending_marks[0].data == pending_mark.data
        assert transformed.pending_marks[0].data_schema == pending_mark.data_schema
        assert transformed.pending_marks[0].severity == LogSeverity.Info

    def test_request_intercept_raises_on_exception(self):
        intercepts.register_llm_request(
            "py_llm_req_raise",
            1,
            False,
            lambda name, request, annotated: raise_runtime_error("boom"),
        )
        try:
            with pytest.raises(RuntimeError, match="callable failed"):
                llm.request_intercepts("raise_llm", make_request())
        finally:
            intercepts.deregister_llm_request("py_llm_req_raise")

    def test_request_intercept_raises_on_invalid_return(self):
        intercepts.register_llm_request("py_llm_req_bad_return", 1, False, lambda name, request, annotated: object())  # type: ignore[arg-type] # ty: ignore[invalid-argument-type]
        try:
            with pytest.raises(RuntimeError, match="must return LLMRequestInterceptOutcome"):
                llm.request_intercepts("bad_return_llm", make_request())
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
    async def test_execution_callback_capture_traceparent_matches_llm_scope(self):
        events = []
        observed = []
        subscribers.register("py_llm_capture_traceparent", events.append)

        async def execution_intercept(_name, request, next_handler):
            observed.append((capture_propagation_context().parent_uuid, capture_traceparent()))
            return await next_handler(request)

        async def provider(_request):
            observed.append((capture_propagation_context().parent_uuid, capture_traceparent()))
            return {"ok": True}

        try:
            intercepts.register_llm_execution("py_llm_capture_traceparent", 10, execution_intercept)
            assert await llm.execute("py_llm_capture_traceparent", make_request(), provider) == {"ok": True}
            await subscribers.flush_async()
            start = _llm_event(events, "py_llm_capture_traceparent", "start")
            expected = f"00-{start.uuid.replace('-', '')}-{start.uuid.replace('-', '')[-16:]}-01"
            assert observed == [(start.uuid, expected), (start.uuid, expected)]
        finally:
            intercepts.deregister_llm_execution("py_llm_capture_traceparent")
            subscribers.deregister("py_llm_capture_traceparent")

    async def test_execution_callback_capture_traceparent_preserves_imported_root(self):
        root_uuid = "018f13f0-7c1a-7a80-8000-000000000701"
        parent_uuid = "018f13f0-7c1a-7a80-8000-000000000702"
        stack = create_scope_stack_from_propagation(PropagationContext(parent_uuid, root_uuid))
        events = []
        observed = []
        subscribers.register("py_llm_capture_propagated_trace_root", events.append)

        async def execution_intercept(_name, request, next_handler):
            observed.append(capture_traceparent())
            return await next_handler(request)

        async def provider(_request):
            observed.append(capture_traceparent())
            return {"ok": True}

        try:
            intercepts.register_llm_execution("py_llm_capture_propagated_trace_root", 10, execution_intercept)
            with use_scope_stack(stack):
                assert await llm.execute("py_llm_propagated_trace_root", make_request(), provider) == {"ok": True}
            await subscribers.flush_async()
            start = _llm_event(events, "py_llm_propagated_trace_root", "start")
            expected = f"00-{root_uuid.replace('-', '')}-{start.uuid.replace('-', '')[-16:]}-01"
            assert observed == [expected, expected]
        finally:
            intercepts.deregister_llm_execution("py_llm_capture_propagated_trace_root")
            subscribers.deregister("py_llm_capture_propagated_trace_root")

    async def test_cancelling_execute_cancels_pending_execution_intercept(self):
        started = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()
        provider_calls: list[LLMRequest] = []
        events: list[Event] = []

        async def middleware(_name, request, next):
            started.set()
            try:
                await release.wait()
                return await next(request)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def provider(request):
            provider_calls.append(request)
            return {"ok": True}

        intercepts.register_llm_execution("py_llm_cancel_intercept", 1, middleware)
        subscribers.register("py_llm_cancel_events", events.append)
        try:
            execution = asyncio.ensure_future(llm.execute("cancel_llm", make_request(), provider))
            await asyncio.wait_for(started.wait(), timeout=1)
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await execution
            await asyncio.wait_for(cancelled.wait(), timeout=1)
            await subscribers.flush_async()
        finally:
            release.set()
            intercepts.deregister_llm_execution("py_llm_cancel_intercept")
            subscribers.deregister("py_llm_cancel_events")

        assert provider_calls == []
        lifecycle = [
            event.scope_category for event in events if isinstance(event, ScopeEvent) and event.name == "cancel_llm"
        ]
        assert lifecycle == ["start", "end"]

    async def test_cancelling_stream_execute_cancels_pending_stream_intercept(self):
        started = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()
        provider_calls: list[LLMRequest] = []
        events: list[Event] = []

        async def middleware(request, next):
            started.set()
            try:
                await release.wait()
                return await next(request)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def provider(request):
            provider_calls.append(request)

            async def generate():
                yield {"token": "unexpected"}

            return generate()

        intercepts.register_llm_stream_execution("py_llm_stream_cancel_intercept", 1, middleware)
        subscribers.register("py_llm_stream_cancel_events", events.append)
        try:
            execution = asyncio.ensure_future(
                llm.stream_execute(
                    "cancel_stream_llm",
                    make_request(),
                    provider,
                    lambda _chunk: None,
                    lambda: {},
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
            intercepts.deregister_llm_stream_execution("py_llm_stream_cancel_intercept")
            subscribers.deregister("py_llm_stream_cancel_events")

        assert provider_calls == []
        lifecycle = [
            event.scope_category
            for event in events
            if isinstance(event, ScopeEvent) and event.name == "cancel_stream_llm"
        ]
        assert lifecycle == ["start", "end"]

    async def test_sync_middleware_preserves_async_caller_context(self):
        request_id = contextvars.ContextVar("llm_middleware_request_id", default="registration")
        observed: list[tuple[str, str]] = []

        def conditional(_request):
            observed.append(("conditional", request_id.get()))
            return None

        def request_intercept(_name, request, annotated):
            observed.append(("request", request_id.get()))
            return LLMRequestInterceptOutcome(request, annotated)

        def execution_intercept(_name, _request, _next):
            observed.append(("execution", request_id.get()))
            return {"ok": True}

        guardrails.register_llm_conditional_execution("py_llm_context_conditional", 1, conditional)
        intercepts.register_llm_request("py_llm_context_request", 1, False, request_intercept)
        intercepts.register_llm_execution("py_llm_context_execution", 1, execution_intercept)
        token = request_id.set("emitter")
        try:
            assert await llm.execute("context_llm", make_request(), lambda _request: {}) == {"ok": True}
            await resolve_async_result(llm.conditional_execution(make_request()))
            standalone = await resolve_async_result(llm.request_intercepts("context_llm_standalone", make_request()))
            assert standalone.request.content == make_request().content
        finally:
            request_id.reset(token)
            intercepts.deregister_llm_execution("py_llm_context_execution")
            intercepts.deregister_llm_request("py_llm_context_request")
            guardrails.deregister_llm_conditional_execution("py_llm_context_conditional")

        assert observed == [
            ("conditional", "emitter"),
            ("request", "emitter"),
            ("execution", "emitter"),
            ("conditional", "emitter"),
            ("request", "emitter"),
        ]

    async def test_sync_stream_intercept_preserves_async_caller_context(self):
        request_id = contextvars.ContextVar("llm_stream_middleware_request_id", default="registration")
        observed: list[tuple[str, str]] = []

        def middleware(request, next):
            observed.append(("callback", request_id.get()))

            async def generate():
                observed.append(("generator-before", request_id.get()))
                await asyncio.sleep(0)
                observed.append(("generator-after", request_id.get()))
                upstream = await next(request)
                async for chunk in upstream:
                    yield chunk

            return generate()

        def provider(_request):
            async def generate():
                yield {"token": "ok"}

            return generate()

        intercepts.register_llm_stream_execution("py_llm_stream_context", 1, middleware)
        token = request_id.set("emitter")
        try:
            stream = await llm.stream_execute(
                "context_stream_llm",
                make_request(),
                provider,
                lambda _chunk: None,
                lambda: {},
            )
            assert [chunk async for chunk in stream] == [{"token": "ok"}]
        finally:
            request_id.reset(token)
            intercepts.deregister_llm_stream_execution("py_llm_stream_context")

        assert observed == [
            ("callback", "emitter"),
            ("generator-before", "emitter"),
            ("generator-after", "emitter"),
        ]

    async def test_custom_stream_iterator_methods_preserve_async_caller_context(self):
        request_id = contextvars.ContextVar("llm_custom_iterator_request_id", default="registration")
        observed: list[tuple[str, str]] = []

        class CustomIterator:
            def __aiter__(self):
                return self

            def __anext__(self):
                observed.append(("anext-sync", request_id.get()))

                async def step():
                    observed.append(("anext-before", request_id.get()))
                    await asyncio.sleep(0)
                    observed.append(("anext-after", request_id.get()))
                    return {"token": "ok"}

                return step()

            def aclose(self):
                observed.append(("aclose-sync", request_id.get()))

                async def close():
                    observed.append(("aclose-before", request_id.get()))
                    await asyncio.sleep(0)
                    observed.append(("aclose-after", request_id.get()))

                return close()

        def middleware(_request, _next):
            observed.append(("callback", request_id.get()))
            return CustomIterator()

        intercepts.register_llm_stream_execution("py_llm_custom_iterator_context", 1, middleware)
        token = request_id.set("emitter")
        try:
            stream = await llm.stream_execute(
                "custom_iterator_context_llm",
                make_request(),
                lambda _request: None,  # ty: ignore[invalid-argument-type]
                lambda _chunk: None,
                lambda: {},
            )
            assert await anext(stream) == {"token": "ok"}
            await stream.aclose()
        finally:
            request_id.reset(token)
            intercepts.deregister_llm_stream_execution("py_llm_custom_iterator_context")

        assert observed[:4] == [
            ("callback", "emitter"),
            ("anext-sync", "emitter"),
            ("anext-before", "emitter"),
            ("anext-after", "emitter"),
        ]
        assert [label for label, _value in observed[-3:]] == [
            "aclose-sync",
            "aclose-before",
            "aclose-after",
        ]
        assert all(value == "emitter" for _label, value in observed)

    async def test_default_lazy_stream_preserves_managed_parent_context(self):
        events = []
        subscribers.register("py_default_lazy_stream_context", events.append)
        owner = scope.push("py-default-lazy-stream-owner", ScopeType.Agent)

        def provider(_request):
            async def generate():
                yield {"parent_uuid": capture_propagation_context().parent_uuid}

            return generate()

        try:
            stream = await llm.stream_execute(
                "py_default_lazy_stream_context",
                make_request(),
                provider,
                lambda _chunk: None,
                lambda: {},
            )
            chunks = [chunk async for chunk in stream]
            await subscribers.flush_async()
        finally:
            scope.pop(owner)
            subscribers.deregister("py_default_lazy_stream_context")

        start = _llm_event(events, "py_default_lazy_stream_context", "start")
        assert chunks == [{"parent_uuid": start.uuid}]

    async def test_terminal_stream_error_close_waits_for_real_producer_cleanup(self):
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()

        class FailingIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("provider stream failed")

            async def aclose(self):
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_finished.set()

        stream = await llm.stream_execute(
            "py_terminal_error_cleanup",
            make_request(),
            lambda _request: FailingIterator(),
            lambda _chunk: None,
            lambda: {},
        )
        with pytest.raises(RuntimeError, match="provider stream failed"):
            await anext(stream)
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)

        closing = asyncio.ensure_future(stream.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        assert not cleanup_finished.is_set()
        release_cleanup.set()
        await asyncio.wait_for(closing, timeout=2)
        assert cleanup_finished.is_set()

    async def test_async_request_intercept_runs_on_originating_loop(self):
        originating_loop = asyncio.get_running_loop()

        async def intercept_fn(_name, request, annotated):
            await asyncio.sleep(0)
            assert asyncio.get_running_loop() is originating_loop
            content = {**request.content, "intercepted": True}
            return LLMRequestInterceptOutcome(LLMRequest(request.headers, content), annotated)

        intercepts.register_llm_request("py_llm_async_request_loop", 1, False, intercept_fn)
        try:
            result = await llm.execute(
                "async_request_llm",
                make_request(),
                lambda request: {"intercepted": request.content["intercepted"]},
            )
        finally:
            intercepts.deregister_llm_request("py_llm_async_request_loop")

        assert result == {"intercepted": True}

    async def test_request_intercept_modifies(self):
        def intercept_fn(name, request, annotated):
            # Request intercepts now operate on LLMRequest
            content = request.content
            content["intercepted"] = True
            return LLMRequestInterceptOutcome(LLMRequest(request.headers, content), annotated)

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

    async def test_execution_intercept_rejects_next_after_settlement(self):
        captured_next = None
        provider_calls = 0

        async def middleware(_name, _request, next):
            nonlocal captured_next
            captured_next = next
            return {"source": "intercept"}

        def provider(_request):
            nonlocal provider_calls
            provider_calls += 1
            return {"source": "provider"}

        intercepts.register_llm_execution("py_llm_late_next", 1, middleware)
        try:
            result = await llm.execute("late_next_llm", make_request(), provider)
            assert result == {"source": "intercept"}
            assert captured_next is not None
            with pytest.raises(RuntimeError, match="execution continuation is no longer active"):
                await captured_next(make_request())
        finally:
            intercepts.deregister_llm_execution("py_llm_late_next")

        assert provider_calls == 0

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

    async def test_stream_execution_intercept_rejects_next_after_settlement(self):
        captured_next = None
        provider_calls = 0

        async def middleware(_request, next):
            nonlocal captured_next
            captured_next = next

            async def replacement():
                yield {"source": "intercept"}

            return replacement()

        def provider(_request):
            nonlocal provider_calls
            provider_calls += 1

            async def stream():
                yield {"source": "provider"}

            return stream()

        intercepts.register_llm_stream_execution("py_llm_stream_late_next", 1, middleware)
        try:
            stream = await llm.stream_execute(
                "late_next_stream_llm", make_request(), provider, lambda _chunk: None, lambda: {}
            )
            assert [chunk async for chunk in stream] == [{"source": "intercept"}]
            assert captured_next is not None
            with pytest.raises(RuntimeError, match="execution continuation is no longer active"):
                await captured_next(make_request())
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_late_next")

        assert provider_calls == 0

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

    async def test_async_response_sanitizer_runs_during_stream_finalization(self):
        events = []
        originating_loop = asyncio.get_running_loop()
        subscribers.register("py_llm_async_stream_sanitizer_sub", events.append)

        async def sanitize_response(response, context) -> dict:
            del context
            await asyncio.sleep(0)
            assert asyncio.get_running_loop() is originating_loop
            return {"sanitized": response["raw"]}

        async def stream_func(request) -> AsyncIterator[dict]:
            del request
            yield {"token": "hello"}

        guardrails.register_llm_sanitize_response("py_llm_async_stream_sanitizer", 1, sanitize_response)
        try:
            stream = await llm.stream_execute(
                "stream_async_response_sanitizer",
                make_request(),
                stream_func,
                lambda _chunk: None,
                lambda: {"raw": True},
            )
            assert [chunk async for chunk in stream] == [{"token": "hello"}]
            await subscribers.flush_async()
        finally:
            guardrails.deregister_llm_sanitize_response("py_llm_async_stream_sanitizer")
            subscribers.deregister("py_llm_async_stream_sanitizer_sub")

        end = _llm_event(events, "stream_async_response_sanitizer", "end")
        assert end.data == {"sanitized": True}

    async def test_stream_response_sanitizer_preserves_emitter_contextvars(self):
        request_id = contextvars.ContextVar("stream_request_id", default="registration")
        observed = []

        async def sanitize_response(response, context):
            del context
            observed.append(request_id.get())
            await asyncio.sleep(0)
            observed.append(request_id.get())
            return response

        async def stream_func(request):
            del request
            yield {"token": "hello"}

        guardrails.register_llm_sanitize_response(
            "py_llm_stream_contextvars",
            1,
            sanitize_response,
        )
        token = request_id.set("caller")
        try:
            stream = await llm.stream_execute(
                "stream_contextvars",
                make_request(),
                stream_func,
                lambda _chunk: None,
                lambda: {"done": True},
            )
            assert [chunk async for chunk in stream] == [{"token": "hello"}]
            await subscribers.flush_async()
        finally:
            request_id.reset(token)
            guardrails.deregister_llm_sanitize_response("py_llm_stream_contextvars")

        assert observed == ["caller", "caller"]

    async def test_stream_execute_aclose_stops_partially_consumed_stream(self):
        producer_closed = asyncio.Event()
        wait_for_more_chunks = asyncio.Event()

        async def stream_func(request):
            try:
                yield {"token": "first"}
                await wait_for_more_chunks.wait()
            finally:
                producer_closed.set()

        stream = await llm.stream_execute(
            "stream_aclose_llm",
            make_request(),
            stream_func,
            lambda chunk: None,
            lambda: {},
        )
        assert await anext(stream) == {"token": "first"}

        await asyncio.wait_for(stream.aclose(), timeout=1)
        assert producer_closed.is_set()
        await asyncio.wait_for(stream.aclose(), timeout=1)
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), timeout=1)

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
            "stream_invalid_iter_llm",
            make_request(),
            lambda request: object(),  # ty: ignore[invalid-argument-type]
            lambda chunk: None,
            lambda: {},
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
        intercepts.register_llm_stream_execution(
            "py_llm_stream_bad_iter",
            1,
            cast(intercepts.LlmStreamExecutionIntercept, lambda request, next: object()),
        )
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

    async def test_stream_execution_intercept_failure_emits_exception_type(self):
        events = []
        subscribers.register("py_llm_stream_intercept_failure_sub", events.append)

        def failing_middleware(request, next):
            raise ValueError("stream intercept boom")

        intercepts.register_llm_stream_execution(
            "py_llm_stream_failure",
            1,
            failing_middleware,
        )
        try:
            with pytest.raises(RuntimeError, match="stream intercept boom"):
                await llm.stream_execute(
                    "stream_intercept_failure_llm",
                    make_request(),
                    lambda request: _single_chunk_stream(),
                    lambda chunk: None,
                    lambda: {},
                )
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_failure")
            await subscribers.flush_async()
            subscribers.deregister("py_llm_stream_intercept_failure_sub")

        metadata = _llm_event(events, "stream_intercept_failure_llm", "end").metadata
        assert isinstance(metadata, dict)
        assert metadata["exception.type"] == "ValueError"

    async def test_stream_execute_collector_failure_raises(self):
        events = []
        subscribers.register("py_llm_stream_collector_failure_sub", events.append)

        def stream_func(request):
            async def gen():
                yield {"token": "hello"}

            return gen()

        try:
            stream = await llm.stream_execute(
                "stream_collector_fail_llm",
                make_request(),
                stream_func,
                lambda chunk: raise_runtime_error("collector boom"),
                lambda: {},
            )
            with pytest.raises(RuntimeError, match="collector boom"):
                await anext(stream)
            await subscribers.flush_async()
        finally:
            subscribers.deregister("py_llm_stream_collector_failure_sub")

        metadata = _llm_event(events, "stream_collector_fail_llm", "end").metadata
        assert isinstance(metadata, dict)
        assert metadata["exception.type"] == "RuntimeError"

    async def test_stream_execute_callback_failure_emits_exception_type(self):
        events = []
        subscribers.register("py_llm_stream_callback_failure_sub", events.append)

        def stream_func(request):
            raise ValueError("stream callback boom")

        async def async_stream_func(request):
            raise TypeError("async stream callback boom")

        try:
            with pytest.raises(RuntimeError, match="stream callback boom"):
                await llm.stream_execute(
                    "stream_callback_fail_llm",
                    make_request(),
                    stream_func,
                    lambda chunk: None,
                    lambda: {},
                )
            with pytest.raises(RuntimeError, match="async stream callback boom"):
                await llm.stream_execute(
                    "async_stream_callback_fail_llm",
                    make_request(),
                    async_stream_func,  # ty: ignore[invalid-argument-type]
                    lambda chunk: None,
                    lambda: {},
                )
            await subscribers.flush_async()
        finally:
            subscribers.deregister("py_llm_stream_callback_failure_sub")

        metadata = _llm_event(events, "stream_callback_fail_llm", "end").metadata
        assert isinstance(metadata, dict)
        assert metadata["exception.type"] == "ValueError"
        metadata = _llm_event(events, "async_stream_callback_fail_llm", "end").metadata
        assert isinstance(metadata, dict)
        assert metadata["exception.type"] == "TypeError"

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
                lambda: object(),  # ty: ignore[invalid-argument-type]
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks == [{"token": "hello"}]
        finally:
            try:
                await subscribers.flush_async()
            finally:
                subscribers.deregister("py_llm_finalizer_fail_sub")

        end = _llm_event(events, "stream_finalizer_fail_llm", "end")
        assert end.data is None

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
                lambda: raise_runtime_error("finalizer boom"),
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks == [{"token": "hello"}]
        finally:
            try:
                await subscribers.flush_async()
            finally:
                subscribers.deregister("py_llm_finalizer_callable_fail_sub")

        end = _llm_event(events, "stream_finalizer_callable_fail_llm", "end")
        assert end.data is None

    async def test_subscriber_exception_does_not_break_streaming(self):
        seen = []
        subscribers.register("py_llm_bad_sub", lambda event: raise_runtime_error("subscriber boom"))
        subscribers.register("py_llm_good_sub", lambda event: seen.append(event.kind))
        try:
            handle = llm.call("llm_subscriber_error", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            try:
                await subscribers.flush_async()
            finally:
                subscribers.deregister("py_llm_bad_sub")
                subscribers.deregister("py_llm_good_sub")

        assert seen == ["scope", "scope"]


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
