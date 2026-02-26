"""Exhaustive tests for NVAgentRT Python bindings."""

import pytest
from nvagentrt import (
    Event,
    EventType,
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    ScopeAttributes,
    ScopeHandle,
    ScopeType,
    ToolAttributes,
    ToolHandle,
    guardrails,
    intercepts,
    llm,
    scope,
    subscribers,
    tools,
)
from nvagentrt._native import SseEvent

# ============================================================================
# Types
# ============================================================================


class TestScopeType:
    def test_all_variants_exist(self):
        variants = [
            ScopeType.Agent,
            ScopeType.Function,
            ScopeType.Tool,
            ScopeType.Llm,
            ScopeType.Retriever,
            ScopeType.Embedder,
            ScopeType.Reranker,
            ScopeType.Guardrail,
            ScopeType.Evaluator,
            ScopeType.Custom,
            ScopeType.Unknown,
        ]
        assert len(variants) == 11

    def test_repr(self):
        assert "Agent" in repr(ScopeType.Agent)


class TestEventType:
    def test_all_variants(self):
        variants = [EventType.Start, EventType.End, EventType.Mark]
        assert len(variants) == 3


class TestScopeAttributes:
    def test_parallel_is_int(self):
        assert isinstance(ScopeAttributes.PARALLEL, int)
        assert ScopeAttributes.PARALLEL == 0b01

    def test_relocatable_is_int(self):
        assert isinstance(ScopeAttributes.RELOCATABLE, int)
        assert ScopeAttributes.RELOCATABLE == 0b10

    def test_construct_from_value(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL)
        assert attrs.is_parallel
        assert not attrs.is_relocatable

    def test_construct_combined(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL | ScopeAttributes.RELOCATABLE)
        assert attrs.is_parallel
        assert attrs.is_relocatable

    def test_or_operator(self):
        a = ScopeAttributes(ScopeAttributes.PARALLEL)
        b = ScopeAttributes(ScopeAttributes.RELOCATABLE)
        combined = a | b
        assert combined.is_parallel
        assert combined.is_relocatable

    def test_value_getter(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL)
        assert attrs.value == ScopeAttributes.PARALLEL


class TestToolAttributes:
    def test_local_is_int(self):
        assert isinstance(ToolAttributes.LOCAL, int)
        assert ToolAttributes.LOCAL == 0b01

    def test_construct(self):
        attrs = ToolAttributes(ToolAttributes.LOCAL)
        assert attrs.is_local

    def test_empty(self):
        attrs = ToolAttributes(0)
        assert not attrs.is_local


class TestLLMAttributes:
    def test_stateless_is_int(self):
        assert isinstance(LLMAttributes.STATELESS, int)

    def test_streaming_is_int(self):
        assert isinstance(LLMAttributes.STREAMING, int)

    def test_construct_combined(self):
        attrs = LLMAttributes(LLMAttributes.STATELESS | LLMAttributes.STREAMING)
        assert attrs.is_stateless
        assert attrs.is_streaming


class TestLLMRequest:
    def test_constructor(self):
        req = LLMRequest("POST", "https://api.example.com", {"Authorization": "Bearer token"}, {"messages": []})
        assert req.method == "POST"
        assert req.url == "https://api.example.com"
        assert req.headers == {"Authorization": "Bearer token"}
        assert req.body == {"messages": []}

    def test_empty_headers(self):
        req = LLMRequest("GET", "https://api.example.com", {}, {"q": "test"})
        assert req.headers == {}

    def test_repr(self):
        req = LLMRequest("POST", "https://api.example.com", {}, {})
        r = repr(req)
        assert "POST" in r
        assert "api.example.com" in r


class TestSseEvent:
    def test_constructor(self):
        event = SseEvent("hello world")
        assert event.data == "hello world"
        assert event.event is None
        assert event.id is None
        assert event.retry is None

    def test_constructor_all_fields(self):
        event = SseEvent("payload", event="chunk", id="42", retry=5000)
        assert event.data == "payload"
        assert event.event == "chunk"
        assert event.id == "42"
        assert event.retry == 5000

    def test_repr(self):
        event = SseEvent("test")
        assert "test" in repr(event)


# ============================================================================
# Scope operations
# ============================================================================


class TestScope:
    def test_get_handle_returns_root(self):
        handle = scope.get_handle()
        assert isinstance(handle, ScopeHandle)
        assert handle.name == "root"

    def test_push_and_pop(self):
        handle = scope.push("test_scope", ScopeType.Agent)
        assert handle.name == "test_scope"
        assert scope.get_handle().name == "test_scope"
        scope.pop(handle)
        assert scope.get_handle().name == "root"

    def test_push_with_attributes(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL)
        handle = scope.push("parallel", ScopeType.Function, attributes=attrs)
        assert handle.name == "parallel"
        assert handle.attributes.is_parallel
        scope.pop(handle)

    def test_push_with_parent(self):
        parent = scope.push("parent", ScopeType.Agent)
        child = scope.push("child", ScopeType.Function, handle=parent)
        assert child.parent_uuid == parent.uuid
        scope.pop(child)
        scope.pop(parent)

    def test_nested_scopes(self):
        s1 = scope.push("level1", ScopeType.Agent)
        s2 = scope.push("level2", ScopeType.Function)
        s3 = scope.push("level3", ScopeType.Tool)
        assert scope.get_handle().name == "level3"
        scope.pop(s3)
        assert scope.get_handle().name == "level2"
        scope.pop(s2)
        assert scope.get_handle().name == "level1"
        scope.pop(s1)
        assert scope.get_handle().name == "root"

    def test_scope_handle_properties(self):
        handle = scope.push("props_test", ScopeType.Retriever)
        assert handle.uuid is not None
        assert handle.name == "props_test"
        assert handle.scope_type == ScopeType.Retriever
        scope.pop(handle)

    def test_event_emission(self):
        scope.event("my_mark")  # Should not raise

    def test_event_with_data(self):
        scope.event("data_mark", data={"key": "value"}, metadata={"version": 1})

    def test_event_with_handle(self):
        handle = scope.push("evt_scope", ScopeType.Agent)
        scope.event("scoped_mark", handle=handle)
        scope.pop(handle)

    def test_pop_invalid_raises(self):
        handle = scope.push("once", ScopeType.Agent)
        scope.pop(handle)
        with pytest.raises(RuntimeError):
            scope.pop(handle)


# ============================================================================
# Subscribers
# ============================================================================


class TestSubscribers:
    def test_register_and_deregister(self):
        events = []
        subscribers.register("py_test_sub", lambda e: events.append(e))
        handle = scope.push("sub_test", ScopeType.Function)
        scope.pop(handle)
        assert subscribers.deregister("py_test_sub")
        assert len(events) >= 2

    def test_subscriber_receives_event_objects(self):
        events = []
        subscribers.register("py_evt_sub", lambda e: events.append(e))
        handle = scope.push("evt_obj_test", ScopeType.Agent)
        scope.pop(handle)
        subscribers.deregister("py_evt_sub")

        assert len(events) >= 2
        for e in events:
            assert isinstance(e, Event)
            assert e.uuid is not None
            assert e.event_type is not None

    def test_duplicate_subscriber_raises(self):
        subscribers.register("py_dup_sub", lambda e: None)
        with pytest.raises(RuntimeError):
            subscribers.register("py_dup_sub", lambda e: None)
        subscribers.deregister("py_dup_sub")

    def test_deregister_nonexistent(self):
        assert not subscribers.deregister("nonexistent_sub")


# ============================================================================
# Tool lifecycle
# ============================================================================


class TestTools:
    def test_call_and_call_end(self):
        handle = tools.call("my_tool", {"input": "data"})
        assert isinstance(handle, ToolHandle)
        assert handle.name == "my_tool"
        tools.call_end(handle, {"output": "result"})

    def test_call_with_attributes(self):
        attrs = ToolAttributes(ToolAttributes.LOCAL)
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


class TestToolsAsync:
    async def test_execute_basic(self):
        # tools.execute wraps a Python callable; use sync func
        def my_func(args):
            return {"result": args["x"] * 2}

        result = await tools.execute("double", {"x": 5}, my_func)
        assert result == {"result": 10}

    async def test_execute_returns_string(self):
        def func(args):
            return "hello"

        result = await tools.execute("str_tool", {}, func)
        assert result == "hello"

    async def test_execute_with_attributes(self):
        def func(args):
            return args

        attrs = ToolAttributes(ToolAttributes.LOCAL)
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

        attrs = ToolAttributes(ToolAttributes.LOCAL)
        result = await tools.execute(
            "async_attr_tool",
            {"key": "value"},
            func,
            attributes=attrs,
        )
        assert result["key"] == "value"


# ============================================================================
# Tool guardrails
# ============================================================================


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
        subscribers.deregister("py_san_req_sub")
        guardrails.deregister_tool_sanitize_request("py_san_req")

        start_events = [e for e in events if e.event_type == EventType.Start]
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

    def test_duplicate_guardrail_raises(self):
        guardrails.register_tool_sanitize_request("py_dup_guard", 1, lambda n, a: a)
        with pytest.raises(RuntimeError):
            guardrails.register_tool_sanitize_request("py_dup_guard", 1, lambda n, a: a)
        guardrails.deregister_tool_sanitize_request("py_dup_guard")

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


# ============================================================================
# Tool intercepts
# ============================================================================


class TestToolIntercepts:
    def test_request_intercept_register_deregister(self):
        intercepts.register_tool_request("py_req_int", 1, False, lambda n, a: a)
        assert intercepts.deregister_tool_request("py_req_int")
        assert not intercepts.deregister_tool_request("py_req_int")

    def test_response_intercept_register_deregister(self):
        intercepts.register_tool_response("py_resp_int", 1, False, lambda n, r: r)
        assert intercepts.deregister_tool_response("py_resp_int")

    def test_execution_intercept_register_deregister(self):
        intercepts.register_tool_execution(
            "py_exec_int",
            1,
            lambda name, args: False,
            lambda args: {"intercepted": True},
        )
        assert intercepts.deregister_tool_execution("py_exec_int")

    def test_duplicate_intercept_raises(self):
        intercepts.register_tool_request("py_dup_int", 1, False, lambda n, a: a)
        with pytest.raises(RuntimeError):
            intercepts.register_tool_request("py_dup_int", 1, False, lambda n, a: a)
        intercepts.deregister_tool_request("py_dup_int")


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

    async def test_response_intercept_modifies_result(self):
        def intercept_fn(name, result):
            result["post_processed"] = True
            return result

        intercepts.register_tool_response("py_resp_mod", 1, False, intercept_fn)

        def func(args):
            return {"output": "raw"}

        result = await tools.execute("resp_tool", {}, func)
        assert result["output"] == "raw"
        assert result["post_processed"] is True

        intercepts.deregister_tool_response("py_resp_mod")

    async def test_execution_intercept_replaces_func(self):
        intercepts.register_tool_execution(
            "py_exec_replace",
            1,
            lambda name, args: True,
            lambda args: {"from_intercept": True},
        )

        def original_func(args):
            return {"from_original": True}

        result = await tools.execute("replaced_tool", {}, original_func)
        assert result["from_intercept"] is True
        assert "from_original" not in result

        intercepts.deregister_tool_execution("py_exec_replace")

    async def test_request_intercept_break_chain(self):
        def first_fn(name, args):
            args["from_first"] = True
            return args

        def second_fn(name, args):
            args["from_second"] = True
            return args

        intercepts.register_tool_request("py_chain1", 1, True, first_fn)  # break_chain=True
        intercepts.register_tool_request("py_chain2", 2, False, second_fn)

        def func(args):
            return args

        result = await tools.execute("chain_tool", {}, func)
        assert result["from_first"] is True
        assert "from_second" not in result

        intercepts.deregister_tool_request("py_chain1")
        intercepts.deregister_tool_request("py_chain2")


# ============================================================================
# LLM lifecycle
# ============================================================================


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


# ============================================================================
# LLM guardrails
# ============================================================================


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


# ============================================================================
# LLM intercepts
# ============================================================================


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


# ============================================================================
# LLM streaming
# ============================================================================


class TestLLMStreaming:
    async def test_stream_execute(self):
        # Stream functions should be sync, returning an async generator
        def stream_func(request):
            async def gen():
                yield 'data: {"token": "hello"}\n\n'
                yield 'data: {"token": "world"}\n\n'

            return gen()

        req = make_request()
        stream = await llm.stream_execute("stream_llm", req, stream_func)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 2


# ============================================================================
# Scope type coverage (all variants)
# ============================================================================


class TestAllScopeTypes:
    @pytest.mark.parametrize(
        "scope_type",
        [
            ScopeType.Agent,
            ScopeType.Function,
            ScopeType.Tool,
            ScopeType.Llm,
            ScopeType.Retriever,
            ScopeType.Embedder,
            ScopeType.Reranker,
            ScopeType.Guardrail,
            ScopeType.Evaluator,
            ScopeType.Custom,
            ScopeType.Unknown,
        ],
    )
    def test_push_with_scope_type(self, scope_type):
        handle = scope.push(f"test_{scope_type}", scope_type)
        assert handle.name.startswith("test_")
        scope.pop(handle)


# ============================================================================
# Event subscriber details
# ============================================================================


class TestSubscriberEventDetails:
    def test_scope_events_have_correct_types(self):
        events = []
        subscribers.register("py_detail_sub", lambda e: events.append(e))
        handle = scope.push("detail_test", ScopeType.Evaluator)
        scope.pop(handle)
        subscribers.deregister("py_detail_sub")

        assert len(events) >= 2
        assert events[0].event_type == EventType.Start
        assert events[1].event_type == EventType.End

    def test_tool_events(self):
        events = []
        subscribers.register("py_tool_evt", lambda e: events.append(e))
        handle = tools.call("evt_tool", {"x": 1})
        tools.call_end(handle, {"y": 2})
        subscribers.deregister("py_tool_evt")

        start_events = [e for e in events if e.event_type == EventType.Start]
        end_events = [e for e in events if e.event_type == EventType.End]
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    def test_llm_events(self):
        events = []
        subscribers.register("py_llm_evt", lambda e: events.append(e))
        req = make_request()
        handle = llm.call("evt_llm", req)
        llm.call_end(handle, {"done": True})
        subscribers.deregister("py_llm_evt")

        start_events = [e for e in events if e.event_type == EventType.Start]
        end_events = [e for e in events if e.event_type == EventType.End]
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    def test_mark_event(self):
        events = []
        subscribers.register("py_mark_evt", lambda e: events.append(e))
        scope.event("test_mark", data={"info": "test"})
        subscribers.deregister("py_mark_evt")

        mark_events = [e for e in events if e.event_type == EventType.Mark]
        assert len(mark_events) >= 1


# ============================================================================
# Handle property access
# ============================================================================


class TestHandleProperties:
    def test_scope_handle_all_properties(self):
        handle = scope.push("prop_test", ScopeType.Embedder)
        assert isinstance(handle.uuid, str)
        assert len(handle.uuid) > 0
        assert handle.name == "prop_test"
        assert handle.scope_type == ScopeType.Embedder
        assert handle.parent_uuid is not None  # root is parent
        # data and metadata are None by default for scope handles
        scope.pop(handle)

    def test_tool_handle_all_properties(self):
        handle = tools.call("prop_tool", {"x": 1}, data={"d": "v"}, metadata={"m": "v"})
        assert isinstance(handle.uuid, str)
        assert handle.name == "prop_tool"
        # data includes sanitized_args from the call
        assert handle.data is not None
        tools.call_end(handle, {})

    def test_llm_handle_all_properties(self):
        req = make_request()
        handle = llm.call("prop_llm", req, data={"d": 1}, metadata={"m": 2})
        assert isinstance(handle.uuid, str)
        assert handle.name == "prop_llm"
        assert handle.data is not None
        llm.call_end(handle, {})

    def test_event_all_properties(self):
        events = []
        subscribers.register("py_prop_evt", lambda e: events.append(e))
        scope.event("prop_mark", data={"key": "val"}, metadata={"meta": "data"})
        subscribers.deregister("py_prop_evt")

        assert len(events) >= 1
        e = events[0]
        assert isinstance(e.uuid, str)
        assert e.name == "prop_mark"
        assert e.event_type == EventType.Mark
        assert e.timestamp is not None
        assert isinstance(e.timestamp, str)
