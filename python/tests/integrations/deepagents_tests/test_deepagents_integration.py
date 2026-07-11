# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Deep Agents NeMo Relay integration."""

from __future__ import annotations

import asyncio
import http.server
import inspect
import json
import threading
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

import nemo_relay

if TYPE_CHECKING:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    import nemo_relay.integrations.deepagents as deepagents_integration


@pytest.fixture(name="deepagents_integration_module", scope="session")
def deepagents_integration_module_fixture() -> types.ModuleType:
    import nemo_relay.integrations.deepagents as deepagents_integration

    return deepagents_integration


@pytest.fixture(name="callback_handler")
def callback_handler_fixture(
    deepagents_integration_module: types.ModuleType,
) -> deepagents_integration.NemoRelayDeepAgentsCallbackHandler:
    return deepagents_integration_module.NemoRelayDeepAgentsCallbackHandler()


def _mock_deepagents_chat_model(responses: list[Any]) -> FakeMessagesListChatModel:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class _MockDeepAgentsChatModel(FakeMessagesListChatModel):
        model: str = "mock-model"

        def bind_tools(self, _tools: Any, *_args: Any, **_kwargs: Any) -> _MockDeepAgentsChatModel:
            return self

    return _MockDeepAgentsChatModel(responses=responses)


class _CollectorRequest(TypedDict):
    path: str
    headers: dict[str, str]
    body: bytes


class _OtelCollectorHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        server = cast("_OtelCollectorServer", self.server)
        server.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": body,
            }
        )
        server.request_event.set()
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        return


class _OtelCollector:
    server: "_OtelCollectorServer"

    def __enter__(self) -> "_OtelCollector":
        self.server = _OtelCollectorServer(("127.0.0.1", 0), _OtelCollectorHandler)
        self.server.requests = []
        self.server.request_event = threading.Event()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1/traces"

    @property
    def body(self) -> bytes:
        return b"".join(request["body"] for request in self.server.requests)

    def wait_for_request(self, timeout: float = 5.0) -> _CollectorRequest:
        assert self.server.request_event.wait(timeout), "timed out waiting for OTLP request"
        return self.server.requests[0]

    def wait_for_spans(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            spans = _decode_otlp_spans(self.body)
            if spans:
                return spans
            time.sleep(0.05)
        raise AssertionError("timed out waiting for OTLP spans")


class _OtelCollectorServer(http.server.ThreadingHTTPServer):
    requests: list[_CollectorRequest]
    request_event: threading.Event


def _any_value_to_python(value: Any) -> Any:
    value_kind = value.WhichOneof("value")
    if value_kind is None:
        return None
    if value_kind == "array_value":
        return [_any_value_to_python(item) for item in value.array_value.values]
    if value_kind == "kvlist_value":
        return {item.key: _any_value_to_python(item.value) for item in value.kvlist_value.values}
    return getattr(value, value_kind)


def _decode_otlp_spans(body: bytes) -> list[dict[str, Any]]:
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    spans: list[dict[str, Any]] = []
    for resource_span in request.resource_spans:
        for scope_span in resource_span.scope_spans:
            spans.extend(
                {
                    "name": span.name,
                    "attributes": {
                        attribute.key: _any_value_to_python(attribute.value) for attribute in span.attributes
                    },
                }
                for span in scope_span.spans
            )
    return spans


def _span_attrs_by_kind(spans: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], span["attributes"])
        for span in spans
        if span["attributes"].get("openinference.span.kind") == kind
    ]


def _has_input_message(attrs: dict[str, Any], role: str, content: str) -> bool:
    index = 0
    while f"llm.input_messages.{index}.message.role" in attrs:
        if (
            attrs[f"llm.input_messages.{index}.message.role"] == role
            and attrs.get(f"llm.input_messages.{index}.message.content") == content
        ):
            return True
        index += 1
    return False


def _filter_mark_events(events: list[nemo_relay.Event]) -> list[nemo_relay.MarkEvent]:
    return [event for event in events if isinstance(event, nemo_relay.MarkEvent)]


def _mark_data(mark: nemo_relay.MarkEvent) -> dict[str, Any]:
    assert isinstance(mark.data, dict)
    return cast(dict[str, Any], mark.data)


def _mark_metadata(mark: nemo_relay.MarkEvent) -> dict[str, Any]:
    assert isinstance(mark.metadata, dict)
    return cast(dict[str, Any], mark.metadata)


def test_before_agent_emits_configuration_mark(
    subscribed_events: list[nemo_relay.Event],
    deepagents_integration_module: types.ModuleType,
):
    middleware = deepagents_integration_module.NemoRelayDeepAgentsMiddleware(
        agent_name="main-agent",
        skills=["/skills/research/"],
        subagents=[{"name": "researcher"}],
        backend_name="StateBackend",
    )

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        middleware.before_agent(MagicMock(name="mock_state"), MagicMock(name="mock_runtime"))

    nemo_relay.subscribers.flush()
    marks = _filter_mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["DeepAgents Skills Configured"]
    assert _mark_metadata(marks[0])["deepagents_kind"] == "skill"
    assert _mark_data(marks[0])["skills"] == ["/skills/research/"]
    assert _mark_data(marks[0])["subagents"] == [{"name": "researcher"}]
    assert _mark_data(marks[0])["backend"] == "StateBackend"


@pytest.mark.parametrize("use_async", [False, True])
def test_model_call_routes_through_langchain_execution_middleware(
    use_async: bool,
    deepagents_integration_module: types.ModuleType,
):
    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain_core.messages import AIMessage, HumanMessage

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    class _RecordingMiddleware(deepagents_integration_module.NemoRelayDeepAgentsMiddleware):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        async def _llm_execute(
            self,
            model_name: str,
            request: nemo_relay.LLMRequest,
            codec: Any,
            response_codec: Any,
            func: Any,
        ) -> Any:
            self.calls.append(
                {
                    "model_name": model_name,
                    "request": request,
                    "codec": codec,
                    "response_codec": response_codec,
                }
            )
            intercepted = nemo_relay.LLMRequest(
                request.headers,
                {
                    **request.content,
                    "model_settings": {"temperature": 0.25},
                },
            )
            return await func(intercepted)

    middleware = _RecordingMiddleware()
    request = ModelRequest(
        model=_mock_deepagents_chat_model([AIMessage(content="unused")]),
        messages=[HumanMessage(content="hello")],
        model_settings={"temperature": 1.0},
    )
    seen_request: dict[str, ModelRequest[Any]] = {}

    def handler(next_request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen_request["request"] = next_request
        return ModelResponse(result=[AIMessage(content="done")])

    async def async_handler(next_request: ModelRequest[Any]) -> ModelResponse[Any]:
        return handler(next_request)

    if use_async:
        response = asyncio.run(middleware.awrap_model_call(request, async_handler))
    else:
        response = middleware.wrap_model_call(request, handler)

    assert response.result[0].content == "done"
    assert seen_request["request"].model_settings == {"temperature": 0.25}
    assert middleware.calls[0]["model_name"] == "mock-model"
    assert isinstance(middleware.calls[0]["codec"], LangChainCodec)
    assert middleware.calls[0]["response_codec"] is middleware.calls[0]["codec"]


@pytest.mark.parametrize("use_async", [False, True])
def test_tool_call_routes_through_langchain_execution_middleware(
    use_async: bool,
    monkeypatch: pytest.MonkeyPatch,
    deepagents_integration_module: types.ModuleType,
):
    from langchain.agents.middleware import ToolCallRequest
    from langchain_core.messages import ToolMessage

    parent_handle = MagicMock()
    mock_tool_execute = AsyncMock()

    async def execute_side_effect(*, func: Any, **_kwargs: Any) -> ToolMessage:
        result = func({"query": "intercepted"})
        if inspect.isawaitable(result):
            return await result
        return result

    mock_tool_execute.side_effect = execute_side_effect
    monkeypatch.setattr(nemo_relay.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_relay.typed, "tool_execute", mock_tool_execute)

    middleware = deepagents_integration_module.NemoRelayDeepAgentsMiddleware()
    request = ToolCallRequest(
        tool_call={"name": "lookup", "args": {"query": "original"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=MagicMock(),
    )
    seen_request: dict[str, ToolCallRequest] = {}

    def handler(next_request: ToolCallRequest) -> ToolMessage:
        seen_request["request"] = next_request
        return ToolMessage(content="done", tool_call_id=next_request.tool_call["id"])

    async def async_handler(next_request: ToolCallRequest) -> ToolMessage:
        return handler(next_request)

    if use_async:
        response = asyncio.run(middleware.awrap_tool_call(request, async_handler))
    else:
        response = middleware.wrap_tool_call(request, handler)

    assert response.content == "done"
    assert seen_request["request"].tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    assert mock_tool_execute.await_args is not None
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle


def test_skill_load_mark_survives_deepagents_middleware(
    subscribed_events: list[nemo_relay.Event],
    deepagents_integration_module: types.ModuleType,
):
    from langchain.agents.middleware import ToolCallRequest
    from langchain_core.messages import ToolMessage

    middleware = deepagents_integration_module.NemoRelayDeepAgentsMiddleware()
    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "args": {"path": "/skills/review/SKILL.md"},
            "id": "call-skill",
        },
        tool=None,
        state={},
        runtime=MagicMock(),
    )

    with nemo_relay.scope.scope("deepagents-skill", nemo_relay.ScopeType.Agent):
        response = middleware.wrap_tool_call(
            request,
            lambda next_request: ToolMessage(
                content="loaded",
                tool_call_id=next_request.tool_call["id"],
            ),
        )

    nemo_relay.subscribers.flush()
    assert response.content == "loaded"
    marks = _filter_mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["skill.load"]
    assert _mark_data(marks[0]) == {"skill_name": "review"}
    assert _mark_metadata(marks[0]) == {
        "skill_load_source": "structured_read",
        "tool_name": "read_file",
    }
    tool_start = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent) and event.name == "read_file" and event.scope_category == "start"
    )
    assert marks[0].parent_uuid == tool_start.uuid


def test_callback_handler_emits_human_in_the_loop_marks(
    subscribed_events: list[nemo_relay.Event],
    callback_handler: deepagents_integration.NemoRelayDeepAgentsCallbackHandler,
):
    from langgraph.callbacks import GraphInterruptEvent, GraphResumeEvent
    from langgraph.types import Interrupt

    run_id = uuid4()
    hitl_request = {
        "action_requests": [
            {
                "name": "edit_file",
                "args": {"file_path": "/workspace/notes.md"},
                "description": "Tool execution requires approval",
            }
        ],
        "review_configs": [{"action_name": "edit_file", "allowed_decisions": ["approve", "reject"]}],
    }

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        callback_handler.on_interrupt(
            GraphInterruptEvent(
                run_id=run_id,
                status="interrupt_after",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
                interrupts=(Interrupt(hitl_request, id="interrupt-1"),),
            )
        )
        callback_handler.on_resume(
            GraphResumeEvent(
                run_id=run_id,
                status="pending",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
            )
        )

    nemo_relay.subscribers.flush()
    marks = _filter_mark_events(subscribed_events)
    assert [mark.name for mark in marks] == [
        "DeepAgents Human In The Loop Interrupt",
        "DeepAgents Human In The Loop Resume",
    ]
    assert _mark_metadata(marks[0])["deepagents_kind"] == "human_in_the_loop"
    assert _mark_data(marks[0])["interrupts"] == [{"id": "interrupt-1", "value": hitl_request}]
    assert _mark_metadata(marks[1])["phase"] == "resume"


def test_callback_handler_falls_back_for_non_hitl_interrupt(
    subscribed_events: list[nemo_relay.Event],
    callback_handler: deepagents_integration.NemoRelayDeepAgentsCallbackHandler,
):
    from langgraph.callbacks import GraphInterruptEvent, GraphResumeEvent
    from langgraph.types import Interrupt

    run_id = uuid4()

    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        callback_handler.on_interrupt(
            GraphInterruptEvent(
                run_id=run_id,
                status="interrupt_after",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
                interrupts=(Interrupt("custom pause", id="interrupt-1"),),
            )
        )
        callback_handler.on_resume(
            GraphResumeEvent(
                run_id=run_id,
                status="pending",
                checkpoint_id="checkpoint-1",
                checkpoint_ns=("parent",),
            )
        )

    nemo_relay.subscribers.flush()
    marks = _filter_mark_events(subscribed_events)
    assert [mark.name for mark in marks] == ["Graph Interrupt", "Graph Resume"]
    assert _mark_metadata(marks[0])["integration"] == "langgraph"
    assert "deepagents_kind" not in _mark_metadata(marks[0])


def test_add_nemo_relay_integration_preserves_backend(deepagents_integration_module: types.ModuleType):
    mock_backend = MagicMock(name="mock_backend")
    mock_compiled_subagent = MagicMock(name="mock_compiled_subagent")
    kwargs = deepagents_integration_module.add_nemo_relay_integration(
        model="mock-model",
        name="main-agent",
        skills=["/skills/main/"],
        backend=mock_backend,
        middleware=[MagicMock(name="mock_middleware")],
        subagents=[
            {"name": "researcher", "description": "Research", "skills": ["/skills/research/"]},
            mock_compiled_subagent,
        ],
    )

    assert kwargs["backend"] is mock_backend
    assert any(
        isinstance(item, deepagents_integration_module.NemoRelayDeepAgentsMiddleware) for item in kwargs["middleware"]
    )
    assert any(
        isinstance(item, deepagents_integration_module.NemoRelayDeepAgentsMiddleware)
        for item in kwargs["subagents"][0]["middleware"]
    )
    assert kwargs["subagents"][1] is mock_compiled_subagent


@pytest.mark.parametrize("use_async", [False, True])
def test_e2e_agent(
    use_async: bool,
    tmp_path: Path,
    subscribed_events: list[nemo_relay.Event],
    deepagents_integration_module: types.ModuleType,
):
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langchain_core.messages import AIMessage, ToolMessage

    reviewer_description = "Reviews filesystem work performed by the main agent."
    reviewer_model = _mock_deepagents_chat_model(
        responses=[
            AIMessage(content="reviewer verified turtle"),
        ]
    )
    model = _mock_deepagents_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/turtle", "content": "shell"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Review the file creation result and report whether turtle was created.",
                            "subagent_type": "reviewer",
                        },
                        "id": "call-2",
                    }
                ],
            ),
            AIMessage(content="created turtle after reviewer verified turtle"),
        ]
    )

    kwargs = deepagents_integration_module.add_nemo_relay_integration(
        model=model,
        tools=[],
        name="main-agent",
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True),
        subagents=[
            {
                "name": "reviewer",
                "description": reviewer_description,
                "system_prompt": "Review the delegated task and return one concise verification sentence.",
                "model": reviewer_model,
                "tools": [],
            }
        ],
    )
    agent = create_deep_agent(**kwargs)

    with nemo_relay.scope.scope("deepagents-request", nemo_relay.ScopeType.Agent):
        input_payload = {"messages": [{"role": "user", "content": "Create a file named turtle."}]}
        if use_async:
            result = asyncio.run(agent.ainvoke(input_payload))
        else:
            result = agent.invoke(input_payload)

    nemo_relay.subscribers.flush()
    assert (tmp_path / "turtle").read_text() == "shell"
    assert result["messages"][-1].content == "created turtle after reviewer verified turtle"
    found_write_file_message = False
    found_subagent_message = False
    for message in result["messages"]:
        if (
            isinstance(message, ToolMessage)
            and message.name == "write_file"
            and message.content == "Updated file /turtle"
        ):
            found_write_file_message = True
        if isinstance(message, ToolMessage) and message.content == "reviewer verified turtle":
            found_subagent_message = True

    assert found_write_file_message
    assert found_subagent_message

    expected_events = [
        "scope.start.deepagents-request",
        "mark..DeepAgents Skills Configured",
        "scope.start.mock-model",
        "scope.end.mock-model",
        "scope.start.write_file",
        "scope.end.write_file",
        "scope.start.mock-model",
        "scope.end.mock-model",
        "scope.start.task",
        "mark..DeepAgents Skills Configured",
        "scope.start.mock-model",
        "scope.end.mock-model",
        "scope.end.task",
        "scope.start.mock-model",
        "scope.end.mock-model",
        "scope.end.deepagents-request",
    ]
    event_strings = [f"{event.kind}.{getattr(event, 'scope_category', '')}.{event.name}" for event in subscribed_events]

    assert event_strings == expected_events


def test_e2e_agent_exports_openinference_output_contract(
    tmp_path: Path,
    deepagents_integration_module: types.ModuleType,
):
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langchain_core.messages import AIMessage

    events: list[nemo_relay.Event] = []
    model = _mock_deepagents_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/turtle", "content": "shell"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="created turtle"),
        ]
    )
    kwargs = deepagents_integration_module.add_nemo_relay_integration(
        model=model,
        tools=[],
        name="main-agent",
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True),
    )
    agent = create_deep_agent(**kwargs)

    with _OtelCollector() as collector:
        config = nemo_relay.OpenInferenceConfig()
        config.endpoint = collector.endpoint
        config.service_name = "deepagents-test"
        subscriber = nemo_relay.OpenInferenceSubscriber(config)
        subscriber_name = f"deepagents_openinference_{uuid4().hex}"
        event_recorder_name = f"deepagents_events_{uuid4().hex}"
        subscriber.register(subscriber_name)
        nemo_relay.subscribers.register(event_recorder_name, events.append)
        try:
            with nemo_relay.scope.scope("deepagents-request", nemo_relay.ScopeType.Agent):
                agent.invoke({"messages": [{"role": "user", "content": "Create a file named turtle."}]})

            nemo_relay.subscribers.flush()
            subscriber.force_flush()
            spans = collector.wait_for_spans()
        finally:
            nemo_relay.subscribers.deregister(event_recorder_name)
            subscriber.deregister(subscriber_name)
            subscriber.shutdown()

        llm_end_events = [
            event
            for event in events
            if isinstance(event, nemo_relay.ScopeEvent) and event.category == "llm" and event.scope_category == "end"
        ]
        assert len(llm_end_events) == 2
        assert all(event.annotated_response is not None for event in llm_end_events)
        first_response = llm_end_events[0].annotated_response
        final_response = llm_end_events[1].annotated_response
        assert first_response is not None
        assert final_response is not None
        assert first_response.tool_calls == [
            {
                "id": "call-1",
                "name": "write_file",
                "arguments": {"content": "shell", "file_path": "/turtle"},
            }
        ]
        assert final_response.response_text() == "created turtle"

        agent_spans = _span_attrs_by_kind(spans, "AGENT")
        llm_spans = _span_attrs_by_kind(spans, "LLM")
        tool_spans = _span_attrs_by_kind(spans, "TOOL")
        assert len(agent_spans) == 1
        assert len(llm_spans) == 2
        assert len(tool_spans) == 1

        first_llm_span = next(
            span for span in llm_spans if "llm.output_messages.0.message.tool_calls.0.tool_call.function.name" in span
        )
        final_llm_span = next(span for span in llm_spans if span.get("llm.output_messages.0.message.content"))
        tool_span = tool_spans[0]
        expected_tool_args = {"content": "shell", "file_path": "/turtle"}
        assert first_llm_span["llm.input_messages.0.message.role"] == "system"
        assert _has_input_message(first_llm_span, "user", "Create a file named turtle.")
        assert first_llm_span["llm.output_messages.0.message.role"] == "assistant"
        assert first_llm_span["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "call-1"
        assert first_llm_span["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "write_file"
        first_llm_args = json.loads(
            first_llm_span["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]
        )
        assert first_llm_args == expected_tool_args
        assert final_llm_span["llm.output_messages.0.message.content"] == "created turtle"
        assert json.loads(tool_span["tool.parameters"]) == expected_tool_args
        assert json.loads(tool_span["tool_call.function.arguments"]) == expected_tool_args
