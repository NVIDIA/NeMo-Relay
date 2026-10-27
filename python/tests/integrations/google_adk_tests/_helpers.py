# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide shared helpers for the Google ADK integration tests."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Sequence
from typing import TYPE_CHECKING, Concatenate, Final, Literal, NamedTuple, ParamSpec, Protocol, TypeAlias, TypeVar
from uuid import uuid4

from typing_extensions import TypeIs, override

import nemo_relay

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.agents.llm_agent import ToolUnion
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.agents.run_config import RunConfig
    from google.adk.events.event import Event
    from google.adk.models import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins import BasePlugin
    from google.adk.runners import Runner
    from google.adk.tools import BaseTool
    from google.adk.tools.base_toolset import BaseToolset
    from google.adk.tools.tool_context import ToolContext
    from google.adk.workflow import BaseNode
    from google.genai import types

    from nemo_relay.integrations.google_adk import NemoRelayPlugin

APP_NAME: Final = "relay_test_app"
HANG_TIMEOUT: Final = 30.0
Category: TypeAlias = Literal["agent", "llm", "tool", "function"]
LlmNext: TypeAlias = Callable[[nemo_relay.LLMRequest], Awaitable[nemo_relay.Json]]
LlmStreamNext: TypeAlias = Callable[[nemo_relay.LLMRequest], Awaitable[AsyncIterator[nemo_relay.Json]]]
ToolNext: TypeAlias = Callable[[nemo_relay.Json], Awaitable[nemo_relay.ToolExecutionResult[nemo_relay.Json]]]
_T = TypeVar("_T")


class ToolCallRecord(NamedTuple):
    args: dict[str, object]
    result: object


def present(value: _T | None) -> _T:
    assert value is not None
    return value


def as_instance(value: object, cls: type[_T]) -> _T:
    assert isinstance(value, cls)
    return value


def _is_object(value: object) -> TypeIs[nemo_relay.JsonObject]:
    return isinstance(value, dict)


def _is_list(value: object) -> TypeIs[list[nemo_relay.Json]]:
    return isinstance(value, list)


def as_object(value: object) -> nemo_relay.JsonObject:
    assert _is_object(value)
    return value


def as_list(value: object) -> list[nemo_relay.Json]:
    assert _is_list(value)
    return value


def metadata_of(event: nemo_relay.ScopeEvent | nemo_relay.MarkEvent) -> nemo_relay.JsonObject:
    return as_object(event.metadata)


def status_of(event: nemo_relay.ScopeEvent) -> nemo_relay.Json:
    return metadata_of(event)["otel.status_code"]


def data_of(event: nemo_relay.ScopeEvent | nemo_relay.MarkEvent) -> nemo_relay.JsonObject:
    return as_object(event.data)


def request_of(start: nemo_relay.ScopeEvent) -> nemo_relay.AnnotatedLLMRequest:
    return present(start.annotated_request)


def response_of(end: nemo_relay.ScopeEvent) -> nemo_relay.AnnotatedLLMResponse:
    return present(end.annotated_response)


class _ScriptedLlmClass(Protocol):
    def __call__(self, *, model: str, call: types.FunctionCall | None, record: ScriptedLlmFactory) -> BaseLlm: ...


class ScriptedLlmFactory:
    """Build scripted models that record their requests and responses on this factory."""

    def __init__(self) -> None:
        self.yielded: list[LlmResponse] = []
        self.seen_requests: list[LlmRequest] = []

    def __call__(self, *, model: str = "scripted-model", call: types.FunctionCall | None = None) -> BaseLlm:
        return _scripted_llm_class()(model=model, call=call, record=self)


def scope_events(events: list[nemo_relay.Event]) -> list[nemo_relay.ScopeEvent]:
    return [event for event in events if isinstance(event, nemo_relay.ScopeEvent)]


def starts(events: list[nemo_relay.Event], category: Category | None = None) -> list[nemo_relay.ScopeEvent]:
    return [
        event
        for event in scope_events(events)
        if event.scope_category == "start" and (category is None or event.category == category)
    ]


def ends(events: list[nemo_relay.Event], category: Category | None = None) -> list[nemo_relay.ScopeEvent]:
    return [
        event
        for event in scope_events(events)
        if event.scope_category == "end" and (category is None or event.category == category)
    ]


def _by_uuid(events: list[nemo_relay.Event]) -> dict[str, nemo_relay.ScopeEvent]:
    return {event.uuid: event for event in starts(events)}


def parent_of(events: list[nemo_relay.Event], event: nemo_relay.ScopeEvent) -> nemo_relay.ScopeEvent:
    parent = _by_uuid(events).get(event.parent_uuid or "")
    assert parent is not None, f"{event.name} has no recorded parent"
    return parent


def marks(events: list[nemo_relay.Event], name: str) -> list[nemo_relay.MarkEvent]:
    return [event for event in events if isinstance(event, nemo_relay.MarkEvent) and event.name == name]


def chunk_indexes(events: list[nemo_relay.Event]) -> list[nemo_relay.Json]:
    return [data_of(chunk)["chunk_index"] for chunk in marks(events, "llm.chunk")]


def assert_balanced(events: list[nemo_relay.Event]) -> None:
    """Assert that every scope that started also ended."""
    assert {event.uuid for event in starts(events)} == {event.uuid for event in ends(events)}


def _only(matches: list[nemo_relay.ScopeEvent], description: str) -> nemo_relay.ScopeEvent:
    assert len(matches) == 1, f"expected one {description}, found {len(matches)}"
    return matches[0]


def named(events: list[nemo_relay.Event], name: str, category: Category | None = None) -> nemo_relay.ScopeEvent:
    return _only([event for event in starts(events, category) if event.name == name], f"start of {name!r}")


def named_end(events: list[nemo_relay.Event], name: str, category: Category | None = None) -> nemo_relay.ScopeEvent:
    return _only([event for event in ends(events, category) if event.name == name], f"end of {name!r}")


def invocation_starts(events: list[nemo_relay.Event]) -> list[nemo_relay.ScopeEvent]:
    return [event for event in starts(events, "agent") if event.name.startswith("adk-invocation:")]


def invocation_start(events: list[nemo_relay.Event]) -> nemo_relay.ScopeEvent:
    invocations = invocation_starts(events)
    assert invocations, "no ADK invocation was recorded"
    return invocations[0]


def final_text(events: list[Event]) -> str | None:
    for event in reversed(events):
        if event.content and not event.partial:
            for part in event.content.parts or []:
                if part.text:
                    return part.text
    return None


def partial_texts(events: list[Event]) -> list[str]:
    return [
        part.text
        for event in events
        if event.partial and event.content
        for part in event.content.parts or []
        if part.text is not None
    ]


def function_responses(events: list[Event]) -> list[types.FunctionResponse]:
    return [response for event in events for response in event.get_function_responses()]


def tool_call(name: str, **args: object) -> types.FunctionCall:
    from google.genai import types

    return types.FunctionCall(name=name, args=args)


def weather_call(location: str = "San Francisco") -> types.FunctionCall:
    return tool_call("get_weather", location=location)


def model_answer(text: str) -> LlmResponse:
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason=types.FinishReason.STOP
    )


@functools.cache
def _scripted_llm_class() -> _ScriptedLlmClass:
    from google.adk.models import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class ScriptedLlm(BaseLlm):
        """Make the scripted tool call unless the last content answers a call.

        Otherwise, answer with the latest user text. A streamed answer arrives as two partial chunks before the final
        response.
        """

        model: str = "scripted-model"
        call: types.FunctionCall | None = None
        record: ScriptedLlmFactory

        # Keep the generated constructor visible when ADK isn't installed.
        if TYPE_CHECKING:

            def __init__(
                self,
                *,
                model: str = "scripted-model",
                call: types.FunctionCall | None = None,
                record: ScriptedLlmFactory,
            ) -> None: ...

        @override
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            self.record.seen_requests.append(llm_request)
            last = llm_request.contents[-1] if llm_request.contents else None
            answered = bool(last and any(part.function_response for part in last.parts or []))
            if self.call is not None and not answered:
                assert self.call.name in llm_request.tools_dict, f"{self.call.name} is not offered to {self.model}"
                response = LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(function_call=self.call.model_copy())]),
                    finish_reason=types.FinishReason.STOP,
                )
                self.record.yielded.append(response)
                yield response
                return

            agent_name = present(llm_request.config.labels)["adk_agent_name"]
            user_text = next(
                (
                    part.text
                    for content in reversed(llm_request.contents)
                    if content.role == "user"
                    for part in content.parts or []
                    if part.text
                ),
                "",
            )
            text = f"Answer from {agent_name} to {user_text!r}."
            if stream:
                for piece in (text[:6], text[6:]):
                    chunk = LlmResponse(
                        content=types.Content(role="model", parts=[types.Part(text=piece)]), partial=True
                    )
                    self.record.yielded.append(chunk)
                    yield chunk
            final = LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                partial=False if stream else None,
                finish_reason=types.FinishReason.STOP,
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=10, candidates_token_count=5, total_token_count=15
                ),
            )
            self.record.yielded.append(final)
            yield final

    return ScriptedLlm


def responding_llm(respond: Callable[[LlmRequest], LlmResponse], *, model: str) -> BaseLlm:
    from google.adk.models import BaseLlm

    class RespondingLlm(BaseLlm):
        @override
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield respond(llm_request)

    return RespondingLlm(model=model)


def get_weather(location: str) -> dict[str, str]:
    """Get the weather for a location."""
    return {"location": location, "forecast": "sunny"}


def json_tool_result(value: dict[str, str | int | float | bool | None]) -> nemo_relay.JsonObject:
    """Return the tool-result wrapper that the ADK codec writes for a flat object of JSON scalars."""
    return {
        "schema": "nemo.relay.tool_result@1",
        "content": json.dumps(value, ensure_ascii=False, sort_keys=True),
        "is_error": False,
        "result": {"kind": "dict", "value": {key: {"kind": "json", "value": item} for key, item in value.items()}},
    }


def weather_result(location: str = "San Francisco") -> nemo_relay.JsonObject:
    return json_tool_result({"location": location, "forecast": "sunny"})


def make_agent(
    scripted_llm: ScriptedLlmFactory,
    name: str,
    *,
    tools: Sequence[ToolUnion | BaseNode] = (),
    sub_agents: Sequence[BaseAgent] = (),
    call: types.FunctionCall | None = None,
) -> LlmAgent:
    from google.adk.agents import LlmAgent

    return LlmAgent(
        name=name,
        model=scripted_llm(model=f"scripted-{name}", call=call),
        instruction="Be brief.",
        tools=list(tools),
        sub_agents=list(sub_agents),
    )


def toolset_of(tools: Sequence[BaseTool], *, prefix: str | None = None) -> BaseToolset:
    from google.adk.tools.base_toolset import BaseToolset

    class Toolset(BaseToolset):
        @override
        async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
            return list(tools)

    return Toolset(tool_name_prefix=prefix)


def tool_witness(records: list[ToolCallRecord]) -> BasePlugin:
    from google.adk.plugins import BasePlugin

    class ToolWitness(BasePlugin):
        @override
        async def after_tool_callback(
            self, *, tool: BaseTool, tool_args: dict[str, object], tool_context: ToolContext, result: dict[str, object]
        ) -> dict[str, object] | None:
            records.append(ToolCallRecord(dict(tool_args), result))
            return None

    return ToolWitness("tool_witness")


def event_signal(reached: asyncio.Event, matches: Callable[[Event], bool]) -> BasePlugin:
    from google.adk.plugins import BasePlugin

    class EventSignal(BasePlugin):
        @override
        async def on_event_callback(self, *, invocation_context: InvocationContext, event: Event) -> Event | None:
            if matches(event):
                reached.set()
            return None

    return EventSignal("event_signal")


def make_runner(
    root_agent: BaseNode,
    *,
    plugins: Sequence[BasePlugin] | None = None,
    context_cache_config: ContextCacheConfig | None = None,
) -> Runner:
    from google.adk.apps import App
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    app = App(
        name=APP_NAME,
        root_agent=root_agent,
        plugins=[NemoRelayPlugin()] if plugins is None else list(plugins),
        context_cache_config=context_cache_config,
    )
    return Runner(app=app, session_service=InMemorySessionService(), artifact_service=InMemoryArtifactService())


async def new_session(runner: Runner) -> str:
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="user")
    return session.id


async def run_turn(
    runner: Runner,
    *,
    session_id: str | None = None,
    message: str | types.Content = "hi",
    run_config: RunConfig | None = None,
) -> list[Event]:
    from google.genai import types

    content = types.UserContent(message) if isinstance(message, str) else message
    return [
        event
        async for event in runner.run_async(
            user_id="user",
            session_id=session_id or await new_session(runner),
            new_message=content,
            run_config=run_config,
        )
    ]


async def run_app(
    root_agent: BaseNode,
    *,
    run_config: RunConfig | None = None,
    plugin: NemoRelayPlugin | None = None,
    extra_plugins: Sequence[BasePlugin] = (),
    message: str = "hi",
    context_cache_config: ContextCacheConfig | None = None,
) -> list[Event]:
    from nemo_relay.integrations.google_adk import NemoRelayPlugin

    runner = make_runner(
        root_agent, plugins=[plugin or NemoRelayPlugin(), *extra_plugins], context_cache_config=context_cache_config
    )
    try:
        return await run_turn(runner, message=message, run_config=run_config)
    finally:
        await runner.close()


async def run_recorded(
    root_agent: BaseNode,
    *,
    run_config: RunConfig | None = None,
    extra_plugins: Sequence[BasePlugin] = (),
    message: str = "hi",
    context_cache_config: ContextCacheConfig | None = None,
) -> list[Event]:
    """Run the agent under a `request` scope, and flush the subscribers after the run ends or fails."""
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            return await run_app(
                root_agent,
                run_config=run_config,
                extra_plugins=extra_plugins,
                message=message,
                context_cache_config=context_cache_config,
            )
    finally:
        await nemo_relay.subscribers.flush_async()


async def hang() -> None:
    await asyncio.Event().wait()


async def run_until_interrupted(
    runner: Runner, session_id: str, reached: asyncio.Event, *, run_config: RunConfig | None = None
) -> None:
    """Run a turn in the current task, and cancel it once `reached` is set.

    Fail when the run ends or times out before `reached` is set.
    """
    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(HANG_TIMEOUT) as deadline:
            waiter = asyncio.ensure_future(reached.wait())

            def interrupt(_: asyncio.Future[Literal[True]]) -> None:
                if reached.is_set():
                    deadline.reschedule(loop.time())

            waiter.add_done_callback(interrupt)
            try:
                await run_turn(runner, session_id=session_id, run_config=run_config)
            finally:
                waiter.cancel()
    except TimeoutError:
        assert reached.is_set(), "the run hung before it reached the interruption point"
    else:
        raise AssertionError("the run ended without reaching the interruption point")


_P = ParamSpec("_P")


@contextlib.contextmanager
def _registered(
    register: Callable[Concatenate[str, _P], object],
    deregister: Callable[[str], object],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> Generator[None, None, None]:
    """Register a component under a unique name for the block, and deregister it when the block exits."""
    name = f"adk-test-{uuid4()}"
    register(name, *args, **kwargs)
    try:
        yield
    finally:
        deregister(name)


def tool_execution_intercept(fn: nemo_relay.ToolExecutionIntercept) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.intercepts.register_tool_execution, nemo_relay.intercepts.deregister_tool_execution, 1, fn
    )


def llm_execution_intercept(fn: nemo_relay.LlmExecutionIntercept) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.intercepts.register_llm_execution, nemo_relay.intercepts.deregister_llm_execution, 1, fn
    )


def llm_stream_execution_intercept(fn: nemo_relay.LlmStreamExecutionIntercept) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.intercepts.register_llm_stream_execution,
        nemo_relay.intercepts.deregister_llm_stream_execution,
        1,
        fn,
    )


def llm_request_intercept(fn: nemo_relay.LlmRequestIntercept) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.intercepts.register_llm_request,
        nemo_relay.intercepts.deregister_llm_request,
        1,
        break_chain=False,
        fn=fn,
    )


def tool_request_intercept(fn: nemo_relay.ToolRequestIntercept) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.intercepts.register_tool_request,
        nemo_relay.intercepts.deregister_tool_request,
        1,
        break_chain=False,
        fn=fn,
    )


def tool_conditional_execution_guardrail(
    fn: nemo_relay.ToolConditionalExecutionGuardrail,
) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.guardrails.register_tool_conditional_execution,
        nemo_relay.guardrails.deregister_tool_conditional_execution,
        1,
        fn,
    )


def llm_conditional_execution_guardrail(
    fn: nemo_relay.LlmConditionalExecutionGuardrail,
) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.guardrails.register_llm_conditional_execution,
        nemo_relay.guardrails.deregister_llm_conditional_execution,
        1,
        fn,
    )


def tool_sanitize_request_guardrail(fn: nemo_relay.ToolSanitizeGuardrail) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.guardrails.register_tool_sanitize_request,
        nemo_relay.guardrails.deregister_tool_sanitize_request,
        1,
        fn,
    )


def tool_sanitize_response_guardrail(fn: nemo_relay.ToolSanitizeGuardrail) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.guardrails.register_tool_sanitize_response,
        nemo_relay.guardrails.deregister_tool_sanitize_response,
        1,
        fn,
    )


def llm_sanitize_request_guardrail(fn: nemo_relay.LlmSanitizeRequestGuardrail) -> AbstractContextManager[None]:
    return _registered(
        nemo_relay.guardrails.register_llm_sanitize_request,
        nemo_relay.guardrails.deregister_llm_sanitize_request,
        1,
        fn,
    )


def invocation_context(agent: BaseAgent, invocation_id: str) -> InvocationContext:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions import InMemorySessionService, Session

    return InvocationContext(
        session_service=InMemorySessionService(),
        invocation_id=invocation_id,
        agent=agent,
        session=Session(id="direct-session", app_name=APP_NAME, user_id="user"),
    )


def with_last_text(content: nemo_relay.JsonObject, text: str) -> nemo_relay.JsonObject:
    """Return a copy of request `content` with `text` in the first part of its last message."""
    edited = copy.deepcopy(content)
    last = as_object(as_list(edited["contents"])[-1])
    as_object(as_list(last["parts"])[0])["text"] = text
    return edited


def text_response(text: str) -> nemo_relay.JsonObject:
    """Return a final `LlmResponse` dump that holds the text, for an execution intercept to answer with."""
    return {"content": {"role": "model", "parts": [{"text": text}]}, "finish_reason": "STOP"}


def _uppercase_part(part: nemo_relay.Json) -> nemo_relay.Json:
    if isinstance(part, dict) and isinstance(text := part.get("text"), str):
        return {**part, "text": text.upper()}
    return part


def uppercase_chunk(chunk: nemo_relay.Json) -> nemo_relay.JsonObject:
    """Uppercase the text parts of a chunk, and return a chunk without content unchanged."""
    chunk = as_object(chunk)
    content = chunk.get("content")
    if not isinstance(content, dict):
        return chunk
    parts = as_list(content.get("parts"))
    return {**chunk, "content": {**content, "parts": [_uppercase_part(part) for part in parts]}}


def stream_intercept(
    transform: Callable[[nemo_relay.Json], nemo_relay.Json | None],
) -> nemo_relay.LlmStreamExecutionIntercept:
    def intercept(
        name: str, request: nemo_relay.LLMRequest, context: nemo_relay.LlmExecutionContext, next_call: LlmStreamNext
    ) -> AsyncIterator[nemo_relay.Json]:
        async def gen() -> AsyncIterator[nemo_relay.Json]:
            async for chunk in await next_call(request):
                transformed = transform(chunk)
                if transformed is not None:
                    yield transformed

        return gen()

    return intercept
