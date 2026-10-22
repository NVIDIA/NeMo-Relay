# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the provider client behind each ADK model class, and record what the client reports about a model call."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated, Any, Final, NamedTuple, ParamSpec, Protocol, TypeAlias, cast

from google.adk.models.google_llm import Gemini
from google.genai import types

from nemo_relay.integrations.google_adk._serialization import CODECS, AdkCodec, Api
from nemo_relay.integrations.google_adk._swap import swap_class

if TYPE_CHECKING:
    import httpx
    from anthropic import AsyncAnthropic, AsyncAnthropicVertex, NotGiven
    from anthropic import AsyncStream as AnthropicStream
    from anthropic.resources.messages import AsyncMessages
    from anthropic.types import (
        Message,
        MessageParam,
        RawMessageStreamEvent,
        ThinkingConfigAdaptiveParam,
        ThinkingConfigDisabledParam,
        ThinkingConfigEnabledParam,
        ToolChoiceParam,
        ToolUnionParam,
    )
    from google.adk.integrations.openai import AzureOpenAIResponsesLlm, OpenAILlm, OpenAIResponsesLlm
    from google.adk.models import BaseLlm
    from google.adk.models.anthropic_llm import AnthropicLlm
    from google.adk.models.apigee_llm import ApigeeLlm, CompletionsHTTPClient
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai.models import AsyncModels
    from litellm import CustomStreamWrapper
    from litellm.types.utils import ModelResponse, ModelResponseStream
    from openai import AsyncOpenAI
    from openai import AsyncStream as OpenAIStream
    from openai.resources.chat.completions import AsyncCompletions
    from openai.resources.responses import AsyncResponses
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.responses import Response, ResponseStreamEvent
    from pydantic import BaseModel

    from nemo_relay import JsonObject

_P = ParamSpec("_P")
_GeminiStream: TypeAlias = AsyncGenerator[types.GenerateContentResponse, None]
_AnthropicStream: TypeAlias = "AnthropicStream[RawMessageStreamEvent]"
_OpenAIStream: TypeAlias = "OpenAIStream[ChatCompletionChunk]"
_Thinking: TypeAlias = (
    "ThinkingConfigEnabledParam | ThinkingConfigDisabledParam | ThinkingConfigAdaptiveParam | NotGiven"
)
_OPENAI_INTEGRATION: Final = "google.adk.integrations.openai"
_ANY: Final = types.FunctionCallingConfigMode.ANY
_COST_FIELDS: Final[
    Annotated[tuple[str, ...], "This tuple names the usage fields that Relay's provider codecs read cost from."]
] = (
    "cost",
    "cost_usd",
)
_FORCED: Final[Mapping[types.FunctionCallingConfigMode, ToolChoiceParam]] = {
    types.FunctionCallingConfigMode.AUTO: {"type": "auto"},
    _ANY: {"type": "any"},
    types.FunctionCallingConfigMode.NONE: {"type": "none"},
}


class Attempt:
    """Carry one model call attempt's edits to the provider client, and collect what the client reports back.

    The edits are an intercept's tool choice edit and the request headers.
    """

    def __init__(self, tool_choice: types.FunctionCallingConfig | None, headers: Mapping[str, str]) -> None:
        self.tool_choice: Annotated[
            types.FunctionCallingConfig | None,
            "This field holds an intercept's function calling config, or None when ADK's config is unchanged.",
        ] = tool_choice
        self.headers: Annotated[
            Mapping[str, str], "This field holds the request headers that the provider client sends with the attempt."
        ] = headers
        self.urls: Annotated[
            list[types.UrlContextMetadata],
            "This field collects the URL context metadata that the first candidate of each Gemini response reports.",
        ] = []
        self.response_id: str | None = None
        self.response_model: Annotated[
            str | None, "This field holds the model that the latest Chat Completions response or chunk names."
        ] = None
        self.provider_cost: Annotated[
            JsonObject | None,
            "This field holds the latest cost fields of the provider's usage, or None when the usage reports no cost.",
        ] = None


CURRENT_ATTEMPT: Final[ContextVar[Attempt | None]] = ContextVar("nemo_relay_google_adk_attempt", default=None)


class Backend(NamedTuple):
    provider: Annotated[
        str,
        "This field holds the provider's LiteLLM name, or an unknown model's class name, and names the LLM scope.",
    ]
    codec: AdkCodec
    client_headers: Annotated[
        bool,
        "This field is true when request headers travel as client extra headers instead of ADK's HTTP options.",
    ]


async def backend_of(model: BaseLlm) -> Backend:
    """Return the provider, codec, and header channel of the model's class.

    Swap the provider client's class so that the client sends each attempt's request headers, records the response
    fields that ADK's `LlmResponse` drops, or both. A Gemini model that uses the interactions API keeps its client
    unswapped. Any other model class gets its class name as the provider and the Gemini codec. Raise `TypeError` for an
    unknown class from ADK's OpenAI integration.
    """
    if (anthropic_llm := anthropic_llm_class()) is not None and isinstance(model, anthropic_llm):
        client = await model._get_anthropic_client()
        swap_class(client.messages, _messages_methods)
        return Backend(_anthropic_provider(client), CODECS[Api.ANTHROPIC_MESSAGES], client_headers=True)
    if (apigee_llm := _apigee_llm()) is not None and isinstance(model, apigee_llm) and _is_apigee_chat(model):
        swap_class(model._completions_http_client, _apigee_completions_methods)
        return Backend("openai", CODECS[Api.OPENAI_CHAT], client_headers=True)
    if isinstance(model, Gemini):
        client = model.api_client
        if not model.use_interactions_api:
            swap_class(client.aio.models, _models_methods)
        provider = "vertex_ai" if client.vertexai else "gemini"
        return Backend(provider, CODECS[Api.GEMINI_GENERATE_CONTENT], client_headers=False)
    if (openai_llm := _openai_llm()) is not None and isinstance(model, openai_llm):
        client = model._openai_client
        swap_class(client.chat.completions, _completions_methods)
        return Backend(_openai_provider(client), CODECS[Api.OPENAI_CHAT], client_headers=True)
    if (responses_llm := _openai_responses_llm()) is not None and isinstance(model, responses_llm):
        client = model._openai_client
        swap_class(client.responses, _responses_methods)
        azure = isinstance(model, _azure_openai_responses_llm())
        provider = "azure" if azure else _openai_provider(client)
        return Backend(provider, CODECS[Api.OPENAI_RESPONSES], client_headers=True)
    if (lite_llm := _lite_llm()) is not None and isinstance(model, lite_llm):
        swap_class(model.llm_client, _lite_llm_client_methods)
        return Backend(_lite_llm_provider(model), CODECS[Api.OPENAI_CHAT], client_headers=False)
    if (openai_class := _openai_integration_class(model)) is not None:
        raise TypeError(f"NeMo Relay does not know the provider API of ADK's model class {openai_class}")
    return Backend(type(model).__name__, CODECS[Api.GEMINI_GENERATE_CONTENT], client_headers=False)


def _anthropic_provider(client: AsyncAnthropic | AsyncAnthropicVertex) -> str:
    """Return `vertex_ai` for a Vertex AI client and `azure_ai` for a Microsoft Foundry client.

    Return `anthropic` for any other client, which includes the AWS and Google Cloud gateways of the Anthropic API.
    """
    from anthropic import AsyncAnthropicFoundry, AsyncAnthropicVertex

    if isinstance(client, AsyncAnthropicVertex):
        return "vertex_ai"
    if isinstance(client, AsyncAnthropicFoundry):
        return "azure_ai"
    return "anthropic"


def _openai_provider(client: AsyncOpenAI) -> str:
    """Return `azure` for an Azure OpenAI client and `openai` for any other client.

    Return `bedrock` for a Bedrock client when the installed SDK defines `AsyncBedrockOpenAI`.
    """
    import openai

    if isinstance(client, openai.AsyncAzureOpenAI):
        return "azure"
    bedrock: type[AsyncOpenAI] | None = getattr(openai, "AsyncBedrockOpenAI", None)
    if bedrock is not None and isinstance(client, bedrock):
        return "bedrock"
    return "openai"


def _lite_llm_provider(model: LiteLlm) -> str:
    """Return the provider that LiteLLM routes the model to.

    Resolve the provider from the model name and the `custom_llm_provider` and `api_base` arguments that the call passes
    to LiteLLM.
    """
    import litellm

    args = model._additional_args
    _, provider, _, _ = litellm.get_llm_provider(
        model=model.model, custom_llm_provider=args.get("custom_llm_provider"), api_base=args.get("api_base")
    )
    return provider


def _is_apigee_chat(model: ApigeeLlm) -> bool:
    return model._api_type is type(model).ApiType.CHAT_COMPLETIONS


def anthropic_llm_class() -> type[AnthropicLlm] | None:
    """Return the `AnthropicLlm` class without importing its module, or None when the module is not loaded."""
    module = sys.modules.get("google.adk.models.anthropic_llm")
    return None if module is None else cast("type[AnthropicLlm]", module.AnthropicLlm)


def _openai_llm() -> type[OpenAILlm] | None:
    module = sys.modules.get(f"{_OPENAI_INTEGRATION}._openai_llm")
    return None if module is None else cast("type[OpenAILlm]", module.OpenAILlm)


def _openai_responses_llm() -> type[OpenAIResponsesLlm] | None:
    module = sys.modules.get(f"{_OPENAI_INTEGRATION}._openai_responses_llm")
    return None if module is None else cast("type[OpenAIResponsesLlm]", module.OpenAIResponsesLlm)


def _azure_openai_responses_llm() -> type[AzureOpenAIResponsesLlm]:
    module = sys.modules[f"{_OPENAI_INTEGRATION}._openai_responses_llm"]
    return cast("type[AzureOpenAIResponsesLlm]", module.AzureOpenAIResponsesLlm)


def _lite_llm() -> type[LiteLlm] | None:
    module = sys.modules.get("google.adk.models.lite_llm")
    return None if module is None else cast("type[LiteLlm]", module.LiteLlm)


def _apigee_llm() -> type[ApigeeLlm] | None:
    module = sys.modules.get("google.adk.models.apigee_llm")
    return None if module is None else cast("type[ApigeeLlm]", module.ApigeeLlm)


def _openai_integration_class(model: BaseLlm) -> str | None:
    """Return the qualified name of the first ADK OpenAI integration class in the model's MRO, or None."""
    for cls in type(model).__mro__:
        if cls.__module__ == _OPENAI_INTEGRATION or cls.__module__.startswith(f"{_OPENAI_INTEGRATION}."):
            return f"{cls.__module__}.{cls.__qualname__}"
    return None


def _record_cost(attempt: Attempt, usage: BaseModel | None) -> None:
    """Record on the attempt the `cost` and `cost_usd` usage fields that ADK's `LlmResponse` drops.

    Leave the attempt unchanged when the usage is missing or reports neither field.
    """
    if usage is None:
        return
    dumped = usage.model_dump(mode="json", exclude_none=True)
    reported: JsonObject = {key: dumped[key] for key in _COST_FIELDS if key in dumped}
    if reported:
        attempt.provider_cost = reported


def _sent_headers(held: Mapping[str, str]) -> dict[str, str]:
    """Return the headers that a provider client sends with the attempt's request headers applied.

    An attempt header replaces each held header with the same name in any case. Outside an attempt, return the held
    headers unchanged.
    """
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None:
        return dict(held)
    names = {key.lower() for key in attempt.headers}
    return {**{key: value for key, value in held.items() if key.lower() not in names}, **attempt.headers}


def _with_headers(kwargs: dict[str, object]) -> dict[str, object]:
    """Return the SDK call's keyword arguments with the attempt's request headers merged into `extra_headers`.

    Return the arguments unchanged outside an attempt or when the attempt has no request headers.
    """
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None or not attempt.headers:
        return kwargs
    held = kwargs.get("extra_headers")
    extra = {} if held is None else cast("Mapping[str, str]", held)
    return {**kwargs, "extra_headers": _sent_headers(extra)}


def _models_methods(cls: type[AsyncModels]) -> Mapping[str, object]:
    """Return `AsyncModels` methods that record the response id and URL context on the current attempt.

    ADK's `LlmResponse` drops the `response_id` and the first candidate's `url_context_metadata`.
    """
    return {
        "generate_content": _recorded(cls.generate_content),
        "generate_content_stream": _recorded_stream(cls.generate_content_stream),
    }


def _record(response: types.GenerateContentResponse) -> None:
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None:
        return
    if response.response_id is not None:
        attempt.response_id = response.response_id
    candidate = next((candidate for candidate in response.candidates or [] if not candidate.index), None)
    if candidate is not None and candidate.url_context_metadata is not None:
        attempt.urls.append(candidate.url_context_metadata)


def _recorded(
    generate: Callable[_P, Awaitable[types.GenerateContentResponse]],
) -> Callable[_P, Awaitable[types.GenerateContentResponse]]:
    async def generate_content(*args: _P.args, **kwargs: _P.kwargs) -> types.GenerateContentResponse:
        response = await generate(*args, **kwargs)
        _record(response)
        return response

    return generate_content


def _recorded_stream(
    generate: Callable[_P, Awaitable[AsyncIterator[types.GenerateContentResponse]]],
) -> Callable[_P, Awaitable[AsyncIterator[types.GenerateContentResponse]]]:
    async def generate_content_stream(
        *args: _P.args, **kwargs: _P.kwargs
    ) -> AsyncIterator[types.GenerateContentResponse]:
        return _recording(cast("_GeminiStream", await generate(*args, **kwargs)))

    return generate_content_stream


async def _recording(responses: _GeminiStream) -> _GeminiStream:
    async with contextlib.aclosing(responses) as stream:
        async for response in stream:
            _record(response)
            yield response


class _Create(Protocol):
    def __call__(self, resource: AsyncMessages, /, **kwargs: object) -> Awaitable[Message | _AnthropicStream]: ...


def _messages_methods(cls: type[AsyncMessages]) -> Mapping[str, object]:
    """Return `AsyncMessages` methods that send the attempt's request headers.

    The methods also record on the attempt the message id and cost that ADK's `LlmResponse` drops.
    """
    from anthropic.types import Message

    original = cast("_Create", cls.create)

    async def create(self: AsyncMessages, **kwargs: object) -> Message | AsyncIterator[RawMessageStreamEvent]:
        result = await original(self, **_with_headers(kwargs))
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return result
        if isinstance(result, Message):
            attempt.response_id = result.id
            _record_cost(attempt, result.usage)
            return result
        return _recorded_messages(result, attempt)

    return {"create": create}


async def _recorded_messages(stream: _AnthropicStream, attempt: Attempt) -> AsyncIterator[RawMessageStreamEvent]:
    from anthropic.types import RawMessageDeltaEvent, RawMessageStartEvent

    try:
        async for event in stream:
            if isinstance(event, RawMessageStartEvent):
                attempt.response_id = event.message.id
                _record_cost(attempt, event.message.usage)
            elif isinstance(event, RawMessageDeltaEvent):
                _record_cost(attempt, event.usage)
            yield event
    finally:
        await stream.close()


class _Complete(Protocol):
    def __call__(
        self, resource: AsyncCompletions, /, **kwargs: object
    ) -> Awaitable[ChatCompletion | _OpenAIStream]: ...


def _completions_methods(cls: type[AsyncCompletions]) -> Mapping[str, object]:
    """Return Chat Completions methods that send the attempt's request headers.

    The methods also record on the attempt the completion id, model, and cost that ADK's `LlmResponse` drops.
    """
    from openai.types.chat import ChatCompletion

    original = cast("_Complete", cls.create)

    async def create(self: AsyncCompletions, **kwargs: object) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        result = await original(self, **_with_headers(kwargs))
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return result
        if isinstance(result, ChatCompletion):
            attempt.response_id = result.id
            attempt.response_model = result.model
            _record_cost(attempt, result.usage)
            return result
        return _recorded_completions(result, attempt)

    return {"create": create}


async def _recorded_completions(stream: _OpenAIStream, attempt: Attempt) -> AsyncIterator[ChatCompletionChunk]:
    try:
        async for chunk in stream:
            attempt.response_id = chunk.id
            attempt.response_model = chunk.model
            _record_cost(attempt, chunk.usage)
            yield chunk
    finally:
        await stream.close()


class _Respond(Protocol):
    def __call__(
        self, resource: AsyncResponses, /, **kwargs: object
    ) -> Awaitable[Response | OpenAIStream[ResponseStreamEvent]]: ...


def _responses_methods(cls: type[AsyncResponses]) -> Mapping[str, object]:
    """Return `AsyncResponses` methods that send the attempt's request headers.

    The methods also record on the attempt the cost that ADK's `LlmResponse` drops.
    """
    from openai.types.responses import Response

    original = cast("_Respond", cls.create)

    async def create(self: AsyncResponses, **kwargs: object) -> Response | AsyncIterator[ResponseStreamEvent]:
        result = await original(self, **_with_headers(kwargs))
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return result
        if isinstance(result, Response):
            _record_cost(attempt, result.usage)
            return result
        return _recorded_responses(result, attempt)

    return {"create": create}


async def _recorded_responses(
    stream: OpenAIStream[ResponseStreamEvent], attempt: Attempt
) -> AsyncIterator[ResponseStreamEvent]:
    from openai.types.responses import ResponseCompletedEvent

    try:
        async for event in stream:
            if isinstance(event, ResponseCompletedEvent):
                _record_cost(attempt, event.response.usage)
            yield event
    finally:
        await stream.close()


class _Post(Protocol):
    def __call__(
        self,
        client: CompletionsHTTPClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        /,
        *,
        timeout: float | None,
    ) -> Awaitable[httpx.Response]: ...


class _Stream(Protocol):
    def __call__(
        self,
        client: CompletionsHTTPClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        /,
        *,
        timeout: float | None,
    ) -> AsyncGenerator[LlmResponse, None]: ...


def _apigee_completions_methods(cls: type[CompletionsHTTPClient]) -> Mapping[str, object]:
    """Return Apigee `CompletionsHTTPClient` methods that send the attempt's request headers.

    Each attempt header replaces the proxy's header of the same name.
    """
    post = cast("_Post", cls._httpx_post_with_retry)
    stream = cast("_Stream", cls._handle_streaming)

    async def _httpx_post_with_retry(
        self: CompletionsHTTPClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        timeout: float | None,
    ) -> httpx.Response:
        return await post(self, url, payload, _sent_headers(headers), timeout=timeout)

    def _handle_streaming(
        self: CompletionsHTTPClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        timeout: float | None,
    ) -> AsyncGenerator[LlmResponse, None]:
        return stream(self, url, payload, _sent_headers(headers), timeout=timeout)

    return {"_httpx_post_with_retry": _httpx_post_with_retry, "_handle_streaming": _handle_streaming}


class _Acompletion(Protocol):
    def __call__(
        self, client: LiteLLMClient, /, model: object, messages: object, tools: object, **kwargs: object
    ) -> Awaitable[ModelResponse | CustomStreamWrapper]: ...


def _lite_llm_client_methods(cls: type[LiteLLMClient]) -> Mapping[str, object]:
    """Return `LiteLLMClient` methods that record the response id and cost on the current attempt.

    ADK's `LlmResponse` drops both fields.
    """
    from litellm.types.utils import ModelResponse

    original = cast("_Acompletion", cls.acompletion)

    async def acompletion(
        self: LiteLLMClient, model: object, messages: object, tools: object, **kwargs: object
    ) -> ModelResponse | AsyncIterator[ModelResponseStream]:
        result = await original(self, model=model, messages=messages, tools=tools, **kwargs)
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return result
        if isinstance(result, ModelResponse):
            attempt.response_id = result.id
            _record_cost(attempt, cast("BaseModel | None", result.get("usage")))
            return result
        return _recorded_chunks(result, attempt)

    return {"acompletion": acompletion}


async def _recorded_chunks(stream: CustomStreamWrapper, attempt: Attempt) -> AsyncIterator[ModelResponseStream]:
    try:
        async for chunk in stream:
            attempt.response_id = chunk.id
            _record_cost(attempt, cast("BaseModel | None", chunk.get("usage")))
            yield chunk
    finally:
        await stream.aclose()


def anthropic_kwargs_methods(cls: type[AnthropicLlm]) -> Mapping[str, object]:
    """Return `AnthropicLlm` methods that apply the attempt's tool choice edit to the Anthropic request.

    ADK builds that request without the function calling config.
    """
    original = cls._build_anthropic_kwargs

    def _build_anthropic_kwargs(
        self: AnthropicLlm,
        llm_request: LlmRequest,
        messages: list[MessageParam],
        tools: Iterable[ToolUnionParam] | NotGiven,
        tool_choice: ToolChoiceParam | NotGiven,
        thinking: _Thinking,
    ) -> dict[str, Any]:
        kwargs = original(self, llm_request, messages, tools, tool_choice, thinking)
        attempt = CURRENT_ATTEMPT.get()
        if attempt is not None and attempt.tool_choice is not None:
            kwargs["tools"], kwargs["tool_choice"] = anthropic_tool_choice(
                attempt.tool_choice, kwargs["tools"], kwargs["tool_choice"]
            )
        return kwargs

    return {"_build_anthropic_kwargs": _build_anthropic_kwargs}


def anthropic_tool_choice(
    config: types.FunctionCallingConfig,
    tools: Iterable[ToolUnionParam] | NotGiven,
    choice: ToolChoiceParam | NotGiven,
) -> tuple[list[ToolUnionParam] | NotGiven, ToolChoiceParam | NotGiven]:
    """Return the Anthropic tools and tool choice for a function calling config.

    Keep only the tools that `allowed_function_names` lists. Force the named tool when the mode is `ANY` with one
    allowed name, and keep the request's own tool choice when the mode is unset. Return `NOT_GIVEN` for both when no
    tools remain. Raise `ValueError` for an undeclared allowed name, for a mode with no Anthropic equivalent, and for a
    mode that requires a tool call when no tools remain.
    """
    from anthropic import NOT_GIVEN, NotGiven

    declared: list[ToolUnionParam] = [] if isinstance(tools, NotGiven) else list(tools)
    names = config.allowed_function_names or []
    if names:
        known = {tool["name"] for tool in declared}
        if missing := [name for name in names if name not in known]:
            raise ValueError(f"ADK allowed_function_names {missing} are not declared on the request")
        declared = [tool for tool in declared if tool["name"] in names]
    mode = config.mode
    forced: ToolChoiceParam | None = None
    if mode is not None and mode is not types.FunctionCallingConfigMode.MODE_UNSPECIFIED:
        if mode not in _FORCED:
            raise ValueError(f"Anthropic has no equivalent of the {mode.value} function calling mode")
        forced = {"type": "tool", "name": names[0]} if mode is _ANY and len(names) == 1 else _FORCED[mode]
    if not declared:
        if forced is not None and forced["type"] == "any":
            raise ValueError("the function calling config requires a tool call but the request declares no tools")
        return NOT_GIVEN, NOT_GIVEN
    return declared, choice if forced is None else forced
