# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM backend for ADK's `LiteLlm` model. Swaps its client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import signature
from types import MethodType
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias, Unpack, cast, get_args

import litellm
from google.genai import types
from litellm.exceptions import BadRequestError
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.litellm_agent.litellm_agent_model_resolver import LITELLM_AGENT_PREFIX
from litellm.integrations.prompt_management_base import PromptManagementBase
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.get_litellm_params import OPTIONAL_KWARGS_KEYS
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params
from litellm.litellm_core_utils.litellm_logging import _init_custom_logger_compatible_class
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import DynamicPromptManagementParamLiteral, ModelResponse, Usage
from pydantic import ValidationError
from typing_extensions import ReadOnly, TypedDict

from nemo_relay.codecs import OpenAIChatCodec
from nemo_relay.integrations.google_adk._backends._attempt import (
    CURRENT_ATTEMPT,
    Attempt,
    Backend,
    ToolChoiceApi,
    provider_tool_choice,
    record_usage,
)
from nemo_relay.integrations.google_adk._serialization import (
    BASE_USAGE,
    BASE_USAGE_KEYS,
    CACHE_WRITE_KEY,
    RECORDED_ID_KEY,
    RECORDED_USAGE_KEY,
    AdkCodec,
    Normalized,
    object_field,
    reported_cost,
    token_counts,
    without_cache,
)
from nemo_relay.integrations.google_adk._swap import Methods, swap_class

if TYPE_CHECKING:
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
    from litellm import CustomStreamWrapper
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.utils import ModelResponseStream, StandardCallbackDynamicParams

    from nemo_relay import JsonObject

_Choice: TypeAlias = Literal["required", "none"]
_ToolChoice: TypeAlias = "str | JsonObject"
_PROVIDER_ARGS: Final[
    Annotated[frozenset[str], "Arguments LiteLLM passes along when it looks up a call's provider."]
] = OPTIONAL_KWARGS_KEYS.difference(signature(litellm.completion).parameters)
_API: Final = "openai_chat"
_CACHE_WRITE: Final = ((CACHE_WRITE_KEY, "cache_write_tokens"),)
_ADK_READ_KEYS: Final[Annotated[tuple[str, ...], "Top-level LiteLLM usage keys where ADK looks for cached tokens."]] = (
    "cached_prompt_tokens",
    "cached_tokens",
    "cache_read_input_tokens",
)
_ADK_WRITE_KEYS: Final[Annotated[tuple[str, ...], "LiteLLM usage keys where ADK looks for cache writes."]] = (
    "cache_creation_input_tokens",
    "cache_write_input_tokens",
)
_USAGE_KEYS: Final = BASE_USAGE_KEYS | {CACHE_WRITE_KEY}
_RECORDED_KEYS: Final = frozenset({RECORDED_USAGE_KEY})
_NATIVE_CODEC: Final = OpenAIChatCodec()
_CHOICES: Final[
    Annotated[Mapping[types.FunctionCallingConfigMode, _Choice | None], "Tool choice per mode. None sends no choice."]
] = {
    types.FunctionCallingConfigMode.AUTO: None,
    types.FunctionCallingConfigMode.ANY: "required",
    types.FunctionCallingConfigMode.NONE: "none",
}


def _function_name(tool: JsonObject) -> str | None:
    """Return function name of `tool`, or None for a built-in tool."""
    function = tool.get("function")
    return cast(str, function["name"]) if isinstance(function, dict) else None


def _named_function(name: str) -> JsonObject:
    return {"type": "function", "function": {"name": name}}


_TOOL_CHOICE: Final[ToolChoiceApi[JsonObject, _ToolChoice | None]] = ToolChoiceApi(
    name="Chat Completions", choices=_CHOICES, named=_named_function, tool_name=_function_name
)
_Callbacks: TypeAlias = list[CustomLogger | str | Callable[..., object]]
_CALLBACK_ARGS: Final[Annotated[frozenset[str], "Callback arguments LiteLLM reads before it loads a credential."]] = (
    frozenset({"callbacks", "success_callback"})
)
_DYNAMIC_PROMPT_ARGS: Final[
    Annotated[tuple[str, ...], "Call arguments that make LiteLLM run prompt management without `prompt_id`."]
] = tuple(DynamicPromptManagementParamLiteral.list_all_params())
_INTEGRATIONS: Final = get_args(litellm._custom_logger_compatible_callbacks_literal)


class _LiteLlmArgs(TypedDict, total=False):
    """`LiteLlm` arguments Relay reads or edits. LiteLLM accepts many more."""

    litellm_credential_name: ReadOnly[str]
    fallbacks: ReadOnly[list[str | dict[str, object]]]
    model_list: ReadOnly[list[dict[str, object]]]
    context_window_fallback_dict: ReadOnly[dict[str, str]]
    prompt_id: ReadOnly[str]
    enable_prompt_caching: ReadOnly[bool]
    api_base: ReadOnly[str]
    base_url: ReadOnly[str]
    custom_llm_provider: ReadOnly[str]
    azure: ReadOnly[bool]
    deployment_id: ReadOnly[str]
    api_key: ReadOnly[str]
    callbacks: ReadOnly[_Callbacks]
    success_callback: ReadOnly[_Callbacks]
    tool_choice: _ToolChoice


def _read(response: JsonObject, recorded: JsonObject, usage_metadata: JsonObject) -> Normalized:
    """Read token usage from ADK's response.

    Fill cache writes from recorded LiteLLM usage when ADK dropped them. Drop ADK's 0 cached tokens when LiteLLM
    reports none.

    Raises:
        ValueError: Prompt count is below cache counts.
        TypeError: A recorded LiteLLM usage field has the wrong type.
    """
    usage = token_counts(usage_metadata, BASE_USAGE) | token_counts(usage_metadata, _CACHE_WRITE)
    if (reported := recorded.get(RECORDED_USAGE_KEY)) is not None:
        litellm_usage = object_field(reported, field="LiteLLM usage")
        details = object_field(litellm_usage.get("prompt_tokens_details"), field="LiteLLM usage prompt_tokens_details")
        if usage.get("cache_read_tokens") == 0 and not (
            _carries(details, ("cached_tokens",)) or _carries(litellm_usage, _ADK_READ_KEYS)
        ):
            del usage["cache_read_tokens"]
        if (
            "cache_write_tokens" not in usage
            and not _carries(litellm_usage, _ADK_WRITE_KEYS)
            and (write := details.get("cache_write_tokens")) is not None
        ):
            if not isinstance(write, int):
                raise TypeError("LiteLLM usage prompt_tokens_details cache_write_tokens must be an integer")
            usage["cache_write_tokens"] = write
    read, write = usage.get("cache_read_tokens"), usage.get("cache_write_tokens")
    if (prompt := usage.get("prompt_tokens")) is not None and (read is not None or write is not None):
        usage["uncached_input_tokens"] = without_cache(prompt, read, write, folded="the cache tokens LiteLLM reports")
    return Normalized(
        usage=usage,
        cost=reported_cost(recorded, codec=_NATIVE_CODEC),
        api_specific={"api": _API},
        usage_keys=_USAGE_KEYS,
        response_keys=frozenset(),
        recorded_keys=_RECORDED_KEYS,
    )


def _carries(fields: JsonObject, keys: Sequence[str]) -> bool:
    """Return whether ADK reads a count from any of `keys` in `fields`."""
    return any(isinstance(fields.get(key), int) for key in keys)


_LITE_LLM_CODEC: Final = AdkCodec(_read)


async def lite_llm_backend(model: LiteLlm, request_model: str | None) -> Backend:
    """Return backend for a `LiteLlm` model and swap its client.

    Returns:
        Backend whose provider is LiteLLM's pick for `request_model`, else for `model.model`, or class name when
        LiteLLM finds none.

    Raises:
        TypeError: LiteLLM settings can switch provider mid-call.
    """
    call = _call(request_model or model.model, cast(_LiteLlmArgs, model._additional_args))
    _require_fixed_route(call)
    swap_class(model.llm_client, _client_methods)
    route = await asyncio.to_thread(_route, call)
    provider = type(model).__name__ if route is None else route
    return Backend(provider, _LITE_LLM_CODEC, client_headers=False, model=call.model)


@dataclass(frozen=True, slots=True)
class _Call:
    """What LiteLLM reads from one call's arguments."""

    model: str
    options: Annotated[_LiteLlmArgs, "Arguments, with the named LiteLLM credential filling missing ones."]
    dynamic: Annotated[
        StandardCallbackDynamicParams, "Callback parameters, which LiteLLM reads before it loads the credential."
    ]


def _call(model: str, args: _LiteLlmArgs) -> _Call:
    return _Call(model, _with_credential(args), initialize_standard_callback_dynamic_params(dict(args)))


def _with_credential(args: _LiteLlmArgs) -> _LiteLlmArgs:
    """Return `args` plus missing values from its named LiteLLM credential.

    Skip credential callbacks: LiteLLM reads callbacks before loading a credential.
    """
    if credential := args.get("litellm_credential_name"):
        values = CredentialAccessor.get_credential_values(credential)
        kept = {key: value for key, value in values.items() if key not in _CALLBACK_ARGS}
        return {**kept, **args}
    return args


def _require_fixed_route(call: _Call) -> None:
    """Raise `TypeError` when LiteLLM can move `call` to another provider while it runs.

    Prompt management that a dynamic argument opens is checked by `_dynamic_prompt_integration`, once the call's
    messages and tools are known.
    """
    options = call.options
    if (options.get("fallbacks") or litellm.model_fallbacks) is not None:
        raise _moving_route("LiteLLM fallbacks")
    if options.get("model_list") is not None:
        raise _moving_route("LiteLLM model_list")
    if call.model in (options.get("context_window_fallback_dict") or {}):
        raise _moving_route("LiteLLM context_window_fallback_dict")
    if call.model.startswith(LITELLM_AGENT_PREFIX):
        raise _moving_route(f"LiteLLM agent model {call.model!r}")
    if (callback := _moving_callback(options)) is not None:
        raise _moving_route(f"LiteLLM callback {type(callback).__name__}")
    if options.get("prompt_id") and (integration := _prompt_integration(call)) is not None:
        raise _moving_route(integration)


def _moving_route(feature: str) -> TypeError:
    return TypeError(
        f"NeMo Relay does not support {feature} on ADK's LiteLlm, which can send a call to another provider"
    )


def _moving_callback(args: _LiteLlmArgs) -> CustomLogger | None:
    """Return a LiteLLM callback whose deployment or agentic loop hook can change a call or its response.

    Check each hook only in the callback lists LiteLLM scans for it.

    Returns:
        First such callback, or None.
    """
    for callback in litellm.callbacks:
        if isinstance(callback, CustomLogger) and _changes_calls(callback):
            return callback
    for callback in [*litellm.callbacks, *(args.get("success_callback") or [])]:
        if isinstance(callback, CustomLogger) and _gates_agentic_loop(callback):
            return callback
    return None


def _prompt_integration(call: _Call) -> str | None:
    """Return the prompt integration LiteLLM hands `call` to once its prompt management gate is open, or None."""
    prompt_id = call.options.get("prompt_id") or None
    if (found := _prefix_logger(call.model)) is not None:
        integration, logger = found
        return f"LiteLLM prompt integration {integration}" if _compiles(logger, prompt_id, call.dynamic) else None
    managers = _prompt_managers(call.options)
    selected = next((manager for manager in managers if _accepts(manager, prompt_id, call.dynamic)), None)
    if selected is None and prompt_id is not None and managers:
        selected = managers[0] if _rewrites_prompt(managers[0], CustomPromptManagement) else None
    return None if selected is None else f"LiteLLM callback {type(selected).__name__}"


def _prompt_managers(options: _LiteLlmArgs) -> list[CustomPromptManagement]:
    registered = [
        *litellm.callbacks,
        *litellm.success_callback,
        *(options.get("callbacks") or []),
        *litellm.failure_callback,
        *litellm._async_success_callback,
        *litellm._async_failure_callback,
    ]
    managers: list[CustomPromptManagement] = []
    for callback in registered:
        if isinstance(callback, CustomPromptManagement) and callback not in managers:
            managers.append(callback)
    return managers


def _rewrites_prompt(logger: CustomLogger, base: type[CustomLogger]) -> bool:
    cls = type(logger)
    return (
        cls.get_chat_completion_prompt is not base.get_chat_completion_prompt
        or cls.async_get_chat_completion_prompt is not base.async_get_chat_completion_prompt
    )


def _prefix_logger(name: str) -> tuple[str, CustomLogger] | None:
    for integration in _INTEGRATIONS:
        if name.startswith(integration):
            logger = _init_custom_logger_compatible_class(integration, internal_usage_cache=None, llm_router=None)
            if logger is not None:
                return integration, logger
    return None


def _compiles(logger: CustomLogger, prompt_id: str | None, params: StandardCallbackDynamicParams) -> bool:
    if isinstance(logger, PromptManagementBase):
        return prompt_id is not None and _accepts(logger, prompt_id, params)
    return _rewrites_prompt(logger, CustomLogger)


def _accepts(manager: PromptManagementBase, prompt_id: str | None, params: StandardCallbackDynamicParams) -> bool:
    """Return whether `manager` runs for `prompt_id`. False when its check raises, as in LiteLLM."""
    try:
        return manager.should_run_prompt_management(
            prompt_id=prompt_id, prompt_spec=None, dynamic_callback_params=params
        )
    except Exception:
        return False


def _dynamic_prompt_integration(
    call: _Call, messages: list[AllMessageValues], tools: list[JsonObject] | None
) -> str | None:
    """Return the prompt integration LiteLLM hands a call without `prompt_id` to, or None.

    LiteLLM opens its prompt management gate for such a call when, after it seeds Anthropic cache control points,
    the call carries a dynamic prompt management argument or names a registered vector store.
    """
    if call.options.get("prompt_id"):
        return None
    params: dict[str, object] = dict(call.options)
    AnthropicCacheControlHook.maybe_seed_default_injection_points(
        non_default_params=params,
        messages=messages,
        model=call.model,
        custom_llm_provider=call.options.get("custom_llm_provider"),
        tools=tools,
        enable_prompt_caching=call.options.get("enable_prompt_caching"),
        api_base=call.options.get("api_base") or call.options.get("base_url"),
    )
    registry = litellm.vector_store_registry
    if any(params.get(argument) for argument in _DYNAMIC_PROMPT_ARGS) or (
        registry is not None and registry.get_vector_store_to_run(non_default_params=params, tools=tools) is not None
    ):
        return _prompt_integration(call)
    return None


def _changes_calls(callback: CustomLogger) -> bool:
    """Return whether `callback` overrides a LiteLLM deployment hook.

    Deployment hooks can change a call or its response.
    """
    hooks = (
        (callback.async_pre_call_deployment_hook, CustomLogger.async_pre_call_deployment_hook),
        (callback.async_post_call_success_deployment_hook, CustomLogger.async_post_call_success_deployment_hook),
        (callback.async_post_call_streaming_deployment_hook, CustomLogger.async_post_call_streaming_deployment_hook),
    )
    return any(_function(hook) is not base for hook, base in hooks)


def _gates_agentic_loop(callback: CustomLogger) -> bool:
    """Return whether `callback` can start a LiteLLM agentic loop."""
    return (
        _function(type(callback).async_should_run_agentic_loop) is not CustomLogger.async_should_run_agentic_loop
        or _function(callback.async_should_run_chat_completion_agentic_loop)
        is not CustomLogger.async_should_run_chat_completion_agentic_loop
    )


def _function(value: object) -> object:
    """Return function behind a bound method, else `value`."""
    return value.__func__ if isinstance(value, MethodType) else value


def _route(call: _Call) -> str | None:
    """Return provider LiteLLM picks for `call`, or None.

    Apply routing settings in LiteLLM's order.
    """
    args = call.options
    name = call.model
    api_base = args.get("api_base")
    base_url = args.get("base_url")
    custom_llm_provider = args.get("custom_llm_provider")
    provider_args = {key: value for key, value in args.items() if key in _PROVIDER_ARGS}
    try:
        if custom_llm_provider is None:
            litellm.get_llm_provider(model=name, api_base=api_base or base_url)
        name = litellm.model_alias_map.get(name, name)
        if args.get("azure") is True:
            custom_llm_provider = "azure"
        if (deployment_id := args.get("deployment_id")) is not None:
            name = deployment_id
            custom_llm_provider = "azure"
        _, provider, _, _ = litellm.get_llm_provider(
            model=name,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base if base_url is None else base_url,
            api_key=args.get("api_key"),
            litellm_params=GenericLiteLLMParams.model_validate(provider_args) if provider_args else None,
        )
    except (BadRequestError, ValidationError):
        return None
    return provider


def _client_methods(cls: type[LiteLLMClient]) -> Methods[LiteLLMClient]:
    """Return `LiteLLMClient` methods that apply current attempt to each call.

    ADK sends LiteLLM only tool choice mode. Methods also record response id and usage on attempt. `LlmResponse`
    drops id and usage cost.
    """
    original = cls.acompletion

    async def acompletion(
        self: LiteLLMClient,
        model: str,
        messages: list[AllMessageValues],
        tools: list[JsonObject] | None,
        **kwargs: Unpack[_LiteLlmArgs],
    ) -> ModelResponse | AsyncIterator[ModelResponseStream]:
        attempt = CURRENT_ATTEMPT.get()
        if attempt is None:
            return await original(self, model=model, messages=messages, tools=tools, **kwargs)
        call = _call(model, kwargs)
        _require_fixed_route(call)
        route = await asyncio.to_thread(_route, call)
        if route is not None and route != attempt.provider:
            raise ValueError(f"LiteLLM routes model {model!r} to {route}, not to recorded provider {attempt.provider}")
        if attempt.tool_choice is not None:
            chosen = provider_tool_choice(
                _TOOL_CHOICE, attempt.tool_choice, tools or [], kwargs.pop("tool_choice", None)
            )
            tools, choice = (None, None) if chosen is None else chosen
            if choice is not None:
                kwargs["tool_choice"] = choice
        if (integration := _dynamic_prompt_integration(call, messages, tools)) is not None:
            raise _moving_route(integration)
        result = await original(self, model=model, messages=messages, tools=tools, **kwargs)
        if isinstance(result, ModelResponse):
            attempt.recorded[RECORDED_ID_KEY] = result.id
            if (usage := _usage(result)) is not None:
                record_usage(attempt, usage.model_copy(update={"cost": _cost(usage)}))
            return result
        recorded = _recorded_completions(result, attempt)
        attempt.streams.append(recorded)
        return recorded

    return {"acompletion": acompletion}


async def _recorded_completions(
    stream: CustomStreamWrapper, attempt: Attempt
) -> AsyncGenerator[ModelResponseStream, None]:
    try:
        async for chunk in stream:
            if chunk.id:
                attempt.recorded.setdefault(RECORDED_ID_KEY, chunk.id)
            if (usage := _usage(chunk)) is not None:
                record_usage(attempt, usage.model_copy(update={"cost": _provider_cost(stream)}))
            yield chunk
    finally:
        await asyncio.shield(stream.aclose())


def _provider_cost(stream: CustomStreamWrapper) -> float | None:
    """Return cost from last usage chunk in `stream` that reports one, or None."""
    for chunk in reversed(cast("list[ModelResponseStream]", stream.chunks)):
        if (usage := _usage(chunk)) is not None and usage.get("cost") is not None:
            return _cost(usage)
    return None


def _cost(usage: Usage) -> float | None:
    """Return `usage` cost, `total_cost` of a breakdown, or None."""
    cost: object = usage.get("cost")
    while isinstance(cost, dict):
        cost = cost.get("total_cost")
    return float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else None


def _usage(response: ModelResponse | ModelResponseStream) -> Usage | None:
    usage: object = response.get("usage")
    return usage if isinstance(usage, Usage) else None
