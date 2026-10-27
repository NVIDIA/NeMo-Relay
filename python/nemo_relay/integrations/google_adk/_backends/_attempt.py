# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model call attempt, carried between managed pipeline and provider client."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final, Generic, NamedTuple, TypeVar, cast

from google.genai import types

from nemo_relay.integrations.google_adk._serialization import RECORDED_USAGE_KEY

if TYPE_CHECKING:
    from pydantic import BaseModel

    from nemo_relay import JsonObject
    from nemo_relay.integrations.google_adk._serialization import AdkCodec

_V = TypeVar("_V")
_ToolT = TypeVar("_ToolT")
_ChoiceT = TypeVar("_ChoiceT")


@dataclass(slots=True, kw_only=True)
class Attempt:
    """One attempt at a model call."""

    tool_choice: Annotated[
        types.FunctionCallingConfig | None,
        "Intercept's function calling config, or None when ADK's config is unchanged.",
    ]
    headers: Annotated[Mapping[str, str], "Request headers provider client sends with attempt."]
    provider: Annotated[str, "Provider name of the attempt's LLM scope."]
    recorded: Annotated[
        JsonObject, "Fields provider client reports and `LlmResponse` drops, carried under `RECORDED_KEY`."
    ] = field(default_factory=dict)
    response_model: Annotated[str | None, "Model named by latest response or chunk."] = None
    streams: Annotated[list[AsyncGenerator[object, None]], "Provider streams handed to ADK, closed with attempt."] = (
        field(default_factory=list)
    )


CURRENT_ATTEMPT: Final[ContextVar[Attempt | None]] = ContextVar("nemo_relay_google_adk_attempt", default=None)


class Backend(NamedTuple):
    provider: Annotated[str, "LLM scope name: provider name, or class name for unknown model."]
    codec: AdkCodec
    client_headers: Annotated[
        bool,
        "True when request headers travel as client extra headers, not ADK's HTTP options.",
    ]
    model: Annotated[str | None, "LLM scope model name: model the class calls, or model's own name for unknown model."]


def usage_fields(usage: BaseModel) -> JsonObject:
    return usage.model_dump(mode="json", exclude_none=True)


def record_usage(attempt: Attempt, usage: BaseModel | None) -> None:
    """Record provider usage on attempt. No change when `usage` is None."""
    if usage is not None:
        attempt.recorded[RECORDED_USAGE_KEY] = usage_fields(usage)


def sent_headers(held: Mapping[str, _V]) -> dict[str, _V | str]:
    """Return held headers with current attempt's request headers applied.

    Attempt header replaces held header of same name, ignoring case. Outside an attempt, return held headers unchanged.
    """
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None:
        return dict[str, _V | str](held)
    names = {key.lower() for key in attempt.headers}
    return {**{key: value for key, value in held.items() if key.lower() not in names}, **attempt.headers}


def with_extra_headers(kwargs: dict[str, object]) -> dict[str, object]:
    """Return SDK call kwargs with current attempt's request headers merged into `extra_headers`.

    Unchanged outside an attempt or when attempt has no request headers.
    """
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None or not attempt.headers:
        return kwargs
    held = kwargs.get("extra_headers")
    extra: Mapping[str, object] = {} if held is None else cast("Mapping[str, object]", held)
    return {**kwargs, "extra_headers": sent_headers(extra)}


@dataclass(frozen=True, slots=True)
class ToolChoiceApi(Generic[_ToolT, _ChoiceT]):
    """How a provider API writes tools and tool choice."""

    name: Annotated[str, "API name shown in errors, such as `Anthropic`."]
    choices: Annotated[
        Mapping[types.FunctionCallingConfigMode, _ChoiceT], "Tool choice for each mode the API can express."
    ]
    named: Annotated[Callable[[str], _ChoiceT], "Tool choice that forces one named function."]
    tool_name: Annotated[Callable[[_ToolT], str | None], "Function name of a tool, or None for a built-in tool."]


def provider_tool_choice(
    api: ToolChoiceApi[_ToolT, _ChoiceT],
    config: types.FunctionCallingConfig,
    tools: Sequence[_ToolT],
    choice: _ChoiceT,
) -> tuple[list[_ToolT], _ChoiceT] | None:
    """Return provider tools and tool choice for a function calling config.

    Keep built-in tools and functions in `allowed_function_names`. Force that function when mode is `ANY` with one
    allowed name. Keep `choice` when mode is unset or `MODE_UNSPECIFIED`.

    Args:
        api: How the provider API writes tools and tool choice.
        config: ADK function calling config to apply.
        tools: Tools the request declares.
        choice: Tool choice the request already carries.

    Returns:
        Remaining tools and tool choice, or None when no tools remain.

    Raises:
        ValueError: Allowed name not declared, mode the API can't express, or `ANY` with no tools left.
    """
    declared = list(tools)
    names = config.allowed_function_names or []
    if names:
        known = {api.tool_name(tool) for tool in declared}
        if missing := [name for name in names if name not in known]:
            raise ValueError(f"ADK allowed_function_names {missing} are not declared on the request")
        declared = [tool for tool in declared if (name := api.tool_name(tool)) is None or name in names]
    mode = config.mode
    if mode is not None and mode is not types.FunctionCallingConfigMode.MODE_UNSPECIFIED:
        if mode not in api.choices:
            raise ValueError(f"{api.name} has no equivalent of the {mode.value} function calling mode")
        forced = mode is types.FunctionCallingConfigMode.ANY and len(names) == 1
        choice = api.named(names[0]) if forced else api.choices[mode]
    if not declared:
        if mode is types.FunctionCallingConfigMode.ANY:
            raise ValueError("the function calling config requires a tool call but the request declares no tools")
        return None
    return declared, choice
