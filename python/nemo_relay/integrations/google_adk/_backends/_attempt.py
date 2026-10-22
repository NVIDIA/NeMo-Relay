# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model call attempt, carried between managed pipeline and provider client."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final, NamedTuple, TypeVar, cast

from nemo_relay.integrations.google_adk._serialization import RECORDED_USAGE_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from google.genai import types
    from pydantic import BaseModel

    from nemo_relay import JsonObject
    from nemo_relay.integrations.google_adk._serialization import AdkCodec

_V = TypeVar("_V")


@dataclass(slots=True, kw_only=True)
class Attempt:
    """Model call attempt: edits for provider client, and what client reports back.

    Edits: intercept's tool choice edit and request headers.
    """

    tool_choice: Annotated[
        types.FunctionCallingConfig | None,
        "Intercept's function calling config, or None when ADK's config is unchanged.",
    ]
    headers: Annotated[Mapping[str, str], "Request headers provider client sends with attempt."]
    recorded: Annotated[
        JsonObject, "Fields provider client reports and `LlmResponse` drops, carried under `RECORDED_KEY`."
    ] = field(default_factory=dict)
    response_model: Annotated[str | None, "Model named by latest response or chunk."] = None


CURRENT_ATTEMPT: Final[ContextVar[Attempt | None]] = ContextVar("nemo_relay_google_adk_attempt", default=None)


class Backend(NamedTuple):
    provider: Annotated[str, "LLM scope name: provider name, or class name for unknown model."]
    codec: AdkCodec
    client_headers: Annotated[
        bool,
        "True when request headers travel as client extra headers, not ADK's HTTP options.",
    ]


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
