# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One model call attempt, carried between the managed pipeline and the provider client."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final, NamedTuple, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from google.genai import types
    from pydantic import BaseModel

    from nemo_relay import Json, JsonObject
    from nemo_relay.integrations.google_adk._serialization import AdkCodec

_V = TypeVar("_V")
_COST_FIELDS: Final[Annotated[tuple[str, ...], "Usage fields Relay's provider codecs read cost from."]] = (
    "cost",
    "cost_usd",
)


@dataclass(slots=True, kw_only=True)
class Attempt:
    """One model call attempt: edits bound for the provider client, and what the client reports back.

    Edits are an intercept's tool choice edit and the request headers.
    """

    tool_choice: Annotated[
        types.FunctionCallingConfig | None,
        "Intercept's function calling config, or None when ADK's config is unchanged.",
    ]
    headers: Annotated[Mapping[str, str], "Request headers the provider client sends with the attempt."]
    urls: Annotated[
        list[types.UrlContextMetadata],
        "URL context metadata reported by the first candidate of each Gemini response.",
    ] = field(default_factory=list)
    response_id: str | None = None
    response_model: Annotated[str | None, "Model named by the latest Chat Completions response or chunk."] = None
    provider_cost: Annotated[
        JsonObject | None,
        "Latest cost fields of the provider's usage, or None when usage reports no cost.",
    ] = None


CURRENT_ATTEMPT: Final[ContextVar[Attempt | None]] = ContextVar("nemo_relay_google_adk_attempt", default=None)


class Backend(NamedTuple):
    provider: Annotated[str, "Provider name the LLM scope takes, or an unknown model's class name."]
    codec: AdkCodec
    client_headers: Annotated[
        bool,
        "True when request headers travel as client extra headers instead of ADK's HTTP options.",
    ]


def usage_fields(usage: BaseModel | None) -> JsonObject | None:
    return None if usage is None else usage.model_dump(mode="json", exclude_none=True)


def record_cost(attempt: Attempt, usage: Mapping[str, Json] | None) -> None:
    """Record on the attempt the `cost` and `cost_usd` usage fields `LlmResponse` drops.

    No change when usage is missing or reports neither field.
    """
    if usage is None:
        return
    reported: JsonObject = {key: usage[key] for key in _COST_FIELDS if key in usage}
    if reported:
        attempt.provider_cost = reported


def sent_headers(held: Mapping[str, _V]) -> dict[str, _V | str]:
    """Return held headers with the attempt's request headers applied.

    An attempt header replaces a held header of the same name in any case. Outside an attempt, return held headers
    unchanged.
    """
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None:
        return dict[str, _V | str](held)
    names = {key.lower() for key in attempt.headers}
    return {**{key: value for key, value in held.items() if key.lower() not in names}, **attempt.headers}


def with_headers(kwargs: dict[str, object]) -> dict[str, object]:
    """Return SDK call kwargs with the attempt's request headers merged into `extra_headers`.

    Unchanged outside an attempt or when the attempt has no request headers.
    """
    attempt = CURRENT_ATTEMPT.get()
    if attempt is None or not attempt.headers:
        return kwargs
    held = kwargs.get("extra_headers")
    extra: Mapping[str, object] = {} if held is None else cast("Mapping[str, object]", held)
    return {**kwargs, "extra_headers": sent_headers(extra)}
