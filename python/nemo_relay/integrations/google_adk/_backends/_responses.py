# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses codec for ADK's `OpenAIResponsesLlm`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from nemo_relay.codecs import OpenAIResponsesCodec
from nemo_relay.integrations.google_adk._serialization import (
    BASE_USAGE,
    BASE_USAGE_KEYS,
    GENAI_USAGE,
    RECORDED_USAGE_KEY,
    AdkCodec,
    Normalized,
    reported_cost,
    token_counts,
    with_uncached,
)

if TYPE_CHECKING:
    from nemo_relay import JsonObject

_API: Final = "openai_responses"
_USAGE_KEYS: Final = BASE_USAGE_KEYS | {"thoughts_token_count"}
_RECORDED_KEYS: Final = frozenset({RECORDED_USAGE_KEY})
_NATIVE_CODEC: Final = OpenAIResponsesCodec()


def _read(response: JsonObject, recorded: JsonObject, usage_metadata: JsonObject) -> Normalized:
    """Read Responses usage. Cached and reasoning counts go under `input_tokens_details` and `output_tokens_details`."""
    counts = token_counts(usage_metadata, BASE_USAGE)
    separate = token_counts(usage_metadata, GENAI_USAGE)
    api_specific: JsonObject = {"api": _API}
    if (cached := counts.get("cache_read_tokens")) is not None:
        api_specific["input_tokens_details"] = {"cached_tokens": cached}
    if (reasoning := separate.get("thoughts_tokens")) is not None:
        api_specific["output_tokens_details"] = {"reasoning_tokens": reasoning}
    return Normalized(
        usage=with_uncached(counts),
        cost=reported_cost(recorded, codec=_NATIVE_CODEC),
        api_specific=api_specific,
        usage_keys=_USAGE_KEYS,
        response_keys=frozenset(),
        recorded_keys=_RECORDED_KEYS,
    )


RESPONSES_CODEC: Final = AdkCodec(_read)
