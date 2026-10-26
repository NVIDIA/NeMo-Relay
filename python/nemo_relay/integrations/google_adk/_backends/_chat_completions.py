# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat Completions codec for ADK's `OpenAILlm` and `ApigeeLlm` in chat completions mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from nemo_relay.codecs import OpenAIChatCodec
from nemo_relay.integrations.google_adk._serialization import (
    BASE_USAGE,
    BASE_USAGE_KEYS,
    RECORDED_USAGE_KEY,
    AdkCodec,
    Normalized,
    reported_cost,
    token_counts,
    with_uncached,
)

if TYPE_CHECKING:
    from nemo_relay import JsonObject

_API: Final = "openai_chat"
_RECORDED_KEYS: Final = frozenset({RECORDED_USAGE_KEY})
_NATIVE_CODEC: Final = OpenAIChatCodec()


def _read(response: JsonObject, recorded: JsonObject, usage_metadata: JsonObject) -> Normalized:
    """Read Chat Completions usage. Completion count already includes any reasoning tokens."""
    return Normalized(
        usage=with_uncached(token_counts(usage_metadata, BASE_USAGE)),
        cost=reported_cost(recorded, codec=_NATIVE_CODEC),
        api_specific={"api": _API},
        usage_keys=BASE_USAGE_KEYS,
        response_keys=frozenset(),
        recorded_keys=_RECORDED_KEYS,
    )


CHAT_CODEC: Final = AdkCodec(_read)
