# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve provider backend for ADK model class. Imports provider module on first use."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.models.google_llm import Gemini

from nemo_relay.integrations.google_adk._backends._apigee import apigee_backend
from nemo_relay.integrations.google_adk._backends._attempt import Backend
from nemo_relay.integrations.google_adk._backends._gemini import GEMINI_CODEC, gemini_backend

if TYPE_CHECKING:
    from google.adk.models import BaseLlm


def _defined_in(cls: type[BaseLlm], module: str) -> bool:
    """Whether any class in `cls` MRO is defined in `module` or a submodule of it."""
    return any(base.__module__ == module or base.__module__.startswith(f"{module}.") for base in cls.__mro__)


def _package(cls: type[BaseLlm]) -> str:
    return cls.__module__.rpartition(".")[0]


def _unknown(cls: type[BaseLlm]) -> TypeError:
    return TypeError(
        f"NeMo Relay does not know the provider API of ADK's model class {cls.__module__}.{cls.__qualname__}"
    )


async def backend_of(model: BaseLlm) -> Backend:
    """Return provider, codec, and header channel for model's class.

    Check ADK's Gemini family first, then classes behind each installed provider SDK. Swap provider client's class so it
    applies each attempt's edits, records response fields `LlmResponse` drops, or both. Raise `TypeError` for a class
    from one of those ADK provider modules that NeMo Relay does not know. Any other class: class name as provider,
    Gemini codec.
    """
    cls = type(model)
    if isinstance(model, ApigeeLlm):
        return apigee_backend(model)
    if isinstance(model, Gemini):
        return gemini_backend(model)
    if _defined_in(cls, ApigeeLlm.__module__) or _defined_in(cls, Gemini.__module__):
        raise _unknown(cls)
    if find_spec("anthropic") is not None:
        from google.adk.models.anthropic_llm import AnthropicLlm

        from nemo_relay.integrations.google_adk._backends._anthropic import anthropic_backend

        if isinstance(model, AnthropicLlm):
            return await anthropic_backend(model)
        if _defined_in(cls, AnthropicLlm.__module__):
            raise _unknown(cls)
    if find_spec("openai") is not None:
        from google.adk.integrations.openai import OpenAILlm, OpenAIResponsesLlm

        from nemo_relay.integrations.google_adk._backends._openai import openai_chat_backend, openai_responses_backend

        if isinstance(model, OpenAILlm):
            return openai_chat_backend(model)
        if isinstance(model, OpenAIResponsesLlm):
            return openai_responses_backend(model)
        if _defined_in(cls, _package(OpenAILlm)):
            raise _unknown(cls)
    return Backend(cls.__name__, GEMINI_CODEC, client_headers=False)
