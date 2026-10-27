# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve provider backend for ADK model class. Imports provider module on first use."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Annotated, Final

from google.adk import integrations as adk_integrations
from google.adk import models as adk_models
from google.adk.integrations import openai as adk_openai
from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini

from nemo_relay.integrations.google_adk._backends._apigee import apigee_backend
from nemo_relay.integrations.google_adk._backends._attempt import Backend
from nemo_relay.integrations.google_adk._backends._gemini import GEMINI_CODEC, gemini_backend

_PROVIDER_PACKAGES: Final[Annotated[tuple[str, ...], "ADK packages defining provider model classes."]] = (
    adk_models.__name__,
    adk_integrations.__name__,
)


def _defined_in(cls: type[BaseLlm], package: str) -> bool:
    """Whether a class in `cls` MRO below `BaseLlm` is defined in `package` or a submodule of it."""
    below = cls.__mro__[: cls.__mro__.index(BaseLlm)]
    return any(base.__module__ == package or base.__module__.startswith(f"{package}.") for base in below)


def _unknown(cls: type[BaseLlm]) -> TypeError:
    return TypeError(
        f"NeMo Relay does not know the provider API of ADK's model class {cls.__module__}.{cls.__qualname__}"
    )


async def backend_of(model: BaseLlm, request_model: str | None) -> Backend:
    """Return provider, model name, codec, and header channel for a model call.

    Swap provider client's class so attempt edits apply and fields `LlmResponse` drops get recorded. Classes
    outside ADK's `models` and `integrations` packages: class name as provider, model's own name, Gemini codec.

    Raises:
        TypeError: ADK `models` or `integrations` class NeMo Relay doesn't support.
    """
    cls = type(model)
    if isinstance(model, ApigeeLlm):
        return apigee_backend(model, request_model)
    if isinstance(model, Gemini):
        return gemini_backend(model, request_model)
    if not any(_defined_in(cls, package) for package in _PROVIDER_PACKAGES):
        return Backend(cls.__name__, GEMINI_CODEC, client_headers=False, model=model.model)
    if _defined_in(cls, adk_openai.__name__):
        if find_spec("openai") is not None:
            from google.adk.integrations.openai import OpenAILlm, OpenAIResponsesLlm

            from nemo_relay.integrations.google_adk._backends._openai import (
                openai_chat_backend,
                openai_responses_backend,
            )

            if isinstance(model, OpenAILlm):
                return openai_chat_backend(model)
            if isinstance(model, OpenAIResponsesLlm):
                return openai_responses_backend(model, request_model)
    elif find_spec("anthropic") is not None:
        from google.adk.models.anthropic_llm import AnthropicLlm

        from nemo_relay.integrations.google_adk._backends._anthropic import anthropic_backend

        if isinstance(model, AnthropicLlm):
            return await anthropic_backend(model, request_model)
    if find_spec("litellm") is not None:
        from google.adk.models.lite_llm import LiteLlm, _ensure_litellm_imported

        if isinstance(model, LiteLlm):
            _ensure_litellm_imported()
            from nemo_relay.integrations.google_adk._backends._lite_llm import lite_llm_backend

            return await lite_llm_backend(model, request_model)
    raise _unknown(cls)
