# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Relay integration for Hermes Agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nemo_relay.integrations.hermes.callbacks import NemoRelayHermesCallbackHandler
from nemo_relay.integrations.hermes.middleware import NemoRelayHermesMiddleware


def add_nemo_relay_integration(
    kwargs: Mapping[str, Any] | None = None,
    *,
    instrument_subagents: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """Attach NeMo Relay observability to a Hermes Agent configuration.

    Use this helper when constructing a Hermes agent to inject Relay
    middleware for tool-call sanitisation, LLM payload guards, and
    lifecycle scope tracking.

    Args:
        kwargs: Existing keyword arguments for the Hermes agent.
        instrument_subagents: Whether to add Relay middleware to
            delegated sub-agents that do not already carry it.
        **overrides: Keyword arguments that override values from *kwargs*.

    Returns:
        dict[str, Any]: Arguments ready to pass to the Hermes agent.
    """
    observed = dict(kwargs or {})
    observed.update(overrides)

    agent_name = observed.get("agent_name", "hermes-agent")
    blocked_tools = list(observed.get("blocked_tools", ()))
    max_tool_args_size = int(observed.get("max_tool_args_size", 65536))

    middleware = list(observed.get("middleware", ()))
    if not any(isinstance(m, NemoRelayHermesMiddleware) for m in middleware):
        middleware.append(
            NemoRelayHermesMiddleware(
                agent_name=agent_name,
                blocked_tools=blocked_tools,
                max_tool_args_size=max_tool_args_size,
            )
        )
    observed["middleware"] = middleware

    callbacks = list(observed.get("callbacks", ()))
    if not any(isinstance(c, NemoRelayHermesCallbackHandler) for c in callbacks):
        callbacks.append(NemoRelayHermesCallbackHandler(agent_name=agent_name))
    observed["callbacks"] = callbacks

    if instrument_subagents:
        subagents = list(observed.get("subagents", ()))
        instrumented = []
        for sub in subagents:
            if isinstance(sub, dict) and not any(
                isinstance(m, NemoRelayHermesMiddleware)
                for m in sub.get("middleware", ())
            ):
                sub = dict(sub)
                sub_mw = list(sub.get("middleware", ()))
                sub_mw.append(
                    NemoRelayHermesMiddleware(
                        agent_name=sub.get("name", "subagent"),
                        blocked_tools=blocked_tools,
                        max_tool_args_size=max_tool_args_size,
                    )
                )
                sub["middleware"] = sub_mw
            instrumented.append(sub)
        observed["subagents"] = instrumented

    return observed


__all__ = [
    "NemoRelayHermesCallbackHandler",
    "NemoRelayHermesMiddleware",
    "add_nemo_relay_integration",
]
