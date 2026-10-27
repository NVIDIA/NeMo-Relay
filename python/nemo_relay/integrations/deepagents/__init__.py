# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Relay integrations for Deep Agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from nemo_relay.integrations.deepagents.callbacks import NemoRelayDeepAgentsCallbackHandler
from nemo_relay.integrations.deepagents.middleware import NemoRelayDeepAgentsMiddleware


def add_nemo_relay_integration(
    kwargs: Mapping[str, Any] | None = None,
    *,
    instrument_subagents: bool = True,
    position: Literal["outer", "inner"] = "inner",
    **overrides: Any,
) -> dict[str, Any]:
    """Attach NeMo Relay observability to ``create_deep_agent`` keyword arguments.

    Use this helper as ``create_deep_agent(**add_nemo_relay_integration(...))``.
    It injects Deep Agents-aware middleware at the top level, adds the same
    middleware to dictionary-style custom subagents that do not inherit parent
    middleware, and leaves any provided backend unchanged.

    Args:
        kwargs: Existing keyword arguments for ``create_deep_agent``.
        instrument_subagents: Whether to add Relay middleware to dictionary
            subagents that do not inherit parent middleware.
        position: Where to add Relay in the user middleware list. ``"inner"``
            adds it at the end by default. ``"outer"`` adds it at the start.
            With ``"outer"``, Relay records tool results after user wrappers
            change them.
            This setting also applies to custom subagents defined as dictionaries.
            Deep Agents controls the order of built-in middleware. Existing
            Relay middleware keeps its position.
        **overrides: Keyword arguments that override values from ``kwargs``.

    Returns:
        dict[str, Any]: Arguments ready to pass to ``create_deep_agent``.

    Raises:
        ValueError: If ``position`` is not ``"outer"`` or ``"inner"``.
    """
    if position not in ("outer", "inner"):
        raise ValueError('position must be "outer" or "inner"')

    observed = dict(kwargs or {})
    observed.update(overrides)

    skills = _string_sequence(observed.get("skills"))
    subagents = list(observed.get("subagents") or ())
    subagent_summaries = [_subagent_summary(subagent) for subagent in subagents]
    backend = observed.get("backend")
    backend_name = type(backend).__name__ if backend is not None else None

    middleware = list(observed.get("middleware") or ())
    _insert_middleware(
        middleware,
        NemoRelayDeepAgentsMiddleware(
            agent_name=observed.get("name"),
            skills=skills,
            subagents=subagent_summaries,
            backend_name=backend_name,
        ),
        position,
    )
    observed["middleware"] = middleware

    if instrument_subagents and subagents:
        observed["subagents"] = [_instrument_subagent(subagent, position) for subagent in subagents]

    return observed


def _insert_middleware(
    middleware: list[Any],
    new_middleware: NemoRelayDeepAgentsMiddleware,
    position: Literal["outer", "inner"],
) -> None:
    if any(isinstance(item, NemoRelayDeepAgentsMiddleware) for item in middleware):
        return
    if position == "outer":
        middleware.insert(0, new_middleware)
    else:
        middleware.append(new_middleware)


def _instrument_subagent(subagent: Any, position: Literal["outer", "inner"]) -> Any:
    if not isinstance(subagent, dict):
        return subagent

    observed = dict(subagent)
    middleware = list(observed.get("middleware") or ())
    _insert_middleware(
        middleware,
        NemoRelayDeepAgentsMiddleware(
            agent_name=observed.get("name"),
            skills=_string_sequence(observed.get("skills")),
            subagents=None,
        ),
        position,
    )
    observed["middleware"] = middleware
    return observed


def _subagent_summary(subagent: Any) -> Mapping[str, Any]:
    if isinstance(subagent, Mapping):
        summary: dict[str, Any] = {"type": "subagent"}
        for key in ("name", "description", "model", "graph_id", "url"):
            value = subagent.get(key)
            if value is not None:
                summary[key] = value
        if "skills" in subagent:
            summary["skills"] = _string_sequence(subagent.get("skills"))
        return summary

    summary = {"type": type(subagent).__name__}
    for attr in ("name", "description", "graph_id", "url"):
        value = getattr(subagent, attr, None)
        if value is not None:
            summary[attr] = value
    return summary


def _string_sequence(value: Any) -> Sequence[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


__all__ = [
    "NemoRelayDeepAgentsCallbackHandler",
    "NemoRelayDeepAgentsMiddleware",
    "add_nemo_relay_integration",
]
