# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Flow integrations for Deep Agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nemo_flow.integrations.deepagents.backend import NemoFlowDeepAgentsBackend, observe_backend
from nemo_flow.integrations.deepagents.callbacks import NemoFlowDeepAgentsCallbackHandler
from nemo_flow.integrations.deepagents.middleware import NemoFlowDeepAgentsMiddleware


def add_nemo_flow_integration(
    kwargs: Mapping[str, Any] | None = None,
    *,
    instrument_subagents: bool = True,
    instrument_backend: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """
    Receives the keyword arguments for ``create_deep_agent`` and returns them with NeMo Flow observability attached.

    Use this helper as ``create_deep_agent(**add_nemo_flow_integration(...))``.
    It injects Deep Agents-aware middleware at the top level, adds the same
    middleware to dictionary-style custom subagents that do not inherit parent
    middleware, and optionally wraps the backend.
    """
    observed = dict(kwargs or {})
    observed.update(overrides)

    skills = _string_sequence(observed.get("skills"))
    subagents = list(observed.get("subagents") or ())
    subagent_summaries = [_subagent_summary(subagent) for subagent in subagents]
    backend = observed.get("backend")

    if isinstance(backend, NemoFlowDeepAgentsBackend):
        observed["backend"] = backend
        backend_name = type(backend.__wrapped__).__name__
    elif instrument_backend and backend is not None:
        observed["backend"] = observe_backend(backend)
        backend_name = type(backend).__name__
    elif backend is not None:
        backend_name = type(backend).__name__
    else:
        backend_name = None

    middleware = list(observed.get("middleware") or ())
    _append_middleware(
        middleware,
        NemoFlowDeepAgentsMiddleware(
            agent_name=observed.get("name"),
            skills=skills,
            subagents=subagent_summaries,
            backend_name=backend_name,
        ),
    )
    observed["middleware"] = middleware

    if instrument_subagents and subagents:
        observed["subagents"] = [_instrument_subagent(subagent) for subagent in subagents]

    return observed


def _append_middleware(middleware: list[Any], new_middleware: NemoFlowDeepAgentsMiddleware) -> None:
    if any(isinstance(item, NemoFlowDeepAgentsMiddleware) for item in middleware):
        return
    middleware.append(new_middleware)


def _instrument_subagent(subagent: Any) -> Any:
    if not isinstance(subagent, dict):
        return subagent

    observed = dict(subagent)
    middleware = list(observed.get("middleware") or ())
    _append_middleware(
        middleware,
        NemoFlowDeepAgentsMiddleware(
            agent_name=observed.get("name"),
            skills=_string_sequence(observed.get("skills")),
            subagents=None,
        ),
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
    "NemoFlowDeepAgentsBackend",
    "NemoFlowDeepAgentsCallbackHandler",
    "NemoFlowDeepAgentsMiddleware",
    "add_nemo_flow_integration",
    "observe_backend",
]
