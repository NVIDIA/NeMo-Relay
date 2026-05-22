# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Deep Agents observability helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import nemo_relay

_logger = logging.getLogger(__name__)


def event_base_name(kind: str) -> str:
    """Return a stable event base name for a Deep Agents category."""
    return {
        "human_in_the_loop": "DeepAgents Human In The Loop",
        "skill": "DeepAgents Skills",
    }.get(kind, "DeepAgents")


def json_safe(value: Any) -> nemo_relay.Json:
    """Return a conservative JSON-compatible value."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes | bytearray):
        return f"<{type(value).__name__}: {len(value)} bytes>"
    return repr(value)


def emit_mark(
    base_name: str,
    kind: str,
    phase: str,
    data: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit a Deep Agents mark event without changing framework behavior."""
    event_metadata: dict[str, Any] = {
        "integration": "deepagents",
        "deepagents_kind": kind,
        "phase": phase,
    }
    if metadata:
        event_metadata.update(metadata)

    try:
        nemo_relay.scope.event(
            f"{base_name} {phase.title()}",
            data=json_safe(data),
            metadata=json_safe(event_metadata),
        )
    except Exception:
        _logger.debug("NeMo Relay: Deep Agents mark emission failed", exc_info=True)
