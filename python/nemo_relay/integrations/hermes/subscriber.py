# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subscriber that bridges NeMo Relay ATOF lifecycle events to Hermes-compatible log entries."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Mapping
from typing import Any

_logger = logging.getLogger(__name__)


class HermesATOFSubscriber:
    """Format NeMo Relay lifecycle events as Hermes-compatible structured log dicts.

    Each event is converted to a dict with keys:
    event_type, scope_name, scope_type, timestamp, data.
    """

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        """Process a single lifecycle event."""
        try:
            formatted = self._format_event(event)
            self._events.append(formatted)
            _logger.debug("Hermes ATOF: %s %s", formatted["event_type"], formatted["scope_name"])
        except Exception:
            _logger.debug("Failed to format event", exc_info=True)

    def _format_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "event_type": event.get("event_type", event.get("type", "unknown")),
            "scope_name": event.get("scope_name", event.get("name", "")),
            "scope_type": event.get("scope_type", ""),
            "timestamp": event.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "data": event.get("data", {}),
        }

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return accumulated events (read-only snapshot)."""
        return list(self._events)

    def clear(self) -> None:
        """Flush accumulated events."""
        self._events.clear()
