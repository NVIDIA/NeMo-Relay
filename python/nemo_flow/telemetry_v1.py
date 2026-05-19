# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable telemetry facade for host-runtime integrations.

This module gives applications a narrow, semver-oriented surface for consuming
NeMo Flow telemetry without depending directly on native event object details.
It is intended for host runtimes that already own execution and need a stable
subscriber/exporter integration point.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import nemo_flow
from nemo_flow import subscribers

logger = logging.getLogger(__name__)

EVENT_SCHEMA_VERSION = "nemo_flow.telemetry.v1"

ErrorPolicy: TypeAlias = Literal["log", "ignore"]
TelemetryEvent: TypeAlias = dict[str, Any]
TelemetryObserver: TypeAlias = Callable[[TelemetryEvent], None]


@dataclass(slots=True)
class Subscription:
    """Handle returned by :func:`register_observer`."""

    name: str
    _active: bool = True

    def deregister(self) -> bool:
        """Deregister the underlying NeMo Flow subscriber."""
        if not self._active:
            return False
        removed = subscribers.deregister(self.name)
        self._active = False
        return removed

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.deregister()


def register_observer(
    name: str,
    callback: TelemetryObserver,
    *,
    error_policy: ErrorPolicy = "log",
) -> Subscription:
    """Register a safe observer for serialized telemetry events.

    Args:
        name: Unique subscriber name.
        callback: Function invoked with a stable event dictionary.
        error_policy: Callback failure handling. ``"log"`` records and
            suppresses callback errors, while ``"ignore"`` suppresses them
            silently. Native subscriber dispatch is fail-open, so observer
            errors are never allowed to interrupt host execution.

    Returns:
        Subscription: Deregistration/context-manager handle.
    """
    if error_policy not in {"log", "ignore"}:
        raise ValueError("error_policy must be one of: log, ignore")

    def _wrapped(event: nemo_flow.Event) -> None:
        try:
            callback(event_to_dict(event))
        except Exception:
            if error_policy == "log":
                logger.exception("NeMo Flow telemetry observer '%s' failed", name)

    subscribers.register(name, _wrapped)
    return Subscription(name=name)


def observer(
    name: str,
    callback: TelemetryObserver,
    *,
    error_policy: ErrorPolicy = "log",
) -> Subscription:
    """Context-manager alias for :func:`register_observer`."""
    return register_observer(name, callback, error_policy=error_policy)


def event_to_json(event: nemo_flow.Event) -> str:
    """Serialize a NeMo Flow event to canonical telemetry JSON."""
    return json.dumps(event_to_dict(event), sort_keys=True, separators=(",", ":"))


def event_to_dict(event: nemo_flow.Event) -> TelemetryEvent:
    """Serialize a NeMo Flow event to the stable telemetry-v1 dictionary shape."""
    kind = _attr(event, "kind")
    if kind == "scope":
        return _scope_event_to_dict(event)
    if kind == "mark":
        return _mark_event_to_dict(event)
    raise TypeError(f"unsupported NeMo Flow event kind: {kind!r}")


def _scope_event_to_dict(event: Any) -> TelemetryEvent:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "kind": "scope",
        "scope_category": _attr(event, "scope_category"),
        "category": _attr(event, "category"),
        "uuid": _attr(event, "uuid"),
        "parent_uuid": _attr(event, "parent_uuid"),
        "name": _attr(event, "name"),
        "timestamp": _attr(event, "timestamp"),
        "data": _json_value(_attr(event, "data")),
        "metadata": _json_value(_attr(event, "metadata")),
        "attributes": _json_value(_attr(event, "attributes")),
        "category_profile": _json_value(_attr(event, "category_profile")),
        "data_schema": _json_value(_attr(event, "data_schema")),
        "annotated_request": _annotated_request_to_dict(_attr(event, "annotated_request")),
        "annotated_response": _annotated_response_to_dict(_attr(event, "annotated_response")),
    }


def _mark_event_to_dict(event: Any) -> TelemetryEvent:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "kind": "mark",
        "scope_category": None,
        "category": _attr(event, "category"),
        "uuid": _attr(event, "uuid"),
        "parent_uuid": _attr(event, "parent_uuid"),
        "name": _attr(event, "name"),
        "timestamp": _attr(event, "timestamp"),
        "data": _json_value(_attr(event, "data")),
        "metadata": _json_value(_attr(event, "metadata")),
        "attributes": None,
        "category_profile": _json_value(_attr(event, "category_profile")),
        "data_schema": _json_value(_attr(event, "data_schema")),
        "annotated_request": None,
        "annotated_response": None,
    }


def _annotated_request_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "messages": _json_value(_attr(value, "messages")),
        "model": _attr(value, "model"),
        "params": _json_value(_attr(value, "params")),
        "tools": _json_value(_attr(value, "tools")),
        "tool_choice": _json_value(_attr(value, "tool_choice")),
        "extra": _json_value(_attr(value, "extra")),
    }


def _annotated_response_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "id": _attr(value, "id"),
        "model": _attr(value, "model"),
        "message": _json_value(_attr(value, "message")),
        "tool_calls": _json_value(_attr(value, "tool_calls")),
        "finish_reason": _attr(value, "finish_reason"),
        "usage": _json_value(_attr(value, "usage")),
        "api_specific": _json_value(_attr(value, "api_specific")),
        "extra": _json_value(_attr(value, "extra")),
        "response_text": _call_method(value, "response_text"),
        "has_tool_calls": _call_method(value, "has_tool_calls"),
    }


def _attr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return None


def _call_method(value: Any, name: str) -> Any:
    try:
        method = getattr(value, name)
        if callable(method):
            return method()
    except Exception:
        return None
    return None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return cast(str, str(value))


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "ErrorPolicy",
    "Subscription",
    "TelemetryEvent",
    "TelemetryObserver",
    "event_to_dict",
    "event_to_json",
    "observer",
    "register_observer",
]
