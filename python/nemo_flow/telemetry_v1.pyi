# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for the stable telemetry-v1 facade."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from nemo_flow import Event

EVENT_SCHEMA_VERSION: str

ErrorPolicy: TypeAlias = Literal["log", "ignore"]
TelemetryEvent: TypeAlias = dict[str, Any]
TelemetryObserver: TypeAlias = Callable[[TelemetryEvent], None]

class Subscription:
    name: str
    def deregister(self) -> bool: ...
    def __enter__(self) -> Subscription: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

def register_observer(
    name: str,
    callback: TelemetryObserver,
    *,
    error_policy: ErrorPolicy = "log",
) -> Subscription: ...
def observer(
    name: str,
    callback: TelemetryObserver,
    *,
    error_policy: ErrorPolicy = "log",
) -> Subscription: ...
def event_to_dict(event: Event) -> TelemetryEvent: ...
def event_to_json(event: Event) -> str: ...
