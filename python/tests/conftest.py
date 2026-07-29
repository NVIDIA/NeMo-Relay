# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for Python tests."""

from __future__ import annotations

import typing
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

if typing.TYPE_CHECKING:
    import nemo_relay


@pytest.fixture(autouse=True)
def isolate_user_plugin_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Prevent local user plugin configuration from affecting test behavior."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


@pytest.fixture(name="subscribed_events")
def subscribed_events_fixture() -> Iterator[list[nemo_relay.Event]]:
    import nemo_relay

    events: list[nemo_relay.Event] = []

    def event_recorder(event: nemo_relay.Event) -> None:
        events.append(event)

    subscriber_name = f"test-{uuid4()}"
    nemo_relay.subscribers.register(subscriber_name, event_recorder)
    yield events
    try:
        nemo_relay.subscribers.flush()
    finally:
        nemo_relay.subscribers.deregister(subscriber_name)
