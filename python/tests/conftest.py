# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for Python tests."""

from __future__ import annotations

import typing
from collections.abc import Iterator
import types
from uuid import uuid4

import pytest

if typing.TYPE_CHECKING:
    import nemo_flow


@pytest.fixture(name="subscribed_events")
def subscribed_events_fixture() -> Iterator[list[nemo_flow.Event]]:
    import nemo_flow

    events: list[nemo_flow.Event] = []

    def event_recorder(event: nemo_flow.Event) -> None:
        events.append(event)

    subscriber_name = f"test-{uuid4()}"
    nemo_flow.subscribers.register(subscriber_name, event_recorder)
    yield events
    nemo_flow.subscribers.deregister(subscriber_name)
