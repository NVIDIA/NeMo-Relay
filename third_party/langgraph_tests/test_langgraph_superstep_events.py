# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LangGraph superstep boundary event emission.

Validates that emit_superstep_start and emit_superstep_end in _nat_nexus.py
emit correct Nexus Mark events with expected name, step, and task_count fields.
"""

from __future__ import annotations

import threading
from typing import Any

import nat_nexus
import pytest
from langgraph._nat_nexus import (  # type: ignore[import-untyped]
    available,
    emit_superstep_end,
    emit_superstep_start,
    pop_graph_scope,
    push_graph_scope,
)
from nat_nexus import create_scope_stack, set_thread_scope_stack


class TestSuperstepEvents:
    """Validate superstep boundary event emission."""

    @pytest.fixture(autouse=True)
    def scope_stack(self):
        """Create an isolated scope stack for each test."""
        stack = create_scope_stack()
        set_thread_scope_stack(stack)
        yield stack

    @pytest.fixture()
    def events(self):
        """Register an event subscriber and collect events."""
        collected: list[Any] = []
        nat_nexus.subscribers.register("test-superstep-collector", lambda e: collected.append(e))
        yield collected
        nat_nexus.subscribers.deregister("test-superstep-collector")

    def test_superstep_start_emits_event(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_superstep_start emits a Mark event with correct name and data fields."""
        graph_handle = push_graph_scope("start_graph")
        emit_superstep_start(step=0, task_count=2)
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Superstep Start" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Superstep Start' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.data["step"] == 0
        assert ev.data["task_count"] == 2

    def test_superstep_end_emits_event(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_superstep_end emits a Mark event with correct name and data fields."""
        graph_handle = push_graph_scope("end_graph")
        emit_superstep_end(step=0, task_count=2)
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Superstep End" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Superstep End' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.data["step"] == 0
        assert ev.data["task_count"] == 2

    def test_superstep_no_scope_no_event(self, scope_stack: Any, events: list[Any]) -> None:
        """No superstep events emitted when no scope stack is active on the thread."""
        results: dict = {}

        def worker() -> None:
            emit_superstep_start(step=1, task_count=3)
            emit_superstep_end(step=1, task_count=3)
            results["start_count"] = len([e for e in events if e.name == "Superstep Start" and e.kind == "Mark"])
            results["end_count"] = len([e for e in events if e.name == "Superstep End" and e.kind == "Mark"])
            results["available"] = available()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert results.get("available") is False, "available() should return False on thread without scope stack"
        assert results.get("start_count", 0) == 0, "No 'Superstep Start' events should be emitted without a scope stack"
        assert results.get("end_count", 0) == 0, "No 'Superstep End' events should be emitted without a scope stack"

    def test_superstep_start_end_ordering(self, scope_stack: Any, events: list[Any]) -> None:
        """'Superstep Start' Mark event appears before 'Superstep End' Mark event in one step."""
        graph_handle = push_graph_scope("ordering_graph")
        emit_superstep_start(step=0, task_count=1)
        emit_superstep_end(step=0, task_count=1)
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.kind == "Mark" and e.name in {"Superstep Start", "Superstep End"}]
        assert len(mark_events) == 2, (
            f"Expected 2 superstep Mark events, got {len(mark_events)}: {[e.name for e in mark_events]}"
        )
        names = [e.name for e in mark_events]
        start_idx = names.index("Superstep Start")
        end_idx = names.index("Superstep End")
        assert start_idx < end_idx, (
            f"'Superstep Start' (idx {start_idx}) should appear before 'Superstep End' (idx {end_idx})"
        )

    def test_superstep_multiple_steps(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_superstep_start and _end carry correct step values across multiple steps."""
        graph_handle = push_graph_scope("multi_step_graph")
        for step in range(3):
            emit_superstep_start(step=step, task_count=2)
            emit_superstep_end(step=step, task_count=2)
        pop_graph_scope(graph_handle)

        start_events = [e for e in events if e.name == "Superstep Start" and e.kind == "Mark"]
        end_events = [e for e in events if e.name == "Superstep End" and e.kind == "Mark"]
        assert len(start_events) == 3, f"Expected 3 'Superstep Start' events, got {len(start_events)}"
        assert len(end_events) == 3, f"Expected 3 'Superstep End' events, got {len(end_events)}"

        for i, ev in enumerate(start_events):
            assert ev.data["step"] == i, f"'Superstep Start' event {i} should have step=={i}, got {ev.data['step']}"
        for i, ev in enumerate(end_events):
            assert ev.data["step"] == i, f"'Superstep End' event {i} should have step=={i}, got {ev.data['step']}"
