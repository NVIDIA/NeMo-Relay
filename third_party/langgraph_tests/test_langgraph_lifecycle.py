# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for LangGraph lifecycle event emission.

Validates that the four lifecycle event helper functions in ``_nemo_flow.py``
emit correct NeMo Flow Mark events with the expected names, event types, and data
fields, and that guard behavior prevents spurious events when no scope stack
is active.

Covers checkpoint save, checkpoint restore, graph interrupt, and graph resume.
"""

from __future__ import annotations

import threading
from collections import namedtuple
from typing import Any

import nemo_flow
import pytest
from langgraph._nemo_flow import (  # type: ignore[import-untyped]
    available,
    emit_checkpoint_restore,
    emit_checkpoint_save,
    emit_graph_interrupt,
    emit_graph_resume,
    pop_graph_scope,
    push_graph_scope,
)
from nemo_flow import create_scope_stack, set_thread_scope_stack


class TestCheckpointEvents:
    """Validate checkpoint save and restore event emission."""

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
        nemo_flow.subscribers.register("test-lifecycle-collector", lambda e: collected.append(e))
        yield collected
        nemo_flow.subscribers.deregister("test-lifecycle-collector")

    # -------------------------------------------------------------------
    # Checkpoint Save events
    # -------------------------------------------------------------------

    def test_checkpoint_save_emits_event(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_checkpoint_save emits a Mark event with correct name and data fields."""
        graph_handle = push_graph_scope("save_graph")
        emit_checkpoint_save(source="loop", step=3, thread_id="thread-1", checkpoint_id="ckpt-abc")
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Checkpoint Save" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Checkpoint Save' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.data["source"] == "loop"
        assert ev.data["step"] == 3
        assert ev.data["thread_id"] == "thread-1"
        assert ev.data["checkpoint_id"] == "ckpt-abc"

    def test_checkpoint_save_all_sources(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_checkpoint_save distinguishes all three source types."""
        graph_handle = push_graph_scope("sources_graph")
        emit_checkpoint_save(source="input", step=0, thread_id="t1", checkpoint_id="ckpt-1")
        emit_checkpoint_save(source="loop", step=1, thread_id="t1", checkpoint_id="ckpt-2")
        emit_checkpoint_save(source="exit", step=2, thread_id="t1", checkpoint_id="ckpt-3")
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Checkpoint Save" and e.kind == "Mark"]
        assert len(mark_events) == 3, f"Expected 3 'Checkpoint Save' Mark events, got {len(mark_events)}"
        sources = [e.data["source"] for e in mark_events]
        assert sources == ["input", "loop", "exit"], f"Expected sources ['input', 'loop', 'exit'], got {sources}"

    def test_no_checkpoint_event_without_checkpointer(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_checkpoint_save produces no events when no scope stack is active."""
        results: dict[str, int] = {}

        def worker() -> None:
            emit_checkpoint_save(source="loop", step=1, thread_id="t1", checkpoint_id="c1")
            results["event_count"] = len([e for e in events if e.name == "Checkpoint Save" and e.kind == "Mark"])
            results["available"] = available()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert results.get("available") is False, "available() should return False on thread without scope stack"
        assert results.get("event_count", 0) == 0, "No 'Checkpoint Save' events should be emitted without a scope stack"

    def test_checkpoint_restore_emits_event(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_checkpoint_restore emits a Mark event with correct name and data fields."""
        graph_handle = push_graph_scope("restore_graph")
        emit_checkpoint_restore(checkpoint_id="ckpt-xyz", thread_id="thread-2", step=5)
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Checkpoint Restore" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Checkpoint Restore' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.data["checkpoint_id"] == "ckpt-xyz"
        assert ev.data["thread_id"] == "thread-2"
        assert ev.data["step"] == 5

    def test_no_restore_event_on_first_run(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_checkpoint_restore produces no events when no scope stack is active."""
        results: dict[str, int] = {}

        def worker() -> None:
            emit_checkpoint_restore(checkpoint_id="ckpt-000", thread_id="t0", step=0)
            results["event_count"] = len([e for e in events if e.name == "Checkpoint Restore" and e.kind == "Mark"])
            results["available"] = available()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert results.get("available") is False, "available() should return False on thread without scope stack"
        assert results.get("event_count", 0) == 0, (
            "No 'Checkpoint Restore' events should be emitted without a scope stack"
        )


class TestInterruptEvents:
    """Validate graph interrupt and resume event emission."""

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
        nemo_flow.subscribers.register("test-interrupt-collector", lambda e: collected.append(e))
        yield collected
        nemo_flow.subscribers.deregister("test-interrupt-collector")

    # -------------------------------------------------------------------
    # Graph Interrupt events
    # -------------------------------------------------------------------

    def test_graph_interrupt_emits_event(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_graph_interrupt emits a Mark event with trigger and interrupts."""
        graph_handle = push_graph_scope("interrupt_graph")
        emit_graph_interrupt(trigger="before", interrupts=[])
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Graph Interrupt" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Graph Interrupt' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.data["trigger"] == "before"
        assert ev.data["interrupts"] == []

    def test_interrupt_trigger_field(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_graph_interrupt serializes interrupt payloads correctly."""
        MockInterrupt = namedtuple("MockInterrupt", ["value", "id"])

        graph_handle = push_graph_scope("payload_graph")
        emit_graph_interrupt(
            trigger="after",
            interrupts=[MockInterrupt(value="please stop", id="intr-1")],
        )
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Graph Interrupt" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Graph Interrupt' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.data["trigger"] == "after"
        assert len(ev.data["interrupts"]) == 1
        intr = ev.data["interrupts"][0]
        assert "please stop" in intr["value"]
        assert intr["id"] == "intr-1"

    def test_scope_survives_interrupt(self, scope_stack: Any, events: list[Any]) -> None:
        """Graph scope is still active when interrupt event fires."""
        graph_handle = push_graph_scope("survive_graph")
        emit_graph_interrupt(trigger="before", interrupts=[])

        mark_events = [e for e in events if e.name == "Graph Interrupt" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Graph Interrupt' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.parent_uuid == graph_handle.uuid, (
            f"Interrupt event parent_uuid ({ev.parent_uuid}) should match "
            f"graph scope UUID ({graph_handle.uuid}) -- scope was active at emission"
        )

        pop_graph_scope(graph_handle)

    # -------------------------------------------------------------------
    # Graph Resume events
    # -------------------------------------------------------------------

    def test_graph_resume_emits_event(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_graph_resume emits a Mark event with serialized resume_values."""
        graph_handle = push_graph_scope("resume_graph")
        emit_graph_resume(resume_values={"answer": 42})
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Graph Resume" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Graph Resume' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert "resume_values" in ev.data, "Event data should contain 'resume_values' key"
        assert "answer" in ev.data["resume_values"], (
            f"'answer' key should appear in serialized resume_values: {ev.data['resume_values']}"
        )

    def test_resume_event_ordering(self, scope_stack: Any, events: list[Any]) -> None:
        """'Checkpoint Restore' appears before 'Graph Resume' in the event list."""
        graph_handle = push_graph_scope("order_graph")
        emit_checkpoint_restore(checkpoint_id="ckpt-order", thread_id="t-order", step=7)
        emit_graph_resume(resume_values={"cmd": "continue"})
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.kind == "Mark" and e.name in {"Checkpoint Restore", "Graph Resume"}]
        assert len(mark_events) == 2, (
            f"Expected 2 Mark events (restore + resume), got {len(mark_events)}: {[e.name for e in mark_events]}"
        )
        names = [e.name for e in mark_events]
        restore_idx = names.index("Checkpoint Restore")
        resume_idx = names.index("Graph Resume")
        assert restore_idx < resume_idx, (
            f"'Checkpoint Restore' (idx {restore_idx}) should appear before "
            f"'Graph Resume' (idx {resume_idx}) in the event list"
        )
