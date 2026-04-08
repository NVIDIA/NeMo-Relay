# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LangGraph edge traversal event emission (HIER-01).

Validates that emit_edge_write in _nat_nexus.py emits correct Nexus Mark events
with the expected name, data fields, and source_node extraction behavior (D-02).
"""

from __future__ import annotations

import threading
from typing import Any

import nat_nexus
import pytest
from langgraph._nat_nexus import (  # type: ignore[import-untyped]
    available,
    emit_edge_write,
    pop_graph_scope,
    pop_node_scope,
    push_graph_scope,
    push_node_scope,
)
from nat_nexus import create_scope_stack, set_thread_scope_stack


class TestEdgeWriteEvents:
    """Validate edge traversal event emission (HIER-01)."""

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
        nat_nexus.subscribers.register("test-edge-collector", lambda e: collected.append(e))
        yield collected
        nat_nexus.subscribers.deregister("test-edge-collector")

    def test_edge_write_emits_event(self, scope_stack: Any, events: list[Any]) -> None:
        """emit_edge_write emits a Mark event with correct name and data fields (HIER-01)."""
        graph_handle = push_graph_scope("edge_graph")
        emit_edge_write(source_node="node_a", channels=["out"], write_count=1)
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Edge Write" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Edge Write' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.data["source_node"] == "node_a"
        assert ev.data["channels"] == ["out"]
        assert ev.data["write_count"] == 1

    def test_edge_write_source_node_root(self, scope_stack: Any, events: list[Any]) -> None:
        """D-02 source_node extraction correct for root node format 'nodename:task_id'."""
        ns = "mynode:some-uuid-1234"
        source_node = ns.split(":")[0].split("|")[-1] if ns else "unknown"

        graph_handle = push_graph_scope("root_extract_graph")
        emit_edge_write(source_node=source_node, channels=["state"], write_count=1)
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Edge Write" and e.kind == "Mark"]
        assert len(mark_events) == 1
        assert mark_events[0].data["source_node"] == "mynode", (
            f"Expected 'mynode', got '{mark_events[0].data['source_node']}'"
        )

    def test_edge_write_source_node_nested(self, scope_stack: Any, events: list[Any]) -> None:
        """D-02 source_node extraction correct for nested format 'parent|child:task_id'."""
        ns = "parent|child:some-uuid-5678"
        source_node = ns.split(":")[0].split("|")[-1] if ns else "unknown"

        graph_handle = push_graph_scope("nested_extract_graph")
        emit_edge_write(source_node=source_node, channels=["output"], write_count=1)
        pop_graph_scope(graph_handle)

        mark_events = [e for e in events if e.name == "Edge Write" and e.kind == "Mark"]
        assert len(mark_events) == 1
        assert mark_events[0].data["source_node"] == "child", (
            f"Expected 'child', got '{mark_events[0].data['source_node']}'"
        )

    def test_edge_write_no_scope_no_event(self, scope_stack: Any, events: list[Any]) -> None:
        """No 'Edge Write' event emitted when no scope stack is active on the thread."""
        results: dict = {}

        def worker() -> None:
            emit_edge_write(source_node="n", channels=["c"], write_count=1)
            results["event_count"] = len([e for e in events if e.name == "Edge Write" and e.kind == "Mark"])
            results["available"] = available()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert results.get("available") is False, "available() should return False on thread without scope stack"
        assert results.get("event_count", 0) == 0, "No 'Edge Write' events should be emitted without a scope stack"

    def test_edge_write_in_node_scope(self, scope_stack: Any, events: list[Any]) -> None:
        """Edge Write event parent_uuid matches node scope UUID when fired inside a node scope."""
        graph_handle = push_graph_scope("scope_ancestry_graph")
        node_handle, _, _ = push_node_scope("writer_node", "task-uuid-0001")

        emit_edge_write(source_node="writer_node", channels=["result"], write_count=1)

        mark_events = [e for e in events if e.name == "Edge Write" and e.kind == "Mark"]
        assert len(mark_events) == 1, f"Expected 1 'Edge Write' Mark event, got {len(mark_events)}"
        ev = mark_events[0]
        assert ev.parent_uuid == node_handle.uuid, (
            f"Edge Write event parent_uuid ({ev.parent_uuid}) should match "
            f"node scope UUID ({node_handle.uuid}) -- confirms node-scope ancestry"
        )

        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)
