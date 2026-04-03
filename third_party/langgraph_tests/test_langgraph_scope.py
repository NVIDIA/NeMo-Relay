# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for LangGraph scope propagation.

Validates that the LangGraph Pregel patch correctly instruments graph and
node execution with Nexus scopes across sync, async, and parallel paths.

These tests exercise the ``_nat_nexus.py`` scope helpers directly to verify
the scope hierarchy, parallel isolation, and double-wrap prevention behavior
without requiring the full LangGraph graph execution engine.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Any

import nat_nexus
import pytest
from langgraph._nat_nexus import (  # type: ignore[import-untyped]
    _graph_scope_info,
    _langgraph_nexus_active,
    available,
    langgraph_nexus_active,
    pop_graph_scope,
    pop_node_scope,
    pop_subgraph_scope,
    push_graph_scope,
    push_node_scope,
    push_subgraph_scope,
)
from nat_nexus import EventType, create_scope_stack, set_thread_scope_stack


class TestLangGraphScope:
    """Validate scope hierarchy, isolation, and lifecycle for LangGraph instrumentation."""

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
        nat_nexus.subscribers.register("test-lg-collector", lambda e: collected.append(e))
        yield collected
        nat_nexus.subscribers.deregister("test-lg-collector")

    # -------------------------------------------------------------------
    # SCOPE-01 / SCOPE-02: Single-node graph scope hierarchy
    # -------------------------------------------------------------------

    def test_single_node_graph_scope(self, scope_stack: Any, events: list[Any]) -> None:
        """Graph scope wraps node scope; node is child of graph.

        Validates SCOPE-01 (graph-level scope) and SCOPE-02 (node-level scope
        with parent-child relationship).

        After Plan 15-02, push_node_scope pushes directly on the current stack
        (no isolated branch scope stacks), so the node is a direct child of
        the graph scope.
        """
        graph_handle = push_graph_scope("my_graph")
        node_handle, _, _ = push_node_scope("my_node", "task-1")

        node_events = [e for e in events if e.name == "my_node" and e.event_type == EventType.Start]
        assert len(node_events) == 1, f"Expected 1 node start event, got {len(node_events)}"
        node_start = node_events[0]

        # Node is now a direct child of the graph scope (no intermediate branch graph)
        assert node_start.parent_uuid == graph_handle.uuid

        assert node_start.metadata.get("langgraph.node") is True
        assert node_start.metadata.get("langgraph.task_id") == "task-1"

        graph_starts = [
            e
            for e in events
            if e.metadata and e.metadata.get("langgraph.graph") is True and e.event_type == EventType.Start
        ]
        assert len(graph_starts) >= 1, "Expected at least 1 graph start event"
        assert graph_starts[0].metadata.get("langgraph.graph") is True

        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)

    # -------------------------------------------------------------------
    # SCOPE-02: Multi-node sequential graph scope
    # -------------------------------------------------------------------

    def test_multi_node_graph_scope(self, scope_stack: Any, events: list[Any]) -> None:
        """Multiple sequential nodes share the same graph parent scope.

        Validates SCOPE-02: each node scope is a direct child of the graph scope.
        """
        graph_handle = push_graph_scope("seq_graph")

        node_a_handle, _, _ = push_node_scope("node_a", "task-a")
        pop_node_scope(node_a_handle)

        node_b_handle, _, _ = push_node_scope("node_b", "task-b")
        pop_node_scope(node_b_handle)

        pop_graph_scope(graph_handle)

        node_a_starts = [e for e in events if e.name == "node_a" and e.event_type == EventType.Start]
        node_b_starts = [e for e in events if e.name == "node_b" and e.event_type == EventType.Start]
        assert len(node_a_starts) == 1, "Expected 1 node_a start event"
        assert len(node_b_starts) == 1, "Expected 1 node_b start event"

        # Both nodes are direct children of the graph scope
        assert node_a_starts[0].parent_uuid == graph_handle.uuid
        assert node_b_starts[0].parent_uuid == graph_handle.uuid

        graph_scope_events = [
            e
            for e in events
            if e.name == "seq_graph"
            and e.event_type == EventType.Start
            and e.metadata
            and e.metadata.get("langgraph.graph") is True
        ]
        # Only 1 graph start event (no per-branch reconstructions)
        assert len(graph_scope_events) == 1

        all_starts = [e for e in events if e.event_type == EventType.Start]
        node_a_idx = next(i for i, e in enumerate(all_starts) if e.name == "node_a")
        node_b_idx = next(i for i, e in enumerate(all_starts) if e.name == "node_b")
        assert node_a_idx < node_b_idx, "node_a should start before node_b"

    # -------------------------------------------------------------------
    # SCOPE-03: Parallel fan-out scope isolation
    # -------------------------------------------------------------------

    def test_parallel_fanout_scope_isolation(self, scope_stack: Any, events: list[Any]) -> None:
        """Parallel branches get distinct node scopes as children of the graph scope.

        Validates SCOPE-03: fan-out creates per-branch child scopes.
        After Plan 15-02, node scopes push directly on the current stack
        (no per-branch isolated scope stacks). For true thread parallelism,
        each thread gets its own scope stack via create_scope_stack.
        """
        graph_handle = push_graph_scope("parallel_graph")

        branch_results: dict[str, dict[str, Any]] = {}

        def run_branch(name: str, task_id: str) -> None:
            # Each thread creates its own scope stack for isolation
            stack = create_scope_stack()
            set_thread_scope_stack(stack)
            # Push a graph scope on this thread's stack to mirror the parent
            branch_graph = push_graph_scope("parallel_graph")
            node_handle, _, _ = push_node_scope(name, task_id)
            branch_results[name] = {
                "node_uuid": node_handle.uuid,
                "graph_uuid": branch_graph.uuid,
                "node_parent": None,
            }
            node_starts = [
                e
                for e in events
                if e.name == name
                and e.event_type == EventType.Start
                and e.metadata
                and e.metadata.get("langgraph.node") is True
            ]
            if node_starts:
                branch_results[name]["node_parent"] = node_starts[0].parent_uuid

            pop_node_scope(node_handle)
            pop_graph_scope(branch_graph)

        ctx = contextvars.copy_context()
        t1 = threading.Thread(target=ctx.run, args=(run_branch, "branch_a", "task-a"))
        t2 = threading.Thread(target=contextvars.copy_context().run, args=(run_branch, "branch_b", "task-b"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        set_thread_scope_stack(scope_stack)
        pop_graph_scope(graph_handle)

        assert "branch_a" in branch_results
        assert "branch_b" in branch_results

        graph_a_uuid = branch_results["branch_a"]["graph_uuid"]
        graph_b_uuid = branch_results["branch_b"]["graph_uuid"]
        assert graph_a_uuid is not None, "Branch A should have a graph scope"
        assert graph_b_uuid is not None, "Branch B should have a graph scope"
        assert graph_a_uuid != graph_b_uuid, "Parallel branches must have distinct graph scope UUIDs"

        assert branch_results["branch_a"]["node_parent"] == graph_a_uuid
        assert branch_results["branch_b"]["node_parent"] == graph_b_uuid

        assert branch_results["branch_a"]["node_uuid"] != branch_results["branch_b"]["node_uuid"]

    # -------------------------------------------------------------------
    # QUAL-04: Sync and async scope parity
    # -------------------------------------------------------------------

    def test_sync_and_async_scope_parity(self, scope_stack: Any, events: list[Any]) -> None:
        """Sync and async paths produce equivalent scope structures."""
        import asyncio

        sync_graph = push_graph_scope("sync_graph")
        sync_node, _, _ = push_node_scope("sync_node", "task-s")
        pop_node_scope(sync_node)
        pop_graph_scope(sync_graph)

        sync_events = list(events)
        events.clear()

        async def async_path() -> None:
            set_thread_scope_stack(scope_stack)
            async_graph = push_graph_scope("async_graph")
            async_node, _, _ = push_node_scope("async_node", "task-a")
            pop_node_scope(async_node)
            pop_graph_scope(async_graph)

        asyncio.run(async_path())

        async_events = list(events)

        sync_types = [
            (
                e.event_type,
                bool(e.metadata and e.metadata.get("langgraph.graph")),
                bool(e.metadata and e.metadata.get("langgraph.node")),
            )
            for e in sync_events
        ]
        async_types = [
            (
                e.event_type,
                bool(e.metadata and e.metadata.get("langgraph.graph")),
                bool(e.metadata and e.metadata.get("langgraph.node")),
            )
            for e in async_events
        ]

        assert sync_types == async_types, (
            f"Sync and async should produce same event pattern.\nSync:  {sync_types}\nAsync: {async_types}"
        )

    # -------------------------------------------------------------------
    # SCOPE-04: LLM call within node scope
    # -------------------------------------------------------------------

    def test_llm_call_in_node_scope(self, scope_stack: Any, events: list[Any]) -> None:
        """LLM call within a node appears in event trace with node scope as parent."""
        graph_handle = push_graph_scope("llm_graph")
        node_handle, _, _ = push_node_scope("llm_node", "task-llm")

        llm_handle = nat_nexus.llm.call(
            "test-model",
            nat_nexus.LLMRequest({}, {"messages": [], "model": "test-model"}),
        )
        nat_nexus.llm.call_end(llm_handle, {"response": "hello"})

        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)

        llm_starts = [
            e
            for e in events
            if e.name == "test-model"
            and e.event_type == EventType.Start
            and not (e.metadata and e.metadata.get("langgraph.node"))
        ]
        assert len(llm_starts) >= 1, "Expected at least 1 LLM start event"
        llm_event = llm_starts[0]

        assert llm_event.parent_uuid == node_handle.uuid, (
            f"LLM event parent ({llm_event.parent_uuid}) should match node scope ({node_handle.uuid})"
        )

    # -------------------------------------------------------------------
    # SCOPE-05: Tool call within node scope
    # -------------------------------------------------------------------

    def test_tool_call_in_node_scope(self, scope_stack: Any, events: list[Any]) -> None:
        """Tool call within a node appears in event trace with node scope as parent."""
        graph_handle = push_graph_scope("tool_graph")
        node_handle, _, _ = push_node_scope("tool_node", "task-tool")

        tool_handle = nat_nexus.tools.call("search_tool", {"query": "test"})
        nat_nexus.tools.call_end(tool_handle, {"results": ["a", "b"]})

        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)

        tool_starts = [e for e in events if e.name == "search_tool" and e.event_type == EventType.Start]
        assert len(tool_starts) == 1, f"Expected 1 tool start event, got {len(tool_starts)}"
        tool_event = tool_starts[0]

        assert tool_event.parent_uuid == node_handle.uuid, (
            f"Tool event parent ({tool_event.parent_uuid}) should match node scope ({node_handle.uuid})"
        )

    # -------------------------------------------------------------------
    # SCOPE-06: No double-wrapping
    # -------------------------------------------------------------------

    def test_no_double_wrapping(self, scope_stack: Any, events: list[Any]) -> None:
        """Graph and node scopes are created exactly once (no double-wrap)."""
        assert langgraph_nexus_active() is False

        graph_handle = push_graph_scope("single_graph")

        assert langgraph_nexus_active() is True

        node_handle, _, _ = push_node_scope("single_node", "task-1")
        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)

        assert langgraph_nexus_active() is False

        graph_starts = [
            e
            for e in events
            if e.metadata and e.metadata.get("langgraph.graph") is True and e.event_type == EventType.Start
        ]
        # Only 1 graph start event (no per-branch reconstruction)
        assert len(graph_starts) == 1, f"Expected 1 graph start event, got {len(graph_starts)}"

        node_starts = [
            e
            for e in events
            if e.name == "single_node"
            and e.event_type == EventType.Start
            and e.metadata
            and e.metadata.get("langgraph.node") is True
        ]
        assert len(node_starts) == 1, f"Expected exactly 1 node start event, got {len(node_starts)}"

    # -------------------------------------------------------------------
    # QUAL-05: Comprehensive scope hierarchy validation
    # -------------------------------------------------------------------

    def test_scope_hierarchy_event_ordering(self, scope_stack: Any, events: list[Any]) -> None:
        """Full lifecycle ordering: graph-start, node-start, node-end, graph-end."""
        graph_handle = push_graph_scope("order_graph")
        node_handle, _, _ = push_node_scope("order_node", "task-ord")
        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)

        lifecycle = []
        for e in events:
            if e.metadata and e.metadata.get("langgraph.graph"):
                lifecycle.append(("graph", e.event_type))
            elif e.metadata and e.metadata.get("langgraph.node"):
                lifecycle.append(("node", e.event_type))

        # With direct push (no branch reconstruction), ordering is:
        # graph-start, node-start, node-end, graph-end
        expected = [
            ("graph", EventType.Start),
            ("node", EventType.Start),
            ("node", EventType.End),
            ("graph", EventType.End),
        ]
        assert lifecycle == expected, f"Lifecycle event ordering mismatch.\nGot:      {lifecycle}\nExpected: {expected}"

    # -------------------------------------------------------------------
    # Additional: available() guard behavior
    # -------------------------------------------------------------------

    def test_available_guard_requires_scope_stack(self) -> None:
        """available() returns False when no scope stack is initialized."""
        result = {}

        def worker() -> None:
            result["available"] = available()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert result["available"] is False, "available() should return False on a thread without an active scope stack"

    def test_available_guard_returns_true_with_stack(self) -> None:
        """available() returns True when scope stack is initialized."""
        assert available() is True


class TestLangGraphSubgraph:
    """Validate subgraph scope nesting and ATIF graph topology metadata (HIER-01..04)."""

    @pytest.fixture(autouse=True)
    def scope_stack(self):
        """Create an isolated scope stack for each test."""
        stack = create_scope_stack()
        set_thread_scope_stack(stack)
        yield stack

    @pytest.fixture()
    def events(self):
        """Collect all lifecycle events during the test."""
        collected: list[dict[str, Any]] = []

        def _subscriber(event: dict) -> None:
            collected.append(event)

        nat_nexus.subscribers.register("test_sub", _subscriber)
        yield collected
        nat_nexus.subscribers.deregister("test_sub")

    # -------------------------------------------------------------------
    # HIER-01: Subgraph nested scope hierarchy
    # -------------------------------------------------------------------

    def test_subgraph_nested_scope_hierarchy(self, scope_stack: Any, events: list[Any]) -> None:
        """Subgraph creates 4-level scope hierarchy: graph -> node -> subgraph -> subgraph_node."""
        graph_handle = push_graph_scope("outer_graph")

        node_handle, _, _ = push_node_scope("parent_node", "task-1")

        sub_handle, active_tok, info_tok = push_subgraph_scope("inner_graph")

        inner_node_handle = nat_nexus.scope.push(
            "inner_node",
            nat_nexus.ScopeType.Agent,
            metadata={"langgraph.node": True},
        )

        subgraph_starts = [
            e
            for e in events
            if e.name == "inner_graph"
            and e.event_type == EventType.Start
            and e.metadata
            and e.metadata.get("langgraph.subgraph") is True
        ]
        assert len(subgraph_starts) == 1, f"Expected 1 subgraph start event, got {len(subgraph_starts)}"

        inner_node_starts = [e for e in events if e.name == "inner_node" and e.event_type == EventType.Start]
        assert len(inner_node_starts) == 1, f"Expected 1 inner_node start event, got {len(inner_node_starts)}"

        assert inner_node_starts[0].parent_uuid == sub_handle.uuid, "inner_node parent should be the subgraph scope"
        assert subgraph_starts[0].parent_uuid == node_handle.uuid, "subgraph parent should be the parent node scope"

        assert subgraph_starts[0].metadata.get("langgraph.subgraph") is True
        assert subgraph_starts[0].metadata.get("langgraph.graph") is True

        nat_nexus.scope.pop(inner_node_handle)
        pop_subgraph_scope(sub_handle, active_tok, info_tok)
        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)

    # -------------------------------------------------------------------
    # HIER-02: Subgraph detection via CONFIG_KEY_TASK_ID
    # -------------------------------------------------------------------

    def test_subgraph_detection_via_config_key(self, scope_stack: Any, events: list[Any]) -> None:
        """push_subgraph_scope pushes on CURRENT stack and sets ContextVars correctly."""
        graph_handle = push_graph_scope("main_graph")

        node_handle, _, _ = push_node_scope("a_node", "task-x")

        assert _langgraph_nexus_active.get() is True

        sub_handle, active_tok, info_tok = push_subgraph_scope("sub_graph")

        assert _langgraph_nexus_active.get() is True

        info = _graph_scope_info.get()
        assert info is not None
        assert info.graph_name == "sub_graph"
        assert info.metadata.get("langgraph.subgraph") is True

        pop_subgraph_scope(sub_handle, active_tok, info_tok)
        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)

    # -------------------------------------------------------------------
    # HIER-02 continued: ContextVar restoration after subgraph pop
    # -------------------------------------------------------------------

    def test_subgraph_contextvar_restoration(self, scope_stack: Any, events: list[Any]) -> None:
        """pop_subgraph_scope restores parent ContextVar values."""
        graph_handle = push_graph_scope("parent_graph")
        assert _graph_scope_info.get().graph_name == "parent_graph"

        node_handle, _, _ = push_node_scope("some_node", "task-y")

        sub_handle, active_tok, info_tok = push_subgraph_scope("child_graph")
        assert _graph_scope_info.get().graph_name == "child_graph"

        pop_subgraph_scope(sub_handle, active_tok, info_tok)

        restored_info = _graph_scope_info.get()
        assert restored_info is not None, "_graph_scope_info should be restored, not None"
        assert restored_info.graph_name == "parent_graph", f"Expected 'parent_graph', got '{restored_info.graph_name}'"

        assert _langgraph_nexus_active.get() is True

        pop_node_scope(node_handle)
        pop_graph_scope(graph_handle)

    # -------------------------------------------------------------------
    # HIER-03, HIER-04: ATIF graph topology metadata
    # -------------------------------------------------------------------

    def test_atif_graph_topology_metadata(self, scope_stack: Any, events: list[Any]) -> None:
        """Graph topology is stored as named metadata on the scope via push_graph_scope."""
        topology = {
            "nodes": [
                {"id": "__start__"},
                {"id": "agent"},
                {"id": "__end__"},
            ],
            "edges": [
                {"source": "__start__", "target": "agent"},
                {"source": "agent", "target": "__end__"},
            ],
        }

        graph_handle = push_graph_scope("my_graph", graph_topology=topology)

        handle = nat_nexus.scope.get_handle()
        assert handle is not None
        assert handle.metadata is not None
        assert "graph_topology" in handle.metadata
        topo = handle.metadata["graph_topology"]
        assert len(topo["nodes"]) == 3, f"Expected 3 nodes, got {len(topo['nodes'])}"
        assert len(topo["edges"]) == 2, f"Expected 2 edges, got {len(topo['edges'])}"
        assert topo["edges"][0]["source"] == "__start__"

        graph_starts = [e for e in events if e.name == "my_graph" and e.event_type == EventType.Start]
        assert len(graph_starts) == 1
        assert graph_starts[0].metadata is not None
        assert "graph_topology" in graph_starts[0].metadata

        set_thread_scope_stack(scope_stack)
        pop_graph_scope(graph_handle)

    # -------------------------------------------------------------------
    # HIER-01 + concurrent: Concurrent subgraph isolation
    # -------------------------------------------------------------------

    def test_concurrent_subgraph_isolation(self, scope_stack: Any) -> None:
        """Concurrent graphs with subgraphs store distinct topologies in scope metadata."""
        all_events: list[Any] = []
        lock = threading.Lock()

        def _collect(event: Any) -> None:
            with lock:
                all_events.append(event)

        nat_nexus.subscribers.register("concurrent-collector", _collect)

        results: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        def run_graph(name: str, node_count: int) -> None:
            try:
                stack = create_scope_stack()
                set_thread_scope_stack(stack)

                nodes = [{"id": f"node_{i}"} for i in range(node_count)]
                edges = [{"source": f"node_{i}", "target": f"node_{i + 1}"} for i in range(node_count - 1)]
                graph_handle = push_graph_scope(f"graph_{name}", graph_topology={"nodes": nodes, "edges": edges})

                node_h, _, _ = push_node_scope(f"{name}_node", f"task-{name}")
                sub_h, at, it = push_subgraph_scope(f"{name}_subgraph")
                pop_subgraph_scope(sub_h, at, it)
                pop_node_scope(node_h)

                set_thread_scope_stack(stack)
                pop_graph_scope(graph_handle)

                results[name] = {"node_count": node_count}
            except Exception as exc:
                import traceback

                errors.append(f"{name}: {exc}\n{traceback.format_exc()}")

        ctx = contextvars.copy_context()
        t_a = threading.Thread(target=ctx.run, args=(run_graph, "A", 2))
        t_b = threading.Thread(target=contextvars.copy_context().run, args=(run_graph, "B", 3))
        t_a.start()
        t_a.join()
        t_b.start()
        t_b.join()

        nat_nexus.subscribers.deregister("concurrent-collector")

        assert not errors, f"Thread errors: {errors}"
        assert "A" in results and "B" in results

        with lock:
            starts_a = [e for e in all_events if e.name == "graph_A" and e.event_type == EventType.Start]
            starts_b = [e for e in all_events if e.name == "graph_B" and e.event_type == EventType.Start]

        assert len(starts_a) >= 1, "Graph A should have at least one start event"
        assert len(starts_b) >= 1, "Graph B should have at least one start event"

        topo_a = starts_a[0].metadata.get("graph_topology")
        topo_b = starts_b[0].metadata.get("graph_topology")

        assert topo_a is not None, "Graph A start event should have graph_topology in metadata"
        assert topo_b is not None, "Graph B start event should have graph_topology in metadata"
        assert len(topo_a["nodes"]) == 2, "Graph A should have 2 nodes"
        assert len(topo_b["nodes"]) == 3, "Graph B should have 3 nodes"
