# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Relay scope operations."""

import asyncio

import pytest

from nemo_relay import (
    LogSeverity,
    MetricKind,
    MetricMeasurement,
    MetricValueType,
    ScopeAttributes,
    ScopeHandle,
    ScopeType,
    scope,
    subscribers,
)


class TestScope:
    def test_get_handle_returns_root(self):
        handle = scope.get_handle()
        assert isinstance(handle, ScopeHandle)
        assert handle.name == "root"

    def test_push_and_pop(self):
        handle = scope.push("test_scope", ScopeType.Agent)
        assert handle.name == "test_scope"
        assert scope.get_handle().name == "test_scope"
        scope.pop(handle)
        assert scope.get_handle().name == "root"

    def test_push_with_attributes(self):
        attrs = ScopeAttributes(ScopeAttributes.PARALLEL)
        handle = scope.push("parallel", ScopeType.Function, attributes=attrs)
        assert handle.name == "parallel"
        assert handle.attributes.is_parallel
        scope.pop(handle)

    def test_push_with_parent(self):
        parent = scope.push("parent", ScopeType.Agent)
        child = scope.push("child", ScopeType.Function, handle=parent)
        assert child.parent_uuid == parent.uuid
        scope.pop(child)
        scope.pop(parent)

    def test_nested_scopes(self):
        s1 = scope.push("level1", ScopeType.Agent)
        s2 = scope.push("level2", ScopeType.Function)
        s3 = scope.push("level3", ScopeType.Tool)
        assert scope.get_handle().name == "level3"
        scope.pop(s3)
        assert scope.get_handle().name == "level2"
        scope.pop(s2)
        assert scope.get_handle().name == "level1"
        scope.pop(s1)
        assert scope.get_handle().name == "root"

    def test_scope_handle_properties(self):
        handle = scope.push("props_test", ScopeType.Retriever)
        assert handle.uuid is not None
        assert handle.name == "props_test"
        assert handle.scope_type == ScopeType.Retriever
        scope.pop(handle)

    def test_event_emission(self):
        scope.event("my_mark")  # Should not raise

    def test_event_with_data(self):
        scope.event("data_mark", data={"key": "value"}, metadata={"version": 1})

    def test_event_with_handle(self):
        handle = scope.push("evt_scope", ScopeType.Agent)
        scope.event("scoped_mark", handle=handle)
        scope.pop(handle)

    def test_event_with_data_schema_and_severity(self, subscribed_events):
        scope.event(
            "structured_log",
            data={"message": "ready"},
            data_schema={"name": "example.log", "version": "1"},
            metadata={"source": "python"},
            severity=LogSeverity.Warn,
        )
        subscribers.flush()

        mark = next(event for event in subscribed_events if event.name == "structured_log")
        assert mark.data_schema == {"name": "example.log", "version": "1"}
        assert mark.severity == LogSeverity.Warn
        assert mark.metadata == {
            "source": "python",
            "nemo_relay.log.severity": "warn",
        }

    def test_metric_emits_validated_metric_mark(self, subscribed_events):
        measurement = MetricMeasurement(
            "relay.tokens",
            MetricKind.Counter,
            MetricValueType.U64,
            12,
            unit="{token}",
            attributes={"model": "test"},
        )

        scope.metric("token_usage", [measurement], metadata={"source": "python"})
        subscribers.flush()

        mark = next(event for event in subscribed_events if event.name == "token_usage")
        assert mark.data_schema == {"name": "nemo.relay.metric_measurements", "version": "1"}
        assert mark.data == {
            "measurements": [
                {
                    "name": "relay.tokens",
                    "kind": "counter",
                    "value_type": "u64",
                    "value": 12,
                    "unit": "{token}",
                    "description": None,
                    "attributes": {"model": "test"},
                    "boundaries": None,
                }
            ]
        }

    def test_metric_rejects_invalid_measurements_atomically(self, subscribed_events):
        valid = MetricMeasurement(
            "relay.requests",
            MetricKind.Counter,
            MetricValueType.U64,
            1,
        )
        invalid = MetricMeasurement(
            "relay.tokens",
            MetricKind.Counter,
            MetricValueType.I64,
            -1,
        )

        with pytest.raises(ValueError, match="does not support value_type i64"):
            scope.metric("invalid_metric", [valid, invalid])
        subscribers.flush()
        assert all(event.name != "invalid_metric" for event in subscribed_events)

    def test_get_handle_preserves_explicit_worker_thread_scope_stack(self):
        import threading

        import nemo_relay

        result = {}

        def worker():
            nemo_relay.set_thread_scope_stack(nemo_relay.create_scope_stack())
            result["name"] = scope.get_handle().name

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert result["name"] == "root"

    def test_pop_invalid_raises(self):
        handle = scope.push("once", ScopeType.Agent)
        scope.pop(handle)
        with pytest.raises(RuntimeError):
            scope.pop(handle)

    def test_scope_ctx_mgr(self):
        with scope.scope("test_scope", ScopeType.Agent) as handle:
            assert handle.name == "test_scope"
            assert scope.get_handle().name == "test_scope"

        assert scope.get_handle().name == "root"

    @pytest.mark.parametrize(
        ("cancel_message", "expected_status_description"),
        [
            (None, "cancelled"),
            ("shutdown", "shutdown"),
        ],
    )
    async def test_scope_ctx_mgr_cancelled_task_sets_error_status(
        self,
        subscribed_events,
        cancel_message,
        expected_status_description,
    ):
        async def cancel_within_scope() -> None:
            with scope.scope("cancelled_scope", ScopeType.Agent):
                task = asyncio.current_task()
                assert task is not None
                if cancel_message is None:
                    task.cancel()
                else:
                    task.cancel(cancel_message)
                await asyncio.sleep(0)

        with pytest.raises(asyncio.CancelledError):
            await cancel_within_scope()

        await subscribers.flush_async()
        assert len(subscribed_events) == 2
        end = subscribed_events[-1]
        assert end.name == "cancelled_scope"
        assert end.kind == "scope"
        assert end.category == "agent"
        assert end.scope_category == "end"
        assert end.metadata == {
            "otel.status_code": "ERROR",
            "otel.status_description": expected_status_description,
        }


class TestAllScopeTypes:
    @pytest.mark.parametrize(
        "scope_type",
        [
            ScopeType.Agent,
            ScopeType.Function,
            ScopeType.Tool,
            ScopeType.Llm,
            ScopeType.Retriever,
            ScopeType.Embedder,
            ScopeType.Reranker,
            ScopeType.Guardrail,
            ScopeType.Evaluator,
            ScopeType.Custom,
            ScopeType.Unknown,
        ],
    )
    def test_push_with_scope_type(self, scope_type):
        handle = scope.push(f"test_{scope_type}", scope_type)
        assert handle.name.startswith("test_")
        scope.pop(handle)
