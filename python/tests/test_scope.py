# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVMagic scope operations."""

import pytest
from nvmagic import (
    ScopeAttributes,
    ScopeHandle,
    ScopeType,
    scope,
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
