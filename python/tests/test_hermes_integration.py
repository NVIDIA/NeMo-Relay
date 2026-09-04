# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hermes Agent integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# We mock the nemo_relay native layer since we cannot build the Rust core
# in a pure-Python test environment.

MOCK_HANDLE = MagicMock(name="ScopeHandle")


@pytest.fixture(autouse=True)
def _mock_relay_native():
    """Patch nemo_relay native calls so the tests run without the Rust runtime."""
    mock_native = MagicMock()
    mock_native.push_scope.return_value = MOCK_HANDLE
    mock_stack_active = MagicMock(return_value=True)

    with patch.dict("sys.modules", {
        "nemo_relay": MagicMock(),
        "nemo_relay._native": mock_native,
        "nemo_relay._context": MagicMock(),
    }):
        # Re-import after patching so references resolve to mocks
        import nemo_relay as nr
        nr.ScopeType = MagicMock()
        nr.ScopeType.Run = "Run"
        nr.ScopeType.Turn = "Turn"
        nr.ScopeType.Function = "Function"
        nr.ScopeType.LLM = "LLM"
        nr.ScopeType.Agent = "Agent"
        nr.ScopeType.Task = "Task"
        nr.scope = MagicMock()
        nr.scope.push.return_value = MOCK_HANDLE
        nr.scope.get_handle.return_value = MOCK_HANDLE
        nr.scope_stack_active = MagicMock(return_value=True)
        nr.guardrails = MagicMock()
        nr.intercepts = MagicMock()
        nr.LLMRequest = MagicMock
        nr.AnnotatedLLMRequest = MagicMock
        nr.LLMRequestInterceptOutcome = MagicMock
        nr.ScopeHandle = MagicMock
        nr.Json = dict

        yield nr


# ---- Callback handler tests -------------------------------------------------

class TestHermesCallbackHandler:
    def _make_handler(self):
        from nemo_relay.integrations.hermes.callbacks import NemoRelayHermesCallbackHandler
        return NemoRelayHermesCallbackHandler(agent_name="test-agent")

    def test_session_scope_hierarchy(self, _mock_relay_native):
        handler = self._make_handler()
        nr = _mock_relay_native

        handler.on_session_start("sess-1")
        nr.scope.push.assert_called()
        assert nr.scope.push.call_args[0][1] == nr.ScopeType.Run

        handler.on_session_end("sess-1")
        nr.scope.pop.assert_called()

    def test_tool_call_scope(self, _mock_relay_native):
        handler = self._make_handler()
        nr = _mock_relay_native

        handler.on_tool_call_start("tc-1", "web_search", {"query": "hello"})
        nr.scope.push.assert_called()
        call_args = nr.scope.push.call_args
        assert call_args[0][1] == nr.ScopeType.Function
        assert call_args[0][0] == "web_search"

        handler.on_tool_call_end("tc-1", result="found 5 results")
        nr.scope.pop.assert_called()

    def test_subagent_scope(self, _mock_relay_native):
        handler = self._make_handler()
        nr = _mock_relay_native

        handler.on_subagent_start("sub-1", "worker")
        nr.scope.push.assert_called()
        assert nr.scope.push.call_args[0][1] == nr.ScopeType.Agent

        handler.on_subagent_end("sub-1")
        nr.scope.pop.assert_called()

    def test_llm_scope(self, _mock_relay_native):
        handler = self._make_handler()
        nr = _mock_relay_native

        handler.on_llm_call_start("llm-1", "gpt-4o", {"messages": []})
        nr.scope.push.assert_called()
        assert nr.scope.push.call_args[0][1] == nr.ScopeType.LLM

        handler.on_llm_call_end("llm-1")
        nr.scope.pop.assert_called()

    def test_task_scope(self, _mock_relay_native):
        handler = self._make_handler()
        nr = _mock_relay_native

        handler.on_task_start("task-1", "daily-healthcheck")
        nr.scope.push.assert_called()
        assert nr.scope.push.call_args[0][1] == nr.ScopeType.Task

        handler.on_task_end("task-1")
        nr.scope.pop.assert_called()

    def test_unknown_event_does_not_crash(self, _mock_relay_native):
        handler = self._make_handler()
        # Closing a scope that was never opened should be a no-op
        handler.on_session_end("nonexistent")


# ---- Middleware tests --------------------------------------------------------

class TestHermesMiddleware:
    def test_redacts_secrets(self):
        from nemo_relay.integrations.hermes.middleware import _redact_secrets
        args = {"query": "hello", "api_key": "sk-secret", "password": "hunter2"}
        result = _redact_secrets(args)
        assert result["query"] == "hello"
        assert result["api_key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"

    def test_blocks_denied_tools(self):
        from nemo_relay.integrations.hermes.middleware import NemoRelayHermesMiddleware
        mw = NemoRelayHermesMiddleware(blocked_tools=["dangerous_tool"])
        with pytest.raises(ValueError, match="blocked"):
            mw._block_denied_tools("dangerous_tool", {})

    def test_allows_permitted_tools(self):
        from nemo_relay.integrations.hermes.middleware import NemoRelayHermesMiddleware
        mw = NemoRelayHermesMiddleware(blocked_tools=["dangerous_tool"])
        result = mw._block_denied_tools("safe_tool", {"x": 1})
        assert result == {"x": 1}

    def test_payload_size_guard_rejects_oversized(self, _mock_relay_native):
        from nemo_relay.integrations.hermes.middleware import NemoRelayHermesMiddleware
        nr = _mock_relay_native
        mw = NemoRelayHermesMiddleware(max_tool_args_size=10)
        big_request = MagicMock()
        big_request.content = {"messages": ["a" * 1000]}
        with pytest.raises(ValueError, match="exceeds configured limit"):
            mw._guard_llm_payload_size("test", big_request, None)

    def test_register_idempotent(self, _mock_relay_native):
        from nemo_relay.integrations.hermes.middleware import NemoRelayHermesMiddleware
        nr = _mock_relay_native
        mw = NemoRelayHermesMiddleware()
        mw.register()
        mw.register()  # second call should be no-op
        assert nr.guardrails.register_tool_sanitize_request.call_count == 1


# ---- add_nemo_relay_integration tests ---------------------------------------

class TestAddIntegration:
    def test_injects_middleware(self):
        from nemo_relay.integrations.hermes import add_nemo_relay_integration
        result = add_nemo_relay_integration({"agent_name": "my-agent"})
        assert "middleware" in result
        assert "callbacks" in result
        assert len(result["middleware"]) == 1
        assert len(result["callbacks"]) == 1

    def test_no_duplicate_middleware(self):
        from nemo_relay.integrations.hermes import add_nemo_relay_integration
        from nemo_relay.integrations.hermes.middleware import NemoRelayHermesMiddleware
        existing = NemoRelayHermesMiddleware()
        result = add_nemo_relay_integration({"middleware": [existing]})
        assert len(result["middleware"]) == 1

    def test_instruments_subagents(self):
        from nemo_relay.integrations.hermes import add_nemo_relay_integration
        subs = [{"name": "worker", "middleware": []}]
        result = add_nemo_relay_integration({"subagents": subs})
        assert len(result["subagents"][0]["middleware"]) == 1


# ---- Subscriber tests -------------------------------------------------------

class TestHermesSubscriber:
    def test_formats_events(self):
        from nemo_relay.integrations.hermes.subscriber import HermesATOFSubscriber
        sub = HermesATOFSubscriber()
        event = {
            "event_type": "scope_start",
            "scope_name": "test",
            "scope_type": "Agent",
            "timestamp": "2026-01-01T00:00:00Z",
            "data": {"key": "value"},
        }
        sub(event)
        assert len(sub.events) == 1
        assert sub.events[0]["event_type"] == "scope_start"
        assert sub.events[0]["scope_name"] == "test"

    def test_clear(self):
        from nemo_relay.integrations.hermes.subscriber import HermesATOFSubscriber
        sub = HermesATOFSubscriber()
        sub({"event_type": "x", "scope_name": "y"})
        sub.clear()
        assert len(sub.events) == 0

    def test_unknown_event_defaults(self):
        from nemo_relay.integrations.hermes.subscriber import HermesATOFSubscriber
        sub = HermesATOFSubscriber()
        sub({})
        assert sub.events[0]["event_type"] == "unknown"
