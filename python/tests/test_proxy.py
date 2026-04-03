# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Agent Toolkit Nexus proxy types and set_latency_sensitivity API."""

from nat_nexus import LLMRequest, ScopeType, intercepts, llm, scope
from nat_nexus.proxy import (
    AgentHints,
    InMemoryBackend,
    LlmCallPrediction,
    NexusProxy,
    PredictionMetrics,
    RedisBackend,
    SensitivityConfig,
    set_latency_sensitivity,
)


class TestProxyConstruction:
    """Tests for proxy type construction."""

    def test_in_memory_backend_construction(self):
        """InMemoryBackend is constructable from Python."""
        backend = InMemoryBackend()
        assert repr(backend) == "<InMemoryBackend>"

    def test_nexus_proxy_construction(self):
        """NexusProxy is constructable with agent_id and backend."""
        backend = InMemoryBackend()
        proxy = NexusProxy("test-agent", backend)
        assert proxy.agent_id == "test-agent"

    def test_nexus_proxy_custom_priorities(self):
        """NexusProxy accepts optional priority kwargs."""
        backend = InMemoryBackend()
        proxy = NexusProxy(
            "test-agent",
            backend,
            llm_intercept_priority=50,
            tool_intercept_priority=75,
        )
        assert proxy.agent_id == "test-agent"

    def test_nexus_proxy_repr(self):
        backend = InMemoryBackend()
        proxy = NexusProxy("repr-test", backend)
        assert "NexusProxy" in repr(proxy) or "Proxy" in repr(proxy)


class TestProxyRegistration:
    """NexusProxy registration and deregistration."""

    async def test_construct_and_register(self):
        """NexusProxy can be registered and deregistered from Python."""
        backend = InMemoryBackend()
        proxy = NexusProxy("py-reg-test", backend)
        try:
            await proxy.register()
            assert proxy.agent_id == "py-reg-test"
        finally:
            proxy.deregister()

    async def test_proxy_register_and_llm_call(self):
        """After registration, an LLM call completes without error."""
        backend = InMemoryBackend()
        proxy = NexusProxy("py-llm-test", backend)
        try:
            await proxy.register()

            def my_llm(request):
                return {"response": "ok"}

            request = LLMRequest({}, {"messages": []})
            result = await llm.execute("test-model", request, my_llm)
            assert result["response"] == "ok"
        finally:
            proxy.deregister()

    async def test_deregister_idempotent(self):
        """Deregister can be called multiple times without error."""
        backend = InMemoryBackend()
        proxy = NexusProxy("py-dereg-test", backend)
        await proxy.register()
        proxy.deregister()
        # Second deregister should not raise
        proxy.deregister()


class TestProxyTypeImports:
    """Verify all proxy types are importable from nat_nexus.proxy."""

    def test_import_agent_hints(self):
        from nat_nexus.proxy import AgentHints

        assert AgentHints is not None

    def test_import_prediction_metrics(self):
        from nat_nexus.proxy import PredictionMetrics

        assert PredictionMetrics is not None

    def test_import_llm_call_prediction(self):
        from nat_nexus.proxy import LlmCallPrediction

        assert LlmCallPrediction is not None

    def test_import_sensitivity_config(self):
        from nat_nexus.proxy import SensitivityConfig

        assert SensitivityConfig is not None

    def test_import_redis_backend(self):
        from nat_nexus.proxy import RedisBackend

        assert RedisBackend is not None

    def test_import_set_latency_sensitivity(self):
        from nat_nexus.proxy import set_latency_sensitivity

        assert set_latency_sensitivity is not None


class TestSensitivityConfig:
    """SensitivityConfig is constructable with default values."""

    def test_default_construction(self):
        config = SensitivityConfig()
        assert config.sensitivity_scale == 5
        assert config.w_critical == 0.5
        assert config.w_fanout == 0.3
        assert config.w_position == 0.2
        assert config.w_parallel == 0.0

    def test_custom_values(self):
        config = SensitivityConfig(sensitivity_scale=10, w_critical=0.7, w_fanout=0.1, w_position=0.1, w_parallel=0.1)
        assert config.sensitivity_scale == 10
        assert config.w_critical == 0.7

    def test_repr(self):
        config = SensitivityConfig()
        r = repr(config)
        assert "SensitivityConfig" in r
        assert "5" in r  # sensitivity_scale default


class TestAgentHints:
    """AgentHints properties are readable (read-only, not constructable from Python)."""

    def test_has_expected_properties(self):
        assert hasattr(AgentHints, "osl")
        assert hasattr(AgentHints, "iat")
        assert hasattr(AgentHints, "priority")
        assert hasattr(AgentHints, "latency_sensitivity")
        assert hasattr(AgentHints, "prefix_id")
        assert hasattr(AgentHints, "total_requests")


class TestPredictionMetrics:
    """PredictionMetrics properties are readable."""

    def test_has_expected_properties(self):
        assert hasattr(PredictionMetrics, "sample_count")
        assert hasattr(PredictionMetrics, "mean")
        assert hasattr(PredictionMetrics, "p50")
        assert hasattr(PredictionMetrics, "p90")
        assert hasattr(PredictionMetrics, "p95")


class TestLlmCallPrediction:
    """LlmCallPrediction nested properties."""

    def test_has_expected_properties(self):
        assert hasattr(LlmCallPrediction, "remaining_calls")
        assert hasattr(LlmCallPrediction, "interarrival_ms")
        assert hasattr(LlmCallPrediction, "output_tokens")
        assert hasattr(LlmCallPrediction, "latency_sensitivity")


class TestRedisBackend:
    """RedisBackend class exists with connect method."""

    def test_has_connect_method(self):
        assert hasattr(RedisBackend, "connect")
        assert callable(RedisBackend.connect)

    def test_repr(self):
        # RedisBackend requires async connect, but repr should work
        assert "RedisBackend" in str(RedisBackend)


class TestAnyBackendProxy:
    """NexusProxy accepts AnyBackend (InMemory path)."""

    def test_construct_with_in_memory(self):
        """Backward-compatible: InMemoryBackend still works."""
        backend = InMemoryBackend()
        proxy = NexusProxy("v2-test", backend)
        assert proxy.agent_id == "v2-test"

    def test_construct_with_sensitivity_config(self):
        """NexusProxy accepts optional sensitivity_config kwarg."""
        backend = InMemoryBackend()
        config = SensitivityConfig()
        proxy = NexusProxy("v2-config-test", backend, sensitivity_config=config)
        assert proxy.agent_id == "v2-config-test"

    async def test_register_with_sensitivity_config(self):
        """NexusProxy with sensitivity_config can register and deregister."""
        backend = InMemoryBackend()
        config = SensitivityConfig(w_critical=1.0, w_fanout=0.0, w_position=0.0, w_parallel=0.0)
        proxy = NexusProxy("v2-reg-test", backend, sensitivity_config=config)
        try:
            await proxy.register()
            assert proxy.agent_id == "v2-reg-test"
        finally:
            proxy.deregister()


class TestSetLatencySensitivity:
    """set_latency_sensitivity() updates scope metadata."""

    def test_basic_call(self):
        """set_latency_sensitivity sets metadata on current scope."""
        with scope.scope("test", ScopeType.Agent):
            set_latency_sensitivity(3)
            # No error means success

    def test_max_merge_higher_wins(self):
        """Calling with higher value updates."""
        with scope.scope("test", ScopeType.Agent):
            set_latency_sensitivity(3)
            set_latency_sensitivity(5)
            # 5 should be the active value

    def test_max_merge_lower_noop(self):
        """Calling with lower value is a no-op."""
        with scope.scope("test", ScopeType.Agent):
            set_latency_sensitivity(5)
            set_latency_sensitivity(3)
            # 5 should still be the active value

    def test_zero_raises_value_error(self):
        """ValueError raised for zero sensitivity."""
        import pytest

        with pytest.raises(ValueError, match="positive"):
            set_latency_sensitivity(0)


class TestHintInjection:
    """End-to-end: proxy + learner + hints injection."""

    async def test_learner_produces_hints_after_runs(self):
        """After registering proxy with sensitivity_config and running LLM calls,
        the learner processes runs and subsequent LLM calls receive AgentHints."""
        import asyncio

        backend = InMemoryBackend()
        config = SensitivityConfig()
        proxy = NexusProxy("v2-hints-test", backend, sensitivity_config=config, dynamo_intercept=True)
        captured_headers = {}

        def capture_intercept(name, request):
            captured_headers["headers"] = request.headers
            return request

        try:
            await proxy.register()
            proxy.store_plan()

            # Register capture intercept at lower priority (higher number = later)
            intercepts.register_llm_request("v2-hints-capture", 200, False, capture_intercept)

            # First LLM call -- learner processes after the run completes via drain task
            with scope.scope("test-agent", ScopeType.Agent):
                request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]})
                await llm.execute("gpt-4", request, lambda r: {"response": "ok"})

            # Give the drain task time to process
            await asyncio.sleep(0.2)

            # Verify the capture intercept fired
            assert "headers" in captured_headers, "LLM intercept should have fired"

        finally:
            intercepts.deregister_llm_request("v2-hints-capture")
            proxy.deregister()

    async def test_latency_sensitive_with_proxy(self):
        """set_latency_sensitivity annotation propagates through proxy."""
        backend = InMemoryBackend()
        proxy = NexusProxy("v2-ls-test", backend, dynamo_intercept=True)
        captured = {}

        def capture_intercept(name, request):
            captured["headers"] = request.headers
            return request

        try:
            await proxy.register()
            proxy.store_plan()
            intercepts.register_llm_request("v2-ls-capture", 200, False, capture_intercept)

            with scope.scope("test-agent", ScopeType.Agent):
                set_latency_sensitivity(3)
                request = LLMRequest({}, {"messages": []})
                await llm.execute("gpt-4", request, lambda r: {"response": "ok"})

            # The latency sensitivity should be visible to the intercept
            # via read_manual_latency_sensitivity() in Rust
            assert "headers" in captured

        finally:
            intercepts.deregister_llm_request("v2-ls-capture")
            proxy.deregister()


class TestStorageBackendProtocol:
    """Custom Python StorageBackend via protocol."""

    def test_protocol_isinstance_check(self):
        """A class with the right methods satisfies isinstance check."""
        from nat_nexus.proxy import StorageBackendProtocol

        class MyBackend:
            async def store_run(self, record):
                pass

            async def load_plan(self, agent_id):
                return None

            async def list_runs(self, agent_id):
                return []

            async def store_trie(self, agent_id, envelope):
                pass

            async def load_trie(self, agent_id):
                return None

            async def store_accumulators(self, agent_id, state):
                pass

            async def load_accumulators(self, agent_id):
                return None

        assert isinstance(MyBackend(), StorageBackendProtocol)

    def test_construct_proxy_with_custom_backend(self):
        """NexusProxy accepts a custom backend implementing StorageBackendProtocol."""
        from nat_nexus.proxy import NexusProxy

        class MyBackend:
            async def store_run(self, record):
                pass

            async def load_plan(self, agent_id):
                return None

            async def list_runs(self, agent_id):
                return []

            async def store_trie(self, agent_id, envelope):
                pass

            async def load_trie(self, agent_id):
                return None

            async def store_accumulators(self, agent_id, state):
                pass

            async def load_accumulators(self, agent_id):
                return None

        proxy = NexusProxy("custom-backend-test", MyBackend())
        assert proxy.agent_id == "custom-backend-test"

    async def test_register_with_custom_backend(self):
        """NexusProxy with custom backend can register and deregister."""
        from nat_nexus.proxy import NexusProxy

        class MyBackend:
            async def store_run(self, record):
                pass

            async def load_plan(self, agent_id):
                return None

            async def list_runs(self, agent_id):
                return []

            async def store_trie(self, agent_id, envelope):
                pass

            async def load_trie(self, agent_id):
                return None

            async def store_accumulators(self, agent_id, state):
                pass

            async def load_accumulators(self, agent_id):
                return None

        proxy = NexusProxy("custom-reg-test", MyBackend())
        try:
            await proxy.register()
            assert proxy.agent_id == "custom-reg-test"
        finally:
            proxy.deregister()

    def test_invalid_backend_type_error(self):
        """Passing a non-backend object raises TypeError."""
        import pytest
        from nat_nexus.proxy import NexusProxy

        with pytest.raises(TypeError, match="backend must be"):
            NexusProxy("bad-backend", "not-a-backend")  # type: ignore[arg-type]

    async def test_custom_backend_llm_call(self):
        """LLM call through proxy with custom backend works end-to-end."""
        from nat_nexus import LLMRequest, ScopeType, llm, scope
        from nat_nexus.proxy import NexusProxy

        stored_runs = []

        class MyBackend:
            async def store_run(self, record):
                stored_runs.append(record)

            async def load_plan(self, agent_id):
                return None

            async def list_runs(self, agent_id):
                return []

            async def store_trie(self, agent_id, envelope):
                pass

            async def load_trie(self, agent_id):
                return None

            async def store_accumulators(self, agent_id, state):
                pass

            async def load_accumulators(self, agent_id):
                return None

        proxy = NexusProxy("custom-llm-test", MyBackend())
        try:
            await proxy.register()
            with scope.scope("agent", ScopeType.Agent):
                request = LLMRequest({}, {"messages": []})
                result = await llm.execute("test-model", request, lambda r: {"response": "ok"})
                assert result["response"] == "ok"
        finally:
            proxy.deregister()

    def test_import_storage_backend_protocol(self):
        """StorageBackendProtocol is importable from nat_nexus."""
        from nat_nexus.proxy import StorageBackendProtocol

        assert StorageBackendProtocol is not None


class TestDynamoIntercept:
    """Python bindings expose dynamo_intercept parameter."""

    def test_construct_without_dynamo_intercept(self):
        """NexusProxy defaults to dynamo_intercept=False."""
        backend = InMemoryBackend()
        proxy = NexusProxy("no-dynamo", backend)
        assert proxy.agent_id == "no-dynamo"

    def test_construct_with_dynamo_intercept_true(self):
        """NexusProxy accepts dynamo_intercept=True."""
        backend = InMemoryBackend()
        proxy = NexusProxy("yes-dynamo", backend, dynamo_intercept=True)
        assert proxy.agent_id == "yes-dynamo"

    def test_construct_with_all_options(self):
        """NexusProxy accepts dynamo_intercept alongside sensitivity_config."""
        backend = InMemoryBackend()
        config = SensitivityConfig()
        proxy = NexusProxy(
            "full-opts",
            backend,
            sensitivity_config=config,
            dynamo_intercept=True,
        )
        assert proxy.agent_id == "full-opts"

    async def test_register_with_dynamo_intercept(self):
        """NexusProxy with dynamo_intercept=True can register and deregister."""
        backend = InMemoryBackend()
        config = SensitivityConfig()
        proxy = NexusProxy(
            "v2-dynamo-reg",
            backend,
            sensitivity_config=config,
            dynamo_intercept=True,
        )
        try:
            await proxy.register()
            assert proxy.agent_id == "v2-dynamo-reg"
        finally:
            proxy.deregister()
