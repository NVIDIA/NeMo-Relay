# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adaptive plugin component API."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

import pytest

from nemo_flow import AnnotatedLLMRequest, JsonObject, LLMRequest, ScopeType, llm, plugin, scope, tools
from nemo_flow import adaptive as adaptive_module
from nemo_flow.adaptive import (
    ADAPTIVE_PLUGIN_KIND,
    AcgConfig,
    AcgStabilityThresholds,
    AdaptiveConfig,
    AdaptiveHintsConfig,
    BackendSpec,
    ComponentSpec,
    StateConfig,
    TelemetryConfig,
    ToolParallelismConfig,
)


class TestAdaptiveConfigHelpers:
    def test_file_covers_native_cache_request_facts_regression(self):
        source = Path(__file__).read_text()
        runtime_call = "runtime" + ".build_cache_request_facts("
        annotated_request = "AnnotatedLLMRequest" + "("
        assert runtime_call in source
        assert annotated_request in source

    def test_backend_helpers(self):
        assert BackendSpec.in_memory().to_dict() == {"kind": "in_memory", "config": {}}
        assert BackendSpec.redis("redis://127.0.0.1:6379").to_dict() == {
            "kind": "redis",
            "config": {"url": "redis://127.0.0.1:6379", "key_prefix": "nemo_flow:"},
        }

    def test_backend_helper_normalizes_nested_dataclass_config(self):
        @dataclass
        class NestedHint:
            path: str
            enabled: bool = True

        backend = BackendSpec(
            kind="custom",
            config={"hints": [NestedHint(path="nvext.agent_hints")]},
        )

        assert backend.to_dict() == {
            "kind": "custom",
            "config": {"hints": [{"path": "nvext.agent_hints", "enabled": True}]},
        }

    def test_section_helpers(self):
        assert TelemetryConfig(learners=["latency_sensitivity"]).to_dict() == {"learners": ["latency_sensitivity"]}
        assert AdaptiveHintsConfig().to_dict()["priority"] == 100
        assert ToolParallelismConfig().to_dict()["mode"] == "observe_only"

    def test_adaptive_component_wraps_as_plugin_component(self):
        wrapped = ComponentSpec(AdaptiveConfig()).to_dict()
        assert wrapped["kind"] == ADAPTIVE_PLUGIN_KIND

    def test_validate_adaptive_plugin_component_warns_missing_state(self):
        report = plugin.validate(
            plugin.PluginConfig(components=[ComponentSpec(AdaptiveConfig(telemetry=TelemetryConfig()))])
        )
        assert any(diag["code"] == "adaptive.section_disabled_missing_state" for diag in report["diagnostics"])

    def test_plugin_component_spec_normalizes_lists_of_dataclasses(self):
        @dataclass
        class ExampleConfig:
            name: str
            weights: list[int]

        component = plugin.ComponentSpec(
            kind="python.example_plugin",
            config={"rules": [ExampleConfig(name="alpha", weights=[1, 2, 3])]},
        )

        assert component.to_dict()["config"] == {
            "rules": [{"name": "alpha", "weights": [1, 2, 3]}],
        }

    def test_set_latency_sensitivity_accepts_positive_integer(self):
        adaptive_module.set_latency_sensitivity(1)

    def test_acg_config_exposes_canonical_threshold_shape(self):
        assert [field.name for field in fields(AcgConfig)] == [
            "provider",
            "observation_window",
            "priority",
            "stability_thresholds",
        ]
        assert AcgStabilityThresholds().to_dict() == {
            "stable_threshold": 0.95,
            "semi_stable_threshold": 0.5,
            "min_observations_for_full_confidence": 20,
        }
        assert AcgConfig(provider="openai").to_dict() == {
            "provider": "openai",
            "observation_window": 100,
            "priority": 50,
            "stability_thresholds": {
                "stable_threshold": 0.95,
                "semi_stable_threshold": 0.5,
                "min_observations_for_full_confidence": 20,
            },
        }


class TestAdaptivePluginConfiguration:
    async def test_adaptive_runtime_build_cache_request_facts_uses_native_bridge(self):
        runtime = adaptive_module.AdaptiveRuntime(
            AdaptiveConfig(
                agent_id="test-adaptive-request-facts",
                state=StateConfig(backend=BackendSpec.in_memory()),
                acg=AcgConfig(provider="anthropic"),
            ).to_dict()
        )

        await runtime.register()
        try:
            facts = runtime.build_cache_request_facts(
                provider="anthropic",
                request_id="00000000-0000-0000-0000-000000000101",
                annotated_request=AnnotatedLLMRequest(
                    [
                        {"role": "system", "content": "You are a careful planner"},
                        {"role": "user", "content": "Find sources about caching"},
                    ],
                    model="claude-sonnet-4-20250514",
                ),
                agent_id="test-adaptive-request-facts",
            )
            assert facts == {
                "missing_facts": ["acg_stability_unavailable"],
                "provider": "anthropic",
                "stable_prefix_length": 0,
            }
        finally:
            runtime.deregister()
            await runtime.shutdown()

    async def test_adaptive_runtime_build_cache_request_facts_supports_openai_provider(self):
        runtime = adaptive_module.AdaptiveRuntime(
            AdaptiveConfig(
                agent_id="test-adaptive-openai-request-facts",
                state=StateConfig(backend=BackendSpec.in_memory()),
                acg=AcgConfig(provider="openai"),
            ).to_dict()
        )

        await runtime.register()
        try:
            facts = runtime.build_cache_request_facts(
                provider="openai",
                request_id="00000000-0000-0000-0000-000000000103",
                annotated_request=AnnotatedLLMRequest(
                    [
                        {"role": "system", "content": "You are a careful planner"},
                        {"role": "user", "content": "Find sources about caching"},
                    ],
                    model="gpt-4.1-mini",
                ),
                agent_id="test-adaptive-openai-request-facts",
            )
            assert facts == {
                "missing_facts": ["acg_stability_unavailable"],
                "provider": "openai",
                "stable_prefix_length": 0,
            }
        finally:
            runtime.deregister()
            await runtime.shutdown()

    def test_adaptive_runtime_bind_scope_requires_registration(self):
        runtime = adaptive_module.AdaptiveRuntime(
            AdaptiveConfig(
                agent_id="test-adaptive-translate-registration",
                state=StateConfig(backend=BackendSpec.in_memory()),
                acg=AcgConfig(provider="anthropic"),
            ).to_dict()
        )

        with scope.scope("adaptive-runtime-registration", ScopeType.Agent) as handle:
            with pytest.raises(RuntimeError, match="must be registered"):
                runtime.bind_scope(handle)

    async def test_adaptive_runtime_bind_scope_passes_through_without_state(self):
        runtime = adaptive_module.AdaptiveRuntime(
            AdaptiveConfig(
                agent_id="test-adaptive-translate-runtime",
                state=StateConfig(backend=BackendSpec.in_memory()),
                acg=AcgConfig(provider="anthropic"),
            ).to_dict()
        )

        await runtime.register()
        try:
            request = LLMRequest(
                {},
                {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "system": "You are helpful.",
                    "model": "claude-sonnet-4-20250514",
                },
            )
            with scope.scope("adaptive-runtime-translate", ScopeType.Agent) as handle:
                runtime.bind_scope(handle)
                translated = llm.request_intercepts("anthropic", request)
                assert translated.content == {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "system": "You are helpful.",
                    "model": "claude-sonnet-4-20250514",
                }
        finally:
            runtime.deregister()
            await runtime.shutdown()

    async def test_configure_report_and_clear(self):
        report = await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        AdaptiveConfig(
                            state=StateConfig(backend=BackendSpec.in_memory()),
                            telemetry=TelemetryConfig(learners=["latency_sensitivity"]),
                            adaptive_hints=AdaptiveHintsConfig(),
                            tool_parallelism=ToolParallelismConfig(),
                        )
                    )
                ]
            )
        )
        try:
            assert report["diagnostics"] == []
            assert plugin.report() == report
        finally:
            plugin.clear()

    async def test_configure_allows_normal_llm_call(self):
        await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        AdaptiveConfig(
                            state=StateConfig(backend=BackendSpec.in_memory()),
                            telemetry=TelemetryConfig(learners=["latency_sensitivity"]),
                            adaptive_hints=AdaptiveHintsConfig(),
                            tool_parallelism=ToolParallelismConfig(),
                        )
                    )
                ]
            )
        )
        try:

            def my_llm(_request: LLMRequest):
                return {"response": "ok"}

            request = LLMRequest({}, {"messages": []})
            result = await llm.execute("test-model", request, my_llm)
            assert result["response"] == "ok"
        finally:
            plugin.clear()

    async def test_python_plugin_is_called_from_core_plugin_system(self):
        class HeaderPlugin:
            def validate(self, plugin_config):
                return [
                    {
                        "level": "warning",
                        "code": "plugin.python_validate_called",
                        "component": "python.test_plugin",
                        "message": f"validated priority={plugin_config.get('priority', 0)}",
                    }
                ]

            def register(self, plugin_config, context):
                priority = plugin_config.get("priority", 33)

                def intercept(_name, request, annotated):
                    headers = dict(request.headers)
                    headers["x-python-plugin"] = f"priority:{priority}"
                    return LLMRequest(headers, request.content), annotated

                async def llm_exec_intercept(_name, request, next_call):
                    response = await next_call(request)
                    response["x-python-llm-exec"] = f"priority:{priority}"
                    return response

                async def llm_stream_exec_intercept(request, next_call):
                    stream = await next_call(request)

                    async def gen():
                        async for chunk in stream:
                            chunk["x-python-llm-stream-exec"] = f"priority:{priority}"
                            yield chunk

                    return gen()

                def tool_request_intercept(_name, args):
                    return {**args, "x-python-tool-plugin": f"priority:{priority}"}

                context.register_llm_request_intercept(
                    "python_header",
                    priority,
                    False,
                    intercept,
                )
                context.register_llm_execution_intercept(
                    "python_exec",
                    priority,
                    llm_exec_intercept,
                )
                context.register_llm_stream_execution_intercept(
                    "python_stream_exec",
                    priority,
                    llm_stream_exec_intercept,
                )
                context.register_tool_request_intercept(
                    "python_tool_request",
                    priority,
                    False,
                    tool_request_intercept,
                )

        plugin.register("python.test_plugin", HeaderPlugin())
        wrapped_config = plugin.PluginConfig(
            components=[
                ComponentSpec(AdaptiveConfig(adaptive_hints=AdaptiveHintsConfig())),
                plugin.ComponentSpec(
                    kind="python.test_plugin",
                    config={"priority": 17},
                ),
            ]
        )
        try:
            report = plugin.validate(wrapped_config)
            assert any(diag["code"] == "plugin.python_validate_called" for diag in report["diagnostics"])

            await plugin.initialize(wrapped_config)

            def my_llm(request: LLMRequest):
                return {
                    "seen_header": request.headers["x-python-plugin"],
                    "seen_exec": request.headers.get("x-missing", "base"),
                }

            request = LLMRequest({}, {"messages": []})
            result = await llm.execute("test-model", request, my_llm)
            assert result["seen_header"] == "priority:17"
            assert result["x-python-llm-exec"] == "priority:17"

            def my_tool(args):
                return args

            tool_result = await tools.execute("search", {"query": "test"}, my_tool)
            assert tool_result["x-python-tool-plugin"] == "priority:17"

            def my_stream_llm(_request: LLMRequest):
                async def gen():
                    yield {"token": "hello"}

                return gen()

            collected: list[JsonObject] = []

            def collector(chunk):
                collected.append(cast(JsonObject, chunk))

            def finalizer():
                return {"count": len(collected)}

            stream = await llm.stream_execute(
                "test-model-stream",
                request,
                my_stream_llm,
                collector,
                finalizer,
            )
            async for chunk in stream:
                assert isinstance(chunk, dict)
                assert chunk["x-python-llm-stream-exec"] == "priority:17"
            assert collected[0]["x-python-llm-stream-exec"] == "priority:17"
        finally:
            plugin.clear()
            plugin.deregister("python.test_plugin")

    def test_list_kinds_includes_registered_plugin(self):
        class MarkerPlugin(plugin.Plugin):
            def validate(self, plugin_config):
                return None

            def register(self, plugin_config, context):
                return None

        plugin.register("python.list_kinds_plugin", MarkerPlugin())
        try:
            assert "python.list_kinds_plugin" in plugin.list_kinds()
        finally:
            plugin.deregister("python.list_kinds_plugin")
