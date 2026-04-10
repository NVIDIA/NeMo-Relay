# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adaptive plugin component API."""

from nemo_flow import LLMRequest, llm, plugin, tools
from nemo_flow.adaptive import (
    ADAPTIVE_PLUGIN_KIND,
    AdaptiveConfig,
    AdaptiveHintsConfig,
    BackendSpec,
    ComponentSpec,
    StateConfig,
    TelemetryConfig,
    ToolParallelismConfig,
)


class TestAdaptiveConfigHelpers:
    def test_backend_helpers(self):
        assert BackendSpec.in_memory().to_dict() == {"kind": "in_memory", "config": {}}
        assert BackendSpec.redis("redis://127.0.0.1:6379").to_dict() == {
            "kind": "redis",
            "config": {"url": "redis://127.0.0.1:6379", "key_prefix": "nemo_flow:"},
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


class TestAdaptivePluginConfiguration:
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

    async def test_python_hosted_plugin_is_called_from_core_plugin_host(self):
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

            collected = []

            def collector(chunk):
                collected.append(chunk)

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
                assert chunk["x-python-llm-stream-exec"] == "priority:17"
            assert collected[0]["x-python-llm-stream-exec"] == "priority:17"
        finally:
            plugin.clear()
            plugin.deregister("python.test_plugin")
