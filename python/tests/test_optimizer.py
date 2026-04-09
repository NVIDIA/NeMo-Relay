# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dynamic optimizer config/runtime API."""

from typing import cast

from nemo_flow import JsonObject, LLMRequest, llm, tools
from nemo_flow.optimizer import (
    BackendSpec,
    ComponentSpec,
    ConfigPolicy,
    DynamoHintsComponent,
    ExternalComponent,
    OptimizerConfig,
    OptimizerRuntime,
    StateConfig,
    TelemetryComponent,
    ToolParallelismComponent,
    deregister_optimizer_plugin,
    register_optimizer_plugin,
    validate_optimizer_config,
)


class TestOptimizerConfigHelpers:
    def test_backend_helpers(self):
        assert BackendSpec.in_memory().to_dict() == {"kind": "in_memory", "config": {}}
        assert BackendSpec.redis("redis://127.0.0.1:6379").to_dict() == {
            "kind": "redis",
            "config": {"url": "redis://127.0.0.1:6379", "key_prefix": "nemo_flow:"},
        }

    def test_component_helpers(self):
        telemetry = TelemetryComponent(learners=["latency_sensitivity"]).to_component().to_dict()
        telemetry_config = cast(JsonObject, telemetry["config"])
        assert telemetry["kind"] == "telemetry"
        assert cast(list[str], telemetry_config["learners"]) == ["latency_sensitivity"]

        dynamo = DynamoHintsComponent().to_component().to_dict()
        dynamo_config = cast(JsonObject, dynamo["config"])
        assert dynamo["kind"] == "dynamo_hints"
        assert cast(int, dynamo_config["priority"]) == 100

        tool = ToolParallelismComponent().to_component().to_dict()
        tool_config = cast(JsonObject, tool["config"])
        assert tool["kind"] == "tool_parallelism"
        assert cast(str, tool_config["mode"]) == "observe_only"

        external = (
            ExternalComponent(
                plugin_kind="python.test_plugin",
                instance_id="plugin-1",
                plugin_config={"priority": 5},
            )
            .to_component()
            .to_dict()
        )
        external_config = cast(JsonObject, external["config"])
        assert external["kind"] == "external_component"
        assert cast(str, external_config["plugin_kind"]) == "python.test_plugin"

    def test_validate_optimizer_config_warns_unknown_component(self):
        report = validate_optimizer_config(OptimizerConfig(components=[ComponentSpec(kind="future_component")]))
        assert any(diag["code"] == "optimizer.unknown_component" for diag in report["diagnostics"])

    def test_runtime_creation_fails_when_unknown_component_is_strict(self):
        config = OptimizerConfig(
            policy=ConfigPolicy(unknown_component="error"),
            components=[ComponentSpec(kind="future_component")],
        )
        try:
            OptimizerRuntime(config)
        except RuntimeError as exc:
            assert "unsupported" in str(exc)
        else:
            raise AssertionError("expected strict config validation failure")


class TestOptimizerRuntime:
    async def test_runtime_register_report_and_shutdown(self):
        runtime = OptimizerRuntime(
            OptimizerConfig(
                state=StateConfig(backend=BackendSpec.in_memory()),
                components=[
                    TelemetryComponent(learners=["latency_sensitivity"]),
                    DynamoHintsComponent(),
                    ToolParallelismComponent(),
                ],
            )
        )
        assert runtime.report()["diagnostics"] == []

        await runtime.register()
        runtime.deregister()
        await runtime.shutdown()

    async def test_runtime_register_allows_normal_llm_call(self):
        runtime = OptimizerRuntime(
            OptimizerConfig(
                state=StateConfig(backend=BackendSpec.in_memory()),
                components=[
                    TelemetryComponent(learners=["latency_sensitivity"]),
                    DynamoHintsComponent(),
                    ToolParallelismComponent(),
                ],
            )
        )
        try:
            await runtime.register()

            def my_llm(_request: LLMRequest):
                return {"response": "ok"}

            request = LLMRequest({}, {"messages": []})
            result = await llm.execute("test-model", request, my_llm)
            assert result["response"] == "ok"
        finally:
            runtime.deregister()

    async def test_python_hosted_plugin_is_called_from_rust(self):
        class HeaderPlugin:
            def validate(self, instance_id, plugin_config):
                return [
                    {
                        "level": "warning",
                        "code": "optimizer.python_plugin_validate_called",
                        "component": "external_component",
                        "message": f"validated {instance_id} with priority={plugin_config.get('priority', 0)}",
                    }
                ]

            def register(self, instance_id, plugin_config, context):
                priority = plugin_config.get("priority", 33)

                def intercept(_name, request, annotated):
                    headers = dict(request.headers)
                    headers["x-python-plugin"] = f"{instance_id}:{priority}"
                    return LLMRequest(headers, request.content), annotated

                async def llm_exec_intercept(_name, request, next_call):
                    response = await next_call(request)
                    response["x-python-llm-exec"] = f"{instance_id}:{priority}"
                    return response

                async def llm_stream_exec_intercept(request, next_call):
                    stream = await next_call(request)

                    async def gen():
                        async for chunk in stream:
                            chunk["x-python-llm-stream-exec"] = f"{instance_id}:{priority}"
                            yield chunk

                    return gen()

                def tool_request_intercept(_name, args):
                    return {**args, "x-python-tool-plugin": f"{instance_id}:{priority}"}

                context.register_llm_request_intercept(
                    f"{instance_id}.python_header",
                    priority,
                    False,
                    intercept,
                )
                context.register_llm_execution_intercept(
                    f"{instance_id}.python_exec",
                    priority,
                    llm_exec_intercept,
                )
                context.register_llm_stream_execution_intercept(
                    f"{instance_id}.python_stream_exec",
                    priority,
                    llm_stream_exec_intercept,
                )
                context.register_tool_request_intercept(
                    f"{instance_id}.python_tool_request",
                    priority,
                    False,
                    tool_request_intercept,
                )

        register_optimizer_plugin("python.test_plugin", HeaderPlugin())
        runtime = OptimizerRuntime(
            OptimizerConfig(
                components=[
                    ExternalComponent(
                        plugin_kind="python.test_plugin",
                        instance_id="plugin-42",
                        plugin_config={"priority": 17},
                    )
                ]
            )
        )
        try:
            report = validate_optimizer_config(
                OptimizerConfig(
                    components=[
                        ExternalComponent(
                            plugin_kind="python.test_plugin",
                            instance_id="plugin-42",
                            plugin_config={"priority": 17},
                        )
                    ]
                )
            )
            assert any(diag["code"] == "optimizer.python_plugin_validate_called" for diag in report["diagnostics"])

            await runtime.register()

            def my_llm(request: LLMRequest):
                return {
                    "seen_header": request.headers["x-python-plugin"],
                    "seen_exec": request.headers.get("x-missing", "base"),
                }

            request = LLMRequest({}, {"messages": []})
            result = await llm.execute("test-model", request, my_llm)
            assert result["seen_header"] == "plugin-42:17"
            assert result["x-python-llm-exec"] == "plugin-42:17"

            def my_tool(args):
                return args

            tool_result = await tools.execute("search", {"query": "test"}, my_tool)
            assert tool_result["x-python-tool-plugin"] == "plugin-42:17"

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
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks[0]["x-python-llm-stream-exec"] == "plugin-42:17"
            assert collected[0]["x-python-llm-stream-exec"] == "plugin-42:17"
        finally:
            runtime.deregister()
            assert deregister_optimizer_plugin("python.test_plugin") is True
