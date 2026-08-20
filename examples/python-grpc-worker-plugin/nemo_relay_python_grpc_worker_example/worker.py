# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example Python worker plugin using the nemo-relay-plugin SDK."""

from __future__ import annotations

from typing import Any

from nemo_relay_plugin import (
    ConfigDiagnostic,
    DiagnosticLevel,
    Json,
    PendingMarkSpec,
    PluginContext,
    ToolExecutionInterceptOutcome,
    ToolNext,
    WorkerPlugin,
    serve_plugin,
)


class ExamplePythonWorker(WorkerPlugin):
    """Small worker plugin that tags tool requests and execution results."""

    plugin_id = "examples.python_grpc_worker"

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        if not isinstance(config, dict):
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="examples.python_grpc_worker.invalid_config",
                    component=self.plugin_id,
                    message="plugin config must be a JSON object",
                )
            ]
        if config.get("reject") is True:
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="examples.python_grpc_worker.rejected",
                    component=self.plugin_id,
                    field="reject",
                    message="Python gRPC worker rejection requested",
                )
            ]
        if "tag" in config and not isinstance(config["tag"], str):
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="examples.python_grpc_worker.invalid_tag",
                    component=self.plugin_id,
                    field="tag",
                    message="tag must be a string",
                )
            ]
        return []

    def register(self, ctx: PluginContext, config: Json) -> None:
        if not isinstance(config, dict):
            raise TypeError("plugin config must be a JSON object")
        if config.get("reject") is True:
            raise ValueError("Python gRPC worker rejection requested")
        tag = config.get("tag", "python_grpc_worker")
        if not isinstance(tag, str):
            raise TypeError("tag must be a string")

        async def inject_event_metadata(event: Any) -> dict[str, Json]:
            if config.get("event_metadata_injector_error") is True:
                raise RuntimeError("Python Event metadata injector error requested")
            if str(event.get("name", "")).startswith("external-plugin-event-metadata-injection"):
                return {"external.injector.transport": "python_grpc_worker"}
            return {}

        async def tag_tool_request(tool_name: str, args: Json) -> Json:
            tagged_args = _tag_json(args, tag)
            await ctx.runtime.emit_mark(
                "examples.python_grpc_worker.tool_request",
                {"tool_name": tool_name, "source": "python-grpc-worker", "tag": tag},
            )
            return tagged_args

        async def tag_tool_execution(
            tool_name: str,
            args: Json,
            next_call: ToolNext,
        ) -> ToolExecutionInterceptOutcome:
            result = await next_call.call(args)
            return ToolExecutionInterceptOutcome(
                result=_tag_json(result.result, tag),
                annotation={
                    "upstream": result.annotation,
                    "worker": {"tool_name": tool_name, "tag": tag},
                },
                pending_marks=[
                    PendingMarkSpec(
                        "examples.python_grpc_worker.tool_execution",
                        data={"tool_name": tool_name, "tag": tag},
                    )
                ],
            )

        ctx.register_event_metadata_injector("example_event_metadata_injector", inject_event_metadata)
        if config.get("event_metadata_injector_only") is True:
            return
        ctx.register_tool_request_intercept("tag_tool_request", tag_tool_request)
        ctx.register_tool_execution_intercept("tag_tool_execution", tag_tool_execution)


def _tag_json(value: Json, tag: str) -> Json:
    if not isinstance(value, dict):
        return value
    metadata = value.get("_nemo_relay_plugin")
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        return value
    return {
        **value,
        "_nemo_relay_plugin": {**metadata, "tag": tag},
    }


async def main() -> None:
    """Entrypoint referenced by relay-plugin.toml."""
    await serve_plugin(ExamplePythonWorker())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
