# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example Python worker plugin using the nemo-relay-plugin SDK."""

from __future__ import annotations

from typing import Any

from nemo_relay_plugin import ConfigDiagnostic, DiagnosticLevel, Json, PluginContext, WorkerPlugin, serve_plugin


class ExamplePythonWorker(WorkerPlugin):
    """Small worker plugin that tags tool request JSON and emits a host mark."""

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

        async def tag_tool_request(tool_name: str, args: Json) -> Json:
            tagged_args = _tag_json(args, tag)
            await ctx.runtime.emit_mark(
                "examples.python_grpc_worker.tool_request",
                {"tool_name": tool_name, "source": "python-grpc-worker", "tag": tag},
            )
            return tagged_args

        ctx.register_tool_request_intercept("tag_tool_request", tag_tool_request)


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
