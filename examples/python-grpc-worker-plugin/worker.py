# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example Python worker plugin using the nemo-relay-plugin SDK."""

from __future__ import annotations

from nemo_relay_plugin import ConfigDiagnostic, DiagnosticLevel, Json, PluginContext, WorkerPlugin, serve_plugin


class ExamplePythonWorker(WorkerPlugin):
    """Small worker plugin that tags tool request JSON and emits a host mark."""

    plugin_id = "examples.python_grpc_worker"

    def validate(self, config: Json) -> list[ConfigDiagnostic]:
        if isinstance(config, dict) and config.get("reject") is True:
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="examples.python_grpc_worker.rejected",
                    component=self.plugin_id,
                    field="reject",
                    message="Python gRPC worker rejection requested",
                )
            ]
        return []

    def register(self, ctx: PluginContext, config: Json) -> None:
        del config

        async def tag_tool_request(tool_name: str, args: Json) -> Json:
            await ctx.runtime.emit_mark(
                "examples.python_grpc_worker.tool_request",
                {"tool_name": tool_name, "source": "python-grpc-worker"},
            )
            return _tag_json(args)

        ctx.register_tool_request_intercept("tag_tool_request", tag_tool_request)


def _tag_json(value: Json) -> Json:
    if isinstance(value, dict):
        return {**value, "python_grpc_worker": True}
    return {"value": value, "python_grpc_worker": True}


async def main() -> None:
    """Entrypoint referenced by relay-plugin.toml."""
    await serve_plugin(ExamplePythonWorker())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
