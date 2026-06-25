<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-relay-plugin

Python authoring SDK for NeMo Relay out-of-process dynamic worker plugins.

Install this package in the Python environment used by a worker manifest with
`load.runtime = "python"`, then expose a `module:function` entrypoint that calls
`serve_plugin`.

```python
from nemo_relay_plugin import Json, PluginContext, WorkerPlugin, serve_plugin


class PolicyPlugin(WorkerPlugin):
    plugin_id = "acme.policy"

    def register(self, ctx: PluginContext, config: Json) -> None:
        async def tag_tool_request(tool_name: str, args: Json) -> Json:
            await ctx.runtime.emit_mark("acme.policy.tool_request", {"tool_name": tool_name})
            if isinstance(args, dict):
                return {**args, "policy": "checked"}
            return {"value": args, "policy": "checked"}

        ctx.register_tool_request_intercept("tag_tool_request", tag_tool_request)


async def main() -> None:
    await serve_plugin(PolicyPlugin())
```

The SDK owns gRPC serving, JSON envelope conversion, callback dispatch,
continuations, host runtime calls, and local scope-stack binding.
