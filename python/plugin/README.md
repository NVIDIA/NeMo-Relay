<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-relay-plugin

Python authoring SDK for NeMo Relay out-of-process dynamic worker plugins.

Declare this package as a dependency of a Python worker project and expose a
`module:function` entrypoint that calls `serve_plugin`. Register the manifest
with `nemo-relay plugins add`; Relay creates a per-plugin virtual environment,
installs `source.manifest_root`, and records that environment for activation.
Enable it with `nemo-relay plugins enable <plugin_id>` after registration. Python
workers without a lifecycle-managed environment are rejected.

A minimal worker plugin looks like this:

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

Set `load.entrypoint` to `your_module:main` in `relay-plugin.toml`. Relay
imports that function and awaits the returned coroutine when it starts the
worker process.

LLM request intercepts return one canonical outcome:

```python
from nemo_relay_plugin import LlmRequestInterceptOutcome, PendingMarkSpec


def intercept(model_name, request, annotated):
    del model_name
    headers = {**request.get("headers", {}), "x-policy": "checked"}
    return LlmRequestInterceptOutcome(
        request={**request, "headers": headers},
        annotated_request=annotated,
        pending_marks=[PendingMarkSpec("acme.policy.checked")],
    )
```

When `annotated` is present, it is authoritative for provider-body content:
leave raw `request["content"]` unchanged, edit normalized fields or provider
extensions through the annotation, and use `request["headers"]` for transport
headers.

The SDK owns gRPC serving, JSON envelope conversion, callback dispatch,
continuations, host runtime calls, and local scope-stack binding. Its private
protobuf bindings are generated from the canonical Relay schema while the
package is built; they are included in installed wheels but are not committed
to the source repository. Installing the published wheel never requires
`protoc` or `grpcio-tools`.

## Callback Concurrency

The gRPC AsyncIO server can keep multiple RPCs in flight. Callback execution is
cooperative: asynchronous callbacks overlap only when they yield control at an
`await`. Synchronous callbacks and synchronous stream iterators run on the
worker event-loop thread. Blocking I/O, `time.sleep`, or long-running CPU work
in those callbacks stalls all worker RPCs. Wrap blocking work in an
asynchronous callback and offload it with `asyncio.to_thread()` or another
appropriate executor.

The SDK does not configure `maximum_concurrent_rpcs`, so gRPC does not enforce
an application-level RPC admission limit.

## Invocation Cancellation

Relay assigns every unary and streaming callback an invocation ID. The host
sends `CancelInvocation` when its managed caller is cancelled, its worker RPC
times out, or it stops consuming a worker-backed stream. The SDK cancels the
matching `asyncio.Task` and reports a structured `worker.cancelled` result.

Cancellation is idempotent. The first request that matches an active callback
returns `accepted = true`; requests for unknown, completed, or already
cancelled IDs return `accepted = false`. Treat acceptance as confirmation that
the SDK found and cancelled the task, not as proof that arbitrary user code has
stopped.

Python task cancellation is cooperative. Async callbacks should allow
`asyncio.CancelledError` to propagate and use `try`/`finally` for cleanup.
Synchronous callbacks run on the event-loop thread and cannot be preempted by
task cancellation. A blocking synchronous callback can delay both the
cancellation RPC and all other worker RPCs, so offload blocking work and make
its own cancellation behavior explicit.

`grpc-v1` workers are expected to implement this best-effort cancellation
contract. Relay remains compatible with older workers that return
`accepted = false`; in that case it still drops the transport request, but it
cannot guarantee worker-side interruption.

Windows ARM64 is not currently supported because `grpcio` does not publish a
usable wheel for that platform. The NeMo Relay workspace skips installation and
tests for this SDK on Windows ARM64 rather than creating a package without its
required gRPC runtime.
