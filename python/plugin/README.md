<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# nemo-relay-plugin

`nemo-relay-plugin` is the Python authoring SDK for NeMo Relay out-of-process
dynamic worker plugins. Use it when plugin code should run in its own Python
process and communicate with Relay through the versioned `grpc-v1` worker
protocol.

## Why Use It?

- **Isolate plugin dependencies**: Run custom policy, middleware, or exporter
  code outside the Relay host process.
- **Use the shared runtime contract**: Register subscribers, guardrails, and
  intercepts through `WorkerPlugin` and `PluginContext`.
- **Call back into Relay safely**: Emit marks, create scopes, and continue
  managed execution through the host runtime handle.
- **Keep worker lifecycle managed**: Let Relay provision the worker environment,
  start the entrypoint, and supply authenticated local endpoints.

## What You Get

- **`WorkerPlugin` and `PluginContext`**: The plugin validation and registration
  contract for worker-owned runtime behavior.
- **`serve_plugin`**: An AsyncIO gRPC server wired to the Relay-managed worker
  environment.
- **Typed runtime helpers**: JSON, event, scope, middleware, continuation, and
  diagnostic types shared with Relay.
- **Generated transport bindings**: Private protobuf bindings included in built
  wheels; published-wheel installation does not require `protoc` or
  `grpcio-tools`.

## Installation

Add the SDK to the Python worker project's dependencies:

```bash
uv add nemo-relay-plugin
```

If you are not using `uv`, install it with `pip`:

```bash
pip install nemo-relay-plugin
```

Declare a `module:function` entrypoint that starts the worker with
`serve_plugin`. Register the plugin manifest through the CLI; Relay creates a
per-plugin virtual environment, installs `source.manifest_root`, and records
that environment for activation:

```bash
nemo-relay plugins add ./relay-plugin.toml
nemo-relay plugins enable <plugin_id>
```

Python workers cannot be loaded directly from `plugins.toml`. They must be
registered through `plugins add`, which provisions the required managed
environment. `plugins remove <plugin_id>` deletes that environment.

## Getting Started

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

For a complete manifest and runnable plugin, see the
[Python gRPC worker plugin example](../../examples/python-grpc-worker-plugin/README.md).

## Request Intercepts

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

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [Build Plugins guide](https://docs.nvidia.com/nemo/relay/build-plugins/about)
- [Python gRPC worker plugin example](../../examples/python-grpc-worker-plugin/README.md)
