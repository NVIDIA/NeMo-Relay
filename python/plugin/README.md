<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay-plugin?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay-plugin/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# nemo-relay-plugin

`nemo-relay-plugin` is the Python SDK for NeMo Relay worker plugins. A worker
runs in its own Python process and calls Relay through
`nvidia.nemo.relay.worker.v1`.

The protocol is `nvidia.nemo.relay.worker.v1`, with contract version `1.0.0`.
Upgrade the host and worker SDK together. Rebuild workers and set these manifest
values: `compat.worker_protocol = "nvidia.nemo.relay.worker.v1"` and
`compat.relay = ">=1.0.0,<2.0"`. Old `grpc-v1` workers cannot connect.
Crate and wheel releases continue to follow the Relay release version.

Callbacks use shared application types. The SDK converts Relay-owned values to
protobuf messages for transport. It preserves application and provider JSON,
tool annotations, pending marks, codec access, cancellation, and scope ownership.

## Authoring Surface

The SDK provides the following types and helpers.

| Surface | Role |
|---|---|
| `WorkerPlugin` and `PluginContext` | Define validation and install all 17 worker-owned subscriber and middleware registrations. |
| `serve_plugin` | Starts an AsyncIO gRPC server from the Relay-managed environment and authenticated local activation endpoints. |
| Typed runtime helpers | Share JSON, event, scope, middleware, continuation, and diagnostic contracts with the Relay host. |
| Tool results | Preserve application results and opaque annotations across tool callbacks and continuations. |
| `LlmExecutionContext` | Provides the selected request codec and, for non-streaming calls, the completed-response codec while the callback is active. |
| Generated transport bindings | Ship private protobuf bindings in the wheel, so installation does not require `protoc` or `grpcio-tools`. |

The worker process isolates Python dependencies and crashes from Relay while preserving
host-owned execution, event, scope, cancellation, and shutdown semantics.

## Propagating Continuation Errors

Re-raise an unchanged error from `ToolNext`, `LlmNext`, or `LlmStreamNext` to
preserve its downstream origin. Typed errors returned by the host remain catchable
as `WorkerSdkError`; stream transport failures retain their gRPC exception type.
Constructing a new exception represents a distinct plugin failure.

The SDK preserves the host-issued continuation identity through callback and
stream responses. The host excludes a forwarded error from plugin callback
failure counts only when that identity matches the current invocation. Existing
error text and execution behavior are preserved. If the identity is missing or does not match, the host uses its normal error
attribution rules.

## Installation

Prebuilt wheels contain generated protobuf bindings and need neither Buf nor
protoc. Source and editable builds require [Buf](https://buf.build/docs/installation/)
1.75.0 or later on `PATH`, or an executable selected with `BUF`. They download
the exact validation schema version recorded in the packaged `buf.lock`. The
first download needs access to the Buf Schema Registry. The source package does
not contain a copy of that upstream schema.


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

## Configuration

Set `[load.max_message_bytes]` in `relay-plugin.toml`. Use `default_bytes` for
the default limit and `by_name` for limits on exact callback registration names.
A callback's limit also applies to its continuations. Other callbacks and control
RPCs keep their own defaults. Host environment overrides take priority. The
default limit is 4 MiB. Restart the host and workers after changing these settings.
Refer to [execution message limits](https://docs.nvidia.com/nemo/relay/build-plugins/workers/grpc-v1-protocol#configure-execution-message-size).

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

Use the optional mark fields to set a data schema, log severity, and category,
as shown in this example:

```python
from nemo_relay_plugin import DataSchema, EventCategory, LogSeverity

await ctx.runtime.emit_mark(
    "acme.policy.decision",
    {"allowed": True},
    data_schema=DataSchema("acme.policy.decision", "1"),
    severity=LogSeverity.INFO,
    category=EventCategory.CUSTOM,
)
```

The original positional `emit_mark(name, data, metadata)` form remains
supported. Relay validates the schema and severity again at the host boundary.
Use `emit_metric(name, measurements, metadata)` to attach the reserved Relay
metric schema; authoritative metric validation remains in Relay after mark
sanitization.

Use `await ctx.runtime.runtime_diagnostics()` in an asynchronous callback to
read an immutable, host-level `RuntimeDiagnostics` snapshot. Each entry has
`code`, `message`, and `count`; `entries` is ordered by code and `get(code)`
looks up one entry. Relay aggregates repeated codes, retains the latest
message, saturates counts, and caps the snapshot at 32 entries. Entries do not
identify the emitting plugin. A host that does not implement this RPC returns gRPC `UNIMPLEMENTED`, which
the SDK reports as an unsupported runtime diagnostics error.

Set `load.entrypoint` to `your_module:main` in `relay-plugin.toml`. Relay
imports that function and awaits the returned coroutine when it starts the
worker process.

## Request Intercepts

LLM request intercepts return one canonical outcome:

```python
from nemo_relay_plugin import AnnotatedLLMRequest, Json, LlmRequestInterceptOutcome, PendingMarkSpec


def intercept(
    model_name: str,
    request: Json,
    annotated: AnnotatedLLMRequest | None,
) -> LlmRequestInterceptOutcome:
    del model_name
    headers = {**request.get("headers", {}), "x-policy": "checked"}
    return LlmRequestInterceptOutcome(
        request={**request, "headers": headers},
        annotated_request=annotated,
        pending_marks=[PendingMarkSpec("acme.policy.checked")],
    )
```

When `annotated` is present, Relay uses it as the source of provider-body content.
Leave raw `request["content"]` unchanged. Edit normalized fields or provider
extensions through the annotation. Use `request["headers"]` for transport headers.

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

Worker v1 implementations are expected to implement this best-effort cancellation
contract. If a worker returns `accepted = false`, Relay still drops the transport
request, but it cannot guarantee worker-side interruption.

Windows ARM64 is not currently supported because `grpcio` does not publish a
usable wheel for that platform. The NeMo Relay workspace skips installation and
tests for this SDK on Windows ARM64 rather than creating a package without its
required gRPC runtime.
