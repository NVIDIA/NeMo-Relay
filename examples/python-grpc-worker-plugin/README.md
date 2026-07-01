<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python gRPC Worker Plugin

This example shows a Python worker plugin using the `nemo-relay-plugin` SDK. It
registers a tool request intercept, emits a mark event through the host runtime,
and returns a mutated JSON tool request.

## Set Up

Run the following commands from this directory.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ../../python/plugin -e .
```

The SDK package owns the generated protobuf stubs and gRPC server setup.

## Register With Relay

Point the CLI at this manifest and enable it:

```bash
nemo-relay plugins add ./relay-plugin.toml
nemo-relay plugins enable examples.python_grpc_worker
```

When launching the gateway, point Relay at the Python interpreter that has
`grpcio` installed:

```bash
NEMO_RELAY_PYTHON="$PWD/.venv/bin/python" nemo-relay gateway
```

You can also reference the manifest manually from `plugins.toml`:

```toml
[[plugins.dynamic]]
manifest = "./examples/python-grpc-worker-plugin/relay-plugin.toml"
config = { tag = "demo" }
```

The worker process is started by Relay through the manifest entrypoint. Enable
the dynamic plugin in `plugins.toml` instead of launching the process directly;
Relay supplies the worker socket, host socket, activation ID, plugin ID, and
activation token environment variables.

Async callbacks are cancelled cooperatively when the host caller times out or
stops consuming a worker stream. Let `asyncio.CancelledError` propagate and put
resource cleanup in `finally` blocks. Synchronous or blocking callback code
cannot be preempted by the SDK; move that work off the event-loop thread and
define its cancellation behavior explicitly.
