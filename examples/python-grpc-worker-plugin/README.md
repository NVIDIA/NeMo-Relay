<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python gRPC Worker Plugin

This example shows a Python worker plugin using the `nemo-relay-plugin` SDK. It
registers a tool request intercept, emits a mark event through the host runtime,
and returns a mutated JSON tool request.

## Register With Relay

Run the following commands from this directory:

```bash
relay_tmp="$(mktemp -d)"
relay_config="$relay_tmp/gateway.toml"
nemo-relay --config "$relay_config" plugins add ./relay-plugin.toml
nemo-relay --config "$relay_config" plugins enable examples.python_grpc_worker
nemo-relay --config "$relay_config" --bind 127.0.0.1:4040
```

After stopping Relay, remove the plugin and its managed environment, then delete
the temporary state:

```bash
nemo-relay --config "$relay_config" plugins remove examples.python_grpc_worker
rm -rf "$relay_tmp"
```

`plugins add` creates an isolated Relay-managed virtual environment and installs
`source.manifest_root` into it with `python -m pip install`. Standard pip index,
proxy, certificate, and wheelhouse environment variables control dependency
resolution. Set `NEMO_RELAY_PYTHON` only when adding the plugin to select a base
Python interpreter; Relay records and reuses the resulting environment during
activation.

Python workers cannot be loaded directly or by adding a manifest reference to
`plugins.toml`. They must be registered through `plugins add`, which provisions
the required environment. `plugins remove` deletes that Relay-managed
environment.

The SDK package owns the generated protobuf stubs and gRPC server setup. Relay
starts the worker through the manifest entrypoint and supplies the worker
socket, host socket, activation ID, and activation token environment variables.
