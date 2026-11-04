<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python gRPC Worker Plugin

This is the Python worker used by the NeMo Relay plugin guide. It supports all 17
`nvidia.nemo.relay.worker.v1` callback types. Configuration selects which callbacks
to register. The example shows annotations, pending marks, codec access, lazy
streams, and cleanup of scopes and cancelled work.

The worker targets Relay 1.x using the `nvidia.nemo.relay.worker.v1` protocol
(contract `1.0.0`). Its manifest requires `compat.relay = ">=1.0.0,<2.0"`.
Its tool continuation returns `ToolExecutionResult`. LLM execution callbacks
receive codec context before the continuation. Non-streaming callbacks can decode
a completed response. Streaming callbacks have request codec access only.

Run the example's own test project from this directory:

```bash
uv run --locked --package nemo-relay-python-grpc-worker-example --group test pytest
```

Each test checks one behavior and can run on its own. The suite builds a wheel
in a clean temporary project. It checks the source digest, JSON Schema,
configuration, and all 17 registration types. Separate tests cover policies,
sanitizers, request changes, continuations, streams, marks, and scopes.

The seventeenth callback type is an optional conditional middleware guardrail.
The activation owns this guardrail. `registration_control.enabled` defaults to `false`. Its default kinds
are `["subscriber"]`, its target is `documentation-controlled-subscriber`, and its
reason is `disabled by documentation plugin`. The callback returns that reason for targets
whose names start with `documentation-controlled-` and returns `None` to leave other
matching targets enabled. The kinds, effective target, and reason must be nonempty. Refer to
[Conditional Middleware Guardrails](../../docs/about-nemo-relay/concepts/conditional-middleware-guardrails.mdx)
for the full runtime and ownership contract.

To run the managed-environment lifecycle from this directory, create temporary
Relay state, add the manifest, and enable the plugin:

```bash
relay_tmp="$(mktemp -d)"
relay_config="$relay_tmp/gateway.toml"
: > "$relay_config"
nemo-relay --config "$relay_config" plugins add ./relay-plugin.toml
nemo-relay --config "$relay_config" plugins enable examples.python_grpc_worker
nemo-relay --config "$relay_config" --bind 127.0.0.1:4040
```

After stopping Relay, run the cleanup commands in the same shell session so
`relay_config` and `relay_tmp` still identify the temporary Relay state.
Removal also deletes the Relay-managed Python environment.

```bash
nemo-relay --config "$relay_config" plugins remove examples.python_grpc_worker
rm -rf -- "$relay_tmp"
```
