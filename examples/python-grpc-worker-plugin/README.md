<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python gRPC Worker Plugin

This package is the complete Python worker used by the plugin authoring guide.
It validates the shared documentation configuration, registers every safe
`grpc-v1` surface, preserves annotations and Relay-owned accounting, uses
invocation-scoped codec proxies, transforms streams lazily, and cleans up marks,
scopes, isolated stacks, and cancelled tasks.

The worker targets the Relay 0.8 `grpc-v1` result contract. Its tool continuation returns
`ToolExecutionResult`. Its execution intercept preserves the application result, carries
the upstream annotation under worker metadata, and adds Relay-owned pending marks.

Run the example's own test project from this directory:

```bash
uv run --locked --group test pytest
```

Each test owns one contract and can be selected independently. The suite builds
this directory as a wheel in a clean temporary project, checks the mandatory
source digest and JSON Schema, validates configuration, asserts all 17
registrations, and then exercises each policy, sanitizer, request rewrite,
continuation, stream, mark, and scope behavior separately.

The seventeenth surface is an optional activation-owned conditional middleware
guardrail. `registration_control.enabled` defaults to `false`; its remaining defaults
are `kinds: ["subscriber"]`, target `documentation-controlled-subscriber`, and reason
`disabled by documentation plugin`. The callback returns that reason for targets
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
