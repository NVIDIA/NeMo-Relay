<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Switchyard Plugin

`nemo-relay-switchyard` is an experimental Relay integration for the
[Switchyard Decision API](https://github.com/NVIDIA-NeMo/Switchyard). Relay calls the
Decision API at request time; the plugin does not currently link against Switchyard Rust
crates or start the Switchyard service.

## Experimental setup

The current integration runs Switchyard as a separate service. See the
[example guide](https://github.com/NVIDIA/NeMo-Relay/tree/main/examples/switchyard) for the
pinned Switchyard revision, worktree setup, E2E commands, and artifact handling.

## Configuration

Enable the plugin in the Relay configuration. Relay owns provider URLs, credentials, exact target
bindings, protocol translation, retries, and the trusted fallback. Switchyard returns a selected
backend ID and routing metadata.

For ATOF-backed routing, configure an enabled HTTP ATOF exporter pointing at the Switchyard
`/v1/atof/events` endpoint. `payload_only` profiles do not require ATOF history.

## Verification

Run commands from the root of the NeMo Relay checkout:

```bash
cargo test -p nemo-relay-switchyard
examples/switchyard/run-real-e2e.sh
```

The real-service script starts the pinned Switchyard server and Relay against deterministic fake
providers. The optional Hermes/Ollama script generates a longer trajectory and exports
ATOF/ATIF/OTEL artifacts:

```bash
examples/switchyard/run-hermes-ollama-smoke.sh
```

The scripts generate ephemeral bearer tokens; no credential values are stored in the repository.

## Future direction

This service boundary is intentional for the current experimental integration. A future
Switchyard library-only implementation can provide an in-process DecisionProvider while retaining
the same versioned request and decision contracts.
