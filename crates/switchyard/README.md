<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Switchyard Plugin

`nemo-relay-switchyard` is an experimental Relay integration for
[Switchyard](https://github.com/NVIDIA-NeMo/Switchyard). Relay calls the Decision API at request
time and uses Switchyard's `switchyard-translation` Rust library for request, buffered-response,
and SSE protocol translation. The plugin does not start the Switchyard service.

## Experimental setup

The current integration runs Switchyard as a separate service. See the
[example guide](https://github.com/NVIDIA/NeMo-Relay/tree/main/examples/switchyard) for the
pinned Switchyard revision, worktree setup, E2E commands, and artifact handling.

## Configuration

Build the Relay CLI with the optional integration enabled:

```bash
cargo build -p nemo-relay-cli --features switchyard
```

Then enable the plugin in Relay configuration. Relay owns provider URLs, credentials, exact target
bindings, dispatch, retries, and the trusted fallback. Switchyard owns routing decisions and
provider-protocol translation.

For ATOF-backed routing, configure an enabled HTTP ATOF exporter pointing at the Switchyard
`/v1/atof/events` endpoint. `payload_only` profiles do not require ATOF history.

## Verification

Run commands from the root of the NeMo Relay checkout:

```bash
cargo test -p nemo-relay-switchyard
cargo test -p nemo-relay-cli --features switchyard switchyard
examples/switchyard/run-real-e2e.sh
```

The manual compatibility smoke test starts the pinned Switchyard server and Relay against
deterministic fake providers. The optional Hermes/Ollama script generates a longer trajectory and
exports ATOF/ATIF/OTEL artifacts:

```bash
examples/switchyard/run-hermes-ollama-smoke.sh
```

The scripts generate ephemeral bearer tokens; no credential values are stored in the repository.

## Future direction

The current routing boundary remains service-based because ATOF accumulation and the Decision API
run in Switchyard. Translation is already library-based. A future in-process DecisionProvider can
replace the service call while retaining the same versioned request and decision contracts.
