<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)

# NeMo Relay Switchyard Plugin

`nemo-relay-switchyard` is NeMo Relay's experimental integration
with the [NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)
Decision API. It adds routing-aware LLM execution intercepts to the Relay
runtime while preserving Relay ownership of provider credentials, target
bindings, dispatch, retries, fallbacks, and observability.

Install it from crates.io, or build it from the NeMo Relay source checkout with
the optional CLI feature while the Switchyard Decision API contract and
service/library boundary are still evolving.

## Why use it?

- **Route through Switchyard decisions**: Select an exact Relay-owned target
  using a versioned Decision API contract.
- **Keep provider protocols stable**: Use Switchyard's translation library for
  OpenAI Chat, OpenAI Responses, and Anthropic Messages request and response
  translation.
- **Preserve Relay execution semantics**: Keep retries, trusted fallbacks,
  credentials, streaming behavior, and optimization accounting in Relay.
- **Support staged rollout**: Run in enforce or observe-only mode with
  explicit target bindings and protocol defaults.

## What you get

- `SwitchyardConfig`: the typed plugin configuration contract.
- `SwitchyardRuntime`: buffered and streaming routing intercepts.
- Decision and target validation for exact backend, model, protocol, and
  endpoint bindings.
- ATOF-backed or payload-only routing context modes.
- Routing marks and model-routing optimization contributions for Relay's
  cumulative accounting pipeline.
- Switchyard-owned protocol translation through the crates.io
  `switchyard-translation` dependency.

## Installation and source build

Add the crate from crates.io:

```bash
cargo add nemo-relay-switchyard
```

To build the optional CLI integration from a NeMo Relay source checkout:

```bash
cargo build -p nemo-relay-cli --features switchyard
cargo test -p nemo-relay-switchyard
```

The resulting CLI includes the Switchyard component only when the `switchyard`
feature is enabled. A default Relay build does not include this experimental
integration.

## Runtime boundary

The current integration calls Switchyard's HTTP Decision API at runtime. Relay
does not start or supervise the Switchyard service. For ATOF-backed profiles,
Switchyard also provides the `/v1/atof/events` ingestion and accumulator
runtime. A reachable service must therefore be configured before routed traffic
is sent.

The current service setup is documented in
[`examples/switchyard/README.md`](../../examples/switchyard/README.md), including
the pinned topic-branch commit, local configuration, compatibility smoke test,
and trajectory workflow.

Translation is already in-process through Switchyard's Rust translation
library. A future in-process DecisionProvider may replace the HTTP Decision API
call without changing the Relay-owned dispatch and observability boundary.

## Configuration and registration

The CLI registers the component when built with `--features switchyard` and
accepts a `[[components]]` entry with `kind = "switchyard"`. A minimal
configuration selects the Decision API and trusted protocol defaults:

```toml
[[components]]
kind = "switchyard"
enabled = true

[components.config]
mode = "enforce"
decision_api_url = "http://127.0.0.1:4000/v1/routing/decision"
decision_profile_id = "my-profile"
context_mode = "payload_only"
request_materialization = "summary_only"

[components.config.default_targets]
openai_chat = "my-openai-target"
openai_responses = "my-responses-target"
anthropic_messages = "my-anthropic-target"
```

For ATOF-backed profiles, configure an enabled Relay ATOF HTTP exporter that
targets the Switchyard `/v1/atof/events` endpoint and use environment-referenced
authentication headers. Keep provider and Decision API credentials outside
tracked configuration files.

## Documentation

- [Switchyard integration examples](../../examples/switchyard/README.md)
- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard)
