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

NeMo Relay 0.6.0 uses a separately running Switchyard Decision API from the
`topic/nemo-relay-integration` branch. The in-process `libsy` decision design
is part of the NeMo Relay 0.7 work and does not apply to NeMo Relay 0.6.0.

Install it from crates.io, or build it from the NeMo Relay source checkout with
the optional CLI feature while the Switchyard Decision API contract and
service/library boundary are still evolving.

Use the plugin to:

- **Route through Switchyard decisions**: Select an exact Relay-owned target
  using a versioned Decision API contract.
- **Keep provider protocols stable**: Use Switchyard's translation library for
  OpenAI Chat, OpenAI Responses, and Anthropic Messages request and response
  translation.
- **Preserve Relay execution semantics**: Keep retries, trusted fallbacks,
  credentials, streaming behavior, and optimization accounting in Relay.
- **Support staged rollout**: Run in enforce or observe-only mode with
  explicit target bindings and protocol defaults.

## Implementation and Runtime Behavior

The plugin includes the following implementation and runtime behavior:

- `SwitchyardConfig`: The typed plugin configuration contract.
- `SwitchyardRuntime`: Buffered and streaming routing intercepts.
- Decision and target validation for exact backend, model, protocol, and
  endpoint bindings.
- ATOF-backed or payload-only routing context modes.
- Routing marks and model-routing optimization contributions for Relay's
  cumulative accounting pipeline.
- Switchyard-owned protocol translation through the
  `switchyard-translation` dependency.

## Installation and Source Build

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

## Runtime Boundary

The current integration calls Switchyard's HTTP Decision API at runtime. Relay
does not start or supervise the Switchyard service. For ATOF-backed profiles,
Switchyard also provides the `/v1/atof/events` ingestion and accumulator
runtime. The service must therefore be running before Relay activates the
plugin. Activation performs a bounded request to the service's `/health`
endpoint and fails if it does not return `{"status":"ok"}`. This requirement
applies to enforce and observe-only rollout modes.

The current service setup is documented in
[`examples/switchyard/README.md`](../../examples/switchyard/README.md), including
the pinned topic-branch commit, local configuration, compatibility smoke test,
and trajectory workflow.

Translation is already in-process through Switchyard's Rust translation
library. A future in-process `DecisionProvider` can replace the HTTP Decision API
call without changing the Relay-owned dispatch and observability boundary.

## Configuration and Registration

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

For ATOF-backed profiles, configure an enabled Relay ATOF HTTP stream sink that
has a unique `name`, targets the Switchyard ingestion URL, and uses
environment-referenced authentication headers. Set `atof_endpoint_name` in the
Switchyard component to that name. Local ATOF JSONL output alone does not
populate the Switchyard accumulator. Keep provider and Decision API credentials
outside tracked configuration files.

## Documentation

For more information, refer to the following resources:

- [Switchyard 0.6.0 setup and validation guide](https://docs.nvidia.com/nemo/relay/v0.6.0/configure-plugins/switchyard/about)
- [Switchyard configuration reference](https://docs.nvidia.com/nemo/relay/v0.6.0/configure-plugins/switchyard/configuration)
- [Switchyard integration examples](../../examples/switchyard/README.md)
- [Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard)
