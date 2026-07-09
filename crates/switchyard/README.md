<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)

# NeMo Relay Switchyard Plugin

`nemo-relay-switchyard` is Relay's first-party remote Decision API integration. It has no Rust
dependency on Switchyard. The current `switchyard.routing_request.v1` and
`switchyard.routing_decision.v1` JSON contracts are represented by local compatibility types
intentionally: the plugin depends on the versioned wire contract, while Relay retains ownership of
its runtime and target-binding boundary. These types are not a placeholder for a future Rust
dependency on Switchyard.

The component supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages on both
buffered and SSE paths. Relay owns target URLs, credentials, exact backend bindings, protocol
translation, bounded provider retries, and the final trusted fallback. Switchyard selects only a
configured backend ID and its exact model/protocol/endpoint tuple.

## Why Use It?

- **Route through a remote Decision API**: Let Switchyard select among Relay-owned backend targets
  without coupling the Relay runtime to Switchyard's implementation.
- **Keep provider boundaries in Relay**: Relay validates exact model, protocol, endpoint, and URL
  bindings before dispatch.
- **Fail open safely**: Decision failures, unsupported extensions, and provider failures use the
  configured same-protocol trusted fallback.
- **Preserve causal evidence**: Routing marks and optimization contributions identify the selected
  model, capable baseline, freshness state, retry attempt, and terminal committed route.

## What You Get

- Remote Decision API request and decision contract compatibility.
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages translation.
- Buffered and streaming dispatch with bounded retries and strict stream commitment.
- Six request-materialization modes and identity-aware ATOF integration.
- Exact backend validation, environment-referenced credentials, and per-protocol fallbacks.

## Installation

The Switchyard component is shipped as part of the NeMo Relay workspace. Enable it through the
Relay plugin configuration described below.

## Configuration

Start from [`examples/switchyard/plugins.toml`](https://github.com/NVIDIA/NeMo-Relay/blob/main/examples/switchyard/plugins.toml). Sensitive
headers contain environment variable names, never secret values. For example:

```bash
export SWITCHYARD_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
export SWITCHYARD_AUTHORIZATION="Bearer ${SWITCHYARD_TOKEN}"
nemo-relay --plugin-config-path examples/switchyard/plugins.toml doctor
```

`context_mode = "payload_only"` requires no history exporter. `context_mode = "atof_required"`
requires an enabled `http_post` ATOF endpoint at the configured (or derived)
`/v1/atof/events` URL, `field_name_policy = "preserve"`, and at least one environment-referenced
authentication header.

## Verification

Run every command in this section from the root of the NeMo Relay repository checkout.

The crate's default test suite includes the fake Decision API/provider E2E, all six request
materialization modes, all buffered and streaming 3×3 protocol combinations, exact target
validation, retry exhaustion, non-retryable fallback, and the stream commit boundary:

```bash
cargo test -p nemo-relay-switchyard
cargo test -p nemo-relay-cli gateway::tests
cargo test -p nemo-relay --features atof-streaming observability::atof::tests
```

The repeatable real-service E2E starts the cumulative Switchyard server and deterministic fake
providers, routes cold and warm buffered requests, and checks the SSE path:

```bash
examples/switchyard/run-real-e2e.sh
```

With Ollama serving `qwen3.6:35b` on `127.0.0.1:11434` and Hermes installed, run a real model
trajectory through Hermes, Relay, and the cumulative Switchyard server:

```bash
examples/switchyard/run-hermes-ollama-smoke.sh
```

Both scripts create a random shared bearer token for the run; no credential value is stored in the
repository. Override `SWITCHYARD_ROOT` when the cumulative worktree is not adjacent to this Relay
worktree.

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)
- [Switchyard Decision API integration example](https://github.com/NVIDIA/NeMo-Relay/tree/main/examples/switchyard)
