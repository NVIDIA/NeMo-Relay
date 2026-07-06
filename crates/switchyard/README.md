<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Switchyard plugin

`nemo-relay-switchyard` is Relay's first-party remote Decision API integration. It has no Rust
dependency on Switchyard: the current `switchyard.routing_request.v1` and
`switchyard.routing_decision.v1` JSON contracts are represented by local compatibility types.

The component supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages on both
buffered and SSE paths. Relay owns target URLs, credentials, exact backend bindings, protocol
translation, bounded provider retries, and the final trusted fallback. Switchyard selects only a
configured backend ID and its exact model/protocol/endpoint tuple.

## Configuration

Start from [`examples/switchyard/plugins.toml`](../../examples/switchyard/plugins.toml). Sensitive
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
