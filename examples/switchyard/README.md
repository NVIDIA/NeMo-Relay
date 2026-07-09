<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard integration examples

These examples exercise the experimental Relay integration with a separately running Switchyard
Decision API service. They are manual, local validation workflows rather than production startup
orchestration.

## Required Switchyard revision

The scripts default to the latest commit currently pinned for the public topic branch:

```text
https://github.com/NVIDIA-NeMo/Switchyard/tree/topic/nemo-relay-integration
5e61cb71ea94fe4f0d365bbc788c9011d42af2e4
```

Create the adjacent checkout from the Relay repository root:

```bash
git fetch upstream topic/nemo-relay-integration
git worktree add --detach ../Switchyard-topic-nemo-relay-integration \
  5e61cb71ea94fe4f0d365bbc788c9011d42af2e4
```

Every real-service script verifies this commit before launching `switchyard-server`. To test a
deliberately different checkout, set both variables explicitly:

```bash
SWITCHYARD_ROOT=/path/to/Switchyard \
SWITCHYARD_EXPECTED_COMMIT=<commit> \
  examples/switchyard/run-real-e2e.sh
```

## Examples

Run these commands from the root of the NeMo Relay checkout.

### Deterministic service E2E

`run-real-e2e.sh` starts Switchyard, Relay, and a fake provider. It verifies cold and warm
StageRouter decisions, buffered routing, SSE routing, and the selected model sequence. It requires
Rust tooling, Python, and `curl`; its temporary logs are removed after a successful run.

```bash
examples/switchyard/run-real-e2e.sh
```

### Hermes and Ollama trajectory

`run-hermes-ollama-smoke.sh` runs a fixed multi-query trajectory through Hermes, Relay, Ollama,
and Switchyard. It requires Docker, Hermes, and the configured local Ollama models. The script
produces ATOF, ATIF, and OTEL artifacts and can leave Phoenix running with
`SWITCHYARD_KEEP_PHOENIX=1`.

```bash
examples/switchyard/run-hermes-ollama-smoke.sh
```

### InferenceHub StageRouter trajectory

`run-inferencehub-stage-router-smoke.sh` runs the Claude Sonnet/Opus trajectory against
InferenceHub. It requires Docker, the Claude CLI, and a secrets file containing
`NV_INFERENCEHUB_KEY`. The key is read from the environment and is never written to the
repository or trajectory payloads.

```bash
INFERENCEHUB_SECRETS_FILE=/path/to/.inference_secrets \
  examples/switchyard/run-inferencehub-stage-router-smoke.sh
```

## Configuration files

- `plugins.toml`: minimal plugin configuration example.
- `real-e2e-plugins.toml` and `real-e2e-profiles.yaml`: deterministic fake-provider E2E.
- `hermes-ollama-plugins.toml` and `hermes-ollama-profiles.yaml`: local Ollama trajectory.
- `inferencehub-stage-router-plugins.toml.in` and `inferencehub-stage-router-profiles.yaml`:
  InferenceHub trajectory templates.
- `fake_upstream.py`: deterministic provider used by the service E2E.
- `otel-collector.yaml`: local OTEL artifact export configuration.

## Runtime model

The scripts launch Switchyard as a separate local process on port `4000`. Relay sends routing
requests to `/v1/routing/decision` and, for ATOF-backed profiles, sends events to
`/v1/atof/events`. Relay owns provider credentials, target bindings, retries, and fallback
behavior; Switchyard owns ATOF accumulation and routing decisions.

The service is not started automatically by Relay outside these examples. A production deployment
must provide a reachable Decision API and configure the Relay plugin with its URL.

## Artifacts and troubleshooting

Trajectory scripts write to `artifacts/` by default. Set `SWITCHYARD_TRAJECTORY_DIR` to choose a
shareable output directory. On failure, logs are preserved and include the verified Switchyard
revision. Do not place API keys or bearer tokens in configuration files; use environment variables
or an untracked secrets file.
