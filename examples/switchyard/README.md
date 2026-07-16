<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard Integration Examples

These examples exercise the experimental Relay integration with a separately running Switchyard
Decision API service and the in-process Switchyard translation library. They are manual, local
validation workflows rather than production startup orchestration.

## Required Switchyard Revision

The scripts default to the latest commit currently pinned for the public topic branch:

```text
https://github.com/NVIDIA-NeMo/Switchyard/tree/topic/nemo-relay-integration
8f9db9a6a47f848cdff1d262276ba25a8ae9cbc8
```

Clone the Switchyard repository next to the Relay checkout, then pin the
required commit:

```bash
git clone --branch topic/nemo-relay-integration \
  https://github.com/NVIDIA-NeMo/Switchyard.git \
  ../Switchyard-topic-nemo-relay-integration
git -C ../Switchyard-topic-nemo-relay-integration checkout --detach \
  8f9db9a6a47f848cdff1d262276ba25a8ae9cbc8
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

The Relay CLI's Switchyard support is compile-time optional. The scripts enable the `switchyard`
feature automatically; custom builds must pass `--features switchyard`.

### Manual Switchyard Compatibility Smoke Test

`run-real-e2e.sh` is a manual compatibility smoke test. It starts the pinned Switchyard server,
Relay, and a fake provider, then verifies cold and warm StageRouter decisions, buffered routing,
SSE routing, and the selected model sequence. It is intended to catch cross-repository service
contract regressions; the CI-safe Relay process regression test covers the faster local behavior
checks. The script requires Rust tooling, Python, and `curl`; its temporary logs are removed after
a successful run.

```bash
examples/switchyard/run-real-e2e.sh
```

### Hermes and Ollama Trajectory

`run-hermes-ollama-smoke.sh` runs a fixed multi-query trajectory through Hermes, Relay, Ollama,
and Switchyard. It requires Docker, Hermes, and the configured local Ollama models. The script
produces ATOF, ATIF, and OTEL artifacts and can leave Phoenix running with
`SWITCHYARD_KEEP_PHOENIX=1`.

```bash
examples/switchyard/run-hermes-ollama-smoke.sh
```

## Configuration Files

The directory includes the following configuration and support files:

- `plugins.toml`: minimal plugin configuration example.
- `real-e2e-plugins.toml` and `real-e2e-profiles.yaml`: deterministic fake-provider E2E.
- `hermes-ollama-plugins.toml` and `hermes-ollama-profiles.yaml`: local Ollama trajectory.
- `fake_upstream.py`: deterministic provider used by the service E2E.
- `otel-collector.yaml`: local OTEL artifact export configuration.

## Runtime Model

The scripts launch Switchyard as a separate local process on port `4000`. Relay sends routing
requests to `/v1/routing/decision` and, for ATOF-backed profiles, sends events to
`/v1/atof/events`. Relay owns provider credentials, target bindings, dispatch, retries, and
fallback behavior. Relay executes provider-protocol translation in process through Switchyard's
translation library; the Switchyard service owns ATOF accumulation and routing decisions. The
Switchyard component selects its HTTP ingestion destination by the observability stream sink name,
not by duplicating the sink URL.

The service is not started automatically by Relay outside these examples. A production deployment
must start a compatible Switchyard service before Relay activates the plugin and configure the
Relay plugin with its Decision API URL. Relay derives the service's root `/health` URL from that
configuration and refuses activation unless it reports `{"status":"ok"}`.

## Artifacts and Troubleshooting

Trajectory scripts write to `artifacts/` by default. Set `SWITCHYARD_TRAJECTORY_DIR` to choose a
shareable output directory. On failure, logs are preserved and include the verified Switchyard
revision. Do not place API keys or bearer tokens in configuration files; use environment variables
or an untracked secrets file.
