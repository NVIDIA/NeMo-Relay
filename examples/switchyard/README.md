<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard libsy example

This example runs Switchyard routing inside NeMo Relay. It starts only:

1. a deterministic local provider;
2. a Relay CLI process built with the `switchyard` feature.

There is no `switchyard-server`, Decision API, health check, or Switchyard ATOF
ingestion service.

From the Relay repository root:

```bash
examples/switchyard/run-real-e2e.sh
examples/switchyard/run-classifier-e2e.sh
```

The first script sends buffered and streaming OpenAI Chat requests through a
seeded, weighted libsy random router. It verifies that both configured provider
models are selected, that Relay performs every physical request, and that a
provider-specific `system_fingerprint` survives the same-protocol stream round
trip. The second script verifies the classifier consultation, weak-tier
selection, structured response format, and buffered and streaming final calls.

The files are:

- `plugins.toml`: version-2 weighted-random configuration;
- `classifier-plugins.toml`: version-2 LLM-classifier configuration using the
  same Relay-owned provider bindings;
- `fake_upstream.py`: deterministic OpenAI-compatible provider;
- `e2e-common.sh`: process and readiness helpers;
- `run-real-e2e.sh`: weighted-random no-service smoke test;
- `run-classifier-e2e.sh`: classifier no-service smoke test.

The example configurations contain no credentials. For real providers, use
each target's `header_env` map and keep secret values outside tracked
configuration.

The classifier configuration demonstrates the two-call `run_stream` lifecycle:
the fake `provider/classifier` returns a structured verdict, then libsy selects
`provider/fast`. Both calls are made by Relay; no Switchyard client or service
is involved.
