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
```

The script sends buffered and streaming OpenAI Chat requests through a seeded,
weighted libsy random router. It verifies that both configured provider models
are selected, that Relay performs every physical request, and that a
provider-specific `system_fingerprint` survives the same-protocol stream round
trip.

The files are:

- `plugins.toml`: version-2 library-only Switchyard configuration;
- `fake_upstream.py`: deterministic OpenAI-compatible provider;
- `e2e-common.sh`: process and readiness helpers;
- `run-real-e2e.sh`: executable no-service smoke test.

`plugins.toml` contains no credentials. For real providers, use each target's
`header_env` map and keep secret values outside tracked configuration.
