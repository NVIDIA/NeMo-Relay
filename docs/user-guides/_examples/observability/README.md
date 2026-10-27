<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Send a representative Relay workload to your tracing platform

Use Python 3.12 and `nemo-relay==0.10.4` in a virtual environment. The default
workload uses the Relay SDK around a real model request and file operations.
Hermes is optional: `--harness hermes` uses its native Relay integration and
requires a separately installed, authenticated Hermes agent.

The same files run on a Brev CPU VM or Linux/macOS host with provider access.
A dedicated host is required if another example installed system-wide policy.

## Run

Start disposable Phoenix in another terminal with
`uvx --from arize-phoenix phoenix serve`. Export `NVIDIA_API_KEY`,
`NVIDIA_MODEL`, and optionally `NVIDIA_BASE_URL` (default:
`https://integrate.api.nvidia.com/v1`). Confirm access to your selected model.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python nemo-relay==0.10.4
.venv/bin/python run.py --destination phoenix --output /tmp/relay-triage --verify
```

Choose a new output directory for every run. `--prepare-only` writes the
configuration and synthetic fixture without starting a workload.
`--redaction-demo` explicitly enables synthetic payloads and email redaction;
the default leaves full payload export disabled.

## Choose a destination

| Destination | Environment |
| --- | --- |
| Phoenix | Optional `RELAY_OTLP_ENDPOINT`, `PHOENIX_AUTHORIZATION` (full Bearer value), `PHOENIX_API_URL` for API verification |
| LangSmith | `RELAY_OTLP_ENDPOINT`, `LANGSMITH_API_KEY`, and either `LANGSMITH_PROJECT` or both `LANGSMITH_AGENT_ID` and `LANGSMITH_AGENT_ENVIRONMENT` |
| Weave | `RELAY_OTLP_ENDPOINT`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` |
| Datadog | `RELAY_OTLP_ENDPOINT` for your Collector; Datadog credentials stay in the Collector |

Use the complete trace URL. HTTPS is required except for loopback;
`--allow-insecure-endpoint` explicitly permits a trusted private HTTP collector.
Supply secrets through your secret manager or hidden interactive input.

## Verify

The runner requires all incident IDs in the handoff, checkout before the report
delay, and closed model/tool scopes beneath an agent in `logs/events.jsonl`.

With `--verify`, it queries Phoenix for this run's trace, reconciles closed local
agent/model/tool scopes with delivered UUIDs, and requires exported parents.
`delivery.json` records the result. Counts vary by workload; verification uses
scope identities rather than assuming one exported span per event.

Hosted LangSmith, Weave, and Datadog delivery must be checked with your own
account. Export-contract tests do not establish authenticated backend delivery.

## Cleanup

Stop only the Phoenix process you started. Retain evidence, then remove your
chosen output directories and virtual environment. Do not remove shared policy.
