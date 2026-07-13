<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export Raw ATOF Events

Use ATOF when the user needs the canonical NeMo Relay lifecycle event stream as
JSONL for local debugging, offline inspection, or delivery to a raw-event
collector. ATOF preserves events; it does not project them into trajectories or
trace spans.

## Default Plugin Path

Prefer plugin-managed lifecycle for reusable process configuration:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "logs"
filename = "events.jsonl"
mode = "append"
```

Use `overwrite` for an isolated one-run artifact and `append` for repeated local
runs. File output remains active when optional streaming endpoints are added.

Use the manual `AtofExporter` API only when the caller needs a custom subscriber
name or explicit registration window. The lifecycle is: create, register, run
instrumented work, force flush, deregister, then shut down.

## Verify

Verify the export with the following checks:

- Confirm the output file exists and contains one JSON object per line.
- Confirm the expected root scope plus tool or LLM lifecycle events are present.
- Check UUID and parent UUID relationships instead of relying only on event
  order.
- Confirm sensitive fields are absent before retaining or transmitting output.
- For streaming endpoints, verify file output separately from endpoint delivery.

Common failures include an unwritable output directory, an invalid mode, an
empty endpoint URL, an unsupported endpoint transport, or shutdown occurring
before pending events flush.

For the complete exporter configuration, see
[ATOF observability](https://docs.nvidia.com/nemo/relay/dev/configure-plugins/observability/atof).
