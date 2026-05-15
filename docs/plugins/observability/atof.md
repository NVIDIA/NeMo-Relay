<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ATOF

Use the `atof` section when you want the raw Agent Trajectory Observability
Format (ATOF) `0.1` event stream written as JSONL.

ATOF JSONL export is useful for local debugging, offline inspection, and
preserving the canonical event stream before it is translated into another
format.

## `plugins.toml` Example

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
mode = "overwrite"
```

This configuration registers the plugin-managed ATOF exporter and writes one
JSON object per lifecycle event to `logs/events.jsonl`.

## Fields

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Must be `true` to write events. |
| `output_directory` | Current working directory | Directory containing the JSONL file. |
| `filename` | Timestamped `nemo-flow-events-*.jsonl` | Explicit output filename. |
| `mode` | `append` | `append` or `overwrite`. |

## Expected Output

Each emitted scope, tool, LLM, middleware, or mark event is written as one ATOF
JSON object per line. For event field semantics, see
[Events](../../about/concepts/events.md).

Register the plugin before instrumented work starts and clear it during
shutdown so file handles flush.

## Manual API

Use the manual `AtofExporter` API when a test or script needs a custom
subscriber name or explicit registration window.

```python
from nemo_flow import AtofExporter, AtofExporterConfig, AtofExporterMode

config = AtofExporterConfig()
config.output_directory = "logs"
config.filename = "events.jsonl"
config.mode = AtofExporterMode.Overwrite

exporter = AtofExporter(config)
exporter.register("atof-jsonl")

# Run instrumented application work here.

exporter.deregister("atof-jsonl")
```

## Common Validation Failures

- `mode` is not `append` or `overwrite`.
- The output directory is not writable at runtime.
- ATOF is enabled in a target that cannot access the native filesystem.
