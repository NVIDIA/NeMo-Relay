<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ATIF

Use the `atif` section when you want one Agent Trajectory Interchange Format
(ATIF) trajectory artifact per top-level agent run.

The plugin-managed ATIF dispatcher watches for direct child scopes with category
`agent`, creates a scope-local exporter for each one, and writes the trajectory
when that agent scope ends. Nested agent scopes remain in the parent
trajectory.

## `plugins.toml` Example

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atif]
enabled = true
agent_name = "Planner"
agent_version = "1.0.0"
model_name = "unknown"
output_directory = "logs"
filename_template = "trajectory-{session_id}.json"
```

This configuration writes a trajectory file such as
`logs/trajectory-<scope-uuid>.json` for each top-level agent scope.

## Fields

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Must be `true` to write trajectories. |
| `agent_name` | `NeMo Flow` | Agent metadata written into the trajectory. |
| `agent_version` | NeMo Flow crate version | Agent version metadata. |
| `model_name` | `unknown` | Default model metadata when no call-level model is present. |
| `tool_definitions` | Omitted | Optional ATIF tool metadata. |
| `extra` | Omitted | Optional ATIF agent metadata. |
| `output_directory` | Current working directory | Directory containing trajectory files. |
| `filename_template` | `nemo-flow-atif-{session_id}.json` | Must contain `{session_id}`. |

## Expected Output

The exporter translates NeMo Flow lifecycle events into ATIF v1.6 trajectory
data. LLM start and end events become model steps, tool events become tool
calls and observations, and scope nesting contributes lineage metadata.

The plugin writes each trajectory when the top-level agent scope closes. If the
plugin is cleared while an agent is still open, teardown flushes the partial
trajectory.

## Manual API

Use the manual `AtifExporter` API when you need explicit collection boundaries
or one exporter object per run.

```python
from nemo_flow import AtifExporter

exporter = AtifExporter("session-1", "agent", "1.0.0", model_name="demo-model")
exporter.register("atif-exporter")

# Run instrumented application work here.

trajectory = exporter.export()
exporter.deregister("atif-exporter")
exporter.clear()
```

## Common Validation Failures

- `filename_template` does not contain `{session_id}`.
- The output directory is not writable at runtime.
- Tool definitions or `extra` metadata are not JSON-compatible.
- The application never opens a top-level `agent` scope, so no trajectory file
  is created.
