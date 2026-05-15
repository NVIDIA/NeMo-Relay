<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow OpenCode Plugin

`nemo-flow-opencode` is a standalone OpenCode server plugin for NeMo Flow
observability. It uses OpenCode's public plugin API and does not require
patching OpenCode. It maps OpenCode activity into NeMo Flow session, LLM, and
tool spans for the generic observability plugin.

For the full guide, see `docs/integrate-frameworks/opencode.md` in the NeMo
Flow documentation.

## Install

Install the plugin with the OpenCode CLI:

```bash
opencode plugin nemo-flow-opencode
```

You can also install the package in the Node.js environment where OpenCode
loads plugins:

```bash
npm install nemo-flow-opencode
```

## Configure

Use the package name in `opencode.json`:

```json
{
  "plugin": [
    [
      "nemo-flow-opencode",
      {
        "enabled": true,
        "logPath": "./.nemoflow/opencode-plugin.log",
        "plugins": {
          "version": 1,
          "components": [
            {
              "kind": "observability",
              "enabled": true,
              "config": {
                "version": 1,
                "atof": {
                  "enabled": true,
                  "output_directory": "./.nemoflow",
                  "filename": "opencode.atof.jsonl"
                },
                "atif": {
                  "enabled": true,
                  "agent_name": "opencode",
                  "output_directory": "./.nemoflow",
                  "filename_template": "opencode-{session_id}.atif.json"
                }
              }
            }
          ]
        }
      }
    ]
  ]
}
```

Fields inside `plugins` are NeMo Flow generic plugin configuration, so they use
`snake_case`. The OpenCode wrapper fields use JavaScript-style names, such as
`logPath`.

## Output

Configure the built-in `observability` component to write:

- ATOF JSONL events with `plugins.components[].config.atof`.
- ATIF trajectory files with `plugins.components[].config.atif`.
- Optional OpenTelemetry or OpenInference traces with
  `plugins.components[].config.opentelemetry` or `openinference`.
- JSONL plugin diagnostics with the OpenCode wrapper `logPath` field.

OpenCode streaming message events are used internally to reconstruct concise
LLM responses. They are not exported as individual ATIF steps.

## Current Limitations

This plugin uses only existing OpenCode hooks. OpenCode does not yet expose an
around-style LLM stream hook or tool execution hook, so the plugin cannot record
exact LLM stream duration, tool error spans for every failure path, request
intercepts, execution intercepts, or conditional guardrail blocking.
