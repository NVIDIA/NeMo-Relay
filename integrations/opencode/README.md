<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow OpenCode Plugin

This package is a standalone OpenCode server plugin for NeMo Flow
observability. It uses OpenCode's public plugin API and does not require
patching OpenCode.

For the illustrated setup guide, see `docs/integrate-frameworks/opencode.md`
in the NeMo Flow source checkout.

## Configuration

Use the plugin from an OpenCode config file. From a NeMo Flow source checkout,
use a file URL:

```json
{
  "plugin": [
    [
      "file:///absolute/path/to/NeMo-Flow/integrations/opencode",
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

When this package is published, replace the file URL with the package name
`nemo-flow-opencode`.

The package loads `nemo-flow-node` and `nemo-flow-node/plugin` dynamically. If
the native Node binding is missing or cannot initialize, the plugin logs one
pass-through warning and does not change OpenCode behavior.

## Compatibility

The plugin declares support for OpenCode plugin APIs through the
`@opencode-ai/plugin` peer dependency. It uses public OpenCode server plugin
hooks available in `@opencode-ai/plugin` `1.14.40` and newer.

## Output

Output is controlled by the generic NeMo Flow `plugins` config. Configure the
built-in `observability` component to write:

- ATOF JSONL events with `plugins.components[].config.atof`.
- ATIF trajectory files with `plugins.components[].config.atif`.
- Optional OpenTelemetry or OpenInference traces with
  `plugins.components[].config.opentelemetry` or `openinference`.
- JSONL plugin diagnostics with the OpenCode wrapper `logPath` field.

## Current Limitations

This plugin uses only existing OpenCode hooks. OpenCode does not yet expose an
around-style LLM stream hook or tool execution hook, so the plugin cannot record
exact LLM stream duration, tool error spans for every failure path, request
intercepts, execution intercepts, or conditional guardrail blocking.
