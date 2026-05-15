<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenCode Plugin

NeMo Flow integrates with OpenCode through the `nemo-flow-opencode` server
plugin. The plugin uses OpenCode's public plugin hooks and does not require a
patched OpenCode checkout.

Use this plugin when you want NeMo Flow observability for OpenCode sessions,
LLM calls, successful tool calls, and session errors. The plugin maps OpenCode
hook payloads into NeMo Flow session, LLM, and tool spans. The generic NeMo
Flow `observability` component controls ATOF, ATIF, OpenTelemetry, and
OpenInference export.

## Requirements

- OpenCode with server plugin support.
- Node.js 20 or newer.
- A NeMo Flow Node.js binding package compatible with `nemo-flow-opencode`.
- Provider credentials configured in OpenCode.

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

OpenCode uses the package name `nemo-flow-opencode` in the `plugin` array.

## Enable and Configure the Plugin

Create or update `opencode.json` in the OpenCode project directory:

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
                  "filename": "opencode.atof.jsonl",
                  "mode": "overwrite"
                },
                "atif": {
                  "enabled": true,
                  "agent_name": "opencode",
                  "output_directory": "./.nemoflow",
                  "filename_template": "opencode-{session_id}.atif.json"
                },
                "opentelemetry": {
                  "enabled": false,
                  "transport": "http_binary",
                  "endpoint": "http://localhost:4318/v1/traces",
                  "service_name": "opencode-nemo-flow"
                },
                "openinference": {
                  "enabled": false,
                  "transport": "http_binary",
                  "endpoint": "http://localhost:6006/v1/traces",
                  "service_name": "opencode-nemo-flow"
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

This example enables filesystem ATOF and ATIF export and leaves OTLP exporters
disabled until you point them at a collector or Phoenix endpoint. Remove
exporter sections you do not use, or set their `enabled` fields to `false`.

- `plugin[][0]` is the OpenCode plugin package name. Use
  `nemo-flow-opencode`.
- `enabled` disables or enables the NeMo Flow OpenCode wrapper without removing
  the plugin entry.
- `logPath` writes JSONL diagnostics for plugin initialization and
  pass-through behavior.
- `plugins` is the generic NeMo Flow plugin configuration document. Use this
  object to configure built-in components such as `observability`.
- `plugins.components[].config.atof` writes raw ATOF JSONL lifecycle events.
- `plugins.components[].config.atif` writes ATIF trajectory JSON files.
- `plugins.components[].config.opentelemetry` sends generic OTLP spans to an
  OpenTelemetry collector when `enabled` is `true`.
- `plugins.components[].config.openinference` sends OpenInference OTLP spans to
  Phoenix or another OpenInference-compatible collector when `enabled` is
  `true`.

## Configuration Key Names

OpenCode wrapper fields use JavaScript-style names, such as `logPath`.

The top-level `plugins` object inside the wrapper is the generic NeMo Flow
plugin config. Fields inside this object use NeMo Flow generic plugin names, so
they are `snake_case` in every binding.

Missing observability sections are disabled. Plugin-host validation or
initialization failures leave OpenCode in pass-through mode and write a warning
to `logPath`.

The ATIF filename placeholder `{session_id}` is the NeMo Flow top-level agent
scope UUID. The OpenCode session ID is recorded in event metadata.

See [Configure the Observability Plugin](../export-observability-data/observability-plugin.md)
for the complete `observability` component schema and exporter-specific fields.

The plugin is passive. It records observability output but does not rewrite
prompts, tool arguments, model requests, or OpenCode execution behavior.

OpenCode streaming message events are used internally to reconstruct concise
LLM responses. They are not exported as individual ATIF steps.

## Known Limitations

The current OpenCode plugin API is enough for passive observability. It is not
enough for NeMo Flow request intercepts, execution intercepts, conditional
blocking, or complete tool error spans because OpenCode does not yet expose
around-style LLM or tool hooks. Future work should add generic OpenCode plugin
hooks upstream before enabling those behaviors.
