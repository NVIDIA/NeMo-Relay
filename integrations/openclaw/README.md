<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-openclaw

`nemo-flow-openclaw` is the NeMo Flow observability plugin package for
OpenClaw. It converts supported OpenClaw hook events into NeMo Flow sessions,
LLM spans, tool spans, and lifecycle marks that the generic NeMo Flow
observability component can export as ATIF JSON, OpenTelemetry spans, and
OpenInference/Phoenix spans.

## Why Use It?

- Observe OpenClaw sessions without patching OpenClaw.
- Export OpenClaw activity into NeMo Flow observability formats.
- Preserve OpenClaw's agent, tool, and LLM lifecycle context where public hooks
  expose enough data.
- Keep ambiguous LLM timing attribution visible through diagnostic marks instead
  of unsafe latency.

## What You Get

- OpenClaw plugin ID `nemo-flow`.
- Generic NeMo Flow plugin initialization through `config.plugins`.
- ATIF JSON export through the built-in `observability` component.
- Optional OpenTelemetry OTLP export.
- Optional OpenInference/Phoenix OTLP export.
- Bounded LLM replay correlation across supported OpenClaw hooks.
- Tool span replay with conservative privacy defaults.
- Admin-scoped `nemoFlow.status` gateway health method.

## Installation

Install the package directly in a Node.js/OpenClaw environment:

```bash
npm install nemo-flow-openclaw
```

For OpenClaw-managed installation, use the OpenClaw CLI:

```bash
openclaw plugins install npm:nemo-flow-openclaw
openclaw gateway restart
```

OpenClaw uses the package `nemo-flow-openclaw` for installation and the plugin
manifest ID `nemo-flow` for configuration.

## Getting Started

Enable the `nemo-flow` plugin ID and grant conversation hook access when
OpenClaw runs with restrictive plugin settings:

```json
{
  "plugins": {
    "allow": ["nemo-flow"],
    "entries": {
      "nemo-flow": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {
          "plugins": {
            "version": 1,
            "components": [
              {
                "kind": "observability",
                "enabled": true,
                "config": {
                  "version": 1,
                  "atif": {
                    "enabled": true,
                    "agent_name": "openclaw",
                    "output_directory": "./nemo-flow-atif"
                  }
                }
              }
            ]
          }
        }
      }
    }
  }
}
```

Plugin configuration lives under `plugins.entries["nemo-flow"].config`. The
OpenClaw wrapper owns `enabled`, `backend`, `capture`, and `correlation`; the
top-level `plugins` object is the generic NeMo Flow plugin configuration
document.

## Configure Outputs

Configure exporters through the generic NeMo Flow `observability` component in
`config.plugins.components`. Missing observability sections are disabled.

Use ATIF-only local export when you want JSON trace files:

```json
{
  "plugins": {
    "version": 1,
    "components": [
      {
        "kind": "observability",
        "enabled": true,
        "config": {
          "version": 1,
          "atif": {
            "enabled": true,
            "agent_name": "openclaw",
            "output_directory": "./nemo-flow-atif"
          }
        }
      }
    ]
  }
}
```

Use OpenInference/Phoenix export when you want Phoenix-compatible LLM traces:

```json
{
  "plugins": {
    "version": 1,
    "components": [
      {
        "kind": "observability",
        "enabled": true,
        "config": {
          "version": 1,
          "openinference": {
            "enabled": true,
            "transport": "http_binary",
            "endpoint": "http://localhost:6006/v1/traces",
            "service_name": "openclaw-nemo-flow"
          }
        }
      }
    ]
  }
}
```

Use OpenTelemetry OTLP export when you want generic OTLP traces:

```json
{
  "plugins": {
    "version": 1,
    "components": [
      {
        "kind": "observability",
        "enabled": true,
        "config": {
          "version": 1,
          "opentelemetry": {
            "enabled": true,
            "transport": "http_binary",
            "endpoint": "http://localhost:4318/v1/traces",
            "service_name": "openclaw-nemo-flow"
          }
        }
      }
    ]
  }
}
```

Privacy defaults capture prompts and responses, and strip tool arguments and
tool results:

```json
{
  "capture": {
    "includePrompts": true,
    "includeResponses": true,
    "stripToolArgs": true,
    "stripToolResults": true
  }
}
```

## Verify the Integration

Inspect the plugin runtime:

```bash
openclaw plugins inspect nemo-flow --runtime --json
```

Run an OpenClaw session with the plugin enabled, then verify the configured
sink:

- ATIF: confirm JSON files appear in the configured
  `plugins.components[].config.atif.output_directory`.
- OpenTelemetry: confirm spans arrive at the configured OTLP collector.
- OpenInference: confirm spans arrive at the configured OpenInference/Phoenix
  endpoint.

The plugin also registers the `operator.admin` scoped gateway method
`nemoFlow.status`. If your CLI is already paired with admin-capable gateway
access, run:

```bash
openclaw gateway call nemoFlow.status --json
```

## Current Limits

The plugin maps supported OpenClaw hook events into NeMo Flow telemetry without
changing OpenClaw execution behavior.

Current OpenClaw public hooks expose request, response, message-write, and
provider timing details through separate event streams. The plugin correlates
those events within the same session, provider, model, and run. When timing
cannot be paired safely, it emits diagnostic marks instead of inventing
latency.

## Troubleshooting

If the plugin does not load, verify the package was installed with
`openclaw plugins install`, `plugins.allow` includes `nemo-flow`,
`plugins.entries["nemo-flow"].enabled` is not disabled, and the gateway was
restarted after configuration changes.

If conversation payloads are missing, verify
`hooks.allowConversationAccess` is enabled for the plugin and the OpenClaw
session emits the relevant LLM, message-write, and tool hooks.

If no export output appears, verify
`plugins.components[].config.atif.output_directory`,
`plugins.components[].config.opentelemetry.endpoint`, or
`plugins.components[].config.openinference.endpoint`, then confirm the
configured collector or output directory is reachable.

## Development

Run these commands from the repository root:

```bash
npm ci --ignore-scripts
npm run build --workspace=nemo-flow-openclaw
npm run typecheck --workspace=nemo-flow-openclaw
npm test --workspace=nemo-flow-openclaw
```

The CI-equivalent repo recipe is:

```bash
just --set ci true test-openclaw
```

Check the package payload before changing package metadata or entrypoints:

```bash
npm run pack:check --workspace=nemo-flow-openclaw
```

`npm run build --workspace=nemo-flow-openclaw` emits production files under
`integrations/openclaw/dist/`. Tests compile to
`integrations/openclaw/.test-dist/` so test artifacts do not enter the
installable package.

The optional live smoke test requires a working installed `nemo-flow-node`
binding:

```bash
npm run test:live --workspace=nemo-flow-openclaw
```
