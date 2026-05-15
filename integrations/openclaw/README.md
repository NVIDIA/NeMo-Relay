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

This public OpenClaw plugin package provides observability support only. It
does not add NeMo Flow security middleware or adaptive optimization behavior to
OpenClaw execution. For middleware-backed behavior, use the patch-based
OpenClaw integration from the NeMo Flow repository.

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

## Configure the Plugin

Enable the `nemo-flow` plugin ID, grant conversation hook access, and place the
OpenClaw plugin configuration under `plugins.entries["nemo-flow"].config`:

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
          "enabled": true,
          "backend": "hooks",
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
                  },
                  "opentelemetry": {
                    "enabled": false,
                    "transport": "http_binary",
                    "endpoint": "http://localhost:4318/v1/traces",
                    "service_name": "openclaw-nemo-flow"
                  },
                  "openinference": {
                    "enabled": false,
                    "transport": "http_binary",
                    "endpoint": "http://localhost:6006/v1/traces",
                    "service_name": "openclaw-nemo-flow"
                  }
                }
              }
            ]
          },
          "capture": {
            "includePrompts": true,
            "includeResponses": true,
            "stripToolArgs": true,
            "stripToolResults": true
          },
          "correlation": {
            "llmOutputGraceMs": 250,
            "recordTtlMs": 600000,
            "maxRecordsPerKey": 32
          }
        }
      }
    }
  }
}
```

This example enables local ATIF export and leaves OTLP exporters disabled until
you point them at a collector or Phoenix endpoint. Remove exporter sections you
do not use, or set their `enabled` fields to `false`.

- `plugins.allow` controls OpenClaw plugin trust and loading.
- `plugins.entries["nemo-flow"].enabled` controls whether OpenClaw starts this
  plugin entry.
- `hooks.allowConversationAccess` lets trusted non-bundled plugins receive
  conversation-sensitive hook payloads such as LLM prompts, LLM responses,
  agent finalization messages, and tool payloads.
- `config.enabled` disables or enables the NeMo Flow OpenClaw wrapper without
  removing the plugin entry. `config.backend` currently supports only `hooks`.
- `config.plugins` is the generic NeMo Flow plugin configuration document. Use
  this object to configure built-in components such as `observability`.
- `config.plugins.components[].config.atif` writes ATIF trajectory JSON files.
  Set `output_directory` to the directory where OpenClaw should write files.
- `config.plugins.components[].config.opentelemetry` sends generic OTLP spans to
  an OpenTelemetry collector when `enabled` is `true`.
- `config.plugins.components[].config.openinference` sends OpenInference OTLP
  spans to Phoenix or another OpenInference-compatible collector when `enabled`
  is `true`.
- `config.capture` controls prompt, response, tool argument, and tool result
  capture. Tool arguments and tool results are stripped by default because they
  often contain user data, local paths, tokens, or large payloads.
- `config.correlation` controls bounded in-memory hook correlation. By default,
  the plugin waits 250 ms for a matching `llm_input` after an `llm_output`,
  keeps correlation records for 600 seconds, and keeps at most 32 records per
  correlation key.

Fields inside `config.plugins` are NeMo Flow generic plugin configuration, so
they use `snake_case` regardless of language. For the full exporter field list,
see the NeMo Flow Observability Plugin schema in the top-level NeMo Flow
documentation at [nvidia.github.io/NeMo-Flow](https://nvidia.github.io/NeMo-Flow/).

## Verify the Integration

Inspect the plugin runtime:

```bash
openclaw plugins inspect nemo-flow --runtime --json
```

Run an OpenClaw session with the plugin enabled, then verify the configured
sink:

- ATIF: confirm JSON files appear in the configured
  `config.plugins.components[].config.atif.output_directory`.
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

It does not change OpenClaw tool execution, provider routing, policy decisions,
or adaptive behavior.

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
`config.plugins.components[].config.atif.output_directory`,
`config.plugins.components[].config.opentelemetry.endpoint`, or
`config.plugins.components[].config.openinference.endpoint`, then confirm the
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
`integrations/openclaw/.test-dist/` from the sibling
`integrations/openclaw/test/` directory so test artifacts do not enter the
installable package or production source tree.

The optional live smoke test requires a working installed `nemo-flow-node`
binding:

```bash
npm run test:live --workspace=nemo-flow-openclaw
```
