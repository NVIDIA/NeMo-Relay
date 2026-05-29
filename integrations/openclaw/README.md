<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-relay-openclaw

`nemo-relay-openclaw` is the NeMo Relay observability plugin package for
OpenClaw. It converts supported OpenClaw hook events into NeMo Relay sessions,
LLM spans, tool spans, and lifecycle marks that the generic NeMo Relay
observability component can export as ATIF JSON, OpenTelemetry spans, and
OpenInference/Phoenix spans. The same generic plugin config path can initialize
Adaptive components for hook-backed telemetry learning.

This public OpenClaw plugin package uses OpenClaw public hooks. It can run
pre-tool conditional guardrails when OpenClaw invokes the before-tool hook, but
it does not rewrite provider routing or model requests. For middleware-backed
behavior that changes execution, use the patch-based OpenClaw integration from
the NeMo Relay repository.

## Why Use It?

- Observe OpenClaw sessions without patching OpenClaw.
- Export OpenClaw activity into NeMo Relay observability formats.
- Preserve OpenClaw's agent, tool, and LLM lifecycle context where public hooks
  expose enough data.
- Keep ambiguous LLM timing attribution visible through diagnostic marks instead
  of unsafe latency.

## What You Get

- OpenClaw plugin ID `nemo-relay`.
- Generic NeMo Relay plugin initialization through `config.plugins`.
- ATIF JSON export through the built-in `observability` component.
- Adaptive plugin initialization through `config.plugins`.
- Optional OpenTelemetry OTLP export.
- Optional OpenInference/Phoenix OTLP export.
- Bounded LLM replay correlation across supported OpenClaw hooks.
- Tool span replay with conservative privacy defaults.
- Admin-scoped `nemoRelay.status` gateway health method.

## Installation

Install the package directly in a Node.js/OpenClaw environment:

```bash
npm install nemo-relay-openclaw
```

For OpenClaw-managed installation, use the OpenClaw CLI:

```bash
openclaw plugins install npm:nemo-relay-openclaw
openclaw gateway restart
```

OpenClaw uses the package `nemo-relay-openclaw` for installation and the plugin
manifest ID `nemo-relay` for configuration.

## Configure the Plugin

Enable the `nemo-relay` plugin ID, grant conversation hook access, and place the
OpenClaw plugin configuration under `plugins.entries["nemo-relay"].config`:

```json
{
  "plugins": {
    "allow": ["nemo-relay"],
    "entries": {
      "nemo-relay": {
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
                    "output_directory": "./nemo-relay-atif"
                  },
                  "opentelemetry": {
                    "enabled": false,
                    "transport": "http_binary",
                    "endpoint": "http://localhost:4318/v1/traces",
                    "service_name": "openclaw-nemo-relay"
                  },
                  "openinference": {
                    "enabled": false,
                    "transport": "http_binary",
                    "endpoint": "http://localhost:6006/v1/traces",
                    "service_name": "openclaw-nemo-relay"
                  }
                }
              },
              {
                "kind": "adaptive",
                "enabled": true,
                "config": {
                  "version": 1,
                  "agent_id": "openclaw",
                  "state": {
                    "backend": {
                      "kind": "in_memory",
                      "config": {}
                    }
                  },
                  "telemetry": {
                    "learners": ["tool_parallelism"]
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
- `plugins.entries["nemo-relay"].enabled` controls whether OpenClaw starts this
  plugin entry.
- `hooks.allowConversationAccess` lets trusted non-bundled plugins receive
  conversation-sensitive hook payloads such as LLM prompts, LLM responses,
  agent finalization messages, and tool payloads.
- `config.enabled` disables or enables the NeMo Relay OpenClaw wrapper without
  removing the plugin entry. `config.backend` currently supports only `hooks`.
- `config.plugins` is the generic NeMo Relay plugin configuration document. Use
  this object to configure built-in components such as `observability` and
  `adaptive`.
- `config.plugins.components[].config.atif` writes ATIF trajectory JSON files.
  Set `output_directory` to the directory where OpenClaw should write files.
- `config.plugins.components[].config.opentelemetry` sends generic OTLP spans to
  an OpenTelemetry collector when `enabled` is `true`.
- `config.plugins.components[].config.openinference` sends OpenInference OTLP
  spans to Phoenix or another OpenInference-compatible collector when `enabled`
  is `true`.
- `config.plugins.components[]` entries with `kind: "adaptive"` initialize the
  Adaptive plugin. In hook-backed OpenClaw mode, adaptive telemetry can consume
  replayed NeMo Relay events, while request-rewrite features such as adaptive
  hints require a managed execution path.
- `config.capture` controls prompt, response, tool argument, and tool result
  capture. Tool arguments and tool results are stripped by default because they
  often contain user data, local paths, tokens, or large payloads.
- `config.correlation` controls bounded in-memory hook correlation. By default,
  the plugin waits 250 ms for a matching `llm_input` after an `llm_output`,
  keeps correlation records for 600 seconds, and keeps at most 32 records per
  correlation key.

Fields inside `config.plugins` are NeMo Relay generic plugin configuration, so
they use `snake_case` regardless of language. For the full exporter field list,
see the NeMo Relay Observability Plugin schema in the top-level NeMo Relay
documentation at [docs.nvidia.com/nemo/relay](https://docs.nvidia.com/nemo/relay).

## Verify the Integration

Inspect the plugin runtime:

```bash
openclaw plugins inspect nemo-relay --runtime --json
```

Run an OpenClaw session with the plugin enabled, then verify the configured
sink:

- ATIF: confirm JSON files appear in the configured
  `config.plugins.components[].config.atif.output_directory`.
- OpenTelemetry: confirm spans arrive at the configured OTLP collector.
- OpenInference: confirm spans arrive at the configured OpenInference/Phoenix
  endpoint.

The plugin also registers the `operator.admin` scoped gateway method
`nemoRelay.status`. If your CLI is already paired with admin-capable gateway
access, run:

```bash
openclaw gateway call nemoRelay.status --json
```

## Current Limits

The plugin maps supported OpenClaw hook events into NeMo Relay telemetry and can
run pre-tool conditional guardrail checks.

It does not rewrite provider routing or provider request payloads.

Current OpenClaw public hooks expose request, response, message-write, and
provider timing details through separate event streams. The plugin correlates
those events within the same session, provider, model, and run. When timing
cannot be paired safely, it emits diagnostic marks instead of inventing
latency.

## Troubleshooting

If the plugin does not load, verify the package was installed with
`openclaw plugins install`, `plugins.allow` includes `nemo-relay`,
`plugins.entries["nemo-relay"].enabled` is not disabled, and the gateway was
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
npm run build --workspace=nemo-relay-openclaw
npm run typecheck --workspace=nemo-relay-openclaw
npm test --workspace=nemo-relay-openclaw
```

The CI-equivalent repo recipe is:

```bash
just --set ci true test-openclaw
```

Check the package payload before changing package metadata or entrypoints:

```bash
npm run pack:check --workspace=nemo-relay-openclaw
```

`npm run build --workspace=nemo-relay-openclaw` emits production files under
`integrations/openclaw/dist/`. Tests compile to
`integrations/openclaw/.test-dist/` from the sibling
`integrations/openclaw/test/` directory so test artifacts do not enter the
installable package or production source tree.

The optional live smoke test requires a working installed `nemo-relay-node`
binding:

```bash
npm run test:live --workspace=nemo-relay-openclaw
```
