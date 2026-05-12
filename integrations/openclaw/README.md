<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-openclaw

`nemo-flow-openclaw` is the NeMo Flow observability plugin package for
OpenClaw. It converts supported OpenClaw hook events into NeMo Flow sessions,
LLM spans, tool spans, lifecycle marks, ATIF JSON, OpenTelemetry spans, and
OpenInference/Phoenix spans.

## Why Use It?

- Observe OpenClaw sessions without patching OpenClaw.
- Export OpenClaw activity into NeMo Flow observability formats.
- Preserve OpenClaw's agent, tool, and LLM lifecycle context where public hooks
  expose enough data.
- Keep ambiguous LLM timing attribution visible through diagnostic marks instead
  of unsafe latency.

## What You Get

- OpenClaw plugin id `nemo-flow`.
- ATIF JSON export enabled by default.
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
manifest id `nemo-flow` for configuration.

## Getting Started

Enable the `nemo-flow` plugin id and grant conversation hook access when
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
        "config": {}
      }
    }
  }
}
```

Plugin configuration lives under `plugins.entries["nemo-flow"].config`.

## Documentation

For full configuration, verification, troubleshooting, and current LLM replay
fidelity limits, use the
[OpenClaw Plugin Guide](../../docs/integrate-frameworks/openclaw-plugin.md).

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
