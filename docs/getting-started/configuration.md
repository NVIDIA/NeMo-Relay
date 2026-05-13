<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configuration

NeMo Flow runtime behavior is configured through API objects and registration calls rather than a global configuration file.

## Core Runtime Setup

Most applications configure NeMo Flow by:

1. Creating or reusing a scope stack.
2. Registering guardrails, intercepts, or subscribers.
3. Calling the managed tool or LLM helpers from the active scope.
4. Deregistering global middleware that should not remain active for the lifetime of the process.

Use scope-local registration when behavior must be tied to one request, session, or agent run.

## Plugin Setup

Plugins use a structured plugin configuration with:

- A version
- One or more component definitions
- Optional component policy

Start with [Basic Guide: Define a Plugin](../build-plugins/basic-guide.md) when you need reusable middleware, subscribers, or adaptive behavior.

## Observability Setup

ATOF exporters, ATIF exporters, OpenTelemetry subscribers, and OpenInference
subscribers can be configured directly through binding-native config objects.
Use the built-in `observability` plugin when you want one plugin component to
own standard exporter setup and teardown. See
[Configure the Observability Plugin](../export-observability-data/observability-plugin.md)
and [Export Observability Data](../export-observability-data/code-examples.md)
for the supported export paths.

NeMo Flow does not require application-level environment variables for normal
runtime use. Configure most behavior through API objects, registration calls, or
plugin configuration.

`OTEL_*` variables are only relevant when the underlying OpenTelemetry exporter
reads endpoint settings from the environment. Prefer explicit config objects in
application code so the active export settings are visible in docs, tests, and
deployment manifests.

## Adaptive Setup

Adaptive optimization is enabled through the adaptive plugin component and binding helper APIs. See [Configure Adaptive Optimization](../use-adaptive-optimization/configure.md).

## CLI Gateway Setup

The `nemo-flow` CLI gateway is the exception to the "no global config file"
rule because it runs outside an application process. Use it when you want local
observability for coding-agent sessions such as Claude Code, Codex, Cursor, or
Hermes.

Run the setup wizard:

```bash
nemo-flow
```

The wizard writes project config at `.nemo-flow/config.toml`, user config at
`$XDG_CONFIG_HOME/nemo-flow/config.toml` or `~/.config/nemo-flow/config.toml`,
or both. System config can be supplied at `/etc/nemo-flow/config.toml` for
shared machines and CI images.

When no explicit `--config` path is passed, CLI config precedence is:

1. Built-in defaults
2. `/etc/nemo-flow/config.toml`
3. Nearest project `.nemo-flow/config.toml`, walking up from the current
   directory
4. `$XDG_CONFIG_HOME/nemo-flow/config.toml`, or
   `~/.config/nemo-flow/config.toml`
5. `NEMO_FLOW_*` environment variables
6. Current command flags

Use the current exporter shape in new config files:

```toml
[exporters.atif]
dir = ".nemo-flow/atif"

[exporters.atof]
dir = ".nemo-flow/atof"
mode = "append"
filename_template = "{session_id}.jsonl"

[exporters.openinference]
endpoint = "http://localhost:6006/v1/traces"

[agents.codex]
command = "codex"
```

For CLI gateway plugin activation, use exactly one source per invocation:
`--plugin-config` JSON, `[plugins].config` inside `config.toml`, or
`plugin.toml` in the discovered system, project, or user config locations.

See [CLI Gateway Quick Start](cli.md) for wizard commands and
[Advanced Guide: Coding-Agent Gateway](../integrate-frameworks/coding-agent-gateway.md)
for daemon mode, transparent runs, hook forwarding, and per-agent guides.
