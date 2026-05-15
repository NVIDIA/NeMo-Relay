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

Start with [Define a Plugin](../build-plugins/basic-guide.md) when you need reusable middleware, subscribers, or adaptive behavior.

The `nemo-flow` CLI gateway reads plugin files named `plugins.toml`. See
[Plugin Configuration Files](../build-plugins/plugin-configuration-files.md)
for file locations, precedence, merge behavior, editor controls, and validation
rules.

## Observability Setup

ATOF exporters, ATIF exporters, OpenTelemetry subscribers, and OpenInference
subscribers can be configured directly through binding-native config objects.
Use the built-in `observability` plugin when you want one plugin component to
own standard exporter setup and teardown. See
[Observability Configuration](../plugins/observability/configuration.md)
and [Observability](../plugins/observability/about.md)
for the supported export paths.

NeMo Flow does not require application-level environment variables for normal
runtime use. Configure most behavior through API objects, registration calls, or
plugin configuration.

`OTEL_*` variables are only relevant when the underlying OpenTelemetry exporter
reads endpoint settings from the environment. Prefer explicit config objects in
application code so the active export settings are visible in docs, tests, and
deployment manifests.

## Adaptive Setup

Adaptive optimization is enabled through the adaptive plugin component and binding helper APIs. See [Adaptive Configuration](../plugins/adaptive/configuration.md).
