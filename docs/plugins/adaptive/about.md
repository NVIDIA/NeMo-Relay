<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive

Use the Adaptive plugin when you want NeMo Flow to collect runtime signals and
activate measured adaptive behavior through the shared plugin system.

Adaptive is a first-party plugin component with kind `adaptive`. It uses the
same runtime model as the rest of NeMo Flow: scopes and managed calls emit
lifecycle events, subscribers and learners observe those events, intercepts can
add guidance, and plugin configuration controls what is active.

The plugin can coordinate:

- Adaptive state for learned runtime signals.
- Telemetry subscribers for adaptive learners.
- Adaptive hints injected into outgoing model requests.
- Tool-parallelism observation or scheduling behavior.
- Adaptive Cache Governor (ACG) prompt-cache planning.
- Component-local validation policy.

## Use Adaptive When

Adaptive is useful when an agent workflow repeats similar work and you want to
observe or tune behavior without hard-coding optimization logic into every
application.

Start here when you need to:

- Collect runtime signals before changing behavior.
- Add model-request hints in a controlled way.
- Plan prompt-cache breakpoints for supported providers.
- Evaluate tool parallelism opportunities.
- Share adaptive state across workers when needed.
- Roll out optimization as a configuration change.

If instrumentation is not in place yet, start with
[Instrument Applications](../../instrument-applications/about.md) or
[Integrate into Frameworks](../../integrate-frameworks/about.md).

## Pages

- [Adaptive Configuration](configuration.md) documents the full plugin
  component shape, validation, activation, teardown, and whole-plugin settings.
- [ACG](acg.md) explains Adaptive Cache Governor configuration and what prompt
  cache planning accomplishes.
- [Adaptive Hints](adaptive-hints.md) explains request hint injection and how
  downstream model paths can consume the hints.

State, telemetry, tool parallelism, and policy are whole-plugin configuration
areas. They are documented on [Adaptive Configuration](configuration.md) rather
than as separate area pages.
