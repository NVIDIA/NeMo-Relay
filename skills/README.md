<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay User Skills

These are user-facing skills for installing, trying, integrating, configuring,
and troubleshooting NeMo Relay. They are intended for application developers,
framework integrators, operators, and users applying Relay to their own agents
and applications.

If you are developing NeMo Relay itself, changing core or binding APIs,
maintaining repository infrastructure, or preparing a Relay contribution, use
the [Relay maintainer skills](../.agents/skills/README.md) in `.agents/skills/`
instead. Compatible coding agents discover that standard directory directly;
`.claude/skills` is a symlink that exposes the same maintainer set to Claude
Code.

## Start Here

Choose the entry point that matches the user's current outcome:

- **New to Relay**: use [`nemo-relay-install`](nemo-relay-install/SKILL.md), then
  [`nemo-relay-get-started`](nemo-relay-get-started/SKILL.md).
- **Want the fastest proof of value**: use
  [`nemo-relay-get-started`](nemo-relay-get-started/SKILL.md). It selects the
  first path compatible with the user's goal and environment: the CLI for a
  generic trial, a maintained integration such as LangChain or LangGraph when
  already present, or language-specific manual integration.
- **Already know where your application calls tools or models**: use
  [`nemo-relay-instrument-calls`](nemo-relay-instrument-calls/SKILL.md).
- **Already emit Relay events and want useful output**: start with
  [`nemo-relay-plugin-observability`](nemo-relay-plugin-observability/SKILL.md).
- **Something is not loading or emitting events**: use
  [`nemo-relay-debug-runtime-integration`](nemo-relay-debug-runtime-integration/SKILL.md).
- **Using Codex Desktop**: prefer the temporary CLI try-now path. Before a
  persistent Codex install changes global provider configuration,
  [`nemo-relay-install`](nemo-relay-install/SKILL.md) warns about the current
  [history-visibility bug](https://github.com/openai/codex/issues/24648) and
  creates a workspace recovery file with undo instructions.

## Onboarding

Use this table for installation and first-value onboarding:

| Skill | Use It When |
|---|---|
| [`nemo-relay-install`](nemo-relay-install/SKILL.md) | Choose and install the CLI, a language package, or a maintained framework or harness integration. |
| [`nemo-relay-get-started`](nemo-relay-get-started/SKILL.md) | Reach a first observable Relay result through the least complicated applicable try-now path. |

## Instrument Applications

Use this table when an application directly owns the execution boundary:

| Skill | Use It When |
|---|---|
| [`nemo-relay-instrument-calls`](nemo-relay-instrument-calls/SKILL.md) | Wrap application-owned tool or LLM/provider calls with scopes and managed execution. |
| [`nemo-relay-instrument-context-isolation`](nemo-relay-instrument-context-isolation/SKILL.md) | Keep scope context isolated across concurrent requests, tasks, threads, or agent runs. |
| [`nemo-relay-instrument-typed-wrappers`](nemo-relay-instrument-typed-wrappers/SKILL.md) | Add typed wrappers or provider codecs while preserving Relay middleware behavior. |

## Configure And Build Plugins

Use this table to select reusable Relay behavior or output:

| Skill | Use It When |
|---|---|
| [`nemo-relay-plugin-observability`](nemo-relay-plugin-observability/SKILL.md) | Inspect or export Relay activity through subscribers, ATOF, ATIF, OpenTelemetry, or OpenInference. This is the recommended first plugin for most users. |
| [`nemo-relay-plugin-adaptive-tuning`](nemo-relay-plugin-adaptive-tuning/SKILL.md) | Configure and measure adaptive hints, tool parallelism, cache behavior, or other adaptive runtime features. |
| [`nemo-relay-plugin-build`](nemo-relay-plugin-build/SKILL.md) | Package reusable runtime behavior as a validated, configuration-driven plugin for applications or integrations. |

## Migrate And Troubleshoot

Use this table for existing integrations that need migration or diagnosis:

| Skill | Use It When |
|---|---|
| [`nemo-relay-migrate-from-flow`](nemo-relay-migrate-from-flow/SKILL.md) | Migrate an application, integration, configuration, or documentation surface from NeMo Flow to NeMo Relay. |
| [`nemo-relay-debug-runtime-integration`](nemo-relay-debug-runtime-integration/SKILL.md) | Diagnose loading failures, inactive scopes, missing events, or plugin and adaptive wiring problems in an application-side integration. |

## Common Journeys

Use these sequences to choose the next workflow step:

1. **Evaluate Relay locally**: install -> get started with Observability -> add
   one goal-aligned plugin.
2. **Instrument an application**: install -> get started -> add one plugin ->
   expand call instrumentation, context isolation, or typed wrappers only when
   the demonstrated boundary is insufficient.
3. **Use an existing framework**: install the maintained integration -> get
   started with its built-in path -> add observability or another plugin based
   on the desired outcome.
4. **Package reusable behavior**: prove the behavior in one application ->
   build a plugin -> validate it in the target integration.
5. **Recover a broken setup**: verify installation -> run the relevant doctor
   checks -> debug the smallest failing runtime boundary.
