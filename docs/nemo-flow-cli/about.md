<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# About

Use this section when you want the `nemo-flow` binary to observe local coding
agent sessions through hooks, a passthrough LLM gateway, and NeMo Flow
observability exporters.

The NeMo Flow CLI is installed by the `nemo-flow-cli` Cargo package. It can run
supported coding agents through a managed local gateway, forward agent hook
payloads into NeMo Flow lifecycle events, route OpenAI-compatible or
Anthropic-compatible model traffic through the gateway, and diagnose local
configuration.

## Start Here When

Use these guides when you need to:

- Observe Claude Code, Codex, Cursor, or Hermes Agent sessions locally.
- Configure coding-agent hooks for NeMo Flow lifecycle events.
- Route model-provider traffic through the local NeMo Flow gateway.
- Export local sessions to Agent Trajectory Interchange Format (ATIF), Agent
  Trajectory Observability Format (ATOF) JSONL, OpenTelemetry, or
  OpenInference.
- Diagnose hook loading, gateway routing, and exporter output.

If you are instrumenting an application or framework directly, use
[Instrument Applications](../instrument-applications/about.md) or
[Integrate into Frameworks](../integrate-frameworks/about.md) instead.

## Agent Harness Support

NeMo Flow CLI support is experimental and observability-focused.

| Agent | Observability | Security | Optimization | Notes |
| --- | --- | --- | --- | --- |
| Claude Code | ✅ Yes | ❌ No | ❌ No | Observability only; no known issues. |
| Codex | ✅ Yes | ❌ No | ❌ No | Observability only; some hooks needed for full feature coverage are missing. |
| Hermes Agent | ✅ Yes | ❌ No | ❌ No | Observability only; no known issues. |
| Cursor | ✅ Yes | ❌ No | ❌ No | Observability only; missing hooks under `cursor-agent` limit feature coverage. |

## Guides

Use these guide links to move from CLI setup into agent-specific instructions.

- [Basic Usage](basic-usage.md) explains gateway routes, transparent runs,
  shared configuration, hook forwarding, and runtime mapping.
- [Claude Code](claude-code.md) covers transparent Claude Code
  runs, Anthropic gateway routing, ATIF verification, and unsupported Claude
  application modes.
- [Codex](codex.md) covers transparent Codex CLI runs, local
  GUI/app caveats, model provider routing, and remote-task limits.
- [Cursor](cursor.md) covers transparent Cursor runs, temporary
  hook patching, GUI and CLI smoke tests, and gateway routing limits.
- [Hermes Agent](hermes.md) covers Hermes shell hook installation,
  dynamic gateway URL handling, session-finalize behavior, and hook consent
  caveats.

Start with [Basic Usage](basic-usage.md), then use the guide for the coding
agent that you want to observe.
