<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Highlights

This page summarizes the notable capabilities in the current release documentation set.

## NeMo Flow 0.2

This release of NeMo Flow release introduces several new components and capabilities.
The complete changelog and release notes can be viewed on [GitHub](https://github.com/NVIDIA/NeMo-Flow/releases).

### NeMo Flow CLI

- Added CLI support for coding-agent harnesses, including Claude Code, Codex, Cursor, and Hermes Agent.
- Enhanced setup and configuration workflows with guided onboarding, environment diagnostics, plugin configuration loading, and plugin editing.

### Integrations

- Added OpenClaw observability support through a first-party `nemo-flow-openclaw` plugin.
- Added middleware and runtime integrations for LangChain, LangGraph, and Deep Agents. These are accessed through the Python `nemo-flow` package through extras.

### Observability

- Introduced a core Observability plugin that globally configures all built-in observability exporters and subscribers.
- Added an Agent Trajectory Observability Format (ATOF) JSONL exporter for structured event output.
- Expanded Codec support for optional annotated LLM request and response fields.

### Security

- Added an example NeMo Guardrails plugin.

### Agent Skill Improvements

- Updated Python binding test guidance for agent skills.
- Added PR template requirements to agent contribution guidance.
