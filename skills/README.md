<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Consumer Skills

This directory contains consumer-facing NeMo Relay skills for application
developers, integrators, and end users.

Public skill directories use a `nemo-relay-` prefix so they remain recognizable
and collision-resistant when exported outside this repository.

Skills in this directory are self-contained. A skill can point to another skill
in this directory, but it must not rely on repository documentation files for
required task guidance. If a skill needs behavior, API, or workflow details from
documentation, embed those details directly in the skill.

Use these skills for tasks such as:

- Getting started with a binding
- Migrating NeMo Flow codebases to NeMo Relay
- Instrumenting tool and LLM calls
- Choosing the current primary documentation track: Rust, Python, or Node.js
- Tuning performance with adaptive features
- Building reusable plugin behavior
- Setting up observability and trace export
- Debugging application-side NeMo Relay integrations

When a skill mentions Go or raw FFI, treat those as source-first advanced
surfaces. Their APIs are tracked in `go/nemo_relay` and `crates/ffi`, but the
primary end-user docs and quick starts focus on Rust, Python, and Node.js.

Maintainer-only repository development skills live in `.agents/skills/`.
The agent-specific directories `.claude/skills`, `.codex/skills`, and
`.cursor/skills` point to that maintainer set.
