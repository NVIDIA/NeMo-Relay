<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Help application developers decide whether and how to start using the NeMo Flow adaptive layer; also handle legacy optimizer wording
---

# Get Started With Adaptive Layer

Use this skill when a user wants the shortest explanation of what the adaptive
layer does and how to take a first step with it. If they say "optimizer",
translate that to the current adaptive/plugin-host model.

## Default Guidance

- Treat adaptive as a config-driven top-level plugin component layered on top of
  NeMo Flow instrumentation.
- Start with the in-memory backend and one built-in section at a time.
- Validate the full plugin config before initialization.
- Add custom plugins only after the baseline adaptive path works.

## First Questions To Answer

- Does the app already emit NeMo Flow events?
- Does it need telemetry-driven learning, LLM hints, tool parallelism, or a
  custom plugin?
- Does it need an in-memory or persistent state backend?

## Use Another Skill When

- you already know the configuration shape you need -> `configure-optimizer-runtime`
- you need to consume the hints/predictions in app logic ->
  `use-optimizer-predictions-and-hints`

## References

- `docs/adaptive-layer.md`
- `docs/adaptive-api-reference.md`
- `docs/online-learning-engine.md`
