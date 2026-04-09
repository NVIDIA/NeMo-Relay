<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Help application developers decide whether and how to start using the NeMo Flow optimizer layer
---

# Get Started With Optimizer

Use this skill when a user wants the shortest explanation of what the optimizer
does and how to take a first step with it.

## Default Guidance

- Treat the optimizer as a config-driven runtime layered on top of NeMo Flow
  instrumentation.
- Start with the in-memory backend and built-in components.
- Validate the config before registering the runtime.
- Add hosted plugins only after the baseline built-in path works.

## First Questions To Answer

- Does the app already emit NeMo Flow events?
- Does it need telemetry-driven learning, LLM hints, tool parallelism, or a
  hosted plugin?
- Does it need an in-memory or persistent state backend?

## Use Another Skill When

- you already know the runtime shape you need -> `configure-optimizer-runtime`
- you need to consume the hints/predictions in app logic ->
  `use-optimizer-predictions-and-hints`

## References

- `docs/optimizer-layer.md`
- `docs/optimizer-api-reference.md`
