<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Consume NeMo Flow adaptive outputs such as hints, predictions, and parallelism guidance in application logic; use this when users still say optimizer
---

# Use Adaptive Predictions And Hints

Use this skill when the adaptive layer is already configured and the
application wants to act on its outputs.

## Focus Areas

- `adaptive_hints` or model request hints injected by adaptive components
- latency sensitivity and scheduling advice
- parallel groups or tool-parallelism guidance
- config reports and diagnostics during rollout

## Rules

- treat adaptive output as guidance unless the consuming API explicitly requires
  stronger semantics
- confirm where the hint is injected or surfaced in the chosen binding
- keep the app behavior understandable when no prediction is available

## References

- `docs/adaptive-layer.md`
- `docs/adaptive-api-reference.md`
- `docs/online-learning-engine.md`
