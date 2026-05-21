---
name: nemo-flow-start-optimizer
description: Deprecated compatibility alias for nemo-flow-tune-performance; retained through v0.3 for users who still ask for optimizer guidance
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Deprecated Compatibility Alias For `nemo-flow-tune-performance`

This legacy skill name is retained through v0.3 for compatibility. NeMo Flow
now describes this area as adaptive tuning or performance tuning, not a
separate optimizer object.

If this alias is selected, continue with the current Phase 2 workflow rather
than looking for a separate optimizer API.

## Compatibility Workflow

1. Translate "optimizer" to adaptive tuning through the first-party plugin
   component with kind `adaptive`.
2. Confirm the app already has scopes, managed tool or LLM calls, and
   observability working.
3. Start with in-memory state and telemetry-only behavior.
4. Run representative traffic and inspect runtime events or reports.
5. Enable only one active tuning surface at a time: adaptive hints, tool
   parallelism, or Adaptive Cache Governor.
6. Compare against a baseline and keep a rollback path.

## Avoid

- Do not introduce a separate optimizer object.
- Do not enable scheduling before tool idempotency and race behavior are known.
- Do not tune from a single run or unrepresentative traffic.

If the user says "optimizer," translate that to current adaptive/plugin
terminology and continue with `nemo-flow-tune-performance`.
