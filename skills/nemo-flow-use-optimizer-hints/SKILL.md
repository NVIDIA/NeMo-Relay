---
name: nemo-flow-use-optimizer-hints
description: Consume NeMo Flow adaptive outputs such as hints, predictions, and parallelism guidance in application logic; use this when users still say optimizer
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Use Adaptive Predictions And Hints

Use this skill when the adaptive layer is already configured and the
application wants to act on its outputs.

## Focus Areas

- `adaptive_hints` or model request hints injected by adaptive components
- Latency sensitivity and scheduling advice
- Parallel groups or tool-parallelism guidance
- Config reports and diagnostics during rollout

## Embedded Hint Semantics

- Adaptive hints are request-intercept behavior. They can inject metadata into a
  configured header or body path; the default body path is
  `nvext.agent_hints`.
- Hint configuration includes `priority`, `break_chain`, `inject_header`, and
  `inject_body_path`. Lower priority values run earlier; adjust priority when
  hints conflict with application intercepts.
- Tool parallelism can run in `observe_only`, `inject_hints`, or `schedule`.
  Use `observe_only` until tool idempotency and race behavior are understood.
- Adaptive cache-governor output is provider-specific prompt-cache planning
  guidance. Use more samples, raise stability thresholds, or switch to
  `passthrough` when cache planning is unstable.
- `set_latency_sensitivity(...)` is a request-local execution hint, not
  persistent adaptive configuration.
- Normal adaptive runtime behavior should come from explicit config objects, not
  environment variables. `NEMO_FLOW_ACG_DEBUG` is for cache-governor diagnostics
  and `NEMO_FLOW_RUN_REDIS_TESTS` is for Redis-backed tests.
- Treat reports and diagnostics as rollout evidence: application results should
  remain unchanged unless scheduling or request metadata changes were
  intentional.

## Rules

- Treat adaptive output as guidance unless the consuming API explicitly requires
  stronger semantics
- Confirm where the hint is injected or surfaced in the chosen binding
- Keep the app behavior understandable when no prediction is available

## Related Skills

- `nemo-flow-start-optimizer`
- `nemo-flow-configure-optimizer`
- `nemo-flow-debug-runtime-integration`
