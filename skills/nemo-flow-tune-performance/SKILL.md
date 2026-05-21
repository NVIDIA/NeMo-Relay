---
name: nemo-flow-tune-performance
description: Plan a measured NeMo Flow adaptive tuning rollout after baseline scopes, tool calls, LLM calls, and observability are working; use this skill to improve latency, tool parallelism, prompt-cache behavior, or model-request behavior from runtime signals
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Tune Performance With Adaptive Behavior

## Use This When

Use this skill when a user has baseline NeMo Flow instrumentation and wants to
improve latency, parallelism, prompt-cache behavior, or model-request behavior
from runtime signals.

## Do Not Use This When

Do not use this skill when the application is not instrumented yet. Start with
`nemo-flow-instrument-calls` or `nemo-flow-start` first.

## Default Guidance

- Observe first, compare against a baseline, then enable one behavior change at
  a time.
- Use the adaptive plugin component rather than inventing a separate optimizer
  object or hand-registering adaptive behavior at every call site.
- Start with in-memory state and telemetry-only behavior for local development.
- Move to persistent state only when learned signals must survive restarts or be
  shared across workers.
- Add active behavior only after representative runtime events show what should
  change.

## Embedded Adaptive Model

- Adaptive behavior is configured through the first-party plugin component with
  kind `adaptive`.
- Adaptive requires existing NeMo Flow scopes, managed tool or LLM calls, and
  lifecycle events because it learns from runtime signals.
- Main configuration areas are state, telemetry, adaptive hints, tool
  parallelism, Adaptive Cache Governor (ACG), and rollout policy.
- State backends are `in_memory` and `redis`.
- Tool-parallelism modes are `observe_only`, `inject_hints`, and `schedule`.
- Adaptive Cache Governor providers are `passthrough`, `anthropic`, and
  `openai`; omit ACG until prompt-cache planning is needed.
- Helper APIs exist in Rust `nemo_flow_adaptive`, Python `nemo_flow.adaptive`,
  and Node.js `nemo-flow-node/adaptive`. Go, WebAssembly, and raw FFI are
  source-first or advanced surfaces.

## Default Path

1. Confirm the app already emits expected scope, tool, and LLM events.
2. Capture a baseline for the workflow you want to improve.
3. Enable adaptive telemetry with in-memory state.
4. Run representative traffic and inspect reports or runtime events.
5. Choose one tuning surface: hints, tool parallelism, or ACG.
6. Enable the smallest behavior change in config.
7. Compare results against the baseline and keep a rollback path.

## Failure Modes To Avoid

- Do not enable scheduling before tool idempotency and race behavior are known.
- Do not enable prompt-cache planning before provider payloads are stable.
- Do not treat adaptive hints as mandatory instructions unless the consuming
  path explicitly defines that contract.
- Do not use environment variables as the primary adaptive configuration model.
- Do not tune from a single run or unrepresentative traffic.

## Use Another Skill When

- You need the exact adaptive config shape -> `nemo-flow-tune-adaptive-config`
- You need to consume adaptive hints or scheduling guidance in app logic ->
  `nemo-flow-tune-adaptive-hints`

## Related Skills

- `nemo-flow-start`
- `nemo-flow-instrument-calls`
- `nemo-flow-setup-observability`
- `nemo-flow-tune-adaptive-config`
- `nemo-flow-tune-adaptive-hints`
