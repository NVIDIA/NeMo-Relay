---
name: nemo-flow-start-optimizer
description: Help application developers decide whether and how to start using the NeMo Flow adaptive layer; also handle legacy optimizer wording
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Get Started With Adaptive Layer

Use this skill when a user wants the shortest explanation of what the adaptive
layer does and how to take a first step with it. If they say "optimizer",
translate that to the current adaptive/plugin model.

## Default Guidance

- Treat adaptive as a config-driven top-level plugin component layered on top of
  NeMo Flow instrumentation.
- Start with the in-memory backend and one built-in section at a time.
- Validate the full plugin config before initialization.
- Add custom plugins only after the baseline adaptive path works.

## Embedded Adaptive Model

- Adaptive is the current name for functionality that users may still call the
  optimizer. Translate optimizer wording to adaptive/plugin wording.
- Adaptive requires existing NeMo Flow instrumentation because it learns from
  emitted scope, tool, and LLM lifecycle events.
- Adaptive can register subscribers for telemetry, LLM request intercepts for
  hints, tool-related behaviors for parallelism guidance or scheduling, LLM
  execution intercepts for cache-governor planning, and state backends for those
  features.
- Main config areas are state, telemetry, adaptive hints, tool parallelism, the
  adaptive cache governor, and rollout policy.
- State backends are `in_memory` and `redis`.
- Tool-parallelism modes are `observe_only`, `inject_hints`, and `schedule`.
- Adaptive cache-governor providers are `passthrough`, `anthropic`, and
  `openai`; omit the cache-governor section until cache planning is needed.
- Helper APIs exist in Python `nemo_flow.adaptive`, Node.js
  `nemo-flow-node/adaptive`, Go `go/nemo_flow/adaptive`, and Rust
  `nemo_flow_adaptive`.
- A safe first rollout is in-memory state plus telemetry, followed by
  representative workflows, report review, optional persistent state, then hints
  or scheduling only after consumers can interpret them.

## First Questions To Answer

- Does the app already emit NeMo Flow events?
- Does it need telemetry-driven learning, LLM hints, tool parallelism, or a
  custom plugin?
- Does it need an in-memory or persistent state backend?

## Use Another Skill When

- You already know the configuration shape you need ->
  `nemo-flow-configure-optimizer`
- You need to consume the hints/predictions in app logic ->
  `nemo-flow-use-optimizer-hints`

## Related Skills

- `nemo-flow-instrument-calls`
- `nemo-flow-configure-optimizer`
- `nemo-flow-use-optimizer-hints`
