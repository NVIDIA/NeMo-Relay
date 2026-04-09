<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Add a new guardrail or intercept type to the NeMo Flow middleware pipeline
---

# Add a Middleware Type

NeMo Flow supports guardrails (validate/gate) and intercepts (transform) at various
pipeline stages. Adding a new middleware type requires changes across all layers.

Use this skill when introducing a new middleware registration surface or adding
middleware behavior to a new pipeline stage.

## Lock The Design First

Decide these before editing code:

- Is this for tools, LLMs, or both?
- Is it a conditional guardrail, sanitize guardrail, request intercept, or
  execution intercept?
- Does it run on request input, inner callable execution, stream chunks, or
  final response output?
- Is the callback fallible, and how should callback failures propagate?
- Does it need both global and scope-local registration?
- What should subscribers observe in `event.input` and `event.output` after this
  middleware runs?

## Pipeline Order

See `docs/middleware-pipeline.md` for the full diagrams.

- **Tool execute**:
  conditional guardrails -> request intercepts -> sanitize request (for events)
  | execution intercept chain(callable) -> sanitize response
- **LLM execute**:
  conditional guardrails -> request intercepts -> sanitize request (for events)
  | execution intercept chain(callable) -> sanitize response

## Core Steps

1. Define or reuse the callback type alias in `crates/core/src/context.rs`.

```rust
pub type MyNewFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
```

2. Add the registry field to `NemoFlowContextState`.

Add a `SortedRegistry<GuardrailEntry<MyNewFn>>` or `SortedRegistry<Intercept<MyNewFn>>`
field to the state struct.

3. Add registration and deregistration APIs in `crates/core/src/api.rs`.

Use the existing `register_guardrail!` or `register_intercept!` macro patterns.
Both global and scope-local variants are needed (via `scope_register_guardrail!`).

4. Add chain execution helpers to `NemoFlowContextState`.

Follow the pattern of `tool_sanitize_request_chain` or `tool_request_intercepts_chain`.

5. Wire the chain into the execute path.

Update `nemo_flow_tool_call_execute` or `nemo_flow_llm_call_execute` in `api.rs`
to call the new chain method at the appropriate pipeline stage.

6. Expose the new middleware surface in every affected binding.

Follow the `add-binding-feature` skill for the cross-binding implementation checklist.

## Required Tests

- [ ] registration and duplicate-name behavior
- [ ] deregistration and no-op missing-name behavior
- [ ] ordering by priority
- [ ] callback error propagation
- [ ] scope-local registration, inheritance, and cleanup on pop
- [ ] event input/output semantics after middleware mutation
- [ ] parity coverage in every affected binding

## Key References

- Pipeline logic: `crates/core/src/api.rs`
- Type aliases: `crates/core/src/context.rs`
- Registry: `crates/core/src/registry.rs`
- Pipeline docs: `docs/middleware-pipeline.md`
- Concepts docs: `docs/concepts.md`
- Validation: `validate-change`
