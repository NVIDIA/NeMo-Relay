---
name: add-middleware
description: Add a new guardrail or intercept type to the NeMo Flow middleware pipeline
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Add a Middleware Type

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

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

See `docs/about/concepts/middleware.md` for the full diagrams.

- **Tool execute**:
  conditional guardrails -> request intercepts -> sanitize request (for events)
  | execution intercept chain(callable) -> sanitize response
- **LLM execute**:
  conditional guardrails -> request intercepts -> sanitize request (for events)
  | execution intercept chain(callable) -> sanitize response

## Core Steps

1. Define or reuse the callback type alias in
   `crates/core/src/api/runtime/callbacks.rs`.

```rust
pub type MyNewFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
```

2. Add the registry field to `NemoFlowContextState` in
   `crates/core/src/api/runtime/state.rs`.

Add a `SortedRegistry<GuardrailEntry<MyNewFn>>` or `SortedRegistry<Intercept<MyNewFn>>`
field to the state struct.

3. Add registration and deregistration APIs in `crates/core/src/api/`.

Use the existing `global_*_registry_api!` and `scope_*_registry_api!` macro
patterns in `crates/core/src/api/registry.rs`. Both global and scope-local
variants are needed unless the design explicitly rules one out.

4. Add chain execution helpers to `NemoFlowContextState` in
   `crates/core/src/api/runtime/state.rs`.

Follow the pattern of `tool_sanitize_request_chain` or `tool_request_intercepts_chain`.

5. Wire the chain into the execute path.

Update `crates/core/src/api/tool.rs` or `crates/core/src/api/llm.rs` to call
the new chain method at the appropriate pipeline stage.

6. Expose the new middleware surface in every affected binding.

Follow the `add-binding-feature` skill for the cross-binding implementation checklist.

## Required Tests

- [ ] Registration and duplicate-name behavior
- [ ] Deregistration and no-op missing-name behavior
- [ ] Ordering by priority
- [ ] Callback error propagation
- [ ] Scope-local registration, inheritance, and cleanup on pop
- [ ] Event input/output semantics after middleware mutation
- [ ] Parity coverage in every affected binding

## Key References

- Pipeline logic: `crates/core/src/api/tool.rs`, `crates/core/src/api/llm.rs`
- Type aliases: `crates/core/src/api/runtime/callbacks.rs`
- Runtime state and chain builders: `crates/core/src/api/runtime/state.rs`
- Scope-local registry merging: `crates/core/src/context/registries.rs`
- Registry: `crates/core/src/registry.rs`
- Pipeline docs: `docs/about/concepts/middleware.md`
- Architecture docs: `docs/about/architecture.md`
- Registration examples: `docs/instrument-applications/advanced-guide.md`
- Validation: `validate-change`
