---
name: add-middleware
description: Add a new guardrail or intercept type to the NeMo Relay middleware pipeline
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Add a Middleware Type

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

NeMo Relay supports guardrails (validate/gate) and intercepts (transform) at various
pipeline stages. Adding a new middleware type requires changes across all layers.

Use this skill when introducing a new middleware registration surface or adding
middleware behavior to a new pipeline stage.

## Lock The Design First

Decide these before editing code:

- Is this for tools, LLMs, marks, scope events, or a combination?
- Is it a conditional guardrail, sanitize guardrail, request intercept, or
  execution intercept?
- Does it run on request input, inner callable execution, stream chunks, or
  final response output?
- Is the callback fallible, and how should callback failures propagate?
- Does it need both global and scope-local registration?
- What should subscribers and exporters observe in the event payload after this
  middleware runs?
- If this is an event sanitizer, which of `data`, `category_profile`, and
  `metadata` may change, and is the event used only as immutable context?

## Pipeline Order

See `docs/about-nemo-relay/concepts/middleware.mdx` for the full diagrams.

- **Tool execute**:
  conditional guardrails -> request intercepts -> sanitize request (for events)
  | execution intercept chain(callable) -> sanitize response
- **LLM execute**:
  conditional guardrails -> request intercepts -> sanitize request (for events)
  | execution intercept chain(callable) -> sanitize response
- **Mark and scope events**:
  specialized tool or LLM sanitizer (when applicable) -> mark or scope event
  sanitizer -> subscriber and exporter dispatch

## Core Steps

1. Define or reuse the callback type alias in
   `crates/core/src/api/runtime/callbacks.rs`.

```rust
pub type MyNewFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
```

2. Add the registry field to `NemoRelayContextState` in
   `crates/core/src/api/runtime/state.rs`.

Add a `SortedRegistry<GuardrailEntry<MyNewFn>>` or `SortedRegistry<Intercept<MyNewFn>>`
field to the state struct.

3. Add registration and deregistration APIs in `crates/core/src/api/`.

Use the existing `global_*_registry_api!` and `scope_*_registry_api!` macro
patterns in `crates/core/src/api/registry.rs`. Both global and scope-local
variants are needed unless the design explicitly rules one out.

4. Add chain execution helpers to `NemoRelayContextState` in
   `crates/core/src/api/runtime/state.rs`.

Follow the pattern of `tool_sanitize_request_chain` or `tool_request_intercepts_chain`.

5. Wire the chain into the execute path.

Update the relevant lifecycle owner to call the new chain method at the
appropriate pipeline stage. Tool and LLM paths live in
`crates/core/src/api/tool.rs` and `crates/core/src/api/llm.rs`; shared mark and
scope event sanitization lives in `crates/core/src/api/shared.rs` and is called
from `crates/core/src/api/scope.rs`.

6. Expose the new middleware surface in every affected binding.

Follow the `add-binding-feature` skill for the cross-binding implementation checklist.

## Required Tests

- [ ] Registration and duplicate-name behavior
- [ ] Deregistration and no-op missing-name behavior
- [ ] Ordering by priority
- [ ] Callback error propagation
- [ ] Scope-local registration, inheritance, and cleanup on pop
- [ ] Event payload semantics after middleware mutation
- [ ] Mark and scope event field semantics, including immutable identity fields
- [ ] Parity coverage in every affected binding

## Key References

- Pipeline logic: `crates/core/src/api/tool.rs`, `crates/core/src/api/llm.rs`
- Type aliases: `crates/core/src/api/runtime/callbacks.rs`
- Runtime state and chain builders: `crates/core/src/api/runtime/state.rs`
- Scope-local registry merging: `crates/core/src/context/registries.rs`
- Registry: `crates/core/src/registry.rs`
- Pipeline docs: `docs/about-nemo-relay/concepts/middleware.mdx`
- Architecture docs: `docs/about-nemo-relay/architecture.mdx`
- Registration examples: `docs/instrument-applications/advanced-guide.mdx`
- Validation: `validate-change`
