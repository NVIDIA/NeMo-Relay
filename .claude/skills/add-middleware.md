<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Add a new guardrail or intercept type to the Nexus middleware pipeline
---

# Adding a New Middleware Type

Nexus supports guardrails (validate/gate) and intercepts (transform) at various
pipeline stages. Adding a new middleware type requires changes across all layers.

## Understanding the Pipeline

See `docs/middleware-pipeline.md` for the full pipeline diagrams. The pipeline order is:

**Tool execute**: conditional guardrails → request intercepts →
sanitize request (for events) | execution intercept chain(callable) → sanitize response

**LLM execute**: conditional guardrails → request intercepts →
sanitize request (for events) | execution intercept chain(callable) → sanitize response

## Steps

### 1. Define the callback type alias in `crates/core/src/context.rs`

```rust
pub type MyNewFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
```

### 2. Add registry field to `NatNexusContextState`

Add a `SortedRegistry<GuardrailEntry<MyNewFn>>` or `SortedRegistry<Intercept<MyNewFn>>`
field to the state struct.

### 3. Add registration macros in `crates/core/src/api.rs`

Use the existing `register_guardrail!` or `register_intercept!` macro patterns.
Both global and scope-local variants are needed (via `scope_register_guardrail!`).

### 4. Add chain method to `NatNexusContextState`

Follow the pattern of `tool_sanitize_request_chain` or `tool_request_intercepts_chain`.

### 5. Wire into the execute pipeline

Update `nat_nexus_tool_call_execute` or `nat_nexus_llm_call_execute` in `api.rs`
to call the new chain method at the appropriate pipeline stage.

### 6. Expose in all bindings

Follow the `add-binding-feature` skill for the cross-binding implementation checklist.

## Key Files

- Pipeline logic: `crates/core/src/api.rs`
- Type aliases: `crates/core/src/context.rs`
- Registry: `crates/core/src/registry.rs`
- Pipeline docs: `docs/middleware-pipeline.md`
- Concepts docs: `docs/concepts.md`
