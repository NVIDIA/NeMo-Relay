---
name: nemo-relay-instrument-calls
description: Wrap application tool calls and LLM/provider calls with NeMo Relay scopes and managed execution APIs
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

<!-- Non-semantic CI refresh marker: 2026-06-06. -->

# Instrument Tool And LLM Calls

Use this skill when an app already has tool functions or model/provider calls and
needs to run them through NeMo Relay correctly.

## Default Guidance

- Put a scope around the natural agent, request, workflow, or graph boundary.
- Use managed execution APIs first:
  - Rust: `tool_call_execute(ToolCallExecuteParams::builder()...)`,
    `llm_call_execute(LlmCallExecuteParams::builder()...)`
  - Python: `tools.execute(...)`, `llm.execute(...)`
  - Node.js: `toolCallExecute(...)`, `llmCallExecute(...)`
  - Go: `tools.Execute(...)`, `llm.Execute(...)` or the top-level wrappers
  - WebAssembly: `toolCallExecute(...)`, `llmCallExecute(...)`
- Use manual lifecycle APIs only when the host framework cannot be wrapped by the
  managed execute helpers.

## Embedded Runtime Semantics

- Managed tool and LLM execution runs conditional-execution guardrails first on
  the raw input. If rejected, the runtime emits a standalone mark event and does
  not run request intercepts or the callable.
- Request intercepts run after conditional guardrails and rewrite the real input
  that reaches execution intercepts and the callback.
- Sanitize-request guardrails affect emitted start-event payloads only. They do
  not rewrite the caller-visible request or arguments.
- Execution intercepts wrap the callback with the middleware `next` pattern and
  may short-circuit by returning their own result.
- Sanitize-response guardrails affect emitted end-event payloads only. The value
  returned to application code remains the raw callback or execution-intercept
  result.
- If execution fails after the start event has been emitted, the runtime still
  emits an end event without a semantic output payload.
- Tool calls are named operations with JSON-compatible arguments and results.
  Keep the original tool callable responsible for business logic; let NeMo Relay
  own lifecycle events, middleware, and metadata.
- LLM calls use an `LLMRequest` made of metadata plus content. Pass model names
  and stable call identifiers when they matter for trace export or diagnostics.
- Manual lifecycle APIs are for framework adapters that already own execution.
  If you use them, every start call needs a matching end or error path with the
  relevant semantic payloads supplied explicitly.
- Partial middleware APIs such as `request_intercepts(...)` and
  `conditional_execution(...)` are for advanced adapters that need one middleware
  family before calling a provider manually.
- Streaming LLM wrappers collect chunks and finalize a response at stream end;
  dropping the stream early can prevent finalizers and subscribers from seeing a
  complete output.

## Checklist

- [ ] Scope boundary chosen before the first tool or LLM call
- [ ] Existing tool function wrapped without losing its original arguments/result
- [ ] Existing LLM/provider call wrapped at the right abstraction layer
- [ ] Optional metadata, attributes, or model name attached where useful
- [ ] Context propagation handled if the call hops threads or async tasks

## Use Another Skill When

- You need traces, ATIF, or export setup -> `nemo-relay-setup-observability`
- You are debugging missing events or load failures ->
  `nemo-relay-debug-runtime-integration`
- You need per-request isolation or worker-pool advice ->
  `nemo-relay-use-context-isolation`
- You need reusable config-activated runtime behavior ->
  `nemo-relay-build-plugin`

## Related Skills

- `nemo-relay-start`
- `nemo-relay-typed-wrappers-codecs`
- `nemo-relay-setup-observability`
- `nemo-relay-build-plugin`
