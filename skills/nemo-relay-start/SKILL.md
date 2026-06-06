---
name: nemo-relay-start
description: Help application developers pick a NeMo Relay binding and get to a first working scope, tool call, and LLM call
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

<!-- Non-semantic CI refresh marker: 2026-06-06. -->

# Get Started With NeMo Relay

Use this skill for first-time users who want the shortest path to a working
example. Rust, Python, and Node.js are the primary quick-start and hosted-docs
paths. Go, WebAssembly, and the raw FFI surface are source-first advanced paths.

## Default Path

- Pick the user's host language first: Rust, Python, or Node.js. If they are
  using Go, WebAssembly, or raw FFI, verify names against tracked source and tests.
- Prefer the managed execution APIs over manual lifecycle APIs.
- Start with one scope, one tool call, and one LLM call.
- Add observability only after the basic flow works.

## Guidance

- **Rust**: use `nemo_relay::api::scope::{push_scope, pop_scope, event}` with
  builder params, then `nemo_relay::api::tool::tool_call_execute(...)` and
  `nemo_relay::api::llm::llm_call_execute(...)`
- **Python**: `uv sync`, then use `nemo_relay.scope.scope(...)`,
  `nemo_relay.tools.execute(...)`, and `nemo_relay.llm.execute(...)`
- **Node.js**: build the addon, then use `withScope(...)`,
  `toolCallExecute(...)`, and `llmCallExecute(...)`
- **Go**: use source-first wrappers such as `scope.Push(...)`,
  `tools.Execute(...)`, `llm.Execute(...)`, or top-level `PushScope(...)`,
  `ToolCallExecute(...)`, and `LlmCallExecute(...)`
- **WebAssembly**: use the generated JS-facing API such as `withScope(...)`,
  `toolCallExecute(...)`, `llmCallExecute(...)`, and `createScopeStack()`
- **FFI**: recommend only for binding or embedding work; verify C names such as
  `nemo_relay_push_scope`, `nemo_relay_tool_call_execute`, and
  `nemo_relay_llm_call_execute` in the current header

## Common Pitfalls

- Calling execute APIs without an active scope
- Skipping the build step for Rust, Python, or Node.js
- Assuming source-first Go/WebAssembly/FFI bindings have the same hosted-doc coverage
  as Rust, Python, and Node.js
- Mixing manual lifecycle APIs into a first example

## Embedded Quick-Start Notes

- Install from packages when building a consumer app: Rust uses `cargo add
  nemo-relay`, Python uses `uv add nemo-relay`, and Node.js uses `npm install
  nemo-relay-node`.
- Use repository setup commands when working from a checkout: Rust builds the
  workspace, Python rebuilds the virtual environment and native extension with
  `uv sync`, and Node.js installs and builds the native addon before tests or
  examples run.
- A first example should register a short-lived subscriber, open an agent scope,
  emit one mark event, run one managed tool call, run one managed LLM call, then
  deregister the subscriber. In Python use `nemo_relay.subscribers`; in Node.js
  use root exports such as `registerSubscriber`; in Rust use
  `nemo_relay::api::subscriber`.
- Success means the app emits scope start/end events plus tool and LLM lifecycle
  events, and the application result remains the provider or tool result.
- Scope handles are explicit in Rust and optional in higher-level Python and
  Node.js helpers when the active scope is already correct. Pass the handle when
  the surrounding framework makes parentage ambiguous.

## Related Skills

- `nemo-relay-instrument-calls`
- `nemo-relay-setup-observability`
- `nemo-relay-debug-runtime-integration`
