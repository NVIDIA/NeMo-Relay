<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code Examples

Use these examples for implementation surfaces:

- [Basic Guide: Adding Scopes](adding-scopes.md)
- [Basic Guide: Wrap Tool Calls](wrap-tool-calls.md)
- [Basic Guide: Wrap LLM Calls](wrap-llm-calls.md)
- [Non-Serializable Data](non-serializable-data.md)
- [Advanced Guide: Using Codecs](using-codecs.md)
- [Advanced Guide: Provider Codecs](provider-codecs.md)
- [Advanced Guide: Provider Response Codecs](provider-response-codecs.md)

## Preferred Integration Order

Choose the highest option your framework boundary supports:

1. Managed execution wrappers.
2. Explicit start and end lifecycle calls.
3. Standalone conditional-execution helpers.
4. Standalone request-intercept helpers.
5. Mark events for milestones that are not full tool or LLM calls.

Managed execution wrappers give NeMo Flow the most complete lifecycle: middleware order, parent-child scope relationships, request and response event payloads, execution intercepts, and subscriber visibility. Fallback helpers are useful when the framework owns the real callback internally.

## Managed Execution Wrappers

Use managed wrappers when the framework exposes a stable callable boundary.

| Surface | Python | Node.js | Rust |
|---|---|---|---|
| Tool execute | `nemo_flow.tools.execute(...)` | `toolCallExecute(...)` | `nemo_flow::api::tool::tool_call_execute` |
| LLM execute | `nemo_flow.llm.execute(...)` | `llmCallExecute(...)` | `nemo_flow::api::llm::llm_call_execute` |
| LLM stream execute | `nemo_flow.typed.llm_stream_execute(...)` | `typedLlmStreamExecute(...)` | `nemo_flow::api::llm::llm_stream_call_execute` |

## Fallback: Explicit API Calls

Use explicit API calls when the framework has start and finish hooks but owns the invocation internally.

What you lose from managed execution wrappers:

- Execution intercepts cannot wrap the real callback.
- The framework must preserve the start handle and call the matching end helper.
- Request intercepts and conditional execution must be invoked explicitly if you still need them.
- Error paths need deliberate end-event or mark-event handling.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow
from nemo_flow import LLMRequest


def framework_tool_started(name: str, args: dict):
    return nemo_flow.tools.call(name, args)


def framework_tool_finished(handle, result: dict) -> None:
    nemo_flow.tools.call_end(handle, result)


def framework_llm_started(provider: str, payload: dict):
    request = LLMRequest({}, payload)
    return nemo_flow.llm.call(provider, request, model_name=payload.get("model"))


def framework_llm_finished(handle, response: dict) -> None:
    nemo_flow.llm.call_end(handle, response)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { LlmRequest, llmCall, llmCallEnd, toolCall, toolCallEnd } from 'nemo-flow-node';

export function frameworkToolStarted(name: string, args: unknown) {
  return toolCall(name, args, null, null, null, null, null);
}

export function frameworkToolFinished(handle: unknown, result: unknown): void {
  toolCallEnd(handle, result, null, null);
}

export function frameworkLlmStarted(provider: string, payload: { model?: string }) {
  const request = new LlmRequest({}, payload);
  return llmCall(provider, request, null, null, null, null, payload.model ?? null);
}

export function frameworkLlmFinished(handle: unknown, response: unknown): void {
  llmCallEnd(handle, response, null, null);
}
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::llm::{llm_call, llm_call_end, LlmCallEndParams, LlmCallParams, LlmRequest};
use nemo_flow::api::tool::{tool_call, tool_call_end, ToolCallEndParams, ToolCallParams};
use serde_json::{json, Value as Json};

let tool_handle = tool_call(
    ToolCallParams::builder()
        .name("search")
        .args(json!({"query": "weather"}))
        .build(),
)?;
tool_call_end(
    ToolCallEndParams::builder()
        .handle(&tool_handle)
        .result(json!({"hits": 2}))
        .build(),
)?;

let request = LlmRequest {
    headers: Default::default(),
    content: json!({"model": "demo-model", "messages": []}),
};
let llm_handle = llm_call(
    LlmCallParams::builder()
        .name("demo-provider")
        .request(&request)
        .model_name("demo-model")
        .build(),
)?;
llm_call_end(
    LlmCallEndParams::builder()
        .handle(&llm_handle)
        .response(Json::String("ok".into()))
        .build(),
)?;
```
:::

::::

## Conditional Execution

Use conditional-execution helpers when the framework needs an allow-or-block decision before it continues down its own invocation path.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow
from nemo_flow import LLMRequest

nemo_flow.tools.conditional_execution("search", {"query": "weather"})
nemo_flow.llm.conditional_execution(LLMRequest({}, {"messages": []}))
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { LlmRequest, llmConditionalExecution, toolConditionalExecution } from 'nemo-flow-node';

await toolConditionalExecution('search', { query: 'weather' });
await llmConditionalExecution(new LlmRequest({}, { messages: [] }));
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::llm::{llm_conditional_execution, LlmRequest};
use nemo_flow::api::tool::tool_conditional_execution;
use serde_json::json;

tool_conditional_execution("search", &json!({"query": "weather"}))?;
let request = LlmRequest { headers: Default::default(), content: json!({"messages": []}) };
llm_conditional_execution(&request)?;
```
:::

::::

## Request Intercepts

Use request-intercept helpers when the framework wants NeMo Flow to rewrite arguments or provider requests before the framework invokes its own downstream code.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow
from nemo_flow import LLMRequest

rewritten_args = nemo_flow.tools.request_intercepts("search", {"query": "weather"})
rewritten_request = nemo_flow.llm.request_intercepts(
    "demo-provider",
    LLMRequest({}, {"messages": []}),
)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { LlmRequest, llmRequestIntercepts, toolRequestIntercepts } from 'nemo-flow-node';

const rewrittenArgs = await toolRequestIntercepts('search', { query: 'weather' });
const rewrittenRequest = await llmRequestIntercepts('demo-provider', new LlmRequest({}, { messages: [] }));
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::llm::{llm_request_intercepts, LlmRequest};
use nemo_flow::api::tool::tool_request_intercepts;
use serde_json::json;

let rewritten_args = tool_request_intercepts("search", json!({"query": "weather"}))?;
let request = LlmRequest { headers: Default::default(), content: json!({"messages": []}) };
let rewritten_request = llm_request_intercepts("demo-provider", request)?;
```
:::

::::

## Mark Events

Use mark events when the framework exposes important milestones but not a full lifecycle boundary.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow

nemo_flow.scope.event("scheduler.retry", data={"attempt": 2})
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { event } from 'nemo-flow-node';

event('scheduler.retry', null, { attempt: 2 }, null);
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::scope::{event, EmitMarkEventParams};
use serde_json::json;

event(
    EmitMarkEventParams::builder()
        .name("scheduler.retry")
        .data(json!({"attempt": 2}))
        .build(),
)?;
```
:::

::::

## Sample Third-Party Integrations

NeMo Flow keeps sample third-party integrations as patch sets under `patches/` and pinned upstream checkouts under `third_party/`.

| Integration | Upstream Checkout |
|---|---|
| Hermes Agent | `third_party/hermes-agent` |
| LangChain | `third_party/langchain` |
| LangChain NVIDIA | `third_party/langchain-nvidia` |
| LangGraph | `third_party/langgraph` |
| OpenClaw | `third_party/openclaw` |
| opencode | `third_party/opencode` |

## Quickstart: Apply Maintained Patches

From the repository root, use the wrapper scripts when you want the maintained
NeMo Flow patches applied to the pinned third-party checkouts:


| Script | Purpose |
|---|---|
| `./scripts/bootstrap-third-party.sh` | Clone pinned third-party upstream checkouts from `third_party/sources.lock`. |
| `./scripts/apply-patches.sh` | Apply NeMo Flow integration patches to third-party checkouts. |
| `./scripts/apply-patches.sh --check` | Ensure the patches apply cleanly to all third-party checkouts. |
| `./scripts/generate-patches.sh` | Regenerate patch files from local third-party checkout changes. |
| `./scripts/build-docs.sh` | Build the documentation site after integration docs change. |

The dry run checks that patches apply cleanly before modifying the local
checkouts. Use manual `git clone`, `git checkout`, and `git apply` commands only
when you need to work on one integration outside the standard wrapper flow.
