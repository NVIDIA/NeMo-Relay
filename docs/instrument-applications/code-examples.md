<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code Examples

Use these examples when you need the direct runtime surfaces behind the application instrumentation guides.

## Invocation API Selection

The following table shows which API to use based on your integration need:

| Need | Preferred API | Use When |
|---|---|---|
| Run a tool with full instrumentation | `tools.execute`, `toolCallExecute`, `tool_call_execute` | Application code owns the callback. |
| Run an LLM call with full instrumentation | `llm.execute`, `llmCallExecute`, `llm_call_execute` | Application code owns the provider call. |
| Run a streaming LLM call | `llm_stream_execute`, `typedLlmStreamExecute`, `llm_stream_call_execute` | You need chunk collection and one final aggregate end event. |
| Emit start/end manually | `call` and `call_end` helpers | A framework owns the real invocation boundary. |
| Emit a checkpoint | `scope.event`, `event` | You need milestone visibility inside an active scope. |
| Attach work to one request | Scope-local registration helpers | Middleware or subscribers should disappear when that scope closes. |

## Manual Tool Lifecycle

Use manual lifecycle calls only when the surrounding code owns the real tool invocation and only exposes reliable start and finish hooks.
If you are replaying events or bridging a framework clock, pass an explicit timestamp to the manual start, end, or mark helpers.
Python accepts timezone-aware `datetime` values, Node.js and WebAssembly accept Unix microseconds since epoch, Rust accepts `DateTime<Utc>`, and Go accepts `time.Time`.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow

handle = nemo_flow.tools.call("search", {"query": "weather"}, data={"attempt": 1})
try:
    result = {"hits": 2}
finally:
    nemo_flow.tools.call_end(handle, result)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { toolCall, toolCallEnd } from 'nemo-flow-node';

const handle = toolCall('search', { query: 'weather' }, null, null, { attempt: 1 }, null, null);
const result = { hits: 2 };
toolCallEnd(handle, result, null, null);
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::tool::{tool_call, tool_call_end, ToolCallEndParams, ToolCallParams};
use serde_json::json;

let handle = tool_call(
    ToolCallParams::builder()
        .name("search")
        .args(json!({"query": "weather"}))
        .data(json!({"attempt": 1}))
        .build(),
)?;

tool_call_end(
    ToolCallEndParams::builder()
        .handle(&handle)
        .result(json!({"hits": 2}))
        .build(),
)?;
```
:::

::::

## Managed LLM Execution

Use managed execution when NeMo Flow should run the full middleware pipeline around the provider call.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow
from nemo_flow import LLMRequest

request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]})


async def invoke(req: LLMRequest):
    return {"text": "hi", "request": req.content}


response = await nemo_flow.llm.execute(
    "demo-provider",
    request,
    invoke,
    model_name="demo-model",
)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { LlmRequest, llmCallExecute } from 'nemo-flow-node';

const request = new LlmRequest({}, { messages: [{ role: 'user', content: 'hello' }] });

const response = await llmCallExecute(
  'demo-provider',
  request,
  async (req: LlmRequest) => ({ text: 'hi', request: req.content }),
  null,
  null,
  null,
  null,
  'demo-model',
);
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::llm::{llm_call_execute, LlmCallExecuteParams, LlmRequest};
use serde_json::json;
use std::sync::Arc;

let request = LlmRequest {
    headers: Default::default(),
    content: json!({"messages": [{"role": "user", "content": "hello"}]}),
};

let response = llm_call_execute(
    LlmCallExecuteParams::builder()
        .name("demo-provider")
        .request(request)
        .func(Arc::new(|req| Box::pin(async move {
            Ok(json!({"text": "hi", "request": req.content}))
        })))
        .model_name("demo-model")
        .build(),
).await?;
```
:::

::::

## Streaming LLM Execution

Use the streaming helper when subscribers need chunk collection plus one final response payload.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from dataclasses import dataclass

from nemo_flow import LLMRequest
from nemo_flow.typed import DataclassCodec, llm_stream_execute


@dataclass
class Chunk:
    delta: str


@dataclass
class FinalResponse:
    text: str


request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]})
collected: list[Chunk] = []


async def stream_impl(_request: LLMRequest):
    yield Chunk(delta="hi")


stream = await llm_stream_execute(
    "demo-provider",
    request,
    stream_impl,
    collector=collected.append,
    finalizer=lambda: FinalResponse(text="".join(chunk.delta for chunk in collected)),
    chunk_json_codec=DataclassCodec(Chunk),
    response_json_codec=DataclassCodec(FinalResponse),
)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { LlmRequest } from 'nemo-flow-node';
import { typedLlmStreamExecute, type Codec } from 'nemo-flow-node/typed';

type Chunk = { delta: string };
type FinalResponse = { text: string };

const chunkCodec: Codec<Chunk> = {
  toJson: (value) => value,
  fromJson: (json) => json as Chunk,
};
const responseCodec: Codec<FinalResponse> = {
  toJson: (value) => value,
  fromJson: (json) => json as FinalResponse,
};

const request = new LlmRequest({}, { messages: [{ role: 'user', content: 'hello' }] });
const collected: Chunk[] = [];

const stream = await typedLlmStreamExecute(
  'demo-provider',
  request,
  async function* () {
    yield { delta: 'hi' };
  },
  (chunk) => collected.push(chunk),
  () => ({ text: collected.map((chunk) => chunk.delta).join('') }),
  chunkCodec,
  responseCodec,
);
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::llm::{
    llm_stream_call_execute, LlmAttributes, LlmRequest, LlmStreamCallExecuteParams,
};
use serde_json::json;

let request = LlmRequest {
    headers: Default::default(),
    content: json!({"messages": [{"role": "user", "content": "hello"}]}),
};

let stream = llm_stream_call_execute(
    LlmStreamCallExecuteParams::builder()
        .name("demo-provider")
        .request(request)
        .func(std::sync::Arc::new(|_req| Box::pin(async move {
            Ok(Box::pin(tokio_stream::iter(vec![Ok(json!({"delta": "hi"}))])))
        })))
        .collector(Box::new(|_chunk| Ok(())))
        .finalizer(Box::new(|| json!({"text": "hi"})))
        .attributes(LlmAttributes::STREAMING)
        .model_name("demo-model")
        .build(),
).await?;
```
:::

::::

## Partial Middleware Calls

These helpers are useful when framework code cannot use managed execution but still wants a request rewrite or block decision.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow
from nemo_flow import LLMRequest

tool_args = nemo_flow.tools.request_intercepts("search", {"query": "weather"})
nemo_flow.tools.conditional_execution("search", tool_args)

llm_request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]})
llm_request = nemo_flow.llm.request_intercepts("demo-provider", llm_request)
nemo_flow.llm.conditional_execution(llm_request)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import {
  LlmRequest,
  llmConditionalExecution,
  llmRequestIntercepts,
  toolConditionalExecution,
  toolRequestIntercepts,
} from 'nemo-flow-node';

const toolArgs = await toolRequestIntercepts('search', { query: 'weather' });
await toolConditionalExecution('search', toolArgs);

const request = new LlmRequest({}, { messages: [{ role: 'user', content: 'hello' }] });
const rewritten = await llmRequestIntercepts('demo-provider', request);
await llmConditionalExecution(rewritten);
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::llm::{llm_conditional_execution, llm_request_intercepts, LlmRequest};
use nemo_flow::api::tool::{tool_conditional_execution, tool_request_intercepts};
use serde_json::json;

let tool_args = tool_request_intercepts("search", json!({"query": "weather"}))?;
tool_conditional_execution("search", &tool_args)?;

let request = LlmRequest {
    headers: Default::default(),
    content: json!({"messages": [{"role": "user", "content": "hello"}]}),
};
let rewritten = llm_request_intercepts("demo-provider", request)?;
llm_conditional_execution(&rewritten)?;
```
:::

::::

## Scope and Context Helpers

Use normal scope helpers first. Reach for explicit stack helpers only when work crosses thread, task, worker, or request boundaries.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from concurrent.futures import ThreadPoolExecutor

import nemo_flow

with nemo_flow.scope.scope("request", nemo_flow.ScopeType.Agent):
    nemo_flow.scope.event("started", data={"ok": True})
    shared = nemo_flow.propagate_scope_to_thread()

    def worker() -> None:
        nemo_flow.set_thread_scope_stack(shared)
        nemo_flow.scope.event("worker-ran")

    with ThreadPoolExecutor() as pool:
        pool.submit(worker).result()
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { ScopeType, createScopeStack, event, setThreadScopeStack, withScope } from 'nemo-flow-node';

const workerStack = createScopeStack();
setThreadScopeStack(workerStack);

await withScope('request', ScopeType.Agent, async (handle) => {
  event('started', handle, { ok: true }, null);
});
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::runtime::{create_scope_stack, set_thread_scope_stack, TASK_SCOPE_STACK};
use nemo_flow::api::scope::{event, EmitMarkEventParams};
use serde_json::json;

let stack = create_scope_stack();
TASK_SCOPE_STACK
    .scope(stack.clone(), async {
        event(EmitMarkEventParams::builder().name("started").data(json!({"ok": true})).build())
    })
    .await?;

std::thread::spawn(move || {
    set_thread_scope_stack(stack);
    // NeMo Flow calls in this thread attach to the same explicit stack.
})
.join()
.unwrap();
```
:::

::::

## Middleware Registration Families

The runtime exposes the same registration families for tool and LLM calls:

- Sanitize-request guardrails change emitted start-event payloads only
- Sanitize-response guardrails change emitted end-event payloads only
- Conditional-execution guardrails return an allow-or-block decision
- Request intercepts change the real request before execution
- Execution intercepts wrap the callback and may post-process or short-circuit
- LLM stream execution intercepts wrap streaming provider callbacks

Every family also has a scope-local surface:

- Python: `nemo_flow.scope_local.register_*`
- Node.js: `scopeRegister*`
- Rust: middleware `scope_register_*` functions under
  `nemo_flow::api::registry`; subscriber scope registration under
  `nemo_flow::api::subscriber`

Use [Advanced Guide: Add Middleware](advanced-guide.md) for an end-to-end policy example and [API Reference](../reference/api/index.md) for symbol-level details.
