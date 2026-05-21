<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Handle Non-Serializable Data

Use this guide when a framework exposes SDK clients, streams, callbacks, file handles, or custom classes at the same boundary where you need NeMo Relay instrumentation.

## What You Build

You will keep non-serializable framework objects in framework-owned storage and pass only stable JSON projections through NeMo Relay middleware and event payloads.

## Before You Start

You need:

- A stable request ID, tool call ID, or framework request object that can key external storage.
- A JSON-compatible projection of the data that guardrails, intercepts, and subscribers need.
- A cleanup point for framework-owned object maps.

## The Constraint

NeMo Relay middleware surfaces operate on JSON-compatible data. Frameworks do not always expose tool or model requests in that form.

## Recommended Strategies

These strategies keep provider and framework data JSON-compatible before it reaches NeMo
Flow.

- Convert provider payloads into a stable request shape before NeMo Relay sees them.
- Preserve opaque framework objects outside the middleware path and pass only the serializable projection into the runtime.
- Store external object references in framework-owned maps keyed by request IDs, not inside NeMo Relay event payloads.
- Use typed wrappers for your application boundary, then serialize at the last responsible moment.

## Concrete Projection Pattern

Keep framework objects in your own map, but send only the JSON projection through NeMo Relay.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from typing import TypedDict

import nemo_relay


class SearchArgs(TypedDict):
    client_id: str
    query: str


class SearchResult(TypedDict):
    hits: int


framework_clients: dict[str, object] = {}
framework_clients["client-1"] = object()


async def invoke(args: SearchArgs) -> SearchResult:
    client = framework_clients[args["client_id"]]
    _ = client  # framework-owned object stays outside NeMo Relay payloads
    return {"hits": 2}


result = await nemo_relay.tools.execute(
    "search",
    SearchArgs(client_id="client-1", query="weather"),
    invoke,
)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { toolCallExecute } from 'nemo-relay-node';

type SearchArgs = { clientId: string; query: string };
type SearchResult = { hits: number };

const frameworkClients = new Map<string, object>();
frameworkClients.set('client-1', {});

const result = await toolCallExecute(
  'search',
  { clientId: 'client-1', query: 'weather' },
  async (args: SearchArgs): Promise<SearchResult> => {
    const client = frameworkClients.get(args.clientId);
    if (!client) {
      throw new Error(`missing client ${args.clientId}`);
    }
    return { hits: 2 };
  },
) as SearchResult;
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_relay::api::tool::{ToolCallExecuteParams, tool_call_execute};
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct FrameworkClient;

let clients: Arc<Mutex<HashMap<String, FrameworkClient>>> = Arc::new(Mutex::new(HashMap::new()));
clients
    .lock()
    .unwrap()
    .insert("client-1".into(), FrameworkClient);

let args = json!({
    "client_id": "client-1",
    "query": "weather"
});

let shared_clients = Arc::clone(&clients);
let result = tool_call_execute(
    ToolCallExecuteParams::builder()
        .name("search")
        .args(args)
        .func(Arc::new(move |input| {
            let local_clients = Arc::clone(&shared_clients);
            Box::pin(async move {
                let client_id = input["client_id"].as_str().unwrap();
                let _client = local_clients.lock().unwrap().get(client_id).cloned().unwrap();
                Ok(json!({"hits": 2}))
            })
        }))
        .build(),
).await?;
```
:::
::::

## Codec Pattern For Provider Payloads

Use an LLM codec when the framework payload is structurally different from the annotated request model you want intercepts to reason about.

- Decode the provider payload into a normalized annotated request.
- Let request intercepts edit the normalized shape.
- Encode the annotated request back to the provider payload before the real call.
- Keep sockets, streams, callbacks, and SDK objects outside the codec result.

## What Not to Do

Avoid these patterns because they make runtime payloads difficult to serialize or
observe.

- Do not place client instances, callbacks, streams, sockets, or open file handles inside JSON event payloads.
- Do not rely on Python or JavaScript object identity surviving the middleware boundary.
- Do not leak framework-internal classes into a plugin config document.

## Practical Workarounds

Use these workarounds when framework data cannot be passed directly through NeMo Relay.

- Replace large objects with IDs and look them up later.
- Emit summarized metadata instead of full request bodies.
- Use request codecs to normalize provider payloads.
- Use manual lifecycle APIs when the framework does not expose a clean execution wrapper.

## Common Failure Cases

These failure cases are common signs that non-serializable data crossed the runtime
boundary.

- A request intercept tries to return a framework SDK object instead of JSON.
- A plugin config stores a callable, client instance, or file handle.
- A subscriber assumes Python or JavaScript object identity survives event serialization.
- A worker thread receives a scope UUID but not the corresponding framework-owned object lookup table.

## Validation Checklist

Use this checklist to confirm the implementation preserves the expected runtime
contract.

- Middleware receives only JSON-compatible values.
- The framework callback can still resolve the original SDK client or stream by ID.
- Subscribers and exporters receive enough metadata to debug the call.
- Cleanup removes object-map entries after the request finishes.
- Redaction happens before payloads reach production subscribers.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Wrap the framework boundary with [Wrap Tool Calls](wrap-tool-calls.md) or [Wrap LLM Calls](wrap-llm-calls.md).
- Use [Provider Codecs](provider-codecs.md) when provider payloads need normalized request or response annotations.
- Use the typed value codec examples in [Using Codecs](using-codecs.md#typed-value-codecs) for structured conversion helpers.
- Use [Add Middleware](../instrument-applications/advanced-guide.md) before adding request transforms.
