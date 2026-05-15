<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wrap Tool Calls

Use this guide when a framework, SDK, or orchestration layer owns tool invocation and you need NeMo Flow to observe and control those calls without changing the framework's public behavior.

## What You Build

You will place a managed NeMo Flow tool execution wrapper at the framework's stable tool boundary. The wrapper emits tool lifecycle events, runs tool middleware, keeps the tool attached to the active scope, and returns the original tool result to the framework.

## Before You Start

You need:

- A framework request or run scope. If the framework does not create one yet, start with [Adding Scopes](adding-scopes.md).
- A stable tool invocation boundary, such as a callback dispatcher, tool registry, or tool adapter.
- A JSON-compatible projection of tool arguments and results.
- A subscriber or exporter that can verify emitted tool events.

## Integration Pattern

Follow this sequence to keep framework work attached to the expected runtime context.

1. Enter or inherit the active framework scope.
2. Capture the current scope handle at the tool boundary.
3. Route the real tool callback through the managed tool execute helper.
4. Keep framework-owned clients, callbacks, streams, and handles outside the emitted JSON payload.
5. Return the tool result exactly as the framework expects.

Managed wrappers are the first choice because NeMo Flow owns the full call boundary. That gives subscribers complete start and end events, lets execution intercepts wrap the real callback, and keeps guardrails and request intercepts in the normal middleware order.

## Concrete Tool Example

The examples below wrap one framework tool callback and attach it to the active parent scope.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from typing import TypedDict

import nemo_flow


class SearchArgs(TypedDict):
    query: str


class SearchResult(TypedDict):
    hits: int
    echo: SearchArgs


async def framework_tool(tool_name: str, raw_args: SearchArgs) -> SearchResult:
    parent = nemo_flow.scope.get_handle()

    async def invoke(args: SearchArgs) -> SearchResult:
        return {"hits": 2, "echo": args}

    return await nemo_flow.tools.execute(
        tool_name,
        raw_args,
        invoke,
        handle=parent,
    )
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { getHandle, toolCallExecute, type ScopeHandle } from 'nemo-flow-node';

type SearchArgs = { query: string };
type SearchResult = { hits: number; echo: SearchArgs };

export async function frameworkTool(toolName: string, rawArgs: SearchArgs): Promise<SearchResult> {
  const parent: ScopeHandle = getHandle();

  return await toolCallExecute(
    toolName,
    rawArgs,
    async (args: SearchArgs): Promise<SearchResult> => ({ hits: 2, echo: args }),
    parent,
    null,
    null,
    null,
  ) as Promise<SearchResult>;
}
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::scope::get_handle;
use nemo_flow::api::tool::{tool_call_execute, ToolCallExecuteParams};
use serde_json::json;
use std::sync::Arc;

async fn run_framework_tool() -> anyhow::Result<serde_json::Value> {
    let parent = get_handle()?;
    let args = json!({"query": "weather"});

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("search")
            .args(args)
            .func(Arc::new(|input| Box::pin(async move {
                Ok(json!({"hits": 2, "echo": input}))
            })))
            .parent(parent)
            .build(),
    )
    .await?;

    Ok(result)
}
```
:::

::::

## When to Use Fallback APIs

Use explicit lifecycle APIs only when the framework owns the real tool invocation internally and exposes only start and finish hooks. In that case, the integration must preserve the returned handle and call the matching end helper on every success and failure path.

Use standalone request-intercept or conditional-execution helpers when the framework needs only partial middleware behavior before it continues down its own invocation path. See [Code Examples](code-examples.md#fallback-explicit-api-calls) for those fallback surfaces.

## Validate the Tool Wrapper

Run one framework tool path and check:

- The application receives the same tool result as before.
- Subscribers see one tool start event and one matching tool end event.
- Tool events share the same root scope UUID as the framework request.
- Global and scope-local tool middleware run exactly once.
- Framework-owned objects do not appear in emitted JSON payloads.

## Common Issues

Check these symptoms first when the workflow does not behave as expected.

- **Tool events appear without parentage**: Pass the active scope handle or ensure the framework tool runs inside a NeMo Flow scope.
- **Middleware does not run**: The framework still calls the real tool callback directly.
- **Payload serialization fails**: Project framework objects into JSON-compatible tool arguments and results before NeMo Flow sees them.
- **A fallback emits incomplete spans**: Manual start and end lifecycle calls must use the same handle.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Add model-provider integration with [Wrap LLM Calls](wrap-llm-calls.md).
- Add request ownership with [Adding Scopes](adding-scopes.md).
- Use [Handle Non-Serializable Data](non-serializable-data.md) when framework objects need ID-based lookup.
