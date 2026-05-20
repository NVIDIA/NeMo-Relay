<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Instrument a Tool Call

Use this guide when you have an application tool callback and want NeMo Flow to emit lifecycle events, apply middleware, and preserve the active agent scope around the call.

## What You Build

You will wrap one existing tool callback with the managed tool execution API. The result is a tool call that:

- Receives JSON-compatible arguments.
- Runs request intercepts, guardrails, execution intercepts, and response guardrails.
- Emits tool start and tool end events.
- Keeps the tool span attached to the current agent or request scope.
- Returns the original tool result to the application.

## Before You Start

Complete one binding Quick Start guide first:

- [Python Quick Start](../getting-started/python/index.md)
- [Node.js Quick Start](../getting-started/nodejs.md)
- [Rust Quick Start](../getting-started/rust.md)

Create a scope for the active request or agent run before adding tool instrumentation. If you have not done that yet, start with [Adding Scopes and Marks](adding-scopes-and-marks.md).

The tool arguments and result must be JSON-compatible. If your framework passes clients, sockets, streams, callbacks, or other opaque objects, use [Handle Non-Serializable Data](../integrate-frameworks/non-serializable-data.md) before you instrument the callback.

## Integration Pattern

Follow this sequence to keep framework work attached to the expected runtime context.

1. Identify the stable tool boundary in your application.
2. Create or inherit a scope for the current agent run, request, or workflow.
3. Register a temporary subscriber while validating the integration.
4. Replace the direct callback invocation with the managed tool execute helper.
5. Pass the active scope handle when the binding supports it.
6. Check that the application result is unchanged and lifecycle events are emitted.

## Minimal Example

The examples below wrap a `search` callback and print emitted events.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import asyncio

import nemo_flow


def log_event(event) -> None:
    print(f"{event.kind} {event.name}")


async def search(args):
    return {
        "query": args["query"],
        "hits": [{"title": "NeMo Flow"}],
    }


async def main() -> None:
    nemo_flow.subscribers.register("instrumentation-check", log_event)

    try:
        with nemo_flow.scope.scope("agent-run", nemo_flow.ScopeType.Agent) as handle:
            result = await nemo_flow.tools.execute(
                "search",
                {"query": "runtime instrumentation"},
                search,
                handle=handle,
            )
            print(result)
    finally:
        nemo_flow.subscribers.deregister("instrumentation-check")


asyncio.run(main())
```
:::

:::{tab-item} Node.js
:sync: node

```js
const {
  ScopeType,
  deregisterSubscriber,
  registerSubscriber,
  toolCallExecute,
  withScope,
} = require("nemo-flow-node");

async function main() {
  registerSubscriber("instrumentation-check", (event) => {
    console.log(`${event.kind} ${event.name}`);
  });

  try {
    await withScope("agent-run", ScopeType.Agent, async (handle) => {
      const result = await toolCallExecute(
        "search",
        { query: "runtime instrumentation" },
        async (args) => ({
          query: args.query,
          hits: [{ title: "NeMo Flow" }],
        }),
        handle,
        null,
        null,
        null,
      );

      console.log(result);
    });
  } finally {
    deregisterSubscriber("instrumentation-check");
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::scope::{
    self, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeType,
};
use nemo_flow::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_flow::api::tool::{tool_call_execute, ToolCallExecuteParams};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    register_subscriber(
        "instrumentation-check",
        Arc::new(|event| {
            println!("{} {}", event.kind(), event.name());
        }),
    )?;

    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("agent-run")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::empty())
            .data(json!({"example": "instrument-tool"}))
            .build(),
    )?;

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("search")
            .args(json!({"query": "runtime instrumentation"}))
            .func(Arc::new(|args| {
                Box::pin(async move {
                    Ok(json!({
                        "query": args["query"],
                        "hits": [{"title": "NeMo Flow"}]
                    }))
                })
            }))
            .parent(handle.clone())
            .build(),
    )
    .await?;

    println!("{result}");

    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())?;
    let _ = deregister_subscriber("instrumentation-check")?;
    Ok(())
}
```
:::

::::

## Validate the Integration

Check both behavior and instrumentation:

- The tool result matches what the application returned before the wrapper was added.
- The subscriber prints an agent or request scope event.
- The subscriber prints tool start and tool end events for `search`.
- Tool start input contains the request arguments after request intercepts and
  sanitize-request guardrails.
- Tool end output contains the tool result after response guardrails.

If only the business result appears, the callback ran but instrumentation did not run. Confirm that the call goes through `tools.execute`, `toolCallExecute`, or `tool_call_execute`.

## Production Checklist

Use this checklist before running the pattern in production traffic.

- Keep tool names stable. Subscribers and downstream exporters use names for filtering and dashboards.
- Keep tool arguments and results JSON-compatible.
- Register temporary debugging subscribers only in development or test environments.
- Pass the parent scope handle when the tool is part of a larger agent, request, or workflow.
- Use middleware names that describe ownership, such as `search.redact_args` or `retrieval.timeout`.

## Common Issues

Check these symptoms first when the workflow does not behave as expected.

- **No events appear**: The application is still calling the tool directly.
- **The tool appears outside the agent scope**: Pass the current scope handle into the managed execute helper.
- **The call fails before execution**: A conditional-execution guardrail rejected the request.
- **Subscribers see different data than the tool receives**: Sanitize guardrails change event payloads, while request intercepts change the real arguments.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Add model-provider instrumentation with [Instrument an LLM Call](instrument-llm-call.md).
- Add policy or transformation with [Add Middleware](advanced-guide.md).
- Export events with [Observability](../plugins/observability/about.md).
- Use [Code Examples](code-examples.md) for manual lifecycle, streaming, scope, and partial middleware API examples.
