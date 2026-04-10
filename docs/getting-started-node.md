<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started: Node.js

This guide takes you from a local build to a minimal scope, tool call, and LLM
call using the Node.js binding.

All examples in this guide use:

- an active NeMo Flow scope
- the managed execution APIs (`toolCallExecute(...)` and `llmCallExecute(...)`)

This guide intentionally does not use the low-level manual lifecycle APIs such
as `toolCall(...)` / `toolCallEnd(...)` or `llmCall(...)` / `llmCallEnd(...)`.

## Prerequisites

- Node.js LTS
- Rust toolchain

## Install and Build

From the repository root:

```bash
cd crates/node
npm install
npm run build
```

## Process Ownership Note

The Node native addon claims NeMo Flow runtime ownership for the current
process when the module loads. Do not load a different native NeMo Flow
binding into the same process. Reusing the Node binding within the same major
version is allowed.

## Minimal Scope and Tool Execution

Run this from `crates/node` so `./index.js` resolves to the generated binding:

```javascript
const {
  ScopeType,
  pushScope,
  popScope,
  toolCallExecute,
} = require("./index.js");

async function main() {
  const handle = pushScope("quickstart-agent", ScopeType.Agent, null, null);

  const result = await toolCallExecute(
    "search",
    { query: "hello" },
    async (args) => ({ results: [`echo:${args.query}`] }),
    null,
    null,
    null,
    null,
  );

  console.log(result);
  popScope(handle);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

## Minimal LLM Execution

```javascript
const {
  ScopeType,
  pushScope,
  popScope,
  llmCallExecute,
} = require("./index.js");

async function main() {
  const handle = pushScope("quickstart-agent", ScopeType.Agent, null, null);

  const response = await llmCallExecute(
    "gpt-4",
    {
      headers: {},
      content: {
        model: "gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      },
    },
    async (request) => ({
      response: "ok",
      messages: request.content.messages,
    }),
    null,
    null,
    null,
    null,
    "gpt-4",
  );

  console.log(response);
  popScope(handle);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

## Add Logging

Register a subscriber before pushing the scope:

```javascript
const { registerSubscriber } = require("./index.js");

registerSubscriber("logger", (event) => {
  console.log(event.name);
});
```

## Common Errors

- Native addon load error
  Re-run `npm install` and `npm run build` in `crates/node`.
- Runtime call outside a scope
  Push a scope before calling `toolCallExecute(...)` or `llmCallExecute(...)`.

## Next Docs

- [Language Bindings](language-bindings.md#nodejs)
- [Observability with OpenTelemetry](observability-with-opentelemetry.md)
- [Observability with OpenInference](observability-with-openinference.md)
- [Middleware Pipeline](middleware-pipeline.md)
- [Typed Wrappers](typed-wrappers.md)
- [Testing](testing.md)
