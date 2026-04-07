<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started: WebAssembly

This guide takes you from a local `wasm-pack` build to a minimal scope, tool
call, and LLM call using the WebAssembly binding.

All examples in this guide use:

- an active Nexus scope
- the managed execution APIs (`toolCallExecute(...)` and `llmCallExecute(...)`)

This guide intentionally does not use the low-level manual lifecycle APIs such
as `toolCall(...)` / `toolCallEnd(...)` or `llmCall(...)` / `llmCallEnd(...)`.

## Prerequisites

- Rust toolchain
- `wasm-pack`
- Node.js if you want to run the generated package locally

## Build

From the repository root:

```bash
wasm-pack build crates/wasm --scope nvidia --target nodejs
```

This produces a `pkg/` directory under `crates/wasm/`.

## Minimal Scope and Tool Execution

Create a JavaScript file next to the generated package and import from
`./pkg/nvidia_nat_nexus_wasm.js`:

```javascript
const wasm = require("./pkg/nvidia_nat_nexus_wasm.js");
const init = wasm.default;
const { pushScope, popScope, toolCallExecute, SCOPE_TYPE_AGENT } = wasm;

async function main() {
  await init();

  const handle = pushScope("quickstart-agent", SCOPE_TYPE_AGENT, null, null, null, null);

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
const wasm = require("./pkg/nvidia_nat_nexus_wasm.js");
const init = wasm.default;
const { pushScope, popScope, llmCallExecute, SCOPE_TYPE_AGENT } = wasm;

async function main() {
  await init();

  const handle = pushScope("quickstart-agent", SCOPE_TYPE_AGENT, null, null, null, null);

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

## Common Errors

- Missing `init()`
  For the `web` target, and harmlessly for `nodejs`, call the default-exported
  `init()` before using the API.
- Import path mismatch
  Make sure you are importing from the generated `pkg/` directory created by
  `wasm-pack`.

## Next Docs

- [Language Bindings](language-bindings.md#webassembly)
- [Observability with OpenTelemetry](observability-with-opentelemetry.md)
- [Observability with OpenInference](observability-with-openinference.md)
- [Middleware Pipeline](middleware-pipeline.md)
- [Context Isolation](context-isolation.md)
- [Testing](testing.md)
