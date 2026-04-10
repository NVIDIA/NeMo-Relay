<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-nemo-flow-wasm

wasm-bindgen WebAssembly bindings for the NeMo Flow core runtime.

The Cargo package name is `nemo-flow-wasm`. The compiled WASM library target
and npm package are NVIDIA-branded (`nemo_flow_wasm` and
`@nvidia/nemo-flow-wasm`).

## Overview

This crate compiles the NeMo Flow core API to WebAssembly via `wasm-bindgen`, making it usable in both browser and Node.js environments. It provides the same API surface as other language bindings, adapted for the single-threaded WASM execution model using `send_wrapper` for thread-safety bridging.

## What It Provides

- **Full API surface** -- Scope management, tool/LLM calls, guardrails, intercepts, subscribers, scope-local middleware, and ATIF export for JavaScript/TypeScript consumers.
- **Browser and Node.js support** -- Works in any environment that supports WebAssembly with `wasm-bindgen` glue code.
- **Async via JS Promises** -- Rust `Future`s are bridged to JavaScript Promises through `wasm-bindgen-futures`.
- **Type conversion** -- Automatic conversion between JS values and Rust types via `serde-wasm-bindgen`.
- **Stream support** -- LLM stream wrapping with JavaScript collector and finalizer callbacks.
- **TypeScript definitions** -- `wasm-pack build` generates `.d.ts` type definitions alongside the `.wasm` and `.js` output.

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | WASM module entry point |
| `src/api.rs` | `#[wasm_bindgen]`-exported API functions |
| `src/types.rs` | JS type wrappers for core types |
| `src/callable.rs` | Wrapping JS functions for Rust callbacks |
| `src/convert.rs` | JS-Rust type conversion via `serde-wasm-bindgen` |
| `src/stream.rs` | LLM stream wrapper for WASM |

## Build

```bash
wasm-pack build crates/wasm --scope nvidia
```

This produces a `pkg/` directory containing `.wasm`, `.js`, and `.d.ts` files ready for bundling or direct import.

To run unit tests:

```bash
cargo test -p nemo-flow-wasm
wasm-pack test --node crates/wasm
```

## Adaptive Config

The WASM binding uses plain JavaScript objects for adaptive config and exposes
the adaptive helpers through `adaptive.js`, with configuration managed through
the generic plugin host.

```javascript
import init from "./pkg/nemo_flow_wasm.js";
import * as adaptive from "./adaptive.js";
import * as plugin from "./plugin.js";

await init();

const config = {
  version: 1,
  state: { backend: { kind: "in_memory", config: {} } },
  telemetry: { learners: ["latency_sensitivity"] },
};

const validation = plugin.validate({
  version: 1,
  components: [adaptive.ComponentSpec(config)],
});
```

## Hosted Plugins

WASM hosted plugins register JavaScript handlers first, then enable themselves
as top-level plugin components in the generic plugin config.

```javascript
import init from "./pkg/nemo_flow_wasm.js";
import * as plugin from "./plugin.js";

await init();

plugin.register("example.header_plugin", {
  register(pluginConfig, context) {
  context.registerToolRequestIntercept(
    "tool",
    25,
    false,
      (_name, args) => ({ ...args, wasmPlugin: "enabled" }),
  );
  },
});

await plugin.initialize({
  version: 1,
  components: [
    plugin.ComponentSpec("example.header_plugin", {}),
  ],
});
```

`context` exposes:

- `registerSubscriber(...)`
- `registerLlmRequestIntercept(...)`
- `registerLlmExecutionIntercept(...)`
- `registerLlmStreamExecutionIntercept(...)`
- `registerToolRequestIntercept(...)`
- `registerToolExecutionIntercept(...)`

Current limitation:

- `registerLlmStreamExecutionIntercept(...)` in the WASM binding produces a
  single-item stream result directly and does not delegate to downstream stream
  handlers. Hosted plugins therefore cannot chain stream execution intercepts
  the same way they can in the Rust, Python, Go, and Node.js bindings.

## Documentation

See [docs/language-bindings.md](../../docs/language-bindings.md) for WASM binding details.
