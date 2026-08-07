<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay Native Plugin SDK

`nemo-relay-plugin` is the Rust authoring SDK and stable ABI for trusted,
in-process NeMo Relay dynamic plugins. Use it to build a Rust `cdylib` that
Relay loads through the versioned native plugin interface.

Native plugins run in the Relay process and are not sandboxed. They should
depend on this crate rather than the host `nemo-relay` runtime crate, keeping
the dynamic-library boundary on the stable C-compatible ABI.

## Why Use It?

- **Author native plugins safely**: Implement `NativePlugin` with typed Rust
  callbacks instead of constructing ABI tables directly.
- **Register real runtime behavior**: Use `PluginContext` for subscribers,
  guardrails, and intercepts.
- **Keep a stable boundary**: Export one versioned native entry point through
  the `nemo_relay_plugin!` or `nemo_relay_plugin_v2!` macro.
- **Use host runtime helpers**: Emit events and manage scope state through the
  high-level `PluginRuntime` wrapper.

## What You Get

- **`NativePlugin`**: Plugin kind, configuration validation, and registration
  lifecycle contract.
- **`PluginContext`**: Component-scoped registration APIs for middleware and
  subscribers.
- **`PluginRuntime`**: Typed helpers for Relay-owned scopes and marks.
- **Versioned native C ABI**: C-compatible host and plugin tables behind the
  safe Rust authoring interface. Manifest native API v1 uses the V3 table and
  native API v2 uses its append-only V4 extension. Plugins must still be
  rebuilt for V3 as described in the
  [0.7 migration guide](https://docs.nvidia.com/nemo/relay/reference/migration-guides#upgrade-to-nemo-relay-07).
- **Raw async middleware**: Completion-based raw registrations for plugins
  that need asynchronous guardrails, intercepts, or event sanitizers. Typed
  Rust callbacks on native API v1 remain synchronous convenience APIs.
- **Safe native API v2 LLM continuations**: Register future-returning Rust
  callbacks, dispatch explicit provider targets, and consume provider events
  as Rust streams without writing C callback or handle-management code.

## Installation

Add the SDK to a Rust dynamic-plugin project:

```bash
cargo add nemo-relay-plugin futures serde_json
```

Configure the library as a dynamic library:

```toml
[lib]
crate-type = ["cdylib"]
```

## Getting Started

Implement `NativePlugin` and export a constructor symbol:

```rust
use nemo_relay_plugin::{Json, NativePlugin, PluginContext, Result};
use serde_json::Map;

struct ExamplePlugin;

impl NativePlugin for ExamplePlugin {
    fn plugin_kind(&self) -> &str {
        "example.native"
    }

    fn register(&mut self, _config: &Map<String, Json>, ctx: &mut PluginContext<'_>) -> Result<()> {
        ctx.register_subscriber("log-events", |event| {
            eprintln!("{}", event.name());
        })
    }
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, || ExamplePlugin);
```

Build the `cdylib`, describe its entry symbol and compatibility in a
`relay-plugin.toml` manifest, then register it through the Relay CLI. See the
complete example for platform-specific artifact and manifest setup.

## Native API v2

Existing plugins continue to use manifest `compat.native_api = "1"` and
`nemo_relay_plugin!`. A plugin that needs Relay-owned provider dispatch exports
only native API v2:

```rust
nemo_relay_plugin::nemo_relay_plugin_v2!(
    nemo_relay_register_plugin,
    || ExamplePlugin
);
```

Set `compat.native_api = "2"` in `relay-plugin.toml`. Rust plugins normally use
`PluginContext::register_async_llm_execution_intercept` and
`PluginContext::register_async_llm_stream_execution_intercept`. The SDK owns the C
callback trampolines, host strings, JSON conversion, panic isolation, output
settlement, cancellation, and handle release.

```rust
use futures::StreamExt;
use nemo_relay_plugin::{
    LlmContinuationInvocationV2, LlmContinuationTargetV2,
    LlmStreamExecutionOutcomeV2,
};

let buffered_target = target.clone();
ctx.register_async_llm_execution_intercept("route", 0, move |_name, request, next| {
    let target = buffered_target.clone();
    async move {
        next.call(LlmContinuationInvocationV2 { request, target })
            .await
            .map_err(|failure| format!("{failure:?}"))
    }
})?;

ctx.register_async_llm_stream_execution_intercept("route-stream", 0, move |_name, request, next| {
    let target = target.clone();
    async move {
        let provider = next
            .open_stream(LlmContinuationInvocationV2 { request, target })
            .await
            .map_err(|failure| format!("{failure:?}"))?;
        let stream = provider.map(|item| item.map_err(|failure| format!("{failure:?}")));
        Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(stream)))
    }
})?;
```

The versioned `register_async_llm_execution_v2` and
`register_async_llm_stream_execution_v2` names remain aliases for the same
native API v2 behavior. Existing synchronous
`register_llm_execution_intercept` and
`register_llm_stream_execution_intercept` callbacks remain unchanged.

To invoke the ordinary untargeted downstream continuation for a buffered
request, call `LlmContinuationV2::call_passthrough`. For streaming, return
`LlmStreamExecutionOutcomeV2::Passthrough(request)`. The call remains inside
its managed Relay LLM lifecycle, but Relay pumps the downstream stream directly
through its bounded queue; provider events do not cross into the plugin merely
to be forwarded.

Continuation operations follow the outer callback mode. A streaming callback
can open targeted streams or pass through to the downstream stream, but it
cannot invoke a buffered continuation. Aggregate a streamed judge or side call
inside the policy when its result is needed before returning the caller stream.

The plugin provides JSON, an absolute target URL, and explicit target headers.
Relay sends the request with HTTP `POST` and binds that transport target to the
current LLM continuation without storing it in `LlmRequest.headers`.
Successful buffered calls return provider JSON up to 16 MiB. Provider
rejections return an HTTP status, bounded body, and safe response headers;
failures without an HTTP response use a small transport-oriented kind. The
plugin owns its retry and fallback policy.

Relay core performs the terminal targeted HTTP request after the remaining LLM
execution intercepts run. This contract is host-independent: it works through
the CLI gateway and through SDK-embedded Relay hosts that call the managed LLM
execution APIs directly.

Streaming dispatch returns `LlmProviderStreamV2`, which implements Rust
`Stream` and cancels unfinished provider work on drop. Targeted streaming
endpoints must return SSE with a JSON value in each `data` frame; the plugin
receives those JSON events rather than raw SSE framing. Provider streams permit
at most one outstanding pull; plugin output and direct pass-through use bounded
queues. Safe callbacks register through the generic V3
asynchronous-middleware APIs and return `Pending`. Relay then polls their Rust
futures and returned streams cooperatively on its Tokio runtime. Each resumed
poll restores the captured Relay continuation and scope context, and a pending
callback does not occupy a blocking worker. Output backpressure parks the task
until the bounded host queue can accept more data. No Rust future, trait object,
`serde_json::Value`, or allocator-owned Rust string crosses the C ABI boundary.

Callback futures and streams must be executor-neutral. A native plugin shared
library can link a different copy of an async runtime than the Relay host, so
host-side polling does not enter plugin-local runtime state. An integration may
instead own and bridge its own runtime explicitly, but it must not assume that
Tokio APIs such as `tokio::spawn` can discover Relay's runtime across the
dynamic-library boundary. The SDK itself has no Tokio dependency.

The raw `PluginContext::host_api_v4` table and generic V3 `Pending`
registration methods remain available for advanced ABI consumers and non-Rust
bindings. V4 adds targeted LLM continuation and host-task operations; it does
not define a separate blocking registration model. Code using the raw tables is
responsible for every callback lifetime, host string, completion, task and
stream settlement, cancellation, and release operation.

The manifest API number is distinct from the internal host-table ABI number:
native API v1 negotiates the V3 host table and native API v2 negotiates V4.
Native plugins are trusted in-process extensions. A v2 plugin owns its target
credentials; Relay transports them but excludes their values from diagnostics
and observability.

Clean plugin teardown unloads the native library normally. If teardown finds
an opaque callback, task, continuation, or stream handle that still owns plugin
code, Relay conservatively keeps only that library mapping loaded for the rest
of the process. All descriptor and handle state is still released. This avoids
unmapping a plugin while an escaped handle is returning through its own code.

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [Build Plugins guide](https://docs.nvidia.com/nemo/relay/build-plugins/about)
- [Rust native plugin example](https://github.com/NVIDIA/NeMo-Relay/blob/main/examples/rust-native-plugin/README.md)
