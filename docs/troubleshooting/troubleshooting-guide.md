<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting Guide

Use this page when a NeMo Flow setup, build, or runtime workflow does not behave as expected.

## Package Or Build Setup Fails

Confirm that your environment matches [Prerequisites](../getting-started/prerequisites.md), then rerun the binding-specific setup command from [Installation](../getting-started/installation.md).

If a command worked previously and now fails, check whether a toolchain update changed the active Rust, Python, Node.js, Go, or WebAssembly version. Recreate generated artifacts after switching versions, especially Python virtual environments, Node.js `node_modules`, Go build caches, and WebAssembly `pkg/` output.

## Rust Workspace Fails To Build

Run the narrowest failing build first:

```bash
cargo build -p nemo-flow
```

If the core crate builds but another crate fails, rerun the binding-specific build or test command from [Testing and Documentation](../contribute/testing-and-docs.md). Binding crates often depend on generated native artifacts, host toolchain headers, or runtime-specific test setup.

## Python Native Module Does Not Import

If `import nemo_flow` fails, rebuild the Python environment and native extension:

```bash
uv sync
uv run pytest python/tests/test_types.py
```

Confirm that Python satisfies the supported version from [Prerequisites](../getting-started/prerequisites.md). If multiple Python installations are available, make sure the `uv` environment is the one running the test or application process.

## Node.js Native Addon Does Not Load

If Node.js reports a missing or incompatible native addon, reinstall dependencies from the repository root so the pretest build can regenerate the local addon:

```bash
npm install
npm test --workspace=nemo-flow-node
```

Use [Node.js Getting Started](../getting-started/nodejs.md) to confirm that the example runs against the generated local build, not a stale package or a globally installed copy.

## Go Tests Cannot Find The FFI Library

The Go binding loads the shared FFI library through CGo. Build the release FFI library before running Go tests, and point the linker and runtime loader at the release target directory:

```bash
cargo build --release -p nemo-flow-ffi
cd go/nemo_flow
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="../../target/release" go test -race -v ./...
```

On macOS, use `DYLD_LIBRARY_PATH="../../target/release"` instead of `LD_LIBRARY_PATH` when the runtime loader cannot find the library.

## WebAssembly Tests Or Package Builds Fail

Confirm that `wasm-pack` is installed and matches the minimum version in [Prerequisites](../getting-started/prerequisites.md), then rerun the WebAssembly build or tests:

```bash
wasm-pack build crates/wasm
wasm-pack test --node crates/wasm
```

If generated files appear stale, remove the WebAssembly package output and rebuild it from the repository root.

## Scope Stack Is Empty

A scope-stack error usually means runtime work is executing outside an active scope or on a thread that did not receive the intended scope stack.

Use [Scopes](../about/concepts/scopes.md) and [Instrument Applications Code Examples](../instrument-applications/code-examples.md#scope-and-context-helpers) to choose the correct scope stack helper for the binding and concurrency model.

## Work Leaks Across Requests

Unexpected shared scopes, middleware, or events across concurrent requests usually means more than one request is using the same scope stack. Create a fresh scope stack per request or agent, and pass that stack into async tasks, threads, callbacks, or worker boundaries that continue the request.

Use [Adding Scopes and Marks](../instrument-applications/adding-scopes-and-marks.md) and [Scopes](../about/concepts/scopes.md) to verify that the root scope matches the request boundary.

## Middleware Does Not Run

Check whether the middleware was registered globally or scope-locally. Scope-local middleware is visible only to the owning scope and descendant scopes, and it is cleaned up when the owning scope closes.

Use [Middleware](../about/concepts/middleware.md), [Add Middleware](../instrument-applications/advanced-guide.md), and [Instrument Applications Code Examples](../instrument-applications/code-examples.md#middleware-registration-families) to verify the expected registration family.

## Middleware Runs In The Wrong Order

Middleware runs by priority after the runtime merges global middleware with scope-local middleware from the active scope chain. If the order is surprising, check the priority assigned to each middleware entry and confirm that similarly named global and scope-local entries are registered in the intended registry.

Use [Middleware](../about/concepts/middleware.md) and [Instrument Applications Code Examples](../instrument-applications/code-examples.md#middleware-registration-families) to compare registration names, priorities, and scope ownership.

## Guardrail Rejections Stop Calls

A guardrail rejection is expected to stop the protected tool or LLM call. Inspect the guardrail result and confirm whether the guardrail was intended to sanitize input, sanitize output, or reject the request completely.

Use [Add Middleware](../instrument-applications/advanced-guide.md) to verify the guardrail family and expected behavior.

## Request Intercept Changes Are Not Visible

Request intercepts transform the request before execution. If the original value still appears downstream, confirm that the intercept returns the changed value in the binding-specific shape and that a later request intercept is not replacing it.

Use [Instrument Applications Code Examples](../instrument-applications/code-examples.md#middleware-registration-families) to compare the expected request intercept callback signature for the binding.

## Execution Intercept Hangs Or Skips The Original Call

Execution intercepts use a middleware-chain pattern. An intercept that never calls `next` intentionally short-circuits the original callable, while an intercept that awaits or calls `next` incorrectly can hang the request.

Use [Middleware](../about/concepts/middleware.md) to confirm when an execution intercept should call `next`, replace the result, or stop the chain.

## Subscribers Do Not Receive Events

Confirm that the subscriber is registered before the runtime emits the events you expect. For scope-local subscribers, confirm that the active scope is the owning scope or a descendant scope.

Use [Subscribers](../about/concepts/subscribers.md), [Events](../about/concepts/events.md), and [Observability](../plugins/observability/about.md) to verify lifecycle timing and event names.

## Events Are Missing Expected Fields

Managed tool and LLM helpers populate semantic fields such as inputs, outputs, model names, and tool call IDs. Manual lifecycle calls require you to provide the relevant params explicitly.

Use [Events](../about/concepts/events.md) and [Instrument Applications Code Examples](../instrument-applications/code-examples.md) to verify the emitted payload shape.

## Agent Trajectory Interchange Format (ATIF) Export Is Empty Or Mixed Across Agents

An empty Agent Trajectory Interchange Format (ATIF) export usually means the
exporter subscribed after the relevant events were emitted, or the export
filter does not match the active `root_uuid`. Mixed trajectories usually mean
multiple agents share a root scope or the export did not filter by root scope.

Use [Agent Trajectory Interchange Format (ATIF)](../plugins/observability/atif.md)
and [Observability](../plugins/observability/about.md) to confirm exporter
setup, event collection timing, and root-scope filtering.

## LLM Stream Output Is Missing The Final Chunk

When wrapping streamed LLM responses, confirm that the stream wrapper receives the provider's terminal event and that the application drains the stream until completion. A stream that is dropped early can prevent finalizers, collectors, or subscribers from seeing the completed output.

Use [Wrap LLM Calls](../integrate-frameworks/wrap-llm-calls.md) and [Provider Response Codecs](../integrate-frameworks/provider-response-codecs.md) to verify the expected stream event format.

## Provider Payloads Fail To Convert

JSON conversion errors usually mean the integration passed a value that cannot be represented in NeMo Flow's JSON model, such as functions, class instances, handles, or provider-specific streaming objects.

Use [Non-Serializable Data](../integrate-frameworks/non-serializable-data.md), [Provider Codecs](../integrate-frameworks/provider-codecs.md), and [Using Codecs](../integrate-frameworks/using-codecs.md) to define explicit conversions for provider-specific payloads.

## Plugin Configuration Does Not Validate

If a plugin fails before runtime execution, validate configuration separately from behavior registration. Check required fields, value types, defaults, and whether the plugin is reading configuration from the expected source.

Use [Validate Configuration](../build-plugins/validate-configuration.md), [Advanced Configuration](../build-plugins/advanced-configuration.md), and [Register Behavior](../build-plugins/register-behavior.md) to isolate configuration problems from runtime behavior problems.

## Adaptive Tuning Does Not Change Behavior

Confirm that adaptive tuning is configured for the component you expect and that
the runtime path actually reaches that component. If behavior does not change,
check whether the configured policy is disabled, scoped too narrowly, or not
connected to the call path under test.

Use [Adaptive Configuration](../plugins/adaptive/configuration.md),
[Adaptive Cache Governor (ACG)](../plugins/adaptive/acg.md), and
[Adaptive Hints](../plugins/adaptive/adaptive-hints.md) to verify component
names and configuration scope.

## Third-Party Patch Does Not Apply

Run the wrapper command from the repository root:

```bash
./scripts/apply-patches.sh --check
```

If the patch still does not apply, confirm that the local checkout is clean and matches the pinned commit in `third_party/sources.lock`.

## Third-Party Integration Behaves Differently From Core APIs

First reproduce the behavior through the closest core or binding-level API. If the core API behaves correctly, inspect the integration wrapper, codec, or provider adapter that translates provider calls into NeMo Flow calls.

Use [Integrate Frameworks](../integrate-frameworks/about.md), [Wrap Tool Calls](../integrate-frameworks/wrap-tool-calls.md), and [Wrap LLM Calls](../integrate-frameworks/wrap-llm-calls.md) to compare the integration path with the core runtime path.
