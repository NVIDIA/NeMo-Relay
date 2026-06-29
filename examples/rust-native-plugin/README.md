<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust Native Dynamic Plugin

This example shows a trusted in-process Rust dynamic plugin using the
high-level `nemo-relay-plugin` SDK. It builds as a `cdylib`, exports a stable
native ABI entry symbol, validates JSON config, registers middleware and
subscribers, emits runtime marks/scopes, and creates an isolated scope stack.

The example intentionally depends on `nemo-relay-plugin`, not on the host
`nemo-relay` runtime crate. Rust DTOs stay inside the plugin crate; the
dynamic-library boundary remains the stable C ABI.

## Build

Run this command from the example directory:

```bash
cargo build
```

Before you register the plugin, replace `<platform-library-file>` in
`relay-plugin.toml` with the file name that `cargo build` creates for your
platform:

| Platform | Library path |
|---|---|
| macOS | `target/debug/libnemo_relay_rust_native_plugin_example.dylib` |
| Linux | `target/debug/libnemo_relay_rust_native_plugin_example.so` |
| Windows | `target/debug/nemo_relay_rust_native_plugin_example.dll` |

## Register With Relay

After updating `load.library`, run these commands from the repository root:

```bash
nemo-relay plugins add ./examples/rust-native-plugin/relay-plugin.toml
nemo-relay plugins enable examples.rust_native_policy
```

You can also reference the manifest manually from `plugins.toml`:

```toml
[[plugins.dynamic]]
manifest = "./examples/rust-native-plugin/relay-plugin.toml"

[plugins.dynamic.config]
tag = "demo"
block_tools = false
block_llms = false
emit_isolated_scope = true
```

The manifest declares the `config_schema` capability and references
`config.schema.json`. After adding the plugin, use the editor for the same
configuration target (`--user`, `--project`, or `--global`) to configure the
fields without loading the native library:

```bash
nemo-relay plugins edit
```

The editor reads the schema file relative to `relay-plugin.toml`. It does not
run the plugin during schema discovery.

Start the gateway normally after the dynamic record is enabled:

```bash
nemo-relay gateway
```

## What the Example Registers

The example registers the following runtime behavior:

- A subscriber that emits a mark when it sees non-plugin scope starts.
- Tool sanitize request/response guardrails for observability payload tagging.
- Conditional execution guardrails for tools and LLMs controlled by config.
- Request and execution intercepts for tools that mutate JSON payloads and call
  continuations.
- LLM sanitize request/response guardrails.
- LLM request, execution, and stream execution intercepts.
- Runtime mark and scope events.
- A plugin-owned isolated scope stack for non-correlated visibility.

Native plugins are not sandboxed. They run in the Relay process and must not
unwind across ABI callbacks.
