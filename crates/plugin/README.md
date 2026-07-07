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
  the `nemo_relay_plugin!` macro.
- **Use host runtime helpers**: Emit events and manage scope state through the
  high-level `PluginRuntime` wrapper.

## What You Get

- **`NativePlugin`**: Plugin kind, configuration validation, and registration
  lifecycle contract.
- **`PluginContext`**: Component-scoped registration APIs for middleware and
  subscribers.
- **`PluginRuntime`**: Typed helpers for Relay-owned scopes and marks.
- **Stable native ABI v1**: C-compatible host and plugin tables behind the
  safe Rust authoring interface.

## Installation

Add the SDK to a Rust dynamic-plugin project:

```bash
cargo add nemo-relay-plugin serde_json
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

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [Build Plugins guide](https://docs.nvidia.com/nemo/relay/build-plugins/about)
- [Rust native plugin example](../../examples/rust-native-plugin/README.md)
