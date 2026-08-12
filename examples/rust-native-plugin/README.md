<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust Native Dynamic Plugin

This project is the complete native plugin used by the authoring guide. Its
configuration, observation, request policy, execution wrappers, and runtime
helpers live in separate source modules. Together they register the subscriber,
all three event sanitizers, five tool surfaces, and six LLM surfaces exposed by
the current typed 0.8.0 SDK.

Run the focused tests and build the shared library from this directory. Every
test has its own configuration input and asserts one validation, schema, or
packaging contract:

```bash
cargo test
cargo build
```

Copy `relay-plugin.toml` to `relay-plugin.local.toml` and replace
`<platform-library-file>` with the debug artifact name:

| Platform | Library Path |
|---|---|
| macOS | `target/debug/libnemo_relay_rust_native_plugin_example.dylib` |
| Linux | `target/debug/libnemo_relay_rust_native_plugin_example.so` |
| Windows | `target/debug/nemo_relay_rust_native_plugin_example.dll` |

Calculate the artifact digest with `shasum -a 256`, `sha256sum`, or
`Get-FileHash -Algorithm SHA256`, then replace `<artifact-sha256>` while keeping
the `sha256:` prefix. The same relative artifact path must appear in
`source.artifact` and `load.library`.

The strict schema documents every feature group and the SDK-owned
`executor.worker_threads` override.
