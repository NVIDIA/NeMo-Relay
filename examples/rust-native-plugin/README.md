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

Run the focused tests and build the shared library from this directory. The
configuration tests isolate validation and schema contracts. The lifecycle test
builds a fresh `cdylib`, materializes a digest-checked manifest, activates the
plugin in a host, executes middleware, observes its runtime mark, and clears
the callbacks before unloading the library:

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

The optional `registration_control` group demonstrates a host-resident,
activation-owned gate. It defaults to `enabled: false`, `kinds: ["subscriber"]`,
`registration_name: "documentation-controlled-subscriber"`, and
`reason: "disabled by documentation plugin"`. The kinds, effective target name,
and reason must be nonempty. Refer to
[Conditional Middleware Guardrails](../../docs/about-nemo-relay/concepts/conditional-middleware-guardrails.mdx)
before enabling it against a discovered runtime target.
