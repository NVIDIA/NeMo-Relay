<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Runtime

Execution, scopes, middleware registries, continuations, and priced codec adapters.
The SDK supplies built-in component composition through explicit runtime services.
This crate participates in the unified NeMo Relay release.

This crate has no production dependency on the public SDK or concrete component
crates. Applications may use it alone without built-in plugins, or install
`composition::RuntimeServices` before starting execution. The public SDK installs
its default services through `nemo_relay::initialize_runtime()` and its plugin
initialization and exporter constructors.

Use `nemo-relay-types` for codec and middleware contracts, `nemo-relay-plugin`
for the non-generic `Plugin` trait and `PluginRegistrar`, and this crate for
codec registration, execution context, and registries. SDK and plugin consumers
share the same runtime globals.

`nemo-relay-plugin-host` owns native and gRPC plugin loading, verification,
validation, activation, and teardown, and depends on this execution engine.
`nemo-relay-logging` owns logging configuration and backends; ordinary producers
use `log`. `nemo-relay-diagnostics` owns bounded diagnostic evidence and readers.
The engine has no production dependency on these three crates.

Provider implementations live in `nemo-relay-codecs`. Both its stateless
`BuiltinLlmNormalizer` and runtime’s `RuntimeLlmNormalizer` implement the
`LlmNormalizer` trait from `nemo-relay-types`. Runtime adds configured pricing
to normalized responses; standalone codecs preserve provider-reported costs.

JSON Schema support, gRPC workers, object storage, and ATOF streaming are built
in to their owning crates. These capabilities require no Cargo feature flags,
including when consuming crates with `default-features = false`.
