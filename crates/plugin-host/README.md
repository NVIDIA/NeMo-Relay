<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-relay-plugin-host

Native and gRPC plugin discovery, configuration, validation, trust verification,
registration, activation, diagnostics, and teardown. Depends on runtime execution
APIs; runtime execution does not depend on plugin hosting.

Plugins receive the shared `PluginRegistrationContext` interface from
`nemo-relay-plugin` during activation. This crate implements it with
`HostPluginRegistrationContext`, which owns registration namespacing,
activation diagnostics, and tracked cleanup.

This crate also owns default built-in pricing, resource metrics, and observability
composition. `initialize_runtime()` installs default services while preserving
services explicitly selected by an embedding application. Plugin initialization
and the `exporter` constructors install these defaults automatically. Exporter
constructors inject runtime codec, scope, and diagnostic adapters internally.
The public `nemo-relay` Rust SDK re-exports these supported entry points.

Runtime-bound exporter fallbacks use the selected `RuntimeServices` pricing
snapshot. Explicitly disabled or custom runtime estimators apply to ATIF,
Full OpenTelemetry, and OpenInference alike. Detached observability exporters
use the standalone model-pricing catalog until a host is supplied.
