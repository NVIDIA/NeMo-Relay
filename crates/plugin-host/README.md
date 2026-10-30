<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-relay-plugin-host

Native and gRPC plugin discovery, configuration, validation, trust verification,
registration, activation, diagnostics, and teardown. Depends on runtime execution
APIs; runtime execution does not depend on plugin hosting.

This crate also owns default built-in pricing, resource metrics, and observability
composition. `initialize_runtime()` installs default services while preserving
services explicitly selected by an embedding application. Plugin initialization
and the `exporter` constructors install these defaults automatically. Exporter
constructors inject runtime codec, scope, and diagnostic adapters internally.
The public `nemo-relay` Rust SDK re-exports these supported entry points.
