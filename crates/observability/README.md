<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Observability

Built-in ATOF, ATIF, and OpenTelemetry exporters and their GenAI and OpenInference
projections. This crate owns export formats, delivery, TLS, and file confinement.

Import configuration and exporter types from `nemo-relay-observability`. For the
Rust SDK, use constructors such as `nemo_relay::exporter::atof(config)`
and `nemo_relay::exporter::opentelemetry(config)`. They supply a runtime host
internally, including subscriber registration, delivery barriers, trace-context
policy, diagnostic reporting, and provider normalization. Standalone exporters
can also be constructed directly; custom runtimes can bind their own host with
`with_host`.

Provider normalization uses the `LlmNormalizer` contract from `nemo-relay-types`.
`nemo-relay-codecs` owns provider implementations and detection;
`nemo-relay-runtime` adds configured pricing through its normalization adapter.
`nemo-relay-plugin-host` supplies runtime adapters to exporters. Runtime hosts
are bound per exporter, without a process-global callback registry.

This package participates in the unified NeMo Relay release and is published
before `nemo-relay-plugin-host` and the `nemo-relay` SDK.

JSON Schema support, gRPC workers, object storage, and ATOF streaming are built
in to their owning crates. These capabilities require no Cargo feature flags,
including when consuming crates with `default-features = false`.
