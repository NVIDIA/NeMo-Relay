<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Configure and use NeMo Flow OpenTelemetry export for OTLP-compatible tracing backends
---

# Export OpenTelemetry Traces

Use this skill when the destination is an OTLP/OpenTelemetry backend such as an
OpenTelemetry Collector, Jaeger, Tempo, or Honeycomb.

## Default Path

- Build the binding-specific `OpenTelemetryConfig`
- Set endpoint, service name, and any required headers
- Construct the subscriber
- Register it before running scoped work
- Deregister, flush, and shut down when the process or subsystem is done

## Things To Confirm

- transport: `http_binary` vs `grpc`
- endpoint and auth headers
- service naming and resource attributes
- whether deterministic flush-before-exit is required
- whether the chosen binding supports the desired transport

## Troubleshooting Focus

- no spans visible
- wrong endpoint or auth headers
- events emitted outside active scopes
- forgetting register/deregister or flush/shutdown steps

## References

- `docs/observability-with-opentelemetry.md`
- `crates/otel/README.md`
