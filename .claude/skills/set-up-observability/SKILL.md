<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Choose and set up the right NeMo Flow observability path for an application
---

# Set Up Observability

Use this skill when an application developer wants visibility into NeMo Flow
activity but has not yet decided which output they need.

## Choose The Output

- **Console or custom event handling**
  Use subscribers.
- **Portable execution trajectories**
  Use `AtifExporter`.
- **General OTLP tracing**
  Use the OpenTelemetry subscriber.
- **OpenInference-aware backends**
  Use the OpenInference subscriber.

## Shared Lifecycle

1. Create the exporter or subscriber.
2. Register it with a unique name.
3. Run NeMo Flow-instrumented work inside scopes.
4. Deregister it.
5. Flush or shut down if the binding supports it and deterministic delivery is needed.

## Use Another Skill When

- you already know you need ATIF -> `export-atif-trajectories`
- you already know you need OTEL -> `export-opentelemetry-traces`
- you already know you need OpenInference -> `export-openinference-traces`
- you are debugging missing telemetry -> `debug-runtime-integration`

## References

- `docs/atif-export.md`
- `docs/observability-with-opentelemetry.md`
- `docs/observability-with-openinference.md`
