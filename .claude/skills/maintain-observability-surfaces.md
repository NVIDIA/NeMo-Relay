<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Maintain or extend NeMo Flow observability surfaces across ATIF, OpenTelemetry, and OpenInference
---

# Maintain Observability Surfaces

Use this skill when changing event fields, exporter behavior, subscriber config,
or binding parity for ATIF, OpenTelemetry, or OpenInference.

## Surfaces To Keep In Sync

- core event model and emitted fields
- `crates/otel` and `crates/openinference`
- FFI and binding-native wrappers where the config or lifecycle is exposed
- Python, Go, Node.js, and WASM config objects and subscriber/exporter methods
- docs under `docs/atif-export.md`,
  `docs/observability-with-opentelemetry.md`, and
  `docs/observability-with-openinference.md`

## Design Checklist

- [ ] Is this an event-model change, exporter-config change, or lifecycle change?
- [ ] Do all bindings expose the same logical knobs and semantics?
- [ ] Are mark events, start/end events, and orphan cases still handled correctly?
- [ ] Do examples and docs reflect the same lifecycle: create, register, run,
  deregister, flush, shutdown?
- [ ] Are span or trajectory fields still derived from the intended event data?

## Validation

- Run the affected Rust crate tests plus `cargo test --workspace` if event fields
  changed.
- Run Python, Go, Node.js, and WASM tests when binding-native config or lifecycle
  changed.
- Update docs and examples in the same branch.

## References

- `docs/atif-export.md`
- `docs/observability-with-opentelemetry.md`
- `docs/observability-with-openinference.md`
- `validate-change`
