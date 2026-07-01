---
name: maintain-observability
description: Maintain or extend NeMo Relay observability surfaces across ATIF, OpenTelemetry, and OpenInference
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Maintain Observability Surfaces

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when changing event fields, exporter behavior, subscriber config,
or binding parity for ATIF, OpenTelemetry, or OpenInference.

## Surfaces To Keep In Sync

- Core event model and emitted fields
- `crates/core/src/observability/atif.rs`
- `crates/core/src/observability/otel.rs`
- `crates/core/src/observability/openinference.rs`
- FFI and binding-native wrappers where the config or lifecycle is exposed
- Python, Go, and Node.js config objects and subscriber/exporter methods
- Docs under `docs/about/concepts/subscribers.md` and
  `docs/export-observability-data/about.md`

## Design Checklist

- [ ] Is this an event-model change, exporter-config change, or lifecycle change?
- [ ] Do all bindings expose the same logical knobs and semantics?
- [ ] Are mark events, start/end events, and orphan cases still handled correctly?
- [ ] Do examples and docs reflect the same lifecycle: create, register, run,
  deregister, flush, shutdown?
- [ ] Are span or trajectory fields still derived from the intended event data?

## Validation

- Run the affected Rust crate tests plus `just test-rust` if event
  fields changed.
- Run `just test-python`, `just test-go`, and `just test-node` when
  binding-native config or lifecycle changed.
- Update docs and examples in the same branch.

## References

- `docs/about/concepts/subscribers.md`
- `docs/export-observability-data/about.md`
- `docs/export-observability-data/code-examples.md`
- `crates/core/src/observability/atif.rs`
- `crates/core/src/observability/otel.rs`
- `crates/core/src/observability/openinference.rs`
- `validate-change`
