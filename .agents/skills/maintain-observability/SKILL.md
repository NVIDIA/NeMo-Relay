---
name: maintain-observability
description: Maintain or extend NeMo Relay observability surfaces across ATIF and typed OpenTelemetry projections
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Maintain Observability Surfaces

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when changing event fields, exporter behavior, subscriber config,
or binding parity for ATIF or the `full`, `gen_ai`, and `openinference`
OpenTelemetry projections.

## Surfaces To Keep In Sync

- Core event model and emitted fields
- `crates/core/src/observability/atif.rs`
- `crates/core/src/observability/otel.rs`
- `crates/core/src/observability/openinference.rs`
- FFI and binding-native wrappers where the config or lifecycle is exposed
- Python, Go, and Node.js config objects and subscriber/exporter methods
- Observability config version 3, where one `opentelemetry` section contains
  typed endpoints and OpenInference has no standalone public surface
- Docs under `docs/about-nemo-relay/concepts/subscribers.mdx` and
  `docs/configure-plugins/observability/`

## Design Checklist

- [ ] Is this an event-model change, exporter-config change, or lifecycle change?
- [ ] Do all bindings expose the same logical knobs and semantics?
- [ ] Does every OpenTelemetry endpoint require a type and nonblank destination?
- [ ] Does each endpoint resolve `header_env` values at activation and reject
  missing, blank, or duplicate headers?
- [ ] Do layered ATOF sink, ATIF storage, and OpenTelemetry endpoint lists
  concatenate with higher-precedence entries first?
- [ ] Are OpenTelemetry and OpenInference dependencies unconditional rather
  than Cargo feature-gated?
- [ ] Does `gen_ai` avoid `nemo_relay.*`, project sanitized LLM instructions
  and messages into the standard content attributes, and emit minimal spans
  for scopes without GenAI semantics so their parentage is preserved?
- [ ] Does `enable_full_payloads` preserve complete sanitized LLM request input
  and annotations while leaving credential removal and sanitizers active?
- [ ] Does Relay derive compliant trace and span IDs consistently across typed
  OpenTelemetry endpoints while preserving lifecycle parentage?
- [ ] Are mark events, start/end events, and orphan cases still handled correctly?
- [ ] Does a sanitized tool result annotation remain opaque under
      `category_profile.tool_result_annotation`, ATIF observation-result
      `extra.tool_result_annotation`, and the single
      `nemo_relay.tool.result.annotation` attribute in `full` and
      `openinference`, while `gen_ai` omits it?
- [ ] Do examples and docs use each exporter's documented flush/deregister
  order before shutdown?
- [ ] Are span or trajectory fields still derived from the intended event data?

## Validation

- Run the affected Rust crate tests plus `just test-rust` if event
  fields changed.
- Run `just test-python`, `just test-go`, and `just test-node` when
  binding-native config or lifecycle changed.
- Update docs and examples in the same branch.

## References

- `docs/about-nemo-relay/concepts/subscribers.mdx`
- `docs/configure-plugins/observability/about.mdx`
- `docs/configure-plugins/observability/opentelemetry.mdx`
- `crates/core/src/observability/atif.rs`
- `crates/core/src/observability/otel.rs`
- `crates/core/src/observability/openinference.rs`
- `validate-change`
