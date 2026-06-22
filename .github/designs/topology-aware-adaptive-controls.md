<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Topology-Aware Adaptive Controls Design

This is a reviewer-facing design note for PR #282. It is intentionally kept out
of the published Fern documentation because it records internal implementation
tradeoffs, benefit gates, and validation samples rather than user-facing usage
instructions.

## Problem

The Adaptive plugin learns from repeated runtime observations. Before this
change, the relevant paths had three avoidable failure modes:

- ACG learning kept consuming observations for a stable prompt profile until
  the observation window was exhausted, even when the profile had already been
  stable for multiple epochs.
- Tool parallelism could retain stale fan-out groups after the observed tool
  cohort shape changed sharply.
- Learned adaptive hints were injected whenever defaults existed, even when the
  learned latency-sensitivity signal was below the configured value needed to
  justify request metadata.

The proposed controls are useful only if they make one of those states
observable and measurably better. If a representative workload does not show
one of the benefit gates below, the control should remain disabled.

## Goals

- Stop ACG learning after repeated stable prompt structure has been observed.
- Discard stale tool-parallelism plans when observed tool cohort shape changes
  sharply.
- Shed learned adaptive hints below a configurable sensitivity threshold while
  preserving manual latency-sensitivity overrides.
- Keep every control disabled by default and observable through existing
  adaptive state, request metadata, and validation reports.

## Non-Goals

- Exact persistent homology or a general-purpose topology library.
- New public Rust, Python, Node.js, Go, WebAssembly, or C FFI topology
  primitives.
- Changes to NeMo Relay scope semantics, event shape, callback execution, or
  user callback return values.
- Public documentation of internal topology algorithms.

## Internal Design

The adaptive crate owns a small internal module, `crate::topology`, with three
bounded primitives:

- `ConvergenceDetector` tracks a fixed history of Betti-like stability
  signatures, drift, and error.
- `DriftDetector` tracks centroid motion for tool cohort feature vectors.
- `GeometricGovernor` adapts a sensitivity threshold for learned hint
  injection.

ACG maps each stability analysis result to:

```text
beta_0 = stable_prefix_length
beta_1 = total_spans - stable_prefix_length
drift  = 1 - stable_prefix_length / total_spans
error  = 1 - average_stability_score
```

The tool-parallelism learner maps observed tool cohorts to a four-value
centroid:

```text
[cohort_count, unique_tool_count, duplicate_reference_ratio, max_cohort_size]
```

Adaptive hints use the governor only for learned hints. A manual
`set_latency_sensitivity()` override still forces hint injection for the current
request.

## Benefit Gates

Each control must satisfy a concrete benefit gate before it is enabled for a
workload:

| Control | Benefit Gate | Observable Signal | Validation |
|---|---|---|---|
| ACG convergence | Stable profiles use fewer observations before decision than the configured observation window while preserving stored stability. | Persisted `StabilityAnalysisResult.converged = true`; later runs reuse cached stability and skip observation repair only after the observations are stored. | `crates/adaptive/tests/integration/topology_convergence_tests.rs` and `crates/adaptive/benches/convergence_bench.rs`. |
| Tool drift | A plan that was learned from an old tool-cohort shape is removed when the next observed cohort shape crosses the configured drift threshold. | The stored `ExecutionPlan` no longer contains stale fan-out groups after drift. | `crates/adaptive/tests/unit/tool_parallelism_learner_tests.rs`. |
| Hint governor | Low-sensitivity learned hints are omitted when below `adaptive_hints.governor.epsilon`, while manual overrides still emit hints. | `nvext.agent_hints` is absent from request headers/body for shed learned hints; manual overrides still add the field. | `crates/adaptive/tests/unit/adaptive_hints_intercept_tests.rs`. |
| Config safety | Invalid thresholds fail before activation. | Plugin validation diagnostics name invalid topology-aware fields. | `crates/adaptive/tests/unit/runtime_tests.rs` and `crates/adaptive/tests/unit/plugin_component_tests.rs`. |

## Sample Evidence

These samples use deterministic fixtures from this change set. They are not
general performance guarantees; they show the expected decision points and the
state a reviewer or operator can inspect.

| Control | Sample Workload | Baseline | With Control | Observable Result |
|---|---|---|---|---|
| ACG convergence | `50` repeated stable prompt observations, `observation_window = 100`, `stability_window = 3`, and `epsilon = 0.001`. | The learner consumes the full fixture before the benchmark decision path ends. | Convergence is declared after the third stable epoch. | `cargo bench -p nemo-relay-adaptive --bench convergence_bench -- --sample-size 10` prints `observations-to-decision: without=50, with=3`. |
| Tool drift | First run observes overlapping `search` and `fetch`; next run observes overlapping `compile`, `test`, and `lint`. | The old stored plan can keep stale `fanout:existing` groups. | The centroid moves from `[1, 2, 0.0, 2]` to `[2, 3, 0.4, 3]`, producing drift above the `0.01` test threshold and rebuilding from an empty plan. | The rebuilt `ExecutionPlan` no longer contains `fanout:existing`. |
| Hint governor | Learned default hints have `latency_sensitivity = 2.0`; governor `epsilon = 10.0`. | Learned hints would be injected whenever defaults exist. | The low-sensitivity learned hint is shed, while a manual `set_latency_sensitivity(11)` override still forces injection. | The shed request has no `nvext.agent_hints` header or body field; the manual request emits `latency_sensitivity = 11.0`. |

## Rollout

All topology-aware fields default to disabled. A rollout should enable one
control at a time, validate representative workloads, and use existing adaptive
state inspection to confirm the observable signals above before enabling the
next control.

Recommended rollout order:

1. Enable ACG convergence only for profiles with stable prompts and compare
   observations-to-decision against the observation window.
2. Enable tool drift only for agents where stale fan-out plans have been seen or
   where tool cohorts are expected to change between phases.
3. Enable hint governor only after learned hints are present and request
   metadata volume matters.

If any gate does not show a benefit on the target workload, leave that control
disabled.
