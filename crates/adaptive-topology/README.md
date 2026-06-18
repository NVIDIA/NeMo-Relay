<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-relay-adaptive-topology

Topology-inspired, manifold, and adaptive-threshold primitives for NeMo Relay.

This crate provides a small deterministic toolkit for observing and reacting to
the shape of runtime data. The signals are intentionally lightweight
approximations for adaptive runtime control; they are not an exact persistent
homology implementation. The crate is designed to work in `no_std`
environments as well as standard Rust builds.

## Module Map

- `governor` — `GeometricGovernor`, a PD controller that adapts a sensitivity
  threshold `epsilon` to keep an effective tick rate near a target.
- `drift` — `DriftDetector`, a centroid velocity tracker that measures
  unexpected drift by comparing each new centroid with the position predicted
  from the previous step.
- `convergence` — `BettiNumbers` and `ConvergenceDetector`, which declare
  convergence when Betti numbers stabilize, drift decays, and error drops.
- `geometry` — `BlockMetadata` summaries and `HierarchicalBlockTree` for
  multi-scale geometric queries and compression hints.
- `manifold` — `ManifoldPoint`, `TimeDelayEmbedder`, `SparseAttentionGraph`,
  and `GeometricConcentrator` for embedding streams and analyzing local
  manifold structure.
- `topology` — `TopologicalShape` and byte-stream Betti approximations for
  shape signatures and verification.

## Example

```rust
use nemo_relay_adaptive_topology::{GeometricGovernor, DriftDetector, ConvergenceDetector, BettiNumbers};

let mut governor = GeometricGovernor::new();
for _ in 0..10 {
    // High deviation should raise the threshold to reduce sensitivity.
    governor.adapt(10_000.0, 0.001);
}

let mut drift = DriftDetector::<3>::new();
drift.update(&[1.0, 0.0, 0.0]);
drift.update(&[2.0, 0.0, 0.0]);
assert!(drift.is_drifting(0.5));

let mut conv = ConvergenceDetector::new(0.001, 3);
conv.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.02);
conv.record_epoch(BettiNumbers::new(1, 0), 0.01, 0.015);
conv.record_epoch(BettiNumbers::new(1, 0), 0.005, 0.01);
assert!(conv.is_converged());
```

## Features

- `std` (default) — enable standard-library support.
- `alloc` — reserved for future allocation-dependent APIs in `no_std`
  builds.
- `serde` — derive `Serialize`/`Deserialize` for public types.
