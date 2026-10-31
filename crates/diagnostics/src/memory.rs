// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inspection of reserved backing and bounded storage occupancy.

use super::{Inner, StoredObservation, lock};
pub use nemo_relay_types::diagnostics::DiagnosticMemoryUsage;

pub(super) fn diagnostic_memory_usage(inner: &Inner) -> DiagnosticMemoryUsage {
    let Some(backend) = &inner.backend else {
        return DiagnosticMemoryUsage::default();
    };
    let state = lock(&backend.state);
    memory_usage_locked(inner, &state)
}

pub(super) fn memory_usage_locked(inner: &Inner, state: &super::State) -> DiagnosticMemoryUsage {
    let backend = inner.backend.as_ref().expect("enabled capture");
    let event_slot_bytes = size_of::<StoredObservation>();
    let detail_bytes = state.detail.used_bytes();
    let detail_logical_bytes = state.detail.logical_bytes();
    DiagnosticMemoryUsage {
        reserved_bytes: backend.reserved_bytes,
        components: state.components.iter().flatten().count(),
        counters: state.counters.iter().flatten().count(),
        events: state.recent.len(),
        event_slot_bytes,
        event_capacity: inner.limits.max_events,
        history_reserved_bytes: state.recent.reserved_bytes(),
        history_occupied_bytes: state.recent.len() * event_slot_bytes,
        detail_bytes,
        detail_capacity_bytes: state.detail.capacity_bytes(),
        detail_reserved_bytes: state.detail.reserved_bytes(),
        detail_metadata_bytes: state.detail.metadata_bytes(),
        detail_logical_bytes,
        detail_slack_bytes: detail_bytes - detail_logical_bytes,
        metadata_bytes: state.metadata_used,
        window_buckets: state.windows.iter().flatten().count(),
        extensions: state.extensions.iter().flatten().count(),
        query_workspaces: backend.queries.as_ref().map_or(0, |pool| pool.occupancy()),
        projections: lock(&backend.projections).slots.iter().flatten().count(),
        projection_workspaces: lock(&backend.publication)
            .iter()
            .filter(|v| v.is_none())
            .count(),
    }
}

// Arc storage contains its two reference counts and aligned Inner payload. This
// includes the inline State/Backend mutex headers; dynamic pools are added once.
pub(super) fn context_layout_bytes() -> usize {
    std::alloc::Layout::new::<[usize; 2]>()
        .extend(std::alloc::Layout::new::<Inner>())
        .expect("diagnostic context layout fits")
        .0
        .pad_to_align()
        .size()
}
