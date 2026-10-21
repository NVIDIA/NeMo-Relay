// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inspection of reserved backing and bounded storage occupancy.

use super::{Inner, StoredObservation, lock};
use serde::Serialize;

#[derive(Debug, Default, Clone, Copy, Serialize)]
/// Backing capacity and current occupancy of one context.
///
/// These gauges describe the bounded backend rather than whole-process RSS.
/// Reserved bytes describe allocated capacity; occupied bytes describe its use
/// and are not additional allocations. History and detail totals are subsets of
/// `reserved_bytes`; logical detail and slack are subsets of `detail_bytes`.
/// All values are zero when collection is disabled.
pub struct DiagnosticMemoryUsage {
    /// Bytes reserved for context-owned backend backing storage.
    pub reserved_bytes: usize,
    /// Occupied active or retained closed component slots.
    pub components: usize,
    /// Occupied lifetime code-counter slots across admitted components.
    pub counters: usize,
    /// Currently retained shared circular-history event headers.
    pub events: usize,
    /// Compiled size of one fixed event slot, excluding separately stored text.
    pub event_slot_bytes: usize,
    /// Event slots admitted for the shared circular history.
    pub event_capacity: usize,
    /// Complete history allocation, including all slots, shared header, and padding.
    pub history_reserved_bytes: usize,
    /// Retained slot bytes; excludes the shared history header and separate text pool.
    pub history_occupied_bytes: usize,
    /// Pool payload bytes occupied in whole reclaimable blocks.
    pub detail_bytes: usize,
    /// Actual payload capacity, rounded down to whole detail blocks.
    pub detail_capacity_bytes: usize,
    /// Complete detail-pool backing: payload capacity plus all block metadata.
    pub detail_reserved_bytes: usize,
    /// Reserved metadata bytes for all detail blocks, occupied or free.
    pub detail_metadata_bytes: usize,
    /// Retained UTF-8 message and correlation bytes, excluding block slack.
    pub detail_logical_bytes: usize,
    /// Unused payload bytes inside occupied blocks; excludes entirely free blocks.
    pub detail_slack_bytes: usize,
    /// Admitted key, ownership, and extension metadata content bytes.
    pub metadata_bytes: usize,
    /// Reserved rolling outcome-bucket slots belonging to components.
    pub window_buckets: usize,
    /// Admitted extension descriptors, including retained closed components.
    pub extensions: usize,
    /// Query leases currently holding preallocated materialization buffers.
    pub query_workspaces: usize,
    /// Optional projection subscriptions currently admitted.
    pub projections: usize,
    /// Projection workspaces currently materializing or delivering evidence.
    pub projection_workspaces: usize,
}

pub(super) fn diagnostic_memory_usage(inner: &Inner) -> DiagnosticMemoryUsage {
    let Some(backend) = &inner.backend else {
        return DiagnosticMemoryUsage::default();
    };
    let state = lock(&backend.state);
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
