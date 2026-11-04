// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup settings and checked admission of backing allocations.

use super::{
    DiagnosticError, ID_BYTES, Projection, PublicationWorkspace, State, context_layout_bytes, read,
};
use serde::Deserialize;
use serde::Serialize;

pub(super) const MAX_DETAIL_BYTES: usize = 4096;

/// Independent capacities of one diagnostic context. All backing allocations are
/// admitted during startup; these limits are not a whole-process RSS ceiling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticLimits {
    /// Active and retained closed registration slots shared by this context.
    pub max_components: usize,
    /// Declared lifetime code-counter slots shared by admitted components.
    pub max_counters: usize,
    /// Shared circular-history headers; zero retains summaries without event history.
    pub max_events: usize,
    /// Aggregate outcome-bucket slots shared by admitted components.
    pub max_window_buckets: usize,
    /// Registered extension descriptors across all retained components.
    pub max_extensions: usize,
    /// Combined admitted component-key, ownership, and extension-description bytes.
    pub max_metadata_bytes: usize,
    /// Bytes admitted per component key or ownership string, at most 128.
    pub max_identity_bytes: usize,
    /// Bytes copied or formatted per observation detail, at most 4096.
    pub max_detail_bytes: usize,
    /// Bytes admitted per request or session correlation field, at most 128.
    pub max_correlation_bytes: usize,
    /// Payload-byte capacity of the shared detail pool, rounded down to whole blocks.
    pub detail_pool_bytes: usize,
    /// Payload bytes per reclaimable detail block, at least four for UTF-8 scalars.
    pub detail_block_bytes: usize,
    /// Simultaneously leased query workspaces; zero disables reads and their pool.
    pub max_queries: usize,
    /// Maximum component rows materialized in one query page.
    pub max_query_components: usize,
    /// Maximum recent-event rows materialized in one query page.
    pub max_query_events: usize,
    /// Code-counter rows materialized per page, independent of store reservations.
    pub max_query_counters: usize,
    /// Outcome buckets materialized per page, independent of store reservations.
    pub max_query_window_buckets: usize,
    /// Serialized bytes per query response, including its envelope and coverage.
    pub max_snapshot_bytes: usize,
    /// Text-arena bytes reserved for each leased query workspace.
    pub query_text_bytes: usize,
    /// Simultaneously registered callbacks; arbitrary callback captures are caller-owned.
    pub max_projections: usize,
    /// Concurrent publications with independent payload and callback-reference storage.
    pub max_projection_workspaces: usize,
    /// Startup ceiling for context backing allocations, excluding whole-process RSS.
    pub max_owned_bytes: usize,
}
impl Default for DiagnosticLimits {
    fn default() -> Self {
        Self {
            max_components: 64,
            max_counters: 2048,
            max_events: 256,
            max_window_buckets: 3840,
            max_extensions: 128,
            max_metadata_bytes: 65536,
            max_identity_bytes: 128,
            max_detail_bytes: 4096,
            max_correlation_bytes: 128,
            detail_pool_bytes: 262144,
            detail_block_bytes: 256,
            max_queries: 4,
            max_query_components: 64,
            max_query_events: 256,
            max_query_counters: 256,
            max_query_window_buckets: 512,
            max_snapshot_bytes: 131072,
            query_text_bytes: 65536,
            max_projections: 16,
            max_projection_workspaces: 4,
            max_owned_bytes: 32 * 1024 * 1024,
        }
    }
}
impl DiagnosticLimits {
    /// Validate capacities and checked backing sizes without allocating a store.
    pub fn validate(&self) -> Result<(), DiagnosticError> {
        if self.max_components == 0
            || self.max_identity_bytes == 0
            || self.max_identity_bytes > ID_BYTES
            || self.max_detail_bytes > MAX_DETAIL_BYTES
            || self.max_correlation_bytes > ID_BYTES
            || self.detail_block_bytes < 4
            || (self.max_queries > 0
                && (self.max_query_components == 0 || self.max_snapshot_bytes < 4096))
            || self.max_projection_workspaces == 0
        {
            return Err(DiagnosticError::InvalidLimits);
        }
        let mut total = State::required_bytes(self)?;
        for bytes in [
            read::QueryPool::required_bytes(self)?,
            self.max_projection_workspaces
                .checked_mul(PublicationWorkspace::required_bytes(self)?)
                .ok_or(DiagnosticError::InvalidLimits)?,
            self.max_projection_workspaces
                .checked_mul(size_of::<Option<PublicationWorkspace>>())
                .ok_or(DiagnosticError::InvalidLimits)?,
            self.max_projections
                .checked_mul(size_of::<Option<(u64, Projection)>>())
                .ok_or(DiagnosticError::InvalidLimits)?,
            context_layout_bytes(),
            ID_BYTES * 2,
        ] {
            total = total
                .checked_add(bytes)
                .ok_or(DiagnosticError::InvalidLimits)?;
        }
        if total > self.max_owned_bytes {
            return Err(DiagnosticError::InvalidLimits);
        }
        Ok(())
    }
}

/// Startup-only runtime setting. Collection is compiled normally and off unless
/// the owner enables this context explicitly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticsSettings {
    /// Enable this context at startup; the default is false.
    pub enabled: bool,
    /// Storage capacities validated before an enabled backend is allocated.
    pub limits: DiagnosticLimits,
}
