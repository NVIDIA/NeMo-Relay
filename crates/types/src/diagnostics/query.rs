// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable selectors and opaque continuation.

use super::{ComponentOwnership, DiagnosticDomain, DiagnosticNumber, PluginDiagnosticFamily};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque continuation bound to source, capture revision, offsets, and scope.
///
/// Scope validation prevents continuation widening; the token is not authority.
/// A host must intersect requested selection with its own authorization scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCursor {
    /// Source incarnation whose retained state was selected.
    pub source_instance_id: Uuid,
    /// Source revision for detecting intervening mutations or reclamation.
    pub sequence: DiagnosticNumber,
    /// Next component-instance position in the bounded source ordering.
    pub component_offset: DiagnosticNumber,
    /// Next recent-event position in the bounded source ordering.
    pub event_offset: DiagnosticNumber,
    /// Source-validated fingerprint of the intersected query scope.
    pub scope: DiagnosticNumber,
}

/// Portable bounded selection; consumers never reset or mutate evidence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticQuery {
    /// Optional required snapshot version; omission selects this runtime's version.
    pub schema_version: Option<u32>,
    /// Select one logical component key across retained registrations.
    pub component_key: Option<String>,
    /// Select one registration incarnation.
    pub component_instance_id: Option<Uuid>,
    /// Select recent events by original refresh generation only.
    /// Lifetime and current-refresh summaries keep their declared scope.
    pub origin_generation: Option<DiagnosticNumber>,
    /// Select exact canonical code counters and events.
    pub code: Option<String>,
    /// Select the authoritative catalogue parent domain.
    pub domain: Option<DiagnosticDomain>,
    /// Select the plugin catalogue family derived from code ownership.
    pub plugin_family: Option<PluginDiagnosticFamily>,
    /// Select event causes only; code counters do not retain cause breakdowns.
    pub failure_class: Option<String>,
    /// Select component rows by current-refresh ownership and events by original ownership.
    /// Selected lifetime counters still span the entire component instance.
    pub ownership: ComponentOwnership,
    /// Select later recent events; summary and code counters remain cumulative.
    pub after_sequence: Option<DiagnosticNumber>,
    /// Requested component-instance row cap, rejected above core limits.
    pub max_components: Option<u32>,
    /// Requested recent-event cap, rejected above core limits.
    pub max_events: Option<u32>,
    /// Requested encoded response cap, rejected above core limits.
    pub max_response_bytes: Option<u32>,
    /// Resume a compatible bounded selection without granting additional scope.
    pub continuation: Option<DiagnosticCursor>,
}
