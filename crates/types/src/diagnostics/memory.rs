// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact diagnostic backing attribution, independent of OS resource usage.
use super::{
    ComponentIdentity, DiagnosticCursor, DiagnosticNumber, ObservationTime, SourceIdentity,
};
use serde::{Deserialize, Serialize};

/// Initial version of the separate memory inspection envelope.
pub const DIAGNOSTIC_MEMORY_VERSION: u32 = 1;

#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Backing capacity and current occupancy of one context.
///
/// These gauges describe the bounded backend rather than whole-process RSS.
/// Reserved bytes describe allocated capacity; occupied bytes describe its use
/// and are not additional allocations. History and detail totals are subsets of
/// `reserved_bytes`; logical detail and slack are subsets of `detail_bytes`.
/// All values are zero when collection is disabled.
pub struct DiagnosticMemoryUsage {
    /// Bytes reserved for context-owned backend backing storage.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub reserved_bytes: usize,
    /// Occupied active or retained closed component slots.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub components: usize,
    /// Occupied lifetime code-counter slots across admitted components.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub counters: usize,
    /// Currently retained shared circular-history event headers.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub events: usize,
    /// Compiled size of one fixed event slot, excluding separately stored text.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub event_slot_bytes: usize,
    /// Event slots admitted for the shared circular history.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub event_capacity: usize,
    /// Complete history allocation, including all slots, shared header, and padding.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub history_reserved_bytes: usize,
    /// Retained slot bytes; excludes the shared history header and separate text pool.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub history_occupied_bytes: usize,
    /// Pool payload bytes occupied in whole reclaimable blocks.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub detail_bytes: usize,
    /// Actual payload capacity, rounded down to whole detail blocks.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub detail_capacity_bytes: usize,
    /// Complete detail-pool backing: payload capacity plus all block metadata.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub detail_reserved_bytes: usize,
    /// Reserved metadata bytes for all detail blocks, occupied or free.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub detail_metadata_bytes: usize,
    /// Retained UTF-8 message and correlation bytes, excluding block slack.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub detail_logical_bytes: usize,
    /// Unused payload bytes inside occupied blocks; excludes entirely free blocks.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub detail_slack_bytes: usize,
    /// Admitted key, ownership, and extension metadata content bytes.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub metadata_bytes: usize,
    /// Reserved rolling outcome-bucket slots belonging to components.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub window_buckets: usize,
    /// Admitted extension descriptors, including retained closed components.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub extensions: usize,
    /// Query leases currently holding preallocated materialization buffers.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub query_workspaces: usize,
    /// Optional projection subscriptions currently admitted.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub projections: usize,
    /// Projection workspaces currently materializing or delivering evidence.
    #[serde(with = "native_number")]
    #[cfg_attr(feature = "schema", schemars(with = "DiagnosticNumber"))]
    pub projection_workspaces: usize,
}

// Preserve the native Rust aggregate API while making every wire quantity lossless.
mod native_number {
    use super::DiagnosticNumber;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
    pub fn serialize<S: Serializer>(value: &usize, s: S) -> Result<S::Ok, S::Error> {
        DiagnosticNumber::new(u64::try_from(*value).map_err(serde::ser::Error::custom)?)
            .serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
        usize::try_from(DiagnosticNumber::deserialize(d)?.get()).map_err(de::Error::custom)
    }
}

/// Occupied backing for one retained observation; these quantities overlap.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EntryMemoryFootprint {
    /// Fixed event slot, excluding text.
    pub header_bytes: DiagnosticNumber,
    /// Whole payload blocks occupied by the single packed text chain.
    pub detail_payload_bytes: DiagnosticNumber,
    /// Metadata for those occupied blocks.
    pub detail_metadata_bytes: DiagnosticNumber,
    /// Retained UTF-8 content, including correlations, without exposing that text.
    pub logical_text_bytes: DiagnosticNumber,
    /// Payload slack within occupied blocks.
    pub detail_slack_bytes: DiagnosticNumber,
    /// Header plus detail payload plus detail metadata.
    pub occupied_bytes: DiagnosticNumber,
}
/// Assigned diagnostic storage for one component incarnation, not its heap or RSS.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ComponentMemoryFootprint {
    /// Occupied registration element, including inline text and padding.
    pub registration_bytes: DiagnosticNumber,
    /// Assigned code-counter elements, including unobserved codes.
    pub counter_bytes: DiagnosticNumber,
    /// Assigned window elements, including unused buckets.
    pub window_bytes: DiagnosticNumber,
    /// Assigned extension descriptors.
    pub extension_bytes: DiagnosticNumber,
    /// Retained observations owned by this registration.
    pub retained_event_count: DiagnosticNumber,
    /// Sum of their complete entry footprints.
    pub retained_event_bytes: DiagnosticNumber,
    /// Sum of assigned backing and retained-event bytes.
    pub occupied_bytes: DiagnosticNumber,
}
/// One retained entry, identified within the enclosing source incarnation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EntryMemoryUsage {
    /// Acceptance sequence, distinct from the capture revision.
    pub sequence: DiagnosticNumber,
    /// Original owning registration; refresh never changes this identity.
    pub component: ComponentIdentity,
    /// Compiled backing footprint; not encoded JSON size.
    #[serde(flatten)]
    pub footprint: EntryMemoryFootprint,
}
/// One retained registration's diagnostic backing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ComponentMemoryUsage {
    /// Registration identity; replacement instances remain distinct.
    pub component: ComponentIdentity,
    /// Whole-registration footprint, unaffected by event filters.
    #[serde(flatten)]
    pub footprint: ComponentMemoryFootprint,
}
/// Whole-selection totals, distinct from the rows present in this page.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemorySelectionTotals {
    /// Matching component registrations across all pages.
    pub components: DiagnosticNumber,
    /// Matching retained entries across all pages.
    pub events: DiagnosticNumber,
    /// Whole-registration backing for matching components.
    pub component_occupied_bytes: DiagnosticNumber,
    /// Backing for matching retained entries (overlaps component totals).
    pub entry_occupied_bytes: DiagnosticNumber,
}
/// Bounded independent memory view; inspection neither records nor freezes history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticMemoryPage {
    /// Separate envelope version, unrelated to snapshot/catalogue versions.
    #[serde(deserialize_with = "memory_version")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "memory_version_schema"))]
    pub schema_version: u32,
    /// Diagnostic source incarnation.
    pub source: SourceIdentity,
    /// Diagnostic capture time, independent of resource collection.
    pub captured_at: ObservationTime,
    /// Retained-state revision used to validate continuation.
    pub sequence: DiagnosticNumber,
    /// Whole-context backing and occupancy, regardless of selection.
    pub aggregate: DiagnosticMemoryUsage,
    /// Complete component attribution, including omitted rows.
    pub component_occupied_bytes: DiagnosticNumber,
    /// Reserved minus complete component attribution; includes free and shared backing.
    pub shared_or_unassigned_reserved_bytes: DiagnosticNumber,
    /// Whole-context retention/admission loss, distinct from omitted page rows.
    pub loss: super::DiagnosticLoss,
    /// Whole-selection totals; byte quantities overlap rather than add.
    pub selection: MemorySelectionTotals,
    /// Totals for the rows in this response only.
    pub page: MemorySelectionTotals,
    /// Bounded selected registration rows.
    pub components: Vec<ComponentMemoryUsage>,
    /// Bounded selected retained entry rows, without detail/correlation text.
    pub entries: Vec<EntryMemoryUsage>,
    /// Independently copied resource context; its manager cache is outside core reservation.
    pub resource: super::ResourceMemoryView,
    /// Mutation-sensitive continuation; snapshot cursors are rejected.
    pub continuation: Option<DiagnosticCursor>,
}
fn memory_version<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u32, D::Error> {
    let version = u32::deserialize(d)?;
    if version != DIAGNOSTIC_MEMORY_VERSION {
        return Err(serde::de::Error::custom(
            "unsupported diagnostic memory schema",
        ));
    }
    Ok(version)
}

#[cfg(feature = "schema")]
fn memory_version_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    let mut schema: schemars::schema::SchemaObject =
        <u32 as schemars::JsonSchema>::json_schema(generator).into();
    schema.enum_values = Some(vec![serde_json::json!(DIAGNOSTIC_MEMORY_VERSION)]);
    schema.into()
}
