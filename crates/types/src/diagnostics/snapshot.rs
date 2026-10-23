// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Component rows and the versioned diagnostic envelope.

use super::{
    ComponentIdentity, DIAGNOSTIC_SCHEMA_VERSION, DiagnosticCounter, DiagnosticCoverage,
    DiagnosticCursor, DiagnosticEvent, DiagnosticLifetimeSummary, DiagnosticNumber,
    DiagnosticRefreshSummary, DiagnosticWindow, ObservationTime, SourceIdentity,
};
use serde::{Deserialize, Deserializer, Serialize, de};

/// One retained row per component registration instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ComponentSnapshot {
    /// Logical responsibility and registration incarnation.
    pub identity: ComponentIdentity,
    /// Source correlation time when the registration was admitted.
    pub registered_at: ObservationTime,
    /// Registration closure; retained summaries remain subject to bounded retention.
    pub closed_at: Option<ObservationTime>,
    /// Whole-component-instance lifetime accounting and source elapsed time.
    pub lifetime: DiagnosticLifetimeSummary,
    /// The sole current refresh period, with explicitly scoped current ownership.
    pub since_refresh: DiagnosticRefreshSummary,
    /// Selected lifetime code counters; no latest-message set is retained per code.
    pub counters: Vec<DiagnosticCounter>,
    /// Admitted bounded component outcome windows, independent of refresh periods.
    pub windows: Vec<DiagnosticWindow>,
}

/// Read evidence from one source incarnation, using the canonical snapshot schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticSnapshot {
    /// Supported envelope version, independent of the catalogue version.
    #[serde(deserialize_with = "deserialize_schema_version")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "schema_version_schema"))]
    pub schema_version: u32,
    /// Stable catalogue revision; unknown names stay opaque in this envelope.
    pub catalogue_version: u32,
    /// Source identity preserved through collection and projection.
    pub source: SourceIdentity,
    /// Source capture time; it is not a collector receipt timestamp.
    pub captured_at: ObservationTime,
    /// Latest source update revision; reads do not increment it.
    pub sequence: DiagnosticNumber,
    /// Whole-context lifetime accounting, independent of component/event selectors.
    /// Hosts must authorize access to this context-wide view before exposing it.
    pub lifetime: DiagnosticLifetimeSummary,
    /// Bounded rows, one per selected retained component instance.
    pub components: Vec<ComponentSnapshot>,
    /// Bounded selected recent evidence.
    pub events: Vec<DiagnosticEvent>,
    /// Availability, scope, integration, and loss limitations.
    pub coverage: DiagnosticCoverage,
    /// Next bounded page, subject to mutation and retention validation.
    pub continuation: Option<DiagnosticCursor>,
}

fn deserialize_schema_version<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    let version = u32::deserialize(deserializer)?;
    if version != DIAGNOSTIC_SCHEMA_VERSION {
        return Err(de::Error::custom("unsupported diagnostic snapshot schema"));
    }
    Ok(version)
}

#[cfg(feature = "schema")]
fn schema_version_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    let mut schema: schemars::schema::SchemaObject =
        <u32 as schemars::JsonSchema>::json_schema(generator).into();
    schema.enum_values = Some(vec![serde_json::json!(DIAGNOSTIC_SCHEMA_VERSION)]);
    schema.into()
}
