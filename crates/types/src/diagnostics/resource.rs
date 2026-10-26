// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded resource-memory context. No diagnostic content enters resource exporters.
use super::{DiagnosticNumber, ObservationTime};
use crate::api::resource_metrics::{
    CapacityUnit, CountUnit, DurationUnit, ResourceLimitEventKind, ResourceLimitResource,
    ResourceMeasurementScope, ResourceOperatingSystem,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Availability of the latest-memory projection, independent of diagnostic reads.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ResourceMemoryAvailability {
    /// Collection or projection is disabled.
    Disabled,
    /// Waiting for a CLI-owned target.
    AwaitingTarget,
    /// No collection has completed.
    NotYetSampled,
    /// A sample is available; inspect its independent staleness flag.
    Available,
    /// Latest collection failed; a prior valid sample may remain.
    Failed,
    /// Runtime/target or process owner was invalidated.
    Invalidated,
    /// A sample exceeds the bounded projection vocabulary or capacity.
    CapacityUnavailable,
    /// The collector platform is unsupported.
    Unsupported,
}
/// Independent baseline series producing this sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ResourceSampleStream {
    /// Configured polling.
    Polling,
    /// Explicit existing resource collect API.
    OnDemand,
}
/// Relationship between the diagnostic process and measured target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ResourceAssociation {
    /// Verified host application process.
    HostProcess,
    /// Verified host root plus sampled descendants.
    HostProcessTree,
    /// Verified different root and descendants; may exclude Relay.
    OtherProcessTree,
    /// System context only.
    System,
    /// No verified root relationship.
    Unverified,
}
/// Field-level scope; snapshot scope does not relabel system/environment data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MemoryFieldScope {
    /// System-visible measurement.
    System,
    /// Selected process or process tree.
    SelectedProcesses,
    /// Owned environment/container/cgroup measurement.
    Environment,
}
/// Exact integers and finite converted decimals retain distinct numeric kinds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ResourceMemoryNumber {
    /// Exact integral value in its declared unit.
    Integer(DiagnosticNumber),
    /// Converted fractional value with the existing resource API's floating precision.
    Decimal(f64),
    /// A collector returned a non-finite number; never substitute zero.
    NonFinite,
}
/// Original unit family and canonical unit. Inspection performs no conversion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "dimension", content = "name", rename_all = "snake_case")]
pub enum ResourceMemoryUnit {
    /// Memory capacity.
    Capacity(#[cfg_attr(feature = "schema", schemars(with = "String"))] CapacityUnit),
    /// Cumulative pressure duration.
    Duration(#[cfg_attr(feature = "schema", schemars(with = "String"))] DurationUnit),
    /// Cumulative events.
    Count(#[cfg_attr(feature = "schema", schemars(with = "String"))] CountUnit),
}
/// One available measurement; an absent field means not reported (unsupported or unreadable).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResourceMemoryMeasurement {
    /// Original precision and numeric kind.
    pub value: ResourceMemoryNumber,
    /// Original configured unit; not necessarily bytes.
    pub unit: ResourceMemoryUnit,
    /// Scope of this specific field.
    pub measurement_scope: MemoryFieldScope,
    /// Readable process count for this field, if the collector reports it.
    pub sampled_processes: Option<DiagnosticNumber>,
}
/// One cumulative environment limit event.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryLimitEvent {
    /// Resource whose limit generated the event.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub resource: ResourceLimitResource,
    /// OS event kind.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub event: ResourceLimitEventKind,
    /// Cumulative count or absent coverage.
    pub count: Option<ResourceMemoryMeasurement>,
}
/// Memory-only projection; fixed fields and eight limit slots bound all copies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResourceMemoryMetrics {
    /// Original `system_used` resource measurement; absent means not reported.
    pub system_used: Option<ResourceMemoryMeasurement>,
    /// Original `system_total` resource measurement; absent means not reported.
    pub system_total: Option<ResourceMemoryMeasurement>,
    /// Original `system_available` resource measurement; absent means not reported.
    pub system_available: Option<ResourceMemoryMeasurement>,
    /// Original `resident` resource measurement; absent means not reported.
    pub resident: Option<ResourceMemoryMeasurement>,
    /// Original `private` resource measurement; absent means not reported.
    pub private: Option<ResourceMemoryMeasurement>,
    /// Original `physical_footprint` resource measurement; absent means not reported.
    pub physical_footprint: Option<ResourceMemoryMeasurement>,
    /// Original `virtual_memory` resource measurement; absent means not reported.
    pub virtual_memory: Option<ResourceMemoryMeasurement>,
    /// Original `peak_resident` resource measurement; absent means not reported.
    pub peak_resident: Option<ResourceMemoryMeasurement>,
    /// Original `limit` resource measurement; absent means not reported.
    pub limit: Option<ResourceMemoryMeasurement>,
    /// Original `environment_accounted` resource measurement; absent means not reported.
    pub environment_accounted: Option<ResourceMemoryMeasurement>,
    /// Original `some_pressure_stall_time` resource measurement; absent means not reported.
    pub some_pressure_stall_time: Option<ResourceMemoryMeasurement>,
    /// Original `full_pressure_stall_time` resource measurement; absent means not reported.
    pub full_pressure_stall_time: Option<ResourceMemoryMeasurement>,
    /// Original `out_of_memory_event_count` resource measurement; absent means not reported.
    pub out_of_memory_event_count: Option<ResourceMemoryMeasurement>,
    /// Retained limit events; unused slots are null, overcapacity rejects publication.
    pub limit_events: [Option<MemoryLimitEvent>; 8],
}
/// Verified root incarnation from the same collection, never PID alone.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResourceProcessIdentity {
    /// OS process identifier.
    pub process_id: u32,
    /// Opaque platform start identity; compare only within the same host/collector.
    pub start_identity: DiagnosticNumber,
}
/// Collection identity and memory values, copied without an unbounded retained graph.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResourceMemorySample {
    /// Collector incarnation; multiple diagnostic sources may share this sample.
    pub collector_instance_id: Uuid,
    /// Active plugin runtime generation.
    pub runtime_generation: DiagnosticNumber,
    /// Collection target generation.
    pub target_generation: DiagnosticNumber,
    /// Independent sampling baseline stream.
    pub stream: ResourceSampleStream,
    /// Shared collection-start order across both streams.
    pub ordinal: DiagnosticNumber,
    /// Collection start on the collector's wall clock.
    pub started_at: ObservationTime,
    /// Collection completion on the collector's wall clock.
    pub completed_at: ObservationTime,
    /// Collection duration measured monotonically.
    pub collection_millis: DiagnosticNumber,
    /// Collector operating system.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub operating_system: ResourceOperatingSystem,
    /// Selected measurement boundary, independent of individual system fields.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub measurement_scope: ResourceMeasurementScope,
    /// Root verified during this collection, or absent for system/unverified context.
    pub root: Option<ResourceProcessIdentity>,
    /// Processes visible at sampling start, when known.
    pub visible_processes: Option<DiagnosticNumber>,
    /// Successfully sampled base process queries, when known.
    pub sampled_processes: Option<DiagnosticNumber>,
    /// Memory-only values; existing resource APIs retain their serialization.
    pub memory: ResourceMemoryMetrics,
}
/// Independent resource result joined after diagnostic capture releases its lock.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResourceMemoryView {
    /// Latest attempt availability; prior valid samples may remain after failure.
    pub availability: ResourceMemoryAvailability,
    /// Exact sample used for this page; may differ on the next page.
    pub sample: Option<ResourceMemorySample>,
    /// Relation to the diagnostic source, verified without retargeting collection.
    pub association: ResourceAssociation,
    /// Local monotonic age since completion, not a remote wall-clock subtraction.
    pub age_millis: Option<DiagnosticNumber>,
    /// Age exceeds the resource plugin's configured freshness threshold.
    pub stale: bool,
    /// Compiled latest-cache slot size, outside core reservation and excluding temporary copies.
    pub projection_slot_bytes: DiagnosticNumber,
}
impl ResourceMemoryView {
    /// Inactive resource collection; constructing this value never initializes it.
    pub fn disabled() -> Self {
        Self {
            availability: ResourceMemoryAvailability::Disabled,
            sample: None,
            association: ResourceAssociation::Unverified,
            age_millis: None,
            stale: false,
            projection_slot_bytes: DiagnosticNumber::new(0),
        }
    }
}
