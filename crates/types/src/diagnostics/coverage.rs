// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Availability, retention, and loss coverage.

use super::{DiagnosticCount, DiagnosticNumber};
use serde::{Deserialize, Serialize};

/// Availability of evidence, without applying deployment or application health.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticAvailability {
    /// Evidence was supplied; other coverage fields still describe completeness.
    #[default]
    Available,
    /// The supported evidence could not currently be obtained.
    Unavailable,
    /// The source does not implement or integrate this capability.
    Unsupported,
    /// No observation has established a value in this retained scope.
    NeverObserved,
    /// The owner intentionally disabled the complete new diagnostic service.
    Disabled,
}

/// Capability coverage, including explicitly deferred producer integrations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCapability {
    /// Bounded capability name, such as `otel`.
    pub name: String,
    /// Availability of evidence from this capability.
    pub availability: DiagnosticAvailability,
    /// Bounded explanatory reason, such as `not_integrated`.
    pub reason: Option<String>,
}

/// Boundaries of returned retained evidence, separate from cumulative counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticRetention {
    /// Oldest sequence in the selected retained event history, if present.
    pub oldest_event_sequence: Option<DiagnosticNumber>,
    /// Latest sequence in the selected retained event history, if present.
    pub latest_event_sequence: Option<DiagnosticNumber>,
    /// Number of returned recent events.
    pub events_retained: DiagnosticNumber,
    /// Number of returned component-instance rows.
    pub components_retained: DiagnosticNumber,
    /// Query limits omitted otherwise matching retained evidence.
    pub query_truncated: bool,
}

/// Source loss state reserved independently of optional evidence capacity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticLoss {
    /// Previously retained events reclaimed from the circular history.
    pub events_evicted: DiagnosticCount,
    /// Accepted observations whose incoming event evidence was not retained.
    pub events_omitted: DiagnosticCount,
    /// Closed component metadata reclaimed with dependent evidence.
    pub components_evicted: DiagnosticCount,
    /// Component registrations rejected by admission limits.
    pub components_rejected: DiagnosticCount,
    /// Code reservations or registrations rejected by admission limits.
    pub keys_rejected: DiagnosticCount,
    /// Observations rejected before canonical counting.
    pub observations_rejected: DiagnosticCount,
    /// Detail shortened within its configured byte allowance.
    pub messages_truncated: DiagnosticCount,
    /// Incoming optional detail omitted without evicting a retained event.
    pub details_omitted: DiagnosticCount,
    /// Correlation fields shortened within their own byte allowance.
    pub correlations_truncated: DiagnosticCount,
    /// Safe correlation fields omitted because detail storage could not fit them.
    pub correlations_omitted: DiagnosticCount,
    /// Metadata registrations rejected by their admission budget.
    pub metadata_rejected: DiagnosticCount,
    /// Accepted summary-only deltas whose classification could not be admitted.
    pub classifications_overflowed: DiagnosticCount,
    /// Window series rejected by their independent admission budget.
    pub windows_rejected: DiagnosticCount,
    /// Query materializations rejected by bounded workspace admission.
    pub queries_rejected: DiagnosticCount,
    /// Recording attempts for closed or replaced component registrations.
    pub stale_observations: DiagnosticCount,
    /// Optional projection deliveries omitted for lack of bounded workspace.
    pub projection_deliveries_omitted: DiagnosticCount,
    /// Projection callbacks whose panic was isolated after acceptance.
    pub projections_panicked: DiagnosticCount,
    /// Cyclic or excessively deep projection chains suppressed.
    pub projections_suppressed: DiagnosticCount,
    /// Unique sequence allocation has exhausted its representation.
    pub sequence_saturated: bool,
}

/// Scope of summary and outcome-window values in a filtered response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSummaryScope {
    /// Lifetime totals cover the whole selected component instance, not matching events.
    /// The separate since-refresh summary names its current period explicitly.
    #[default]
    WholeComponentLifetime,
}

/// Explicit evidence, retention, saturation, and producer integration coverage.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCoverage {
    /// Availability of this diagnostic source context.
    pub availability: DiagnosticAvailability,
    /// Scope of component summaries and component outcome windows.
    pub summary_scope: DiagnosticSummaryScope,
    /// Bounded selected history boundaries and query omission.
    pub retention: DiagnosticRetention,
    /// Reserved cumulative loss and capacity accounting.
    pub loss: DiagnosticLoss,
    /// Producer integrations whose support or absence is explicitly advertised.
    pub capabilities: Vec<DiagnosticCapability>,
}
