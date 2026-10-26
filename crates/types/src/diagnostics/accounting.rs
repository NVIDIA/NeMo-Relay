// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed summaries, code counters, and bounded outcome windows.

use super::{
    ComponentOwnership, DiagnosticAvailability, DiagnosticNumber, DiagnosticOutcome,
    ObservationTime,
};
use serde::{Deserialize, Serialize};

/// Magnitude of an occurrence or loss counter, including overflow visibility.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCount {
    /// Exact count unless saturation has occurred.
    pub value: DiagnosticNumber,
    /// Whether an overflowing increment made the value a lower bound.
    pub saturated: bool,
}

impl DiagnosticCount {
    /// Add occurrences, preserving overflow as an explicit lower-bound marker.
    pub fn add(&mut self, amount: u64) {
        let (value, overflowed) = self.value.get().overflowing_add(amount);
        if overflowed {
            self.value = DiagnosticNumber::new(u64::MAX);
            self.saturated = true;
        } else {
            self.value = DiagnosticNumber::new(value);
        }
    }
}

/// Fixed numeric accounting for one explicitly identified lifetime or refresh scope.
///
/// Recent-evidence loss cannot roll back accepted counts. Saturated counts remain
/// explicit lower bounds. Code and event filters never redefine the summary scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticTotals {
    /// Accepted recording calls, one per call regardless of occurrence delta.
    pub recorded_events_total: DiagnosticCount,
    /// Accepted occurrence deltas across all outcomes and codes.
    pub occurrences_total: DiagnosticCount,
    /// Previously retained events owned by this component or context and evicted.
    pub history_evictions_total: DiagnosticCount,
    /// Occurrences whose owning operation explicitly succeeded.
    pub successes_total: DiagnosticCount,
    /// Occurrences whose owning operation explicitly failed.
    pub failures_total: DiagnosticCount,
    /// Occurrences without an established operation outcome.
    pub unspecified_total: DiagnosticCount,
}

impl DiagnosticTotals {
    /// Add one accepted recording call and its observed occurrence delta.
    ///
    /// The recording boundary validates the positive delta before updating every
    /// applicable scope. This helper does not perform admission or retain evidence.
    pub fn record(&mut self, outcome: DiagnosticOutcome, count: u64) {
        self.recorded_events_total.add(1);
        self.occurrences_total.add(count);
        match outcome {
            DiagnosticOutcome::Success => self.successes_total.add(count),
            DiagnosticOutcome::Failure => self.failures_total.add(count),
            DiagnosticOutcome::Unknown => self.unspecified_total.add(count),
        }
    }
}

/// Fixed lifetime totals and monotonic timing for a context or component instance.
///
/// Context totals survive component refresh, replacement, and metadata reclamation.
/// Component totals survive refresh for the same registration instance. Neither is
/// reconstructed from retained events or reset by a filtered read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticLifetimeSummary {
    /// Whole-context or whole-component-instance accounting, as named by the view.
    pub totals: DiagnosticTotals,
    /// Monotonic elapsed lifetime measured by the source at capture.
    pub elapsed_millis: DiagnosticNumber,
    /// Wall-clock correlation time of the latest observed lifetime failure.
    pub last_failure_at: Option<ObservationTime>,
    /// Monotonic elapsed time since that failure; absence never means recovery.
    pub since_last_failure_millis: Option<DiagnosticNumber>,
}

/// One current resettable counting period, with no retained historical summary.
///
/// The generation marks a refresh boundary within a component instance. Ownership
/// describes this current period; it does not relabel the component lifetime totals.
/// Totals count observations recorded after the refresh boundary, including work
/// carrying an older captured origin, and evictions performed after the boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticRefreshSummary {
    /// Component-local generation increased by a successful refresh.
    pub generation: DiagnosticNumber,
    /// Wall-clock correlation anchor for the successful refresh.
    pub started_at: ObservationTime,
    /// Monotonic elapsed duration since the refresh boundary.
    pub elapsed_millis: DiagnosticNumber,
    /// Current configuration and owner attribution for this counting period.
    pub ownership: ComponentOwnership,
    /// Counts for this current period, independently reserved from lifetime totals.
    pub totals: DiagnosticTotals,
}

/// Lifetime occurrences for one code in its containing component instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCounter {
    /// Canonical code; the component instance comes from the containing row.
    pub code: String,
    /// Accepted occurrences, independent of optional recent detail.
    pub occurrences: DiagnosticCount,
    /// Source time when this retained counter was last updated.
    pub last_observed_at: ObservationTime,
}

/// An aggregate bucket using component-relative elapsed time, never raw process ticks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticWindowBucket {
    /// Inclusive elapsed offset from the component's monotonic start.
    pub start_elapsed_millis: DiagnosticNumber,
    /// Exclusive nominal bucket end; coverage describes the observed portion.
    pub end_elapsed_millis: DiagnosticNumber,
    /// Explicit successful-operation occurrences observed in this interval.
    pub successes: DiagnosticCount,
    /// Explicit failed-operation occurrences observed in this interval.
    pub failures: DiagnosticCount,
    /// Occurrences with no established operation outcome in this interval.
    pub unspecified: DiagnosticCount,
}

/// Actual observation interval and limitations of a fixed-bucket window.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticWindowCoverage {
    /// Availability of the admitted window series.
    pub availability: DiagnosticAvailability,
    /// Earliest covered monotonic elapsed offset within this component lifetime.
    pub start_elapsed_millis: DiagnosticNumber,
    /// Latest covered monotonic elapsed offset at capture time.
    pub end_elapsed_millis: DiagnosticNumber,
    /// The first returned bucket is only partly inside the covered interval.
    pub partial_start: bool,
    /// The last returned bucket is still being observed or partly selected.
    pub partial_end: bool,
    /// Known loss prevents treating the aggregate as a complete interval.
    pub incomplete: bool,
}

/// Fixed-resolution outcome series for the containing component instance.
///
/// Batched deltas belong to their observation interval. Edge buckets cannot
/// establish an exact arbitrary rolling interval or reconstruct occurrence times.
/// Refreshing the component does not replace or duplicate this window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticWindow {
    /// Duration of each fixed bucket.
    pub resolution_millis: DiagnosticNumber,
    /// Configured bucket slots; memory is independent of observation rate.
    pub bucket_capacity: u32,
    /// Bounded selected aggregates; idle covered intervals may have zero values.
    pub buckets: Vec<DiagnosticWindowBucket>,
    /// Actual interval and edge limitations.
    pub coverage: DiagnosticWindowCoverage,
}
