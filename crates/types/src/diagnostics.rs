// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable diagnostic evidence shared by Relay runtimes, tools, and adapters.
//!
//! These types describe observations rather than readiness or deployment health.
//! They contain no runtime state. Producers enforce resource limits before
//! recording; consumers inspect coverage before interpreting absent evidence.
//!
//! Potentially growing integers use DiagnosticNumber, encoded as canonical
//! unsigned decimal JSON strings to preserve every u64 value in JavaScript.
//! Numbers, signs, leading zeroes, fractions, whitespace, and values outside
//! u64 are rejected. Unknown object fields support additive evolution. Unknown
//! catalogue names remain opaque in supported envelopes; unsupported schema
//! versions require adapter negotiation.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use uuid::Uuid;

/// Initial version of the canonical diagnostic snapshot envelope.
pub const DIAGNOSTIC_SCHEMA_VERSION: u32 = 1;

/// Initial stable typed-catalogue revision, independent of the envelope version.
pub const DIAGNOSTIC_CATALOGUE_VERSION: u32 = 1;

mod catalogue;
pub use catalogue::*;

/// UTC wall-clock time when the source observed evidence or captured a snapshot.
///
/// This timestamp supports correlation, not elapsed time across machines.
/// Collectors track receipt time with their own clocks.
pub type ObservationTime = DateTime<Utc>;

/// Unsigned diagnostic integer serialized losslessly as a decimal string.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiagnosticNumber(u64);

impl DiagnosticNumber {
    /// Construct a diagnostic integer without losing precision.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the complete Rust integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for DiagnosticNumber {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<DiagnosticNumber> for u64 {
    fn from(value: DiagnosticNumber) -> Self {
        value.0
    }
}

impl fmt::Display for DiagnosticNumber {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Serialize for DiagnosticNumber {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for DiagnosticNumber {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        let canonical = value == "0"
            || (value.len() <= 20
                && value
                    .as_bytes()
                    .first()
                    .is_some_and(|byte| matches!(byte, b'1'..=b'9'))
                && value.bytes().all(|byte| byte.is_ascii_digit()));
        if !canonical {
            return Err(de::Error::custom(
                "expected a canonical unsigned decimal u64 string",
            ));
        }
        value.parse::<u64>().map(Self).map_err(de::Error::custom)
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for DiagnosticNumber {
    fn schema_name() -> String {
        "DiagnosticNumber".into()
    }

    fn json_schema(_generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        use schemars::schema::{InstanceType, Schema, SchemaObject, StringValidation};

        // Every 20-digit alternative fixes the maximum prefix and lowers one
        // digit; shorter canonical integers are always within u64's range.
        let maximum = u64::MAX.to_string();
        let mut alternatives = vec!["0".to_owned(), "[1-9][0-9]{0,18}".to_owned()];
        for (index, digit) in maximum.bytes().enumerate() {
            let first = if index == 0 { b'1' } else { b'0' };
            if digit > first {
                alternatives.push(format!(
                    "{}[{}-{}][0-9]{{{}}}",
                    &maximum[..index],
                    char::from(first),
                    char::from(digit - 1),
                    maximum.len() - index - 1,
                ));
            }
        }
        alternatives.push(maximum);

        Schema::Object(SchemaObject {
            instance_type: Some(InstanceType::String.into()),
            string: Some(Box::new(StringValidation {
                max_length: Some(20),
                min_length: Some(1),
                pattern: Some(format!("^({})$", alternatives.join("|"))),
            })),
            ..Default::default()
        })
    }
}

/// Identity of one Relay process or explicitly owned runtime incarnation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SourceIdentity {
    /// Unique incarnation assigned before logging initialization.
    pub instance_id: Uuid,
    /// Descriptive operating system process ID, never an identity key.
    pub process_id: Option<u32>,
    /// Producer-defined role, such as embedded runtime, gateway, broker, or worker.
    pub process_role: String,
    /// Relay package version producing the evidence.
    pub relay_version: String,
}

/// Identity of one registered component within a diagnostic source.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ComponentIdentity {
    /// Stable logical name, excluding request and session identifiers.
    pub key: String,
    /// Unique registration instance, distinguishing replacement components.
    pub instance_id: Uuid,
    /// Source-local registration generation, increasing when a logical key is replaced.
    pub generation: DiagnosticNumber,
}

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

/// Severity of evidence, independent of an operation's outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    /// Detailed instrumentation useful for tracing execution.
    Trace,
    /// Developer-oriented evidence.
    Debug,
    /// Informational observation.
    Info,
    /// An unexpected observation.
    Warning,
    /// A failed operation.
    Error,
}

/// Observed operation outcome; severity does not imply this value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticOutcome {
    /// No operation outcome was observed.
    #[default]
    Unknown,
    /// The observed operation succeeded.
    Success,
    /// The observed operation failed.
    Failure,
}

/// Optional safe correlation identifiers, excluded from aggregation keys.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCorrelation {
    /// Trajectory session associated with the observation, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Request associated with the observation, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Bounded attribution for a component refresh or an originating observation.
///
/// Core bounds strings on admission. Nonempty fields also act as an intersection
/// selector in a query; request and session IDs belong in correlation instead.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ComponentOwnership {
    /// Subscriber registration associated with this attribution, when applicable.
    pub subscriber_instance_id: Option<Uuid>,
    /// Plugin registration associated with this attribution, when applicable.
    pub plugin_instance_id: Option<Uuid>,
    /// Bounded identity of the committed configuration.
    pub configuration_id: Option<String>,
    /// Bounded runtime responsibility for an embedded or process owner.
    pub responsibility: Option<String>,
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

/// Original bounded configuration attribution carried by one observation.
///
/// Captured origin metadata can survive a refresh without retaining a historical
/// counter or window store. It is not a separately queryable aggregate identity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticOrigin {
    /// Refresh generation associated with the originating work or observation.
    pub refresh_generation: DiagnosticNumber,
    /// Original bounded owner and configuration attribution.
    pub ownership: ComponentOwnership,
}

/// Cause metadata resolved from registered built-in or extension descriptors.
///
/// Unknown future names remain opaque strings in a supported schema. Core and
/// transport adapters enforce their explicit byte limits before materialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ClassificationWire {
    /// Bounded stable built-in name or registered namespaced extension name.
    pub canonical_code: String,
    /// Bounded cause name; absent for successful observations and notices.
    pub failure_class: Option<String>,
}

impl From<Classification> for ClassificationWire {
    fn from(value: Classification) -> Self {
        Self {
            canonical_code: value.code().canonical_name().into(),
            failure_class: value.failure_class().map(|class| class.as_str().into()),
        }
    }
}

/// Safe optional recent evidence, independent of the accepted numeric summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticDetail {
    /// Producer-approved text, copied only within admitted byte capacity.
    pub message: String,
    /// Whether the original text exceeded its retained byte allowance.
    pub truncated: bool,
}

/// Fixed-size facts about an observed plugin operation, never counter dimensions.
///
/// The producer supplies these only from known execution and lifecycle evidence.
/// An absent value remains unknown; mechanism is never inferred from a code name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PluginObservationContext {
    /// Observed native ABI or dynamic-worker execution mechanism, when known.
    pub execution: Option<PluginExecution>,
    /// Observed validation, activation, runtime, or policy lifecycle phase.
    pub phase: Option<PluginPhase>,
}

/// One recent observation with its actual class and original configuration attribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticEvent {
    /// Unique source-local update sequence; it is not an occurrence counter.
    pub sequence: DiagnosticNumber,
    /// Source observation time, preserved through projections.
    pub observed_at: ObservationTime,
    /// Registration that owned the original observation.
    pub component: ComponentIdentity,
    /// Original configuration attribution, preserved through component refresh.
    pub origin: DiagnosticOrigin,
    /// Canonical event classification; never reconstructed from a latest message.
    pub classification: ClassificationWire,
    /// Bounded typed plugin-operation facts, excluded from aggregation keys.
    #[serde(default)]
    pub context: PluginObservationContext,
    /// Evidence severity, independent of its observed operation outcome.
    pub severity: DiagnosticSeverity,
    /// Explicit observed operation outcome.
    pub outcome: DiagnosticOutcome,
    /// Bounded optional evidence; absence does not imply a missing counter.
    pub detail: Option<DiagnosticDetail>,
    /// Input occurrence delta represented by this recording call.
    pub count: DiagnosticNumber,
    /// Safe correlation, excluded from aggregation identity.
    pub correlation: DiagnosticCorrelation,
}

impl DiagnosticEvent {
    /// Borrow the canonical code without allocating a second representation.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.classification.canonical_code
    }
    /// Borrow retained detail, when available.
    #[must_use]
    pub fn message(&self) -> Option<&str> {
        self.detail.as_ref().map(|detail| detail.message.as_str())
    }
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
pub struct DiagnosticQuery {
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
    /// Select the plugin mechanism derived from catalogue entries.
    pub plugin_family: Option<PluginDiagnosticFamily>,
    /// Select event causes only; code counters do not retain cause breakdowns.
    pub failure_class: Option<String>,
    /// Select component rows by current-refresh ownership and events by original ownership.
    /// Selected lifetime counters still span the entire component instance.
    pub ownership: ComponentOwnership,
    /// Select later recent events; summary and code counters remain cumulative.
    pub after_sequence: Option<DiagnosticNumber>,
    /// Requested component-instance row cap, further constrained by core limits.
    pub max_components: Option<u32>,
    /// Requested recent-event cap, further constrained by core limits.
    pub max_events: Option<u32>,
    /// Requested encoded response cap, further constrained by core limits.
    pub max_response_bytes: Option<u32>,
    /// Resume a compatible bounded selection without granting additional scope.
    pub continuation: Option<DiagnosticCursor>,
}

/// Read evidence from one source incarnation, using the canonical snapshot schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticSnapshot {
    /// Supported envelope version, independent of the catalogue version.
    #[serde(deserialize_with = "deserialize_schema_version")]
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
