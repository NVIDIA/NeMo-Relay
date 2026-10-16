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
//! u64 are rejected. Unknown object fields support additive evolution; closed
//! enums and unsupported schema versions require adapter negotiation.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use uuid::Uuid;

/// Version of the diagnostic snapshot contract defined in this module.
pub const DIAGNOSTIC_SCHEMA_VERSION: u32 = 1;

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
                ..Default::default()
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

/// An observation retained in the source's bounded recent-event history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticEvent {
    /// Source-local sequence; inspect coverage for saturation.
    pub sequence: DiagnosticNumber,
    /// Source observation time, preserved through projections.
    pub observed_at: ObservationTime,
    /// Component that owns the evidence.
    pub component: ComponentIdentity,
    /// Stable producer-defined classification suitable for machine checks.
    pub code: String,
    /// Severity independent of operation outcome.
    pub severity: DiagnosticSeverity,
    /// Observed operation outcome.
    pub outcome: DiagnosticOutcome,
    /// Bounded, producer-approved human-readable detail.
    pub message: String,
    /// Occurrences represented by this observation.
    pub count: DiagnosticNumber,
    /// Correlation preserved without creating new aggregate identities.
    pub correlation: DiagnosticCorrelation,
}

/// Cumulative occurrences for one registered component and diagnostic code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCounter {
    /// Stable classification counted by this aggregate.
    pub code: String,
    /// Counter epoch; replacing snapshots must not add their cumulative counts.
    pub epoch: Uuid,
    /// Occurrences; reading a snapshot does not increment this value.
    pub occurrences: DiagnosticCount,
    /// Source time of the last counted observation.
    pub last_observed_at: ObservationTime,
    /// Latest counted detail for existing report projections.
    pub last_message: String,
}

/// Retained diagnostic evidence for one registered component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ComponentSnapshot {
    /// Logical name and registration incarnation.
    pub identity: ComponentIdentity,
    /// Source time when this component registered.
    pub registered_at: ObservationTime,
    /// Source time when this component registration closed, if observed.
    pub closed_at: Option<ObservationTime>,
    /// Cumulative occurrence aggregates retained within source limits.
    pub counters: Vec<DiagnosticCounter>,
}

/// Evidence availability, without inferring deployment health.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticAvailability {
    /// Evidence was supplied; coverage still determines completeness.
    #[default]
    Available,
    /// Evidence could not currently be obtained from its source.
    Unavailable,
    /// The source does not implement this capability.
    Unsupported,
    /// No observation has yet established a value.
    NeverObserved,
}

/// Retention boundaries of a returned view, separate from cumulative counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticRetention {
    /// Oldest event sequence in the returned history, if any.
    pub oldest_event_sequence: Option<DiagnosticNumber>,
    /// Latest event sequence in the returned history, if any.
    pub latest_event_sequence: Option<DiagnosticNumber>,
    /// Number of events returned by this view.
    pub events_retained: DiagnosticNumber,
    /// Number of component snapshots returned by this view.
    pub components_retained: DiagnosticNumber,
    /// Whether query bounds omitted otherwise matching retained evidence.
    pub query_truncated: bool,
}

/// Reserved source metadata, writable even when evidence budgets are full.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticLoss {
    /// Events removed from bounded recent history.
    pub events_evicted: DiagnosticCount,
    /// Closed component snapshots removed from retained state.
    pub components_evicted: DiagnosticCount,
    /// Component registrations rejected by resource limits.
    pub components_rejected: DiagnosticCount,
    /// New component/code aggregates rejected by resource limits.
    pub keys_rejected: DiagnosticCount,
    /// Observations rejected before recording.
    pub observations_rejected: DiagnosticCount,
    /// Observation messages shortened to configured size limits.
    pub messages_truncated: DiagnosticCount,
    /// Projection callbacks that panicked after an observation was retained.
    pub projections_panicked: DiagnosticCount,
    /// Nested projections suppressed to prevent a reentrant failure loop.
    pub projections_suppressed: DiagnosticCount,
    /// Whether the source event sequence can no longer increase uniquely.
    pub sequence_saturated: bool,
}

/// Evidence availability and explicit retention or loss limitations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCoverage {
    /// Whether evidence is supported, observed, and obtainable.
    pub availability: DiagnosticAvailability,
    /// Boundaries and query omissions of the returned view.
    pub retention: DiagnosticRetention,
    /// Cumulative source loss and saturation metadata.
    pub loss: DiagnosticLoss,
}

/// Read-only evidence from one diagnostic source incarnation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticSnapshot {
    /// Contract version; adapters negotiate unsupported versions.
    pub schema_version: u32,
    /// Source identity preserved through collection and projection.
    pub source: SourceIdentity,
    /// Source capture time, not the collector's receipt time.
    pub captured_at: ObservationTime,
    /// Latest source event sequence, even if that event was evicted.
    pub sequence: DiagnosticNumber,
    /// Evidence including retained closed components.
    pub components: Vec<ComponentSnapshot>,
    /// Bounded recent history selected by the query.
    pub events: Vec<DiagnosticEvent>,
    /// Limitations to consider before interpreting absent evidence.
    pub coverage: DiagnosticCoverage,
}
