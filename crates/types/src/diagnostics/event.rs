// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Classification, optional context, and diagnostic evidence.

use super::{
    BuiltinDiagnosticCode, ClassificationError, ComponentIdentity, DiagnosticNumber,
    DiagnosticOrigin, FailureClass, FailureCode, NoticeCode, ObservationTime,
    PluginObservationContext, SuccessCode,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

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

/// Optional caller correlation identifiers, excluded from aggregation keys.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticCorrelation<'a> {
    /// Trajectory session associated with the observation, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Cow<'a, str>>,
    /// Request associated with the observation, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<Cow<'a, str>>,
}

/// Shared classification payload for producers, projections, and decoded evidence.
///
/// Typed constructors validate built-in code/cause pairs and borrow their stable
/// names. Decoded future names remain opaque. Core validates recording input and
/// resolves admitted identifiers before changing counters or retaining evidence.
/// Transport adapters must bound received names before materializing owned data.
/// Private fields prevent mutation after construction. Explicit opaque-name
/// construction and deserialization preserve evidence without asserting validity.
///
/// ```compile_fail
/// use nemo_relay_types::diagnostics::{Classification, SuccessCode};
/// let mut value = Classification::success(SuccessCode::SinkInitialized);
/// value.canonical_code = "different.name".into();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Classification {
    /// Stable built-in name or registered namespaced extension name.
    canonical_code: Cow<'static, str>,
    /// Cause name; absent for successful observations and notices.
    failure_class: Option<Cow<'static, str>>,
}

impl Classification {
    /// Preserve names from a wire payload or an admitted extension descriptor.
    /// This does not validate catalogue membership or authorize recording. Core
    /// revalidates every recording input, including values decoded with Serde.
    pub fn from_opaque_names(
        canonical_code: Cow<'static, str>,
        failure_class: Option<Cow<'static, str>>,
    ) -> Self {
        Self {
            canonical_code,
            failure_class,
        }
    }
    /// Consume the payload without cloning borrowed names or owned buffers.
    pub fn into_names(self) -> (Cow<'static, str>, Option<Cow<'static, str>>) {
        (self.canonical_code, self.failure_class)
    }
    /// Validate the required cause and the catalogue entry's permitted classes.
    pub fn new(
        code: BuiltinDiagnosticCode,
        failure_class: Option<FailureClass>,
    ) -> Result<Self, ClassificationError> {
        if !code.allows_class(failure_class) {
            return Err(ClassificationError);
        }
        Ok(Self {
            canonical_code: Cow::Borrowed(code.canonical_name()),
            failure_class: failure_class.map(|class| Cow::Borrowed(class.as_str())),
        })
    }
    /// Classify a failure using an explicitly supported cause.
    pub fn failure(
        code: FailureCode,
        failure_class: FailureClass,
    ) -> Result<Self, ClassificationError> {
        Self::new(BuiltinDiagnosticCode::Failure(code), Some(failure_class))
    }
    /// Classify an observed success, without clearing previous failures.
    #[must_use]
    pub const fn success(code: SuccessCode) -> Self {
        Self {
            canonical_code: Cow::Borrowed(code.canonical_name()),
            failure_class: None,
        }
    }
    /// Classify an observed notice, without inventing a failure cause.
    #[must_use]
    pub const fn notice(code: NoticeCode) -> Self {
        Self {
            canonical_code: Cow::Borrowed(code.canonical_name()),
            failure_class: None,
        }
    }
    /// Borrow the canonical name, including an opaque future name.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.canonical_code
    }
    /// Borrow the observed cause name without interpreting unfamiliar values.
    #[must_use]
    pub fn failure_class(&self) -> Option<&str> {
        self.failure_class.as_deref()
    }
}

/// Restricted optional recent evidence, independent of the accepted numeric summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticDetail {
    /// Text copied within admitted byte capacity; bounding does not sanitize it.
    pub message: String,
    /// Whether the original text exceeded its retained byte allowance.
    pub truncated: bool,
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
    /// Authenticated remote submission provenance, absent for local observations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteDiagnosticOrigin>,
    /// Canonical event classification; never reconstructed from a latest message.
    pub classification: Classification,
    /// Bounded plugin-operation wire facts, excluded from aggregation keys.
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
    /// Restricted caller correlation, excluded from aggregation identity.
    pub correlation: DiagnosticCorrelation<'static>,
}

/// Fixed-size provenance retained by an authorized ingestion adapter.
///
/// Producer wall time is a claim with unknown clock skew. Host summaries and
/// windows use host receipt time; clocks from different processes are never
/// subtracted. Session rotation assigns a fresh source before sequence reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RemoteDiagnosticOrigin {
    /// Host-issued diagnostic session/source incarnation.
    pub source_instance_id: uuid::Uuid,
    /// Host-issued plugin activation incarnation, distinct from the session.
    pub activation_instance_id: uuid::Uuid,
    /// Ordered terminal submission sequence; not the host event sequence.
    pub submission_sequence: DiagnosticNumber,
    /// Optional producer-reported wall time, with unknown clock uncertainty.
    pub observed_at: Option<ObservationTime>,
    /// Host wall time at acceptance, also used as the event observation time.
    pub received_at: ObservationTime,
    /// Host monotonic elapsed time at acceptance, relative to context startup.
    pub accepted_elapsed_millis: DiagnosticNumber,
}

impl DiagnosticEvent {
    /// Borrow the canonical code without allocating a second representation.
    #[must_use]
    pub fn code(&self) -> &str {
        self.classification.code()
    }
    /// Borrow retained detail, when available.
    #[must_use]
    pub fn message(&self) -> Option<&str> {
        self.detail.as_ref().map(|detail| detail.message.as_str())
    }
}

/// Allowlisted operational output, constructed independently of retained detail.
///
/// This payload deliberately has no message, caller correlation, component name,
/// plugin context, or configuration attribution. Unknown/extension codes are not
/// automatically authorized for output by the recording service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticOutputSummary {
    /// Source-local sequence of the accepted observation.
    pub sequence: DiagnosticNumber,
    /// Time at the recording boundary.
    pub observed_at: ObservationTime,
    /// Service-generated registration identity, without its caller-supplied name.
    pub component_instance_id: uuid::Uuid,
    /// Validated built-in code and cause only.
    pub classification: Classification,
    /// Approved enum severity.
    pub severity: DiagnosticSeverity,
    /// Observed enum outcome.
    pub outcome: DiagnosticOutcome,
    /// Original occurrence delta.
    pub count: DiagnosticNumber,
}
impl DiagnosticOutputSummary {
    /// Stable built-in catalogue name.
    pub fn code(&self) -> &str {
        self.classification.code()
    }
}
