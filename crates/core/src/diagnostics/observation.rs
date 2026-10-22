// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Producer registrations, handles, observations, and recording receipts.

use super::{
    BuiltinDiagnosticCode, Classification, ComponentHandle, ComponentOwnership,
    DiagnosticCorrelation, DiagnosticOutcome, DiagnosticSeverity, FailureClass,
    PluginObservationContext,
};
use std::fmt;
use std::num::NonZeroU64;
use uuid::Uuid;

/// Fixed-duration component outcome windows; no code/class Cartesian product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSettings {
    /// Positive duration of one bucket on the source monotonic clock.
    pub resolution_millis: u64,
    /// Positive circular bucket-slot reservation for the component instance.
    pub bucket_count: usize,
}
impl Default for WindowSettings {
    fn default() -> Self {
        Self {
            resolution_millis: 5000,
            bucket_count: 60,
        }
    }
}

/// Expected codes and attribution are admitted together with the summary.
#[derive(Debug, Clone)]
pub struct ComponentRegistration<'a> {
    /// Stable logical responsibility, excluding request and session identifiers.
    pub key: &'a str,
    /// Complete built-in vocabulary whose counters are reserved at admission.
    pub builtin_codes: &'a [BuiltinDiagnosticCode],
    /// Initial configuration attribution, bounded on admission.
    pub ownership: ComponentOwnership,
    /// Optional fixed-resolution outcome series admitted with the component.
    pub window: Option<WindowSettings>,
}
impl<'a> ComponentRegistration<'a> {
    /// Declare a vocabulary with empty attribution and no outcome window.
    pub fn new(key: &'a str, builtin_codes: &'a [BuiltinDiagnosticCode]) -> Self {
        Self {
            key,
            builtin_codes,
            ownership: ComponentOwnership::default(),
            window: None,
        }
    }
}

/// Extension vocabulary is scoped to a producer namespace and admitted once.
#[derive(Debug, Clone)]
pub struct ExtensionRegistration<'a> {
    /// Producer namespace outside reserved built-in namespaces.
    pub namespace: &'a str,
    /// Stable local name combined with the namespace at admission.
    pub name: &'a str,
    /// Safe bounded explanation retained once in registration metadata.
    pub description: &'a str,
    /// Permitted causes; successful or unspecified extensions admit no class.
    pub allowed_classes: &'a [FailureClass],
    /// Declared outcome used to validate extension class requirements.
    pub outcome: DiagnosticOutcome,
}
/// A registered extension identity. A stale handle cannot create a new code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensionHandle {
    pub(super) context_id: Uuid,
    pub(super) component: ComponentHandle,
    pub(super) slot: usize,
    pub(super) generation: u64,
}

/// Borrowed producer-approved detail. A Display source is formatted into capped
/// storage only when the enabled recorder invokes its lazy observation factory.
#[derive(Clone, Copy)]
pub enum DiagnosticDetailRef<'a> {
    /// Supply no optional text evidence.
    None,
    /// Borrow restricted text for a capped copy; bounding does not redact secrets.
    Text(&'a str),
    /// Format into capped scratch only when enabled; retain no source value.
    Display(&'a dyn fmt::Display),
}
impl fmt::Debug for DiagnosticDetailRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "None",
            Self::Text(_) => "Text(..)",
            Self::Display(_) => "Display(..)",
        })
    }
}
impl<'a> From<&'a str> for DiagnosticDetailRef<'a> {
    fn from(value: &'a str) -> Self {
        Self::Text(value)
    }
}

/// An admitted custom cause descriptor belonging to one registered extension.
/// Its immutable metadata is bounded and does not create another counter series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensionClassHandle {
    pub(super) extension: ExtensionHandle,
    pub(super) index: usize,
}

#[derive(Debug, Clone)]
pub(super) enum ObservationCode {
    Builtin(Classification),
    Extension(ExtensionHandle, Option<FailureClass>),
    OpaqueExtension(ExtensionHandle, ExtensionClassHandle),
    Overflow,
}

/// One observation. Its count is a positive input delta, never a cumulative total.
///
/// ```compile_fail
/// use nemo_relay::diagnostics::{Classification, Observation, SuccessCode};
/// // Constructors require NonZeroU64, so a zero integer is not a valid input.
/// let observation = Observation::new(
///     Classification::success(SuccessCode::SinkInitialized), "initialized", 0,
/// );
/// ```
#[derive(Debug, Clone)]
pub struct Observation<'a> {
    pub(super) code: ObservationCode,
    // Only core's authorized transport adapter can supply remote identity.
    pub(crate) remote: Option<nemo_relay_types::diagnostics::RemoteDiagnosticOrigin>,
    /// Optional producer-approved explanation; bounding does not perform redaction.
    pub detail: DiagnosticDetailRef<'a>,
    /// Positive input occurrence delta, never a cumulative report total.
    pub count: NonZeroU64,
    /// Operational evidence severity, independent of deployment health.
    pub severity: DiagnosticSeverity,
    /// Explicit observed operation outcome for summaries and windows.
    pub outcome: DiagnosticOutcome,
    /// Borrowed caller correlation, never an aggregate key.
    pub correlation: DiagnosticCorrelation<'a>,
    /// Plugin execution and phase names, validated before fixed-size core retention.
    pub context: PluginObservationContext,
}
impl<'a> Observation<'a> {
    /// Author built-in evidence with severity/outcome defaults from its code.
    ///
    /// The type guarantees a positive count; recording validates the reservation. Constructing
    /// this value does not retain its borrowed detail or source error.
    pub fn new(
        classification: Classification,
        detail: impl Into<DiagnosticDetailRef<'a>>,
        count: NonZeroU64,
    ) -> Self {
        let (severity, outcome) =
            match BuiltinDiagnosticCode::from_canonical_name(classification.code()) {
                Some(BuiltinDiagnosticCode::Failure(_)) => {
                    (DiagnosticSeverity::Error, DiagnosticOutcome::Failure)
                }
                Some(BuiltinDiagnosticCode::Success(_)) => {
                    (DiagnosticSeverity::Info, DiagnosticOutcome::Success)
                }
                _ => (DiagnosticSeverity::Info, DiagnosticOutcome::Unknown),
            };
        Self {
            code: ObservationCode::Builtin(classification),
            remote: None,
            detail: detail.into(),
            count,
            severity,
            outcome,
            correlation: DiagnosticCorrelation::default(),
            context: PluginObservationContext::default(),
        }
    }
    /// Author evidence using an admitted extension handle and supported cause.
    ///
    /// Recording validates ownership, descriptor generation, and class. A stale
    /// handle cannot reserve a replacement code or create another counter.
    pub fn extension(
        handle: ExtensionHandle,
        class: Option<FailureClass>,
        outcome: DiagnosticOutcome,
        detail: impl Into<DiagnosticDetailRef<'a>>,
        count: NonZeroU64,
    ) -> Self {
        Self {
            code: ObservationCode::Extension(handle, class),
            remote: None,
            detail: detail.into(),
            count,
            severity: if outcome == DiagnosticOutcome::Failure {
                DiagnosticSeverity::Error
            } else {
                DiagnosticSeverity::Info
            },
            outcome,
            correlation: DiagnosticCorrelation::default(),
            context: PluginObservationContext::default(),
        }
    }
    /// Author an extension failure with its own admitted immutable cause name.
    /// Handles from another code/registration or expired metadata are rejected.
    pub fn extension_with_class(
        handle: ExtensionHandle,
        class: ExtensionClassHandle,
        detail: impl Into<DiagnosticDetailRef<'a>>,
        count: NonZeroU64,
    ) -> Self {
        Self {
            code: ObservationCode::OpaqueExtension(handle, class),
            remote: None,
            detail: detail.into(),
            count,
            severity: DiagnosticSeverity::Error,
            outcome: DiagnosticOutcome::Failure,
            correlation: DiagnosticCorrelation::default(),
            context: PluginObservationContext::default(),
        }
    }
    /// Explicit summary-only admission when extension vocabulary could not fit.
    /// This is not a way to reinterpret an invalid or undeclared code.
    pub fn classification_overflow(outcome: DiagnosticOutcome, count: NonZeroU64) -> Self {
        Self {
            code: ObservationCode::Overflow,
            remote: None,
            detail: DiagnosticDetailRef::None,
            count,
            severity: DiagnosticSeverity::Warning,
            outcome,
            correlation: DiagnosticCorrelation::default(),
            context: PluginObservationContext::default(),
        }
    }
    /// Attach known plugin execution and lifecycle facts without another series.
    pub fn with_plugin_context(mut self, context: PluginObservationContext) -> Self {
        self.context = context;
        self
    }
    /// Set the observed outcome without changing classification or health policy.
    pub fn with_outcome(mut self, outcome: DiagnosticOutcome) -> Self {
        self.outcome = outcome;
        self
    }
    /// Set output severity independently of operation outcome and cause.
    pub fn with_severity(mut self, severity: DiagnosticSeverity) -> Self {
        self.severity = severity;
        self
    }
    /// Attach safe borrowed correlation to be capped outside aggregate keys.
    pub fn with_correlation(mut self, correlation: DiagnosticCorrelation<'a>) -> Self {
        self.correlation = correlation;
        self
    }
}
/// Explicit authoring name for a borrowed observation.
pub type DiagnosticObservation<'a> = Observation<'a>;

pub use nemo_relay_types::diagnostics::{EvidenceDisposition, RecordOutcome, RecordReceipt};
