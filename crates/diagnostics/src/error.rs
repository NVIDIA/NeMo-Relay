// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission, availability, and post-acceptance delivery errors.

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
/// Validation, admission, or availability failure, independent of operational logging.
pub enum DiagnosticError {
    /// Reads report configured disablement instead of available empty evidence.
    #[error("diagnostics is disabled by startup configuration")]
    Disabled,
    /// Enabled capacities are invalid, overflow arithmetic, or exceed the byte ceiling.
    #[error("invalid diagnostics capacity profile")]
    InvalidLimits,
    /// An admitted startup backing allocation could not be reserved.
    #[error("diagnostics backing allocation failed")]
    Allocation,
    /// The logical key is empty, oversized, or outside the supported syntax.
    #[error("invalid diagnostic component key")]
    InvalidComponentKey,
    /// The code is undeclared or its handle names another/stale registration.
    #[error("invalid or undeclared diagnostic code")]
    InvalidCode,
    /// The code or extension does not permit the supplied cause.
    #[error("invalid diagnostic classification")]
    InvalidClassification,
    /// Plugin context contains a name unsupported by this recording runtime.
    #[error("invalid or unsupported diagnostic plugin context")]
    InvalidContext,
    /// Admission cannot fit another component without displacing active state.
    #[error("diagnostic component capacity exhausted")]
    ComponentCapacity,
    /// Declared built-in or extension counter reservations cannot fit.
    #[error("diagnostic code-counter capacity exhausted")]
    KeyCapacity,
    /// A key, ownership value, or descriptor exceeds metadata capacity.
    #[error("diagnostic metadata capacity exhausted")]
    MetadataCapacity,
    /// The requested outcome series cannot reserve its bucket storage.
    #[error("diagnostic window capacity exhausted")]
    WindowCapacity,
    /// The handle names a closed or replaced registration.
    #[error("diagnostic component is closed or stale")]
    ComponentClosed,
    /// A sequence or generation cannot advance without identity reuse.
    #[error("diagnostic sequence exhausted")]
    SequenceExhausted,
    /// Startup intentionally disabled query materialization with max_queries = 0.
    #[error("diagnostic queries are disabled")]
    QueryDisabled,
    /// Every bounded query workspace is currently leased.
    #[error("diagnostic query capacity exhausted")]
    QueryCapacity,
    /// The requested snapshot envelope version is unsupported.
    #[error("unsupported diagnostic snapshot schema")]
    UnsupportedSchema,
    /// A selector or page request exceeds its admission constraints.
    #[error("diagnostic query is invalid")]
    InvalidQuery,
    /// Continuation has the wrong source/scope or its captured state changed.
    #[error("diagnostic continuation is invalid or its capture changed")]
    InvalidCursor,
    /// A selected atomic row or required envelope cannot fit the query budgets.
    #[error("diagnostic query page cannot hold a required row or envelope")]
    SnapshotCapacity,
    /// Backing ownership could not be reconciled; no partial accounting is returned.
    #[error("diagnostic memory accounting is inconsistent")]
    InconsistentAccounting,
    /// Subscription slots or unique registration IDs are exhausted.
    #[error("diagnostic projection capacity exhausted")]
    ProjectionCapacity,
    /// The process context was already installed or consumed, even while disabled.
    #[error("the process diagnostic context is already installed or used")]
    AlreadyInitialized,
}
impl DiagnosticError {
    /// Safe transport-neutral meaning for read/startup failures. Recording-only
    /// errors are internal here; they must never be mislabeled invalid input.
    pub fn read_reason(self) -> super::DiagnosticReadReason {
        use super::DiagnosticReadReason as Reason;
        match self {
            Self::Disabled => Reason::Disabled,
            Self::QueryDisabled => Reason::QueryDisabled,
            Self::QueryCapacity => Reason::Busy,
            Self::InvalidQuery => Reason::InvalidQuery,
            Self::UnsupportedSchema => Reason::UnsupportedSchema,
            Self::InvalidCursor => Reason::InvalidCursor,
            Self::SnapshotCapacity => Reason::SnapshotCapacity,
            Self::InvalidLimits => Reason::InvalidLimits,
            Self::Allocation => Reason::AllocationFailed,
            _ => Reason::Internal,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
/// Subscriber conversion or delivery failure after canonical acceptance.
pub enum ProjectionError {
    /// The conversion or output representation exceeded its explicit byte ceiling.
    #[error("projection payload exceeds its budget")]
    PayloadTooLarge,
    /// The bounded formatter could not serialize the accepted observation.
    #[error("projection serialization failed")]
    SerializationFailed,
    /// Delivery failed without undoing accepted canonical state.
    #[error("projection delivery failed")]
    DeliveryFailed,
}
