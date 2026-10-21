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

/// Initial version of the canonical diagnostic snapshot envelope.
pub const DIAGNOSTIC_SCHEMA_VERSION: u32 = 1;

/// Initial stable typed-catalogue revision, independent of the envelope version.
pub const DIAGNOSTIC_CATALOGUE_VERSION: u32 = 1;

mod catalogue;
pub use catalogue::{
    BuiltinDiagnosticCode, ClassificationError, CommonPluginFailureCode, DiagnosticClassifier,
    DiagnosticDomain, FailureClass, FailureCode, LoggingFailureCode, NoticeCode,
    PluginDiagnosticFamily, PluginExecution, PluginFailureCode, PluginPhase, SuccessCode,
};
mod number;
pub use number::DiagnosticNumber;
mod identity;
pub use identity::{
    ComponentIdentity, ComponentOwnership, DiagnosticOrigin, ObservationTime, SourceIdentity,
};
mod accounting;
pub use accounting::{
    DiagnosticCount, DiagnosticCounter, DiagnosticLifetimeSummary, DiagnosticRefreshSummary,
    DiagnosticTotals, DiagnosticWindow, DiagnosticWindowBucket, DiagnosticWindowCoverage,
};
mod event;
pub use event::{
    Classification, DiagnosticCorrelation, DiagnosticDetail, DiagnosticEvent, DiagnosticOutcome,
    DiagnosticSeverity,
};
mod context;
pub use context::{PluginExecutionName, PluginObservationContext, PluginPhaseName};
mod coverage;
pub use coverage::{
    DiagnosticAvailability, DiagnosticCapability, DiagnosticCoverage, DiagnosticLoss,
    DiagnosticRetention, DiagnosticSummaryScope,
};
mod query;
pub use query::{DiagnosticCursor, DiagnosticQuery};
mod snapshot;
pub use snapshot::{ComponentSnapshot, DiagnosticSnapshot};
