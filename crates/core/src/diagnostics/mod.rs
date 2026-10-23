// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, process-local diagnostic evidence, independent of logger startup.
//!
//! Startup composition chooses collection once. Disabled handles skip lazy work;
//! enabled components reserve summaries before recording. Recent evidence uses a
//! shared circular history and a reclaimable detail pool. Neither history loss
//! nor output delivery changes accepted numeric totals. Diagnostic evidence never
//! assigns readiness, recovery, or deployment health.
//!
//! The backing-storage budget excludes caller-owned inputs, producer handles,
//! independently decoded snapshot copies, allocator overhead, and user callback
//! captures. Existing operational output and legacy reports have separate budgets.
//!
//! # Startup and recording
//!
//! Select settings once before giving producers their handles. Install the chosen
//! context with [`Diagnostics::install_process_default`] before using convenience
//! constructors, or inject it explicitly into logging and plugin composition.
//! Neither path requires logging to initialize first.
//!
//! ```
//! use nemo_relay::diagnostics::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let diagnostics = Diagnostics::from_settings(
//!     SourceIdentity {
//!         instance_id: uuid::Uuid::now_v7(),
//!         process_id: Some(std::process::id()),
//!         process_role: "embedded".into(),
//!         relay_version: env!("CARGO_PKG_VERSION").into(),
//!     },
//!     DiagnosticsSettings { enabled: true, ..Default::default() },
//! )?;
//! let code = BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkWriteFailed));
//! let recorder = diagnostics.register_component(ComponentRegistration::new("example", &[code]))?;
//! recorder.record_lazy(|| Observation::new(
//!     Classification::failure(FailureCode::Logging(LoggingFailureCode::SinkWriteFailed), FailureClass::Io)
//!         .expect("supported catalogue pair"),
//!     "Producer-approved safe detail",
//!     std::num::NonZeroU64::MIN, // occurrence delta, never a cumulative total
//! ))?;
//! let page = diagnostics.snapshot(&DiagnosticQuery::default())?;
//! let bounded_json: &[u8] = page.as_bytes();
//! # assert!(!bounded_json.is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! Disabled contexts use the same producer calls and skip their lazy factories;
//! reads return [`DiagnosticError::Disabled`]. Held snapshot leases use a reserved
//! query workspace until dropped. Decoding [`SnapshotLease::to_snapshot`] instead
//! creates an independent caller-owned DTO outside the service's memory budget.

/// Producer-owned pure mappings from existing typed errors to catalogue values.
pub mod classifiers;
pub use classifiers::PluginErrorClassifier;
/// Configuration-driven service and bounded copies for embedded language runtimes.
pub mod embedded;
mod lifecycle;
mod read;
mod request;
pub use request::{MAX_DIAGNOSTIC_QUERY_BYTES, decode_diagnostic_query};
mod settings;
mod state;
mod storage;
use settings::MAX_DETAIL_BYTES;
pub use settings::{DiagnosticLimits, DiagnosticsSettings};
mod observation;
use observation::ObservationCode;
pub use observation::{
    ComponentRegistration, DiagnosticDetailRef, DiagnosticObservation, EvidenceDisposition,
    ExtensionClassHandle, ExtensionHandle, ExtensionRegistration, Observation, RecordOutcome,
    RecordReceipt, WindowSettings,
};
mod error;
pub use error::{DiagnosticError, ProjectionError};
mod context;
use context::{Backend, Inner, lock};
pub use context::{
    ComponentOwner, ComponentRecorder, Diagnostics, DiagnosticsReader, OriginRecorder,
};
mod projection;
pub use projection::ProjectionGuard;
use projection::{
    Projection, Projections, PublicationLease, PublicationWorkspace, publish, take_publication,
};
mod recording;
use recording::{bounded_str, prepare_detail, record_observation};
mod memory;
use lifecycle::{close_component, refresh_component, register_component, register_extension};
pub use memory::DiagnosticMemoryUsage;
use memory::{context_layout_bytes, diagnostic_memory_usage};
pub use nemo_relay_types::diagnostics::{
    BuiltinDiagnosticCode, Classification, ClassificationError, CliFailureCode,
    CommonPluginFailureCode, ComponentIdentity, ComponentOwnership, ComponentSnapshot,
    DIAGNOSTIC_CATALOGUE_VERSION, DIAGNOSTIC_DISCOVERY_VERSION, DIAGNOSTIC_SCHEMA_VERSION,
    DaemonFailureCode, DiagnosticAvailability, DiagnosticCapability, DiagnosticClassifier,
    DiagnosticCorrelation, DiagnosticCount, DiagnosticCounter, DiagnosticCoverage,
    DiagnosticCursor, DiagnosticDetail, DiagnosticDiscovery, DiagnosticDomain, DiagnosticEvent,
    DiagnosticLifetimeSummary, DiagnosticLoss, DiagnosticNumber, DiagnosticOrigin,
    DiagnosticOutcome, DiagnosticOutputSummary, DiagnosticProducerSupport, DiagnosticQuery,
    DiagnosticReadReason, DiagnosticRefreshSummary, DiagnosticRetention, DiagnosticSeverity,
    DiagnosticSnapshot, DiagnosticSummaryScope, DiagnosticTotals, DiagnosticWindow,
    DiagnosticWindowBucket, DiagnosticWindowCoverage, FailureClass, FailureCode,
    LoggingFailureCode, NoticeCode, ObservationTime, PluginDiagnosticFamily, PluginExecution,
    PluginExecutionName, PluginFailureCode, PluginObservationContext, PluginOperationFailureCode,
    PluginPhase, PluginPhaseName, RemoteDiagnosticOrigin, SourceIdentity, SuccessCode,
};
pub use read::SnapshotLease;
use state::{
    CapturedOrigin, ComponentHandle, ComponentSlot, CounterIndex, CounterSlot, DESCRIPTION_BYTES,
    ExtensionSlot, ID_BYTES, LifetimeState, Ownership, RefreshState, State, StoredCode,
    StoredFailureClass, StoredObservation, StoredPluginContext, WindowReservation, WindowSlot,
    empty_slots, text,
};

pub use nemo_relay_types::diagnostics::{
    ComponentMemoryFootprint, ComponentMemoryUsage, DIAGNOSTIC_MEMORY_VERSION,
    DiagnosticMemoryPage, EntryMemoryFootprint, EntryMemoryUsage, MemoryFieldScope,
    MemoryLimitEvent, MemorySelectionTotals, ResourceAssociation, ResourceMemoryAvailability,
    ResourceMemoryMeasurement, ResourceMemoryMetrics, ResourceMemoryNumber, ResourceMemorySample,
    ResourceMemoryUnit, ResourceMemoryView, ResourceProcessIdentity, ResourceSampleStream,
};
pub use read::{ComponentMemoryView, EntryMemoryView, MemoryInspectionLease};

pub use read::{ComponentView, EventView};
