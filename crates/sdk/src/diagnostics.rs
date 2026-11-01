// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic collection and inspection contracts for embedded Rust applications.

pub use nemo_relay_diagnostics::{
    BuiltinDiagnosticCode, Classification, ClassificationError, CliFailureCode,
    CommonPluginFailureCode, ComponentIdentity, ComponentMemoryFootprint, ComponentMemoryUsage,
    ComponentMemoryView, ComponentOwner, ComponentOwnership, ComponentRecorder,
    ComponentRegistration, ComponentSnapshot, ComponentView, DIAGNOSTIC_CATALOGUE_VERSION,
    DIAGNOSTIC_DISCOVERY_VERSION, DIAGNOSTIC_MEMORY_VERSION, DIAGNOSTIC_SCHEMA_VERSION,
    DaemonFailureCode, DiagnosticAvailability, DiagnosticCapability, DiagnosticClassifier,
    DiagnosticCorrelation, DiagnosticCount, DiagnosticCounter, DiagnosticCoverage,
    DiagnosticCursor, DiagnosticDetail, DiagnosticDetailRef, DiagnosticDiscovery, DiagnosticDomain,
    DiagnosticError, DiagnosticEvent, DiagnosticLifetimeSummary, DiagnosticLimits, DiagnosticLoss,
    DiagnosticMemoryPage, DiagnosticMemoryUsage, DiagnosticNumber, DiagnosticObservation,
    DiagnosticOrigin, DiagnosticOutcome, DiagnosticOutputSummary, DiagnosticProducerSupport,
    DiagnosticQuery, DiagnosticReadReason, DiagnosticRefreshSummary, DiagnosticRetention,
    DiagnosticSeverity, DiagnosticSnapshot, DiagnosticSummaryScope, DiagnosticTotals,
    DiagnosticWindow, DiagnosticWindowBucket, DiagnosticWindowCoverage, Diagnostics,
    DiagnosticsReader, DiagnosticsSettings, EntryMemoryFootprint, EntryMemoryUsage,
    EntryMemoryView, EventView, EvidenceDisposition, ExtensionClassHandle, ExtensionHandle,
    ExtensionRegistration, FailureClass, FailureCode, LoggingFailureCode,
    MAX_DIAGNOSTIC_QUERY_BYTES, MemoryFieldScope, MemoryInspectionLease, MemoryLimitEvent,
    MemorySelectionTotals, NoticeCode, Observation, ObservationTime, OriginRecorder,
    PluginDiagnosticFamily, PluginErrorClassifier, PluginExecution, PluginExecutionName,
    PluginFailureCode, PluginObservationContext, PluginOperationFailureCode, PluginPhase,
    PluginPhaseName, ProjectionError, ProjectionGuard, RecordOutcome, RecordReceipt,
    RemoteDiagnosticOrigin, ResourceAssociation, ResourceMemoryAvailability,
    ResourceMemoryMeasurement, ResourceMemoryMetrics, ResourceMemoryNumber, ResourceMemorySample,
    ResourceMemoryUnit, ResourceMemoryView, ResourceProcessIdentity, ResourceSampleStream,
    SnapshotLease, SourceIdentity, SuccessCode, WindowSettings, decode_diagnostic_query,
};

// Keep all shared DTO leaf types reachable alongside the service facade.
pub use nemo_relay_types::diagnostics::*;
