// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration-driven embedded service and bounded language-adapter copies.
//!
//! Readers are inspection authority. No automatic formatting or error path exposes
//! retained detail, and no snapshot is connected to an event/exporter pipeline.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Serialize;
#[cfg(test)]
use serde_json::Value as Json;

use super::{
    DiagnosticError, DiagnosticQuery, Diagnostics, DiagnosticsReader, DiagnosticsSettings,
    SourceIdentity,
};

/// Maximum encoded query accepted by an embedded reader, before JSON parsing.
pub const MAX_BINDING_QUERY_BYTES: usize = super::MAX_DIAGNOSTIC_QUERY_BYTES;
/// Query fields whose canonical JSON value is a lossless unsigned integer.
/// Binding adapters use this descriptor when accepting native integer inputs.
pub const DIAGNOSTIC_QUERY_INTEGER_FIELDS: &[&str] = &[
    "origin_generation",
    "after_sequence",
    "sequence",
    "component_offset",
    "event_offset",
    "scope",
];
/// Independent simultaneous page-copy/foreign-conversion budget per service.
pub const BINDING_COPY_CONCURRENCY: usize = 2;

/// Stable service/reader failure. Messages never contain retained evidence or query input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EmbeddedDiagnosticError {
    /// Reads never initialize a service implicitly.
    #[error("not_initialized")]
    NotInitialized,
    /// An inherited service/reader cannot safely use parent-process locks.
    #[error("fork_requires_exec")]
    ForkRequiresExec,
    /// Startup-only settings cannot change through plugin reload.
    #[error("configuration_conflict")]
    ConfigurationConflict,
    /// Shared bounded request admission result.
    #[error("{0}")]
    Request(#[from] super::DiagnosticReadReason),
    /// Core admission or query result.
    #[error("{0}")]
    Core(#[from] DiagnosticError),
}
impl EmbeddedDiagnosticError {
    /// Machine-readable reason shared by the native bindings and C ABI.
    pub fn reason(self) -> &'static str {
        match self {
            Self::NotInitialized => "not_initialized",
            Self::ForkRequiresExec => "fork_requires_exec",
            Self::ConfigurationConflict => "configuration_conflict",
            Self::Request(reason) => reason.as_str(),
            Self::Core(DiagnosticError::AlreadyInitialized) => "configuration_conflict",
            Self::Core(error) => error.read_reason().as_str(),
        }
    }
}

type Result<T> = std::result::Result<T, EmbeddedDiagnosticError>;
static SERVICE: OnceLock<EmbeddedService> = OnceLock::new();
static STARTUP: Mutex<()> = Mutex::new(());
static OWNER_PID: AtomicU32 = AtomicU32::new(0);

fn check_process(pid: u32) -> Result<()> {
    if pid != 0 && pid != std::process::id() {
        Err(EmbeddedDiagnosticError::ForkRequiresExec)
    } else {
        Ok(())
    }
}

struct EmbeddedService {
    context: Diagnostics,
    reader: EmbeddedDiagnosticsReader,
}

impl EmbeddedService {
    fn new(context: Diagnostics) -> Result<Self> {
        context
            .limits()
            .max_snapshot_bytes
            .checked_mul(BINDING_COPY_CONCURRENCY)
            .ok_or(DiagnosticError::InvalidLimits)?;
        let mut buffers = Vec::with_capacity(BINDING_COPY_CONCURRENCY);
        if context.reader().availability() == super::DiagnosticAvailability::Available
            && context.limits().max_queries > 0
        {
            for _ in 0..BINDING_COPY_CONCURRENCY {
                let mut buffer = Vec::new();
                buffer
                    .try_reserve_exact(context.limits().max_snapshot_bytes)
                    .map_err(|_| DiagnosticError::Allocation)?;
                buffers.push(buffer);
            }
        }
        let reader = EmbeddedDiagnosticsReader {
            inner: context.reader(),
            copies: Arc::new(Mutex::new(buffers)),
            pid: std::process::id(),
        };
        Ok(Self { context, reader })
    }
}

/// Resolve once before plugin validation/activation. Equivalent settings reuse
/// the service; a conflicting request leaves all existing state untouched.
pub(crate) fn start(settings: DiagnosticsSettings) -> Result<&'static Diagnostics> {
    check_process(OWNER_PID.load(Ordering::Acquire))?;
    // Publish before acquiring any service lock, including concurrent first startup.
    OWNER_PID
        .compare_exchange(0, std::process::id(), Ordering::AcqRel, Ordering::Acquire)
        .ok();
    check_process(OWNER_PID.load(Ordering::Acquire))?;
    let _startup = STARTUP.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(service) = SERVICE.get() {
        if (service.context.reader().availability() == super::DiagnosticAvailability::Available)
            != settings.enabled
            || service.context.limits() != &settings.limits
        {
            return Err(EmbeddedDiagnosticError::ConfigurationConflict);
        }
        return Ok(&service.context);
    }
    settings.limits.validate()?;
    // Allocate the adapter budget before selecting the process default. A failure
    // cannot leave a partially initialized service or consume the default.
    let existing = Diagnostics::selected_process_default();
    if let Some(context) = existing
        && ((context.reader().availability() == super::DiagnosticAvailability::Available)
            != settings.enabled
            || context.limits() != &settings.limits)
    {
        return Err(EmbeddedDiagnosticError::ConfigurationConflict);
    }
    let context = match existing {
        Some(context) => context.clone(),
        None => Diagnostics::from_settings(
            SourceIdentity {
                instance_id: uuid::Uuid::now_v7(),
                process_id: Some(std::process::id()),
                process_role: "embedded".into(),
                relay_version: env!("CARGO_PKG_VERSION").into(),
            },
            settings,
        )?,
    };
    let service = EmbeddedService::new(context)?;
    if existing.is_none() {
        Diagnostics::install_context(service.context.clone())?;
    }
    SERVICE
        .set(service)
        .map_err(|_| EmbeddedDiagnosticError::ConfigurationConflict)?;
    let service = SERVICE.get().expect("service installed under startup lock");
    // Optional attachment cannot disable startup or replace the embedding logger.
    let _ = crate::logging::attach_default_diagnostics(&service.context);
    Ok(&service.context)
}

/// Obtain an inspection capability without initializing the service or logger.
/// The service is selected by ordinary plugin-host initialization configuration.
pub fn reader() -> Result<EmbeddedDiagnosticsReader> {
    check_process(OWNER_PID.load(Ordering::Acquire))?;
    SERVICE
        .get()
        .map(|service| service.reader.clone())
        .ok_or(EmbeddedDiagnosticError::NotInitialized)
}

/// Read-only embedded capability. Clones share the service's copy admission pool.
/// Debug output deliberately contains no snapshot, query, or restricted detail.
#[derive(Clone)]
pub struct EmbeddedDiagnosticsReader {
    inner: DiagnosticsReader,
    copies: Arc<Mutex<Vec<Vec<u8>>>>,
    pid: u32,
}
impl std::fmt::Debug for EmbeddedDiagnosticsReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedDiagnosticsReader")
            .finish_non_exhaustive()
    }
}

/// Independent budgets and coverage of the binding adapter; excludes language
/// object overhead and caller-retained copies, and is not an RSS measurement.
#[derive(Debug, Serialize)]
pub struct EmbeddedDiagnosticInfo {
    /// Canonical core discovery and source identity.
    pub discovery: super::DiagnosticDiscovery,
    /// Configured capacities, even when disabled.
    pub limits: super::DiagnosticLimits,
    /// Core allocated backing and occupancy.
    pub memory: super::DiagnosticMemoryUsage,
    /// Concurrent adapter conversion limit.
    pub binding_copy_concurrency: usize,
    /// Maximum payload in each adapter copy.
    pub binding_max_page_bytes: usize,
    /// Total adapter payload reservation, independent of core's memory ceiling.
    pub binding_reserved_bytes: usize,
    /// Logging attachment availability. Pre-startup log records are never replayed.
    pub logging_coverage: &'static str,
}
impl EmbeddedDiagnosticsReader {
    /// Inspect capacities without allocating a core query workspace.
    pub fn info(&self) -> Result<EmbeddedDiagnosticInfo> {
        check_process(self.pid)?;
        let limits = self.inner.limits().clone();
        let enabled = self.inner.availability() == super::DiagnosticAvailability::Available;
        Ok(EmbeddedDiagnosticInfo {
            discovery: self.inner.discovery(),
            memory: self.inner.memory_usage(),
            binding_copy_concurrency: BINDING_COPY_CONCURRENCY,
            binding_max_page_bytes: limits.max_snapshot_bytes,
            binding_reserved_bytes: if enabled && limits.max_queries > 0 {
                limits
                    .max_snapshot_bytes
                    .checked_mul(BINDING_COPY_CONCURRENCY)
                    .ok_or(DiagnosticError::InvalidLimits)?
            } else {
                0
            },
            logging_coverage: crate::logging::default_diagnostics_coverage(),
            limits,
        })
    }
    /// Capture one admitted copy. Drop the core lease before foreign conversion;
    /// hold this page until conversion finishes to retain adapter admission.
    pub fn snapshot(&self, query: &DiagnosticQuery) -> Result<EmbeddedDiagnosticPage> {
        self.read_page(query, false)
    }
    /// Inspect memory through the same bounded adapter-copy pool.
    pub fn inspect_memory(&self, query: &DiagnosticQuery) -> Result<EmbeddedDiagnosticPage> {
        self.read_page(query, true)
    }
    /// Bounded canonical JSON query for the separate memory view.
    pub fn inspect_memory_json(&self, query: &str) -> Result<EmbeddedDiagnosticPage> {
        check_process(self.pid)?;
        self.inspect_memory(&decode_query(query)?)
    }
    fn read_page(&self, query: &DiagnosticQuery, memory: bool) -> Result<EmbeddedDiagnosticPage> {
        check_process(self.pid)?;
        if self.inner.availability() == super::DiagnosticAvailability::Disabled {
            return Err(DiagnosticError::Disabled.into());
        }
        if self.inner.limits().max_queries == 0 {
            return Err(DiagnosticError::QueryDisabled.into());
        }
        let buffer = self
            .copies
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pop()
            .ok_or(DiagnosticError::QueryCapacity)?;
        let mut page = EmbeddedDiagnosticPage {
            buffer: Some(buffer),
            copies: self.copies.clone(),
            pid: self.pid,
        };
        let snapshot;
        let inspection;
        let bytes = if memory {
            inspection = self.inner.inspect_memory(query)?;
            inspection.as_bytes()
        } else {
            snapshot = self.inner.snapshot(query)?;
            snapshot.as_bytes()
        };
        let buffer = page.buffer.as_mut().expect("admitted buffer");
        // Core's max_snapshot_bytes is also the admitted copy capacity.
        if bytes.len() > buffer.capacity() {
            return Err(DiagnosticError::SnapshotCapacity.into());
        }
        buffer.extend_from_slice(bytes);
        Ok(page)
    }
    /// Bounded canonical JSON query. Missing fields use the core defaults;
    /// unknown fields and oversized input are rejected before capture.
    pub fn snapshot_json(&self, query: &str) -> Result<EmbeddedDiagnosticPage> {
        check_process(self.pid)?;
        self.snapshot(&decode_query(query)?)
    }
}

/// Adapter-owned page payload. Its permit is returned deterministically on drop.
/// Foreign results are caller-owned copies and must never retain this object.
pub struct EmbeddedDiagnosticPage {
    buffer: Option<Vec<u8>>,
    copies: Arc<Mutex<Vec<Vec<u8>>>>,
    pid: u32,
}
impl EmbeddedDiagnosticPage {
    /// Borrow canonical JSON with lossless decimal-string integers.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(self.buffer.as_ref().expect("live page"))
            .expect("core emits UTF-8 JSON")
    }
}
impl Drop for EmbeddedDiagnosticPage {
    fn drop(&mut self) {
        if check_process(self.pid).is_ok() {
            let mut buffer = self.buffer.take().expect("live page");
            buffer.clear();
            self.copies
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(buffer);
        }
    }
}

/// Decode bounded input without copying an unbounded foreign query graph.
pub fn decode_query(input: &str) -> Result<DiagnosticQuery> {
    super::decode_diagnostic_query(input).map_err(Into::into)
}

/// Exact schema fields holding decimal-string DiagnosticNumber values. Binding
/// object conversion uses these fields only on core-produced canonical pages,
/// never on arbitrary application JSON; text and opaque names stay strings.
#[doc(hidden)]
pub fn is_diagnostic_number_field(field: &str) -> bool {
    matches!(
        field,
        "reserved_bytes"
            | "components"
            | "counters"
            | "events"
            | "event_slot_bytes"
            | "event_capacity"
            | "history_reserved_bytes"
            | "history_occupied_bytes"
            | "detail_bytes"
            | "detail_capacity_bytes"
            | "detail_reserved_bytes"
            | "detail_metadata_bytes"
            | "detail_logical_bytes"
            | "detail_slack_bytes"
            | "metadata_bytes"
            | "window_buckets"
            | "extensions"
            | "query_workspaces"
            | "projections"
            | "projection_workspaces"
            | "header_bytes"
            | "detail_payload_bytes"
            | "logical_text_bytes"
            | "occupied_bytes"
            | "registration_bytes"
            | "counter_bytes"
            | "window_bytes"
            | "extension_bytes"
            | "retained_event_count"
            | "retained_event_bytes"
            | "component_occupied_bytes"
            | "entry_occupied_bytes"
            | "shared_or_unassigned_reserved_bytes"
            | "runtime_generation"
            | "target_generation"
            | "ordinal"
            | "collection_millis"
            | "start_identity"
            | "visible_processes"
            | "sampled_processes"
            | "age_millis"
            | "projection_slot_bytes"
            | "value"
            | "sequence"
            | "generation"
            | "refresh_generation"
            | "component_offset"
            | "event_offset"
            | "scope"
            | "count"
            | "submission_sequence"
            | "accepted_elapsed_millis"
            | "elapsed_millis"
            | "since_last_failure_millis"
            | "start_elapsed_millis"
            | "end_elapsed_millis"
            | "resolution_millis"
            | "oldest_event_sequence"
            | "latest_event_sequence"
            | "events_retained"
            | "components_retained"
            | "max_components"
            | "max_events"
            | "max_query_components"
            | "max_query_events"
            | "detail_pool_bytes"
            | "max_queries"
            | "max_snapshot_bytes"
            | "max_owned_bytes"
    )
}

/// Return the binding query contract as JSON for the language wrappers.
#[doc(hidden)]
pub fn binding_query_contract_json() -> String {
    serde_json::json!({
        "max_query_bytes": MAX_BINDING_QUERY_BYTES,
        "numeric_query_fields": DIAGNOSTIC_QUERY_INTEGER_FIELDS,
    })
    .to_string()
}

#[cfg(test)]
#[path = "../../tests/coverage/diagnostics_embedded_tests.rs"]
mod tests;
