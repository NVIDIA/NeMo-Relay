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
//!     1, // occurrence delta, never a cumulative total
//! ))?;
//! let page = diagnostics.snapshot(DiagnosticQuery::default())?;
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

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Instant;
use uuid::Uuid;

/// Producer-owned pure mappings from existing typed errors to catalogue values.
pub mod classifiers;
pub use classifiers::*;
mod lifecycle;
use lifecycle::{close_component, refresh_component, register_component, register_extension};
mod read;
mod state;
mod storage;

pub use nemo_relay_types::diagnostics::*;
pub use read::SnapshotLease;
use state::*;

const MAX_DETAIL_BYTES: usize = 4096;
const MAX_PROJECTION_DEPTH: usize = 32;

/// Independent capacities of one diagnostic context. All backing allocations are
/// admitted during startup; these limits are not a whole-process RSS ceiling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticLimits {
    /// Active and retained closed registration slots shared by this context.
    pub max_components: usize,
    /// Declared lifetime code-counter slots shared by admitted components.
    pub max_counters: usize,
    /// Shared circular-history headers; zero retains summaries without event history.
    pub max_events: usize,
    /// Aggregate outcome-bucket slots shared by admitted components.
    pub max_window_buckets: usize,
    /// Registered extension descriptors across all retained components.
    pub max_extensions: usize,
    /// Combined admitted component-key, ownership, and extension-description bytes.
    pub max_metadata_bytes: usize,
    /// Bytes admitted per component key or ownership string, at most 128.
    pub max_identity_bytes: usize,
    /// Bytes copied or formatted per observation detail, at most 4096.
    pub max_detail_bytes: usize,
    /// Bytes admitted per request or session correlation field, at most 128.
    pub max_correlation_bytes: usize,
    /// Payload-byte capacity of the shared detail pool, rounded down to whole blocks.
    pub detail_pool_bytes: usize,
    /// Payload bytes per reclaimable detail block, at least four for UTF-8 scalars.
    pub detail_block_bytes: usize,
    /// Simultaneously leased query workspaces; held leases retain their permits.
    pub max_queries: usize,
    /// Maximum component rows materialized in one query page.
    pub max_query_components: usize,
    /// Maximum recent-event rows materialized in one query page.
    pub max_query_events: usize,
    /// Serialized bytes per query response, including its envelope and coverage.
    pub max_snapshot_bytes: usize,
    /// Text-arena bytes reserved for each leased query workspace.
    pub query_text_bytes: usize,
    /// Simultaneously registered callbacks; arbitrary callback captures are caller-owned.
    pub max_projections: usize,
    /// Concurrent publications with independent payload and callback-reference storage.
    pub max_projection_workspaces: usize,
    /// Startup ceiling for context backing allocations, excluding whole-process RSS.
    pub max_owned_bytes: usize,
}
impl Default for DiagnosticLimits {
    fn default() -> Self {
        Self {
            max_components: 64,
            max_counters: 2048,
            max_events: 256,
            max_window_buckets: 3840,
            max_extensions: 128,
            max_metadata_bytes: 65536,
            max_identity_bytes: 128,
            max_detail_bytes: 4096,
            max_correlation_bytes: 128,
            detail_pool_bytes: 262144,
            detail_block_bytes: 256,
            max_queries: 4,
            max_query_components: 64,
            max_query_events: 256,
            max_snapshot_bytes: 524288,
            query_text_bytes: 524288,
            max_projections: 16,
            max_projection_workspaces: 4,
            max_owned_bytes: 32 * 1024 * 1024,
        }
    }
}
impl DiagnosticLimits {
    fn validate(&self) -> Result<(), DiagnosticError> {
        if self.max_components == 0
            || self.max_identity_bytes == 0
            || self.max_identity_bytes > ID_BYTES
            || self.max_detail_bytes > MAX_DETAIL_BYTES
            || self.max_correlation_bytes > ID_BYTES
            || self.detail_block_bytes < 4
            || self.max_queries == 0
            || self.max_query_components == 0
            || self.max_snapshot_bytes < 4096
            || self.max_projection_workspaces == 0
        {
            return Err(DiagnosticError::InvalidLimits);
        }
        let mut total = State::required_bytes(self)?;
        for bytes in [
            read::QueryPool::required_bytes(self)?,
            self.max_projection_workspaces
                .checked_mul(PublicationWorkspace::required_bytes(self)?)
                .ok_or(DiagnosticError::InvalidLimits)?,
            self.max_projection_workspaces
                .checked_mul(size_of::<Option<PublicationWorkspace>>())
                .ok_or(DiagnosticError::InvalidLimits)?,
            self.max_projections
                .checked_mul(size_of::<Option<(u64, Projection)>>())
                .ok_or(DiagnosticError::InvalidLimits)?,
            context_layout_bytes(),
            ID_BYTES * 2,
        ] {
            total = total
                .checked_add(bytes)
                .ok_or(DiagnosticError::InvalidLimits)?;
        }
        if total > self.max_owned_bytes {
            return Err(DiagnosticError::InvalidLimits);
        }
        Ok(())
    }
}

/// Startup-only runtime setting. Collection is compiled normally and off unless
/// the owner enables this context explicitly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticsSettings {
    /// Enable this context at startup; the default is false.
    pub enabled: bool,
    /// Storage capacities validated before an enabled backend is allocated.
    pub limits: DiagnosticLimits,
}

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
    context_id: Uuid,
    component: ComponentHandle,
    slot: usize,
    generation: u64,
}

/// Borrowed producer-approved detail. A Display source is formatted into capped
/// storage only when the enabled recorder invokes its lazy observation factory.
#[derive(Clone, Copy)]
pub enum DiagnosticDetailRef<'a> {
    /// Supply no optional text evidence.
    None,
    /// Borrow already-safe text for a capped copy; bounding does not redact secrets.
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

/// Correlation is borrowed and capped, and never establishes an aggregate key.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiagnosticCorrelationRef<'a> {
    /// Safe optional session identifier, capped independently of message detail.
    pub session_id: Option<&'a str>,
    /// Safe optional request identifier, excluded from aggregate identity.
    pub request_id: Option<&'a str>,
}
/// An admitted custom cause descriptor belonging to one registered extension.
/// Its immutable metadata is bounded and does not create another counter series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensionClassHandle {
    extension: ExtensionHandle,
    index: usize,
}

#[derive(Debug, Clone, Copy)]
enum ObservationCode {
    Builtin(Classification),
    Extension(ExtensionHandle, Option<FailureClass>),
    OpaqueExtension(ExtensionHandle, ExtensionClassHandle),
    Overflow,
}

/// One observation. Its count is an input delta, never a cumulative report total.
#[derive(Debug, Clone, Copy)]
pub struct Observation<'a> {
    code: ObservationCode,
    /// Optional producer-approved explanation; bounding does not perform redaction.
    pub detail: DiagnosticDetailRef<'a>,
    /// Positive input occurrence delta, never a cumulative report total.
    pub count: u64,
    /// Operational evidence severity, independent of deployment health.
    pub severity: DiagnosticSeverity,
    /// Explicit observed operation outcome for summaries and windows.
    pub outcome: DiagnosticOutcome,
    /// Borrowed safe correlation, never an aggregate key.
    pub correlation: DiagnosticCorrelationRef<'a>,
    /// Fixed-size plugin execution and phase facts, excluded from aggregate keys.
    pub context: PluginObservationContext,
}
impl<'a> Observation<'a> {
    /// Author built-in evidence with severity/outcome defaults from its code.
    ///
    /// Recording validates the count and declared code reservation. Constructing
    /// this value does not retain its borrowed detail or source error.
    pub fn new(
        classification: Classification,
        detail: impl Into<DiagnosticDetailRef<'a>>,
        count: u64,
    ) -> Self {
        let (severity, outcome) = match classification.code() {
            BuiltinDiagnosticCode::Failure(_) => {
                (DiagnosticSeverity::Error, DiagnosticOutcome::Failure)
            }
            BuiltinDiagnosticCode::Success(_) => {
                (DiagnosticSeverity::Info, DiagnosticOutcome::Success)
            }
            _ => (DiagnosticSeverity::Info, DiagnosticOutcome::Unknown),
        };
        Self {
            code: ObservationCode::Builtin(classification),
            detail: detail.into(),
            count,
            severity,
            outcome,
            correlation: DiagnosticCorrelationRef::default(),
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
        count: u64,
    ) -> Self {
        Self {
            code: ObservationCode::Extension(handle, class),
            detail: detail.into(),
            count,
            severity: if outcome == DiagnosticOutcome::Failure {
                DiagnosticSeverity::Error
            } else {
                DiagnosticSeverity::Info
            },
            outcome,
            correlation: DiagnosticCorrelationRef::default(),
            context: PluginObservationContext::default(),
        }
    }
    /// Author an extension failure with its own admitted immutable cause name.
    /// Handles from another code/registration or expired metadata are rejected.
    pub fn extension_with_class(
        handle: ExtensionHandle,
        class: ExtensionClassHandle,
        detail: impl Into<DiagnosticDetailRef<'a>>,
        count: u64,
    ) -> Self {
        Self {
            code: ObservationCode::OpaqueExtension(handle, class),
            detail: detail.into(),
            count,
            severity: DiagnosticSeverity::Error,
            outcome: DiagnosticOutcome::Failure,
            correlation: DiagnosticCorrelationRef::default(),
            context: PluginObservationContext::default(),
        }
    }
    /// Explicit summary-only admission when extension vocabulary could not fit.
    /// This is not a way to reinterpret an invalid or undeclared code.
    pub fn classification_overflow(outcome: DiagnosticOutcome, count: u64) -> Self {
        Self {
            code: ObservationCode::Overflow,
            detail: DiagnosticDetailRef::None,
            count,
            severity: DiagnosticSeverity::Warning,
            outcome,
            correlation: DiagnosticCorrelationRef::default(),
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
    pub fn with_correlation(mut self, correlation: DiagnosticCorrelationRef<'a>) -> Self {
        self.correlation = correlation;
        self
    }
}
/// Explicit authoring name for a borrowed observation.
pub type DiagnosticObservation<'a> = Observation<'a>;

/// Evidence retention is separate from numeric acceptance and output delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceDisposition {
    /// The header and supplied optional evidence fit retention budgets.
    Retained,
    /// The retained event has shortened or omitted optional detail/correlation.
    Truncated,
    /// Numeric acceptance succeeded without retaining an incoming event header.
    Dropped,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Accepted lifetime and current-summary update, independent of retention and delivery.
pub struct RecordReceipt {
    /// Unique source-local accepted-update sequence, never an occurrence total.
    pub sequence: u64,
    /// Updated code total, absent for summary-only classification overflow.
    pub code_total: Option<DiagnosticCount>,
    /// Evidence retention disposition, independent of projection delivery.
    pub evidence: EvidenceDisposition,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Accepted collection or intentional startup disablement.
pub enum RecordOutcome {
    /// Accepted numeric state and the independent retention disposition.
    Recorded(RecordReceipt),
    /// No factory, receipt, sequence, or loss accounting was produced.
    Disabled,
}

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
    /// Recording requires a positive input occurrence delta.
    #[error("an observation requires a positive occurrence count")]
    InvalidCount,
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
    /// Every bounded query workspace is currently leased.
    #[error("diagnostic query capacity exhausted")]
    QueryCapacity,
    /// A selector or page request exceeds its admission constraints.
    #[error("diagnostic query is invalid")]
    InvalidQuery,
    /// Continuation has the wrong source/scope or its captured state changed.
    #[error("diagnostic continuation is invalid or its capture changed")]
    InvalidCursor,
    /// The page cannot fit its required envelope in the response budget.
    #[error("diagnostic response budget cannot hold its envelope")]
    SnapshotCapacity,
    /// Subscription slots or unique registration IDs are exhausted.
    #[error("diagnostic projection capacity exhausted")]
    ProjectionCapacity,
    /// The process context was already installed or consumed, even while disabled.
    #[error("the process diagnostic context is already installed or used")]
    AlreadyInitialized,
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

#[derive(Clone)]
/// Independent diagnostics context with immutable startup enablement.
///
/// Clones share this context only. State and budgets are independent of logger
/// lifetime and of any other explicitly isolated embedded contexts.
pub struct Diagnostics {
    inner: Arc<Inner>,
}
struct Inner {
    context_id: Uuid,
    source: SourceIdentity,
    limits: DiagnosticLimits,
    started: Instant,
    backend: Option<Backend>,
}
struct Backend {
    state: Mutex<State>,
    queries: Arc<read::QueryPool>,
    projections: Mutex<Projections>,
    publication: Mutex<Box<[Option<PublicationWorkspace>]>>,
    reserved_bytes: usize,
}
impl Inner {
    fn tick(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}
impl fmt::Debug for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Diagnostics")
            .field("source", &self.inner.source)
            .field("enabled", &self.inner.backend.is_some())
            .finish_non_exhaustive()
    }
}
#[derive(Clone)]
/// Read capability for one context, without recording or lifecycle authority.
///
/// Clones keep the context alive. Materialization requires a bounded lease;
/// disabled collection is explicit instead of fabricated empty evidence.
pub struct DiagnosticsReader {
    inner: Arc<Inner>,
}
impl fmt::Debug for DiagnosticsReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiagnosticsReader")
            .field("source", &self.inner.source)
            .finish_non_exhaustive()
    }
}
#[derive(Clone)]
/// Producer capability scoped to one admitted registration.
///
/// Clones share its lease; the last recorder drop closes the component.
/// Producers cannot refresh or replace registrations. Disabled recorders
/// have no active identity and skip lazy observation factories.
pub struct ComponentRecorder {
    inner: Arc<Inner>,
    lease: Option<Arc<ComponentLease>>,
}
impl fmt::Debug for ComponentRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentRecorder")
            .field("identity", &self.identity())
            .finish_non_exhaustive()
    }
}
struct ComponentLease {
    inner: Weak<Inner>,
    handle: ComponentHandle,
    identity: ComponentIdentity,
}
impl Drop for ComponentLease {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            let _ = close_component(&inner, self.handle);
        }
    }
}
/// Restricted lifecycle capability returned only to the registration owner.
pub struct ComponentOwner {
    inner: Weak<Inner>,
    handle: Option<ComponentHandle>,
}
impl ComponentOwner {
    /// Close this registration without erasing retained evidence.
    ///
    /// Returns false for already closed/unavailable or disabled handles. Later
    /// recording receives an explicit closed-handle failure.
    pub fn close(&self) -> Result<bool, DiagnosticError> {
        match (self.inner.upgrade(), self.handle) {
            (Some(inner), Some(handle)) => close_component(&inner, handle),
            _ => Ok(false),
        }
    }
    /// Reset the current since-refresh summary after a committed reload.
    ///
    /// Lifetime totals, code counters, rolling windows, and retained evidence
    /// remain available. The generation marks the new observation-time period;
    /// captured operations retain their original attribution. Disabled handles
    /// return None without allocating or changing diagnostic state.
    pub fn refresh(
        &self,
        ownership: ComponentOwnership,
    ) -> Result<Option<DiagnosticNumber>, DiagnosticError> {
        let Some(inner) = self.inner.upgrade() else {
            return Err(DiagnosticError::ComponentClosed);
        };
        let Some(handle) = self.handle else {
            return Ok(None);
        };
        refresh_component(&inner, handle, ownership).map(|value| Some(value.into()))
    }
}
/// An operation's bounded original configuration attribution.
///
/// A refresh never invalidates this recorder. Observations update lifetime and
/// current since-refresh totals at recording time while preserving their origin
/// on retained events and projections. Closing the registration remains explicit.
#[derive(Clone, Debug)]
pub struct OriginRecorder {
    recorder: ComponentRecorder,
    origin: Option<CapturedOrigin>,
}
impl OriginRecorder {
    /// Record one completion with its captured origin and optional projections.
    /// The synchronous factory is skipped while collection is disabled.
    pub fn record_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.recorder.record_inner_lazy(make, true, self.origin)
    }
    /// Record with captured origin while bypassing optional output projections.
    pub fn record_without_projection_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.recorder.record_inner_lazy(make, false, self.origin)
    }
}

type Projection =
    Arc<dyn Fn(&DiagnosticEvent) -> Result<(), ProjectionError> + Send + Sync + 'static>;
struct Projections {
    next_id: u64,
    slots: Box<[Option<(u64, Projection)>]>,
}
/// Bounded projection subscription, or inert disabled-mode handle.
///
/// Dropping unsubscribes future publication. A callback copied before removal
/// may still finish; adapters can add stronger shutdown ordering when needed.
pub struct ProjectionGuard {
    inner: Weak<Inner>,
    id: Option<u64>,
}
impl fmt::Debug for ProjectionGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectionGuard").finish_non_exhaustive()
    }
}
impl Drop for ProjectionGuard {
    fn drop(&mut self) {
        if let (Some(inner), Some(id)) = (self.inner.upgrade(), self.id)
            && let Some(backend) = &inner.backend
        {
            let removed = {
                let mut projections = lock(&backend.projections);
                projections
                    .slots
                    .iter_mut()
                    .find(|slot| slot.as_ref().is_some_and(|(key, _)| *key == id))
                    .and_then(Option::take)
            };
            // Captured user values can reenter logging/diagnostics on drop.
            drop(removed);
        }
    }
}

static PROCESS_DIAGNOSTICS: OnceLock<Diagnostics> = OnceLock::new();
impl Diagnostics {
    /// Construct a context using the startup owner's selected mode and capacities.
    ///
    /// Enabled mode validates arithmetic and admits backend storage before use.
    /// The circular-history allocation uses the global allocation-error handler
    /// on allocation failure; fallible pool reservations return an error.
    /// Disabled mode retains lightweight source/settings state without registry,
    /// history, detail, query, or projection pools. Source role/version metadata
    /// is capped at 128 UTF-8 bytes; the supplied UUID is preserved.
    pub fn from_settings(
        source: SourceIdentity,
        settings: DiagnosticsSettings,
    ) -> Result<Self, DiagnosticError> {
        let source = SourceIdentity {
            instance_id: source.instance_id,
            process_id: source.process_id,
            process_role: bounded_str(&source.process_role, ID_BYTES).to_owned(),
            relay_version: bounded_str(&source.relay_version, ID_BYTES).to_owned(),
        };
        let backend = if settings.enabled {
            settings.limits.validate()?;
            let state = State::new(&settings.limits)?;
            let queries =
                read::QueryPool::new(&settings.limits).map_err(|_| DiagnosticError::Allocation)?;
            let mut publication = empty_slots(settings.limits.max_projection_workspaces)?;
            for slot in &mut publication {
                *slot = Some(PublicationWorkspace::new(&settings.limits)?);
            }
            let projections = Projections {
                next_id: 0,
                slots: empty_slots(settings.limits.max_projections)?,
            };
            let reserved_bytes = state
                .reserved_bytes()
                .checked_add(queries.reserved_bytes())
                .and_then(|v| {
                    v.checked_add(
                        publication
                            .iter()
                            .flatten()
                            .map(PublicationWorkspace::reserved_bytes)
                            .sum::<usize>(),
                    )
                })
                .and_then(|v| {
                    v.checked_add(projections.slots.len() * size_of::<Option<(u64, Projection)>>())
                })
                .and_then(|v| {
                    v.checked_add(publication.len() * size_of::<Option<PublicationWorkspace>>())
                })
                .and_then(|v| v.checked_add(context_layout_bytes()))
                .and_then(|v| {
                    v.checked_add(source.process_role.capacity() + source.relay_version.capacity())
                })
                .ok_or(DiagnosticError::InvalidLimits)?;
            if reserved_bytes > settings.limits.max_owned_bytes {
                return Err(DiagnosticError::InvalidLimits);
            }
            Some(Backend {
                state: Mutex::new(state),
                queries,
                projections: Mutex::new(projections),
                publication: Mutex::new(publication),
                reserved_bytes,
            })
        } else {
            None
        };
        Ok(Self {
            inner: Arc::new(Inner {
                context_id: Uuid::now_v7(),
                source,
                limits: settings.limits,
                started: Instant::now(),
                backend,
            }),
        })
    }
    /// Install before handing out default handles. Once used, inert handles are
    /// never silently upgraded to a different startup mode.
    pub fn install_process_default(
        source: SourceIdentity,
        settings: DiagnosticsSettings,
    ) -> Result<&'static Self, DiagnosticError> {
        if PROCESS_DIAGNOSTICS.get().is_some() {
            return Err(DiagnosticError::AlreadyInitialized);
        }
        let context = Self::from_settings(source, settings)?;
        PROCESS_DIAGNOSTICS
            .set(context)
            .map_err(|_| DiagnosticError::AlreadyInitialized)?;
        Ok(PROCESS_DIAGNOSTICS.get().expect("installed default"))
    }
    /// Return the installed process context or initialize a disabled default.
    ///
    /// Install enabled settings before the first lookup. Early disabled handles
    /// are never implicitly upgraded by logger or plugin construction.
    pub fn process_default() -> &'static Self {
        PROCESS_DIAGNOSTICS.get_or_init(Self::default)
    }
    /// Configured capacities, independently of current occupancy or disablement.
    pub fn limits(&self) -> &DiagnosticLimits {
        &self.inner.limits
    }
    /// Source incarnation shared by every observation from this context.
    pub fn source(&self) -> &SourceIdentity {
        &self.inner.source
    }
    /// Issue a read capability without materializing or allocating a query page.
    pub fn reader(&self) -> DiagnosticsReader {
        DiagnosticsReader {
            inner: Arc::clone(&self.inner),
        }
    }
    /// Capture a bounded page through this context's reader.
    ///
    /// The lease holds a query permit until dropped. Reads never reset counters
    /// or pin history; configured disablement is returned explicitly.
    pub fn snapshot(&self, query: DiagnosticQuery) -> Result<SnapshotLease, DiagnosticError> {
        self.reader().snapshot(query)
    }
    /// Reserve a component vocabulary and return its producer lease.
    ///
    /// Admission cannot displace active summaries. The final recorder drop closes
    /// the component. Use [`Self::register_owned_component`] for explicit close
    /// and refresh authority. Disabled mode returns an inert recorder.
    pub fn register_component(
        &self,
        registration: ComponentRegistration<'_>,
    ) -> Result<ComponentRecorder, DiagnosticError> {
        self.register_owned_component(registration)
            .map(|(_, recorder)| recorder)
    }
    /// Reserve a component and separately issue owner and producer capabilities.
    ///
    /// The owner can close or rotate this registration. Disabled mode issues inert
    /// handles without an active identity or backend reservation. Admission
    /// failures are explicit and recorded in reserved loss accounting.
    pub fn register_owned_component(
        &self,
        registration: ComponentRegistration<'_>,
    ) -> Result<(ComponentOwner, ComponentRecorder), DiagnosticError> {
        let Some(backend) = &self.inner.backend else {
            return Ok((
                ComponentOwner {
                    inner: Arc::downgrade(&self.inner),
                    handle: None,
                },
                ComponentRecorder {
                    inner: Arc::clone(&self.inner),
                    lease: None,
                },
            ));
        };
        let mut state = lock(&backend.state);
        let admitted = register_component(&mut state, &self.inner, registration);
        let (handle, identity) = match admitted {
            Ok(value) => value,
            Err(error) => {
                state.loss.components_rejected.add(1);
                return Err(error);
            }
        };
        Ok((
            ComponentOwner {
                inner: Arc::downgrade(&self.inner),
                handle: Some(handle),
            },
            ComponentRecorder {
                inner: Arc::clone(&self.inner),
                lease: Some(Arc::new(ComponentLease {
                    inner: Arc::downgrade(&self.inner),
                    handle,
                    identity,
                })),
            },
        ))
    }
    /// Attach a bounded synchronous subscriber to future accepted typed events.
    ///
    /// Callbacks run outside the store lock, may run concurrently, and have isolated
    /// panics. Disabled mode retains no callback and returns an inert guard.
    /// Arbitrary callback captures and retained copies remain caller-owned.
    pub fn subscribe_projection(
        &self,
        callback: impl Fn(&DiagnosticEvent) + Send + Sync + 'static,
    ) -> Result<ProjectionGuard, DiagnosticError> {
        self.subscribe_fallible_projection(move |event| {
            callback(event);
            Ok(())
        })
    }
    /// Attach a subscriber that reports bounded conversion or delivery failure.
    ///
    /// Errors update projection-loss accounting without undoing numeric acceptance.
    /// Separate publication workspaces cannot pin history. Disabled mode retains
    /// no callback and returns an inert guard.
    pub fn subscribe_fallible_projection(
        &self,
        callback: impl Fn(&DiagnosticEvent) -> Result<(), ProjectionError> + Send + Sync + 'static,
    ) -> Result<ProjectionGuard, DiagnosticError> {
        let Some(backend) = &self.inner.backend else {
            return Ok(ProjectionGuard {
                inner: Weak::new(),
                id: None,
            });
        };
        let mut projections = lock(&backend.projections);
        let index = projections
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or(DiagnosticError::ProjectionCapacity)?;
        let id = projections
            .next_id
            .checked_add(1)
            .ok_or(DiagnosticError::ProjectionCapacity)?;
        projections.next_id = id;
        projections.slots[index] = Some((id, Arc::new(callback)));
        Ok(ProjectionGuard {
            inner: Arc::downgrade(&self.inner),
            id: Some(id),
        })
    }
    /// Inspect reserved backend bytes and occupancy without changing evidence.
    ///
    /// Disabled contexts report zero backend capacity. Lightweight facade state,
    /// caller-owned copies, thread stacks, and independent outputs are excluded.
    pub fn memory_usage(&self) -> DiagnosticMemoryUsage {
        diagnostic_memory_usage(&self.inner)
    }
}
impl Default for Diagnostics {
    fn default() -> Self {
        Self::from_settings(
            SourceIdentity {
                instance_id: Uuid::now_v7(),
                process_id: Some(std::process::id()),
                process_role: "runtime".into(),
                relay_version: env!("CARGO_PKG_VERSION").into(),
            },
            DiagnosticsSettings::default(),
        )
        .expect("disabled construction needs no diagnostic backend")
    }
}
#[derive(Debug, Default, Clone, Copy, Serialize)]
/// Backing capacity and current occupancy of one context.
///
/// These gauges describe the bounded backend rather than whole-process RSS.
/// All values are zero when collection is disabled.
pub struct DiagnosticMemoryUsage {
    /// Bytes reserved for context-owned backend backing storage.
    pub reserved_bytes: usize,
    /// Occupied active or retained closed component slots.
    pub components: usize,
    /// Occupied lifetime code-counter slots across admitted components.
    pub counters: usize,
    /// Currently retained shared circular-history event headers.
    pub events: usize,
    /// Pool payload bytes occupied in whole reclaimable blocks.
    pub detail_bytes: usize,
    /// Admitted key, ownership, and extension metadata content bytes.
    pub metadata_bytes: usize,
    /// Reserved rolling outcome-bucket slots belonging to components.
    pub window_buckets: usize,
    /// Admitted extension descriptors, including retained closed components.
    pub extensions: usize,
    /// Query leases currently holding preallocated materialization buffers.
    pub query_workspaces: usize,
    /// Optional projection subscriptions currently admitted.
    pub projections: usize,
    /// Projection workspaces currently materializing or delivering evidence.
    pub projection_workspaces: usize,
}

impl DiagnosticsReader {
    /// Inspect reserved backing bytes and occupancy without modifying evidence.
    pub fn memory_usage(&self) -> DiagnosticMemoryUsage {
        diagnostic_memory_usage(&self.inner)
    }
    /// Inspect configured capacities separately from actual backing allocation.
    pub fn limits(&self) -> &DiagnosticLimits {
        &self.inner.limits
    }
    /// Capture a bounded consistent page with continuation and coverage.
    ///
    /// A held lease occupies one workspace. Serialization follows store unlocking;
    /// polling neither increments counters nor resets refresh summaries. Disabled collection
    /// returns [`DiagnosticError::Disabled`].
    pub fn snapshot(&self, query: DiagnosticQuery) -> Result<SnapshotLease, DiagnosticError> {
        read::capture(&self.inner, query)
    }
    /// Report configured availability without materializing a snapshot.
    ///
    /// Available means enabled, not complete producer integration or deployment health.
    pub fn availability(&self) -> DiagnosticAvailability {
        if self.inner.backend.is_some() {
            DiagnosticAvailability::Available
        } else {
            DiagnosticAvailability::Disabled
        }
    }
}
impl ComponentRecorder {
    /// Borrow the original registration identity, or None while disabled.
    ///
    /// Closure or retention cannot relabel this handle as another registration.
    pub fn identity(&self) -> Option<&ComponentIdentity> {
        self.lease.as_ref().map(|lease| &lease.identity)
    }
    /// Read the same source context without granting lifecycle authority.
    pub fn reader(&self) -> DiagnosticsReader {
        DiagnosticsReader {
            inner: Arc::clone(&self.inner),
        }
    }
    /// Capture bounded configuration attribution for an operation's completion.
    ///
    /// Refreshing does not pin history or invalidate the captured origin.
    /// Disabled collection returns an inert recorder without allocating.
    pub fn capture_origin(&self) -> Result<OriginRecorder, DiagnosticError> {
        let origin = if let (Some(backend), Some(lease)) = (&self.inner.backend, &self.lease) {
            let state = lock(&backend.state);
            let component = state
                .component(lease.handle)
                .filter(|v| v.closed_at.is_none())
                .ok_or(DiagnosticError::ComponentClosed)?;
            Some(component.origin())
        } else {
            None
        };
        Ok(OriginRecorder {
            recorder: self.clone(),
            origin,
        })
    }
    /// Admit namespaced extension metadata and its component code counter.
    ///
    /// The handle retains registration/descriptor generation scope. Disabled mode
    /// returns None without backend allocation. Explicit capacity failure may be
    /// reported using a summary-only classification overflow observation.
    pub fn register_extension(
        &self,
        registration: ExtensionRegistration<'_>,
    ) -> Result<Option<ExtensionHandle>, DiagnosticError> {
        let (Some(backend), Some(lease)) = (&self.inner.backend, &self.lease) else {
            return Ok(None);
        };
        let mut state = lock(&backend.state);
        register_extension(&mut state, &self.inner, lease.handle, registration).map(Some)
    }
    /// Admit a custom failure-class name for a registered extension. Each code
    /// has four fixed class slots, each at most 32 bytes, within metadata budgets.
    /// Disabled contexts return None without registering metadata.
    pub fn register_extension_class(
        &self,
        extension: ExtensionHandle,
        name: &str,
    ) -> Result<Option<ExtensionClassHandle>, DiagnosticError> {
        let (Some(backend), Some(lease)) = (&self.inner.backend, &self.lease) else {
            return Ok(None);
        };
        lifecycle::register_extension_class(
            &mut lock(&backend.state),
            &self.inner,
            lease.handle,
            extension,
            name,
        )
        .map(Some)
    }
    /// Build and record one observation synchronously only when enabled.
    ///
    /// Reserved summaries accept positive deltas independently of history pressure.
    /// Projections follow store unlocking. Disabled mode skips the factory and
    /// returns [`RecordOutcome::Disabled`] without loss accounting.
    pub fn record_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.record_inner_lazy(make, true, None)
    }
    /// Use the same recording boundary while bypassing every projection.
    ///
    /// Sink feedback and existing operational emitters use this path to avoid output
    /// recursion or duplicates. Disabled mode skips the factory.
    pub fn record_without_projection_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.record_inner_lazy(make, false, None)
    }
    /// Record constructed evidence and attempt optional projections.
    ///
    /// Disabled mode discards it but cannot undo caller construction work.
    /// Built-in producers use [`Self::record_lazy`] for diagnostic-only work.
    pub fn record(&self, observation: Observation<'_>) -> Result<RecordOutcome, DiagnosticError> {
        self.record_lazy(|| observation)
    }
    /// Record constructed evidence without any projection callback.
    ///
    /// Prefer [`Self::record_without_projection_lazy`] to skip construction while
    /// disabled. Counting and retention use the same canonical boundary.
    pub fn record_without_projection(
        &self,
        observation: Observation<'_>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.record_without_projection_lazy(|| observation)
    }
    fn record_inner_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
        project: bool,
        origin: Option<CapturedOrigin>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        let (Some(backend), Some(lease)) = (&self.inner.backend, &self.lease) else {
            return Ok(RecordOutcome::Disabled);
        };
        let observation = make();
        let mut formatted = [0u8; MAX_DETAIL_BYTES];
        let (detail, truncated) = prepare_detail(
            observation.detail,
            &mut formatted,
            self.inner.limits.max_detail_bytes,
        );
        let projection_requested =
            project && lock(&backend.projections).slots.iter().any(Option::is_some);
        let publication = if projection_requested {
            take_publication(backend)
        } else {
            None
        };
        let mut publication = publication;
        let recorded = {
            let mut state = lock(&backend.state);
            let result = record_observation(
                &mut state,
                &self.inner,
                lease.handle,
                origin,
                &observation,
                detail,
                truncated,
                projection_requested,
                publication.as_mut().map(|(_, workspace)| workspace),
            );
            if result.is_err() {
                state.loss.observations_rejected.add(1);
            }
            result
        };
        if let Some((index, mut workspace)) = publication {
            if recorded.is_ok() && workspace.ready {
                publish(&self.inner, &mut workspace);
            }
            workspace.reset();
            lock(&backend.publication)[index] = Some(workspace);
        }
        recorded.map(RecordOutcome::Recorded)
    }
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
fn bounded_str(value: &str, max: usize) -> &str {
    let mut end = value.len().min(max);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn prepare_detail<'a>(
    detail: DiagnosticDetailRef<'a>,
    buffer: &'a mut [u8],
    max: usize,
) -> (Option<&'a str>, bool) {
    match detail {
        DiagnosticDetailRef::None => (None, false),
        DiagnosticDetailRef::Text(text) => {
            let value = bounded_str(text, max);
            (Some(value), value.len() != text.len())
        }
        DiagnosticDetailRef::Display(value) => {
            struct BoundedFormat<'a> {
                buffer: &'a mut [u8],
                len: usize,
                truncated: bool,
            }
            impl fmt::Write for BoundedFormat<'_> {
                fn write_str(&mut self, value: &str) -> fmt::Result {
                    let part = bounded_str(value, self.buffer.len().saturating_sub(self.len));
                    self.buffer[self.len..self.len + part.len()].copy_from_slice(part.as_bytes());
                    self.len += part.len();
                    if part.len() < value.len() {
                        self.truncated = true;
                        Err(fmt::Error)
                    } else {
                        Ok(())
                    }
                }
            }
            let mut output = BoundedFormat {
                buffer: &mut buffer[..max],
                len: 0,
                truncated: false,
            };
            let failed = fmt::write(&mut output, format_args!("{value}")).is_err();
            let len = output.len;
            let truncated = output.truncated || failed;
            (
                Some(std::str::from_utf8(&buffer[..len]).expect("formatter copies complete UTF-8")),
                truncated,
            )
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep the borrowed recording inputs explicit at the single acceptance boundary."
)]
fn record_observation(
    state: &mut State,
    inner: &Inner,
    handle: ComponentHandle,
    requested_origin: Option<CapturedOrigin>,
    observation: &Observation<'_>,
    detail: Option<&str>,
    truncated: bool,
    projection_requested: bool,
    publication: Option<&mut PublicationWorkspace>,
) -> Result<RecordReceipt, DiagnosticError> {
    if observation.count == 0 {
        return Err(DiagnosticError::InvalidCount);
    }
    let component = match state
        .component(handle)
        .filter(|v| v.closed_at.is_none())
        .copied()
    {
        Some(component) => component,
        None => {
            state.loss.stale_observations.add(1);
            return Err(DiagnosticError::ComponentClosed);
        }
    };
    let origin = requested_origin.unwrap_or_else(|| component.origin());
    let (code, class) = match observation.code {
        ObservationCode::Builtin(classification) => (
            Some(StoredCode::Builtin(classification.code())),
            classification
                .failure_class()
                .map(StoredFailureClass::Builtin),
        ),
        ObservationCode::Extension(extension, class) => {
            if extension.context_id != inner.context_id || extension.component != handle {
                return Err(DiagnosticError::InvalidCode);
            }
            let descriptor = state
                .extensions
                .get(extension.slot)
                .and_then(Option::as_ref)
                .filter(|v| v.generation == extension.generation && v.component == handle)
                .ok_or(DiagnosticError::InvalidCode)?;
            if descriptor.outcome == DiagnosticOutcome::Failure {
                if class.is_none_or(|class| {
                    descriptor.allowed_classes & lifecycle::class_bit(class) == 0
                }) {
                    return Err(DiagnosticError::InvalidClassification);
                }
            } else if class.is_some() {
                return Err(DiagnosticError::InvalidClassification);
            }
            (
                Some(StoredCode::Extension {
                    slot: extension.slot,
                    generation: extension.generation,
                }),
                class.map(StoredFailureClass::Builtin),
            )
        }
        ObservationCode::OpaqueExtension(extension, class) => {
            if extension.context_id != inner.context_id
                || extension.component != handle
                || class.extension != extension
            {
                return Err(DiagnosticError::InvalidClassification);
            }
            let descriptor = state
                .extensions
                .get(extension.slot)
                .and_then(Option::as_ref)
                .filter(|v| v.generation == extension.generation && v.component == handle)
                .ok_or(DiagnosticError::InvalidCode)?;
            let cause = StoredFailureClass::Extension(class);
            if descriptor.outcome != DiagnosticOutcome::Failure || state.class_name(cause).is_none()
            {
                return Err(DiagnosticError::InvalidClassification);
            }
            (
                Some(StoredCode::Extension {
                    slot: extension.slot,
                    generation: extension.generation,
                }),
                Some(cause),
            )
        }
        ObservationCode::Overflow => (None, None),
    };
    let counter_index = if let Some(code) = code {
        Some(
            state
                .counters
                .iter()
                .position(|v| {
                    v.as_ref()
                        .is_some_and(|v| v.component == handle && v.code == code)
                })
                .ok_or(DiagnosticError::InvalidCode)?,
        )
    } else {
        None
    };
    let sequence = state.next_sequence()?;
    let observed_at = Utc::now();
    let tick = inner.tick();
    state
        .lifetime
        .record(observation.outcome, observation.count, observed_at, tick);
    let current = state.components[handle.slot]
        .as_mut()
        .expect("validated component");
    current
        .lifetime
        .record(observation.outcome, observation.count, observed_at, tick);
    current
        .refresh
        .totals
        .record(observation.outcome, observation.count);
    state.record_window(handle, tick, observation.outcome, observation.count);
    let code_total = counter_index.map(|index| {
        let counter = state.counters[index].as_mut().expect("declared counter");
        counter.occurrences.add(observation.count);
        counter.last_observed_at = Some(observed_at);
        counter.occurrences
    });
    let Some(code) = code else {
        state.loss.classifications_overflowed.add(observation.count);
        state.loss.events_omitted.add(1);
        return Ok(RecordReceipt {
            sequence,
            code_total: None,
            evidence: EvidenceDisposition::Dropped,
        });
    };
    if truncated {
        state.loss.messages_truncated.add(1);
    }
    let session = observation
        .correlation
        .session_id
        .map(|value| bounded_str(value, inner.limits.max_correlation_bytes));
    let request = observation
        .correlation
        .request_id
        .map(|value| bounded_str(value, inner.limits.max_correlation_bytes));
    let session_truncated = observation
        .correlation
        .session_id
        .zip(session)
        .is_some_and(|(a, b)| a.len() != b.len());
    let request_truncated = observation
        .correlation
        .request_id
        .zip(request)
        .is_some_and(|(a, b)| a.len() != b.len());
    if session_truncated {
        state.loss.correlations_truncated.add(1);
    }
    if request_truncated {
        state.loss.correlations_truncated.add(1);
    }
    if let Some(workspace) = publication {
        workspace.fill(
            state,
            &component,
            origin,
            code,
            class,
            sequence,
            observed_at,
            observation,
            detail,
            truncated,
            session,
            request,
        );
    } else if projection_requested {
        state.loss.projection_deliveries_omitted.add(1);
    }
    let evidence = if inner.limits.max_events == 0 {
        state.loss.events_omitted.add(1);
        EvidenceDisposition::Dropped
    } else {
        let retained_detail =
            detail.and_then(|text| state.allocate_detail(text, inner.limits.max_detail_bytes));
        if detail.is_some() && retained_detail.is_none() {
            state.loss.details_omitted.add(1);
        }
        let retained_session = session
            .and_then(|text| state.allocate_detail(text, inner.limits.max_correlation_bytes));
        let retained_request = request
            .and_then(|text| state.allocate_detail(text, inner.limits.max_correlation_bytes));
        let correlation_omitted = (session.is_some() && retained_session.is_none()) as u64
            + (request.is_some() && retained_request.is_none()) as u64;
        state.loss.correlations_omitted.add(correlation_omitted);
        let event = StoredObservation {
            sequence,
            observed_at,
            count: observation.count,
            component: handle,
            origin,
            code,
            failure_class: class,
            context: observation.context,
            severity: observation.severity,
            outcome: observation.outcome,
            detail: retained_detail,
            detail_truncated: truncated,
            session_id: retained_session,
            request_id: retained_request,
        };
        if let Some(evicted) = state.recent.push_back(event) {
            state.evict_event(evicted);
        }
        if truncated || detail.is_some() && retained_detail.is_none() || correlation_omitted > 0 {
            EvidenceDisposition::Truncated
        } else {
            EvidenceDisposition::Retained
        }
    };
    Ok(RecordReceipt {
        sequence,
        code_total,
        evidence,
    })
}

struct PublicationWorkspace {
    event: DiagnosticEvent,
    detail: String,
    class: String,
    session: String,
    request: String,
    origin_configuration: String,
    origin_responsibility: String,
    callbacks: Box<[Option<Projection>]>,
    ready: bool,
}
fn reserved_string(capacity: usize) -> Result<String, DiagnosticError> {
    let mut value = String::new();
    value
        .try_reserve_exact(capacity)
        .map_err(|_| DiagnosticError::Allocation)?;
    Ok(value)
}
impl PublicationWorkspace {
    fn required_bytes(limits: &DiagnosticLimits) -> Result<usize, DiagnosticError> {
        let text = ID_BYTES
            .checked_mul(4)
            .and_then(|v| v.checked_add(32))
            .and_then(|v| v.checked_add(limits.max_detail_bytes))
            .and_then(|v| v.checked_add(limits.max_correlation_bytes.checked_mul(2)?))
            .ok_or(DiagnosticError::InvalidLimits)?;
        text.checked_add(
            limits
                .max_projections
                .checked_mul(size_of::<Option<Projection>>())
                .ok_or(DiagnosticError::InvalidLimits)?,
        )
        .ok_or(DiagnosticError::InvalidLimits)
    }
    fn new(limits: &DiagnosticLimits) -> Result<Self, DiagnosticError> {
        Ok(Self {
            event: DiagnosticEvent {
                sequence: 0.into(),
                observed_at: Utc::now(),
                component: ComponentIdentity {
                    key: reserved_string(ID_BYTES)?,
                    instance_id: Uuid::nil(),
                    generation: 0.into(),
                },
                origin: DiagnosticOrigin {
                    refresh_generation: 0.into(),
                    ownership: ComponentOwnership::default(),
                },
                classification: ClassificationWire {
                    canonical_code: reserved_string(ID_BYTES)?,
                    failure_class: None,
                },
                severity: DiagnosticSeverity::Info,
                outcome: DiagnosticOutcome::Unknown,
                context: PluginObservationContext::default(),
                detail: None,
                count: 0.into(),
                correlation: DiagnosticCorrelation::default(),
            },
            detail: reserved_string(limits.max_detail_bytes)?,
            class: reserved_string(32)?,
            session: reserved_string(limits.max_correlation_bytes)?,
            request: reserved_string(limits.max_correlation_bytes)?,
            origin_configuration: reserved_string(ID_BYTES)?,
            origin_responsibility: reserved_string(ID_BYTES)?,
            callbacks: empty_slots(limits.max_projections)?,
            ready: false,
        })
    }
    fn reserved_bytes(&self) -> usize {
        self.event.component.key.capacity()
            + self.event.classification.canonical_code.capacity()
            + self.detail.capacity()
            + self.class.capacity()
            + self.session.capacity()
            + self.request.capacity()
            + self.origin_configuration.capacity()
            + self.origin_responsibility.capacity()
            + self.callbacks.len() * size_of::<Option<Projection>>()
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Copy validated observation fields into the existing projection workspace."
    )]
    fn fill(
        &mut self,
        state: &State,
        component: &ComponentSlot,
        origin: CapturedOrigin,
        code: StoredCode,
        class: Option<StoredFailureClass>,
        sequence: u64,
        observed_at: ObservationTime,
        observation: &Observation<'_>,
        detail: Option<&str>,
        truncated: bool,
        session: Option<&str>,
        request: Option<&str>,
    ) {
        self.event.sequence = sequence.into();
        self.event.observed_at = observed_at;
        self.event.component.key.clear();
        self.event.component.key.push_str(component.key.as_str());
        self.event.component.instance_id = component.instance_id;
        self.event.component.generation = component.handle.generation.into();
        self.event.origin.refresh_generation = origin.generation.into();
        self.event.origin.ownership.subscriber_instance_id =
            origin.ownership.subscriber_instance_id;
        self.event.origin.ownership.plugin_instance_id = origin.ownership.plugin_instance_id;
        if let Some(value) = origin.ownership.configuration_id {
            self.origin_configuration.clear();
            self.origin_configuration.push_str(value.as_str());
            self.event.origin.ownership.configuration_id =
                Some(std::mem::take(&mut self.origin_configuration));
        }
        if let Some(value) = origin.ownership.responsibility {
            self.origin_responsibility.clear();
            self.origin_responsibility.push_str(value.as_str());
            self.event.origin.ownership.responsibility =
                Some(std::mem::take(&mut self.origin_responsibility));
        }
        self.event.classification.canonical_code.clear();
        self.event
            .classification
            .canonical_code
            .push_str(state.code_name(code).expect("admitted code"));
        if let Some(class) = class {
            self.class.clear();
            self.class
                .push_str(state.class_name(class).expect("admitted class"));
            self.event.classification.failure_class = Some(std::mem::take(&mut self.class));
        }
        self.event.context = observation.context;
        self.event.severity = observation.severity;
        self.event.outcome = observation.outcome;
        self.event.count = observation.count.into();
        if let Some(detail) = detail {
            self.detail.clear();
            self.detail.push_str(detail);
            self.event.detail = Some(DiagnosticDetail {
                message: std::mem::take(&mut self.detail),
                truncated,
            });
        }
        if let Some(value) = session {
            self.session.clear();
            self.session.push_str(value);
            self.event.correlation.session_id = Some(std::mem::take(&mut self.session));
        }
        if let Some(value) = request {
            self.request.clear();
            self.request.push_str(value);
            self.event.correlation.request_id = Some(std::mem::take(&mut self.request));
        }
        self.ready = true;
    }
    fn reset(&mut self) {
        if let Some(value) = self.event.origin.ownership.configuration_id.take() {
            self.origin_configuration = value;
            self.origin_configuration.clear();
        }
        if let Some(value) = self.event.origin.ownership.responsibility.take() {
            self.origin_responsibility = value;
            self.origin_responsibility.clear();
        }
        if let Some(value) = self.event.detail.take() {
            self.detail = value.message;
            self.detail.clear();
        }
        if let Some(value) = self.event.classification.failure_class.take() {
            self.class = value;
            self.class.clear();
        }
        if let Some(value) = self.event.correlation.session_id.take() {
            self.session = value;
            self.session.clear();
        }
        if let Some(value) = self.event.correlation.request_id.take() {
            self.request = value;
            self.request.clear();
        }
        for callback in &mut self.callbacks {
            *callback = None;
        }
        self.ready = false;
    }
}
fn take_publication(backend: &Backend) -> Option<(usize, PublicationWorkspace)> {
    let (index, mut workspace) = {
        let mut pool = lock(&backend.publication);
        let index = pool.iter().position(Option::is_some)?;
        (index, pool[index].take().expect("available workspace"))
    };
    let projections = lock(&backend.projections);
    for (target, source) in workspace.callbacks.iter_mut().zip(projections.slots.iter()) {
        *target = source.as_ref().map(|(_, callback)| Arc::clone(callback));
    }
    Some((index, workspace))
}
thread_local! {static PROJECTING:RefCell<(usize,[usize;MAX_PROJECTION_DEPTH])>=const{RefCell::new((0,[0;MAX_PROJECTION_DEPTH]))};}
struct ProjectionScope;
impl Drop for ProjectionScope {
    fn drop(&mut self) {
        PROJECTING.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.0 -= 1;
        });
    }
}
fn publish(inner: &Inner, workspace: &mut PublicationWorkspace) {
    let service = std::ptr::from_ref(inner) as usize;
    let entered = PROJECTING.with(|stack| {
        let mut stack = stack.borrow_mut();
        if stack.0 == MAX_PROJECTION_DEPTH || stack.1[..stack.0].contains(&service) {
            false
        } else {
            let index = stack.0;
            stack.1[index] = service;
            stack.0 += 1;
            true
        }
    });
    let Some(backend) = &inner.backend else {
        return;
    };
    if !entered {
        lock(&backend.state).loss.projections_suppressed.add(1);
        return;
    }
    let _scope = ProjectionScope;
    for callback in workspace.callbacks.iter().flatten() {
        match catch_unwind(AssertUnwindSafe(|| callback(&workspace.event))) {
            Ok(Ok(())) => {}
            Ok(Err(_)) => lock(&backend.state)
                .loss
                .projection_deliveries_omitted
                .add(1),
            Err(_) => lock(&backend.state).loss.projections_panicked.add(1),
        }
    }
}

fn diagnostic_memory_usage(inner: &Inner) -> DiagnosticMemoryUsage {
    let Some(backend) = &inner.backend else {
        return DiagnosticMemoryUsage::default();
    };
    let state = lock(&backend.state);
    DiagnosticMemoryUsage {
        reserved_bytes: backend.reserved_bytes,
        components: state.components.iter().flatten().count(),
        counters: state.counters.iter().flatten().count(),
        events: state.recent.len(),
        detail_bytes: state.detail.used_bytes(),
        metadata_bytes: state.metadata_used,
        window_buckets: state.windows.iter().flatten().count(),
        extensions: state.extensions.iter().flatten().count(),
        query_workspaces: backend.queries.occupancy(),
        projections: lock(&backend.projections).slots.iter().flatten().count(),
        projection_workspaces: lock(&backend.publication)
            .iter()
            .filter(|v| v.is_none())
            .count(),
    }
}

// Arc storage contains its two reference counts and aligned Inner payload. This
// includes the inline State/Backend mutex headers; dynamic pools are added once.
fn context_layout_bytes() -> usize {
    std::alloc::Layout::new::<[usize; 2]>()
        .extend(std::alloc::Layout::new::<Inner>())
        .expect("diagnostic context layout fits")
        .0
        .pad_to_align()
        .size()
}
