// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local diagnostics independent of logging, plugins, and async runtimes.
//!
//! Components receive scoped recording handles. Readers can inspect bounded evidence
//! without changing counters or extending component lifetimes.
//! A projection receives accepted observations after the state lock has been released;
//! observations describe evidence without inferring health or recovery policy.
//! Messages and correlation fields must already be safe for diagnostic exposure:
//! truncation bounds their size and does not remove credentials or user payloads.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use chrono::Utc;
use uuid::Uuid;

mod compatibility;
pub(crate) use compatibility::{legacy_component_key, legacy_diagnostic_code};

pub use nemo_relay_types::diagnostics::*;

const MAX_IDENTITY_BYTES: usize = 128;
const MAX_CODE_BYTES: usize = 128;
const MAX_PROJECTION_DEPTH: usize = 32;

thread_local! {
    // Independent services can forward to each other. Suppress only a service
    // already in this thread's projection chain, or an excessively deep chain.
    static PROJECTING: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Resource budgets for one independently owned diagnostic service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticLimits {
    /// Active and retained closed components, combined.
    pub max_components: usize,
    /// Distinct diagnostic codes retained by each component.
    pub max_keys_per_component: usize,
    /// Recent observations retained for inspection.
    pub max_events: usize,
    /// UTF-8 bytes retained in each message, including a truncation marker.
    pub max_message_bytes: usize,
    /// UTF-8 bytes retained in each correlation field.
    pub max_correlation_bytes: usize,
    /// Combined encoded bytes of retained component and event evidence.
    pub max_retained_bytes: usize,
    /// Maximum encoded size of a query response, including its envelope.
    pub max_snapshot_bytes: usize,
    /// Maximum simultaneously attached operational-log projections.
    pub max_projections: usize,
}

impl Default for DiagnosticLimits {
    fn default() -> Self {
        Self {
            max_components: 128,
            max_keys_per_component: 64,
            max_events: 256,
            max_message_bytes: 4_096,
            max_correlation_bytes: 128,
            max_retained_bytes: 512 * 1_024,
            max_snapshot_bytes: 512 * 1_024,
            max_projections: 16,
        }
    }
}

/// A query narrows the retained view without changing the underlying evidence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiagnosticQuery {
    /// Match a logical component key across retained registration generations.
    pub component_key: Option<String>,
    /// Match one component registration incarnation.
    pub component_instance_id: Option<Uuid>,
    /// Retain only this diagnostic code in events and counters.
    pub code: Option<String>,
    /// Return events strictly after this sequence. Counters remain cumulative.
    pub after_sequence: Option<u64>,
}

/// A validated bounded diagnostic code. Correlation belongs in separate fields.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DiagnosticCode(String);

impl DiagnosticCode {
    /// Validate bounded lowercase code segments separated by dots.
    pub fn new(value: impl Into<String>) -> Result<Self, DiagnosticError> {
        let value = value.into();
        if valid_code(&value) {
            Ok(Self(value))
        } else {
            Err(DiagnosticError::InvalidCode)
        }
    }

    /// Borrow the validated code.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for DiagnosticCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// One counted observation, accepted atomically with its counter update.
#[derive(Debug, Clone)]
pub struct DiagnosticObservation {
    /// Stable code; identifiers such as requests and sessions belong in correlation.
    pub code: String,
    /// Severity of the observed evidence, independent of deployment health policy.
    pub severity: DiagnosticSeverity,
    /// Observed operation outcome, independent of deployment health policy.
    pub outcome: DiagnosticOutcome,
    /// Caller-sanitized explanation; the recorder bounds bytes but does not redact.
    pub message: String,
    /// Occurrences represented by this observation; the count must be nonzero.
    pub count: u64,
    /// Bounded correlation fields, excluded from aggregation keys.
    pub correlation: DiagnosticCorrelation,
}

impl DiagnosticObservation {
    /// Build a counted observation of failed operations.
    pub fn failure(code: impl Into<String>, message: impl Into<String>, count: u64) -> Self {
        Self {
            code: code.into(),
            severity: DiagnosticSeverity::Error,
            outcome: DiagnosticOutcome::Failure,
            message: message.into(),
            count,
            correlation: DiagnosticCorrelation::default(),
        }
    }

    /// Build a counted observation of successful operations.
    pub fn success(code: impl Into<String>, message: impl Into<String>, count: u64) -> Self {
        Self {
            code: code.into(),
            severity: DiagnosticSeverity::Info,
            outcome: DiagnosticOutcome::Success,
            message: message.into(),
            count,
            correlation: DiagnosticCorrelation::default(),
        }
    }

    /// Build a counted observation without classifying its operation outcome.
    pub fn occurrence(
        code: impl Into<String>,
        severity: DiagnosticSeverity,
        message: impl Into<String>,
        count: u64,
    ) -> Self {
        Self {
            code: code.into(),
            severity,
            outcome: DiagnosticOutcome::Unknown,
            message: message.into(),
            count,
            correlation: DiagnosticCorrelation::default(),
        }
    }

    /// Attach separately bounded correlation fields.
    pub fn with_correlation(mut self, correlation: DiagnosticCorrelation) -> Self {
        self.correlation = correlation;
        self
    }
}

/// Short alias for integrations constructing observations directly.
pub type Observation = DiagnosticObservation;

/// Acceptance into local memory is independent of delivery through log sinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordReceipt {
    /// Process-local accepted-update sequence.
    pub sequence: u64,
    /// Cumulative occurrences for this component and code, never a second count.
    pub occurrence_count: u64,
    /// The cumulative count exceeded its finite representation.
    pub count_saturated: bool,
}

/// Validation and capacity failures reported without relying on a logger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiagnosticError {
    #[error("invalid diagnostic component key")]
    /// Component key is empty, too long, or contains unsupported characters.
    InvalidComponentKey,
    #[error("invalid diagnostic code")]
    /// Diagnostic code is empty, too long, or has invalid code segments.
    InvalidCode,
    #[error("an occurrence must have a nonzero count")]
    /// An observation has zero occurrences.
    InvalidCount,
    #[error("diagnostic component capacity exhausted")]
    /// No component slot is available after evicting retained closed components.
    ComponentCapacity,
    #[error("diagnostic key capacity exhausted")]
    /// This component has retained its maximum distinct diagnostic codes.
    KeyCapacity,
    #[error("diagnostic retained byte capacity exhausted")]
    /// The atomic update cannot fit the retained-evidence byte budget.
    ByteCapacity,
    #[error("diagnostic component is closed or no longer retained")]
    /// This registration is closed or has already been evicted.
    ComponentClosed,
    #[error("diagnostic sequence exhausted")]
    /// No unique sequence or registration generation can be allocated.
    SequenceExhausted,
    #[error("diagnostic snapshot budget cannot hold its envelope")]
    /// The configured query budget cannot hold the required snapshot envelope.
    SnapshotCapacity,
    #[error("diagnostic projection capacity exhausted")]
    /// No projection slot or callback registration identifier is available.
    ProjectionCapacity,
}

/// Independently owned diagnostic foundation. Clones share only this service.
#[derive(Clone)]
pub struct Diagnostics {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Diagnostics {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Diagnostics")
            .field("source", &self.inner.source)
            .field("limits", &self.inner.limits)
            .finish_non_exhaustive()
    }
}

struct Inner {
    source: SourceIdentity,
    limits: DiagnosticLimits,
    state: Mutex<State>,
    projections: Mutex<Projections>,
}

#[derive(Default)]
struct State {
    sequence: u64,
    generation: u64,
    components: BTreeMap<Uuid, ComponentSnapshot>,
    component_bytes: BTreeMap<Uuid, usize>,
    events: VecDeque<DiagnosticEvent>,
    event_bytes: VecDeque<usize>,
    // Wide accounting keeps transient replacement totals exact before eviction.
    retained_bytes: u128,
    loss: DiagnosticLoss,
}

type Projection = Arc<dyn Fn(&DiagnosticEvent) + Send + Sync + 'static>;

#[derive(Default)]
struct Projections {
    next_id: u64,
    callbacks: BTreeMap<u64, Projection>,
}

/// Read-only access does not own or extend the lifetime of a component.
#[derive(Clone)]
pub struct DiagnosticsReader {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for DiagnosticsReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DiagnosticsReader")
            .field("source", &self.inner.source)
            .finish_non_exhaustive()
    }
}

/// A component registration. Last-handle drop closes it and retains evidence.
#[derive(Clone)]
pub struct ComponentRecorder {
    lease: Arc<ComponentLease>,
}

impl std::fmt::Debug for ComponentRecorder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ComponentRecorder")
            .field("identity", &self.lease.identity)
            .finish_non_exhaustive()
    }
}

struct ComponentLease {
    inner: Arc<Inner>,
    identity: ComponentIdentity,
}

impl Drop for ComponentLease {
    fn drop(&mut self) {
        close_component(&self.inner, self.identity.instance_id);
    }
}

/// Detaches an observation projection when dropped. It cannot keep a service alive.
/// A callback cloned by an in-flight publisher can still complete after detachment.
pub struct ProjectionGuard {
    inner: Weak<Inner>,
    id: u64,
}

impl std::fmt::Debug for ProjectionGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProjectionGuard")
            .finish_non_exhaustive()
    }
}

impl Drop for ProjectionGuard {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            lock(&inner.projections).callbacks.remove(&self.id);
        }
    }
}

impl Diagnostics {
    /// Construct before logging or runtime initialization. Oversized source labels
    /// are bounded like messages; source incarnation and process ID are preserved.
    pub fn new(mut source: SourceIdentity, limits: DiagnosticLimits) -> Self {
        let mut loss = DiagnosticLoss::default();
        bound_field(&mut source.process_role, MAX_IDENTITY_BYTES, &mut loss);
        bound_field(&mut source.relay_version, MAX_IDENTITY_BYTES, &mut loss);
        Self {
            inner: Arc::new(Inner {
                source,
                limits,
                state: Mutex::new(State {
                    loss,
                    ..State::default()
                }),
                projections: Mutex::new(Projections::default()),
            }),
        }
    }

    /// Optional process service for existing APIs that do not accept explicit ownership.
    pub fn process_default() -> &'static Self {
        process_diagnostics()
    }

    /// Source identity shared by every observation from this service.
    pub fn source(&self) -> &SourceIdentity {
        &self.inner.source
    }

    /// Alias for the source identity used by integration adapters.
    pub fn source_identity(&self) -> &SourceIdentity {
        self.source()
    }

    /// Obtain read-only access without extending a component registration lifetime.
    pub fn reader(&self) -> DiagnosticsReader {
        DiagnosticsReader {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Build a bounded snapshot. Reads never update evidence or coverage counters.
    pub fn snapshot(&self, query: DiagnosticQuery) -> Result<DiagnosticSnapshot, DiagnosticError> {
        self.reader().snapshot(query)
    }

    /// Register a fresh generation, even when the logical key was used before.
    pub fn register_component(
        &self,
        key: impl Into<String>,
    ) -> Result<ComponentRecorder, DiagnosticError> {
        let key = key.into();
        let mut state = lock(&self.inner.state);
        if !valid_component_key(&key) {
            increment(&mut state.loss.components_rejected, 1);
            return Err(DiagnosticError::InvalidComponentKey);
        }
        while state.components.len() >= self.inner.limits.max_components {
            if !evict_closed_component(&mut state, None) {
                increment(&mut state.loss.components_rejected, 1);
                return Err(DiagnosticError::ComponentCapacity);
            }
        }
        let Some(generation) = state.generation.checked_add(1) else {
            state.loss.sequence_saturated = true;
            increment(&mut state.loss.components_rejected, 1);
            return Err(DiagnosticError::SequenceExhausted);
        };
        let sequence = next_sequence(&mut state)?;
        let identity = ComponentIdentity {
            key,
            instance_id: Uuid::now_v7(),
            generation: generation.into(),
        };
        state.components.insert(
            identity.instance_id,
            ComponentSnapshot {
                identity: identity.clone(),
                registered_at: Utc::now(),
                closed_at: None,
                counters: Vec::new(),
            },
        );
        update_component_bytes(&mut state, identity.instance_id);
        if !enforce_byte_budget(&mut state, &self.inner.limits, identity.instance_id) {
            remove_component(&mut state, identity.instance_id);
            increment(&mut state.loss.components_rejected, 1);
            return Err(DiagnosticError::ByteCapacity);
        }
        state.generation = generation;
        state.sequence = sequence;
        Ok(ComponentRecorder {
            lease: Arc::new(ComponentLease {
                inner: Arc::clone(&self.inner),
                identity,
            }),
        })
    }

    /// Callbacks run synchronously after releasing all diagnostic locks. Concurrent
    /// writers may invoke callbacks concurrently and in a different sequence order. They must
    /// be bounded and avoid capturing strong references that form ownership cycles.
    /// Nested delivery to a different service is allowed; service cycles and chains
    /// deeper than 32 are suppressed and counted in coverage metadata.
    pub fn subscribe_projection(
        &self,
        callback: impl Fn(&DiagnosticEvent) + Send + Sync + 'static,
    ) -> Result<ProjectionGuard, DiagnosticError> {
        let mut projections = lock(&self.inner.projections);
        if projections.callbacks.len() >= self.inner.limits.max_projections {
            return Err(DiagnosticError::ProjectionCapacity);
        }
        let Some(id) = projections.next_id.checked_add(1) else {
            return Err(DiagnosticError::ProjectionCapacity);
        };
        projections.next_id = id;
        projections.callbacks.insert(id, Arc::new(callback));
        Ok(ProjectionGuard {
            inner: Arc::downgrade(&self.inner),
            id,
        })
    }
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self::new(
            SourceIdentity {
                instance_id: Uuid::now_v7(),
                process_id: Some(std::process::id()),
                process_role: "runtime".to_owned(),
                relay_version: env!("CARGO_PKG_VERSION").to_owned(),
            },
            DiagnosticLimits::default(),
        )
    }
}

/// The legacy process default is lazy and never reset by logger reconfiguration.
pub fn process_diagnostics() -> &'static Diagnostics {
    static DIAGNOSTICS: OnceLock<Diagnostics> = OnceLock::new();
    DIAGNOSTICS.get_or_init(Diagnostics::default)
}

impl ComponentRecorder {
    /// Identity of this particular component registration.
    pub fn identity(&self) -> &ComponentIdentity {
        &self.lease.identity
    }

    /// Obtain read-only access without extending a component registration lifetime.
    pub fn reader(&self) -> DiagnosticsReader {
        DiagnosticsReader {
            inner: Arc::clone(&self.lease.inner),
        }
    }

    /// Read this registration's evidence, including closed state if retained.
    pub fn snapshot(&self) -> Option<ComponentSnapshot> {
        self.reader()
            .component_snapshot(self.lease.identity.instance_id)
    }

    /// Explicit close is idempotent and preserves retained observations.
    pub fn close(&self) -> bool {
        close_component(&self.lease.inner, self.lease.identity.instance_id)
    }

    /// Atomically count failed operations and record the observation.
    pub fn observe_failure(
        &self,
        code: impl Into<String>,
        message: impl Into<String>,
        count: u64,
        correlation: DiagnosticCorrelation,
    ) -> Result<RecordReceipt, DiagnosticError> {
        self.record(
            DiagnosticObservation::failure(code, message, count).with_correlation(correlation),
        )
    }

    /// Atomically count successful operations and record the observation.
    pub fn observe_success(
        &self,
        code: impl Into<String>,
        message: impl Into<String>,
        count: u64,
        correlation: DiagnosticCorrelation,
    ) -> Result<RecordReceipt, DiagnosticError> {
        self.record(
            DiagnosticObservation::success(code, message, count).with_correlation(correlation),
        )
    }

    /// Count observed evidence without classifying its operation outcome.
    pub fn observe_occurrence(
        &self,
        code: impl Into<String>,
        severity: DiagnosticSeverity,
        message: impl Into<String>,
        count: u64,
        correlation: DiagnosticCorrelation,
    ) -> Result<RecordReceipt, DiagnosticError> {
        self.record(
            DiagnosticObservation::occurrence(code, severity, message, count)
                .with_correlation(correlation),
        )
    }

    /// Accept an observation and notify attached projections after releasing locks.
    pub fn record(
        &self,
        observation: DiagnosticObservation,
    ) -> Result<RecordReceipt, DiagnosticError> {
        self.record_inner(observation, true)
    }

    /// Sink failures use this path to preserve evidence without projecting into
    /// the same failed sink or recursively trying other attached logger adapters.
    pub fn record_without_projection(
        &self,
        observation: DiagnosticObservation,
    ) -> Result<RecordReceipt, DiagnosticError> {
        self.record_inner(observation, false)
    }

    fn record_inner(
        &self,
        mut observation: DiagnosticObservation,
        project: bool,
    ) -> Result<RecordReceipt, DiagnosticError> {
        let inner = &self.lease.inner;
        let id = self.lease.identity.instance_id;
        let mut state = lock(&inner.state);
        if !valid_code(&observation.code) {
            increment(&mut state.loss.keys_rejected, 1);
            return reject(&mut state, DiagnosticError::InvalidCode);
        }
        if observation.count == 0 {
            return reject(&mut state, DiagnosticError::InvalidCount);
        }
        let Some(existing) = state.components.get(&id) else {
            return reject(&mut state, DiagnosticError::ComponentClosed);
        };
        if existing.closed_at.is_some() {
            return reject(&mut state, DiagnosticError::ComponentClosed);
        }
        let known_code = existing
            .counters
            .iter()
            .any(|entry| entry.code == observation.code);
        if !known_code && existing.counters.len() >= inner.limits.max_keys_per_component {
            increment(&mut state.loss.keys_rejected, 1);
            return reject(&mut state, DiagnosticError::KeyCapacity);
        }
        let sequence = next_sequence(&mut state)?;
        bound_field(
            &mut observation.message,
            inner.limits.max_message_bytes,
            &mut state.loss,
        );
        for value in [
            &mut observation.correlation.session_id,
            &mut observation.correlation.request_id,
        ]
        .into_iter()
        .flatten()
        {
            bound_field(value, inner.limits.max_correlation_bytes, &mut state.loss);
        }
        let now = Utc::now();
        let previous = state
            .components
            .get(&id)
            .cloned()
            .expect("component checked above");
        let component = state
            .components
            .get_mut(&id)
            .expect("component checked above");

        if let Some(counter) = component
            .counters
            .iter_mut()
            .find(|entry| entry.code == observation.code)
        {
            increment(&mut counter.occurrences, observation.count);
            counter.last_observed_at = now;
            counter.last_message.clone_from(&observation.message);
        } else {
            component.counters.push(DiagnosticCounter {
                code: observation.code.clone(),
                epoch: id,
                occurrences: DiagnosticCount {
                    value: observation.count.into(),
                    saturated: false,
                },
                last_observed_at: now,
                last_message: observation.message.clone(),
            });
            component
                .counters
                .sort_by(|left, right| left.code.cmp(&right.code));
        }

        let (occurrence_count, count_saturated) = counter_receipt(component, &observation.code);
        let event = DiagnosticEvent {
            sequence: sequence.into(),
            observed_at: now,
            component: self.lease.identity.clone(),
            code: observation.code,
            severity: observation.severity,
            outcome: observation.outcome,
            message: observation.message,
            count: observation.count.into(),
            correlation: observation.correlation,
        };
        update_component_bytes(&mut state, id);
        push_event(&mut state, &inner.limits, event.clone());
        if !enforce_byte_budget(&mut state, &inner.limits, id) {
            state.components.insert(id, previous);
            update_component_bytes(&mut state, id);
            return reject(&mut state, DiagnosticError::ByteCapacity);
        }
        state.sequence = sequence;
        drop(state);
        if project {
            publish(inner, &event);
        }
        Ok(RecordReceipt {
            sequence,
            occurrence_count,
            count_saturated,
        })
    }
}

impl DiagnosticsReader {
    /// Source identity shared by every observation from this service.
    pub fn source(&self) -> &SourceIdentity {
        &self.inner.source
    }

    /// Read one registration within the retained-evidence budget, or return none.
    /// The versioned snapshot response additionally enforces the query byte budget.
    pub fn component_snapshot(&self, instance_id: Uuid) -> Option<ComponentSnapshot> {
        lock(&self.inner.state)
            .components
            .get(&instance_id)
            .cloned()
    }

    /// Build a bounded snapshot. Reads never update evidence or coverage counters.
    pub fn snapshot(&self, query: DiagnosticQuery) -> Result<DiagnosticSnapshot, DiagnosticError> {
        let state = lock(&self.inner.state);
        let mut components = state
            .components
            .values()
            .filter(|component| component_matches(&component.identity, &query))
            .cloned()
            .collect::<Vec<_>>();
        for component in &mut components {
            component
                .counters
                .retain(|counter| query.code.as_ref().is_none_or(|code| code == &counter.code));
        }
        components.sort_by(|left, right| {
            left.identity
                .key
                .cmp(&right.identity.key)
                .then(left.identity.generation.cmp(&right.identity.generation))
        });
        let events = state
            .events
            .iter()
            .filter(|event| component_matches(&event.component, &query))
            .filter(|event| query.code.as_ref().is_none_or(|code| code == &event.code))
            .filter(|event| {
                query
                    .after_sequence
                    .is_none_or(|sequence| event.sequence.get() > sequence)
            })
            .cloned()
            .collect::<Vec<_>>();
        let retained = DiagnosticRetention::default();
        let availability = if state.sequence == 0 {
            DiagnosticAvailability::NeverObserved
        } else {
            DiagnosticAvailability::Available
        };
        let mut snapshot = DiagnosticSnapshot {
            schema_version: DIAGNOSTIC_SCHEMA_VERSION,
            source: self.inner.source.clone(),
            captured_at: Utc::now(),
            sequence: state.sequence.into(),
            components,
            events,
            coverage: DiagnosticCoverage {
                availability,
                retention: retained,
                loss: state.loss.clone(),
            },
        };
        drop(state);
        update_snapshot_retention(&mut snapshot);
        while encoded_bytes(&snapshot) > self.inner.limits.max_snapshot_bytes {
            snapshot.coverage.retention.query_truncated = true;
            if !snapshot.events.is_empty() {
                snapshot.events.remove(0);
            } else if snapshot.components.pop().is_none() {
                return Err(DiagnosticError::SnapshotCapacity);
            }
            update_snapshot_retention(&mut snapshot);
        }
        Ok(snapshot)
    }
}

fn update_snapshot_retention(snapshot: &mut DiagnosticSnapshot) {
    snapshot.coverage.retention.oldest_event_sequence =
        snapshot.events.first().map(|event| event.sequence);
    snapshot.coverage.retention.latest_event_sequence =
        snapshot.events.last().map(|event| event.sequence);
    snapshot.coverage.retention.events_retained = (snapshot.events.len() as u64).into();
    snapshot.coverage.retention.components_retained = (snapshot.components.len() as u64).into();
}

fn component_matches(component: &ComponentIdentity, query: &DiagnosticQuery) -> bool {
    query
        .component_key
        .as_ref()
        .is_none_or(|key| key == &component.key)
        && query
            .component_instance_id
            .is_none_or(|id| id == component.instance_id)
}

fn counter_receipt(component: &ComponentSnapshot, code: &str) -> (u64, bool) {
    component
        .counters
        .iter()
        .find(|counter| counter.code == code)
        .map(|counter| {
            (
                counter.occurrences.value.get(),
                counter.occurrences.saturated,
            )
        })
        .unwrap_or((0, false))
}

fn next_sequence(state: &mut State) -> Result<u64, DiagnosticError> {
    match state.sequence.checked_add(1) {
        Some(sequence) => Ok(sequence),
        None => {
            state.loss.sequence_saturated = true;
            increment(&mut state.loss.observations_rejected, 1);
            Err(DiagnosticError::SequenceExhausted)
        }
    }
}

fn reject<T>(state: &mut State, error: DiagnosticError) -> Result<T, DiagnosticError> {
    increment(&mut state.loss.observations_rejected, 1);
    Err(error)
}

fn increment(count: &mut DiagnosticCount, amount: u64) {
    match count.value.get().checked_add(amount) {
        Some(value) => count.value = value.into(),
        None => {
            count.value = u64::MAX.into();
            count.saturated = true;
        }
    }
}

fn push_event(state: &mut State, limits: &DiagnosticLimits, event: DiagnosticEvent) {
    let bytes = encoded_bytes(&event);
    state.events.push_back(event);
    state.event_bytes.push_back(bytes);
    state.retained_bytes += bytes as u128;
    while state.events.len() > limits.max_events {
        evict_oldest_event(state);
    }
}

fn evict_closed_component(state: &mut State, protected: Option<Uuid>) -> bool {
    let oldest = state
        .components
        .iter()
        .filter(|(id, component)| Some(**id) != protected && component.closed_at.is_some())
        .min_by_key(|(_, component)| component.identity.generation)
        .map(|(id, _)| *id);
    if let Some(id) = oldest {
        remove_component(state, id);
        increment(&mut state.loss.components_evicted, 1);
        true
    } else {
        false
    }
}

fn enforce_byte_budget(state: &mut State, limits: &DiagnosticLimits, protected: Uuid) -> bool {
    while state.retained_bytes > limits.max_retained_bytes as u128 {
        if !evict_oldest_event(state) && !evict_closed_component(state, Some(protected)) {
            return false;
        }
    }
    true
}

// Serialize only the changed component; unrelated components and previously
// retained events never participate in an observation's byte-size calculation.
fn update_component_bytes(state: &mut State, id: Uuid) {
    let bytes = encoded_bytes(&state.components[&id]);
    let previous = state.component_bytes.insert(id, bytes).unwrap_or(0);
    state.retained_bytes = state.retained_bytes - previous as u128 + bytes as u128;
}

fn remove_component(state: &mut State, id: Uuid) {
    state.components.remove(&id);
    if let Some(bytes) = state.component_bytes.remove(&id) {
        state.retained_bytes -= bytes as u128;
    }
}

fn evict_oldest_event(state: &mut State) -> bool {
    if state.events.pop_front().is_none() {
        return false;
    }
    let bytes = state
        .event_bytes
        .pop_front()
        .expect("retained event has a byte size");
    state.retained_bytes -= bytes as u128;
    increment(&mut state.loss.events_evicted, 1);
    true
}

fn encoded_bytes(value: &impl serde::Serialize) -> usize {
    // All shared diagnostic DTOs serialize without user-supplied serializer hooks.
    serde_json::to_vec(value).map_or(usize::MAX, |value| value.len())
}

fn close_component(inner: &Inner, id: Uuid) -> bool {
    let mut state = lock(&inner.state);
    let Some(component) = state.components.get_mut(&id) else {
        return false;
    };
    if component.closed_at.is_some() {
        return false;
    }
    component.closed_at = Some(Utc::now());
    update_component_bytes(&mut state, id);
    if let Ok(sequence) = next_sequence(&mut state) {
        state.sequence = sequence;
    }
    // Closing increases the encoded timestamp size. Preserve the closed component
    // if possible, while allowing bounded retention to evict it under byte pressure.
    while state.retained_bytes > inner.limits.max_retained_bytes as u128 {
        if !evict_oldest_event(&mut state) && !evict_closed_component(&mut state, None) {
            break;
        }
    }
    true
}

fn publish(inner: &Inner, event: &DiagnosticEvent) {
    PROJECTING.with(|projecting| {
        let service = std::ptr::from_ref(inner) as usize;
        let mut chain = projecting.borrow_mut();
        if chain.len() >= MAX_PROJECTION_DEPTH || chain.contains(&service) {
            drop(chain);
            increment(&mut lock(&inner.state).loss.projections_suppressed, 1);
            return;
        }
        chain.push(service);
        drop(chain);
        struct Reset<'a>(&'a RefCell<Vec<usize>>);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.borrow_mut().pop();
            }
        }
        let _reset = Reset(projecting);
        let callbacks = lock(&inner.projections)
            .callbacks
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for callback in callbacks {
            // A projection failure cannot undo an accepted observation or poison state.
            if catch_unwind(AssertUnwindSafe(|| callback(event))).is_err() {
                increment(&mut lock(&inner.state).loss.projections_panicked, 1);
            }
        }
    });
}

fn bound_field(value: &mut String, max_bytes: usize, loss: &mut DiagnosticLoss) {
    if value.len() <= max_bytes {
        return;
    }
    increment(&mut loss.messages_truncated, 1);
    let marker = if max_bytes >= '…'.len_utf8() {
        "…"
    } else {
        ""
    };
    let mut end = max_bytes.saturating_sub(marker.len());
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
    value.push_str(marker);
}

fn valid_code(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_CODE_BYTES
        && value.split('.').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        })
}

fn valid_component_key(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTITY_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
        })
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
