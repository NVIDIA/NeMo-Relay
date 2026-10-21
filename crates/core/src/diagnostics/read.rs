// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leased query materialization. Copied pages and serialized output use reserved
//! workspaces; owned DTO decoding is an explicit caller-owned operation.

use std::collections::TryReserveError;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::ser::{Serialize, SerializeSeq, SerializeStruct, Serializer};
use uuid::Uuid;

use super::storage::CappedWriter;
use super::*;

#[derive(Clone, Copy)]
struct TextRef {
    offset: usize,
    len: usize,
}

struct PageIdentity {
    key: TextRef,
    instance_id: Uuid,
    generation: DiagnosticNumber,
}
struct PageOwnership {
    subscriber_instance_id: Option<Uuid>,
    plugin_instance_id: Option<Uuid>,
    configuration_id: Option<TextRef>,
    responsibility: Option<TextRef>,
}
struct PageComponent {
    source_position: usize,
    text_start: usize,
    identity: PageIdentity,
    ownership: PageOwnership,
    registered_at: DateTime<Utc>,
    closed_at: Option<DateTime<Utc>>,
    lifetime: DiagnosticLifetimeSummary,
    refresh_generation: DiagnosticNumber,
    refresh_started_at: ObservationTime,
    refresh_elapsed_millis: DiagnosticNumber,
    refresh_totals: DiagnosticTotals,
    counter_start: usize,
    counter_len: usize,
    window_start: usize,
    window_len: usize,
}
struct PageCounter {
    code: TextRef,
    occurrences: DiagnosticCount,
    last_observed_at: DateTime<Utc>,
}
struct PageWindow {
    resolution_millis: DiagnosticNumber,
    bucket_capacity: u32,
    bucket_start: usize,
    bucket_len: usize,
    coverage: DiagnosticWindowCoverage,
}
struct PageEvent {
    source_position: usize,
    text_start: usize,
    sequence: DiagnosticNumber,
    observed_at: DateTime<Utc>,
    component: PageIdentity,
    origin_generation: DiagnosticNumber,
    ownership: PageOwnership,
    code: TextRef,
    failure_class: Option<TextRef>,
    context: StoredPluginContext,
    severity: DiagnosticSeverity,
    outcome: DiagnosticOutcome,
    detail: Option<TextRef>,
    detail_truncated: bool,
    count: DiagnosticNumber,
    session_id: Option<TextRef>,
    request_id: Option<TextRef>,
}

/// One preallocated query page. The public DTO graph is deliberately not held
/// here: text references and flat arrays resolve through borrowed serializers.
pub(super) struct PageWorkspace {
    components: Vec<PageComponent>,
    counters: Vec<PageCounter>,
    windows: Vec<PageWindow>,
    buckets: Vec<DiagnosticWindowBucket>,
    events: Vec<PageEvent>,
    text: Box<[u8]>,
    text_len: usize,
    output: Box<[u8]>,
    output_len: usize,
}

impl PageWorkspace {
    fn new(limits: &DiagnosticLimits) -> Result<Self, TryReserveError> {
        Ok(Self {
            components: reserved(limits.max_query_components)?,
            counters: reserved(limits.max_counters)?,
            windows: reserved(limits.max_query_components)?,
            buckets: reserved(limits.max_window_buckets)?,
            events: reserved(limits.max_query_events)?,
            text: bytes(limits.query_text_bytes)?,
            text_len: 0,
            output: bytes(limits.max_snapshot_bytes)?,
            output_len: 0,
        })
    }
    fn clear(&mut self) {
        self.components.clear();
        self.counters.clear();
        self.windows.clear();
        self.buckets.clear();
        self.events.clear();
        self.text_len = 0;
        self.output_len = 0;
    }
    fn reserved_bytes(&self) -> usize {
        self.components.capacity() * std::mem::size_of::<PageComponent>()
            + self.counters.capacity() * std::mem::size_of::<PageCounter>()
            + self.windows.capacity() * std::mem::size_of::<PageWindow>()
            + self.buckets.capacity() * std::mem::size_of::<DiagnosticWindowBucket>()
            + self.events.capacity() * std::mem::size_of::<PageEvent>()
            + self.text.len()
            + self.output.len()
    }
    fn copy_text(&mut self, value: &str) -> Option<TextRef> {
        let end = self.text_len.checked_add(value.len())?;
        if end > self.text.len() {
            return None;
        }
        let reference = TextRef {
            offset: self.text_len,
            len: value.len(),
        };
        self.text[self.text_len..end].copy_from_slice(value.as_bytes());
        self.text_len = end;
        Some(reference)
    }
    fn string(&self, reference: TextRef) -> &str {
        std::str::from_utf8(&self.text[reference.offset..reference.offset + reference.len])
            .expect("copied diagnostic text is UTF-8")
    }
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>, TryReserveError> {
    let mut result = Vec::new();
    result.try_reserve_exact(capacity)?;
    Ok(result)
}
fn bytes(capacity: usize) -> Result<Box<[u8]>, TryReserveError> {
    let mut result = reserved(capacity)?;
    result.resize(capacity, 0u8);
    Ok(result.into_boxed_slice())
}

pub(super) struct QueryPool {
    slots: Mutex<Box<[Option<PageWorkspace>]>>,
    reserved_bytes: usize,
}
impl QueryPool {
    pub(super) fn required_bytes(limits: &DiagnosticLimits) -> Result<usize, DiagnosticError> {
        let mut per_page = 0usize;
        for (count, size) in [
            (
                limits.max_query_components,
                std::mem::size_of::<PageComponent>(),
            ),
            (limits.max_counters, std::mem::size_of::<PageCounter>()),
            (
                limits.max_query_components,
                std::mem::size_of::<PageWindow>(),
            ),
            (
                limits.max_window_buckets,
                std::mem::size_of::<DiagnosticWindowBucket>(),
            ),
            (limits.max_query_events, std::mem::size_of::<PageEvent>()),
            (limits.query_text_bytes, 1),
            (limits.max_snapshot_bytes, 1),
        ] {
            per_page = per_page
                .checked_add(
                    count
                        .checked_mul(size)
                        .ok_or(DiagnosticError::InvalidLimits)?,
                )
                .ok_or(DiagnosticError::InvalidLimits)?;
        }
        let header = std::alloc::Layout::new::<[usize; 2]>()
            .extend(std::alloc::Layout::new::<Self>())
            .map_err(|_| DiagnosticError::InvalidLimits)?
            .0
            .pad_to_align()
            .size();
        let slots = limits
            .max_queries
            .checked_mul(std::mem::size_of::<Option<PageWorkspace>>())
            .ok_or(DiagnosticError::InvalidLimits)?;
        header
            .checked_add(slots)
            .and_then(|n| {
                limits
                    .max_queries
                    .checked_mul(per_page)
                    .and_then(|pages| n.checked_add(pages))
            })
            .ok_or(DiagnosticError::InvalidLimits)
    }
    pub(super) fn new(limits: &DiagnosticLimits) -> Result<Arc<Self>, TryReserveError> {
        let mut slots = reserved(limits.max_queries)?;
        for _ in 0..limits.max_queries {
            slots.push(Some(PageWorkspace::new(limits)?));
        }
        let reserved_bytes = std::alloc::Layout::new::<[usize; 2]>()
            .extend(std::alloc::Layout::new::<Self>())
            .expect("query pool layout fits")
            .0
            .pad_to_align()
            .size()
            + slots.capacity() * std::mem::size_of::<Option<PageWorkspace>>()
            + slots
                .iter()
                .flatten()
                .map(PageWorkspace::reserved_bytes)
                .sum::<usize>();
        Ok(Arc::new(Self {
            slots: Mutex::new(slots.into_boxed_slice()),
            reserved_bytes,
        }))
    }
    pub(super) fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }
    pub(super) fn occupancy(&self) -> usize {
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|slot| slot.is_none())
            .count()
    }
    fn acquire(self: &Arc<Self>) -> Option<SnapshotLease> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let (index, slot) = slots.iter_mut().enumerate().find(|(_, s)| s.is_some())?;
        let mut workspace = slot.take()?;
        workspace.clear();
        Some(SnapshotLease {
            pool: Arc::clone(self),
            index,
            workspace: Some(workspace),
        })
    }
}

/// A bounded serialized page. Dropping the lease returns its workspace permit;
/// it owns copied bytes and never pins retained history or a component lease.
pub struct SnapshotLease {
    pool: Arc<QueryPool>,
    index: usize,
    workspace: Option<PageWorkspace>,
}
impl SnapshotLease {
    /// Borrow the canonical JSON response while retaining its workspace permit.
    pub fn as_bytes(&self) -> &[u8] {
        let page = self.workspace.as_ref().expect("live query workspace");
        &page.output[..page.output_len]
    }
    /// Decode an independently owned snapshot. These new allocations belong to
    /// the caller and are outside the diagnostic service's storage guarantee.
    pub fn to_snapshot(&self) -> Result<DiagnosticSnapshot, serde_json::Error> {
        serde_json::from_slice(self.as_bytes())
    }
}
impl std::fmt::Debug for SnapshotLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotLease")
            .field("bytes", &self.as_bytes().len())
            .finish_non_exhaustive()
    }
}
impl Drop for SnapshotLease {
    fn drop(&mut self) {
        let mut slots = self.pool.slots.lock().unwrap_or_else(|e| e.into_inner());
        slots[self.index] = self.workspace.take();
    }
}

struct IdentityView<'a> {
    page: &'a PageWorkspace,
    identity: &'a PageIdentity,
}
impl Serialize for IdentityView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("ComponentIdentity", 3)?;
        out.serialize_field("key", self.page.string(self.identity.key))?;
        out.serialize_field("instance_id", &self.identity.instance_id)?;
        out.serialize_field("generation", &self.identity.generation)?;
        out.end()
    }
}
struct OwnershipView<'a> {
    page: &'a PageWorkspace,
    ownership: &'a PageOwnership,
}
impl Serialize for OwnershipView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("ComponentOwnership", 4)?;
        out.serialize_field(
            "subscriber_instance_id",
            &self.ownership.subscriber_instance_id,
        )?;
        out.serialize_field("plugin_instance_id", &self.ownership.plugin_instance_id)?;
        out.serialize_field(
            "configuration_id",
            &self.ownership.configuration_id.map(|v| self.page.string(v)),
        )?;
        out.serialize_field(
            "responsibility",
            &self.ownership.responsibility.map(|v| self.page.string(v)),
        )?;
        out.end()
    }
}
struct CountersView<'a> {
    page: &'a PageWorkspace,
    start: usize,
    len: usize,
}
impl Serialize for CountersView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_seq(Some(self.len))?;
        for counter in &self.page.counters[self.start..self.start + self.len] {
            out.serialize_element(&CounterView {
                page: self.page,
                counter,
            })?;
        }
        out.end()
    }
}
struct CounterView<'a> {
    page: &'a PageWorkspace,
    counter: &'a PageCounter,
}
impl Serialize for CounterView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("DiagnosticCounter", 3)?;
        out.serialize_field("code", self.page.string(self.counter.code))?;
        out.serialize_field("occurrences", &self.counter.occurrences)?;
        out.serialize_field("last_observed_at", &self.counter.last_observed_at)?;
        out.end()
    }
}
struct WindowsView<'a> {
    page: &'a PageWorkspace,
    start: usize,
    len: usize,
}
impl Serialize for WindowsView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_seq(Some(self.len))?;
        for window in &self.page.windows[self.start..self.start + self.len] {
            out.serialize_element(&WindowView {
                page: self.page,
                window,
            })?;
        }
        out.end()
    }
}
struct WindowView<'a> {
    page: &'a PageWorkspace,
    window: &'a PageWindow,
}
impl Serialize for WindowView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("DiagnosticWindow", 4)?;
        out.serialize_field("resolution_millis", &self.window.resolution_millis)?;
        out.serialize_field("bucket_capacity", &self.window.bucket_capacity)?;
        out.serialize_field(
            "buckets",
            &self.page.buckets
                [self.window.bucket_start..self.window.bucket_start + self.window.bucket_len],
        )?;
        out.serialize_field("coverage", &self.window.coverage)?;
        out.end()
    }
}
struct ComponentsView<'a> {
    page: &'a PageWorkspace,
}
impl Serialize for ComponentsView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_seq(Some(self.page.components.len()))?;
        for component in &self.page.components {
            out.serialize_element(&ComponentView {
                page: self.page,
                component,
            })?;
        }
        out.end()
    }
}
struct ComponentView<'a> {
    page: &'a PageWorkspace,
    component: &'a PageComponent,
}
impl Serialize for ComponentView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let row = self.component;
        let mut out = s.serialize_struct("ComponentSnapshot", 7)?;
        out.serialize_field(
            "identity",
            &IdentityView {
                page: self.page,
                identity: &row.identity,
            },
        )?;
        out.serialize_field("registered_at", &row.registered_at)?;
        out.serialize_field("closed_at", &row.closed_at)?;
        out.serialize_field("lifetime", &row.lifetime)?;
        out.serialize_field(
            "since_refresh",
            &RefreshView {
                page: self.page,
                component: row,
            },
        )?;
        out.serialize_field(
            "counters",
            &CountersView {
                page: self.page,
                start: row.counter_start,
                len: row.counter_len,
            },
        )?;
        out.serialize_field(
            "windows",
            &WindowsView {
                page: self.page,
                start: row.window_start,
                len: row.window_len,
            },
        )?;
        out.end()
    }
}
struct RefreshView<'a> {
    page: &'a PageWorkspace,
    component: &'a PageComponent,
}
impl Serialize for RefreshView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let row = self.component;
        let mut out = s.serialize_struct("DiagnosticRefreshSummary", 5)?;
        out.serialize_field("generation", &row.refresh_generation)?;
        out.serialize_field("started_at", &row.refresh_started_at)?;
        out.serialize_field("elapsed_millis", &row.refresh_elapsed_millis)?;
        out.serialize_field(
            "ownership",
            &OwnershipView {
                page: self.page,
                ownership: &row.ownership,
            },
        )?;
        out.serialize_field("totals", &row.refresh_totals)?;
        out.end()
    }
}
struct EventsView<'a> {
    page: &'a PageWorkspace,
}
impl Serialize for EventsView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_seq(Some(self.page.events.len()))?;
        for event in &self.page.events {
            out.serialize_element(&EventView {
                page: self.page,
                event,
            })?;
        }
        out.end()
    }
}
struct EventView<'a> {
    page: &'a PageWorkspace,
    event: &'a PageEvent,
}
impl Serialize for EventView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let e = self.event;
        let mut out = s.serialize_struct("DiagnosticEvent", 11)?;
        out.serialize_field("sequence", &e.sequence)?;
        out.serialize_field("observed_at", &e.observed_at)?;
        out.serialize_field(
            "component",
            &IdentityView {
                page: self.page,
                identity: &e.component,
            },
        )?;
        out.serialize_field(
            "origin",
            &OriginView {
                page: self.page,
                event: e,
            },
        )?;
        out.serialize_field(
            "classification",
            &ClassificationView {
                page: self.page,
                event: e,
            },
        )?;
        out.serialize_field("context", &e.context)?;
        out.serialize_field("severity", &e.severity)?;
        out.serialize_field("outcome", &e.outcome)?;
        out.serialize_field(
            "detail",
            &e.detail.map(|text| DetailView {
                message: self.page.string(text),
                truncated: e.detail_truncated,
            }),
        )?;
        out.serialize_field("count", &e.count)?;
        out.serialize_field(
            "correlation",
            &CorrelationView {
                page: self.page,
                event: e,
            },
        )?;
        out.end()
    }
}
struct OriginView<'a> {
    page: &'a PageWorkspace,
    event: &'a PageEvent,
}
impl Serialize for OriginView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("DiagnosticOrigin", 2)?;
        out.serialize_field("refresh_generation", &self.event.origin_generation)?;
        out.serialize_field(
            "ownership",
            &OwnershipView {
                page: self.page,
                ownership: &self.event.ownership,
            },
        )?;
        out.end()
    }
}
struct ClassificationView<'a> {
    page: &'a PageWorkspace,
    event: &'a PageEvent,
}
impl Serialize for ClassificationView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("Classification", 2)?;
        out.serialize_field("canonical_code", self.page.string(self.event.code))?;
        out.serialize_field(
            "failure_class",
            &self.event.failure_class.map(|v| self.page.string(v)),
        )?;
        out.end()
    }
}
struct DetailView<'a> {
    message: &'a str,
    truncated: bool,
}
impl Serialize for DetailView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("DiagnosticDetail", 2)?;
        out.serialize_field("message", self.message)?;
        out.serialize_field("truncated", &self.truncated)?;
        out.end()
    }
}
struct CorrelationView<'a> {
    page: &'a PageWorkspace,
    event: &'a PageEvent,
}
impl Serialize for CorrelationView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let len = usize::from(self.event.session_id.is_some())
            + usize::from(self.event.request_id.is_some());
        let mut out = s.serialize_struct("DiagnosticCorrelation", len)?;
        if let Some(id) = self.event.session_id {
            out.serialize_field("session_id", self.page.string(id))?;
        }
        if let Some(id) = self.event.request_id {
            out.serialize_field("request_id", self.page.string(id))?;
        }
        out.end()
    }
}

struct Capture {
    captured_at: ObservationTime,
    sequence: u64,
    availability: DiagnosticAvailability,
    loss: DiagnosticLoss,
    lifetime: DiagnosticLifetimeSummary,
    scope: u64,
    next_component: usize,
    next_event: usize,
    component_end: usize,
    event_end: usize,
    truncated: bool,
}
impl Capture {
    fn continuation(&self, source: &SourceIdentity) -> Option<DiagnosticCursor> {
        if self.loss.sequence_saturated
            || (self.next_component >= self.component_end && self.next_event >= self.event_end)
        {
            return None;
        }
        Some(DiagnosticCursor {
            source_instance_id: source.instance_id,
            sequence: self.sequence.into(),
            component_offset: (self.next_component as u64).into(),
            event_offset: (self.next_event as u64).into(),
            scope: self.scope.into(),
        })
    }
}
struct SnapshotView<'a> {
    page: &'a PageWorkspace,
    source: &'a SourceIdentity,
    capture: &'a Capture,
}
impl Serialize for SnapshotView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("DiagnosticSnapshot", 10)?;
        out.serialize_field("schema_version", &DIAGNOSTIC_SCHEMA_VERSION)?;
        out.serialize_field("catalogue_version", &DIAGNOSTIC_CATALOGUE_VERSION)?;
        out.serialize_field("source", self.source)?;
        out.serialize_field("captured_at", &self.capture.captured_at)?;
        out.serialize_field("sequence", &DiagnosticNumber::new(self.capture.sequence))?;
        out.serialize_field("lifetime", &self.capture.lifetime)?;
        out.serialize_field("components", &ComponentsView { page: self.page })?;
        out.serialize_field("events", &EventsView { page: self.page })?;
        out.serialize_field(
            "coverage",
            &CoverageView {
                page: self.page,
                capture: self.capture,
            },
        )?;
        out.serialize_field("continuation", &self.capture.continuation(self.source))?;
        out.end()
    }
}
struct CoverageView<'a> {
    page: &'a PageWorkspace,
    capture: &'a Capture,
}
impl Serialize for CoverageView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut out = s.serialize_struct("DiagnosticCoverage", 5)?;
        out.serialize_field("availability", &self.capture.availability)?;
        out.serialize_field(
            "summary_scope",
            &DiagnosticSummaryScope::WholeComponentLifetime,
        )?;
        out.serialize_field(
            "retention",
            &DiagnosticRetention {
                oldest_event_sequence: self.page.events.first().map(|e| e.sequence),
                latest_event_sequence: self.page.events.last().map(|e| e.sequence),
                events_retained: (self.page.events.len() as u64).into(),
                components_retained: (self.page.components.len() as u64).into(),
                query_truncated: self.capture.truncated,
            },
        )?;
        out.serialize_field("loss", &self.capture.loss)?;
        out.serialize_field(
            "capabilities",
            &[CapabilityView {
                name: "otel",
                availability: DiagnosticAvailability::Unsupported,
                reason: "not_integrated",
            }],
        )?;
        out.end()
    }
}
#[derive(serde::Serialize)]
struct CapabilityView<'a> {
    name: &'a str,
    availability: DiagnosticAvailability,
    reason: &'a str,
}

pub(super) fn capture(
    inner: &Inner,
    query: DiagnosticQuery,
) -> Result<SnapshotLease, DiagnosticError> {
    let backend = inner.backend.as_ref().ok_or(DiagnosticError::Disabled)?;
    validate_query(&query, &inner.limits)?;
    let Some(mut lease) = backend.queries.acquire() else {
        super::lock(&backend.state).loss.queries_rejected.add(1);
        return Err(DiagnosticError::QueryCapacity);
    };
    let page = lease.workspace.as_mut().expect("acquired query workspace");
    let scope = query_scope(&query, inner.context_id);
    let state = super::lock(&backend.state);
    let captured_at = Utc::now();
    let tick = inner.tick();
    let (mut component_position, mut event_position) = (0, 0);
    if let Some(cursor) = &query.continuation {
        if state.loss.sequence_saturated {
            return Err(DiagnosticError::InvalidCursor);
        }
        if cursor.source_instance_id != inner.source.instance_id
            || cursor.sequence.get() != state.sequence
            || cursor.scope.get() != scope
        {
            return Err(DiagnosticError::InvalidCursor);
        }
        component_position = usize::try_from(cursor.component_offset.get())
            .map_err(|_| DiagnosticError::InvalidCursor)?;
        event_position = usize::try_from(cursor.event_offset.get())
            .map_err(|_| DiagnosticError::InvalidCursor)?;
        if component_position > state.components.len() || event_position > state.recent.len() {
            return Err(DiagnosticError::InvalidCursor);
        }
    }
    let component_limit = query
        .max_components
        .map_or(inner.limits.max_query_components, |n| {
            (n as usize).min(inner.limits.max_query_components)
        });
    let event_limit = query.max_events.map_or(inner.limits.max_query_events, |n| {
        (n as usize).min(inner.limits.max_query_events)
    });
    let mut truncated = false;
    if component_limit == 0 {
        component_position = state.components.len();
    }
    while component_position < state.components.len() {
        let position = component_position;
        let Some(component) = state.components[position].as_ref() else {
            component_position += 1;
            continue;
        };
        if !component_matches(component, &query) {
            component_position += 1;
            continue;
        }
        if page.components.len() == component_limit {
            truncated = true;
            break;
        }
        if copy_component(page, &state, position, component, &query, tick).is_none() {
            if page.components.is_empty() {
                return Err(DiagnosticError::SnapshotCapacity);
            }
            truncated = true;
            break;
        }
        component_position += 1;
    }
    if event_limit == 0 {
        event_position = state.recent.len();
    }
    while event_position < state.recent.len() {
        let position = event_position;
        let event = state
            .recent
            .get(position)
            .expect("retained history position");
        if !event_matches(&state, event, &query) {
            event_position += 1;
            continue;
        }
        if page.events.len() == event_limit {
            truncated = true;
            break;
        }
        if copy_event(page, &state, event, position).is_none() {
            if page.components.is_empty() && page.events.is_empty() {
                return Err(DiagnosticError::SnapshotCapacity);
            }
            truncated = true;
            break;
        }
        event_position += 1;
    }
    let mut capture = Capture {
        captured_at,
        sequence: state.sequence,
        availability: if state.loss.sequence_saturated {
            DiagnosticAvailability::Unavailable
        } else if state.lifetime.totals.recorded_events_total.value.get() == 0 {
            DiagnosticAvailability::NeverObserved
        } else {
            DiagnosticAvailability::Available
        },
        loss: state.loss.clone(),
        lifetime: lifetime_summary(&state.lifetime, tick, tick),
        scope,
        next_component: component_position,
        next_event: event_position,
        component_end: state.components.len(),
        event_end: state.recent.len(),
        truncated,
    };
    drop(state);
    let max_bytes = query
        .max_response_bytes
        .map_or(inner.limits.max_snapshot_bytes, |n| {
            (n as usize).min(inner.limits.max_snapshot_bytes)
        });
    loop {
        let mut counter = CountingWriter(0);
        serde_json::to_writer(
            &mut counter,
            &SnapshotView {
                page,
                source: &inner.source,
                capture: &capture,
            },
        )
        .map_err(|_| DiagnosticError::SnapshotCapacity)?;
        if counter.0 <= max_bytes {
            break;
        }
        capture.truncated = true;
        if let Some(event) = page.events.pop() {
            capture.next_event = event.source_position;
            page.text_len = event.text_start;
        } else if let Some(component) = page.components.pop() {
            capture.next_component = component.source_position;
            page.counters.truncate(component.counter_start);
            let bucket_start = page
                .windows
                .get(component.window_start)
                .map_or(page.buckets.len(), |w| w.bucket_start);
            page.windows.truncate(component.window_start);
            page.buckets.truncate(bucket_start);
            page.text_len = component.text_start;
        } else {
            return Err(DiagnosticError::SnapshotCapacity);
        }
        if page.events.is_empty() && page.components.is_empty() {
            return Err(DiagnosticError::SnapshotCapacity);
        }
    }
    let mut output = std::mem::take(&mut page.output);
    let (serialized, output_len) = {
        let mut writer = CappedWriter::new(&mut output[..max_bytes]);
        let serialized = serde_json::to_writer(
            &mut writer,
            &SnapshotView {
                page,
                source: &inner.source,
                capture: &capture,
            },
        );
        (serialized, writer.len())
    };
    page.output = output;
    serialized.map_err(|_| DiagnosticError::SnapshotCapacity)?;
    page.output_len = output_len;
    Ok(lease)
}

fn validate_query(
    query: &DiagnosticQuery,
    limits: &DiagnosticLimits,
) -> Result<(), DiagnosticError> {
    let fields = [
        query.component_key.as_deref(),
        query.ownership.configuration_id.as_deref(),
        query.ownership.responsibility.as_deref(),
    ];
    if fields
        .into_iter()
        .flatten()
        .any(|v| v.len() > limits.max_identity_bytes)
    {
        return Err(DiagnosticError::InvalidQuery);
    }
    // Built-in catalogue names are admitted independently of identity limits.
    // Extensions retain their own code-name and cause-name admission bounds.
    if query.code.as_deref().is_some_and(|name| {
        BuiltinDiagnosticCode::from_canonical_name(name).is_none()
            && name.len() > limits.max_identity_bytes
    }) || query
        .failure_class
        .as_deref()
        .is_some_and(|name| FailureClass::from_canonical_name(name).is_none() && name.len() > 32)
    {
        return Err(DiagnosticError::InvalidQuery);
    }
    Ok(())
}
fn query_scope(query: &DiagnosticQuery, context_id: Uuid) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    context_id.hash(&mut hash);
    query.component_key.hash(&mut hash);
    query.component_instance_id.hash(&mut hash);
    query.origin_generation.hash(&mut hash);
    query.code.hash(&mut hash);
    query.domain.hash(&mut hash);
    query.plugin_family.hash(&mut hash);
    query.failure_class.hash(&mut hash);
    query.ownership.hash(&mut hash);
    query.after_sequence.hash(&mut hash);
    query.max_components.hash(&mut hash);
    query.max_events.hash(&mut hash);
    query.max_response_bytes.hash(&mut hash);
    hash.finish()
}
fn identity_matches(component: &state::ComponentSlot, query: &DiagnosticQuery) -> bool {
    query
        .component_key
        .as_deref()
        .is_none_or(|value| component.key.as_str() == value)
        && query
            .component_instance_id
            .is_none_or(|value| component.instance_id == value)
}
fn component_matches(component: &state::ComponentSlot, query: &DiagnosticQuery) -> bool {
    identity_matches(component, query) && component.refresh.ownership.matches(&query.ownership)
}
fn code_matches(state: &state::State, code: state::StoredCode, query: &DiagnosticQuery) -> bool {
    if query
        .code
        .as_deref()
        .is_some_and(|v| state.code_name(code) != Some(v))
    {
        return false;
    }
    if let Some(domain) = query.domain
        && !matches!(code,state::StoredCode::Builtin(c) if c.domain()==domain)
    {
        return false;
    }
    if let Some(family) = query.plugin_family
        && !matches!(code,state::StoredCode::Builtin(c) if c.plugin_family()==Some(family))
    {
        return false;
    }
    true
}
fn event_matches(
    state: &state::State,
    event: &state::StoredObservation,
    query: &DiagnosticQuery,
) -> bool {
    let Some(component) = state.component(event.component) else {
        return false;
    };
    identity_matches(component, query)
        && event.origin.ownership.matches(&query.ownership)
        && query
            .origin_generation
            .is_none_or(|value| event.origin.generation == value.get())
        && code_matches(state, event.code, query)
        && query.failure_class.as_deref().is_none_or(|value| {
            event
                .failure_class
                .and_then(|class| state.class_name(class))
                == Some(value)
        })
        && query
            .after_sequence
            .is_none_or(|value| event.sequence > value.get())
}
fn copy_identity(
    page: &mut PageWorkspace,
    component: &state::ComponentSlot,
) -> Option<PageIdentity> {
    Some(PageIdentity {
        key: page.copy_text(component.key.as_str())?,
        instance_id: component.instance_id,
        generation: component.handle.generation.into(),
    })
}
fn copy_ownership(page: &mut PageWorkspace, ownership: &state::Ownership) -> Option<PageOwnership> {
    Some(PageOwnership {
        subscriber_instance_id: ownership.subscriber_instance_id,
        plugin_instance_id: ownership.plugin_instance_id,
        configuration_id: ownership
            .configuration_id
            .as_ref()
            .map(|v| page.copy_text(v.as_str()))
            .transpose_option()?,
        responsibility: ownership
            .responsibility
            .as_ref()
            .map(|v| page.copy_text(v.as_str()))
            .transpose_option()?,
    })
}
// Option<Option<T>> distinguishes absent source metadata from failed admission.
trait TransposeOption<T> {
    fn transpose_option(self) -> Option<Option<T>>;
}
impl<T> TransposeOption<T> for Option<Option<T>> {
    fn transpose_option(self) -> Option<Option<T>> {
        match self {
            Some(Some(v)) => Some(Some(v)),
            Some(None) => None,
            None => Some(None),
        }
    }
}
fn lifetime_summary(state: &LifetimeState, elapsed: u64, tick: u64) -> DiagnosticLifetimeSummary {
    DiagnosticLifetimeSummary {
        totals: state.totals,
        elapsed_millis: elapsed.into(),
        last_failure_at: state.last_failure_at,
        since_last_failure_millis: state
            .last_failure_tick
            .map(|value| tick.saturating_sub(value).into()),
    }
}
fn copy_component(
    page: &mut PageWorkspace,
    state: &state::State,
    position: usize,
    component: &state::ComponentSlot,
    query: &DiagnosticQuery,
    tick: u64,
) -> Option<()> {
    let checkpoint = (
        page.text_len,
        page.counters.len(),
        page.windows.len(),
        page.buckets.len(),
    );
    let result = (|| {
        let identity = copy_identity(page, component)?;
        let ownership = copy_ownership(page, &component.refresh.ownership)?;
        let counter_start = page.counters.len();
        for counter in state
            .counters
            .iter()
            .flatten()
            .filter(|c| c.component == component.handle)
        {
            let Some(observed) = counter.last_observed_at else {
                continue;
            };
            if !code_matches(state, counter.code, query) {
                continue;
            }
            if page.counters.len() == page.counters.capacity() {
                return None;
            }
            let code = page.copy_text(state.code_name(counter.code)?)?;
            page.counters.push(PageCounter {
                code,
                occurrences: counter.occurrences,
                last_observed_at: observed,
            });
        }
        let window_start = page.windows.len();
        if let Some(reservation) = component.window {
            if page.windows.len() == page.windows.capacity() {
                return None;
            }
            let window = copy_window(page, state, component, reservation, tick)?;
            page.windows.push(window);
        }
        let end_tick = component.closed_tick.unwrap_or(tick);
        let lifetime = lifetime_summary(
            &component.lifetime,
            end_tick.saturating_sub(component.registered_tick),
            tick,
        );
        page.components.push(PageComponent {
            source_position: position,
            text_start: checkpoint.0,
            identity,
            ownership,
            registered_at: component.registered_at,
            closed_at: component.closed_at,
            lifetime,
            refresh_generation: component.refresh.generation.into(),
            refresh_started_at: component.refresh.started_at,
            refresh_elapsed_millis: end_tick
                .saturating_sub(component.refresh.started_tick)
                .into(),
            refresh_totals: component.refresh.totals,
            counter_start,
            counter_len: page.counters.len() - counter_start,
            window_start,
            window_len: page.windows.len() - window_start,
        });
        Some(())
    })();
    if result.is_none() {
        page.text_len = checkpoint.0;
        page.counters.truncate(checkpoint.1);
        page.windows.truncate(checkpoint.2);
        page.buckets.truncate(checkpoint.3);
    }
    result
}
fn copy_window(
    page: &mut PageWorkspace,
    state: &state::State,
    component: &state::ComponentSlot,
    reservation: state::WindowReservation,
    tick: u64,
) -> Option<PageWindow> {
    if reservation.len == 0 || reservation.resolution_millis == 0 {
        return None;
    }
    let end = component
        .closed_tick
        .unwrap_or(tick)
        .saturating_sub(component.registered_tick);
    let resolution = reservation.resolution_millis;
    let mut last = end / resolution;
    if component.closed_tick.is_some() && end > 0 && end.is_multiple_of(resolution) {
        // Millisecond ticks can put an accepted observation and closure at the
        // same boundary. Keep that occupied bucket even though coverage ends
        // at its start; only discard an unobserved terminal bucket.
        let terminal = state
            .windows
            .get(reservation.start + (last % reservation.len as u64) as usize)
            .and_then(Option::as_ref);
        if terminal.is_none_or(|slot| slot.interval != Some(last)) {
            last = last.saturating_sub(1);
        }
    }
    let first = last.saturating_sub(reservation.len.saturating_sub(1) as u64);
    let bucket_start = page.buckets.len();
    let mut incomplete = false;
    for interval in first..=last {
        if page.buckets.len() == page.buckets.capacity() {
            return None;
        }
        let slot = state
            .windows
            .get(reservation.start + (interval % (reservation.len as u64)) as usize)
            .and_then(Option::as_ref)?;
        let (successes, failures, unspecified) = if slot.interval == Some(interval) {
            (slot.successes, slot.failures, slot.unspecified)
        } else {
            (
                DiagnosticCount::default(),
                DiagnosticCount::default(),
                DiagnosticCount::default(),
            )
        };
        incomplete |= successes.saturated || failures.saturated || unspecified.saturated;
        page.buckets.push(DiagnosticWindowBucket {
            start_elapsed_millis: interval.saturating_mul(resolution).into(),
            end_elapsed_millis: interval.saturating_add(1).saturating_mul(resolution).into(),
            successes,
            failures,
            unspecified,
        });
    }
    Some(PageWindow {
        resolution_millis: resolution.into(),
        bucket_capacity: u32::try_from(reservation.len).ok()?,
        bucket_start,
        bucket_len: page.buckets.len() - bucket_start,
        coverage: DiagnosticWindowCoverage {
            availability: if component.lifetime.totals.recorded_events_total.value.get() == 0 {
                DiagnosticAvailability::NeverObserved
            } else {
                DiagnosticAvailability::Available
            },
            start_elapsed_millis: first.saturating_mul(resolution).into(),
            end_elapsed_millis: end.into(),
            partial_start: false,
            partial_end: component.closed_tick.is_none()
                || end == 0
                || !end.is_multiple_of(resolution)
                || last.saturating_mul(resolution) == end,
            incomplete,
        },
    })
}
fn copy_detail(
    page: &mut PageWorkspace,
    state: &state::State,
    handle: storage::DetailHandle,
) -> Option<TextRef> {
    let len = state.detail.byte_len(handle)?;
    let end = page.text_len.checked_add(len)?;
    if end > page.text.len() {
        return None;
    }
    let reference = TextRef {
        offset: page.text_len,
        len,
    };
    let mut writer = CappedWriter::new(&mut page.text[page.text_len..end]);
    state.detail.copy_into(handle, &mut writer).ok()?;
    page.text_len = end;
    Some(reference)
}
fn copy_event(
    page: &mut PageWorkspace,
    state: &state::State,
    event: &state::StoredObservation,
    position: usize,
) -> Option<()> {
    let text_start = page.text_len;
    let result = (|| {
        let component = copy_identity(page, state.component(event.component)?)?;
        let code = page.copy_text(state.code_name(event.code)?)?;
        let failure_class = event
            .failure_class
            .map(|class| page.copy_text(state.class_name(class)?))
            .transpose_option()?;
        let detail = event
            .detail
            .map(|v| copy_detail(page, state, v))
            .transpose_option()?;
        let session_id = event
            .session_id
            .map(|v| copy_detail(page, state, v))
            .transpose_option()?;
        let request_id = event
            .request_id
            .map(|v| copy_detail(page, state, v))
            .transpose_option()?;
        let ownership = copy_ownership(page, &event.origin.ownership)?;
        page.events.push(PageEvent {
            source_position: position,
            text_start,
            sequence: event.sequence.into(),
            observed_at: event.observed_at,
            component,
            origin_generation: event.origin.generation.into(),
            ownership,
            code,
            failure_class,
            context: event.context,
            severity: event.severity,
            outcome: event.outcome,
            detail,
            detail_truncated: event.detail_truncated,
            count: event.count.into(),
            session_id,
            request_id,
        });
        Some(())
    })();
    if result.is_none() {
        page.text_len = text_start;
    }
    result
}

struct CountingWriter(usize);
impl std::io::Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_windows_preserve_observations_at_the_closing_boundary() {
        for (resolution, observed, closed, last_start, partial_end) in [
            (1, 2, 2, 2, true),
            (1, 1, 2, 1, false),
            (2, 2, 3, 2, true),
            (1, 0, 0, 0, true),
        ] {
            let diagnostics = Diagnostics::from_settings(
                SourceIdentity {
                    instance_id: Uuid::nil(),
                    process_id: None,
                    process_role: "test".into(),
                    relay_version: "test".into(),
                },
                DiagnosticsSettings {
                    enabled: true,
                    ..Default::default()
                },
            )
            .unwrap();
            let code = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
            let recorder = diagnostics
                .register_component(ComponentRegistration {
                    window: Some(WindowSettings {
                        resolution_millis: resolution,
                        bucket_count: 4,
                    }),
                    ..ComponentRegistration::new("sink", &[code])
                })
                .unwrap();
            recorder
                .record_without_projection_lazy(|| {
                    Observation::new(
                        Classification::success(SuccessCode::SinkInitialized),
                        "success",
                        7,
                    )
                })
                .unwrap();
            {
                // Inject exact monotonic offsets into private state so this
                // regression never depends on scheduler or sleep precision.
                let backend = diagnostics.inner.backend.as_ref().unwrap();
                let mut state = lock(&backend.state);
                let handle = recorder.lease.as_ref().unwrap().handle;
                let component = state.components[handle.slot].as_mut().unwrap();
                component.registered_tick = 0;
                component.closed_tick = Some(closed);
                component.closed_at = Some(Utc::now());
                for slot in state.windows.iter_mut().flatten() {
                    slot.interval = None;
                }
                state.record_window(handle, observed, DiagnosticOutcome::Success, 7);
            }
            let snapshot = diagnostics
                .snapshot(DiagnosticQuery::default())
                .unwrap()
                .to_snapshot()
                .unwrap();
            let component = &snapshot.components[0];
            let window = &component.windows[0];
            assert_eq!(component.lifetime.totals.successes_total.value.get(), 7);
            assert_eq!(snapshot.events[0].count.get(), 7);
            assert_eq!(
                window
                    .buckets
                    .iter()
                    .map(|b| b.successes.value.get())
                    .sum::<u64>(),
                7
            );
            assert_eq!(
                window.buckets.last().unwrap().start_elapsed_millis.get(),
                last_start
            );
            assert_eq!(window.coverage.end_elapsed_millis.get(), closed);
            assert_eq!(window.coverage.partial_end, partial_end);
        }
    }
}
