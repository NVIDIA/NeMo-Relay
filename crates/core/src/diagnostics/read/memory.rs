// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Derive occupancy during capture. Scratch and rows share the existing query lease.

use super::*;
use crate::diagnostics::{
    ComponentMemoryFootprint, DIAGNOSTIC_MEMORY_VERSION, DiagnosticMemoryPage,
    DiagnosticMemoryUsage, EntryMemoryFootprint, MemorySelectionTotals, ResourceMemoryView,
};

#[derive(Default, Clone, Copy)]
struct Charge {
    registration: usize,
    counters: usize,
    windows: usize,
    extensions: usize,
    events: usize,
    event_bytes: usize,
}
impl Charge {
    fn total(self) -> Result<usize, DiagnosticError> {
        [
            self.registration,
            self.counters,
            self.windows,
            self.extensions,
            self.event_bytes,
        ]
        .into_iter()
        .try_fold(0usize, add)
    }
    fn footprint(self) -> Result<ComponentMemoryFootprint, DiagnosticError> {
        Ok(ComponentMemoryFootprint {
            registration_bytes: number(self.registration),
            counter_bytes: number(self.counters),
            window_bytes: number(self.windows),
            extension_bytes: number(self.extensions),
            retained_event_count: number(self.events),
            retained_event_bytes: number(self.event_bytes),
            occupied_bytes: number(self.total()?),
        })
    }
}
fn add(a: usize, b: usize) -> Result<usize, DiagnosticError> {
    a.checked_add(b)
        .ok_or(DiagnosticError::InconsistentAccounting)
}
fn number(n: usize) -> DiagnosticNumber {
    DiagnosticNumber::new(n as u64)
}
struct ComponentRow {
    position: usize,
    identity: PageIdentity,
    footprint: ComponentMemoryFootprint,
}
struct EntryRow {
    position: usize,
    identity: PageIdentity,
    sequence: DiagnosticNumber,
    footprint: EntryMemoryFootprint,
}
pub(super) struct Workspace {
    metadata: Option<Metadata>,
    charges: Vec<Charge>,
    components: Vec<ComponentRow>,
    entries: Vec<EntryRow>,
}
impl Workspace {
    pub(super) fn new(limits: &DiagnosticLimits) -> Result<Self, TryReserveError> {
        let mut charges = reserved(limits.max_components)?;
        charges.resize(limits.max_components, Charge::default());
        Ok(Self {
            metadata: None,
            charges,
            components: reserved(limits.max_query_components)?,
            entries: reserved(limits.max_query_events)?,
        })
    }
    pub(super) fn clear(&mut self) {
        self.metadata = None;
        self.components.clear();
        self.entries.clear();
    }
    pub(super) fn required_bytes(l: &DiagnosticLimits) -> Result<usize, DiagnosticError> {
        [
            (l.max_components, size_of::<Charge>()),
            (l.max_query_components, size_of::<ComponentRow>()),
            (l.max_query_events, size_of::<EntryRow>()),
        ]
        .into_iter()
        .try_fold(0usize, |n, (count, size)| {
            n.checked_add(
                count
                    .checked_mul(size)
                    .ok_or(DiagnosticError::InvalidLimits)?,
            )
            .ok_or(DiagnosticError::InvalidLimits)
        })
    }
    pub(super) fn reserved_bytes(&self) -> usize {
        self.charges.capacity() * size_of::<Charge>()
            + self.components.capacity() * size_of::<ComponentRow>()
            + self.entries.capacity() * size_of::<EntryRow>()
    }
}

/// Borrowed component row from copied query storage. Never borrows live history.
pub struct ComponentMemoryView<'a> {
    page: &'a PageWorkspace,
    row: &'a ComponentRow,
}
impl ComponentMemoryView<'_> {
    /// Logical component key.
    pub fn key(&self) -> &str {
        self.page.string(self.row.identity.key)
    }
    /// Registration incarnation.
    pub fn instance_id(&self) -> Uuid {
        self.row.identity.instance_id
    }
    /// Registration generation.
    pub fn generation(&self) -> DiagnosticNumber {
        self.row.identity.generation
    }
    /// Exact assigned backing, independent of event selection.
    pub fn footprint(&self) -> &ComponentMemoryFootprint {
        &self.row.footprint
    }
}
impl Serialize for ComponentMemoryView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        struct View<'a> {
            component: IdentityView<'a>,
            #[serde(flatten)]
            footprint: &'a ComponentMemoryFootprint,
        }
        View {
            component: IdentityView::new(self.page, &self.row.identity),
            footprint: &self.row.footprint,
        }
        .serialize(s)
    }
}
/// Borrowed retained-entry footprint from copied query storage, without its text.
pub struct EntryMemoryView<'a> {
    page: &'a PageWorkspace,
    row: &'a EntryRow,
}
impl EntryMemoryView<'_> {
    /// Original owner key.
    pub fn key(&self) -> &str {
        self.page.string(self.row.identity.key)
    }
    /// Original registration incarnation.
    pub fn instance_id(&self) -> Uuid {
        self.row.identity.instance_id
    }
    /// Original registration generation.
    pub fn generation(&self) -> DiagnosticNumber {
        self.row.identity.generation
    }
    /// Source acceptance sequence.
    pub fn sequence(&self) -> DiagnosticNumber {
        self.row.sequence
    }
    /// Compiled entry footprint, not serialized size.
    pub fn footprint(&self) -> &EntryMemoryFootprint {
        &self.row.footprint
    }
}
impl Serialize for EntryMemoryView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        struct View<'a> {
            sequence: DiagnosticNumber,
            component: IdentityView<'a>,
            #[serde(flatten)]
            footprint: &'a EntryMemoryFootprint,
        }
        View {
            sequence: self.row.sequence,
            component: IdentityView::new(self.page, &self.row.identity),
            footprint: &self.row.footprint,
        }
        .serialize(s)
    }
}
struct Metadata {
    captured_at: ObservationTime,
    sequence: u64,
    scope: u64,
    aggregate: DiagnosticMemoryUsage,
    loss: DiagnosticLoss,
    occupied: usize,
    shared: usize,
    selection: MemorySelectionTotals,
    next_component: usize,
    next_event: usize,
    component_end: usize,
    event_end: usize,
    saturated: bool,
    resource: ResourceMemoryView,
}
/// Memory inspection holds the same bounded permit as an ordinary snapshot.
/// Borrowed iterators refer only to the copied page; dropping releases all storage.
pub struct MemoryInspectionLease {
    lease: SnapshotLease,
    components: usize,
    entries: usize,
}
impl std::fmt::Debug for MemoryInspectionLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryInspectionLease")
            .field("bytes", &self.as_bytes().len())
            .finish_non_exhaustive()
    }
}
impl MemoryInspectionLease {
    fn metadata(&self) -> &Metadata {
        self.lease
            .workspace
            .as_ref()
            .expect("live lease")
            .memory
            .metadata
            .as_ref()
            .expect("captured memory metadata")
    }
    /// Canonical bounded JSON; no allocation.
    pub fn as_bytes(&self) -> &[u8] {
        self.lease.as_bytes()
    }
    /// Explicit caller-owned decode, outside the service budget.
    pub fn to_memory_page(&self) -> Result<DiagnosticMemoryPage, serde_json::Error> {
        serde_json::from_slice(self.as_bytes())
    }
    /// Allocation-free traversal of copied component rows.
    pub fn components(&self) -> impl ExactSizeIterator<Item = ComponentMemoryView<'_>> {
        let page = self.lease.workspace.as_ref().expect("live lease");
        page.memory.components[..self.components]
            .iter()
            .map(move |row| ComponentMemoryView { page, row })
    }
    /// Allocation-free traversal of copied entry rows.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = EntryMemoryView<'_>> {
        let page = self.lease.workspace.as_ref().expect("live lease");
        page.memory.entries[..self.entries]
            .iter()
            .map(move |row| EntryMemoryView { page, row })
    }
    /// Whole-context aggregate at the same capture revision.
    pub fn aggregate(&self) -> &DiagnosticMemoryUsage {
        &self.metadata().aggregate
    }
    /// Complete attributed backing, including rows omitted by selection/paging.
    pub fn component_occupied_bytes(&self) -> DiagnosticNumber {
        number(self.metadata().occupied)
    }
    /// Shared and free reserved backing.
    pub fn shared_or_unassigned_reserved_bytes(&self) -> DiagnosticNumber {
        number(self.metadata().shared)
    }
    /// Independently copied resource context; no collection or allocation.
    pub fn resource(&self) -> &ResourceMemoryView {
        &self.metadata().resource
    }
}

struct View<'a> {
    page: &'a PageWorkspace,
    source: &'a SourceIdentity,
    metadata: &'a Metadata,
    components: usize,
    entries: usize,
}
impl<'a> View<'a> {
    fn prefix(
        page: &'a PageWorkspace,
        source: &'a SourceIdentity,
        metadata: &'a Metadata,
        rows: usize,
    ) -> Self {
        Self {
            page,
            source,
            metadata,
            components: rows.min(page.memory.components.len()),
            entries: rows
                .saturating_sub(page.memory.components.len())
                .min(page.memory.entries.len()),
        }
    }
}
impl Serialize for View<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let m = self.metadata;
        let components = &self.page.memory.components[..self.components];
        let entries = &self.page.memory.entries[..self.entries];
        let next_c = self
            .page
            .memory
            .components
            .get(self.components)
            .map_or(m.next_component, |r| r.position);
        let next_e = self
            .page
            .memory
            .entries
            .get(self.entries)
            .map_or(m.next_event, |r| r.position);
        let continuation = (!m.saturated && (next_c < m.component_end || next_e < m.event_end))
            .then(|| DiagnosticCursor {
                source_instance_id: self.source.instance_id,
                sequence: m.sequence.into(),
                component_offset: number(next_c),
                event_offset: number(next_e),
                scope: m.scope.into(),
            });
        let totals = MemorySelectionTotals {
            components: number(components.len()),
            events: number(entries.len()),
            component_occupied_bytes: components
                .iter()
                .map(|r| r.footprint.occupied_bytes.get())
                .sum::<u64>()
                .into(),
            entry_occupied_bytes: entries
                .iter()
                .map(|r| r.footprint.occupied_bytes.get())
                .sum::<u64>()
                .into(),
        };
        struct Rows<'a>(&'a View<'a>, bool);
        impl Serialize for Rows<'_> {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                let v = self.0;
                let mut seq =
                    s.serialize_seq(Some(if self.1 { v.components } else { v.entries }))?;
                if self.1 {
                    for row in &v.page.memory.components[..v.components] {
                        seq.serialize_element(&ComponentMemoryView { page: v.page, row })?;
                    }
                } else {
                    for row in &v.page.memory.entries[..v.entries] {
                        seq.serialize_element(&EntryMemoryView { page: v.page, row })?;
                    }
                }
                seq.end()
            }
        }
        let mut out = s.serialize_struct("DiagnosticMemoryPage", 14)?;
        out.serialize_field("schema_version", &DIAGNOSTIC_MEMORY_VERSION)?;
        out.serialize_field("source", self.source)?;
        out.serialize_field("captured_at", &m.captured_at)?;
        out.serialize_field("sequence", &DiagnosticNumber::new(m.sequence))?;
        out.serialize_field("aggregate", &m.aggregate)?;
        out.serialize_field("loss", &m.loss)?;
        out.serialize_field("component_occupied_bytes", &number(m.occupied))?;
        out.serialize_field("shared_or_unassigned_reserved_bytes", &number(m.shared))?;
        out.serialize_field("selection", &m.selection)?;
        out.serialize_field("page", &totals)?;
        out.serialize_field("components", &Rows(self, true))?;
        out.serialize_field("entries", &Rows(self, false))?;
        out.serialize_field("resource", &m.resource)?;
        out.serialize_field("continuation", &continuation)?;
        out.end()
    }
}

pub(in crate::diagnostics) fn capture(
    inner: &Inner,
    query: &DiagnosticQuery,
) -> Result<MemoryInspectionLease, DiagnosticError> {
    let backend = inner.backend.as_ref().ok_or(DiagnosticError::Disabled)?;
    validate_query(query, &inner.limits)?;
    let queries = backend
        .queries
        .as_ref()
        .ok_or(DiagnosticError::QueryDisabled)?;
    let Some(mut lease) = queries.acquire() else {
        super::super::lock(&backend.state)
            .loss
            .queries_rejected
            .add(1);
        return Err(DiagnosticError::QueryCapacity);
    };
    let page = lease.workspace.as_mut().expect("live lease");
    page.memory.charges.fill(Charge::default());
    // Include view identity in the same context/selection fingerprint.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    query_scope(query, inner.context_id).hash(&mut hasher);
    "memory".hash(&mut hasher);
    let scope = hasher.finish();
    let state = super::super::lock(&backend.state);
    let (next_c, mut next_e) = positions(inner, query, &state, scope)?;
    let component_limit = query
        .max_components
        .map_or(inner.limits.max_query_components, |n| n as usize);
    let entry_limit = query
        .max_events
        .map_or(inner.limits.max_query_events, |n| n as usize);
    assigned_charges(&state, &mut page.memory.charges)?;
    let mut selection = MemorySelectionTotals::default();
    let mut entry_blocked = false;
    if entry_limit == 0 {
        next_e = state.recent.len();
    }
    for i in 0..state.recent.len() {
        let e = state.recent.get(i).expect("history position");
        let c = state
            .component(e.component)
            .ok_or(DiagnosticError::InconsistentAccounting)?;
        let (payload, metadata, logical) = match e.text.payload {
            None => (0, 0, 0),
            Some(handle) => state
                .detail
                .footprint(handle)
                .ok_or(DiagnosticError::InconsistentAccounting)?,
        };
        let bytes = add(
            size_of::<state::StoredObservation>(),
            add(payload, metadata)?,
        )?;
        let charge = &mut page.memory.charges[e.component.slot];
        charge.events = add(charge.events, 1)?;
        charge.event_bytes = add(charge.event_bytes, bytes)?;
        if !event_matches(&state, e, query) {
            if i >= next_e && !entry_blocked {
                next_e = i + 1;
            }
            continue;
        }
        selection.events = (selection.events.get() + 1).into();
        selection.entry_occupied_bytes =
            (selection.entry_occupied_bytes.get() + bytes as u64).into();
        if i < next_e || entry_blocked {
            continue;
        }
        if page.memory.entries.len() == entry_limit {
            entry_blocked = true;
            continue;
        }
        let Some(identity) = copy_identity(page, c) else {
            entry_blocked = true;
            continue;
        };
        page.memory.entries.push(EntryRow {
            position: i,
            identity,
            sequence: e.sequence.into(),
            footprint: EntryMemoryFootprint {
                header_bytes: number(size_of::<state::StoredObservation>()),
                detail_payload_bytes: number(payload),
                detail_metadata_bytes: number(metadata),
                logical_text_bytes: number(logical),
                detail_slack_bytes: number(
                    payload
                        .checked_sub(logical)
                        .ok_or(DiagnosticError::InconsistentAccounting)?,
                ),
                occupied_bytes: number(bytes),
            },
        });
        next_e = i + 1;
    }
    let (occupied, next_c, component_blocked) =
        capture_components(&state, query, page, next_c, component_limit, &mut selection)?;
    if (component_blocked || entry_blocked)
        && page.memory.components.is_empty()
        && page.memory.entries.is_empty()
    {
        return Err(DiagnosticError::SnapshotCapacity);
    }
    let aggregate = super::super::memory::memory_usage_locked(inner, &state);
    let mut metadata = Metadata {
        captured_at: Utc::now(),
        sequence: state.sequence,
        scope,
        shared: aggregate
            .reserved_bytes
            .checked_sub(occupied)
            .ok_or(DiagnosticError::InconsistentAccounting)?,
        aggregate,
        loss: state.loss.clone(),
        occupied,
        selection,
        next_component: next_c,
        next_event: next_e,
        component_end: state.components.len(),
        event_end: state.recent.len(),
        saturated: state.loss.sequence_saturated,
        resource: ResourceMemoryView::disabled(),
    };
    drop(state);
    // Independent subsystem copy after releasing diagnostic state. Never collect here.
    metadata.resource =
        crate::resource_metrics::manager::latest_memory_view(inner.source.process_id);
    let max_bytes = query
        .max_response_bytes
        .map_or(inner.limits.max_snapshot_bytes, |n| n as usize);
    let rows = match fit(page, &inner.source, &metadata, max_bytes) {
        Err(DiagnosticError::SnapshotCapacity) if metadata.resource.sample.is_some() => {
            // Resource context must not prevent an otherwise fitting diagnostic page.
            metadata.resource = ResourceMemoryView {
                availability:
                    nemo_relay_types::diagnostics::ResourceMemoryAvailability::CapacityUnavailable,
                projection_slot_bytes: metadata.resource.projection_slot_bytes,
                ..ResourceMemoryView::disabled()
            };
            fit(page, &inner.source, &metadata, max_bytes)?
        }
        result => result?,
    };
    let components = rows.min(page.memory.components.len());
    let entries = rows.saturating_sub(page.memory.components.len());
    let mut output = std::mem::take(&mut page.output);
    let (result, len) = {
        let mut writer = CappedWriter::new(&mut output[..max_bytes]);
        let result = serde_json::to_writer(
            &mut writer,
            &View::prefix(page, &inner.source, &metadata, rows),
        );
        (result, writer.len())
    };
    page.output = output;
    page.output_len = len;
    result.map_err(|_| DiagnosticError::SnapshotCapacity)?;
    page.memory.metadata = Some(metadata);
    Ok(MemoryInspectionLease {
        lease,
        components,
        entries,
    })
}

fn positions(
    inner: &Inner,
    query: &DiagnosticQuery,
    state: &state::State,
    scope: u64,
) -> Result<(usize, usize), DiagnosticError> {
    let mut next_c = 0usize;
    let mut next_e = 0usize;
    if let Some(c) = &query.continuation {
        if state.loss.sequence_saturated
            || c.source_instance_id != inner.source.instance_id
            || c.sequence.get() != state.sequence
            || c.scope.get() != scope
        {
            return Err(DiagnosticError::InvalidCursor);
        }
        next_c = usize::try_from(c.component_offset.get())
            .map_err(|_| DiagnosticError::InvalidCursor)?;
        next_e =
            usize::try_from(c.event_offset.get()).map_err(|_| DiagnosticError::InvalidCursor)?;
        if next_c > state.components.len() || next_e > state.recent.len() {
            return Err(DiagnosticError::InvalidCursor);
        }
    }
    Ok((next_c, next_e))
}

fn assigned_charges(state: &state::State, charges: &mut [Charge]) -> Result<(), DiagnosticError> {
    for (i, c) in state.components.iter().enumerate() {
        if let Some(c) = c {
            let charge = &mut charges[i];
            charge.registration = size_of::<Option<state::ComponentSlot>>();
            charge.windows = c
                .window
                .map_or(0, |w| w.len * size_of::<Option<state::WindowSlot>>());
        }
    }
    for c in state.counters.iter().flatten() {
        if state.component(c.component).is_none() {
            return Err(DiagnosticError::InconsistentAccounting);
        }
        let charge = &mut charges[c.component.slot];
        charge.counters = add(charge.counters, size_of::<Option<state::CounterSlot>>())?;
    }
    for x in state.extensions.iter().flatten() {
        if state.component(x.component).is_none() {
            return Err(DiagnosticError::InconsistentAccounting);
        }
        let charge = &mut charges[x.component.slot];
        charge.extensions = add(charge.extensions, size_of::<Option<state::ExtensionSlot>>())?;
    }
    Ok(())
}

fn fit(
    page: &PageWorkspace,
    source: &SourceIdentity,
    metadata: &Metadata,
    max_bytes: usize,
) -> Result<usize, DiagnosticError> {
    fitting_prefix(
        page.memory.components.len() + page.memory.entries.len(),
        |rows| {
            let exceeded = Cell::new(false);
            let mut count = CountingWriter::new(max_bytes, &exceeded);
            serde_json::to_writer(&mut count, &View::prefix(page, source, metadata, rows))
                .map_err(|_| DiagnosticError::SnapshotCapacity)?;
            Ok(!exceeded.get())
        },
    )
}

fn capture_components(
    state: &state::State,
    query: &DiagnosticQuery,
    page: &mut PageWorkspace,
    mut next_c: usize,
    component_limit: usize,
    selection: &mut MemorySelectionTotals,
) -> Result<(usize, usize, bool), DiagnosticError> {
    let mut occupied = 0;
    let mut component_blocked = false;
    if component_limit == 0 {
        next_c = state.components.len();
    }
    for (i, c) in state.components.iter().enumerate() {
        let Some(c) = c else {
            if i >= next_c && !component_blocked {
                next_c = i + 1;
            }
            continue;
        };
        let charge = page.memory.charges[i];
        let total = charge.total()?;
        occupied = add(occupied, total)?;
        if !component_matches(c, query) {
            if i >= next_c && !component_blocked {
                next_c = i + 1;
            }
            continue;
        }
        selection.components = (selection.components.get() + 1).into();
        selection.component_occupied_bytes =
            (selection.component_occupied_bytes.get() + total as u64).into();
        if i < next_c || component_blocked {
            continue;
        }
        if page.memory.components.len() == component_limit {
            component_blocked = true;
            continue;
        }
        let Some(identity) = copy_identity(page, c) else {
            component_blocked = true;
            continue;
        };
        page.memory.components.push(ComponentRow {
            position: i,
            identity,
            footprint: charge.footprint()?,
        });
        next_c = i + 1;
    }
    Ok((occupied, next_c, component_blocked))
}
