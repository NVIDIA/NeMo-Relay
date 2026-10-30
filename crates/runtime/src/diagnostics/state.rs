// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-capacity local state. No wire DTO graph is retained in this store.

use std::num::NonZeroUsize;

use super::storage::{BoundedHistory, DetailHandle, DetailPool};
use super::{
    BuiltinDiagnosticCode, ComponentOwnership, DiagnosticCount, DiagnosticError, DiagnosticLimits,
    DiagnosticLoss, DiagnosticOutcome, DiagnosticSeverity, DiagnosticTotals, FailureClass,
    ObservationTime, PluginExecution, PluginObservationContext, PluginPhase,
};
use serde::Serialize;
use uuid::Uuid;

pub(super) const ID_BYTES: usize = 128;
pub(super) const DESCRIPTION_BYTES: usize = 256;

/// Library-managed fixed-capacity UTF-8; no allocation on construction or copy.
pub(super) type Text<const N: usize> = arrayvec::ArrayString<N>;

pub(super) fn text<const N: usize>(value: &str) -> Result<Text<N>, DiagnosticError> {
    Text::from(value).map_err(|_| DiagnosticError::MetadataCapacity)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ComponentHandle {
    pub slot: usize,
    pub generation: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StoredCode {
    Builtin(BuiltinDiagnosticCode),
    Extension { slot: usize, generation: u64 },
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Ownership {
    pub subscriber_instance_id: Option<Uuid>,
    pub plugin_instance_id: Option<Uuid>,
    pub configuration_id: Option<Text<ID_BYTES>>,
    pub responsibility: Option<Text<ID_BYTES>>,
}
impl Ownership {
    pub fn new(value: &ComponentOwnership) -> Result<Self, DiagnosticError> {
        Ok(Self {
            subscriber_instance_id: value.subscriber_instance_id,
            plugin_instance_id: value.plugin_instance_id,
            configuration_id: value.configuration_id.as_deref().map(text).transpose()?,
            responsibility: value.responsibility.as_deref().map(text).transpose()?,
        })
    }
    pub fn bytes(&self) -> usize {
        self.configuration_id.map_or(0, |s| s.len()) + self.responsibility.map_or(0, |s| s.len())
    }
    pub fn matches(&self, query: &ComponentOwnership) -> bool {
        query
            .subscriber_instance_id
            .is_none_or(|v| self.subscriber_instance_id == Some(v))
            && query
                .plugin_instance_id
                .is_none_or(|v| self.plugin_instance_id == Some(v))
            && query.configuration_id.as_deref().is_none_or(|v| {
                self.configuration_id
                    .as_ref()
                    .is_some_and(|s| s.as_str() == v)
            })
            && query.responsibility.as_deref().is_none_or(|v| {
                self.responsibility
                    .as_ref()
                    .is_some_and(|s| s.as_str() == v)
            })
    }
}

/// Numeric state and latest-failure timing for one lifetime. Context state is
/// never reset when a component is refreshed, closed, or reclaimed.
#[derive(Clone, Copy, Default)]
pub(super) struct LifetimeState {
    pub totals: DiagnosticTotals,
    pub last_failure_at: Option<ObservationTime>,
    pub last_failure_tick: Option<u64>,
}
impl LifetimeState {
    pub fn record(
        &mut self,
        outcome: DiagnosticOutcome,
        count: u64,
        observed_at: ObservationTime,
        tick: u64,
    ) {
        self.totals.record(outcome, count);
        if outcome == DiagnosticOutcome::Failure {
            self.last_failure_at = Some(observed_at);
            self.last_failure_tick = Some(tick);
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct RefreshState {
    pub generation: u64,
    pub started_at: ObservationTime,
    pub started_tick: u64,
    pub ownership: Ownership,
    pub totals: DiagnosticTotals,
}

/// Original attribution copied into bounded event headers or caller-owned
/// operation handles. It cannot pin backend storage after a refresh.
#[derive(Clone, Copy, Debug)]
pub(super) struct CapturedOrigin {
    pub generation: u64,
    pub ownership: Ownership,
}

#[derive(Clone, Copy)]
pub(super) struct ComponentSlot {
    pub handle: ComponentHandle,
    pub instance_id: Uuid,
    pub key: Text<ID_BYTES>,
    pub registered_at: ObservationTime,
    pub registered_tick: u64,
    pub closed_at: Option<ObservationTime>,
    pub closed_tick: Option<u64>,
    pub lifetime: LifetimeState,
    pub refresh: RefreshState,
    pub window: Option<WindowReservation>,
    pub counter_head: Option<CounterIndex>,
}
impl ComponentSlot {
    pub fn origin(&self) -> CapturedOrigin {
        CapturedOrigin {
            generation: self.refresh.generation,
            ownership: self.refresh.ownership,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct WindowReservation {
    pub start: usize,
    pub len: usize,
    pub resolution_millis: u64,
}
/// Compact optional index into the preallocated counter pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CounterIndex(NonZeroUsize);
impl CounterIndex {
    pub fn new(slot: usize) -> Self {
        Self(
            slot.checked_add(1)
                .and_then(NonZeroUsize::new)
                .expect("admitted counter index"),
        )
    }
    pub fn slot(self) -> usize {
        self.0.get() - 1
    }
}

#[derive(Clone, Copy)]
pub(super) struct CounterSlot {
    pub component: ComponentHandle,
    pub next: Option<CounterIndex>,
    pub code: StoredCode,
    pub occurrences: DiagnosticCount,
    pub last_observed_at: Option<ObservationTime>,
}
#[derive(Clone, Copy)]
pub(super) struct WindowSlot {
    pub interval: Option<u64>,
    pub successes: DiagnosticCount,
    pub failures: DiagnosticCount,
    pub unspecified: DiagnosticCount,
}
#[derive(Clone, Copy)]
pub(super) struct ExtensionSlot {
    pub generation: u64,
    pub component: ComponentHandle,
    pub name: Text<ID_BYTES>,
    pub description: Text<DESCRIPTION_BYTES>,
    pub allowed_classes: enumset::EnumSet<FailureClass>,
    pub opaque_classes: [Option<Text<32>>; 4],
    pub outcome: DiagnosticOutcome,
}
impl ExtensionSlot {
    pub fn metadata_bytes(&self) -> usize {
        self.name.len()
            + self.description.len()
            + self
                .opaque_classes
                .iter()
                .flatten()
                .map(Text::len)
                .sum::<usize>()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StoredFailureClass {
    Builtin(FailureClass),
    Extension {
        slot: usize,
        generation: u64,
        index: u8,
    },
}
/// Validated compact plugin context retained independently of caller-owned names.
#[derive(Clone, Copy, Default, Serialize)]
pub(super) struct StoredPluginContext {
    pub execution: Option<PluginExecution>,
    pub phase: Option<PluginPhase>,
}
impl TryFrom<&PluginObservationContext> for StoredPluginContext {
    type Error = DiagnosticError;

    fn try_from(value: &PluginObservationContext) -> Result<Self, Self::Error> {
        Ok(Self {
            execution: value
                .execution
                .as_ref()
                .map(|name| name.known().ok_or(DiagnosticError::InvalidContext))
                .transpose()?,
            phase: value
                .phase
                .as_ref()
                .map(|name| name.known().ok_or(DiagnosticError::InvalidContext))
                .transpose()?,
        })
    }
}
impl From<StoredPluginContext> for PluginObservationContext {
    fn from(value: StoredPluginContext) -> Self {
        Self {
            execution: value.execution.map(Into::into),
            phase: value.phase.map(Into::into),
        }
    }
}

/// Logical offsets into one packed message/correlation allocation. Admission
/// caps their combined length at 4096 + 128 + 128 bytes.
#[derive(Clone, Copy)]
pub(super) struct TextRange {
    pub start: u16,
    pub len: u16,
}
#[derive(Clone, Copy, Default)]
pub(super) struct RetainedText {
    pub payload: Option<DetailHandle>,
    pub detail: Option<TextRange>,
    pub session_id: Option<TextRange>,
    pub request_id: Option<TextRange>,
}

#[derive(Clone, Copy)]
pub(super) struct StoredObservation {
    pub sequence: u64,
    pub observed_at: ObservationTime,
    pub remote: Option<nemo_relay_types::diagnostics::RemoteDiagnosticOrigin>,
    pub count: u64,
    pub component: ComponentHandle,
    pub origin: CapturedOrigin,
    pub code: StoredCode,
    pub failure_class: Option<StoredFailureClass>,
    pub context: StoredPluginContext,
    pub severity: DiagnosticSeverity,
    pub outcome: DiagnosticOutcome,
    pub text: RetainedText,
    pub detail_truncated: bool,
}

pub(super) struct State {
    pub sequence: u64,
    pub generation: u64,
    pub components: Box<[Option<ComponentSlot>]>,
    pub lifetime: LifetimeState,
    pub counters: Box<[Option<CounterSlot>]>,
    pub windows: Box<[Option<WindowSlot>]>,
    pub extensions: Box<[Option<ExtensionSlot>]>,
    pub recent: BoundedHistory<StoredObservation>,
    pub detail: DetailPool,
    // Logical admission bytes for current registration/ownership and extension
    // text. Original event attribution is inline in preallocated event slots.
    pub metadata_used: usize,
    pub loss: DiagnosticLoss,
}

pub(super) fn empty_slots<T>(n: usize) -> Result<Box<[Option<T>]>, DiagnosticError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .map_err(|_| DiagnosticError::Allocation)?;
    result.resize_with(n, || None);
    Ok(result.into_boxed_slice())
}
impl State {
    pub fn required_bytes(limits: &DiagnosticLimits) -> Result<usize, DiagnosticError> {
        let mut total = BoundedHistory::<StoredObservation>::required_bytes(limits.max_events)
            .map_err(|_| DiagnosticError::InvalidLimits)?;
        for (count, bytes) in [
            (limits.max_components, size_of::<Option<ComponentSlot>>()),
            (limits.max_counters, size_of::<Option<CounterSlot>>()),
            (limits.max_window_buckets, size_of::<Option<WindowSlot>>()),
            (limits.max_extensions, size_of::<Option<ExtensionSlot>>()),
        ] {
            total = total
                .checked_add(
                    count
                        .checked_mul(bytes)
                        .ok_or(DiagnosticError::InvalidLimits)?,
                )
                .ok_or(DiagnosticError::InvalidLimits)?;
        }
        total
            .checked_add(
                DetailPool::required_bytes(limits.detail_pool_bytes, limits.detail_block_bytes)
                    .map_err(|_| DiagnosticError::InvalidLimits)?,
            )
            .ok_or(DiagnosticError::InvalidLimits)
    }
    pub fn new(limits: &DiagnosticLimits) -> Result<Self, DiagnosticError> {
        Ok(Self {
            sequence: 0,
            generation: 0,
            components: empty_slots(limits.max_components)?,
            lifetime: LifetimeState::default(),
            counters: empty_slots(limits.max_counters)?,
            windows: empty_slots(limits.max_window_buckets)?,
            extensions: empty_slots(limits.max_extensions)?,
            recent: BoundedHistory::new(limits.max_events),
            detail: DetailPool::new(limits.detail_pool_bytes, limits.detail_block_bytes)
                .map_err(|_| DiagnosticError::Allocation)?,
            metadata_used: 0,
            loss: DiagnosticLoss::default(),
        })
    }
    pub fn reserved_bytes(&self) -> usize {
        self.components.len() * size_of::<Option<ComponentSlot>>()
            + self.counters.len() * size_of::<Option<CounterSlot>>()
            + self.windows.len() * size_of::<Option<WindowSlot>>()
            + self.extensions.len() * size_of::<Option<ExtensionSlot>>()
            + self.recent.reserved_bytes()
            + self.detail.reserved_bytes()
    }
    pub fn next_sequence(&mut self) -> Result<u64, DiagnosticError> {
        match self.sequence.checked_add(1) {
            Some(n) => {
                self.sequence = n;
                Ok(n)
            }
            None => {
                self.loss.sequence_saturated = true;
                Err(DiagnosticError::SequenceExhausted)
            }
        }
    }
    pub fn next_generation(&mut self) -> Result<u64, DiagnosticError> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(DiagnosticError::SequenceExhausted)?;
        self.generation = generation;
        Ok(generation)
    }
    pub fn component(&self, handle: ComponentHandle) -> Option<&ComponentSlot> {
        self.components
            .get(handle.slot)?
            .as_ref()
            .filter(|v| v.handle == handle)
    }
    /// Populate already admitted slots once, preserving the existing pool order.
    pub fn reserve_builtin_counters(
        &mut self,
        handle: ComponentHandle,
        builtin_codes: &[BuiltinDiagnosticCode],
    ) -> Option<CounterIndex> {
        let mut counter_head = None;
        let mut counter_tail: Option<CounterIndex> = None;
        let mut codes = builtin_codes.iter();
        for index in 0..self.counters.len() {
            if self.counters[index].is_some() {
                continue;
            }
            let Some(code) = codes.next() else {
                break;
            };
            let current = CounterIndex::new(index);
            self.counters[index] = Some(CounterSlot {
                component: handle,
                next: None,
                code: StoredCode::Builtin(*code),
                occurrences: DiagnosticCount::default(),
                last_observed_at: None,
            });
            if let Some(previous) = counter_tail {
                self.counters[previous.slot()]
                    .as_mut()
                    .expect("reserved counter")
                    .next = Some(current);
            } else {
                counter_head = Some(current);
            }
            counter_tail = Some(current);
        }
        debug_assert_eq!(codes.len(), 0, "all builtin counters were admitted");
        counter_head
    }
    /// Visit only the owning registration's reservations, retaining pool order.
    pub fn component_counters(
        &self,
        handle: ComponentHandle,
    ) -> impl Iterator<Item = (CounterIndex, &CounterSlot)> {
        let mut next = self
            .component(handle)
            .and_then(|component| component.counter_head);
        std::iter::from_fn(move || {
            let index = next.take()?;
            let counter = self.counters.get(index.slot())?.as_ref()?;
            if counter.component != handle {
                return None;
            }
            next = counter.next;
            Some((index, counter))
        })
        .take(self.counters.len())
    }
    /// Insert a newly admitted extension counter into its component's ordered chain.
    pub fn link_counter(&mut self, handle: ComponentHandle, index: CounterIndex) {
        let mut previous: Option<CounterIndex> = None;
        let mut next = self
            .component(handle)
            .expect("admitted component")
            .counter_head;
        while let Some(current) = next {
            if current.slot() > index.slot() {
                break;
            }
            previous = next;
            next = self.counters[current.slot()]
                .as_ref()
                .expect("reserved counter")
                .next;
        }
        self.counters[index.slot()]
            .as_mut()
            .expect("admitted counter")
            .next = next;
        if let Some(previous) = previous {
            self.counters[previous.slot()]
                .as_mut()
                .expect("reserved counter")
                .next = Some(index);
        } else {
            self.components[handle.slot]
                .as_mut()
                .expect("admitted component")
                .counter_head = Some(index);
        }
    }
    pub fn code_name(&self, code: StoredCode) -> Option<&str> {
        match code {
            StoredCode::Builtin(code) => Some(code.canonical_name()),
            StoredCode::Extension { slot, generation } => self
                .extensions
                .get(slot)?
                .as_ref()
                .filter(|v| v.generation == generation)
                .map(|v| v.name.as_str()),
        }
    }
    pub fn class_name(&self, class: StoredFailureClass) -> Option<&str> {
        match class {
            StoredFailureClass::Builtin(class) => Some(class.as_str()),
            // Public ownership/context checks have already run. The unique
            // descriptor generation still protects retained references on reuse.
            StoredFailureClass::Extension {
                slot,
                generation,
                index,
            } => self
                .extensions
                .get(slot)?
                .as_ref()
                .filter(|value| value.generation == generation)?
                .opaque_classes
                .get(usize::from(index))?
                .as_ref()
                .map(Text::as_str),
        }
    }
    pub fn evict_event(&mut self, event: StoredObservation) {
        if let Some(component) = self
            .components
            .get_mut(event.component.slot)
            .and_then(Option::as_mut)
            .filter(|value| value.handle == event.component)
        {
            component.lifetime.totals.history_evictions_total.add(1);
            // The current period measures eviction activity since its refresh.
            component.refresh.totals.history_evictions_total.add(1);
        }
        self.lifetime.totals.history_evictions_total.add(1);
        self.loss.events_evicted.add(1);
        if let Some(handle) = event.text.payload {
            self.detail.free(handle);
        }
    }
    pub fn evict_oldest(&mut self) -> bool {
        if let Some(event) = self.recent.pop_front() {
            self.evict_event(event);
            true
        } else {
            false
        }
    }
    pub fn evict_closed_component(&mut self) -> bool {
        let candidate = self
            .components
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                value
                    .as_ref()
                    .filter(|value| value.closed_at.is_some())
                    .map(|value| (index, value.registered_tick))
            })
            .min_by_key(|value| value.1)
            .map(|value| value.0);
        let Some(index) = candidate else {
            return false;
        };
        self.reclaim_component(index, true)
    }
    pub(super) fn discard_unobserved_component(&mut self, handle: ComponentHandle) -> bool {
        if self.component(handle).is_none_or(|component| {
            component.lifetime.totals.recorded_events_total.value.get() != 0
        }) {
            return false;
        }
        self.reclaim_component(handle.slot, false)
    }
    fn reclaim_component(&mut self, index: usize, count_eviction: bool) -> bool {
        let component = *self.components[index].as_ref().expect("selected component");
        let handle = component.handle;
        // Charge evidence loss while the original registration is still present.
        // Visit each original entry once. Popping first guarantees a free
        // slot for each survivor, without shifting the remaining event headers.
        for _ in 0..self.recent.len() {
            let event = self.recent.pop_front().expect("retained event");
            if event.component == handle {
                self.evict_event(event);
            } else {
                let displaced = self.recent.push_back(event);
                debug_assert!(displaced.is_none(), "a popped slot must remain available");
            }
        }
        self.components[index] = None;
        self.metadata_used = self
            .metadata_used
            .saturating_sub(component.key.len() + component.refresh.ownership.bytes());
        let mut next = component.counter_head;
        while let Some(index) = next {
            let counter = self.counters[index.slot()]
                .take()
                .expect("reserved counter");
            debug_assert_eq!(counter.component, handle);
            next = counter.next;
        }
        if let Some(window) = component.window {
            self.windows[window.start..window.start + window.len].fill(None);
        }
        for slot in &mut self.extensions {
            if slot.as_ref().is_some_and(|value| value.component == handle)
                && let Some(old) = slot.take()
            {
                self.metadata_used = self.metadata_used.saturating_sub(old.metadata_bytes());
            }
        }
        if count_eviction {
            self.loss.components_evicted.add(1);
        }
        let _ = self.next_sequence();
        true
    }
    /// Admit message, session, and request in priority order, before evicting
    /// anything. An intrinsically impossible field cannot displace history.
    pub fn retain_event_text(&mut self, fields: [Option<&str>; 3]) -> RetainedText {
        let mut selected = [""; 3];
        let mut ranges = [None; 3];
        let mut plan = None;
        let mut len = 0u16;
        for (index, field) in fields.into_iter().enumerate() {
            let Some(field) = field else { continue };
            let Ok(field_len) = u16::try_from(field.len()) else {
                continue;
            };
            let Some(end) = len.checked_add(field_len) else {
                continue;
            };
            selected[index] = field;
            if let Some(candidate) = self.detail.plan_parts(selected) {
                plan = Some(candidate);
                ranges[index] = Some(TextRange {
                    start: len,
                    len: field_len,
                });
                len = end;
            } else {
                selected[index] = "";
            }
        }
        if ranges.iter().all(Option::is_none) {
            return RetainedText::default();
        }
        let plan = plan.expect("admitted fields have a storage plan");
        let payload = loop {
            if let Some(handle) = self.detail.allocate_plan(&plan) {
                break handle;
            }
            if !self.evict_oldest() {
                return RetainedText::default();
            }
        };
        RetainedText {
            payload: Some(payload),
            detail: ranges[0],
            session_id: ranges[1],
            request_id: ranges[2],
        }
    }
    pub fn record_window(
        &mut self,
        handle: ComponentHandle,
        tick: u64,
        outcome: DiagnosticOutcome,
        count: u64,
    ) {
        let Some(component) = self.component(handle).copied() else {
            return;
        };
        let Some(window) = component.window else {
            return;
        };
        let interval = tick.saturating_sub(component.registered_tick) / window.resolution_millis;
        let index = window.start + (interval % (window.len as u64)) as usize;
        let slot = self.windows[index].as_mut().expect("window reservation");
        if slot.interval != Some(interval) {
            slot.interval = Some(interval);
            slot.successes = DiagnosticCount::default();
            slot.failures = DiagnosticCount::default();
            slot.unspecified = DiagnosticCount::default();
        }
        match outcome {
            DiagnosticOutcome::Success => slot.successes.add(count),
            DiagnosticOutcome::Failure => slot.failures.add(count),
            _ => slot.unspecified.add(count),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::{
        ComponentRegistration, Diagnostics, DiagnosticsSettings, ExtensionClassHandle, lock,
    };

    #[test]
    fn inline_text_checks_utf8_byte_capacity_without_allocating_backing() {
        assert_eq!(text::<8>("🙂🙂").unwrap().as_str(), "🙂🙂");
        assert_eq!(text::<8>("🙂🙂x"), Err(DiagnosticError::MetadataCapacity));
        assert_eq!(text::<8>("€€").unwrap().len(), 6);
        assert_eq!(text::<0>("").unwrap().as_str(), "");
        assert_eq!(text::<0>("x"), Err(DiagnosticError::MetadataCapacity));
    }

    #[test]
    fn component_counter_reservations_visit_only_the_owning_chain() {
        let diagnostics = Diagnostics::from_settings(
            Diagnostics::default().source().clone(),
            DiagnosticsSettings {
                enabled: true,
                limits: DiagnosticLimits {
                    max_counters: 64 * BuiltinDiagnosticCode::all().len(),
                    ..Default::default()
                },
            },
        )
        .unwrap();
        let codes = BuiltinDiagnosticCode::all();
        let recorders: Vec<_> = (0..64)
            .map(|index| {
                diagnostics
                    .register_component(ComponentRegistration::new(
                        &format!("component.{index}"),
                        codes,
                    ))
                    .unwrap()
            })
            .collect();
        let state = lock(&diagnostics.inner.backend.as_ref().unwrap().state);
        for recorder in &recorders {
            let handle = recorder.lease.as_ref().unwrap().handle;
            let counters: Vec<_> = state.component_counters(handle).collect();
            assert_eq!(counters.len(), codes.len());
            assert!(
                counters
                    .iter()
                    .all(|(_, counter)| counter.component == handle)
            );
            assert!(
                counters
                    .windows(2)
                    .all(|pair| pair[0].0.slot() < pair[1].0.slot())
            );
        }
        let last = recorders.last().unwrap().lease.as_ref().unwrap().handle;
        assert!(state.component_counters(last).next().unwrap().0.slot() > 1000);
    }

    #[test]
    fn stored_extension_causes_are_compact_and_generation_checked() {
        assert!(size_of::<StoredFailureClass>() < size_of::<ExtensionClassHandle>());
        let mut state = State::new(&DiagnosticLimits::default()).unwrap();
        state.extensions[0] = Some(ExtensionSlot {
            generation: 1,
            component: ComponentHandle {
                slot: 0,
                generation: 1,
            },
            name: text("acme.failure").unwrap(),
            description: text("custom").unwrap(),
            allowed_classes: enumset::EnumSet::new(),
            opaque_classes: [Some(text("original").unwrap()), None, None, None],
            outcome: DiagnosticOutcome::Failure,
        });
        let original = StoredFailureClass::Extension {
            slot: 0,
            generation: 1,
            index: 0,
        };
        assert_eq!(state.class_name(original), Some("original"));
        let descriptor = state.extensions[0].as_mut().unwrap();
        descriptor.generation = 2;
        descriptor.component = ComponentHandle {
            slot: 1,
            generation: 2,
        };
        descriptor.opaque_classes[0] = Some(text("replacement").unwrap());
        assert_eq!(state.class_name(original), None);
        assert_eq!(
            state.class_name(StoredFailureClass::Extension {
                slot: 0,
                generation: 2,
                index: 0
            }),
            Some("replacement")
        );
    }

    #[test]
    #[ignore = "informational private layout and reserved-backing inspection"]
    fn diagnostic_storage_layout_profile() {
        println!(
            "event={} origin={} ownership={} stored_cause={} optional_cause={} detail_handle={} optional_detail_handle={} counter_slot={} optional_counter_slot={} window_slot={} optional_window_slot={}",
            size_of::<StoredObservation>(),
            size_of::<CapturedOrigin>(),
            size_of::<Ownership>(),
            size_of::<StoredFailureClass>(),
            size_of::<Option<StoredFailureClass>>(),
            size_of::<DetailHandle>(),
            size_of::<Option<DetailHandle>>(),
            size_of::<CounterSlot>(),
            size_of::<Option<CounterSlot>>(),
            size_of::<WindowSlot>(),
            size_of::<Option<WindowSlot>>(),
        );
        for (name, offset) in [
            ("origin", std::mem::offset_of!(StoredObservation, origin)),
            (
                "cause",
                std::mem::offset_of!(StoredObservation, failure_class),
            ),
            ("text", std::mem::offset_of!(StoredObservation, text)),
        ] {
            println!("event field {name} offset={offset}");
        }
        let diagnostics = Diagnostics::from_settings(
            Diagnostics::default().source().clone(),
            DiagnosticsSettings {
                enabled: true,
                ..Default::default()
            },
        )
        .unwrap();
        println!(
            "default reserved backing={}",
            diagnostics.memory_usage().reserved_bytes
        );
    }
}
