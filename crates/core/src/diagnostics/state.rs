// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-capacity local state. No wire DTO graph is retained in this store.

use super::storage::{BoundedHistory, DetailHandle, DetailPool};
use super::*;

pub(super) const ID_BYTES: usize = 128;
pub(super) const DESCRIPTION_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Text<const N: usize> {
    bytes: [u8; N],
    len: usize,
}
impl<const N: usize> Text<N> {
    pub fn new(value: &str) -> Result<Self, DiagnosticError> {
        if value.len() > N {
            return Err(DiagnosticError::MetadataCapacity);
        }
        let mut text = Self {
            bytes: [0; N],
            len: value.len(),
        };
        text.bytes[..value.len()].copy_from_slice(value.as_bytes());
        Ok(text)
    }
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).expect("text is copied from UTF-8")
    }
    pub fn len(&self) -> usize {
        self.len
    }
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
            configuration_id: value
                .configuration_id
                .as_deref()
                .map(Text::new)
                .transpose()?,
            responsibility: value.responsibility.as_deref().map(Text::new).transpose()?,
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
#[derive(Clone, Copy)]
pub(super) struct CounterSlot {
    pub component: ComponentHandle,
    pub code: StoredCode,
    pub occurrences: DiagnosticCount,
    pub last_observed_at: Option<ObservationTime>,
}
#[derive(Clone, Copy)]
pub(super) struct WindowSlot {
    pub component: ComponentHandle,
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
    pub allowed_classes: u32,
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
    Extension(ExtensionClassHandle),
}
#[derive(Clone, Copy)]
pub(super) struct StoredObservation {
    pub sequence: u64,
    pub observed_at: ObservationTime,
    pub count: u64,
    pub component: ComponentHandle,
    pub origin: CapturedOrigin,
    pub code: StoredCode,
    pub failure_class: Option<StoredFailureClass>,
    pub context: PluginObservationContext,
    pub severity: DiagnosticSeverity,
    pub outcome: DiagnosticOutcome,
    pub detail: Option<DetailHandle>,
    pub detail_truncated: bool,
    pub session_id: Option<DetailHandle>,
    pub request_id: Option<DetailHandle>,
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
            StoredFailureClass::Extension(handle) => self
                .extensions
                .get(handle.extension.slot)?
                .as_ref()
                .filter(|value| {
                    value.generation == handle.extension.generation
                        && value.component == handle.extension.component
                })?
                .opaque_classes
                .get(handle.index)?
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
        for handle in [event.detail, event.session_id, event.request_id]
            .into_iter()
            .flatten()
        {
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
        let component = *self.components[index].as_ref().expect("selected component");
        let handle = component.handle;
        // Charge evidence loss while the original registration is still present.
        let mut event = 0;
        while event < self.recent.len() {
            if self
                .recent
                .get(event)
                .is_some_and(|value| value.component == handle)
            {
                let old = self.recent.remove(event).expect("retained event");
                self.evict_event(old);
            } else {
                event += 1;
            }
        }
        self.components[index] = None;
        self.metadata_used = self
            .metadata_used
            .saturating_sub(component.key.len() + component.refresh.ownership.bytes());
        for counter in &mut self.counters {
            if counter
                .as_ref()
                .is_some_and(|value| value.component == handle)
            {
                *counter = None;
            }
        }
        for bucket in &mut self.windows {
            if bucket
                .as_ref()
                .is_some_and(|value| value.component == handle)
            {
                *bucket = None;
            }
        }
        for slot in &mut self.extensions {
            if slot.as_ref().is_some_and(|value| value.component == handle)
                && let Some(old) = slot.take()
            {
                self.metadata_used = self.metadata_used.saturating_sub(old.metadata_bytes());
            }
        }
        self.loss.components_evicted.add(1);
        let _ = self.next_sequence();
        true
    }
    pub fn allocate_detail(&mut self, value: &str, max: usize) -> Option<DetailHandle> {
        if !self.detail.can_fit(value, max) {
            return None;
        }
        loop {
            if let Some(handle) = self.detail.allocate(value, max) {
                return Some(handle);
            }
            if !self.evict_oldest() {
                return None;
            }
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
