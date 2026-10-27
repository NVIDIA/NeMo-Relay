// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic component reservations and owner-authorized refresh transitions.

use super::{
    BuiltinDiagnosticCode, ComponentHandle, ComponentIdentity, ComponentOwnership,
    ComponentRegistration, ComponentSlot, CounterIndex, CounterSlot, DESCRIPTION_BYTES,
    DiagnosticCount, DiagnosticError, DiagnosticLimits, DiagnosticOutcome, DiagnosticTotals,
    ExtensionClassHandle, ExtensionHandle, ExtensionRegistration, ExtensionSlot, FailureClass,
    ID_BYTES, Inner, LifetimeState, Ownership, RefreshState, State, StoredCode, WindowReservation,
    WindowSlot, lock, text,
};
use chrono::Utc;
use uuid::Uuid;

fn valid_name(value: &str, max: usize) -> bool {
    !value.is_empty()
        && value.len() <= max
        && value
            .bytes()
            .all(|byte| matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' | b'.'))
        && value.split('.').all(|part| !part.is_empty())
}
fn ownership(
    value: &ComponentOwnership,
    limits: &DiagnosticLimits,
) -> Result<Ownership, DiagnosticError> {
    if value
        .configuration_id
        .as_ref()
        .is_some_and(|v| v.len() > limits.max_identity_bytes)
        || value
            .responsibility
            .as_ref()
            .is_some_and(|v| v.len() > limits.max_identity_bytes)
    {
        return Err(DiagnosticError::MetadataCapacity);
    }
    Ownership::new(value)
}
fn free_window_start(state: &State, len: usize) -> Option<usize> {
    if len == 0 {
        return Some(0);
    }
    if len > state.windows.len() {
        return None;
    }
    let mut free = 0;
    for (index, slot) in state.windows.iter().enumerate() {
        if slot.is_none() {
            free += 1;
        } else {
            free = 0;
        }
        if free == len {
            return Some(index + 1 - len);
        }
    }
    None
}
fn ensure_revision(state: &mut State) -> Result<(), DiagnosticError> {
    if state.sequence == u64::MAX || state.generation == u64::MAX {
        state.loss.sequence_saturated = true;
        Err(DiagnosticError::SequenceExhausted)
    } else {
        Ok(())
    }
}
fn note_admission(state: &mut State, error: DiagnosticError) -> DiagnosticError {
    match error {
        DiagnosticError::MetadataCapacity => state.loss.metadata_rejected.add(1),
        DiagnosticError::WindowCapacity => state.loss.windows_rejected.add(1),
        DiagnosticError::KeyCapacity => state.loss.keys_rejected.add(1),
        _ => {}
    }
    error
}
fn component_admission(
    state: &State,
    inner: &Inner,
    metadata: usize,
    codes: usize,
    windows: usize,
) -> Option<DiagnosticError> {
    if !state.components.iter().any(Option::is_none) {
        Some(DiagnosticError::ComponentCapacity)
    } else if state.counters.iter().filter(|slot| slot.is_none()).count() < codes {
        Some(DiagnosticError::KeyCapacity)
    } else if free_window_start(state, windows).is_none() {
        Some(DiagnosticError::WindowCapacity)
    } else if state
        .metadata_used
        .checked_add(metadata)
        .is_none_or(|n| n > inner.limits.max_metadata_bytes)
    {
        Some(DiagnosticError::MetadataCapacity)
    } else {
        None
    }
}

pub(super) fn register_component(
    state: &mut State,
    inner: &Inner,
    registration: ComponentRegistration<'_>,
) -> Result<(ComponentHandle, ComponentIdentity), DiagnosticError> {
    if !valid_name(registration.key, inner.limits.max_identity_bytes) {
        return Err(DiagnosticError::InvalidComponentKey);
    }
    if registration.builtin_codes.len() > state.counters.len() {
        return Err(note_admission(state, DiagnosticError::KeyCapacity));
    }
    for (index, code) in registration.builtin_codes.iter().enumerate() {
        if registration.builtin_codes[..index].contains(code) {
            return Err(DiagnosticError::InvalidCode);
        }
    }
    let key = text(registration.key)?;
    let attribution =
        ownership(&registration.ownership, &inner.limits).map_err(|e| note_admission(state, e))?;
    let metadata = key
        .len()
        .checked_add(attribution.bytes())
        .ok_or(DiagnosticError::MetadataCapacity)?;
    if metadata > inner.limits.max_metadata_bytes {
        return Err(note_admission(state, DiagnosticError::MetadataCapacity));
    }
    let window_count = registration.window.map_or(0, |v| v.bucket_count);
    if registration
        .window
        .is_some_and(|v| v.bucket_count == 0 || v.resolution_millis == 0)
        || window_count > state.windows.len()
    {
        return Err(note_admission(state, DiagnosticError::WindowCapacity));
    }
    ensure_revision(state)?;
    while let Some(error) = component_admission(
        state,
        inner,
        metadata,
        registration.builtin_codes.len(),
        window_count,
    ) {
        // Only closed registrations can yield reservations to another component.
        if state.evict_closed_component() {
            continue;
        }
        return Err(note_admission(state, error));
    }
    let slot = state
        .components
        .iter()
        .position(Option::is_none)
        .expect("component admitted");
    let window_start = free_window_start(state, window_count).expect("window admitted");
    let generation = state.next_generation()?;
    state.next_sequence()?;
    let now = Utc::now();
    let tick = inner.tick();
    let handle = ComponentHandle { slot, generation };
    let instance_id = Uuid::now_v7();
    let window = registration.window.map(|v| WindowReservation {
        start: window_start,
        len: v.bucket_count,
        resolution_millis: v.resolution_millis,
    });
    let counter_head = state.reserve_builtin_counters(handle, registration.builtin_codes);
    if let Some(window) = window {
        for index in window.start..window.start + window.len {
            state.windows[index] = Some(WindowSlot {
                interval: None,
                successes: DiagnosticCount::default(),
                failures: DiagnosticCount::default(),
                unspecified: DiagnosticCount::default(),
            });
        }
    }
    state.components[slot] = Some(ComponentSlot {
        handle,
        instance_id,
        key,
        registered_at: now,
        registered_tick: tick,
        closed_at: None,
        closed_tick: None,
        lifetime: LifetimeState::default(),
        refresh: RefreshState {
            generation: 1,
            started_at: now,
            started_tick: tick,
            ownership: attribution,
            totals: DiagnosticTotals::default(),
        },
        window,
        counter_head,
    });
    state.metadata_used += metadata;
    Ok((
        handle,
        ComponentIdentity {
            key: registration.key.to_owned(),
            instance_id,
            generation: generation.into(),
        },
    ))
}

pub(super) fn register_extension(
    state: &mut State,
    inner: &Inner,
    handle: ComponentHandle,
    registration: ExtensionRegistration<'_>,
) -> Result<ExtensionHandle, DiagnosticError> {
    state
        .component(handle)
        .filter(|value| value.closed_at.is_none())
        .ok_or(DiagnosticError::ComponentClosed)?;
    if !valid_name(registration.namespace, inner.limits.max_identity_bytes)
        || !valid_name(registration.name, inner.limits.max_identity_bytes)
    {
        return Err(DiagnosticError::InvalidCode);
    }
    if matches!(
        registration.namespace.split('.').next(),
        Some(
            "runtime"
                | "plugin"
                | "upstream"
                | "observability"
                | "logging"
                | "binding"
                | "otel"
                | "atif"
                | "configuration"
                | "deployment"
        )
    ) {
        return Err(DiagnosticError::InvalidCode);
    }
    let len = registration
        .namespace
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_add(registration.name.len()))
        .ok_or(DiagnosticError::MetadataCapacity)?;
    if len > inner.limits.max_identity_bytes || registration.description.len() > DESCRIPTION_BYTES {
        return Err(note_admission(state, DiagnosticError::MetadataCapacity));
    }
    let mut bytes = [0u8; ID_BYTES];
    bytes[..registration.namespace.len()].copy_from_slice(registration.namespace.as_bytes());
    bytes[registration.namespace.len()] = b'.';
    bytes[registration.namespace.len() + 1..len].copy_from_slice(registration.name.as_bytes());
    let name = std::str::from_utf8(&bytes[..len]).expect("admitted ASCII name");
    if BuiltinDiagnosticCode::from_canonical_name(name).is_some() {
        return Err(DiagnosticError::InvalidCode);
    }
    let mut classes = enumset::EnumSet::new();
    for class in registration.allowed_classes {
        if !classes.insert(*class) {
            return Err(DiagnosticError::InvalidClassification);
        }
    }
    if registration.outcome != DiagnosticOutcome::Failure && !classes.is_empty() {
        return Err(DiagnosticError::InvalidClassification);
    }
    for (slot, value) in state.extensions.iter().enumerate() {
        if let Some(value) = value.filter(|v| v.component == handle && v.name.as_str() == name) {
            if value.description.as_str() != registration.description
                || value.allowed_classes != classes
                || value.outcome != registration.outcome
            {
                return Err(DiagnosticError::InvalidCode);
            }
            return Ok(ExtensionHandle {
                context_id: inner.context_id,
                component: handle,
                slot,
                generation: value.generation,
            });
        }
    }
    let metadata = len
        .checked_add(registration.description.len())
        .ok_or(DiagnosticError::MetadataCapacity)?;
    if metadata > inner.limits.max_metadata_bytes {
        return Err(note_admission(state, DiagnosticError::MetadataCapacity));
    }
    ensure_revision(state)?;
    admit_extension(state, inner, metadata)?;
    let slot = state
        .extensions
        .iter()
        .position(Option::is_none)
        .expect("extension admitted");
    let counter = state
        .counters
        .iter()
        .position(Option::is_none)
        .expect("counter admitted");
    let generation = state.next_generation()?;
    state.next_sequence()?;
    state.extensions[slot] = Some(ExtensionSlot {
        generation,
        component: handle,
        name: text(name)?,
        description: text(registration.description)?,
        allowed_classes: classes,
        opaque_classes: [None; 4],
        outcome: registration.outcome,
    });
    state.counters[counter] = Some(CounterSlot {
        component: handle,
        next: None,
        code: StoredCode::Extension { slot, generation },
        occurrences: DiagnosticCount::default(),
        last_observed_at: None,
    });
    state.link_counter(handle, CounterIndex::new(counter));
    state.metadata_used += metadata;
    Ok(ExtensionHandle {
        context_id: inner.context_id,
        component: handle,
        slot,
        generation,
    })
}

pub(super) fn close_component(
    inner: &Inner,
    handle: ComponentHandle,
) -> Result<bool, DiagnosticError> {
    let Some(backend) = &inner.backend else {
        return Ok(false);
    };
    let mut state = lock(&backend.state);
    let old = *state
        .component(handle)
        .ok_or(DiagnosticError::ComponentClosed)?;
    if old.closed_at.is_some() {
        return Ok(false);
    }
    state.next_sequence()?;
    let now = Utc::now();
    let tick = inner.tick();
    let component = state.components[handle.slot]
        .as_mut()
        .expect("validated component");
    component.closed_at = Some(now);
    component.closed_tick = Some(tick);
    Ok(true)
}

/// Reset the single current-period summary while preserving component/context
/// lifetimes, code totals, rolling windows, and original retained event evidence.
pub(super) fn refresh_component(
    inner: &Inner,
    handle: ComponentHandle,
    new_ownership: ComponentOwnership,
) -> Result<u64, DiagnosticError> {
    let Some(backend) = &inner.backend else {
        return Err(DiagnosticError::Disabled);
    };
    let mut state = lock(&backend.state);
    let attribution = ownership(&new_ownership, &inner.limits)
        .map_err(|error| note_admission(&mut state, error))?;
    let old = *state
        .component(handle)
        .filter(|value| value.closed_at.is_none())
        .ok_or(DiagnosticError::ComponentClosed)?;
    ensure_revision(&mut state)?;
    let generation = old
        .refresh
        .generation
        .checked_add(1)
        .ok_or(DiagnosticError::SequenceExhausted)?;
    while state
        .metadata_used
        .saturating_sub(old.refresh.ownership.bytes())
        .checked_add(attribution.bytes())
        .is_none_or(|value| value > inner.limits.max_metadata_bytes)
    {
        if state.evict_closed_component() {
            continue;
        }
        return Err(note_admission(
            &mut state,
            DiagnosticError::MetadataCapacity,
        ));
    }
    state.next_sequence()?;
    state.metadata_used = state.metadata_used - old.refresh.ownership.bytes() + attribution.bytes();
    let now = Utc::now();
    let tick = inner.tick();
    let component = state.components[handle.slot]
        .as_mut()
        .expect("validated component");
    component.refresh = RefreshState {
        generation,
        started_at: now,
        started_tick: tick,
        ownership: attribution,
        totals: DiagnosticTotals::default(),
    };
    Ok(generation)
}

pub(super) fn register_extension_class(
    state: &mut State,
    inner: &Inner,
    component: ComponentHandle,
    extension: ExtensionHandle,
    name: &str,
) -> Result<ExtensionClassHandle, DiagnosticError> {
    if extension.context_id != inner.context_id || extension.component != component {
        return Err(DiagnosticError::InvalidCode);
    }
    state
        .component(component)
        .filter(|value| value.closed_at.is_none())
        .ok_or(DiagnosticError::ComponentClosed)?;
    // Class descriptors are immutable bounded snake_case names, not arbitrary
    // error text. Their fixed slots never create additional aggregate series.
    if name.is_empty()
        || name.len() > 32
        || !name.as_bytes()[0].is_ascii_lowercase()
        || !name
            .bytes()
            .all(|byte| matches!(byte,b'a'..=b'z'|b'0'..=b'9'|b'_'))
    {
        return Err(DiagnosticError::InvalidClassification);
    }
    let descriptor = *state
        .extensions
        .get(extension.slot)
        .and_then(Option::as_ref)
        .filter(|value| value.component == component && value.generation == extension.generation)
        .ok_or(DiagnosticError::InvalidCode)?;
    if descriptor.outcome != DiagnosticOutcome::Failure {
        return Err(DiagnosticError::InvalidClassification);
    }
    if let Some(index) = descriptor
        .opaque_classes
        .iter()
        .position(|value| value.as_ref().is_some_and(|value| value.as_str() == name))
    {
        return Ok(ExtensionClassHandle { extension, index });
    }
    // A known portable class must use its built-in value, rather than a second
    // descriptor with the same spelling and potentially different admission.
    if FailureClass::from_canonical_name(name).is_some() {
        return Err(DiagnosticError::InvalidClassification);
    }
    let index = descriptor
        .opaque_classes
        .iter()
        .position(Option::is_none)
        .ok_or_else(|| note_admission(state, DiagnosticError::MetadataCapacity))?;
    ensure_revision(state)?;
    while state
        .metadata_used
        .checked_add(name.len())
        .is_none_or(|value| value > inner.limits.max_metadata_bytes)
    {
        if state.evict_closed_component() {
            continue;
        }
        return Err(note_admission(state, DiagnosticError::MetadataCapacity));
    }
    let text = text(name)?;
    state.next_sequence()?;
    state.extensions[extension.slot]
        .as_mut()
        .expect("validated extension")
        .opaque_classes[index] = Some(text);
    state.metadata_used += name.len();
    Ok(ExtensionClassHandle { extension, index })
}

fn admit_extension(
    state: &mut State,
    inner: &Inner,
    metadata: usize,
) -> Result<(), DiagnosticError> {
    loop {
        let error = if !state.extensions.iter().any(Option::is_none)
            || !state.counters.iter().any(Option::is_none)
        {
            Some(DiagnosticError::KeyCapacity)
        } else if state
            .metadata_used
            .checked_add(metadata)
            .is_none_or(|n| n > inner.limits.max_metadata_bytes)
        {
            Some(DiagnosticError::MetadataCapacity)
        } else {
            None
        };
        let Some(error) = error else {
            break;
        };
        if state.evict_closed_component() {
            continue;
        }
        return Err(note_admission(state, error));
    }
    Ok(())
}
