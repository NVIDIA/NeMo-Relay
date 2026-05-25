// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::api::event::Event;
use crate::api::runtime::EventSubscriberFn;
use crate::api::runtime::ScopeStackHandle;
use crate::api::runtime::current_scope_stack;
use crate::api::runtime::global_context;
use crate::api::shared::ensure_runtime_owner;
use crate::error::{FlowError, Result};
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Event as TracingEvent, Metadata, Subscriber as TracingSubscriber};
use tracing_subscriber::layer::Context as LayerContext;
use uuid::Uuid;

/// `tracing` target used by NeMo Relay lifecycle event records.
///
/// External tracing layers can filter on this target before decoding records
/// with [`event_from_tracing`].
pub const EVENT_TRACE_TARGET: &str = "nemo_relay::events";

/// `tracing` field containing the canonical ATOF JSON representation.
///
/// The field is part of NeMo Relay's Rust tracing integration contract. Prefer
/// [`event_from_tracing`] over reading this field directly when a library wants
/// a canonical [`Event`].
pub const EVENT_JSON_FIELD: &str = "event.json";

/// Callback-backed NeMo Relay event subscriber.
///
/// This adapter consumes the structured `tracing` records emitted by the core
/// runtime, reconstructs the canonical NeMo Relay [`Event`], and invokes the
/// configured callback. It implements [`tracing::Subscriber`] and
/// [`tracing_subscriber::Layer`] so Rust hosts can install it directly as a
/// tracing collector or compose it into an existing subscriber stack.
pub struct Subscriber {
    callback: EventSubscriberFn,
    next_span_id: Mutex<u64>,
}

impl Subscriber {
    /// Create a tracing-compatible subscriber from a NeMo Relay event callback.
    pub fn new(callback: EventSubscriberFn) -> Self {
        Self {
            callback,
            next_span_id: Mutex::new(1),
        }
    }

    /// Return a clone of the callback used by this subscriber.
    pub fn callback(&self) -> EventSubscriberFn {
        self.callback.clone()
    }

    /// Convert this subscriber back into its callback.
    pub fn into_callback(self) -> EventSubscriberFn {
        self.callback
    }

    fn observe_tracing_event(&self, event: &TracingEvent<'_>) {
        if let Some(event) = event_from_tracing(event) {
            (self.callback)(&event);
        }
    }
}

impl TracingSubscriber for Subscriber {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        is_nemo_relay_event(metadata)
    }

    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if is_nemo_relay_event(metadata) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        let mut next_span_id = self
            .next_span_id
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let id = *next_span_id;
        *next_span_id = next_span_id.saturating_add(1);
        Id::from_u64(id)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &TracingEvent<'_>) {
        self.observe_tracing_event(event);
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

impl<S> tracing_subscriber::Layer<S> for Subscriber
where
    S: TracingSubscriber,
{
    fn on_event(&self, event: &TracingEvent<'_>, _ctx: LayerContext<'_, S>) {
        self.observe_tracing_event(event);
    }
}

/// Create a tracing-subscriber layer that consumes NeMo Relay lifecycle events.
///
/// This is the canonical Rust integration point for libraries that want to
/// receive NeMo Relay events through `tracing-subscriber` composition instead
/// of registering directly in the NeMo Relay subscriber registry.
pub fn tracing_layer(callback: EventSubscriberFn) -> Subscriber {
    Subscriber::new(callback)
}

/// Return `true` when tracing metadata belongs to a NeMo Relay lifecycle event.
///
/// External layers can use this as a cheap filter before calling
/// [`event_from_tracing`].
pub fn is_nemo_relay_event(metadata: &Metadata<'_>) -> bool {
    metadata.target() == EVENT_TRACE_TARGET
}

/// Decode a canonical NeMo Relay [`Event`] from a `tracing` event record.
///
/// Returns `None` when the tracing event is not a NeMo Relay lifecycle record or
/// when the record does not contain a valid canonical event payload.
pub fn event_from_tracing(event: &TracingEvent<'_>) -> Option<Event> {
    if !is_nemo_relay_event(event.metadata()) {
        return None;
    }

    let mut visitor = EventJsonVisitor::default();
    event.record(&mut visitor);
    let event_json = visitor.event_json?;
    serde_json::from_str::<Event>(&event_json).ok()
}

#[derive(Default)]
struct EventJsonVisitor {
    event_json: Option<String>,
}

impl Visit for EventJsonVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == EVENT_JSON_FIELD {
            self.event_json = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == EVENT_JSON_FIELD && self.event_json.is_none() {
            self.event_json = Some(format!("{value:?}"));
        }
    }
}

/// Closeable handle for an anonymous global event subscriber.
///
/// Dropping the handle performs a best-effort deregistration. Call
/// [`SubscriberHandle::close`] when the caller needs to know whether the
/// subscriber was removed.
pub struct SubscriberHandle {
    id: Uuid,
    closed: AtomicBool,
}

impl SubscriberHandle {
    fn new(id: Uuid) -> Self {
        Self {
            id,
            closed: AtomicBool::new(false),
        }
    }

    /// Return the runtime-generated identifier for this subscription.
    ///
    /// The identifier is opaque and is intended for diagnostics only.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Deregister this subscriber.
    ///
    /// Returns `true` when the subscriber was still registered and was removed.
    /// Returns `false` when the handle had already been closed.
    pub fn close(&self) -> Result<bool> {
        if self
            .closed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(false);
        }

        match close_global_subscription(self.id) {
            Ok(removed) => Ok(removed),
            Err(error) => {
                self.closed.store(false, Ordering::Release);
                Err(error)
            }
        }
    }
}

impl Drop for SubscriberHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Closeable handle for an anonymous scope-local event subscriber.
///
/// Dropping the handle performs a best-effort deregistration. Scope pop also
/// removes the subscriber automatically, so closing after scope cleanup returns
/// `Ok(false)`.
pub struct ScopeSubscriberHandle {
    scope_uuid: Uuid,
    id: Uuid,
    scope_stack: ScopeStackHandle,
    closed: AtomicBool,
}

impl ScopeSubscriberHandle {
    fn new(scope_uuid: Uuid, id: Uuid, scope_stack: ScopeStackHandle) -> Self {
        Self {
            scope_uuid,
            id,
            scope_stack,
            closed: AtomicBool::new(false),
        }
    }

    /// Return the runtime-generated identifier for this subscription.
    ///
    /// The identifier is opaque and is intended for diagnostics only.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Return the UUID of the scope that owns this subscription.
    pub fn scope_uuid(&self) -> Uuid {
        self.scope_uuid
    }

    /// Deregister this scope-local subscriber.
    ///
    /// Returns `true` when the subscriber was still registered and was removed.
    /// Returns `false` when the handle had already been closed or the owning
    /// scope has already been popped.
    pub fn close(&self) -> Result<bool> {
        if self
            .closed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(false);
        }

        match close_scope_subscription(&self.scope_stack, self.scope_uuid, self.id) {
            Ok(removed) => Ok(removed),
            Err(error) => {
                self.closed.store(false, Ordering::Release);
                Err(error)
            }
        }
    }
}

impl Drop for ScopeSubscriberHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Register an anonymous global lifecycle event subscriber.
///
/// The returned handle owns the registration and can be closed explicitly or
/// dropped to deregister the subscriber.
pub fn subscribe(callback: EventSubscriberFn) -> Result<SubscriberHandle> {
    ensure_runtime_owner()?;
    let id = Uuid::now_v7();
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    state.insert_anonymous_event_subscriber(id, callback);
    Ok(SubscriberHandle::new(id))
}

/// Register an anonymous scope-local lifecycle event subscriber.
///
/// The returned handle owns the registration and captures the active scope
/// stack so it can be closed even if another scope stack is current later.
pub fn scope_subscribe(
    scope_uuid: &Uuid,
    callback: EventSubscriberFn,
) -> Result<ScopeSubscriberHandle> {
    ensure_runtime_owner()?;
    let id = Uuid::now_v7();
    let scope_stack = current_scope_stack();
    {
        let mut guard = scope_stack
            .write()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let registries = guard
            .local_registries_mut(scope_uuid)
            .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
        registries.anonymous_event_subscribers.insert(id, callback);
    }
    Ok(ScopeSubscriberHandle::new(*scope_uuid, id, scope_stack))
}

fn close_global_subscription(id: Uuid) -> Result<bool> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(state.remove_anonymous_event_subscriber(&id))
}

fn close_scope_subscription(
    scope_stack: &ScopeStackHandle,
    scope_uuid: Uuid,
    id: Uuid,
) -> Result<bool> {
    ensure_runtime_owner()?;
    let mut guard = scope_stack
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let registries = match guard.local_registries_mut(&scope_uuid) {
        Some(registries) => registries,
        None => return Ok(false),
    };
    Ok(registries.anonymous_event_subscribers.remove(&id).is_some())
}

/// Register a global lifecycle event subscriber.
///
/// The subscriber is added to the process-wide registry and receives every
/// emitted scope, tool, LLM, and mark event until it is deregistered.
///
/// # Parameters
/// - `name`: Unique subscriber name in the global registry.
/// - `callback`: Subscriber callback invoked for each emitted event.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the subscriber was registered.
///
/// # Errors
/// Returns [`FlowError::AlreadyExists`] when another global subscriber is
/// already registered under the same name.
///
/// # Notes
/// Global subscribers remain active across scopes until explicitly removed.
pub fn register_subscriber(name: &str, callback: EventSubscriberFn) -> Result<()> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    state.register_event_subscriber(name, callback)
}

/// Deregister a global lifecycle event subscriber.
///
/// This removes the named subscriber from the process-wide registry.
///
/// # Parameters
/// - `name`: Global subscriber name to remove.
///
/// # Returns
/// A [`Result`] containing `true` when a subscriber was removed and `false`
/// when the name was not registered.
///
/// # Errors
/// Returns an error when the global registry lock cannot be acquired safely.
///
/// # Notes
/// Deregistration affects only future event delivery.
pub fn deregister_subscriber(name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(state.deregister_event_subscriber(name))
}

/// Register a scope-local lifecycle event subscriber.
///
/// The subscriber remains active only while the target scope is still present
/// on the active scope stack.
///
/// # Parameters
/// - `scope_uuid`: UUID of the owning scope.
/// - `name`: Unique subscriber name within the owning scope.
/// - `callback`: Subscriber callback invoked for events emitted under that
///   scope hierarchy.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the subscriber was registered.
///
/// # Errors
/// Returns [`FlowError::NotFound`] when the scope does not exist on the active
/// stack and [`FlowError::AlreadyExists`] when the scope already owns a
/// subscriber with the same name.
///
/// # Notes
/// Scope-local subscribers are removed automatically when the owning scope is
/// popped.
pub fn scope_register_subscriber(
    scope_uuid: &uuid::Uuid,
    name: &str,
    callback: EventSubscriberFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    if registries.event_subscribers.contains_key(name) {
        return Err(FlowError::AlreadyExists(format!(
            "{name} subscriber already exists"
        )));
    }
    registries
        .event_subscribers
        .insert(name.to_string(), callback);
    Ok(())
}

/// Deregister a scope-local lifecycle event subscriber.
///
/// This removes the named subscriber from the registry attached to a specific
/// active scope.
///
/// # Parameters
/// - `scope_uuid`: UUID of the owning scope.
/// - `name`: Scope-local subscriber name to remove.
///
/// # Returns
/// A [`Result`] containing `true` when a subscriber was removed and `false`
/// when the name was not registered on that scope.
///
/// # Errors
/// Returns [`FlowError::NotFound`] when the scope does not exist on the active
/// stack.
///
/// # Notes
/// Deregistration affects only future event delivery for that scope.
pub fn scope_deregister_subscriber(scope_uuid: &uuid::Uuid, name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    Ok(registries.event_subscribers.remove(name).is_some())
}
