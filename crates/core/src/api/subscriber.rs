// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::api::shared::ensure_runtime_owner;
use crate::context::callbacks::EventSubscriberFn;
use crate::context::global::global_context;
use crate::context::scope_stack::current_scope_stack;
use crate::error::{FlowError, Result};

pub fn register_subscriber(name: &str, callback: EventSubscriberFn) -> Result<()> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    if state.event_subscribers.contains_key(name) {
        return Err(FlowError::AlreadyExists(format!(
            "{name} subscriber already exists"
        )));
    }
    state.event_subscribers.insert(name.to_string(), callback);
    Ok(())
}

pub fn deregister_subscriber(name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(state.event_subscribers.remove(name).is_some())
}

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

pub fn scope_deregister_subscriber(scope_uuid: &uuid::Uuid, name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    Ok(registries.event_subscribers.remove(name).is_some())
}
