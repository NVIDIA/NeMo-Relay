// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::api::shared::{ensure_runtime_owner, resolve_parent_uuid, snapshot_event_subscribers};
use crate::context::global::global_context;
use crate::context::scope_stack::{
    current_scope_stack, task_scope_push, task_scope_remove, task_scope_top,
};
use crate::context::state::NemoFlowContextState;
use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::types::scope::{ScopeAttributes, ScopeHandle, ScopeType};

pub fn get_handle() -> Result<ScopeHandle> {
    ensure_runtime_owner()?;
    Ok(task_scope_top())
}

pub fn push_scope(
    name: &str,
    scope_type: ScopeType,
    parent: Option<&ScopeHandle>,
    attributes: ScopeAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<ScopeHandle> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(parent);
    let (handle, event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let handle =
            state.create_scope_handle(name, parent_uuid, scope_type, attributes, data, metadata);
        let event = state.build_scope_start_event(&handle);
        (handle, event, subscribers)
    };
    task_scope_push(handle.clone());
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(handle)
}

pub fn pop_scope(handle_uuid: &uuid::Uuid) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let (scope, event, subscribers) = {
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let top = scope_guard.top();
        if top.uuid != *handle_uuid {
            if scope_guard.find(handle_uuid).is_some() {
                return Err(FlowError::InvalidArgument(
                    "scope handle is not at the top of the stack".into(),
                ));
            }
            return Err(FlowError::NotFound("scope handle not found".into()));
        }
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let scope = top.clone();
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let event = state.end_scope_handle(&scope);
        (scope, event, subscribers)
    };
    let removed = task_scope_remove(handle_uuid)?;
    debug_assert_eq!(removed.uuid, scope.uuid);
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

pub fn event(
    name: &str,
    parent: Option<&ScopeHandle>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(parent);
    let (event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let event = state.create_event(name, parent_uuid, data, metadata);
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}
