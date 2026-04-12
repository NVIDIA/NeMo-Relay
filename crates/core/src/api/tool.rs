// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use crate::api::scope::event;
use crate::api::shared::{ensure_runtime_owner, resolve_parent_uuid, snapshot_event_subscribers};
use crate::context::callbacks::ToolExecutionNextFn;
use crate::context::global::global_context;
use crate::context::scope_stack::current_scope_stack;
use crate::context::state::NemoFlowContextState;
use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::types::scope::ScopeHandle;
use crate::types::tool::{ToolAttributes, ToolHandle};

pub fn tool_call(
    name: &str,
    args: Json,
    parent: Option<&ScopeHandle>,
    attributes: ToolAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    tool_call_id: Option<String>,
) -> Result<ToolHandle> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(parent);
    let (handle, event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.tool_sanitize_request_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;

        let sanitized_args = state.tool_sanitize_request_chain(name, args, &scope_locals);
        let handle =
            state.create_tool_handle(name, parent_uuid, attributes, data, metadata, tool_call_id);
        let event = state.build_tool_start_event(&handle, Some(sanitized_args));
        (handle, event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(handle)
}

pub fn tool_call_end(
    handle: &ToolHandle,
    result: Json,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let (event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.tool_sanitize_response_guardrails
        });
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;

        let sanitized_result =
            state.tool_sanitize_response_chain(&handle.name, result, &scope_locals);
        let event = state.end_tool_handle(handle, data, metadata, Some(sanitized_result));
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

fn emit_tool_end_without_output(
    handle: &ToolHandle,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let (event, subscribers) = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_subscribers = scope_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(scope_subscribers)?;
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let event = state.end_tool_handle(handle, data, metadata, None);
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

pub async fn tool_call_execute(
    name: &str,
    args: Json,
    func: ToolExecutionNextFn,
    parent: Option<ScopeHandle>,
    attributes: ToolAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<Json> {
    ensure_runtime_owner()?;
    {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
            &registries.tool_conditional_execution_guardrails
        });
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        if let Some(error) = state.tool_conditional_execution_chain(name, &args, &scope_locals)? {
            drop(state);
            drop(scope_guard);
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(object) = rejection_data.as_object_mut() {
                object.insert("rejected".into(), json!(true));
                object.insert("rejection_reason".into(), json!(&error));
            }
            let _ = event(
                name,
                parent.as_ref(),
                Some(rejection_data),
                metadata.clone(),
            );
            return Err(FlowError::GuardrailRejected(error));
        }
    }

    let intercepted_args = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registries| &registries.tool_request_intercepts);
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.tool_request_intercepts_chain(name, args, &scope_locals)?
    };

    let handle = tool_call(
        name,
        intercepted_args.clone(),
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
        None,
    )?;

    let execution = {
        let scope_stack = current_scope_stack();
        let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registries| &registries.tool_execution_intercepts);
        let context = global_context();
        let state = context
            .read()
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        state.tool_build_execution_chain(name, func, &scope_locals)
    };

    match execution(intercepted_args).await {
        Ok(result) => {
            tool_call_end(&handle, result.clone(), data, metadata)?;
            Ok(result)
        }
        Err(error) => {
            let _ = emit_tool_end_without_output(&handle, data, metadata);
            Err(error)
        }
    }
}

pub fn tool_request_intercepts(name: &str, args: Json) -> Result<Json> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
    let scope_locals = scope_guard
        .collect_scope_local_registries(|registries| &registries.tool_request_intercepts);
    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    state.tool_request_intercepts_chain(name, args, &scope_locals)
}

pub fn tool_conditional_execution(name: &str, args: &Json) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
    let scope_locals = scope_guard.collect_scope_local_registries(|registries| {
        &registries.tool_conditional_execution_guardrails
    });
    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    if let Some(error) = state.tool_conditional_execution_chain(name, args, &scope_locals)? {
        return Err(FlowError::GuardrailRejected(error));
    }
    Ok(())
}
