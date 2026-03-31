// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API for the Nexus runtime.
//!
//! This module contains all top-level functions that language bindings and
//! application code call. The API is organized into several groups:
//!
//! - **Scope operations** — [`nat_nexus_get_handle`], [`nat_nexus_push_scope`],
//!   [`nat_nexus_pop_scope`], [`nat_nexus_event`]
//! - **Tool lifecycle** — [`nat_nexus_tool_call`], [`nat_nexus_tool_call_end`],
//!   [`nat_nexus_tool_call_execute`]
//! - **LLM lifecycle** — [`nat_nexus_llm_call`], [`nat_nexus_llm_call_end`],
//!   [`nat_nexus_llm_call_execute`], [`nat_nexus_llm_stream_call_execute`]
//! - **Guardrail registration** — `nat_nexus_register_*_guardrail` /
//!   `nat_nexus_deregister_*_guardrail` for tool and LLM sanitize/conditional guardrails
//! - **Intercept registration** — `nat_nexus_register_*_intercept` /
//!   `nat_nexus_deregister_*_intercept` for tool and LLM request/response/execution intercepts
//! - **Subscriber registration** — [`nat_nexus_register_subscriber`],
//!   [`nat_nexus_deregister_subscriber`]
//! - **Standalone middleware chains** — [`nat_nexus_tool_request_intercepts`],
//!   [`nat_nexus_tool_conditional_execution`], [`nat_nexus_tool_response_intercepts`],
//!   [`nat_nexus_llm_request_intercepts`], [`nat_nexus_llm_conditional_execution`]
//!
//! All functions operate on the global context singleton returned by
//! [`global_context`].

use std::pin::Pin;

use serde_json::json;
use tokio_stream::Stream;
use uuid::Uuid;

use crate::context::*;
use crate::error::{NexusError, Result};
use crate::json::Json;
use crate::stream::LlmStreamWrapper;
use crate::types::*;

/// Resolves the parent UUID: uses the explicit parent if provided, otherwise
/// falls back to the current top of the scope stack.
fn resolve_parent_uuid(parent: Option<&ScopeHandle>) -> Option<Uuid> {
    Some(
        parent
            .map(|h| h.uuid)
            .unwrap_or_else(|| task_scope_top().uuid),
    )
}

// ---------------------------------------------------------------------------
// Macros for register/deregister API generation
// ---------------------------------------------------------------------------

macro_rules! guardrail_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(name: &str, priority: i32, guardrail: $fn_type) -> Result<()> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| NexusError::Internal(e.to_string()))?;
            state
                .$field
                .register(
                    name.to_string(),
                    GuardrailEntry {
                        priority,
                        guardrail,
                    },
                )
                .map_err(|e| NexusError::AlreadyExists(e))
        }

        /// Deregister the guardrail by name. Returns `true` if it existed.
        pub fn $deregister_name(name: &str) -> Result<bool> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| NexusError::Internal(e.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

macro_rules! intercept_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(
            name: &str,
            priority: i32,
            break_chain: bool,
            callable: $fn_type,
        ) -> Result<()> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| NexusError::Internal(e.to_string()))?;
            state
                .$field
                .register(
                    name.to_string(),
                    Intercept {
                        priority,
                        break_chain,
                        callable,
                    },
                )
                .map_err(|e| NexusError::AlreadyExists(e))
        }

        /// Deregister the intercept by name. Returns `true` if it existed.
        pub fn $deregister_name(name: &str) -> Result<bool> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| NexusError::Internal(e.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

macro_rules! execution_intercept_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(name: &str, priority: i32, callable: $fn_type) -> Result<()> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| NexusError::Internal(e.to_string()))?;
            state
                .$field
                .register(name.to_string(), ExecutionIntercept { priority, callable })
                .map_err(|e| NexusError::AlreadyExists(e))
        }

        /// Deregister the execution intercept by name. Returns `true` if it existed.
        pub fn $deregister_name(name: &str) -> Result<bool> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| NexusError::Internal(e.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

// ---------------------------------------------------------------------------
// Tool guardrail registrations
//
// Each pair generates:
//   - `nat_nexus_register_*`: registers a named guardrail with a priority.
//     Returns AlreadyExists if the name is taken.
//   - `nat_nexus_deregister_*`: removes a guardrail by name.
//     Returns Ok(true) if it existed, Ok(false) otherwise.
// ---------------------------------------------------------------------------

guardrail_registry_api!(
    /// Register a tool request sanitize guardrail that transforms tool arguments before execution.
    ///
    /// Callback signature: `(tool_name: &str, args: Json) -> Json`.
    ///
    /// # Errors
    ///
    /// Returns [`NexusError::AlreadyExists`] if a guardrail with the given name is already registered.
    nat_nexus_register_tool_sanitize_request_guardrail,
    nat_nexus_deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);

guardrail_registry_api!(
    /// Register a tool response sanitize guardrail that transforms tool results after execution.
    ///
    /// Callback signature: `(tool_name: &str, result: Json) -> Json`.
    nat_nexus_register_tool_sanitize_response_guardrail,
    nat_nexus_deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);

guardrail_registry_api!(
    /// Register a tool conditional execution guardrail that can reject tool calls.
    ///
    /// Callback signature: `(tool_name: &str, args: &Json) -> Option<String>`.
    /// Return `None` to allow execution, `Some(reason)` to reject it.
    nat_nexus_register_tool_conditional_execution_guardrail,
    nat_nexus_deregister_tool_conditional_execution_guardrail,
    tool_conditional_execution_guardrails,
    ToolConditionalFn
);

// ---------------------------------------------------------------------------
// Tool intercept registrations
//
// Each pair generates register/deregister for intercepts that transform
// data flowing through the tool call pipeline.
// ---------------------------------------------------------------------------

intercept_registry_api!(
    /// Register a tool request intercept that transforms arguments before sanitize guardrails.
    ///
    /// Callback signature: `(tool_name: &str, args: Json) -> Json`.
    /// Set `break_chain = true` to prevent lower-priority intercepts from running.
    nat_nexus_register_tool_request_intercept,
    nat_nexus_deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);

intercept_registry_api!(
    /// Register a tool response intercept that transforms the result after execution.
    ///
    /// Callback signature: `(tool_name: &str, result: Json) -> Json`.
    /// Set `break_chain = true` to prevent lower-priority intercepts from running.
    nat_nexus_register_tool_response_intercept,
    nat_nexus_deregister_tool_response_intercept,
    tool_response_intercepts,
    ToolInterceptFn
);

execution_intercept_registry_api!(
    /// Register a tool execution intercept following the middleware chain pattern.
    ///
    /// Callback signature: `(args: Json, next: ToolExecutionNextFn) -> Future<Result<Json>>`.
    /// Call `next(args)` to continue the chain, or skip it to short-circuit.
    nat_nexus_register_tool_execution_intercept,
    nat_nexus_deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionFn
);

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

guardrail_registry_api!(
    /// Register an LLM request sanitize guardrail that transforms the request before execution.
    ///
    /// Callback signature: `(request: LLMRequest) -> LLMRequest`.
    nat_nexus_register_llm_sanitize_request_guardrail,
    nat_nexus_deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);

guardrail_registry_api!(
    /// Register an LLM response sanitize guardrail that transforms the response after execution.
    ///
    /// Callback signature: `(response: Json) -> Json`.
    nat_nexus_register_llm_sanitize_response_guardrail,
    nat_nexus_deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);

guardrail_registry_api!(
    /// Register an LLM conditional execution guardrail that can reject LLM calls.
    ///
    /// Callback signature: `(request: &LLMRequest) -> Option<String>`.
    /// Return `None` to allow execution, `Some(reason)` to reject it.
    nat_nexus_register_llm_conditional_execution_guardrail,
    nat_nexus_deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

intercept_registry_api!(
    /// Register an LLM request intercept that transforms the request before sanitize guardrails.
    ///
    /// Callback signature: `(request: LLMRequest) -> LLMRequest`.
    /// Set `break_chain = true` to prevent lower-priority intercepts from running.
    nat_nexus_register_llm_request_intercept,
    nat_nexus_deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);

execution_intercept_registry_api!(
    /// Register an LLM execution intercept following the middleware chain pattern.
    ///
    /// Callback signature: `(request: LLMRequest, next: LlmExecutionNextFn) -> Future<Result<Json>>`.
    /// Call `next(request)` to continue the chain, or skip it to short-circuit.
    nat_nexus_register_llm_execution_intercept,
    nat_nexus_deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionFn
);

execution_intercept_registry_api!(
    /// Register an LLM streaming execution intercept following the middleware chain pattern.
    ///
    /// Callback signature: `(request: LLMRequest, next: LlmStreamExecutionNextFn) -> Future<Result<Stream>>`.
    /// Call `next(request)` to continue the chain, or skip it to short-circuit.
    nat_nexus_register_llm_stream_execution_intercept,
    nat_nexus_deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionFn
);

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Registers a named event subscriber that will be called for every lifecycle event.
///
/// Returns [`NexusError::AlreadyExists`] if a subscriber with the given name
/// is already registered.
pub fn nat_nexus_register_subscriber(name: &str, callback: EventSubscriberFn) -> Result<()> {
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| NexusError::Internal(e.to_string()))?;
    if state.event_subscribers.contains_key(name) {
        return Err(NexusError::AlreadyExists(format!(
            "{name} subscriber already exists"
        )));
    }
    state.event_subscribers.insert(name.to_string(), callback);
    Ok(())
}

/// Deregisters an event subscriber by name. Returns `true` if it existed, `false` otherwise.
pub fn nat_nexus_deregister_subscriber(name: &str) -> Result<bool> {
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| NexusError::Internal(e.to_string()))?;
    Ok(state.event_subscribers.remove(name).is_some())
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations
// ---------------------------------------------------------------------------

macro_rules! scope_local_guardrail_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(scope_uuid: &Uuid, name: &str, priority: i32, guardrail: $fn_type) -> Result<()> {
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| NexusError::NotFound(format!("scope {scope_uuid} not found")))?;
            regs.$field
                .register(
                    name.to_string(),
                    GuardrailEntry { priority, guardrail },
                )
                .map_err(|e| NexusError::AlreadyExists(e))
        }

        /// Deregister a scope-local guardrail by name. Returns `true` if it existed.
        pub fn $deregister_name(scope_uuid: &Uuid, name: &str) -> Result<bool> {
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| NexusError::NotFound(format!("scope {scope_uuid} not found")))?;
            Ok(regs.$field.deregister(name))
        }
    };
}

macro_rules! scope_local_intercept_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(
            scope_uuid: &Uuid,
            name: &str,
            priority: i32,
            break_chain: bool,
            callable: $fn_type,
        ) -> Result<()> {
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| NexusError::NotFound(format!("scope {scope_uuid} not found")))?;
            regs.$field
                .register(
                    name.to_string(),
                    Intercept { priority, break_chain, callable },
                )
                .map_err(|e| NexusError::AlreadyExists(e))
        }

        /// Deregister a scope-local intercept by name. Returns `true` if it existed.
        pub fn $deregister_name(scope_uuid: &Uuid, name: &str) -> Result<bool> {
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| NexusError::NotFound(format!("scope {scope_uuid} not found")))?;
            Ok(regs.$field.deregister(name))
        }
    };
}

macro_rules! scope_local_execution_intercept_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(scope_uuid: &Uuid, name: &str, priority: i32, callable: $fn_type) -> Result<()> {
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| NexusError::NotFound(format!("scope {scope_uuid} not found")))?;
            regs.$field
                .register(name.to_string(), ExecutionIntercept { priority, callable })
                .map_err(|e| NexusError::AlreadyExists(e))
        }

        /// Deregister a scope-local execution intercept by name. Returns `true` if it existed.
        pub fn $deregister_name(scope_uuid: &Uuid, name: &str) -> Result<bool> {
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| NexusError::NotFound(format!("scope {scope_uuid} not found")))?;
            Ok(regs.$field.deregister(name))
        }
    };
}

// Tool guardrails — scope-local
scope_local_guardrail_registry_api!(
    /// Register a scope-local tool request sanitize guardrail.
    nat_nexus_scope_register_tool_sanitize_request_guardrail,
    nat_nexus_scope_deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);
scope_local_guardrail_registry_api!(
    /// Register a scope-local tool response sanitize guardrail.
    nat_nexus_scope_register_tool_sanitize_response_guardrail,
    nat_nexus_scope_deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);
scope_local_guardrail_registry_api!(
    /// Register a scope-local tool conditional execution guardrail.
    nat_nexus_scope_register_tool_conditional_execution_guardrail,
    nat_nexus_scope_deregister_tool_conditional_execution_guardrail,
    tool_conditional_execution_guardrails,
    ToolConditionalFn
);

// Tool intercepts — scope-local
scope_local_intercept_registry_api!(
    /// Register a scope-local tool request intercept.
    nat_nexus_scope_register_tool_request_intercept,
    nat_nexus_scope_deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);
scope_local_intercept_registry_api!(
    /// Register a scope-local tool response intercept.
    nat_nexus_scope_register_tool_response_intercept,
    nat_nexus_scope_deregister_tool_response_intercept,
    tool_response_intercepts,
    ToolInterceptFn
);
scope_local_execution_intercept_registry_api!(
    /// Register a scope-local tool execution intercept.
    nat_nexus_scope_register_tool_execution_intercept,
    nat_nexus_scope_deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionFn
);

// LLM guardrails — scope-local
scope_local_guardrail_registry_api!(
    /// Register a scope-local LLM request sanitize guardrail.
    nat_nexus_scope_register_llm_sanitize_request_guardrail,
    nat_nexus_scope_deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);
scope_local_guardrail_registry_api!(
    /// Register a scope-local LLM response sanitize guardrail.
    nat_nexus_scope_register_llm_sanitize_response_guardrail,
    nat_nexus_scope_deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);
scope_local_guardrail_registry_api!(
    /// Register a scope-local LLM conditional execution guardrail.
    nat_nexus_scope_register_llm_conditional_execution_guardrail,
    nat_nexus_scope_deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);

// LLM intercepts — scope-local
scope_local_intercept_registry_api!(
    /// Register a scope-local LLM request intercept.
    nat_nexus_scope_register_llm_request_intercept,
    nat_nexus_scope_deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);
scope_local_execution_intercept_registry_api!(
    /// Register a scope-local LLM execution intercept.
    nat_nexus_scope_register_llm_execution_intercept,
    nat_nexus_scope_deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionFn
);
scope_local_execution_intercept_registry_api!(
    /// Register a scope-local LLM streaming execution intercept.
    nat_nexus_scope_register_llm_stream_execution_intercept,
    nat_nexus_scope_deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionFn
);

// Scope-local subscriber registration

/// Registers a scope-local event subscriber.
pub fn nat_nexus_scope_register_subscriber(
    scope_uuid: &Uuid,
    name: &str,
    callback: EventSubscriberFn,
) -> Result<()> {
    let ss = current_scope_stack();
    let mut guard = ss.write().expect("scope stack lock poisoned");
    let regs = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| NexusError::NotFound(format!("scope {scope_uuid} not found")))?;
    if regs.event_subscribers.contains_key(name) {
        return Err(NexusError::AlreadyExists(format!(
            "{name} subscriber already exists"
        )));
    }
    regs.event_subscribers.insert(name.to_string(), callback);
    Ok(())
}

/// Deregisters a scope-local event subscriber. Returns `true` if it existed.
pub fn nat_nexus_scope_deregister_subscriber(scope_uuid: &Uuid, name: &str) -> Result<bool> {
    let ss = current_scope_stack();
    let mut guard = ss.write().expect("scope stack lock poisoned");
    let regs = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| NexusError::NotFound(format!("scope {scope_uuid} not found")))?;
    Ok(regs.event_subscribers.remove(name).is_some())
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Returns a clone of the current top scope handle from the scope stack.
///
/// Always succeeds because the root scope is always present.
pub fn nat_nexus_get_handle() -> Result<ScopeHandle> {
    Ok(task_scope_top())
}

/// Creates a new scope and pushes it onto the scope stack.
///
/// Emits a `Start` event to all subscribers. If `parent` is `None`, the current
/// top of the scope stack is used as the parent. Optional `data` and `metadata`
/// payloads are attached to the new scope handle.
///
/// Returns the new [`ScopeHandle`].
pub fn nat_nexus_push_scope(
    name: &str,
    scope_type: ScopeType,
    parent: Option<&ScopeHandle>,
    attributes: ScopeAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<ScopeHandle> {
    let parent_uuid = resolve_parent_uuid(parent);
    let handle = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let root_uuid = Some(ss_guard.root_uuid());
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.create_scope_handle(
            name,
            parent_uuid,
            scope_type,
            attributes,
            root_uuid,
            data,
            metadata,
            &sl_subs,
        )
    };
    task_scope_push(handle.clone());
    Ok(handle)
}

/// Removes a scope from the scope stack by UUID and emits an `End` event.
///
/// Returns [`NexusError::NotFound`] if the UUID is not in the stack.
pub fn nat_nexus_pop_scope(handle_uuid: &Uuid) -> Result<()> {
    // Emit the End event while still holding the read lock so scope-local
    // subscribers (including those on the scope being popped) are dispatched.
    let ss = current_scope_stack();
    {
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let root_uuid = Some(ss_guard.root_uuid());
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let scope = ss_guard
            .find(handle_uuid)
            .ok_or_else(|| NexusError::NotFound("scope handle not found".into()))?
            .clone();
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.end_scope_handle(&scope, root_uuid, &sl_subs);
    }
    // Now remove the scope (takes a write lock internally).
    task_scope_remove(handle_uuid)?;
    Ok(())
}

/// Emits a standalone marker event to all subscribers.
///
/// This is a lightweight way to record application-specific events (e.g.,
/// checkpoints, metrics) without creating a scope or handle.
pub fn nat_nexus_event(
    name: &str,
    parent: Option<&ScopeHandle>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    let parent_uuid = resolve_parent_uuid(parent);
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let root_uuid = Some(ss_guard.root_uuid());
    let sl_subs = ss_guard.collect_scope_local_subscribers();
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;
    state.create_event(name, parent_uuid, data, metadata, root_uuid, &sl_subs);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begins a tool call: runs request sanitize guardrails, creates a tool handle,
/// and emits a `Start` event.
///
/// The sanitized arguments are stored in the event's `input` field.
/// Call [`nat_nexus_tool_call_end`] when the tool completes.
pub fn nat_nexus_tool_call(
    name: &str,
    args: Json,
    parent: Option<&ScopeHandle>,
    attributes: ToolAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    tool_call_id: Option<String>,
) -> Result<ToolHandle> {
    let parent_uuid = resolve_parent_uuid(parent);
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let root_uuid = Some(ss_guard.root_uuid());
    let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_sanitize_request_guardrails);
    let sl_subs = ss_guard.collect_scope_local_subscribers();
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;

    let sanitized_args = state.tool_sanitize_request_chain(name, args, &sl);

    Ok(state.create_tool_handle(
        name,
        parent_uuid,
        attributes,
        data,
        metadata,
        tool_call_id,
        Some(sanitized_args),
        root_uuid,
        &sl_subs,
    ))
}

/// Ends a tool call: runs response sanitize guardrails and emits an `End` event.
///
/// The sanitized result is stored in the event's `output` field.
pub fn nat_nexus_tool_call_end(
    handle: &ToolHandle,
    result: Json,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let root_uuid = Some(ss_guard.root_uuid());
    let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_sanitize_response_guardrails);
    let sl_subs = ss_guard.collect_scope_local_subscribers();
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;

    let sanitized_result = state.tool_sanitize_response_chain(&handle.name, result, &sl);

    state.end_tool_handle(
        handle,
        data,
        metadata,
        Some(sanitized_result),
        root_uuid,
        &sl_subs,
    );
    Ok(())
}

/// Executes a complete tool call lifecycle: conditional guardrails (on the raw
/// request), request intercepts, sanitize guardrails, execution (with middleware
/// chain of execution intercepts), response intercepts, and sanitize response
/// guardrails.
///
/// Conditional execution guardrails run **before** request intercepts so that
/// they gate on the unmodified input. On rejection, only a standalone `Mark`
/// event is emitted (no `Start`/`End` pair).
///
/// This is the high-level function that orchestrates the full middleware pipeline.
/// Returns [`NexusError::GuardrailRejected`] if a conditional guardrail rejects the call.
pub async fn nat_nexus_tool_call_execute(
    name: &str,
    args: Json,
    func: ToolExecutionNextFn,
    parent: Option<ScopeHandle>,
    attributes: ToolAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<Json> {
    // Conditional guardrails — run on the raw args before any transformation
    {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl =
            ss_guard.collect_scope_local_registries(|r| &r.tool_conditional_execution_guardrails);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        if let Some(err) = state.tool_conditional_execution_chain(name, &args, &sl) {
            drop(state);
            drop(ss_guard);
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(obj) = rejection_data.as_object_mut() {
                obj.insert("rejected".into(), json!(true));
                obj.insert("rejection_reason".into(), json!(&err));
            }
            let _ = nat_nexus_event(name, parent.as_ref(), Some(rejection_data), metadata);
            return Err(NexusError::GuardrailRejected(err));
        }
    }

    // Request intercepts
    let intercepted_args = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_request_intercepts);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.tool_request_intercepts_chain(name, args, &sl)
    };

    // Tool call start (scope-local sanitize request guardrails are picked up inside nat_nexus_tool_call)
    let handle = nat_nexus_tool_call(
        name,
        intercepted_args.clone(),
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
        None,
    )?;

    // Execution chain — build middleware chain under lock, release, then await
    let exec_future = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_execution_intercepts);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.tool_build_execution_chain(func, &sl)
    };
    let result = exec_future(intercepted_args).await?;

    // Response intercepts
    let result = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_response_intercepts);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.tool_response_intercepts_chain(name, result, &sl)
    };

    // Tool call end (scope-local sanitize response guardrails are picked up inside nat_nexus_tool_call_end)
    nat_nexus_tool_call_end(&handle, result.clone(), data, metadata)?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begins an LLM call: runs request sanitize guardrails on the [`LLMRequest`],
/// creates an LLM handle, and emits a `Start` event.
///
/// The sanitized request is stored in the event's `input` field.
/// Call [`nat_nexus_llm_call_end`] when the LLM call completes.
#[allow(clippy::too_many_arguments)]
pub fn nat_nexus_llm_call(
    name: &str,
    request: &LLMRequest,
    parent: Option<&ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
) -> Result<LLMHandle> {
    let parent_uuid = resolve_parent_uuid(parent);
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let root_uuid = Some(ss_guard.root_uuid());
    let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_sanitize_request_guardrails);
    let sl_subs = ss_guard.collect_scope_local_subscribers();
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;

    let sanitized_request = state.llm_sanitize_request_chain(request.clone(), &sl);
    let input = serde_json::to_value(&sanitized_request).unwrap_or(Json::Null);

    Ok(state.create_llm_handle(
        name,
        parent_uuid,
        attributes,
        data,
        metadata,
        model_name,
        Some(input),
        root_uuid,
        &sl_subs,
    ))
}

/// Ends an LLM call: runs response sanitize guardrails and emits an `End` event.
///
/// The sanitized response data is stored in the event's `output` field.
pub fn nat_nexus_llm_call_end(
    handle: &LLMHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let root_uuid = Some(ss_guard.root_uuid());
    let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_sanitize_response_guardrails);
    let sl_subs = ss_guard.collect_scope_local_subscribers();
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;

    let sanitized_response = state.llm_sanitize_response_chain(response, &sl);

    state.end_llm_handle(
        handle,
        data,
        metadata,
        Some(sanitized_response),
        root_uuid,
        &sl_subs,
    );
    Ok(())
}

/// Executes a complete non-streaming LLM call lifecycle: conditional guardrails,
/// request intercepts, sanitize guardrails, execution (with optional intercept
/// override), and sanitize response guardrails.
///
/// The entire pipeline operates on [`LLMRequest`]. Conditional execution
/// guardrails run **before** request intercepts so that they gate on the
/// unmodified input. On rejection, only a standalone `Mark` event is emitted
/// (no `Start`/`End` pair).
///
/// Returns [`NexusError::GuardrailRejected`] if a conditional guardrail rejects the call.
#[allow(clippy::too_many_arguments)]
pub async fn nat_nexus_llm_call_execute(
    name: &str,
    request: LLMRequest,
    func: LlmExecutionNextFn,
    parent: Option<ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
) -> Result<Json> {
    // Conditional guardrails — check on unmodified request
    {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl =
            ss_guard.collect_scope_local_registries(|r| &r.llm_conditional_execution_guardrails);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        if let Some(err) = state.llm_conditional_execution_chain(&request, &sl) {
            drop(state);
            drop(ss_guard);
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(obj) = rejection_data.as_object_mut() {
                obj.insert("rejected".into(), json!(true));
                obj.insert("rejection_reason".into(), json!(&err));
            }
            let _ = nat_nexus_event(name, parent.as_ref(), Some(rejection_data), metadata);
            return Err(NexusError::GuardrailRejected(err));
        }
    }

    // Request intercepts
    let intercepted_request = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_request_intercepts);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.llm_request_intercepts_chain(request, &sl)
    };

    // LLM call start (sanitize guardrails happen inside nat_nexus_llm_call)
    let handle = nat_nexus_llm_call(
        name,
        &intercepted_request,
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
        model_name,
    )?;

    // Execution chain — build middleware chain under lock, release, then await
    let exec_future = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_execution_intercepts);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.llm_build_execution_chain(func, &sl)
    };
    let response = exec_future(intercepted_request).await?;

    // LLM call end (sanitize response guardrails happen inside nat_nexus_llm_call_end)
    nat_nexus_llm_call_end(&handle, response.clone(), data, metadata)?;

    Ok(response)
}

/// Executes a complete streaming LLM call lifecycle.
///
/// Similar to [`nat_nexus_llm_call_execute`] but returns a
/// [`Stream`] of Json chunks. Conditional execution guardrails run
/// **before** request intercepts so that they gate on the unmodified
/// input. On rejection, only a standalone `Mark` event is emitted
/// (no `Start`/`End` pair).
///
/// The returned stream is wrapped in [`LlmStreamWrapper`] which feeds
/// each chunk to the `collector`, and on stream exhaustion calls the
/// `finalizer` to produce the aggregated response. That response then
/// flows through sanitize response guardrails before the `End` event
/// is emitted.
///
/// - `collector` — called with each chunk (Json); use this to accumulate
///   streaming tokens or forward them to another sink.
/// - `finalizer` — called once when the stream is exhausted; returns the
///   aggregated response as [`Json`].
///
/// Returns [`NexusError::GuardrailRejected`] if a conditional guardrail rejects the call.
#[allow(clippy::too_many_arguments)]
pub async fn nat_nexus_llm_stream_call_execute(
    name: &str,
    request: LLMRequest,
    func: LlmStreamExecutionNextFn,
    collector: Box<dyn FnMut(Json) + Send>,
    finalizer: Box<dyn FnOnce() -> Json + Send>,
    parent: Option<ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
) -> Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>> {
    // Conditional guardrails — check on unmodified request
    {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl =
            ss_guard.collect_scope_local_registries(|r| &r.llm_conditional_execution_guardrails);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        if let Some(err) = state.llm_conditional_execution_chain(&request, &sl) {
            drop(state);
            drop(ss_guard);
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(obj) = rejection_data.as_object_mut() {
                obj.insert("rejected".into(), json!(true));
                obj.insert("rejection_reason".into(), json!(&err));
            }
            let _ = nat_nexus_event(name, parent.as_ref(), Some(rejection_data), metadata);
            return Err(NexusError::GuardrailRejected(err));
        }
    }

    // Request intercepts
    let intercepted_request = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_request_intercepts);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.llm_request_intercepts_chain(request, &sl)
    };

    // LLM call start (sanitize guardrails happen inside nat_nexus_llm_call)
    let handle = nat_nexus_llm_call(
        name,
        &intercepted_request,
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
        model_name,
    )?;

    // Stream execution chain — build middleware chain under lock, release, then await
    let exec_future = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_stream_execution_intercepts);
        let ctx = global_context();
        let state = ctx
            .read()
            .map_err(|e| NexusError::Internal(e.to_string()))?;
        state.llm_stream_build_execution_chain(func, &sl)
    };
    let raw_stream = exec_future(intercepted_request).await?;

    // Wrap in LlmStreamWrapper which handles collector/finalizer and END event
    let wrapper = LlmStreamWrapper::new(raw_stream, handle, collector, finalizer, data, metadata);
    Ok(Box::pin(wrapper))
}

// ---------------------------------------------------------------------------
// Standalone middleware chain functions
// ---------------------------------------------------------------------------

/// Runs the registered tool request intercept chain on the given arguments.
///
/// Returns the transformed arguments after all intercepts have been applied.
/// This allows invoking request intercepts independently of the full
/// [`nat_nexus_tool_call_execute`] pipeline.
pub fn nat_nexus_tool_request_intercepts(name: &str, args: Json) -> Result<Json> {
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_request_intercepts);
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;
    Ok(state.tool_request_intercepts_chain(name, args, &sl))
}

/// Runs the registered tool conditional execution guardrail chain.
///
/// Returns `Ok(())` if all guardrails pass, or
/// [`Err(NexusError::GuardrailRejected(reason))`](NexusError::GuardrailRejected)
/// if any guardrail rejects the call.
pub fn nat_nexus_tool_conditional_execution(name: &str, args: &Json) -> Result<()> {
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_conditional_execution_guardrails);
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;
    if let Some(err) = state.tool_conditional_execution_chain(name, args, &sl) {
        return Err(NexusError::GuardrailRejected(err));
    }
    Ok(())
}

/// Runs the registered tool response intercept chain on the given result.
///
/// Returns the transformed result after all intercepts have been applied.
/// This allows invoking response intercepts independently of the full
/// [`nat_nexus_tool_call_execute`] pipeline.
pub fn nat_nexus_tool_response_intercepts(name: &str, result: Json) -> Result<Json> {
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_response_intercepts);
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;
    Ok(state.tool_response_intercepts_chain(name, result, &sl))
}

/// Runs the registered LLM request intercept chain on the given [`LLMRequest`].
///
/// Returns the transformed [`LLMRequest`] after all intercepts have been applied.
/// This allows invoking request intercepts independently of the full
/// [`nat_nexus_llm_call_execute`] pipeline.
pub fn nat_nexus_llm_request_intercepts(request: LLMRequest) -> Result<LLMRequest> {
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_request_intercepts);
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;
    Ok(state.llm_request_intercepts_chain(request, &sl))
}

/// Runs the registered LLM conditional execution guardrail chain.
///
/// Returns `Ok(())` if all guardrails pass, or
/// [`Err(NexusError::GuardrailRejected(reason))`](NexusError::GuardrailRejected)
/// if any guardrail rejects the call.
pub fn nat_nexus_llm_conditional_execution(request: &LLMRequest) -> Result<()> {
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_conditional_execution_guardrails);
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| NexusError::Internal(e.to_string()))?;
    if let Some(err) = state.llm_conditional_execution_chain(request, &sl) {
        return Err(NexusError::GuardrailRejected(err));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::await_holding_lock, clippy::type_complexity)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    // Serialize all API tests since they share global state.
    // Using std::sync::Mutex (not tokio) is intentional — these are single-threaded
    // tokio tests and the lock serializes access to the process-wide global context.
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    fn reset_global() {
        let ctx = global_context();
        let mut state = ctx.write().unwrap();
        *state = NatNexusContextState::new();
    }

    #[test]
    fn test_push_pop_scope() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        // Root scope is always present
        let root = nat_nexus_get_handle().unwrap();
        assert_eq!(root.name, "root");

        let handle = nat_nexus_push_scope(
            "test_scope",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(nat_nexus_get_handle().unwrap().name, "test_scope");
        nat_nexus_pop_scope(&handle.uuid).unwrap();

        // After pop, root scope is on top again
        assert_eq!(nat_nexus_get_handle().unwrap().name, "root");
    }

    #[test]
    fn test_subscriber_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        let count = Arc::new(AtomicU32::new(0));
        let c = count.clone();
        nat_nexus_register_subscriber(
            "test_sub",
            Box::new(move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            }),
        )
        .unwrap();

        // Duplicate should fail
        assert!(nat_nexus_register_subscriber("test_sub", Box::new(|_| {}),).is_err());

        // Push scope emits event
        let handle = nat_nexus_push_scope(
            "s",
            ScopeType::Function,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);

        nat_nexus_pop_scope(&handle.uuid).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 2);

        // Deregister
        assert!(nat_nexus_deregister_subscriber("test_sub").unwrap());
        assert!(!nat_nexus_deregister_subscriber("test_sub").unwrap());
    }

    #[test]
    fn test_tool_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_tool_sanitize_request_guardrail("g1", 10, Box::new(|_name, args| args))
            .unwrap();

        // Duplicate fails
        assert!(nat_nexus_register_tool_sanitize_request_guardrail(
            "g1",
            10,
            Box::new(|_name, args| args),
        )
        .is_err());

        assert!(nat_nexus_deregister_tool_sanitize_request_guardrail("g1").unwrap());
    }

    // -- Scope hierarchy --

    #[test]
    fn test_nested_scopes() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let s1 = nat_nexus_push_scope(
            "level1",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(nat_nexus_get_handle().unwrap().name, "level1");

        let s2 = nat_nexus_push_scope(
            "level2",
            ScopeType::Function,
            Some(&s1),
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(nat_nexus_get_handle().unwrap().name, "level2");
        assert_eq!(s2.parent_uuid, Some(s1.uuid));

        nat_nexus_pop_scope(&s2.uuid).unwrap();
        assert_eq!(nat_nexus_get_handle().unwrap().name, "level1");

        nat_nexus_pop_scope(&s1.uuid).unwrap();
        assert_eq!(nat_nexus_get_handle().unwrap().name, "root");
    }

    #[test]
    fn test_pop_nonexistent_scope() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        let result = nat_nexus_pop_scope(&Uuid::new_v4());
        assert!(result.is_err());
    }

    #[test]
    fn test_scope_attributes_propagated() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        let handle = nat_nexus_push_scope(
            "parallel_scope",
            ScopeType::Agent,
            None,
            ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE,
            None,
            None,
        )
        .unwrap();
        assert!(handle.attributes.contains(ScopeAttributes::PARALLEL));
        assert!(handle.attributes.contains(ScopeAttributes::RELOCATABLE));
        nat_nexus_pop_scope(&handle.uuid).unwrap();
    }

    // -- Event emission --

    #[test]
    fn test_event_emission() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "evt_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push((e.name.clone(), e.event_type));
            }),
        )
        .unwrap();

        nat_nexus_event("my_mark", None, Some(json!({"x": 1})), None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].0, Some("my_mark".into()));
        assert_eq!(captured[0].1, crate::types::EventType::Mark);

        drop(captured);
        nat_nexus_deregister_subscriber("evt_test").unwrap();
    }

    // -- Tool lifecycle --

    #[test]
    fn test_tool_call_and_end() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "tool_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.event_type);
            }),
        )
        .unwrap();

        let handle = nat_nexus_tool_call(
            "my_tool",
            json!({"input": "data"}),
            None,
            ToolAttributes::empty(),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(handle.name, "my_tool");

        nat_nexus_tool_call_end(&handle, json!({"output": "result"}), None, None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], crate::types::EventType::Start);
        assert_eq!(captured[1], crate::types::EventType::End);

        drop(captured);
        nat_nexus_deregister_subscriber("tool_test").unwrap();
    }

    #[test]
    fn test_tool_call_with_sanitize_guardrail() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        // Register a sanitizer that adds a field
        nat_nexus_register_tool_sanitize_request_guardrail(
            "sanitizer",
            1,
            Box::new(|_name, mut args| {
                args.as_object_mut()
                    .unwrap()
                    .insert("sanitized".into(), json!(true));
                args
            }),
        )
        .unwrap();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "tool_san_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let handle = nat_nexus_tool_call(
            "my_tool",
            json!({"input": "data"}),
            None,
            ToolAttributes::empty(),
            None,
            None,
            None,
        )
        .unwrap();

        // The start event input should contain sanitized args
        let captured = events.lock().unwrap();
        let start_event = &captured[0];
        let input = start_event.input.as_ref().unwrap();
        assert_eq!(input["sanitized"], true);
        assert_eq!(input["input"], "data");

        drop(captured);
        nat_nexus_tool_call_end(&handle, json!("ok"), None, None).unwrap();
        nat_nexus_deregister_subscriber("tool_san_test").unwrap();
        nat_nexus_deregister_tool_sanitize_request_guardrail("sanitizer").unwrap();
    }

    #[test]
    fn test_tool_call_end_with_sanitize_response_guardrail() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_tool_sanitize_response_guardrail(
            "resp_sanitizer",
            1,
            Box::new(|_name, mut result| {
                result
                    .as_object_mut()
                    .unwrap()
                    .insert("cleaned".into(), json!(true));
                result
            }),
        )
        .unwrap();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "tool_resp_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let handle = nat_nexus_tool_call(
            "tool",
            json!({}),
            None,
            ToolAttributes::empty(),
            None,
            None,
            None,
        )
        .unwrap();
        nat_nexus_tool_call_end(&handle, json!({"output": "raw"}), None, None).unwrap();

        let captured = events.lock().unwrap();
        let end_event = &captured[1];
        let output = end_event.output.as_ref().unwrap();
        assert_eq!(output["cleaned"], true);
        assert_eq!(output["output"], "raw");

        drop(captured);
        nat_nexus_deregister_subscriber("tool_resp_test").unwrap();
        nat_nexus_deregister_tool_sanitize_response_guardrail("resp_sanitizer").unwrap();
    }

    // -- Tool call execute (async) --

    #[tokio::test]
    async fn test_tool_call_execute_basic() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let func: ToolExecutionNextFn =
            Box::new(|args| Box::pin(async move { Ok(json!({"result": args["input"]})) }));

        let result = nat_nexus_tool_call_execute(
            "exec_tool",
            json!({"input": "hello"}),
            func,
            None,
            ToolAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["result"], "hello");
    }

    #[tokio::test]
    async fn test_tool_call_execute_with_request_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_tool_request_intercept(
            "req_intercept",
            1,
            false,
            Box::new(|_name, mut args| {
                args.as_object_mut()
                    .unwrap()
                    .insert("added_by_intercept".into(), json!(true));
                args
            }),
        )
        .unwrap();

        let func: ToolExecutionNextFn = Box::new(|args| Box::pin(async move { Ok(args) }));

        let result = nat_nexus_tool_call_execute(
            "tool",
            json!({"original": true}),
            func,
            None,
            ToolAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["original"], true);
        assert_eq!(result["added_by_intercept"], true);

        nat_nexus_deregister_tool_request_intercept("req_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_with_response_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_tool_response_intercept(
            "resp_intercept",
            1,
            false,
            Box::new(|_name, mut result| {
                result
                    .as_object_mut()
                    .unwrap()
                    .insert("response_intercepted".into(), json!(true));
                result
            }),
        )
        .unwrap();

        let func: ToolExecutionNextFn =
            Box::new(|_args| Box::pin(async move { Ok(json!({"output": "raw"})) }));

        let result = nat_nexus_tool_call_execute(
            "tool",
            json!({}),
            func,
            None,
            ToolAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["output"], "raw");
        assert_eq!(result["response_intercepted"], true);

        nat_nexus_deregister_tool_response_intercept("resp_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "tool_reject_sub",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        nat_nexus_register_tool_conditional_execution_guardrail(
            "blocker",
            1,
            Box::new(|_name, _args| Some("forbidden tool".into())),
        )
        .unwrap();

        let func: ToolExecutionNextFn =
            Box::new(|_args| Box::pin(async move { Ok(json!({"should_not_reach": true})) }));

        let result = nat_nexus_tool_call_execute(
            "tool",
            json!({}),
            func,
            None,
            ToolAttributes::empty(),
            None,
            None,
        )
        .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            NexusError::GuardrailRejected(msg) => assert_eq!(msg, "forbidden tool"),
            e => panic!("expected GuardrailRejected, got {e:?}"),
        }

        // Verify standalone Mark event with rejection data (no Start/End pair)
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].event_type, crate::types::EventType::Mark);
        let mark_data = captured[0].data.as_ref().unwrap();
        assert_eq!(mark_data["rejected"], true);
        assert_eq!(mark_data["rejection_reason"], "forbidden tool");

        drop(captured);
        nat_nexus_deregister_subscriber("tool_reject_sub").unwrap();
        nat_nexus_deregister_tool_conditional_execution_guardrail("blocker").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_with_execution_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_tool_execution_intercept(
            "exec_intercept",
            1,
            Arc::new(|_args: Json, _next: ToolExecutionNextFn| {
                Box::pin(async move { Ok(json!({"from_intercept": true})) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();

        let func: ToolExecutionNextFn =
            Box::new(|_args| Box::pin(async move { Ok(json!({"from_original": true})) }));

        let result = nat_nexus_tool_call_execute(
            "tool",
            json!({}),
            func,
            None,
            ToolAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap();

        // Execution intercept should have replaced the original function
        assert_eq!(result["from_intercept"], true);
        assert!(result.get("from_original").is_none());

        nat_nexus_deregister_tool_execution_intercept("exec_intercept").unwrap();
    }

    // -- LLM lifecycle --

    #[test]
    fn test_llm_call_and_end() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "llm_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.event_type);
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let handle = nat_nexus_llm_call(
            "my_llm",
            &request,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(handle.name, "my_llm");

        nat_nexus_llm_call_end(&handle, json!({"response": "ok"}), None, None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], crate::types::EventType::Start);
        assert_eq!(captured[1], crate::types::EventType::End);

        drop(captured);
        nat_nexus_deregister_subscriber("llm_test").unwrap();
    }

    #[test]
    fn test_llm_call_with_sanitize_guardrail() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_llm_sanitize_request_guardrail(
            "llm_sanitizer",
            1,
            Box::new(|mut req: LLMRequest| {
                req.headers.insert("X-Sanitized".into(), json!("true"));
                req
            }),
        )
        .unwrap();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "llm_san_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let handle = nat_nexus_llm_call(
            "llm",
            &request,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
        )
        .unwrap();

        let captured = events.lock().unwrap();
        let start_event = &captured[0];
        // Sanitized request should be in input
        let input = start_event.input.as_ref().unwrap();
        assert_eq!(input["headers"]["X-Sanitized"], "true");

        drop(captured);
        nat_nexus_llm_call_end(&handle, json!("ok"), None, None).unwrap();
        nat_nexus_deregister_subscriber("llm_san_test").unwrap();
        nat_nexus_deregister_llm_sanitize_request_guardrail("llm_sanitizer").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_basic() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let func: LlmExecutionNextFn =
            Box::new(|req: LLMRequest| Box::pin(async move { Ok(json!({"echo": req.content})) }));

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": [{"role": "user", "content": "hi"}]}),
        };
        let content = request.content.clone();

        let result = nat_nexus_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["echo"], content);
    }

    #[tokio::test]
    async fn test_llm_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "llm_reject_sub",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        nat_nexus_register_llm_conditional_execution_guardrail(
            "llm_blocker",
            1,
            Box::new(|_req: &LLMRequest| Some("blocked by policy".into())),
        )
        .unwrap();

        let func: LlmExecutionNextFn = Box::new(|_req| Box::pin(async move { Ok(json!({})) }));

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        let result = nat_nexus_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
        )
        .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            NexusError::GuardrailRejected(msg) => assert_eq!(msg, "blocked by policy"),
            e => panic!("expected GuardrailRejected, got {e:?}"),
        }

        // Verify standalone Mark event with rejection data (no Start/End pair)
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].event_type, crate::types::EventType::Mark);
        let mark_data = captured[0].data.as_ref().unwrap();
        assert_eq!(mark_data["rejected"], true);
        assert_eq!(mark_data["rejection_reason"], "blocked by policy");

        drop(captured);
        nat_nexus_deregister_subscriber("llm_reject_sub").unwrap();
        nat_nexus_deregister_llm_conditional_execution_guardrail("llm_blocker").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_with_request_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_llm_request_intercept(
            "llm_req_intercept",
            1,
            false,
            Box::new(|mut req: LLMRequest| {
                req.headers.insert("intercepted".into(), json!(true));
                req
            }),
        )
        .unwrap();

        let func: LlmExecutionNextFn = Box::new(|req: LLMRequest| {
            let saw = req
                .headers
                .get("intercepted")
                .cloned()
                .unwrap_or(Json::Null);
            Box::pin(async move { Ok(json!({"saw_intercepted": saw})) })
        });

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        let result = nat_nexus_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["saw_intercepted"], true);

        nat_nexus_deregister_llm_request_intercept("llm_req_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_with_execution_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_llm_execution_intercept(
            "llm_exec_intercept",
            1,
            Arc::new(|_req: LLMRequest, _next: LlmExecutionNextFn| {
                Box::pin(async move { Ok(json!({"from_intercept": true})) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();

        let func: LlmExecutionNextFn =
            Box::new(|_req| Box::pin(async move { Ok(json!({"from_original": true})) }));

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        let result = nat_nexus_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["from_intercept"], true);
        assert!(result.get("from_original").is_none());

        nat_nexus_deregister_llm_execution_intercept("llm_exec_intercept").unwrap();
    }

    // -- All guardrail/intercept registration pairs --

    #[test]
    fn test_tool_sanitize_response_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_tool_sanitize_response_guardrail("g1", 1, Box::new(|_n, r| r)).unwrap();
        assert!(
            nat_nexus_register_tool_sanitize_response_guardrail("g1", 1, Box::new(|_n, r| r))
                .is_err()
        );
        assert!(nat_nexus_deregister_tool_sanitize_response_guardrail("g1").unwrap());
        assert!(!nat_nexus_deregister_tool_sanitize_response_guardrail("g1").unwrap());
    }

    #[test]
    fn test_tool_conditional_execution_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_tool_conditional_execution_guardrail("g1", 1, Box::new(|_n, _a| None))
            .unwrap();
        assert!(nat_nexus_register_tool_conditional_execution_guardrail(
            "g1",
            1,
            Box::new(|_n, _a| None)
        )
        .is_err());
        assert!(nat_nexus_deregister_tool_conditional_execution_guardrail("g1").unwrap());
    }

    #[test]
    fn test_tool_request_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_tool_request_intercept("i1", 1, false, Box::new(|_n, a| a)).unwrap();
        assert!(
            nat_nexus_register_tool_request_intercept("i1", 1, false, Box::new(|_n, a| a)).is_err()
        );
        assert!(nat_nexus_deregister_tool_request_intercept("i1").unwrap());
    }

    #[test]
    fn test_tool_response_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_tool_response_intercept("i1", 1, false, Box::new(|_n, r| r)).unwrap();
        assert!(
            nat_nexus_register_tool_response_intercept("i1", 1, false, Box::new(|_n, r| r))
                .is_err()
        );
        assert!(nat_nexus_deregister_tool_response_intercept("i1").unwrap());
    }

    #[test]
    fn test_tool_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_tool_execution_intercept(
            "i1",
            1,
            Arc::new(|a: Json, _next: ToolExecutionNextFn| {
                Box::pin(async move { Ok(a) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();
        assert!(nat_nexus_register_tool_execution_intercept(
            "i1",
            1,
            Arc::new(
                |a: Json, _next: ToolExecutionNextFn| Box::pin(async move { Ok(a) })
                    as Pin<
                        Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>,
                    >
            ),
        )
        .is_err());
        assert!(nat_nexus_deregister_tool_execution_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_sanitize_request_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_llm_sanitize_request_guardrail("g1", 1, Box::new(|r| r)).unwrap();
        assert!(
            nat_nexus_register_llm_sanitize_request_guardrail("g1", 1, Box::new(|r| r)).is_err()
        );
        assert!(nat_nexus_deregister_llm_sanitize_request_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_sanitize_response_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_llm_sanitize_response_guardrail("g1", 1, Box::new(|r| r)).unwrap();
        assert!(
            nat_nexus_register_llm_sanitize_response_guardrail("g1", 1, Box::new(|r| r)).is_err()
        );
        assert!(nat_nexus_deregister_llm_sanitize_response_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_conditional_execution_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_llm_conditional_execution_guardrail("g1", 1, Box::new(|_r| None))
            .unwrap();
        assert!(nat_nexus_register_llm_conditional_execution_guardrail(
            "g1",
            1,
            Box::new(|_r| None)
        )
        .is_err());
        assert!(nat_nexus_deregister_llm_conditional_execution_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_request_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_llm_request_intercept("i1", 1, false, Box::new(|r| r)).unwrap();
        assert!(nat_nexus_register_llm_request_intercept("i1", 1, false, Box::new(|r| r)).is_err());
        assert!(nat_nexus_deregister_llm_request_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_llm_execution_intercept(
            "i1",
            1,
            Arc::new(|_request: LLMRequest, _next: LlmExecutionNextFn| {
                Box::pin(async move { Ok(json!({})) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();
        assert!(nat_nexus_deregister_llm_execution_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_stream_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nat_nexus_register_llm_stream_execution_intercept(
            "i1",
            1,
            Arc::new(|_request: LLMRequest, _next: LlmStreamExecutionNextFn| {
                Box::pin(async move {
                    let stream: Pin<Box<dyn Stream<Item = crate::error::Result<Json>> + Send>> =
                        Box::pin(tokio_stream::empty());
                    Ok(stream)
                })
                    as Pin<
                        Box<
                            dyn std::future::Future<
                                    Output = crate::error::Result<
                                        Pin<
                                            Box<
                                                dyn Stream<Item = crate::error::Result<Json>>
                                                    + Send,
                                            >,
                                        >,
                                    >,
                                > + Send,
                        >,
                    >
            }),
        )
        .unwrap();
        assert!(nat_nexus_deregister_llm_stream_execution_intercept("i1").unwrap());
    }

    // -- Deregister non-existent returns false --

    #[test]
    fn test_deregister_nonexistent_subscriber() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nat_nexus_deregister_subscriber("nonexistent").unwrap());
    }

    #[test]
    fn test_deregister_nonexistent_guardrails() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nat_nexus_deregister_tool_sanitize_request_guardrail("nope").unwrap());
        assert!(!nat_nexus_deregister_tool_sanitize_response_guardrail("nope").unwrap());
        assert!(!nat_nexus_deregister_tool_conditional_execution_guardrail("nope").unwrap());
        assert!(!nat_nexus_deregister_llm_sanitize_request_guardrail("nope").unwrap());
        assert!(!nat_nexus_deregister_llm_sanitize_response_guardrail("nope").unwrap());
        assert!(!nat_nexus_deregister_llm_conditional_execution_guardrail("nope").unwrap());
    }

    #[test]
    fn test_deregister_nonexistent_intercepts() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nat_nexus_deregister_tool_request_intercept("nope").unwrap());
        assert!(!nat_nexus_deregister_tool_response_intercept("nope").unwrap());
        assert!(!nat_nexus_deregister_tool_execution_intercept("nope").unwrap());
        assert!(!nat_nexus_deregister_llm_request_intercept("nope").unwrap());
        assert!(!nat_nexus_deregister_llm_execution_intercept("nope").unwrap());
        assert!(!nat_nexus_deregister_llm_stream_execution_intercept("nope").unwrap());
    }

    // -- LLM stream call execute --

    #[tokio::test]
    async fn test_llm_stream_call_execute_basic() {
        use tokio_stream::StreamExt;

        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let func: LlmStreamExecutionNextFn = Box::new(|_request: LLMRequest| {
            Box::pin(async move {
                let items = vec![Ok(json!({"token": "hello"})), Ok(json!({"token": "world"}))];
                let stream: Pin<Box<dyn Stream<Item = crate::error::Result<Json>> + Send>> =
                    Box::pin(tokio_stream::iter(items));
                Ok(stream)
            })
                as Pin<
                    Box<
                        dyn std::future::Future<
                                Output = crate::error::Result<
                                    Pin<Box<dyn Stream<Item = crate::error::Result<Json>> + Send>>,
                                >,
                            > + Send,
                    >,
                >
        });

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        let collected = Arc::new(Mutex::new(Vec::new()));
        let cc = collected.clone();
        let collector: Box<dyn FnMut(Json) + Send> = Box::new(move |chunk| {
            cc.lock().unwrap().push(chunk);
        });
        let fc = collected.clone();
        let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(move || {
            let chunks = fc.lock().unwrap();
            Json::Array(chunks.clone())
        });

        let mut stream = nat_nexus_llm_stream_call_execute(
            "llm",
            request,
            func,
            collector,
            finalizer,
            None,
            LLMAttributes::STREAMING,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }

        // Should have received 2 chunks
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0]["token"], "hello");
        assert_eq!(chunks[1]["token"], "world");
    }

    #[tokio::test]
    async fn test_llm_stream_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nat_nexus_register_subscriber(
            "stream_reject_sub",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        nat_nexus_register_llm_conditional_execution_guardrail(
            "stream_blocker",
            1,
            Box::new(|_req: &LLMRequest| Some("stream blocked".into())),
        )
        .unwrap();

        let func: LlmStreamExecutionNextFn = Box::new(|_request: LLMRequest| {
            Box::pin(async move {
                let stream: Pin<Box<dyn Stream<Item = crate::error::Result<Json>> + Send>> =
                    Box::pin(tokio_stream::empty());
                Ok(stream)
            })
                as Pin<
                    Box<
                        dyn std::future::Future<
                                Output = crate::error::Result<
                                    Pin<Box<dyn Stream<Item = crate::error::Result<Json>> + Send>>,
                                >,
                            > + Send,
                    >,
                >
        });

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        let collector: Box<dyn FnMut(Json) + Send> = Box::new(|_| {});
        let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| Json::Null);

        let result = nat_nexus_llm_stream_call_execute(
            "llm",
            request,
            func,
            collector,
            finalizer,
            None,
            LLMAttributes::STREAMING,
            None,
            None,
            None,
        )
        .await;

        match result {
            Err(NexusError::GuardrailRejected(msg)) => assert_eq!(msg, "stream blocked"),
            Err(e) => panic!("expected GuardrailRejected, got {e:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }

        // Verify standalone Mark event with rejection data (no Start/End pair)
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].event_type, crate::types::EventType::Mark);
        let mark_data = captured[0].data.as_ref().unwrap();
        assert_eq!(mark_data["rejected"], true);
        assert_eq!(mark_data["rejection_reason"], "stream blocked");

        drop(captured);
        nat_nexus_deregister_subscriber("stream_reject_sub").unwrap();
        nat_nexus_deregister_llm_conditional_execution_guardrail("stream_blocker").unwrap();
    }

    // -- Tool call with explicit parent --

    #[test]
    fn test_tool_call_with_parent() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let scope = nat_nexus_push_scope(
            "parent",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        let handle = nat_nexus_tool_call(
            "tool",
            json!({}),
            Some(&scope),
            ToolAttributes::empty(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(handle.parent_uuid, Some(scope.uuid));
        nat_nexus_tool_call_end(&handle, json!({}), None, None).unwrap();
        nat_nexus_pop_scope(&scope.uuid).unwrap();
    }

    // -- LLM call with attributes --

    #[test]
    fn test_llm_call_with_attributes() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let handle = nat_nexus_llm_call(
            "llm",
            &request,
            None,
            LLMAttributes::STATELESS | LLMAttributes::STREAMING,
            Some(json!({"custom": "data"})),
            Some(json!({"meta": "info"})),
            None,
        )
        .unwrap();

        assert!(handle.attributes.contains(LLMAttributes::STATELESS));
        assert!(handle.attributes.contains(LLMAttributes::STREAMING));
        nat_nexus_llm_call_end(&handle, json!({}), None, None).unwrap();
    }

    // -- Standalone middleware chain tests --

    #[test]
    fn test_tool_request_intercepts_standalone() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_tool_request_intercept(
            "add_field",
            10,
            false,
            Box::new(|_name, mut args| {
                if let Some(obj) = args.as_object_mut() {
                    obj.insert("injected".into(), json!(true));
                }
                args
            }),
        )
        .unwrap();

        let result = nat_nexus_tool_request_intercepts("tool", json!({"key": "value"})).unwrap();
        assert_eq!(result["key"], "value");
        assert_eq!(result["injected"], true);

        nat_nexus_deregister_tool_request_intercept("add_field").unwrap();
    }

    #[test]
    fn test_tool_conditional_execution_standalone_pass() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        // No guardrails registered — should pass
        assert!(nat_nexus_tool_conditional_execution("tool", &json!({})).is_ok());
    }

    #[test]
    fn test_tool_conditional_execution_standalone_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_tool_conditional_execution_guardrail(
            "blocker",
            1,
            Box::new(|_name, _args| Some("blocked".into())),
        )
        .unwrap();

        match nat_nexus_tool_conditional_execution("tool", &json!({})) {
            Err(NexusError::GuardrailRejected(msg)) => assert_eq!(msg, "blocked"),
            other => panic!("expected GuardrailRejected, got {other:?}"),
        }

        nat_nexus_deregister_tool_conditional_execution_guardrail("blocker").unwrap();
    }

    #[test]
    fn test_tool_response_intercepts_standalone() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_tool_response_intercept(
            "wrap",
            10,
            false,
            Box::new(|_name, result| json!({"wrapped": result})),
        )
        .unwrap();

        let result = nat_nexus_tool_response_intercepts("tool", json!("hello")).unwrap();
        assert_eq!(result["wrapped"], "hello");

        nat_nexus_deregister_tool_response_intercept("wrap").unwrap();
    }

    #[test]
    fn test_llm_request_intercepts_standalone() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_llm_request_intercept(
            "add_field",
            10,
            false,
            Box::new(|mut request: LLMRequest| {
                request
                    .content
                    .as_object_mut()
                    .unwrap()
                    .insert("intercepted".into(), json!(true));
                request
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let result = nat_nexus_llm_request_intercepts(request).unwrap();
        assert_eq!(result.content["intercepted"], true);
        assert_eq!(result.content["messages"], json!([]));

        nat_nexus_deregister_llm_request_intercept("add_field").unwrap();
    }

    #[test]
    fn test_llm_conditional_execution_standalone_pass() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        assert!(nat_nexus_llm_conditional_execution(&request).is_ok());
    }

    #[test]
    fn test_llm_conditional_execution_standalone_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nat_nexus_register_llm_conditional_execution_guardrail(
            "blocker",
            1,
            Box::new(|_req| Some("llm blocked".into())),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        match nat_nexus_llm_conditional_execution(&request) {
            Err(NexusError::GuardrailRejected(msg)) => assert_eq!(msg, "llm blocked"),
            other => panic!("expected GuardrailRejected, got {other:?}"),
        }

        nat_nexus_deregister_llm_conditional_execution_guardrail("blocker").unwrap();
    }
}
