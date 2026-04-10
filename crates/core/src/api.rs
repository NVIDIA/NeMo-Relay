// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API for the NeMo Flow runtime.
//!
//! This module contains all top-level functions that language bindings and
//! application code call. The API is organized into several groups:
//!
//! - **Scope operations** — [`nemo_flow_get_handle`], [`nemo_flow_push_scope`],
//!   [`nemo_flow_pop_scope`], [`nemo_flow_event`]
//! - **Tool lifecycle** — [`nemo_flow_tool_call`], [`nemo_flow_tool_call_end`],
//!   [`nemo_flow_tool_call_execute`]
//! - **LLM lifecycle** — [`nemo_flow_llm_call`], [`nemo_flow_llm_call_end`],
//!   [`nemo_flow_llm_call_execute`], [`nemo_flow_llm_stream_call_execute`]
//! - **Guardrail registration** — `nemo_flow_register_*_guardrail` /
//!   `nemo_flow_deregister_*_guardrail` for tool and LLM sanitize/conditional guardrails
//! - **Intercept registration** — `nemo_flow_register_*_intercept` /
//!   `nemo_flow_deregister_*_intercept` for tool and LLM request/execution intercepts
//! - **Subscriber registration** — [`nemo_flow_register_subscriber`],
//!   [`nemo_flow_deregister_subscriber`]
//! - **Standalone middleware chains** — [`nemo_flow_tool_request_intercepts`],
//!   [`nemo_flow_tool_conditional_execution`],
//!   [`nemo_flow_llm_request_intercepts`], [`nemo_flow_llm_conditional_execution`]
//!
//! All functions operate on the global context singleton returned by
//! [`global_context`].

use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;
use tokio_stream::Stream;
use uuid::Uuid;

use crate::codec::{AnnotatedLLMRequest, AnnotatedLLMResponse, LlmCodec, LlmResponseCodec};
use crate::context::*;
use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::shared_runtime::ensure_process_runtime_owner;
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

fn snapshot_event_subscribers(
    scope_local_subscribers: Vec<EventSubscriberFn>,
) -> Result<Vec<EventSubscriberFn>> {
    let ctx = global_context();
    let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
    Ok(state.collect_event_subscribers(&scope_local_subscribers))
}

fn ensure_runtime_owner() -> Result<()> {
    ensure_process_runtime_owner()
}

// ---------------------------------------------------------------------------
// Macros for register/deregister API generation
// ---------------------------------------------------------------------------

macro_rules! guardrail_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(name: &str, priority: i32, guardrail: $fn_type) -> Result<()> {
            ensure_runtime_owner()?;
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| FlowError::Internal(e.to_string()))?;
            state
                .$field
                .register(
                    name.to_string(),
                    GuardrailEntry {
                        priority,
                        guardrail,
                    },
                )
                .map_err(|e| FlowError::AlreadyExists(e))
        }

        /// Deregister the guardrail by name. Returns `true` if it existed.
        pub fn $deregister_name(name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| FlowError::Internal(e.to_string()))?;
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
            ensure_runtime_owner()?;
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| FlowError::Internal(e.to_string()))?;
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
                .map_err(|e| FlowError::AlreadyExists(e))
        }

        /// Deregister the intercept by name. Returns `true` if it existed.
        pub fn $deregister_name(name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| FlowError::Internal(e.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

macro_rules! execution_intercept_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(name: &str, priority: i32, callable: $fn_type) -> Result<()> {
            ensure_runtime_owner()?;
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| FlowError::Internal(e.to_string()))?;
            state
                .$field
                .register(name.to_string(), ExecutionIntercept { priority, callable })
                .map_err(|e| FlowError::AlreadyExists(e))
        }

        /// Deregister the execution intercept by name. Returns `true` if it existed.
        pub fn $deregister_name(name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| FlowError::Internal(e.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

// ---------------------------------------------------------------------------
// Tool guardrail registrations
//
// Each pair generates:
//   - `nemo_flow_register_*`: registers a named guardrail with a priority.
//     Returns AlreadyExists if the name is taken.
//   - `nemo_flow_deregister_*`: removes a guardrail by name.
//     Returns Ok(true) if it existed, Ok(false) otherwise.
// ---------------------------------------------------------------------------

guardrail_registry_api!(
    /// Register a tool request sanitize guardrail that rewrites the payload recorded on tool
    /// lifecycle events.
    ///
    /// Callback signature: `(tool_name: &str, args: Json) -> Json`.
    /// In the managed `*_execute` APIs, sanitize guardrails are observability-oriented:
    /// they affect the `Start` event payload, but request intercepts still control the
    /// actual execution input.
    ///
    /// This callback is infallible. Handle failures inside the callback itself.
    ///
    /// # Errors
    ///
    /// Returns [`FlowError::AlreadyExists`] if a guardrail with the given name is already registered.
    nemo_flow_register_tool_sanitize_request_guardrail,
    nemo_flow_deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);

guardrail_registry_api!(
    /// Register a tool response sanitize guardrail that rewrites the payload recorded on tool
    /// lifecycle events.
    ///
    /// Callback signature: `(tool_name: &str, result: Json) -> Json`.
    /// In the managed `*_execute` APIs, sanitize guardrails affect the `End` event payload,
    /// not the value returned to the caller.
    ///
    /// This callback is infallible. Handle failures inside the callback itself.
    nemo_flow_register_tool_sanitize_response_guardrail,
    nemo_flow_deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);

guardrail_registry_api!(
    /// Register a tool conditional execution guardrail that can reject tool calls.
    ///
    /// Callback signature: `(tool_name: &str, args: &Json) -> Result<Option<String>>`.
    /// Return `Ok(None)` to allow execution, `Ok(Some(reason))` to reject it.
    /// Returning `Err(...)` aborts the originating NeMo Flow call.
    nemo_flow_register_tool_conditional_execution_guardrail,
    nemo_flow_deregister_tool_conditional_execution_guardrail,
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
    /// Callback signature: `(tool_name: &str, args: Json) -> Result<Json>`.
    /// Set `break_chain = true` to prevent lower-priority intercepts from running.
    /// Returning `Err(...)` aborts the originating NeMo Flow call.
    nemo_flow_register_tool_request_intercept,
    nemo_flow_deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);

execution_intercept_registry_api!(
    /// Register a tool execution intercept following the middleware chain pattern.
    ///
    /// Callback signature: `(args: Json, next: ToolExecutionNextFn) -> Future<Result<Json>>`.
    /// Call `next(args)` to continue the chain, or skip it to short-circuit.
    nemo_flow_register_tool_execution_intercept,
    nemo_flow_deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionFn
);

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

guardrail_registry_api!(
    /// Register an LLM request sanitize guardrail that rewrites the payload recorded on LLM
    /// lifecycle events.
    ///
    /// Callback signature: `(request: LLMRequest) -> LLMRequest`.
    /// In the managed `*_execute` APIs, sanitize guardrails affect the `Start` event payload,
    /// while request intercepts still control the request passed into execution.
    ///
    /// This callback is infallible. Handle failures inside the callback itself.
    nemo_flow_register_llm_sanitize_request_guardrail,
    nemo_flow_deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);

guardrail_registry_api!(
    /// Register an LLM response sanitize guardrail that rewrites the payload recorded on LLM
    /// lifecycle events.
    ///
    /// Callback signature: `(response: Json) -> Json`.
    /// In the managed `*_execute` APIs, sanitize guardrails affect the `End` event payload,
    /// not the value returned to the caller.
    ///
    /// This callback is infallible. Handle failures inside the callback itself.
    nemo_flow_register_llm_sanitize_response_guardrail,
    nemo_flow_deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);

guardrail_registry_api!(
    /// Register an LLM conditional execution guardrail that can reject LLM calls.
    ///
    /// Callback signature: `(request: &LLMRequest) -> Result<Option<String>>`.
    /// Return `Ok(None)` to allow execution, `Ok(Some(reason))` to reject it.
    /// Returning `Err(...)` aborts the originating NeMo Flow call.
    nemo_flow_register_llm_conditional_execution_guardrail,
    nemo_flow_deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

intercept_registry_api!(
    /// Register an LLM request intercept that transforms the request before sanitize guardrails.
    ///
    /// Callback signature: `(request: LLMRequest) -> Result<LLMRequest>`.
    /// Set `break_chain = true` to prevent lower-priority intercepts from running.
    /// Returning `Err(...)` aborts the originating NeMo Flow call.
    nemo_flow_register_llm_request_intercept,
    nemo_flow_deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);

execution_intercept_registry_api!(
    /// Register an LLM execution intercept following the middleware chain pattern.
    ///
    /// Callback signature: `(request: LLMRequest, next: LlmExecutionNextFn) -> Future<Result<Json>>`.
    /// Call `next(request)` to continue the chain, or skip it to short-circuit.
    nemo_flow_register_llm_execution_intercept,
    nemo_flow_deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionFn
);

execution_intercept_registry_api!(
    /// Register an LLM streaming execution intercept following the middleware chain pattern.
    ///
    /// Callback signature: `(request: LLMRequest, next: LlmStreamExecutionNextFn) -> Future<Result<Stream>>`.
    /// Call `next(request)` to continue the chain, or skip it to short-circuit.
    nemo_flow_register_llm_stream_execution_intercept,
    nemo_flow_deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionFn
);

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Registers a named event subscriber that will be called for every lifecycle event.
///
/// Subscriber callbacks run synchronously on the calling thread, but the
/// runtime snapshots the subscriber list and releases its locks before
/// invoking them.
///
/// Returns [`FlowError::AlreadyExists`] if a subscriber with the given name
/// is already registered.
pub fn nemo_flow_register_subscriber(name: &str, callback: EventSubscriberFn) -> Result<()> {
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| FlowError::Internal(e.to_string()))?;
    if state.event_subscribers.contains_key(name) {
        return Err(FlowError::AlreadyExists(format!(
            "{name} subscriber already exists"
        )));
    }
    state.event_subscribers.insert(name.to_string(), callback);
    Ok(())
}

/// Deregisters an event subscriber by name. Returns `true` if it existed, `false` otherwise.
pub fn nemo_flow_deregister_subscriber(name: &str) -> Result<bool> {
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| FlowError::Internal(e.to_string()))?;
    let removed = state.event_subscribers.remove(name).is_some();
    Ok(removed)
}

// ---------------------------------------------------------------------------
// Request intercept pipeline with Codec decode/encode
// ---------------------------------------------------------------------------

/// Runs the full request intercept pipeline with optional Codec decode/encode.
///
/// 1. If Codec provided: decode LLMRequest -> AnnotatedLLMRequest
/// 2. Run intercept chain (single unified registry, priority-sorted)
/// 3. If Codec provided and annotated was produced: encode back to LLMRequest,
///    preserving headers from the post-intercept request
///
/// Used by both [`nemo_flow_llm_call_execute`] and [`nemo_flow_llm_stream_call_execute`].
fn run_request_intercepts_with_codec(
    name: &str,
    request: LLMRequest,
    codec: Option<Arc<dyn LlmCodec>>,
) -> Result<(LLMRequest, Option<Arc<AnnotatedLLMRequest>>)> {
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_request_intercepts);

    let ctx = global_context();
    let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;

    // Clone original for encode step (merge-not-replace needs pre-intercept content)
    let original = request.clone();

    // Decode: if Codec provided, translate opaque request to structured form
    let annotated = match &codec {
        Some(c) => Some(c.decode(&request)?),
        None => None,
    };

    // Run unified intercept chain
    let (intercepted_request, intercepted_annotated) =
        state.llm_request_intercepts_chain(name, request, annotated, &sl)?;

    // Encode: merge structured changes back into opaque request
    match (codec, intercepted_annotated) {
        (Some(c), Some(ann)) => {
            let mut encoded = c.encode(&ann, &original)?;
            // Preserve header modifications from intercepts
            encoded.headers = intercepted_request.headers;
            Ok((encoded, Some(Arc::new(ann))))
        }
        _ => Ok((intercepted_request, None)),
    }
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations
// ---------------------------------------------------------------------------

macro_rules! scope_local_guardrail_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(scope_uuid: &Uuid, name: &str, priority: i32, guardrail: $fn_type) -> Result<()> {
            ensure_runtime_owner()?;
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            regs.$field
                .register(
                    name.to_string(),
                    GuardrailEntry {
                        priority,
                        guardrail,
                    },
                )
                .map_err(|e| FlowError::AlreadyExists(e))
        }

        /// Deregister a scope-local guardrail by name. Returns `true` if it existed.
        pub fn $deregister_name(scope_uuid: &Uuid, name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
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
            ensure_runtime_owner()?;
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            regs.$field
                .register(
                    name.to_string(),
                    Intercept {
                        priority,
                        break_chain,
                        callable,
                    },
                )
                .map_err(|e| FlowError::AlreadyExists(e))
        }

        /// Deregister a scope-local intercept by name. Returns `true` if it existed.
        pub fn $deregister_name(scope_uuid: &Uuid, name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            Ok(regs.$field.deregister(name))
        }
    };
}

macro_rules! scope_local_execution_intercept_registry_api {
    ($(#[$reg_meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$reg_meta])*
        pub fn $register_name(scope_uuid: &Uuid, name: &str, priority: i32, callable: $fn_type) -> Result<()> {
            ensure_runtime_owner()?;
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            regs.$field
                .register(name.to_string(), ExecutionIntercept { priority, callable })
                .map_err(|e| FlowError::AlreadyExists(e))
        }

        /// Deregister a scope-local execution intercept by name. Returns `true` if it existed.
        pub fn $deregister_name(scope_uuid: &Uuid, name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let ss = current_scope_stack();
            let mut guard = ss.write().expect("scope stack lock poisoned");
            let regs = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            Ok(regs.$field.deregister(name))
        }
    };
}

// Tool guardrails — scope-local
scope_local_guardrail_registry_api!(
    /// Register a scope-local tool request sanitize guardrail.
    nemo_flow_scope_register_tool_sanitize_request_guardrail,
    nemo_flow_scope_deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);
scope_local_guardrail_registry_api!(
    /// Register a scope-local tool response sanitize guardrail.
    nemo_flow_scope_register_tool_sanitize_response_guardrail,
    nemo_flow_scope_deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);
scope_local_guardrail_registry_api!(
    /// Register a scope-local tool conditional execution guardrail.
    nemo_flow_scope_register_tool_conditional_execution_guardrail,
    nemo_flow_scope_deregister_tool_conditional_execution_guardrail,
    tool_conditional_execution_guardrails,
    ToolConditionalFn
);

// Tool intercepts — scope-local
scope_local_intercept_registry_api!(
    /// Register a scope-local tool request intercept.
    nemo_flow_scope_register_tool_request_intercept,
    nemo_flow_scope_deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);
scope_local_execution_intercept_registry_api!(
    /// Register a scope-local tool execution intercept.
    nemo_flow_scope_register_tool_execution_intercept,
    nemo_flow_scope_deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionFn
);

// LLM guardrails — scope-local
scope_local_guardrail_registry_api!(
    /// Register a scope-local LLM request sanitize guardrail.
    nemo_flow_scope_register_llm_sanitize_request_guardrail,
    nemo_flow_scope_deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);
scope_local_guardrail_registry_api!(
    /// Register a scope-local LLM response sanitize guardrail.
    nemo_flow_scope_register_llm_sanitize_response_guardrail,
    nemo_flow_scope_deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);
scope_local_guardrail_registry_api!(
    /// Register a scope-local LLM conditional execution guardrail.
    nemo_flow_scope_register_llm_conditional_execution_guardrail,
    nemo_flow_scope_deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);

// LLM intercepts — scope-local
scope_local_intercept_registry_api!(
    /// Register a scope-local LLM request intercept.
    nemo_flow_scope_register_llm_request_intercept,
    nemo_flow_scope_deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);
scope_local_execution_intercept_registry_api!(
    /// Register a scope-local LLM execution intercept.
    nemo_flow_scope_register_llm_execution_intercept,
    nemo_flow_scope_deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionFn
);
scope_local_execution_intercept_registry_api!(
    /// Register a scope-local LLM streaming execution intercept.
    nemo_flow_scope_register_llm_stream_execution_intercept,
    nemo_flow_scope_deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionFn
);

// Scope-local subscriber registration

/// Registers a scope-local event subscriber.
pub fn nemo_flow_scope_register_subscriber(
    scope_uuid: &Uuid,
    name: &str,
    callback: EventSubscriberFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    let ss = current_scope_stack();
    let mut guard = ss.write().expect("scope stack lock poisoned");
    let regs = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    if regs.event_subscribers.contains_key(name) {
        return Err(FlowError::AlreadyExists(format!(
            "{name} subscriber already exists"
        )));
    }
    regs.event_subscribers.insert(name.to_string(), callback);
    Ok(())
}

/// Deregisters a scope-local event subscriber. Returns `true` if it existed.
pub fn nemo_flow_scope_deregister_subscriber(scope_uuid: &Uuid, name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let ss = current_scope_stack();
    let mut guard = ss.write().expect("scope stack lock poisoned");
    let regs = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    Ok(regs.event_subscribers.remove(name).is_some())
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Returns a clone of the current top scope handle from the scope stack.
///
/// Always succeeds because the root scope is always present.
pub fn nemo_flow_get_handle() -> Result<ScopeHandle> {
    ensure_runtime_owner()?;
    Ok(task_scope_top())
}

/// Creates a new scope and pushes it onto the scope stack.
///
/// Emits a `Start` event to all subscribers after the scope has been pushed.
/// If `parent` is `None`, the current top of the scope stack is used as the
/// parent. Optional `data` and `metadata` payloads are attached to the new
/// scope handle.
///
/// Returns the new [`ScopeHandle`].
pub fn nemo_flow_push_scope(
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
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        let handle =
            state.create_scope_handle(name, parent_uuid, scope_type, attributes, data, metadata);
        let event = state.build_scope_start_event(&handle);
        (handle, event, subscribers)
    };
    task_scope_push(handle.clone());
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(handle)
}

/// Removes a scope from the scope stack by UUID and emits an `End` event after
/// the scope has been removed.
///
/// Returns [`FlowError::NotFound`] if the UUID is not in the stack, or
/// [`FlowError::InvalidArgument`] if it does not identify the current top
/// scope.
pub fn nemo_flow_pop_scope(handle_uuid: &Uuid) -> Result<()> {
    ensure_runtime_owner()?;
    let ss = current_scope_stack();
    let (scope, event, subscribers) = {
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let top = ss_guard.top();
        if top.uuid != *handle_uuid {
            if ss_guard.find(handle_uuid).is_some() {
                return Err(FlowError::InvalidArgument(
                    "scope handle is not at the top of the stack".into(),
                ));
            }
            return Err(FlowError::NotFound("scope handle not found".into()));
        }
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let scope = top.clone();
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        let event = state.end_scope_handle(&scope);
        (scope, event, subscribers)
    };
    let removed = task_scope_remove(handle_uuid)?;
    debug_assert_eq!(removed.uuid, scope.uuid);
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

/// Emits a standalone marker event to all subscribers.
///
/// This is a lightweight way to record application-specific events (e.g.,
/// checkpoints, metrics) without creating a scope or handle.
pub fn nemo_flow_event(
    name: &str,
    parent: Option<&ScopeHandle>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(parent);
    let (event, subscribers) = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        let event = state.create_event(name, parent_uuid, data, metadata);
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begins a tool call: runs request sanitize guardrails, creates a tool handle,
/// and emits a `Start` event.
///
/// The sanitized arguments are stored in the event's `input` field.
/// Call [`nemo_flow_tool_call_end`] when the tool completes.
pub fn nemo_flow_tool_call(
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
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_sanitize_request_guardrails);
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;

        let sanitized_args = state.tool_sanitize_request_chain(name, args, &sl);
        let handle =
            state.create_tool_handle(name, parent_uuid, attributes, data, metadata, tool_call_id);
        let event = state.build_tool_start_event(&handle, Some(sanitized_args));
        (handle, event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(handle)
}

/// Ends a tool call: runs response sanitize guardrails and emits an `End` event.
///
/// The sanitized result is stored in the event's `output` field.
pub fn nemo_flow_tool_call_end(
    handle: &ToolHandle,
    result: Json,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let (event, subscribers) = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_sanitize_response_guardrails);
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;

        let sanitized_result = state.tool_sanitize_response_chain(&handle.name, result, &sl);
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
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        let event = state.end_tool_handle(handle, data, metadata, None);
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

/// Executes a complete tool call lifecycle: conditional guardrails (on the raw
/// request), request intercepts, sanitize guardrails for lifecycle event
/// payloads, execution (with middleware chain of execution intercepts), and
/// sanitize response guardrails for lifecycle event payloads.
///
/// Conditional execution guardrails run **before** request intercepts so that
/// they gate on the unmodified input. On rejection, only a standalone `Mark`
/// event is emitted (no `Start`/`End` pair).
///
/// This is the high-level function that orchestrates the full middleware pipeline.
/// Returns [`FlowError::GuardrailRejected`] if a conditional guardrail rejects the call.
pub async fn nemo_flow_tool_call_execute(
    name: &str,
    args: Json,
    func: ToolExecutionNextFn,
    parent: Option<ScopeHandle>,
    attributes: ToolAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<Json> {
    ensure_runtime_owner()?;
    // Conditional guardrails — run on the raw args before any transformation
    {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl =
            ss_guard.collect_scope_local_registries(|r| &r.tool_conditional_execution_guardrails);
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        if let Some(err) = state.tool_conditional_execution_chain(name, &args, &sl)? {
            drop(state);
            drop(ss_guard);
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(obj) = rejection_data.as_object_mut() {
                obj.insert("rejected".into(), json!(true));
                obj.insert("rejection_reason".into(), json!(&err));
            }
            let _ = nemo_flow_event(name, parent.as_ref(), Some(rejection_data), metadata);
            return Err(FlowError::GuardrailRejected(err));
        }
    }

    // Request intercepts
    let intercepted_args = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_request_intercepts);
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        state.tool_request_intercepts_chain(name, args, &sl)?
    };

    // Tool call start (scope-local sanitize request guardrails are picked up inside nemo_flow_tool_call)
    let handle = nemo_flow_tool_call(
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
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        state.tool_build_execution_chain(name, func, &sl)
    };
    let exec_result = exec_future(intercepted_args).await;
    drop(exec_future);
    match exec_result {
        Ok(result) => {
            // Tool call end (scope-local sanitize response guardrails are picked up inside nemo_flow_tool_call_end)
            nemo_flow_tool_call_end(&handle, result.clone(), data, metadata)?;
            Ok(result)
        }
        Err(err) => {
            let _ = emit_tool_end_without_output(&handle, data, metadata);
            Err(err)
        }
    }
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begins an LLM call: runs request sanitize guardrails on the [`LLMRequest`],
/// creates an LLM handle, and emits a `Start` event.
///
/// The sanitized request is stored in the event's `input` field.
/// Call [`nemo_flow_llm_call_end`] when the LLM call completes.
#[allow(clippy::too_many_arguments)]
pub fn nemo_flow_llm_call(
    name: &str,
    request: &LLMRequest,
    parent: Option<&ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    annotated_request: Option<Arc<AnnotatedLLMRequest>>,
) -> Result<LLMHandle> {
    ensure_runtime_owner()?;
    let parent_uuid = resolve_parent_uuid(parent);
    let (handle, event, subscribers) = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_sanitize_request_guardrails);
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;

        let sanitized_request = state.llm_sanitize_request_chain(request.clone(), &sl);
        let input = serde_json::to_value(&sanitized_request).unwrap_or(Json::Null);
        let handle =
            state.create_llm_handle(name, parent_uuid, attributes, data, metadata, model_name);
        let event = state.build_llm_start_event(&handle, Some(input), annotated_request);
        (handle, event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(handle)
}

/// Ends an LLM call: runs response sanitize guardrails and emits an `End` event.
///
/// The sanitized response data is stored in the event's `output` field.
pub fn nemo_flow_llm_call_end(
    handle: &LLMHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
    annotated_response: Option<Arc<AnnotatedLLMResponse>>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let (event, subscribers) = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_sanitize_response_guardrails);
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;

        let sanitized_response = state.llm_sanitize_response_chain(response, &sl);
        let event = state.end_llm_handle(
            handle,
            data,
            metadata,
            Some(sanitized_response),
            annotated_response,
        );
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

fn emit_llm_end_without_output(
    handle: &LLMHandle,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    ensure_runtime_owner()?;
    let (event, subscribers) = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl_subs = ss_guard.collect_scope_local_subscribers();
        let subscribers = snapshot_event_subscribers(sl_subs)?;
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;

        let event = state.end_llm_handle(handle, data, metadata, None, None);
        (event, subscribers)
    };
    NemoFlowContextState::emit_event(&event, &subscribers);
    Ok(())
}

/// Executes a complete non-streaming LLM call lifecycle: conditional guardrails,
/// request intercepts, sanitize guardrails for lifecycle event payloads,
/// execution (with optional intercept override), and sanitize response
/// guardrails for lifecycle event payloads.
///
/// The entire pipeline operates on [`LLMRequest`]. Conditional execution
/// guardrails run **before** request intercepts so that they gate on the
/// unmodified input. On rejection, only a standalone `Mark` event is emitted
/// (no `Start`/`End` pair).
///
/// Returns [`FlowError::GuardrailRejected`] if a conditional guardrail rejects the call.
#[allow(clippy::too_many_arguments)]
pub async fn nemo_flow_llm_call_execute(
    name: &str,
    request: LLMRequest,
    func: LlmExecutionNextFn,
    parent: Option<ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    codec: Option<Arc<dyn LlmCodec>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
) -> Result<Json> {
    ensure_runtime_owner()?;
    // Conditional guardrails — check on unmodified request
    {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl =
            ss_guard.collect_scope_local_registries(|r| &r.llm_conditional_execution_guardrails);
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        if let Some(err) = state.llm_conditional_execution_chain(&request, &sl)? {
            drop(state);
            drop(ss_guard);
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(obj) = rejection_data.as_object_mut() {
                obj.insert("rejected".into(), json!(true));
                obj.insert("rejection_reason".into(), json!(&err));
            }
            let _ = nemo_flow_event(name, parent.as_ref(), Some(rejection_data), metadata);
            return Err(FlowError::GuardrailRejected(err));
        }
    }

    // Request intercepts with optional Codec decode/encode
    let (intercepted_request, annotated_request) =
        run_request_intercepts_with_codec(name, request, codec)?;

    // LLM call start (sanitize guardrails happen inside nemo_flow_llm_call)
    let handle = nemo_flow_llm_call(
        name,
        &intercepted_request,
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
        model_name,
        annotated_request,
    )?;

    // Execution chain — build middleware chain under lock, release, then await
    let exec_future = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_execution_intercepts);
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        state.llm_build_execution_chain(name, func, &sl)
    };
    let exec_result = exec_future(intercepted_request).await;
    drop(exec_future);
    match exec_result {
        Ok(response) => {
            // Decode response before sanitize (annotated reflects raw API response)
            let annotated_response = response_codec
                .as_ref()
                .and_then(|c| c.decode_response(&response).ok())
                .map(Arc::new);
            // LLM call end (sanitize response guardrails happen inside nemo_flow_llm_call_end)
            nemo_flow_llm_call_end(
                &handle,
                response.clone(),
                data,
                metadata,
                annotated_response,
            )?;
            Ok(response)
        }
        Err(err) => {
            let _ = emit_llm_end_without_output(&handle, data, metadata);
            Err(err)
        }
    }
}

/// Executes a complete streaming LLM call lifecycle.
///
/// Similar to [`nemo_flow_llm_call_execute`] but returns a
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
///   streaming tokens or forward them to another sink. This callback is
///   fallible: return `Ok(())` to continue or `Err(FlowError)` to terminate
///   the stream.
/// - `finalizer` — called once when the stream is exhausted; returns the
///   aggregated response as [`Json`]. This callback is infallible.
///
/// Returns [`FlowError::GuardrailRejected`] if a conditional guardrail rejects the call.
#[allow(clippy::too_many_arguments)]
pub async fn nemo_flow_llm_stream_call_execute(
    name: &str,
    request: LLMRequest,
    func: LlmStreamExecutionNextFn,
    collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
    finalizer: Box<dyn FnOnce() -> Json + Send>,
    parent: Option<ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    codec: Option<Arc<dyn LlmCodec>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
) -> Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>> {
    ensure_runtime_owner()?;
    // Conditional guardrails — check on unmodified request
    {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl =
            ss_guard.collect_scope_local_registries(|r| &r.llm_conditional_execution_guardrails);
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        if let Some(err) = state.llm_conditional_execution_chain(&request, &sl)? {
            drop(state);
            drop(ss_guard);
            let mut rejection_data = data.clone().unwrap_or_else(|| json!({}));
            if let Some(obj) = rejection_data.as_object_mut() {
                obj.insert("rejected".into(), json!(true));
                obj.insert("rejection_reason".into(), json!(&err));
            }
            let _ = nemo_flow_event(name, parent.as_ref(), Some(rejection_data), metadata);
            return Err(FlowError::GuardrailRejected(err));
        }
    }

    // Request intercepts with optional Codec decode/encode
    let (intercepted_request, annotated_request) =
        run_request_intercepts_with_codec(name, request, codec)?;

    // LLM call start (sanitize guardrails happen inside nemo_flow_llm_call)
    let handle = nemo_flow_llm_call(
        name,
        &intercepted_request,
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
        model_name,
        annotated_request,
    )?;

    // Stream execution chain — build middleware chain under lock, release, then await
    let exec_future = {
        let ss = current_scope_stack();
        let ss_guard = ss.read().expect("scope stack lock poisoned");
        let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_stream_execution_intercepts);
        let ctx = global_context();
        let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
        state.llm_stream_build_execution_chain(name, func, &sl)
    };
    let exec_result = exec_future(intercepted_request).await;
    drop(exec_future);
    match exec_result {
        Ok(raw_stream) => {
            // Wrap in LlmStreamWrapper which handles collector/finalizer and END event
            let wrapper = LlmStreamWrapper::new(
                raw_stream,
                handle,
                collector,
                finalizer,
                data,
                metadata,
                response_codec,
            );
            let wrapped_stream =
                Box::pin(wrapper) as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>;
            Ok(wrapped_stream)
        }
        Err(err) => {
            let _ = emit_llm_end_without_output(&handle, data, metadata);
            Err(err)
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone middleware chain functions
// ---------------------------------------------------------------------------

/// Runs the registered tool request intercept chain on the given arguments.
///
/// Returns the transformed arguments after all intercepts have been applied.
/// This allows invoking request intercepts independently of the full
/// [`nemo_flow_tool_call_execute`] pipeline.
pub fn nemo_flow_tool_request_intercepts(name: &str, args: Json) -> Result<Json> {
    ensure_runtime_owner()?;
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_request_intercepts);
    let ctx = global_context();
    let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
    state.tool_request_intercepts_chain(name, args, &sl)
}

/// Runs the registered tool conditional execution guardrail chain.
///
/// Returns `Ok(())` if all guardrails pass, or
/// [`Err(FlowError::GuardrailRejected(reason))`](FlowError::GuardrailRejected)
/// if any guardrail rejects the call.
pub fn nemo_flow_tool_conditional_execution(name: &str, args: &Json) -> Result<()> {
    ensure_runtime_owner()?;
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.tool_conditional_execution_guardrails);
    let ctx = global_context();
    let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
    if let Some(err) = state.tool_conditional_execution_chain(name, args, &sl)? {
        return Err(FlowError::GuardrailRejected(err));
    }
    Ok(())
}

/// Runs the registered LLM request intercept chain on the given [`LLMRequest`].
///
/// Returns the transformed [`LLMRequest`] after all intercepts have been applied.
/// This allows invoking request intercepts independently of the full
/// [`nemo_flow_llm_call_execute`] pipeline.
pub fn nemo_flow_llm_request_intercepts(name: &str, request: LLMRequest) -> Result<LLMRequest> {
    ensure_runtime_owner()?;
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_request_intercepts);
    let ctx = global_context();
    let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
    let (req, _ann) = state.llm_request_intercepts_chain(name, request, None, &sl)?;
    Ok(req)
}

/// Runs the registered LLM conditional execution guardrail chain.
///
/// Returns `Ok(())` if all guardrails pass, or
/// [`Err(FlowError::GuardrailRejected(reason))`](FlowError::GuardrailRejected)
/// if any guardrail rejects the call.
pub fn nemo_flow_llm_conditional_execution(request: &LLMRequest) -> Result<()> {
    ensure_runtime_owner()?;
    let ss = current_scope_stack();
    let ss_guard = ss.read().expect("scope stack lock poisoned");
    let sl = ss_guard.collect_scope_local_registries(|r| &r.llm_conditional_execution_guardrails);
    let ctx = global_context();
    let state = ctx.read().map_err(|e| FlowError::Internal(e.to_string()))?;
    if let Some(err) = state.llm_conditional_execution_chain(request, &sl)? {
        return Err(FlowError::GuardrailRejected(err));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::await_holding_lock, clippy::type_complexity)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::shared_runtime::{reset_runtime_owner_for_tests, runtime_owner_test_mutex};

    struct TestMutexProxy;

    impl TestMutexProxy {
        fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'static, ()>> {
            runtime_owner_test_mutex().lock()
        }
    }

    static TEST_MUTEX: TestMutexProxy = TestMutexProxy;

    fn reset_global() {
        reset_runtime_owner_for_tests();
        let ctx = global_context();
        let mut state = ctx.write().unwrap();
        *state = NemoFlowContextState::new();
    }

    #[test]
    fn test_push_pop_scope() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        // Root scope is always present
        let root = nemo_flow_get_handle().unwrap();
        assert_eq!(root.name, "root");

        let handle = nemo_flow_push_scope(
            "test_scope",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(nemo_flow_get_handle().unwrap().name, "test_scope");
        nemo_flow_pop_scope(&handle.uuid).unwrap();

        // After pop, root scope is on top again
        assert_eq!(nemo_flow_get_handle().unwrap().name, "root");
    }

    #[test]
    fn test_subscriber_callbacks_run_outside_runtime_locks() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        let lock_checks = Arc::new(Mutex::new(Vec::new()));
        let captured = lock_checks.clone();
        nemo_flow_register_subscriber(
            "lock_probe",
            Arc::new(move |_| {
                let scope_stack_writable =
                    crate::context::current_scope_stack().try_write().is_ok();
                let global_context_writable = crate::context::global_context().try_write().is_ok();
                captured
                    .lock()
                    .unwrap()
                    .push((scope_stack_writable, global_context_writable));
            }),
        )
        .unwrap();

        let handle = nemo_flow_push_scope(
            "probe",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        nemo_flow_pop_scope(&handle.uuid).unwrap();

        let checks = lock_checks.lock().unwrap();
        assert_eq!(checks.len(), 2);
        assert!(
            checks
                .iter()
                .all(|(scope_ok, global_ok)| *scope_ok && *global_ok)
        );

        drop(checks);
        nemo_flow_deregister_subscriber("lock_probe").unwrap();
    }

    #[test]
    fn test_scope_events_observe_post_mutation_active_handle() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        let observations = Arc::new(Mutex::new(Vec::new()));
        let captured = observations.clone();
        nemo_flow_register_subscriber(
            "scope_visibility",
            Arc::new(move |event: &crate::types::Event| {
                if event.scope_type() != Some(ScopeType::Agent) {
                    return;
                }
                if event.name() != "visible_scope" {
                    return;
                }
                let active_uuid = nemo_flow_get_handle().unwrap().uuid;
                captured
                    .lock()
                    .unwrap()
                    .push((event.kind().to_string(), active_uuid));
            }),
        )
        .unwrap();

        let handle = nemo_flow_push_scope(
            "visible_scope",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        let parent_uuid = handle.parent_uuid.unwrap();
        nemo_flow_pop_scope(&handle.uuid).unwrap();

        let observations = observations.lock().unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0], ("ScopeStart".to_string(), handle.uuid));
        assert_eq!(observations[1], ("ScopeEnd".to_string(), parent_uuid));

        drop(observations);
        nemo_flow_deregister_subscriber("scope_visibility").unwrap();
    }

    #[test]
    fn test_subscriber_registration() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();
        let count = Arc::new(AtomicU32::new(0));
        let c = count.clone();
        nemo_flow_register_subscriber(
            "test_sub",
            Arc::new(move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            }),
        )
        .unwrap();

        // Duplicate should fail
        assert!(nemo_flow_register_subscriber("test_sub", Arc::new(|_| {}),).is_err());

        // Push scope emits event
        let handle = nemo_flow_push_scope(
            "s",
            ScopeType::Function,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);

        nemo_flow_pop_scope(&handle.uuid).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 2);

        // Deregister
        assert!(nemo_flow_deregister_subscriber("test_sub").unwrap());
        assert!(!nemo_flow_deregister_subscriber("test_sub").unwrap());
    }

    #[test]
    fn test_tool_guardrail_registration() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();
        nemo_flow_register_tool_sanitize_request_guardrail("g1", 10, Box::new(|_name, args| args))
            .unwrap();

        // Duplicate fails
        assert!(
            nemo_flow_register_tool_sanitize_request_guardrail(
                "g1",
                10,
                Box::new(|_name, args| args),
            )
            .is_err()
        );

        assert!(nemo_flow_deregister_tool_sanitize_request_guardrail("g1").unwrap());
    }

    // -- Scope hierarchy --

    #[test]
    fn test_nested_scopes() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        let s1 = nemo_flow_push_scope(
            "level1",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(nemo_flow_get_handle().unwrap().name, "level1");

        let s2 = nemo_flow_push_scope(
            "level2",
            ScopeType::Function,
            Some(&s1),
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(nemo_flow_get_handle().unwrap().name, "level2");
        assert_eq!(s2.parent_uuid, Some(s1.uuid));

        nemo_flow_pop_scope(&s2.uuid).unwrap();
        assert_eq!(nemo_flow_get_handle().unwrap().name, "level1");

        nemo_flow_pop_scope(&s1.uuid).unwrap();
        assert_eq!(nemo_flow_get_handle().unwrap().name, "root");
    }

    #[test]
    fn test_pop_nonexistent_scope() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();
        let result = nemo_flow_pop_scope(&Uuid::new_v4());
        assert!(result.is_err());
    }

    #[test]
    fn test_pop_non_top_scope_rejected_without_end_event() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        nemo_flow_register_subscriber(
            "capture_scope_events",
            Arc::new(move |event: &crate::types::Event| {
                captured
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), event.kind().to_string()));
            }),
        )
        .unwrap();

        let parent = nemo_flow_push_scope(
            "parent",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        let child = nemo_flow_push_scope(
            "child",
            ScopeType::Function,
            Some(&parent),
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();

        let err = nemo_flow_pop_scope(&parent.uuid).unwrap_err();
        assert!(matches!(err, FlowError::InvalidArgument(_)));
        assert_eq!(nemo_flow_get_handle().unwrap().uuid, child.uuid);

        let events = events.lock().unwrap();
        let parent_end_count = events
            .iter()
            .filter(|(name, event_kind)| name == "parent" && *event_kind == "ScopeEnd")
            .count();
        assert_eq!(parent_end_count, 0);

        drop(events);
        nemo_flow_pop_scope(&child.uuid).unwrap();
        nemo_flow_pop_scope(&parent.uuid).unwrap();
        nemo_flow_deregister_subscriber("capture_scope_events").unwrap();
    }

    #[test]
    fn test_scope_attributes_propagated() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();
        let handle = nemo_flow_push_scope(
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
        nemo_flow_pop_scope(&handle.uuid).unwrap();
    }

    // -- Event emission --

    #[test]
    fn test_event_emission() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nemo_flow_register_subscriber(
            "evt_test",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock()
                    .unwrap()
                    .push((e.name().to_string(), e.kind().to_string()));
            }),
        )
        .unwrap();

        nemo_flow_event("my_mark", None, Some(json!({"x": 1})), None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].0, "my_mark");
        assert_eq!(captured[0].1, "Mark");

        drop(captured);
        nemo_flow_deregister_subscriber("evt_test").unwrap();
    }

    // -- Tool lifecycle --

    #[test]
    fn test_tool_call_and_end() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nemo_flow_register_subscriber(
            "tool_test",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.kind().to_string());
            }),
        )
        .unwrap();

        let handle = nemo_flow_tool_call(
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

        nemo_flow_tool_call_end(&handle, json!({"output": "result"}), None, None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], "ToolStart");
        assert_eq!(captured[1], "ToolEnd");

        drop(captured);
        nemo_flow_deregister_subscriber("tool_test").unwrap();
    }

    #[test]
    fn test_tool_call_with_sanitize_guardrail() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        // Register a sanitizer that adds a field
        nemo_flow_register_tool_sanitize_request_guardrail(
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
        nemo_flow_register_subscriber(
            "tool_san_test",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let handle = nemo_flow_tool_call(
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
        let input = start_event.input().unwrap();
        assert_eq!(input["sanitized"], true);
        assert_eq!(input["input"], "data");

        drop(captured);
        nemo_flow_tool_call_end(&handle, json!("ok"), None, None).unwrap();
        nemo_flow_deregister_subscriber("tool_san_test").unwrap();
        nemo_flow_deregister_tool_sanitize_request_guardrail("sanitizer").unwrap();
    }

    #[test]
    fn test_tool_call_end_with_sanitize_response_guardrail() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        nemo_flow_register_tool_sanitize_response_guardrail(
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
        nemo_flow_register_subscriber(
            "tool_resp_test",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let handle = nemo_flow_tool_call(
            "tool",
            json!({}),
            None,
            ToolAttributes::empty(),
            None,
            None,
            None,
        )
        .unwrap();
        nemo_flow_tool_call_end(&handle, json!({"output": "raw"}), None, None).unwrap();

        let captured = events.lock().unwrap();
        let end_event = &captured[1];
        let output = end_event.output().unwrap();
        assert_eq!(output["cleaned"], true);
        assert_eq!(output["output"], "raw");

        drop(captured);
        nemo_flow_deregister_subscriber("tool_resp_test").unwrap();
        nemo_flow_deregister_tool_sanitize_response_guardrail("resp_sanitizer").unwrap();
    }

    // -- Tool call execute (async) --

    #[tokio::test]
    async fn test_tool_call_execute_basic() {
        let _lock = runtime_owner_test_mutex().lock().unwrap();
        reset_global();

        let func: ToolExecutionNextFn =
            Arc::new(|args| Box::pin(async move { Ok(json!({"result": args["input"]})) }));

        let result = nemo_flow_tool_call_execute(
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
    async fn test_tool_call_execute_failure_emits_end_event() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        nemo_flow_register_subscriber(
            "tool_exec_failure_sub",
            Arc::new(move |e: &crate::types::Event| {
                captured.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let func: ToolExecutionNextFn =
            Arc::new(|_args| Box::pin(async move { Err(FlowError::Internal("boom".into())) }));

        let err = nemo_flow_tool_call_execute(
            "failing_tool",
            json!({"input": true}),
            func,
            None,
            ToolAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap_err();

        assert!(matches!(err, FlowError::Internal(_)));

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].kind(), "ToolStart");
        assert_eq!(captured[1].kind(), "ToolEnd");
        assert_eq!(captured[0].uuid(), captured[1].uuid());
        assert!(captured[1].output().is_none());

        drop(captured);
        nemo_flow_deregister_subscriber("tool_exec_failure_sub").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_with_request_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_tool_request_intercept(
            "req_intercept",
            1,
            false,
            Box::new(|_name, mut args| {
                args.as_object_mut()
                    .unwrap()
                    .insert("added_by_intercept".into(), json!(true));
                Ok(args)
            }),
        )
        .unwrap();

        let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

        let result = nemo_flow_tool_call_execute(
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

        nemo_flow_deregister_tool_request_intercept("req_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nemo_flow_register_subscriber(
            "tool_reject_sub",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        nemo_flow_register_tool_conditional_execution_guardrail(
            "blocker",
            1,
            Box::new(|_name, _args| Ok(Some("forbidden tool".into()))),
        )
        .unwrap();

        let func: ToolExecutionNextFn =
            Arc::new(|_args| Box::pin(async move { Ok(json!({"should_not_reach": true})) }));

        let result = nemo_flow_tool_call_execute(
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
            FlowError::GuardrailRejected(msg) => assert_eq!(msg, "forbidden tool"),
            e => panic!("expected GuardrailRejected, got {e:?}"),
        }

        // Verify standalone Mark event with rejection data (no Start/End pair)
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].kind(), "Mark");
        let mark_data = captured[0].data().unwrap();
        assert_eq!(mark_data["rejected"], true);
        assert_eq!(mark_data["rejection_reason"], "forbidden tool");

        drop(captured);
        nemo_flow_deregister_subscriber("tool_reject_sub").unwrap();
        nemo_flow_deregister_tool_conditional_execution_guardrail("blocker").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_with_execution_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_tool_execution_intercept(
            "exec_intercept",
            1,
            Arc::new(|_name: &str, _args: Json, _next: ToolExecutionNextFn| {
                Box::pin(async move { Ok(json!({"from_intercept": true})) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();

        let func: ToolExecutionNextFn =
            Arc::new(|_args| Box::pin(async move { Ok(json!({"from_original": true})) }));

        let result = nemo_flow_tool_call_execute(
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

        nemo_flow_deregister_tool_execution_intercept("exec_intercept").unwrap();
    }

    // -- LLM lifecycle --

    #[test]
    fn test_llm_call_and_end() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nemo_flow_register_subscriber(
            "llm_test",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.kind().to_string());
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let handle = nemo_flow_llm_call(
            "my_llm",
            &request,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(handle.name, "my_llm");

        nemo_flow_llm_call_end(&handle, json!({"response": "ok"}), None, None, None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], "LLMStart");
        assert_eq!(captured[1], "LLMEnd");

        drop(captured);
        nemo_flow_deregister_subscriber("llm_test").unwrap();
    }

    #[test]
    fn test_llm_call_with_sanitize_guardrail() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_llm_sanitize_request_guardrail(
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
        nemo_flow_register_subscriber(
            "llm_san_test",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let handle = nemo_flow_llm_call(
            "llm",
            &request,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let captured = events.lock().unwrap();
        let start_event = &captured[0];
        // Sanitized request should be in input
        let input = start_event.input().unwrap();
        assert_eq!(input["headers"]["X-Sanitized"], "true");

        drop(captured);
        nemo_flow_llm_call_end(&handle, json!("ok"), None, None, None).unwrap();
        nemo_flow_deregister_subscriber("llm_san_test").unwrap();
        nemo_flow_deregister_llm_sanitize_request_guardrail("llm_sanitizer").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_basic() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let func: LlmExecutionNextFn =
            Arc::new(|req: LLMRequest| Box::pin(async move { Ok(json!({"echo": req.content})) }));

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": [{"role": "user", "content": "hi"}]}),
        };
        let content = request.content.clone();

        let result = nemo_flow_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["echo"], content);
    }

    #[tokio::test]
    async fn test_llm_call_execute_failure_emits_end_event() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        nemo_flow_register_subscriber(
            "llm_exec_failure_sub",
            Arc::new(move |e: &crate::types::Event| {
                captured.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let func: LlmExecutionNextFn =
            Arc::new(|_req| Box::pin(async move { Err(FlowError::Internal("boom".into())) }));

        let err = nemo_flow_llm_call_execute(
            "failing_llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap_err();

        assert!(matches!(err, FlowError::Internal(_)));

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].kind(), "LLMStart");
        assert_eq!(captured[1].kind(), "LLMEnd");
        assert_eq!(captured[0].uuid(), captured[1].uuid());
        assert!(captured[1].output().is_none());

        drop(captured);
        nemo_flow_deregister_subscriber("llm_exec_failure_sub").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nemo_flow_register_subscriber(
            "llm_reject_sub",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        nemo_flow_register_llm_conditional_execution_guardrail(
            "llm_blocker",
            1,
            Box::new(|_req: &LLMRequest| Ok(Some("blocked by policy".into()))),
        )
        .unwrap();

        let func: LlmExecutionNextFn = Arc::new(|_req| Box::pin(async move { Ok(json!({})) }));

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        let result = nemo_flow_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
            None,
            None,
        )
        .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            FlowError::GuardrailRejected(msg) => assert_eq!(msg, "blocked by policy"),
            e => panic!("expected GuardrailRejected, got {e:?}"),
        }

        // Verify standalone Mark event with rejection data (no Start/End pair)
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].kind(), "Mark");
        let mark_data = captured[0].data().unwrap();
        assert_eq!(mark_data["rejected"], true);
        assert_eq!(mark_data["rejection_reason"], "blocked by policy");

        drop(captured);
        nemo_flow_deregister_subscriber("llm_reject_sub").unwrap();
        nemo_flow_deregister_llm_conditional_execution_guardrail("llm_blocker").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_with_request_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_llm_request_intercept(
            "llm_req_intercept",
            1,
            false,
            Box::new(|_name: &str, mut req: LLMRequest, annotated| {
                req.headers.insert("intercepted".into(), json!(true));
                Ok((req, annotated))
            }),
        )
        .unwrap();

        let func: LlmExecutionNextFn = Arc::new(|req: LLMRequest| {
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

        let result = nemo_flow_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["saw_intercepted"], true);

        nemo_flow_deregister_llm_request_intercept("llm_req_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_with_execution_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_llm_execution_intercept(
            "llm_exec_intercept",
            1,
            Arc::new(|_name: &str, _req: LLMRequest, _next: LlmExecutionNextFn| {
                Box::pin(async move { Ok(json!({"from_intercept": true})) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();

        let func: LlmExecutionNextFn =
            Arc::new(|_req| Box::pin(async move { Ok(json!({"from_original": true})) }));

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        let result = nemo_flow_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["from_intercept"], true);
        assert!(result.get("from_original").is_none());

        nemo_flow_deregister_llm_execution_intercept("llm_exec_intercept").unwrap();
    }

    // -- All guardrail/intercept registration pairs --

    #[test]
    fn test_tool_sanitize_response_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_tool_sanitize_response_guardrail("g1", 1, Box::new(|_n, r| r)).unwrap();
        assert!(
            nemo_flow_register_tool_sanitize_response_guardrail("g1", 1, Box::new(|_n, r| r))
                .is_err()
        );
        assert!(nemo_flow_deregister_tool_sanitize_response_guardrail("g1").unwrap());
        assert!(!nemo_flow_deregister_tool_sanitize_response_guardrail("g1").unwrap());
    }

    #[test]
    fn test_tool_conditional_execution_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_tool_conditional_execution_guardrail(
            "g1",
            1,
            Box::new(|_n, _a| Ok(None)),
        )
        .unwrap();
        assert!(
            nemo_flow_register_tool_conditional_execution_guardrail(
                "g1",
                1,
                Box::new(|_n, _a| Ok(None))
            )
            .is_err()
        );
        assert!(nemo_flow_deregister_tool_conditional_execution_guardrail("g1").unwrap());
    }

    #[test]
    fn test_tool_request_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_tool_request_intercept("i1", 1, false, Box::new(|_n, a| Ok(a))).unwrap();
        assert!(
            nemo_flow_register_tool_request_intercept("i1", 1, false, Box::new(|_n, a| Ok(a)))
                .is_err()
        );
        assert!(nemo_flow_deregister_tool_request_intercept("i1").unwrap());
    }

    #[test]
    fn test_tool_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_tool_execution_intercept(
            "i1",
            1,
            Arc::new(|_name: &str, a: Json, _next: ToolExecutionNextFn| {
                Box::pin(async move { Ok(a) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();
        assert!(
            nemo_flow_register_tool_execution_intercept(
                "i1",
                1,
                Arc::new(
                    |_name: &str, a: Json, _next: ToolExecutionNextFn| Box::pin(
                        async move { Ok(a) }
                    )
                        as Pin<
                            Box<
                                dyn std::future::Future<Output = crate::error::Result<Json>> + Send,
                            >,
                        >
                ),
            )
            .is_err()
        );
        assert!(nemo_flow_deregister_tool_execution_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_sanitize_request_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_llm_sanitize_request_guardrail("g1", 1, Box::new(|r| r)).unwrap();
        assert!(
            nemo_flow_register_llm_sanitize_request_guardrail("g1", 1, Box::new(|r| r)).is_err()
        );
        assert!(nemo_flow_deregister_llm_sanitize_request_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_sanitize_response_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_llm_sanitize_response_guardrail("g1", 1, Box::new(|r| r)).unwrap();
        assert!(
            nemo_flow_register_llm_sanitize_response_guardrail("g1", 1, Box::new(|r| r)).is_err()
        );
        assert!(nemo_flow_deregister_llm_sanitize_response_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_conditional_execution_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_llm_conditional_execution_guardrail("g1", 1, Box::new(|_r| Ok(None)))
            .unwrap();
        assert!(
            nemo_flow_register_llm_conditional_execution_guardrail(
                "g1",
                1,
                Box::new(|_r| Ok(None))
            )
            .is_err()
        );
        assert!(nemo_flow_deregister_llm_conditional_execution_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_request_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_llm_request_intercept(
            "i1",
            1,
            false,
            Box::new(|_name: &str, r, a| Ok((r, a))),
        )
        .unwrap();
        assert!(
            nemo_flow_register_llm_request_intercept(
                "i1",
                1,
                false,
                Box::new(|_name: &str, r, a| Ok((r, a)))
            )
            .is_err()
        );
        assert!(nemo_flow_deregister_llm_request_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_llm_execution_intercept(
            "i1",
            1,
            Arc::new(
                |_name: &str, _request: LLMRequest, _next: LlmExecutionNextFn| {
                    Box::pin(async move { Ok(json!({})) })
                        as Pin<
                            Box<
                                dyn std::future::Future<Output = crate::error::Result<Json>> + Send,
                            >,
                        >
                },
            ),
        )
        .unwrap();
        assert!(nemo_flow_deregister_llm_execution_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_stream_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nemo_flow_register_llm_stream_execution_intercept(
            "i1",
            1,
            Arc::new(
                |_name: &str, _request: LLMRequest, _next: LlmStreamExecutionNextFn| {
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
                },
            ),
        )
        .unwrap();
        assert!(nemo_flow_deregister_llm_stream_execution_intercept("i1").unwrap());
    }

    // -- Deregister non-existent returns false --

    #[test]
    fn test_deregister_nonexistent_subscriber() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nemo_flow_deregister_subscriber("nonexistent").unwrap());
    }

    #[test]
    fn test_deregister_nonexistent_guardrails() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nemo_flow_deregister_tool_sanitize_request_guardrail("nope").unwrap());
        assert!(!nemo_flow_deregister_tool_sanitize_response_guardrail("nope").unwrap());
        assert!(!nemo_flow_deregister_tool_conditional_execution_guardrail("nope").unwrap());
        assert!(!nemo_flow_deregister_llm_sanitize_request_guardrail("nope").unwrap());
        assert!(!nemo_flow_deregister_llm_sanitize_response_guardrail("nope").unwrap());
        assert!(!nemo_flow_deregister_llm_conditional_execution_guardrail("nope").unwrap());
    }

    #[test]
    fn test_deregister_nonexistent_intercepts() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nemo_flow_deregister_tool_request_intercept("nope").unwrap());
        assert!(!nemo_flow_deregister_tool_execution_intercept("nope").unwrap());
        assert!(!nemo_flow_deregister_llm_request_intercept("nope").unwrap());
        assert!(!nemo_flow_deregister_llm_execution_intercept("nope").unwrap());
        assert!(!nemo_flow_deregister_llm_stream_execution_intercept("nope").unwrap());
    }

    // -- LLM stream call execute --

    #[tokio::test]
    async fn test_llm_stream_call_execute_basic() {
        use tokio_stream::StreamExt;

        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let func: LlmStreamExecutionNextFn = Arc::new(|_request: LLMRequest| {
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
        let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(move |chunk| {
            cc.lock().unwrap().push(chunk);
            Ok(())
        });
        let fc = collected.clone();
        let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(move || {
            let chunks = fc.lock().unwrap();
            Json::Array(chunks.clone())
        });

        let mut stream = nemo_flow_llm_stream_call_execute(
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
    async fn test_llm_stream_call_execute_setup_failure_emits_end_event() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        nemo_flow_register_subscriber(
            "llm_stream_exec_failure_sub",
            Arc::new(move |e: &crate::types::Event| {
                captured.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let func: LlmStreamExecutionNextFn =
            Arc::new(|_req| Box::pin(async move { Err(FlowError::Internal("boom".into())) }));

        let result = nemo_flow_llm_stream_call_execute(
            "failing_stream_llm",
            request,
            func,
            Box::new(|_chunk| Ok(())),
            Box::new(|| json!({"unused": true})),
            None,
            LLMAttributes::empty(),
            None,
            None,
            None,
            None,
            None,
        )
        .await;

        assert!(matches!(result, Err(FlowError::Internal(_))));

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].kind(), "LLMStart");
        assert_eq!(captured[1].kind(), "LLMEnd");
        assert_eq!(captured[0].uuid(), captured[1].uuid());
        assert!(captured[1].output().is_none());

        drop(captured);
        nemo_flow_deregister_subscriber("llm_stream_exec_failure_sub").unwrap();
    }

    #[tokio::test]
    async fn test_llm_stream_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nemo_flow_register_subscriber(
            "stream_reject_sub",
            Arc::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        nemo_flow_register_llm_conditional_execution_guardrail(
            "stream_blocker",
            1,
            Box::new(|_req: &LLMRequest| Ok(Some("stream blocked".into()))),
        )
        .unwrap();

        let func: LlmStreamExecutionNextFn = Arc::new(|_request: LLMRequest| {
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

        let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_| Ok(()));
        let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| Json::Null);

        let result = nemo_flow_llm_stream_call_execute(
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
            None,
            None,
        )
        .await;

        match result {
            Err(FlowError::GuardrailRejected(msg)) => assert_eq!(msg, "stream blocked"),
            Err(e) => panic!("expected GuardrailRejected, got {e:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }

        // Verify standalone Mark event with rejection data (no Start/End pair)
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].kind(), "Mark");
        let mark_data = captured[0].data().unwrap();
        assert_eq!(mark_data["rejected"], true);
        assert_eq!(mark_data["rejection_reason"], "stream blocked");

        drop(captured);
        nemo_flow_deregister_subscriber("stream_reject_sub").unwrap();
        nemo_flow_deregister_llm_conditional_execution_guardrail("stream_blocker").unwrap();
    }

    // -- Tool call with explicit parent --

    #[test]
    fn test_tool_call_with_parent() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let scope = nemo_flow_push_scope(
            "parent",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        let handle = nemo_flow_tool_call(
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
        nemo_flow_tool_call_end(&handle, json!({}), None, None).unwrap();
        nemo_flow_pop_scope(&scope.uuid).unwrap();
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
        let handle = nemo_flow_llm_call(
            "llm",
            &request,
            None,
            LLMAttributes::STATELESS | LLMAttributes::STREAMING,
            Some(json!({"custom": "data"})),
            Some(json!({"meta": "info"})),
            None,
            None,
        )
        .unwrap();

        assert!(handle.attributes.contains(LLMAttributes::STATELESS));
        assert!(handle.attributes.contains(LLMAttributes::STREAMING));
        nemo_flow_llm_call_end(&handle, json!({}), None, None, None).unwrap();
    }

    // -- Standalone middleware chain tests --

    #[test]
    fn test_tool_request_intercepts_standalone() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_tool_request_intercept(
            "add_field",
            10,
            false,
            Box::new(|_name, mut args| {
                if let Some(obj) = args.as_object_mut() {
                    obj.insert("injected".into(), json!(true));
                }
                Ok(args)
            }),
        )
        .unwrap();

        let result = nemo_flow_tool_request_intercepts("tool", json!({"key": "value"})).unwrap();
        assert_eq!(result["key"], "value");
        assert_eq!(result["injected"], true);

        nemo_flow_deregister_tool_request_intercept("add_field").unwrap();
    }

    #[test]
    fn test_tool_conditional_execution_standalone_pass() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        // No guardrails registered — should pass
        assert!(nemo_flow_tool_conditional_execution("tool", &json!({})).is_ok());
    }

    #[test]
    fn test_tool_conditional_execution_standalone_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_tool_conditional_execution_guardrail(
            "blocker",
            1,
            Box::new(|_name, _args| Ok(Some("blocked".into()))),
        )
        .unwrap();

        match nemo_flow_tool_conditional_execution("tool", &json!({})) {
            Err(FlowError::GuardrailRejected(msg)) => assert_eq!(msg, "blocked"),
            other => panic!("expected GuardrailRejected, got {other:?}"),
        }

        nemo_flow_deregister_tool_conditional_execution_guardrail("blocker").unwrap();
    }

    #[test]
    fn test_tool_request_intercepts_standalone_callback_error() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_tool_request_intercept(
            "broken",
            1,
            false,
            Box::new(|_name, _args| Err(FlowError::Internal("tool intercept failed".into()))),
        )
        .unwrap();

        match nemo_flow_tool_request_intercepts("tool", json!({})) {
            Err(FlowError::Internal(msg)) => assert_eq!(msg, "tool intercept failed"),
            other => panic!("expected Internal error, got {other:?}"),
        }

        nemo_flow_deregister_tool_request_intercept("broken").unwrap();
    }

    #[test]
    fn test_tool_conditional_execution_standalone_callback_error() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_tool_conditional_execution_guardrail(
            "broken",
            1,
            Box::new(|_name, _args| Err(FlowError::Internal("tool conditional failed".into()))),
        )
        .unwrap();

        match nemo_flow_tool_conditional_execution("tool", &json!({})) {
            Err(FlowError::Internal(msg)) => assert_eq!(msg, "tool conditional failed"),
            other => panic!("expected Internal error, got {other:?}"),
        }

        nemo_flow_deregister_tool_conditional_execution_guardrail("broken").unwrap();
    }

    #[test]
    fn test_llm_request_intercepts_standalone() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_llm_request_intercept(
            "add_field",
            10,
            false,
            Box::new(|_name: &str, mut request: LLMRequest, annotated| {
                request
                    .content
                    .as_object_mut()
                    .unwrap()
                    .insert("intercepted".into(), json!(true));
                Ok((request, annotated))
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        let result = nemo_flow_llm_request_intercepts("test_llm", request).unwrap();
        assert_eq!(result.content["intercepted"], true);
        assert_eq!(result.content["messages"], json!([]));

        nemo_flow_deregister_llm_request_intercept("add_field").unwrap();
    }

    #[test]
    fn test_llm_conditional_execution_standalone_pass() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        assert!(nemo_flow_llm_conditional_execution(&request).is_ok());
    }

    #[test]
    fn test_llm_conditional_execution_standalone_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_llm_conditional_execution_guardrail(
            "blocker",
            1,
            Box::new(|_req| Ok(Some("llm blocked".into()))),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };
        match nemo_flow_llm_conditional_execution(&request) {
            Err(FlowError::GuardrailRejected(msg)) => assert_eq!(msg, "llm blocked"),
            other => panic!("expected GuardrailRejected, got {other:?}"),
        }

        nemo_flow_deregister_llm_conditional_execution_guardrail("blocker").unwrap();
    }

    #[test]
    fn test_llm_request_intercepts_standalone_callback_error() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_llm_request_intercept(
            "broken",
            1,
            false,
            Box::new(|_name, _request, _annotated| {
                Err(FlowError::Internal("llm request intercept failed".into()))
            }),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        match nemo_flow_llm_request_intercepts("test_llm", request) {
            Err(FlowError::Internal(msg)) => assert_eq!(msg, "llm request intercept failed"),
            other => panic!("expected Internal error, got {other:?}"),
        }

        nemo_flow_deregister_llm_request_intercept("broken").unwrap();
    }

    #[test]
    fn test_llm_conditional_execution_standalone_callback_error() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nemo_flow_register_llm_conditional_execution_guardrail(
            "broken",
            1,
            Box::new(|_request| Err(FlowError::Internal("llm conditional failed".into()))),
        )
        .unwrap();

        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        };

        match nemo_flow_llm_conditional_execution(&request) {
            Err(FlowError::Internal(msg)) => assert_eq!(msg, "llm conditional failed"),
            other => panic!("expected Internal error, got {other:?}"),
        }

        nemo_flow_deregister_llm_conditional_execution_guardrail("broken").unwrap();
    }
}
