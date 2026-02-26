//! Public API for the NVAgentRT runtime.
//!
//! This module contains all top-level functions that language bindings and
//! application code call. The API is organized into several groups:
//!
//! - **Scope operations** — [`nv_agentrt_get_handle`], [`nv_agentrt_push_scope`],
//!   [`nv_agentrt_pop_scope`], [`nv_agentrt_event`]
//! - **Tool lifecycle** — [`nv_agentrt_tool_call`], [`nv_agentrt_tool_call_end`],
//!   [`nv_agentrt_tool_call_execute`]
//! - **LLM lifecycle** — [`nv_agentrt_llm_call`], [`nv_agentrt_llm_call_end`],
//!   [`nv_agentrt_llm_call_execute`], [`nv_agentrt_llm_stream_call_execute`]
//! - **Guardrail registration** — `nv_agentrt_register_*_guardrail` /
//!   `nv_agentrt_deregister_*_guardrail` for tool and LLM sanitize/conditional guardrails
//! - **Intercept registration** — `nv_agentrt_register_*_intercept` /
//!   `nv_agentrt_deregister_*_intercept` for tool and LLM request/response/execution intercepts
//! - **Subscriber registration** — [`nv_agentrt_register_subscriber`],
//!   [`nv_agentrt_deregister_subscriber`]
//!
//! All functions operate on the global context singleton returned by
//! [`global_context`].

use std::pin::Pin;

use serde_json::json;
use tokio_stream::Stream;
use uuid::Uuid;

use crate::context::*;
use crate::error::{AgentRtError, Result};
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
    ($register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        pub fn $register_name(name: &str, priority: i32, guardrail: $fn_type) -> Result<()> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| AgentRtError::Internal(e.to_string()))?;
            state
                .$field
                .register(
                    name.to_string(),
                    GuardrailEntry {
                        priority,
                        guardrail,
                    },
                )
                .map_err(|e| AgentRtError::AlreadyExists(e))
        }

        pub fn $deregister_name(name: &str) -> Result<bool> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| AgentRtError::Internal(e.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

macro_rules! intercept_registry_api {
    ($register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        pub fn $register_name(
            name: &str,
            priority: i32,
            break_chain: bool,
            callable: $fn_type,
        ) -> Result<()> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| AgentRtError::Internal(e.to_string()))?;
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
                .map_err(|e| AgentRtError::AlreadyExists(e))
        }

        pub fn $deregister_name(name: &str) -> Result<bool> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| AgentRtError::Internal(e.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

macro_rules! execution_intercept_registry_api {
    ($register_name:ident, $deregister_name:ident, $field:ident, $cond_type:ty, $fn_type:ty) => {
        pub fn $register_name(
            name: &str,
            priority: i32,
            conditional: $cond_type,
            callable: $fn_type,
        ) -> Result<()> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| AgentRtError::Internal(e.to_string()))?;
            state
                .$field
                .register(
                    name.to_string(),
                    ExecutionIntercept {
                        priority,
                        conditional,
                        callable,
                    },
                )
                .map_err(|e| AgentRtError::AlreadyExists(e))
        }

        pub fn $deregister_name(name: &str) -> Result<bool> {
            let ctx = global_context();
            let mut state = ctx
                .write()
                .map_err(|e| AgentRtError::Internal(e.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

// ---------------------------------------------------------------------------
// Tool guardrail registrations
//
// Each pair generates:
//   - `nv_agentrt_register_*`: registers a named guardrail with a priority.
//     Returns AlreadyExists if the name is taken.
//   - `nv_agentrt_deregister_*`: removes a guardrail by name.
//     Returns Ok(true) if it existed, Ok(false) otherwise.
// ---------------------------------------------------------------------------

// Registers a tool request sanitize guardrail that transforms tool arguments before execution.
// Callback signature: `(tool_name: &str, args: Json) -> Json`.
//
// Errors: Returns `AgentRtError::AlreadyExists` if a guardrail with the given name is already registered.
// deregister: Deregisters a tool request sanitize guardrail by name.
guardrail_registry_api!(
    nv_agentrt_register_tool_sanitize_request_guardrail,
    nv_agentrt_deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);

// Registers a tool response sanitize guardrail that transforms tool results after execution.
// Callback signature: `(tool_name: &str, result: Json) -> Json`.
// deregister: Deregisters a tool response sanitize guardrail by name.
guardrail_registry_api!(
    nv_agentrt_register_tool_sanitize_response_guardrail,
    nv_agentrt_deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);

// Registers a tool conditional execution guardrail that can reject tool calls.
// Callback signature: `(tool_name: &str, args: &Json) -> Option<rejection_reason>`.
// Return `None` to allow, `Some(reason)` to reject.
// deregister: Deregisters a tool conditional execution guardrail by name.
guardrail_registry_api!(
    nv_agentrt_register_tool_conditional_execution_guardrail,
    nv_agentrt_deregister_tool_conditional_execution_guardrail,
    tool_conditional_execution_guardrails,
    ToolConditionalFn
);

// ---------------------------------------------------------------------------
// Tool intercept registrations
//
// Each pair generates register/deregister for intercepts that transform
// data flowing through the tool call pipeline.
// ---------------------------------------------------------------------------

// Registers a tool request intercept that transforms arguments before sanitize guardrails.
// Callback signature: `(tool_name: &str, args: Json) -> Json`.
// Set `break_chain = true` to prevent subsequent intercepts from running.
// deregister: Deregisters a tool request intercept by name.
intercept_registry_api!(
    nv_agentrt_register_tool_request_intercept,
    nv_agentrt_deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);

// Registers a tool response intercept that transforms the result after execution.
// Callback signature: `(tool_name: &str, result: Json) -> Json`.
// Set `break_chain = true` to prevent subsequent intercepts from running.
// deregister: Deregisters a tool response intercept by name.
intercept_registry_api!(
    nv_agentrt_register_tool_response_intercept,
    nv_agentrt_deregister_tool_response_intercept,
    tool_response_intercepts,
    ToolInterceptFn
);

// Registers a tool execution intercept that conditionally replaces the tool's execution function.
// The `conditional` is checked first: `(tool_name: &str, args: &Json) -> bool`.
// If it returns `true`, the `callable` is used instead: `(args: Json) -> Future<Result<Json>>`.
// deregister: Deregisters a tool execution intercept by name.
execution_intercept_registry_api!(
    nv_agentrt_register_tool_execution_intercept,
    nv_agentrt_deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionConditionalFn,
    ToolExecutionFn
);

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

// Registers an LLM request sanitize guardrail that transforms the request before execution.
// Callback signature: `(request: LLMRequest) -> LLMRequest`.
// deregister: Deregisters an LLM request sanitize guardrail by name.
guardrail_registry_api!(
    nv_agentrt_register_llm_sanitize_request_guardrail,
    nv_agentrt_deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);

// Registers an LLM response sanitize guardrail that transforms the response after execution.
// Callback signature: `(response: Json) -> Json`.
// deregister: Deregisters an LLM response sanitize guardrail by name.
guardrail_registry_api!(
    nv_agentrt_register_llm_sanitize_response_guardrail,
    nv_agentrt_deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);

// Registers an LLM conditional execution guardrail that can reject LLM calls.
// Callback signature: `(request: &LLMRequest) -> Option<rejection_reason>`.
// Return `None` to allow, `Some(reason)` to reject.
// deregister: Deregisters an LLM conditional execution guardrail by name.
guardrail_registry_api!(
    nv_agentrt_register_llm_conditional_execution_guardrail,
    nv_agentrt_deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

// Registers an LLM request intercept that transforms the request before sanitize guardrails.
// Callback signature: `(request: LLMRequest) -> LLMRequest`.
// deregister: Deregisters an LLM request intercept by name.
intercept_registry_api!(
    nv_agentrt_register_llm_request_intercept,
    nv_agentrt_deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);

// Registers an LLM response intercept that transforms the response after execution.
// Callback signature: `(response: Json) -> Json`.
// deregister: Deregisters an LLM response intercept by name.
intercept_registry_api!(
    nv_agentrt_register_llm_response_intercept,
    nv_agentrt_deregister_llm_response_intercept,
    llm_response_intercepts,
    LlmResponseInterceptFn
);

// Registers an LLM stream response intercept applied to each SSE event during streaming.
// Callback signature: `(event: SseEvent) -> SseEvent`.
// deregister: Deregisters an LLM stream response intercept by name.
intercept_registry_api!(
    nv_agentrt_register_llm_stream_response_intercept,
    nv_agentrt_deregister_llm_stream_response_intercept,
    llm_stream_response_intercepts,
    LlmStreamResponseInterceptFn
);

// Registers an LLM execution intercept that conditionally replaces the LLM execution function.
// The `conditional` is checked first: `(request: &LLMRequest) -> bool`.
// If it returns `true`, the `callable` is used: `(request: LLMRequest) -> Future<Result<Json>>`.
// deregister: Deregisters an LLM execution intercept by name.
execution_intercept_registry_api!(
    nv_agentrt_register_llm_execution_intercept,
    nv_agentrt_deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionConditionalFn,
    LlmExecutionFn
);

// Registers an LLM streaming execution intercept that conditionally replaces the stream execution function.
// The `conditional` is checked first: `(request: &LLMRequest) -> bool`.
// If it returns `true`, the `callable` is used: `(request: LLMRequest) -> Future<Result<Stream>>`.
// deregister: Deregisters an LLM streaming execution intercept by name.
execution_intercept_registry_api!(
    nv_agentrt_register_llm_stream_execution_intercept,
    nv_agentrt_deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionConditionalFn,
    LlmStreamExecutionFn
);

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Registers a named event subscriber that will be called for every lifecycle event.
///
/// Returns [`AgentRtError::AlreadyExists`] if a subscriber with the given name
/// is already registered.
pub fn nv_agentrt_register_subscriber(name: &str, callback: EventSubscriberFn) -> Result<()> {
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
    if state.event_subscribers.contains_key(name) {
        return Err(AgentRtError::AlreadyExists(format!(
            "{name} subscriber already exists"
        )));
    }
    state.event_subscribers.insert(name.to_string(), callback);
    Ok(())
}

/// Deregisters an event subscriber by name. Returns `true` if it existed, `false` otherwise.
pub fn nv_agentrt_deregister_subscriber(name: &str) -> Result<bool> {
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
    Ok(state.event_subscribers.remove(name).is_some())
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Returns a clone of the current top scope handle from the scope stack.
///
/// Always succeeds because the root scope is always present.
pub fn nv_agentrt_get_handle() -> Result<ScopeHandle> {
    Ok(task_scope_top())
}

/// Creates a new scope and pushes it onto the scope stack.
///
/// Emits a `Start` event to all subscribers. If `parent` is `None`, the current
/// top of the scope stack is used as the parent.
///
/// Returns the new [`ScopeHandle`].
pub fn nv_agentrt_push_scope(
    name: &str,
    scope_type: ScopeType,
    parent: Option<&ScopeHandle>,
    attributes: ScopeAttributes,
) -> Result<ScopeHandle> {
    let parent_uuid = resolve_parent_uuid(parent);
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
    let handle = state.create_scope_handle(name, parent_uuid, scope_type, attributes);
    task_scope_push(handle.clone());
    Ok(handle)
}

/// Removes a scope from the scope stack by UUID and emits an `End` event.
///
/// Returns [`AgentRtError::NotFound`] if the UUID is not in the stack.
pub fn nv_agentrt_pop_scope(handle_uuid: &Uuid) -> Result<()> {
    let scope = task_scope_remove(handle_uuid)?;
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
    state.end_scope_handle(&scope);
    Ok(())
}

/// Emits a standalone marker event to all subscribers.
///
/// This is a lightweight way to record application-specific events (e.g.,
/// checkpoints, metrics) without creating a scope or handle.
pub fn nv_agentrt_event(
    name: &str,
    parent: Option<&ScopeHandle>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    let parent_uuid = resolve_parent_uuid(parent);
    let ctx = global_context();
    let state = ctx
        .read()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;
    state.create_event(name, parent_uuid, data, metadata);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begins a tool call: runs request sanitize guardrails, creates a tool handle,
/// and emits a `Start` event.
///
/// The sanitized arguments are stored in the handle's `data` under `"sanitized_args"`.
/// Call [`nv_agentrt_tool_call_end`] when the tool completes.
pub fn nv_agentrt_tool_call(
    name: &str,
    args: Json,
    parent: Option<&ScopeHandle>,
    attributes: ToolAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<ToolHandle> {
    let parent_uuid = resolve_parent_uuid(parent);
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;

    let sanitized_args = state.tool_sanitize_request_chain(name, args);
    let mut data = data.unwrap_or_else(|| json!({}));
    if let Some(obj) = data.as_object_mut() {
        obj.insert("sanitized_args".to_string(), sanitized_args);
    }

    Ok(state.create_tool_handle(name, parent_uuid, attributes, Some(data), metadata))
}

/// Ends a tool call: runs response sanitize guardrails and emits an `End` event.
///
/// The sanitized result is stored in the event's `data` under `"sanitized_result"`.
pub fn nv_agentrt_tool_call_end(
    handle: &ToolHandle,
    result: Json,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;

    let sanitized_result = state.tool_sanitize_response_chain(&handle.name, result);
    let mut data = data.unwrap_or_else(|| json!({}));
    if let Some(obj) = data.as_object_mut() {
        obj.insert("sanitized_result".to_string(), sanitized_result);
    }

    state.end_tool_handle(handle, Some(data), metadata);
    Ok(())
}

/// Executes a complete tool call lifecycle: request intercepts, sanitize guardrails,
/// conditional guardrails, execution (with optional execution intercept override),
/// response intercepts, and sanitize response guardrails.
///
/// This is the high-level function that orchestrates the full middleware pipeline.
/// Returns [`AgentRtError::GuardrailRejected`] if a conditional guardrail rejects the call.
pub async fn nv_agentrt_tool_call_execute(
    name: &str,
    args: Json,
    func: ToolExecutionFn,
    parent: Option<ScopeHandle>,
    attributes: ToolAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<Json> {
    // Request intercepts
    let intercepted_args = {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        state.tool_request_intercepts_chain(name, args)
    };

    // Tool call start
    let handle = nv_agentrt_tool_call(
        name,
        intercepted_args.clone(),
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
    )?;

    // Conditional guardrails
    {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        if let Some(err) = state.tool_conditional_execution_chain(name, &intercepted_args) {
            return Err(AgentRtError::GuardrailRejected(err));
        }
    }

    // Execution chain — find intercept under lock, release, then await
    let exec_future = {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        if state.tool_find_execution_intercept(name, &intercepted_args) {
            state.tool_call_execution_intercept(name, intercepted_args)
        } else {
            func(intercepted_args)
        }
    };
    let result = exec_future.await?;

    // Response intercepts
    let result = {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        state.tool_response_intercepts_chain(name, result)
    };

    // Tool call end
    nv_agentrt_tool_call_end(&handle, result.clone(), data, metadata)?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begins an LLM call: runs request sanitize guardrails, creates an LLM handle,
/// and emits a `Start` event.
///
/// The sanitized request is stored in the handle's `data` under `"sanitized_request"`.
/// Call [`nv_agentrt_llm_call_end`] when the LLM call completes.
pub fn nv_agentrt_llm_call(
    name: &str,
    request: &LLMRequest,
    parent: Option<&ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<LLMHandle> {
    let parent_uuid = resolve_parent_uuid(parent);
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;

    let sanitized_request = state.llm_sanitize_request_chain(request.clone());
    let mut data = data.unwrap_or_else(|| json!({}));
    if let Some(obj) = data.as_object_mut() {
        obj.insert(
            "sanitized_request".to_string(),
            serde_json::to_value(&sanitized_request).unwrap_or(Json::Null),
        );
    }

    Ok(state.create_llm_handle(name, parent_uuid, attributes, Some(data), metadata))
}

/// Ends an LLM call: runs response sanitize guardrails and emits an `End` event.
pub fn nv_agentrt_llm_call_end(
    handle: &LLMHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    let ctx = global_context();
    let mut state = ctx
        .write()
        .map_err(|e| AgentRtError::Internal(e.to_string()))?;

    let sanitized_response = state.llm_sanitize_response_chain(response);
    let mut data = data.unwrap_or_else(|| json!({}));
    if let Some(obj) = data.as_object_mut() {
        obj.insert("sanitized_result".to_string(), sanitized_response);
    }

    state.end_llm_handle(handle, Some(data), metadata);
    Ok(())
}

/// Executes a complete non-streaming LLM call lifecycle: request intercepts,
/// sanitize guardrails, conditional guardrails, execution (with optional intercept
/// override), response intercepts, and sanitize response guardrails.
///
/// Returns [`AgentRtError::GuardrailRejected`] if a conditional guardrail rejects the call.
pub async fn nv_agentrt_llm_call_execute(
    name: &str,
    request: LLMRequest,
    func: LlmExecutionFn,
    parent: Option<ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<Json> {
    // Request intercepts
    let intercepted_request = {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        state.llm_request_intercepts_chain(request)
    };

    // LLM call start
    let handle = nv_agentrt_llm_call(
        name,
        &intercepted_request,
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
    )?;

    // Conditional guardrails
    {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        if let Some(err) = state.llm_conditional_execution_chain(&intercepted_request) {
            return Err(AgentRtError::GuardrailRejected(err));
        }
    }

    // Execution chain — find intercept under lock, release, then await
    let exec_future = {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        if state.llm_find_execution_intercept(&intercepted_request) {
            state.llm_call_execution_intercept(intercepted_request)
        } else {
            func(intercepted_request)
        }
    };
    let response = exec_future.await?;

    // Response intercepts
    let response = {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        state.llm_response_intercepts_chain(response)
    };

    // LLM call end
    nv_agentrt_llm_call_end(&handle, response.clone(), data, metadata)?;

    Ok(response)
}

/// Executes a complete streaming LLM call lifecycle.
///
/// Similar to [`nv_agentrt_llm_call_execute`] but returns a
/// [`Stream`] of SSE text chunks. The returned stream is
/// wrapped in [`LlmStreamWrapper`] which handles SSE parsing, per-event
/// intercepts, event aggregation, and automatic `End` event emission when
/// the stream is exhausted.
///
/// Returns [`AgentRtError::GuardrailRejected`] if a conditional guardrail rejects the call.
pub async fn nv_agentrt_llm_stream_call_execute(
    name: &str,
    request: LLMRequest,
    func: LlmStreamExecutionFn,
    parent: Option<ScopeHandle>,
    attributes: LLMAttributes,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
    // Request intercepts
    let intercepted_request = {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        state.llm_request_intercepts_chain(request)
    };

    // LLM call start
    let handle = nv_agentrt_llm_call(
        name,
        &intercepted_request,
        parent.as_ref(),
        attributes,
        data.clone(),
        metadata.clone(),
    )?;

    // Conditional guardrails
    {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        if let Some(err) = state.llm_conditional_execution_chain(&intercepted_request) {
            return Err(AgentRtError::GuardrailRejected(err));
        }
    }

    // Stream execution chain — find intercept under lock, release, then await
    let exec_future = {
        let ctx = global_context();
        let mut state = ctx
            .write()
            .map_err(|e| AgentRtError::Internal(e.to_string()))?;
        if state.llm_stream_find_execution_intercept(&intercepted_request) {
            state.llm_stream_call_execution_intercept(intercepted_request)
        } else {
            func(intercepted_request)
        }
    };
    let raw_stream = exec_future.await?;

    // Wrap in LlmStreamWrapper which handles parsing, intercepts, and END event
    let wrapper = LlmStreamWrapper::new(raw_stream, handle, data, metadata);
    Ok(Box::pin(wrapper))
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
        *state = NVAgentRTContextState::new();
    }

    #[test]
    fn test_push_pop_scope() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        // Root scope is always present
        let root = nv_agentrt_get_handle().unwrap();
        assert_eq!(root.name, "root");

        let handle = nv_agentrt_push_scope(
            "test_scope",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
        )
        .unwrap();
        assert_eq!(nv_agentrt_get_handle().unwrap().name, "test_scope");
        nv_agentrt_pop_scope(&handle.uuid).unwrap();

        // After pop, root scope is on top again
        assert_eq!(nv_agentrt_get_handle().unwrap().name, "root");
    }

    #[test]
    fn test_subscriber_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        let count = Arc::new(AtomicU32::new(0));
        let c = count.clone();
        nv_agentrt_register_subscriber(
            "test_sub",
            Box::new(move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            }),
        )
        .unwrap();

        // Duplicate should fail
        assert!(nv_agentrt_register_subscriber("test_sub", Box::new(|_| {}),).is_err());

        // Push scope emits event
        let handle =
            nv_agentrt_push_scope("s", ScopeType::Function, None, ScopeAttributes::empty())
                .unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);

        nv_agentrt_pop_scope(&handle.uuid).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 2);

        // Deregister
        assert!(nv_agentrt_deregister_subscriber("test_sub").unwrap());
        assert!(!nv_agentrt_deregister_subscriber("test_sub").unwrap());
    }

    #[test]
    fn test_tool_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_tool_sanitize_request_guardrail("g1", 10, Box::new(|_name, args| args))
            .unwrap();

        // Duplicate fails
        assert!(nv_agentrt_register_tool_sanitize_request_guardrail(
            "g1",
            10,
            Box::new(|_name, args| args),
        )
        .is_err());

        assert!(nv_agentrt_deregister_tool_sanitize_request_guardrail("g1").unwrap());
    }

    // -- Scope hierarchy --

    #[test]
    fn test_nested_scopes() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let s1 = nv_agentrt_push_scope("level1", ScopeType::Agent, None, ScopeAttributes::empty())
            .unwrap();
        assert_eq!(nv_agentrt_get_handle().unwrap().name, "level1");

        let s2 = nv_agentrt_push_scope(
            "level2",
            ScopeType::Function,
            Some(&s1),
            ScopeAttributes::empty(),
        )
        .unwrap();
        assert_eq!(nv_agentrt_get_handle().unwrap().name, "level2");
        assert_eq!(s2.parent_uuid, Some(s1.uuid));

        nv_agentrt_pop_scope(&s2.uuid).unwrap();
        assert_eq!(nv_agentrt_get_handle().unwrap().name, "level1");

        nv_agentrt_pop_scope(&s1.uuid).unwrap();
        assert_eq!(nv_agentrt_get_handle().unwrap().name, "root");
    }

    #[test]
    fn test_pop_nonexistent_scope() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        let result = nv_agentrt_pop_scope(&Uuid::new_v4());
        assert!(result.is_err());
    }

    #[test]
    fn test_scope_attributes_propagated() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        let handle = nv_agentrt_push_scope(
            "parallel_scope",
            ScopeType::Agent,
            None,
            ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE,
        )
        .unwrap();
        assert!(handle.attributes.contains(ScopeAttributes::PARALLEL));
        assert!(handle.attributes.contains(ScopeAttributes::RELOCATABLE));
        nv_agentrt_pop_scope(&handle.uuid).unwrap();
    }

    // -- Event emission --

    #[test]
    fn test_event_emission() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nv_agentrt_register_subscriber(
            "evt_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push((e.name.clone(), e.event_type));
            }),
        )
        .unwrap();

        nv_agentrt_event("my_mark", None, Some(json!({"x": 1})), None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].0, Some("my_mark".into()));
        assert_eq!(captured[0].1, crate::types::EventType::Mark);

        drop(captured);
        nv_agentrt_deregister_subscriber("evt_test").unwrap();
    }

    // -- Tool lifecycle --

    #[test]
    fn test_tool_call_and_end() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nv_agentrt_register_subscriber(
            "tool_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.event_type);
            }),
        )
        .unwrap();

        let handle = nv_agentrt_tool_call(
            "my_tool",
            json!({"input": "data"}),
            None,
            ToolAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(handle.name, "my_tool");

        nv_agentrt_tool_call_end(&handle, json!({"output": "result"}), None, None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], crate::types::EventType::Start);
        assert_eq!(captured[1], crate::types::EventType::End);

        drop(captured);
        nv_agentrt_deregister_subscriber("tool_test").unwrap();
    }

    #[test]
    fn test_tool_call_with_sanitize_guardrail() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        // Register a sanitizer that adds a field
        nv_agentrt_register_tool_sanitize_request_guardrail(
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
        nv_agentrt_register_subscriber(
            "tool_san_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let handle = nv_agentrt_tool_call(
            "my_tool",
            json!({"input": "data"}),
            None,
            ToolAttributes::empty(),
            None,
            None,
        )
        .unwrap();

        // The start event data should contain sanitized_args
        let captured = events.lock().unwrap();
        let start_event = &captured[0];
        let data = start_event.data.as_ref().unwrap();
        assert_eq!(data["sanitized_args"]["sanitized"], true);
        assert_eq!(data["sanitized_args"]["input"], "data");

        drop(captured);
        nv_agentrt_tool_call_end(&handle, json!("ok"), None, None).unwrap();
        nv_agentrt_deregister_subscriber("tool_san_test").unwrap();
        nv_agentrt_deregister_tool_sanitize_request_guardrail("sanitizer").unwrap();
    }

    #[test]
    fn test_tool_call_end_with_sanitize_response_guardrail() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_tool_sanitize_response_guardrail(
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
        nv_agentrt_register_subscriber(
            "tool_resp_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let handle =
            nv_agentrt_tool_call("tool", json!({}), None, ToolAttributes::empty(), None, None)
                .unwrap();
        nv_agentrt_tool_call_end(&handle, json!({"output": "raw"}), None, None).unwrap();

        let captured = events.lock().unwrap();
        let end_event = &captured[1];
        let data = end_event.data.as_ref().unwrap();
        assert_eq!(data["sanitized_result"]["cleaned"], true);
        assert_eq!(data["sanitized_result"]["output"], "raw");

        drop(captured);
        nv_agentrt_deregister_subscriber("tool_resp_test").unwrap();
        nv_agentrt_deregister_tool_sanitize_response_guardrail("resp_sanitizer").unwrap();
    }

    // -- Tool call execute (async) --

    #[tokio::test]
    async fn test_tool_call_execute_basic() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let func: ToolExecutionFn =
            Box::new(|args| Box::pin(async move { Ok(json!({"result": args["input"]})) }));

        let result = nv_agentrt_tool_call_execute(
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

        nv_agentrt_register_tool_request_intercept(
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

        let func: ToolExecutionFn = Box::new(|args| Box::pin(async move { Ok(args) }));

        let result = nv_agentrt_tool_call_execute(
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

        nv_agentrt_deregister_tool_request_intercept("req_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_with_response_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_tool_response_intercept(
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

        let func: ToolExecutionFn =
            Box::new(|_args| Box::pin(async move { Ok(json!({"output": "raw"})) }));

        let result = nv_agentrt_tool_call_execute(
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

        nv_agentrt_deregister_tool_response_intercept("resp_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_tool_conditional_execution_guardrail(
            "blocker",
            1,
            Box::new(|_name, _args| Some("forbidden tool".into())),
        )
        .unwrap();

        let func: ToolExecutionFn =
            Box::new(|_args| Box::pin(async move { Ok(json!({"should_not_reach": true})) }));

        let result = nv_agentrt_tool_call_execute(
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
            AgentRtError::GuardrailRejected(msg) => assert_eq!(msg, "forbidden tool"),
            e => panic!("expected GuardrailRejected, got {e:?}"),
        }

        nv_agentrt_deregister_tool_conditional_execution_guardrail("blocker").unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_execute_with_execution_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_tool_execution_intercept(
            "exec_intercept",
            1,
            Box::new(|_name: &str, _args: &Json| true),
            Box::new(|_args: Json| {
                Box::pin(async move { Ok(json!({"from_intercept": true})) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();

        let func: ToolExecutionFn =
            Box::new(|_args| Box::pin(async move { Ok(json!({"from_original": true})) }));

        let result = nv_agentrt_tool_call_execute(
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

        nv_agentrt_deregister_tool_execution_intercept("exec_intercept").unwrap();
    }

    // -- LLM lifecycle --

    #[test]
    fn test_llm_call_and_end() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let events = Arc::new(Mutex::new(Vec::new()));
        let ec = events.clone();
        nv_agentrt_register_subscriber(
            "llm_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.event_type);
            }),
        )
        .unwrap();

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com/v1/chat".into(),
            headers: serde_json::Map::new(),
            body: json!({"messages": []}),
        };
        let handle =
            nv_agentrt_llm_call("my_llm", &request, None, LLMAttributes::empty(), None, None)
                .unwrap();
        assert_eq!(handle.name, "my_llm");

        nv_agentrt_llm_call_end(&handle, json!({"response": "ok"}), None, None).unwrap();

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], crate::types::EventType::Start);
        assert_eq!(captured[1], crate::types::EventType::End);

        drop(captured);
        nv_agentrt_deregister_subscriber("llm_test").unwrap();
    }

    #[test]
    fn test_llm_call_with_sanitize_guardrail() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_llm_sanitize_request_guardrail(
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
        nv_agentrt_register_subscriber(
            "llm_san_test",
            Box::new(move |e: &crate::types::Event| {
                ec.lock().unwrap().push(e.clone());
            }),
        )
        .unwrap();

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };
        let handle =
            nv_agentrt_llm_call("llm", &request, None, LLMAttributes::empty(), None, None).unwrap();

        let captured = events.lock().unwrap();
        let start_event = &captured[0];
        let data = start_event.data.as_ref().unwrap();
        // Sanitized request should be in data
        let sanitized_req = &data["sanitized_request"];
        assert_eq!(sanitized_req["headers"]["X-Sanitized"], "true");

        drop(captured);
        nv_agentrt_llm_call_end(&handle, json!("ok"), None, None).unwrap();
        nv_agentrt_deregister_subscriber("llm_san_test").unwrap();
        nv_agentrt_deregister_llm_sanitize_request_guardrail("llm_sanitizer").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_basic() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let func: LlmExecutionFn =
            Box::new(|req: LLMRequest| Box::pin(async move { Ok(json!({"model": req.url})) }));

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };

        let result = nv_agentrt_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["model"], "https://api.example.com");
    }

    #[tokio::test]
    async fn test_llm_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_llm_conditional_execution_guardrail(
            "llm_blocker",
            1,
            Box::new(|_req: &LLMRequest| Some("blocked by policy".into())),
        )
        .unwrap();

        let func: LlmExecutionFn = Box::new(|_req| Box::pin(async move { Ok(json!({})) }));

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };

        let result = nv_agentrt_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
        )
        .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            AgentRtError::GuardrailRejected(msg) => assert_eq!(msg, "blocked by policy"),
            e => panic!("expected GuardrailRejected, got {e:?}"),
        }

        nv_agentrt_deregister_llm_conditional_execution_guardrail("llm_blocker").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_with_request_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_llm_request_intercept(
            "llm_req_intercept",
            1,
            false,
            Box::new(|mut req: LLMRequest| {
                req.url = "https://intercepted.example.com".into();
                req
            }),
        )
        .unwrap();

        let func: LlmExecutionFn =
            Box::new(|req: LLMRequest| Box::pin(async move { Ok(json!({"called_url": req.url})) }));

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://original.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };

        let result = nv_agentrt_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["called_url"], "https://intercepted.example.com");

        nv_agentrt_deregister_llm_request_intercept("llm_req_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_with_response_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_llm_response_intercept(
            "llm_resp_intercept",
            1,
            false,
            Box::new(|mut resp: Json| {
                resp.as_object_mut()
                    .unwrap()
                    .insert("response_modified".into(), json!(true));
                resp
            }),
        )
        .unwrap();

        let func: LlmExecutionFn =
            Box::new(|_req| Box::pin(async move { Ok(json!({"original": true})) }));

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };

        let result = nv_agentrt_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["original"], true);
        assert_eq!(result["response_modified"], true);

        nv_agentrt_deregister_llm_response_intercept("llm_resp_intercept").unwrap();
    }

    #[tokio::test]
    async fn test_llm_call_execute_with_execution_intercept() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_llm_execution_intercept(
            "llm_exec_intercept",
            1,
            Box::new(|_req: &LLMRequest| true),
            Box::new(|_req: LLMRequest| {
                Box::pin(async move { Ok(json!({"from_intercept": true})) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();

        let func: LlmExecutionFn =
            Box::new(|_req| Box::pin(async move { Ok(json!({"from_original": true})) }));

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };

        let result = nv_agentrt_llm_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::empty(),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result["from_intercept"], true);
        assert!(result.get("from_original").is_none());

        nv_agentrt_deregister_llm_execution_intercept("llm_exec_intercept").unwrap();
    }

    // -- All guardrail/intercept registration pairs --

    #[test]
    fn test_tool_sanitize_response_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_tool_sanitize_response_guardrail("g1", 1, Box::new(|_n, r| r)).unwrap();
        assert!(
            nv_agentrt_register_tool_sanitize_response_guardrail("g1", 1, Box::new(|_n, r| r))
                .is_err()
        );
        assert!(nv_agentrt_deregister_tool_sanitize_response_guardrail("g1").unwrap());
        assert!(!nv_agentrt_deregister_tool_sanitize_response_guardrail("g1").unwrap());
    }

    #[test]
    fn test_tool_conditional_execution_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_tool_conditional_execution_guardrail("g1", 1, Box::new(|_n, _a| None))
            .unwrap();
        assert!(nv_agentrt_register_tool_conditional_execution_guardrail(
            "g1",
            1,
            Box::new(|_n, _a| None)
        )
        .is_err());
        assert!(nv_agentrt_deregister_tool_conditional_execution_guardrail("g1").unwrap());
    }

    #[test]
    fn test_tool_request_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_tool_request_intercept("i1", 1, false, Box::new(|_n, a| a)).unwrap();
        assert!(
            nv_agentrt_register_tool_request_intercept("i1", 1, false, Box::new(|_n, a| a))
                .is_err()
        );
        assert!(nv_agentrt_deregister_tool_request_intercept("i1").unwrap());
    }

    #[test]
    fn test_tool_response_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_tool_response_intercept("i1", 1, false, Box::new(|_n, r| r)).unwrap();
        assert!(
            nv_agentrt_register_tool_response_intercept("i1", 1, false, Box::new(|_n, r| r))
                .is_err()
        );
        assert!(nv_agentrt_deregister_tool_response_intercept("i1").unwrap());
    }

    #[test]
    fn test_tool_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_tool_execution_intercept(
            "i1",
            1,
            Box::new(|_n: &str, _a: &Json| false),
            Box::new(|a: Json| {
                Box::pin(async move { Ok(a) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();
        assert!(nv_agentrt_register_tool_execution_intercept(
            "i1",
            1,
            Box::new(|_n: &str, _a: &Json| false),
            Box::new(|a: Json| Box::pin(async move { Ok(a) })
                as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>),
        )
        .is_err());
        assert!(nv_agentrt_deregister_tool_execution_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_sanitize_request_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_llm_sanitize_request_guardrail("g1", 1, Box::new(|r| r)).unwrap();
        assert!(
            nv_agentrt_register_llm_sanitize_request_guardrail("g1", 1, Box::new(|r| r)).is_err()
        );
        assert!(nv_agentrt_deregister_llm_sanitize_request_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_sanitize_response_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_llm_sanitize_response_guardrail("g1", 1, Box::new(|r| r)).unwrap();
        assert!(
            nv_agentrt_register_llm_sanitize_response_guardrail("g1", 1, Box::new(|r| r)).is_err()
        );
        assert!(nv_agentrt_deregister_llm_sanitize_response_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_conditional_execution_guardrail_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_llm_conditional_execution_guardrail("g1", 1, Box::new(|_r| None))
            .unwrap();
        assert!(nv_agentrt_register_llm_conditional_execution_guardrail(
            "g1",
            1,
            Box::new(|_r| None)
        )
        .is_err());
        assert!(nv_agentrt_deregister_llm_conditional_execution_guardrail("g1").unwrap());
    }

    #[test]
    fn test_llm_request_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_llm_request_intercept("i1", 1, false, Box::new(|r| r)).unwrap();
        assert!(
            nv_agentrt_register_llm_request_intercept("i1", 1, false, Box::new(|r| r)).is_err()
        );
        assert!(nv_agentrt_deregister_llm_request_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_response_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_llm_response_intercept("i1", 1, false, Box::new(|r| r)).unwrap();
        assert!(
            nv_agentrt_register_llm_response_intercept("i1", 1, false, Box::new(|r| r)).is_err()
        );
        assert!(nv_agentrt_deregister_llm_response_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_stream_response_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_llm_stream_response_intercept("i1", 1, false, Box::new(|e| e)).unwrap();
        assert!(
            nv_agentrt_register_llm_stream_response_intercept("i1", 1, false, Box::new(|e| e))
                .is_err()
        );
        assert!(nv_agentrt_deregister_llm_stream_response_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_llm_execution_intercept(
            "i1",
            1,
            Box::new(|_r: &LLMRequest| false),
            Box::new(|_r: LLMRequest| {
                Box::pin(async move { Ok(json!({})) })
                    as Pin<Box<dyn std::future::Future<Output = crate::error::Result<Json>> + Send>>
            }),
        )
        .unwrap();
        assert!(nv_agentrt_deregister_llm_execution_intercept("i1").unwrap());
    }

    #[test]
    fn test_llm_stream_execution_intercept_registration() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        nv_agentrt_register_llm_stream_execution_intercept(
            "i1",
            1,
            Box::new(|_r: &LLMRequest| false),
            Box::new(|_r: LLMRequest| {
                Box::pin(async move {
                    let stream: Pin<Box<dyn Stream<Item = crate::error::Result<String>> + Send>> =
                        Box::pin(tokio_stream::empty());
                    Ok(stream)
                })
                    as Pin<
                        Box<
                            dyn std::future::Future<
                                    Output = crate::error::Result<
                                        Pin<
                                            Box<
                                                dyn Stream<Item = crate::error::Result<String>>
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
        assert!(nv_agentrt_deregister_llm_stream_execution_intercept("i1").unwrap());
    }

    // -- Deregister non-existent returns false --

    #[test]
    fn test_deregister_nonexistent_subscriber() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nv_agentrt_deregister_subscriber("nonexistent").unwrap());
    }

    #[test]
    fn test_deregister_nonexistent_guardrails() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nv_agentrt_deregister_tool_sanitize_request_guardrail("nope").unwrap());
        assert!(!nv_agentrt_deregister_tool_sanitize_response_guardrail("nope").unwrap());
        assert!(!nv_agentrt_deregister_tool_conditional_execution_guardrail("nope").unwrap());
        assert!(!nv_agentrt_deregister_llm_sanitize_request_guardrail("nope").unwrap());
        assert!(!nv_agentrt_deregister_llm_sanitize_response_guardrail("nope").unwrap());
        assert!(!nv_agentrt_deregister_llm_conditional_execution_guardrail("nope").unwrap());
    }

    #[test]
    fn test_deregister_nonexistent_intercepts() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();
        assert!(!nv_agentrt_deregister_tool_request_intercept("nope").unwrap());
        assert!(!nv_agentrt_deregister_tool_response_intercept("nope").unwrap());
        assert!(!nv_agentrt_deregister_tool_execution_intercept("nope").unwrap());
        assert!(!nv_agentrt_deregister_llm_request_intercept("nope").unwrap());
        assert!(!nv_agentrt_deregister_llm_response_intercept("nope").unwrap());
        assert!(!nv_agentrt_deregister_llm_stream_response_intercept("nope").unwrap());
        assert!(!nv_agentrt_deregister_llm_execution_intercept("nope").unwrap());
        assert!(!nv_agentrt_deregister_llm_stream_execution_intercept("nope").unwrap());
    }

    // -- LLM stream call execute --

    #[tokio::test]
    async fn test_llm_stream_call_execute_basic() {
        use tokio_stream::StreamExt;

        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let func: LlmStreamExecutionFn = Box::new(|_req: LLMRequest| {
            Box::pin(async move {
                let items = vec![
                    Ok("data: {\"token\": \"hello\"}\n\n".to_string()),
                    Ok("data: {\"token\": \"world\"}\n\n".to_string()),
                ];
                let stream: Pin<Box<dyn Stream<Item = crate::error::Result<String>> + Send>> =
                    Box::pin(tokio_stream::iter(items));
                Ok(stream)
            })
                as Pin<
                    Box<
                        dyn std::future::Future<
                                Output = crate::error::Result<
                                    Pin<
                                        Box<dyn Stream<Item = crate::error::Result<String>> + Send>,
                                    >,
                                >,
                            > + Send,
                    >,
                >
        });

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };

        let mut stream = nv_agentrt_llm_stream_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::STREAMING,
            None,
            None,
        )
        .await
        .unwrap();

        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }

        // Should have received 2 SSE events
        assert!(chunks.len() >= 2);
    }

    #[tokio::test]
    async fn test_llm_stream_call_execute_conditional_rejection() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        nv_agentrt_register_llm_conditional_execution_guardrail(
            "stream_blocker",
            1,
            Box::new(|_req: &LLMRequest| Some("stream blocked".into())),
        )
        .unwrap();

        let func: LlmStreamExecutionFn = Box::new(|_req: LLMRequest| {
            Box::pin(async move {
                let stream: Pin<Box<dyn Stream<Item = crate::error::Result<String>> + Send>> =
                    Box::pin(tokio_stream::empty());
                Ok(stream)
            })
                as Pin<
                    Box<
                        dyn std::future::Future<
                                Output = crate::error::Result<
                                    Pin<
                                        Box<dyn Stream<Item = crate::error::Result<String>> + Send>,
                                    >,
                                >,
                            > + Send,
                    >,
                >
        });

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };

        let result = nv_agentrt_llm_stream_call_execute(
            "llm",
            request,
            func,
            None,
            LLMAttributes::STREAMING,
            None,
            None,
        )
        .await;

        match result {
            Err(AgentRtError::GuardrailRejected(msg)) => assert_eq!(msg, "stream blocked"),
            Err(e) => panic!("expected GuardrailRejected, got {e:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }

        nv_agentrt_deregister_llm_conditional_execution_guardrail("stream_blocker").unwrap();
    }

    // -- Tool call with explicit parent --

    #[test]
    fn test_tool_call_with_parent() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let scope =
            nv_agentrt_push_scope("parent", ScopeType::Agent, None, ScopeAttributes::empty())
                .unwrap();
        let handle = nv_agentrt_tool_call(
            "tool",
            json!({}),
            Some(&scope),
            ToolAttributes::empty(),
            None,
            None,
        )
        .unwrap();

        assert_eq!(handle.parent_uuid, Some(scope.uuid));
        nv_agentrt_tool_call_end(&handle, json!({}), None, None).unwrap();
        nv_agentrt_pop_scope(&scope.uuid).unwrap();
    }

    // -- LLM call with attributes --

    #[test]
    fn test_llm_call_with_attributes() {
        let _lock = TEST_MUTEX.lock().unwrap();
        reset_global();

        let request = LLMRequest {
            method: "POST".into(),
            url: "https://api.example.com".into(),
            headers: serde_json::Map::new(),
            body: json!({}),
        };
        let handle = nv_agentrt_llm_call(
            "llm",
            &request,
            None,
            LLMAttributes::STATELESS | LLMAttributes::STREAMING,
            Some(json!({"custom": "data"})),
            Some(json!({"meta": "info"})),
        )
        .unwrap();

        assert!(handle.attributes.contains(LLMAttributes::STATELESS));
        assert!(handle.attributes.contains(LLMAttributes::STREAMING));
        nv_agentrt_llm_call_end(&handle, json!({}), None, None).unwrap();
    }
}
