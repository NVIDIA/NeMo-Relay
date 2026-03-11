// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public NAPI API functions for the NVMagic Node.js bindings.
//!
//! This module exposes the full agent runtime API to JavaScript/TypeScript:
//! scope stack management, tool and LLM lifecycle operations, guardrail and
//! intercept registration/deregistration, and event subscriber management.
//! All functions are annotated with `#[napi]` and their doc comments appear
//! in the generated `index.d.ts` TypeScript definitions.

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction};
use napi_derive::napi;
use serde_json::Value as Json;
use tokio_stream::StreamExt;

use nvmagic_core as core;
use nvmagic_core::types as core_types;

use crate::callable;
use crate::convert::{opt_json, to_napi_err};
use crate::stream::LlmStream;
use crate::types::*;

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Creates a new isolated scope stack.
#[napi]
pub fn create_scope_stack() -> JsScopeStack {
    JsScopeStack {
        inner: nvmagic_core::create_scope_stack(),
    }
}

/// Returns the current execution context's scope stack handle.
#[napi]
pub fn current_scope_stack() -> JsScopeStack {
    JsScopeStack {
        inner: nvmagic_core::current_scope_stack(),
    }
}

/// Binds a scope stack to the current thread.
#[napi]
pub fn set_thread_scope_stack(stack: &JsScopeStack) {
    nvmagic_core::set_thread_scope_stack(stack.inner.clone());
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Get the handle for the current top-of-stack execution scope.
///
/// Returns the `JsScopeHandle` for the innermost active scope on the current task's scope stack.
/// Throws if the scope stack is empty.
#[napi]
pub fn get_handle() -> Result<JsScopeHandle> {
    core::nvmagic_get_handle()
        .map(JsScopeHandle::from)
        .map_err(to_napi_err)
}

/// Push a new execution scope onto the scope stack.
///
/// Creates a child scope with the given `name` and `scopeType`. If `handle` is provided,
/// the new scope is parented to that scope; otherwise it is parented to the current top scope.
/// Optional `attributes` is a bitfield of scope attribute flags (e.g., `SCOPE_ATTR_PARALLEL`).
/// Returns the handle for the newly created scope.
#[napi]
pub fn push_scope(
    name: String,
    scope_type: ScopeType,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
) -> Result<JsScopeHandle> {
    let attrs = core_types::ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    core::nvmagic_push_scope(&name, scope_type.into(), handle.map(|h| &h.inner), attrs)
        .map(JsScopeHandle::from)
        .map_err(to_napi_err)
}

/// Pop an execution scope from the scope stack.
///
/// Removes the scope identified by `handle` from the stack and emits an end event.
/// Throws if the handle does not match the current top scope.
#[napi]
pub fn pop_scope(handle: &JsScopeHandle) -> Result<()> {
    core::nvmagic_pop_scope(&handle.inner.uuid).map_err(to_napi_err)
}

/// Emit a custom mark event on the current scope.
///
/// Emits a named event with optional `data` and `metadata` payloads. If `handle` is provided,
/// the event is associated with that scope; otherwise it uses the current top scope.
#[napi]
pub fn event(
    name: String,
    handle: Option<&JsScopeHandle>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    core::nvmagic_event(
        &name,
        handle.map(|h| &h.inner),
        opt_json(data),
        opt_json(metadata),
    )
    .map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begin a tool call, running request guardrails and intercepts.
///
/// Registers a tool invocation with the given `name` and `args`. Guardrails and request
/// intercepts are applied before the call proceeds. Returns a `JsToolHandle` that must
/// be passed to `toolCallEnd()` when the tool finishes. Optional `handle` specifies the
/// parent scope; `attributes` is a bitfield (e.g., `TOOL_ATTR_LOCAL`).
#[napi]
pub fn tool_call(
    name: String,
    args: Json,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    tool_call_id: Option<String>,
) -> Result<JsToolHandle> {
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    core::nvmagic_tool_call(
        &name,
        args,
        handle.map(|h| &h.inner),
        attrs,
        opt_json(data),
        opt_json(metadata),
        tool_call_id,
    )
    .map(JsToolHandle::from)
    .map_err(to_napi_err)
}

/// End a tool call, running response guardrails and intercepts.
///
/// Signals that the tool call identified by `handle` has completed with the given `result`.
/// Response guardrails and intercepts are applied to the result.
#[napi]
pub fn tool_call_end(
    handle: &JsToolHandle,
    result: Json,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    core::nvmagic_tool_call_end(&handle.inner, result, opt_json(data), opt_json(metadata))
        .map_err(to_napi_err)
}

/// Execute a tool call end-to-end with full lifecycle management.
///
/// Runs conditional-execution guardrails (on raw args) → request intercepts →
/// sanitize-request guardrails → execution intercepts → `func` → response
/// intercepts → sanitize-response guardrails. On rejection, only a standalone
/// Mark event is emitted (no Start/End pair) and `GuardrailRejected` is returned.
/// Returns the final (possibly intercepted) tool result.
#[napi]
pub async fn tool_call_execute(
    name: String,
    args: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<Json> {
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(core::task_scope_top);
    let exec_fn = callable::wrap_js_tool_exec_fn(func);
    let default_fn: nvmagic_core::ToolExecutionNextFn = Box::new(move |args| exec_fn(args));
    let scope_stack = nvmagic_core::current_scope_stack();

    nvmagic_core::TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            core::nvmagic_tool_call_execute(
                &name,
                args,
                default_fn,
                Some(parent),
                attrs,
                opt_json(data),
                opt_json(metadata),
            )
            .await
            .map_err(to_napi_err)
        })
        .await
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begin an LLM call, running request guardrails and intercepts.
///
/// Registers an LLM invocation with the given provider `name` and native request payload.
/// Returns a `JsLLMHandle` that must be passed to `llmCallEnd()` when the response is received.
/// Optional `handle` specifies the parent scope; `attributes` is a bitfield
/// (e.g., `LLM_ATTR_STREAMING`). Optional `toRequest` converts native JSON to a formal
/// `JsLLMRequest` for guardrails; defaults to the identity converter.
#[allow(clippy::too_many_arguments)]
#[napi]
pub fn llm_call(
    name: String,
    native: Json,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    to_request: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
) -> Result<JsLLMHandle> {
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let to_request_fn = to_request.map(wrap_js_to_request);
    core::nvmagic_llm_call(
        &name,
        &native,
        handle.map(|h| &h.inner),
        attrs,
        opt_json(data),
        opt_json(metadata),
        model_name,
        to_request_fn.as_ref(),
    )
    .map(JsLLMHandle::from)
    .map_err(to_napi_err)
}

/// End an LLM call, running response guardrails and intercepts.
///
/// Signals that the LLM call identified by `handle` has completed with the given `response`.
/// Response guardrails and intercepts are applied to the response. Optional `toResponse`
/// converts native JSON to a formal `JsLLMResponse` for guardrails; defaults to the
/// identity converter.
#[napi]
pub fn llm_call_end(
    handle: &JsLLMHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
    to_response: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
) -> Result<()> {
    let to_response_fn = to_response.map(wrap_js_to_response);
    core::nvmagic_llm_call_end(
        &handle.inner,
        response,
        opt_json(data),
        opt_json(metadata),
        to_response_fn.as_ref(),
    )
    .map_err(to_napi_err)
}

/// Execute an LLM call end-to-end with full lifecycle management.
///
/// Runs conditional-execution guardrails (on raw request) → request intercepts →
/// sanitize-request guardrails → execution intercepts → `func` → response
/// intercepts → sanitize-response guardrails. On rejection, only a standalone
/// Mark event is emitted (no Start/End pair) and `GuardrailRejected` is returned.
/// Returns the final (possibly intercepted) LLM response.
#[allow(clippy::too_many_arguments)]
#[napi]
pub async fn llm_call_execute(
    name: String,
    native: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    to_request: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    to_response: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
) -> Result<Json> {
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(core::task_scope_top);
    let exec_fn = callable::wrap_js_llm_exec_fn(func);
    let default_fn: nvmagic_core::LlmExecutionNextFn = Box::new(move |req| exec_fn(req));
    let to_request_fn = to_request.map(wrap_js_to_request);
    let to_response_fn = to_response.map(wrap_js_to_response);
    let scope_stack = nvmagic_core::current_scope_stack();

    nvmagic_core::TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            core::nvmagic_llm_call_execute(
                &name,
                native,
                default_fn,
                Some(parent),
                attrs,
                opt_json(data),
                opt_json(metadata),
                model_name,
                to_request_fn,
                to_response_fn,
            )
            .await
            .map_err(to_napi_err)
        })
        .await
}

/// Execute a streaming LLM call end-to-end with full lifecycle management.
///
/// Like `llmCallExecute`, conditional-execution guardrails run first on the raw request.
/// Returns an `LlmStream` whose `next()` method yields response chunks incrementally.
/// The `func` callback receives the native request as JSON and its response is streamed back.
/// Stream-level intercepts are applied to each chunk.
///
/// The optional `collector` callback is invoked with each intercepted chunk as JSON,
/// allowing the caller to accumulate chunks for aggregation. The optional `finalizer`
/// callback is invoked once when the stream is exhausted and must return a JSON value
/// representing the aggregated response.
#[allow(clippy::too_many_arguments)]
#[napi]
pub async fn llm_stream_call_execute(
    name: String,
    native: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    collector: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    finalizer: Option<ThreadsafeFunction<(), ErrorStrategy::Fatal>>,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    to_request: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    to_response: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
) -> Result<LlmStream> {
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(core::task_scope_top);

    // For stream execution, we need the stream-specific wrapper
    let exec_fn = callable::wrap_js_llm_exec_fn(func);

    let wrapped_collector: Box<dyn FnMut(Json) + Send> = match collector {
        Some(cb) => callable::wrap_js_collector_fn(cb),
        None => Box::new(|_: Json| {}),
    };

    let wrapped_finalizer: Box<dyn FnOnce() -> Json + Send> = match finalizer {
        Some(cb) => callable::wrap_js_finalizer_fn(cb),
        None => Box::new(|| Json::Null),
    };

    // Bridge LlmExecutionFn -> LlmStreamExecutionNextFn (FnOnce)
    let default_fn: nvmagic_core::LlmStreamExecutionNextFn = Box::new(move |native| {
        let fut = exec_fn(native);
        Box::pin(async move {
            let result = fut.await?;
            let stream = tokio_stream::once(Ok(result));
            Ok(Box::pin(stream)
                as std::pin::Pin<
                    Box<dyn tokio_stream::Stream<Item = nvmagic_core::Result<Json>> + Send>,
                >)
        })
    });

    let to_request_fn = to_request.map(wrap_js_to_request);
    let to_response_fn = to_response.map(wrap_js_to_response);
    let scope_stack = nvmagic_core::current_scope_stack();

    nvmagic_core::TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            let rust_stream = core::nvmagic_llm_stream_call_execute(
                &name,
                native,
                default_fn,
                wrapped_collector,
                wrapped_finalizer,
                Some(parent),
                attrs,
                opt_json(data),
                opt_json(metadata),
                model_name,
                to_request_fn,
                to_response_fn,
            )
            .await
            .map_err(to_napi_err)?;

            let (tx, rx) = tokio::sync::mpsc::channel(32);
            tokio::spawn(async move {
                let mut stream = rust_stream;
                while let Some(item) = stream.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
            });

            Ok(LlmStream {
                receiver: tokio::sync::Mutex::new(rx),
            })
        })
        .await
}

// ---------------------------------------------------------------------------
// Tool guardrail registrations
// ---------------------------------------------------------------------------

macro_rules! napi_guardrail_tool_api {
    ($(#[doc = $reg_doc:expr])* $register_name:ident,
     $(#[doc = $dereg_doc:expr])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[doc = $reg_doc])*
        #[napi]
        pub fn $register_name(
            name: String,
            priority: i32,
            guardrail: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
        ) -> Result<()> {
            $core_register(&name, priority, $wrapper(guardrail)).map_err(to_napi_err)
        }

        $(#[doc = $dereg_doc])*
        #[napi]
        pub fn $deregister_name(name: String) -> Result<bool> {
            $core_deregister(&name).map_err(to_napi_err)
        }
    };
}

napi_guardrail_tool_api!(
    /// Register a guardrail that sanitizes tool request arguments before execution.
    ///
    /// The `guardrail` callback receives `(toolName, args)` and must return sanitized args.
    /// Higher `priority` values run first. Throws if a guardrail with the same `name` already exists.
    register_tool_sanitize_request_guardrail,
    /// Deregister a tool request sanitization guardrail by name.
    ///
    /// Returns `true` if a guardrail with that name was found and removed.
    deregister_tool_sanitize_request_guardrail,
    core::nvmagic_register_tool_sanitize_request_guardrail,
    core::nvmagic_deregister_tool_sanitize_request_guardrail,
    callable::wrap_js_tool_fn
);

napi_guardrail_tool_api!(
    /// Register a guardrail that sanitizes tool response data after execution.
    ///
    /// The `guardrail` callback receives `(toolName, result)` and must return sanitized result.
    /// Higher `priority` values run first. Throws if a guardrail with the same `name` already exists.
    register_tool_sanitize_response_guardrail,
    /// Deregister a tool response sanitization guardrail by name.
    ///
    /// Returns `true` if a guardrail with that name was found and removed.
    deregister_tool_sanitize_response_guardrail,
    core::nvmagic_register_tool_sanitize_response_guardrail,
    core::nvmagic_deregister_tool_sanitize_response_guardrail,
    callable::wrap_js_tool_fn
);

/// Register a guardrail that conditionally gates tool execution.
///
/// The `guardrail` callback receives `(toolName, args)` and must return `null` to allow
/// execution or a rejection reason string to block it. Higher `priority` values run first.
#[napi]
pub fn register_tool_conditional_execution_guardrail(
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_tool_conditional_execution_guardrail(
        &name,
        priority,
        callable::wrap_js_tool_conditional_fn(guardrail),
    )
    .map_err(to_napi_err)
}

/// Deregister a tool conditional execution guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed.
#[napi]
pub fn deregister_tool_conditional_execution_guardrail(name: String) -> Result<bool> {
    core::nvmagic_deregister_tool_conditional_execution_guardrail(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Tool intercept registrations
// ---------------------------------------------------------------------------

macro_rules! napi_intercept_tool_api {
    ($(#[doc = $reg_doc:expr])* $register_name:ident,
     $(#[doc = $dereg_doc:expr])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[doc = $reg_doc])*
        #[napi]
        pub fn $register_name(
            name: String,
            priority: i32,
            break_chain: bool,
            callable: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
        ) -> Result<()> {
            $core_register(&name, priority, break_chain, $wrapper(callable)).map_err(to_napi_err)
        }

        $(#[doc = $dereg_doc])*
        #[napi]
        pub fn $deregister_name(name: String) -> Result<bool> {
            $core_deregister(&name).map_err(to_napi_err)
        }
    };
}

napi_intercept_tool_api!(
    /// Register an intercept that transforms tool request arguments.
    ///
    /// The `callable` receives `(toolName, args)` and returns transformed args. If `breakChain`
    /// is `true`, no lower-priority intercepts run after this one. Higher `priority` values run first.
    register_tool_request_intercept,
    /// Deregister a tool request intercept by name.
    ///
    /// Returns `true` if an intercept with that name was found and removed.
    deregister_tool_request_intercept,
    core::nvmagic_register_tool_request_intercept,
    core::nvmagic_deregister_tool_request_intercept,
    callable::wrap_js_tool_fn
);

napi_intercept_tool_api!(
    /// Register an intercept that transforms tool response data.
    ///
    /// The `callable` receives `(toolName, result)` and returns transformed result. If `breakChain`
    /// is `true`, no lower-priority intercepts run after this one. Higher `priority` values run first.
    register_tool_response_intercept,
    /// Deregister a tool response intercept by name.
    ///
    /// Returns `true` if an intercept with that name was found and removed.
    deregister_tool_response_intercept,
    core::nvmagic_register_tool_response_intercept,
    core::nvmagic_deregister_tool_response_intercept,
    callable::wrap_js_tool_fn
);

/// Register a tool execution intercept following the middleware chain pattern.
///
/// The `conditional` callback receives `(toolName, args)` and returns `true` if this intercept
/// should handle execution. If it matches, `callable` receives the args and a `next` function.
/// Call `next(args)` to invoke the next intercept or original implementation; skip calling
/// `next` to short-circuit the chain.
#[napi]
pub fn register_tool_execution_intercept(
    name: String,
    priority: i32,
    conditional: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_tool_execution_intercept(
        &name,
        priority,
        callable::wrap_js_tool_exec_conditional_fn(conditional),
        callable::wrap_js_tool_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister a tool execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_tool_execution_intercept(name: String) -> Result<bool> {
    core::nvmagic_deregister_tool_execution_intercept(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register a guardrail that sanitizes LLM request data before execution.
///
/// The `guardrail` callback receives the LLM request as JSON and must return the sanitized request.
/// Higher `priority` values run first. Throws if a guardrail with the same `name` already exists.
#[napi]
pub fn register_llm_sanitize_request_guardrail(
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_llm_sanitize_request_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_sanitize_request_fn(guardrail),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM request sanitization guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed.
#[napi]
pub fn deregister_llm_sanitize_request_guardrail(name: String) -> Result<bool> {
    core::nvmagic_deregister_llm_sanitize_request_guardrail(&name).map_err(to_napi_err)
}

/// Register a guardrail that sanitizes LLM response data after execution.
///
/// The `guardrail` callback receives the LLM response as a `JsLLMResponse` and must return
/// the sanitized response. Higher `priority` values run first. Throws if a guardrail with
/// the same `name` already exists.
#[napi]
pub fn register_llm_sanitize_response_guardrail(
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_llm_sanitize_response_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_response_fn(guardrail),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM response sanitization guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed.
#[napi]
pub fn deregister_llm_sanitize_response_guardrail(name: String) -> Result<bool> {
    core::nvmagic_deregister_llm_sanitize_response_guardrail(&name).map_err(to_napi_err)
}

/// Register a guardrail that conditionally gates LLM execution.
///
/// The `guardrail` callback receives the LLM request as JSON and must return `null` to allow
/// execution or a rejection reason string to block it. Higher `priority` values run first.
#[napi]
pub fn register_llm_conditional_execution_guardrail(
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_llm_conditional_execution_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_conditional_fn(guardrail),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM conditional execution guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed.
#[napi]
pub fn deregister_llm_conditional_execution_guardrail(name: String) -> Result<bool> {
    core::nvmagic_deregister_llm_conditional_execution_guardrail(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an intercept that transforms LLM request data.
///
/// The `callable` receives the native LLM request as JSON and returns a transformed request.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
#[napi]
pub fn register_llm_request_intercept(
    name: String,
    priority: i32,
    break_chain: bool,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_llm_request_intercept(
        &name,
        priority,
        break_chain,
        callable::wrap_js_json_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM request intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_request_intercept(name: String) -> Result<bool> {
    core::nvmagic_deregister_llm_request_intercept(&name).map_err(to_napi_err)
}

/// Register an intercept that transforms LLM response data.
///
/// The `callable` receives the LLM response as a `JsLLMResponse` and returns a transformed response.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
#[napi]
pub fn register_llm_response_intercept(
    name: String,
    priority: i32,
    break_chain: bool,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_llm_response_intercept(
        &name,
        priority,
        break_chain,
        callable::wrap_js_llm_response_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM response intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_response_intercept(name: String) -> Result<bool> {
    core::nvmagic_deregister_llm_response_intercept(&name).map_err(to_napi_err)
}

/// Register an intercept that transforms individual chunks in a streaming LLM response.
///
/// The `callable` receives each chunk as a JSON value and returns the transformed chunk.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
#[napi]
pub fn register_llm_stream_response_intercept(
    name: String,
    priority: i32,
    break_chain: bool,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_llm_stream_response_intercept(
        &name,
        priority,
        break_chain,
        callable::wrap_js_json_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM stream response intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_stream_response_intercept(name: String) -> Result<bool> {
    core::nvmagic_deregister_llm_stream_response_intercept(&name).map_err(to_napi_err)
}

/// Register an LLM execution intercept following the middleware chain pattern.
///
/// The `conditional` callback receives the LLM request as JSON and returns `true` if this
/// intercept should handle execution. If it matches, `callable` receives the request and a
/// `next` function. Call `next(request)` to invoke the next intercept or original
/// implementation; skip calling `next` to short-circuit the chain.
#[napi]
pub fn register_llm_execution_intercept(
    name: String,
    priority: i32,
    conditional: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_llm_execution_intercept(
        &name,
        priority,
        callable::wrap_js_llm_exec_conditional_fn(conditional),
        callable::wrap_js_llm_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_execution_intercept(name: String) -> Result<bool> {
    core::nvmagic_deregister_llm_execution_intercept(&name).map_err(to_napi_err)
}

/// Register a streaming LLM execution intercept following the middleware chain pattern.
///
/// The `conditional` callback receives the LLM request as JSON and returns `true` if this
/// intercept should handle execution. If it matches, `callable` receives the request and a
/// `next` function. Call `next(request)` to invoke the next intercept or original streaming
/// implementation; skip calling `next` to short-circuit the chain.
#[napi]
pub fn register_llm_stream_execution_intercept(
    name: String,
    priority: i32,
    conditional: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_llm_stream_execution_intercept(
        &name,
        priority,
        callable::wrap_js_llm_exec_conditional_fn(conditional),
        callable::wrap_js_llm_stream_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM stream execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_stream_execution_intercept(name: String) -> Result<bool> {
    core::nvmagic_deregister_llm_stream_execution_intercept(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Register a named event subscriber that receives all lifecycle events.
///
/// The `callback` receives each event as a JSON-serialized `JsEvent` object. Events are
/// delivered asynchronously and non-blocking. Throws if a subscriber with the same `name`
/// already exists.
#[napi]
pub fn register_subscriber(
    name: String,
    callback: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nvmagic_register_subscriber(&name, callable::wrap_js_event_subscriber(callback))
        .map_err(to_napi_err)
}

/// Deregister an event subscriber by name.
///
/// Returns `true` if a subscriber with that name was found and removed.
#[napi]
pub fn deregister_subscriber(name: String) -> Result<bool> {
    core::nvmagic_deregister_subscriber(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Run the registered tool request intercept chain on the given arguments.
/// Returns the transformed arguments.
#[napi]
pub fn tool_request_intercepts(name: String, args: Json) -> Result<Json> {
    core::nvmagic_tool_request_intercepts(&name, args).map_err(to_napi_err)
}

/// Run the registered tool conditional execution guardrail chain.
/// Throws if any guardrail rejects.
#[napi]
pub fn tool_conditional_execution(name: String, args: Json) -> Result<()> {
    core::nvmagic_tool_conditional_execution(&name, &args).map_err(to_napi_err)
}

/// Run the registered tool response intercept chain on the given result.
/// Returns the transformed result.
#[napi]
pub fn tool_response_intercepts(name: String, result: Json) -> Result<Json> {
    core::nvmagic_tool_response_intercepts(&name, result).map_err(to_napi_err)
}

/// Run the registered LLM request intercept chain on the given native request.
/// Returns the transformed native request as JSON.
#[napi]
pub fn llm_request_intercepts(native: Json) -> Result<Json> {
    core::nvmagic_llm_request_intercepts(native).map_err(to_napi_err)
}

/// Run the registered LLM conditional execution guardrail chain.
/// Throws if any guardrail rejects. Optional `toRequest` converts native JSON to a formal
/// `JsLLMRequest` for guardrails; defaults to the identity converter.
#[napi]
pub fn llm_conditional_execution(
    native: Json,
    to_request: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
) -> Result<()> {
    let to_request_fn = to_request.map(wrap_js_to_request);
    core::nvmagic_llm_conditional_execution(&native, to_request_fn.as_ref()).map_err(to_napi_err)
}

/// Run the registered LLM response intercept chain on the given response.
/// Returns the transformed response.
#[napi]
pub fn llm_response_intercepts(response: &JsLLMResponse) -> Result<JsLLMResponse> {
    let core_response = response.inner.clone();
    let result = core::nvmagic_llm_response_intercepts(core_response).map_err(to_napi_err)?;
    Ok(JsLLMResponse { inner: result })
}

// ---------------------------------------------------------------------------
// ATIF Exporter
// ---------------------------------------------------------------------------

/// An ATIF (Agent Trajectory Interchange Format) exporter that collects lifecycle events
/// and exports them as a structured trajectory.
///
/// Create an instance with session and agent metadata, then register it as an event subscriber.
/// When ready, call `exportJson()` to serialize the collected trajectory.
#[napi]
pub struct JsAtifExporter {
    inner: nvmagic_core::atif::AtifExporter,
}

#[napi]
impl JsAtifExporter {
    /// Create a new ATIF exporter.
    ///
    /// `sessionId` identifies the session. `agentName` and `agentVersion` describe the agent.
    /// Optional `modelName` records the LLM model used.
    #[napi(constructor)]
    pub fn new(
        session_id: String,
        agent_name: String,
        agent_version: String,
        model_name: Option<String>,
    ) -> napi::Result<Self> {
        let agent_info = nvmagic_core::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: None,
            extra: None,
        };
        Ok(Self {
            inner: nvmagic_core::atif::AtifExporter::new(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    ///
    /// Throws if a subscriber with the same `name` already exists.
    #[napi]
    pub fn register(&self, name: String) -> napi::Result<()> {
        let subscriber = self.inner.subscriber();
        nvmagic_core::nvmagic_register_subscriber(&name, subscriber)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Deregister this exporter's event subscriber by name.
    ///
    /// Returns `true` if a subscriber with that name was found and removed.
    #[napi]
    pub fn deregister(&self, name: String) -> napi::Result<bool> {
        nvmagic_core::nvmagic_deregister_subscriber(&name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Export the collected trajectory as a JSON string.
    ///
    /// If `rootUuid` is provided, only the subtree rooted at that scope is exported.
    /// Returns a JSON-serialized `AtifTrajectory`.
    #[napi]
    pub fn export_json(&self, root_uuid: Option<String>) -> napi::Result<String> {
        let root = root_uuid
            .map(|s| uuid::Uuid::parse_str(&s))
            .transpose()
            .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
        let trajectory = self.inner.export(root);
        serde_json::to_string(&trajectory).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Clear all collected events from the exporter.
    #[napi]
    pub fn clear(&self) {
        self.inner.clear();
    }
}

// ---------------------------------------------------------------------------
// LLM converter helpers
// ---------------------------------------------------------------------------

use std::sync::Arc;

/// Wraps a JS `toRequest` callback into a boxed `ToRequestFn`.
fn wrap_js_to_request(
    tsfn: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> core_types::ToRequestFn {
    let tsfn = Arc::new(tsfn);
    Box::new(move |native: &Json| {
        let func = tsfn.clone();
        let native_clone = native.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            native_clone,
            napi::threadsafe_function::ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        let result = rx.recv().unwrap_or(Json::Null);
        serde_json::from_value(result).unwrap_or_else(|_| core_types::LLMRequest {
            headers: serde_json::Map::new(),
            content: native.clone(),
        })
    })
}

/// Wraps a JS `toResponse` callback into a boxed `ToResponseFn`.
fn wrap_js_to_response(
    tsfn: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> core_types::ToResponseFn {
    let tsfn = Arc::new(tsfn);
    Box::new(move |native: &Json| {
        let func = tsfn.clone();
        let native_clone = native.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            native_clone,
            napi::threadsafe_function::ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        let result = rx.recv().unwrap_or(Json::Null);
        serde_json::from_value(result).unwrap_or_else(|_| core_types::LLMResponse {
            data: native.clone(),
        })
    })
}
