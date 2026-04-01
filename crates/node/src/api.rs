// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public NAPI API functions for the Nexus Node.js bindings.
//!
//! This module exposes the full agent runtime API to JavaScript/TypeScript:
//! scope stack management, tool and LLM lifecycle operations, guardrail and
//! intercept registration/deregistration, and event subscriber management.
//! All functions are annotated with `#[napi]` and their doc comments appear
//! in the generated `index.d.ts` TypeScript definitions.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex as StdMutex;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::JsObject;
use napi_derive::napi;
use serde_json::Value as Json;
use tokio_stream::StreamExt;

use nvidia_nat_nexus_core as core;
use nvidia_nat_nexus_core::types as core_types;

use crate::callable;
use crate::convert::{opt_json, to_napi_err};
use crate::stream::LlmStream;
use crate::types::*;

// ---------------------------------------------------------------------------
// Stream channel registry — enables JS async generators to push chunks to Rust
// ---------------------------------------------------------------------------

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(0);

type StreamSender = tokio::sync::mpsc::UnboundedSender<nvidia_nat_nexus_core::Result<Json>>;

static STREAM_CHANNELS: std::sync::LazyLock<StdMutex<HashMap<u64, StreamSender>>> =
    std::sync::LazyLock::new(|| StdMutex::new(HashMap::new()));

fn register_stream_channel(id: u64, tx: StreamSender) {
    STREAM_CHANNELS.lock().unwrap().insert(id, tx);
}

/// Push a chunk into the stream identified by `streamId`.
/// Called from JavaScript during async generator iteration.
#[napi]
pub fn push_stream_chunk(stream_id: f64, chunk: Json) -> bool {
    let id = stream_id as u64;
    if let Some(tx) = STREAM_CHANNELS.lock().unwrap().get(&id) {
        tx.send(Ok(chunk)).is_ok()
    } else {
        false
    }
}

/// Signal that a stream is complete. Drops the sender so the Rust
/// receiver sees the channel as closed.
#[napi]
pub fn end_stream(stream_id: f64) {
    let id = stream_id as u64;
    STREAM_CHANNELS.lock().unwrap().remove(&id);
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Creates a new isolated scope stack.
#[napi]
pub fn create_scope_stack() -> JsScopeStack {
    JsScopeStack {
        inner: nvidia_nat_nexus_core::create_scope_stack(),
    }
}

/// Returns the current execution context's scope stack handle.
#[napi]
pub fn current_scope_stack() -> JsScopeStack {
    JsScopeStack {
        inner: nvidia_nat_nexus_core::current_scope_stack(),
    }
}

/// Binds a scope stack to the current thread.
#[napi]
pub fn set_thread_scope_stack(stack: &JsScopeStack) {
    nvidia_nat_nexus_core::set_thread_scope_stack(stack.inner.clone());
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `setThreadScopeStack` has been called on the current
/// thread, or the caller is inside a task-local scope. Returns `false` when
/// only the auto-created default is present.
#[napi]
pub fn scope_stack_active() -> bool {
    nvidia_nat_nexus_core::scope_stack_active()
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
    core::nat_nexus_get_handle()
        .map(JsScopeHandle::from)
        .map_err(to_napi_err)
}

/// Push a new execution scope onto the scope stack.
///
/// Creates a child scope with the given `name` and `scopeType`. If `handle` is provided,
/// the new scope is parented to that scope; otherwise it is parented to the current top scope.
/// Optional `attributes` is a bitfield of scope attribute flags (e.g., `SCOPE_ATTR_PARALLEL`).
/// Optional `data` is a JSON application data payload attached to the scope.
/// Optional `metadata` is a JSON metadata payload attached to the scope.
/// Returns the handle for the newly created scope.
#[napi]
pub fn push_scope(
    name: String,
    scope_type: ScopeType,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<JsScopeHandle> {
    let attrs = core_types::ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    core::nat_nexus_push_scope(
        &name,
        scope_type.into(),
        handle.map(|h| &h.inner),
        attrs,
        opt_json(data),
        opt_json(metadata),
    )
    .map(JsScopeHandle::from)
    .map_err(to_napi_err)
}

/// Pop an execution scope from the scope stack.
///
/// Removes the scope identified by `handle` from the stack and emits an end event.
/// Throws if the handle does not match the current top scope.
#[napi]
pub fn pop_scope(handle: &JsScopeHandle) -> Result<()> {
    core::nat_nexus_pop_scope(&handle.inner.uuid).map_err(to_napi_err)
}

/// Push a scope, run a callback, then pop the scope automatically.
///
/// Creates a child scope with the given `name` and `scopeType`, invokes the
/// `callback` with the new scope handle, and guarantees that the scope is popped
/// when the callback completes (whether it returns normally, throws, or returns a
/// rejected Promise). Supports both synchronous and async (Promise-returning)
/// callbacks.
///
/// Optional `handle` sets the parent scope; `attributes` is a bitfield of scope
/// attribute flags; `data` and `metadata` are JSON payloads attached to the scope.
///
/// Returns a Promise that resolves with the callback's return value.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>")]
pub fn with_scope(
    env: Env,
    name: String,
    scope_type: ScopeType,
    callback: napi::JsFunction,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<JsObject> {
    let attrs = core_types::ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let scope_handle = core::nat_nexus_push_scope(
        &name,
        scope_type.into(),
        handle.map(|h| &h.inner),
        attrs,
        opt_json(data),
        opt_json(metadata),
    )
    .map(JsScopeHandle::from)
    .map_err(to_napi_err)?;

    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();
    let scope_uuid = scope_handle.inner.uuid;
    let scope_name = scope_handle.inner.name.clone();
    let scope_type_int: u32 = ScopeType::from(scope_handle.inner.scope_type) as u32;
    let scope_attrs = scope_handle.inner.attributes.bits();
    let scope_parent_uuid = scope_handle.inner.parent_uuid.map(|u| u.to_string());

    // Create a promise-aware wrapper so we handle both sync and async callbacks.
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callback).map_err(|e| {
            // Pop scope before propagating error
            let _ = core::nat_nexus_pop_scope(&scope_uuid);
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );

    env.execute_tokio_future(
        async move {
            nvidia_nat_nexus_core::TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    let handle_json = serde_json::json!({
                        "uuid": scope_uuid.to_string(),
                        "name": scope_name,
                        "scopeType": scope_type_int,
                        "attributes": scope_attrs,
                        "parentUuid": scope_parent_uuid,
                    });

                    let result = pa_fn.call(handle_json).await;
                    // Always pop the scope, even on error
                    let _ = core::nat_nexus_pop_scope(&scope_uuid);
                    result.map_err(to_napi_err)
                })
                .await
        },
        |_env, result| Ok(result),
    )
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
    core::nat_nexus_event(
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
    core::nat_nexus_tool_call(
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
    core::nat_nexus_tool_call_end(&handle.inner, result, opt_json(data), opt_json(metadata))
        .map_err(to_napi_err)
}

/// Execute a tool call end-to-end with full lifecycle management.
///
/// Runs conditional-execution guardrails (on raw args) → request intercepts →
/// sanitize-request guardrails → execution intercepts → `func` → response
/// intercepts → sanitize-response guardrails. On rejection, only a standalone
/// Mark event is emitted (no Start/End pair) and `GuardrailRejected` is returned.
/// Returns the final (possibly intercepted) tool result.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>")]
pub fn tool_call_execute(
    env: Env,
    name: String,
    args: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<JsObject> {
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(core::task_scope_top);
    let exec_fn = callable::wrap_js_tool_exec_fn(func);
    let default_fn: nvidia_nat_nexus_core::ToolExecutionNextFn =
        Box::new(move |args| exec_fn(args));
    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();

    env.execute_tokio_future(
        async move {
            nvidia_nat_nexus_core::TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core::nat_nexus_tool_call_execute(
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
        },
        |_env, result| Ok(result),
    )
}

/// Execute a tool call end-to-end, supporting both sync and async (Promise-returning) callbacks.
///
/// Same lifecycle as `toolCallExecute` (guardrails → intercepts → func → response processing),
/// but transparently handles JS callbacks that return Promises. Uses `napi_is_promise` to detect
/// Promise return values and resolves them before continuing the pipeline.
///
/// Accepts a raw `JsFunction` instead of `ThreadsafeFunction` so it can create a
/// promise-aware wrapper with access to `Env`.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>")]
pub fn tool_call_execute_async(
    env: Env,
    name: String,
    args: Json,
    func: JsFunction,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<JsObject> {
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(core::task_scope_top);
    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();

    // Create promise-aware wrapper — this must happen on the JS thread (we have Env).
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &func).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );

    let exec_fn: nvidia_nat_nexus_core::ToolExecutionNextFn = Box::new(move |args| {
        let pa_fn = pa_fn.clone();
        Box::pin(async move { pa_fn.call(args).await })
    });

    env.execute_tokio_future(
        async move {
            nvidia_nat_nexus_core::TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core::nat_nexus_tool_call_execute(
                        &name,
                        args,
                        exec_fn,
                        Some(parent),
                        attrs,
                        opt_json(data),
                        opt_json(metadata),
                    )
                    .await
                    .map_err(to_napi_err)
                })
                .await
        },
        |_env, result| Ok(result),
    )
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begin an LLM call, running request guardrails and intercepts.
///
/// Registers an LLM invocation with the given provider `name` and request payload.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LLMRequest` schema. Returns a `JsLLMHandle` that must be passed to `llmCallEnd()`
/// when the response is received. Optional `handle` specifies the parent scope; `attributes`
/// is a bitfield (e.g., `LLM_ATTR_STREAMING`).
#[allow(clippy::too_many_arguments)]
#[napi]
pub fn llm_call(
    name: String,
    request: Json,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
) -> Result<JsLLMHandle> {
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let llm_request: core_types::LLMRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
    core::nat_nexus_llm_call(
        &name,
        &llm_request,
        handle.map(|h| &h.inner),
        attrs,
        opt_json(data),
        opt_json(metadata),
        model_name,
    )
    .map(JsLLMHandle::from)
    .map_err(to_napi_err)
}

/// End an LLM call, running response guardrails and intercepts.
///
/// Signals that the LLM call identified by `handle` has completed with the given `response`.
/// Response guardrails and intercepts are applied to the response.
#[napi]
pub fn llm_call_end(
    handle: &JsLLMHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<()> {
    core::nat_nexus_llm_call_end(&handle.inner, response, opt_json(data), opt_json(metadata))
        .map_err(to_napi_err)
}

/// Execute an LLM call end-to-end with full lifecycle management.
///
/// Runs conditional-execution guardrails (on raw request) → request intercepts →
/// sanitize-request guardrails → execution intercepts → `func` → response
/// intercepts → sanitize-response guardrails. On rejection, only a standalone
/// Mark event is emitted (no Start/End pair) and `GuardrailRejected` is returned.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LLMRequest` schema. Returns the final (possibly intercepted) LLM response.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>")]
pub fn llm_call_execute(
    env: Env,
    name: String,
    request: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
) -> Result<JsObject> {
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(core::task_scope_top);
    let llm_request: core_types::LLMRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
    let exec_fn = callable::wrap_js_llm_exec_fn(func);
    let default_fn: nvidia_nat_nexus_core::LlmExecutionNextFn = Box::new(move |req| exec_fn(req));
    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();

    env.execute_tokio_future(
        async move {
            nvidia_nat_nexus_core::TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core::nat_nexus_llm_call_execute(
                        &name,
                        llm_request,
                        default_fn,
                        Some(parent),
                        attrs,
                        opt_json(data),
                        opt_json(metadata),
                        model_name,
                    )
                    .await
                    .map_err(to_napi_err)
                })
                .await
        },
        |_env, result| Ok(result),
    )
}

/// Execute a streaming LLM call end-to-end with full lifecycle management.
///
/// Like `llmCallExecute`, conditional-execution guardrails run first on the raw request.
/// Returns an `LlmStream` whose `next()` method yields response chunks incrementally.
/// The `func` callback receives the request as JSON and its response is streamed back.
/// Stream-level intercepts are applied to each chunk.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LLMRequest` schema.
///
/// The optional `collector` callback is invoked with each intercepted chunk as JSON,
/// allowing the caller to accumulate chunks for aggregation. The optional `finalizer`
/// callback is invoked once when the stream is exhausted and must return a JSON value
/// representing the aggregated response.
#[allow(clippy::too_many_arguments)]
#[napi]
pub async fn llm_stream_call_execute(
    name: String,
    request: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    collector: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    finalizer: Option<ThreadsafeFunction<(), ErrorStrategy::Fatal>>,
    handle: Option<&JsScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
) -> Result<LlmStream> {
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(core::task_scope_top);
    let llm_request: core_types::LLMRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;

    let wrapped_collector: Box<dyn FnMut(Json) -> nvidia_nat_nexus_core::Result<()> + Send> =
        match collector {
            Some(cb) => callable::wrap_js_collector_fn(cb),
            None => Box::new(|_: Json| Ok(())),
        };

    let wrapped_finalizer: Box<dyn FnOnce() -> Json + Send> = match finalizer {
        Some(cb) => callable::wrap_js_finalizer_fn(cb),
        None => Box::new(|| Json::Null),
    };

    // Push-based stream bridge: JS iterates the async generator on the
    // event loop and pushes each chunk into Rust via `pushStreamChunk`.
    // We create an unbounded channel here and pass the stream ID to JS
    // so it knows where to send chunks.
    let func = std::sync::Arc::new(func);
    let default_fn: nvidia_nat_nexus_core::LlmStreamExecutionNextFn =
        Box::new(move |req: core_types::LLMRequest| {
            let stream_id = NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed);
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            register_stream_channel(stream_id, tx);

            // Serialize the LLMRequest to JSON and wrap with streamId so JS can extract both
            let req_json = serde_json::to_value(&req).unwrap_or(Json::Null);
            let wrapper = serde_json::json!({
                "__nat_nexus_native": req_json,
                "__nat_nexus_stream_id": stream_id,
            });

            // NonBlocking: queue the call on the JS event loop and return immediately.
            // The JS function starts async iteration and pushes chunks via pushStreamChunk.
            func.call(wrapper, ThreadsafeFunctionCallMode::NonBlocking);

            Box::pin(async move {
                let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
                Ok(Box::pin(stream)
                    as std::pin::Pin<
                        Box<
                            dyn tokio_stream::Stream<Item = nvidia_nat_nexus_core::Result<Json>>
                                + Send,
                        >,
                    >)
            })
        });

    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();

    nvidia_nat_nexus_core::TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            let rust_stream = core::nat_nexus_llm_stream_call_execute(
                &name,
                llm_request,
                default_fn,
                wrapped_collector,
                wrapped_finalizer,
                Some(parent),
                attrs,
                opt_json(data),
                opt_json(metadata),
                model_name,
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
    core::nat_nexus_register_tool_sanitize_request_guardrail,
    core::nat_nexus_deregister_tool_sanitize_request_guardrail,
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
    core::nat_nexus_register_tool_sanitize_response_guardrail,
    core::nat_nexus_deregister_tool_sanitize_response_guardrail,
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
    core::nat_nexus_register_tool_conditional_execution_guardrail(
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
    core::nat_nexus_deregister_tool_conditional_execution_guardrail(&name).map_err(to_napi_err)
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
    core::nat_nexus_register_tool_request_intercept,
    core::nat_nexus_deregister_tool_request_intercept,
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
    core::nat_nexus_register_tool_response_intercept,
    core::nat_nexus_deregister_tool_response_intercept,
    callable::wrap_js_tool_fn
);

/// Register a tool execution intercept following the middleware chain pattern.
///
/// The `callable` receives the args and a `next` function. Call `next(args)` to invoke
/// the next intercept or original implementation; skip calling `next` to short-circuit
/// the chain.
#[napi]
pub fn register_tool_execution_intercept(
    name: String,
    priority: i32,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nat_nexus_register_tool_execution_intercept(
        &name,
        priority,
        callable::wrap_js_tool_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister a tool execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_tool_execution_intercept(name: String) -> Result<bool> {
    core::nat_nexus_deregister_tool_execution_intercept(&name).map_err(to_napi_err)
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
    core::nat_nexus_register_llm_sanitize_request_guardrail(
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
    core::nat_nexus_deregister_llm_sanitize_request_guardrail(&name).map_err(to_napi_err)
}

/// Register a guardrail that sanitizes LLM response data after execution.
///
/// The `guardrail` callback receives the LLM response as a JSON value and must return
/// the sanitized response as JSON. Higher `priority` values run first. Throws if a guardrail
/// with the same `name` already exists.
#[napi]
pub fn register_llm_sanitize_response_guardrail(
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nat_nexus_register_llm_sanitize_response_guardrail(
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
    core::nat_nexus_deregister_llm_sanitize_response_guardrail(&name).map_err(to_napi_err)
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
    core::nat_nexus_register_llm_conditional_execution_guardrail(
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
    core::nat_nexus_deregister_llm_conditional_execution_guardrail(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an intercept that transforms LLM request data.
///
/// The `callable` receives the `LLMRequest` (as JSON) and returns a transformed request.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
#[napi]
pub fn register_llm_request_intercept(
    name: String,
    priority: i32,
    break_chain: bool,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nat_nexus_register_llm_request_intercept(
        &name,
        priority,
        break_chain,
        callable::wrap_js_llm_request_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM request intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_request_intercept(name: String) -> Result<bool> {
    core::nat_nexus_deregister_llm_request_intercept(&name).map_err(to_napi_err)
}

/// Register an LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original implementation; skip calling `next` to
/// short-circuit the chain.
#[napi]
pub fn register_llm_execution_intercept(
    name: String,
    priority: i32,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nat_nexus_register_llm_execution_intercept(
        &name,
        priority,
        callable::wrap_js_llm_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_execution_intercept(name: String) -> Result<bool> {
    core::nat_nexus_deregister_llm_execution_intercept(&name).map_err(to_napi_err)
}

/// Register a streaming LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original streaming implementation; skip calling `next`
/// to short-circuit the chain.
#[napi]
pub fn register_llm_stream_execution_intercept(
    name: String,
    priority: i32,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core::nat_nexus_register_llm_stream_execution_intercept(
        &name,
        priority,
        callable::wrap_js_llm_stream_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister an LLM stream execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_stream_execution_intercept(name: String) -> Result<bool> {
    core::nat_nexus_deregister_llm_stream_execution_intercept(&name).map_err(to_napi_err)
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
    core::nat_nexus_register_subscriber(&name, callable::wrap_js_event_subscriber(callback))
        .map_err(to_napi_err)
}

/// Deregister an event subscriber by name.
///
/// Returns `true` if a subscriber with that name was found and removed.
#[napi]
pub fn deregister_subscriber(name: String) -> Result<bool> {
    core::nat_nexus_deregister_subscriber(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations — Tool
// ---------------------------------------------------------------------------

macro_rules! napi_scope_guardrail_tool_api {
    ($(#[doc = $reg_doc:expr])* $register_name:ident,
     $(#[doc = $dereg_doc:expr])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[doc = $reg_doc])*
        #[napi]
        pub fn $register_name(
            scope_uuid: String,
            name: String,
            priority: i32,
            guardrail: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
        ) -> Result<()> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
            $core_register(&uuid, &name, priority, $wrapper(guardrail)).map_err(to_napi_err)
        }

        $(#[doc = $dereg_doc])*
        #[napi]
        pub fn $deregister_name(scope_uuid: String, name: String) -> Result<bool> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
            $core_deregister(&uuid, &name).map_err(to_napi_err)
        }
    };
}

napi_scope_guardrail_tool_api!(
    /// Register a scope-local guardrail that sanitizes tool request arguments before execution.
    ///
    /// The `guardrail` callback receives `(toolName, args)` and must return sanitized args.
    /// Higher `priority` values run first. Throws if a guardrail with the same `name` already exists
    /// on the specified scope.
    scope_register_tool_sanitize_request_guardrail,
    /// Deregister a scope-local tool request sanitization guardrail by name.
    ///
    /// Returns `true` if a guardrail with that name was found and removed from the specified scope.
    scope_deregister_tool_sanitize_request_guardrail,
    core::nat_nexus_scope_register_tool_sanitize_request_guardrail,
    core::nat_nexus_scope_deregister_tool_sanitize_request_guardrail,
    callable::wrap_js_tool_fn
);

napi_scope_guardrail_tool_api!(
    /// Register a scope-local guardrail that sanitizes tool response data after execution.
    ///
    /// The `guardrail` callback receives `(toolName, result)` and must return sanitized result.
    /// Higher `priority` values run first. Throws if a guardrail with the same `name` already exists
    /// on the specified scope.
    scope_register_tool_sanitize_response_guardrail,
    /// Deregister a scope-local tool response sanitization guardrail by name.
    ///
    /// Returns `true` if a guardrail with that name was found and removed from the specified scope.
    scope_deregister_tool_sanitize_response_guardrail,
    core::nat_nexus_scope_register_tool_sanitize_response_guardrail,
    core::nat_nexus_scope_deregister_tool_sanitize_response_guardrail,
    callable::wrap_js_tool_fn
);

/// Register a scope-local guardrail that conditionally gates tool execution.
///
/// The `guardrail` callback receives `(toolName, args)` and must return `null` to allow
/// execution or a rejection reason string to block it. Higher `priority` values run first.
#[napi]
pub fn scope_register_tool_conditional_execution_guardrail(
    scope_uuid: String,
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_tool_conditional_execution_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_tool_conditional_fn(guardrail),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local tool conditional execution guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_tool_conditional_execution_guardrail(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_tool_conditional_execution_guardrail(&uuid, &name)
        .map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — Tool
// ---------------------------------------------------------------------------

macro_rules! napi_scope_intercept_tool_api {
    ($(#[doc = $reg_doc:expr])* $register_name:ident,
     $(#[doc = $dereg_doc:expr])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[doc = $reg_doc])*
        #[napi]
        pub fn $register_name(
            scope_uuid: String,
            name: String,
            priority: i32,
            break_chain: bool,
            callable: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
        ) -> Result<()> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
            $core_register(&uuid, &name, priority, break_chain, $wrapper(callable))
                .map_err(to_napi_err)
        }

        $(#[doc = $dereg_doc])*
        #[napi]
        pub fn $deregister_name(scope_uuid: String, name: String) -> Result<bool> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
            $core_deregister(&uuid, &name).map_err(to_napi_err)
        }
    };
}

napi_scope_intercept_tool_api!(
    /// Register a scope-local intercept that transforms tool request arguments.
    ///
    /// The `callable` receives `(toolName, args)` and returns transformed args. If `breakChain`
    /// is `true`, no lower-priority intercepts run after this one. Higher `priority` values run first.
    scope_register_tool_request_intercept,
    /// Deregister a scope-local tool request intercept by name.
    ///
    /// Returns `true` if an intercept with that name was found and removed from the specified scope.
    scope_deregister_tool_request_intercept,
    core::nat_nexus_scope_register_tool_request_intercept,
    core::nat_nexus_scope_deregister_tool_request_intercept,
    callable::wrap_js_tool_fn
);

napi_scope_intercept_tool_api!(
    /// Register a scope-local intercept that transforms tool response data.
    ///
    /// The `callable` receives `(toolName, result)` and returns transformed result. If `breakChain`
    /// is `true`, no lower-priority intercepts run after this one. Higher `priority` values run first.
    scope_register_tool_response_intercept,
    /// Deregister a scope-local tool response intercept by name.
    ///
    /// Returns `true` if an intercept with that name was found and removed from the specified scope.
    scope_deregister_tool_response_intercept,
    core::nat_nexus_scope_register_tool_response_intercept,
    core::nat_nexus_scope_deregister_tool_response_intercept,
    callable::wrap_js_tool_fn
);

/// Register a scope-local tool execution intercept following the middleware chain pattern.
///
/// The `callable` receives the args and a `next` function. Call `next(args)` to invoke
/// the next intercept or original implementation; skip calling `next` to short-circuit
/// the chain.
#[napi]
pub fn scope_register_tool_execution_intercept(
    scope_uuid: String,
    name: String,
    priority: i32,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_tool_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_tool_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local tool execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_tool_execution_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_tool_execution_intercept(&uuid, &name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations — LLM
// ---------------------------------------------------------------------------

/// Register a scope-local guardrail that sanitizes LLM request data before execution.
///
/// The `guardrail` callback receives the LLM request as JSON and must return the sanitized request.
/// Higher `priority` values run first. Throws if a guardrail with the same `name` already exists
/// on the specified scope.
#[napi]
pub fn scope_register_llm_sanitize_request_guardrail(
    scope_uuid: String,
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_llm_sanitize_request_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_sanitize_request_fn(guardrail),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local LLM request sanitization guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_sanitize_request_guardrail(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_llm_sanitize_request_guardrail(&uuid, &name)
        .map_err(to_napi_err)
}

/// Register a scope-local guardrail that sanitizes LLM response data after execution.
///
/// The `guardrail` callback receives the LLM response as a JSON value and must return
/// the sanitized response as JSON. Higher `priority` values run first. Throws if a guardrail
/// with the same `name` already exists on the specified scope.
#[napi]
pub fn scope_register_llm_sanitize_response_guardrail(
    scope_uuid: String,
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_llm_sanitize_response_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_response_fn(guardrail),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local LLM response sanitization guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_sanitize_response_guardrail(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_llm_sanitize_response_guardrail(&uuid, &name)
        .map_err(to_napi_err)
}

/// Register a scope-local guardrail that conditionally gates LLM execution.
///
/// The `guardrail` callback receives the LLM request as JSON and must return `null` to allow
/// execution or a rejection reason string to block it. Higher `priority` values run first.
#[napi]
pub fn scope_register_llm_conditional_execution_guardrail(
    scope_uuid: String,
    name: String,
    priority: i32,
    guardrail: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_llm_conditional_execution_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_conditional_fn(guardrail),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local LLM conditional execution guardrail by name.
///
/// Returns `true` if a guardrail with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_conditional_execution_guardrail(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_llm_conditional_execution_guardrail(&uuid, &name)
        .map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — LLM
// ---------------------------------------------------------------------------

/// Register a scope-local intercept that transforms LLM request data.
///
/// The `callable` receives the `LLMRequest` (as JSON) and returns a transformed request.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
#[napi]
pub fn scope_register_llm_request_intercept(
    scope_uuid: String,
    name: String,
    priority: i32,
    break_chain: bool,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_llm_request_intercept(
        &uuid,
        &name,
        priority,
        break_chain,
        callable::wrap_js_llm_request_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local LLM request intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_request_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_llm_request_intercept(&uuid, &name).map_err(to_napi_err)
}

/// Register a scope-local LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original implementation; skip calling `next` to
/// short-circuit the chain.
#[napi]
pub fn scope_register_llm_execution_intercept(
    scope_uuid: String,
    name: String,
    priority: i32,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_llm_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_execution_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_llm_execution_intercept(&uuid, &name).map_err(to_napi_err)
}

/// Register a scope-local streaming LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original streaming implementation; skip calling `next`
/// to short-circuit the chain.
#[napi]
pub fn scope_register_llm_stream_execution_intercept(
    scope_uuid: String,
    name: String,
    priority: i32,
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_llm_stream_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_stream_exec_intercept_fn(callable),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local LLM stream execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_stream_execution_intercept(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_llm_stream_execution_intercept(&uuid, &name)
        .map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registrations
// ---------------------------------------------------------------------------

/// Register a scope-local named event subscriber that receives lifecycle events
/// for the specified scope.
///
/// The `callback` receives each event as a JSON-serialized `JsEvent` object. Events are
/// delivered asynchronously and non-blocking. Throws if a subscriber with the same `name`
/// already exists on the specified scope.
#[napi]
pub fn scope_register_subscriber(
    scope_uuid: String,
    name: String,
    callback: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_register_subscriber(
        &uuid,
        &name,
        callable::wrap_js_event_subscriber(callback),
    )
    .map_err(to_napi_err)
}

/// Deregister a scope-local event subscriber by name.
///
/// Returns `true` if a subscriber with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_subscriber(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core::nat_nexus_scope_deregister_subscriber(&uuid, &name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Run the registered tool request intercept chain on the given arguments.
/// Returns the transformed arguments.
#[napi]
pub fn tool_request_intercepts(name: String, args: Json) -> Result<Json> {
    core::nat_nexus_tool_request_intercepts(&name, args).map_err(to_napi_err)
}

/// Run the registered tool conditional execution guardrail chain.
/// Throws if any guardrail rejects.
#[napi]
pub fn tool_conditional_execution(name: String, args: Json) -> Result<()> {
    core::nat_nexus_tool_conditional_execution(&name, &args).map_err(to_napi_err)
}

/// Run the registered tool response intercept chain on the given result.
/// Returns the transformed result.
#[napi]
pub fn tool_response_intercepts(name: String, result: Json) -> Result<Json> {
    core::nat_nexus_tool_response_intercepts(&name, result).map_err(to_napi_err)
}

/// Run the registered LLM request intercept chain on the given request.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LLMRequest` schema. Returns the transformed request as JSON.
#[napi]
pub fn llm_request_intercepts(name: String, request: Json) -> Result<Json> {
    let llm_request: core_types::LLMRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
    core::nat_nexus_llm_request_intercepts(&name, llm_request)
        .map(|r| serde_json::to_value(&r).unwrap_or(Json::Null))
        .map_err(to_napi_err)
}

/// Run the registered LLM conditional execution guardrail chain.
/// Throws if any guardrail rejects. The `request` should be a JSON object with `headers`
/// and `content` fields matching the `LLMRequest` schema.
#[napi]
pub fn llm_conditional_execution(request: Json) -> Result<()> {
    let llm_request: core_types::LLMRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LLMRequest: {e}")))?;
    core::nat_nexus_llm_conditional_execution(&llm_request).map_err(to_napi_err)
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
    inner: nvidia_nat_nexus_core::atif::AtifExporter,
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
        let agent_info = nvidia_nat_nexus_core::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: None,
            extra: None,
        };
        Ok(Self {
            inner: nvidia_nat_nexus_core::atif::AtifExporter::new(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    ///
    /// Throws if a subscriber with the same `name` already exists.
    #[napi]
    pub fn register(&self, name: String) -> napi::Result<()> {
        let subscriber = self.inner.subscriber();
        nvidia_nat_nexus_core::nat_nexus_register_subscriber(&name, subscriber)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Deregister this exporter's event subscriber by name.
    ///
    /// Returns `true` if a subscriber with that name was found and removed.
    #[napi]
    pub fn deregister(&self, name: String) -> napi::Result<bool> {
        nvidia_nat_nexus_core::nat_nexus_deregister_subscriber(&name)
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
