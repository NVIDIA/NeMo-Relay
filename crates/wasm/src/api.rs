// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level NVAgentRT API functions exposed to JavaScript via `wasm_bindgen`.
//!
//! This module contains all public entry points for:
//!
//! - **Scope management** -- push/pop hierarchical execution scopes and emit
//!   custom events.
//! - **Tool lifecycle** -- begin, end, and execute tool calls with full
//!   middleware pipeline support (guardrails and intercepts).
//! - **LLM lifecycle** -- begin, end, and execute LLM calls with full
//!   middleware pipeline support.
//! - **Guardrail registration** -- register and deregister sanitize-request,
//!   sanitize-response, and conditional-execution guardrails for both tools
//!   and LLMs.
//! - **Intercept registration** -- register and deregister request, response,
//!   execution, and stream-response intercepts for both tools and LLMs.
//! - **Event subscribers** -- register and deregister lifecycle event
//!   subscribers.
//!
//! All functions use `JsValue` for JSON payloads and return `Result<T, JsValue>`
//! where errors are thrown as JavaScript exceptions.

use js_sys::Function;
use wasm_bindgen::prelude::*;

use nvagentrt_core::types as core_types;

use crate::callable;
use crate::convert::*;
use crate::stream::WasmLlmStream;
use crate::types::*;

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Returns the handle of the current (topmost) scope on the scope stack.
///
/// Throws if the scope stack is empty.
#[wasm_bindgen(js_name = "getHandle")]
pub fn nvagentrt_get_handle() -> Result<WasmScopeHandle, JsValue> {
    nvagentrt_core::nvagentrt_get_handle()
        .map(WasmScopeHandle::from)
        .map_err(to_js_err)
}

/// Pushes a new scope onto the scope stack and returns its handle.
///
/// - `name` - Human-readable scope name.
/// - `scope_type` - Integer scope type constant (e.g. `SCOPE_TYPE_AGENT`).
/// - `parent` - Optional parent scope handle; uses the current top if omitted.
/// - `attributes` - Optional bitfield of scope attribute flags.
#[wasm_bindgen(js_name = "pushScope")]
pub fn nvagentrt_push_scope(
    name: &str,
    scope_type: i32,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
) -> Result<WasmScopeHandle, JsValue> {
    let attrs = core_types::ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    nvagentrt_core::nvagentrt_push_scope(
        name,
        i32_to_scope_type(scope_type),
        parent.as_ref().map(|h| &h.inner),
        attrs,
    )
    .map(WasmScopeHandle::from)
    .map_err(to_js_err)
}

/// Pops the scope identified by `handle` from the scope stack.
///
/// Throws if the handle does not match the current top of the stack.
#[wasm_bindgen(js_name = "popScope")]
pub fn nvagentrt_pop_scope(handle: &WasmScopeHandle) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_pop_scope(&handle.inner.uuid).map_err(to_js_err)
}

/// Emits a custom event to all registered subscribers.
///
/// - `name` - Event name.
/// - `parent` - Optional parent scope handle for the event.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
#[wasm_bindgen(js_name = "event")]
pub fn nvagentrt_event(
    name: &str,
    parent: Option<WasmScopeHandle>,
    data: JsValue,
    metadata: JsValue,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_event(
        name,
        parent.as_ref().map(|h| &h.inner),
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
    )
    .map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begins a tool call, returning a `WasmToolHandle` for the active invocation.
///
/// Runs request guardrails and intercepts on the arguments before returning.
///
/// - `name` - Tool name.
/// - `args` - JSON arguments to the tool.
/// - `parent` - Optional parent scope handle.
/// - `attributes` - Optional bitfield of tool attribute flags.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
#[wasm_bindgen(js_name = "toolCall")]
pub fn nvagentrt_tool_call(
    name: &str,
    args: JsValue,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    tool_call_id: Option<String>,
) -> Result<WasmToolHandle, JsValue> {
    let args_json = js_to_json(&args)?;
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    nvagentrt_core::nvagentrt_tool_call(
        name,
        args_json,
        parent.as_ref().map(|h| &h.inner),
        attrs,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
        tool_call_id,
    )
    .map(WasmToolHandle::from)
    .map_err(to_js_err)
}

/// Ends an active tool call, running response guardrails and intercepts.
///
/// - `handle` - The tool handle returned by `toolCall`.
/// - `result` - JSON result of the tool execution.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
#[wasm_bindgen(js_name = "toolCallEnd")]
pub fn nvagentrt_tool_call_end(
    handle: &WasmToolHandle,
    result: JsValue,
    data: JsValue,
    metadata: JsValue,
) -> Result<(), JsValue> {
    let result_json = js_to_json(&result)?;
    nvagentrt_core::nvagentrt_tool_call_end(
        &handle.inner,
        result_json,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
    )
    .map_err(to_js_err)
}

/// Executes a full tool call lifecycle through the middleware pipeline.
///
/// Runs conditional-execution guardrails (on raw args) → request intercepts →
/// sanitize-request guardrails → execution intercepts → `func` → response
/// intercepts → sanitize-response guardrails. On rejection, only a standalone
/// Mark event is emitted (no Start/End pair) and `GuardrailRejected` is returned.
///
/// - `name` - Tool name.
/// - `args` - JSON arguments to the tool.
/// - `func` - JavaScript function `(args) => result | Promise<result>` to execute.
/// - `parent` - Optional parent scope handle.
/// - `attributes` - Optional bitfield of tool attribute flags.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
#[wasm_bindgen(js_name = "toolCallExecute")]
pub async fn nvagentrt_tool_call_execute(
    name: &str,
    args: JsValue,
    func: Function,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
) -> Result<JsValue, JsValue> {
    let args_json = js_to_json(&args)?;
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent_handle = parent
        .map(|h| h.inner)
        .unwrap_or_else(nvagentrt_core::task_scope_top);
    let exec_fn = callable::wrap_js_tool_exec_fn(func);
    let default_fn: nvagentrt_core::ToolExecutionNextFn = Box::new(move |args| exec_fn(args));

    let result = nvagentrt_core::nvagentrt_tool_call_execute(
        name,
        args_json,
        default_fn,
        Some(parent_handle),
        attrs,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
    )
    .await
    .map_err(to_js_err)?;

    Ok(json_to_js(&result))
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begins an LLM call, returning a `WasmLLMHandle` for the active invocation.
///
/// Runs request guardrails and intercepts on the request before returning.
///
/// - `name` - LLM provider/model name.
/// - `native` - The native LLM request as a JSON value.
/// - `parent` - Optional parent scope handle.
/// - `attributes` - Optional bitfield of LLM attribute flags.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
/// - `model_name` - Optional model name string.
/// - `to_request` - Optional JS function `(native) => { headers, content }` to convert
///   native JSON to a formal LLMRequest.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "llmCall")]
pub fn nvagentrt_llm_call(
    name: &str,
    native: JsValue,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    model_name: Option<String>,
    to_request: Option<Function>,
) -> Result<WasmLLMHandle, JsValue> {
    let native_json = js_to_json(&native)?;
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let to_req = to_request.map(wrap_wasm_to_request);
    nvagentrt_core::nvagentrt_llm_call(
        name,
        &native_json,
        parent.as_ref().map(|h| &h.inner),
        attrs,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
        model_name,
        to_req.as_ref(),
    )
    .map(WasmLLMHandle::from)
    .map_err(to_js_err)
}

/// Ends an active LLM call, running response guardrails and intercepts.
///
/// - `handle` - The LLM handle returned by `llmCall`.
/// - `response` - JSON response from the LLM.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
/// - `to_response` - Optional JS function `(native) => { data }` to convert
///   native JSON to a formal LLMResponse.
#[wasm_bindgen(js_name = "llmCallEnd")]
pub fn nvagentrt_llm_call_end(
    handle: &WasmLLMHandle,
    response: JsValue,
    data: JsValue,
    metadata: JsValue,
    to_response: Option<Function>,
) -> Result<(), JsValue> {
    let response_json = js_to_json(&response)?;
    let to_resp = to_response.map(wrap_wasm_to_response);
    nvagentrt_core::nvagentrt_llm_call_end(
        &handle.inner,
        response_json,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
        to_resp.as_ref(),
    )
    .map_err(to_js_err)
}

/// Executes a full LLM call lifecycle through the middleware pipeline.
///
/// Runs conditional-execution guardrails (on raw request) → request intercepts →
/// sanitize-request guardrails → execution intercepts → `func` → response
/// intercepts → sanitize-response guardrails. On rejection, only a standalone
/// Mark event is emitted (no Start/End pair) and `GuardrailRejected` is returned.
///
/// - `name` - LLM provider/model name.
/// - `native` - The native LLM request as a JSON value.
/// - `func` - JavaScript function `(native) => result | Promise<result>` to execute.
/// - `parent` - Optional parent scope handle.
/// - `attributes` - Optional bitfield of LLM attribute flags.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
/// - `model_name` - Optional model name string.
/// - `to_request` - Optional JS function `(native) => { headers, content }`.
/// - `to_response` - Optional JS function `(native) => { data }`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "llmCallExecute")]
pub async fn nvagentrt_llm_call_execute(
    name: &str,
    native: JsValue,
    func: Function,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    model_name: Option<String>,
    to_request: Option<Function>,
    to_response: Option<Function>,
) -> Result<JsValue, JsValue> {
    let native_json = js_to_json(&native)?;
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent_handle = parent
        .map(|h| h.inner)
        .unwrap_or_else(nvagentrt_core::task_scope_top);
    let exec_fn = callable::wrap_js_llm_exec_fn(func);
    let default_fn: nvagentrt_core::LlmExecutionNextFn = Box::new(move |native| exec_fn(native));

    let result = nvagentrt_core::nvagentrt_llm_call_execute(
        name,
        native_json,
        default_fn,
        Some(parent_handle),
        attrs,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
        model_name,
        to_request.map(wrap_wasm_to_request),
        to_response.map(wrap_wasm_to_response),
    )
    .await
    .map_err(to_js_err)?;

    Ok(json_to_js(&result))
}

/// Executes a streaming LLM call lifecycle through the middleware pipeline.
///
/// Like `llmCallExecute`, conditional-execution guardrails run first on the raw
/// request. Returns a `WasmLlmStream` whose `next()` method yields response
/// chunks incrementally. Stream-level intercepts are applied to each chunk.
///
/// - `name` - LLM provider/model name.
/// - `native` - The native LLM request as a JSON value.
/// - `func` - JavaScript function `(native) => result | Promise<result>` to execute.
/// - `collector` - Optional JavaScript function `(chunk) => void` called with each
///   intercepted Json chunk for accumulation.
/// - `finalizer` - Optional JavaScript function `() => object` called once when the
///   stream is exhausted to produce the aggregated response.
/// - `parent` - Optional parent scope handle.
/// - `attributes` - Optional bitfield of LLM attribute flags.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
/// - `model_name` - Optional model name string.
/// - `to_request` - Optional JS function `(native) => { headers, content }`.
/// - `to_response` - Optional JS function `(native) => { data }`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "llmStreamCallExecute")]
pub async fn nvagentrt_llm_stream_call_execute(
    name: &str,
    native: JsValue,
    func: Function,
    collector: Option<Function>,
    finalizer: Option<Function>,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    model_name: Option<String>,
    to_request: Option<Function>,
    to_response: Option<Function>,
) -> Result<WasmLlmStream, JsValue> {
    let native_json = js_to_json(&native)?;
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent_handle = parent
        .map(|h| h.inner)
        .unwrap_or_else(nvagentrt_core::task_scope_top);
    let exec_fn = callable::wrap_js_llm_exec_fn(func);

    let wrapped_collector: Box<dyn FnMut(serde_json::Value) + Send> = match collector {
        Some(cb) => callable::wrap_js_collector_fn(cb),
        None => Box::new(|_: serde_json::Value| {}),
    };

    let wrapped_finalizer: Box<dyn FnOnce() -> serde_json::Value + Send> = match finalizer {
        Some(cb) => callable::wrap_js_finalizer_fn(cb),
        None => Box::new(|| serde_json::Value::Null),
    };

    // Bridge LlmExecutionFn -> LlmStreamExecutionNextFn (FnOnce)
    let default_fn: nvagentrt_core::LlmStreamExecutionNextFn = Box::new(move |native| {
        let fut = exec_fn(native);
        Box::pin(async move {
            let result = fut.await?;
            let stream = tokio_stream::once(Ok(result));
            Ok(Box::pin(stream)
                as std::pin::Pin<
                    Box<
                        dyn tokio_stream::Stream<Item = nvagentrt_core::Result<serde_json::Value>>
                            + Send,
                    >,
                >)
        })
    });

    let rust_stream = nvagentrt_core::nvagentrt_llm_stream_call_execute(
        name,
        native_json,
        default_fn,
        wrapped_collector,
        wrapped_finalizer,
        Some(parent_handle),
        attrs,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
        model_name,
        to_request.map(wrap_wasm_to_request),
        to_response.map(wrap_wasm_to_response),
    )
    .await
    .map_err(to_js_err)?;

    use tokio_stream::StreamExt;
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    wasm_bindgen_futures::spawn_local(async move {
        let mut stream = rust_stream;
        while let Some(item) = stream.next().await {
            if tx.send(item).await.is_err() {
                break;
            }
        }
    });

    Ok(WasmLlmStream {
        receiver: tokio::sync::Mutex::new(rx),
    })
}

// ---------------------------------------------------------------------------
// Guardrail registrations
// ---------------------------------------------------------------------------

/// Registers a guardrail that sanitizes tool request arguments before execution.
///
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(name, args) => sanitizedArgs`.
#[wasm_bindgen(js_name = "registerToolSanitizeRequestGuardrail")]
pub fn register_tool_sanitize_request_guardrail(
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_tool_sanitize_request_guardrail(
        name,
        priority,
        callable::wrap_js_tool_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered tool sanitize-request guardrail by name.
///
/// Returns `true` if the guardrail was found and removed.
#[wasm_bindgen(js_name = "deregisterToolSanitizeRequestGuardrail")]
pub fn deregister_tool_sanitize_request_guardrail(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_tool_sanitize_request_guardrail(name).map_err(to_js_err)
}

/// Registers a guardrail that sanitizes tool response data after execution.
///
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(name, result) => sanitizedResult`.
#[wasm_bindgen(js_name = "registerToolSanitizeResponseGuardrail")]
pub fn register_tool_sanitize_response_guardrail(
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_tool_sanitize_response_guardrail(
        name,
        priority,
        callable::wrap_js_tool_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered tool sanitize-response guardrail by name.
///
/// Returns `true` if the guardrail was found and removed.
#[wasm_bindgen(js_name = "deregisterToolSanitizeResponseGuardrail")]
pub fn deregister_tool_sanitize_response_guardrail(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_tool_sanitize_response_guardrail(name).map_err(to_js_err)
}

/// Registers a guardrail that conditionally gates tool execution.
///
/// The guardrail function returns `null` to allow execution or a rejection
/// reason string to block it.
///
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(name, args) => string | null`.
#[wasm_bindgen(js_name = "registerToolConditionalExecutionGuardrail")]
pub fn register_tool_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_tool_conditional_execution_guardrail(
        name,
        priority,
        callable::wrap_js_tool_conditional_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered tool conditional-execution guardrail by name.
///
/// Returns `true` if the guardrail was found and removed.
#[wasm_bindgen(js_name = "deregisterToolConditionalExecutionGuardrail")]
pub fn deregister_tool_conditional_execution_guardrail(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_tool_conditional_execution_guardrail(name)
        .map_err(to_js_err)
}

// Tool intercepts

/// Registers an intercept that transforms tool request arguments.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(name, args) => transformedArgs`.
#[wasm_bindgen(js_name = "registerToolRequestIntercept")]
pub fn register_tool_request_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    func: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_tool_request_intercept(
        name,
        priority,
        break_chain,
        callable::wrap_js_tool_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered tool request intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterToolRequestIntercept")]
pub fn deregister_tool_request_intercept(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_tool_request_intercept(name).map_err(to_js_err)
}

/// Registers an intercept that transforms tool response data.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(name, result) => transformedResult`.
#[wasm_bindgen(js_name = "registerToolResponseIntercept")]
pub fn register_tool_response_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    func: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_tool_response_intercept(
        name,
        priority,
        break_chain,
        callable::wrap_js_tool_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered tool response intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterToolResponseIntercept")]
pub fn deregister_tool_response_intercept(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_tool_response_intercept(name).map_err(to_js_err)
}

/// Registers a tool execution intercept following the middleware chain pattern.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `conditional` - JS function `(name, args) => boolean` that decides whether to intercept.
/// - `exec_fn` - JS function `(args, next) => result | Promise<result>` — intercept function.
///   Call `await next(args)` to invoke the next intercept or original implementation.
#[wasm_bindgen(js_name = "registerToolExecutionIntercept")]
pub fn register_tool_execution_intercept(
    name: &str,
    priority: i32,
    conditional: Function,
    exec_fn: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_tool_execution_intercept(
        name,
        priority,
        callable::wrap_js_tool_exec_conditional_fn(conditional),
        callable::wrap_js_tool_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered tool execution intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterToolExecutionIntercept")]
pub fn deregister_tool_execution_intercept(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_tool_execution_intercept(name).map_err(to_js_err)
}

// LLM guardrails

/// Registers a guardrail that sanitizes LLM request data before the call.
///
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(request) => sanitizedRequest`.
#[wasm_bindgen(js_name = "registerLlmSanitizeRequestGuardrail")]
pub fn register_llm_sanitize_request_guardrail(
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_llm_sanitize_request_guardrail(
        name,
        priority,
        callable::wrap_js_llm_sanitize_request_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM sanitize-request guardrail by name.
///
/// Returns `true` if the guardrail was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmSanitizeRequestGuardrail")]
pub fn deregister_llm_sanitize_request_guardrail(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_llm_sanitize_request_guardrail(name).map_err(to_js_err)
}

/// Registers a guardrail that sanitizes LLM response data after the call.
///
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(response) => sanitizedResponse`.
#[wasm_bindgen(js_name = "registerLlmSanitizeResponseGuardrail")]
pub fn register_llm_sanitize_response_guardrail(
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_llm_sanitize_response_guardrail(
        name,
        priority,
        callable::wrap_js_llm_response_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM sanitize-response guardrail by name.
///
/// Returns `true` if the guardrail was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmSanitizeResponseGuardrail")]
pub fn deregister_llm_sanitize_response_guardrail(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_llm_sanitize_response_guardrail(name).map_err(to_js_err)
}

/// Registers a guardrail that conditionally gates LLM execution.
///
/// The guardrail function returns `null` to allow execution or a rejection
/// reason string to block it.
///
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(request) => string | null`.
#[wasm_bindgen(js_name = "registerLlmConditionalExecutionGuardrail")]
pub fn register_llm_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_llm_conditional_execution_guardrail(
        name,
        priority,
        callable::wrap_js_llm_conditional_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM conditional-execution guardrail by name.
///
/// Returns `true` if the guardrail was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmConditionalExecutionGuardrail")]
pub fn deregister_llm_conditional_execution_guardrail(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_llm_conditional_execution_guardrail(name)
        .map_err(to_js_err)
}

// LLM intercepts

/// Registers an intercept that transforms LLM request data (native Json).
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(native) => transformedNative`.
#[wasm_bindgen(js_name = "registerLlmRequestIntercept")]
pub fn register_llm_request_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    func: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_llm_request_intercept(
        name,
        priority,
        break_chain,
        callable::wrap_js_json_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM request intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmRequestIntercept")]
pub fn deregister_llm_request_intercept(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_llm_request_intercept(name).map_err(to_js_err)
}

/// Registers an intercept that transforms LLM response data.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(response) => transformedResponse`.
#[wasm_bindgen(js_name = "registerLlmResponseIntercept")]
pub fn register_llm_response_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    func: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_llm_response_intercept(
        name,
        priority,
        break_chain,
        callable::wrap_js_llm_response_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM response intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmResponseIntercept")]
pub fn deregister_llm_response_intercept(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_llm_response_intercept(name).map_err(to_js_err)
}

/// Registers an intercept that transforms individual chunks in a streaming LLM response.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(chunk) => transformedChunk` (Json).
#[wasm_bindgen(js_name = "registerLlmStreamResponseIntercept")]
pub fn register_llm_stream_response_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    func: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_llm_stream_response_intercept(
        name,
        priority,
        break_chain,
        callable::wrap_js_stream_response_intercept_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM stream response intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmStreamResponseIntercept")]
pub fn deregister_llm_stream_response_intercept(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_llm_stream_response_intercept(name).map_err(to_js_err)
}

/// Registers an LLM execution intercept following the middleware chain pattern.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `conditional` - JS function `(native) => boolean` that decides whether to intercept.
/// - `exec_fn` - JS function `(native, next) => result | Promise<result>` — intercept function.
///   Call `await next(native)` to invoke the next intercept or original implementation.
#[wasm_bindgen(js_name = "registerLlmExecutionIntercept")]
pub fn register_llm_execution_intercept(
    name: &str,
    priority: i32,
    conditional: Function,
    exec_fn: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_llm_execution_intercept(
        name,
        priority,
        callable::wrap_js_llm_exec_conditional_fn(conditional),
        callable::wrap_js_llm_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM execution intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmExecutionIntercept")]
pub fn deregister_llm_execution_intercept(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_llm_execution_intercept(name).map_err(to_js_err)
}

/// Registers a streaming LLM execution intercept following the middleware chain pattern.
///
/// The execution function result is wrapped into a single-item stream internally.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `conditional` - JS function `(native) => boolean` that decides whether to intercept.
/// - `exec_fn` - JS function `(native, next) => result | Promise<result>` — intercept function.
///   Call `await next(native)` to invoke the next intercept or original streaming implementation.
#[wasm_bindgen(js_name = "registerLlmStreamExecutionIntercept")]
pub fn register_llm_stream_execution_intercept(
    name: &str,
    priority: i32,
    conditional: Function,
    exec_fn: Function,
) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_llm_stream_execution_intercept(
        name,
        priority,
        callable::wrap_js_llm_exec_conditional_fn(conditional),
        callable::wrap_js_llm_stream_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM stream execution intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmStreamExecutionIntercept")]
pub fn deregister_llm_stream_execution_intercept(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_llm_stream_execution_intercept(name).map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Registers an event subscriber that receives lifecycle events.
///
/// - `name` - Unique subscriber name.
/// - `callback` - JS function `(event) => void` called for each event.
#[wasm_bindgen(js_name = "registerSubscriber")]
pub fn register_subscriber(name: &str, callback: Function) -> Result<(), JsValue> {
    nvagentrt_core::nvagentrt_register_subscriber(
        name,
        callable::wrap_js_event_subscriber(callback),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered event subscriber by name.
///
/// Returns `true` if the subscriber was found and removed.
#[wasm_bindgen(js_name = "deregisterSubscriber")]
pub fn deregister_subscriber(name: &str) -> Result<bool, JsValue> {
    nvagentrt_core::nvagentrt_deregister_subscriber(name).map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Creates a new isolated scope stack.
#[wasm_bindgen(js_name = "createScopeStack")]
pub fn create_scope_stack() -> WasmScopeStack {
    WasmScopeStack {
        inner: nvagentrt_core::create_scope_stack(),
    }
}

/// Returns the current thread's scope stack handle.
#[wasm_bindgen(js_name = "currentScopeStack")]
pub fn current_scope_stack() -> WasmScopeStack {
    WasmScopeStack {
        inner: nvagentrt_core::current_scope_stack(),
    }
}

/// Binds a scope stack to the current thread.
#[wasm_bindgen(js_name = "setThreadScopeStack")]
pub fn set_thread_scope_stack(stack: &WasmScopeStack) {
    nvagentrt_core::set_thread_scope_stack(stack.inner.clone());
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Runs the registered tool request intercept chain on the given arguments.
#[wasm_bindgen(js_name = "toolRequestIntercepts")]
pub fn nvagentrt_tool_request_intercepts_wasm(
    name: &str,
    args: JsValue,
) -> Result<JsValue, JsValue> {
    let args_json = js_to_json(&args)?;
    let result =
        nvagentrt_core::nvagentrt_tool_request_intercepts(name, args_json).map_err(to_js_err)?;
    Ok(json_to_js(&result))
}

/// Runs the registered tool conditional execution guardrail chain.
#[wasm_bindgen(js_name = "toolConditionalExecution")]
pub fn nvagentrt_tool_conditional_execution_wasm(name: &str, args: JsValue) -> Result<(), JsValue> {
    let args_json = js_to_json(&args)?;
    nvagentrt_core::nvagentrt_tool_conditional_execution(name, &args_json).map_err(to_js_err)
}

/// Runs the registered tool response intercept chain on the given result.
#[wasm_bindgen(js_name = "toolResponseIntercepts")]
pub fn nvagentrt_tool_response_intercepts_wasm(
    name: &str,
    result: JsValue,
) -> Result<JsValue, JsValue> {
    let result_json = js_to_json(&result)?;
    let transformed =
        nvagentrt_core::nvagentrt_tool_response_intercepts(name, result_json).map_err(to_js_err)?;
    Ok(json_to_js(&transformed))
}

/// Runs the registered LLM request intercept chain on the given native Json.
#[wasm_bindgen(js_name = "llmRequestIntercepts")]
pub fn nvagentrt_llm_request_intercepts_wasm(native: JsValue) -> Result<JsValue, JsValue> {
    let native_json = js_to_json(&native)?;
    let result =
        nvagentrt_core::nvagentrt_llm_request_intercepts(native_json).map_err(to_js_err)?;
    Ok(json_to_js(&result))
}

/// Runs the registered LLM conditional execution guardrail chain.
///
/// - `native` - The native LLM request as a JSON value.
/// - `to_request` - Optional JS function `(native) => { headers, content }`.
#[wasm_bindgen(js_name = "llmConditionalExecution")]
pub fn nvagentrt_llm_conditional_execution_wasm(
    native: JsValue,
    to_request: Option<Function>,
) -> Result<(), JsValue> {
    let native_json = js_to_json(&native)?;
    let to_req = to_request.map(wrap_wasm_to_request);
    nvagentrt_core::nvagentrt_llm_conditional_execution(&native_json, to_req.as_ref())
        .map_err(to_js_err)
}

/// Runs the registered LLM response intercept chain on the given response.
#[wasm_bindgen(js_name = "llmResponseIntercepts")]
pub fn nvagentrt_llm_response_intercepts_wasm(
    response: &WasmLLMResponse,
) -> Result<WasmLLMResponse, JsValue> {
    let result = nvagentrt_core::nvagentrt_llm_response_intercepts(response.inner.clone())
        .map_err(to_js_err)?;
    Ok(WasmLLMResponse { inner: result })
}

// ---------------------------------------------------------------------------
// ATIF exporter
// ---------------------------------------------------------------------------

/// ATIF trajectory exporter for collecting events and producing ATIF JSON.
#[wasm_bindgen]
pub struct WasmAtifExporter {
    inner: nvagentrt_core::atif::AtifExporter,
}

#[wasm_bindgen]
impl WasmAtifExporter {
    /// Creates a new ATIF exporter.
    #[wasm_bindgen(constructor)]
    pub fn new(
        session_id: String,
        agent_name: String,
        agent_version: String,
        model_name: Option<String>,
    ) -> Self {
        let agent_info = nvagentrt_core::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: None,
            extra: None,
        };
        Self {
            inner: nvagentrt_core::atif::AtifExporter::new(session_id, agent_info),
        }
    }

    /// Registers the exporter as an event subscriber.
    pub fn register(&self, name: &str) -> Result<(), JsValue> {
        let subscriber = self.inner.subscriber();
        nvagentrt_core::nvagentrt_register_subscriber(name, subscriber)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deregisters the exporter subscriber.
    pub fn deregister(&self, name: &str) -> Result<bool, JsValue> {
        nvagentrt_core::nvagentrt_deregister_subscriber(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Exports collected events as an ATIF trajectory JSON string.
    pub fn export_json(&self, root_uuid: Option<String>) -> Result<String, JsValue> {
        let root = root_uuid
            .map(|s| uuid::Uuid::parse_str(&s))
            .transpose()
            .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
        let trajectory = self.inner.export(root);
        serde_json::to_string(&trajectory).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Clears all collected events.
    pub fn clear(&self) {
        self.inner.clear();
    }
}

// ---------------------------------------------------------------------------
// LLM converter helpers
// ---------------------------------------------------------------------------

/// Wrapper around a JS `Function` that is `Send + Sync` for WASM's
/// single-threaded execution model.
struct SendSyncFunction(send_wrapper::SendWrapper<Function>);

// Safety: WASM is single-threaded; SendWrapper ensures the inner Function is
// only accessed from the main thread where it was created.
unsafe impl Send for SendSyncFunction {}
unsafe impl Sync for SendSyncFunction {}

/// Wraps an optional JS `to_request` function into a boxed `ToRequestFn`.
fn wrap_wasm_to_request(func: Function) -> nvagentrt_core::ToRequestFn {
    let wrapper = SendSyncFunction(send_wrapper::SendWrapper::new(func));
    Box::new(move |native: &serde_json::Value| {
        let js_native = json_to_js(native);
        match wrapper.0.call1(&JsValue::NULL, &js_native) {
            Ok(result) => {
                let result_json = js_to_json(&result).unwrap_or(serde_json::Value::Null);
                serde_json::from_value(result_json).unwrap_or_else(|_| {
                    nvagentrt_core::types::LLMRequest {
                        headers: serde_json::Map::new(),
                        content: native.clone(),
                    }
                })
            }
            Err(_) => nvagentrt_core::types::LLMRequest {
                headers: serde_json::Map::new(),
                content: native.clone(),
            },
        }
    })
}

/// Wraps an optional JS `to_response` function into a boxed `ToResponseFn`.
fn wrap_wasm_to_response(func: Function) -> nvagentrt_core::ToResponseFn {
    let wrapper = SendSyncFunction(send_wrapper::SendWrapper::new(func));
    Box::new(move |native: &serde_json::Value| {
        let js_native = json_to_js(native);
        match wrapper.0.call1(&JsValue::NULL, &js_native) {
            Ok(result) => {
                let result_json = js_to_json(&result).unwrap_or(serde_json::Value::Null);
                serde_json::from_value(result_json).unwrap_or_else(|_| {
                    nvagentrt_core::types::LLMResponse {
                        data: native.clone(),
                    }
                })
            }
            Err(_) => nvagentrt_core::types::LLMResponse {
                data: native.clone(),
            },
        }
    })
}
