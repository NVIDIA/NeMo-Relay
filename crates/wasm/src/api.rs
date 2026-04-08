// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level Nexus API functions exposed to JavaScript via `wasm_bindgen`.
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
//!   and execution intercepts for tools; request and execution intercepts for
//!   LLMs.
//! - **Event subscribers** -- register and deregister lifecycle event
//!   subscribers.
//!
//! All functions use `JsValue` for JSON payloads and return `Result<T, JsValue>`
//! where errors are thrown as JavaScript exceptions.

use std::collections::HashMap;
use std::sync::Arc;

use js_sys::Function;
use serde::{Deserialize, Serialize};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use nvidia_nat_nexus_core::types as core_types;

use crate::callable;
use crate::convert::*;
use crate::stream::WasmLlmStream;
use crate::types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmOpenTelemetryConfig {
    transport: Option<String>,
    endpoint: Option<String>,
    headers: Option<HashMap<String, String>>,
    resource_attributes: Option<HashMap<String, String>>,
    service_name: Option<String>,
    service_namespace: Option<String>,
    service_version: Option<String>,
    instrumentation_scope: Option<String>,
    timeout_millis: Option<u32>,
}

impl Default for WasmOpenTelemetryConfig {
    fn default() -> Self {
        Self {
            transport: Some("http_binary".to_string()),
            endpoint: None,
            headers: Some(HashMap::new()),
            resource_attributes: Some(HashMap::new()),
            service_name: Some("nat-nexus".to_string()),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: Some("nvidia-nat-nexus-otel".to_string()),
            timeout_millis: Some(3_000),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmOpenInferenceConfig {
    transport: Option<String>,
    endpoint: Option<String>,
    headers: Option<HashMap<String, String>>,
    resource_attributes: Option<HashMap<String, String>>,
    service_name: Option<String>,
    service_namespace: Option<String>,
    service_version: Option<String>,
    instrumentation_scope: Option<String>,
    timeout_millis: Option<u32>,
}

impl Default for WasmOpenInferenceConfig {
    fn default() -> Self {
        Self {
            transport: Some("http_binary".to_string()),
            endpoint: None,
            headers: Some(HashMap::new()),
            resource_attributes: Some(HashMap::new()),
            service_name: Some("nat-nexus".to_string()),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: Some("nvidia-nat-nexus-openinference".to_string()),
            timeout_millis: Some(3_000),
        }
    }
}

fn build_otel_config(
    config: Option<WasmOpenTelemetryConfig>,
) -> Result<nvidia_nat_nexus_otel::OpenTelemetryConfig, JsValue> {
    let config = config.unwrap_or_default();
    let transport = config
        .transport
        .unwrap_or_else(|| "http_binary".to_string());
    let service_name = config
        .service_name
        .unwrap_or_else(|| "nat-nexus".to_string());
    let instrumentation_scope = config
        .instrumentation_scope
        .unwrap_or_else(|| "nvidia-nat-nexus-otel".to_string());
    let timeout_millis = config.timeout_millis.unwrap_or(3_000);

    let mut otel_config = match transport.as_str() {
        "http_binary" => nvidia_nat_nexus_otel::OpenTelemetryConfig::http_binary(service_name),
        "grpc" => nvidia_nat_nexus_otel::OpenTelemetryConfig::grpc(service_name),
        other => {
            return Err(JsValue::from_str(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}",
            )));
        }
    }
    .with_instrumentation_scope(instrumentation_scope)
    .with_timeout(std::time::Duration::from_millis(timeout_millis.into()));

    if let Some(endpoint) = config.endpoint {
        otel_config = otel_config.with_endpoint(endpoint);
    }
    if let Some(namespace) = config.service_namespace {
        otel_config = otel_config.with_service_namespace(namespace);
    }
    if let Some(version) = config.service_version {
        otel_config = otel_config.with_service_version(version);
    }
    for (key, value) in config.headers.unwrap_or_default() {
        otel_config = otel_config.with_header(key, value);
    }
    for (key, value) in config.resource_attributes.unwrap_or_default() {
        otel_config = otel_config.with_resource_attribute(key, value);
    }
    Ok(otel_config)
}

fn build_openinference_config(
    config: Option<WasmOpenInferenceConfig>,
) -> Result<nvidia_nat_nexus_openinference::OpenInferenceConfig, JsValue> {
    let config = config.unwrap_or_default();
    let transport = config
        .transport
        .unwrap_or_else(|| "http_binary".to_string());
    let service_name = config
        .service_name
        .unwrap_or_else(|| "nat-nexus".to_string());
    let instrumentation_scope = config
        .instrumentation_scope
        .unwrap_or_else(|| "nvidia-nat-nexus-openinference".to_string());
    let timeout_millis = config.timeout_millis.unwrap_or(3_000);

    let transport = match transport.as_str() {
        "http_binary" => nvidia_nat_nexus_openinference::OtlpTransport::HttpBinary,
        "grpc" => nvidia_nat_nexus_openinference::OtlpTransport::Grpc,
        other => {
            return Err(JsValue::from_str(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}",
            )));
        }
    };

    let mut openinference_config = nvidia_nat_nexus_openinference::OpenInferenceConfig::new()
        .with_transport(transport)
        .with_service_name(service_name)
        .with_instrumentation_scope(instrumentation_scope)
        .with_timeout(std::time::Duration::from_millis(timeout_millis.into()));

    if let Some(endpoint) = config.endpoint {
        openinference_config = openinference_config.with_endpoint(endpoint);
    }
    if let Some(namespace) = config.service_namespace {
        openinference_config = openinference_config.with_service_namespace(namespace);
    }
    if let Some(version) = config.service_version {
        openinference_config = openinference_config.with_service_version(version);
    }
    for (key, value) in config.headers.unwrap_or_default() {
        openinference_config = openinference_config.with_header(key, value);
    }
    for (key, value) in config.resource_attributes.unwrap_or_default() {
        openinference_config = openinference_config.with_resource_attribute(key, value);
    }
    Ok(openinference_config)
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Returns the handle of the current (topmost) scope on the scope stack.
///
/// Throws if the scope stack is empty.
#[wasm_bindgen(js_name = "getHandle")]
pub fn nat_nexus_get_handle() -> Result<WasmScopeHandle, JsValue> {
    nvidia_nat_nexus_core::nat_nexus_get_handle()
        .map(WasmScopeHandle::from)
        .map_err(to_js_err)
}

/// Pushes a new scope onto the scope stack and returns its handle.
///
/// - `name` - Human-readable scope name.
/// - `scope_type` - Integer scope type constant (e.g. `SCOPE_TYPE_AGENT`).
/// - `parent` - Optional parent scope handle; uses the current top if omitted.
/// - `attributes` - Optional bitfield of scope attribute flags.
/// - `data` - Optional JSON application data payload.
/// - `metadata` - Optional JSON metadata payload.
#[wasm_bindgen(js_name = "pushScope")]
pub fn nat_nexus_push_scope(
    name: &str,
    scope_type: i32,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
) -> Result<WasmScopeHandle, JsValue> {
    let attrs = core_types::ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    nvidia_nat_nexus_core::nat_nexus_push_scope(
        name,
        i32_to_scope_type(scope_type),
        parent.as_ref().map(|h| &h.inner),
        attrs,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
    )
    .map(WasmScopeHandle::from)
    .map_err(to_js_err)
}

/// Pops the scope identified by `handle` from the scope stack.
///
/// Throws if the handle does not match the current top of the stack.
#[wasm_bindgen(js_name = "popScope")]
pub fn nat_nexus_pop_scope(handle: &WasmScopeHandle) -> Result<(), JsValue> {
    nvidia_nat_nexus_core::nat_nexus_pop_scope(&handle.inner.uuid).map_err(to_js_err)
}

/// Returns the most recent callback error that could not be surfaced through a direct exception.
#[wasm_bindgen(js_name = "getLastCallbackError")]
pub fn nat_nexus_get_last_callback_error() -> Option<String> {
    get_last_callback_error()
}

/// Clears the most recent callback error recorded by the WASM binding.
#[wasm_bindgen(js_name = "clearLastCallbackError")]
pub fn nat_nexus_clear_last_callback_error() {
    clear_last_callback_error();
}

/// Pushes a scope, invokes the callback, then pops the scope automatically.
///
/// Creates a child scope with the given `name` and `scope_type`, calls the
/// `callback` with a `WasmScopeHandle`, and guarantees the scope is popped
/// when the callback returns (whether normally or by throwing). If the callback
/// returns a `Promise`, the scope is popped after the Promise settles.
///
/// - `name` - Human-readable scope name.
/// - `scope_type` - Integer scope type constant (e.g. `SCOPE_TYPE_AGENT`).
/// - `callback` - A JS function `(handle) => result` or `(handle) => Promise<result>`.
/// - `parent` - Optional parent scope handle; uses the current top if omitted.
/// - `attributes` - Optional bitfield of scope attribute flags.
/// - `data` - Optional JSON application data payload.
/// - `metadata` - Optional JSON metadata payload.
#[wasm_bindgen(js_name = "withScope")]
pub fn nat_nexus_with_scope(
    name: &str,
    scope_type: i32,
    callback: &Function,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
) -> Result<JsValue, JsValue> {
    let attrs = core_types::ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let scope_handle = nvidia_nat_nexus_core::nat_nexus_push_scope(
        name,
        i32_to_scope_type(scope_type),
        parent.as_ref().map(|h| &h.inner),
        attrs,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
    )
    .map(WasmScopeHandle::from)
    .map_err(to_js_err)?;

    let scope_uuid = scope_handle.inner.uuid;

    // Call the callback with the scope handle.
    let scope_handle_js: JsValue = scope_handle.into();
    let result = callback.call1(&JsValue::NULL, &scope_handle_js);

    match result {
        Ok(ref val) if val.has_type::<js_sys::Promise>() => {
            // Callback returned a Promise — defer pop to settlement.
            let promise: JsValue = val.clone();

            let then_uuid = scope_uuid;
            let then_cb = Closure::once(move |resolved: JsValue| -> JsValue {
                let _ = nvidia_nat_nexus_core::nat_nexus_pop_scope(&then_uuid);
                resolved
            });

            let catch_uuid = scope_uuid;
            let catch_cb = Closure::once(move |rejected: JsValue| -> JsValue {
                let _ = nvidia_nat_nexus_core::nat_nexus_pop_scope(&catch_uuid);
                // Re-throw by returning a rejected promise
                js_sys::Promise::reject(&rejected).into()
            });

            // Chain .then(onFulfilled, onRejected) via JS interop.
            let then_fn: Function = then_cb.into_js_value().unchecked_into();
            let catch_fn: Function = catch_cb.into_js_value().unchecked_into();
            let then_method: Function =
                js_sys::Reflect::get(&promise, &"then".into())?.unchecked_into();
            let chained = then_method.call2(&promise, &then_fn, &catch_fn)?;
            Ok(chained)
        }
        Ok(val) => {
            // Synchronous return — pop immediately.
            let _ = nvidia_nat_nexus_core::nat_nexus_pop_scope(&scope_uuid);
            Ok(val)
        }
        Err(err) => {
            // Callback threw — pop and propagate the error.
            let _ = nvidia_nat_nexus_core::nat_nexus_pop_scope(&scope_uuid);
            Err(err)
        }
    }
}

/// Emits a custom event to all registered subscribers.
///
/// - `name` - Event name.
/// - `parent` - Optional parent scope handle for the event.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
#[wasm_bindgen(js_name = "event")]
pub fn nat_nexus_event(
    name: &str,
    parent: Option<WasmScopeHandle>,
    data: JsValue,
    metadata: JsValue,
) -> Result<(), JsValue> {
    nvidia_nat_nexus_core::nat_nexus_event(
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
pub fn nat_nexus_tool_call(
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
    nvidia_nat_nexus_core::nat_nexus_tool_call(
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
pub fn nat_nexus_tool_call_end(
    handle: &WasmToolHandle,
    result: JsValue,
    data: JsValue,
    metadata: JsValue,
) -> Result<(), JsValue> {
    let result_json = js_to_json(&result)?;
    nvidia_nat_nexus_core::nat_nexus_tool_call_end(
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
pub async fn nat_nexus_tool_call_execute(
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
        .unwrap_or_else(nvidia_nat_nexus_core::task_scope_top);
    let exec_fn = callable::wrap_js_tool_exec_fn(func);
    let default_fn: nvidia_nat_nexus_core::ToolExecutionNextFn =
        Arc::new(move |args| exec_fn(args));

    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();
    let data_json = opt_js_to_json(&data)?;
    let metadata_json = opt_js_to_json(&metadata)?;
    let result = nvidia_nat_nexus_core::TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            nvidia_nat_nexus_core::nat_nexus_tool_call_execute(
                name,
                args_json,
                default_fn,
                Some(parent_handle),
                attrs,
                data_json,
                metadata_json,
            )
            .await
        })
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
/// - `request` - The LLM request as a JSON value with `{ headers, content }` shape.
/// - `parent` - Optional parent scope handle.
/// - `attributes` - Optional bitfield of LLM attribute flags.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
/// - `model_name` - Optional model name string.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "llmCall")]
pub fn nat_nexus_llm_call(
    name: &str,
    request: JsValue,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    model_name: Option<String>,
) -> Result<WasmLLMHandle, JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: core_types::LLMRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(nvidia_nat_nexus_core::NexusError::Internal(e.to_string())))?;
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    nvidia_nat_nexus_core::nat_nexus_llm_call(
        name,
        &llm_request,
        parent.as_ref().map(|h| &h.inner),
        attrs,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
        model_name,
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
#[wasm_bindgen(js_name = "llmCallEnd")]
pub fn nat_nexus_llm_call_end(
    handle: &WasmLLMHandle,
    response: JsValue,
    data: JsValue,
    metadata: JsValue,
) -> Result<(), JsValue> {
    let response_json = js_to_json(&response)?;
    nvidia_nat_nexus_core::nat_nexus_llm_call_end(
        &handle.inner,
        response_json,
        opt_js_to_json(&data)?,
        opt_js_to_json(&metadata)?,
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
/// - `request` - The LLM request as a JSON value with `{ headers, content }` shape.
/// - `func` - JavaScript function `(request) => result | Promise<result>` to execute.
/// - `parent` - Optional parent scope handle.
/// - `attributes` - Optional bitfield of LLM attribute flags.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
/// - `model_name` - Optional model name string.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "llmCallExecute")]
pub async fn nat_nexus_llm_call_execute(
    name: &str,
    request: JsValue,
    func: Function,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    model_name: Option<String>,
) -> Result<JsValue, JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: core_types::LLMRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(nvidia_nat_nexus_core::NexusError::Internal(e.to_string())))?;
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent_handle = parent
        .map(|h| h.inner)
        .unwrap_or_else(nvidia_nat_nexus_core::task_scope_top);
    let exec_fn = callable::wrap_js_llm_exec_fn(func);
    let default_fn: nvidia_nat_nexus_core::LlmExecutionNextFn =
        Arc::new(move |request| exec_fn(request));

    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();
    let data_json = opt_js_to_json(&data)?;
    let metadata_json = opt_js_to_json(&metadata)?;
    let result = nvidia_nat_nexus_core::TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            nvidia_nat_nexus_core::nat_nexus_llm_call_execute(
                name,
                llm_request,
                default_fn,
                Some(parent_handle),
                attrs,
                data_json,
                metadata_json,
                model_name,
            )
            .await
        })
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
/// - `request` - The LLM request as a JSON value with `{ headers, content }` shape.
/// - `func` - JavaScript function `(request) => result | Promise<result>` to execute.
/// - `collector` - Optional JavaScript function `(chunk) => void` called with each
///   intercepted Json chunk for accumulation.
/// - `finalizer` - Optional JavaScript function `() => object` called once when the
///   stream is exhausted to produce the aggregated response.
/// - `parent` - Optional parent scope handle.
/// - `attributes` - Optional bitfield of LLM attribute flags.
/// - `data` - Optional JSON data payload.
/// - `metadata` - Optional JSON metadata payload.
/// - `model_name` - Optional model name string.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "llmStreamCallExecute")]
pub async fn nat_nexus_llm_stream_call_execute(
    name: &str,
    request: JsValue,
    func: Function,
    collector: Option<Function>,
    finalizer: Option<Function>,
    parent: Option<WasmScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    model_name: Option<String>,
) -> Result<WasmLlmStream, JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: core_types::LLMRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(nvidia_nat_nexus_core::NexusError::Internal(e.to_string())))?;
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent_handle = parent
        .map(|h| h.inner)
        .unwrap_or_else(nvidia_nat_nexus_core::task_scope_top);
    let exec_fn = callable::wrap_js_llm_exec_fn(func);

    let wrapped_collector: Box<
        dyn FnMut(serde_json::Value) -> nvidia_nat_nexus_core::Result<()> + Send,
    > = match collector {
        Some(cb) => callable::wrap_js_collector_fn(cb),
        None => Box::new(|_: serde_json::Value| Ok(())),
    };

    let wrapped_finalizer: Box<dyn FnOnce() -> serde_json::Value + Send> = match finalizer {
        Some(cb) => callable::wrap_js_finalizer_fn(cb),
        None => Box::new(|| serde_json::Value::Null),
    };

    // Bridge LlmExecutionFn -> LlmStreamExecutionNextFn
    let default_fn: nvidia_nat_nexus_core::LlmStreamExecutionNextFn = Arc::new(move |request| {
        let fut = exec_fn(request);
        Box::pin(async move {
            let result = fut.await?;
            let stream = tokio_stream::once(Ok(result));
            Ok(Box::pin(stream)
                as std::pin::Pin<
                    Box<
                        dyn tokio_stream::Stream<
                                Item = nvidia_nat_nexus_core::Result<serde_json::Value>,
                            > + Send,
                    >,
                >)
        })
    });

    let scope_stack = nvidia_nat_nexus_core::current_scope_stack();
    let data_json = opt_js_to_json(&data)?;
    let metadata_json = opt_js_to_json(&metadata)?;
    let rust_stream = nvidia_nat_nexus_core::TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            nvidia_nat_nexus_core::nat_nexus_llm_stream_call_execute(
                name,
                llm_request,
                default_fn,
                wrapped_collector,
                wrapped_finalizer,
                Some(parent_handle),
                attrs,
                data_json,
                metadata_json,
                model_name,
            )
            .await
        })
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
    nvidia_nat_nexus_core::nat_nexus_register_tool_sanitize_request_guardrail(
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
    nvidia_nat_nexus_core::nat_nexus_deregister_tool_sanitize_request_guardrail(name)
        .map_err(to_js_err)
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
    nvidia_nat_nexus_core::nat_nexus_register_tool_sanitize_response_guardrail(
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
    nvidia_nat_nexus_core::nat_nexus_deregister_tool_sanitize_response_guardrail(name)
        .map_err(to_js_err)
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
    nvidia_nat_nexus_core::nat_nexus_register_tool_conditional_execution_guardrail(
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
    nvidia_nat_nexus_core::nat_nexus_deregister_tool_conditional_execution_guardrail(name)
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
    nvidia_nat_nexus_core::nat_nexus_register_tool_request_intercept(
        name,
        priority,
        break_chain,
        callable::wrap_js_tool_request_intercept_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered tool request intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterToolRequestIntercept")]
pub fn deregister_tool_request_intercept(name: &str) -> Result<bool, JsValue> {
    nvidia_nat_nexus_core::nat_nexus_deregister_tool_request_intercept(name).map_err(to_js_err)
}

/// Registers a tool execution intercept following the middleware chain pattern.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `exec_fn` - JS function `(args, next) => result | Promise<result>` — intercept function.
///   Call `await next(args)` to invoke the next intercept or original implementation.
#[wasm_bindgen(js_name = "registerToolExecutionIntercept")]
pub fn register_tool_execution_intercept(
    name: &str,
    priority: i32,
    exec_fn: Function,
) -> Result<(), JsValue> {
    nvidia_nat_nexus_core::nat_nexus_register_tool_execution_intercept(
        name,
        priority,
        callable::wrap_js_tool_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered tool execution intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterToolExecutionIntercept")]
pub fn deregister_tool_execution_intercept(name: &str) -> Result<bool, JsValue> {
    nvidia_nat_nexus_core::nat_nexus_deregister_tool_execution_intercept(name).map_err(to_js_err)
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
    nvidia_nat_nexus_core::nat_nexus_register_llm_sanitize_request_guardrail(
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
    nvidia_nat_nexus_core::nat_nexus_deregister_llm_sanitize_request_guardrail(name)
        .map_err(to_js_err)
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
    nvidia_nat_nexus_core::nat_nexus_register_llm_sanitize_response_guardrail(
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
    nvidia_nat_nexus_core::nat_nexus_deregister_llm_sanitize_response_guardrail(name)
        .map_err(to_js_err)
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
    nvidia_nat_nexus_core::nat_nexus_register_llm_conditional_execution_guardrail(
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
    nvidia_nat_nexus_core::nat_nexus_deregister_llm_conditional_execution_guardrail(name)
        .map_err(to_js_err)
}

// LLM intercepts

/// Registers an intercept that transforms LLM request data (`LLMRequest`).
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(request) => transformedRequest`.
#[wasm_bindgen(js_name = "registerLlmRequestIntercept")]
pub fn register_llm_request_intercept(
    name: &str,
    priority: i32,
    break_chain: bool,
    func: Function,
) -> Result<(), JsValue> {
    nvidia_nat_nexus_core::nat_nexus_register_llm_request_intercept(
        name,
        priority,
        break_chain,
        callable::wrap_js_llm_request_intercept_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM request intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmRequestIntercept")]
pub fn deregister_llm_request_intercept(name: &str) -> Result<bool, JsValue> {
    nvidia_nat_nexus_core::nat_nexus_deregister_llm_request_intercept(name).map_err(to_js_err)
}

/// Registers an LLM execution intercept following the middleware chain pattern.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `exec_fn` - JS function `(native, next) => result | Promise<result>` — intercept function.
///   Call `await next(native)` to invoke the next intercept or original implementation.
#[wasm_bindgen(js_name = "registerLlmExecutionIntercept")]
pub fn register_llm_execution_intercept(
    name: &str,
    priority: i32,
    exec_fn: Function,
) -> Result<(), JsValue> {
    nvidia_nat_nexus_core::nat_nexus_register_llm_execution_intercept(
        name,
        priority,
        callable::wrap_js_llm_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM execution intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmExecutionIntercept")]
pub fn deregister_llm_execution_intercept(name: &str) -> Result<bool, JsValue> {
    nvidia_nat_nexus_core::nat_nexus_deregister_llm_execution_intercept(name).map_err(to_js_err)
}

/// Registers a streaming LLM execution intercept following the middleware chain pattern.
///
/// The execution function result is wrapped into a single-item stream internally.
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `exec_fn` - JS function `(native, next) => result | Promise<result>` — intercept function.
///   Call `await next(native)` to invoke the next intercept or original streaming implementation.
#[wasm_bindgen(js_name = "registerLlmStreamExecutionIntercept")]
pub fn register_llm_stream_execution_intercept(
    name: &str,
    priority: i32,
    exec_fn: Function,
) -> Result<(), JsValue> {
    nvidia_nat_nexus_core::nat_nexus_register_llm_stream_execution_intercept(
        name,
        priority,
        callable::wrap_js_llm_stream_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a previously registered LLM stream execution intercept by name.
///
/// Returns `true` if the intercept was found and removed.
#[wasm_bindgen(js_name = "deregisterLlmStreamExecutionIntercept")]
pub fn deregister_llm_stream_execution_intercept(name: &str) -> Result<bool, JsValue> {
    nvidia_nat_nexus_core::nat_nexus_deregister_llm_stream_execution_intercept(name)
        .map_err(to_js_err)
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
    nvidia_nat_nexus_core::nat_nexus_register_subscriber(
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
    nvidia_nat_nexus_core::nat_nexus_deregister_subscriber(name).map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations — Tool
// ---------------------------------------------------------------------------

/// Registers a scope-local guardrail that sanitizes tool request arguments before execution.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(name, args) => sanitizedArgs`.
#[wasm_bindgen(js_name = "scopeRegisterToolSanitizeRequestGuardrail")]
pub fn scope_register_tool_sanitize_request_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_tool_sanitize_request_guardrail(
        &uuid,
        name,
        priority,
        callable::wrap_js_tool_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local tool sanitize-request guardrail by name.
///
/// Returns `true` if the guardrail was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterToolSanitizeRequestGuardrail")]
pub fn scope_deregister_tool_sanitize_request_guardrail(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_tool_sanitize_request_guardrail(&uuid, name)
        .map_err(to_js_err)
}

/// Registers a scope-local guardrail that sanitizes tool response data after execution.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(name, result) => sanitizedResult`.
#[wasm_bindgen(js_name = "scopeRegisterToolSanitizeResponseGuardrail")]
pub fn scope_register_tool_sanitize_response_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_tool_sanitize_response_guardrail(
        &uuid,
        name,
        priority,
        callable::wrap_js_tool_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local tool sanitize-response guardrail by name.
///
/// Returns `true` if the guardrail was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterToolSanitizeResponseGuardrail")]
pub fn scope_deregister_tool_sanitize_response_guardrail(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_tool_sanitize_response_guardrail(&uuid, name)
        .map_err(to_js_err)
}

/// Registers a scope-local guardrail that conditionally gates tool execution.
///
/// The guardrail function returns `null` to allow execution or a rejection
/// reason string to block it.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(name, args) => string | null`.
#[wasm_bindgen(js_name = "scopeRegisterToolConditionalExecutionGuardrail")]
pub fn scope_register_tool_conditional_execution_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_tool_conditional_execution_guardrail(
        &uuid,
        name,
        priority,
        callable::wrap_js_tool_conditional_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local tool conditional-execution guardrail by name.
///
/// Returns `true` if the guardrail was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterToolConditionalExecutionGuardrail")]
pub fn scope_deregister_tool_conditional_execution_guardrail(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_tool_conditional_execution_guardrail(
        &uuid, name,
    )
    .map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — Tool
// ---------------------------------------------------------------------------

/// Registers a scope-local intercept that transforms tool request arguments.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(name, args) => transformedArgs`.
#[wasm_bindgen(js_name = "scopeRegisterToolRequestIntercept")]
pub fn scope_register_tool_request_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    break_chain: bool,
    func: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_tool_request_intercept(
        &uuid,
        name,
        priority,
        break_chain,
        callable::wrap_js_tool_request_intercept_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local tool request intercept by name.
///
/// Returns `true` if the intercept was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterToolRequestIntercept")]
pub fn scope_deregister_tool_request_intercept(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_tool_request_intercept(&uuid, name)
        .map_err(to_js_err)
}

/// Registers a scope-local tool execution intercept following the middleware chain pattern.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `exec_fn` - JS function `(args, next) => result | Promise<result>` -- intercept function.
///   Call `await next(args)` to invoke the next intercept or original implementation.
#[wasm_bindgen(js_name = "scopeRegisterToolExecutionIntercept")]
pub fn scope_register_tool_execution_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    exec_fn: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_tool_execution_intercept(
        &uuid,
        name,
        priority,
        callable::wrap_js_tool_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local tool execution intercept by name.
///
/// Returns `true` if the intercept was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterToolExecutionIntercept")]
pub fn scope_deregister_tool_execution_intercept(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_tool_execution_intercept(&uuid, name)
        .map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations — LLM
// ---------------------------------------------------------------------------

/// Registers a scope-local guardrail that sanitizes LLM request data before the call.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(request) => sanitizedRequest`.
#[wasm_bindgen(js_name = "scopeRegisterLlmSanitizeRequestGuardrail")]
pub fn scope_register_llm_sanitize_request_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_llm_sanitize_request_guardrail(
        &uuid,
        name,
        priority,
        callable::wrap_js_llm_sanitize_request_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local LLM sanitize-request guardrail by name.
///
/// Returns `true` if the guardrail was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterLlmSanitizeRequestGuardrail")]
pub fn scope_deregister_llm_sanitize_request_guardrail(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_llm_sanitize_request_guardrail(&uuid, name)
        .map_err(to_js_err)
}

/// Registers a scope-local guardrail that sanitizes LLM response data after the call.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(response) => sanitizedResponse`.
#[wasm_bindgen(js_name = "scopeRegisterLlmSanitizeResponseGuardrail")]
pub fn scope_register_llm_sanitize_response_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_llm_sanitize_response_guardrail(
        &uuid,
        name,
        priority,
        callable::wrap_js_llm_response_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local LLM sanitize-response guardrail by name.
///
/// Returns `true` if the guardrail was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterLlmSanitizeResponseGuardrail")]
pub fn scope_deregister_llm_sanitize_response_guardrail(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_llm_sanitize_response_guardrail(&uuid, name)
        .map_err(to_js_err)
}

/// Registers a scope-local guardrail that conditionally gates LLM execution.
///
/// The guardrail function returns `null` to allow execution or a rejection
/// reason string to block it.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique guardrail name.
/// - `priority` - Execution priority (lower runs first).
/// - `guardrail` - JS function `(request) => string | null`.
#[wasm_bindgen(js_name = "scopeRegisterLlmConditionalExecutionGuardrail")]
pub fn scope_register_llm_conditional_execution_guardrail(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_llm_conditional_execution_guardrail(
        &uuid,
        name,
        priority,
        callable::wrap_js_llm_conditional_fn(guardrail),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local LLM conditional-execution guardrail by name.
///
/// Returns `true` if the guardrail was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterLlmConditionalExecutionGuardrail")]
pub fn scope_deregister_llm_conditional_execution_guardrail(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_llm_conditional_execution_guardrail(
        &uuid, name,
    )
    .map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — LLM
// ---------------------------------------------------------------------------

/// Registers a scope-local intercept that transforms LLM request data (`LLMRequest`).
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(request) => transformedRequest`.
#[wasm_bindgen(js_name = "scopeRegisterLlmRequestIntercept")]
pub fn scope_register_llm_request_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    break_chain: bool,
    func: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_llm_request_intercept(
        &uuid,
        name,
        priority,
        break_chain,
        callable::wrap_js_llm_request_intercept_fn(func),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local LLM request intercept by name.
///
/// Returns `true` if the intercept was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterLlmRequestIntercept")]
pub fn scope_deregister_llm_request_intercept(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_llm_request_intercept(&uuid, name)
        .map_err(to_js_err)
}

/// Registers a scope-local LLM execution intercept following the middleware chain pattern.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `exec_fn` - JS function `(native, next) => result | Promise<result>` -- intercept function.
///   Call `await next(native)` to invoke the next intercept or original implementation.
#[wasm_bindgen(js_name = "scopeRegisterLlmExecutionIntercept")]
pub fn scope_register_llm_execution_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    exec_fn: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_llm_execution_intercept(
        &uuid,
        name,
        priority,
        callable::wrap_js_llm_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local LLM execution intercept by name.
///
/// Returns `true` if the intercept was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterLlmExecutionIntercept")]
pub fn scope_deregister_llm_execution_intercept(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_llm_execution_intercept(&uuid, name)
        .map_err(to_js_err)
}

/// Registers a scope-local streaming LLM execution intercept following the middleware chain pattern.
///
/// The execution function result is wrapped into a single-item stream internally.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `exec_fn` - JS function `(native, next) => result | Promise<result>` -- intercept function.
///   Call `await next(native)` to invoke the next intercept or original streaming implementation.
#[wasm_bindgen(js_name = "scopeRegisterLlmStreamExecutionIntercept")]
pub fn scope_register_llm_stream_execution_intercept(
    scope_uuid: &str,
    name: &str,
    priority: i32,
    exec_fn: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_llm_stream_execution_intercept(
        &uuid,
        name,
        priority,
        callable::wrap_js_llm_stream_exec_intercept_fn(exec_fn),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local LLM stream execution intercept by name.
///
/// Returns `true` if the intercept was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterLlmStreamExecutionIntercept")]
pub fn scope_deregister_llm_stream_execution_intercept(
    scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_llm_stream_execution_intercept(&uuid, name)
        .map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registrations
// ---------------------------------------------------------------------------

/// Registers a scope-local event subscriber that receives lifecycle events
/// for the specified scope.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique subscriber name.
/// - `callback` - JS function `(event) => void` called for each event.
#[wasm_bindgen(js_name = "scopeRegisterSubscriber")]
pub fn scope_register_subscriber(
    scope_uuid: &str,
    name: &str,
    callback: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_register_subscriber(
        &uuid,
        name,
        callable::wrap_js_event_subscriber(callback),
    )
    .map_err(to_js_err)
}

/// Removes a scope-local event subscriber by name.
///
/// Returns `true` if the subscriber was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterSubscriber")]
pub fn scope_deregister_subscriber(scope_uuid: &str, name: &str) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    nvidia_nat_nexus_core::nat_nexus_scope_deregister_subscriber(&uuid, name).map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Creates a new isolated scope stack.
#[wasm_bindgen(js_name = "createScopeStack")]
pub fn create_scope_stack() -> WasmScopeStack {
    WasmScopeStack {
        inner: nvidia_nat_nexus_core::create_scope_stack(),
    }
}

/// Returns the current thread's scope stack handle.
#[wasm_bindgen(js_name = "currentScopeStack")]
pub fn current_scope_stack() -> WasmScopeStack {
    WasmScopeStack {
        inner: nvidia_nat_nexus_core::current_scope_stack(),
    }
}

/// Binds a scope stack to the current thread.
#[wasm_bindgen(js_name = "setThreadScopeStack")]
pub fn set_thread_scope_stack(stack: &WasmScopeStack) {
    nvidia_nat_nexus_core::set_thread_scope_stack(stack.inner.clone());
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `setThreadScopeStack` has been called. Returns `false`
/// when only the auto-created default is present.
#[wasm_bindgen(js_name = "scopeStackActive")]
pub fn scope_stack_active() -> bool {
    nvidia_nat_nexus_core::scope_stack_active()
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Runs the registered tool request intercept chain on the given arguments.
#[wasm_bindgen(js_name = "toolRequestIntercepts")]
pub fn nat_nexus_tool_request_intercepts_wasm(
    name: &str,
    args: JsValue,
) -> Result<JsValue, JsValue> {
    let args_json = js_to_json(&args)?;
    let result = nvidia_nat_nexus_core::nat_nexus_tool_request_intercepts(name, args_json)
        .map_err(to_js_err)?;
    Ok(json_to_js(&result))
}

/// Runs the registered tool conditional execution guardrail chain.
#[wasm_bindgen(js_name = "toolConditionalExecution")]
pub fn nat_nexus_tool_conditional_execution_wasm(name: &str, args: JsValue) -> Result<(), JsValue> {
    let args_json = js_to_json(&args)?;
    nvidia_nat_nexus_core::nat_nexus_tool_conditional_execution(name, &args_json).map_err(to_js_err)
}

/// Runs the registered LLM request intercept chain on the given `LLMRequest`.
#[wasm_bindgen(js_name = "llmRequestIntercepts")]
pub fn nat_nexus_llm_request_intercepts_wasm(
    name: &str,
    request: JsValue,
) -> Result<JsValue, JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: core_types::LLMRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(nvidia_nat_nexus_core::NexusError::Internal(e.to_string())))?;
    let result = nvidia_nat_nexus_core::nat_nexus_llm_request_intercepts(name, llm_request)
        .map_err(to_js_err)?;
    let result_json = serde_json::to_value(&result)
        .map_err(|e| to_js_err(nvidia_nat_nexus_core::NexusError::Internal(e.to_string())))?;
    Ok(json_to_js(&result_json))
}

/// Runs the registered LLM conditional execution guardrail chain.
///
/// - `request` - The LLM request as a JSON value with `{ headers, content }` shape.
#[wasm_bindgen(js_name = "llmConditionalExecution")]
pub fn nat_nexus_llm_conditional_execution_wasm(request: JsValue) -> Result<(), JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: core_types::LLMRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(nvidia_nat_nexus_core::NexusError::Internal(e.to_string())))?;
    nvidia_nat_nexus_core::nat_nexus_llm_conditional_execution(&llm_request).map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// ATIF exporter
// ---------------------------------------------------------------------------

/// ATIF trajectory exporter for collecting events and producing ATIF JSON.
#[wasm_bindgen]
pub struct WasmAtifExporter {
    inner: nvidia_nat_nexus_core::atif::AtifExporter,
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
        let agent_info = nvidia_nat_nexus_core::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: None,
            extra: None,
        };
        Self {
            inner: nvidia_nat_nexus_core::atif::AtifExporter::new(session_id, agent_info),
        }
    }

    /// Registers the exporter as an event subscriber.
    pub fn register(&self, name: &str) -> Result<(), JsValue> {
        let subscriber = self.inner.subscriber();
        nvidia_nat_nexus_core::nat_nexus_register_subscriber(name, subscriber)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deregisters the exporter subscriber.
    pub fn deregister(&self, name: &str) -> Result<bool, JsValue> {
        nvidia_nat_nexus_core::nat_nexus_deregister_subscriber(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Exports collected events as an ATIF trajectory JSON string.
    pub fn export_json(&self) -> Result<String, JsValue> {
        let trajectory = self.inner.export();
        serde_json::to_string(&trajectory).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Clears all collected events.
    pub fn clear(&self) {
        self.inner.clear();
    }
}

/// Returns a default OpenTelemetry config object that can be mutated in JS
/// before constructing `OpenTelemetrySubscriber`.
#[wasm_bindgen(js_name = "defaultOpenTelemetryConfig")]
pub fn default_open_telemetry_config() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&WasmOpenTelemetryConfig::default())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// OpenTelemetry-backed event subscriber.
#[wasm_bindgen(js_name = OpenTelemetrySubscriber)]
pub struct WasmOpenTelemetrySubscriber {
    inner: nvidia_nat_nexus_otel::OpenTelemetrySubscriber,
}

#[wasm_bindgen(js_class = OpenTelemetrySubscriber)]
impl WasmOpenTelemetrySubscriber {
    /// Creates a new OpenTelemetry subscriber from a config object.
    ///
    /// Expected object shape:
    /// `{ transport, endpoint, headers, resource_attributes, service_name,
    /// service_namespace, service_version, instrumentation_scope, timeout_millis }`
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<JsValue>) -> Result<WasmOpenTelemetrySubscriber, JsValue> {
        let config = match config {
            Some(value) if !value.is_undefined() && !value.is_null() => Some(
                serde_wasm_bindgen::from_value(value)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            ),
            _ => None,
        };

        let inner = nvidia_nat_nexus_otel::OpenTelemetrySubscriber::new(build_otel_config(config)?)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Registers this subscriber globally with the given name.
    pub fn register(&self, name: &str) -> Result<(), JsValue> {
        self.inner
            .register(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deregisters a subscriber by name.
    pub fn deregister(&self, name: &str) -> Result<bool, JsValue> {
        self.inner
            .deregister(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Force a flush of finished spans through the exporter.
    #[wasm_bindgen(js_name = "forceFlush")]
    pub fn force_flush(&self) -> Result<(), JsValue> {
        self.inner
            .force_flush()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Shut down the underlying tracer provider.
    pub fn shutdown(&self) -> Result<(), JsValue> {
        self.inner
            .shutdown()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Returns a default OpenInference config object that can be mutated in JS
/// before constructing `OpenInferenceSubscriber`.
#[wasm_bindgen(js_name = "defaultOpenInferenceConfig")]
pub fn default_open_inference_config() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&WasmOpenInferenceConfig::default())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// OpenInference-backed event subscriber.
#[wasm_bindgen(js_name = OpenInferenceSubscriber)]
pub struct WasmOpenInferenceSubscriber {
    inner: nvidia_nat_nexus_openinference::OpenInferenceSubscriber,
}

#[wasm_bindgen(js_class = OpenInferenceSubscriber)]
impl WasmOpenInferenceSubscriber {
    /// Creates a new OpenInference subscriber from a config object.
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<JsValue>) -> Result<WasmOpenInferenceSubscriber, JsValue> {
        let config = match config {
            Some(value) if !value.is_undefined() && !value.is_null() => Some(
                serde_wasm_bindgen::from_value(value)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            ),
            _ => None,
        };

        let inner = nvidia_nat_nexus_openinference::OpenInferenceSubscriber::new(
            build_openinference_config(config)?,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn register(&self, name: &str) -> Result<(), JsValue> {
        self.inner
            .register(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn deregister(&self, name: &str) -> Result<bool, JsValue> {
        self.inner
            .deregister(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "forceFlush")]
    pub fn force_flush(&self) -> Result<(), JsValue> {
        self.inner
            .force_flush()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn shutdown(&self) -> Result<(), JsValue> {
        self.inner
            .shutdown()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_config_defaults_match_expected_values() {
        let otel_config = WasmOpenTelemetryConfig::default();
        assert_eq!(otel_config.transport.as_deref(), Some("http_binary"));
        assert_eq!(otel_config.service_name.as_deref(), Some("nat-nexus"));
        assert_eq!(
            otel_config.instrumentation_scope.as_deref(),
            Some("nvidia-nat-nexus-otel")
        );
        assert_eq!(otel_config.timeout_millis, Some(3_000));

        let openinference_config = WasmOpenInferenceConfig::default();
        assert_eq!(
            openinference_config.transport.as_deref(),
            Some("http_binary")
        );
        assert_eq!(
            openinference_config.service_name.as_deref(),
            Some("nat-nexus")
        );
        assert_eq!(
            openinference_config.instrumentation_scope.as_deref(),
            Some("nvidia-nat-nexus-openinference")
        );
        assert_eq!(openinference_config.timeout_millis, Some(3_000));
    }

    #[test]
    fn config_builders_accept_explicit_overrides() {
        assert!(build_otel_config(Some(WasmOpenTelemetryConfig {
            transport: Some("grpc".to_string()),
            endpoint: Some("http://localhost:4317".to_string()),
            headers: Some(HashMap::from([(
                "authorization".to_string(),
                "Bearer token".to_string()
            )])),
            resource_attributes: Some(HashMap::from([(
                "deployment.environment".to_string(),
                "test".to_string(),
            )])),
            service_name: Some("demo-agent".to_string()),
            service_namespace: Some("agents".to_string()),
            service_version: Some("1.2.3".to_string()),
            instrumentation_scope: Some("demo-scope".to_string()),
            timeout_millis: Some(1_250),
        }))
        .is_ok());

        assert!(build_openinference_config(Some(WasmOpenInferenceConfig {
            transport: Some("grpc".to_string()),
            endpoint: Some("http://localhost:4317".to_string()),
            headers: Some(HashMap::from([(
                "authorization".to_string(),
                "Bearer token".to_string()
            )])),
            resource_attributes: Some(HashMap::from([(
                "deployment.environment".to_string(),
                "test".to_string(),
            )])),
            service_name: Some("demo-agent".to_string()),
            service_namespace: Some("agents".to_string()),
            service_version: Some("1.2.3".to_string()),
            instrumentation_scope: Some("demo-scope".to_string()),
            timeout_millis: Some(1_250),
        }))
        .is_ok());
    }

    #[test]
    fn wasm_atif_exporter_exports_full_trajectory_without_root_parameter() {
        let exporter = WasmAtifExporter::new(
            "session-wasm".to_string(),
            "test-agent".to_string(),
            "1.0.0".to_string(),
            Some("demo-model".to_string()),
        );
        let export: serde_json::Value =
            serde_json::from_str(&exporter.export_json().unwrap()).unwrap();
        assert_eq!(export["session_id"], "session-wasm");
        assert_eq!(export["agent"]["name"], "test-agent");
        assert!(export["steps"].as_array().unwrap().is_empty());

        exporter.clear();
        let cleared: serde_json::Value =
            serde_json::from_str(&exporter.export_json().unwrap()).unwrap();
        assert!(cleared["steps"].as_array().unwrap().is_empty());
    }
}
