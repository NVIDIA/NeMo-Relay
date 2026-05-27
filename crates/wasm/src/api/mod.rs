// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level NeMo Relay API functions exposed to JavaScript via `wasm_bindgen`.
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
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use js_sys::{Function, Reflect};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;

use nemo_relay::api::llm as relay_llm_api;
use nemo_relay::api::llm::{LlmAttributes, LlmRequest as CoreLlmRequest};
use nemo_relay::api::registry as relay_registry_api;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use nemo_relay::api::runtime::{
    TASK_SCOPE_STACK, create_scope_stack as create_scope_stack_handle,
    current_scope_stack as current_scope_stack_handle, scope_stack_active as scope_stack_is_active,
    set_thread_scope_stack as bind_thread_scope_stack, task_scope_top,
};
use nemo_relay::api::scope as relay_scope_api;
use nemo_relay::api::scope::{ScopeAttributes, ScopeHandle as CoreScopeHandle};
use nemo_relay::api::subscriber as relay_subscriber_api;
use nemo_relay::api::tool as relay_tool_api;
use nemo_relay::api::tool::ToolAttributes;
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError,
    PluginRegistration as ComponentRegistration, PluginRegistrationContext,
    active_plugin_report as active_plugin_report_impl,
    clear_plugin_configuration as clear_plugin_configuration_impl,
    deregister_plugin as deregister_plugin_impl, initialize_plugins as initialize_plugins_impl,
    list_plugin_kinds as list_plugin_kinds_impl, register_plugin as register_plugin_impl,
    validate_plugin_config as validate_plugin_config_impl,
};
use nemo_relay_adaptive::plugin_component::register_adaptive_component;

use crate::callable;
use crate::convert::{
    js_to_json, json_to_js, opt_js_to_json, opt_js_to_timestamp_micros, to_js_err,
};
use crate::stream::LlmStream;
#[cfg(test)]
pub use crate::types::{LLM_STATEFUL, LLM_STREAMING, SCOPE_PARALLEL, TOOL_REMOTE};
use crate::types::{LlmHandle, ScopeHandle, ScopeStack, ScopeType, ToolHandle};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
            service_name: Some("nemo-relay".to_string()),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: Some("nemo-relay-otel".to_string()),
            timeout_millis: Some(3_000),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
            service_name: Some("nemo-relay".to_string()),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: Some("nemo-relay-openinference".to_string()),
            timeout_millis: Some(3_000),
        }
    }
}

#[wasm_bindgen(typescript_custom_section)]
const TYPESCRIPT_SHARED_DECLARATIONS: &str = r#"
/** One JSON scalar value accepted by the root WebAssembly API. */
export type JsonPrimitive = string | number | boolean | null;
/** A JSON object with recursively JSON-serializable values. */
export interface JsonObject {
  [key: string]: Json;
}
/** A JSON array with recursively JSON-serializable values. */
export interface JsonArray extends Array<Json> {}
/** Any JSON-serializable value accepted by the root WebAssembly API. */
export type Json = JsonPrimitive | JsonObject | JsonArray;

/** Mutable configuration object for `OpenTelemetrySubscriber`. */
export interface OpenTelemetryConfig {
  transport?: string | null;
  endpoint?: string | null;
  headers?: Record<string, string> | null;
  resourceAttributes?: Record<string, string> | null;
  serviceName?: string | null;
  serviceNamespace?: string | null;
  serviceVersion?: string | null;
  instrumentationScope?: string | null;
  timeoutMillis?: number | null;
}

/** Mutable configuration object for `OpenInferenceSubscriber`. */
export interface OpenInferenceConfig {
  transport?: string | null;
  endpoint?: string | null;
  headers?: Record<string, string> | null;
  resourceAttributes?: Record<string, string> | null;
  serviceName?: string | null;
  serviceNamespace?: string | null;
  serviceVersion?: string | null;
  instrumentationScope?: string | null;
  timeoutMillis?: number | null;
}
"#;

fn clone_scope_handle_arg(handle: &JsValue) -> Result<Option<CoreScopeHandle>, JsValue> {
    if handle.is_null() || handle.is_undefined() {
        return Ok(None);
    }

    // Preferred path would be `handle.clone().dyn_into::<ScopeHandle>()`, but
    // wasm-bindgen does not currently implement `JsCast` for this exported
    // class in this crate, so the safe fallback here is to rebuild the core
    // handle from the JS-visible fields instead of using raw pointer access.
    let conversion_err = || JsValue::from_str("expected instance of ScopeHandle");
    let get_field =
        |name: &str| Reflect::get(handle, &JsValue::from_str(name)).map_err(|_| conversion_err());

    let uuid = get_field("uuid")?
        .as_string()
        .ok_or_else(conversion_err)
        .and_then(|uuid| Uuid::parse_str(&uuid).map_err(|_| conversion_err()))?;
    let name = get_field("name")?.as_string().ok_or_else(conversion_err)?;
    let scope_type = match get_field("scopeType")?
        .as_f64()
        .ok_or_else(conversion_err)? as u32
    {
        0 => ScopeType::Agent,
        1 => ScopeType::Function,
        2 => ScopeType::Tool,
        3 => ScopeType::Llm,
        4 => ScopeType::Retriever,
        5 => ScopeType::Embedder,
        6 => ScopeType::Reranker,
        7 => ScopeType::Guardrail,
        8 => ScopeType::Evaluator,
        9 => ScopeType::Custom,
        10 => ScopeType::Unknown,
        _ => return Err(conversion_err()),
    };
    let attributes = ScopeAttributes::from_bits_truncate(
        get_field("attributes")?
            .as_f64()
            .ok_or_else(conversion_err)? as u32,
    );
    let parent_uuid = match get_field("parentUuid")? {
        value if value.is_null() || value.is_undefined() => None,
        value => Some(
            value
                .as_string()
                .ok_or_else(conversion_err)
                .and_then(|uuid| Uuid::parse_str(&uuid).map_err(|_| conversion_err()))?,
        ),
    };
    let data = opt_js_to_json(&get_field("data")?)?;
    let metadata = opt_js_to_json(&get_field("metadata")?)?;

    Ok(Some(
        CoreScopeHandle::builder()
            .uuid(uuid)
            .scope_type(scope_type.into())
            .name(name)
            .data_opt(data)
            .metadata_opt(metadata)
            .attributes(attributes)
            .parent_uuid_opt(parent_uuid)
            .build(),
    ))
}

fn build_otel_config(
    config: Option<WasmOpenTelemetryConfig>,
) -> Result<nemo_relay::observability::otel::OpenTelemetryConfig, JsValue> {
    let config = config.unwrap_or_default();
    let transport = config
        .transport
        .unwrap_or_else(|| "http_binary".to_string());
    let service_name = config
        .service_name
        .unwrap_or_else(|| "nemo-relay".to_string());
    let instrumentation_scope = config
        .instrumentation_scope
        .unwrap_or_else(|| "nemo-relay-otel".to_string());
    let timeout_millis = config.timeout_millis.unwrap_or(3_000);

    let mut otel_config = match transport.as_str() {
        "http_binary" => {
            nemo_relay::observability::otel::OpenTelemetryConfig::http_binary(service_name)
        }
        "grpc" => nemo_relay::observability::otel::OpenTelemetryConfig::grpc(service_name),
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
) -> Result<nemo_relay::observability::openinference::OpenInferenceConfig, JsValue> {
    let config = config.unwrap_or_default();
    let transport = config
        .transport
        .unwrap_or_else(|| "http_binary".to_string());
    let service_name = config
        .service_name
        .unwrap_or_else(|| "nemo-relay".to_string());
    let instrumentation_scope = config
        .instrumentation_scope
        .unwrap_or_else(|| "nemo-relay-openinference".to_string());
    let timeout_millis = config.timeout_millis.unwrap_or(3_000);

    let transport = match transport.as_str() {
        "http_binary" => nemo_relay::observability::openinference::OtlpTransport::HttpBinary,
        "grpc" => nemo_relay::observability::openinference::OtlpTransport::Grpc,
        other => {
            return Err(JsValue::from_str(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}",
            )));
        }
    };

    let mut openinference_config =
        nemo_relay::observability::openinference::OpenInferenceConfig::new()
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
pub fn get_handle() -> Result<ScopeHandle, JsValue> {
    relay_scope_api::get_handle()
        .map(ScopeHandle::from)
        .map_err(to_js_err)
}

/// Pushes a new scope onto the scope stack and returns its handle.
///
/// - `name` - Human-readable scope name.
/// - `scope_type` - Scope type enum value.
/// - `parent` - Optional parent scope handle; uses the current top if omitted.
/// - `attributes` - Optional bitfield of scope attribute flags.
/// - `data` - Optional JSON application payload stored on the scope handle.
/// - `metadata` - Optional JSON metadata payload recorded on the start event.
/// - `input` - Optional semantic JSON payload exported on the scope start event.
/// - `timestamp` - Optional Unix microseconds timestamp recorded as the handle
///   start time and start event timestamp. Must be a safe integer number; omitted
///   values use the current runtime time.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "pushScope")]
pub fn push_scope(
    name: &str,
    #[wasm_bindgen(js_name = "scopeType")] scope_type: ScopeType,
    #[wasm_bindgen(unchecked_param_type = "ScopeHandle | null | undefined")] handle: JsValue,
    attributes: Option<u32>,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] input: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number | null | undefined")] timestamp: Option<f64>,
) -> Result<ScopeHandle, JsValue> {
    let attrs = ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let handle = clone_scope_handle_arg(&handle)?;
    let timestamp = opt_js_to_timestamp_micros(timestamp)?;
    relay_scope_api::push_scope(
        relay_scope_api::PushScopeParams::builder()
            .name(name)
            .scope_type(scope_type.into())
            .parent_opt(handle.as_ref())
            .attributes(attrs)
            .data_opt(opt_js_to_json(&data)?)
            .metadata_opt(opt_js_to_json(&metadata)?)
            .input_opt(opt_js_to_json(&input)?)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map(ScopeHandle::from)
    .map_err(to_js_err)
}

/// Pops the scope identified by `handle` from the scope stack.
/// Optional `output` is a semantic JSON payload exported on the scope end event.
/// Optional `timestamp` is a Unix microseconds timestamp recorded on the scope end event.
/// It must be a safe integer number; omitted values use the runtime default end timestamp.
///
/// Throws if the handle does not match the current top of the stack.
#[wasm_bindgen(js_name = "popScope")]
pub fn pop_scope(
    handle: &ScopeHandle,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] output: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number | null | undefined")] timestamp: Option<f64>,
) -> Result<(), JsValue> {
    let timestamp = opt_js_to_timestamp_micros(timestamp)?;
    relay_scope_api::pop_scope(
        relay_scope_api::PopScopeParams::builder()
            .handle_uuid(&handle.inner.uuid)
            .output_opt(opt_js_to_json(&output)?)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_js_err)
}

/// Returns the most recent callback error that could not be surfaced through a direct exception.
#[wasm_bindgen(
    js_name = "getLastCallbackError",
    unchecked_return_type = "string | null"
)]
pub fn get_last_callback_error() -> JsValue {
    match crate::convert::get_last_callback_error() {
        Some(error) => error
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL),
        None => JsValue::NULL,
    }
}

/// Clears the most recent callback error recorded by the WebAssembly binding.
#[wasm_bindgen(js_name = "clearLastCallbackError")]
pub fn clear_last_callback_error() {
    crate::convert::clear_last_callback_error();
}

/// Pushes a scope, invokes the callback, then pops the scope automatically.
///
/// Creates a child scope with the given `name` and `scope_type`, calls the
/// `callback` with a `ScopeHandle`, and guarantees the scope is popped
/// when the callback returns (whether normally or by throwing). If the callback
/// returns a `Promise`, the scope is popped after the Promise settles.
///
/// - `name` - Human-readable scope name.
/// - `scope_type` - Scope type enum value.
/// - `callback` - A JS function `(handle) => result` or `(handle) => Promise<result>`.
/// - `parent` - Optional parent scope handle; uses the current top if omitted.
/// - `attributes` - Optional bitfield of scope attribute flags.
/// - `data` - Optional JSON application data payload.
/// - `metadata` - Optional JSON metadata payload.
/// - `input` - Optional semantic JSON payload exported on the scope start event.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "withScope", unchecked_return_type = "Promise<unknown>")]
pub fn with_scope(
    name: &str,
    #[wasm_bindgen(js_name = "scopeType")] scope_type: ScopeType,
    #[wasm_bindgen(unchecked_param_type = "(handle: ScopeHandle) => any")] callback: &Function,
    #[wasm_bindgen(unchecked_param_type = "ScopeHandle | null | undefined")] handle: JsValue,
    attributes: Option<u32>,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] input: JsValue,
) -> Result<js_sys::Promise, JsValue> {
    let attrs = ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let handle = clone_scope_handle_arg(&handle)?;
    let scope_handle = relay_scope_api::push_scope(
        relay_scope_api::PushScopeParams::builder()
            .name(name)
            .scope_type(scope_type.into())
            .parent_opt(handle.as_ref())
            .attributes(attrs)
            .data_opt(opt_js_to_json(&data)?)
            .metadata_opt(opt_js_to_json(&metadata)?)
            .input_opt(opt_js_to_json(&input)?)
            .build(),
    )
    .map(ScopeHandle::from)
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
                let _ = relay_scope_api::pop_scope(
                    relay_scope_api::PopScopeParams::builder()
                        .handle_uuid(&then_uuid)
                        .build(),
                );
                resolved
            });

            let catch_uuid = scope_uuid;
            let catch_cb = Closure::once(move |rejected: JsValue| -> JsValue {
                let _ = relay_scope_api::pop_scope(
                    relay_scope_api::PopScopeParams::builder()
                        .handle_uuid(&catch_uuid)
                        .build(),
                );
                // Re-throw by returning a rejected promise
                js_sys::Promise::reject(&rejected).into()
            });

            // Chain .then(onFulfilled, onRejected) via JS interop.
            let then_fn: Function = then_cb.into_js_value().unchecked_into();
            let catch_fn: Function = catch_cb.into_js_value().unchecked_into();
            let then_method: Function =
                js_sys::Reflect::get(&promise, &"then".into())?.unchecked_into();
            let chained = then_method.call2(&promise, &then_fn, &catch_fn)?;
            Ok(chained.unchecked_into())
        }
        Ok(val) => {
            // Synchronous return — pop immediately.
            let _ = relay_scope_api::pop_scope(
                relay_scope_api::PopScopeParams::builder()
                    .handle_uuid(&scope_uuid)
                    .build(),
            );
            Ok(js_sys::Promise::resolve(&val))
        }
        Err(err) => {
            // Callback threw — pop and propagate the error.
            let _ = relay_scope_api::pop_scope(
                relay_scope_api::PopScopeParams::builder()
                    .handle_uuid(&scope_uuid)
                    .build(),
            );
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
/// - `timestamp` - Optional Unix microseconds timestamp recorded on the mark event.
///   Must be a safe integer number; omitted values use the current runtime time.
#[wasm_bindgen(js_name = "event")]
pub fn event(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "ScopeHandle | null | undefined")] handle: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number | null | undefined")] timestamp: Option<f64>,
) -> Result<(), JsValue> {
    let handle = clone_scope_handle_arg(&handle)?;
    let timestamp = opt_js_to_timestamp_micros(timestamp)?;
    relay_scope_api::event(
        relay_scope_api::EmitMarkEventParams::builder()
            .name(name)
            .parent_opt(handle.as_ref())
            .data_opt(opt_js_to_json(&data)?)
            .metadata_opt(opt_js_to_json(&metadata)?)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begins a tool call, returning a `ToolHandle` for the active invocation.
///
/// Applies sanitize-request guardrails to the emitted start-event payload before
/// returning. Request and execution intercepts run only through `toolCallExecute`.
///
/// - `name` - Tool name.
/// - `args` - JSON arguments to the tool. These become the start-event data
///   after sanitize-request guardrails.
/// - `parent` - Optional parent scope handle; uses the current top if omitted.
/// - `attributes` - Optional bitfield of tool attribute flags.
/// - `data` - Optional JSON application payload stored on the tool handle.
/// - `metadata` - Optional JSON metadata payload recorded on the start event.
/// - `tool_call_id` - Optional provider correlation ID recorded in the tool
///   event category profile.
/// - `timestamp` - Optional Unix microseconds timestamp recorded as the handle
///   start time and start event timestamp. Must be a safe integer number; omitted
///   values use the current runtime time.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "toolCall")]
pub fn tool_call(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "Json")] args: JsValue,
    #[wasm_bindgen(unchecked_param_type = "ScopeHandle | null | undefined")] handle: JsValue,
    attributes: Option<u32>,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(
        js_name = "toolCallId",
        unchecked_param_type = "string | null | undefined"
    )]
    tool_call_id: Option<String>,
    #[wasm_bindgen(unchecked_param_type = "number | null | undefined")] timestamp: Option<f64>,
) -> Result<ToolHandle, JsValue> {
    let args_json = js_to_json(&args)?;
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let handle = clone_scope_handle_arg(&handle)?;
    let timestamp = opt_js_to_timestamp_micros(timestamp)?;
    relay_tool_api::tool_call(
        relay_tool_api::ToolCallParams::builder()
            .name(name)
            .args(args_json)
            .parent_opt(handle.as_ref())
            .attributes(attrs)
            .data_opt(opt_js_to_json(&data)?)
            .metadata_opt(opt_js_to_json(&metadata)?)
            .tool_call_id_opt(tool_call_id)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map(ToolHandle::from)
    .map_err(to_js_err)
}

/// Ends an active tool call, applying sanitize-response guardrails.
///
/// Response intercepts run only through `toolCallExecute`.
///
/// - `handle` - The tool handle returned by `toolCall`.
/// - `result` - JSON result of the tool execution. This becomes the end-event
///   data after sanitize-response guardrails unless it sanitizes to JSON null.
/// - `data` - Optional JSON data payload used when the sanitized result is JSON null.
/// - `metadata` - Optional JSON metadata payload recorded on the end event.
/// - `timestamp` - Optional Unix microseconds timestamp recorded on the tool end
///   event. Must be a safe integer number; omitted values use the runtime
///   default end timestamp.
#[wasm_bindgen(js_name = "toolCallEnd")]
pub fn tool_call_end(
    handle: &ToolHandle,
    #[wasm_bindgen(unchecked_param_type = "Json")] result: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number | null | undefined")] timestamp: Option<f64>,
) -> Result<(), JsValue> {
    let result_json = js_to_json(&result)?;
    let timestamp = opt_js_to_timestamp_micros(timestamp)?;
    relay_tool_api::tool_call_end(
        relay_tool_api::ToolCallEndParams::builder()
            .handle(&handle.inner)
            .result(result_json)
            .data_opt(opt_js_to_json(&data)?)
            .metadata_opt(opt_js_to_json(&metadata)?)
            .timestamp_opt(timestamp)
            .build(),
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
#[wasm_bindgen(js_name = "toolCallExecute", unchecked_return_type = "unknown")]
pub async fn tool_call_execute(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "Json")] args: JsValue,
    #[wasm_bindgen(unchecked_param_type = "(arg: Json) => any")] func: Function,
    #[wasm_bindgen(unchecked_param_type = "ScopeHandle | null | undefined")] handle: JsValue,
    attributes: Option<u32>,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
) -> Result<JsValue, JsValue> {
    let args_json = js_to_json(&args)?;
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent_handle = clone_scope_handle_arg(&handle)?.unwrap_or_else(task_scope_top);
    let exec_fn = callable::wrap_js_tool_exec_fn(func);
    let default_fn: ToolExecutionNextFn = Arc::new(move |args| exec_fn(args));

    let scope_stack = current_scope_stack_handle();
    let data_json = opt_js_to_json(&data)?;
    let metadata_json = opt_js_to_json(&metadata)?;
    let result = TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            relay_tool_api::tool_call_execute(
                relay_tool_api::ToolCallExecuteParams::builder()
                    .name(name)
                    .args(args_json)
                    .func(default_fn)
                    .parent(parent_handle)
                    .attributes(attrs)
                    .data_opt(data_json)
                    .metadata_opt(metadata_json)
                    .build(),
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

/// Begins an LLM call, returning a `LlmHandle` for the active invocation.
///
/// Applies sanitize-request guardrails to the emitted start-event payload before
/// returning. Request and execution intercepts run only through `llmCallExecute`.
///
/// - `name` - LLM provider/model name.
/// - `request` - The LLM request as a JSON value with `{ headers, content }`
///   shape. This becomes the start-event data after sanitize-request guardrails.
/// - `parent` - Optional parent scope handle; uses the current top if omitted.
/// - `attributes` - Optional bitfield of LLM attribute flags.
/// - `data` - Optional JSON application payload stored on the LLM handle.
/// - `metadata` - Optional JSON metadata payload recorded on the start event.
/// - `model_name` - Optional model name string recorded in the LLM event
///   category profile.
/// - `timestamp` - Optional Unix microseconds timestamp recorded as the handle
///   start time and start event timestamp. Must be a safe integer number; omitted
///   values use the current runtime time.
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
#[wasm_bindgen(js_name = "llmCall")]
pub fn llm_call(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "Json")] request: JsValue,
    #[wasm_bindgen(unchecked_param_type = "ScopeHandle | null | undefined")] handle: JsValue,
    attributes: Option<u32>,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(
        js_name = "modelName",
        unchecked_param_type = "string | null | undefined"
    )]
    model_name: Option<String>,
    #[wasm_bindgen(unchecked_param_type = "number | null | undefined")] timestamp: Option<f64>,
) -> Result<LlmHandle, JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: CoreLlmRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(FlowError::Internal(e.to_string())))?;
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let handle = clone_scope_handle_arg(&handle)?;
    let timestamp = opt_js_to_timestamp_micros(timestamp)?;
    let params = relay_llm_api::LlmCallParams::builder()
        .name(name)
        .request(&llm_request)
        .parent_opt(handle.as_ref())
        .attributes(attrs)
        .data_opt(opt_js_to_json(&data)?)
        .metadata_opt(opt_js_to_json(&metadata)?)
        .model_name_opt(model_name)
        .timestamp_opt(timestamp)
        .build();
    relay_llm_api::llm_call(params)
        .map(LlmHandle::from)
        .map_err(to_js_err)
}

/// Ends an active LLM call, applying sanitize-response guardrails.
///
/// Response intercepts run only through `llmCallExecute`.
///
/// - `handle` - The LLM handle returned by `llmCall`.
/// - `response` - JSON response from the LLM. This becomes the end-event data
///   after sanitize-response guardrails unless it sanitizes to JSON null.
/// - `data` - Optional JSON data payload used when the sanitized response is JSON null.
/// - `metadata` - Optional JSON metadata payload recorded on the end event.
/// - `timestamp` - Optional Unix microseconds timestamp recorded on the LLM end
///   event. Must be a safe integer number; omitted values use the runtime
///   default end timestamp.
#[wasm_bindgen(js_name = "llmCallEnd")]
pub fn llm_call_end(
    handle: &LlmHandle,
    #[wasm_bindgen(unchecked_param_type = "Json")] response: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number | null | undefined")] timestamp: Option<f64>,
) -> Result<(), JsValue> {
    let response_json = js_to_json(&response)?;
    let timestamp = opt_js_to_timestamp_micros(timestamp)?;
    relay_llm_api::llm_call_end(
        relay_llm_api::LlmCallEndParams::builder()
            .handle(&handle.inner)
            .response(response_json)
            .data_opt(opt_js_to_json(&data)?)
            .metadata_opt(opt_js_to_json(&metadata)?)
            .timestamp_opt(timestamp)
            .build(),
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
/// - `codec_decode` - Optional JS decode function for annotated-aware request intercepts.
/// - `codec_encode` - Optional JS encode function for annotated-aware request intercepts.
/// - `response_codec_decode` - Optional JS decode function used to attach
///   annotated response data to emitted end events.
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
#[wasm_bindgen(js_name = "llmCallExecute", unchecked_return_type = "unknown")]
pub async fn llm_call_execute(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "Json")] request: JsValue,
    #[wasm_bindgen(unchecked_param_type = "(arg: Json) => any")] func: Function,
    #[wasm_bindgen(unchecked_param_type = "ScopeHandle | null | undefined")] handle: JsValue,
    attributes: Option<u32>,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(
        js_name = "modelName",
        unchecked_param_type = "string | null | undefined"
    )]
    model_name: Option<String>,
    #[wasm_bindgen(
        js_name = "codecDecode",
        unchecked_param_type = "((arg: Json) => any) | null | undefined"
    )]
    codec_decode: Option<Function>,
    #[wasm_bindgen(
        js_name = "codecEncode",
        unchecked_param_type = "((arg: Json) => any) | null | undefined"
    )]
    codec_encode: Option<Function>,
    #[wasm_bindgen(
        js_name = "responseCodecDecode",
        unchecked_param_type = "((arg: Json) => any) | null | undefined"
    )]
    response_codec_decode: Option<Function>,
) -> Result<JsValue, JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: CoreLlmRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(FlowError::Internal(e.to_string())))?;
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent_handle = clone_scope_handle_arg(&handle)?.unwrap_or_else(task_scope_top);
    let exec_fn = callable::wrap_js_llm_exec_fn(func);
    let default_fn: LlmExecutionNextFn = Arc::new(move |request| exec_fn(request));
    let codec = match (codec_decode, codec_encode) {
        (Some(d), Some(e)) => Some(callable::wrap_js_codec(d, e)),
        _ => None,
    };
    let response_codec = response_codec_decode.map(callable::wrap_js_response_codec);

    let scope_stack = current_scope_stack_handle();
    let data_json = opt_js_to_json(&data)?;
    let metadata_json = opt_js_to_json(&metadata)?;
    let result = TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            let params = relay_llm_api::LlmCallExecuteParams::builder()
                .name(name)
                .request(llm_request)
                .func(default_fn)
                .parent(parent_handle)
                .attributes(attrs)
                .data_opt(data_json)
                .metadata_opt(metadata_json)
                .model_name_opt(model_name)
                .codec_opt(codec)
                .response_codec_opt(response_codec)
                .build();
            relay_llm_api::llm_call_execute(params).await
        })
        .await
        .map_err(to_js_err)?;

    Ok(json_to_js(&result))
}

/// Executes a streaming LLM call lifecycle through the middleware pipeline.
///
/// Like `llmCallExecute`, conditional-execution guardrails run first on the raw
/// request. Returns a `LlmStream` whose `next()` method yields response
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
/// - `codec_decode` - Optional JS decode function for annotated-aware request intercepts.
/// - `codec_encode` - Optional JS encode function for annotated-aware request intercepts.
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
#[wasm_bindgen(js_name = "llmStreamCallExecute")]
pub async fn llm_stream_call_execute(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "Json")] request: JsValue,
    #[wasm_bindgen(unchecked_param_type = "(arg: Json) => any")] func: Function,
    #[wasm_bindgen(unchecked_param_type = "((arg: Json) => any) | null | undefined")]
    collector: Option<Function>,
    #[wasm_bindgen(unchecked_param_type = "(() => any) | null | undefined")] finalizer: Option<
        Function,
    >,
    #[wasm_bindgen(unchecked_param_type = "ScopeHandle | null | undefined")] handle: JsValue,
    attributes: Option<u32>,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Json | null | undefined")] metadata: JsValue,
    #[wasm_bindgen(
        js_name = "modelName",
        unchecked_param_type = "string | null | undefined"
    )]
    model_name: Option<String>,
    #[wasm_bindgen(
        js_name = "codecDecode",
        unchecked_param_type = "((arg: Json) => any) | null | undefined"
    )]
    codec_decode: Option<Function>,
    #[wasm_bindgen(
        js_name = "codecEncode",
        unchecked_param_type = "((arg: Json) => any) | null | undefined"
    )]
    codec_encode: Option<Function>,
    #[wasm_bindgen(
        js_name = "responseCodecDecode",
        unchecked_param_type = "((arg: Json) => any) | null | undefined"
    )]
    response_codec_decode: Option<Function>,
) -> Result<LlmStream, JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: CoreLlmRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(FlowError::Internal(e.to_string())))?;
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent_handle = clone_scope_handle_arg(&handle)?.unwrap_or_else(task_scope_top);
    let exec_fn = callable::wrap_js_llm_exec_fn(func);

    let wrapped_collector: Box<dyn FnMut(serde_json::Value) -> FlowResult<()> + Send> =
        match collector {
            Some(cb) => callable::wrap_js_collector_fn(cb),
            None => Box::new(|_: serde_json::Value| Ok(())),
        };

    let wrapped_finalizer: Box<dyn FnOnce() -> serde_json::Value + Send> = match finalizer {
        Some(cb) => callable::wrap_js_finalizer_fn(cb),
        None => Box::new(|| serde_json::Value::Null),
    };

    // Bridge LlmExecutionFn -> LlmStreamExecutionNextFn
    let default_fn: LlmStreamExecutionNextFn = Arc::new(move |request| {
        let fut = exec_fn(request);
        Box::pin(async move {
            let result = fut.await?;
            let stream = tokio_stream::once(Ok(result));
            Ok(Box::pin(stream)
                as std::pin::Pin<
                    Box<dyn tokio_stream::Stream<Item = FlowResult<serde_json::Value>> + Send>,
                >)
        })
    });

    let codec = match (codec_decode, codec_encode) {
        (Some(d), Some(e)) => Some(callable::wrap_js_codec(d, e)),
        _ => None,
    };
    let response_codec = response_codec_decode.map(callable::wrap_js_response_codec);
    let scope_stack = current_scope_stack_handle();
    let data_json = opt_js_to_json(&data)?;
    let metadata_json = opt_js_to_json(&metadata)?;
    let rust_stream = TASK_SCOPE_STACK
        .scope(scope_stack, async move {
            let params = relay_llm_api::LlmStreamCallExecuteParams::builder()
                .name(name)
                .request(llm_request)
                .func(default_fn)
                .collector(wrapped_collector)
                .finalizer(wrapped_finalizer)
                .parent(parent_handle)
                .attributes(attrs)
                .data_opt(data_json)
                .metadata_opt(metadata_json)
                .model_name_opt(model_name)
                .codec_opt(codec)
                .response_codec_opt(response_codec)
                .build();
            relay_llm_api::llm_stream_call_execute(params).await
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

    Ok(LlmStream {
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
    #[wasm_bindgen(unchecked_param_type = "(name: string, args: Json) => any")] guardrail: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_tool_sanitize_request_guardrail(
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
    relay_registry_api::deregister_tool_sanitize_request_guardrail(name).map_err(to_js_err)
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
    #[wasm_bindgen(unchecked_param_type = "(name: string, result: Json) => any")]
    guardrail: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_tool_sanitize_response_guardrail(
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
    relay_registry_api::deregister_tool_sanitize_response_guardrail(name).map_err(to_js_err)
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
    #[wasm_bindgen(unchecked_param_type = "(name: string, args: Json) => string | null")] guardrail: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_tool_conditional_execution_guardrail(
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
    relay_registry_api::deregister_tool_conditional_execution_guardrail(name).map_err(to_js_err)
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
    #[wasm_bindgen(js_name = "breakChain")] break_chain: bool,
    #[wasm_bindgen(
        js_name = "callable",
        unchecked_param_type = "(name: string, args: Json) => any"
    )]
    func: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_tool_request_intercept(
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
    relay_registry_api::deregister_tool_request_intercept(name).map_err(to_js_err)
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
    #[wasm_bindgen(
        js_name = "callable",
        unchecked_param_type = "(args: Json, next: (...args: any[]) => any) => any"
    )]
    exec_fn: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_tool_execution_intercept(
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
    relay_registry_api::deregister_tool_execution_intercept(name).map_err(to_js_err)
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
    #[wasm_bindgen(unchecked_param_type = "(request: Json) => any")] guardrail: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_llm_sanitize_request_guardrail(
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
    relay_registry_api::deregister_llm_sanitize_request_guardrail(name).map_err(to_js_err)
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
    #[wasm_bindgen(unchecked_param_type = "(response: Json) => any")] guardrail: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_llm_sanitize_response_guardrail(
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
    relay_registry_api::deregister_llm_sanitize_response_guardrail(name).map_err(to_js_err)
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
    #[wasm_bindgen(unchecked_param_type = "(request: Json) => string | null")] guardrail: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_llm_conditional_execution_guardrail(
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
    relay_registry_api::deregister_llm_conditional_execution_guardrail(name).map_err(to_js_err)
}

// LLM intercepts

/// Registers an intercept that transforms LLM request data (`LlmRequest`).
///
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(request) => transformedRequest`.
#[wasm_bindgen(js_name = "registerLlmRequestIntercept")]
pub fn register_llm_request_intercept(
    name: &str,
    priority: i32,
    #[wasm_bindgen(js_name = "breakChain")] break_chain: bool,
    #[wasm_bindgen(js_name = "callable", unchecked_param_type = "(request: Json) => any")]
    func: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_llm_request_intercept(
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
    relay_registry_api::deregister_llm_request_intercept(name).map_err(to_js_err)
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
    #[wasm_bindgen(
        js_name = "callable",
        unchecked_param_type = "(request: Json, next: (...args: any[]) => any) => any"
    )]
    exec_fn: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_llm_execution_intercept(
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
    relay_registry_api::deregister_llm_execution_intercept(name).map_err(to_js_err)
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
    #[wasm_bindgen(
        js_name = "callable",
        unchecked_param_type = "(request: Json, next: (...args: any[]) => any) => any"
    )]
    exec_fn: Function,
) -> Result<(), JsValue> {
    relay_registry_api::register_llm_stream_execution_intercept(
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
    relay_registry_api::deregister_llm_stream_execution_intercept(name).map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

enum SubscriptionInner {
    Global(relay_subscriber_api::SubscriberHandle),
    Scope(relay_subscriber_api::ScopeSubscriberHandle),
}

fn close_subscription_inner(inner: &SubscriptionInner) -> FlowResult<bool> {
    match inner {
        SubscriptionInner::Global(handle) => handle.close(),
        SubscriptionInner::Scope(handle) => handle.close(),
    }
}

/// Closeable event subscriber registration.
#[wasm_bindgen(js_name = Subscription)]
pub struct Subscription {
    inner: Arc<Mutex<Option<SubscriptionInner>>>,
}

#[wasm_bindgen(js_class = Subscription)]
impl Subscription {
    /// Close this subscription.
    ///
    /// Returns `true` when a live subscriber was removed and `false` when the
    /// subscription was already closed or scope cleanup already removed it.
    #[wasm_bindgen(js_name = close)]
    pub fn close(&self) -> Result<bool, JsValue> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("subscription lock poisoned"))?;
        let Some(inner) = guard.as_ref() else {
            return Ok(false);
        };
        let removed = close_subscription_inner(inner).map_err(to_js_err)?;
        *guard = None;
        Ok(removed)
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        if let Some(inner) = guard.take() {
            let _ = close_subscription_inner(&inner);
        }
    }
}

/// Registers an event subscriber that receives lifecycle events.
///
/// - `name` - Unique subscriber name.
/// - `callback` - JS function `(event) => void` called for each event.
#[wasm_bindgen(js_name = "registerSubscriber")]
pub fn register_subscriber(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "(event: Json) => any")] callback: Function,
) -> Result<(), JsValue> {
    relay_subscriber_api::register_subscriber(name, callable::wrap_js_event_subscriber(callback))
        .map_err(to_js_err)
}

/// Registers an anonymous event subscriber and returns a closeable handle.
///
/// - `callback` - JS function `(event) => void` called for each event.
#[wasm_bindgen(js_name = "subscribe")]
pub fn subscribe(
    #[wasm_bindgen(unchecked_param_type = "(event: Json) => any")] callback: Function,
) -> Result<Subscription, JsValue> {
    relay_subscriber_api::subscribe(callable::wrap_js_event_subscriber(callback))
        .map(|handle| Subscription {
            inner: Arc::new(Mutex::new(Some(SubscriptionInner::Global(handle)))),
        })
        .map_err(to_js_err)
}

/// Removes a previously registered event subscriber by name.
///
/// Returns `true` if the subscriber was found and removed.
#[wasm_bindgen(js_name = "deregisterSubscriber")]
pub fn deregister_subscriber(name: &str) -> Result<bool, JsValue> {
    relay_subscriber_api::deregister_subscriber(name).map_err(to_js_err)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(unchecked_param_type = "(name: string, args: Json) => any")] guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_tool_sanitize_request_guardrail(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_tool_sanitize_request_guardrail(&uuid, name)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(unchecked_param_type = "(name: string, result: Json) => any")]
    guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_tool_sanitize_response_guardrail(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_tool_sanitize_response_guardrail(&uuid, name)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(unchecked_param_type = "(name: string, args: Json) => string | null")] guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_tool_conditional_execution_guardrail(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_tool_conditional_execution_guardrail(&uuid, name)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(js_name = "breakChain")] break_chain: bool,
    #[wasm_bindgen(
        js_name = "callable",
        unchecked_param_type = "(name: string, args: Json) => any"
    )]
    func: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_tool_request_intercept(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_tool_request_intercept(&uuid, name).map_err(to_js_err)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(
        js_name = "callable",
        unchecked_param_type = "(args: Json, next: (...args: any[]) => any) => any"
    )]
    exec_fn: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_tool_execution_intercept(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_tool_execution_intercept(&uuid, name).map_err(to_js_err)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(unchecked_param_type = "(request: Json) => any")] guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_llm_sanitize_request_guardrail(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_llm_sanitize_request_guardrail(&uuid, name)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(unchecked_param_type = "(response: Json) => any")] guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_llm_sanitize_response_guardrail(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_llm_sanitize_response_guardrail(&uuid, name)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(unchecked_param_type = "(request: Json) => string | null")] guardrail: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_llm_conditional_execution_guardrail(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_llm_conditional_execution_guardrail(&uuid, name)
        .map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — LLM
// ---------------------------------------------------------------------------

/// Registers a scope-local intercept that transforms LLM request data (`LlmRequest`).
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `name` - Unique intercept name.
/// - `priority` - Execution priority (lower runs first).
/// - `break_chain` - If `true`, stops further intercepts from running after this one.
/// - `func` - JS function `(request) => transformedRequest`.
#[wasm_bindgen(js_name = "scopeRegisterLlmRequestIntercept")]
pub fn scope_register_llm_request_intercept(
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(js_name = "breakChain")] break_chain: bool,
    #[wasm_bindgen(js_name = "callable", unchecked_param_type = "(request: Json) => any")]
    func: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_llm_request_intercept(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_llm_request_intercept(&uuid, name).map_err(to_js_err)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(
        js_name = "callable",
        unchecked_param_type = "(request: Json, next: (...args: any[]) => any) => any"
    )]
    exec_fn: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_llm_execution_intercept(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_llm_execution_intercept(&uuid, name).map_err(to_js_err)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    priority: i32,
    #[wasm_bindgen(
        js_name = "callable",
        unchecked_param_type = "(request: Json, next: (...args: any[]) => any) => any"
    )]
    exec_fn: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_register_llm_stream_execution_intercept(
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_registry_api::scope_deregister_llm_stream_execution_intercept(&uuid, name)
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
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "(event: Json) => any")] callback: Function,
) -> Result<(), JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_subscriber_api::scope_register_subscriber(
        &uuid,
        name,
        callable::wrap_js_event_subscriber(callback),
    )
    .map_err(to_js_err)
}

/// Registers an anonymous scope-local event subscriber and returns a closeable handle.
///
/// - `scope_uuid` - UUID of the scope to register on.
/// - `callback` - JS function `(event) => void` called for each event.
#[wasm_bindgen(js_name = "scopeSubscribe")]
pub fn scope_subscribe(
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    #[wasm_bindgen(unchecked_param_type = "(event: Json) => any")] callback: Function,
) -> Result<Subscription, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_subscriber_api::scope_subscribe(&uuid, callable::wrap_js_event_subscriber(callback))
        .map(|handle| Subscription {
            inner: Arc::new(Mutex::new(Some(SubscriptionInner::Scope(handle)))),
        })
        .map_err(to_js_err)
}

/// Removes a scope-local event subscriber by name.
///
/// Returns `true` if the subscriber was found and removed from the specified scope.
#[wasm_bindgen(js_name = "scopeDeregisterSubscriber")]
pub fn scope_deregister_subscriber(
    #[wasm_bindgen(js_name = "scopeUuid")] scope_uuid: &str,
    name: &str,
) -> Result<bool, JsValue> {
    let uuid = uuid::Uuid::parse_str(scope_uuid)
        .map_err(|e| JsValue::from_str(&format!("invalid UUID: {e}")))?;
    relay_subscriber_api::scope_deregister_subscriber(&uuid, name).map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Creates a new isolated scope stack.
#[wasm_bindgen(js_name = "createScopeStack")]
pub fn create_scope_stack() -> ScopeStack {
    ScopeStack {
        inner: create_scope_stack_handle(),
    }
}

/// Returns the current thread's scope stack handle.
#[wasm_bindgen(js_name = "currentScopeStack")]
pub fn current_scope_stack() -> ScopeStack {
    ScopeStack {
        inner: current_scope_stack_handle(),
    }
}

/// Binds a scope stack to the current thread.
#[wasm_bindgen(js_name = "setThreadScopeStack")]
pub fn set_thread_scope_stack(stack: &ScopeStack) {
    bind_thread_scope_stack(stack.inner.clone());
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `setThreadScopeStack` has been called. Returns `false`
/// when only the auto-created default is present.
#[wasm_bindgen(js_name = "scopeStackActive")]
pub fn scope_stack_active() -> bool {
    scope_stack_is_active()
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Runs the registered tool request intercept chain on the given arguments.
#[wasm_bindgen(js_name = "toolRequestIntercepts", unchecked_return_type = "Json")]
pub fn tool_request_intercepts_wasm(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "Json")] args: JsValue,
) -> Result<JsValue, JsValue> {
    let args_json = js_to_json(&args)?;
    let result = relay_tool_api::tool_request_intercepts(name, args_json).map_err(to_js_err)?;
    Ok(json_to_js(&result))
}

/// Runs the registered tool conditional execution guardrail chain.
#[wasm_bindgen(js_name = "toolConditionalExecution")]
pub fn tool_conditional_execution_wasm(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "Json")] args: JsValue,
) -> Result<(), JsValue> {
    let args_json = js_to_json(&args)?;
    relay_tool_api::tool_conditional_execution(name, &args_json).map_err(to_js_err)
}

/// Runs the registered LLM request intercept chain on the given `LlmRequest`.
#[wasm_bindgen(js_name = "llmRequestIntercepts", unchecked_return_type = "Json")]
pub fn llm_request_intercepts_wasm(
    name: &str,
    #[wasm_bindgen(unchecked_param_type = "Json")] request: JsValue,
) -> Result<JsValue, JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: CoreLlmRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(FlowError::Internal(e.to_string())))?;
    let result = relay_llm_api::llm_request_intercepts(name, llm_request).map_err(to_js_err)?;
    let result_json =
        serde_json::to_value(&result).map_err(|e| to_js_err(FlowError::Internal(e.to_string())))?;
    Ok(json_to_js(&result_json))
}

/// Runs the registered LLM conditional execution guardrail chain.
///
/// - `request` - The LLM request as a JSON value with `{ headers, content }` shape.
#[wasm_bindgen(js_name = "llmConditionalExecution")]
pub fn llm_conditional_execution_wasm(
    #[wasm_bindgen(unchecked_param_type = "Json")] request: JsValue,
) -> Result<(), JsValue> {
    let request_json = js_to_json(&request)?;
    let llm_request: CoreLlmRequest = serde_json::from_value(request_json)
        .map_err(|e| to_js_err(FlowError::Internal(e.to_string())))?;
    relay_llm_api::llm_conditional_execution(&llm_request).map_err(to_js_err)
}

// ---------------------------------------------------------------------------
// ATIF exporter
// ---------------------------------------------------------------------------

/// ATIF trajectory exporter for collecting events and producing ATIF JSON.
#[wasm_bindgen(js_name = AtifExporter)]
pub struct AtifExporter {
    inner: nemo_relay::observability::atif::AtifExporter,
}

#[wasm_bindgen(js_class = AtifExporter)]
impl AtifExporter {
    /// Creates a new ATIF exporter.
    #[allow(non_snake_case)]
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(js_name = "sessionId")] session_id: String,
        #[wasm_bindgen(js_name = "agentName")] agent_name: String,
        #[wasm_bindgen(js_name = "agentVersion")] agent_version: String,
        #[wasm_bindgen(
            js_name = "modelName",
            unchecked_param_type = "string | null | undefined"
        )]
        model_name: Option<String>,
    ) -> Self {
        let agent_info = nemo_relay::observability::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: None,
            extra: None,
        };
        Self {
            inner: nemo_relay::observability::atif::AtifExporter::new(session_id, agent_info),
        }
    }

    /// Registers the exporter as an event subscriber.
    pub fn register(&self, name: &str) -> Result<(), JsValue> {
        let subscriber = self.inner.subscriber();
        relay_subscriber_api::register_subscriber(name, subscriber)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deregisters the exporter subscriber.
    pub fn deregister(&self, name: &str) -> Result<bool, JsValue> {
        relay_subscriber_api::deregister_subscriber(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Exports collected events as an ATIF trajectory JSON string.
    #[wasm_bindgen(js_name = "exportJson")]
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
#[wasm_bindgen(
    js_name = "defaultOpenTelemetryConfig",
    unchecked_return_type = "OpenTelemetryConfig"
)]
pub fn default_open_telemetry_config() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&WasmOpenTelemetryConfig::default())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// OpenTelemetry-backed event subscriber.
#[wasm_bindgen(js_name = OpenTelemetrySubscriber)]
pub struct OpenTelemetrySubscriber {
    inner: nemo_relay::observability::otel::OpenTelemetrySubscriber,
}

#[wasm_bindgen(js_class = OpenTelemetrySubscriber)]
impl OpenTelemetrySubscriber {
    /// Creates a new OpenTelemetry subscriber from a config object.
    ///
    /// Expected object shape:
    /// `{ transport, endpoint, headers, resourceAttributes, serviceName,
    /// serviceNamespace, serviceVersion, instrumentationScope, timeoutMillis }`
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(unchecked_param_type = "OpenTelemetryConfig | null | undefined")]
        config: Option<JsValue>,
    ) -> Result<OpenTelemetrySubscriber, JsValue> {
        let config = match config {
            Some(value) if !value.is_undefined() && !value.is_null() => Some(
                serde_wasm_bindgen::from_value(value)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            ),
            _ => None,
        };

        let inner = nemo_relay::observability::otel::OpenTelemetrySubscriber::new(
            build_otel_config(config)?,
        )
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
#[wasm_bindgen(
    js_name = "defaultOpenInferenceConfig",
    unchecked_return_type = "OpenInferenceConfig"
)]
pub fn default_open_inference_config() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&WasmOpenInferenceConfig::default())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

fn ensure_adaptive_component_registered() -> Result<(), JsValue> {
    register_adaptive_component().map_err(to_js_err)
}

/// Validate a plugin config document and return a structured diagnostics report.
#[wasm_bindgen(js_name = "validatePluginConfig", unchecked_return_type = "Json")]
pub fn validate_plugin_config(
    #[wasm_bindgen(unchecked_param_type = "Json")] config: JsValue,
) -> Result<JsValue, JsValue> {
    ensure_adaptive_component_registered()?;
    let config: PluginConfig = serde_wasm_bindgen::from_value(config)?;
    serde_wasm_bindgen::to_value(&validate_plugin_config_impl(&config))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[derive(Clone)]
#[wasm_bindgen(js_name = "PluginContext", skip_typescript)]
/// Plugin registration context exposed to JavaScript plugins.
///
/// Plugin implementations receive this object during registration and use it
/// to install namespaced subscribers, guardrails, and intercepts that can be
/// cleaned up automatically on deregistration.
pub struct PluginContext {
    registrations: Arc<Mutex<Vec<ComponentRegistration>>>,
    namespace_prefix: String,
}

impl PluginContext {
    fn drain_registrations(&self) -> Result<Vec<ComponentRegistration>, JsValue> {
        let mut guard = self
            .registrations
            .lock()
            .map_err(|e| JsValue::from_str(&format!("plugin context lock poisoned: {e}")))?;
        Ok(std::mem::take(&mut *guard))
    }

    fn push_registration(&self, registration: ComponentRegistration) -> Result<(), JsValue> {
        self.registrations
            .lock()
            .map_err(|e| JsValue::from_str(&format!("failed to acquire registrations lock: {e}")))?
            .push(registration);
        Ok(())
    }

    fn qualify_name(&self, name: &str) -> String {
        format!("{}{}", self.namespace_prefix, name)
    }
}

#[wasm_bindgen(js_class = PluginContext)]
impl PluginContext {
    /// Register a lifecycle subscriber scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Subscriber name relative to the plugin namespace.
    /// - `callback`: JavaScript callback that receives serialized lifecycle events.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerSubscriber)]
    pub fn register_subscriber(
        &self,
        name: &str,
        #[wasm_bindgen(unchecked_param_type = "(event: Json) => any")] callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_subscriber_api::register_subscriber(
            &qualified_name,
            crate::callable::wrap_js_event_subscriber(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_subscriber_api::deregister_subscriber(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "subscriber deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register a tool request sanitizer scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Guardrail name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that receives the tool name and JSON arguments.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerToolSanitizeRequestGuardrail)]
    pub fn register_tool_sanitize_request_guardrail(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(unchecked_param_type = "(name: string, args: Json) => any")]
        callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_tool_sanitize_request_guardrail(
            &qualified_name,
            priority,
            crate::callable::wrap_js_tool_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_tool_sanitize_request_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "tool sanitize request guardrail deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register a tool response sanitizer scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Guardrail name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that receives the tool name and JSON result.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerToolSanitizeResponseGuardrail)]
    pub fn register_tool_sanitize_response_guardrail(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(unchecked_param_type = "(name: string, result: Json) => any")]
        callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_tool_sanitize_response_guardrail(
            &qualified_name,
            priority,
            crate::callable::wrap_js_tool_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_tool_sanitize_response_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "tool sanitize response guardrail deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register a tool conditional-execution guardrail scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Guardrail name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that returns `null` to allow execution
    ///   or a string reason to reject the call.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerToolConditionalExecutionGuardrail)]
    pub fn register_tool_conditional_execution_guardrail(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(unchecked_param_type = "(name: string, args: Json) => string | null")]
        callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_tool_conditional_execution_guardrail(
            &qualified_name,
            priority,
            crate::callable::wrap_js_tool_conditional_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_tool_conditional_execution_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "tool conditional execution guardrail deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register an LLM request sanitizer scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Guardrail name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that receives and returns a JSON request.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerLlmSanitizeRequestGuardrail)]
    pub fn register_llm_sanitize_request_guardrail(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(unchecked_param_type = "(request: Json) => any")] callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_llm_sanitize_request_guardrail(
            &qualified_name,
            priority,
            crate::callable::wrap_js_llm_sanitize_request_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_llm_sanitize_request_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "llm sanitize request guardrail deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register an LLM response sanitizer scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Guardrail name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that receives and returns a JSON response.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerLlmSanitizeResponseGuardrail)]
    pub fn register_llm_sanitize_response_guardrail(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(unchecked_param_type = "(response: Json) => any")] callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_llm_sanitize_response_guardrail(
            &qualified_name,
            priority,
            crate::callable::wrap_js_llm_response_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_llm_sanitize_response_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "llm sanitize response guardrail deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register an LLM conditional-execution guardrail scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Guardrail name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that returns `null` to allow execution
    ///   or a string reason to reject the call.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerLlmConditionalExecutionGuardrail)]
    pub fn register_llm_conditional_execution_guardrail(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(unchecked_param_type = "(request: Json) => string | null")]
        callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_llm_conditional_execution_guardrail(
            &qualified_name,
            priority,
            crate::callable::wrap_js_llm_conditional_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_llm_conditional_execution_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "llm conditional execution guardrail deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register an LLM request intercept scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Intercept name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `break_chain`: Whether later request intercepts should be skipped.
    /// - `callback`: JavaScript callback that receives and returns a JSON request.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerLlmRequestIntercept)]
    pub fn register_llm_request_intercept(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(js_name = "breakChain")] break_chain: bool,
        #[wasm_bindgen(unchecked_param_type = "(request: Json) => any")] callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_llm_request_intercept(
            &qualified_name,
            priority,
            break_chain,
            crate::callable::wrap_js_llm_request_intercept_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_llm_request_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "llm request intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register a non-streaming LLM execution intercept scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Intercept name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that receives the JSON request plus a
    ///   continuation for the remaining execution chain.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerLlmExecutionIntercept)]
    pub fn register_llm_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(
            unchecked_param_type = "(request: Json, next: (...args: any[]) => any) => any"
        )]
        callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_llm_execution_intercept(
            &qualified_name,
            priority,
            crate::callable::wrap_js_llm_exec_intercept_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_llm_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "llm execution intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register a streaming LLM execution intercept scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Intercept name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that receives the JSON request plus a
    ///   continuation for the remaining execution chain.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerLlmStreamExecutionIntercept)]
    pub fn register_llm_stream_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(
            unchecked_param_type = "(request: Json, next: (...args: any[]) => any) => any"
        )]
        callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_llm_stream_execution_intercept(
            &qualified_name,
            priority,
            crate::callable::wrap_js_llm_stream_exec_intercept_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_llm_stream_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "llm stream execution intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register a tool request intercept scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Intercept name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `break_chain`: Whether later request intercepts should be skipped.
    /// - `callback`: JavaScript callback that receives the tool name and JSON arguments.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerToolRequestIntercept)]
    pub fn register_tool_request_intercept(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(js_name = "breakChain")] break_chain: bool,
        #[wasm_bindgen(unchecked_param_type = "(name: string, args: Json) => any")]
        callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_tool_request_intercept(
            &qualified_name,
            priority,
            break_chain,
            crate::callable::wrap_js_tool_request_intercept_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_tool_request_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "tool request intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }

    /// Register a tool execution intercept scoped to the current plugin namespace.
    ///
    /// # Parameters
    /// - `name`: Intercept name relative to the plugin namespace.
    /// - `priority`: Execution priority, where lower values run first.
    /// - `callback`: JavaScript callback that receives JSON arguments plus a
    ///   continuation for the remaining execution chain.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    #[wasm_bindgen(js_name = registerToolExecutionIntercept)]
    pub fn register_tool_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        #[wasm_bindgen(
            unchecked_param_type = "(args: Json, next: (...args: any[]) => any) => any"
        )]
        callback: Function,
    ) -> Result<(), JsValue> {
        let qualified_name = self.qualify_name(name);
        relay_registry_api::register_tool_execution_intercept(
            &qualified_name,
            priority,
            crate::callable::wrap_js_tool_exec_intercept_fn(callback),
        )
        .map_err(to_js_err)?;

        let name_owned = qualified_name;
        self.push_registration(ComponentRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                relay_registry_api::deregister_tool_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|e| {
                        PluginError::RegistrationFailed(format!(
                            "tool execution intercept deregistration failed: {e}"
                        ))
                    })
            }),
        ))
    }
}

struct WasmPlugin {
    plugin_kind: String,
    validate: Option<send_wrapper::SendWrapper<Function>>,
    register: send_wrapper::SendWrapper<Function>,
}

// SAFETY: The `validate` and `register` functions are wrapped in `SendWrapper`,
// which enforces access from the thread that created them. Cross-thread access
// will panic rather than allow undefined behavior.
unsafe impl Send for WasmPlugin {}
// SAFETY: The same `SendWrapper` invariant applies for shared references; the
// wrapped callbacks are only invoked on their originating thread.
unsafe impl Sync for WasmPlugin {}

impl Plugin for WasmPlugin {
    fn plugin_kind(&self) -> &str {
        &self.plugin_kind
    }

    fn validate(
        &self,
        plugin_config: &serde_json::Map<String, serde_json::Value>,
    ) -> Vec<ConfigDiagnostic> {
        let Some(validate) = &self.validate else {
            return vec![];
        };
        let plugin_config_js = json_to_js(&serde_json::Value::Object(plugin_config.clone()));
        let this_arg = JsValue::NULL;
        let validation = validate.call1(&this_arg, &plugin_config_js);
        match validation {
            Ok(value) if value.is_null() || value.is_undefined() => vec![],
            Ok(value) => serde_wasm_bindgen::from_value::<Vec<ConfigDiagnostic>>(value)
                .unwrap_or_else(|e| {
                    vec![ConfigDiagnostic {
                        level: DiagnosticLevel::Error,
                        code: "plugin.validate_failed".into(),
                        component: Some(self.plugin_kind.clone()),
                        field: None,
                        message: format!(
                            "WebAssembly plugin validate returned invalid diagnostics: {e}"
                        ),
                    }]
                }),
            Err(err) => vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.validate_failed".into(),
                component: Some(self.plugin_kind.clone()),
                field: None,
                message: err
                    .as_string()
                    .unwrap_or_else(|| "WebAssembly plugin validate failed".to_string()),
            }],
        }
    }

    fn register<'a>(
        &'a self,
        plugin_config: &serde_json::Map<String, serde_json::Value>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), PluginError>> + Send + 'a>> {
        let namespace_prefix = ctx.qualify_name("");
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let plugin_context = PluginContext {
                registrations: Arc::new(Mutex::new(vec![])),
                namespace_prefix,
            };
            let plugin_context_js = JsValue::from(plugin_context.clone());
            let plugin_config_js = json_to_js(&serde_json::Value::Object(plugin_config));
            self.register
                .call2(&JsValue::NULL, &plugin_config_js, &plugin_context_js)
                .map_err(|err| {
                    PluginError::RegistrationFailed(
                        err.as_string()
                            .unwrap_or_else(|| "WebAssembly plugin register failed".to_string()),
                    )
                })?;

            ctx.extend_registrations(plugin_context.drain_registrations().map_err(|err| {
                PluginError::RegistrationFailed(err.as_string().unwrap_or_else(|| {
                    "failed to drain WebAssembly plugin registrations".to_string()
                }))
            })?);
            Ok(())
        })
    }
}

#[allow(non_snake_case)]
#[wasm_bindgen(js_name = "registerPlugin")]
/// Register a plugin backed by JavaScript callbacks.
///
/// `validate` receives one component-local config object and returns diagnostics.
/// `register` receives `(pluginConfig, context)` and should attach subscribers,
/// guardrails, or intercepts through the provided plugin context.
pub fn register_plugin(
    #[wasm_bindgen(js_name = "pluginKind", unchecked_param_type = "string")] plugin_kind: String,
    #[wasm_bindgen(unchecked_param_type = "((...args: any[]) => any) | null | undefined")] validate: Option<Function>,
    #[wasm_bindgen(unchecked_param_type = "(...args: any[]) => any")] register: Function,
) -> Result<(), JsValue> {
    ensure_adaptive_component_registered()?;
    register_plugin_impl(Arc::new(WasmPlugin {
        plugin_kind,
        validate: validate.map(send_wrapper::SendWrapper::new),
        register: send_wrapper::SendWrapper::new(register),
    }))
    .map_err(to_js_err)
}

#[wasm_bindgen(js_name = "deregisterPlugin")]
#[allow(non_snake_case)]
/// Deregister a previously registered plugin kind.
pub fn deregister_plugin(
    #[wasm_bindgen(js_name = "pluginKind", unchecked_param_type = "string")] plugin_kind: String,
) -> bool {
    deregister_plugin_impl(&plugin_kind)
}

#[wasm_bindgen(js_name = "initializePlugins", unchecked_return_type = "Json")]
/// Validate and activate a plugin configuration.
///
/// Replaces the current active plugin configuration and rolls back partial
/// registration on failure.
pub async fn initialize_plugins(
    #[wasm_bindgen(unchecked_param_type = "Json")] config: JsValue,
) -> Result<JsValue, JsValue> {
    ensure_adaptive_component_registered()?;
    let config: PluginConfig = serde_wasm_bindgen::from_value(config)?;
    let report = initialize_plugins_impl(config).await.map_err(to_js_err)?;
    serde_wasm_bindgen::to_value(&report).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen(js_name = "clearPluginConfiguration")]
/// Clear the active plugin configuration while leaving plugin kinds registered.
pub fn clear_plugin_configuration() -> Result<(), JsValue> {
    clear_plugin_configuration_impl().map_err(to_js_err)
}

#[wasm_bindgen(js_name = "activePluginReport", unchecked_return_type = "Json | null")]
/// Return the last successfully activated plugin report, if any.
pub fn active_plugin_report() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&active_plugin_report_impl())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen(js_name = "listPluginKinds", unchecked_return_type = "string[]")]
/// List the plugin kinds currently registered with the runtime.
pub fn list_plugin_kinds() -> Result<JsValue, JsValue> {
    ensure_adaptive_component_registered()?;
    serde_wasm_bindgen::to_value(&list_plugin_kinds_impl())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// OpenInference-backed event subscriber.
#[wasm_bindgen(js_name = OpenInferenceSubscriber)]
pub struct OpenInferenceSubscriber {
    inner: nemo_relay::observability::openinference::OpenInferenceSubscriber,
}

#[wasm_bindgen(js_class = OpenInferenceSubscriber)]
impl OpenInferenceSubscriber {
    /// Creates a new OpenInference subscriber from a config object.
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(unchecked_param_type = "OpenInferenceConfig | null | undefined")]
        config: Option<JsValue>,
    ) -> Result<OpenInferenceSubscriber, JsValue> {
        let config = match config {
            Some(value) if !value.is_undefined() && !value.is_null() => Some(
                serde_wasm_bindgen::from_value(value)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            ),
            _ => None,
        };

        let inner = nemo_relay::observability::openinference::OpenInferenceSubscriber::new(
            build_openinference_config(config)?,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Register the subscriber under a runtime-global name.
    ///
    /// # Parameters
    /// - `name`: Subscriber name to register.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when registration succeeds.
    pub fn register(&self, name: &str) -> Result<(), JsValue> {
        self.inner
            .register(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deregister the subscriber by name.
    ///
    /// # Parameters
    /// - `name`: Subscriber name to deregister.
    ///
    /// # Returns
    /// A JavaScript `Result` containing `true` when a registration was removed.
    pub fn deregister(&self, name: &str) -> Result<bool, JsValue> {
        self.inner
            .deregister(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Flush any buffered telemetry.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when the flush completes.
    #[wasm_bindgen(js_name = "forceFlush")]
    pub fn force_flush(&self) -> Result<(), JsValue> {
        self.inner
            .force_flush()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Shut down the subscriber and release owned resources.
    ///
    /// # Returns
    /// A JavaScript `Result` that is `Ok(())` when shutdown completes.
    pub fn shutdown(&self) -> Result<(), JsValue> {
        self.inner
            .shutdown()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/api_tests.rs"]
mod tests;
