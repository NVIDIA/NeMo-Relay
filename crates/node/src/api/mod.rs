// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public NAPI API functions for the NeMo Relay Node.js bindings.
//!
//! This module exposes the full agent runtime API to JavaScript/TypeScript:
//! scope stack management, tool and LLM lifecycle operations, guardrail and
//! intercept registration/deregistration, and event subscriber management.
//! All functions are annotated with `#[napi]` and their doc comments appear
//! in the generated `index.d.ts` TypeScript definitions.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue};
use napi_derive::napi;
use serde::Deserialize;
use serde_json::Value as Json;
use tokio_stream::StreamExt;

use nemo_relay::api::llm as core_llm_api;
use nemo_relay::api::llm::{LlmAttributes, LlmRequest};
use nemo_relay::api::registry as core_registry_api;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use nemo_relay::api::runtime::{
    TASK_SCOPE_STACK, create_scope_stack as create_scope_stack_handle,
    current_scope_stack as current_scope_stack_handle, scope_stack_active as scope_stack_is_active,
    set_thread_scope_stack as bind_thread_scope_stack, task_scope_top,
};
use nemo_relay::api::scope as core_scope_api;
use nemo_relay::api::scope::ScopeAttributes;
use nemo_relay::api::subscriber as core_subscriber_api;
use nemo_relay::api::tool as core_tool_api;
use nemo_relay::api::tool::ToolAttributes;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::Usage;
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError, PluginRegistration,
    PluginRegistrationContext, active_plugin_report as active_plugin_report_impl,
    clear_plugin_configuration as clear_plugin_configuration_impl,
    deregister_plugin as deregister_plugin_impl, initialize_plugins as initialize_plugins_impl,
    list_plugin_kinds as list_plugin_kinds_impl, register_plugin as register_plugin_impl,
    validate_plugin_config as validate_plugin_config_impl,
};
use nemo_relay::shared_runtime::initialize_shared_runtime_binding;
use nemo_relay_adaptive::acg::{
    AgentIdentity, CacheRequestFacts, CacheTelemetryEvent, CacheTelemetryProvider,
};
use nemo_relay_adaptive::context_helpers::set_latency_sensitivity as adaptive_set_latency_sensitivity;
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use nemo_relay_adaptive::{AdaptiveConfig, AdaptiveRuntime as CoreAdaptiveRuntime};
use nemo_relay_pii_redaction::component::register_pii_redaction_component;

use crate::callable;
use crate::convert::{
    callback_json, clear_last_callback_error as clear_recorded_callback_error,
    get_last_callback_error as get_recorded_callback_error, opt_json, parse_timestamp_micros,
    to_napi_err,
};
use crate::stream::LlmStream;
use crate::types::{LlmHandle, ScopeHandle, ScopeStack, ScopeType, ToolHandle};

#[napi::module_init]
fn init() {
    initialize_shared_runtime_binding("node")
        .expect("node runtime ownership initialization should succeed");
    register_adaptive_component()
        .expect("node adaptive plugin component registration should succeed");
    register_pii_redaction_component()
        .expect("node pii redaction plugin component registration should succeed");
}

fn parse_string_map(
    value: Option<Json>,
    field_name: &str,
) -> napi::Result<HashMap<String, String>> {
    let Some(value) = value else {
        return Ok(HashMap::new());
    };
    let Json::Object(map) = value else {
        return Err(napi::Error::from_reason(format!(
            "{field_name} must be an object of string values",
        )));
    };
    let mut out = HashMap::with_capacity(map.len());
    for (key, value) in map {
        let Json::String(value) = value else {
            return Err(napi::Error::from_reason(format!(
                "{field_name} must be an object of string values",
            )));
        };
        out.insert(key, value);
    }
    Ok(out)
}

fn otel_status_metadata(status_code: &'static str, status_message: Option<String>) -> Json {
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "otel.status_code".to_string(),
        Json::String(status_code.to_string()),
    );
    if let Some(status_message) = status_message {
        metadata.insert(
            "otel.status_description".to_string(),
            Json::String(status_message),
        );
    }
    Json::Object(metadata)
}

fn build_otel_config(
    options: Option<OpenTelemetryConfig>,
) -> napi::Result<nemo_relay::observability::otel::OpenTelemetryConfig> {
    let options = options.unwrap_or_default();
    let transport = options
        .transport
        .unwrap_or_else(|| "http_binary".to_string());
    let service_name = options
        .service_name
        .unwrap_or_else(|| "nemo-relay".to_string());
    let instrumentation_scope = options
        .instrumentation_scope
        .unwrap_or_else(|| "nemo-relay-otel".to_string());
    let timeout_millis = options.timeout_millis.unwrap_or(3_000);

    let mut config = match transport.as_str() {
        "http_binary" => {
            nemo_relay::observability::otel::OpenTelemetryConfig::http_binary(service_name)
        }
        "grpc" => nemo_relay::observability::otel::OpenTelemetryConfig::grpc(service_name),
        other => {
            return Err(napi::Error::from_reason(format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}",
            )));
        }
    }
    .with_instrumentation_scope(instrumentation_scope)
    .with_timeout(std::time::Duration::from_millis(timeout_millis.into()));

    if let Some(endpoint) = options.endpoint {
        config = config.with_endpoint(endpoint);
    }
    if let Some(namespace) = options.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = options.service_version {
        config = config.with_service_version(version);
    }
    for (key, value) in parse_string_map(options.headers, "headers")? {
        config = config.with_header(key, value);
    }
    for (key, value) in parse_string_map(options.resource_attributes, "resourceAttributes")? {
        config = config.with_resource_attribute(key, value);
    }

    Ok(config)
}

fn build_atof_config(
    options: Option<AtofExporterConfig>,
) -> napi::Result<nemo_relay::observability::atof::AtofExporterConfig> {
    let options = options.unwrap_or_default();
    let mut config = nemo_relay::observability::atof::AtofExporterConfig::new();

    if let Some(output_directory) = options.output_directory {
        config = config.with_output_directory(PathBuf::from(output_directory));
    }
    if let Some(filename) = options.filename {
        config = config.with_filename(filename);
    }
    if let Some(mode) = options.mode {
        let Some(mode) = nemo_relay::observability::atof::AtofExporterMode::parse(&mode) else {
            return Err(napi::Error::from_reason(
                "mode must be 'append' or 'overwrite'",
            ));
        };
        config = config.with_mode(mode);
    }
    let mut endpoints = Vec::new();
    for endpoint in options.endpoints.unwrap_or_default() {
        let transport = endpoint
            .transport
            .unwrap_or_else(|| "http_post".to_string());
        let Some(transport) =
            nemo_relay::observability::atof::AtofEndpointTransport::parse(&transport)
        else {
            return Err(napi::Error::from_reason(
                "endpoint transport must be 'http_post', 'websocket', or 'ndjson'",
            ));
        };
        let mut endpoint_config =
            nemo_relay::observability::atof::AtofEndpointConfig::new(endpoint.url, transport);
        if let Some(timeout_millis) = endpoint.timeout_millis {
            endpoint_config = endpoint_config.with_timeout_millis(timeout_millis.into());
        }
        if let Some(field_name_policy) = endpoint.field_name_policy {
            let Some(field_name_policy) =
                nemo_relay::observability::atof::AtofEndpointFieldNamePolicy::parse(
                    &field_name_policy,
                )
            else {
                return Err(napi::Error::from_reason(
                    "endpoint field_name_policy must be 'preserve' or 'replace_dots'",
                ));
            };
            endpoint_config = endpoint_config.with_field_name_policy(field_name_policy);
        }
        for (key, value) in parse_string_map(endpoint.headers, "endpoint.headers")? {
            endpoint_config = endpoint_config.with_header(key, value);
        }
        endpoints.push(endpoint_config);
    }
    config = config.with_endpoints(endpoints);

    Ok(config)
}

fn build_openinference_config(
    options: Option<OpenInferenceConfig>,
) -> napi::Result<nemo_relay::observability::openinference::OpenInferenceConfig> {
    let options = options.unwrap_or_default();
    let transport = options
        .transport
        .unwrap_or_else(|| "http_binary".to_string());
    let service_name = options
        .service_name
        .unwrap_or_else(|| "nemo-relay".to_string());
    let instrumentation_scope = options
        .instrumentation_scope
        .unwrap_or_else(|| "nemo-relay-openinference".to_string());
    let timeout_millis = options.timeout_millis.unwrap_or(3_000);

    let transport = match transport.as_str() {
        "http_binary" => nemo_relay::observability::openinference::OtlpTransport::HttpBinary,
        "grpc" => nemo_relay::observability::openinference::OtlpTransport::Grpc,
        other => {
            return Err(napi::Error::from_reason(format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}",
            )));
        }
    };

    let mut config = nemo_relay::observability::openinference::OpenInferenceConfig::new()
        .with_transport(transport)
        .with_service_name(service_name)
        .with_instrumentation_scope(instrumentation_scope)
        .with_timeout(std::time::Duration::from_millis(timeout_millis.into()));

    if let Some(endpoint) = options.endpoint {
        config = config.with_endpoint(endpoint);
    }
    if let Some(namespace) = options.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = options.service_version {
        config = config.with_service_version(version);
    }
    for (key, value) in parse_string_map(options.headers, "headers")? {
        config = config.with_header(key, value);
    }
    for (key, value) in parse_string_map(options.resource_attributes, "resourceAttributes")? {
        config = config.with_resource_attribute(key, value);
    }

    Ok(config)
}

// ---------------------------------------------------------------------------
// Stream channel registry — enables JS async generators to push chunks to Rust
// ---------------------------------------------------------------------------

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(0);

type StreamSender = tokio::sync::mpsc::UnboundedSender<FlowResult<Json>>;
type RustJsonStream = std::pin::Pin<Box<dyn tokio_stream::Stream<Item = FlowResult<Json>> + Send>>;

static STREAM_CHANNELS: std::sync::LazyLock<StdMutex<HashMap<u64, StreamSender>>> =
    std::sync::LazyLock::new(|| StdMutex::new(HashMap::new()));

fn register_stream_channel(id: u64, tx: StreamSender) {
    STREAM_CHANNELS.lock().unwrap().insert(id, tx);
}

fn remove_stream_channel(id: u64) {
    STREAM_CHANNELS.lock().unwrap().remove(&id);
}

fn ensure_stream_callback_queued(id: u64, status: napi::Status) -> FlowResult<()> {
    if status == napi::Status::Ok {
        return Ok(());
    }

    remove_stream_channel(id);
    Err(FlowError::Internal(format!(
        "failed to queue JS stream producer callback: {status:?}",
    )))
}

async fn forward_stream_to_channel(
    mut stream: RustJsonStream,
    tx: tokio::sync::mpsc::Sender<FlowResult<Json>>,
) {
    while let Some(item) = stream.next().await {
        if tx.send(item).await.is_err() {
            break;
        }
    }
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
    remove_stream_channel(id);
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum PromiseAwareKey {
    GlobalToolExecution(String),
    GlobalLlmExecution(String),
    GlobalLlmStreamExecution(String),
    ScopeToolExecution { scope_uuid: String, name: String },
    ScopeLlmExecution { scope_uuid: String, name: String },
    ScopeLlmStreamExecution { scope_uuid: String, name: String },
}

impl PromiseAwareKey {
    fn scope_uuid(&self) -> Option<&str> {
        match self {
            Self::ScopeToolExecution { scope_uuid, .. }
            | Self::ScopeLlmExecution { scope_uuid, .. }
            | Self::ScopeLlmStreamExecution { scope_uuid, .. } => Some(scope_uuid),
            Self::GlobalToolExecution(_)
            | Self::GlobalLlmExecution(_)
            | Self::GlobalLlmStreamExecution(_) => None,
        }
    }
}

static PROMISE_AWARE_REGISTRATIONS: std::sync::LazyLock<
    StdMutex<HashMap<PromiseAwareKey, std::sync::Arc<crate::promise_call::PromiseAwareFn>>>,
> = std::sync::LazyLock::new(|| StdMutex::new(HashMap::new()));

fn remember_promise_aware(
    key: PromiseAwareKey,
    pa_fn: std::sync::Arc<crate::promise_call::PromiseAwareFn>,
) {
    if let Some(previous) = PROMISE_AWARE_REGISTRATIONS
        .lock()
        .unwrap()
        .insert(key, pa_fn)
    {
        previous.close();
    }
}

fn forget_promise_aware(key: &PromiseAwareKey) {
    if let Some(pa_fn) = PROMISE_AWARE_REGISTRATIONS.lock().unwrap().remove(key) {
        pa_fn.close();
    }
}

fn forget_scope_local_promise_aware(scope_uuid: &str) {
    let mut registrations = PROMISE_AWARE_REGISTRATIONS.lock().unwrap();
    let keys = registrations
        .keys()
        .filter(|key| key.scope_uuid() == Some(scope_uuid))
        .cloned()
        .collect::<Vec<_>>();

    for key in keys {
        let registration = registrations.remove(&key);
        if let Some(pa_fn) = registration {
            pa_fn.close();
        }
    }
}

/// # Safety
/// Both `env` and `value` must contain valid N-API handles that point to live
/// JavaScript objects in the same environment. The caller must also ensure the
/// environment is not in a pending exception state.
fn js_unknown_from_raw<T: NapiRaw>(env: &Env, value: &T) -> JsUnknown {
    unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) }
}

fn json_callback_tsfn(
    env: &Env,
    func: &JsFunction,
) -> napi::Result<ThreadsafeFunction<Json, ErrorStrategy::Fatal>> {
    let mut tsfn = func
        .create_threadsafe_function::<Json, Json, _, ErrorStrategy::Fatal>(0, |ctx| {
            Ok(vec![ctx.value])
        })?;
    tsfn.unref(env)?;
    Ok(tsfn)
}

fn build_plugin_context(
    env: &Env,
    namespace_prefix: String,
    registrations: Arc<StdMutex<Vec<PluginRegistration>>>,
) -> napi::Result<JsObject> {
    let mut context = env.create_object()?;

    let subscriber_regs = registrations.clone();
    let subscriber_namespace = namespace_prefix.clone();
    let register_subscriber = env.create_function_from_closure(
        "__nemo_relay_adaptive_register_subscriber",
        move |ctx| {
            let name = format!("{}{}", subscriber_namespace, ctx.get::<String>(0)?);
            let callback = ctx.get::<JsFunction>(1)?;
            let tsfn = json_callback_tsfn(ctx.env, &callback)?;
            core_subscriber_api::register_subscriber(
                &name,
                callable::wrap_js_event_subscriber(tsfn),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            subscriber_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_subscriber_api::deregister_subscriber(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "subscriber deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property("registerSubscriber", register_subscriber)?;

    let tool_sanitize_request_regs = registrations.clone();
    let tool_sanitize_request_namespace = namespace_prefix.clone();
    let register_tool_sanitize_request_guardrail = env.create_function_from_closure(
        "__nemo_relay_plugin_register_tool_sanitize_request_guardrail",
        move |ctx| {
            let name = format!(
                "{}{}",
                tool_sanitize_request_namespace,
                ctx.get::<String>(0)?
            );
            let priority = ctx.get::<i32>(1)?;
            let callback =
                ctx.get::<ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>>(2)?;
            core_registry_api::register_tool_sanitize_request_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_fn(callback),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            tool_sanitize_request_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_tool_sanitize_request_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "tool sanitize request guardrail deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerToolSanitizeRequestGuardrail",
        register_tool_sanitize_request_guardrail,
    )?;

    let tool_sanitize_response_regs = registrations.clone();
    let tool_sanitize_response_namespace = namespace_prefix.clone();
    let register_tool_sanitize_response_guardrail = env.create_function_from_closure(
        "__nemo_relay_plugin_register_tool_sanitize_response_guardrail",
        move |ctx| {
            let name = format!(
                "{}{}",
                tool_sanitize_response_namespace,
                ctx.get::<String>(0)?
            );
            let priority = ctx.get::<i32>(1)?;
            let callback =
                ctx.get::<ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>>(2)?;
            core_registry_api::register_tool_sanitize_response_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_fn(callback),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            tool_sanitize_response_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_tool_sanitize_response_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "tool sanitize response guardrail deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerToolSanitizeResponseGuardrail",
        register_tool_sanitize_response_guardrail,
    )?;

    let tool_conditional_regs = registrations.clone();
    let tool_conditional_namespace = namespace_prefix.clone();
    let register_tool_conditional_execution_guardrail = env.create_function_from_closure(
        "__nemo_relay_plugin_register_tool_conditional_execution_guardrail",
        move |ctx| {
            let name = format!("{}{}", tool_conditional_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback =
                ctx.get::<ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>>(2)?;
            core_registry_api::register_tool_conditional_execution_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_conditional_fn(callback),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            tool_conditional_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_tool_conditional_execution_guardrail(
                            &name_clone,
                        )
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "tool conditional execution guardrail deregistration failed: {e}"
                            ))
                        })
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerToolConditionalExecutionGuardrail",
        register_tool_conditional_execution_guardrail,
    )?;

    let llm_sanitize_request_regs = registrations.clone();
    let llm_sanitize_request_namespace = namespace_prefix.clone();
    let register_llm_sanitize_request_guardrail = env.create_function_from_closure(
        "__nemo_relay_plugin_register_llm_sanitize_request_guardrail",
        move |ctx| {
            let name = format!(
                "{}{}",
                llm_sanitize_request_namespace,
                ctx.get::<String>(0)?
            );
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>(2)?;
            core_registry_api::register_llm_sanitize_request_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_sanitize_request_fn(callback),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            llm_sanitize_request_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_llm_sanitize_request_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "llm sanitize request guardrail deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerLlmSanitizeRequestGuardrail",
        register_llm_sanitize_request_guardrail,
    )?;

    let llm_sanitize_response_regs = registrations.clone();
    let llm_sanitize_response_namespace = namespace_prefix.clone();
    let register_llm_sanitize_response_guardrail = env.create_function_from_closure(
        "__nemo_relay_plugin_register_llm_sanitize_response_guardrail",
        move |ctx| {
            let name = format!(
                "{}{}",
                llm_sanitize_response_namespace,
                ctx.get::<String>(0)?
            );
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>(2)?;
            core_registry_api::register_llm_sanitize_response_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_response_fn(callback),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            llm_sanitize_response_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_llm_sanitize_response_guardrail(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "llm sanitize response guardrail deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerLlmSanitizeResponseGuardrail",
        register_llm_sanitize_response_guardrail,
    )?;

    let llm_conditional_regs = registrations.clone();
    let llm_conditional_namespace = namespace_prefix.clone();
    let register_llm_conditional_execution_guardrail = env.create_function_from_closure(
        "__nemo_relay_plugin_register_llm_conditional_execution_guardrail",
        move |ctx| {
            let name = format!("{}{}", llm_conditional_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>(2)?;
            core_registry_api::register_llm_conditional_execution_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_conditional_fn(callback),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            llm_conditional_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_llm_conditional_execution_guardrail(
                            &name_clone,
                        )
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "llm conditional execution guardrail deregistration failed: {e}"
                            ))
                        })
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerLlmConditionalExecutionGuardrail",
        register_llm_conditional_execution_guardrail,
    )?;

    let llm_regs = registrations.clone();
    let llm_request_namespace = namespace_prefix.clone();
    let register_llm_request_intercept = env.create_function_from_closure(
        "__nemo_relay_adaptive_register_llm_request_intercept",
        move |ctx| {
            let name = format!("{}{}", llm_request_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let break_chain = ctx.get::<bool>(2)?;
            let callback = ctx.get::<JsFunction>(3)?;
            let tsfn = json_callback_tsfn(ctx.env, &callback)?;
            core_registry_api::register_llm_request_intercept(
                &name,
                priority,
                break_chain,
                callable::wrap_js_llm_request_intercept_fn(tsfn),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            llm_regs.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    core_registry_api::deregister_llm_request_intercept(&name_clone)
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "llm request intercept deregistration failed: {e}"
                            ))
                        })
                }),
            ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerLlmRequestIntercept",
        register_llm_request_intercept,
    )?;

    let llm_exec_regs = registrations.clone();
    let llm_exec_namespace = namespace_prefix.clone();
    let register_llm_execution_intercept = env.create_function_from_closure(
        "__nemo_relay_adaptive_register_llm_execution_intercept",
        move |ctx| {
            let name = format!("{}{}", llm_exec_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<JsFunction>(2)?;
            let promise_fn = Arc::new(crate::promise_call::PromiseAwareFn::new(
                ctx.env, &callback,
            )?);
            core_registry_api::register_llm_execution_intercept(
                &name,
                priority,
                callable::wrap_js_llm_exec_intercept_fn(promise_fn.clone()),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            llm_exec_regs.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    let result = core_registry_api::deregister_llm_execution_intercept(&name_clone)
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "llm execution intercept deregistration failed: {e}"
                            ))
                        });
                    promise_fn.close();
                    result
                }),
            ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerLlmExecutionIntercept",
        register_llm_execution_intercept,
    )?;

    let llm_stream_exec_regs = registrations.clone();
    let llm_stream_namespace = namespace_prefix.clone();
    let register_llm_stream_execution_intercept = env.create_function_from_closure(
        "__nemo_relay_adaptive_register_llm_stream_execution_intercept",
        move |ctx| {
            let name = format!("{}{}", llm_stream_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<JsFunction>(2)?;
            let promise_fn = Arc::new(crate::promise_call::PromiseAwareFn::new(
                ctx.env, &callback,
            )?);
            core_registry_api::register_llm_stream_execution_intercept(
                &name,
                priority,
                callable::wrap_js_llm_stream_exec_intercept_fn(promise_fn.clone()),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            llm_stream_exec_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        let result = core_registry_api::deregister_llm_stream_execution_intercept(
                            &name_clone,
                        )
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "llm stream execution intercept deregistration failed: {e}"
                            ))
                        });
                        promise_fn.close();
                        result
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerLlmStreamExecutionIntercept",
        register_llm_stream_execution_intercept,
    )?;

    let tool_request_regs = registrations.clone();
    let tool_request_namespace = namespace_prefix.clone();
    let register_tool_request_intercept = env.create_function_from_closure(
        "__nemo_relay_adaptive_register_tool_request_intercept",
        move |ctx| {
            let name = format!("{}{}", tool_request_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let break_chain = ctx.get::<bool>(2)?;
            let callback =
                ctx.get::<ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>>(3)?;
            core_registry_api::register_tool_request_intercept(
                &name,
                priority,
                break_chain,
                callable::wrap_js_tool_request_intercept_fn(callback),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            tool_request_regs
                .lock()
                .unwrap()
                .push(PluginRegistration::new(
                    "plugin",
                    name_clone.clone(),
                    Box::new(move || {
                        core_registry_api::deregister_tool_request_intercept(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "tool request intercept deregistration failed: {e}"
                                ))
                            })
                    }),
                ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerToolRequestIntercept",
        register_tool_request_intercept,
    )?;

    let tool_regs = registrations.clone();
    let tool_exec_namespace = namespace_prefix;
    let register_tool_execution_intercept = env.create_function_from_closure(
        "__nemo_relay_adaptive_register_tool_execution_intercept",
        move |ctx| {
            let name = format!("{}{}", tool_exec_namespace, ctx.get::<String>(0)?);
            let priority = ctx.get::<i32>(1)?;
            let callback = ctx.get::<JsFunction>(2)?;
            let promise_fn = Arc::new(crate::promise_call::PromiseAwareFn::new(
                ctx.env, &callback,
            )?);
            core_registry_api::register_tool_execution_intercept(
                &name,
                priority,
                callable::wrap_js_tool_exec_intercept_fn(promise_fn.clone()),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            tool_regs.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    let result =
                        core_registry_api::deregister_tool_execution_intercept(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "tool execution intercept deregistration failed: {e}"
                                ))
                            });
                    promise_fn.close();
                    result
                }),
            ));
            ctx.env.get_undefined()
        },
    )?;
    context.set_named_property(
        "registerToolExecutionIntercept",
        register_tool_execution_intercept,
    )?;

    Ok(context)
}

struct NodePluginRegisterCall {
    plugin_config: Json,
    namespace_prefix: String,
    registrations: Arc<StdMutex<Vec<PluginRegistration>>>,
}

/// # Safety
/// `env` and `reference` must remain valid for the entire lifetime of this
/// struct. `reference` must be a valid N-API reference created for a live
/// JavaScript function in `env`, and `env` must not be used after the
/// corresponding Node.js environment has been torn down.
struct PersistentJsFunction {
    env: napi::sys::napi_env,
    reference: napi::sys::napi_ref,
}

// SAFETY: `PersistentJsFunction` only stores raw N-API handles. Callers are
// responsible for constructing it from a live environment and function
// reference, and all access goes back through that same environment.
unsafe impl Send for PersistentJsFunction {}
// SAFETY: The same invariants as `Send` apply. The struct does not provide
// interior mutation beyond the N-API reference lifecycle managed by Node.
unsafe impl Sync for PersistentJsFunction {}

impl PersistentJsFunction {
    fn new(env: &Env, func: &JsFunction) -> napi::Result<Self> {
        let mut reference = ptr::null_mut();
        // SAFETY: `env.raw()` and `func.raw()` are live N-API handles provided
        // by napi-rs for the current environment. `reference` points to valid
        // writable storage for the created reference.
        let status =
            unsafe { napi::sys::napi_create_reference(env.raw(), func.raw(), 1, &mut reference) };
        if status == napi::sys::Status::napi_ok {
            Ok(Self {
                env: env.raw(),
                reference,
            })
        } else {
            Err(napi::Error::from_reason(format!(
                "failed to create JS function reference: {:?}",
                napi::Status::from(status)
            )))
        }
    }

    fn call_validate(&self, plugin_config: &Json) -> napi::Result<Json> {
        // SAFETY: `self.env` was captured from a live N-API environment when
        // this persistent reference was created and remains valid while the
        // binding module is alive.
        let mut value = ptr::null_mut();
        // SAFETY: `self.reference` is a valid reference created by
        // `napi_create_reference`; `value` is writable storage for the
        // resolved function object.
        let status =
            unsafe { napi::sys::napi_get_reference_value(self.env, self.reference, &mut value) };
        if status != napi::sys::Status::napi_ok {
            return Err(napi::Error::from_reason(format!(
                "failed to borrow JS function reference: {:?}",
                napi::Status::from(status)
            )));
        }

        // SAFETY: `value` came from `napi_get_reference_value` for a function
        // reference owned by this struct, so it is a live JS function handle.
        let func = unsafe { JsFunction::from_raw_unchecked(self.env, value) };
        let config = unsafe {
            JsUnknown::from_raw_unchecked(
                self.env,
                Json::to_napi_value(self.env, plugin_config.clone())?,
            )
        };
        let returned = func.call(None, &[config])?;
        // SAFETY: `returned` is the live result of invoking `func` in the same
        // environment stored on this struct.
        unsafe { Option::<Json>::from_napi_value(self.env, returned.raw()) }.map(callback_json)
    }
}

impl Drop for PersistentJsFunction {
    fn drop(&mut self) {
        // SAFETY: `self.reference` was created by `napi_create_reference` for
        // `self.env` and is deleted exactly once here during drop.
        let _ = unsafe { napi::sys::napi_delete_reference(self.env, self.reference) };
    }
}

struct NodePlugin {
    plugin_kind: String,
    validate: Option<PersistentJsFunction>,
    register: ThreadsafeFunction<NodePluginRegisterCall, ErrorStrategy::Fatal>,
}

impl Plugin for NodePlugin {
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
        match validate.call_validate(&Json::Object(plugin_config.clone())) {
            Ok(Json::Null) => vec![],
            Ok(value) => {
                serde_json::from_value::<Vec<ConfigDiagnostic>>(value).unwrap_or_else(|e| {
                    vec![ConfigDiagnostic {
                        level: DiagnosticLevel::Error,
                        code: "plugin.validate_failed".into(),
                        component: Some(self.plugin_kind.clone()),
                        field: None,
                        message: format!("JS plugin validate returned invalid diagnostics: {e}"),
                    }]
                })
            }
            Err(e) => vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.validate_failed".into(),
                component: Some(self.plugin_kind.clone()),
                field: None,
                message: format!("JS plugin validate failed: {e}"),
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
            let registrations = Arc::new(StdMutex::new(Vec::<PluginRegistration>::new()));
            let payload = NodePluginRegisterCall {
                plugin_config: Json::Object(plugin_config),
                namespace_prefix,
                registrations: registrations.clone(),
            };
            let (tx, rx) = std::sync::mpsc::sync_channel::<std::result::Result<(), String>>(1);
            let status = self.register.call_with_return_value(
                payload,
                ThreadsafeFunctionCallMode::NonBlocking,
                move |_val: JsUnknown| {
                    let _ = tx.send(Ok(()));
                    Ok(())
                },
            );
            if status != napi::Status::Ok {
                return Err(PluginError::RegistrationFailed(format!(
                    "failed to queue JS plugin register callback: {status:?}"
                )));
            }
            rx.recv()
                .map_err(|_| {
                    PluginError::RegistrationFailed(
                        "JS plugin register completion channel closed".into(),
                    )
                })?
                .map_err(PluginError::RegistrationFailed)?;

            let drained = std::mem::take(&mut *registrations.lock().map_err(|e| {
                PluginError::RegistrationFailed(format!("plugin registrations lock poisoned: {e}"))
            })?);
            ctx.extend_registrations(drained);
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Creates a new isolated scope stack.
#[napi]
pub fn create_scope_stack() -> ScopeStack {
    ScopeStack {
        inner: create_scope_stack_handle(),
    }
}

/// Returns the current execution context's scope stack handle.
#[napi]
pub fn current_scope_stack() -> ScopeStack {
    ScopeStack {
        inner: current_scope_stack_handle(),
    }
}

/// Binds a scope stack to the current thread.
#[napi]
pub fn set_thread_scope_stack(stack: &ScopeStack) {
    bind_thread_scope_stack(stack.inner.clone());
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `setThreadScopeStack` has been called on the current
/// thread, or the caller is inside a task-local scope. Returns `false` when
/// only the auto-created default is present.
#[napi]
pub fn scope_stack_active() -> bool {
    scope_stack_is_active()
}

/// Returns the most recent callback error that could not be surfaced through a direct exception.
///
/// This is primarily used for sanitize/intercept/finalizer callback paths whose
/// core callback signatures cannot return `Result`.
#[napi]
pub fn get_last_callback_error() -> Option<String> {
    get_recorded_callback_error()
}

/// Clears the most recent callback error recorded by the Node binding.
#[napi]
pub fn clear_last_callback_error() {
    clear_recorded_callback_error();
}

/// Internal test helper: invoke a closed JS tool callback wrapper and return the fallback value.
#[napi(js_name = "__testClosedToolCallback")]
pub fn test_closed_tool_callback(
    callback: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
    name: String,
    args: Json,
) -> Json {
    clear_recorded_callback_error();
    let _ = callback.clone().abort();
    let wrapped = callable::wrap_js_tool_fn(callback);
    wrapped(&name, args)
}

/// Internal test helper: invoke a closed JS LLM sanitize-request wrapper and return the fallback request.
#[napi(js_name = "__testClosedLlmSanitizeRequestCallback")]
pub fn test_closed_llm_sanitize_request_callback(
    callback: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    request: Json,
) -> Result<Json> {
    clear_recorded_callback_error();
    let _ = callback.clone().abort();
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    let wrapped = callable::wrap_js_llm_sanitize_request_fn(callback);
    Ok(serde_json::to_value(wrapped(llm_request)).unwrap_or(Json::Null))
}

/// Internal test helper: invoke a closed JS LLM sanitize-response wrapper and return the fallback response.
#[napi(js_name = "__testClosedLlmResponseCallback")]
pub fn test_closed_llm_response_callback(
    callback: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    response: Json,
) -> Json {
    clear_recorded_callback_error();
    let _ = callback.clone().abort();
    let wrapped = callable::wrap_js_llm_response_fn(callback);
    wrapped(response)
}

/// Internal test helper: invoke a closed JS collector wrapper and surface the queue failure.
#[napi(js_name = "__testClosedCollectorCallback")]
pub fn test_closed_collector_callback(
    callback: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    chunk: Json,
) -> Result<()> {
    clear_recorded_callback_error();
    let _ = callback.clone().abort();
    let mut wrapped = callable::wrap_js_collector_fn(callback);
    wrapped(chunk).map_err(to_napi_err)
}

/// Internal test helper: invoke a closed JS finalizer wrapper and return the fallback value.
#[napi(js_name = "__testClosedFinalizerCallback")]
pub fn test_closed_finalizer_callback(
    callback: ThreadsafeFunction<(), ErrorStrategy::Fatal>,
) -> Json {
    clear_recorded_callback_error();
    let _ = callback.clone().abort();
    let wrapped = callable::wrap_js_finalizer_fn(callback);
    wrapped()
}

/// Internal test helper: exercise the PromiseAwareFn closed-call path.
#[napi(
    js_name = "__testClosedPromiseAwareCall",
    ts_return_type = "Promise<unknown>"
)]
pub fn test_closed_promise_aware_call(env: Env, func: JsFunction) -> Result<JsObject> {
    let promise_aware = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &func).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );
    promise_aware.close();

    env.execute_tokio_future(
        async move { promise_aware.call(Json::Null).await.map_err(to_napi_err) },
        |_env, result| Ok(result),
    )
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Get the handle for the current top-of-stack execution scope.
///
/// Returns the `ScopeHandle` for the innermost active scope on the current task's scope stack.
/// Throws if the scope stack is empty.
#[napi]
pub fn get_handle() -> Result<ScopeHandle> {
    core_scope_api::get_handle()
        .map(ScopeHandle::from)
        .map_err(to_napi_err)
}

/// Push a new execution scope onto the scope stack.
///
/// Creates a child scope with the given `name` and `scopeType`. If `handle` is provided,
/// the new scope is parented to that scope; otherwise it is parented to the current top scope.
/// Optional `attributes` is a bitfield of scope attribute flags.
/// Optional `data` is a JSON application payload stored on the scope handle.
/// Optional `metadata` is a JSON metadata payload recorded on the scope start event.
/// Optional `input` is a semantic JSON payload exported on the scope start event.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded as the handle
/// start time and start event timestamp. It must be a safe integer number; omit it
/// to use the current runtime time.
/// Returns the handle for the newly created scope.
#[napi]
#[allow(clippy::too_many_arguments)]
pub fn push_scope(
    name: String,
    scope_type: ScopeType,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    input: Option<Json>,
    timestamp: Option<f64>,
) -> Result<ScopeHandle> {
    let attrs = ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let timestamp = parse_timestamp_micros(timestamp)?;
    core_scope_api::push_scope(
        core_scope_api::PushScopeParams::builder()
            .name(name.as_str())
            .scope_type(scope_type.into())
            .parent_opt(handle.map(|h| &h.inner))
            .attributes(attrs)
            .data_opt(opt_json(data))
            .metadata_opt(opt_json(metadata))
            .input_opt(opt_json(input))
            .timestamp_opt(timestamp)
            .build(),
    )
    .map(ScopeHandle::from)
    .map_err(to_napi_err)
}

/// Pop an execution scope from the scope stack.
///
/// Removes the scope identified by `handle` from the stack and emits an end event.
/// Optional `output` is a semantic JSON payload exported on the scope end event.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded on the end event.
/// It must be a safe integer number; omit it to use the runtime default end timestamp.
/// Optional `metadata` is a JSON metadata payload recorded on the scope end event.
/// Throws if the handle does not match the current top scope.
#[napi]
pub fn pop_scope(
    handle: &ScopeHandle,
    output: Option<Json>,
    timestamp: Option<f64>,
    metadata: Option<Json>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    core_scope_api::pop_scope(
        core_scope_api::PopScopeParams::builder()
            .handle_uuid(&handle.inner.uuid)
            .output_opt(opt_json(output))
            .timestamp_opt(timestamp)
            .metadata_opt(opt_json(metadata))
            .build(),
    )
    .map_err(to_napi_err)?;
    forget_scope_local_promise_aware(&handle.inner.uuid.to_string());
    Ok(())
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
/// attribute flags; `data` is stored on the scope handle; `metadata` is recorded
/// on the start event; and `input` is exported as the semantic start-event payload.
///
/// Returns a Promise that resolves with the callback's return value.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>")]
pub fn with_scope(
    env: Env,
    name: String,
    scope_type: ScopeType,
    callback: napi::JsFunction,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    input: Option<Json>,
) -> Result<JsObject> {
    let attrs = ScopeAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let scope_handle = core_scope_api::push_scope(
        core_scope_api::PushScopeParams::builder()
            .name(name.as_str())
            .scope_type(scope_type.into())
            .parent_opt(handle.map(|h| &h.inner))
            .attributes(attrs)
            .data_opt(opt_json(data))
            .metadata_opt(opt_json(metadata))
            .input_opt(opt_json(input))
            .build(),
    )
    .map(ScopeHandle::from)
    .map_err(to_napi_err)?;

    let scope_stack = current_scope_stack_handle();
    let scope_uuid = scope_handle.inner.uuid;
    // Hand the callback a real `ScopeHandle` instance, matching the Rust,
    // Python bindings, so it can be passed back into `event`,
    // `toolCallExecute`, and `llmCallExecute`. The instance is materialized on
    // the JS thread because a `napi_wrap`'d handle cannot cross the
    // threadsafe-function boundary as plain JSON.
    let callback_handle = scope_handle.inner.clone();

    // Create a promise-aware wrapper so we handle both sync and async callbacks.
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callback).map_err(|e| {
            let status_message = format!("failed to create PromiseAwareFn: {e}");
            let _ = core_scope_api::pop_scope(
                core_scope_api::PopScopeParams::builder()
                    .handle_uuid(&scope_uuid)
                    .metadata_opt(Some(otel_status_metadata(
                        "ERROR",
                        Some(status_message.clone()),
                    )))
                    .build(),
            );
            napi::Error::from_reason(status_message)
        })?,
    );

    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    let build_handle: crate::promise_call::Arg0Builder =
                        Box::new(move |env: &Env| {
                            let raw = unsafe {
                                <ScopeHandle as ToNapiValue>::to_napi_value(
                                    env.raw(),
                                    ScopeHandle::from(callback_handle),
                                )?
                            };
                            Ok(unsafe { JsUnknown::from_raw_unchecked(env.raw(), raw) })
                        });

                    let result = pa_fn.call_with_arg0(build_handle).await;
                    let metadata = match &result {
                        Ok(_) => otel_status_metadata("OK", None),
                        Err(error) => otel_status_metadata("ERROR", Some(error.to_string())),
                    };
                    // Always pop the scope, even on error.
                    if core_scope_api::pop_scope(
                        core_scope_api::PopScopeParams::builder()
                            .handle_uuid(&scope_uuid)
                            .metadata_opt(Some(metadata))
                            .build(),
                    )
                    .is_ok()
                    {
                        forget_scope_local_promise_aware(&scope_uuid.to_string());
                    }
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
/// Optional `timestamp` is a Unix timestamp in microseconds recorded on the mark event.
/// It must be a safe integer number; omit it to use the current runtime time.
#[napi]
pub fn event(
    name: String,
    handle: Option<&ScopeHandle>,
    data: Option<Json>,
    metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    core_scope_api::event(
        core_scope_api::EmitMarkEventParams::builder()
            .name(&name)
            .parent_opt(handle.map(|h| &h.inner))
            .data_opt(opt_json(data))
            .metadata_opt(opt_json(metadata))
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begin a manual tool call lifecycle span.
///
/// Registers a tool invocation with the given `name` and `args`. Sanitize-request
/// guardrails are applied to the emitted start-event payload; request and execution
/// intercepts run only through `toolCallExecute`. Returns a `ToolHandle` that must
/// be passed to `toolCallEnd()` when the tool finishes. Optional `handle` specifies
/// the parent scope; `attributes` is a bitfield; `data` is stored on the handle;
/// `metadata` is recorded on the start event; and `toolCallId` is recorded in the
/// tool event category profile. Optional `timestamp` is a Unix timestamp in
/// microseconds recorded as the handle start time and start event timestamp. It must
/// be a safe integer number; omit it to use the current runtime time.
#[napi]
#[allow(clippy::too_many_arguments)]
pub fn tool_call(
    name: String,
    args: Json,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    tool_call_id: Option<String>,
    timestamp: Option<f64>,
) -> Result<ToolHandle> {
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let timestamp = parse_timestamp_micros(timestamp)?;
    core_tool_api::tool_call(
        core_tool_api::ToolCallParams::builder()
            .name(name.as_str())
            .args(args)
            .parent_opt(handle.map(|h| &h.inner))
            .attributes(attrs)
            .data_opt(opt_json(data))
            .metadata_opt(opt_json(metadata))
            .tool_call_id_opt(tool_call_id)
            .timestamp_opt(timestamp)
            .build(),
    )
    .map(ToolHandle::from)
    .map_err(to_napi_err)
}

/// End a manual tool call lifecycle span.
///
/// Signals that the tool call identified by `handle` has completed with the given `result`.
/// Sanitize-response guardrails are applied to the emitted end-event payload; response
/// intercepts run only through `toolCallExecute`. Optional `data` is used when the
/// sanitized result is JSON null, and optional `metadata` is recorded on the end event.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded on the end event.
/// It must be a safe integer number; omit it to use the runtime default end timestamp.
#[napi]
pub fn tool_call_end(
    handle: &ToolHandle,
    result: Json,
    data: Option<Json>,
    metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    core_tool_api::tool_call_end(
        core_tool_api::ToolCallEndParams::builder()
            .handle(&handle.inner)
            .result(result)
            .data_opt(opt_json(data))
            .metadata_opt(opt_json(metadata))
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_napi_err)
}

/// Execute a tool call end-to-end with full lifecycle management.
///
/// Runs conditional-execution guardrails (on raw args) → request intercepts →
/// sanitize-request guardrails for the emitted `Start` event payload →
/// execution intercepts → `func` → sanitize-response guardrails for the emitted
/// `End` event payload. On rejection, only a standalone Mark event is emitted
/// (no Start/End pair) and `GuardrailRejected` is returned. Returns the final
/// execution result; sanitize guardrails do not rewrite the caller-visible value.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>")]
pub fn tool_call_execute(
    env: Env,
    name: String,
    args: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<JsObject> {
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(task_scope_top);
    let exec_fn = callable::wrap_js_tool_exec_fn(func);
    let default_fn: ToolExecutionNextFn = std::sync::Arc::new(move |args| exec_fn(args));
    let scope_stack = current_scope_stack_handle();

    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core_tool_api::tool_call_execute(
                        core_tool_api::ToolCallExecuteParams::builder()
                            .name(name)
                            .args(args)
                            .func(default_fn)
                            .parent(parent)
                            .attributes(attrs)
                            .data_opt(opt_json(data))
                            .metadata_opt(opt_json(metadata))
                            .build(),
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
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<JsObject> {
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(task_scope_top);
    let scope_stack = current_scope_stack_handle();

    // Create promise-aware wrapper — this must happen on the JS thread (we have Env).
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &func).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );

    let exec_fn: ToolExecutionNextFn = std::sync::Arc::new(move |args| {
        let pa_fn = pa_fn.clone();
        Box::pin(async move { pa_fn.call(args).await })
    });

    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core_tool_api::tool_call_execute(
                        core_tool_api::ToolCallExecuteParams::builder()
                            .name(name)
                            .args(args)
                            .func(exec_fn)
                            .parent(parent)
                            .attributes(attrs)
                            .data_opt(opt_json(data))
                            .metadata_opt(opt_json(metadata))
                            .build(),
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

/// Begin a manual LLM call lifecycle span.
///
/// Registers an LLM invocation with the given provider `name` and request payload.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LlmRequest` schema. Returns an `LlmHandle` that must be passed to `llmCallEnd()`
/// when the response is received. Sanitize-request guardrails are applied to the emitted
/// start-event payload; request and execution intercepts run only through `llmCallExecute`.
/// Optional `handle` specifies the parent scope; `attributes` is a bitfield; `data` is
/// stored on the handle; `metadata` is recorded on the start event; and `modelName` is
/// recorded in the LLM event category profile. Optional `timestamp` is a Unix timestamp
/// in microseconds recorded as the handle start time and start event timestamp. It must
/// be a safe integer number; omit it to use the current runtime time.
#[allow(clippy::too_many_arguments)]
#[napi]
pub fn llm_call(
    name: String,
    request: Json,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    timestamp: Option<f64>,
) -> Result<LlmHandle> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let timestamp = parse_timestamp_micros(timestamp)?;
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    let params = core_llm_api::LlmCallParams::builder()
        .name(&name)
        .request(&llm_request)
        .parent_opt(handle.map(|h| &h.inner))
        .attributes(attrs)
        .data_opt(opt_json(data))
        .metadata_opt(opt_json(metadata))
        .model_name_opt(model_name)
        .timestamp_opt(timestamp)
        .build();
    core_llm_api::llm_call(params)
        .map(LlmHandle::from)
        .map_err(to_napi_err)
}

/// End a manual LLM call lifecycle span.
///
/// Signals that the LLM call identified by `handle` has completed with the given `response`.
/// Sanitize-response guardrails are applied to the emitted end-event payload; response
/// intercepts run only through `llmCallExecute`. Optional `data` is used when the
/// sanitized response is JSON null, and optional `metadata` is recorded on the end event.
/// Optional `timestamp` is a Unix timestamp in microseconds recorded on the end event.
/// It must be a safe integer number; omit it to use the runtime default end timestamp.
#[napi]
pub fn llm_call_end(
    handle: &LlmHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    core_llm_api::llm_call_end(
        core_llm_api::LlmCallEndParams::builder()
            .handle(&handle.inner)
            .response(response)
            .data_opt(opt_json(data))
            .metadata_opt(opt_json(metadata))
            .timestamp_opt(timestamp)
            .build(),
    )
    .map_err(to_napi_err)
}

/// Execute an LLM call end-to-end with full lifecycle management.
///
/// Runs conditional-execution guardrails (on raw request) → request intercepts →
/// sanitize-request guardrails for the emitted `Start` event payload →
/// execution intercepts → `func` → sanitize-response guardrails for the emitted
/// `End` event payload. On rejection, only a standalone Mark event is emitted
/// (no Start/End pair) and `GuardrailRejected` is returned. The `request`
/// should be a JSON object with `headers` and `content` fields matching the
/// `LlmRequest` schema. Returns the final execution response; sanitize
/// guardrails do not rewrite the caller-visible value.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>")]
pub fn llm_call_execute(
    env: Env,
    name: String,
    request: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    codec_decode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    codec_encode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    response_codec_decode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
) -> Result<JsObject> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(task_scope_top);
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    let exec_fn = callable::wrap_js_llm_exec_fn(func);
    let default_fn: LlmExecutionNextFn = std::sync::Arc::new(move |req| exec_fn(req));
    let codec = match (codec_decode, codec_encode) {
        (Some(d), Some(e)) => Some(callable::wrap_js_codec(d, e)),
        _ => None,
    };
    let response_codec = response_codec_decode.map(callable::wrap_js_response_codec);
    let scope_stack = current_scope_stack_handle();

    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    let params = core_llm_api::LlmCallExecuteParams::builder()
                        .name(name)
                        .request(llm_request)
                        .func(default_fn)
                        .parent(parent)
                        .attributes(attrs)
                        .data_opt(opt_json(data))
                        .metadata_opt(opt_json(metadata))
                        .model_name_opt(model_name)
                        .codec_opt(codec)
                        .response_codec_opt(response_codec)
                        .build();
                    core_llm_api::llm_call_execute(params)
                        .await
                        .map_err(to_napi_err)
                })
                .await
        },
        |_env, result| Ok(result),
    )
}

/// Execute an LLM call end-to-end, supporting both sync and async (Promise-returning) callbacks.
///
/// Same lifecycle as `llmCallExecute` (guardrails → intercepts → func → response processing),
/// but transparently handles JS callbacks that return Promises.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<unknown>")]
pub fn llm_call_execute_async(
    env: Env,
    name: String,
    request: Json,
    func: JsFunction,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    codec_decode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    codec_encode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    response_codec_decode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
) -> Result<JsObject> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(task_scope_top);
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    let scope_stack = current_scope_stack_handle();

    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &func).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );

    let exec_fn: LlmExecutionNextFn = std::sync::Arc::new(move |req| {
        let pa_fn = pa_fn.clone();
        let req_json = serde_json::to_value(&req).unwrap_or(Json::Null);
        Box::pin(async move { pa_fn.call(req_json).await })
    });

    let codec = match (codec_decode, codec_encode) {
        (Some(d), Some(e)) => Some(callable::wrap_js_codec(d, e)),
        _ => None,
    };
    let response_codec = response_codec_decode.map(callable::wrap_js_response_codec);

    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    let params = core_llm_api::LlmCallExecuteParams::builder()
                        .name(name)
                        .request(llm_request)
                        .func(exec_fn)
                        .parent(parent)
                        .attributes(attrs)
                        .data_opt(opt_json(data))
                        .metadata_opt(opt_json(metadata))
                        .model_name_opt(model_name)
                        .codec_opt(codec)
                        .response_codec_opt(response_codec)
                        .build();
                    core_llm_api::llm_call_execute(params)
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
/// Sanitize-request guardrails only affect the emitted `Start` event payload, and
/// sanitize-response guardrails only affect the aggregated `End` event payload.
/// Returns an `LlmStream` whose `next()` method yields response chunks incrementally.
/// The `func` callback receives the intercepted request as JSON and its response is streamed back.
/// Stream-level intercepts are applied to each chunk.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LlmRequest` schema.
///
/// The optional `collector` callback is invoked with each intercepted chunk as JSON,
/// allowing the caller to accumulate chunks for aggregation. The optional `finalizer`
/// callback is invoked once when the stream is exhausted and must return a JSON value
/// representing the aggregated response.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<LlmStream>")]
pub fn llm_stream_call_execute(
    env: Env,
    name: String,
    request: Json,
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    collector: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    finalizer: Option<ThreadsafeFunction<(), ErrorStrategy::Fatal>>,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    codec_decode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    codec_encode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    response_codec_decode: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
) -> Result<JsObject> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(task_scope_top);
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;

    let wrapped_collector: Box<dyn FnMut(Json) -> FlowResult<()> + Send> = match collector {
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
    let default_fn: LlmStreamExecutionNextFn = std::sync::Arc::new(move |req: LlmRequest| {
        let stream_id = NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        register_stream_channel(stream_id, tx);

        // Serialize the LlmRequest to JSON and wrap with streamId so JS can extract both
        let req_json = serde_json::to_value(&req).unwrap_or(Json::Null);
        let wrapper = serde_json::json!({
            "__nemo_relay_native": req_json,
            "__nemo_relay_stream_id": stream_id,
        });

        // NonBlocking: queue the call on the JS event loop and return immediately.
        // The JS function starts async iteration and pushes chunks via pushStreamChunk.
        let call_status = func.call(wrapper, ThreadsafeFunctionCallMode::NonBlocking);

        Box::pin(async move {
            ensure_stream_callback_queued(stream_id, call_status)?;

            let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
            Ok(Box::pin(stream)
                as std::pin::Pin<
                    Box<dyn tokio_stream::Stream<Item = FlowResult<Json>> + Send>,
                >)
        })
    });

    let codec = match (codec_decode, codec_encode) {
        (Some(d), Some(e)) => Some(callable::wrap_js_codec(d, e)),
        _ => None,
    };
    let response_codec = response_codec_decode.map(callable::wrap_js_response_codec);
    let scope_stack = current_scope_stack_handle();

    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    let params = core_llm_api::LlmStreamCallExecuteParams::builder()
                        .name(name)
                        .request(llm_request)
                        .func(default_fn)
                        .collector(wrapped_collector)
                        .finalizer(wrapped_finalizer)
                        .parent(parent)
                        .attributes(attrs)
                        .data_opt(opt_json(data))
                        .metadata_opt(opt_json(metadata))
                        .model_name_opt(model_name)
                        .codec_opt(codec)
                        .response_codec_opt(response_codec)
                        .build();
                    let rust_stream = core_llm_api::llm_stream_call_execute(params)
                        .await
                        .map_err(to_napi_err)?;

                    let (tx, rx) = tokio::sync::mpsc::channel(32);
                    tokio::spawn(forward_stream_to_channel(rust_stream, tx));

                    Ok(LlmStream {
                        receiver: tokio::sync::Mutex::new(rx),
                    })
                })
                .await
        },
        |_env, result| Ok(result),
    )
}

// ---------------------------------------------------------------------------
// Tool guardrail registrations
// ---------------------------------------------------------------------------

macro_rules! napi_guardrail_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
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
    core_registry_api::register_tool_sanitize_request_guardrail,
    core_registry_api::deregister_tool_sanitize_request_guardrail,
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
    core_registry_api::register_tool_sanitize_response_guardrail,
    core_registry_api::deregister_tool_sanitize_response_guardrail,
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
    core_registry_api::register_tool_conditional_execution_guardrail(
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
    core_registry_api::deregister_tool_conditional_execution_guardrail(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Tool intercept registrations
// ---------------------------------------------------------------------------

macro_rules! napi_intercept_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
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
    core_registry_api::register_tool_request_intercept,
    core_registry_api::deregister_tool_request_intercept,
    callable::wrap_js_tool_request_intercept_fn
);

/// Register a tool execution intercept following the middleware chain pattern.
///
/// The `callable` receives the args and a `next` function. Call `next(args)` to invoke
/// the next intercept or original implementation; skip calling `next` to short-circuit
/// the chain.
#[napi]
pub fn register_tool_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(args: Json, next: (args: Json) => Json | Promise<Json>) => { result: Json; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }> } | Promise<{ result: Json; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }> }>"
    )]
    callable: JsFunction,
) -> Result<()> {
    let key = PromiseAwareKey::GlobalToolExecution(name.clone());
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );
    core_registry_api::register_tool_execution_intercept(
        &name,
        priority,
        callable::wrap_js_tool_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(to_napi_err)?;
    remember_promise_aware(key, pa_fn);
    Ok(())
}

/// Deregister a tool execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_tool_execution_intercept(name: String) -> Result<bool> {
    let key = PromiseAwareKey::GlobalToolExecution(name.clone());
    let removed =
        core_registry_api::deregister_tool_execution_intercept(&name).map_err(to_napi_err)?;
    if removed {
        forget_promise_aware(&key);
    }
    Ok(removed)
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
    core_registry_api::register_llm_sanitize_request_guardrail(
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
    core_registry_api::deregister_llm_sanitize_request_guardrail(&name).map_err(to_napi_err)
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
    core_registry_api::register_llm_sanitize_response_guardrail(
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
    core_registry_api::deregister_llm_sanitize_response_guardrail(&name).map_err(to_napi_err)
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
    core_registry_api::register_llm_conditional_execution_guardrail(
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
    core_registry_api::deregister_llm_conditional_execution_guardrail(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an intercept that transforms LLM request data.
///
/// The `callable` receives the `LlmRequest` (as JSON) and returns a transformed request.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
#[napi]
pub fn register_llm_request_intercept(
    name: String,
    priority: i32,
    break_chain: bool,
    #[napi(
        ts_arg_type = "(args: { name: string; request: Json; annotated: Json | null }) => { request: Json; annotated?: Json | null; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }>; optimizationContributions?: Array<{ id?: string; sequence?: number; producer: string; kind: string; applied: boolean; model_transition?: { baseline?: { model: string; provider?: string }; effective?: { model: string; provider?: string } }; token_impact?: { baseline?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; effective?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; saved?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; quality?: 'observed' | 'estimated'; estimation_method?: string }; payload_schema?: { name: string; version: string }; payload?: Json; [key: string]: Json | undefined }> }"
    )]
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core_registry_api::register_llm_request_intercept(
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
    core_registry_api::deregister_llm_request_intercept(&name).map_err(to_napi_err)
}

/// Register an LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original implementation; skip calling `next` to
/// short-circuit the chain.
#[napi]
pub fn register_llm_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    callable: JsFunction,
) -> Result<()> {
    let key = PromiseAwareKey::GlobalLlmExecution(name.clone());
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );
    core_registry_api::register_llm_execution_intercept(
        &name,
        priority,
        callable::wrap_js_llm_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(to_napi_err)?;
    remember_promise_aware(key, pa_fn);
    Ok(())
}

/// Deregister an LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_execution_intercept(name: String) -> Result<bool> {
    let key = PromiseAwareKey::GlobalLlmExecution(name.clone());
    let removed =
        core_registry_api::deregister_llm_execution_intercept(&name).map_err(to_napi_err)?;
    if removed {
        forget_promise_aware(&key);
    }
    Ok(removed)
}

/// Register a streaming LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original streaming implementation; in Node the
/// returned promise resolves to an array of downstream JSON chunks. Skip calling
/// `next` to short-circuit the chain.
#[napi]
pub fn register_llm_stream_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    callable: JsFunction,
) -> Result<()> {
    let key = PromiseAwareKey::GlobalLlmStreamExecution(name.clone());
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );
    core_registry_api::register_llm_stream_execution_intercept(
        &name,
        priority,
        callable::wrap_js_llm_stream_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(to_napi_err)?;
    remember_promise_aware(key, pa_fn);
    Ok(())
}

/// Deregister an LLM stream execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_stream_execution_intercept(name: String) -> Result<bool> {
    let key = PromiseAwareKey::GlobalLlmStreamExecution(name.clone());
    let removed =
        core_registry_api::deregister_llm_stream_execution_intercept(&name).map_err(to_napi_err)?;
    if removed {
        forget_promise_aware(&key);
    }
    Ok(removed)
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Register a named event subscriber that receives all lifecycle events.
///
/// The `callback` receives each event as the canonical JSON event object. Events are
/// delivered asynchronously and non-blocking. Throws if a subscriber with the same `name`
/// already exists.
#[napi]
pub fn register_subscriber(
    name: String,
    callback: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    core_subscriber_api::register_subscriber(&name, callable::wrap_js_event_subscriber(callback))
        .map_err(to_napi_err)
}

/// Deregister an event subscriber by name.
///
/// Future emissions stop seeing the subscriber. Already queued event snapshots may
/// still run. Returns `true` if a subscriber with that name was found and removed.
#[napi]
pub fn deregister_subscriber(name: String) -> Result<bool> {
    core_subscriber_api::deregister_subscriber(&name).map_err(to_napi_err)
}

/// Wait for native subscriber callbacks queued before this call to finish.
///
/// JavaScript subscribers are queued through Node's `ThreadsafeFunction`; callers that
/// need JS callback side effects should await an event-loop tick after this returns.
#[napi]
pub fn flush_subscribers() -> Result<()> {
    core_subscriber_api::flush_subscribers().map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations — Tool
// ---------------------------------------------------------------------------

macro_rules! napi_scope_guardrail_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
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
    core_registry_api::scope_register_tool_sanitize_request_guardrail,
    core_registry_api::scope_deregister_tool_sanitize_request_guardrail,
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
    core_registry_api::scope_register_tool_sanitize_response_guardrail,
    core_registry_api::scope_deregister_tool_sanitize_response_guardrail,
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
    core_registry_api::scope_register_tool_conditional_execution_guardrail(
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
    core_registry_api::scope_deregister_tool_conditional_execution_guardrail(&uuid, &name)
        .map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — Tool
// ---------------------------------------------------------------------------

macro_rules! napi_scope_intercept_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
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
    core_registry_api::scope_register_tool_request_intercept,
    core_registry_api::scope_deregister_tool_request_intercept,
    callable::wrap_js_tool_request_intercept_fn
);

/// Register a scope-local tool execution intercept following the middleware chain pattern.
///
/// The `callable` receives the args and a `next` function. Call `next(args)` to invoke
/// the next intercept or original implementation; skip calling `next` to short-circuit
/// the chain.
#[napi]
pub fn scope_register_tool_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(args: Json, next: (args: Json) => Json | Promise<Json>) => { result: Json; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }> } | Promise<{ result: Json; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }> }>"
    )]
    callable: JsFunction,
) -> Result<()> {
    let key = PromiseAwareKey::ScopeToolExecution {
        scope_uuid: scope_uuid.clone(),
        name: name.clone(),
    };
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );
    core_registry_api::scope_register_tool_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_tool_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(to_napi_err)?;
    remember_promise_aware(key, pa_fn);
    Ok(())
}

/// Deregister a scope-local tool execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_tool_execution_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let key = PromiseAwareKey::ScopeToolExecution {
        scope_uuid: scope_uuid.clone(),
        name: name.clone(),
    };
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    let removed = core_registry_api::scope_deregister_tool_execution_intercept(&uuid, &name)
        .map_err(to_napi_err)?;
    if removed {
        forget_promise_aware(&key);
    }
    Ok(removed)
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
    core_registry_api::scope_register_llm_sanitize_request_guardrail(
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
    core_registry_api::scope_deregister_llm_sanitize_request_guardrail(&uuid, &name)
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
    core_registry_api::scope_register_llm_sanitize_response_guardrail(
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
    core_registry_api::scope_deregister_llm_sanitize_response_guardrail(&uuid, &name)
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
    core_registry_api::scope_register_llm_conditional_execution_guardrail(
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
    core_registry_api::scope_deregister_llm_conditional_execution_guardrail(&uuid, &name)
        .map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local intercept registrations — LLM
// ---------------------------------------------------------------------------

/// Register a scope-local intercept that transforms LLM request data.
///
/// The `callable` receives the `LlmRequest` (as JSON) and returns a transformed request.
/// If `breakChain` is `true`, no lower-priority intercepts run after this one.
/// Higher `priority` values run first.
#[napi]
pub fn scope_register_llm_request_intercept(
    scope_uuid: String,
    name: String,
    priority: i32,
    break_chain: bool,
    #[napi(
        ts_arg_type = "(args: { name: string; request: Json; annotated: Json | null }) => { request: Json; annotated?: Json | null; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }>; optimizationContributions?: Array<{ id?: string; sequence?: number; producer: string; kind: string; applied: boolean; model_transition?: { baseline?: { model: string; provider?: string }; effective?: { model: string; provider?: string } }; token_impact?: { baseline?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; effective?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; saved?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; quality?: 'observed' | 'estimated'; estimation_method?: string }; payload_schema?: { name: string; version: string }; payload?: Json; [key: string]: Json | undefined }> }"
    )]
    callable: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_llm_request_intercept(
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
    core_registry_api::scope_deregister_llm_request_intercept(&uuid, &name).map_err(to_napi_err)
}

/// Register a scope-local LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original implementation; skip calling `next` to
/// short-circuit the chain.
#[napi]
pub fn scope_register_llm_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    callable: JsFunction,
) -> Result<()> {
    let key = PromiseAwareKey::ScopeLlmExecution {
        scope_uuid: scope_uuid.clone(),
        name: name.clone(),
    };
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );
    core_registry_api::scope_register_llm_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(to_napi_err)?;
    remember_promise_aware(key, pa_fn);
    Ok(())
}

/// Deregister a scope-local LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_execution_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let key = PromiseAwareKey::ScopeLlmExecution {
        scope_uuid: scope_uuid.clone(),
        name: name.clone(),
    };
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    let removed = core_registry_api::scope_deregister_llm_execution_intercept(&uuid, &name)
        .map_err(to_napi_err)?;
    if removed {
        forget_promise_aware(&key);
    }
    Ok(removed)
}

/// Register a scope-local streaming LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original streaming implementation; in Node the
/// returned promise resolves to an array of downstream JSON chunks. Skip calling
/// `next` to short-circuit the chain.
#[napi]
pub fn scope_register_llm_stream_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    callable: JsFunction,
) -> Result<()> {
    let key = PromiseAwareKey::ScopeLlmStreamExecution {
        scope_uuid: scope_uuid.clone(),
        name: name.clone(),
    };
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );
    core_registry_api::scope_register_llm_stream_execution_intercept(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_stream_exec_intercept_fn(pa_fn.clone()),
    )
    .map_err(to_napi_err)?;
    remember_promise_aware(key, pa_fn);
    Ok(())
}

/// Deregister a scope-local LLM stream execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_stream_execution_intercept(
    scope_uuid: String,
    name: String,
) -> Result<bool> {
    let key = PromiseAwareKey::ScopeLlmStreamExecution {
        scope_uuid: scope_uuid.clone(),
        name: name.clone(),
    };
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    let removed = core_registry_api::scope_deregister_llm_stream_execution_intercept(&uuid, &name)
        .map_err(to_napi_err)?;
    if removed {
        forget_promise_aware(&key);
    }
    Ok(removed)
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registrations
// ---------------------------------------------------------------------------

/// Register a scope-local named event subscriber that receives lifecycle events
/// for the specified scope.
///
/// The `callback` receives each event as the canonical JSON event object. Events are
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
    core_subscriber_api::scope_register_subscriber(
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
    core_subscriber_api::scope_deregister_subscriber(&uuid, &name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Run the registered tool request intercept chain on the given arguments.
/// Returns the transformed arguments.
#[napi(ts_return_type = "Promise<unknown>")]
pub fn tool_request_intercepts(env: Env, name: String, args: Json) -> Result<JsObject> {
    let scope_stack = current_scope_stack_handle();
    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core_tool_api::tool_request_intercepts(&name, args).map_err(to_napi_err)
                })
                .await
        },
        |_env, result| Ok(result),
    )
}

/// Run the registered tool conditional execution guardrail chain.
/// Throws if any guardrail rejects.
#[napi(ts_return_type = "Promise<void>")]
pub fn tool_conditional_execution(env: Env, name: String, args: Json) -> Result<JsObject> {
    let scope_stack = current_scope_stack_handle();
    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core_tool_api::tool_conditional_execution(&name, &args).map_err(to_napi_err)
                })
                .await
        },
        |env, _| env.get_undefined(),
    )
}

/// Run the registered LLM request intercept chain on the given request.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LlmRequest` schema. Returns the transformed request as JSON.
#[napi(
    ts_return_type = "Promise<{ request: Json; annotated: Json | null; pendingMarks: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }>; optimizationContributions: Array<{ id?: string; sequence?: number; producer: string; kind: string; applied: boolean; model_transition?: { baseline?: { model: string; provider?: string }; effective?: { model: string; provider?: string } }; token_impact?: { baseline?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; effective?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; saved?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; quality?: 'observed' | 'estimated'; estimation_method?: string }; payload_schema?: { name: string; version: string }; payload?: Json; [key: string]: Json | undefined }> }>"
)]
pub fn llm_request_intercepts(env: Env, name: String, request: Json) -> Result<JsObject> {
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    let scope_stack = current_scope_stack_handle();
    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core_llm_api::llm_request_intercepts(&name, llm_request)
                        .map(|r| {
                            serde_json::json!({
                                "request": r.request,
                                "annotated": r.annotated_request,
                                "pendingMarks": callable::js_pending_marks(r.pending_marks),
                                "optimizationContributions": r.optimization_contributions,
                            })
                        })
                        .map_err(to_napi_err)
                })
                .await
        },
        |_env, result| Ok(result),
    )
}

/// Run the registered LLM conditional execution guardrail chain.
/// Throws if any guardrail rejects. The `request` should be a JSON object with `headers`
/// and `content` fields matching the `LlmRequest` schema.
#[napi(ts_return_type = "Promise<void>")]
pub fn llm_conditional_execution(env: Env, request: Json) -> Result<JsObject> {
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    let scope_stack = current_scope_stack_handle();
    env.execute_tokio_future(
        async move {
            TASK_SCOPE_STACK
                .scope(scope_stack, async move {
                    core_llm_api::llm_conditional_execution(&llm_request).map_err(to_napi_err)
                })
                .await
        },
        |env, _| env.get_undefined(),
    )
}

// ---------------------------------------------------------------------------
// Agent Trajectory Interchange Format (ATIF) Exporter
// ---------------------------------------------------------------------------

/// An Agent Trajectory Interchange Format (ATIF) exporter that collects lifecycle
/// events and exports them as a structured trajectory.
///
/// Create an instance with session and agent metadata, then register it as an event subscriber.
/// When ready, call `exportJson()` to serialize the collected trajectory.
#[napi]
pub struct AtifExporter {
    inner: nemo_relay::observability::atif::AtifExporter,
}

#[napi]
impl AtifExporter {
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
        let agent_info = nemo_relay::observability::atif::AtifAgentInfo {
            name: agent_name,
            version: agent_version,
            model_name,
            tool_definitions: None,
            extra: None,
        };
        Ok(Self {
            inner: nemo_relay::observability::atif::AtifExporter::new(session_id, agent_info),
        })
    }

    /// Register this exporter as an event subscriber with the given name.
    ///
    /// Throws if a subscriber with the same `name` already exists.
    #[napi]
    pub fn register(&self, name: String) -> napi::Result<()> {
        let subscriber = self.inner.subscriber();
        core_subscriber_api::register_subscriber(&name, subscriber)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Deregister this exporter's event subscriber by name.
    ///
    /// Returns `true` if a subscriber with that name was found and removed.
    #[napi]
    pub fn deregister(&self, name: String) -> napi::Result<bool> {
        core_subscriber_api::deregister_subscriber(&name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Export the collected trajectory as a JSON string.
    ///
    /// Returns a JSON-serialized `AtifTrajectory`.
    #[napi]
    pub fn export_json(&self) -> napi::Result<String> {
        let trajectory = self
            .inner
            .try_export()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&trajectory).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Clear all collected events from the exporter.
    #[napi]
    pub fn clear(&self) {
        self.inner.clear();
    }
}

/// Mutable configuration object for `AtofExporter`.
#[napi(object)]
#[derive(Default)]
pub struct AtofExporterConfig {
    /// Output directory. Defaults to the current working directory.
    pub output_directory: Option<String>,
    /// `"append"` (default) or `"overwrite"`.
    pub mode: Option<String>,
    /// Output filename. Defaults to `nemo-relay-events-YYYY-MM-DD-HH.MM.SS.jsonl`.
    pub filename: Option<String>,
    /// Streaming endpoints that receive every raw ATOF event.
    pub endpoints: Option<Vec<AtofEndpointConfig>>,
}

/// Mutable configuration object for one ATOF streaming endpoint.
#[napi(object)]
#[derive(Default)]
pub struct AtofEndpointConfig {
    /// Endpoint URL.
    pub url: String,
    /// `"http_post"` (default), `"websocket"`, or `"ndjson"`.
    pub transport: Option<String>,
    /// Extra endpoint headers as string key/value pairs.
    pub headers: Option<Json>,
    /// Per-endpoint timeout in milliseconds.
    pub timeout_millis: Option<u32>,
    /// Field name policy applied before sending events.
    pub field_name_policy: Option<String>,
}

/// Filesystem-backed Agent Trajectory Observability Format (ATOF) JSONL event exporter.
#[napi]
pub struct AtofExporter {
    inner: nemo_relay::observability::atof::AtofExporter,
}

#[napi]
impl AtofExporter {
    /// Create a new Agent Trajectory Observability Format (ATOF) JSONL exporter
    /// from a config object.
    #[napi(constructor)]
    pub fn new(config: Option<AtofExporterConfig>) -> napi::Result<Self> {
        let inner = nemo_relay::observability::atof::AtofExporter::new(build_atof_config(config)?)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Return the JSONL output path.
    #[napi(getter)]
    pub fn path(&self) -> String {
        self.inner.path().to_string_lossy().into_owned()
    }

    /// Register this exporter globally with the given name.
    #[napi]
    pub fn register(&self, name: String) -> napi::Result<()> {
        self.inner
            .register(&name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Deregister a subscriber by name.
    #[napi]
    pub fn deregister(&self, name: String) -> napi::Result<bool> {
        self.inner
            .deregister(&name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Flush the output file.
    #[napi]
    pub fn force_flush(&self) -> napi::Result<()> {
        self.inner
            .force_flush()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Shut down the exporter by flushing output.
    #[napi]
    pub fn shutdown(&self) -> napi::Result<()> {
        self.inner
            .shutdown()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Mutable configuration object for `OpenTelemetrySubscriber`.
#[napi(object)]
#[derive(Default)]
pub struct OpenTelemetryConfig {
    /// `"http_binary"` (default) or `"grpc"`.
    pub transport: Option<String>,
    /// OTLP endpoint, such as `http://localhost:4318/v1/traces`.
    pub endpoint: Option<String>,
    /// Extra exporter headers/metadata as string key/value pairs.
    pub headers: Option<Json>,
    /// Extra OpenTelemetry resource attributes as string key/value pairs.
    pub resource_attributes: Option<Json>,
    /// `service.name` resource attribute. Defaults to `"nemo-relay"`.
    pub service_name: Option<String>,
    /// Optional `service.namespace` resource attribute.
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    pub service_version: Option<String>,
    /// Instrumentation scope name. Defaults to `"nemo-relay-otel"`.
    pub instrumentation_scope: Option<String>,
    /// Export timeout in milliseconds. Defaults to `3000`.
    pub timeout_millis: Option<u32>,
}

/// Mutable configuration object for `OpenInferenceSubscriber`.
#[napi(object)]
#[derive(Default)]
pub struct OpenInferenceConfig {
    /// `"http_binary"` (default) or `"grpc"`.
    pub transport: Option<String>,
    /// OTLP endpoint, such as `http://localhost:4318/v1/traces`.
    pub endpoint: Option<String>,
    /// Extra exporter headers/metadata as string key/value pairs.
    pub headers: Option<Json>,
    /// Extra OpenInference resource attributes as string key/value pairs.
    pub resource_attributes: Option<Json>,
    /// `service.name` resource attribute. Defaults to `"nemo-relay"`.
    pub service_name: Option<String>,
    /// Optional `service.namespace` resource attribute.
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    pub service_version: Option<String>,
    /// Instrumentation scope name. Defaults to `"nemo-relay-openinference"`.
    pub instrumentation_scope: Option<String>,
    /// Export timeout in milliseconds. Defaults to `3000`.
    pub timeout_millis: Option<u32>,
}

/// OpenTelemetry-backed event subscriber.
#[napi]
pub struct OpenTelemetrySubscriber {
    inner: nemo_relay::observability::otel::OpenTelemetrySubscriber,
}

#[napi]
impl OpenTelemetrySubscriber {
    /// Create a new OpenTelemetry subscriber from a config object.
    #[napi(constructor)]
    pub fn new(config: Option<OpenTelemetryConfig>) -> napi::Result<Self> {
        let inner = nemo_relay::observability::otel::OpenTelemetrySubscriber::new(
            build_otel_config(config)?,
        )
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Register this subscriber globally with the given name.
    #[napi]
    pub fn register(&self, name: String) -> napi::Result<()> {
        self.inner
            .register(&name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Deregister a subscriber by name.
    #[napi]
    pub fn deregister(&self, name: String) -> napi::Result<bool> {
        self.inner
            .deregister(&name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Force a flush of finished spans through the exporter.
    #[napi]
    pub fn force_flush(&self) -> napi::Result<()> {
        self.inner
            .force_flush()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Shut down the underlying tracer provider.
    #[napi]
    pub fn shutdown(&self) -> napi::Result<()> {
        self.inner
            .shutdown()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// OpenInference-backed event subscriber.
#[napi]
pub struct OpenInferenceSubscriber {
    inner: nemo_relay::observability::openinference::OpenInferenceSubscriber,
}

#[napi]
impl OpenInferenceSubscriber {
    /// Create a new OpenInference subscriber from a config object.
    #[napi(constructor)]
    pub fn new(config: Option<OpenInferenceConfig>) -> napi::Result<Self> {
        let inner = nemo_relay::observability::openinference::OpenInferenceSubscriber::new(
            build_openinference_config(config)?,
        )
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Register this subscriber globally with the given name.
    #[napi]
    pub fn register(&self, name: String) -> napi::Result<()> {
        self.inner
            .register(&name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Deregister a subscriber by name.
    #[napi]
    pub fn deregister(&self, name: String) -> napi::Result<bool> {
        self.inner
            .deregister(&name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Force a flush of finished spans through the exporter.
    #[napi]
    pub fn force_flush(&self) -> napi::Result<()> {
        self.inner
            .force_flush()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Shut down the underlying tracer provider.
    #[napi]
    pub fn shutdown(&self) -> napi::Result<()> {
        self.inner
            .shutdown()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CacheTelemetryOptions {
    provider: String,
    #[serde(alias = "request_id")]
    request_id: String,
    usage: Option<Json>,
    #[serde(default, alias = "request_facts")]
    request_facts: Option<Json>,
    #[serde(alias = "agent_id")]
    agent_id: String,
    #[serde(alias = "template_version")]
    template_version: String,
    #[serde(alias = "toolset_hash")]
    toolset_hash: String,
    #[serde(alias = "model_family")]
    model_family: String,
    #[serde(alias = "tenant_scope")]
    tenant_scope: String,
    timestamp: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CacheRequestFactsOptions {
    provider: String,
    #[serde(alias = "request_id")]
    request_id: String,
    #[serde(alias = "annotated_request")]
    annotated_request: Json,
    #[serde(alias = "agent_id")]
    agent_id: String,
    timestamp: Option<String>,
}

enum NodeAdaptiveRuntimeState {
    Pending {
        config: AdaptiveConfig,
        report: nemo_relay::plugin::ConfigReport,
    },
    Ready(CoreAdaptiveRuntime),
}

/// Owned adaptive runtime that can register adaptive features outside the plugin system.
#[napi]
pub struct AdaptiveRuntime {
    inner: Arc<tokio::sync::Mutex<Option<NodeAdaptiveRuntimeState>>>,
}

#[napi]
impl AdaptiveRuntime {
    /// Create an adaptive runtime wrapper from config.
    ///
    /// The runtime is constructed lazily when `register()` is awaited.
    #[napi(constructor)]
    pub fn new(config: Json) -> napi::Result<Self> {
        let config: AdaptiveConfig = serde_json::from_value(config)
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
        let report = validate_adaptive_config_or_err(&config)?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(Some(
                NodeAdaptiveRuntimeState::Pending { config, report },
            ))),
        })
    }

    /// Register all configured adaptive runtime features.
    ///
    /// `register()` and `shutdown()` both temporarily take ownership of the
    /// runtime state, so concurrent calls are mutually exclusive by design.
    /// Once either operation takes the state, another concurrent registration or
    /// shutdown attempt fails with "adaptive runtime already shut down". This
    /// prevents double-registration and shutdown-during-registration races.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn register(&self, env: Env) -> napi::Result<JsObject> {
        let inner = self.inner.clone();
        env.execute_tokio_future(
            async move {
                let state = {
                    let mut guard = inner.lock().await;
                    guard.take().ok_or_else(|| {
                        napi::Error::from_reason("adaptive runtime already shut down")
                    })?
                };

                let (result, next_state) = match state {
                    NodeAdaptiveRuntimeState::Pending { config, report } => {
                        match CoreAdaptiveRuntime::new(config.clone()).await {
                            Ok(mut runtime) => {
                                let result = runtime
                                    .register()
                                    .await
                                    .map_err(|error| napi::Error::from_reason(error.to_string()));
                                (result, Some(NodeAdaptiveRuntimeState::Ready(runtime)))
                            }
                            Err(error) => (
                                Err(napi::Error::from_reason(error.to_string())),
                                Some(NodeAdaptiveRuntimeState::Pending { config, report }),
                            ),
                        }
                    }
                    NodeAdaptiveRuntimeState::Ready(mut runtime) => {
                        let result = runtime
                            .register()
                            .await
                            .map_err(|error| napi::Error::from_reason(error.to_string()));
                        (result, Some(NodeAdaptiveRuntimeState::Ready(runtime)))
                    }
                };

                let mut guard = inner.lock().await;
                *guard = next_state;
                result
            },
            |env, _| env.get_undefined(),
        )
    }

    /// Deregister all previously registered adaptive runtime features.
    #[napi]
    pub fn deregister(&self) -> napi::Result<()> {
        let mut guard = self.inner.try_lock().map_err(|_| {
            napi::Error::from_reason(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("adaptive runtime already shut down"))?;
        match state {
            NodeAdaptiveRuntimeState::Pending { .. } => Ok(()),
            NodeAdaptiveRuntimeState::Ready(runtime) => runtime
                .deregister()
                .map_err(|error| napi::Error::from_reason(error.to_string())),
        }
    }

    /// Shut down the adaptive runtime and consume its Rust runtime state.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn shutdown(&self, env: Env) -> napi::Result<JsObject> {
        let inner = self.inner.clone();
        env.execute_tokio_future(
            async move {
                let state = {
                    let mut guard = inner.lock().await;
                    guard.take().ok_or_else(|| {
                        napi::Error::from_reason("adaptive runtime already shut down")
                    })?
                };
                match state {
                    NodeAdaptiveRuntimeState::Pending { .. } => Ok(()),
                    NodeAdaptiveRuntimeState::Ready(runtime) => runtime
                        .shutdown()
                        .await
                        .map_err(|error| napi::Error::from_reason(error.to_string())),
                }
            },
            |env, _| env.get_undefined(),
        )
    }

    /// Block until the telemetry drain has processed pending events.
    #[napi(js_name = "waitForIdle")]
    pub fn wait_for_idle(&self) -> napi::Result<()> {
        let guard = self.inner.try_lock().map_err(|_| {
            napi::Error::from_reason(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("adaptive runtime already shut down"))?;
        match state {
            NodeAdaptiveRuntimeState::Pending { .. } => Ok(()),
            NodeAdaptiveRuntimeState::Ready(runtime) => {
                runtime.wait_for_idle();
                Ok(())
            }
        }
    }

    /// Return the validation report captured during runtime construction.
    #[napi]
    pub fn report(&self) -> napi::Result<Json> {
        let guard = self.inner.try_lock().map_err(|_| {
            napi::Error::from_reason(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("adaptive runtime already shut down"))?;
        let report = match state {
            NodeAdaptiveRuntimeState::Pending { report, .. } => report,
            NodeAdaptiveRuntimeState::Ready(runtime) => runtime.report(),
        };
        serde_json::to_value(report).map_err(|error| napi::Error::from_reason(error.to_string()))
    }

    /// Bind the runtime's ACG request rewrite to a scope.
    #[napi(js_name = "bindScope")]
    pub fn bind_scope(&self, scope_handle: &ScopeHandle) -> napi::Result<()> {
        let mut guard = self.inner.try_lock().map_err(|_| {
            napi::Error::from_reason(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("adaptive runtime already shut down"))?;
        match state {
            NodeAdaptiveRuntimeState::Pending { .. } => Err(napi::Error::from_reason(
                "adaptive runtime must be registered before binding ACG request intercepts",
            )),
            NodeAdaptiveRuntimeState::Ready(runtime) => runtime
                .bind_scope(scope_handle.inner.uuid)
                .map_err(|error| napi::Error::from_reason(error.to_string())),
        }
    }

    /// Build cache request facts for an annotated LLM request.
    #[napi(js_name = "buildCacheRequestFacts")]
    pub fn build_cache_request_facts(&self, options: Json) -> napi::Result<Option<Json>> {
        let options: CacheRequestFactsOptions = serde_json::from_value(options)
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
        let _request_id = parse_cache_telemetry_request_id(&options.request_id)?;
        let _timestamp = parse_cache_telemetry_timestamp(options.timestamp.as_deref())?;
        let annotated_request: AnnotatedLlmRequest =
            serde_json::from_value(options.annotated_request).map_err(|error| {
                napi::Error::from_reason(format!("invalid annotatedRequest: {error}"))
            })?;
        let guard = self.inner.try_lock().map_err(|_| {
            napi::Error::from_reason(
                "adaptive runtime is locked by an async operation; try again after await completes",
            )
        })?;
        let state = guard
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("adaptive runtime already shut down"))?;
        match state {
            NodeAdaptiveRuntimeState::Pending { .. } => Err(napi::Error::from_reason(
                "adaptive runtime must be registered before building cache request facts",
            )),
            NodeAdaptiveRuntimeState::Ready(runtime) => Ok(runtime
                .build_cache_request_facts(&options.agent_id, &options.provider, &annotated_request)
                .map(|facts| serde_json::to_value(&facts))
                .transpose()
                .map_err(|error| napi::Error::from_reason(error.to_string()))?),
        }
    }
}

fn validate_adaptive_config_or_err(
    config: &AdaptiveConfig,
) -> napi::Result<nemo_relay::plugin::ConfigReport> {
    let report = CoreAdaptiveRuntime::validate_config(config);
    if report.has_errors() {
        let joined = report
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
            .map(|diagnostic| diagnostic.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(napi::Error::from_reason(joined));
    }
    Ok(report)
}

fn parse_cache_telemetry_provider(provider: &str) -> napi::Result<CacheTelemetryProvider> {
    match provider {
        "anthropic" => Ok(CacheTelemetryProvider::Anthropic),
        "openai" => Ok(CacheTelemetryProvider::OpenAI),
        other => Err(napi::Error::from_reason(format!(
            "unsupported provider: {other}",
        ))),
    }
}

fn parse_cache_telemetry_request_id(request_id: &str) -> napi::Result<uuid::Uuid> {
    uuid::Uuid::parse_str(request_id)
        .map_err(|error| napi::Error::from_reason(format!("invalid requestId UUID: {error}")))
}

fn parse_cache_telemetry_timestamp(timestamp: Option<&str>) -> napi::Result<DateTime<Utc>> {
    match timestamp {
        Some(value) => DateTime::parse_from_rfc3339(value)
            .map(|value| value.with_timezone(&Utc))
            .map_err(|error| napi::Error::from_reason(format!("invalid timestamp: {error}"))),
        None => Ok(Utc::now()),
    }
}

fn parse_cache_request_facts(value: Option<Json>) -> napi::Result<Option<CacheRequestFacts>> {
    let Some(value) = value else {
        return Ok(None);
    };
    serde_json::from_value(value)
        .map(Some)
        .map_err(|error| napi::Error::from_reason(format!("invalid requestFacts: {error}")))
}

/// Validate an adaptive config document without constructing a runtime.
#[napi(js_name = "validateAdaptiveConfig")]
pub fn validate_adaptive_config(config: Json) -> napi::Result<Json> {
    let config: AdaptiveConfig = serde_json::from_value(config)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(CoreAdaptiveRuntime::validate_config(&config))
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Build one adaptive cache telemetry event from normalized usage.
#[napi(js_name = "buildCacheTelemetryEvent")]
pub fn build_cache_telemetry_event(options: Json) -> napi::Result<Option<Json>> {
    let options: CacheTelemetryOptions = serde_json::from_value(options)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    let provider = parse_cache_telemetry_provider(&options.provider)?;
    let request_id = parse_cache_telemetry_request_id(&options.request_id)?;
    let timestamp = parse_cache_telemetry_timestamp(options.timestamp.as_deref())?;
    let Some(usage_json) = options.usage else {
        return Ok(None);
    };
    let usage: Usage = serde_json::from_value(usage_json)
        .map_err(|error| napi::Error::from_reason(format!("invalid usage: {error}")))?;
    let request_facts = parse_cache_request_facts(options.request_facts)?;
    let agent_identity = AgentIdentity {
        agent_id: options.agent_id,
        template_version: options.template_version,
        toolset_hash: options.toolset_hash,
        model_family: options.model_family,
        tenant_scope: options.tenant_scope,
    };
    CacheTelemetryEvent::from_usage(
        request_id,
        agent_identity,
        provider,
        &usage,
        timestamp,
        request_facts.as_ref(),
    )
    .map(|event| serde_json::to_value(&event))
    .transpose()
    .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Set manual latency sensitivity on the current scope.
#[napi(js_name = "setLatencySensitivity")]
pub fn set_latency_sensitivity(value: u32) -> napi::Result<()> {
    if value == 0 {
        return Err(napi::Error::from_reason(
            "sensitivity must be positive (> 0)",
        ));
    }
    adaptive_set_latency_sensitivity(value)
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Validate a plugin config document and return a structured diagnostics report.
#[napi]
pub fn validate_plugin_config(config: Json) -> napi::Result<Json> {
    let config: PluginConfig =
        serde_json::from_value(config).map_err(|e| napi::Error::from_reason(e.to_string()))?;
    serde_json::to_value(validate_plugin_config_impl(&config))
        .map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Register a plugin backed by JavaScript callbacks.
///
/// `validate` receives `(pluginConfig)` and should return a diagnostics array.
/// `register` receives `(pluginConfig, context)` and should use the context methods
/// to attach subscribers or intercepts. Both callbacks must be synchronous.
#[napi]
pub fn register_plugin(
    env: Env,
    plugin_kind: String,
    validate: Option<JsFunction>,
    register: JsFunction,
) -> napi::Result<()> {
    let validate_tsfn = match validate {
        Some(func) => Some(PersistentJsFunction::new(&env, &func)?),
        None => None,
    };
    let mut register_tsfn = register
        .create_threadsafe_function::<NodePluginRegisterCall, JsUnknown, _, ErrorStrategy::Fatal>(
            0,
            move |ctx: napi::threadsafe_function::ThreadSafeCallContext<NodePluginRegisterCall>| {
                let plugin_config = unsafe {
                    JsUnknown::from_raw_unchecked(
                        ctx.env.raw(),
                        Json::to_napi_value(ctx.env.raw(), ctx.value.plugin_config)?,
                    )
                };
                let plugin_context = build_plugin_context(
                    &ctx.env,
                    ctx.value.namespace_prefix,
                    ctx.value.registrations,
                )?;
                Ok(vec![
                    plugin_config,
                    js_unknown_from_raw(&ctx.env, &plugin_context),
                ])
            },
        )?;
    register_tsfn.unref(&env)?;

    register_plugin_impl(Arc::new(NodePlugin {
        plugin_kind,
        validate: validate_tsfn,
        register: register_tsfn,
    }))
    .map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Deregister a plugin by kind.
#[napi]
pub fn deregister_plugin(plugin_kind: String) -> bool {
    deregister_plugin_impl(&plugin_kind)
}

/// Initialize the active global plugin components.
#[napi]
pub async fn initialize_plugins(config: Json) -> napi::Result<Json> {
    let config: PluginConfig =
        serde_json::from_value(config).map_err(|e| napi::Error::from_reason(e.to_string()))?;
    let report = initialize_plugins_impl(config)
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    serde_json::to_value(&report).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Clear the active global plugin configuration.
#[napi]
pub fn clear_plugin_configuration() -> napi::Result<()> {
    clear_plugin_configuration_impl().map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Return the last successfully configured plugin report.
#[napi]
pub fn active_plugin_report() -> napi::Result<Option<Json>> {
    active_plugin_report_impl()
        .map(|report| serde_json::to_value(&report))
        .transpose()
        .map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// List registered plugin kinds.
#[napi]
pub fn list_plugin_kinds() -> Vec<String> {
    list_plugin_kinds_impl()
}
