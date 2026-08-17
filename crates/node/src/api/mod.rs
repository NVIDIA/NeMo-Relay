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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::task::{Context, Poll};

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue};
use napi_derive::napi;
use serde::Deserialize;
use serde_json::Value as Json;
use tokio_stream::{Stream, StreamExt};

use nemo_relay::api::llm as core_llm_api;
use nemo_relay::api::llm::{LlmAttributes, LlmRequest};
use nemo_relay::api::registry as core_registry_api;
use nemo_relay::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, capture_nested_publication_buffer, with_nested_publication_buffer,
};
use nemo_relay::api::runtime::{
    EventSanitizeFn, LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, LlmStreamInner,
    ScopeStackHandle as CoreScopeStackHandle, ToolExecutionNextFn,
};
use nemo_relay::api::runtime::{
    TASK_SCOPE_STACK, capture_propagation_context as capture_propagation_context_handle,
    capture_propagation_context_with_root as capture_propagation_context_with_root_handle,
    capture_traceparent as capture_traceparent_handle,
    create_scope_stack as create_scope_stack_handle,
    create_scope_stack_from_propagation as create_scope_stack_from_propagation_handle,
    current_scope_stack as current_scope_stack_handle, scope_stack_active as scope_stack_is_active,
    set_thread_scope_stack as bind_thread_scope_stack, task_scope_top,
    with_scope_stack as with_scope_stack_handle,
};
use nemo_relay::api::scope as core_scope_api;
use nemo_relay::api::scope::ScopeAttributes;
use nemo_relay::api::subscriber as core_subscriber_api;
use nemo_relay::api::tool as core_tool_api;
use nemo_relay::api::tool::ToolAttributes;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::Usage;
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec as CoreDynamicPluginActivationSpec, DynamicPluginKind,
    PluginHostActivation as CorePluginHostActivation,
};
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
use crate::callback_factory;
use crate::convert::{
    callback_json, clear_last_callback_error as clear_recorded_callback_error,
    get_last_callback_error as get_recorded_callback_error, opt_json, parse_timestamp_micros,
    record_callback_error, to_napi_err,
};
use crate::promise_call::PromiseAwareFn;
use crate::promise_call::with_publication_callback_context;
use crate::stream::LlmStream;
use crate::types::{
    LlmHandle, ScopeHandle, ScopeStack, ScopeType, ToolExecutionResult, ToolHandle,
};

static NODE_ENVIRONMENT_COUNT: AtomicUsize = AtomicUsize::new(0);
static NODE_ENVIRONMENT_LIFECYCLE_LOCK: StdMutex<()> = StdMutex::new(());

fn register_node_environment() -> FlowResult<()> {
    let _guard = NODE_ENVIRONMENT_LIFECYCLE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    nemo_relay::logging::initialize_default_logging()?;
    NODE_ENVIRONMENT_COUNT.fetch_add(1, Ordering::AcqRel);
    Ok(())
}

fn cleanup_node_environment() {
    let _guard = NODE_ENVIRONMENT_LIFECYCLE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if NODE_ENVIRONMENT_COUNT.fetch_sub(1, Ordering::AcqRel) == 1
        && let Err(error) = nemo_relay::logging::shutdown_default_logging()
    {
        eprintln!("nemo-relay: operational logging shutdown failed: {error}");
    }
}

fn effective_scope_context(
    env: &Env,
) -> napi::Result<(
    nemo_relay::api::runtime::ScopeStackHandle,
    Option<PublicationBuffer>,
)> {
    Ok(callback_factory::callback_scope_stack(env)?
        .unwrap_or_else(|| (current_scope_stack_handle(), None)))
}

fn effective_scope_stack(env: &Env) -> napi::Result<nemo_relay::api::runtime::ScopeStackHandle> {
    effective_scope_context(env).map(|(scope_stack, _)| scope_stack)
}

fn with_effective_scope_stack<T>(env: &Env, callback: impl FnOnce() -> T) -> napi::Result<T> {
    let (scope_stack, publication_buffer) = effective_scope_context(env)?;
    Ok(with_scope_stack_handle(scope_stack, || {
        with_nested_publication_buffer(publication_buffer, callback)
    }))
}

fn effective_scope_top(
    scope_stack: &nemo_relay::api::runtime::ScopeStackHandle,
) -> nemo_relay::api::scope::ScopeHandle {
    with_scope_stack_handle(scope_stack.clone(), task_scope_top)
}

#[napi::module_init]
fn init() {
    initialize_shared_runtime_binding("node")
        .expect("node runtime ownership initialization should succeed");
    register_adaptive_component()
        .expect("node adaptive plugin component registration should succeed");
    register_pii_redaction_component()
        .expect("node pii redaction plugin component registration should succeed");
}

#[cfg(not(test))]
#[napi_derive::module_exports]
fn install_well_known_symbol_methods(exports: JsObject, mut env: Env) -> napi::Result<()> {
    register_node_environment().map_err(to_napi_err)?;
    if let Err(error) = env.add_env_cleanup_hook((), |_| cleanup_node_environment()) {
        cleanup_node_environment();
        return Err(error);
    }
    let activation: JsFunction = exports.get_named_property("DynamicPluginActivation")?;
    let activation = activation.coerce_to_object()?;
    let mut prototype: JsObject = activation.get_named_property("prototype")?;
    let symbol: JsFunction = env.get_global()?.get_named_property("Symbol")?;
    let symbol = symbol.coerce_to_object()?;
    let async_dispose: napi::JsSymbol = symbol.get_named_property("asyncDispose")?;
    let close: JsFunction = prototype.get_named_property("close")?;
    prototype.set_property(async_dispose, close)?;
    prototype.delete_named_property("[Symbol.asyncDispose]")?;
    Ok(())
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

fn parse_otel_type(value: &str) -> napi::Result<nemo_relay::observability::OpenTelemetryType> {
    match value {
        "full" => Ok(nemo_relay::observability::OpenTelemetryType::Full),
        "gen_ai" => Ok(nemo_relay::observability::OpenTelemetryType::GenAi),
        "openinference" => Ok(nemo_relay::observability::OpenTelemetryType::OpenInference),
        other => Err(napi::Error::from_reason(format!(
            "type must be 'full', 'gen_ai', or 'openinference', got {other:?}",
        ))),
    }
}

fn parse_otel_transport(
    value: Option<String>,
) -> napi::Result<nemo_relay::observability::otel::OtlpTransport> {
    match value.as_deref().unwrap_or("http_binary") {
        "http_binary" => Ok(nemo_relay::observability::otel::OtlpTransport::HttpBinary),
        "grpc" => Ok(nemo_relay::observability::otel::OtlpTransport::Grpc),
        other => Err(napi::Error::from_reason(format!(
            "transport must be 'http_binary' or 'grpc', got {other:?}",
        ))),
    }
}

fn parse_mark_projection(
    value: Option<String>,
) -> napi::Result<nemo_relay::observability::MarkProjection> {
    serde_json::from_value(Json::String(value.unwrap_or_else(|| "inherit".to_string())))
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

fn parse_attribute_mappings(
    value: Option<Json>,
) -> napi::Result<Vec<nemo_relay::observability::OtlpAttributeMapping>> {
    let mappings = match value {
        Some(value) => serde_json::from_value(value)
            .map_err(|error| napi::Error::from_reason(format!("attributeMappings: {error}")))?,
        None => Vec::new(),
    };
    nemo_relay::observability::validate_attribute_mappings(&mappings)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    Ok(mappings)
}

fn build_otel_config(
    options: OpenTelemetryConfig,
) -> napi::Result<nemo_relay::observability::otel::OpenTelemetryConfig> {
    let otel_type = parse_otel_type(&options.r#type)?;
    let endpoint = options.endpoint.trim().to_string();
    if endpoint.is_empty() {
        return Err(napi::Error::from_reason(
            "endpoint must be a nonblank string",
        ));
    }
    let transport = parse_otel_transport(options.transport)?;
    let service_name = options
        .service_name
        .unwrap_or_else(|| "unknown_service".to_string());
    let instrumentation_scope = options
        .instrumentation_scope
        .unwrap_or_else(|| "opentelemetry".to_string());
    let timeout_millis = options.timeout_millis.unwrap_or(3_000);

    let mut config = nemo_relay::observability::otel::OpenTelemetryConfig::new(otel_type, endpoint)
        .with_transport(transport)
        .with_service_name(service_name)
        .with_instrumentation_scope(instrumentation_scope)
        .with_timeout(std::time::Duration::from_millis(timeout_millis.into()));

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
    config = config
        .with_mark_projection(parse_mark_projection(options.mark_projection)?)
        .with_mark_exclude_names(
            options
                .mark_exclude_names
                .unwrap_or_else(nemo_relay::observability::default_mark_exclude_names),
        )
        .with_attribute_mappings(parse_attribute_mappings(options.attribute_mappings)?);
    Ok(config)
}

fn build_atof_config(
    options: Option<AtofExporterConfig>,
) -> napi::Result<nemo_relay::observability::atof::AtofExporterConfig> {
    let options = options.unwrap_or_default();
    match options.r#type.as_deref().unwrap_or("file") {
        "file" => {
            let mut config = nemo_relay::observability::atof::AtofExporterConfig::new();
            if let Some(output_directory) = options.output_directory {
                config = config.with_output_directory(PathBuf::from(output_directory));
            }
            if let Some(filename) = options.filename {
                config = config.with_filename(filename);
            }
            if let Some(mode) = options.mode {
                let Some(mode) = nemo_relay::observability::atof::AtofExporterMode::parse(&mode)
                else {
                    return Err(napi::Error::from_reason(
                        "mode must be 'append' or 'overwrite'",
                    ));
                };
                config = config.with_mode(mode);
            }
            Ok(config)
        }
        "stream" => {
            let url = options
                .url
                .ok_or_else(|| napi::Error::from_reason("stream sink requires url"))?;
            let transport = options.transport.unwrap_or_else(|| "http_post".to_string());
            let Some(transport) =
                nemo_relay::observability::atof::AtofEndpointTransport::parse(&transport)
            else {
                return Err(napi::Error::from_reason(
                    "stream transport must be 'http_post', 'websocket', or 'ndjson'",
                ));
            };
            let mut sink =
                nemo_relay::observability::atof::AtofStreamSinkConfig::new(url, transport);
            if let Some(timeout_millis) = options.timeout_millis {
                sink = sink.with_timeout_millis(timeout_millis.into());
            }
            if let Some(field_name_policy) = options.field_name_policy {
                let Some(field_name_policy) =
                    nemo_relay::observability::atof::AtofEndpointFieldNamePolicy::parse(
                        &field_name_policy,
                    )
                else {
                    return Err(napi::Error::from_reason(
                        "stream field_name_policy must be 'preserve' or 'replace_dots'",
                    ));
                };
                sink = sink.with_field_name_policy(field_name_policy);
            }
            for (key, value) in parse_string_map(options.headers, "headers")? {
                sink = sink.with_header(key, value);
            }
            for (key, variable) in parse_string_map(options.header_env, "headerEnv")? {
                sink = sink.with_header_env(key, variable);
            }
            Ok(nemo_relay::observability::atof::AtofExporterConfig::new().with_stream_sink(sink))
        }
        _ => Err(napi::Error::from_reason(
            "ATOF sink type must be 'file' or 'stream'",
        )),
    }
}

// ---------------------------------------------------------------------------
// Stream channel registry — enables JS async generators to push chunks to Rust
// ---------------------------------------------------------------------------

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(0);

type StreamSender = tokio::sync::mpsc::UnboundedSender<FlowResult<Json>>;
type RustJsonStream = LlmJsonStream;

struct StreamChannel {
    sender: StreamSender,
    cancelled: AtomicBool,
    closed: tokio::sync::watch::Sender<Option<std::result::Result<(), String>>>,
}

static STREAM_CHANNELS: std::sync::LazyLock<StdMutex<HashMap<u64, Arc<StreamChannel>>>> =
    std::sync::LazyLock::new(|| StdMutex::new(HashMap::new()));

fn register_stream_channel(
    id: u64,
    tx: StreamSender,
) -> tokio::sync::watch::Receiver<Option<std::result::Result<(), String>>> {
    let (closed, closed_rx) = tokio::sync::watch::channel(None);
    STREAM_CHANNELS.lock().unwrap().insert(
        id,
        Arc::new(StreamChannel {
            sender: tx,
            cancelled: AtomicBool::new(false),
            closed,
        }),
    );
    closed_rx
}

fn finish_stream_channel(id: u64, result: std::result::Result<(), String>) {
    if let Some(channel) = STREAM_CHANNELS.lock().unwrap().remove(&id) {
        channel.closed.send_replace(Some(result));
    }
}

fn cancel_stream_channel(id: u64) {
    if let Some(channel) = STREAM_CHANNELS.lock().unwrap().get(&id) {
        channel.cancelled.store(true, Ordering::Release);
    }
}

fn ensure_stream_callback_queued(id: u64, status: napi::Status) -> FlowResult<()> {
    if status == napi::Status::Ok {
        return Ok(());
    }

    finish_stream_channel(
        id,
        Err(format!(
            "failed to queue JS stream producer callback: {status:?}"
        )),
    );
    Err(FlowError::Internal(format!(
        "failed to queue JS stream producer callback: {status:?}",
    )))
}

async fn forward_stream_to_channel(
    mut stream: RustJsonStream,
    tx: tokio::sync::mpsc::Sender<FlowResult<Json>>,
    mut cancel: tokio::sync::watch::Receiver<bool>,
    closed: tokio::sync::watch::Sender<Option<std::result::Result<(), String>>>,
) {
    loop {
        if *cancel.borrow() {
            break;
        }
        let item = tokio::select! {
            _ = cancel.changed() => break,
            item = stream.next() => item,
        };
        let Some(item) = item else {
            break;
        };
        tokio::select! {
            _ = cancel.changed() => break,
            result = tx.send(item) => {
                if result.is_err() {
                    break;
                }
            }
        }
    }
    closed.send_replace(Some(
        stream.close().await.map_err(|error| error.to_string()),
    ));
}

struct NodePushStream {
    receiver: tokio_stream::wrappers::UnboundedReceiverStream<FlowResult<Json>>,
    stream_id: u64,
    closed: tokio::sync::watch::Receiver<Option<std::result::Result<(), String>>>,
}

impl Stream for NodePushStream {
    type Item = FlowResult<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_next(cx)
    }
}

impl Drop for NodePushStream {
    fn drop(&mut self) {
        cancel_stream_channel(self.stream_id);
    }
}

impl LlmStreamInner for NodePushStream {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = FlowResult<()>> + Send + '_>> {
        let stream_id = self.stream_id;
        let mut closed = self.get_mut().closed.clone();
        Box::pin(async move {
            cancel_stream_channel(stream_id);
            while closed.borrow().is_none() {
                closed.changed().await.map_err(|_| {
                    FlowError::Internal("JS stream cleanup task ended early".into())
                })?;
            }
            closed
                .borrow()
                .clone()
                .expect("close state checked above")
                .map_err(FlowError::Internal)
        })
    }
}

/// Push a chunk into the stream identified by `streamId`.
/// Called from JavaScript during async generator iteration.
#[napi]
pub fn push_stream_chunk(stream_id: f64, chunk: Json) -> bool {
    let id = stream_id as u64;
    if let Some(channel) = STREAM_CHANNELS.lock().unwrap().get(&id) {
        !channel.cancelled.load(Ordering::Acquire) && channel.sender.send(Ok(chunk)).is_ok()
    } else {
        false
    }
}

/// Signal that a stream is complete. Drops the sender so the Rust
/// receiver sees the channel as closed.
#[napi]
pub fn end_stream(env: Env, stream_id: f64) -> napi::Result<()> {
    let id = stream_id as u64;
    finish_stream_channel(id, Ok(()));
    callback_factory::expire_callback_context(&env)
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

struct ScopedStreamCall {
    request: Json,
    scope_stack: CoreScopeStackHandle,
    publication_buffer: Option<PublicationBuffer>,
    propagation_parent_uuid: String,
}

fn scoped_stream_callback_tsfn(
    env: &Env,
    func: &JsFunction,
) -> napi::Result<ThreadsafeFunction<ScopedStreamCall, ErrorStrategy::Fatal>> {
    let callback = callback_factory::wrap_scoped_stream_callback(env, func)?;
    let mut tsfn = callback.create_threadsafe_function(
        0,
        |ctx: napi::threadsafe_function::ThreadSafeCallContext<ScopedStreamCall>| {
            let request = unsafe {
                JsUnknown::from_raw_unchecked(
                    ctx.env.raw(),
                    Json::to_napi_value(ctx.env.raw(), ctx.value.request)?,
                )
            };
            let scope_stack = ScopeStack {
                inner: ctx.value.scope_stack,
                publication_buffer: ctx.value.publication_buffer,
            }
            .into_instance(ctx.env)?;
            Ok(vec![
                request,
                unsafe { JsUnknown::from_raw_unchecked(ctx.env.raw(), scope_stack.raw()) },
                ctx.env
                    .create_string(&ctx.value.propagation_parent_uuid)?
                    .into_unknown(),
            ])
        },
    )?;
    tsfn.unref(env)?;
    Ok(tsfn)
}

fn middleware_tool_callback_tsfn(
    env: &Env,
    func: &JsFunction,
) -> napi::Result<ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>> {
    let callback = callable::safe_middleware_callback(env, func)?;
    let mut tsfn = callback.create_threadsafe_function(
        0,
        |ctx: napi::threadsafe_function::ThreadSafeCallContext<(String, Json)>| {
            let name = ctx.env.create_string_from_std(ctx.value.0)?;
            let args = unsafe {
                JsUnknown::from_raw_unchecked(
                    ctx.env.raw(),
                    Json::to_napi_value(ctx.env.raw(), ctx.value.1)?,
                )
            };
            Ok(vec![js_unknown_from_raw(&ctx.env, &name), args])
        },
    )?;
    tsfn.unref(env)?;
    Ok(tsfn)
}

fn middleware_json_callback_tsfn(
    env: &Env,
    func: &JsFunction,
) -> napi::Result<ThreadsafeFunction<Json, ErrorStrategy::Fatal>> {
    let callback = callable::safe_middleware_callback(env, func)?;
    let mut tsfn = callback.create_threadsafe_function(
        0,
        |ctx: napi::threadsafe_function::ThreadSafeCallContext<Json>| Ok(vec![ctx.value]),
    )?;
    tsfn.unref(env)?;
    Ok(tsfn)
}

fn middleware_llm_sanitize_request_callback_tsfn(
    env: &Env,
    func: &JsFunction,
) -> napi::Result<
    ThreadsafeFunction<(Json, callable::JsLlmSanitizeRequestContext), ErrorStrategy::Fatal>,
> {
    let callback = callable::safe_middleware_callback(env, func)?;
    let mut tsfn = callback.create_threadsafe_function(
        0,
        |ctx: napi::threadsafe_function::ThreadSafeCallContext<(
            Json,
            callable::JsLlmSanitizeRequestContext,
        )>| {
            let first = unsafe {
                JsUnknown::from_raw_unchecked(
                    ctx.env.raw(),
                    Json::to_napi_value(ctx.env.raw(), ctx.value.0)?,
                )
            };
            let context = callable::js_llm_sanitize_request_context_to_napi(&ctx.env, ctx.value.1)?;
            Ok(vec![first, context])
        },
    )?;
    tsfn.unref(env)?;
    Ok(tsfn)
}

fn middleware_llm_sanitize_response_callback_tsfn(
    env: &Env,
    func: &JsFunction,
) -> napi::Result<
    ThreadsafeFunction<(Json, callable::JsLlmSanitizeResponseContext), ErrorStrategy::Fatal>,
> {
    let callback = callable::safe_middleware_callback(env, func)?;
    let mut tsfn = callback.create_threadsafe_function(
        0,
        |ctx: napi::threadsafe_function::ThreadSafeCallContext<(
            Json,
            callable::JsLlmSanitizeResponseContext,
        )>| {
            let first = unsafe {
                JsUnknown::from_raw_unchecked(
                    ctx.env.raw(),
                    Json::to_napi_value(ctx.env.raw(), ctx.value.0)?,
                )
            };
            let context =
                callable::js_llm_sanitize_response_context_to_napi(&ctx.env, ctx.value.1)?;
            Ok(vec![first, context])
        },
    )?;
    tsfn.unref(env)?;
    Ok(tsfn)
}

#[allow(clippy::too_many_arguments)]
fn add_plugin_event_sanitizer(
    env: &Env,
    context: &mut JsObject,
    property: &str,
    namespace_prefix: String,
    registrations: Arc<StdMutex<Vec<PluginRegistration>>>,
    register: fn(&str, i32, EventSanitizeFn) -> FlowResult<()>,
    deregister: fn(&str) -> FlowResult<bool>,
    label: &'static str,
) -> napi::Result<()> {
    let function = env.create_function_from_closure(property, move |ctx| {
        let name = format!("{}{}", namespace_prefix, ctx.get::<String>(0)?);
        let priority = ctx.get::<i32>(1)?;
        let callback = ctx.get::<JsFunction>(2)?;
        register(&name, priority, node_event_sanitize_fn(ctx.env, &callback)?)
            .map_err(to_napi_err)?;
        let name_clone = name.clone();
        registrations.lock().unwrap().push(PluginRegistration::new(
            "plugin",
            name_clone.clone(),
            Box::new(move || {
                deregister(&name_clone).map(|_| ()).map_err(|error| {
                    PluginError::RegistrationFailed(format!(
                        "{label} deregistration failed: {error}"
                    ))
                })
            }),
        ));
        ctx.env.get_undefined()
    })?;
    context.set_named_property(property, function)
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
            core_subscriber_api::register_subscriber(
                &name,
                callable::wrap_js_event_subscriber(ctx.env, name.clone(), callback)?,
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

    add_plugin_event_sanitizer(
        env,
        &mut context,
        "registerMarkSanitizeGuardrail",
        namespace_prefix.clone(),
        registrations.clone(),
        core_registry_api::register_mark_sanitize_guardrail,
        core_registry_api::deregister_mark_sanitize_guardrail,
        "mark sanitize guardrail",
    )?;
    add_plugin_event_sanitizer(
        env,
        &mut context,
        "registerScopeSanitizeStartGuardrail",
        namespace_prefix.clone(),
        registrations.clone(),
        core_registry_api::register_scope_sanitize_start_guardrail,
        core_registry_api::deregister_scope_sanitize_start_guardrail,
        "scope start sanitize guardrail",
    )?;
    add_plugin_event_sanitizer(
        env,
        &mut context,
        "registerScopeSanitizeEndGuardrail",
        namespace_prefix.clone(),
        registrations.clone(),
        core_registry_api::register_scope_sanitize_end_guardrail,
        core_registry_api::deregister_scope_sanitize_end_guardrail,
        "scope end sanitize guardrail",
    )?;

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
            let callback = ctx.get::<JsFunction>(2)?;
            core_registry_api::register_tool_sanitize_request_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_sanitize_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
            let callback = ctx.get::<JsFunction>(2)?;
            core_registry_api::register_tool_sanitize_response_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_sanitize_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
            let callback = ctx.get::<JsFunction>(2)?;
            core_registry_api::register_tool_conditional_execution_guardrail(
                &name,
                priority,
                callable::wrap_js_tool_conditional_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
            let callback = ctx.get::<JsFunction>(2)?;
            core_registry_api::register_llm_sanitize_request_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_sanitize_request_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
            let callback = ctx.get::<JsFunction>(2)?;
            core_registry_api::register_llm_sanitize_response_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_sanitize_response_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
            let callback = ctx.get::<JsFunction>(2)?;
            core_registry_api::register_llm_conditional_execution_guardrail(
                &name,
                priority,
                callable::wrap_js_llm_conditional_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
            core_registry_api::register_llm_request_intercept(
                &name,
                priority,
                break_chain,
                callable::wrap_js_llm_request_intercept_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
            core_registry_api::register_llm_execution_intercept(
                &name,
                priority,
                callable::wrap_js_llm_exec_intercept_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            llm_exec_regs.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    core_registry_api::deregister_llm_execution_intercept(&name_clone)
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "llm execution intercept deregistration failed: {e}"
                            ))
                        })
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
            core_registry_api::register_llm_stream_execution_intercept(
                &name,
                priority,
                callable::wrap_js_llm_stream_exec_intercept_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
                        core_registry_api::deregister_llm_stream_execution_intercept(&name_clone)
                            .map(|_| ())
                            .map_err(|e| {
                                PluginError::RegistrationFailed(format!(
                                    "llm stream execution intercept deregistration failed: {e}"
                                ))
                            })
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
            let callback = ctx.get::<JsFunction>(3)?;
            core_registry_api::register_tool_request_intercept(
                &name,
                priority,
                break_chain,
                callable::wrap_js_tool_request_intercept_promise_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
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
            core_registry_api::register_tool_execution_intercept(
                &name,
                priority,
                callable::wrap_js_tool_exec_intercept_fn(Arc::new(PromiseAwareFn::new(
                    ctx.env, &callback,
                )?)),
            )
            .map_err(to_napi_err)?;

            let name_clone = name.clone();
            tool_regs.lock().unwrap().push(PluginRegistration::new(
                "plugin",
                name_clone.clone(),
                Box::new(move || {
                    core_registry_api::deregister_tool_execution_intercept(&name_clone)
                        .map(|_| ())
                        .map_err(|e| {
                            PluginError::RegistrationFailed(format!(
                                "tool execution intercept deregistration failed: {e}"
                            ))
                        })
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
pub(crate) struct PersistentJsFunction {
    env: napi::sys::napi_env,
    reference: napi::sys::napi_ref,
    cleanup: napi::sys::napi_threadsafe_function,
}

// SAFETY: Direct function access is restricted by callers to the registration
// thread. Releasing the cleanup TSFN is thread-safe, and its event-loop
// finalizer deletes the N-API reference.
unsafe impl Send for PersistentJsFunction {}
// SAFETY: The same invariants as `Send` apply. The stored handles are immutable,
// and reference deletion is serialized by the cleanup TSFN finalizer.
unsafe impl Sync for PersistentJsFunction {}

unsafe extern "C" fn delete_persistent_js_function_reference(
    env: napi::sys::napi_env,
    finalize_data: *mut std::ffi::c_void,
    _finalize_hint: *mut std::ffi::c_void,
) {
    if !env.is_null() && !finalize_data.is_null() {
        // SAFETY: `finalize_data` is the live N-API reference passed to the
        // cleanup TSFN at creation. This finalizer runs once on Node's event
        // loop after the final TSFN release.
        let _ =
            unsafe { napi::sys::napi_delete_reference(env, finalize_data as napi::sys::napi_ref) };
    }
}

unsafe extern "C" fn persistent_js_function_cleanup_call(
    _env: napi::sys::napi_env,
    _js_callback: napi::sys::napi_value,
    _context: *mut std::ffi::c_void,
    _data: *mut std::ffi::c_void,
) {
}

impl PersistentJsFunction {
    fn new(env: &Env, func: &JsFunction) -> napi::Result<Self> {
        let mut reference = ptr::null_mut();
        // SAFETY: `env.raw()` and `func.raw()` are live N-API handles provided
        // by napi-rs for the current environment. `reference` points to valid
        // writable storage for the created reference.
        let status =
            unsafe { napi::sys::napi_create_reference(env.raw(), func.raw(), 1, &mut reference) };
        if status != napi::sys::Status::napi_ok {
            return Err(napi::Error::from_reason(format!(
                "failed to create JS function reference: {:?}",
                napi::Status::from(status)
            )));
        }

        let mut resource_name = ptr::null_mut();
        let resource_name_bytes = b"nemo_relay_persistent_js_function\0";
        let status = unsafe {
            napi::sys::napi_create_string_utf8(
                env.raw(),
                resource_name_bytes.as_ptr().cast(),
                resource_name_bytes.len() - 1,
                &mut resource_name,
            )
        };
        if status != napi::sys::Status::napi_ok {
            let _ = unsafe { napi::sys::napi_delete_reference(env.raw(), reference) };
            return Err(napi::Error::from_reason(format!(
                "failed to create persistent JS function cleanup resource name: {:?}",
                napi::Status::from(status)
            )));
        }

        let mut cleanup = ptr::null_mut();
        let status = unsafe {
            napi::sys::napi_create_threadsafe_function(
                env.raw(),
                ptr::null_mut(),
                ptr::null_mut(),
                resource_name,
                0,
                1,
                reference.cast(),
                Some(delete_persistent_js_function_reference),
                ptr::null_mut(),
                Some(persistent_js_function_cleanup_call),
                &mut cleanup,
            )
        };
        if status != napi::sys::Status::napi_ok {
            let _ = unsafe { napi::sys::napi_delete_reference(env.raw(), reference) };
            return Err(napi::Error::from_reason(format!(
                "failed to create persistent JS function cleanup handle: {:?}",
                napi::Status::from(status)
            )));
        }

        let status = unsafe { napi::sys::napi_unref_threadsafe_function(env.raw(), cleanup) };
        if status != napi::sys::Status::napi_ok {
            let _ = unsafe {
                napi::sys::napi_release_threadsafe_function(
                    cleanup,
                    napi::sys::ThreadsafeFunctionReleaseMode::release,
                )
            };
            return Err(napi::Error::from_reason(format!(
                "failed to unref persistent JS function cleanup handle: {:?}",
                napi::Status::from(status)
            )));
        }

        Ok(Self {
            env: env.raw(),
            reference,
            cleanup,
        })
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

    fn call_json(&self, argument: Json) -> napi::Result<Json> {
        let mut value = ptr::null_mut();
        // SAFETY: `self.reference` is a live N-API reference created in
        // `self.env`, and `value` is writable storage for the borrowed
        // function value.
        let status =
            unsafe { napi::sys::napi_get_reference_value(self.env, self.reference, &mut value) };
        if status != napi::sys::Status::napi_ok {
            return Err(napi::Error::from_reason("failed to borrow codec function"));
        }
        // SAFETY: `value` was resolved from this struct's function reference,
        // so it is a live function value in `self.env` for this call.
        let func = unsafe { JsFunction::from_raw_unchecked(self.env, value) };
        // SAFETY: `Json::to_napi_value` created this argument in `self.env`,
        // so wrapping it as `JsUnknown` is valid for the immediate callback.
        let argument = unsafe {
            JsUnknown::from_raw_unchecked(self.env, Json::to_napi_value(self.env, argument)?)
        };
        let returned = func.call(None, &[argument])?;
        // SAFETY: `returned` is the live result of invoking `func` in this environment.
        unsafe { Option::<Json>::from_napi_value(self.env, returned.raw()) }.map(callback_json)
    }
}

fn node_event_sanitize_fn(env: &Env, func: &JsFunction) -> napi::Result<EventSanitizeFn> {
    // The registry and queued snapshots own the only callback references.
    // PromiseAwareFn releases its TSFN on the last drop, so deregistration
    // preserves already-snapshotted publication while still cleaning up
    // deterministically once that work finishes.
    let callback = Arc::new(crate::promise_call::PromiseAwareFn::new(env, func)?);
    Ok(callable::wrap_js_event_sanitize_promise_fn(callback))
}

type NodeLlmCodec = (
    Arc<dyn nemo_relay::codec::traits::LlmCodec>,
    Vec<Arc<PersistentJsFunction>>,
);
type NodeLlmResponseCodec = (
    Arc<dyn nemo_relay::codec::traits::LlmResponseCodec>,
    Vec<Arc<PersistentJsFunction>>,
);

fn node_llm_codec(
    env: &Env,
    decode: &JsFunction,
    encode: &JsFunction,
) -> napi::Result<NodeLlmCodec> {
    let direct_decode = Arc::new(PersistentJsFunction::new(env, decode)?);
    let direct_encode = Arc::new(PersistentJsFunction::new(env, encode)?);
    let references = vec![direct_decode.clone(), direct_encode.clone()];
    let register_thread = std::thread::current().id();

    let mut decode_tsfn = decode.create_threadsafe_function(
        0,
        |ctx: napi::threadsafe_function::ThreadSafeCallContext<Json>| Ok(vec![ctx.value]),
    )?;
    decode_tsfn.unref(env)?;
    let mut encode_tsfn = encode.create_threadsafe_function(
        0,
        |ctx: napi::threadsafe_function::ThreadSafeCallContext<Json>| Ok(vec![ctx.value]),
    )?;
    encode_tsfn.unref(env)?;

    Ok((
        callable::wrap_js_codec(
            decode_tsfn,
            encode_tsfn,
            register_thread,
            Arc::new(move |argument| {
                direct_decode.call_json(argument).map_err(|error| {
                    FlowError::Internal(format!("JS codec decode callback failed: {error}"))
                })
            }),
            Arc::new(move |argument| {
                direct_encode.call_json(argument).map_err(|error| {
                    FlowError::Internal(format!("JS codec encode callback failed: {error}"))
                })
            }),
        ),
        references,
    ))
}

fn node_llm_response_codec(env: &Env, decode: &JsFunction) -> napi::Result<NodeLlmResponseCodec> {
    let direct_decode = Arc::new(PersistentJsFunction::new(env, decode)?);
    let references = vec![direct_decode.clone()];
    let register_thread = std::thread::current().id();
    let mut decode_tsfn = decode.create_threadsafe_function(
        0,
        |ctx: napi::threadsafe_function::ThreadSafeCallContext<Json>| Ok(vec![ctx.value]),
    )?;
    decode_tsfn.unref(env)?;
    Ok((
        callable::wrap_js_response_codec(
            decode_tsfn,
            register_thread,
            Arc::new(move |argument| {
                direct_decode.call_json(argument).map_err(|error| {
                    FlowError::Internal(format!(
                        "JS response codec decode callback failed: {error}"
                    ))
                })
            }),
        ),
        references,
    ))
}

impl Drop for PersistentJsFunction {
    fn drop(&mut self) {
        // SAFETY: N-API permits releasing a TSFN from any thread. Its finalizer
        // runs on the event loop and deletes `self.reference` exactly once.
        let _ = unsafe {
            napi::sys::napi_release_threadsafe_function(
                self.cleanup,
                napi::sys::ThreadsafeFunctionReleaseMode::release,
            )
        };
    }
}

struct NodePluginValidateCallback {
    direct: PersistentJsFunction,
    thread_safe: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    registration_thread: std::thread::ThreadId,
}

impl NodePluginValidateCallback {
    fn call(&self, plugin_config: Json) -> napi::Result<Json> {
        if std::thread::current().id() == self.registration_thread {
            return self.direct.call_validate(&plugin_config);
        }

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let status = self.thread_safe.call_with_return_value(
            plugin_config,
            ThreadsafeFunctionCallMode::Blocking,
            move |value: Option<Json>| {
                let result = callable::unwrap_middleware_result(
                    callback_json(value),
                    "JS plugin validate callback failed",
                )
                .map_err(|error| napi::Error::from_reason(error.to_string()));
                let _ = tx.send(result);
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(napi::Error::from_reason(format!(
                "failed to queue JS plugin validate callback: {status:?}"
            )));
        }
        rx.recv()
            .map_err(|_| napi::Error::from_reason("JS plugin validate completion channel closed"))?
    }
}

struct NodePlugin {
    plugin_kind: String,
    validate: Option<NodePluginValidateCallback>,
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
        match validate.call(Json::Object(plugin_config.clone())) {
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

/// Transport-neutral Relay causal context for application-managed transport.
#[napi(object)]
pub struct PropagationContext {
    pub version: u32,
    pub root_uuid: Option<String>,
    pub parent_uuid: String,
}

fn propagation_context_from_napi(
    context: PropagationContext,
) -> napi::Result<nemo_relay::api::runtime::PropagationContext> {
    let root_uuid = context
        .root_uuid
        .as_deref()
        .map(uuid::Uuid::parse_str)
        .transpose()
        .map_err(|error| napi::Error::from_reason(format!("invalid root UUID: {error}")))?;
    let parent_uuid = uuid::Uuid::parse_str(&context.parent_uuid)
        .map_err(|error| napi::Error::from_reason(format!("invalid parent UUID: {error}")))?;
    let version = u16::try_from(context.version)
        .map_err(|_| napi::Error::from_reason("propagation context version is out of range"))?;
    let context = nemo_relay::api::runtime::PropagationContext {
        version,
        root_uuid,
        parent_uuid,
    };
    context
        .validate()
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    Ok(context)
}

fn propagation_context_to_napi(
    context: nemo_relay::api::runtime::PropagationContext,
) -> PropagationContext {
    PropagationContext {
        version: u32::from(context.version),
        root_uuid: context.root_uuid.map(|uuid| uuid.to_string()),
        parent_uuid: context.parent_uuid.to_string(),
    }
}

/// Creates a new isolated scope stack.
#[napi]
pub fn create_scope_stack() -> ScopeStack {
    ScopeStack {
        inner: create_scope_stack_handle(),
        publication_buffer: None,
    }
}

/// Capture the current Relay causal parent for application-managed transport.
#[napi]
pub fn capture_propagation_context(env: Env) -> napi::Result<PropagationContext> {
    if let Some(parent_uuid) = callback_factory::callback_propagation_parent_uuid(&env)? {
        let parent_uuid = uuid::Uuid::parse_str(&parent_uuid)
            .map_err(|error| napi::Error::from_reason(format!("invalid parent UUID: {error}")))?;
        return Ok(propagation_context_to_napi(
            nemo_relay::api::runtime::PropagationContext {
                version: nemo_relay::api::runtime::PropagationContext::VERSION,
                root_uuid: None,
                parent_uuid,
            },
        ));
    }
    with_effective_scope_stack(&env, capture_propagation_context_handle)?
        .map(propagation_context_to_napi)
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Capture the current parent with an optional stable application session root.
#[napi]
pub fn capture_propagation_context_with_root(
    env: Env,
    root_uuid: Option<String>,
) -> napi::Result<PropagationContext> {
    let root_uuid = root_uuid
        .as_deref()
        .map(uuid::Uuid::parse_str)
        .transpose()
        .map_err(|error| napi::Error::from_reason(format!("invalid root UUID: {error}")))?;
    if let Some(parent_uuid) = callback_factory::callback_propagation_parent_uuid(&env)? {
        let parent_uuid = uuid::Uuid::parse_str(&parent_uuid)
            .map_err(|error| napi::Error::from_reason(format!("invalid parent UUID: {error}")))?;
        return Ok(propagation_context_to_napi(
            nemo_relay::api::runtime::PropagationContext {
                version: nemo_relay::api::runtime::PropagationContext::VERSION,
                root_uuid,
                parent_uuid,
            },
        ));
    }
    with_effective_scope_stack(&env, || {
        capture_propagation_context_with_root_handle(root_uuid)
    })?
    .map(propagation_context_to_napi)
    .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Capture the current Relay context as a W3C `traceparent` value.
#[napi]
pub fn capture_traceparent(env: Env) -> napi::Result<String> {
    if let Some(parent_uuid) = callback_factory::callback_propagation_parent_uuid(&env)? {
        let parent_uuid = uuid::Uuid::parse_str(&parent_uuid)
            .map_err(|error| napi::Error::from_reason(format!("invalid parent UUID: {error}")))?;
        let root_uuid = with_effective_scope_stack(&env, capture_traceparent_handle)
            .ok()
            .and_then(|result| result.ok())
            .and_then(|traceparent| {
                traceparent
                    .get(3..35)
                    .and_then(|value| uuid::Uuid::parse_str(value).ok())
            })
            .unwrap_or(parent_uuid);
        return nemo_relay::api::runtime::PropagationContext {
            version: nemo_relay::api::runtime::PropagationContext::VERSION,
            root_uuid: Some(root_uuid),
            parent_uuid,
        }
        .to_traceparent()
        .map_err(|error| napi::Error::from_reason(error.to_string()));
    }
    with_effective_scope_stack(&env, capture_traceparent_handle)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Serialize a Relay causal context to the JSON wire format.
#[napi]
pub fn propagation_context_to_json(context: PropagationContext) -> napi::Result<String> {
    propagation_context_from_napi(context)?
        .to_json()
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Convert a rooted Relay propagation context to a W3C `traceparent` value.
#[napi]
pub fn propagation_context_to_traceparent(context: PropagationContext) -> napi::Result<String> {
    propagation_context_from_napi(context)?
        .to_traceparent()
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Deserialize and validate a Relay causal context from the JSON wire format.
#[napi]
pub fn propagation_context_from_json(value: String) -> napi::Result<PropagationContext> {
    nemo_relay::api::runtime::PropagationContext::from_json(&value)
        .map(propagation_context_to_napi)
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Create an isolated scope stack seeded from a received propagation context.
#[napi]
pub fn create_scope_stack_from_propagation(
    context: PropagationContext,
) -> napi::Result<ScopeStack> {
    create_scope_stack_from_propagation_handle(&propagation_context_from_napi(context)?)
        .map(ScopeStack::from)
        .map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Run a callback with an isolated scope stack installed.
///
/// The caller's stack is restored immediately after callback invocation. When
/// the callback returns a Promise, the requested stack remains active for that
/// Promise until it settles and is then expired for inherited detached work.
/// Use this helper instead of `setThreadScopeStack` to isolate concurrent async
/// branches.
#[napi]
pub fn with_scope_stack(
    env: Env,
    stack: &ScopeStack,
    callback: JsFunction,
) -> napi::Result<JsUnknown> {
    if let Some(value) = callback_factory::with_callback_scope_stack(&env, stack, &callback)? {
        return Ok(value);
    }
    with_scope_stack_handle(stack.inner.clone(), || {
        with_nested_publication_buffer(stack.publication_buffer.clone(), || {
            callback.call::<JsUnknown>(None, &[])
        })
    })
}

/// Returns the current execution context's scope stack handle.
#[napi]
pub fn current_scope_stack(env: Env) -> napi::Result<ScopeStack> {
    if let Some((inner, publication_buffer)) = callback_factory::callback_scope_stack(&env)? {
        return Ok(ScopeStack {
            inner,
            publication_buffer,
        });
    }
    Ok(ScopeStack::from(current_scope_stack_handle()))
}

/// Binds a scope stack to the current thread or async resource.
///
/// This mutates the current execution resource. Use `withScopeStack` when
/// concurrent asynchronous branches need isolated stack replacements.
#[napi]
pub fn set_thread_scope_stack(env: Env, stack: &ScopeStack) -> napi::Result<()> {
    if callback_factory::set_callback_scope_stack(&env, stack)? {
        return Ok(());
    }
    bind_thread_scope_stack(stack.inner.clone());
    Ok(())
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `setThreadScopeStack` has been called on the current
/// thread, or the caller is inside a task-local scope. Returns `false` when
/// only the auto-created default is present.
#[napi]
pub fn scope_stack_active(env: Env) -> napi::Result<bool> {
    if callback_factory::callback_scope_stack(&env)?.is_some() {
        return Ok(true);
    }
    Ok(scope_stack_is_active())
}

/// Returns the most recent callback error that could not be surfaced through a direct exception.
///
/// This is primarily used for sanitize callback paths that omit observability
/// payloads and cannot surface their errors directly.
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
pub async fn test_closed_tool_callback(
    callback: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
    name: String,
    args: Json,
) -> Result<Json> {
    clear_recorded_callback_error();
    let _ = callback.clone().abort();
    let wrapped = callable::wrap_js_tool_fn(callback);
    let fallback = args.clone();
    match wrapped(name, args).await {
        Ok(value) => Ok(value),
        Err(error) => {
            record_callback_error(error.to_string());
            Ok(fallback)
        }
    }
}

/// Internal test helper: model a closed JS LLM request sanitizer.
#[napi(js_name = "__testClosedLlmSanitizeRequestCallback")]
pub fn test_closed_llm_sanitize_request_callback(
    callback: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    request: Json,
) -> Result<Option<Json>> {
    clear_recorded_callback_error();
    let _ = callback.clone().abort();
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    drop(llm_request);
    record_callback_error("nemo_relay: failed to queue JS LLM sanitize request callback");
    Ok(None)
}

/// Internal test helper: model a closed JS LLM response sanitizer.
#[napi(js_name = "__testClosedLlmResponseCallback")]
pub fn test_closed_llm_response_callback(
    callback: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    response: Json,
) -> Option<Json> {
    clear_recorded_callback_error();
    let _ = callback.clone().abort();
    drop(response);
    record_callback_error("nemo_relay: failed to queue JS LLM sanitize response callback");
    None
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

/// Internal test helper: exercise PromiseAwareFn queue and conversion failures.
#[napi(
    js_name = "__testClosedPromiseAwareCall",
    ts_return_type = "Promise<unknown>"
)]
pub fn test_closed_promise_aware_call(
    env: Env,
    func: JsFunction,
    force_conversion_failure: Option<bool>,
) -> Result<JsObject> {
    let promise_aware = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &func).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );
    if !force_conversion_failure.unwrap_or(false) {
        promise_aware.close();
    }

    env.execute_tokio_future(
        async move {
            if force_conversion_failure.unwrap_or(false) {
                promise_aware
                    .call_with_arg0(Box::new(|_| {
                        Err(napi::Error::from_reason(
                            "forced PromiseAwareFn conversion failure",
                        ))
                    }))
                    .await
            } else {
                promise_aware.call(Json::Null).await
            }
            .map_err(to_napi_err)
        },
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
pub fn get_handle(env: Env) -> Result<ScopeHandle> {
    with_effective_scope_stack(&env, core_scope_api::get_handle)?
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
    env: Env,
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
    with_effective_scope_stack(&env, || {
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
    })?
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
    env: Env,
    handle: &ScopeHandle,
    output: Option<Json>,
    timestamp: Option<f64>,
    metadata: Option<Json>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_scope_api::pop_scope(
            core_scope_api::PopScopeParams::builder()
                .handle_uuid(&handle.inner.uuid)
                .output_opt(opt_json(output))
                .timestamp_opt(timestamp)
                .metadata_opt(opt_json(metadata))
                .build(),
        )
    })?
    .map_err(to_napi_err)?;
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
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    let scope_handle = with_scope_stack_handle(scope_stack.clone(), || {
        with_nested_publication_buffer(publication_buffer.clone(), || {
            core_scope_api::push_scope(
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
        })
    })
    .map(ScopeHandle::from)
    .map_err(to_napi_err)?;

    let scope_uuid = scope_handle.inner.uuid;
    // Hand the callback a real `ScopeHandle` instance, matching the Rust,
    // Python bindings, so it can be passed back into `event`,
    // `toolCallExecute`, and `llmCallExecute`. The instance is materialized on
    // the JS thread because a `napi_wrap`'d handle cannot cross the
    // threadsafe-function boundary as plain JSON.
    let callback_handle = scope_handle.inner.clone();

    // Create a promise-aware wrapper so we handle both sync and async callbacks.
    let error_publication_buffer = publication_buffer.clone();
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callback).map_err(|e| {
            let status_message = format!("failed to create PromiseAwareFn: {e}");
            let _ = with_scope_stack_handle(scope_stack.clone(), || {
                with_nested_publication_buffer(error_publication_buffer.clone(), || {
                    core_scope_api::pop_scope(
                        core_scope_api::PopScopeParams::builder()
                            .handle_uuid(&scope_uuid)
                            .metadata_opt(Some(otel_status_metadata(
                                "ERROR",
                                Some(status_message.clone()),
                            )))
                            .build(),
                    )
                })
            });
            napi::Error::from_reason(status_message)
        })?,
    );

    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
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
                                Err(error) => {
                                    otel_status_metadata("ERROR", Some(error.to_string()))
                                }
                            };
                            // Always pop the scope, even on error.
                            let _ = core_scope_api::pop_scope(
                                core_scope_api::PopScopeParams::builder()
                                    .handle_uuid(&scope_uuid)
                                    .metadata_opt(Some(metadata))
                                    .build(),
                            );
                            result.map_err(to_napi_err)
                        })
                        .await
                },
            )
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
    env: Env,
    name: String,
    handle: Option<&ScopeHandle>,
    data: Option<Json>,
    metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_scope_api::event(
            core_scope_api::EmitMarkEventParams::builder()
                .name(&name)
                .parent_opt(handle.map(|h| &h.inner))
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
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
    env: Env,
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
    with_effective_scope_stack(&env, || {
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
    })?
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
    env: Env,
    handle: &ToolHandle,
    result: ToolExecutionResult,
    data: Option<Json>,
    metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_tool_api::tool_call_end(
            core_tool_api::ToolCallEndParams::builder()
                .handle(&handle.inner)
                .execution_result(result.into())
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
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
#[napi(ts_return_type = "Promise<ToolExecutionResult>")]
pub fn tool_call_execute(
    env: Env,
    name: String,
    args: Json,
    #[napi(ts_arg_type = "(arg: Json) => ToolExecutionResult")] func: JsFunction,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<JsObject> {
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));
    let callback = callable::safe_execution_callback(&env, &func)?;
    let exec_fn = callable::wrap_js_tool_exec_fn(json_callback_tsfn(&env, &callback)?);
    let default_fn: ToolExecutionNextFn = std::sync::Arc::new(move |args| exec_fn(args));

    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
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
                            .map(ToolExecutionResult::from)
                            .map_err(to_napi_err)
                        })
                        .await
                },
            )
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
#[napi(ts_return_type = "Promise<ToolExecutionResult>")]
pub fn tool_call_execute_async(
    env: Env,
    name: String,
    args: Json,
    #[napi(
        ts_arg_type = "(arg: Json, signal: AbortSignal) => ToolExecutionResult | Promise<ToolExecutionResult>"
    )]
    func: JsFunction,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
) -> Result<JsObject> {
    let attrs = ToolAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));

    // Create promise-aware wrapper — this must happen on the JS thread (we have Env).
    let pa_fn = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &func).map_err(|e| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {e}"))
        })?,
    );

    let exec_fn: ToolExecutionNextFn = std::sync::Arc::new(move |args| {
        let pa_fn = pa_fn.clone();
        Box::pin(async move {
            let result = pa_fn.call(args).await?;
            serde_json::from_value(result).map_err(|error| {
                FlowError::Internal(format!(
                    "tool execution callback must return ToolExecutionResult: {error}"
                ))
            })
        })
    });

    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
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
                            .map(ToolExecutionResult::from)
                            .map_err(to_napi_err)
                        })
                        .await
                },
            )
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
    env: Env,
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
    with_effective_scope_stack(&env, || core_llm_api::llm_call(params))?
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
    env: Env,
    handle: &LlmHandle,
    response: Json,
    data: Option<Json>,
    metadata: Option<Json>,
    timestamp: Option<f64>,
) -> Result<()> {
    let timestamp = parse_timestamp_micros(timestamp)?;
    with_effective_scope_stack(&env, || {
        core_llm_api::llm_call_end(
            core_llm_api::LlmCallEndParams::builder()
                .handle(&handle.inner)
                .response(response)
                .data_opt(opt_json(data))
                .metadata_opt(opt_json(metadata))
                .timestamp_opt(timestamp)
                .build(),
        )
    })?
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
    #[napi(ts_arg_type = "(arg: Json) => any")] func: JsFunction,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_decode: Option<JsFunction>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_encode: Option<JsFunction>,
    #[napi(ts_arg_type = "(arg: Json) => any")] response_codec_decode: Option<JsFunction>,
) -> Result<JsObject> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    let callback = callable::safe_execution_callback(&env, &func)?;
    let exec_fn = callable::wrap_js_llm_exec_fn(json_callback_tsfn(&env, &callback)?);
    let default_fn: LlmExecutionNextFn = std::sync::Arc::new(move |req| exec_fn(req));
    let mut codec_references = Vec::new();
    let codec = match (codec_decode.as_ref(), codec_encode.as_ref()) {
        (Some(d), Some(e)) => {
            let (codec, references) = node_llm_codec(&env, d, e)?;
            codec_references.extend(references);
            Some(codec)
        }
        (None, None) => None,
        _ => {
            return Err(napi::Error::from_reason(
                "codecDecode and codecEncode must be provided together",
            ));
        }
    };
    let response_codec = response_codec_decode
        .as_ref()
        .map(|decode| node_llm_response_codec(&env, decode))
        .transpose()?
        .map(|(codec, references)| {
            codec_references.extend(references);
            codec
        });
    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
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
            )
            .await
        },
        move |_env, result| {
            drop(codec_references);
            Ok(result)
        },
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
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_decode: Option<JsFunction>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_encode: Option<JsFunction>,
    #[napi(ts_arg_type = "(arg: Json) => any")] response_codec_decode: Option<JsFunction>,
) -> Result<JsObject> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
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

    let mut codec_references = Vec::new();
    let codec = match (codec_decode.as_ref(), codec_encode.as_ref()) {
        (Some(d), Some(e)) => {
            let (codec, references) = node_llm_codec(&env, d, e)?;
            codec_references.extend(references);
            Some(codec)
        }
        (None, None) => None,
        _ => {
            return Err(napi::Error::from_reason(
                "codecDecode and codecEncode must be provided together",
            ));
        }
    };
    let response_codec = response_codec_decode
        .as_ref()
        .map(|decode| node_llm_response_codec(&env, decode))
        .transpose()?
        .map(|(codec, references)| {
            codec_references.extend(references);
            codec
        });

    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
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
            )
            .await
        },
        move |_env, result| {
            drop(codec_references);
            Ok(result)
        },
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
/// callback is invoked once when the stream is exhausted or closed early and
/// must return a JSON value representing the aggregated response. Consumers
/// that stop reading early must await `stream.close()` to wait for producer
/// cleanup and surface cleanup errors.
#[allow(clippy::too_many_arguments)]
#[napi(ts_return_type = "Promise<LlmStream>")]
pub fn llm_stream_call_execute(
    env: Env,
    name: String,
    request: Json,
    func: JsFunction,
    collector: Option<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    finalizer: Option<ThreadsafeFunction<(), ErrorStrategy::Fatal>>,
    handle: Option<&ScopeHandle>,
    attributes: Option<u32>,
    data: Option<Json>,
    metadata: Option<Json>,
    model_name: Option<String>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_decode: Option<JsFunction>,
    #[napi(ts_arg_type = "(arg: Json) => any")] codec_encode: Option<JsFunction>,
    #[napi(ts_arg_type = "(arg: Json) => any")] response_codec_decode: Option<JsFunction>,
) -> Result<JsObject> {
    let attrs = LlmAttributes::from_bits_truncate(attributes.unwrap_or(0));
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    let parent = handle
        .map(|h| h.inner.clone())
        .unwrap_or_else(|| effective_scope_top(&scope_stack));
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
    let func = std::sync::Arc::new(scoped_stream_callback_tsfn(&env, &func)?);
    let default_fn: LlmStreamExecutionNextFn = std::sync::Arc::new(move |req: LlmRequest| {
        let propagation_parent_uuid = match capture_propagation_context_handle() {
            Ok(context) => context.parent_uuid.to_string(),
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let stream_id = NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let closed = register_stream_channel(stream_id, tx);
        let scope_stack = current_scope_stack_handle();
        let publication_buffer = capture_nested_publication_buffer();

        // Serialize the LlmRequest to JSON and wrap with streamId so JS can extract both
        let req_json = serde_json::to_value(&req).unwrap_or(Json::Null);
        let wrapper = serde_json::json!({
            "__nemo_relay_native": req_json,
            "__nemo_relay_stream_id": stream_id,
        });

        // NonBlocking: queue the call on the JS event loop and return immediately.
        // The JS function starts async iteration and pushes chunks via pushStreamChunk.
        let call_status = func.call(
            ScopedStreamCall {
                request: wrapper,
                scope_stack,
                publication_buffer,
                propagation_parent_uuid,
            },
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Box::pin(async move {
            ensure_stream_callback_queued(stream_id, call_status)?;

            Ok(LlmJsonStream::from_closeable(NodePushStream {
                receiver: tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
                stream_id,
                closed,
            }))
        })
    });

    let mut codec_references = Vec::new();
    let codec = match (codec_decode.as_ref(), codec_encode.as_ref()) {
        (Some(d), Some(e)) => {
            let (codec, references) = node_llm_codec(&env, d, e)?;
            codec_references.extend(references);
            Some(codec)
        }
        (None, None) => None,
        _ => {
            return Err(napi::Error::from_reason(
                "codecDecode and codecEncode must be provided together",
            ));
        }
    };
    let response_codec = response_codec_decode
        .as_ref()
        .map(|decode| node_llm_response_codec(&env, decode))
        .transpose()?
        .map(|(codec, references)| {
            codec_references.extend(references);
            codec
        });
    let completion_codec_references = codec_references.clone();
    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id.clone(),
                publication_buffer.clone(),
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
                            let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
                            let (closed, closed_rx) = tokio::sync::watch::channel(None);
                            tokio::spawn(with_publication_callback_context(
                                publication_context_id,
                                publication_buffer,
                                forward_stream_to_channel(rust_stream, tx, cancel_rx, closed),
                            ));

                            Ok(LlmStream {
                                receiver: tokio::sync::Mutex::new(rx),
                                cancel,
                                closed: closed_rx,
                                codec_references,
                            })
                        })
                        .await
                },
            )
            .await
        },
        move |_env, result| {
            drop(completion_codec_references);
            Ok(result)
        },
    )
}

// ---------------------------------------------------------------------------
// Tool guardrail registrations
// ---------------------------------------------------------------------------

macro_rules! napi_event_guardrail_api {
    ($register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path) => {
        /// Register an event sanitize guardrail.
        ///
        /// The callback may return fields directly or in a Promise. Scope and mark
        /// calls queue the event and return synchronously; publication resumes after
        /// the Promise settles. Callback, serialization, conversion, or invalid-result
        /// failures clear the emitted event fields and record the error for
        /// `getLastCallbackError()`. Await `flushSubscribers()` before inspecting
        /// either the delivered event or that error.
        #[napi]
        pub fn $register_name(
            env: Env,
            name: String,
            priority: i32,
            #[napi(
                ts_arg_type = "(event: Json, fields: EventSanitizeFields) => EventSanitizeFields | Promise<EventSanitizeFields>"
            )]
            guardrail: JsFunction,
        ) -> Result<()> {
            $core_register(&name, priority, node_event_sanitize_fn(&env, &guardrail)?)
                .map_err(to_napi_err)
        }

        #[napi]
        pub fn $deregister_name(name: String) -> Result<bool> {
            $core_deregister(&name).map_err(to_napi_err)
        }
    };
}

napi_event_guardrail_api!(
    register_mark_sanitize_guardrail,
    deregister_mark_sanitize_guardrail,
    core_registry_api::register_mark_sanitize_guardrail,
    core_registry_api::deregister_mark_sanitize_guardrail
);
napi_event_guardrail_api!(
    register_scope_sanitize_start_guardrail,
    deregister_scope_sanitize_start_guardrail,
    core_registry_api::register_scope_sanitize_start_guardrail,
    core_registry_api::deregister_scope_sanitize_start_guardrail
);
napi_event_guardrail_api!(
    register_scope_sanitize_end_guardrail,
    deregister_scope_sanitize_end_guardrail,
    core_registry_api::register_scope_sanitize_end_guardrail,
    core_registry_api::deregister_scope_sanitize_end_guardrail
);

macro_rules! napi_guardrail_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[doc = $reg_doc])*
        #[napi]
        pub fn $register_name(
            env: Env,
            name: String,
            priority: i32,
            #[napi(
                ts_arg_type = "(toolName: string, value: Json) => Json | Promise<Json>"
            )]
            guardrail: JsFunction,
        ) -> Result<()> {
            let callback = Arc::new(PromiseAwareFn::new(&env, &guardrail)?);
            $core_register(
                &name,
                priority,
                callable::wrap_js_tool_sanitize_promise_fn(callback),
            )
            .map_err(to_napi_err)
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
    /// If the callback throws, Relay omits the emitted payload and records the error
    /// for `getLastCallbackError()`.
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
    /// If the callback throws, Relay omits the emitted payload and records the error
    /// for `getLastCallbackError()`.
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
/// If the callback throws, the managed call rejects and the protected callback does not run.
#[napi]
pub fn register_tool_conditional_execution_guardrail(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(toolName: string, args: Json) => string | null | Promise<string | null>"
    )]
    guardrail: JsFunction,
) -> Result<()> {
    let callback = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &guardrail).map_err(|error| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {error}"))
        })?,
    );
    core_registry_api::register_tool_conditional_execution_guardrail(
        &name,
        priority,
        callable::wrap_js_tool_conditional_promise_fn(callback),
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
            env: Env,
            name: String,
            priority: i32,
            break_chain: bool,
            #[napi(
                ts_arg_type = "(toolName: string, args: Json) => Json | Promise<Json>"
            )]
            callable: JsFunction,
        ) -> Result<()> {
            let callback = std::sync::Arc::new(
                crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|error| {
                    napi::Error::from_reason(format!("failed to create PromiseAwareFn: {error}"))
                })?,
            );
            $core_register(
                &name,
                priority,
                break_chain,
                callable::wrap_js_tool_request_intercept_promise_fn(callback),
            )
            .map_err(to_napi_err)
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
    /// If the callback throws, the managed call rejects and later middleware does not run.
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
/// the chain. `next` may be called repeatedly or concurrently while `callable` is
/// pending; each call receives an isolated scope-stack branch, and unfinished or
/// later calls reject after `callable` settles.
#[napi]
pub fn register_tool_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(args: Json, next: (args: Json) => ToolExecutionResult | Promise<ToolExecutionResult>) => { result: Json; annotation?: Json; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }> } | Promise<{ result: Json; annotation?: Json; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }> }>"
    )]
    callable: JsFunction,
) -> Result<()> {
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
    Ok(())
}

/// Deregister a tool execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_tool_execution_intercept(name: String) -> Result<bool> {
    core_registry_api::deregister_tool_execution_intercept(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register a guardrail that sanitizes LLM request data before execution.
///
/// The `guardrail` callback receives `(request, context)` and must return the sanitized request,
/// or `null` to omit the observability payload. Lower `priority` values run first. Throws if a
/// guardrail with the same `name` already exists. If the callback throws, Relay omits the payload
/// and annotation, continues publication, and records the error for `getLastCallbackError()`.
#[napi]
pub fn register_llm_sanitize_request_guardrail(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, context: import('./plugin').LlmSanitizeRequestContext) => Json | null | Promise<Json | null>"
    )]
    guardrail: JsFunction,
) -> Result<()> {
    core_registry_api::register_llm_sanitize_request_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_sanitize_request_promise_fn(Arc::new(PromiseAwareFn::new(
            &env, &guardrail,
        )?)),
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
/// The `guardrail` callback receives `(response, context)` and must return the sanitized response,
/// or `null` to omit the observability payload. Lower `priority` values run first. Throws if a
/// guardrail with the same `name` already exists. If the callback throws, Relay omits the payload
/// and annotation, continues publication, and records the error for `getLastCallbackError()`.
#[napi]
pub fn register_llm_sanitize_response_guardrail(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(response: Json, context: import('./plugin').LlmSanitizeResponseContext) => Json | null | Promise<Json | null>"
    )]
    guardrail: JsFunction,
) -> Result<()> {
    core_registry_api::register_llm_sanitize_response_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_sanitize_response_promise_fn(Arc::new(PromiseAwareFn::new(
            &env, &guardrail,
        )?)),
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
/// If the callback throws, the managed call rejects and the protected callback does not run.
#[napi]
pub fn register_llm_conditional_execution_guardrail(
    env: Env,
    name: String,
    priority: i32,
    #[napi(ts_arg_type = "(request: Json) => string | null | Promise<string | null>")]
    guardrail: JsFunction,
) -> Result<()> {
    let callback = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &guardrail).map_err(|error| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {error}"))
        })?,
    );
    core_registry_api::register_llm_conditional_execution_guardrail(
        &name,
        priority,
        callable::wrap_js_llm_conditional_promise_fn(callback),
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
/// If the callback throws, the managed call rejects and later middleware does not run.
#[napi]
pub fn register_llm_request_intercept(
    env: Env,
    name: String,
    priority: i32,
    break_chain: bool,
    #[napi(
        ts_arg_type = "(args: { name: string; request: Json; annotated: Json | null }) => import('./plugin').LlmRequestInterceptOutcome | Promise<import('./plugin').LlmRequestInterceptOutcome>"
    )]
    callable: JsFunction,
) -> Result<()> {
    let callback = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|error| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {error}"))
        })?,
    );
    core_registry_api::register_llm_request_intercept(
        &name,
        priority,
        break_chain,
        callable::wrap_js_llm_request_intercept_promise_fn(callback),
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
/// short-circuit the chain. `next` may be called repeatedly or concurrently while
/// `callable` is pending; each call receives an isolated scope-stack branch, and
/// unfinished or later calls reject after `callable` settles.
#[napi]
pub fn register_llm_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, next: (request: Json) => Json | Promise<Json>) => Json | Promise<Json>"
    )]
    callable: JsFunction,
) -> Result<()> {
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
    Ok(())
}

/// Deregister an LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_execution_intercept(name: String) -> Result<bool> {
    core_registry_api::deregister_llm_execution_intercept(&name).map_err(to_napi_err)
}

/// Register a streaming LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original streaming implementation; in Node the
/// returned promise resolves to an array of downstream JSON chunks. Skip calling
/// `next` to short-circuit the chain. `next` may be called repeatedly or concurrently
/// while `callable` is pending; each call receives an isolated scope-stack branch,
/// and unfinished or later calls reject after the returned interceptor stream settles.
/// A downstream stream returned successfully by `next` keeps its normal lifetime.
#[napi]
pub fn register_llm_stream_execution_intercept(
    env: Env,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, next: (request: Json) => Promise<Json[]>) => Json | Json[] | Promise<Json | Json[]>"
    )]
    callable: JsFunction,
) -> Result<()> {
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
    Ok(())
}

/// Deregister an LLM stream execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed.
#[napi]
pub fn deregister_llm_stream_execution_intercept(name: String) -> Result<bool> {
    core_registry_api::deregister_llm_stream_execution_intercept(&name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Register a named event subscriber that receives all lifecycle events.
///
/// The `callback` receives each event as the canonical JSON event object and may return a
/// Promise. Events are delivered asynchronously and non-blocking. Callback failures are
/// isolated, reported to stderr and `getLastCallbackError()`, and do not reject
/// `flushSubscribers()`. Throws if a subscriber with the same `name` already exists.
#[napi]
pub fn register_subscriber(
    env: Env,
    name: String,
    #[napi(ts_arg_type = "(event: Json) => void | Promise<void>")] callback: JsFunction,
) -> Result<()> {
    let callback = callable::wrap_js_event_subscriber(&env, name.clone(), callback)?;
    core_subscriber_api::register_subscriber(&name, callback).map_err(to_napi_err)
}

/// Deregister an event subscriber by name.
///
/// Future emissions stop seeing the subscriber. Already queued event snapshots may
/// still run. Returns `true` if a subscriber with that name was found and removed.
#[napi]
pub fn deregister_subscriber(name: String) -> Result<bool> {
    core_subscriber_api::deregister_subscriber(&name).map_err(to_napi_err)
}

/// Return a Promise that resolves when native and JavaScript subscriber callbacks and
/// managed terminal publications registered before this call finish.
///
/// Call this function outside subscribers, event sanitizers, conditional
/// guardrails, and request or execution intercepts. A queued tool or LLM
/// observability sanitizer may call it, but the Promise resolves without
/// waiting for its own publication.
///
/// Awaiting this Promise does not block the Node event loop while Promise-returning event
/// sanitizers settle or queued JavaScript subscriber callbacks run. Native events emitted by a
/// JavaScript subscriber are separate publications and may require another flush.
///
/// The Promise rejects if the blocking task fails or the core subscriber flush returns an error.
/// Callers should handle errors when awaiting it.
#[napi(ts_return_type = "Promise<void>")]
pub fn flush_subscribers(env: Env) -> Result<JsObject> {
    let reentrant = crate::callback_factory::publication_callback_active(&env)?;
    env.execute_tokio_future(
        async move {
            if reentrant {
                return Ok(());
            }
            tokio::task::spawn_blocking(|| {
                core_subscriber_api::flush_subscribers()?;
                callable::flush_js_subscriber_callbacks()
            })
            .await
            .map_err(|error| to_napi_err(FlowError::Internal(error.to_string())))?
            .map_err(to_napi_err)
        },
        |env, _| env.get_undefined(),
    )
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations — Tool
// ---------------------------------------------------------------------------

macro_rules! napi_scope_event_guardrail_api {
    ($register_name:ident, $deregister_name:ident, $core_register:path, $core_deregister:path) => {
        /// Register a scope-local event sanitize guardrail.
        ///
        /// The callback may return fields directly or in a Promise. Scope and mark
        /// calls queue the event and return synchronously; publication resumes after
        /// the Promise settles. Callback, serialization, conversion, or invalid-result
        /// failures clear the emitted event fields and record the error for
        /// `getLastCallbackError()`. Await `flushSubscribers()` before inspecting
        /// either the delivered event or that error.
        #[napi]
        pub fn $register_name(
            env: Env,
            scope_uuid: String,
            name: String,
            priority: i32,
            #[napi(
                ts_arg_type = "(event: Json, fields: EventSanitizeFields) => EventSanitizeFields | Promise<EventSanitizeFields>"
            )]
            guardrail: JsFunction,
        ) -> Result<()> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
            $core_register(
                &uuid,
                &name,
                priority,
                node_event_sanitize_fn(&env, &guardrail)?,
            )
            .map_err(to_napi_err)
        }

        #[napi]
        pub fn $deregister_name(scope_uuid: String, name: String) -> Result<bool> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
            $core_deregister(&uuid, &name).map_err(to_napi_err)
        }
    };
}

napi_scope_event_guardrail_api!(
    scope_register_mark_sanitize_guardrail,
    scope_deregister_mark_sanitize_guardrail,
    core_registry_api::scope_register_mark_sanitize_guardrail,
    core_registry_api::scope_deregister_mark_sanitize_guardrail
);
napi_scope_event_guardrail_api!(
    scope_register_scope_sanitize_start_guardrail,
    scope_deregister_scope_sanitize_start_guardrail,
    core_registry_api::scope_register_scope_sanitize_start_guardrail,
    core_registry_api::scope_deregister_scope_sanitize_start_guardrail
);
napi_scope_event_guardrail_api!(
    scope_register_scope_sanitize_end_guardrail,
    scope_deregister_scope_sanitize_end_guardrail,
    core_registry_api::scope_register_scope_sanitize_end_guardrail,
    core_registry_api::scope_deregister_scope_sanitize_end_guardrail
);

macro_rules! napi_scope_guardrail_tool_api {
    ($(#[doc = $reg_doc:expr_2021])* $register_name:ident,
     $(#[doc = $dereg_doc:expr_2021])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:path) => {
        $(#[doc = $reg_doc])*
        #[napi]
        pub fn $register_name(
            env: Env,
            scope_uuid: String,
            name: String,
            priority: i32,
            #[napi(
                ts_arg_type = "(toolName: string, value: Json) => Json | Promise<Json>"
            )]
            guardrail: JsFunction,
        ) -> Result<()> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
            let callback = Arc::new(PromiseAwareFn::new(&env, &guardrail)?);
            $core_register(
                &uuid,
                &name,
                priority,
                callable::wrap_js_tool_sanitize_promise_fn(callback),
            )
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

napi_scope_guardrail_tool_api!(
    /// Register a scope-local guardrail that sanitizes tool request arguments before execution.
    ///
    /// The `guardrail` callback receives `(toolName, args)` and must return sanitized args.
    /// Higher `priority` values run first. Throws if a guardrail with the same `name` already exists
    /// on the specified scope.
    /// If the callback throws, Relay omits the emitted payload and records the error
    /// for `getLastCallbackError()`.
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
    /// If the callback throws, Relay omits the emitted payload and records the error
    /// for `getLastCallbackError()`.
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
/// If the callback throws, the managed call rejects and the protected callback does not run.
#[napi]
pub fn scope_register_tool_conditional_execution_guardrail(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(toolName: string, args: Json) => string | null | Promise<string | null>"
    )]
    guardrail: JsFunction,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_tool_conditional_execution_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_tool_conditional_promise_fn(std::sync::Arc::new(
            crate::promise_call::PromiseAwareFn::new(&env, &guardrail).map_err(|error| {
                napi::Error::from_reason(format!("failed to create PromiseAwareFn: {error}"))
            })?,
        )),
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
            env: Env,
            scope_uuid: String,
            name: String,
            priority: i32,
            break_chain: bool,
            #[napi(
                ts_arg_type = "(toolName: string, args: Json) => Json | Promise<Json>"
            )]
            callable: JsFunction,
        ) -> Result<()> {
            let uuid = uuid::Uuid::parse_str(&scope_uuid)
                .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
            let callback = std::sync::Arc::new(
                crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|error| {
                    napi::Error::from_reason(format!("failed to create PromiseAwareFn: {error}"))
                })?,
            );
            $core_register(
                &uuid,
                &name,
                priority,
                break_chain,
                callable::wrap_js_tool_request_intercept_promise_fn(callback),
            )
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
    /// If the callback throws, the managed call rejects and later middleware does not run.
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
/// the chain. `next` may be called repeatedly or concurrently while `callable` is
/// pending; each call receives an isolated scope-stack branch, and unfinished or
/// later calls reject after `callable` settles.
#[napi]
pub fn scope_register_tool_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(args: Json, next: (args: Json) => ToolExecutionResult | Promise<ToolExecutionResult>) => { result: Json; annotation?: Json; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }> } | Promise<{ result: Json; annotation?: Json; pendingMarks?: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }> }>"
    )]
    callable: JsFunction,
) -> Result<()> {
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
    Ok(())
}

/// Deregister a scope-local tool execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_tool_execution_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_tool_execution_intercept(&uuid, &name).map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local guardrail registrations — LLM
// ---------------------------------------------------------------------------

/// Register a scope-local guardrail that sanitizes LLM request data before execution.
///
/// The `guardrail` callback receives `(request, context)` and must return the sanitized request,
/// or `null` to omit the observability payload. Lower `priority` values run first. Throws if a
/// guardrail with the same `name` already exists on the specified scope. If the callback throws,
/// Relay omits the payload and annotation, continues publication, and records the error for
/// `getLastCallbackError()`.
#[napi]
pub fn scope_register_llm_sanitize_request_guardrail(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, context: import('./plugin').LlmSanitizeRequestContext) => Json | null | Promise<Json | null>"
    )]
    guardrail: JsFunction,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_llm_sanitize_request_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_sanitize_request_promise_fn(Arc::new(PromiseAwareFn::new(
            &env, &guardrail,
        )?)),
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
/// The `guardrail` callback receives `(response, context)` and must return the sanitized response,
/// or `null` to omit the observability payload. Lower `priority` values run first. Throws if a
/// guardrail with the same `name` already exists on the specified scope. If the callback throws,
/// Relay omits the payload and annotation, continues publication, and records the error for
/// `getLastCallbackError()`.
#[napi]
pub fn scope_register_llm_sanitize_response_guardrail(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(response: Json, context: import('./plugin').LlmSanitizeResponseContext) => Json | null | Promise<Json | null>"
    )]
    guardrail: JsFunction,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_llm_sanitize_response_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_sanitize_response_promise_fn(Arc::new(PromiseAwareFn::new(
            &env, &guardrail,
        )?)),
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
/// If the callback throws, the managed call rejects and the protected callback does not run.
#[napi]
pub fn scope_register_llm_conditional_execution_guardrail(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(ts_arg_type = "(request: Json) => string | null | Promise<string | null>")]
    guardrail: JsFunction,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_register_llm_conditional_execution_guardrail(
        &uuid,
        &name,
        priority,
        callable::wrap_js_llm_conditional_promise_fn(std::sync::Arc::new(
            crate::promise_call::PromiseAwareFn::new(&env, &guardrail).map_err(|error| {
                napi::Error::from_reason(format!("failed to create PromiseAwareFn: {error}"))
            })?,
        )),
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
/// If the callback throws, the managed call rejects and later middleware does not run.
#[napi]
pub fn scope_register_llm_request_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    break_chain: bool,
    #[napi(
        ts_arg_type = "(args: { name: string; request: Json; annotated: Json | null }) => import('./plugin').LlmRequestInterceptOutcome | Promise<import('./plugin').LlmRequestInterceptOutcome>"
    )]
    callable: JsFunction,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    let callback = std::sync::Arc::new(
        crate::promise_call::PromiseAwareFn::new(&env, &callable).map_err(|error| {
            napi::Error::from_reason(format!("failed to create PromiseAwareFn: {error}"))
        })?,
    );
    core_registry_api::scope_register_llm_request_intercept(
        &uuid,
        &name,
        priority,
        break_chain,
        callable::wrap_js_llm_request_intercept_promise_fn(callback),
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
/// short-circuit the chain. `next` may be called repeatedly or concurrently while
/// `callable` is pending; each call receives an isolated scope-stack branch, and
/// unfinished or later calls reject after `callable` settles.
#[napi]
pub fn scope_register_llm_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, next: (request: Json) => Json | Promise<Json>) => Json | Promise<Json>"
    )]
    callable: JsFunction,
) -> Result<()> {
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
    Ok(())
}

/// Deregister a scope-local LLM execution intercept by name.
///
/// Returns `true` if an intercept with that name was found and removed from the specified scope.
#[napi]
pub fn scope_deregister_llm_execution_intercept(scope_uuid: String, name: String) -> Result<bool> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_llm_execution_intercept(&uuid, &name).map_err(to_napi_err)
}

/// Register a scope-local streaming LLM execution intercept following the middleware chain pattern.
///
/// The `callable` receives the request and a `next` function. Call `next(request)` to
/// invoke the next intercept or original streaming implementation; in Node the
/// returned promise resolves to an array of downstream JSON chunks. Skip calling
/// `next` to short-circuit the chain. `next` may be called repeatedly or concurrently
/// while `callable` is pending; each call receives an isolated scope-stack branch,
/// and unfinished or later calls reject after the returned interceptor stream settles.
/// A downstream stream returned successfully by `next` keeps its normal lifetime.
#[napi]
pub fn scope_register_llm_stream_execution_intercept(
    env: Env,
    scope_uuid: String,
    name: String,
    priority: i32,
    #[napi(
        ts_arg_type = "(request: Json, next: (request: Json) => Promise<Json[]>) => Json | Json[] | Promise<Json | Json[]>"
    )]
    callable: JsFunction,
) -> Result<()> {
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
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    core_registry_api::scope_deregister_llm_stream_execution_intercept(&uuid, &name)
        .map_err(to_napi_err)
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registrations
// ---------------------------------------------------------------------------

/// Register a scope-local named event subscriber that receives lifecycle events
/// for the specified scope.
///
/// The `callback` receives each event as the canonical JSON event object and may return a
/// Promise. Events are delivered asynchronously and non-blocking. Callback failures are
/// isolated, reported to stderr and `getLastCallbackError()`, and do not reject
/// `flushSubscribers()`. Throws if a subscriber with the same `name` already exists on the
/// specified scope.
#[napi]
pub fn scope_register_subscriber(
    env: Env,
    scope_uuid: String,
    name: String,
    #[napi(ts_arg_type = "(event: Json) => void | Promise<void>")] callback: JsFunction,
) -> Result<()> {
    let uuid = uuid::Uuid::parse_str(&scope_uuid)
        .map_err(|e| napi::Error::from_reason(format!("invalid UUID: {e}")))?;
    let callback = callable::wrap_js_event_subscriber(&env, name.clone(), callback)?;
    core_subscriber_api::scope_register_subscriber(&uuid, &name, callback).map_err(to_napi_err)
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
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_tool_api::tool_request_intercepts(&name, args)
                                .await
                                .map_err(to_napi_err)
                        })
                        .await
                },
            )
            .await
        },
        |_env, result| Ok(result),
    )
}

/// Run the registered tool conditional execution guardrail chain.
/// Throws if any guardrail rejects.
#[napi(ts_return_type = "Promise<void>")]
pub fn tool_conditional_execution(env: Env, name: String, args: Json) -> Result<JsObject> {
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_tool_api::tool_conditional_execution(&name, &args)
                                .await
                                .map_err(to_napi_err)
                        })
                        .await
                },
            )
            .await
        },
        |env, _| env.get_undefined(),
    )
}

/// Run the registered LLM request intercept chain on the given request.
/// The `request` should be a JSON object with `headers` and `content` fields matching
/// the `LlmRequest` schema. Returns the transformed request as JSON.
#[napi(
    ts_return_type = "Promise<{ request: Json; annotated: Json | null; pendingMarks: Array<{ name: string; category?: string | null; categoryProfile?: Json; data?: Json; metadata?: Json }>; optimizationContributions: Array<{ id?: string; sequence?: number; producer: string; kind: 'input_compression' | 'model_routing' | (string & {}); applied: boolean; model_transition?: { baseline?: { model: string; provider?: string }; effective?: { model: string; provider?: string } }; token_impact?: { baseline?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; effective?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; saved?: { prompt_tokens?: number; completion_tokens?: number; cache_read_tokens?: number; cache_write_tokens?: number; total_tokens?: number }; quality?: 'observed' | 'estimated'; estimation_method?: string }; payload_schema?: { name: string; version: string }; payload?: Json; [key: string]: Json | undefined }> }>"
)]
pub fn llm_request_intercepts(env: Env, name: String, request: Json) -> Result<JsObject> {
    let llm_request: LlmRequest = serde_json::from_value(request)
        .map_err(|e| napi::Error::from_reason(format!("invalid LlmRequest: {e}")))?;
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_llm_api::llm_request_intercepts(&name, llm_request)
                                .await
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
            )
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
    let publication_context_id = callback_factory::event_sanitizer_callback_context_id(&env)?;
    let (scope_stack, publication_buffer) = effective_scope_context(&env)?;
    env.execute_tokio_future(
        async move {
            with_publication_callback_context(
                publication_context_id,
                publication_buffer,
                async move {
                    TASK_SCOPE_STACK
                        .scope(scope_stack, async move {
                            core_llm_api::llm_conditional_execution(&llm_request)
                                .await
                                .map_err(to_napi_err)
                        })
                        .await
                },
            )
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

/// One tagged sink configuration for `AtofExporter`.
#[napi(object)]
#[derive(Default)]
pub struct AtofExporterConfig {
    /// Sink type: `"file"` (default) or `"stream"`.
    pub r#type: Option<String>,
    /// Output directory. Defaults to the current working directory.
    pub output_directory: Option<String>,
    /// `"append"` (default) or `"overwrite"`.
    pub mode: Option<String>,
    /// Output filename. Defaults to `nemo-relay-events-YYYY-MM-DD-HH.MM.SS.jsonl`.
    pub filename: Option<String>,
    /// Stream endpoint URL. Required when `type` is `"stream"`.
    pub url: Option<String>,
    /// `"http_post"` (default), `"websocket"`, or `"ndjson"`.
    pub transport: Option<String>,
    /// Extra stream headers as string key/value pairs.
    pub headers: Option<Json>,
    /// Header names mapped to environment variables that supply their values.
    pub header_env: Option<Json>,
    /// Per-stream timeout in milliseconds.
    pub timeout_millis: Option<u32>,
    /// Field name policy applied before sending stream events.
    pub field_name_policy: Option<String>,
}

/// Single-sink Agent Trajectory Observability Format (ATOF) exporter.
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

    /// Return the JSONL output path, or `null` for a stream sink.
    #[napi(getter)]
    pub fn path(&self) -> Option<String> {
        self.inner
            .path()
            .map(|path| path.to_string_lossy().into_owned())
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

    /// Outside subscriber and middleware callbacks, wait for queued subscriber delivery, then
    /// flush the file sink or ask the stream sink to drain for up to its timeout. A stream timeout
    /// is logged and does not by itself return an error.
    #[napi]
    pub fn force_flush(&self) -> napi::Result<()> {
        self.inner
            .force_flush()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Outside subscriber and middleware callbacks, wait for queued subscriber delivery, then
    /// flush the file sink or ask the stream sink to drain and close up to its timeout. A stream
    /// timeout is logged and does not by itself return an error.
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
    /// `"full"`, `"gen_ai"`, or `"openinference"`.
    #[napi(ts_type = "\"full\" | \"gen_ai\" | \"openinference\"")]
    pub r#type: String,
    /// `"http_binary"` (default) or `"grpc"`.
    pub transport: Option<String>,
    /// OTLP endpoint, such as `http://localhost:4318/v1/traces`.
    pub endpoint: String,
    /// Extra exporter headers/metadata as string key/value pairs.
    pub headers: Option<Json>,
    /// Extra OpenTelemetry resource attributes as string key/value pairs.
    pub resource_attributes: Option<Json>,
    /// `service.name` resource attribute. Defaults to `"unknown_service"`.
    pub service_name: Option<String>,
    /// Optional `service.namespace` resource attribute.
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    pub service_version: Option<String>,
    /// Instrumentation scope name. Defaults to `"opentelemetry"`.
    pub instrumentation_scope: Option<String>,
    /// Export timeout in milliseconds. Defaults to `3000`.
    pub timeout_millis: Option<u32>,
    /// Mark projection for full and OpenInference exporters. Defaults to `"inherit"`.
    #[napi(ts_type = "\"inherit\" | \"event\" | \"tool\"")]
    pub mark_projection: Option<String>,
    /// Mark names excluded from full and OpenInference projections.
    pub mark_exclude_names: Option<Vec<String>>,
    /// Attribute aliases for full and OpenInference projections.
    pub attribute_mappings: Option<Json>,
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
    pub fn new(config: OpenTelemetryConfig) -> napi::Result<Self> {
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
    let validate_callback = match validate {
        Some(func) => {
            let direct = PersistentJsFunction::new(&env, &func)?;
            let callback = callable::safe_middleware_callback(&env, &func)?;
            let mut thread_safe = callback.create_threadsafe_function(
                0,
                |ctx: napi::threadsafe_function::ThreadSafeCallContext<Json>| Ok(vec![ctx.value]),
            )?;
            thread_safe.unref(&env)?;
            Some(NodePluginValidateCallback {
                direct,
                thread_safe,
                registration_thread: std::thread::current().id(),
            })
        }
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
        validate: validate_callback,
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

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct NodeDynamicPluginActivationSpec {
    #[serde(alias = "plugin_id")]
    plugin_id: String,
    kind: DynamicPluginKind,
    #[serde(alias = "manifest_ref")]
    manifest_ref: String,
    #[serde(default, alias = "environment_ref")]
    environment_ref: Option<String>,
    #[serde(default)]
    config: serde_json::Map<String, Json>,
}

impl From<NodeDynamicPluginActivationSpec> for CoreDynamicPluginActivationSpec {
    fn from(spec: NodeDynamicPluginActivationSpec) -> Self {
        Self {
            plugin_id: spec.plugin_id,
            kind: spec.kind,
            manifest_ref: spec.manifest_ref,
            environment_ref: spec.environment_ref,
            config: spec.config,
        }
    }
}

/// Owned dynamic plugin activation.
///
/// Keep this object alive while code may invoke callbacks registered by the
/// dynamic plugins. Call `close()` for deterministic cleanup; garbage
/// collection performs the same cleanup as a defensive fallback.
#[napi]
pub struct DynamicPluginActivation {
    close_state: Arc<DynamicPluginCloseState>,
    report: Json,
}

type DynamicPluginTeardownResult = std::result::Result<(), String>;

enum DynamicPluginCloseStatus {
    Active(Option<CorePluginHostActivation>),
    Closing,
    Closed,
}

struct DynamicPluginCloseState {
    status: StdMutex<DynamicPluginCloseStatus>,
    completion: tokio::sync::watch::Sender<Option<DynamicPluginTeardownResult>>,
}

impl DynamicPluginCloseState {
    fn new(activation: CorePluginHostActivation) -> Self {
        let (completion, _) = tokio::sync::watch::channel(None);
        Self {
            status: StdMutex::new(DynamicPluginCloseStatus::Active(Some(activation))),
            completion,
        }
    }

    fn active(&self) -> bool {
        let status = self
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &*status {
            DynamicPluginCloseStatus::Active(activation) => activation.is_some(),
            DynamicPluginCloseStatus::Closing | DynamicPluginCloseStatus::Closed => false,
        }
    }

    fn begin_close(self: &Arc<Self>, log_finalizer_error: bool) {
        let activation = {
            let mut status = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match &mut *status {
                DynamicPluginCloseStatus::Active(activation) => {
                    let activation = activation.take();
                    *status = DynamicPluginCloseStatus::Closing;
                    activation
                }
                DynamicPluginCloseStatus::Closing | DynamicPluginCloseStatus::Closed => None,
            }
        };
        let Some(activation) = activation else {
            return;
        };

        // Keep the activation outside the spawned closure so a thread-spawn
        // failure cannot drop it and synchronously run teardown on the JS thread.
        let activation = Arc::new(StdMutex::new(Some(activation)));
        let worker_activation = Arc::clone(&activation);
        let close_state = Arc::clone(self);
        let spawn = std::thread::Builder::new()
            .name("nemo-relay-node-plugin-teardown".into())
            .spawn(move || {
                let activation = worker_activation
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                let result = match activation {
                    Some(activation) => {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            activation.clear()
                        }))
                        .map_err(|_| "dynamic plugin teardown task panicked".to_string())
                        .and_then(|result| result.map_err(|error| error.to_string()))
                    }
                    None => Err("dynamic plugin teardown task lost its activation".to_string()),
                };
                if log_finalizer_error && let Err(error) = &result {
                    eprintln!("nemo_relay: dynamic plugin finalizer teardown failed: {error}");
                }
                close_state.finish(result);
            });

        if let Err(error) = spawn {
            // Cleanup must never fall back to the JS thread. Retain the
            // activation for process lifetime if no teardown thread can start.
            if let Some(activation) = activation
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
            {
                std::mem::forget(activation);
            }
            let error = format!("failed to start dynamic plugin teardown task: {error}");
            if log_finalizer_error {
                eprintln!("nemo_relay: dynamic plugin finalizer teardown failed: {error}");
            }
            self.finish(Err(error));
        }
    }

    fn finish(&self, result: DynamicPluginTeardownResult) {
        *self
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = DynamicPluginCloseStatus::Closed;
        self.completion.send_replace(Some(result));
    }

    async fn wait_for_close(&self) -> DynamicPluginTeardownResult {
        let mut completion = self.completion.subscribe();
        loop {
            if let Some(result) = completion.borrow().clone() {
                return result;
            }
            if completion.changed().await.is_err() {
                return Err(
                    "dynamic plugin teardown result channel closed unexpectedly".to_string()
                );
            }
        }
    }
}

#[napi]
impl DynamicPluginActivation {
    /// Return the validation report produced by activation.
    #[napi(getter)]
    pub fn report(&self) -> Json {
        self.report.clone()
    }

    /// Return whether this activation handle has not begun teardown.
    ///
    /// `false` does not guarantee another process-wide activation can start;
    /// failed teardown may intentionally retain the activation owner.
    #[napi(getter)]
    pub fn active(&self) -> napi::Result<bool> {
        Ok(self.close_state.active())
    }

    /// Clear plugin callbacks before unloading libraries and workers.
    ///
    /// This method is idempotent, including when concurrent callers race to
    /// close the same activation.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn close(&self, env: Env) -> napi::Result<JsObject> {
        let close_state = Arc::clone(&self.close_state);
        close_state.begin_close(false);
        env.execute_tokio_future(
            async move {
                close_state
                    .wait_for_close()
                    .await
                    .map_err(napi::Error::from_reason)
            },
            |env, _| env.get_undefined(),
        )
    }

    /// Supply the structured disposal signature to napi-rs declaration generation.
    ///
    /// Module initialization installs `close()` under the actual well-known
    /// symbol and removes this string-named declaration shim from the prototype.
    #[napi(js_name = "[Symbol.asyncDispose]", ts_return_type = "Promise<void>")]
    pub fn async_dispose(&self, env: Env) -> napi::Result<JsObject> {
        self.close(env)
    }
}

impl Drop for DynamicPluginActivation {
    fn drop(&mut self) {
        self.close_state.begin_close(true);
    }
}

/// Initialize with explicitly resolved dynamic plugins.
///
/// `config` is layered over discovered `plugins.toml` files and may contain
/// statically registered components; dynamic components are activated after
/// that effective base configuration. At least one dynamic plugin is required.
/// Static-only callers should use `initializePlugins`. The returned object owns
/// all loaded libraries and worker processes. Its validation report is available
/// through the `report` property.
#[napi]
pub async fn initialize_with_dynamic_plugins(
    config: Json,
    specs: Json,
) -> napi::Result<DynamicPluginActivation> {
    let config: PluginConfig = serde_json::from_value(config)
        .map_err(|error| napi::Error::from_reason(format!("invalid plugin config: {error}")))?;
    let specs: Vec<NodeDynamicPluginActivationSpec> =
        serde_json::from_value(specs).map_err(|error| {
            napi::Error::from_reason(format!("invalid dynamic plugin specs: {error}"))
        })?;
    let specs = specs.into_iter().map(Into::into).collect::<Vec<_>>();
    let (activation, report) =
        CorePluginHostActivation::activate_with_discovered_config(config, specs)
            .await
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    let report = serde_json::to_value(report)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    Ok(DynamicPluginActivation {
        close_state: Arc::new(DynamicPluginCloseState::new(activation)),
        report,
    })
}

/// Clear the active global plugin configuration.
#[napi]
pub fn clear_plugin_configuration() -> napi::Result<()> {
    clear_plugin_configuration_impl().map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Return the active plugin report or one retained after a teardown failure with runtime diagnostics.
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
