// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenInference subscriber support for NeMo Relay.
//!
//! This crate adapts NeMo Relay lifecycle events into OpenInference trace spans:
//!
//! - scope/tool/LLM `Start` events open spans
//! - matching `End` events close spans
//! - `Mark` events become span events on the active parent span when possible
//! - orphan marks fall back to zero-duration spans so they still reach OTLP
//!
//! The public API is intentionally small:
//!
//! - [`OpenInferenceConfig`] configures the OTLP exporter and OpenInference metadata
//! - [`OpenInferenceSubscriber`] exposes a NeMo Relay [`EventSubscriberFn`] and
//!   convenience `register` / `deregister` / `force_flush` / `shutdown` methods

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::api::event::{Event, ScopeCategory};
use crate::api::runtime::EventSubscriberFn;
use crate::api::scope::ScopeType;
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::codec::request::{
    AnnotatedLlmRequest, ContentPart, Message, MessageContent, ToolDefinition,
};
use crate::codec::response::{
    AnnotatedLlmResponse, FinishReason, ResponseToolCall, Usage, estimate_cost_for_provider,
};
use crate::error::FlowError;
use crate::json::Json;
use chrono::{DateTime, Utc};
use openinference_semantic_conventions::SpanKind as OpenInferenceSpanKind;
use openinference_semantic_conventions::attributes as oi;
use opentelemetry::trace::{
    Span as _, SpanContext, SpanKind, TraceContextExt, Tracer, TracerProvider as _,
};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, Span};
use serde::Serialize;
use uuid::Uuid;

#[cfg(target_arch = "wasm32")]
use async_trait::async_trait;
#[cfg(target_arch = "wasm32")]
use opentelemetry_http::{
    Bytes, HttpClient, HttpError, Request as HttpRequest, Response as HttpResponse,
};
#[cfg(not(target_arch = "wasm32"))]
use opentelemetry_otlp::WithTonicConfig;
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{JsFuture, spawn_local};
#[cfg(target_arch = "wasm32")]
use web_sys::{Request as WebRequest, RequestInit};

/// Result type for the OpenInference subscriber crate.
pub type Result<T> = std::result::Result<T, OpenInferenceError>;

/// Errors produced while configuring or operating the OpenInference subscriber.
#[derive(Debug, thiserror::Error)]
pub enum OpenInferenceError {
    /// The tonic gRPC exporter requires an active Tokio runtime.
    #[error("the OTLP gRPC exporter requires an active Tokio runtime")]
    MissingTokioRuntime,
    /// The requested transport is not available on this target.
    #[error("the OTLP {transport} transport is not supported on this target")]
    UnsupportedTransport {
        /// Human-readable transport label used in the error message.
        transport: &'static str,
    },
    /// Failed to parse a configured gRPC metadata header.
    #[error("invalid OTLP gRPC header {key:?}: {message}")]
    InvalidGrpcHeader {
        /// Header name that failed to parse.
        key: String,
        /// Parser failure message.
        message: String,
    },
    /// Failed to build the OTLP exporter.
    #[error("failed to build the OTLP exporter: {0}")]
    ExporterBuild(String),
    /// The underlying tracer provider returned an error.
    #[error("OpenInference tracer provider error: {0}")]
    Provider(String),
    /// Registration errors from the core runtime.
    #[error(transparent)]
    Core(#[from] FlowError),
}

/// Supported OTLP trace transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OtlpTransport {
    /// OTLP/HTTP protobuf, typically `http://host:4318/v1/traces`.
    #[default]
    HttpBinary,
    /// OTLP/gRPC, typically `http://host:4317`.
    Grpc,
}

/// Configuration for the OpenInference subscriber.
#[derive(Debug, Clone)]
pub struct OpenInferenceConfig {
    endpoint: Option<String>,
    headers: HashMap<String, String>,
    resource_attributes: HashMap<String, String>,
    service_name: String,
    service_namespace: Option<String>,
    service_version: Option<String>,
    instrumentation_scope: String,
    timeout: Duration,
    transport: OtlpTransport,
}

impl Default for OpenInferenceConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            headers: HashMap::new(),
            resource_attributes: HashMap::new(),
            service_name: "nemo-relay".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nemo-relay-openinference".to_string(),
            timeout: Duration::from_secs(3),
            transport: OtlpTransport::HttpBinary,
        }
    }
}

impl OpenInferenceConfig {
    /// Creates a config with sensible defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects the OTLP transport.
    pub fn with_transport(mut self, transport: OtlpTransport) -> Self {
        self.transport = transport;
        self
    }

    /// Sets the `service.name` resource attribute.
    pub fn with_service_name(mut self, service_name: impl Into<String>) -> Self {
        self.service_name = service_name.into();
        self
    }

    /// Overrides the OTLP endpoint. If unset, exporter defaults and OTEL_* env vars apply.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Adds a header/metadata entry for the exporter.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Adds a resource attribute as a string key/value pair.
    pub fn with_resource_attribute(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.resource_attributes.insert(key.into(), value.into());
        self
    }

    /// Sets the OTLP request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the service namespace resource attribute.
    pub fn with_service_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.service_namespace = Some(namespace.into());
        self
    }

    /// Sets the service version resource attribute.
    pub fn with_service_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = Some(version.into());
        self
    }

    /// Sets the instrumentation scope name used for emitted spans.
    pub fn with_instrumentation_scope(mut self, scope: impl Into<String>) -> Self {
        self.instrumentation_scope = scope.into();
        self
    }
}

/// OpenInference-backed NeMo Relay subscriber.
#[derive(Clone)]
pub struct OpenInferenceSubscriber {
    inner: Arc<Inner>,
}

struct Inner {
    processor: Arc<Mutex<OpenInferenceEventProcessor>>,
    subscriber: EventSubscriberFn,
}

impl OpenInferenceSubscriber {
    /// Builds a subscriber backed by a new OTLP tracer provider.
    pub fn new(config: OpenInferenceConfig) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        if config.transport == OtlpTransport::Grpc && tokio::runtime::Handle::try_current().is_err()
        {
            return Err(OpenInferenceError::MissingTokioRuntime);
        }
        #[cfg(target_arch = "wasm32")]
        if config.transport == OtlpTransport::Grpc {
            return Err(OpenInferenceError::UnsupportedTransport { transport: "gRPC" });
        }

        let provider = build_tracer_provider(&config)?;
        Ok(Self::from_tracer_provider_with_scope(
            provider,
            config.instrumentation_scope,
        ))
    }

    /// Builds a subscriber from an already-configured tracer provider.
    pub fn from_tracer_provider(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
    ) -> Self {
        Self::from_tracer_provider_with_scope(provider, instrumentation_scope.into())
    }

    fn from_tracer_provider_with_scope(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
    ) -> Self {
        let processor = Arc::new(Mutex::new(OpenInferenceEventProcessor::new(
            provider,
            instrumentation_scope,
        )));
        let processor_for_callback = Arc::clone(&processor);
        let subscriber: EventSubscriberFn = Arc::new(move |event: &Event| {
            let Ok(mut guard) = processor_for_callback.lock() else {
                // Observability should not take down the host process if the
                // subscriber state was previously poisoned.
                return;
            };
            guard.process(event);
        });

        Self {
            inner: Arc::new(Inner {
                processor,
                subscriber,
            }),
        }
    }

    /// Returns the raw NeMo Relay subscriber callback for custom registration flows.
    pub fn subscriber(&self) -> EventSubscriberFn {
        Arc::clone(&self.inner.subscriber)
    }

    /// Registers this subscriber globally with the NeMo Relay runtime.
    pub fn register(&self, name: &str) -> Result<()> {
        register_subscriber(name, self.subscriber()).map_err(Into::into)
    }

    /// Deregisters a previously-registered global subscriber by name.
    pub fn deregister(&self, name: &str) -> Result<bool> {
        deregister_subscriber(name).map_err(Into::into)
    }

    /// Flushes finished spans through the underlying tracer provider.
    pub fn force_flush(&self) -> Result<()> {
        flush_subscribers()?;
        let guard = self.inner.processor.lock().map_err(|_| {
            OpenInferenceError::Provider("the subscriber state lock was poisoned".to_string())
        })?;
        guard.force_flush()
    }

    /// Shuts down the underlying tracer provider.
    ///
    /// Call `deregister(...)` first if the subscriber is still registered with NeMo Relay.
    pub fn shutdown(&self) -> Result<()> {
        flush_subscribers()?;
        let guard = self.inner.processor.lock().map_err(|_| {
            OpenInferenceError::Provider("the subscriber state lock was poisoned".to_string())
        })?;
        guard.shutdown()
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, Copy, Default)]
struct WasmHttpClient;

#[cfg(target_arch = "wasm32")]
#[async_trait]
impl HttpClient for WasmHttpClient {
    async fn send_bytes(
        &self,
        request: HttpRequest<Bytes>,
    ) -> std::result::Result<HttpResponse<Bytes>, HttpError> {
        let (parts, body) = request.into_parts();

        let request = {
            let request_url = parts.uri.to_string();
            let init = RequestInit::new();
            init.set_method(parts.method.as_str());
            if !body.is_empty() {
                let body_bytes = js_sys::Uint8Array::from(body.as_ref());
                init.set_body_opt_u8_array(Some(&body_bytes));
            }

            let request =
                WebRequest::new_with_str_and_init(&request_url, &init).map_err(js_error)?;
            let request_headers = request.headers();
            for (name, value) in &parts.headers {
                let value = value
                    .to_str()
                    .map_err(|e| http_error(format!("invalid OTLP HTTP header {name}: {e}")))?;
                request_headers
                    .set(name.as_str(), value)
                    .map_err(js_error)?;
            }
            request
        };

        let fetch_promise = if let Some(window) = web_sys::window() {
            window.fetch_with_request(&request)
        } else {
            let global = js_sys::global();
            let fetch = js_sys::Reflect::get(&global, &JsValue::from_str("fetch"))
                .map_err(js_error)?
                .dyn_into::<js_sys::Function>()
                .map_err(js_error)?;
            fetch.call1(&global, &request).map_err(js_error)?.into()
        };
        // Waiting on the fetch promise from a synchronous wasm call stack can deadlock
        // Node/browser event processing, so dispatch the request asynchronously.
        spawn_local(async move {
            if let Err(error) = JsFuture::from(fetch_promise).await {
                web_sys::console::warn_1(&JsValue::from_str(&format!(
                    "OpenInference OTLP/HTTP export failed: {error:?}"
                )));
            }
        });

        HttpResponse::builder()
            .status(202)
            .body(Bytes::new())
            .map_err(|e| http_error(e.to_string()))
    }
}

#[cfg(target_arch = "wasm32")]
fn js_error(value: JsValue) -> HttpError {
    http_error(
        value
            .as_string()
            .unwrap_or_else(|| format!("JavaScript error: {value:?}")),
    )
}

#[cfg(target_arch = "wasm32")]
fn http_error(message: impl Into<String>) -> HttpError {
    Box::new(std::io::Error::other(message.into()))
}

fn build_tracer_provider(config: &OpenInferenceConfig) -> Result<SdkTracerProvider> {
    let exporter = match config.transport {
        OtlpTransport::HttpBinary => {
            #[cfg(not(target_arch = "wasm32"))]
            install_rustls_crypto_provider();
            let mut builder = SpanExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_timeout(config.timeout);
            if let Some(endpoint) = &config.endpoint {
                builder = builder.with_endpoint(endpoint.clone());
            }
            if !config.headers.is_empty() {
                builder = builder.with_headers(config.headers.clone());
            }
            #[cfg(target_arch = "wasm32")]
            {
                builder = builder.with_http_client(WasmHttpClient);
            }
            builder
                .build()
                .map_err(|e| OpenInferenceError::ExporterBuild(e.to_string()))?
        }
        #[cfg(not(target_arch = "wasm32"))]
        OtlpTransport::Grpc => {
            let mut builder = SpanExporter::builder()
                .with_tonic()
                .with_protocol(Protocol::Grpc)
                .with_timeout(config.timeout);
            if let Some(endpoint) = &config.endpoint {
                builder = builder.with_endpoint(endpoint.clone());
            }
            if !config.headers.is_empty() {
                builder = builder.with_metadata(build_grpc_metadata(&config.headers)?);
            }
            builder
                .build()
                .map_err(|e| OpenInferenceError::ExporterBuild(e.to_string()))?
        }
        #[cfg(target_arch = "wasm32")]
        OtlpTransport::Grpc => {
            return Err(OpenInferenceError::UnsupportedTransport { transport: "gRPC" });
        }
    };

    let mut resource_attributes = vec![KeyValue::new("service.name", config.service_name.clone())];
    if let Some(service_namespace) = &config.service_namespace {
        resource_attributes.push(KeyValue::new(
            "service.namespace",
            service_namespace.clone(),
        ));
    }
    if let Some(service_version) = &config.service_version {
        resource_attributes.push(KeyValue::new("service.version", service_version.clone()));
    }
    for (key, value) in &config.resource_attributes {
        resource_attributes.push(KeyValue::new(key.clone(), value.clone()));
    }

    // Disable per-span attribute caps. OpenInference emits many flat
    // `llm.input_messages.*` attributes on long conversations; the OTel SDK
    // default (128) silently drops attributes added last in the span's
    // lifecycle, notably `llm.token_count.*` emitted at span end.
    let builder = SdkTracerProvider::builder()
        .with_resource(
            Resource::builder_empty()
                .with_attributes(resource_attributes)
                .build(),
        )
        .with_max_attributes_per_span(u32::MAX)
        .with_max_attributes_per_event(u32::MAX);

    #[cfg(not(target_arch = "wasm32"))]
    {
        if Handle::try_current().is_ok() {
            Ok(builder.with_batch_exporter(exporter).build())
        } else {
            Ok(builder.with_simple_exporter(exporter).build())
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(builder.with_simple_exporter(exporter).build())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn install_rustls_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

#[cfg(not(target_arch = "wasm32"))]
fn build_grpc_metadata(headers: &HashMap<String, String>) -> Result<MetadataMap> {
    let mut metadata = MetadataMap::new();
    for (key, value) in headers {
        let metadata_key = MetadataKey::from_bytes(key.as_bytes()).map_err(|e| {
            OpenInferenceError::InvalidGrpcHeader {
                key: key.clone(),
                message: e.to_string(),
            }
        })?;
        let metadata_value = MetadataValue::try_from(value.as_str()).map_err(|e| {
            OpenInferenceError::InvalidGrpcHeader {
                key: key.clone(),
                message: e.to_string(),
            }
        })?;
        metadata.insert(metadata_key, metadata_value);
    }
    Ok(metadata)
}

struct ActiveSpan {
    span: Span,
    span_context: SpanContext,
}

struct OpenInferenceEventProcessor {
    active_spans: HashMap<Uuid, ActiveSpan>,
    provider: SdkTracerProvider,
    tracer: SdkTracer,
}

impl OpenInferenceEventProcessor {
    fn new(provider: SdkTracerProvider, instrumentation_scope: String) -> Self {
        let tracer = provider.tracer(instrumentation_scope);
        Self {
            active_spans: HashMap::new(),
            provider,
            tracer,
        }
    }

    fn process(&mut self, event: &Event) {
        match event.scope_category() {
            Some(ScopeCategory::Start) => self.process_start(event),
            Some(ScopeCategory::End) => self.process_end(event),
            None => self.process_mark(event),
        }
    }

    fn force_flush(&self) -> Result<()> {
        self.provider
            .force_flush()
            .map_err(|e| OpenInferenceError::Provider(e.to_string()))
    }

    fn shutdown(&self) -> Result<()> {
        self.provider
            .shutdown()
            .map_err(|e| OpenInferenceError::Provider(e.to_string()))
    }

    fn process_start(&mut self, event: &Event) {
        let mut span = self
            .tracer
            .span_builder(span_name(event))
            .with_kind(span_kind(event))
            .with_start_time(to_system_time(*event.timestamp()))
            .start_with_context(&self.tracer, &self.parent_context(event));
        span.set_attributes(start_attributes(event));
        let span_context = local_parent_span_context(span.span_context());
        self.active_spans
            .insert(event.uuid(), ActiveSpan { span, span_context });
    }

    fn process_end(&mut self, event: &Event) {
        let Some(mut active_span) = self.active_spans.remove(&event.uuid()) else {
            return;
        };
        active_span.span.set_attributes(end_attributes(event));
        active_span
            .span
            .end_with_timestamp(to_system_time(*event.timestamp()));
    }

    fn process_mark(&mut self, event: &Event) {
        let mark_name = event.name().to_string();
        let timestamp = to_system_time(*event.timestamp());
        let attributes = mark_attributes(event);

        if let Some(parent_span) = self.find_parent_span_mut(event) {
            parent_span
                .span
                .add_event_with_timestamp(mark_name, timestamp, attributes);
            return;
        }

        let mut span = self
            .tracer
            .span_builder(format!("mark:{mark_name}"))
            .with_kind(SpanKind::Internal)
            .with_start_time(timestamp)
            .start_with_context(&self.tracer, &self.parent_context(event));
        let mut span_attributes = attributes;
        span_attributes.push(KeyValue::new(
            oi::OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKind::Chain,
        ));
        span_attributes.push(KeyValue::new("nemo_relay.mark.orphan", true));
        span.set_attributes(span_attributes);
        span.end_with_timestamp(timestamp);
    }

    fn parent_context(&self, event: &Event) -> Context {
        self.find_parent_span(event)
            .map(|active_span| {
                Context::new().with_remote_span_context(active_span.span_context.clone())
            })
            .unwrap_or_default()
    }

    fn parent_span_uuid(&self, event: &Event) -> Option<Uuid> {
        event
            .parent_uuid()
            .filter(|uuid| self.active_spans.contains_key(uuid))
    }

    fn find_parent_span(&self, event: &Event) -> Option<&ActiveSpan> {
        self.parent_span_uuid(event)
            .and_then(|uuid| self.active_spans.get(&uuid))
    }

    fn find_parent_span_mut(&mut self, event: &Event) -> Option<&mut ActiveSpan> {
        self.parent_span_uuid(event)
            .and_then(|uuid| self.active_spans.get_mut(&uuid))
    }
}

fn span_kind(event: &Event) -> SpanKind {
    match semantic_scope_type(event) {
        Some(ScopeType::Llm) => SpanKind::Client,
        Some(
            ScopeType::Tool | ScopeType::Retriever | ScopeType::Embedder | ScopeType::Reranker,
        ) => SpanKind::Client,
        _ => SpanKind::Internal,
    }
}

fn span_name(event: &Event) -> String {
    event.name().to_string()
}

fn semantic_scope_type(event: &Event) -> Option<ScopeType> {
    event.scope_type()
}

fn scope_type_name(scope_type: Option<ScopeType>) -> &'static str {
    match scope_type {
        Some(ScopeType::Agent) => "agent",
        Some(ScopeType::Function) => "function",
        Some(ScopeType::Tool) => "tool",
        Some(ScopeType::Llm) => "llm",
        Some(ScopeType::Retriever) => "retriever",
        Some(ScopeType::Embedder) => "embedder",
        Some(ScopeType::Reranker) => "reranker",
        Some(ScopeType::Guardrail) => "guardrail",
        Some(ScopeType::Evaluator) => "evaluator",
        Some(ScopeType::Custom) => "custom",
        Some(ScopeType::Unknown) | None => "unknown",
    }
}

fn start_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = common_attributes(event);
    let is_llm = event
        .category()
        .is_some_and(|category| category.as_str() == "llm");
    if is_llm {
        // Final span metadata should reflect the completed event, especially for mixed-fidelity
        // Hermes flows where the request can be exact but the terminal error is lossy.
        attributes.retain(|attribute| attribute.key.as_str() != oi::METADATA.as_str());
    }
    let handle_attributes = event.attributes();
    if handle_attributes.is_some_and(|attributes| !attributes.is_empty()) {
        push_serialized(
            &mut attributes,
            "nemo_relay.handle_attributes_json",
            handle_attributes,
        );
    }
    if event
        .category()
        .is_none_or(|category| category.as_str() != "llm")
    {
        push_serialized(
            &mut attributes,
            "nemo_relay.start.input_json",
            event.input(),
        );
    }
    if event
        .category()
        .is_some_and(|category| category.as_str() == "tool")
    {
        attributes.push(KeyValue::new(oi::tool::NAME, event.name().to_string()));
        attributes.push(KeyValue::new(
            oi::tool_call::function::NAME,
            event.name().to_string(),
        ));
    }

    if let Some((input, mime_type)) = openinference_input_value(event) {
        attributes.push(KeyValue::new(oi::input::VALUE, input.clone()));
        attributes.push(KeyValue::new(oi::input::MIME_TYPE, mime_type));

        if event
            .category()
            .is_some_and(|category| category.as_str() == "tool")
        {
            attributes.push(KeyValue::new(oi::tool::PARAMETERS, input.clone()));
            attributes.push(KeyValue::new(oi::tool_call::function::ARGUMENTS, input));
        }
    }
    if is_llm {
        push_llm_request_attributes(&mut attributes, event);
    }
    attributes
}

fn end_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    let is_llm = event
        .category()
        .is_some_and(|category| category.as_str() == "llm");

    if let Some(metadata) = event.metadata().and_then(to_json_string) {
        attributes.push(KeyValue::new(oi::METADATA, metadata));
    }

    push_serialized(
        &mut attributes,
        "nemo_relay.end.output_json",
        event.output(),
    );
    if let Some((output, mime_type)) = openinference_output_value(event) {
        attributes.push(KeyValue::new(oi::output::VALUE, output));
        attributes.push(KeyValue::new(oi::output::MIME_TYPE, mime_type));
    }
    let fallback_usage = if is_llm {
        usage_from_manual_llm_output(event.output())
    } else {
        None
    };
    let usage = event
        .annotated_response()
        .and_then(|response| response.usage.as_ref())
        .or(fallback_usage.as_ref());
    if is_llm && let Some(usage) = usage {
        if let Some(v) = usage.prompt_tokens {
            attributes.push(KeyValue::new(oi::llm::token_count::PROMPT, v as i64));
        }
        if let Some(v) = usage.completion_tokens {
            attributes.push(KeyValue::new(oi::llm::token_count::COMPLETION, v as i64));
        }
        if let Some(v) = usage.total_tokens {
            attributes.push(KeyValue::new(oi::llm::token_count::TOTAL, v as i64));
        }
        if let Some(v) = usage.cache_read_tokens {
            attributes.push(KeyValue::new(
                oi::llm::token_count::prompt_details::CACHE_READ,
                v as i64,
            ));
        }
        if let Some(v) = usage.cache_write_tokens {
            attributes.push(KeyValue::new(
                oi::llm::token_count::prompt_details::CACHE_WRITE,
                v as i64,
            ));
        }
    }
    if is_llm && let Some(cost_total) = cost_total_from_llm_event(event, fallback_usage.as_ref()) {
        attributes.push(KeyValue::new(oi::llm::cost::TOTAL, cost_total));
    }
    if is_llm {
        push_llm_response_attributes(&mut attributes, event);
    }
    attributes
}

fn push_llm_request_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(request) = event.annotated_request() {
        push_annotated_request_attributes(attributes, request);
        return;
    }

    let Some(input) = event.input().and_then(replay_llm_payload) else {
        return;
    };
    if let Some(provider) = input.get("provider").and_then(Json::as_str) {
        attributes.push(KeyValue::new(oi::llm::PROVIDER, provider.to_string()));
    }
    if let Some(system) = input.get("systemPrompt").and_then(display_text_from_json) {
        attributes.push(KeyValue::new(oi::llm::SYSTEM, system));
    }
    push_replay_input_messages(attributes, input);
}

fn push_llm_response_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(response) = event.annotated_response() {
        push_annotated_response_attributes(attributes, response);
        return;
    }

    let Some(output) = event.output().and_then(replay_llm_response) else {
        return;
    };
    push_replay_response_attributes(attributes, output);
}

fn push_annotated_request_attributes(
    attributes: &mut Vec<KeyValue>,
    request: &AnnotatedLlmRequest,
) {
    if let Some(system) = request.system_prompt() {
        attributes.push(KeyValue::new(oi::llm::SYSTEM, system.to_string()));
    }
    if let Some(params) = request.params.as_ref().and_then(to_json_string) {
        attributes.push(KeyValue::new(oi::llm::INVOCATION_PARAMETERS, params));
    }
    push_annotated_input_messages(attributes, &request.messages);
    if let Some(tools) = request.tools.as_deref() {
        push_annotated_tools(attributes, tools);
    }
}

fn push_annotated_response_attributes(
    attributes: &mut Vec<KeyValue>,
    response: &AnnotatedLlmResponse,
) {
    if let Some(reason) = response.finish_reason.as_ref() {
        attributes.push(KeyValue::new(
            "llm.finish_reason",
            finish_reason_value(reason),
        ));
    }

    let has_message = response.message.is_some()
        || response
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty());
    if has_message {
        attributes.push(KeyValue::new(
            "llm.output_messages.0.message.role",
            "assistant",
        ));
    }
    if let Some(content) = response.message.as_ref().and_then(message_content_text) {
        attributes.push(KeyValue::new(
            "llm.output_messages.0.message.content",
            content,
        ));
    }
    if let Some(tool_calls) = response.tool_calls.as_deref() {
        push_response_tool_calls(attributes, 0, tool_calls);
    }
}

fn push_annotated_input_messages(attributes: &mut Vec<KeyValue>, messages: &[Message]) {
    for (index, message) in messages.iter().enumerate() {
        let (role, content) = match message {
            Message::System { content, .. } => ("system", Some(content)),
            Message::User { content, .. } => ("user", Some(content)),
            Message::Assistant { content, .. } => ("assistant", content.as_ref()),
            Message::Tool { content, .. } => ("tool", Some(content)),
        };
        push_message_role(attributes, "llm.input_messages", index, role);
        if let Some(content) = content {
            push_message_text_content(attributes, "llm.input_messages", index, content);
        }
    }
}

fn push_annotated_tools(attributes: &mut Vec<KeyValue>, tools: &[ToolDefinition]) {
    for (index, tool) in tools.iter().enumerate() {
        if let Some(json) = to_json_string(tool) {
            attributes.push(KeyValue::new(
                format!("llm.tools.{index}.tool.json_schema"),
                json,
            ));
        }
    }
}

fn push_response_tool_calls(
    attributes: &mut Vec<KeyValue>,
    message_index: usize,
    tool_calls: &[ResponseToolCall],
) {
    for (call_index, tool_call) in tool_calls.iter().enumerate() {
        push_output_tool_call(
            attributes,
            message_index,
            call_index,
            Some(tool_call.id.as_str()),
            Some(tool_call.name.as_str()),
            to_json_string(&tool_call.arguments),
        );
    }
}

fn push_message_role(
    attributes: &mut Vec<KeyValue>,
    prefix: &'static str,
    index: usize,
    role: &str,
) {
    attributes.push(KeyValue::new(
        format!("{prefix}.{index}.message.role"),
        role.to_string(),
    ));
}

fn push_message_text_content(
    attributes: &mut Vec<KeyValue>,
    prefix: &'static str,
    index: usize,
    content: &MessageContent,
) {
    if let Some(text) = message_content_text(content) {
        attributes.push(KeyValue::new(
            format!("{prefix}.{index}.message.content"),
            text,
        ));
    }
}

fn message_content_text(content: &MessageContent) -> Option<String> {
    match content {
        MessageContent::Text(text) => display_text_from_string(text),
        MessageContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
            if text.is_empty() { None } else { Some(text) }
        }
    }
}

fn replay_llm_payload(input: &Json) -> Option<&Json> {
    let content = input.as_object().and_then(|object| object.get("content"))?;
    let content_object = content.as_object()?;
    is_openclaw_replay_payload(content_object).then_some(content)
}

fn replay_llm_response(output: &Json) -> Option<&Json> {
    output
        .as_object()
        .and_then(|object| object.get("openclaw"))
        .and_then(Json::as_object)
        .map(|_| output)
}

fn is_openclaw_replay_payload(content: &serde_json::Map<String, Json>) -> bool {
    content
        .get("source")
        .and_then(Json::as_str)
        .is_some_and(|source| source.starts_with("openclaw."))
        || content.contains_key("placeholderRequest")
}

fn push_replay_input_messages(attributes: &mut Vec<KeyValue>, input: &Json) {
    if let Some(messages) = input.get("messages").and_then(Json::as_array) {
        for (index, message) in messages.iter().enumerate() {
            push_replay_input_message(attributes, index, message);
        }
        return;
    }
    if let Some(prompt) = input.get("prompt").and_then(display_text_from_json) {
        push_message_role(attributes, "llm.input_messages", 0, "user");
        attributes.push(KeyValue::new(
            "llm.input_messages.0.message.content",
            prompt,
        ));
    }
}

fn push_replay_input_message(attributes: &mut Vec<KeyValue>, index: usize, message: &Json) {
    let Some(object) = message.as_object() else {
        return;
    };
    if !object.contains_key("role") && !object.contains_key("content") {
        return;
    }
    let role = object.get("role").and_then(Json::as_str).unwrap_or("user");
    push_message_role(attributes, "llm.input_messages", index, role);
    if let Some(text) = object.get("content").and_then(display_text_from_json) {
        attributes.push(KeyValue::new(
            format!("llm.input_messages.{index}.message.content"),
            text,
        ));
    }
}

fn push_replay_response_attributes(attributes: &mut Vec<KeyValue>, output: &Json) {
    if output.get("role").is_none()
        && output.get("content").is_none()
        && output.get("tool_calls").is_none()
    {
        return;
    }
    let role = output
        .get("role")
        .and_then(Json::as_str)
        .unwrap_or("assistant");
    push_message_role(attributes, "llm.output_messages", 0, role);
    if let Some(content) = output.get("content").and_then(display_text_from_json) {
        attributes.push(KeyValue::new(
            "llm.output_messages.0.message.content",
            content,
        ));
    }
    if let Some(tool_calls) = output.get("tool_calls").and_then(Json::as_array) {
        push_raw_output_tool_calls(attributes, 0, tool_calls);
    }
}

fn push_raw_output_tool_calls(
    attributes: &mut Vec<KeyValue>,
    message_index: usize,
    tool_calls: &[Json],
) {
    for (call_index, tool_call) in tool_calls.iter().enumerate() {
        push_output_tool_call(
            attributes,
            message_index,
            call_index,
            tool_call.get("id").and_then(Json::as_str),
            raw_tool_call_name(tool_call),
            raw_tool_call_arguments(tool_call).and_then(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| to_json_string(value))
            }),
        );
    }
}

fn raw_tool_call_name(tool_call: &Json) -> Option<&str> {
    tool_call
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Json::as_str)
        .or_else(|| tool_call.get("name").and_then(Json::as_str))
        .or_else(|| tool_call.get("toolName").and_then(Json::as_str))
}

fn raw_tool_call_arguments(tool_call: &Json) -> Option<&Json> {
    tool_call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .or_else(|| tool_call.get("arguments"))
        .or_else(|| tool_call.get("input"))
}

fn push_output_tool_call(
    attributes: &mut Vec<KeyValue>,
    message_index: usize,
    call_index: usize,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<String>,
) {
    if let Some(id) = id {
        attributes.push(KeyValue::new(
            format!(
                "llm.output_messages.{message_index}.message.tool_calls.{call_index}.tool_call.id"
            ),
            id.to_string(),
        ));
    }
    if let Some(name) = name {
        attributes.push(KeyValue::new(
            format!(
                "llm.output_messages.{message_index}.message.tool_calls.{call_index}.tool_call.function.name"
            ),
            name.to_string(),
        ));
    }
    if let Some(arguments) = arguments {
        attributes.push(KeyValue::new(
            format!(
                "llm.output_messages.{message_index}.message.tool_calls.{call_index}.tool_call.function.arguments"
            ),
            arguments,
        ));
    }
}

fn finish_reason_value(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Complete => "complete".to_string(),
        FinishReason::Length => "length".to_string(),
        FinishReason::ToolUse => "tool_use".to_string(),
        FinishReason::ContentFilter => "content_filter".to_string(),
        FinishReason::Unknown(reason) => reason.clone(),
    }
}

fn cost_total_from_manual_llm_output(output: Option<&Json>) -> Option<f64> {
    let object = output?.as_object()?;
    let usage = object.get("usage").and_then(Json::as_object);
    let token_usage = object.get("token_usage").and_then(Json::as_object);
    usage
        .and_then(cost_total_from_usage)
        .or_else(|| token_usage.and_then(cost_total_from_usage))
}

fn cost_total_from_llm_event(event: &Event, fallback_usage: Option<&Usage>) -> Option<f64> {
    cost_total_from_manual_llm_output(event.output())
        .or_else(|| {
            let response = event.annotated_response()?;
            let usage = response.usage.as_ref()?;
            usage.cost.as_ref()?.total_for_currency("USD")
        })
        .or_else(|| {
            let response = event.annotated_response()?;
            let usage = response.usage.as_ref()?;
            let model_name = response.model.as_deref().or_else(|| event.model_name())?;
            estimate_cost_for_provider(Some(event.name()), model_name, usage)
                .and_then(|cost| cost.total_for_currency("USD"))
        })
        .or_else(|| {
            let usage = fallback_usage?;
            let model_name = event
                .model_name()
                .or_else(|| model_name_from_manual_llm_output(event.output()))?;
            estimate_cost_for_provider(Some(event.name()), model_name, usage)
                .and_then(|cost| cost.total_for_currency("USD"))
        })
}

fn model_name_from_manual_llm_output(output: Option<&Json>) -> Option<&str> {
    output?.as_object()?.get("model").and_then(Json::as_str)
}

fn cost_total_from_usage(usage: &serde_json::Map<String, Json>) -> Option<f64> {
    usage
        .get("cost_usd")
        .and_then(Json::as_f64)
        .or_else(|| usage.get("cost")?.as_object()?.get("total")?.as_f64())
}

fn usage_from_manual_llm_output(output: Option<&Json>) -> Option<Usage> {
    let object = output?.as_object()?;
    let usage = object.get("usage").and_then(Json::as_object);
    let token_usage = object.get("token_usage").and_then(Json::as_object);
    if usage.is_none() && token_usage.is_none() {
        return None;
    }

    let prompt_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &["prompt_tokens", "input_tokens", "inputTokens", "input"],
    );
    let completion_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &[
            "completion_tokens",
            "output_tokens",
            "completionTokens",
            "outputTokens",
            "output",
        ],
    );
    let reported_total_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &["total_tokens", "totalTokens", "total"],
    );
    let cache_read_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &[
            "cache_read_tokens",
            "cached_tokens",
            "cache_read_input_tokens",
            "cacheReadTokens",
            "cachedTokens",
            "cacheReadInputTokens",
            "cacheRead",
        ],
    )
    .or_else(|| {
        first_nested_u64_from_manual_usage(
            usage,
            token_usage,
            "input_tokens_details",
            "cached_tokens",
        )
    })
    .or_else(|| {
        first_nested_u64_from_manual_usage(
            usage,
            token_usage,
            "prompt_tokens_details",
            "cached_tokens",
        )
    });
    let cache_write_tokens = first_u64_from_manual_usage(
        usage,
        token_usage,
        &[
            "cache_write_tokens",
            "cache_creation_input_tokens",
            "cacheWriteTokens",
            "cacheCreationInputTokens",
            "cacheWrite",
        ],
    );

    if prompt_tokens.is_none()
        && completion_tokens.is_none()
        && reported_total_tokens.is_none()
        && cache_read_tokens.is_none()
        && cache_write_tokens.is_none()
    {
        return None;
    }
    let total_tokens =
        normalize_total_tokens(reported_total_tokens, prompt_tokens, completion_tokens);

    Some(Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cache_read_tokens,
        cache_write_tokens,
        cost: None,
    })
}

fn normalize_total_tokens(
    total_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
) -> Option<u64> {
    let total_tokens = total_tokens?;
    let minimum_total = prompt_tokens
        .unwrap_or(0)
        .saturating_add(completion_tokens.unwrap_or(0));
    if minimum_total == 0 || total_tokens >= minimum_total {
        Some(total_tokens)
    } else {
        None
    }
}

fn first_u64_from_manual_usage(
    usage: Option<&serde_json::Map<String, Json>>,
    token_usage: Option<&serde_json::Map<String, Json>>,
    keys: &[&str],
) -> Option<u64> {
    usage
        .and_then(|value| first_u64(value, keys))
        .or_else(|| token_usage.and_then(|value| first_u64(value, keys)))
}

fn first_nested_u64_from_manual_usage(
    usage: Option<&serde_json::Map<String, Json>>,
    token_usage: Option<&serde_json::Map<String, Json>>,
    parent_key: &str,
    child_key: &str,
) -> Option<u64> {
    usage
        .and_then(|value| nested_u64(value, parent_key, child_key))
        .or_else(|| token_usage.and_then(|value| nested_u64(value, parent_key, child_key)))
}

fn nested_u64(
    usage: &serde_json::Map<String, Json>,
    parent_key: &str,
    child_key: &str,
) -> Option<u64> {
    usage
        .get(parent_key)
        .and_then(Json::as_object)
        .and_then(|details| details.get(child_key))
        .and_then(Json::as_u64)
}

fn first_u64(usage: &serde_json::Map<String, Json>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Json::as_u64))
}

fn mark_attributes(event: &Event) -> Vec<KeyValue> {
    let handle_attributes = event.attributes();
    let mut attributes = vec![
        KeyValue::new("nemo_relay.mark.uuid", event.uuid().to_string()),
        KeyValue::new(
            "nemo_relay.mark.parent_uuid",
            event
                .parent_uuid()
                .map(|uuid| uuid.to_string())
                .unwrap_or_default(),
        ),
    ];
    push_serialized(
        &mut attributes,
        "nemo_relay.mark.attributes_json",
        handle_attributes,
    );
    push_serialized(&mut attributes, "nemo_relay.mark.data_json", event.data());
    push_serialized(
        &mut attributes,
        "nemo_relay.mark.metadata_json",
        event.metadata(),
    );
    attributes
}

fn common_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = vec![
        KeyValue::new(
            oi::OPENINFERENCE_SPAN_KIND,
            openinference_span_kind(semantic_scope_type(event)),
        ),
        KeyValue::new("nemo_relay.uuid", event.uuid().to_string()),
        KeyValue::new(
            "nemo_relay.parent_uuid",
            event
                .parent_uuid()
                .map(|uuid| uuid.to_string())
                .unwrap_or_default(),
        ),
        KeyValue::new(
            "nemo_relay.scope_type",
            scope_type_name(semantic_scope_type(event)),
        ),
    ];

    if let Some(model_name) = event.model_name() {
        attributes.push(KeyValue::new(oi::llm::MODEL_NAME, model_name.to_string()));
    }
    if let Some(tool_call_id) = event.tool_call_id() {
        attributes.push(KeyValue::new(oi::tool_call::ID, tool_call_id.to_string()));
    }
    if let Some(metadata) = event.metadata().and_then(to_json_string) {
        attributes.push(KeyValue::new(oi::METADATA, metadata));
    }

    attributes
}

fn openinference_span_kind(scope_type: Option<ScopeType>) -> OpenInferenceSpanKind {
    match scope_type {
        Some(ScopeType::Agent) => OpenInferenceSpanKind::Agent,
        Some(ScopeType::Tool) => OpenInferenceSpanKind::Tool,
        Some(ScopeType::Llm) => OpenInferenceSpanKind::Llm,
        Some(ScopeType::Retriever) => OpenInferenceSpanKind::Retriever,
        Some(ScopeType::Embedder) => OpenInferenceSpanKind::Embedding,
        Some(ScopeType::Reranker) => OpenInferenceSpanKind::Reranker,
        Some(ScopeType::Guardrail) => OpenInferenceSpanKind::Guardrail,
        Some(ScopeType::Evaluator) => OpenInferenceSpanKind::Evaluator,
        Some(ScopeType::Function | ScopeType::Custom | ScopeType::Unknown) | None => {
            OpenInferenceSpanKind::Chain
        }
    }
}

fn push_serialized<T: Serialize + ?Sized>(
    attributes: &mut Vec<KeyValue>,
    key: &'static str,
    value: Option<&T>,
) {
    if let Some(value) = value
        && let Ok(json) = serde_json::to_string(value)
    {
        attributes.push(KeyValue::new(key, json));
    }
}

fn openinference_input_value(event: &Event) -> Option<(String, &'static str)> {
    let input = event.input()?;

    if event
        .category()
        .is_some_and(|category| category.as_str() == "llm")
    {
        return llm_input_display_value(input)
            .map(|display| (display, "text/plain"))
            .or_else(|| sanitized_llm_input_json(input).map(|json| (json, "application/json")));
    }

    to_json_string(input).map(|json| (json, "application/json"))
}

fn openinference_output_value(event: &Event) -> Option<(String, &'static str)> {
    let output = event.output()?;
    display_text_from_json(output)
        .map(|display| (display, "text/plain"))
        .or_else(|| to_json_string(output).map(|json| (json, "application/json")))
}

fn llm_input_display_value(input: &Json) -> Option<String> {
    let content = match input {
        Json::Object(object) => object.get("content").unwrap_or(input),
        _ => input,
    };

    content
        .get("messages")
        .and_then(display_text_from_messages)
        .or_else(|| display_text_from_json(content))
}

fn sanitized_llm_input_json(input: &Json) -> Option<String> {
    match input {
        Json::Object(object) => {
            let mut sanitized = object.clone();
            sanitized.remove("headers");
            to_json_string(&Json::Object(sanitized))
        }
        _ => to_json_string(input),
    }
}

fn display_text_from_json(value: &Json) -> Option<String> {
    match value {
        Json::String(text) => display_text_from_string(text),
        Json::Object(object) => {
            for key in ["content", "summary", "message", "text", "prompt"] {
                if let Some(display) = object.get(key).and_then(display_text_from_json) {
                    return Some(display);
                }
            }
            object
                .get("output")
                .and_then(display_text_from_openai_responses_output)
                .or_else(|| {
                    object
                        .get("choices")
                        .and_then(display_text_from_chat_choices)
                })
                .or_else(|| {
                    object
                        .get("tool_calls")
                        .and_then(display_text_from_tool_calls)
                })
        }
        Json::Array(items) => display_text_from_content_blocks(items),
        _ => None,
    }
}

fn display_text_from_openai_responses_output(value: &Json) -> Option<String> {
    let items = value.as_array()?;
    let mut entries = Vec::new();
    let mut tool_names = Vec::new();
    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        match object.get("type").and_then(Json::as_str) {
            Some("message") => {
                if let Some(content) = object
                    .get("content")
                    .and_then(display_text_from_openai_responses_content)
                {
                    entries.push(content);
                }
            }
            Some("function_call") => {
                if let Some(name) = object.get("name").and_then(Json::as_str) {
                    tool_names.push(name.to_string());
                }
            }
            _ => {}
        }
    }
    if !tool_names.is_empty() {
        entries.push(format!("Requested tools: {}", tool_names.join(", ")));
    }
    let text = entries.join("\n").trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn display_text_from_openai_responses_content(value: &Json) -> Option<String> {
    let content = value.as_array()?;
    let text = content
        .iter()
        .filter_map(|part| {
            let object = part.as_object()?;
            match object.get("type").and_then(Json::as_str) {
                Some("output_text" | "text") => object.get("text").and_then(Json::as_str),
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn display_text_from_messages(value: &Json) -> Option<String> {
    let messages = value.as_array()?;
    let text = messages
        .iter()
        .filter_map(display_text_from_message)
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn display_text_from_message(value: &Json) -> Option<String> {
    let role = value
        .get("role")
        .and_then(Json::as_str)
        .unwrap_or("message");
    if role == "tool" {
        return Some("tool: Tool result omitted".to_string());
    }
    let display = value
        .get("content")
        .and_then(display_text_from_json)
        .or_else(|| {
            value
                .get("tool_calls")
                .and_then(display_text_from_tool_calls)
        })?;
    Some(format!("{role}: {display}"))
}

fn display_text_from_string(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(parsed) = serde_json::from_str::<Json>(trimmed)
        && let Some(display) = display_text_from_json(&parsed)
    {
        return Some(display);
    }
    Some(trimmed.to_string())
}

fn display_text_from_chat_choices(value: &Json) -> Option<String> {
    let choices = value.as_array()?;
    for choice in choices {
        let Some(message) = choice.get("message") else {
            continue;
        };
        let content = message.get("content").and_then(display_text_from_json);
        let tool_calls = message
            .get("tool_calls")
            .and_then(display_text_from_tool_calls);
        match (content, tool_calls) {
            (Some(content), Some(tool_calls)) => return Some(format!("{content}\n{tool_calls}")),
            (Some(content), None) => return Some(content),
            (None, Some(tool_calls)) => return Some(tool_calls),
            (None, None) => {}
        }
    }
    None
}

fn display_text_from_content_blocks(items: &[Json]) -> Option<String> {
    let mut entries = items
        .iter()
        .filter_map(content_block_display_text)
        .collect::<Vec<_>>();
    let tool_calls = items.iter().filter_map(tool_call_name).collect::<Vec<_>>();
    if !tool_calls.is_empty() {
        entries.push(format!("Requested tools: {}", tool_calls.join(", ")));
    }
    let text = entries
        .into_iter()
        .filter(|item| !item.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn content_block_display_text(item: &Json) -> Option<String> {
    if let Some(text) = item.as_str() {
        return Some(text.to_string());
    }
    if item.get("stripped").and_then(Json::as_bool) == Some(true) {
        return None;
    }
    if let Some("thinking" | "reasoning" | "toolResult" | "tool_result") =
        item.get("type").and_then(Json::as_str)
    {
        return None;
    }
    item.get("text").and_then(Json::as_str).map(str::to_string)
}

fn display_text_from_tool_calls(value: &Json) -> Option<String> {
    let calls = value.as_array()?;
    let names = calls.iter().filter_map(tool_call_name).collect::<Vec<_>>();
    if names.is_empty() {
        None
    } else {
        Some(format!("Requested tools: {}", names.join(", ")))
    }
}

fn tool_call_name(value: &Json) -> Option<String> {
    value
        .get("name")
        .and_then(Json::as_str)
        .or_else(|| value.get("toolName").and_then(Json::as_str))
        .or_else(|| {
            value
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Json::as_str)
        })
        .map(str::to_string)
}

fn to_json_string<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_string(value).ok()
}

fn local_parent_span_context(span_context: &SpanContext) -> SpanContext {
    SpanContext::new(
        span_context.trace_id(),
        span_context.span_id(),
        span_context.trace_flags(),
        false,
        span_context.trace_state().clone(),
    )
}

fn to_system_time(timestamp: DateTime<Utc>) -> SystemTime {
    let seconds = timestamp.timestamp();
    let nanos = timestamp.timestamp_subsec_nanos();
    if seconds >= 0 {
        UNIX_EPOCH + Duration::new(seconds as u64, nanos)
    } else if nanos == 0 {
        UNIX_EPOCH - Duration::new(seconds.unsigned_abs(), 0)
    } else {
        UNIX_EPOCH - Duration::new(seconds.unsigned_abs() - 1, 1_000_000_000 - nanos)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/openinference_tests.rs"]
mod tests;
