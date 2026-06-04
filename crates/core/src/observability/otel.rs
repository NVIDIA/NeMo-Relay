// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry subscriber support for NeMo Relay.
//!
//! This crate adapts NeMo Relay lifecycle events into OpenTelemetry trace spans:
//!
//! - scope/tool/LLM `Start` events open spans
//! - matching `End` events close spans
//! - `Mark` events become span events on the active parent span when possible
//! - orphan marks fall back to zero-duration spans so they still reach OTLP
//!
//! The public API is intentionally small:
//!
//! - [`OpenTelemetryConfig`] configures the OTLP exporter and resource metadata
//! - [`OpenTelemetrySubscriber`] exposes a NeMo Relay [`EventSubscriberFn`] and
//!   convenience `register` / `deregister` / `force_flush` / `shutdown` methods

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::api::event::Event;
use crate::api::event::ScopeCategory;
use crate::api::runtime::EventSubscriberFn;
use crate::api::scope::ScopeType;
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::error::FlowError;
use chrono::{DateTime, Utc};
use opentelemetry::trace::{
    Span as _, SpanContext, SpanKind, Status, TraceContextExt, Tracer, TracerProvider as _,
};

use crate::json::Json;
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

/// Result type for the OpenTelemetry subscriber crate.
pub type Result<T> = std::result::Result<T, OpenTelemetryError>;

/// Errors produced while configuring or operating the OpenTelemetry subscriber.
#[derive(Debug, thiserror::Error)]
pub enum OpenTelemetryError {
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
    #[error("OpenTelemetry tracer provider error: {0}")]
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

/// Configuration for the OpenTelemetry subscriber.
#[derive(Debug, Clone)]
pub struct OpenTelemetryConfig {
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

impl Default for OpenTelemetryConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            headers: HashMap::new(),
            resource_attributes: HashMap::new(),
            service_name: "nemo-relay".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nemo-relay-otel".to_string(),
            timeout: Duration::from_secs(3),
            transport: OtlpTransport::HttpBinary,
        }
    }
}

impl OpenTelemetryConfig {
    /// Creates an HTTP OTLP config for the given service name.
    pub fn http_binary(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            transport: OtlpTransport::HttpBinary,
            ..Self::default()
        }
    }

    /// Creates a gRPC OTLP config for the given service name.
    pub fn grpc(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            transport: OtlpTransport::Grpc,
            ..Self::default()
        }
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

/// OpenTelemetry-backed NeMo Relay subscriber.
#[derive(Clone)]
pub struct OpenTelemetrySubscriber {
    inner: Arc<Inner>,
}

struct Inner {
    processor: Arc<Mutex<OtelEventProcessor>>,
    subscriber: EventSubscriberFn,
}

impl OpenTelemetrySubscriber {
    /// Builds a subscriber backed by a new OTLP tracer provider.
    pub fn new(config: OpenTelemetryConfig) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        if config.transport == OtlpTransport::Grpc && tokio::runtime::Handle::try_current().is_err()
        {
            return Err(OpenTelemetryError::MissingTokioRuntime);
        }
        #[cfg(target_arch = "wasm32")]
        if config.transport == OtlpTransport::Grpc {
            return Err(OpenTelemetryError::UnsupportedTransport { transport: "gRPC" });
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
        let processor = Arc::new(Mutex::new(OtelEventProcessor::new(
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
            OpenTelemetryError::Provider("the subscriber state lock was poisoned".to_string())
        })?;
        guard.force_flush()
    }

    /// Shuts down the underlying tracer provider.
    ///
    /// Call `deregister(...)` first if the subscriber is still registered with NeMo Relay.
    pub fn shutdown(&self) -> Result<()> {
        flush_subscribers()?;
        let guard = self.inner.processor.lock().map_err(|_| {
            OpenTelemetryError::Provider("the subscriber state lock was poisoned".to_string())
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
                    "OpenTelemetry OTLP/HTTP export failed: {error:?}"
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

fn build_tracer_provider(config: &OpenTelemetryConfig) -> Result<SdkTracerProvider> {
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
                .map_err(|e| OpenTelemetryError::ExporterBuild(e.to_string()))?
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
                .map_err(|e| OpenTelemetryError::ExporterBuild(e.to_string()))?
        }
        #[cfg(target_arch = "wasm32")]
        OtlpTransport::Grpc => {
            return Err(OpenTelemetryError::UnsupportedTransport { transport: "gRPC" });
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

    // Disable per-span attribute caps. Consumers may emit large attribute
    // sets on long-running spans; the OTel SDK default (128) silently drops
    // attributes added last in the span's lifecycle.
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
            OpenTelemetryError::InvalidGrpcHeader {
                key: key.clone(),
                message: e.to_string(),
            }
        })?;
        let metadata_value = MetadataValue::try_from(value.as_str()).map_err(|e| {
            OpenTelemetryError::InvalidGrpcHeader {
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

struct OtelEventProcessor {
    active_spans: HashMap<Uuid, ActiveSpan>,
    provider: SdkTracerProvider,
    tracer: SdkTracer,
}

impl OtelEventProcessor {
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
            .map_err(|e| OpenTelemetryError::Provider(e.to_string()))
    }

    fn shutdown(&self) -> Result<()> {
        self.provider
            .shutdown()
            .map_err(|e| OpenTelemetryError::Provider(e.to_string()))
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

        if let Some(metadata) = event.metadata() && let Some(status_code) = metadata.get("otel.status_code").and_then(Json::as_str) {
            let status = match status_code {
                "OK" => Status::Ok,
                "ERROR" => Status::error(
                    metadata
                        .get("otel.status_message")
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string(),
                ),
                "UNSET" => Status::Unset,
                other => {
                    eprintln!("Unrecognized OTEL status code in event metadata: {other}");
                    Status::Unset
                }
            };
            active_span.span.set_status(status);
        }
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
    let handle_attributes = event.attributes();
    push_serialized(
        &mut attributes,
        "nemo_relay.handle_attributes_json",
        handle_attributes,
    );
    push_serialized(&mut attributes, "nemo_relay.start.data_json", event.data());
    push_serialized(
        &mut attributes,
        "nemo_relay.start.metadata_json",
        event.metadata(),
    );
    push_serialized(
        &mut attributes,
        "nemo_relay.start.input_json",
        event.input(),
    );
    attributes
}

fn end_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    push_serialized(&mut attributes, "nemo_relay.end.data_json", event.data());

    let metadata = event.metadata();
    push_serialized(&mut attributes, "nemo_relay.end.metadata_json", metadata);
    push_serialized(
        &mut attributes,
        "nemo_relay.end.output_json",
        event.output(),
    );

    attributes
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
        attributes.push(KeyValue::new(
            "nemo_relay.model_name",
            model_name.to_string(),
        ));
    }
    if let Some(tool_call_id) = event.tool_call_id() {
        attributes.push(KeyValue::new(
            "nemo_relay.tool_call_id",
            tool_call_id.to_string(),
        ));
    }

    attributes
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
#[path = "../../tests/unit/observability/otel_tests.rs"]
mod tests;
