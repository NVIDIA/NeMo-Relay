// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry subscriber support for Nexus.
//!
//! This crate adapts Nexus lifecycle events into OpenTelemetry trace spans:
//!
//! - scope/tool/LLM `Start` events open spans
//! - matching `End` events close spans
//! - `Mark` events become span events on the active parent span when possible
//! - orphan marks fall back to zero-duration spans so they still reach OTLP
//!
//! The public API is intentionally small:
//!
//! - [`OpenTelemetryConfig`] configures the OTLP exporter and resource metadata
//! - [`OpenTelemetrySubscriber`] exposes a Nexus [`EventSubscriberFn`] and
//!   convenience `register` / `deregister` / `force_flush` / `shutdown` methods

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use nvidia_nat_nexus_core::{
    nat_nexus_deregister_subscriber, nat_nexus_register_subscriber, Event, EventSubscriberFn,
    HandleAttributes, NexusError, ScopeType, ToolAttributes,
};
use opentelemetry::trace::{
    Span as _, SpanContext, SpanKind, TraceContextExt, Tracer, TracerProvider as _,
};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, Span};
use opentelemetry_sdk::Resource;
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
use wasm_bindgen_futures::{spawn_local, JsFuture};
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
    UnsupportedTransport { transport: &'static str },
    /// Failed to parse a configured gRPC metadata header.
    #[error("invalid OTLP gRPC header {key:?}: {message}")]
    InvalidGrpcHeader { key: String, message: String },
    /// Failed to build the OTLP exporter.
    #[error("failed to build the OTLP exporter: {0}")]
    ExporterBuild(String),
    /// The underlying tracer provider returned an error.
    #[error("OpenTelemetry tracer provider error: {0}")]
    Provider(String),
    /// Registration errors from Nexus core.
    #[error(transparent)]
    Nexus(#[from] NexusError),
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
            service_name: "nat-nexus".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "nvidia-nat-nexus-otel".to_string(),
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

/// OpenTelemetry-backed Nexus subscriber.
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

    /// Returns the raw Nexus subscriber callback for custom registration flows.
    pub fn subscriber(&self) -> EventSubscriberFn {
        Arc::clone(&self.inner.subscriber)
    }

    /// Registers this subscriber globally with the Nexus runtime.
    pub fn register(&self, name: &str) -> Result<()> {
        nat_nexus_register_subscriber(name, self.subscriber()).map_err(Into::into)
    }

    /// Deregisters a previously-registered global subscriber by name.
    pub fn deregister(&self, name: &str) -> Result<bool> {
        nat_nexus_deregister_subscriber(name).map_err(Into::into)
    }

    /// Flushes finished spans through the underlying tracer provider.
    pub fn force_flush(&self) -> Result<()> {
        let guard = self.inner.processor.lock().map_err(|_| {
            OpenTelemetryError::Provider("the subscriber state lock was poisoned".to_string())
        })?;
        guard.force_flush()
    }

    /// Shuts down the underlying tracer provider.
    ///
    /// Call `deregister(...)` first if the subscriber is still registered with Nexus.
    pub fn shutdown(&self) -> Result<()> {
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

    let builder = SdkTracerProvider::builder().with_resource(
        Resource::builder_empty()
            .with_attributes(resource_attributes)
            .build(),
    );

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
        match event {
            Event::ScopeStart(_) | Event::ToolStart(_) | Event::LLMStart(_) => {
                self.process_start(event)
            }
            Event::ScopeEnd(_) | Event::ToolEnd(_) | Event::LLMEnd(_) => self.process_end(event),
            Event::Mark(_) => self.process_mark(event),
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
        span_attributes.push(KeyValue::new("nexus.mark.orphan", true));
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
        Some(ScopeType::Tool)
            if matches!(
                event.attributes(),
                Some(HandleAttributes::Tool(attributes)) if attributes.contains(ToolAttributes::LOCAL)
            ) =>
        {
            SpanKind::Internal
        }
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
    match event {
        Event::ScopeStart(inner) => Some(inner.scope_type),
        Event::ScopeEnd(inner) => Some(inner.scope_type),
        Event::ToolStart(_) | Event::ToolEnd(_) => Some(ScopeType::Tool),
        Event::LLMStart(_) | Event::LLMEnd(_) => Some(ScopeType::Llm),
        Event::Mark(_) => None,
    }
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
        "nexus.handle_attributes_json",
        handle_attributes.as_ref(),
    );
    push_serialized(&mut attributes, "nexus.start.data_json", event.data());
    push_serialized(
        &mut attributes,
        "nexus.start.metadata_json",
        event.metadata(),
    );
    push_serialized(&mut attributes, "nexus.start.input_json", event.input());
    attributes
}

fn end_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    push_serialized(&mut attributes, "nexus.end.data_json", event.data());
    push_serialized(&mut attributes, "nexus.end.metadata_json", event.metadata());
    push_serialized(&mut attributes, "nexus.end.output_json", event.output());
    attributes
}

fn mark_attributes(event: &Event) -> Vec<KeyValue> {
    let handle_attributes = event.attributes();
    let mut attributes = vec![
        KeyValue::new("nexus.mark.uuid", event.uuid().to_string()),
        KeyValue::new(
            "nexus.mark.parent_uuid",
            event
                .parent_uuid()
                .map(|uuid| uuid.to_string())
                .unwrap_or_default(),
        ),
    ];
    push_serialized(
        &mut attributes,
        "nexus.mark.attributes_json",
        handle_attributes.as_ref(),
    );
    push_serialized(&mut attributes, "nexus.mark.data_json", event.data());
    push_serialized(
        &mut attributes,
        "nexus.mark.metadata_json",
        event.metadata(),
    );
    attributes
}

fn common_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = vec![
        KeyValue::new("nexus.uuid", event.uuid().to_string()),
        KeyValue::new(
            "nexus.parent_uuid",
            event
                .parent_uuid()
                .map(|uuid| uuid.to_string())
                .unwrap_or_default(),
        ),
        KeyValue::new(
            "nexus.scope_type",
            scope_type_name(semantic_scope_type(event)),
        ),
    ];

    if let Some(model_name) = event.model_name() {
        attributes.push(KeyValue::new("nexus.model_name", model_name.to_string()));
    }
    if let Some(tool_call_id) = event.tool_call_id() {
        attributes.push(KeyValue::new(
            "nexus.tool_call_id",
            tool_call_id.to_string(),
        ));
    }

    attributes
}

fn push_serialized<T: Serialize>(
    attributes: &mut Vec<KeyValue>,
    key: &'static str,
    value: Option<&T>,
) {
    if let Some(value) = value {
        if let Ok(json) = serde_json::to_string(value) {
            attributes.push(KeyValue::new(key, json));
        }
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
mod tests {
    use super::*;
    use nvidia_nat_nexus_core::{Json, ScopeType};
    use opentelemetry_sdk::trace::InMemorySpanExporterBuilder;
    use serde_json::json;
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::sync::{Mutex, OnceLock};
    use std::thread;
    use uuid::Uuid;

    static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_mutex() -> &'static Mutex<()> {
        TEST_MUTEX.get_or_init(|| Mutex::new(()))
    }

    fn reset_global() {
        let context = nvidia_nat_nexus_core::global_context();
        *context.write().unwrap() = nvidia_nat_nexus_core::NatNexusContextState::new();
    }

    fn make_provider() -> (
        SdkTracerProvider,
        opentelemetry_sdk::trace::InMemorySpanExporter,
    ) {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        (provider, exporter)
    }

    fn attr_map(attributes: &[KeyValue]) -> HashMap<String, String> {
        attributes
            .iter()
            .map(|attribute| {
                (
                    attribute.key.as_str().to_string(),
                    attribute.value.to_string(),
                )
            })
            .collect()
    }

    fn make_start_event(
        uuid: Uuid,
        parent_uuid: Option<Uuid>,
        name: &str,
        scope_type: ScopeType,
        input: Option<Json>,
    ) -> Event {
        match scope_type {
            ScopeType::Tool => Event::tool_start(
                parent_uuid,
                uuid,
                name,
                None,
                None,
                nvidia_nat_nexus_core::ToolAttributes::empty(),
                input,
                None,
            ),
            ScopeType::Llm => Event::llm_start(
                parent_uuid,
                uuid,
                name,
                None,
                None,
                nvidia_nat_nexus_core::LLMAttributes::empty(),
                input,
                None,
            ),
            _ => Event::scope_start(
                parent_uuid,
                uuid,
                name,
                None,
                None,
                nvidia_nat_nexus_core::ScopeAttributes::empty(),
                scope_type,
            ),
        }
    }

    #[test]
    fn config_defaults_and_builder_overrides_are_applied() {
        let config = OpenTelemetryConfig::http_binary("demo-agent")
            .with_endpoint("http://localhost:4318/v1/traces")
            .with_header("authorization", "Bearer token")
            .with_resource_attribute("deployment.environment", "test")
            .with_service_namespace("agents")
            .with_service_version("1.2.3")
            .with_instrumentation_scope("demo-scope")
            .with_timeout(Duration::from_millis(1250));

        assert_eq!(config.transport, OtlpTransport::HttpBinary);
        assert_eq!(
            config.endpoint.as_deref(),
            Some("http://localhost:4318/v1/traces")
        );
        assert_eq!(
            config.headers.get("authorization"),
            Some(&"Bearer token".into())
        );
        assert_eq!(
            config.resource_attributes.get("deployment.environment"),
            Some(&"test".into())
        );
        assert_eq!(config.service_name, "demo-agent");
        assert_eq!(config.service_namespace.as_deref(), Some("agents"));
        assert_eq!(config.service_version.as_deref(), Some("1.2.3"));
        assert_eq!(config.instrumentation_scope, "demo-scope");
        assert_eq!(config.timeout, Duration::from_millis(1250));

        let defaults = OpenTelemetryConfig::default();
        assert_eq!(defaults.transport, OtlpTransport::HttpBinary);
        assert_eq!(defaults.service_name, "nat-nexus");
        assert_eq!(defaults.instrumentation_scope, "nvidia-nat-nexus-otel");
        assert_eq!(defaults.timeout, Duration::from_secs(3));
        assert!(defaults.headers.is_empty());
        assert!(defaults.resource_attributes.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn grpc_config_requires_a_tokio_runtime() {
        let err = match OpenTelemetrySubscriber::new(OpenTelemetryConfig::grpc("demo-agent")) {
            Ok(_) => panic!("gRPC construction should require a Tokio runtime"),
            Err(err) => err,
        };
        assert!(matches!(err, OpenTelemetryError::MissingTokioRuntime));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn invalid_grpc_headers_are_rejected() {
        let err = build_grpc_metadata(&HashMap::from([(
            "bad key".to_string(),
            "value".to_string(),
        )]))
        .expect_err("invalid metadata key should fail");
        assert!(matches!(err, OpenTelemetryError::InvalidGrpcHeader { .. }));
    }

    #[test]
    fn subscriber_registration_and_provider_lifecycle_methods_work() {
        let (provider, _exporter) = make_provider();
        let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "test-scope");
        let name = format!("otel_test_{}", Uuid::new_v4().simple());

        subscriber.register(&name).unwrap();
        assert!(subscriber.deregister(&name).unwrap());
        assert!(!subscriber.deregister(&name).unwrap());
        subscriber.force_flush().unwrap();
        subscriber.shutdown().unwrap();
    }

    #[test]
    fn registered_subscriber_emits_spans_for_scope_push_pop_and_marks() {
        let _guard = test_mutex().lock().unwrap();
        reset_global();

        let (provider, exporter) = make_provider();
        let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "e2e-scope");
        let name = format!("otel_e2e_{}", Uuid::new_v4().simple());

        subscriber.register(&name).unwrap();
        let handle = nvidia_nat_nexus_core::nat_nexus_push_scope(
            "otel_scope",
            ScopeType::Agent,
            None,
            nvidia_nat_nexus_core::ScopeAttributes::empty(),
            Some(json!({"scope": true})),
            Some(json!({"phase": "start"})),
        )
        .unwrap();
        nvidia_nat_nexus_core::nat_nexus_event(
            "otel_mark",
            Some(&handle),
            Some(json!({"step": 1})),
            Some(json!({"source": "rust-test"})),
        )
        .unwrap();
        nvidia_nat_nexus_core::nat_nexus_pop_scope(&handle.uuid).unwrap();

        assert!(subscriber.deregister(&name).unwrap());
        subscriber.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);

        let span = &spans[0];
        assert_eq!(span.name.as_ref(), "otel_scope");
        assert_eq!(span.events.events.len(), 1);
        assert_eq!(span.events.events[0].name.as_ref(), "otel_mark");

        let attributes = attr_map(&span.attributes);
        assert_eq!(
            attributes.get("nexus.start.data_json"),
            Some(&"{\"scope\":true}".to_string())
        );
        assert_eq!(
            attributes.get("nexus.start.metadata_json"),
            Some(&"{\"phase\":\"start\"}".to_string())
        );

        let event_attributes = attr_map(&span.events.events[0].attributes);
        assert_eq!(
            event_attributes.get("nexus.mark.data_json"),
            Some(&"{\"step\":1}".to_string())
        );
        assert_eq!(
            event_attributes.get("nexus.mark.metadata_json"),
            Some(&"{\"source\":\"rust-test\"}".to_string())
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn http_config_exports_scope_push_pop_and_marks_without_tokio_runtime() {
        let _guard = test_mutex().lock().unwrap();
        reset_global();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
        let (request_tx, request_rx) = mpsc::channel();

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut bytes = Vec::new();
            let mut buf = [0_u8; 4096];

            let (header_end, content_length) = loop {
                let read = stream.read(&mut buf).unwrap();
                if read == 0 {
                    panic!("collector closed before receiving an OTLP request");
                }
                bytes.extend_from_slice(&buf[..read]);

                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let header_end = header_end + 4;
                    let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers_text
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            if name.eq_ignore_ascii_case("content-length") {
                                value.trim().parse::<usize>().ok()
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    break (header_end, content_length);
                }
            };

            while bytes.len() < header_end + content_length {
                let read = stream.read(&mut buf).unwrap();
                if read == 0 {
                    break;
                }
                bytes.extend_from_slice(&buf[..read]);
            }

            let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
            let request_line = headers_text.lines().next().unwrap();
            let path = request_line.split_whitespace().nth(1).unwrap().to_string();
            let content_type = headers_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-type") {
                        Some(value.trim().to_string())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            let body = bytes[header_end..header_end + content_length].to_vec();

            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            request_tx.send((path, content_type, body)).unwrap();
        });

        let config = OpenTelemetryConfig::http_binary("demo-agent").with_endpoint(endpoint);
        let subscriber = OpenTelemetrySubscriber::new(config).unwrap();
        let name = format!("otel_http_{}", Uuid::new_v4().simple());

        subscriber.register(&name).unwrap();
        let handle = nvidia_nat_nexus_core::nat_nexus_push_scope(
            "otel_scope",
            ScopeType::Agent,
            None,
            nvidia_nat_nexus_core::ScopeAttributes::empty(),
            Some(json!({"scope": true})),
            None,
        )
        .unwrap();
        nvidia_nat_nexus_core::nat_nexus_event(
            "otel_mark",
            Some(&handle),
            Some(json!({"step": 1})),
            Some(json!({"source": "rust-http"})),
        )
        .unwrap();
        nvidia_nat_nexus_core::nat_nexus_pop_scope(&handle.uuid).unwrap();

        assert!(subscriber.deregister(&name).unwrap());
        subscriber.force_flush().unwrap();

        let (path, content_type, body) = request_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("expected an OTLP request");
        assert_eq!(path, "/v1/traces");
        assert_eq!(content_type, "application/x-protobuf");
        assert!(!body.is_empty());
    }

    #[test]
    fn records_span_start_mark_and_end() {
        let (provider, exporter) = make_provider();
        let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
        let root_uuid = Uuid::new_v4();

        let start = make_start_event(
            root_uuid,
            None,
            "search",
            ScopeType::Tool,
            Some(json!({"query": "hello"})),
        );
        processor.process(&start);

        let mark = Event::mark(
            Some(root_uuid),
            Uuid::new_v4(),
            "checkpoint",
            Some(json!({"step": 1})),
            None,
        );
        processor.process(&mark);

        let end = Event::tool_end(
            None,
            root_uuid,
            "search",
            None,
            None,
            nvidia_nat_nexus_core::ToolAttributes::empty(),
            Some(json!({"result": "ok"})),
            None,
        );
        processor.process(&end);

        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name.as_ref(), "search");
        assert_eq!(span.events.events.len(), 1);
        assert_eq!(span.events.events[0].name.as_ref(), "checkpoint");

        let attributes = attr_map(&span.attributes);
        assert_eq!(attributes.get("nexus.uuid"), Some(&root_uuid.to_string()));
        assert_eq!(
            attributes.get("nexus.start.input_json"),
            Some(&"{\"query\":\"hello\"}".to_string())
        );
        assert_eq!(
            attributes.get("nexus.end.output_json"),
            Some(&"{\"result\":\"ok\"}".to_string())
        );
    }

    #[test]
    fn preserves_parent_child_relationships() {
        let (provider, exporter) = make_provider();
        let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());

        let root_uuid = Uuid::new_v4();
        let child_uuid = Uuid::new_v4();

        processor.process(&make_start_event(
            root_uuid,
            None,
            "agent",
            ScopeType::Agent,
            None,
        ));
        processor.process(&make_start_event(
            child_uuid,
            Some(root_uuid),
            "model-call",
            ScopeType::Llm,
            None,
        ));
        processor.process(&Event::llm_end(
            Some(root_uuid),
            child_uuid,
            "model-call",
            None,
            None,
            nvidia_nat_nexus_core::LLMAttributes::empty(),
            None,
            None,
        ));
        processor.process(&Event::scope_end(
            None,
            root_uuid,
            "agent",
            None,
            None,
            nvidia_nat_nexus_core::ScopeAttributes::empty(),
            ScopeType::Agent,
        ));

        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 2);
        let parent = spans
            .iter()
            .find(|span| span.name.as_ref() == "agent")
            .unwrap();
        let child = spans
            .iter()
            .find(|span| span.name.as_ref() == "model-call")
            .unwrap();

        assert_eq!(
            child.span_context.trace_id(),
            parent.span_context.trace_id()
        );
        assert_eq!(child.parent_span_id, parent.span_context.span_id());
        assert!(!child.parent_span_is_remote);
    }

    #[test]
    fn orphan_marks_become_zero_duration_spans() {
        let (provider, exporter) = make_provider();
        let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
        let mark = Event::mark(
            None,
            Uuid::new_v4(),
            "detached",
            Some(json!({"kind": "standalone"})),
            None,
        );

        processor.process(&mark);
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name.as_ref(), "mark:detached");
        assert_eq!(span.start_time, span.end_time);

        let attributes = attr_map(&span.attributes);
        assert_eq!(
            attributes.get("nexus.mark.orphan"),
            Some(&"true".to_string())
        );
    }

    #[test]
    fn semantic_scope_type_and_span_kind_follow_event_variants() {
        let scope_event = Event::scope_start(
            None,
            Uuid::new_v4(),
            "guardrail",
            None,
            None,
            nvidia_nat_nexus_core::ScopeAttributes::empty(),
            ScopeType::Guardrail,
        );
        assert_eq!(
            semantic_scope_type(&scope_event),
            Some(ScopeType::Guardrail)
        );
        assert_eq!(span_kind(&scope_event), SpanKind::Internal);

        let local_tool = Event::tool_start(
            None,
            Uuid::new_v4(),
            "search",
            None,
            None,
            nvidia_nat_nexus_core::ToolAttributes::LOCAL,
            Some(json!({"query": "hello"})),
            None,
        );
        assert_eq!(semantic_scope_type(&local_tool), Some(ScopeType::Tool));
        assert_eq!(span_kind(&local_tool), SpanKind::Internal);

        let llm_event = Event::llm_end(
            None,
            Uuid::new_v4(),
            "model-call",
            None,
            None,
            nvidia_nat_nexus_core::LLMAttributes::empty(),
            Some(json!({"result": "hello"})),
            None,
        );
        assert_eq!(semantic_scope_type(&llm_event), Some(ScopeType::Llm));
        assert_eq!(span_kind(&llm_event), SpanKind::Client);

        let mark = Event::mark(None, Uuid::new_v4(), "checkpoint", None, None);
        assert_eq!(semantic_scope_type(&mark), None);
        assert_eq!(span_kind(&mark), SpanKind::Internal);
    }

    #[test]
    fn pre_epoch_timestamps_round_trip_through_system_time() {
        let timestamp = DateTime::parse_from_rfc3339("1969-12-31T23:59:58.500000000Z")
            .unwrap()
            .with_timezone(&Utc);

        assert_eq!(
            to_system_time(timestamp),
            UNIX_EPOCH - Duration::new(1, 500_000_000)
        );
    }
}
