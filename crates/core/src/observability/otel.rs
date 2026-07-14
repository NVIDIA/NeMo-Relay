// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry subscriber support for NeMo Relay.
//!
//! This crate adapts NeMo Relay lifecycle events into OpenTelemetry trace spans:
//!
//! - scope/tool/LLM `Start` events open spans
//! - matching `End` events close spans
//! - `Mark` events become span events by default, with an optional visible child-span projection
//! - orphan marks fall back to zero-duration spans so they still reach OTLP
//!
//! The public API is intentionally small:
//!
//! - [`OpenTelemetryConfig`] configures the OTLP exporter and resource metadata
//! - [`OpenTelemetrySubscriber`] exposes a NeMo Relay [`EventSubscriberFn`] and
//!   convenience `register` / `deregister` / `force_flush` / `shutdown` methods

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{
    MarkProjection, OtlpAttributeMapping, apply_attribute_mappings, attribute_mapping_aliases,
    default_mark_exclude_names, effective_mark_projection, estimate_cost_for_response_or_model,
    estimate_cost_for_response_or_requested_model, manual, model_name_for_llm_event,
    push_serialized_top_level_attributes, push_top_level_json_attributes,
    validate_attribute_mappings,
};
use crate::api::event::{Event, EventNormalizationExt, ScopeCategory};
use crate::api::runtime::EventSubscriberFn;
use crate::api::scope::ScopeType;
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::codec::response::CostEstimate;
use crate::error::FlowError;
use chrono::{DateTime, Utc};
use opentelemetry::trace::{
    Span as _, SpanContext, SpanKind, TraceContextExt, Tracer, TracerProvider as _,
};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, Span};
use uuid::Uuid;

const COMPLETED_SPAN_CONTEXT_LIMIT: usize = 4096;

use opentelemetry_otlp::WithTonicConfig;
use tokio::runtime::Handle;
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};

/// Result type for the OpenTelemetry subscriber crate.
pub type Result<T> = std::result::Result<T, OpenTelemetryError>;

/// Errors produced while configuring or operating the OpenTelemetry subscriber.
#[derive(Debug, thiserror::Error)]
pub enum OpenTelemetryError {
    /// The tonic gRPC exporter requires an active Tokio runtime.
    #[error("the OTLP gRPC exporter requires an active Tokio runtime")]
    MissingTokioRuntime,
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
    /// Attribute mapping configuration was invalid.
    #[error("invalid attribute mappings: {0}")]
    InvalidAttributeMappings(String),
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
    mark_projection: MarkProjection,
    mark_exclude_names: Vec<String>,
    attribute_mappings: Vec<OtlpAttributeMapping>,
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
            mark_projection: MarkProjection::default(),
            mark_exclude_names: default_mark_exclude_names(),
            attribute_mappings: Vec::new(),
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

    /// Selects how point-in-time marks are represented in exported traces.
    pub fn with_mark_projection(mut self, mark_projection: MarkProjection) -> Self {
        self.mark_projection = mark_projection;
        self
    }

    /// Excludes named marks from tool projection while preserving their native
    /// event representation. The default excludes high-volume `llm.chunk`
    /// marks.
    pub fn with_mark_exclude_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.mark_exclude_names = names.into_iter().map(Into::into).collect();
        self
    }

    /// Adds a typed attribute copy after event payload projection.
    pub fn with_attribute_mapping(
        mut self,
        key: impl Into<String>,
        alias: impl Into<String>,
    ) -> Self {
        self.attribute_mappings
            .push(OtlpAttributeMapping::new(key, alias));
        self
    }

    /// Replaces the configured typed attribute copies.
    pub fn with_attribute_mappings<I>(mut self, mappings: I) -> Self
    where
        I: IntoIterator<Item = OtlpAttributeMapping>,
    {
        self.attribute_mappings = mappings.into_iter().collect();
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
        if config.transport == OtlpTransport::Grpc && tokio::runtime::Handle::try_current().is_err()
        {
            return Err(OpenTelemetryError::MissingTokioRuntime);
        }
        validate_attribute_mappings(&config.attribute_mappings)
            .map_err(OpenTelemetryError::InvalidAttributeMappings)?;

        let provider = build_tracer_provider(&config)?;
        Ok(Self::from_tracer_provider_with_scope(
            provider,
            config.instrumentation_scope,
            config.mark_projection,
            config.mark_exclude_names,
            config.attribute_mappings,
        ))
    }

    /// Builds a subscriber from an already-configured tracer provider.
    pub fn from_tracer_provider(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
    ) -> Self {
        Self::from_tracer_provider_with_scope(
            provider,
            instrumentation_scope.into(),
            MarkProjection::default(),
            default_mark_exclude_names(),
            Vec::new(),
        )
    }

    /// Builds a subscriber from a tracer provider with an explicit mark projection.
    pub fn from_tracer_provider_with_mark_projection(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
        mark_projection: MarkProjection,
    ) -> Self {
        Self::from_tracer_provider_with_scope(
            provider,
            instrumentation_scope.into(),
            mark_projection,
            default_mark_exclude_names(),
            Vec::new(),
        )
    }

    /// Builds a subscriber with explicit mark projection and exclusion names.
    pub fn from_tracer_provider_with_mark_projection_and_exclusions<I, S>(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
        mark_projection: MarkProjection,
        mark_exclude_names: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::from_tracer_provider_with_scope(
            provider,
            instrumentation_scope.into(),
            mark_projection,
            mark_exclude_names.into_iter().map(Into::into).collect(),
            Vec::new(),
        )
    }

    /// Builds a subscriber from a tracer provider with typed attribute copies.
    pub fn from_tracer_provider_with_attribute_mappings<I>(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
        attribute_mappings: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = OtlpAttributeMapping>,
    {
        let attribute_mappings = attribute_mappings.into_iter().collect::<Vec<_>>();
        validate_attribute_mappings(&attribute_mappings)
            .map_err(OpenTelemetryError::InvalidAttributeMappings)?;
        Ok(Self::from_tracer_provider_with_scope(
            provider,
            instrumentation_scope.into(),
            MarkProjection::default(),
            default_mark_exclude_names(),
            attribute_mappings,
        ))
    }

    fn from_tracer_provider_with_scope(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        mark_projection: MarkProjection,
        mark_exclude_names: Vec<String>,
        attribute_mappings: Vec<OtlpAttributeMapping>,
    ) -> Self {
        let processor = Arc::new(Mutex::new(
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
                provider,
                instrumentation_scope,
                mark_projection,
                mark_exclude_names,
                attribute_mappings,
            ),
        ));
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
            builder
                .build()
                .map_err(|e| OpenTelemetryError::ExporterBuild(e.to_string()))?
        }
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

    if Handle::try_current().is_ok() {
        Ok(builder.with_batch_exporter(exporter).build())
    } else {
        Ok(builder.with_simple_exporter(exporter).build())
    }
}

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
    projected_attributes: Vec<KeyValue>,
}

struct OtelEventProcessor {
    active_spans: HashMap<Uuid, ActiveSpan>,
    completed_span_contexts: HashMap<Uuid, SpanContext>,
    completed_span_order: VecDeque<Uuid>,
    provider: SdkTracerProvider,
    tracer: SdkTracer,
    mark_projection: MarkProjection,
    mark_exclude_names: Vec<String>,
    attribute_mappings: Vec<OtlpAttributeMapping>,
}

impl OtelEventProcessor {
    #[cfg(test)]
    fn new(provider: SdkTracerProvider, instrumentation_scope: String) -> Self {
        Self::new_with_mark_projection(provider, instrumentation_scope, MarkProjection::default())
    }

    #[cfg(test)]
    fn new_with_mark_projection(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        mark_projection: MarkProjection,
    ) -> Self {
        Self::new_with_mark_projection_and_exclusions(
            provider,
            instrumentation_scope,
            mark_projection,
            default_mark_exclude_names(),
        )
    }

    #[cfg(test)]
    fn new_with_mark_projection_and_exclusions(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        mark_projection: MarkProjection,
        mark_exclude_names: Vec<String>,
    ) -> Self {
        Self::new_with_mark_projection_and_exclusions_and_mappings(
            provider,
            instrumentation_scope,
            mark_projection,
            mark_exclude_names,
            Vec::new(),
        )
    }

    fn new_with_mark_projection_and_exclusions_and_mappings(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        mark_projection: MarkProjection,
        mark_exclude_names: Vec<String>,
        attribute_mappings: Vec<OtlpAttributeMapping>,
    ) -> Self {
        let tracer = provider.tracer(instrumentation_scope);
        Self {
            active_spans: HashMap::new(),
            completed_span_contexts: HashMap::new(),
            completed_span_order: VecDeque::new(),
            provider,
            tracer,
            mark_projection,
            mark_exclude_names,
            attribute_mappings,
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
        self.remove_completed_span_context(event.uuid());
        let mut span = self
            .tracer
            .span_builder(span_name(event))
            .with_kind(span_kind(event))
            .with_start_time(to_system_time(*event.timestamp()))
            .start_with_context(&self.tracer, &self.parent_context(event));
        let attributes = start_attributes(event);
        span.set_attributes(attributes.clone());
        let span_context = local_parent_span_context(span.span_context());
        self.active_spans.insert(
            event.uuid(),
            ActiveSpan {
                span,
                span_context,
                projected_attributes: attributes,
            },
        );
    }

    fn process_end(&mut self, event: &Event) {
        let Some(mut active_span) = self.active_spans.remove(&event.uuid()) else {
            return;
        };
        self.record_completed_span_context(event.uuid(), active_span.span_context.clone());

        super::set_span_status_from_event_metadata(&mut active_span.span, event);
        let mut attributes = end_attributes(event);
        let mut projected_attributes = active_span.projected_attributes;
        projected_attributes.extend(attributes.iter().cloned());
        attributes.extend(attribute_mapping_aliases(
            &projected_attributes,
            &self.attribute_mappings,
        ));
        active_span.span.set_attributes(attributes);
        active_span
            .span
            .end_with_timestamp(to_system_time(*event.timestamp()));
    }

    fn process_mark(&mut self, event: &Event) {
        if effective_mark_projection(event, self.mark_projection, &self.mark_exclude_names)
            == MarkProjection::Tool
        {
            self.process_mark_as_tool(event);
            return;
        }
        let mark_name = event.name().to_string();
        let timestamp = to_system_time(*event.timestamp());
        let mut attributes = mark_attributes(event);

        if self.find_parent_span(event).is_some() {
            apply_attribute_mappings(&mut attributes, &self.attribute_mappings);
            let parent_span = self
                .find_parent_span_mut(event)
                .expect("parent span was present during mark projection");
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
        attributes.push(KeyValue::new("nemo_relay.mark.orphan", true));
        apply_attribute_mappings(&mut attributes, &self.attribute_mappings);
        span.set_attributes(attributes);
        span.end_with_timestamp(timestamp);
    }

    fn process_mark_as_tool(&mut self, event: &Event) {
        let timestamp = to_system_time(*event.timestamp());
        let orphan = self.find_parent_span(event).is_none();
        let mut attributes = mark_attributes(event);
        attributes.push(KeyValue::new("nemo_relay.mark.projection", "tool"));
        attributes.push(KeyValue::new("nemo_relay.scope_type", "tool"));
        if orphan {
            attributes.push(KeyValue::new("nemo_relay.mark.orphan", true));
        }
        apply_attribute_mappings(&mut attributes, &self.attribute_mappings);

        let mut span = self
            .tracer
            .span_builder(format!("mark:{}", event.name()))
            .with_kind(SpanKind::Internal)
            .with_start_time(timestamp)
            .start_with_context(&self.tracer, &self.parent_context(event));
        span.set_attributes(attributes);
        span.end_with_timestamp(timestamp);
    }

    fn parent_context(&self, event: &Event) -> Context {
        if let Some(active_span) = self.find_parent_span(event) {
            return Context::new().with_remote_span_context(active_span.span_context.clone());
        }
        event
            .parent_uuid()
            .and_then(|uuid| self.completed_span_contexts.get(&uuid))
            .map(|span_context| Context::new().with_remote_span_context(span_context.clone()))
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

    fn remove_completed_span_context(&mut self, uuid: Uuid) {
        self.completed_span_contexts.remove(&uuid);
        self.completed_span_order
            .retain(|completed_uuid| *completed_uuid != uuid);
    }

    fn record_completed_span_context(&mut self, uuid: Uuid, span_context: SpanContext) {
        if self
            .completed_span_contexts
            .insert(uuid, span_context)
            .is_none()
        {
            self.completed_span_order.push_back(uuid);
        }
        while self.completed_span_order.len() > COMPLETED_SPAN_CONTEXT_LIMIT {
            if let Some(expired) = self.completed_span_order.pop_front() {
                self.completed_span_contexts.remove(&expired);
            }
        }
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
    push_serialized_top_level_attributes(
        &mut attributes,
        "nemo_relay.handle_attributes",
        event.attributes(),
    );
    push_top_level_json_attributes(&mut attributes, "nemo_relay.start.data", event.data());
    push_top_level_json_attributes(
        &mut attributes,
        "nemo_relay.start.metadata",
        event.metadata(),
    );
    push_top_level_json_attributes(&mut attributes, "nemo_relay.start.input", event.input());
    attributes
}

fn end_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    push_top_level_json_attributes(&mut attributes, "nemo_relay.end.data", event.data());
    push_top_level_json_attributes(&mut attributes, "nemo_relay.end.metadata", event.metadata());
    push_top_level_json_attributes(&mut attributes, "nemo_relay.end.output", event.output());
    if event
        .category()
        .is_some_and(|category| category.as_str() == "llm")
        && let Some((cost, currency)) = cost_from_llm_event(event)
    {
        attributes.push(KeyValue::new("nemo_relay.llm.cost.total", cost));
        attributes.push(KeyValue::new("nemo_relay.llm.cost.currency", currency));
    }
    if let Some(response) = event.annotated_response()
        && let Some(summary) = response.optimization_summary.as_ref()
    {
        push_optimization_attributes(&mut attributes, summary);
    }
    attributes
}

fn push_optimization_attributes(
    attributes: &mut Vec<KeyValue>,
    summary: &crate::codec::optimization::LlmOptimizationSummary,
) {
    if let Some(model) = summary.baseline_model.as_ref() {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.baseline_model",
            model.model.clone(),
        ));
    }
    if let Some(model) = summary.effective_model.as_ref() {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.effective_model",
            model.model.clone(),
        ));
    }
    if let Some(tokens) = summary.tokens_saved.prompt_tokens {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.prompt_tokens_saved",
            i64::try_from(tokens).unwrap_or(i64::MAX),
        ));
    }
    if let Some(tokens) = summary.tokens_saved.total_tokens {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.total_tokens_saved",
            i64::try_from(tokens).unwrap_or(i64::MAX),
        ));
    }
    if let Some(cost) = summary
        .baseline_cost
        .as_ref()
        .and_then(|cost| cost.total_or_component_sum())
    {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.baseline_cost",
            cost,
        ));
    }
    if let Some(cost) = summary.baseline_cost.as_ref() {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.baseline_cost_currency",
            cost.currency.clone(),
        ));
        if let Some(source) = cost.pricing_source.as_ref() {
            attributes.push(KeyValue::new(
                "nemo_relay.llm.optimization.baseline_pricing_source",
                source.clone(),
            ));
        }
        if let Some(as_of) = cost.pricing_as_of.as_ref() {
            attributes.push(KeyValue::new(
                "nemo_relay.llm.optimization.baseline_pricing_as_of",
                as_of.clone(),
            ));
        }
    }
    if let Some(cost) = summary
        .actual_cost
        .as_ref()
        .and_then(|cost| cost.total_or_component_sum())
    {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.actual_cost",
            cost,
        ));
    }
    if let Some(cost) = summary.actual_cost.as_ref() {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.actual_cost_currency",
            cost.currency.clone(),
        ));
        if let Some(source) = cost.pricing_source.as_ref() {
            attributes.push(KeyValue::new(
                "nemo_relay.llm.optimization.actual_pricing_source",
                source.clone(),
            ));
        }
        if let Some(as_of) = cost.pricing_as_of.as_ref() {
            attributes.push(KeyValue::new(
                "nemo_relay.llm.optimization.actual_pricing_as_of",
                as_of.clone(),
            ));
        }
    }
    if let Some(cost) = summary.estimated_cost_saved {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.estimated_cost_saved",
            cost,
        ));
        if let Some(currency) = summary.currency.as_ref() {
            attributes.push(KeyValue::new(
                "nemo_relay.llm.optimization.estimated_cost_saved_currency",
                currency.clone(),
            ));
        }
    }
    if let Some(currency) = summary.currency.as_ref() {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.currency",
            currency.clone(),
        ));
    }
    attributes.push(KeyValue::new(
        "nemo_relay.llm.optimization.status",
        match summary.status {
            crate::codec::optimization::LlmOptimizationSummaryStatus::Complete => "complete",
            crate::codec::optimization::LlmOptimizationSummaryStatus::Partial => "partial",
        },
    ));
    if let Some(source) = summary
        .baseline_cost
        .as_ref()
        .and_then(|cost| cost.pricing_source.as_ref())
        .or_else(|| {
            summary
                .actual_cost
                .as_ref()
                .and_then(|cost| cost.pricing_source.as_ref())
        })
    {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.pricing_source",
            source.clone(),
        ));
    }
    if let Some(as_of) = summary
        .baseline_cost
        .as_ref()
        .and_then(|cost| cost.pricing_as_of.as_ref())
        .or_else(|| {
            summary
                .actual_cost
                .as_ref()
                .and_then(|cost| cost.pricing_as_of.as_ref())
        })
    {
        attributes.push(KeyValue::new(
            "nemo_relay.llm.optimization.pricing_as_of",
            as_of.clone(),
        ));
    }
}

fn cost_from_llm_event(event: &Event) -> Option<(f64, String)> {
    if let Some(response) = event.normalized_llm_response() {
        let response = response.as_ref();
        if let Some(usage) = response.usage.as_ref() {
            if let Some(cost) = usage.cost.as_ref() {
                return cost_total_and_currency(cost);
            }
            if let Some(cost) = estimate_cost_for_response_or_requested_model(
                event,
                response.model.as_deref(),
                usage,
            ) {
                return cost_total_and_currency(&cost);
            }
        }
    }
    if let Some(cost) =
        manual::cost_from_manual_llm_output(event.output(), manual::ManualCostPolicy::AnyCurrency)
    {
        return Some(cost);
    }
    let usage = manual::usage_from_manual_llm_output(event.output())?;
    estimate_cost_for_response_or_model(
        Some(event.name()),
        event.model_name(),
        manual::model_name_from_manual_llm_output(event.output()),
        &usage,
    )
    .and_then(|cost| cost_total_and_currency(&cost))
}

fn cost_total_and_currency(cost: &CostEstimate) -> Option<(f64, String)> {
    Some((cost.total_or_component_sum()?, cost.currency.clone()))
}

fn mark_attributes(event: &Event) -> Vec<KeyValue> {
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
    push_serialized_top_level_attributes(
        &mut attributes,
        "nemo_relay.mark.attributes",
        event.attributes(),
    );
    push_top_level_json_attributes(&mut attributes, "nemo_relay.mark.data", event.data());
    push_top_level_json_attributes(
        &mut attributes,
        "nemo_relay.mark.metadata",
        event.metadata(),
    );
    if let Some(category) = event.category() {
        attributes.push(KeyValue::new(
            "nemo_relay.mark.category",
            category.as_str().to_string(),
        ));
    }
    push_serialized_top_level_attributes(
        &mut attributes,
        "nemo_relay.mark.category_profile",
        event.category_profile(),
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

    if let Some(model_name) = model_name_for_llm_event(event) {
        attributes.push(KeyValue::new("nemo_relay.model_name", model_name));
    }
    if let Some(tool_call_id) = event.tool_call_id() {
        attributes.push(KeyValue::new(
            "nemo_relay.tool_call_id",
            tool_call_id.to_string(),
        ));
    }

    attributes
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
