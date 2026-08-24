// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry subscriber support for NeMo Relay.
//!
//! This crate adapts NeMo Relay lifecycle events into the selected `full`,
//! `gen_ai`, or `openinference` OpenTelemetry trace projection. Scope start and
//! end events open and close supported spans. Non-metric mark behavior is fixed
//! by the selected projection; reserved metric-schema marks are not projected
//! into traces.
//!
//! The public API is intentionally small:
//!
//! - [`OpenTelemetryConfig`] configures the OTLP exporter and resource metadata
//! - [`OpenTelemetrySubscriber`] exposes a NeMo Relay [`EventSubscriberFn`] and
//!   convenience `register` / `deregister` / `force_flush` / `shutdown` methods

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::otel_signal::{
    MetricMarkClassification, SignalRuntimeDiagnostics, classify_metric_mark,
    should_relog_runtime_diagnostic,
};
use super::{
    MarkProjection, OpenTelemetryRuntimeDiagnostics, OpenTelemetryType, OtlpAttributeMapping,
    apply_attribute_mappings, attribute_mapping_aliases, attribute_mapping_inputs,
    default_mark_exclude_names, effective_mark_projection, estimate_cost_for_response_or_model,
    estimate_cost_for_response_or_requested_model, manual, model_name_for_llm_event,
    promote_event_metadata_attributes, push_serialized_top_level_attributes,
    push_session_identity_attributes, push_tool_result_annotation_attribute,
    push_top_level_json_attributes, relay_span_id, relay_trace_id, validate_attribute_mappings,
    validate_metadata_promotion_prefixes,
};
use crate::api::event::{Event, EventNormalizationExt, ScopeCategory};
use crate::api::runtime::{EventSubscriberFn, current_scope_stack};
use crate::api::scope::ScopeType;
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::codec::response::CostEstimate;
use crate::error::FlowError;
use chrono::{DateTime, Utc};
use opentelemetry::trace::{
    Span as _, SpanContext, SpanId, SpanKind, TraceContextExt, TraceFlags, TraceId, TraceState,
    Tracer, TracerProvider as _,
};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::{
    Protocol, SpanExporter as OtlpSpanExporter, WithExportConfig, WithHttpConfig,
};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::trace::{
    BatchConfigBuilder, BatchSpanProcessor, IdGenerator, RandomIdGenerator, SdkTracer,
    SdkTracerProvider, Span, SpanData, SpanExporter, SpanProcessor,
};
use uuid::Uuid;

use crate::plugin::OTEL_RUNTIME_DELIVERY_FAILURE_MARKER;

pub(super) const COMPLETED_SPAN_CONTEXT_LIMIT: usize = 4096;

use opentelemetry_otlp::WithTonicConfig;
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};

thread_local! {
    static PENDING_RELAY_IDS: RefCell<Option<(TraceId, SpanId)>> = const { RefCell::new(None) };
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RelayIdGenerator;

impl IdGenerator for RelayIdGenerator {
    fn new_trace_id(&self) -> TraceId {
        PENDING_RELAY_IDS
            .with(|ids| ids.borrow().map(|(trace_id, _)| trace_id))
            .unwrap_or_else(|| RandomIdGenerator::default().new_trace_id())
    }

    fn new_span_id(&self) -> SpanId {
        PENDING_RELAY_IDS
            .with(|ids| ids.borrow().map(|(_, span_id)| span_id))
            .unwrap_or_else(|| RandomIdGenerator::default().new_span_id())
    }
}

fn with_relay_ids<T>(uuid: Uuid, build: impl FnOnce() -> T) -> T {
    struct ResetPendingIds;

    impl Drop for ResetPendingIds {
        fn drop(&mut self) {
            PENDING_RELAY_IDS.with(|ids| {
                ids.replace(None);
            });
        }
    }

    PENDING_RELAY_IDS.with(|ids| {
        ids.replace(Some((
            super::relay_trace_id(uuid),
            super::relay_span_id(uuid),
        )));
    });
    let _reset = ResetPendingIds;
    build()
}

/// Result type for the OpenTelemetry subscriber crate.
pub type Result<T> = std::result::Result<T, OpenTelemetryError>;

pub(super) fn normalize_shutdown_result(result: OTelSdkResult) -> OTelSdkResult {
    match result {
        Err(OTelSdkError::AlreadyShutdown) => Ok(()),
        result => result,
    }
}

/// Errors produced while configuring or operating the OpenTelemetry subscriber.
#[derive(Debug, thiserror::Error)]
pub enum OpenTelemetryError {
    /// Failed to parse a configured gRPC metadata header.
    #[error("invalid OTLP gRPC header {key:?}: {message}")]
    InvalidGrpcHeader {
        /// Header name that failed to parse.
        key: String,
        /// Parser failure message.
        message: String,
    },
    /// A configured OTLP header name or value is invalid.
    #[error("invalid OTLP header {key:?}: {message}")]
    InvalidHeader {
        /// Header name that failed validation.
        key: String,
        /// Validation failure message.
        message: String,
    },
    /// Process-global OTLP header variables cannot be isolated per endpoint.
    #[error(
        "{variable} is not supported because process-global OTLP headers can leak across endpoints; use the endpoint headers or header_env configuration"
    )]
    GlobalHeaderEnvironmentUnsupported {
        /// Environment variable that was set.
        variable: &'static str,
    },
    /// Failed to build the OTLP exporter.
    #[error("failed to build the OTLP exporter: {0}")]
    ExporterBuild(String),
    /// The underlying tracer provider returned an error.
    #[error("OpenTelemetry tracer provider error: {0}")]
    TraceProvider(String),
    /// The underlying logger provider returned an error.
    #[error("OpenTelemetry log provider error: {0}")]
    LogProvider(String),
    /// The underlying meter provider returned an error.
    #[error("OpenTelemetry metric provider error: {0}")]
    MetricProvider(String),
    /// Attribute mapping configuration was invalid.
    #[error("invalid attribute mappings: {0}")]
    InvalidAttributeMappings(String),
    /// Metadata promotion prefix configuration was invalid.
    #[error("invalid metadata promotion prefixes: {0}")]
    InvalidMetadataPromotionPrefixes(String),
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

/// Completes a bare OTLP/HTTP base URL with the standard trace signal path.
#[doc(hidden)]
pub fn resolve_http_trace_endpoint(endpoint: &str) -> Cow<'_, str> {
    let Ok(mut parsed) = reqwest::Url::parse(endpoint) else {
        return Cow::Borrowed(endpoint);
    };

    // A trailing slash deliberately selects the collector root. A bare authority
    // is the only form that receives the conventional OTLP trace path.
    let has_explicit_root_path = endpoint
        .split(['?', '#'])
        .next()
        .is_some_and(|url| url.ends_with('/'));
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.path() != "/"
        || has_explicit_root_path
    {
        return Cow::Borrowed(endpoint);
    }
    parsed.set_path("/v1/traces");
    Cow::Owned(parsed.into())
}

/// Configuration for the OpenTelemetry subscriber.
#[derive(Debug, Clone)]
pub struct OpenTelemetryConfig {
    otel_type: OpenTelemetryType,
    endpoint: String,
    headers: HashMap<String, String>,
    resource_attributes: HashMap<String, String>,
    service_name: String,
    service_namespace: Option<String>,
    service_version: Option<String>,
    instrumentation_scope: String,
    mark_projection: MarkProjection,
    mark_exclude_names: Vec<String>,
    attribute_mappings: Vec<OtlpAttributeMapping>,
    promote_metadata_prefixes: Vec<String>,
    timeout: Duration,
    transport: OtlpTransport,
    max_queue_size: Option<usize>,
    max_export_batch_size: Option<usize>,
    scheduled_delay: Option<Duration>,
}

impl OpenTelemetryConfig {
    fn default_values() -> Self {
        Self {
            otel_type: OpenTelemetryType::Full,
            endpoint: String::new(),
            headers: HashMap::new(),
            resource_attributes: HashMap::new(),
            service_name: "unknown_service".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "opentelemetry".to_string(),
            mark_projection: MarkProjection::default(),
            mark_exclude_names: default_mark_exclude_names(),
            attribute_mappings: Vec::new(),
            promote_metadata_prefixes: Vec::new(),
            timeout: Duration::from_secs(3),
            transport: OtlpTransport::HttpBinary,
            max_queue_size: None,
            max_export_batch_size: None,
            scheduled_delay: None,
        }
    }

    /// Creates a typed OpenTelemetry exporter for a required OTLP endpoint.
    pub fn new(otel_type: OpenTelemetryType, endpoint: impl Into<String>) -> Self {
        Self {
            otel_type,
            endpoint: endpoint.into(),
            ..Self::default_values()
        }
    }

    /// Creates an HTTP OTLP config for the given service name.
    #[cfg(test)]
    pub(crate) fn http_binary(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            transport: OtlpTransport::HttpBinary,
            ..Self::default_values()
        }
    }

    /// Creates a gRPC OTLP config for the given service name.
    #[cfg(test)]
    pub(crate) fn grpc(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            transport: OtlpTransport::Grpc,
            ..Self::default_values()
        }
    }

    /// Overrides the OTLP endpoint. If unset, exporter defaults and OTEL_* env vars apply.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
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

    /// Adds a header/metadata entry for the exporter.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    #[cfg(test)]
    pub(crate) fn header(&self, key: &str) -> Option<&str> {
        self.headers.get(key).map(String::as_str)
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

    /// Overrides the batch processor queue size for this endpoint.
    pub(crate) fn with_max_queue_size(mut self, max_queue_size: usize) -> Self {
        self.max_queue_size = Some(max_queue_size);
        self
    }

    /// Overrides the maximum export batch size for this endpoint.
    pub(crate) fn with_max_export_batch_size(mut self, max_export_batch_size: usize) -> Self {
        self.max_export_batch_size = Some(max_export_batch_size);
        self
    }

    /// Overrides the maximum delay before exporting a non-full batch.
    pub(crate) fn with_scheduled_delay(mut self, scheduled_delay: Duration) -> Self {
        self.scheduled_delay = Some(scheduled_delay);
        self
    }

    #[cfg(test)]
    pub(crate) fn batch_overrides(&self) -> (Option<usize>, Option<usize>, Option<Duration>) {
        (
            self.max_queue_size,
            self.max_export_batch_size,
            self.scheduled_delay,
        )
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

    /// Excludes named marks from tool projection.
    pub fn with_mark_exclude_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.mark_exclude_names = names.into_iter().map(Into::into).collect();
        self
    }

    /// Adds a projected OpenTelemetry attribute alias.
    pub fn with_attribute_mapping(
        mut self,
        key: impl Into<String>,
        alias: impl Into<String>,
    ) -> Self {
        self.attribute_mappings
            .push(OtlpAttributeMapping::new(key, alias));
        self
    }

    /// Replaces projected OpenTelemetry attribute aliases.
    pub fn with_attribute_mappings<I>(mut self, mappings: I) -> Self
    where
        I: IntoIterator<Item = OtlpAttributeMapping>,
    {
        self.attribute_mappings = mappings.into_iter().collect();
        self
    }

    /// Selects literal Event metadata prefixes copied to OTLP attributes.
    pub fn with_promote_metadata_prefixes<I, S>(mut self, prefixes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.promote_metadata_prefixes = prefixes.into_iter().map(Into::into).collect();
        self
    }
}

#[cfg(test)]
impl Default for OpenTelemetryConfig {
    fn default() -> Self {
        Self::default_values()
    }
}

/// OpenTelemetry-backed NeMo Relay subscriber.
#[derive(Clone)]
pub struct OpenTelemetrySubscriber {
    inner: Arc<Inner>,
}

/// Options for constructing an OpenTelemetry subscriber from an existing tracer provider.
#[derive(Debug, Clone)]
pub struct OpenTelemetrySubscriberOptions {
    /// How mark events are projected into the trace.
    pub mark_projection: MarkProjection,
    /// Mark names excluded from tool projection.
    pub mark_exclude_names: Vec<String>,
    /// Typed OTLP attributes copied to alias keys.
    pub attribute_mappings: Vec<OtlpAttributeMapping>,
    /// Literal Event metadata prefixes copied to OTLP attributes.
    pub promote_metadata_prefixes: Vec<String>,
}

impl Default for OpenTelemetrySubscriberOptions {
    fn default() -> Self {
        Self {
            mark_projection: MarkProjection::default(),
            mark_exclude_names: default_mark_exclude_names(),
            attribute_mappings: Vec::new(),
            promote_metadata_prefixes: Vec::new(),
        }
    }
}

struct Inner {
    // Keep `provider` before `_runtime`: the provider must be dropped before
    // `ExporterRuntime` joins and tears down its Tokio runtime. Do not reorder
    // these fields.
    provider: SdkTracerProvider,
    runtime_diagnostics: SignalRuntimeDiagnostics,
    subscriber: EventSubscriberFn,
    _runtime: Option<ExporterRuntime>,
}

struct ExporterRuntime {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
    runtime_diagnostics: SignalRuntimeDiagnostics,
}

impl Drop for ExporterRuntime {
    fn drop(&mut self) {
        self.stop.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl OpenTelemetrySubscriber {
    /// Builds a subscriber backed by a new OTLP tracer provider.
    pub fn new(config: OpenTelemetryConfig) -> Result<Self> {
        Self::new_with_runtime_diagnostics(config, None)
    }

    pub(crate) fn new_for_plugin(
        config: OpenTelemetryConfig,
        endpoint_index: usize,
    ) -> Result<Self> {
        Self::new_with_runtime_diagnostics(
            config,
            Some(format!("opentelemetry.traces[{endpoint_index}].endpoint")),
        )
    }

    fn new_with_runtime_diagnostics(
        config: OpenTelemetryConfig,
        diagnostic_field: Option<String>,
    ) -> Result<Self> {
        if config.endpoint.trim().is_empty() {
            return Err(OpenTelemetryError::ExporterBuild(
                "endpoint must be a nonblank string".to_string(),
            ));
        }
        validate_attribute_mappings(&config.attribute_mappings)
            .map_err(OpenTelemetryError::InvalidAttributeMappings)?;
        validate_metadata_promotion_prefixes(&config.promote_metadata_prefixes)
            .map_err(OpenTelemetryError::InvalidMetadataPromotionPrefixes)?;
        reject_global_header_environment()?;
        validate_headers(&config.headers)?;
        let runtime_diagnostics = SignalRuntimeDiagnostics::new(diagnostic_field);
        let (provider, runtime) =
            build_owned_tracer_provider(config.clone(), runtime_diagnostics.clone())?;
        Ok(Self::from_tracer_provider_with_scope_and_type(
            provider,
            config.instrumentation_scope,
            config.otel_type,
            config.mark_projection,
            config.mark_exclude_names,
            config.attribute_mappings,
            config.promote_metadata_prefixes,
            Some(runtime),
        ))
    }

    /// Builds a subscriber from an already-configured tracer provider.
    pub fn from_tracer_provider(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
    ) -> Self {
        Self::from_tracer_provider_with_type(
            provider,
            instrumentation_scope,
            OpenTelemetryType::Full,
        )
    }

    /// Builds a typed subscriber from an already-configured tracer provider.
    pub fn from_tracer_provider_with_type(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
        otel_type: OpenTelemetryType,
    ) -> Self {
        let instrumentation_scope = instrumentation_scope.into();
        Self::from_tracer_provider_with_scope_and_type(
            provider,
            instrumentation_scope,
            otel_type,
            MarkProjection::default(),
            default_mark_exclude_names(),
            Vec::new(),
            Vec::new(),
            None,
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
        Self::from_tracer_provider_with_options(
            provider,
            instrumentation_scope,
            OpenTelemetrySubscriberOptions {
                attribute_mappings,
                ..Default::default()
            },
        )
    }

    /// Builds a subscriber from a tracer provider with composable projection options.
    pub fn from_tracer_provider_with_options(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
        options: OpenTelemetrySubscriberOptions,
    ) -> Result<Self> {
        validate_attribute_mappings(&options.attribute_mappings)
            .map_err(OpenTelemetryError::InvalidAttributeMappings)?;
        validate_metadata_promotion_prefixes(&options.promote_metadata_prefixes)
            .map_err(OpenTelemetryError::InvalidMetadataPromotionPrefixes)?;
        Ok(Self::from_tracer_provider_with_scope_and_type(
            provider,
            instrumentation_scope.into(),
            OpenTelemetryType::Full,
            options.mark_projection,
            options.mark_exclude_names,
            options.attribute_mappings,
            options.promote_metadata_prefixes,
            None,
        ))
    }

    /// Builds a typed subscriber from a tracer provider with projection options.
    pub fn from_tracer_provider_with_type_and_options(
        provider: SdkTracerProvider,
        instrumentation_scope: impl Into<String>,
        otel_type: OpenTelemetryType,
        options: OpenTelemetrySubscriberOptions,
    ) -> Result<Self> {
        validate_attribute_mappings(&options.attribute_mappings)
            .map_err(OpenTelemetryError::InvalidAttributeMappings)?;
        validate_metadata_promotion_prefixes(&options.promote_metadata_prefixes)
            .map_err(OpenTelemetryError::InvalidMetadataPromotionPrefixes)?;
        Ok(Self::from_tracer_provider_with_scope_and_type(
            provider,
            instrumentation_scope.into(),
            otel_type,
            options.mark_projection,
            options.mark_exclude_names,
            options.attribute_mappings,
            options.promote_metadata_prefixes,
            None,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn from_tracer_provider_with_scope_and_type(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        otel_type: OpenTelemetryType,
        mark_projection: MarkProjection,
        mark_exclude_names: Vec<String>,
        attribute_mappings: Vec<OtlpAttributeMapping>,
        promote_metadata_prefixes: Vec<String>,
        runtime: Option<ExporterRuntime>,
    ) -> Self {
        let runtime_diagnostics = runtime
            .as_ref()
            .map(|runtime| runtime.runtime_diagnostics.clone())
            .unwrap_or_else(|| SignalRuntimeDiagnostics::new(None));
        let processor = Arc::new(Mutex::new(
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider.clone(),
                instrumentation_scope,
                otel_type,
                mark_projection,
                mark_exclude_names,
                attribute_mappings,
                promote_metadata_prefixes,
                runtime_diagnostics.clone(),
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
                provider,
                runtime_diagnostics,
                subscriber,
                _runtime: runtime,
            }),
        }
    }

    /// Returns the raw NeMo Relay subscriber callback for custom registration flows.
    pub fn subscriber(&self) -> EventSubscriberFn {
        Arc::clone(&self.inner.subscriber)
    }

    /// Return a bounded snapshot of runtime diagnostics for this subscriber.
    pub fn runtime_diagnostics(&self) -> OpenTelemetryRuntimeDiagnostics {
        self.inner.runtime_diagnostics.snapshot()
    }

    /// Registers this subscriber globally with the NeMo Relay runtime.
    pub fn register(&self, name: &str) -> Result<()> {
        register_subscriber(name, self.subscriber())?;
        log::info!(
            target: "nemo_relay.observability",
            event = "exporter_registered",
            exporter = "opentelemetry",
            subscriber = name;
            "OpenTelemetry exporter registered"
        );
        Ok(())
    }

    /// Deregisters a previously-registered global subscriber by name.
    pub fn deregister(&self, name: &str) -> Result<bool> {
        let removed = deregister_subscriber(name)?;
        if removed {
            log::info!(
                target: "nemo_relay.observability",
                event = "subscriber_deregistered",
                subscriber = name;
                "Observability subscriber deregistered"
            );
        }
        Ok(removed)
    }

    /// Flushes finished spans through the underlying tracer provider.
    pub fn force_flush(&self) -> Result<()> {
        flush_subscribers()?;
        self.inner
            .provider
            .force_flush()
            .map_err(|error| OpenTelemetryError::TraceProvider(error.to_string()))
    }

    /// Shuts down the underlying tracer provider.
    ///
    /// Call `deregister(...)` first if the subscriber is still registered with NeMo Relay.
    pub fn shutdown(&self) -> Result<()> {
        let barrier_error = flush_subscribers().err().map(OpenTelemetryError::Core);
        let provider_result = self.shutdown_provider();
        if let Some(error) = barrier_error {
            return Err(error);
        }
        provider_result
    }

    pub(crate) fn shutdown_provider(&self) -> Result<()> {
        let provider_result = self.inner.provider.shutdown();
        if provider_result.is_ok() {
            log::info!(
                target: "nemo_relay.observability",
                event = "exporter_shutdown",
                exporter = "opentelemetry";
                "OpenTelemetry exporter shut down"
            );
        }
        normalize_shutdown_result(provider_result)
            .map_err(|error| OpenTelemetryError::TraceProvider(error.to_string()))
    }
}

fn build_owned_tracer_provider(
    config: OpenTelemetryConfig,
    runtime_diagnostics: SignalRuntimeDiagnostics,
) -> Result<(SdkTracerProvider, ExporterRuntime)> {
    let (result_sender, result_receiver) = mpsc::sync_channel(1);
    let (stop_sender, stop_receiver) = mpsc::channel();
    let provider_diagnostics = runtime_diagnostics.clone();
    let runtime_thread = thread::Builder::new()
        .name("nemo-relay-otlp".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = result_sender
                        .send(Err(OpenTelemetryError::ExporterBuild(error.to_string())));
                    return;
                }
            };
            let provider = {
                let _guard = runtime.enter();
                build_tracer_provider(&config, provider_diagnostics)
            };
            let keep_runtime_alive = provider.is_ok();
            let _ = result_sender.send(provider);
            if keep_runtime_alive {
                let _ = stop_receiver.recv();
            }
        })
        .map_err(|error| OpenTelemetryError::ExporterBuild(error.to_string()))?;
    let provider = result_receiver.recv().map_err(|error| {
        OpenTelemetryError::ExporterBuild(format!("exporter runtime stopped unexpectedly: {error}"))
    })??;
    Ok((
        provider,
        ExporterRuntime {
            stop: Some(stop_sender),
            thread: Some(runtime_thread),
            runtime_diagnostics,
        },
    ))
}

fn reject_global_header_environment() -> Result<()> {
    for variable in [
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ] {
        if std::env::var_os(variable).is_some_and(|value| !value.is_empty()) {
            return Err(OpenTelemetryError::GlobalHeaderEnvironmentUnsupported { variable });
        }
    }
    Ok(())
}

pub(crate) fn validate_headers(headers: &HashMap<String, String>) -> Result<()> {
    let mut normalized = HashSet::new();
    for (key, value) in headers {
        let normalized_key = key.to_ascii_lowercase();
        if !normalized.insert(normalized_key) {
            return Err(OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: "header names must be unique ignoring ASCII case".to_string(),
            });
        }
        reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
            OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: error.to_string(),
            }
        })?;
        reqwest::header::HeaderValue::from_str(value).map_err(|error| {
            OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: error.to_string(),
            }
        })?;
    }
    Ok(())
}

fn build_tracer_provider(
    config: &OpenTelemetryConfig,
    runtime_diagnostics: SignalRuntimeDiagnostics,
) -> Result<SdkTracerProvider> {
    let exporter = match config.transport {
        OtlpTransport::HttpBinary => {
            let mut builder = OtlpSpanExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_timeout(config.timeout);
            builder =
                builder.with_endpoint(resolve_http_trace_endpoint(&config.endpoint).into_owned());
            if !config.headers.is_empty() {
                builder = builder.with_headers(config.headers.clone());
            }
            builder
                .build()
                .map_err(|e| OpenTelemetryError::ExporterBuild(e.to_string()))?
        }
        OtlpTransport::Grpc => {
            let mut builder = OtlpSpanExporter::builder()
                .with_tonic()
                .with_protocol(Protocol::Grpc)
                .with_timeout(config.timeout);
            builder = builder.with_endpoint(config.endpoint.clone());
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
        .with_id_generator(RelayIdGenerator)
        .with_max_attributes_per_span(u32::MAX)
        .with_max_attributes_per_event(u32::MAX);

    let mut batch_config = BatchConfigBuilder::default();
    if let Some(max_queue_size) = config.max_queue_size {
        batch_config = batch_config.with_max_queue_size(max_queue_size);
    }
    if let Some(max_export_batch_size) = config.max_export_batch_size {
        batch_config = batch_config.with_max_export_batch_size(max_export_batch_size);
    }
    if let Some(scheduled_delay) = config.scheduled_delay {
        batch_config = batch_config.with_scheduled_delay(scheduled_delay);
    }
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        exporter,
        config.endpoint.clone(),
        runtime_diagnostics,
        batch_config.build(),
    );
    Ok(builder.with_span_processor(processor).build())
}

#[derive(Debug)]
struct CountingSpanExporter<E> {
    inner: E,
    accepted_spans: Arc<AtomicU64>,
    diagnostics: Arc<TraceDeliveryDiagnostics>,
}

impl<E: SpanExporter> SpanExporter for CountingSpanExporter<E> {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.accepted_spans
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        let result = self.inner.export(batch).await;
        if let Err(error) = &result {
            self.diagnostics.record_export_failure(error);
        } else {
            self.diagnostics.record_export_success();
        }
        result
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

#[derive(Debug)]
struct TraceDeliveryDiagnostics {
    endpoint: String,
    runtime_diagnostics: SignalRuntimeDiagnostics,
    export_failures: AtomicU64,
    unresolved_export_failure: AtomicBool,
}

impl TraceDeliveryDiagnostics {
    fn new(endpoint: String, runtime_diagnostics: SignalRuntimeDiagnostics) -> Self {
        Self {
            endpoint,
            runtime_diagnostics,
            export_failures: AtomicU64::new(0),
            unresolved_export_failure: AtomicBool::new(false),
        }
    }

    fn record_export_failure(&self, error: &impl std::fmt::Display) {
        self.export_failures.fetch_add(1, Ordering::Relaxed);
        self.unresolved_export_failure
            .store(true, Ordering::Relaxed);
        self.runtime_diagnostics.record(
            "otel.traces_export_failed",
            format!(
                "OpenTelemetry trace export to endpoint {} failed: {error}",
                self.endpoint
            ),
            1,
        );
    }

    fn record_export_success(&self) {
        self.unresolved_export_failure
            .store(false, Ordering::Relaxed);
    }

    fn unresolved_failure_summary(&self) -> Option<String> {
        self.unresolved_export_failure
            .load(Ordering::Relaxed)
            .then(|| {
                let failures = self.export_failures.load(Ordering::Relaxed);
                format!(
                    "otel.traces_export_failed ({failures}): export to endpoint {} has not recovered",
                    self.endpoint
                )
            })
    }
}

#[derive(Debug)]
struct DiagnosticBatchSpanProcessor {
    inner: BatchSpanProcessor,
    completed_spans: AtomicU64,
    accepted_spans: Arc<AtomicU64>,
    diagnostics: Arc<TraceDeliveryDiagnostics>,
    diagnostic_reported: AtomicBool,
}

impl DiagnosticBatchSpanProcessor {
    fn new_with_batch_config<E: SpanExporter + 'static>(
        exporter: E,
        endpoint: String,
        runtime_diagnostics: SignalRuntimeDiagnostics,
        batch_config: opentelemetry_sdk::trace::BatchConfig,
    ) -> Self {
        let accepted_spans = Arc::new(AtomicU64::new(0));
        let diagnostics = Arc::new(TraceDeliveryDiagnostics::new(endpoint, runtime_diagnostics));
        let exporter = CountingSpanExporter {
            inner: exporter,
            accepted_spans: Arc::clone(&accepted_spans),
            diagnostics: Arc::clone(&diagnostics),
        };
        Self {
            inner: BatchSpanProcessor::builder(exporter)
                .with_batch_config(batch_config)
                .build(),
            completed_spans: AtomicU64::new(0),
            accepted_spans,
            diagnostics,
            diagnostic_reported: AtomicBool::new(false),
        }
    }

    fn record_dropped_spans(&self) -> u64 {
        let dropped = self
            .completed_spans
            .load(Ordering::Relaxed)
            .saturating_sub(self.accepted_spans.load(Ordering::Relaxed));
        if dropped == 0 || self.diagnostic_reported.swap(true, Ordering::Relaxed) {
            return dropped;
        }
        self.diagnostics.runtime_diagnostics.record(
            "otel.spans_dropped",
            format!(
                "OpenTelemetry dropped {dropped} spans before export to endpoint {} because the batch queue was full",
                self.diagnostics.endpoint
            ),
            dropped,
        );
        dropped
    }
}

impl SpanProcessor for DiagnosticBatchSpanProcessor {
    fn on_start(&self, span: &mut Span, cx: &Context) {
        self.inner.on_start(span, cx);
    }

    fn on_end(&self, span: SpanData) {
        self.completed_spans.fetch_add(1, Ordering::Relaxed);
        self.inner.on_end(span);
    }

    fn force_flush(&self) -> OTelSdkResult {
        let result = self.inner.force_flush();
        if result.is_ok()
            && let Some(summary) = self.diagnostics.unresolved_failure_summary()
        {
            return Err(OTelSdkError::InternalFailure(summary));
        }
        result
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        let result = self.inner.shutdown_with_timeout(timeout);
        if result.is_ok() {
            let dropped = self.record_dropped_spans();
            let export_failure = self.diagnostics.unresolved_failure_summary();
            if (dropped > 0 || export_failure.is_some())
                && self.diagnostics.runtime_diagnostics.has_plugin_mirror()
            {
                let mut diagnostics = Vec::new();
                if dropped > 0 {
                    diagnostics.push(format!("otel.spans_dropped ({dropped})"));
                }
                if let Some(export_failure) = export_failure {
                    diagnostics.push(export_failure);
                }
                return Err(OTelSdkError::InternalFailure(format!(
                    "{OTEL_RUNTIME_DELIVERY_FAILURE_MARKER}: {}",
                    diagnostics.join(", ")
                )));
            }
        }
        result
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
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

pub(super) struct ActiveSpan {
    span: Span,
    span_context: SpanContext,
    start_model_name: Option<String>,
    projected_attributes: Vec<KeyValue>,
    projection_attribute_keys: HashSet<String>,
    start_promoted_metadata: Vec<KeyValue>,
    descendant_error_type: Option<String>,
    descendant_exception_type: Option<String>,
}

pub(super) struct OtelEventProcessor {
    pub(super) active_spans: HashMap<Uuid, ActiveSpan>,
    pub(super) completed_span_contexts: HashMap<Uuid, SpanContext>,
    pub(super) completed_span_order: VecDeque<Uuid>,
    #[cfg(test)]
    provider: SdkTracerProvider,
    tracer: SdkTracer,
    otel_type: OpenTelemetryType,
    mark_projection: MarkProjection,
    mark_exclude_names: Vec<String>,
    attribute_mappings: Vec<OtlpAttributeMapping>,
    promote_metadata_prefixes: Vec<String>,
    invalid_metric_count: u64,
    runtime_diagnostics: SignalRuntimeDiagnostics,
}

impl OtelEventProcessor {
    #[cfg(test)]
    fn new(provider: SdkTracerProvider, instrumentation_scope: String) -> Self {
        Self::new_with_mark_projection(provider, instrumentation_scope, MarkProjection::default())
    }

    #[cfg(test)]
    pub(super) fn new_openinference(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
    ) -> Self {
        Self::new_with_mark_projection_and_exclusions_and_mappings(
            provider,
            instrumentation_scope,
            OpenTelemetryType::OpenInference,
            MarkProjection::default(),
            default_mark_exclude_names(),
            Vec::new(),
        )
    }

    #[cfg(test)]
    pub(super) fn new_openinference_with_mark_projection(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        mark_projection: MarkProjection,
    ) -> Self {
        Self::new_with_mark_projection_and_exclusions_and_mappings(
            provider,
            instrumentation_scope,
            OpenTelemetryType::OpenInference,
            mark_projection,
            default_mark_exclude_names(),
            Vec::new(),
        )
    }

    #[cfg(test)]
    pub(super) fn new_openinference_with_mark_projection_and_exclusions(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        mark_projection: MarkProjection,
        mark_exclude_names: Vec<String>,
    ) -> Self {
        Self::new_with_mark_projection_and_exclusions_and_mappings(
            provider,
            instrumentation_scope,
            OpenTelemetryType::OpenInference,
            mark_projection,
            mark_exclude_names,
            Vec::new(),
        )
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
            OpenTelemetryType::Full,
            mark_projection,
            mark_exclude_names,
            Vec::new(),
        )
    }

    #[cfg(test)]
    fn new_with_mark_projection_and_exclusions_and_mappings(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        otel_type: OpenTelemetryType,
        mark_projection: MarkProjection,
        mark_exclude_names: Vec<String>,
        attribute_mappings: Vec<OtlpAttributeMapping>,
    ) -> Self {
        Self::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
            provider,
            instrumentation_scope,
            otel_type,
            mark_projection,
            mark_exclude_names,
            attribute_mappings,
            Vec::new(),
            SignalRuntimeDiagnostics::new(None),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
        provider: SdkTracerProvider,
        instrumentation_scope: String,
        otel_type: OpenTelemetryType,
        mark_projection: MarkProjection,
        mark_exclude_names: Vec<String>,
        attribute_mappings: Vec<OtlpAttributeMapping>,
        promote_metadata_prefixes: Vec<String>,
        runtime_diagnostics: SignalRuntimeDiagnostics,
    ) -> Self {
        let tracer = provider.tracer(instrumentation_scope);
        Self {
            active_spans: HashMap::new(),
            completed_span_contexts: HashMap::new(),
            completed_span_order: VecDeque::new(),
            #[cfg(test)]
            provider,
            tracer,
            otel_type,
            mark_projection,
            mark_exclude_names,
            attribute_mappings,
            promote_metadata_prefixes,
            invalid_metric_count: 0,
            runtime_diagnostics,
        }
    }

    pub(super) fn process(&mut self, event: &Event) {
        match event.scope_category() {
            Some(ScopeCategory::Start) => self.process_start(event),
            Some(ScopeCategory::End) => self.process_end(event),
            None => self.process_mark(event),
        }
    }

    #[cfg(test)]
    pub(super) fn force_flush(&self) -> Result<()> {
        self.provider
            .force_flush()
            .map_err(|e| OpenTelemetryError::TraceProvider(e.to_string()))
    }

    fn process_start(&mut self, event: &Event) {
        self.remove_completed_span_context(event.uuid());
        let parent_context = self.parent_context(event);
        let is_trace_root = !parent_context.span().span_context().is_valid();
        let start_model_name = model_name_for_llm_event(event);
        let span_name = match self.otel_type {
            OpenTelemetryType::Full => span_name(event),
            OpenTelemetryType::GenAi => super::otel_genai::span_name(event),
            OpenTelemetryType::OpenInference => super::openinference::span_name(event),
        };
        let span_kind = match self.otel_type {
            OpenTelemetryType::Full => span_kind(event),
            OpenTelemetryType::GenAi => super::otel_genai::span_kind(event),
            OpenTelemetryType::OpenInference => super::openinference::span_kind(event),
        };
        let mut span = with_relay_ids(event.uuid(), || {
            self.tracer
                .span_builder(span_name)
                .with_kind(span_kind)
                .with_start_time(to_system_time(*event.timestamp()))
                .start_with_context(&self.tracer, &parent_context)
        });
        let mut attributes = match self.otel_type {
            OpenTelemetryType::Full => start_attributes(event),
            OpenTelemetryType::GenAi => super::otel_genai::start_attributes(event),
            OpenTelemetryType::OpenInference => super::openinference::start_attributes(event),
        };
        if self.otel_type == OpenTelemetryType::Full && start_model_name.is_some() {
            attributes.retain(|attribute| attribute.key.as_str() != "nemo_relay.model_name");
        }
        if self.otel_type == OpenTelemetryType::OpenInference && start_model_name.is_some() {
            super::openinference::remove_start_model_name(&mut attributes);
        }
        if self.otel_type != OpenTelemetryType::GenAi && is_trace_root {
            push_session_identity_attributes(&mut attributes, event);
        }
        // Snapshot keys claimed by the projection so promoted metadata cannot
        // replace them when the span completes.
        let mut projection_attribute_keys = attributes
            .iter()
            .map(|attribute| attribute.key.as_str().to_string())
            .collect::<HashSet<_>>();
        if self.otel_type != OpenTelemetryType::GenAi {
            projection_attribute_keys.extend(
                self.attribute_mappings
                    .iter()
                    .map(|mapping| mapping.alias.clone()),
            );
        }
        let projected_attributes = if self.otel_type == OpenTelemetryType::GenAi {
            Vec::new()
        } else {
            attribute_mapping_inputs(&attributes, &self.attribute_mappings)
        };
        let mut start_promoted_metadata = Vec::new();
        self.promote_metadata(
            &mut start_promoted_metadata,
            event,
            &projection_attribute_keys,
        );
        span.set_attributes(attributes);
        let span_context = local_parent_span_context(span.span_context());
        self.active_spans.insert(
            event.uuid(),
            ActiveSpan {
                span,
                span_context,
                start_model_name,
                projected_attributes,
                projection_attribute_keys,
                start_promoted_metadata,
                descendant_error_type: None,
                descendant_exception_type: None,
            },
        );
    }

    fn process_end(&mut self, event: &Event) {
        let Some(mut active_span) = self.active_spans.remove(&event.uuid()) else {
            return;
        };
        self.record_completed_span_context(event.uuid(), active_span.span_context.clone());

        super::set_span_status_from_event_metadata(&mut active_span.span, event);
        let mut attributes = match self.otel_type {
            OpenTelemetryType::Full => end_attributes(event),
            OpenTelemetryType::GenAi => super::otel_genai::end_attributes(event),
            OpenTelemetryType::OpenInference => super::openinference::end_attributes(event),
        };
        let is_error = metadata_string(event, "otel.status_code") == Some("ERROR");
        let explicit_error_type = metadata_string(event, "error.type");
        let error_type = is_error.then(|| {
            explicit_error_type
                .map(ToOwned::to_owned)
                .or(active_span.descendant_error_type.take())
                .unwrap_or_else(|| "_OTHER".to_string())
        });
        let exception_type = is_error
            .then(|| {
                metadata_string(event, "exception.type")
                    .map(ToOwned::to_owned)
                    .or(active_span.descendant_exception_type.take())
            })
            .flatten();
        if matches!(
            self.otel_type,
            OpenTelemetryType::Full | OpenTelemetryType::GenAi
        ) && let Some(error_type) = error_type.as_ref()
        {
            attributes.retain(|attribute| attribute.key.as_str() != "error.type");
            attributes.push(KeyValue::new("error.type", error_type.clone()));
        }
        if let Some(exception_type) = exception_type.as_ref() {
            active_span.span.add_event_with_timestamp(
                "exception",
                to_system_time(*event.timestamp()),
                vec![KeyValue::new("exception.type", exception_type.clone())],
            );
        }
        let end_model_name =
            model_name_for_llm_event(event).or_else(|| active_span.start_model_name.take());
        if self.otel_type == OpenTelemetryType::Full
            && let Some(model_name) = end_model_name.clone()
        {
            attributes.push(KeyValue::new("nemo_relay.model_name", model_name));
        }
        if self.otel_type == OpenTelemetryType::OpenInference
            && let Some(model_name) = end_model_name
        {
            super::openinference::push_model_name(&mut attributes, model_name);
        }
        if self.otel_type != OpenTelemetryType::GenAi && !self.attribute_mappings.is_empty() {
            let mut projected_attributes = active_span.projected_attributes;
            projected_attributes.extend(attributes.iter().cloned());
            attributes.extend(attribute_mapping_aliases(
                &projected_attributes,
                &self.attribute_mappings,
            ));
        }
        // Scope-end key presence is authoritative even when the final value
        // cannot be represented as an OTLP attribute. Do not restore a stale
        // promoted value retained from scope start.
        active_span.projection_attribute_keys.extend(
            attributes
                .iter()
                .map(|attribute| attribute.key.as_str().to_string()),
        );
        let end_metadata = event.metadata().and_then(crate::json::Json::as_object);
        active_span.start_promoted_metadata.retain(|attribute| {
            let key = attribute.key.as_str();
            !active_span.projection_attribute_keys.contains(key)
                && !end_metadata.is_some_and(|metadata| metadata.contains_key(key))
        });
        attributes.extend(active_span.start_promoted_metadata);
        self.promote_metadata(
            &mut attributes,
            event,
            &active_span.projection_attribute_keys,
        );
        if is_error && let Some(parent_span) = self.find_parent_span_mut(event) {
            if parent_span.descendant_error_type.is_none() {
                parent_span.descendant_error_type = error_type;
            }
            if parent_span.descendant_exception_type.is_none() {
                parent_span.descendant_exception_type = exception_type;
            }
        }
        active_span.span.set_attributes(attributes);
        active_span
            .span
            .end_with_timestamp(to_system_time(*event.timestamp()));
    }

    fn process_mark(&mut self, event: &Event) {
        match classify_metric_mark(event) {
            MetricMarkClassification::NotMetric => {}
            MetricMarkClassification::Valid(_) => return,
            MetricMarkClassification::Invalid(error) => {
                self.invalid_metric_count = self.invalid_metric_count.saturating_add(1);
                let diagnostic_count = self.runtime_diagnostics.record(
                    "otel.metric_mark_invalid",
                    format!(
                        "OpenTelemetry metric mark {:?} was dropped atomically: {error}",
                        event.name()
                    ),
                    1,
                );
                if should_relog_runtime_diagnostic(diagnostic_count) {
                    log::warn!(
                        target: "nemo_relay.observability",
                        event = "otel_metric_mark_rejected",
                        mark_name = event.name();
                        "OpenTelemetry metric mark was dropped atomically: {error}"
                    );
                }
                return;
            }
        }
        if self.otel_type == OpenTelemetryType::GenAi {
            return;
        }
        if effective_mark_projection(event, self.mark_projection, &self.mark_exclude_names)
            == MarkProjection::Tool
        {
            self.process_mark_as_tool(event);
            return;
        }
        let mark_name = event.name().to_string();
        let timestamp = to_system_time(*event.timestamp());
        let mut attributes = self.mark_attributes(event);
        if event.name() == "session.start" {
            push_session_identity_attributes(&mut attributes, event);
        }

        if self.find_parent_span(event).is_some() {
            apply_attribute_mappings(&mut attributes, &self.attribute_mappings);
            self.promote_metadata(&mut attributes, event, &HashSet::new());
            let parent_span = self
                .find_parent_span_mut(event)
                .expect("parent span was present during mark projection");
            parent_span
                .span
                .add_event_with_timestamp(mark_name, timestamp, attributes);
            return;
        }

        let mut span = with_relay_ids(event.uuid(), || {
            self.tracer
                .span_builder(format!("mark:{mark_name}"))
                .with_kind(SpanKind::Internal)
                .with_start_time(timestamp)
                .start_with_context(&self.tracer, &self.parent_context(event))
        });
        if self.otel_type == OpenTelemetryType::OpenInference {
            super::openinference::push_orphan_mark_attributes(&mut attributes);
        } else {
            attributes.push(KeyValue::new("nemo_relay.mark.orphan", true));
        }
        apply_attribute_mappings(&mut attributes, &self.attribute_mappings);
        self.promote_metadata(&mut attributes, event, &HashSet::new());
        span.set_attributes(attributes);
        span.end_with_timestamp(timestamp);
    }

    fn process_mark_as_tool(&mut self, event: &Event) {
        let timestamp = to_system_time(*event.timestamp());
        let orphan = self.find_parent_span(event).is_none();
        let mut attributes = self.mark_attributes(event);
        if event.name() == "session.start" {
            push_session_identity_attributes(&mut attributes, event);
        }
        attributes.push(KeyValue::new("nemo_relay.mark.projection", "tool"));
        if self.otel_type == OpenTelemetryType::OpenInference {
            super::openinference::push_tool_mark_attributes(&mut attributes, event);
        } else {
            attributes.push(KeyValue::new("nemo_relay.scope_type", "tool"));
        }
        if orphan {
            attributes.push(KeyValue::new("nemo_relay.mark.orphan", true));
        }
        apply_attribute_mappings(&mut attributes, &self.attribute_mappings);
        self.promote_metadata(&mut attributes, event, &HashSet::new());

        let mut span = with_relay_ids(event.uuid(), || {
            self.tracer
                .span_builder(format!("mark:{}", event.name()))
                .with_kind(SpanKind::Internal)
                .with_start_time(timestamp)
                .start_with_context(&self.tracer, &self.parent_context(event))
        });
        span.set_attributes(attributes);
        span.end_with_timestamp(timestamp);
    }

    // Report unsupported metadata by key without exposing its value or
    // interrupting Event export.
    fn promote_metadata(
        &self,
        attributes: &mut Vec<KeyValue>,
        event: &Event,
        protected_keys: &HashSet<String>,
    ) {
        let mut issues = promote_event_metadata_attributes(
            attributes,
            event,
            &self.promote_metadata_prefixes,
            protected_keys,
        );

        issues.sort_by(|left, right| left.key.cmp(&right.key));
        for issue in issues {
            let diagnostic_code =
                format!("otel.metadata_promotion_value_unsupported.{}", issue.key);
            let diagnostic_count = self.runtime_diagnostics.record(
                diagnostic_code,
                format!(
                    "OpenTelemetry metadata attribute {:?} was not promoted: {}",
                    issue.key, issue.reason
                ),
                1,
            );
            if should_relog_runtime_diagnostic(diagnostic_count) {
                log::warn!(
                    target: "nemo_relay.observability",
                    event = "otel_metadata_promotion_value_unsupported",
                    metadata_key = issue.key.as_str();
                    "OpenTelemetry metadata attribute was not promoted: {}",
                    issue.reason
                );
            }
        }
    }
    fn mark_attributes(&self, event: &Event) -> Vec<KeyValue> {
        match self.otel_type {
            OpenTelemetryType::Full => mark_attributes(event),
            OpenTelemetryType::OpenInference => super::openinference::mark_attributes(event),
            OpenTelemetryType::GenAi => Vec::new(),
        }
    }

    fn parent_context(&self, event: &Event) -> Context {
        if let Some(active_span) = self.find_parent_span(event) {
            return Context::new().with_remote_span_context(active_span.span_context.clone());
        }
        if let Some(span_context) = event
            .parent_uuid()
            .and_then(|uuid| self.completed_span_contexts.get(&uuid))
        {
            return Context::new().with_remote_span_context(span_context.clone());
        }
        let Some(parent_uuid) = event.parent_uuid() else {
            return Context::new();
        };
        let stack = current_scope_stack();
        let stack = stack.read().expect("scope stack lock poisoned");
        if !stack.is_propagated_parent(parent_uuid) {
            return Context::new();
        }
        let root_uuid = stack.root_uuid();
        Context::new().with_remote_span_context(SpanContext::new(
            relay_trace_id(root_uuid),
            relay_span_id(parent_uuid),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        ))
    }

    fn parent_span_uuid(&self, event: &Event) -> Option<Uuid> {
        let parent_uuid = event.parent_uuid()?;
        self.active_spans
            .contains_key(&parent_uuid)
            .then_some(parent_uuid)
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

fn metadata_string<'a>(event: &'a Event, key: &str) -> Option<&'a str> {
    event.metadata()?.get(key)?.as_str()
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
    push_tool_result_annotation_attribute(&mut attributes, event);
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
    crate::observability::push_common_optimization_attributes(attributes, summary);
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

pub(super) fn to_system_time(timestamp: DateTime<Utc>) -> SystemTime {
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
