// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry log export for sanitized Relay mark events.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use opentelemetry::logs::{AnyValue, LogRecord as _, Logger as _, LoggerProvider as _, Severity};
use opentelemetry::trace::{SpanContext, TraceFlags, TraceState};
use opentelemetry::{InstrumentationScope, Key};
use opentelemetry_otlp::{
    LogExporter as OtlpLogExporter, Protocol, WithExportConfig, WithHttpConfig, WithTonicConfig,
};
use opentelemetry_sdk::logs::{
    BatchConfigBuilder, BatchLogProcessor, LogBatch, LogExporter, LogProcessor, SdkLogRecord,
    SdkLogger, SdkLoggerProvider,
};
use opentelemetry_sdk::{Resource, error::OTelSdkResult};
use serde_json::{Map, Value as Json};
use uuid::Uuid;

use crate::api::event::{ATOF_VERSION, Event, LOG_SEVERITY_METADATA_KEY, LogSeverity};
use crate::api::runtime::EventSubscriberFn;
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::observability::{relay_span_id, relay_trace_id};
use crate::plugin::OTEL_RUNTIME_DELIVERY_FAILURE_MARKER;

use super::OpenTelemetryRuntimeDiagnostics;
use super::otel::{
    DEFAULT_COMPLETED_SPAN_CONTEXT_TTL, OpenTelemetryError, OtlpTransport, Result,
    normalize_shutdown_result,
};
use super::otel_signal::{
    MetricMarkClassification, SignalExporterRuntime, SignalRuntimeDiagnostics, build_grpc_metadata,
    build_in_owned_runtime, classify_metric_mark, reject_signal_header_environment,
    resolve_header_env, resolve_http_signal_endpoint, should_relog_runtime_diagnostic,
    signal_resource, validate_signal_headers,
};

const DEFAULT_MAX_QUEUE_SIZE: usize = 2_048;
const DEFAULT_MAX_EXPORT_BATCH_SIZE: usize = 512;
const DEFAULT_SCHEDULED_DELAY: Duration = Duration::from_secs(1);

/// Configuration for an OTLP log subscriber.
#[derive(Debug, Clone)]
pub struct OpenTelemetryLogConfig {
    endpoint: String,
    headers: HashMap<String, String>,
    header_env: HashMap<String, String>,
    resource_attributes: HashMap<String, String>,
    service_name: String,
    service_namespace: Option<String>,
    service_version: Option<String>,
    instrumentation_scope: String,
    timeout: Duration,
    transport: OtlpTransport,
    minimum_severity: LogSeverity,
    max_queue_size: usize,
    max_export_batch_size: usize,
    scheduled_delay: Duration,
    completed_span_context_ttl: Duration,
    diagnostic_field: Option<String>,
}

impl OpenTelemetryLogConfig {
    /// Create a log exporter for a required OTLP endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            headers: HashMap::new(),
            header_env: HashMap::new(),
            resource_attributes: HashMap::new(),
            service_name: "unknown_service".to_string(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: "opentelemetry".to_string(),
            timeout: Duration::from_secs(3),
            transport: OtlpTransport::HttpBinary,
            minimum_severity: LogSeverity::Info,
            max_queue_size: DEFAULT_MAX_QUEUE_SIZE,
            max_export_batch_size: DEFAULT_MAX_EXPORT_BATCH_SIZE,
            scheduled_delay: DEFAULT_SCHEDULED_DELAY,
            completed_span_context_ttl: DEFAULT_COMPLETED_SPAN_CONTEXT_TTL,
            diagnostic_field: None,
        }
    }

    /// Select the OTLP transport.
    pub fn with_transport(mut self, transport: OtlpTransport) -> Self {
        self.transport = transport;
        self
    }

    /// Add an exporter header or gRPC metadata entry.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Map an exporter header name to the environment variable supplying its value.
    pub fn with_header_env(mut self, key: impl Into<String>, variable: impl Into<String>) -> Self {
        self.header_env.insert(key.into(), variable.into());
        self
    }

    /// Add an OpenTelemetry resource attribute.
    pub fn with_resource_attribute(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.resource_attributes.insert(key.into(), value.into());
        self
    }

    /// Set the `service.name` resource attribute.
    pub fn with_service_name(mut self, service_name: impl Into<String>) -> Self {
        self.service_name = service_name.into();
        self
    }

    /// Set the optional `service.namespace` resource attribute.
    pub fn with_service_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.service_namespace = Some(namespace.into());
        self
    }

    /// Set the optional `service.version` resource attribute.
    pub fn with_service_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = Some(version.into());
        self
    }

    /// Set the instrumentation scope name.
    pub fn with_instrumentation_scope(mut self, scope: impl Into<String>) -> Self {
        self.instrumentation_scope = scope.into();
        self
    }

    /// Set the OTLP request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the minimum mark severity exported as a log.
    pub fn with_minimum_severity(mut self, severity: LogSeverity) -> Self {
        self.minimum_severity = severity;
        self
    }

    /// Set the maximum queued log records.
    pub fn with_max_queue_size(mut self, max_queue_size: usize) -> Self {
        self.max_queue_size = max_queue_size;
        self
    }

    /// Set the maximum records in one export batch.
    pub fn with_max_export_batch_size(mut self, max_export_batch_size: usize) -> Self {
        self.max_export_batch_size = max_export_batch_size;
        self
    }

    /// Set the maximum delay before exporting a partial batch.
    pub fn with_scheduled_delay(mut self, delay: Duration) -> Self {
        self.scheduled_delay = delay;
        self
    }

    /// Sets how long completed scopes retain log parent context for late marks.
    ///
    /// The value must be greater than zero. Subscriber construction fails with
    /// [`OpenTelemetryError::ExporterBuild`] when the TTL is zero.
    pub fn with_completed_span_context_ttl(mut self, ttl: Duration) -> Self {
        self.completed_span_context_ttl = ttl;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.endpoint.trim().is_empty() {
            return Err(OpenTelemetryError::ExporterBuild(
                "endpoint must be a nonblank string".to_string(),
            ));
        }
        if self.timeout.is_zero() {
            return Err(OpenTelemetryError::ExporterBuild(
                "timeout must be greater than 0".to_string(),
            ));
        }
        if self.max_queue_size == 0 {
            return Err(OpenTelemetryError::ExporterBuild(
                "max_queue_size must be greater than 0".to_string(),
            ));
        }
        if self.max_export_batch_size == 0 || self.max_export_batch_size > self.max_queue_size {
            return Err(OpenTelemetryError::ExporterBuild(
                "max_export_batch_size must be greater than 0 and no greater than max_queue_size"
                    .to_string(),
            ));
        }
        if self.scheduled_delay.is_zero() {
            return Err(OpenTelemetryError::ExporterBuild(
                "scheduled_delay must be greater than 0".to_string(),
            ));
        }
        if self.completed_span_context_ttl.is_zero() {
            return Err(OpenTelemetryError::ExporterBuild(
                "completed_span_context_ttl must be greater than 0".to_string(),
            ));
        }
        reject_signal_header_environment("OTEL_EXPORTER_OTLP_LOGS_HEADERS")?;
        validate_signal_headers(&self.headers)
    }
}

/// Resolve an OTLP/HTTP endpoint for the logs signal.
pub fn resolve_http_log_endpoint(endpoint: &str) -> Cow<'_, str> {
    resolve_http_signal_endpoint(endpoint, "logs")
}

/// OpenTelemetry log-backed Relay event subscriber.
#[derive(Clone)]
pub struct OpenTelemetryLogSubscriber {
    inner: Arc<LogSubscriberInner>,
}

struct LogSubscriberInner {
    // Drop the processor and logger before the provider, then stop its runtime.
    _processor: Arc<Mutex<LogEventProcessor>>,
    provider: SdkLoggerProvider,
    delivery_diagnostics: Arc<LogDeliveryDiagnostics>,
    runtime_diagnostics: SignalRuntimeDiagnostics,
    subscriber: EventSubscriberFn,
    _runtime: SignalExporterRuntime,
}

impl OpenTelemetryLogSubscriber {
    /// Build an OTLP log subscriber with an independently owned provider.
    pub fn new(config: OpenTelemetryLogConfig) -> Result<Self> {
        Self::new_with_runtime_diagnostics(config)
    }

    pub(crate) fn new_for_plugin(
        mut config: OpenTelemetryLogConfig,
        endpoint_index: usize,
    ) -> Result<Self> {
        config.diagnostic_field = Some(format!(
            "opentelemetry.logs.endpoints[{endpoint_index}].endpoint"
        ));
        Self::new_with_runtime_diagnostics(config)
    }

    fn new_with_runtime_diagnostics(mut config: OpenTelemetryLogConfig) -> Result<Self> {
        config.validate()?;
        config.headers = resolve_header_env(&config.headers, &config.header_env)?;
        validate_signal_headers(&config.headers)?;
        let minimum_severity = config.minimum_severity;
        let completed_span_context_ttl = config.completed_span_context_ttl;
        let instrumentation_scope = config.instrumentation_scope.clone();
        let runtime_diagnostics = SignalRuntimeDiagnostics::new(config.diagnostic_field.clone());
        let delivery_diagnostics = Arc::new(LogDeliveryDiagnostics::new(
            config.endpoint.clone(),
            runtime_diagnostics.clone(),
        ));
        let provider_diagnostics = Arc::clone(&delivery_diagnostics);
        let (provider, runtime) = build_in_owned_runtime("nemo-relay-otlp-logs", move || {
            build_log_provider(&config, provider_diagnostics)
        })?;
        let logger = provider.logger(instrumentation_scope);
        let processor = Arc::new(Mutex::new(LogEventProcessor::new_with_runtime_diagnostics(
            logger,
            minimum_severity,
            completed_span_context_ttl,
            runtime_diagnostics.clone(),
        )));
        let callback_processor = Arc::clone(&processor);
        let callback_recovery_warned = Arc::new(AtomicBool::new(false));
        let callback_recovery_warned_for_callback = Arc::clone(&callback_recovery_warned);
        let subscriber: EventSubscriberFn = Arc::new(move |event| {
            let mut processor = match callback_processor.lock() {
                Ok(processor) => processor,
                Err(poisoned) => {
                    if !callback_recovery_warned_for_callback.swap(true, Ordering::Relaxed) {
                        log::warn!(
                            target: "nemo_relay.observability",
                            event = "otel_log_processor_lock_recovered";
                            "OpenTelemetry log subscriber recovered a poisoned processor lock"
                        );
                    }
                    poisoned.into_inner()
                }
            };
            processor.process(event);
        });
        Ok(Self {
            inner: Arc::new(LogSubscriberInner {
                _processor: processor,
                provider,
                delivery_diagnostics,
                runtime_diagnostics,
                subscriber,
                _runtime: runtime,
            }),
        })
    }

    /// Return the raw Relay subscriber callback.
    pub fn subscriber(&self) -> EventSubscriberFn {
        Arc::clone(&self.inner.subscriber)
    }

    /// Return a bounded snapshot of runtime diagnostics for this subscriber.
    pub fn runtime_diagnostics(&self) -> OpenTelemetryRuntimeDiagnostics {
        self.inner.runtime_diagnostics.snapshot()
    }

    /// Register the subscriber globally.
    pub fn register(&self, name: &str) -> Result<()> {
        register_subscriber(name, self.subscriber())?;
        Ok(())
    }

    /// Deregister a previously registered subscriber.
    pub fn deregister(&self, name: &str) -> Result<bool> {
        Ok(deregister_subscriber(name)?)
    }

    /// Flush queued Relay events and the OTLP log processor.
    ///
    /// After a successful flush, runtime diagnostics include queue drops observed so far.
    pub fn force_flush(&self) -> Result<()> {
        flush_subscribers()?;
        self.inner
            .provider
            .force_flush()
            .map_err(|error| OpenTelemetryError::LogProvider(error.to_string()))
    }

    /// Shut down the OTLP logger provider.
    ///
    /// Deregister this subscriber before calling shutdown.
    pub fn shutdown(&self) -> Result<()> {
        let barrier = flush_subscribers().map_err(OpenTelemetryError::Core);
        let provider = normalize_shutdown_result(self.inner.provider.shutdown())
            .map_err(|error| OpenTelemetryError::LogProvider(error.to_string()));
        barrier.and(provider)
    }

    pub(crate) fn shutdown_provider(&self) -> Result<()> {
        normalize_shutdown_result(self.inner.provider.shutdown())
            .map_err(|error| OpenTelemetryError::LogProvider(error.to_string()))
    }

    pub(crate) fn delivery_failure_summary(&self) -> Option<String> {
        self.inner.delivery_diagnostics.failure_summary()
    }
}

fn build_log_provider(
    config: &OpenTelemetryLogConfig,
    diagnostics: Arc<LogDeliveryDiagnostics>,
) -> Result<SdkLoggerProvider> {
    let exporter = match config.transport {
        OtlpTransport::HttpBinary => {
            let mut builder = OtlpLogExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_timeout(config.timeout)
                .with_endpoint(resolve_http_log_endpoint(&config.endpoint).into_owned());
            if !config.headers.is_empty() {
                builder = builder.with_headers(config.headers.clone());
            }
            builder
                .build()
                .map_err(|error| OpenTelemetryError::ExporterBuild(error.to_string()))?
        }
        OtlpTransport::Grpc => {
            let mut builder = OtlpLogExporter::builder()
                .with_tonic()
                .with_protocol(Protocol::Grpc)
                .with_timeout(config.timeout)
                .with_endpoint(config.endpoint.clone());
            if !config.headers.is_empty() {
                builder = builder.with_metadata(build_grpc_metadata(&config.headers)?);
            }
            builder
                .build()
                .map_err(|error| OpenTelemetryError::ExporterBuild(error.to_string()))?
        }
    };

    let batch_config = BatchConfigBuilder::default()
        .with_max_queue_size(config.max_queue_size)
        .with_max_export_batch_size(config.max_export_batch_size)
        .with_scheduled_delay(config.scheduled_delay)
        .build();
    let exporter = DiagnosticLogExporter {
        inner: exporter,
        diagnostics: Arc::clone(&diagnostics),
    };
    let processor = BatchLogProcessor::builder(exporter)
        .with_batch_config(batch_config)
        .build();
    let processor = DiagnosticBatchLogProcessor {
        inner: processor,
        diagnostics,
    };
    Ok(SdkLoggerProvider::builder()
        .with_resource(signal_resource(
            &config.service_name,
            config.service_namespace.as_deref(),
            config.service_version.as_deref(),
            &config.resource_attributes,
        ))
        .with_log_processor(processor)
        .build())
}

#[derive(Debug)]
struct LogDeliveryDiagnostics {
    emitted: AtomicU64,
    accepted: AtomicU64,
    export_failures: AtomicU64,
    reported_queue_drops: AtomicU64,
    endpoint: String,
    runtime_diagnostics: SignalRuntimeDiagnostics,
}

impl LogDeliveryDiagnostics {
    fn new(endpoint: String, runtime_diagnostics: SignalRuntimeDiagnostics) -> Self {
        Self {
            emitted: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            export_failures: AtomicU64::new(0),
            reported_queue_drops: AtomicU64::new(0),
            endpoint,
            runtime_diagnostics,
        }
    }

    fn record_export_failure(&self, error: &impl std::fmt::Display) {
        self.export_failures.fetch_add(1, Ordering::Relaxed);
        self.runtime_diagnostics.record(
            "otel.logs_export_failed",
            format!(
                "OpenTelemetry log export to endpoint {} failed: {error}",
                self.endpoint
            ),
            1,
        );
    }

    fn record_queue_drops(&self) -> u64 {
        let dropped = self
            .emitted
            .load(Ordering::Relaxed)
            .saturating_sub(self.accepted.load(Ordering::Relaxed));
        let mut reported = self.reported_queue_drops.load(Ordering::Relaxed);
        while dropped > reported {
            match self.reported_queue_drops.compare_exchange_weak(
                reported,
                dropped,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.runtime_diagnostics.record(
                        "otel.logs_dropped",
                        format!(
                            "OpenTelemetry dropped logs before export to endpoint {} because the batch queue was full",
                            self.endpoint
                        ),
                        dropped - reported,
                    );
                    break;
                }
                Err(current) => reported = current,
            }
        }
        dropped
    }

    fn failure_summary(&self) -> Option<String> {
        let dropped = self.record_queue_drops();
        let export_failures = self.export_failures.load(Ordering::Relaxed);
        (dropped > 0 || export_failures > 0).then(|| {
            format!("otel.logs_dropped ({dropped}), otel.logs_export_failed ({export_failures})")
        })
    }
}

#[derive(Debug)]
struct DiagnosticLogExporter<E> {
    inner: E,
    diagnostics: Arc<LogDeliveryDiagnostics>,
}

impl<E: LogExporter> LogExporter for DiagnosticLogExporter<E> {
    async fn export(&self, batch: LogBatch<'_>) -> OTelSdkResult {
        self.diagnostics
            .accepted
            .fetch_add(batch.iter().count() as u64, Ordering::Relaxed);
        let result = self.inner.export(batch).await;
        if let Err(error) = &result {
            self.diagnostics.record_export_failure(error);
        }
        result
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn event_enabled(&self, level: Severity, target: &str, name: Option<&str>) -> bool {
        self.inner.event_enabled(level, target, name)
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

#[derive(Debug)]
struct DiagnosticBatchLogProcessor {
    inner: BatchLogProcessor,
    diagnostics: Arc<LogDeliveryDiagnostics>,
}

impl LogProcessor for DiagnosticBatchLogProcessor {
    fn emit(&self, record: &mut SdkLogRecord, instrumentation: &InstrumentationScope) {
        self.diagnostics.emitted.fetch_add(1, Ordering::Relaxed);
        self.inner.emit(record, instrumentation);
    }

    fn force_flush(&self) -> OTelSdkResult {
        let result = self.inner.force_flush();
        if result.is_ok() {
            self.diagnostics.record_queue_drops();
        }
        result
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        let result = self.inner.shutdown_with_timeout(timeout);
        if result.is_ok() {
            let dropped = self.diagnostics.record_queue_drops();
            let export_failures = self.diagnostics.export_failures.load(Ordering::Relaxed);
            if (dropped > 0 || export_failures > 0)
                && self.diagnostics.runtime_diagnostics.has_plugin_mirror()
            {
                return Err(opentelemetry_sdk::error::OTelSdkError::InternalFailure(
                    format!(
                        "{OTEL_RUNTIME_DELIVERY_FAILURE_MARKER}: otel.logs_dropped ({dropped}), otel.logs_export_failed ({export_failures})"
                    ),
                ));
            }
        }
        result
    }

    fn event_enabled(&self, level: Severity, target: &str, name: Option<&str>) -> bool {
        self.inner.event_enabled(level, target, name)
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

struct CompletedLogContext {
    closed_at: DateTime<Utc>,
    span_context: SpanContext,
}

struct ScopeLineage {
    active: HashMap<Uuid, SpanContext>,
    completed: HashMap<Uuid, CompletedLogContext>,
    completed_expiry_index: BTreeMap<DateTime<Utc>, HashSet<Uuid>>,
}

impl ScopeLineage {
    fn new() -> Self {
        Self {
            active: HashMap::new(),
            completed: HashMap::new(),
            completed_expiry_index: BTreeMap::new(),
        }
    }

    fn process_start(&mut self, event: &Event) {
        self.remove_completed(event.uuid());
        let parent = self.parent_context(event);
        let trace_id = parent
            .as_ref()
            .map(SpanContext::trace_id)
            .unwrap_or_else(|| {
                relay_trace_id(event.propagation_root_uuid().unwrap_or(event.uuid()))
            });
        self.active.insert(
            event.uuid(),
            SpanContext::new(
                trace_id,
                relay_span_id(event.uuid()),
                TraceFlags::SAMPLED,
                false,
                TraceState::default(),
            ),
        );
    }

    fn process_end(&mut self, event: &Event) {
        let Some(context) = self.active.remove(&event.uuid()) else {
            return;
        };
        self.record_completed(event.uuid(), *event.timestamp(), context);
    }

    fn parent_context(&self, event: &Event) -> Option<SpanContext> {
        let parent_uuid = event.parent_uuid()?;
        if let Some(context) = self.active.get(&parent_uuid) {
            return Some(context.clone());
        }
        if let Some(context) = self.completed.get(&parent_uuid) {
            return Some(context.span_context.clone());
        }
        event.propagation_root_uuid().map(|root_uuid| {
            SpanContext::new(
                relay_trace_id(root_uuid),
                relay_span_id(parent_uuid),
                TraceFlags::SAMPLED,
                true,
                TraceState::default(),
            )
        })
    }

    fn remove_completed(&mut self, uuid: Uuid) {
        if let Some(context) = self.completed.remove(&uuid) {
            self.remove_expiry_index_entry(uuid, context.closed_at);
        }
    }

    fn remove_expiry_index_entry(&mut self, uuid: Uuid, closed_at: DateTime<Utc>) {
        let remove_bucket = self
            .completed_expiry_index
            .get_mut(&closed_at)
            .is_some_and(|uuids| {
                uuids.remove(&uuid);
                uuids.is_empty()
            });
        if remove_bucket {
            self.completed_expiry_index.remove(&closed_at);
        }
    }

    fn record_completed(
        &mut self,
        uuid: Uuid,
        closed_at: DateTime<Utc>,
        span_context: SpanContext,
    ) {
        self.remove_completed(uuid);
        self.completed.insert(
            uuid,
            CompletedLogContext {
                closed_at,
                span_context,
            },
        );
        self.completed_expiry_index
            .entry(closed_at)
            .or_default()
            .insert(uuid);
    }

    fn expire_completed(&mut self, timestamp: DateTime<Utc>, ttl: Duration) -> u64 {
        let mut expired_count = 0;
        while let Some((closed_at, _)) = self.completed_expiry_index.first_key_value() {
            let closed_at = *closed_at;
            if !timestamp
                .signed_duration_since(closed_at)
                .to_std()
                .is_ok_and(|age| age > ttl)
            {
                break;
            }
            let uuids = self
                .completed_expiry_index
                .remove(&closed_at)
                .expect("completed log expiry bucket exists");
            for uuid in uuids {
                if self.completed.remove(&uuid).is_some() {
                    expired_count += 1;
                }
            }
        }
        expired_count
    }
}

struct LogEventProcessor {
    logger: SdkLogger,
    minimum_severity: LogSeverity,
    lineage: ScopeLineage,
    invalid_severity_count: u64,
    invalid_metric_count: u64,
    runtime_diagnostics: SignalRuntimeDiagnostics,
    completed_span_context_ttl: Duration,
}

impl LogEventProcessor {
    #[cfg(test)]
    fn new(
        logger: SdkLogger,
        minimum_severity: LogSeverity,
        diagnostic_field: Option<String>,
    ) -> Self {
        Self::new_with_runtime_diagnostics(
            logger,
            minimum_severity,
            DEFAULT_COMPLETED_SPAN_CONTEXT_TTL,
            SignalRuntimeDiagnostics::new(diagnostic_field),
        )
    }

    fn new_with_runtime_diagnostics(
        logger: SdkLogger,
        minimum_severity: LogSeverity,
        completed_span_context_ttl: Duration,
        runtime_diagnostics: SignalRuntimeDiagnostics,
    ) -> Self {
        Self {
            logger,
            minimum_severity,
            lineage: ScopeLineage::new(),
            invalid_severity_count: 0,
            invalid_metric_count: 0,
            runtime_diagnostics,
            completed_span_context_ttl,
        }
    }

    fn process(&mut self, event: &Event) {
        let expired_count = self
            .lineage
            .expire_completed(*event.timestamp(), self.completed_span_context_ttl);
        if expired_count > 0 {
            let count = self.runtime_diagnostics.record(
                "otel.completed_log_context_expired",
                format!("OpenTelemetry expired {expired_count} completed log contexts"),
                expired_count,
            );
            if should_relog_runtime_diagnostic(count) {
                log::warn!(
                    target: "nemo_relay.observability",
                    event = "otel_completed_log_context_expired",
                    expired_count;
                    "OpenTelemetry expired completed log contexts"
                );
            }
        }
        match event.scope_category() {
            Some(crate::api::event::ScopeCategory::Start) => {
                self.lineage.process_start(event);
            }
            Some(crate::api::event::ScopeCategory::End) => self.lineage.process_end(event),
            None => self.process_mark(event),
        }
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
        let severity = match mark_severity(event) {
            Ok(severity) => severity,
            Err(error) => {
                self.invalid_severity_count = self.invalid_severity_count.saturating_add(1);
                let diagnostic_count = self.runtime_diagnostics.record(
                    "otel.log_mark_invalid_severity",
                    format!(
                        "OpenTelemetry log mark {:?} was dropped: {error}",
                        event.name()
                    ),
                    1,
                );
                if should_relog_runtime_diagnostic(diagnostic_count) {
                    log::warn!(
                        target: "nemo_relay.observability",
                        event = "otel_log_invalid_severity",
                        mark_name = event.name();
                        "OpenTelemetry log mark was dropped: {error}"
                    );
                }
                return;
            }
        };
        if severity < self.minimum_severity {
            return;
        }

        let mut record = self.logger.create_log_record();
        record.set_timestamp(super::otel::to_system_time(*event.timestamp()));
        record.set_observed_timestamp(SystemTime::now());
        let (otel_severity, severity_text) = otel_severity(severity);
        record.set_severity_number(otel_severity);
        record.set_severity_text(severity_text);
        if let Some(body) = event.data().and_then(json_body) {
            record.set_body(body);
        }
        add_log_attributes(&mut record, event);
        if let Some(context) = self.lineage.parent_context(event) {
            record.set_trace_context(
                context.trace_id(),
                context.span_id(),
                Some(context.trace_flags()),
            );
        }
        self.logger.emit(record);
    }
}

fn mark_severity(event: &Event) -> std::result::Result<LogSeverity, String> {
    let Some(value) = event
        .metadata()
        .and_then(Json::as_object)
        .and_then(|metadata| metadata.get(LOG_SEVERITY_METADATA_KEY))
    else {
        return Ok(LogSeverity::Info);
    };
    let value = value.as_str().ok_or_else(|| {
        format!("{LOG_SEVERITY_METADATA_KEY} must be a string after sanitization")
    })?;
    value
        .parse::<LogSeverity>()
        .map_err(|error| error.to_string())
}

fn otel_severity(severity: LogSeverity) -> (Severity, &'static str) {
    match severity {
        LogSeverity::Trace => (Severity::Trace, "TRACE"),
        LogSeverity::Debug => (Severity::Debug, "DEBUG"),
        LogSeverity::Info => (Severity::Info, "INFO"),
        LogSeverity::Warn => (Severity::Warn, "WARN"),
        LogSeverity::Error => (Severity::Error, "ERROR"),
    }
}

fn add_log_attributes(record: &mut opentelemetry_sdk::logs::SdkLogRecord, event: &Event) {
    record.add_attribute("nemo_relay.atof.version", ATOF_VERSION);
    record.add_attribute("nemo_relay.mark.name", event.name().to_string());
    record.add_attribute("nemo_relay.mark.uuid", event.uuid().to_string());
    if let Some(root_uuid) = event.propagation_root_uuid() {
        record.add_attribute("nemo_relay.session.instance_id", root_uuid.to_string());
    }
    if let Some(parent_uuid) = event.parent_uuid() {
        record.add_attribute("nemo_relay.mark.parent_uuid", parent_uuid.to_string());
    }
    if let Some(category) = event.category() {
        record.add_attribute("nemo_relay.mark.category", category.as_str().to_string());
    }
    if let Some(profile) = event.category_profile()
        && let Ok(value) = serde_json::to_value(profile)
        && let Some(value) = json_any_value(&value, true)
    {
        record.add_attribute("nemo_relay.mark.category_profile", value);
    }
    if let Some(schema) = event.data_schema() {
        record.add_attribute("nemo_relay.mark.data_schema.name", schema.name.clone());
        record.add_attribute(
            "nemo_relay.mark.data_schema.version",
            schema.version.clone(),
        );
    }
    if let Some(metadata) = sanitized_metadata(event.metadata())
        && let Some(value) = json_any_value(&metadata, true)
    {
        record.add_attribute("nemo_relay.mark.metadata", value);
    }
}

fn sanitized_metadata(metadata: Option<&Json>) -> Option<Json> {
    let metadata = metadata?.clone();
    match metadata {
        Json::Object(mut object) => {
            object.remove(LOG_SEVERITY_METADATA_KEY);
            (!object.is_empty()).then_some(Json::Object(object))
        }
        other => Some(other),
    }
}

fn json_body(value: &Json) -> Option<AnyValue> {
    (!value.is_null())
        .then(|| json_any_value(value, true))
        .flatten()
}

fn json_any_value(value: &Json, nested: bool) -> Option<AnyValue> {
    match value {
        Json::Null => nested.then(|| AnyValue::from("null")),
        Json::Bool(value) => Some(AnyValue::Boolean(*value)),
        Json::String(value) => Some(AnyValue::from(value.clone())),
        Json::Number(value) => {
            if let Some(value) = value.as_i64() {
                Some(AnyValue::Int(value))
            } else if let Some(value) = value.as_u64() {
                i64::try_from(value)
                    .map(AnyValue::Int)
                    .ok()
                    .or_else(|| Some(AnyValue::from(value.to_string())))
            } else {
                value.as_f64().map(AnyValue::Double)
            }
        }
        Json::Array(values) => Some(AnyValue::ListAny(Box::new(
            values
                .iter()
                .filter_map(|value| json_any_value(value, true))
                .collect(),
        ))),
        Json::Object(values) => Some(AnyValue::Map(Box::new(json_map(values)))),
    }
}

fn json_map(values: &Map<String, Json>) -> HashMap<Key, AnyValue> {
    values
        .iter()
        .filter_map(|(key, value)| {
            json_any_value(value, true).map(|value| (Key::new(key.clone()), value))
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/observability/otel_logs_tests.rs"]
mod tests;
