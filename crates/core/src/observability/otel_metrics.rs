// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry metric export for Relay metric marks.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::api::event::{
    AttributeValue, Event, MetricAttributes, MetricKind, MetricValue, MetricValueType,
    ValidatedMetricMeasurement,
};
#[cfg(test)]
use crate::api::event::{MetricEnvelope, MetricMeasurement};
use crate::api::runtime::EventSubscriberFn;
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter, MeterProvider as _, UpDownCounter};
use opentelemetry::{Array, InstrumentationScope, KeyValue, Value};
use opentelemetry_otlp::{
    MetricExporter as OtlpMetricExporter, Protocol, WithExportConfig, WithHttpConfig,
    WithTonicConfig,
};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider, Stream, Temporality};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::Value as Json;
use sha2::{Digest, Sha256};

use super::OpenTelemetryRuntimeDiagnostics;
use super::otel::{OpenTelemetryError, OtlpTransport, Result, normalize_shutdown_result};
use super::otel_signal::{
    MetricMarkClassification, SignalExporterRuntime, SignalRuntimeDiagnostics, build_grpc_metadata,
    build_in_owned_runtime, classify_metric_mark, reject_signal_header_environment,
    resolve_header_env, resolve_http_signal_endpoint, should_relog_runtime_diagnostic,
    signal_resource, validate_signal_headers,
};

const DEFAULT_EXPORT_INTERVAL: Duration = Duration::from_secs(60);
const DEFAULT_MAX_INSTRUMENTS: usize = 256;
const DEFAULT_CARDINALITY_LIMIT: usize = 2_000;

/// Preferred aggregation temporality for OTLP metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricTemporality {
    /// Accumulate values from process start.
    #[default]
    Cumulative,
    /// Export values recorded since the previous collection when supported.
    Delta,
    /// Favor delta aggregation for counters and histograms to reduce memory.
    LowMemory,
}

impl MetricTemporality {
    /// Return the canonical config value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cumulative => "cumulative",
            Self::Delta => "delta",
            Self::LowMemory => "low_memory",
        }
    }

    fn sdk(self) -> Temporality {
        match self {
            Self::Cumulative => Temporality::Cumulative,
            Self::Delta => Temporality::Delta,
            Self::LowMemory => Temporality::LowMemory,
        }
    }
}

impl std::str::FromStr for MetricTemporality {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cumulative" => Ok(Self::Cumulative),
            "delta" => Ok(Self::Delta),
            "low_memory" | "lowmemory" => Ok(Self::LowMemory),
            other => Err(format!(
                "invalid metric temporality {other:?}; expected cumulative, delta, or low_memory"
            )),
        }
    }
}

/// Configuration for an OTLP metric subscriber.
#[derive(Debug, Clone)]
pub struct OpenTelemetryMetricConfig {
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
    export_interval: Duration,
    temporality: MetricTemporality,
    max_instruments: usize,
    cardinality_limit: usize,
    diagnostic_field: Option<String>,
}

impl OpenTelemetryMetricConfig {
    /// Create a metric exporter for a required OTLP endpoint.
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
            export_interval: DEFAULT_EXPORT_INTERVAL,
            temporality: MetricTemporality::Cumulative,
            max_instruments: DEFAULT_MAX_INSTRUMENTS,
            cardinality_limit: DEFAULT_CARDINALITY_LIMIT,
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

    /// Set the interval between metric collections.
    pub fn with_export_interval(mut self, interval: Duration) -> Self {
        self.export_interval = interval;
        self
    }

    /// Set the preferred aggregation temporality.
    pub fn with_temporality(mut self, temporality: MetricTemporality) -> Self {
        self.temporality = temporality;
        self
    }

    /// Set the maximum number of distinct instrument names retained by this endpoint.
    pub fn with_max_instruments(mut self, max_instruments: usize) -> Self {
        self.max_instruments = max_instruments;
        self
    }

    /// Set the SDK series cardinality limit per instrument.
    pub fn with_cardinality_limit(mut self, cardinality_limit: usize) -> Self {
        self.cardinality_limit = cardinality_limit;
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
        if self.export_interval.is_zero() {
            return Err(OpenTelemetryError::ExporterBuild(
                "export_interval must be greater than 0".to_string(),
            ));
        }
        if self.max_instruments == 0 {
            return Err(OpenTelemetryError::ExporterBuild(
                "max_instruments must be greater than 0".to_string(),
            ));
        }
        if self.cardinality_limit == 0 {
            return Err(OpenTelemetryError::ExporterBuild(
                "cardinality_limit must be greater than 0".to_string(),
            ));
        }
        if self.cardinality_limit == usize::MAX {
            return Err(OpenTelemetryError::ExporterBuild(
                "cardinality_limit must be less than usize::MAX".to_string(),
            ));
        }
        reject_signal_header_environment("OTEL_EXPORTER_OTLP_METRICS_HEADERS")?;
        validate_signal_headers(&self.headers)
    }
}

/// Resolve an OTLP/HTTP endpoint for the metrics signal.
pub fn resolve_http_metric_endpoint(endpoint: &str) -> Cow<'_, str> {
    resolve_http_signal_endpoint(endpoint, "metrics")
}

/// OpenTelemetry metric-backed Relay event subscriber.
#[derive(Clone)]
pub struct OpenTelemetryMetricSubscriber {
    inner: Arc<MetricSubscriberInner>,
}

struct MetricSubscriberInner {
    // Drop instruments and meter before the provider, then stop its runtime.
    _processor: Arc<Mutex<MetricEventProcessor>>,
    processor_lock_recovery_warned: Arc<AtomicBool>,
    provider: SdkMeterProvider,
    delivery_diagnostics: Arc<MetricDeliveryDiagnostics>,
    runtime_diagnostics: SignalRuntimeDiagnostics,
    subscriber: EventSubscriberFn,
    _runtime: SignalExporterRuntime,
}

impl OpenTelemetryMetricSubscriber {
    /// Build an OTLP metric subscriber with an independently owned provider.
    pub fn new(config: OpenTelemetryMetricConfig) -> Result<Self> {
        Self::new_with_runtime_diagnostics(config)
    }

    pub(crate) fn new_for_plugin(
        mut config: OpenTelemetryMetricConfig,
        endpoint_index: usize,
    ) -> Result<Self> {
        config.diagnostic_field = Some(format!(
            "opentelemetry.metrics.endpoints[{endpoint_index}].endpoint"
        ));
        Self::new_with_runtime_diagnostics(config)
    }

    fn new_with_runtime_diagnostics(mut config: OpenTelemetryMetricConfig) -> Result<Self> {
        config.validate()?;
        config.headers = resolve_header_env(&config.headers, &config.header_env)?;
        validate_signal_headers(&config.headers)?;
        let instrumentation_scope = config.instrumentation_scope.clone();
        let max_instruments = config.max_instruments;
        let cardinality_limit = config.cardinality_limit;
        let runtime_diagnostics = SignalRuntimeDiagnostics::new(config.diagnostic_field.clone());
        let delivery_diagnostics = Arc::new(MetricDeliveryDiagnostics::new(
            config.endpoint.clone(),
            runtime_diagnostics.clone(),
        ));
        let provider_diagnostics = Arc::clone(&delivery_diagnostics);
        let (provider, runtime) = build_in_owned_runtime("nemo-relay-otlp-metrics", move || {
            build_metric_provider(&config, provider_diagnostics)
        })?;
        let meter =
            provider.meter_with_scope(InstrumentationScope::builder(instrumentation_scope).build());
        let processor = Arc::new(Mutex::new(
            MetricEventProcessor::new_with_runtime_diagnostics(
                meter,
                max_instruments,
                cardinality_limit,
                runtime_diagnostics.clone(),
            ),
        ));
        let callback_processor = Arc::clone(&processor);
        let processor_lock_recovery_warned = Arc::new(AtomicBool::new(false));
        let callback_recovery_warned_for_callback = Arc::clone(&processor_lock_recovery_warned);
        let subscriber: EventSubscriberFn = Arc::new(move |event| {
            process_metric_event(
                &callback_processor,
                &callback_recovery_warned_for_callback,
                event,
            );
        });
        Ok(Self {
            inner: Arc::new(MetricSubscriberInner {
                _processor: processor,
                processor_lock_recovery_warned,
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

    pub(crate) fn process_validated(
        &self,
        event: &Event,
        measurements: &[ValidatedMetricMeasurement],
    ) {
        process_validated_metric_measurements(
            &self.inner._processor,
            &self.inner.processor_lock_recovery_warned,
            event,
            measurements,
        );
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

    /// Collect and export current metric aggregates immediately.
    pub fn force_flush(&self) -> Result<()> {
        flush_subscribers()?;
        self.inner
            .provider
            .force_flush()
            .map_err(|error| OpenTelemetryError::MetricProvider(error.to_string()))
    }

    /// Shut down the meter provider, including its final collection.
    ///
    /// Deregister this subscriber before calling shutdown.
    pub fn shutdown(&self) -> Result<()> {
        let barrier = flush_subscribers().map_err(OpenTelemetryError::Core);
        let provider = normalize_shutdown_result(self.inner.provider.shutdown())
            .map_err(|error| OpenTelemetryError::MetricProvider(error.to_string()));
        barrier.and(provider)
    }

    pub(crate) fn shutdown_provider(&self) -> Result<()> {
        normalize_shutdown_result(self.inner.provider.shutdown())
            .map_err(|error| OpenTelemetryError::MetricProvider(error.to_string()))
    }

    pub(crate) fn delivery_failure_summary(&self) -> Option<String> {
        self.inner.delivery_diagnostics.failure_summary()
    }
}

fn build_metric_provider(
    config: &OpenTelemetryMetricConfig,
    diagnostics: Arc<MetricDeliveryDiagnostics>,
) -> Result<SdkMeterProvider> {
    let temporality = config.temporality.sdk();
    let exporter = match config.transport {
        OtlpTransport::HttpBinary => {
            let mut builder = OtlpMetricExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_temporality(temporality)
                .with_timeout(config.timeout)
                .with_endpoint(resolve_http_metric_endpoint(&config.endpoint).into_owned());
            if !config.headers.is_empty() {
                builder = builder.with_headers(config.headers.clone());
            }
            builder
                .build()
                .map_err(|error| OpenTelemetryError::ExporterBuild(error.to_string()))?
        }
        OtlpTransport::Grpc => {
            let mut builder = OtlpMetricExporter::builder()
                .with_tonic()
                .with_protocol(Protocol::Grpc)
                .with_temporality(temporality)
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

    let exporter = DiagnosticMetricExporter {
        inner: exporter,
        diagnostics,
    };
    let reader = PeriodicReader::builder(exporter)
        .with_interval(config.export_interval)
        .build();
    let cardinality_limit = config.cardinality_limit;
    Ok(SdkMeterProvider::builder()
        .with_resource(signal_resource(
            &config.service_name,
            config.service_namespace.as_deref(),
            config.service_version.as_deref(),
            &config.resource_attributes,
        ))
        .with_reader(reader)
        .with_view(move |instrument| {
            Stream::builder()
                .with_name(instrument.name().to_string())
                .with_cardinality_limit(cardinality_limit)
                .build()
                .ok()
        })
        .build())
}

#[derive(Debug)]
struct DiagnosticMetricExporter<E> {
    inner: E,
    diagnostics: Arc<MetricDeliveryDiagnostics>,
}

#[derive(Debug)]
struct MetricDeliveryDiagnostics {
    endpoint: String,
    runtime_diagnostics: SignalRuntimeDiagnostics,
    export_failures: AtomicU64,
}

impl MetricDeliveryDiagnostics {
    fn new(endpoint: String, runtime_diagnostics: SignalRuntimeDiagnostics) -> Self {
        Self {
            endpoint,
            runtime_diagnostics,
            export_failures: AtomicU64::new(0),
        }
    }

    fn failure_summary(&self) -> Option<String> {
        let failures = self.export_failures.load(Ordering::Relaxed);
        (failures > 0).then(|| format!("otel.metrics_export_failed ({failures})"))
    }

    fn record_export_failure(&self, error: &impl std::fmt::Display) -> u64 {
        let failure_count = self.export_failures.fetch_add(1, Ordering::Relaxed) + 1;
        self.runtime_diagnostics.record(
            "otel.metrics_export_failed",
            format!(
                "OpenTelemetry metric export to endpoint {} failed: {error}",
                self.endpoint
            ),
            1,
        );
        failure_count
    }
}

impl<E: PushMetricExporter> PushMetricExporter for DiagnosticMetricExporter<E> {
    async fn export(&self, metrics: &ResourceMetrics) -> OTelSdkResult {
        let result = self.inner.export(metrics).await;
        if let Err(error) = &result {
            let failure_count = self.diagnostics.record_export_failure(error);
            if should_relog_runtime_diagnostic(failure_count) {
                log::error!(
                    target: "nemo_relay.observability",
                    event = "otel_metrics_export_failed",
                    endpoint = self.diagnostics.endpoint.as_str();
                    "OpenTelemetry metric export failed: {error}"
                );
            }
        }
        result
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        let result = self.inner.shutdown_with_timeout(timeout);
        if result.is_ok() && self.diagnostics.runtime_diagnostics.has_plugin_mirror() {
            let failures = self.diagnostics.export_failures.load(Ordering::Relaxed);
            if failures > 0 {
                return Err(opentelemetry_sdk::error::OTelSdkError::InternalFailure(
                    format!(
                        "{}: otel.metrics_export_failed ({failures})",
                        crate::plugin::OTEL_RUNTIME_DELIVERY_FAILURE_MARKER
                    ),
                ));
            }
        }
        result
    }

    fn temporality(&self) -> Temporality {
        self.inner.temporality()
    }
}

#[derive(Debug, Clone)]
struct MetricDescriptor {
    kind: MetricKind,
    value_type: MetricValueType,
    unit: Option<String>,
    description: Option<String>,
    boundaries: Option<Vec<f64>>,
}

impl MetricDescriptor {
    fn from_measurement(measurement: &ValidatedMetricMeasurement) -> Self {
        Self {
            kind: measurement.descriptor.kind,
            value_type: measurement.value.value_type(),
            unit: measurement.descriptor.unit.clone(),
            description: measurement.descriptor.description.clone(),
            boundaries: measurement
                .descriptor
                .boundaries
                .as_ref()
                .map(|boundaries| boundaries.values()),
        }
    }

    fn has_same_identity(&self, other: &Self) -> bool {
        // Description and boundaries are advisory OpenTelemetry fields. The first
        // descriptor to create an instrument supplies them for that process.
        self.kind == other.kind && self.value_type == other.value_type && self.unit == other.unit
    }
}

enum CachedInstrument {
    U64Counter(Counter<u64>),
    F64Counter(Counter<f64>),
    I64UpDownCounter(UpDownCounter<i64>),
    F64UpDownCounter(UpDownCounter<f64>),
    U64Gauge(Gauge<u64>),
    I64Gauge(Gauge<i64>),
    F64Gauge(Gauge<f64>),
    U64Histogram(Histogram<u64>),
    F64Histogram(Histogram<f64>),
}

struct InstrumentEntry {
    descriptor: MetricDescriptor,
    instrument: CachedInstrument,
    attribute_sets: HashSet<[u8; 32]>,
}

struct MetricEventProcessor {
    meter: Meter,
    instruments: HashMap<String, InstrumentEntry>,
    max_instruments: usize,
    rejected_marks: u64,
    runtime_diagnostics: SignalRuntimeDiagnostics,
    cardinality_limit: usize,
}

impl MetricEventProcessor {
    #[cfg(test)]
    fn new(
        meter: Meter,
        max_instruments: usize,
        diagnostic_field: Option<String>,
        cardinality_limit: usize,
    ) -> Self {
        Self::new_with_runtime_diagnostics(
            meter,
            max_instruments,
            cardinality_limit,
            SignalRuntimeDiagnostics::new(diagnostic_field),
        )
    }

    fn new_with_runtime_diagnostics(
        meter: Meter,
        max_instruments: usize,
        cardinality_limit: usize,
        runtime_diagnostics: SignalRuntimeDiagnostics,
    ) -> Self {
        Self {
            meter,
            instruments: HashMap::new(),
            max_instruments,
            rejected_marks: 0,
            runtime_diagnostics,
            cardinality_limit,
        }
    }

    #[cfg(test)]
    fn process(&mut self, event: &Event) {
        self.process_classification(event, classify_metric_mark(event));
    }

    fn process_classification(&mut self, event: &Event, classification: MetricMarkClassification) {
        let measurements = match classification {
            MetricMarkClassification::NotMetric => return,
            MetricMarkClassification::Valid(measurements) => measurements,
            MetricMarkClassification::Invalid(error) => {
                self.reject(event, MetricRejection::InvalidEnvelope, error);
                return;
            }
        };
        self.process_validated(event, &measurements);
    }

    fn process_validated(&mut self, event: &Event, measurements: &[ValidatedMetricMeasurement]) {
        let mut measurements = measurements.to_vec();
        if let Some(root_uuid) = event.propagation_root_uuid() {
            for measurement in &mut measurements {
                measurement.attributes.insert(
                    "nemo_relay.session.instance_id".to_string(),
                    AttributeValue::String(root_uuid.to_string()),
                );
            }
        }
        if let Err(error) = self.record_envelope(&measurements) {
            self.reject(event, error.kind, error.message);
        }
    }

    fn record_envelope(
        &mut self,
        measurements: &[ValidatedMetricMeasurement],
    ) -> std::result::Result<(), MetricRecordError> {
        let mut proposed: HashMap<String, (&ValidatedMetricMeasurement, MetricDescriptor)> =
            HashMap::new();
        for measurement in measurements {
            let key = measurement.descriptor.descriptor_key();
            let descriptor = MetricDescriptor::from_measurement(measurement);
            if let Some(existing) = self.instruments.get(&key)
                && !existing.descriptor.has_same_identity(&descriptor)
            {
                return Err(MetricRecordError::new(
                    MetricRejection::DescriptorConflict,
                    format!(
                        "metric {:?} conflicts with its existing instrument descriptor",
                        measurement.descriptor.name.as_str()
                    ),
                ));
            }
            proposed.entry(key).or_insert((measurement, descriptor));
        }

        let new_count = proposed
            .keys()
            .filter(|key| !self.instruments.contains_key(*key))
            .count();
        if self.instruments.len().saturating_add(new_count) > self.max_instruments {
            return Err(MetricRecordError::new(
                MetricRejection::InstrumentLimit,
                format!(
                    "metric mark exceeds the endpoint limit of {} distinct instruments",
                    self.max_instruments
                ),
            ));
        }

        for (key, (_measurement, descriptor)) in proposed {
            if !self.instruments.contains_key(&key) {
                let instrument = build_instrument(&self.meter, &key, &descriptor);
                self.instruments.insert(
                    key,
                    InstrumentEntry {
                        descriptor,
                        instrument,
                        attribute_sets: HashSet::new(),
                    },
                );
            }
        }

        for measurement in measurements {
            let key = measurement.descriptor.descriptor_key();
            let entry = self
                .instruments
                .get_mut(&key)
                .expect("metric instrument was preflighted and constructed");
            if let Some(attribute_fingerprint) =
                metric_attribute_set_fingerprint(&measurement.attributes)
                && !entry.attribute_sets.contains(&attribute_fingerprint)
            {
                if entry.attribute_sets.len() >= self.cardinality_limit {
                    self.runtime_diagnostics.record(
                        "otel.metric_cardinality_limit",
                        format!(
                            "OpenTelemetry metric {:?} exceeded the endpoint cardinality limit of {}; additional attribute sets use the SDK overflow series",
                            measurement.descriptor.name.as_str(),
                            self.cardinality_limit
                        ),
                        1,
                    );
                } else {
                    entry.attribute_sets.insert(attribute_fingerprint);
                }
            }
            record_measurement(&entry.instrument, measurement);
        }
        Ok(())
    }

    fn reject(&mut self, event: &Event, kind: MetricRejection, error: String) {
        self.rejected_marks = self.rejected_marks.saturating_add(1);
        let diagnostic_count = self.runtime_diagnostics.record(
            kind.code(),
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
    }
}

fn process_metric_event(
    processor: &Mutex<MetricEventProcessor>,
    recovery_warned: &AtomicBool,
    event: &Event,
) {
    let classification = classify_metric_mark(event);
    let mut processor = lock_metric_processor(processor, recovery_warned);
    processor.process_classification(event, classification);
}

fn process_validated_metric_measurements(
    processor: &Mutex<MetricEventProcessor>,
    recovery_warned: &AtomicBool,
    event: &Event,
    measurements: &[ValidatedMetricMeasurement],
) {
    let mut processor = lock_metric_processor(processor, recovery_warned);
    processor.process_validated(event, measurements);
}

fn lock_metric_processor<'a>(
    processor: &'a Mutex<MetricEventProcessor>,
    recovery_warned: &AtomicBool,
) -> MutexGuard<'a, MetricEventProcessor> {
    match processor.lock() {
        Ok(processor) => processor,
        Err(poisoned) => {
            if !recovery_warned.swap(true, Ordering::Relaxed) {
                log::warn!(
                    target: "nemo_relay.observability",
                    event = "otel_metric_processor_lock_recovered";
                    "OpenTelemetry metric subscriber recovered a poisoned processor lock"
                );
            }
            poisoned.into_inner()
        }
    }
}

fn metric_attribute_set_fingerprint(attributes: &MetricAttributes) -> Option<[u8; 32]> {
    if attributes.is_empty() {
        return None;
    }

    let mut hasher = Sha256::new();
    for (key, value) in attributes.iter() {
        hash_metric_attribute_bytes(&mut hasher, key.as_bytes());
        match value {
            AttributeValue::String(value) => {
                hasher.update(b"s");
                hash_metric_attribute_bytes(&mut hasher, value.as_bytes());
            }
            AttributeValue::Bool(value) => {
                hasher.update(b"b");
                hasher.update([u8::from(*value)]);
            }
            AttributeValue::I64(value) => {
                hasher.update(b"i");
                hasher.update(value.to_be_bytes());
            }
            AttributeValue::F64(value) => {
                hasher.update(b"f");
                hasher.update(value.get().to_bits().to_be_bytes());
            }
            AttributeValue::StringArray(values) => {
                hasher.update(b"S");
                hash_metric_attribute_count(&mut hasher, values.len());
                for value in values {
                    hash_metric_attribute_bytes(&mut hasher, value.as_bytes());
                }
            }
            AttributeValue::BoolArray(values) => {
                hasher.update(b"B");
                hash_metric_attribute_count(&mut hasher, values.len());
                for value in values {
                    hasher.update([u8::from(*value)]);
                }
            }
            AttributeValue::I64Array(values) => {
                hasher.update(b"I");
                hash_metric_attribute_count(&mut hasher, values.len());
                for value in values {
                    hasher.update(value.to_be_bytes());
                }
            }
            AttributeValue::F64Array(values) => {
                hasher.update(b"F");
                hash_metric_attribute_count(&mut hasher, values.len());
                for value in values {
                    hasher.update(value.get().to_bits().to_be_bytes());
                }
            }
        }
    }
    Some(hasher.finalize().into())
}

fn hash_metric_attribute_count(hasher: &mut Sha256, count: usize) {
    hasher.update((count as u64).to_be_bytes());
}

fn hash_metric_attribute_bytes(hasher: &mut Sha256, value: &[u8]) {
    hash_metric_attribute_count(hasher, value.len());
    hasher.update(value);
}

#[derive(Debug, Clone, Copy)]
enum MetricRejection {
    InvalidEnvelope,
    DescriptorConflict,
    InstrumentLimit,
}

impl MetricRejection {
    const fn code(self) -> &'static str {
        match self {
            Self::InvalidEnvelope => "otel.metric_mark_invalid",
            Self::DescriptorConflict => "otel.metric_descriptor_conflict",
            Self::InstrumentLimit => "otel.metric_instrument_limit",
        }
    }
}

struct MetricRecordError {
    kind: MetricRejection,
    message: String,
}

impl MetricRecordError {
    fn new(kind: MetricRejection, message: String) -> Self {
        Self { kind, message }
    }
}

fn build_instrument(meter: &Meter, name: &str, descriptor: &MetricDescriptor) -> CachedInstrument {
    match descriptor.kind {
        MetricKind::Counter => build_counter(meter, name, descriptor),
        MetricKind::UpDownCounter => build_up_down_counter(meter, name, descriptor),
        MetricKind::Gauge => build_gauge(meter, name, descriptor),
        MetricKind::Histogram => build_histogram(meter, name, descriptor),
    }
}

macro_rules! configured_instrument {
    ($builder:expr, $descriptor:expr) => {{
        let mut builder = $builder;
        if let Some(description) = $descriptor.description.clone() {
            builder = builder.with_description(description);
        }
        if let Some(unit) = $descriptor.unit.clone() {
            builder = builder.with_unit(unit);
        }
        builder
    }};
}

fn build_counter(meter: &Meter, name: &str, descriptor: &MetricDescriptor) -> CachedInstrument {
    match descriptor.value_type {
        MetricValueType::U64 => CachedInstrument::U64Counter(
            configured_instrument!(meter.u64_counter(name.to_string()), descriptor).build(),
        ),
        MetricValueType::F64 => CachedInstrument::F64Counter(
            configured_instrument!(meter.f64_counter(name.to_string()), descriptor).build(),
        ),
        MetricValueType::I64 => unreachable!("validated counter has a supported value type"),
    }
}

fn build_up_down_counter(
    meter: &Meter,
    name: &str,
    descriptor: &MetricDescriptor,
) -> CachedInstrument {
    match descriptor.value_type {
        MetricValueType::I64 => CachedInstrument::I64UpDownCounter(
            configured_instrument!(meter.i64_up_down_counter(name.to_string()), descriptor).build(),
        ),
        MetricValueType::F64 => CachedInstrument::F64UpDownCounter(
            configured_instrument!(meter.f64_up_down_counter(name.to_string()), descriptor).build(),
        ),
        MetricValueType::U64 => {
            unreachable!("validated up/down counter has a supported value type")
        }
    }
}

fn build_gauge(meter: &Meter, name: &str, descriptor: &MetricDescriptor) -> CachedInstrument {
    match descriptor.value_type {
        MetricValueType::U64 => CachedInstrument::U64Gauge(
            configured_instrument!(meter.u64_gauge(name.to_string()), descriptor).build(),
        ),
        MetricValueType::I64 => CachedInstrument::I64Gauge(
            configured_instrument!(meter.i64_gauge(name.to_string()), descriptor).build(),
        ),
        MetricValueType::F64 => CachedInstrument::F64Gauge(
            configured_instrument!(meter.f64_gauge(name.to_string()), descriptor).build(),
        ),
    }
}

fn build_histogram(meter: &Meter, name: &str, descriptor: &MetricDescriptor) -> CachedInstrument {
    match descriptor.value_type {
        MetricValueType::U64 => {
            let mut builder =
                configured_instrument!(meter.u64_histogram(name.to_string()), descriptor);
            if let Some(boundaries) = descriptor.boundaries.clone() {
                builder = builder.with_boundaries(boundaries);
            }
            CachedInstrument::U64Histogram(builder.build())
        }
        MetricValueType::F64 => {
            let mut builder =
                configured_instrument!(meter.f64_histogram(name.to_string()), descriptor);
            if let Some(boundaries) = descriptor.boundaries.clone() {
                builder = builder.with_boundaries(boundaries);
            }
            CachedInstrument::F64Histogram(builder.build())
        }
        MetricValueType::I64 => unreachable!("validated histogram has a supported value type"),
    }
}

fn record_measurement(instrument: &CachedInstrument, measurement: &ValidatedMetricMeasurement) {
    let attributes = metric_attributes(&measurement.attributes);
    match (instrument, measurement.value) {
        (CachedInstrument::U64Counter(instrument), MetricValue::U64(value)) => {
            instrument.add(value, &attributes);
        }
        (CachedInstrument::F64Counter(instrument), MetricValue::F64(value)) => {
            instrument.add(value.get(), &attributes);
        }
        (CachedInstrument::I64UpDownCounter(instrument), MetricValue::I64(value)) => {
            instrument.add(value, &attributes);
        }
        (CachedInstrument::F64UpDownCounter(instrument), MetricValue::F64(value)) => {
            instrument.add(value.get(), &attributes);
        }
        (CachedInstrument::U64Gauge(instrument), MetricValue::U64(value)) => {
            instrument.record(value, &attributes);
        }
        (CachedInstrument::I64Gauge(instrument), MetricValue::I64(value)) => {
            instrument.record(value, &attributes);
        }
        (CachedInstrument::F64Gauge(instrument), MetricValue::F64(value)) => {
            instrument.record(value.get(), &attributes);
        }
        (CachedInstrument::U64Histogram(instrument), MetricValue::U64(value)) => {
            instrument.record(value, &attributes);
        }
        (CachedInstrument::F64Histogram(instrument), MetricValue::F64(value)) => {
            instrument.record(value.get(), &attributes);
        }
        _ => unreachable!("cached instrument matches its validated metric value"),
    }
}

fn metric_attributes(attributes: &MetricAttributes) -> Vec<KeyValue> {
    attributes
        .iter()
        .map(|(key, value)| KeyValue::new(key.clone(), metric_attribute_value(value)))
        .collect()
}

fn metric_attribute_value(value: &AttributeValue) -> Value {
    match value {
        AttributeValue::String(value) => Value::String(value.clone().into()),
        AttributeValue::Bool(value) => Value::Bool(*value),
        AttributeValue::I64(value) => Value::I64(*value),
        AttributeValue::F64(value) => Value::F64(value.get()),
        AttributeValue::StringArray(values) => Value::Array(Array::String(
            values.iter().cloned().map(Into::into).collect(),
        )),
        AttributeValue::BoolArray(values) => Value::Array(Array::Bool(values.clone())),
        AttributeValue::I64Array(values) => Value::Array(Array::I64(values.clone())),
        AttributeValue::F64Array(values) => {
            Value::Array(Array::F64(values.iter().map(|value| value.get()).collect()))
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/otel_metrics_tests.rs"]
mod tests;
