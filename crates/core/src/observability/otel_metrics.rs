// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry metric export for Relay metric marks.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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

use super::otel::{OpenTelemetryError, OtlpTransport, Result};
use super::otel_signal::{
    MetricMarkClassification, SignalExporterRuntime, build_grpc_metadata, build_in_owned_runtime,
    classify_metric_mark, record_signal_runtime_diagnostic, reject_signal_header_environment,
    resolve_http_signal_endpoint, signal_resource, validate_signal_headers,
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
    provider: SdkMeterProvider,
    delivery_diagnostics: Arc<MetricDeliveryDiagnostics>,
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

    fn new_with_runtime_diagnostics(config: OpenTelemetryMetricConfig) -> Result<Self> {
        config.validate()?;
        let instrumentation_scope = config.instrumentation_scope.clone();
        let max_instruments = config.max_instruments;
        let diagnostic_field = config.diagnostic_field.clone();
        let cardinality_limit = config.cardinality_limit;
        let delivery_diagnostics = Arc::new(MetricDeliveryDiagnostics::new(
            config.endpoint.clone(),
            config.diagnostic_field.clone(),
        ));
        let provider_diagnostics = Arc::clone(&delivery_diagnostics);
        let (provider, runtime) = build_in_owned_runtime("nemo-relay-otlp-metrics", move || {
            build_metric_provider(&config, provider_diagnostics)
        })?;
        let meter =
            provider.meter_with_scope(InstrumentationScope::builder(instrumentation_scope).build());
        let processor = Arc::new(Mutex::new(MetricEventProcessor::new(
            meter,
            max_instruments,
            diagnostic_field,
            cardinality_limit,
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
                            event = "otel_metric_processor_lock_recovered";
                            "OpenTelemetry metric subscriber recovered a poisoned processor lock"
                        );
                    }
                    poisoned.into_inner()
                }
            };
            processor.process(event);
        });
        Ok(Self {
            inner: Arc::new(MetricSubscriberInner {
                _processor: processor,
                provider,
                delivery_diagnostics,
                subscriber,
                _runtime: runtime,
            }),
        })
    }

    /// Return the raw Relay subscriber callback.
    pub fn subscriber(&self) -> EventSubscriberFn {
        Arc::clone(&self.inner.subscriber)
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
            .map_err(|error| OpenTelemetryError::Provider(error.to_string()))
    }

    /// Shut down the meter provider, including its final collection.
    ///
    /// Deregister this subscriber before calling shutdown.
    pub fn shutdown(&self) -> Result<()> {
        let barrier = flush_subscribers().map_err(OpenTelemetryError::Core);
        let provider = self
            .inner
            .provider
            .shutdown()
            .map_err(|error| OpenTelemetryError::Provider(error.to_string()));
        barrier.and(provider)
    }

    pub(crate) fn shutdown_provider(&self) -> Result<()> {
        self.inner
            .provider
            .shutdown()
            .map_err(|error| OpenTelemetryError::Provider(error.to_string()))
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
    diagnostic_field: Option<String>,
    export_failures: AtomicU64,
}

impl MetricDeliveryDiagnostics {
    fn new(endpoint: String, diagnostic_field: Option<String>) -> Self {
        Self {
            endpoint,
            diagnostic_field,
            export_failures: AtomicU64::new(0),
        }
    }

    fn failure_summary(&self) -> Option<String> {
        let failures = self.export_failures.load(Ordering::Relaxed);
        (failures > 0).then(|| format!("otel.metrics_export_failed ({failures})"))
    }
}

impl<E: PushMetricExporter> PushMetricExporter for DiagnosticMetricExporter<E> {
    async fn export(&self, metrics: &ResourceMetrics) -> OTelSdkResult {
        let result = self.inner.export(metrics).await;
        if let Err(error) = &result {
            self.diagnostics
                .export_failures
                .fetch_add(1, Ordering::Relaxed);
            record_signal_runtime_diagnostic(
                "otel.metrics_export_failed",
                self.diagnostics.diagnostic_field.clone(),
                format!(
                    "OpenTelemetry metric export to endpoint {} failed: {error}",
                    self.diagnostics.endpoint
                ),
                1,
            );
        }
        result
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        let result = self.inner.shutdown_with_timeout(timeout);
        if result.is_ok() && self.diagnostics.diagnostic_field.is_some() {
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
    attribute_sets: HashSet<String>,
}

struct MetricEventProcessor {
    meter: Meter,
    instruments: HashMap<String, InstrumentEntry>,
    max_instruments: usize,
    rejected_marks: u64,
    diagnostic_field: Option<String>,
    cardinality_limit: usize,
}

impl MetricEventProcessor {
    fn new(
        meter: Meter,
        max_instruments: usize,
        diagnostic_field: Option<String>,
        cardinality_limit: usize,
    ) -> Self {
        Self {
            meter,
            instruments: HashMap::new(),
            max_instruments,
            rejected_marks: 0,
            diagnostic_field,
            cardinality_limit,
        }
    }

    fn process(&mut self, event: &Event) {
        let envelope = match classify_metric_mark(event) {
            MetricMarkClassification::NotMetric => return,
            MetricMarkClassification::Valid(envelope) => envelope,
            MetricMarkClassification::Invalid(error) => {
                self.reject(event, MetricRejection::InvalidEnvelope, error);
                return;
            }
        };
        if let Err(error) = self.record_envelope(&envelope) {
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
            if let Some(attribute_key) = metric_attribute_set_key(&measurement.attributes)
                && !entry.attribute_sets.contains(&attribute_key)
            {
                if entry.attribute_sets.len() >= self.cardinality_limit {
                    record_signal_runtime_diagnostic(
                        "otel.metric_cardinality_limit",
                        self.diagnostic_field.clone(),
                        format!(
                            "OpenTelemetry metric {:?} exceeded the endpoint cardinality limit of {}; additional attribute sets use the SDK overflow series",
                            measurement.descriptor.name.as_str(),
                            self.cardinality_limit
                        ),
                        1,
                    );
                } else {
                    entry.attribute_sets.insert(attribute_key);
                }
            }
            record_measurement(&entry.instrument, measurement);
        }
        Ok(())
    }

    fn reject(&mut self, event: &Event, kind: MetricRejection, error: String) {
        self.rejected_marks = self.rejected_marks.saturating_add(1);
        if self.rejected_marks == 1 {
            log::warn!(
                target: "nemo_relay.observability",
                event = "otel_metric_mark_rejected",
                mark_name = event.name();
                "OpenTelemetry metric mark was dropped atomically: {error}"
            );
        }
        record_signal_runtime_diagnostic(
            kind.code(),
            self.diagnostic_field.clone(),
            format!(
                "OpenTelemetry metric mark {:?} was dropped atomically: {error}",
                event.name()
            ),
            1,
        );
    }
}

fn metric_attribute_set_key(attributes: &MetricAttributes) -> Option<String> {
    if attributes.is_empty() {
        return None;
    }
    Some(format!("{attributes:?}"))
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
