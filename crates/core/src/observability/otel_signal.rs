// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for independently owned OTLP signal providers.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use opentelemetry::KeyValue;
use opentelemetry_sdk::{
    Resource,
    error::{OTelSdkError, OTelSdkResult},
    resource::TelemetryResourceDetector,
};
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};
use uuid::Uuid;

use crate::api::event::{
    Event, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION, MetricEnvelope, ScopeCategory,
    ValidatedMetricMeasurement,
};
use crate::plugin::{RuntimeDiagnostic, record_active_plugin_runtime_diagnostic};

use super::otel::{OpenTelemetryError, Result};
use super::{MetadataPromotionIssue, promote_event_metadata_attributes};

// Each dynamic pipeline owns a runtime worker and a supervising thread. Keep the
// per-endpoint log/metric budget conservative; the base provider is not counted.
pub(super) const MAX_DYNAMIC_SIGNAL_PIPELINES: usize = 16;

const MAX_RUNTIME_DIAGNOSTICS: usize = 32;
const MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS: usize = 1_024;
pub(super) const TELEMETRY_SDK_RESOURCE_ATTRIBUTE_KEYS: [&str; 3] = [
    "telemetry.sdk.name",
    "telemetry.sdk.language",
    "telemetry.sdk.version",
];

/// Reject resource attributes owned by the OpenTelemetry SDK detector.
pub(super) fn validate_telemetry_sdk_resource_attributes(
    resource_attributes: &HashMap<String, String>,
) -> Result<()> {
    for key in TELEMETRY_SDK_RESOURCE_ATTRIBUTE_KEYS {
        if resource_attributes.contains_key(key) {
            return Err(OpenTelemetryError::ExporterBuild(format!(
                "resource attribute {key:?} is set automatically by the OpenTelemetry SDK and must not be configured"
            )));
        }
    }
    Ok(())
}

/// Build Relay's OTLP resource with only SDK-provided telemetry identity.
pub(super) fn telemetry_resource(attributes: impl IntoIterator<Item = KeyValue>) -> Resource {
    Resource::builder_empty()
        .with_detector(Box::new(TelemetryResourceDetector))
        .with_attributes(attributes)
        .build()
}

/// A bounded aggregate describing an OpenTelemetry runtime problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenTelemetryRuntimeDiagnostic {
    /// Stable identifier for the diagnostic condition.
    pub code: String,
    /// Distinct messages recorded for this condition, retained within a bounded budget.
    pub message: String,
    /// Total number of occurrences recorded for this condition.
    pub count: u64,
}

/// Snapshot of bounded runtime diagnostics recorded by an OpenTelemetry subscriber.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenTelemetryRuntimeDiagnostics {
    diagnostics: Vec<OpenTelemetryRuntimeDiagnostic>,
}

impl OpenTelemetryRuntimeDiagnostics {
    /// Return diagnostics in stable code order.
    pub fn entries(&self) -> &[OpenTelemetryRuntimeDiagnostic] {
        &self.diagnostics
    }

    /// Return a diagnostic by its stable code.
    pub fn get(&self, code: &str) -> Option<&OpenTelemetryRuntimeDiagnostic> {
        self.diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == code)
    }
}

#[derive(Debug, Default)]
struct RuntimeDiagnosticState {
    diagnostics: BTreeMap<String, RuntimeDiagnosticEntry>,
}

#[derive(Debug)]
struct RuntimeDiagnosticEntry {
    diagnostic: OpenTelemetryRuntimeDiagnostic,
    messages: BTreeSet<String>,
}

/// Shared runtime-diagnostic recorder for one independently owned OTLP subscriber.
#[derive(Debug, Clone)]
pub(super) struct SignalRuntimeDiagnostics {
    state: Arc<Mutex<RuntimeDiagnosticState>>,
    plugin_field: Option<String>,
}

impl SignalRuntimeDiagnostics {
    pub(super) fn new(plugin_field: Option<String>) -> Self {
        Self {
            state: Arc::new(Mutex::new(RuntimeDiagnosticState::default())),
            plugin_field,
        }
    }

    pub(super) fn record(&self, code: impl Into<String>, message: String, count: u64) -> u64 {
        let code = code.into();
        let count = count.max(1);
        let message = truncate_runtime_diagnostic_message(message);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let total = if let Some(diagnostic) = state.diagnostics.get_mut(&code) {
            diagnostic.diagnostic.count = diagnostic.diagnostic.count.saturating_add(count);
            if diagnostic.messages.insert(message.clone()) {
                let combined = combine_runtime_diagnostic_messages(&diagnostic.messages);
                if combined.chars().count() <= MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS {
                    diagnostic.diagnostic.message = combined;
                } else {
                    diagnostic.messages.remove(&message);
                }
            }
            diagnostic.diagnostic.count
        } else if state.diagnostics.len() < MAX_RUNTIME_DIAGNOSTICS {
            let mut messages = BTreeSet::new();
            messages.insert(message.clone());
            state.diagnostics.insert(
                code.clone(),
                RuntimeDiagnosticEntry {
                    diagnostic: OpenTelemetryRuntimeDiagnostic {
                        code: code.clone(),
                        message: message.clone(),
                        count,
                    },
                    messages,
                },
            );
            count
        } else {
            return 0;
        };
        drop(state);

        record_signal_runtime_diagnostic(&code, self.plugin_field.clone(), message, count);
        total
    }

    pub(super) fn snapshot(&self) -> OpenTelemetryRuntimeDiagnostics {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        OpenTelemetryRuntimeDiagnostics {
            diagnostics: state
                .diagnostics
                .values()
                .map(|entry| entry.diagnostic.clone())
                .collect(),
        }
    }

    pub(super) fn has_plugin_mirror(&self) -> bool {
        self.plugin_field.is_some()
    }
}

pub(super) fn should_relog_runtime_diagnostic(count: u64) -> bool {
    count.is_power_of_two()
}

/// Report capacity fallback without including high-cardinality resource keys.
pub(super) fn record_resource_pipeline_limit(diagnostics: &SignalRuntimeDiagnostics, signal: &str) {
    let message = format!(
        "OpenTelemetry {signal} resource pipeline limit of {MAX_DYNAMIC_SIGNAL_PIPELINES} reached; new resources use the configured base resource"
    );
    let count = diagnostics.record("otel.resource_metadata_pipeline_limit", message.clone(), 1);
    if should_relog_runtime_diagnostic(count) {
        log::warn!(
            target: "nemo_relay.observability",
            event = "otel_resource_metadata_pipeline_limit";
            "{message}"
        );
    }
}

/// Retry a batch-processor control message while the data queue is transiently full.
///
/// The SDK sends flush and shutdown control messages with `try_send`. A full data queue can
/// therefore reject the control message before it reaches the worker, even though the worker
/// will make room shortly. Retrying that specific condition preserves the flush/shutdown
/// barrier without masking exporter or shutdown failures.
pub(super) fn retry_batch_processor_channel_full(
    timeout: Duration,
    mut operation: impl FnMut() -> OTelSdkResult,
) -> OTelSdkResult {
    const RETRY_DELAY: Duration = Duration::from_millis(1);

    let started = Instant::now();
    loop {
        let result = operation();
        let is_channel_full = matches!(
            &result,
            Err(OTelSdkError::InternalFailure(message))
                if message.contains("ChannelFull")
                    || message.to_ascii_lowercase().contains("channel is full")
        );
        let elapsed = started.elapsed();
        if !is_channel_full || elapsed >= timeout {
            return result;
        }

        thread::sleep(RETRY_DELAY.min(timeout - elapsed));
    }
}

fn truncate_runtime_diagnostic_message(message: String) -> String {
    if message.chars().count() <= MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS {
        return message;
    }
    let mut truncated = message
        .chars()
        .take(MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS - 1)
        .collect::<String>();
    truncated.push('…');
    truncated
}

fn combine_runtime_diagnostic_messages(messages: &BTreeSet<String>) -> String {
    messages
        .iter()
        .fold(String::new(), |mut combined, message| {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(message);
            combined
        })
}

pub(super) enum MetricMarkClassification {
    NotMetric,
    Valid(Vec<ValidatedMetricMeasurement>),
    Invalid(String),
}

pub(super) fn classify_metric_mark(event: &Event) -> MetricMarkClassification {
    if event.scope_category().is_some() {
        return MetricMarkClassification::NotMetric;
    }
    let Some(schema) = event.data_schema() else {
        return MetricMarkClassification::NotMetric;
    };
    if schema.name != METRIC_DATA_SCHEMA_NAME {
        return MetricMarkClassification::NotMetric;
    }
    if schema.version != METRIC_DATA_SCHEMA_VERSION {
        return MetricMarkClassification::Invalid(format!(
            "unsupported metric schema version {:?}",
            schema.version
        ));
    }
    let measurements = match event
        .data()
        .cloned()
        .ok_or_else(|| "metric mark data is missing".to_string())
        .and_then(|data| {
            serde_json::from_value::<MetricEnvelope>(data)
                .map_err(|error| format!("invalid metric envelope: {error}"))
        })
        .and_then(|envelope| {
            envelope
                .validated_measurements()
                .map_err(|error| error.to_string())
        }) {
        Ok(measurements) => measurements,
        Err(error) => return MetricMarkClassification::Invalid(error),
    };
    MetricMarkClassification::Valid(measurements)
}

/// Tokio runtime retained for the lifetime of an OTLP provider.
pub(super) struct SignalExporterRuntime {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for SignalExporterRuntime {
    fn drop(&mut self) {
        self.stop.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Construct a provider inside a dedicated Tokio runtime and retain that runtime.
pub(super) fn build_in_owned_runtime<T, F>(
    thread_name: &str,
    build: F,
) -> Result<(T, SignalExporterRuntime)>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let (result_sender, result_receiver) = mpsc::sync_channel(1);
    let (stop_sender, stop_receiver) = mpsc::channel();
    let runtime_thread = thread::Builder::new()
        .name(thread_name.to_string())
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
            let result = {
                let _guard = runtime.enter();
                build()
            };
            if result_sender.send(result).is_err() {
                return;
            }
            let _ = stop_receiver.recv();
        })
        .map_err(|error| OpenTelemetryError::ExporterBuild(error.to_string()))?;

    let value = result_receiver.recv().map_err(|error| {
        OpenTelemetryError::ExporterBuild(format!("exporter runtime stopped unexpectedly: {error}"))
    })??;
    Ok((
        value,
        SignalExporterRuntime {
            stop: Some(stop_sender),
            thread: Some(runtime_thread),
        },
    ))
}

pub(super) fn validate_signal_headers(headers: &HashMap<String, String>) -> Result<()> {
    let mut normalized = HashSet::new();
    for (key, value) in headers {
        if key.trim().is_empty() || key.trim() != key {
            return Err(OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: "header name must be nonblank and have no surrounding whitespace"
                    .to_string(),
            });
        }
        if value.trim().is_empty() || value.trim() != value {
            return Err(OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: "header value must be nonblank and have no surrounding whitespace"
                    .to_string(),
            });
        }
        if !normalized.insert(key.to_ascii_lowercase()) {
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

pub(super) fn resolve_header_env(
    headers: &HashMap<String, String>,
    header_env: &HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    let mut normalized = HashSet::new();
    for key in headers.keys() {
        normalized.insert(key.to_ascii_lowercase());
    }

    for (key, variable) in header_env {
        if !normalized.insert(key.to_ascii_lowercase()) {
            return Err(OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message:
                    "header names must be unique across headers and header_env ignoring ASCII case"
                        .to_string(),
            });
        }
        reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
            OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: error.to_string(),
            }
        })?;
        if variable.trim().is_empty()
            || variable.trim() != variable
            || variable.contains(['\0', '='])
        {
            return Err(OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: "header_env must name a nonblank environment variable without surrounding whitespace, '=' or NUL"
                    .to_string(),
            });
        }
    }

    let mut resolved = headers.clone();
    for (key, variable) in header_env {
        let value = match std::env::var(variable) {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => {
                return Err(OpenTelemetryError::InvalidHeader {
                    key: key.clone(),
                    message: format!("environment variable {variable:?} is not set"),
                });
            }
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(OpenTelemetryError::InvalidHeader {
                    key: key.clone(),
                    message: format!("environment variable {variable:?} is not valid Unicode"),
                });
            }
        };
        if value.trim().is_empty() || value.trim() != value {
            return Err(OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: format!(
                    "environment variable {variable:?} must contain a nonblank value without surrounding whitespace"
                ),
            });
        }
        reqwest::header::HeaderValue::from_str(&value).map_err(|_| {
            OpenTelemetryError::InvalidHeader {
                key: key.clone(),
                message: format!(
                    "environment variable {variable:?} does not contain a valid header value"
                ),
            }
        })?;
        resolved.insert(key.clone(), value);
    }
    Ok(resolved)
}

pub(super) fn reject_signal_header_environment(signal_variable: &'static str) -> Result<()> {
    for variable in ["OTEL_EXPORTER_OTLP_HEADERS", signal_variable] {
        if std::env::var_os(variable).is_some_and(|value| !value.is_empty()) {
            return Err(OpenTelemetryError::GlobalHeaderEnvironmentUnsupported { variable });
        }
    }
    Ok(())
}

pub(super) fn resolve_http_signal_endpoint<'a>(endpoint: &'a str, signal: &str) -> Cow<'a, str> {
    let Ok(mut parsed) = reqwest::Url::parse(endpoint) else {
        return Cow::Borrowed(endpoint);
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        return Cow::Borrowed(endpoint);
    }

    let path = parsed.path();
    if path == "/" {
        parsed.set_path(&format!("/v1/{signal}"));
        return Cow::Owned(parsed.into());
    }

    if path == "/v1/traces" || path.ends_with("/v1/traces") {
        let prefix = path.strip_suffix("/v1/traces").unwrap_or_default();
        parsed.set_path(&format!("{prefix}/v1/{signal}"));
        return Cow::Owned(parsed.into());
    }
    Cow::Borrowed(endpoint)
}

pub(super) fn build_grpc_metadata(headers: &HashMap<String, String>) -> Result<MetadataMap> {
    let mut metadata = MetadataMap::new();
    for (key, value) in headers {
        let key = MetadataKey::from_bytes(key.as_bytes()).map_err(|error| {
            OpenTelemetryError::InvalidGrpcHeader {
                key: key.clone(),
                message: error.to_string(),
            }
        })?;
        let value = MetadataValue::try_from(value.as_str()).map_err(|error| {
            OpenTelemetryError::InvalidGrpcHeader {
                key: key.to_string(),
                message: error.to_string(),
            }
        })?;
        metadata.insert(key, value);
    }
    Ok(metadata)
}

pub(super) fn record_signal_runtime_diagnostic(
    code: &str,
    field: Option<String>,
    message: String,
    count: u64,
) {
    record_active_plugin_runtime_diagnostic(RuntimeDiagnostic {
        code: code.to_string(),
        component: "observability".to_string(),
        field,
        message,
        session_id: None,
        count,
    });
}

pub(super) fn signal_resource(
    service_name: &str,
    service_namespace: Option<&str>,
    service_version: Option<&str>,
    resource_attributes: &HashMap<String, String>,
) -> Resource {
    telemetry_resource(signal_resource_attributes(
        Some(service_name),
        service_namespace,
        service_version,
        resource_attributes,
    ))
}

pub(super) fn signal_resource_attributes(
    service_name: Option<&str>,
    service_namespace: Option<&str>,
    service_version: Option<&str>,
    resource_attributes: &HashMap<String, String>,
) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    if let Some(service_name) = service_name {
        attributes.push(KeyValue::new("service.name", service_name.to_string()));
    }
    if let Some(namespace) = service_namespace {
        attributes.push(KeyValue::new("service.namespace", namespace.to_string()));
    }
    if let Some(version) = service_version {
        attributes.push(KeyValue::new("service.version", version.to_string()));
    }
    attributes.extend(
        resource_attributes
            .iter()
            .map(|(key, value)| KeyValue::new(key.clone(), value.clone())),
    );
    attributes
}

pub(super) fn canonical_resource_key(attributes: &[KeyValue]) -> String {
    let mut entries = attributes
        .iter()
        .map(|attribute| format!("{}={:?}", attribute.key.as_str(), attribute.value))
        .collect::<Vec<_>>();
    entries.sort();
    entries.join("\u{1f}")
}

pub(super) fn promoted_signal_resource_attributes(
    event: &Event,
    prefixes: &[String],
    service_name: Option<&str>,
    service_namespace: Option<&str>,
    service_version: Option<&str>,
    resource_attributes: &HashMap<String, String>,
    runtime_diagnostics: &SignalRuntimeDiagnostics,
) -> Option<(String, Vec<KeyValue>)> {
    if prefixes.is_empty()
        || event.scope_category() != Some(ScopeCategory::Start)
        || (event.parent_uuid().is_some() && event.propagation_parent_uuid() == event.parent_uuid())
    {
        return None;
    }
    let mut attributes = signal_resource_attributes(
        service_name,
        service_namespace,
        service_version,
        resource_attributes,
    );
    let mut protected_keys = attributes
        .iter()
        .map(|attribute| attribute.key.as_str().to_string())
        .collect::<HashSet<_>>();
    protected_keys.extend(
        TELEMETRY_SDK_RESOURCE_ATTRIBUTE_KEYS
            .iter()
            .map(|key| (*key).to_string()),
    );
    let promotion =
        promote_event_metadata_attributes(&mut attributes, event, prefixes, &protected_keys);
    record_metadata_promotion_issues(runtime_diagnostics, promotion.issues, "resource_metadata");
    let key = canonical_resource_key(&attributes);
    let base_key = canonical_resource_key(&signal_resource_attributes(
        service_name,
        service_namespace,
        service_version,
        resource_attributes,
    ));
    (key != base_key).then_some((key, attributes))
}

fn record_metadata_promotion_issues(
    runtime_diagnostics: &SignalRuntimeDiagnostics,
    mut issues: Vec<MetadataPromotionIssue>,
    kind: &str,
) {
    issues.sort_by(|left, right| left.key.cmp(&right.key));
    for issue in issues {
        let diagnostic_code = format!("otel.{kind}_promotion_value_unsupported.{}", issue.key);
        let diagnostic_count = runtime_diagnostics.record(
            diagnostic_code,
            format!(
                "OpenTelemetry {kind} attribute {:?} was not promoted: {}",
                issue.key, issue.reason
            ),
            1,
        );
        if should_relog_runtime_diagnostic(diagnostic_count) {
            log::warn!(
                target: "nemo_relay.observability",
                event = "otel_metadata_promotion_value_unsupported",
                metadata_key = issue.key.as_str();
                "OpenTelemetry {kind} attribute was not promoted: {}",
                issue.reason
            );
        }
    }
}

struct CompletedResourceRoute<T> {
    closed_at: DateTime<Utc>,
    route: T,
}

/// Tracks the resource selected by a root scope so child scopes and late marks
/// use the same signal provider.
pub(super) struct SignalResourceLineage<T> {
    active: HashMap<Uuid, T>,
    completed: HashMap<Uuid, CompletedResourceRoute<T>>,
    completed_expiry_index: BTreeMap<DateTime<Utc>, HashSet<Uuid>>,
}

impl<T: Clone> SignalResourceLineage<T> {
    pub(super) fn new() -> Self {
        Self {
            active: HashMap::new(),
            completed: HashMap::new(),
            completed_expiry_index: BTreeMap::new(),
        }
    }

    pub(super) fn process(
        &mut self,
        event: &Event,
        root_route: Option<T>,
        completed_context_ttl: Duration,
    ) -> Option<T> {
        self.expire_completed(*event.timestamp(), completed_context_ttl);
        match event.scope_category() {
            Some(ScopeCategory::Start) => {
                self.remove_completed(event.uuid());
                let route = self.parent_route(event).or(root_route);
                if let Some(route) = route.clone() {
                    self.active.insert(event.uuid(), route);
                }
                route
            }
            Some(ScopeCategory::End) => {
                let route = self.active.remove(&event.uuid());
                if let Some(route) = route.clone() {
                    self.record_completed(event.uuid(), *event.timestamp(), route);
                }
                route
            }
            None => self.parent_route(event).or(root_route),
        }
    }

    pub(super) fn existing_route(&self, event: &Event) -> Option<T> {
        if event.scope_category() == Some(ScopeCategory::End)
            && let Some(route) = self.active.get(&event.uuid())
        {
            return Some(route.clone());
        }
        self.parent_route(event)
    }

    fn parent_route(&self, event: &Event) -> Option<T> {
        let parent_uuid = event.parent_uuid()?;
        self.active.get(&parent_uuid).cloned().or_else(|| {
            self.completed
                .get(&parent_uuid)
                .map(|context| context.route.clone())
        })
    }

    fn remove_completed(&mut self, uuid: Uuid) {
        if let Some(context) = self.completed.remove(&uuid) {
            let remove_bucket = self
                .completed_expiry_index
                .get_mut(&context.closed_at)
                .is_some_and(|uuids| {
                    uuids.remove(&uuid);
                    uuids.is_empty()
                });
            if remove_bucket {
                self.completed_expiry_index.remove(&context.closed_at);
            }
        }
    }

    fn record_completed(&mut self, uuid: Uuid, closed_at: DateTime<Utc>, route: T) {
        self.remove_completed(uuid);
        self.completed
            .insert(uuid, CompletedResourceRoute { closed_at, route });
        self.completed_expiry_index
            .entry(closed_at)
            .or_default()
            .insert(uuid);
    }

    fn expire_completed(&mut self, timestamp: DateTime<Utc>, ttl: Duration) {
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
                .expect("completed resource-route expiry bucket exists");
            for uuid in uuids {
                self.completed.remove(&uuid);
            }
        }
    }
}
