// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for independently owned OTLP signal providers.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};

use crate::api::event::{
    Event, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION, MetricEnvelope,
    ValidatedMetricMeasurement,
};
use crate::plugin::{RuntimeDiagnostic, record_active_plugin_runtime_diagnostic};

use super::otel::{OpenTelemetryError, Result};

const MAX_RUNTIME_DIAGNOSTICS: usize = 32;
const MAX_RUNTIME_DIAGNOSTIC_MESSAGE_CHARS: usize = 1_024;

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
    let mut attributes = vec![KeyValue::new("service.name", service_name.to_string())];
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
    Resource::builder_empty()
        .with_attributes(attributes)
        .build()
}
