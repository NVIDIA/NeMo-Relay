// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for independently owned OTLP signal providers.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::thread;

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};

use crate::api::event::{
    Event, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION, MetricEnvelope,
};
use crate::plugin::{RuntimeDiagnostic, record_active_plugin_runtime_diagnostic};

use super::otel::{OpenTelemetryError, Result};

pub(super) enum MetricMarkClassification {
    NotMetric,
    Valid(MetricEnvelope),
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
    let envelope = match event
        .data()
        .cloned()
        .ok_or_else(|| "metric mark data is missing".to_string())
        .and_then(|data| {
            serde_json::from_value::<MetricEnvelope>(data)
                .map_err(|error| format!("invalid metric envelope: {error}"))
        })
        .and_then(|envelope| {
            envelope
                .validate()
                .map_err(|error| error.to_string())
                .map(|()| envelope)
        }) {
        Ok(envelope) => envelope,
        Err(error) => return MetricMarkClassification::Invalid(error),
    };
    MetricMarkClassification::Valid(envelope)
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
    if field.is_none() {
        return;
    }
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
