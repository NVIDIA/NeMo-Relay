// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core activation services for the crate-owned observability component.

#[cfg(test)]
#[allow(unused_imports)]
use crate::api::event::{Event, LogSeverity, ScopeCategory, ValidatedMetricMeasurement};
#[cfg(test)]
#[allow(unused_imports)]
use crate::api::runtime::{EventSubscriberFn, current_scope_stack, global_context};
#[cfg(test)]
#[allow(unused_imports)]
use crate::api::scope::ScopeType;
#[cfg(test)]
#[allow(unused_imports)]
use crate::api::subscriber::{
    flush_subscribers, scope_deregister_subscriber, try_scope_deregister_subscriber,
    try_scope_register_subscriber,
};
#[cfg(test)]
#[allow(unused_imports)]
use crate::config_editor::{
    EditorConfig, EditorFieldKind, EditorFieldSpec, EditorListItemSpec, EditorSchema,
    EditorTaggedUnionSpec, EditorVariantSpec,
};
#[cfg(test)]
#[allow(unused_imports)]
use crate::error::FlowError;
#[cfg(test)]
#[allow(unused_imports)]
use crate::plugin::{
    ATIF_RUNTIME_DELIVERY_FAILURE_MARKER, ConfigDiagnostic, ConfigPolicy, DiagnosticLevel,
    OTEL_RUNTIME_DELIVERY_FAILURE_MARKER, Plugin, PluginComponentSpec, PluginError,
    PluginRegistration, PluginRegistrationCleanupOutcome, PluginRegistrationContext,
    Result as PluginResult, UnsupportedBehavior, apply_global_config_policy, deregister_plugin,
    register_builtin_plugin,
};
#[cfg(test)]
#[allow(unused_imports)]
use crate::plugin::{RuntimeDiagnostic, record_active_plugin_runtime_diagnostic};
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::OpenTelemetryTlsConfig;
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::atif::{AtifAgentInfo, AtifExporter};
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::atof::{
    AtofEndpointFieldNamePolicy, AtofEndpointTransport, AtofExporter,
    AtofExporterConfig as CoreAtofExporterConfig, AtofExporterMode, AtofFileSinkConfig,
    AtofSinkConfig as CoreAtofSinkConfig, AtofStreamSinkConfig,
};
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::header_file::{PlaintextPolicy, redacted_origin};
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::otel::{
    OpenTelemetryConfig as CoreOpenTelemetryConfig,
    OpenTelemetryFileSinkConfig as CoreOpenTelemetryFileSinkConfig, OpenTelemetrySubscriber,
    OtlpFileSinkMode, OtlpFileSinkSettings, OtlpTransport, absolute_otlp_file_sink_directory,
    default_otlp_file_sink_filename, otlp_file_sink_directory_is_blank,
    resolve_http_trace_endpoint, validate_otlp_file_sink_filename,
};
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::otel_file::OtlpFileFormat;
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::otel_logs::{
    OpenTelemetryLogConfig as CoreOpenTelemetryLogConfig, OpenTelemetryLogSubscriber,
    resolve_http_log_endpoint,
};
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::otel_metrics::{
    MetricTemporality, OpenTelemetryMetricConfig as CoreOpenTelemetryMetricConfig,
    OpenTelemetryMetricSubscriber, resolve_http_metric_endpoint,
};
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::otel_signal::{
    MetricMarkClassification, classify_metric_mark, validate_signal_headers,
};
#[allow(unused_imports)]
pub(crate) use nemo_relay_observability::plugin::*;
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::private_file::atomic_private_write;
#[cfg(test)]
#[allow(unused_imports)]
use nemo_relay_observability::{
    MarkProjection, OpenTelemetryType, OtlpAttributeMapping, default_mark_exclude_names,
    validate_attribute_mappings, validate_metadata_promotion_prefixes,
};
#[cfg(test)]
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};
#[cfg(test)]
#[allow(unused_imports)]
use serde_json::{Map, Value as Json};
#[cfg(test)]
#[allow(unused_imports)]
use std::borrow::Cow;
#[cfg(test)]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[cfg(test)]
#[allow(unused_imports)]
use std::future::Future;
#[cfg(test)]
#[allow(unused_imports)]
use std::net::IpAddr;
#[cfg(test)]
#[allow(unused_imports)]
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(test)]
#[allow(unused_imports)]
use std::path::{Component, Path, PathBuf};
#[cfg(test)]
#[allow(unused_imports)]
use std::pin::Pin;
#[cfg(test)]
#[allow(unused_imports)]
use std::sync::atomic::AtomicU8;
#[cfg(test)]
#[allow(unused_imports)]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(test)]
#[allow(unused_imports)]
use std::sync::{Arc, Mutex};
#[cfg(test)]
#[allow(unused_imports)]
use std::time::Duration;
#[cfg(test)]
#[allow(unused_imports)]
use uuid::Uuid;

pub(crate) fn register_observability_component() -> crate::plugin::Result<()> {
    crate::plugin::register_builtin_plugin(std::sync::Arc::new(ObservabilityPlugin::new(
        crate::observability_runtime::core_host(),
    )))
}
#[cfg(test)]
fn deregister_observability_component() -> bool {
    crate::plugin::deregister_plugin(OBSERVABILITY_PLUGIN_KIND)
}
#[cfg(test)]
#[path = "../../integration/observability/plugin_component_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../integration/observability/otel_tls_tests.rs"]
mod tls_tests;

pub(crate) fn shutdown_opentelemetry_subscribers(
    subscribers: &[Arc<OpenTelemetrySubscriber>],
    host: &dyn nemo_relay_observability::host::ObservabilityHost,
) -> Option<OpenTelemetryShutdownFailure> {
    let mut errors = Vec::new();
    if let Err(error) = host.flush_subscribers() {
        errors.push(OpenTelemetryShutdownIssue::other(
            nemo_relay_observability::otel::OpenTelemetryError::Core(error),
        ));
    }
    errors.extend(shutdown_opentelemetry_providers(subscribers));
    shutdown_failure_from_errors(errors)
}

pub(crate) fn shutdown_opentelemetry_providers(
    subscribers: &[Arc<OpenTelemetrySubscriber>],
) -> Vec<OpenTelemetryShutdownIssue> {
    let mut errors = Vec::new();
    for subscriber in subscribers {
        if let Err(error) = subscriber.shutdown_provider() {
            errors.push(OpenTelemetryShutdownIssue::trace(error));
        }
    }
    errors
}
