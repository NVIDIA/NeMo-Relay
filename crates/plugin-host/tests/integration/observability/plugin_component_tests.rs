// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core plugin-host integration with the observability component.

use super::*;
use crate::observability_test_support::otel::ExporterInitializationError;
use crate::test_support::plugin_host::{
    test_close_plugin_host, test_initialize_plugin_host_exact, test_validate_static_plugin_config,
};
use nemo_relay_plugin_host::plugin_config_schema;
use nemo_relay_plugin_host::{PluginComponentSpec, PluginConfig, list_plugin_kinds, lookup_plugin};
use nemo_relay_runtime::api::event::{
    BaseEvent, DataSchema, EventCategory, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION,
    MarkEvent, ScopeEvent,
};
use nemo_relay_runtime::api::runtime::{
    NemoRelayContextState, PropagationContext, ThreadScopeStackBinding, capture_thread_scope_stack,
    create_scope_stack_from_propagation, global_context, restore_thread_scope_stack,
    set_thread_scope_stack,
};
use nemo_relay_runtime::api::scope::{PopScopeParams, PushScopeParams};
use nemo_relay_runtime::api::subscriber::scope_deregister_subscriber;
use nemo_relay_runtime::config_editor::EditorFieldKind;
use serde_json::json;
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

struct RestoreThreadScopeStackGuard(ThreadScopeStackBinding);

impl Drop for RestoreThreadScopeStackGuard {
    fn drop(&mut self) {
        restore_thread_scope_stack(self.0.clone());
    }
}

// Restores test-mutated environment variables, including during unwinding.
struct EnvironmentGuard(Vec<(String, Option<OsString>)>);

impl EnvironmentGuard {
    fn capture(variables: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self(
            variables
                .into_iter()
                .map(|variable| {
                    let variable = variable.as_ref().to_string();
                    let value = std::env::var_os(&variable);
                    (variable, value)
                })
                .collect(),
        )
    }
}

impl Drop for EnvironmentGuard {
    fn drop(&mut self) {
        // SAFETY: the observability mutex serializes test-only environment changes.
        unsafe {
            for (variable, value) in &self.0 {
                match value {
                    Some(value) => std::env::set_var(variable, value),
                    None => std::env::remove_var(variable),
                }
            }
        }
    }
}

#[derive(Debug)]
struct ShutdownFailureSpanProcessor {
    message: String,
    shutdown_calls: Arc<AtomicUsize>,
}

impl opentelemetry_sdk::trace::SpanProcessor for ShutdownFailureSpanProcessor {
    fn on_start(&self, _span: &mut opentelemetry_sdk::trace::Span, _cx: &opentelemetry::Context) {}

    fn on_end(&self, _span: opentelemetry_sdk::trace::SpanData) {}

    fn force_flush(&self) -> opentelemetry_sdk::error::OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> opentelemetry_sdk::error::OTelSdkResult {
        self.shutdown_calls.fetch_add(1, Ordering::SeqCst);
        Err(opentelemetry_sdk::error::OTelSdkError::InternalFailure(
            self.message.clone(),
        ))
    }
}

fn temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("nemo-relay-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path.canonicalize().unwrap()
}

#[test]
fn automatic_otel_skips_all_disallowed_plaintext_signals_without_headers() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let _environment = plaintext_preflight_environment();
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://collector.example:4318",
        );
    }

    let config = plugin_config(json!({"version": 4}));
    let report = futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    let skipped = report
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "observability.automatic_otel_signal_skipped")
        .collect::<Vec<_>>();
    assert_eq!(skipped.len(), 3, "diagnostics: {:?}", report.diagnostics);
    for signal in ["traces", "logs", "metrics"] {
        assert!(
            skipped.iter().any(|diagnostic| {
                diagnostic
                    .message
                    .contains(&format!("OpenTelemetry {signal} exporter"))
                    && diagnostic.field.as_deref() == Some("OTEL_EXPORTER_OTLP_ENDPOINT")
                    && diagnostic
                        .message
                        .contains("plaintext destination rejected by transport policy")
            }),
            "{signal}: {skipped:?}"
        );
    }
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .is_empty()
    );
    crate::test_support::plugin_host::test_close_plugin_host().unwrap();
}

#[test]
fn automatic_otel_rejection_reports_redacted_trace_source_and_keeps_valid_metrics() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let variables = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_SDK_DISABLED",
        "OTEL_TRACES_EXPORTER",
        "OTEL_METRICS_EXPORTER",
    ];
    let _environment = EnvironmentGuard::capture(variables);
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        for variable in variables {
            std::env::remove_var(variable);
        }
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://relay-user:relay-secret@collector.example:4318/private/traces?token=relay-query-secret",
        );
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            "http://127.0.0.1:4318/v1/metrics",
        );
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
            "authorization=Bearer%20relay-header-secret",
        );
    }

    let config = plugin_config(json!({"version": 4, "opentelemetry": {"enabled": true}}));
    let report = futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    let skipped = report
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "observability.automatic_otel_signal_skipped")
        .collect::<Vec<_>>();
    assert_eq!(skipped.len(), 1, "diagnostics: {:?}", report.diagnostics);
    let diagnostic = skipped[0];
    assert_eq!(diagnostic.level, DiagnosticLevel::Warning);
    assert_eq!(
        diagnostic.field.as_deref(),
        Some("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    );
    assert!(diagnostic.message.contains("traces"));
    assert!(diagnostic.message.contains("http://collector.example:4318"));
    assert!(
        diagnostic
            .message
            .contains("insecure_http_hosts or NEMO_RELAY_OTEL_INSECURE_HTTP_HOSTS")
    );
    for secret in [
        "relay-user",
        "relay-secret",
        "/private/traces",
        "relay-query-secret",
        "relay-header-secret",
    ] {
        assert!(
            !diagnostic.message.contains(secret),
            "diagnostic exposed {secret}"
        );
    }
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry.automatic")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn automatic_otel_reports_generic_source_and_accepts_allowlisted_named_host() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let variables = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_SDK_DISABLED",
        "OTEL_TRACES_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_METRICS_EXPORTER",
    ];
    let _environment = EnvironmentGuard::capture(variables);
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        for variable in variables {
            std::env::remove_var(variable);
        }
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://collector.example:4318/private?token=relay-query-secret",
        );
    }
    let config = plugin_config(json!({"version": 4, "opentelemetry": {"enabled": true}}));
    let report =
        futures::executor::block_on(test_initialize_plugin_host_exact(config.clone())).unwrap();
    let skipped = report
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "observability.automatic_otel_signal_skipped")
        .collect::<Vec<_>>();
    assert_eq!(skipped.len(), 3, "diagnostics: {:?}", report.diagnostics);
    for diagnostic in skipped {
        assert_eq!(
            diagnostic.field.as_deref(),
            Some("OTEL_EXPORTER_OTLP_ENDPOINT")
        );
        assert!(diagnostic.message.contains("http://collector.example:4318"));
        assert!(!diagnostic.message.contains("/private"));
        assert!(!diagnostic.message.contains("relay-query-secret"));
    }
    assert!(
        !global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry.automatic")
    );
    test_close_plugin_host().unwrap();

    let allowed = plugin_config(json!({
        "version": 4,
        "insecure_http_hosts": ["collector.example"],
        "opentelemetry": {"enabled": true}
    }));
    let report = futures::executor::block_on(test_initialize_plugin_host_exact(allowed)).unwrap();
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "observability.automatic_otel_signal_skipped" }),
        "diagnostics: {:?}",
        report.diagnostics
    );
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry.automatic")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn automatic_insecure_host_override_does_not_allow_configured_exporters() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let variables = [
        AUTOMATIC_INSECURE_HTTP_HOSTS_VARIABLE,
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_SDK_DISABLED",
        "OTEL_TRACES_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_METRICS_EXPORTER",
    ];
    let _environment = EnvironmentGuard::capture(variables);
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        for variable in variables {
            std::env::remove_var(variable);
        }
        std::env::set_var(
            AUTOMATIC_INSECURE_HTTP_HOSTS_VARIABLE,
            "collector.example, other.example",
        );
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://collector.example:4318",
        );
    }
    let automatic = plugin_config(json!({
        "version": 4,
        "opentelemetry": {"enabled": true}
    }));
    let report = futures::executor::block_on(test_initialize_plugin_host_exact(automatic)).unwrap();
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "observability.automatic_otel_signal_skipped" })
    );
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry.automatic")
    );
    test_close_plugin_host().unwrap();

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::remove_var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") };
    let configured = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{"type": "full", "endpoint": "http://collector.example:4318"}]
        }
    }));
    let error = futures::executor::block_on(test_initialize_plugin_host_exact(configured))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("OTLP trace endpoints require HTTPS"),
        "{error}"
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn invalid_automatic_insecure_host_override_is_ignored_without_failing_activation() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let variables = [
        AUTOMATIC_INSECURE_HTTP_HOSTS_VARIABLE,
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_SDK_DISABLED",
        "OTEL_TRACES_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_METRICS_EXPORTER",
    ];
    let _environment = EnvironmentGuard::capture(variables);
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        for variable in variables {
            std::env::remove_var(variable);
        }
        std::env::set_var(
            AUTOMATIC_INSECURE_HTTP_HOSTS_VARIABLE,
            "collector.example,https://private.example/token?secret=hidden",
        );
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://collector.example:4318",
        );
    }
    let config = plugin_config(json!({"version": 4, "opentelemetry": {"enabled": true}}));
    let report = futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.invalid_automatic_insecure_http_hosts"
            && diagnostic.level == DiagnosticLevel::Warning
            && diagnostic.field.as_deref() == Some(AUTOMATIC_INSECURE_HTTP_HOSTS_VARIABLE)
            && !diagnostic.message.contains("hidden")
    }));
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "observability.automatic_otel_signal_skipped" })
    );
    assert!(
        !global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry.automatic")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn automatic_otel_skip_keeps_a_valid_configured_destination_active() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let variables = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_SDK_DISABLED",
        "OTEL_TRACES_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_METRICS_EXPORTER",
    ];
    let _environment = EnvironmentGuard::capture(variables);
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        for variable in variables {
            std::env::remove_var(variable);
        }
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://collector.example:4318",
        );
    }
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{"type": "full", "endpoint": "http://127.0.0.1:4318"}]
        }
    }));
    let report = futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.automatic_otel_signal_skipped"
            && diagnostic.level == DiagnosticLevel::Warning
    }));
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn automatic_otel_exporter_is_added_to_configured_relay_endpoints() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let endpoint = "OTEL_EXPORTER_OTLP_ENDPOINT";
    let _environment = EnvironmentGuard::capture([endpoint]);

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::set_var(endpoint, "http://127.0.0.1:4318") };

    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://127.0.0.1:4319"
            }]
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let state = global_context();
    let state = state.read().unwrap();
    assert_eq!(state.event_subscribers.len(), 2);
    drop(state);
    crate::test_support::plugin_host::test_close_plugin_host().unwrap();
}

#[test]
fn automatic_otel_global_headers_survive_skipped_configured_endpoints() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let (endpoint, captures) = start_http_capture_server(3);
    let _environment = EnvironmentGuard::capture([
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_HEADERS",
    ]);
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", &endpoint);
        std::env::set_var("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json");
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "authorization=Bearer%20automatic-token",
        );
    }
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{"type": "gen_ai", "endpoint": endpoint}],
            "logs": {"enabled": true},
            "metrics": {"enabled": true}
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    let agent = push_agent("automatic-header-regression");
    nemo_relay_runtime::api::scope::event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("sample.log")
            .data(json!({"message": "exported"}))
            .build(),
    )
    .unwrap();
    nemo_relay_runtime::api::scope::event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("sample")
            .data(json!({"measurements": [{
                "name": "sample.counter", "kind": "counter", "value_type": "u64", "value": 1
            }]}))
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
    )
    .unwrap();
    pop(&agent);
    flush_subscribers().unwrap();
    crate::test_support::plugin_host::test_close_plugin_host().unwrap();
    let requests = wait_for_captures(&captures, 3);
    assert_eq!(requests.len(), 3);
    for path in ["/v1/traces", "/v1/logs", "/v1/metrics"] {
        let request = requests
            .iter()
            .find(|request| request.headers.contains(path))
            .unwrap();
        assert!(
            request
                .headers
                .contains("authorization: Bearer automatic-token")
        );
        assert!(!request.body.contains("automatic-token"));
    }
}

#[test]
fn automatic_otel_malformed_percent_headers_skip_exporters_without_disclosing_credentials() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let endpoint = "OTEL_EXPORTER_OTLP_ENDPOINT";
    let generic = "OTEL_EXPORTER_OTLP_HEADERS";
    let signal_headers = [
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
        "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
    ];
    let _environment =
        EnvironmentGuard::capture([endpoint, generic].into_iter().chain(signal_headers));
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(endpoint, "http://127.0.0.1:4318");
        for variable in signal_headers {
            std::env::remove_var(variable);
        }
    }
    for malformed in ["%ZZ", "%", "%2", "%2Z"] {
        // SAFETY: the observability mutex serializes test-only environment changes.
        unsafe {
            std::env::set_var(
                generic,
                format!("authorization=Bearer%20secret-token,x-bad={malformed}"),
            );
        }
        let errors = [
            OpenTelemetrySubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
            .err()
            .unwrap(),
            OpenTelemetryLogSubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
            .err()
            .unwrap(),
            OpenTelemetryMetricSubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .map(|exporter| exporter.with_host(crate::observability_runtime::runtime_host()))
            .err()
            .unwrap(),
        ];
        for error in errors {
            let message = error.to_string();
            assert!(message.contains("malformed percent escape"));
            assert!(!message.contains("secret-token"));
            assert!(!message.contains(malformed));
        }
    }
    reset_runtime();
    let collector = TcpListener::bind("127.0.0.1:0").unwrap();
    collector.set_nonblocking(true).unwrap();
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(
            endpoint,
            format!("http://{}", collector.local_addr().unwrap()),
        );
    }
    let report = futures::executor::block_on(test_initialize_plugin_host_exact(plugin_config(
        json!({"version": 4}),
    )))
    .unwrap();
    let skipped = report
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "observability.automatic_otel_signal_skipped")
        .collect::<Vec<_>>();
    assert_eq!(skipped.len(), 3, "diagnostics: {:?}", report.diagnostics);
    for diagnostic in skipped {
        assert_eq!(diagnostic.field.as_deref(), Some(endpoint));
        assert!(!diagnostic.message.contains("secret-token"));
        assert!(!diagnostic.message.contains("%2Z"));
    }
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .is_empty()
    );
    let agent = push_agent("malformed-header-regression");
    pop(&agent);
    flush_subscribers().unwrap();
    crate::test_support::plugin_host::test_close_plugin_host().unwrap();
    assert!(
        matches!(collector.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
    );
}

#[test]
fn disabled_opentelemetry_section_suppresses_automatic_exporters() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let endpoint = "OTEL_EXPORTER_OTLP_ENDPOINT";
    let _environment = EnvironmentGuard::capture([endpoint]);

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::set_var(endpoint, "http://127.0.0.1:4318") };

    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {"enabled": false}
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let state = global_context();
    assert!(state.read().unwrap().event_subscribers.is_empty());
    crate::test_support::plugin_host::test_close_plugin_host().unwrap();
}

#[test]
fn enabled_empty_opentelemetry_section_allows_automatic_exporters() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let endpoint = "OTEL_EXPORTER_OTLP_ENDPOINT";
    let disabled = "OTEL_SDK_DISABLED";
    let _environment = EnvironmentGuard::capture([endpoint, disabled]);

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(endpoint, "http://127.0.0.1:4318");
        std::env::remove_var(disabled);
    }
    assert!(automatic_otlp_signals().is_some());

    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {"enabled": true}
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let state = global_context();
    assert_eq!(state.read().unwrap().event_subscribers.len(), 1);
    crate::test_support::plugin_host::test_close_plugin_host().unwrap();
}

#[derive(Clone, Debug)]
struct HttpCapture {
    headers: String,
    body: String,
}

fn start_http_capture_server(expected_requests: usize) -> (String, Arc<Mutex<Vec<HttpCapture>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let captures = Arc::new(Mutex::new(Vec::new()));
    let thread_captures = Arc::clone(&captures);
    std::thread::spawn(move || {
        for _ in 0..expected_requests {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0_u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            let headers = String::from_utf8_lossy(&request);
            let length = headers
                .lines()
                .find_map(|line| {
                    line.split_once(':').and_then(|(name, value)| {
                        name.eq_ignore_ascii_case("content-length")
                            .then_some(value.trim())
                    })
                })
                .unwrap()
                .parse::<usize>()
                .unwrap();
            let mut body = vec![0_u8; length];
            stream.read_exact(&mut body).unwrap();
            thread_captures.lock().unwrap().push(HttpCapture {
                headers: headers.into_owned(),
                body: String::from_utf8(body).unwrap(),
            });
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
        }
    });
    (url, captures)
}

fn start_http_status_server(
    status: &'static str,
) -> (String, std::thread::JoinHandle<std::io::Result<()>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || -> std::io::Result<()> {
        listener.set_nonblocking(true)?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let (mut stream, _) = loop {
            match listener.accept() {
                Ok(connection) => break connection,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() >= deadline {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "test HTTP server did not receive a request",
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(error) => return Err(error),
            }
        };
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
        let mut request = Vec::new();
        let mut byte = [0_u8; 1];
        while !request.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte)?;
            request.push(byte[0]);
        }
        let headers = String::from_utf8_lossy(&request);
        let length = headers
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then_some(value.trim())
                })
            })
            .and_then(|value| value.parse::<usize>().ok());
        let Some(length) = length else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "test HTTP request did not include Content-Length",
            ));
        };
        let mut body = vec![0_u8; length];
        stream.read_exact(&mut body)?;
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Length: 0\r\nETag: \"test-etag\"\r\nConnection: close\r\n\r\n"
        )?;
        stream.flush()
    });
    (url, server)
}

fn wait_for_captures(captures: &Arc<Mutex<Vec<HttpCapture>>>, expected: usize) -> Vec<HttpCapture> {
    for _ in 0..100 {
        let snapshot = captures.lock().unwrap().clone();
        if snapshot.len() >= expected {
            return snapshot;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    captures.lock().unwrap().clone()
}

fn reset_runtime() {
    log::set_max_level(log::LevelFilter::Info);
    let _ = test_close_plugin_host();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn start_otlp_capture_server() -> (String, mpsc::Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut byte = [0_u8; 1];
        while !request.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
        }
        let headers = String::from_utf8_lossy(&request);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then_some(value.trim())
                })
            })
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let mut body = vec![0_u8; content_length];
        stream.read_exact(&mut body).unwrap();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
        sender.send(body).unwrap();
    });
    (endpoint, receiver)
}

fn component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
        enabled: true,
        config,
    }
}

fn plugin_config(config: Json) -> PluginConfig {
    PluginConfig {
        version: 1,
        components: vec![component(config)],
        policy: Default::default(),
    }
}

fn push_agent(name: &str) -> nemo_relay_runtime::api::scope::ScopeHandle {
    nemo_relay_runtime::api::scope::push_scope(
        PushScopeParams::builder()
            .name(name)
            .scope_type(ScopeType::Agent)
            .input(json!({"agent": name}))
            .build(),
    )
    .unwrap()
}

fn push_function(name: &str) -> nemo_relay_runtime::api::scope::ScopeHandle {
    nemo_relay_runtime::api::scope::push_scope(
        PushScopeParams::builder()
            .name(name)
            .scope_type(ScopeType::Function)
            .input(json!({"function": name}))
            .build(),
    )
    .unwrap()
}

fn pop(handle: &nemo_relay_runtime::api::scope::ScopeHandle) {
    nemo_relay_runtime::api::scope::pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .output(json!({"done": handle.name}))
            .build(),
    )
    .unwrap();
}

fn schema_has_property(schema: &Json, name: &str) -> bool {
    schema_property(schema, name).is_some()
}

fn schema_property<'a>(schema: &'a Json, name: &str) -> Option<&'a Json> {
    match schema {
        Json::Object(object) => {
            if let Some(property) = object
                .get("properties")
                .and_then(Json::as_object)
                .and_then(|properties| properties.get(name))
            {
                return Some(property);
            }
            object
                .values()
                .find_map(|value| schema_property(value, name))
        }
        Json::Array(values) => values.iter().find_map(|value| schema_property(value, name)),
        _ => None,
    }
}

#[test]
fn default_config_and_component_conversion_cover_public_shape() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let defaults = ObservabilityConfig::default();
    assert_eq!(defaults.version, 4);
    assert!(!defaults.enable_full_payloads);
    assert!(defaults.atof.is_none());
    assert!(defaults.atif.is_none());
    assert!(defaults.opentelemetry.is_none());

    let atof = AtofSectionConfig::default();
    assert!(!atof.enabled);
    assert!(atof.sinks.is_empty());

    assert_default_stream_sink_shape();

    let atif = AtifSectionConfig::default();
    assert!(!atif.enabled);
    assert_eq!(atif.agent_name, "NeMo Relay");
    assert_eq!(atif.agent_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(atif.model_name, "unknown");
    assert_eq!(atif.filename_template, "nemo-relay-atif-{session_id}.json");

    let otel = OpenTelemetrySectionConfig {
        enabled: true,
        file_sinks: Vec::new(),
        endpoints: vec![OpenTelemetryEndpointConfig {
            tls: None,
            otel_type: OpenTelemetryType::Full,
            endpoint: "http://localhost:4318/v1/traces".to_string(),
            transport: default_otlp_transport(),
            service_name: default_otel_service_name(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: default_otel_instrumentation_scope(),
            timeout_millis: default_timeout_millis(),
            max_queue_size: None,
            max_export_batch_size: None,
            scheduled_delay_millis: None,
            completed_span_context_ttl_millis: None,
            headers: HashMap::new(),
            header_env: HashMap::new(),
            header_file: HashMap::new(),
            resource_attributes: HashMap::new(),
            mark_projection: MarkProjection::default(),
            mark_exclude_names: default_mark_exclude_names(),
            attribute_mappings: Vec::new(),
            promote_metadata_prefixes: Vec::new(),
            promote_resource_metadata_prefixes: Vec::new(),
        }],
        logs: None,
        metrics: None,
    };

    let generic: PluginComponentSpec = ComponentSpec::new(ObservabilityConfig {
        atof: Some(atof),
        atif: Some(atif),
        opentelemetry: Some(otel),
        ..ObservabilityConfig::default()
    })
    .into();
    assert_eq!(generic.kind, OBSERVABILITY_PLUGIN_KIND);
    assert!(generic.enabled);
    assert_eq!(generic.config["version"], json!(4));
    assert_eq!(generic.config["atif"]["agent_name"], json!("NeMo Relay"));
    assert_endpoint_batch_fields_omitted(&generic.config["opentelemetry"]["traces"][0]);
    let legacy: OpenTelemetrySectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "endpoints": generic.config["opentelemetry"]["traces"].clone(),
    }))
    .expect("legacy endpoints alias should deserialize");
    assert_eq!(
        serde_json::to_value(legacy).unwrap()["traces"],
        generic.config["opentelemetry"]["traces"]
    );

    assert_endpoint_batch_fields_deserialize();
}

fn assert_endpoint_batch_fields_omitted(serialized_endpoint: &Json) {
    for field in [
        "max_queue_size",
        "max_export_batch_size",
        "scheduled_delay_millis",
    ] {
        assert!(serialized_endpoint.get(field).is_none());
    }
}

fn assert_endpoint_batch_fields_deserialize() {
    let endpoint: OpenTelemetryEndpointConfig = serde_json::from_value(json!({
        "type": "full",
        "endpoint": "http://localhost:4318/v1/traces",
        "max_queue_size": 4096,
        "max_export_batch_size": 256,
        "scheduled_delay_millis": 750,
    }))
    .unwrap();
    assert_eq!(endpoint.max_queue_size, Some(4096));
    assert_eq!(endpoint.max_export_batch_size, Some(256));
    assert_eq!(endpoint.scheduled_delay_millis, Some(750));
}

#[test]
fn full_payload_policy_activates_and_clears_with_the_plugin() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let config = plugin_config(json!({"enable_full_payloads": true}));
    assert!(!test_validate_static_plugin_config(&config).has_errors());
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .observability_full_payloads_enabled
    );

    test_close_plugin_host().unwrap();
    assert!(
        !global_context()
            .read()
            .unwrap()
            .observability_full_payloads_enabled
    );
}

fn assert_default_stream_sink_shape() {
    let parsed_atof: AtofSectionConfig = serde_json::from_value(json!({
        "sinks": [{"type": "stream", "name": "switchyard", "url": "http://localhost/events"}]
    }))
    .unwrap();
    let AtofSinkSectionConfig::Stream(stream) = &parsed_atof.sinks[0] else {
        panic!("expected stream sink");
    };
    assert_eq!(stream.name.as_deref(), Some("switchyard"));
    assert_eq!(stream.transport, "http_post");
    assert_eq!(stream.field_name_policy, "preserve");
}

fn plaintext_preflight_environment() -> EnvironmentGuard {
    let mut variables: Vec<String> = std::env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| name.starts_with("OTEL_"))
        .collect();
    variables.extend(
        [
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_HEADERS",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            "OTEL_TRACES_EXPORTER",
            "OTEL_LOGS_EXPORTER",
            "OTEL_METRICS_EXPORTER",
            "OTEL_SDK_DISABLED",
            "NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG",
        ]
        .into_iter()
        .map(str::to_string),
    );
    let environment = EnvironmentGuard::capture(&variables);
    // SAFETY: callers hold the observability mutex until this guard restores the environment.
    unsafe {
        for variable in variables {
            std::env::remove_var(variable);
        }
        std::env::set_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "1");
    }
    environment
}

fn plaintext_preflight_configuration(
    signal: &str,
    destination: &str,
    headers: Json,
) -> PluginConfig {
    let mut endpoint = json!({"endpoint": destination});
    endpoint
        .as_object_mut()
        .unwrap()
        .extend(headers.as_object().unwrap().clone());
    let section = match signal {
        "traces" => {
            endpoint["type"] = json!("full");
            json!({"opentelemetry": {"enabled": true, "endpoints": [endpoint]}})
        }
        "logs" | "metrics" => json!({
            "opentelemetry": {"enabled": true, signal: {"enabled": true, "endpoints": [endpoint]}}
        }),
        "atof" => {
            endpoint.as_object_mut().unwrap().remove("endpoint");
            endpoint["url"] = json!(destination);
            endpoint["type"] = json!("stream");
            json!({"atof": {"enabled": true, "sinks": [endpoint]}})
        }
        "atif" => {
            endpoint["type"] = json!("http");
            json!({"atif": {"enabled": true, "storage": [endpoint]}})
        }
        _ => panic!("unexpected test signal {signal}"),
    };
    let mut configuration = section;
    configuration["version"] = json!(4);
    plugin_config(configuration)
}

fn plaintext_preflight_diagnostics(configuration: &PluginConfig) -> Vec<ConfigDiagnostic> {
    nemo_relay_plugin_host::validate_exact(configuration.clone())
        .config
        .diagnostics
        .into_iter()
        .filter(|diagnostic| diagnostic.code == "observability.unsafe_plaintext_destination")
        .collect()
}

#[test]
fn plaintext_preflight_rejects_all_five_destinations_and_header_sources() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _environment = plaintext_preflight_environment();
    for signal in ["traces", "logs", "metrics", "atof", "atif"] {
        for headers in [
            json!({"headers": {"authorization": "synthetic-inline-secret"}}),
            json!({"header_env": {"authorization": "UNREAD_SYNTHETIC_TOKEN"}}),
            json!({"header_file": {"authorization": "/synthetic/unread-token"}}),
            json!({}),
        ] {
            let has_headers = headers != json!({});
            let configuration = plaintext_preflight_configuration(
                signal,
                "http://collector.example:4318/private-path?token=synthetic-query-secret",
                headers,
            );
            let diagnostics = plaintext_preflight_diagnostics(&configuration);
            if signal != "traces" && !has_headers {
                assert!(diagnostics.is_empty(), "{signal}: {diagnostics:?}");
                continue;
            }
            assert_eq!(diagnostics.len(), 1, "{signal}: {diagnostics:?}");
            let diagnostic = &diagnostics[0];
            assert_eq!(diagnostic.level, DiagnosticLevel::Error);
            let component = match signal {
                "atof" | "atif" => signal.to_string(),
                _ => format!("opentelemetry.{signal}"),
            };
            let field = match signal {
                "atof" => "sinks[0].url",
                "atif" => "storage[0].endpoint",
                _ => "endpoints[0].endpoint",
            };
            assert_eq!(diagnostic.component.as_deref(), Some(component.as_str()));
            assert_eq!(diagnostic.field.as_deref(), Some(field));
            assert!(diagnostic.message.contains("http://collector.example:4318"));
            assert!(!diagnostic.message.contains("synthetic-query-secret"));
            assert!(!diagnostic.message.contains("private-path"));
            assert!(!diagnostic.message.contains("synthetic-inline-secret"));
            #[cfg(feature = "__skip-implicit-config")]
            {
                let merged = nemo_relay_plugin_host::validate(configuration, None).unwrap();
                assert!(
                    merged.config.diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == "observability.unsafe_plaintext_destination"
                            && diagnostic.component.as_deref() == Some(component.as_str())
                            && diagnostic.field.as_deref() == Some(field)
                            && diagnostic.level == DiagnosticLevel::Error
                    }),
                    "{signal}: {:?}",
                    merged.config.diagnostics
                );
            }
        }
    }
}

#[test]
fn plaintext_preflight_preserves_https_loopback_and_explicit_host_permissions() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _environment = plaintext_preflight_environment();
    for signal in ["traces", "logs", "metrics", "atof", "atif"] {
        for destination in [
            "https://collector.example:4318/ingest",
            "http://localhost:4318/ingest",
            "http://127.0.0.1:4318/ingest",
            "http://[::1]:4318/ingest",
            "http://collector.example:4318/ingest",
        ] {
            let mut configuration = plaintext_preflight_configuration(
                signal,
                destination,
                json!({"headers": {"authorization": "synthetic-token"}}),
            );
            if destination == "http://collector.example:4318/ingest" {
                configuration.components[0]
                    .config
                    .insert("insecure_http_hosts".into(), json!(["COLLECTOR.EXAMPLE"]));
            }
            let report = nemo_relay_plugin_host::validate_exact(configuration);
            assert!(
                !report.config.has_errors(),
                "{signal} {destination}: {:?}",
                report.config.diagnostics
            );
        }
    }
}

#[test]
fn plaintext_preflight_identifies_later_endpoints_and_redacts_credentials() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _environment = plaintext_preflight_environment();
    for signal in ["traces", "logs", "metrics", "atof", "atif"] {
        let mut configuration = plaintext_preflight_configuration(
            signal,
            "http://synthetic-user:synthetic-password@collector.example:4318/private-token?key=query-secret",
            json!({"headers": {"authorization": "synthetic-header-secret"}}),
        );
        let section = &mut configuration.components[0].config;
        let endpoints = match signal {
            "traces" => section["opentelemetry"]["endpoints"]
                .as_array_mut()
                .unwrap(),
            "logs" | "metrics" => section["opentelemetry"][signal]["endpoints"]
                .as_array_mut()
                .unwrap(),
            "atof" => section["atof"]["sinks"].as_array_mut().unwrap(),
            "atif" => section["atif"]["storage"].as_array_mut().unwrap(),
            _ => unreachable!(),
        };
        let mut valid = endpoints[0].clone();
        valid[if signal == "atof" { "url" } else { "endpoint" }] =
            json!("https://safe.example:4318/ingest");
        endpoints.insert(0, valid);
        let diagnostics = plaintext_preflight_diagnostics(&configuration);
        assert_eq!(diagnostics.len(), 1, "{signal}: {diagnostics:?}");
        let diagnostic = &diagnostics[0];
        assert!(diagnostic.field.as_deref().unwrap().contains("[1]"));
        let rendered = serde_json::to_string(diagnostic).unwrap();
        for secret in [
            "synthetic-user",
            "synthetic-password",
            "private-token",
            "query-secret",
            "synthetic-header-secret",
        ] {
            assert!(!rendered.contains(secret), "{signal}: {rendered}");
        }
    }
}

#[test]
fn plaintext_preflight_checks_inherited_log_and_metric_destinations() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _environment = plaintext_preflight_environment();
    let configuration = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{"type": "full", "endpoint": "http://collector.example:4318/v1/traces", "headers": {"authorization": "synthetic-token"}}],
            "logs": {"enabled": true},
            "metrics": {"enabled": true}
        }
    }));
    let diagnostics = plaintext_preflight_diagnostics(&configuration);
    assert_eq!(diagnostics.len(), 3, "{diagnostics:?}");
    for signal in ["traces", "logs", "metrics"] {
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.component.as_deref() == Some(format!("opentelemetry.{signal}").as_str())
                && diagnostic.field.as_deref() == Some("endpoints[0].endpoint")
        }));
    }
}

#[test]
fn plaintext_preflight_checks_automatic_destinations_and_honors_exporter_disabling() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _environment = plaintext_preflight_environment();
    let configuration = plugin_config(json!({"version": 4}));
    for signal in ["TRACES", "LOGS", "METRICS"] {
        let endpoint_variable = format!("OTEL_EXPORTER_OTLP_{signal}_ENDPOINT");
        let exporter_variable = format!("OTEL_{signal}_EXPORTER");
        // SAFETY: the observability mutex serializes test-only environment changes.
        unsafe {
            std::env::set_var(
                &endpoint_variable,
                "http://collector.example:4318/private?token=synthetic-secret",
            );
            std::env::set_var(
                "OTEL_EXPORTER_OTLP_HEADERS",
                "authorization=synthetic-header-secret",
            );
        }
        let diagnostics = plaintext_preflight_diagnostics(&configuration);
        assert_eq!(diagnostics.len(), 1, "{signal}: {diagnostics:?}");
        assert_eq!(
            diagnostics[0].component.as_deref(),
            Some(format!("opentelemetry.automatic.{}", signal.to_lowercase()).as_str())
        );
        assert_eq!(
            diagnostics[0].field.as_deref(),
            Some("endpoints[0].endpoint")
        );
        assert!(!diagnostics[0].message.contains("synthetic-secret"));
        for setting in ["none", "console"] {
            // SAFETY: the observability mutex serializes test-only environment changes.
            unsafe {
                std::env::set_var(&exporter_variable, setting);
            }
            assert!(plaintext_preflight_diagnostics(&configuration).is_empty());
        }
        // SAFETY: the observability mutex serializes test-only environment changes.
        unsafe {
            std::env::remove_var(&exporter_variable);
            std::env::set_var("OTEL_SDK_DISABLED", "true");
        }
        assert!(plaintext_preflight_diagnostics(&configuration).is_empty());
        // SAFETY: the observability mutex serializes test-only environment changes.
        unsafe {
            std::env::remove_var("OTEL_SDK_DISABLED");
            std::env::remove_var(&endpoint_variable);
        }
    }
    // A generic endpoint applies the same transport policy to all three signals without headers.
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://collector.example:4318",
        );
        std::env::remove_var("OTEL_EXPORTER_OTLP_HEADERS");
    }
    let diagnostics = plaintext_preflight_diagnostics(&configuration);
    assert_eq!(diagnostics.len(), 3, "{diagnostics:?}");
    for signal in ["traces", "logs", "metrics"] {
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.component.as_deref()
                == Some(format!("opentelemetry.automatic.{signal}").as_str())
        }));
    }
}

#[test]
fn plaintext_preflight_ignores_disabled_components_and_sections() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _environment = plaintext_preflight_environment();
    for signal in ["traces", "logs", "metrics", "atof", "atif"] {
        let mut configuration = plaintext_preflight_configuration(
            signal,
            "http://collector.example:4318",
            json!({"headers": {"authorization": "synthetic-token"}}),
        );
        configuration.components[0].enabled = false;
        assert!(plaintext_preflight_diagnostics(&configuration).is_empty());
        configuration.components[0].enabled = true;
        let section = &mut configuration.components[0].config;
        let section_name = if ["traces", "logs", "metrics"].contains(&signal) {
            "opentelemetry"
        } else {
            signal
        };
        section[section_name]["enabled"] = json!(false);
        assert!(plaintext_preflight_diagnostics(&configuration).is_empty());
        if ["logs", "metrics"].contains(&signal) {
            configuration.components[0].config["opentelemetry"]["enabled"] = json!(true);
            configuration.components[0].config["opentelemetry"][signal]["enabled"] = json!(false);
            assert!(plaintext_preflight_diagnostics(&configuration).is_empty());
        }
    }
}

#[test]
fn plaintext_preflight_errors_do_not_block_partial_opentelemetry_activation() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _environment = plaintext_preflight_environment();
    for signal in ["traces", "logs", "metrics"] {
        reset_runtime();
        let mut configuration = plaintext_preflight_configuration(
            signal,
            "http://collector.example:4318/ingest",
            json!({"headers": {"authorization": "synthetic-token"}}),
        );
        let section = &mut configuration.components[0].config;
        let endpoints = if signal == "traces" {
            section["opentelemetry"]["endpoints"]
                .as_array_mut()
                .unwrap()
        } else {
            section["opentelemetry"][signal]["endpoints"]
                .as_array_mut()
                .unwrap()
        };
        let mut valid = endpoints[0].clone();
        valid["endpoint"] = json!("https://safe.example:4318/ingest");
        endpoints.push(valid);
        assert!(
            nemo_relay_plugin_host::validate_exact(configuration.clone())
                .config
                .has_errors()
        );
        futures::executor::block_on(test_initialize_plugin_host_exact(configuration)).unwrap();
        assert_eq!(global_context().read().unwrap().event_subscribers.len(), 1);
        test_close_plugin_host().unwrap();
    }
}

#[test]
fn plaintext_preflight_uses_websocket_policy_for_atof_streams() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let _environment = plaintext_preflight_environment();
    for (destination, rejected) in [
        ("ws://collector.example:8080/events", true),
        ("wss://collector.example:8080/events", false),
        ("ws://localhost:8080/events", false),
    ] {
        let mut configuration = plaintext_preflight_configuration(
            "atof",
            destination,
            json!({"headers": {"authorization": "synthetic-token"}}),
        );
        configuration.components[0].config["atof"]["sinks"][0]["transport"] = json!("websocket");
        assert_eq!(
            !plaintext_preflight_diagnostics(&configuration).is_empty(),
            rejected,
            "{destination}"
        );
    }
}

fn test_opentelemetry_endpoint() -> OpenTelemetryEndpointConfig {
    OpenTelemetryEndpointConfig {
        tls: None,
        otel_type: OpenTelemetryType::Full,
        endpoint: "http://localhost:4318/v1/traces".to_string(),
        transport: default_otlp_transport(),
        service_name: default_otel_service_name(),
        service_namespace: None,
        service_version: None,
        instrumentation_scope: default_otel_instrumentation_scope(),
        timeout_millis: default_timeout_millis(),
        max_queue_size: None,
        max_export_batch_size: None,
        scheduled_delay_millis: None,
        completed_span_context_ttl_millis: None,
        headers: HashMap::new(),
        header_env: HashMap::new(),
        header_file: HashMap::new(),
        resource_attributes: HashMap::new(),
        mark_projection: MarkProjection::default(),
        mark_exclude_names: default_mark_exclude_names(),
        attribute_mappings: Vec::new(),
        promote_metadata_prefixes: Vec::new(),
        promote_resource_metadata_prefixes: Vec::new(),
    }
}

fn test_signal_endpoint() -> OpenTelemetrySignalEndpointConfig {
    OpenTelemetrySignalEndpointConfig {
        tls: None,
        endpoint: "http://localhost:4318/v1/logs".to_string(),
        transport: default_otlp_transport(),
        headers: HashMap::new(),
        header_env: HashMap::new(),
        header_file: HashMap::new(),
        resource_attributes: HashMap::new(),
        promote_resource_metadata_prefixes: Vec::new(),
        service_name: default_otel_service_name(),
        service_namespace: None,
        service_version: None,
        instrumentation_scope: default_otel_instrumentation_scope(),
        timeout_millis: default_timeout_millis(),
    }
}

#[test]
fn invalid_batch_config_identifies_the_endpoint_during_activation() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "https://jaeger-local:4318/v1/traces",
                    "max_queue_size": 256
                },
                {
                    "type": "full",
                    "endpoint": "https://tempo-prod:4319/v1/traces",
                    "max_queue_size": 0
                },
                {
                    "type": "full",
                    "endpoint": "https://compliance:4320/v1/traces",
                    "max_queue_size": 128
                }
            ]
        }
    }));

    let report = test_validate_static_plugin_config(&config);
    assert!(!report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[1].max_queue_size")
            && diagnostic.level == DiagnosticLevel::Warning
            && diagnostic.code == "observability.invalid_otel_endpoint"
            && diagnostic
                .message
                .contains("OpenTelemetry endpoints[1].max_queue_size must be greater than 0")
    }));

    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    let report = crate::test_support::plugin_host::test_plugin_host_report().unwrap();
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[1].max_queue_size")
            && diagnostic.level == DiagnosticLevel::Warning
            && diagnostic.code == "observability.invalid_otel_endpoint"
    }));
    test_close_plugin_host().unwrap();
}

#[test]
fn all_invalid_trace_batch_configs_still_block_activation() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "https://jaeger-local:4318/v1/traces",
                "max_queue_size": 0
            }]
        }
    }));

    assert!(!test_validate_static_plugin_config(&config).has_errors());
    let error = futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires at least one valid trace, log, or metric endpoint")
    );
    assert!(crate::test_support::plugin_host::test_plugin_host_report().is_none());
}

#[test]
fn plain_http_trace_endpoint_activates_only_when_host_is_listed() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let config = |insecure_http_hosts: Json| {
        plugin_config(json!({
            "version": 4,
            "insecure_http_hosts": insecure_http_hosts,
            "opentelemetry": {
                "enabled": true,
                "traces": [{
                    "type": "gen_ai",
                    "endpoint": "http://host.docker.internal:8082/v1/traces",
                    "headers": {"authorization": "Bearer token"}
                }]
            }
        }))
    };

    let listed = config(json!(["HOST.DOCKER.INTERNAL"]));
    assert!(!test_validate_static_plugin_config(&listed).has_errors());
    futures::executor::block_on(test_initialize_plugin_host_exact(listed)).unwrap();
    test_close_plugin_host().unwrap();

    reset_runtime();
    let error = futures::executor::block_on(test_initialize_plugin_host_exact(config(json!([]))))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("requires at least one valid trace, log, or metric endpoint"),
        "{error}"
    );
    assert!(
        error.contains("traces[0] (http://host.docker.internal:8082)"),
        "{error}"
    );
    assert!(
        error.contains("OTLP trace endpoints require HTTPS"),
        "{error}"
    );
    assert!(!error.contains("Bearer token"), "{error}");

    reset_runtime();
    let malformed = config(json!(["http://host.docker.internal"]));
    let report = test_validate_static_plugin_config(&malformed);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "observability.invalid_insecure_http_hosts"),
        "{:?}",
        report.diagnostics
    );
    let error = futures::executor::block_on(test_initialize_plugin_host_exact(malformed))
        .unwrap_err()
        .to_string();
    assert!(error.contains("bare hostname or IP address"), "{error}");
}

#[test]
fn skipped_endpoint_reasons_never_echo_endpoint_userinfo() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "traces": [{
                "type": "gen_ai",
                "endpoint": "ftp://user:secret-token@collector.example/x"
            }],
            "logs": {"enabled": true}
        }
    }));
    let error = futures::executor::block_on(test_initialize_plugin_host_exact(config))
        .unwrap_err()
        .to_string();
    assert!(error.contains("traces[0]"), "{error}");
    assert!(error.contains("logs[0]"), "{error}");
    assert!(!error.contains("secret-token"), "{error}");

    for (endpoint, secret, logs) in [
        (
            "https://user:p^ss@collector.example/v1/traces",
            "p^ss",
            false,
        ),
        (
            "https://collector.example/v1/traces?api_key=s3cret\n",
            "s3cret",
            false,
        ),
        (
            "http://collector.internal:4318/t/SECRET-KEY/ingest",
            "SECRET-KEY",
            true,
        ),
    ] {
        reset_runtime();
        let config = plugin_config(json!({
            "version": 4,
            "opentelemetry": {
                "enabled": true,
                "traces": [{"type": "gen_ai", "endpoint": endpoint}],
                "logs": {"enabled": logs}
            }
        }));
        let error = futures::executor::block_on(test_initialize_plugin_host_exact(config))
            .unwrap_err()
            .to_string();
        assert!(error.contains("traces[0]"), "{error}");
        assert!(!error.contains(secret), "{error}");
    }
}

#[test]
fn file_only_sink_directory_creation_failure_reports_the_file_error() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let directory = tempfile::tempdir().unwrap();
    let not_a_directory = directory.path().join("not-a-directory");
    fs::write(&not_a_directory, b"file").unwrap();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "file_sinks": [{
                "output_directory": not_a_directory,
                "filename": "trace.jsonl"
            }]
        }
    }));

    let error = futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("file_sinks[0]"), "{message}");
    assert!(message.contains("not-a-directory"), "{message}");
    assert!(
        message.contains("failed to create OTLP output directory"),
        "{message}"
    );
    assert!(
        !message.contains("requires at least one valid trace, log, or metric endpoint"),
        "{message}"
    );
}

#[test]
fn mixed_file_sink_directory_failure_is_reported_in_activation_diagnostics() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let directory = tempfile::tempdir().unwrap();
    let not_a_directory = directory.path().join("not-a-directory");
    fs::write(&not_a_directory, b"file").unwrap();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://127.0.0.1:4318/v1/traces"
            }],
            "file_sinks": [{
                "output_directory": not_a_directory,
                "filename": "trace.jsonl"
            }]
        }
    }));

    let report = futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.invalid_otel_file_sink"
            && diagnostic.field.as_deref() == Some("file_sinks[0]")
            && diagnostic.message.contains("not-a-directory")
    }));
    test_close_plugin_host().unwrap();
}

#[test]
fn opentelemetry_registration_rejects_an_empty_endpoint_list() {
    let mut context = PluginRegistrationContext::new();
    let error = register_opentelemetry(
        OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: Vec::new(),
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
        &PlaintextPolicy::default(),
        &mut context,
        false,
        &crate::observability_runtime::runtime_host(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("at least one endpoint"));
}

#[test]
fn opentelemetry_endpoint_header_env_rejects_missing_and_duplicate_headers() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let variable = "NEMO_RELAY_TEST_MISSING_OTEL_HEADER_ENV";
    unsafe { std::env::remove_var(variable) };
    let report = test_validate_static_plugin_config(&plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://localhost:4318/v1/traces",
                "headers": {"Authorization": "inline"},
                "header_env": {
                    "authorization": variable,
                    "x-api-key": variable
                }
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[0].header_env.authorization")
            && diagnostic.message.contains("both headers and header_env")
    }));
    let activation =
        futures::executor::block_on(test_initialize_plugin_host_exact(plugin_config(json!({
            "opentelemetry": {
                "enabled": true,
                "endpoints": [{
                    "type": "full",
                    "endpoint": "http://localhost:4318/v1/traces",
                    "header_env": {"x-api-key": variable}
                }]
            }
        }))));
    assert!(activation.is_err());

    futures::executor::block_on(test_initialize_plugin_host_exact(plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "http://localhost:4318/v1/traces",
                    "header_env": {"x-api-key": variable}
                },
                {
                    "type": "gen_ai",
                    "endpoint": "http://localhost:4319/v1/traces"
                }
            ]
        }
    }))))
    .unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn invalid_log_endpoint_keeps_valid_signal_subscriber() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let variable = "NEMO_RELAY_TEST_MISSING_OTEL_LOG_HEADER_ENV";
    unsafe { std::env::remove_var(variable) };
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "logs": {
                "enabled": true,
                "endpoints": [
                    {
                        "endpoint": "http://localhost:4318/v1/logs",
                        "header_env": {"authorization": variable}
                    },
                    {"endpoint": "http://localhost:4319/v1/logs"}
                ]
            }
        }
    }));

    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn invalid_metric_endpoint_keeps_valid_signal_subscriber() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let variable = "NEMO_RELAY_TEST_MISSING_OTEL_METRIC_HEADER_ENV";
    unsafe { std::env::remove_var(variable) };
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "metrics": {
                "enabled": true,
                "endpoints": [
                    {
                        "endpoint": "http://localhost:4318/v1/metrics",
                        "header_env": {"authorization": variable}
                    },
                    {"endpoint": "http://localhost:4319/v1/metrics"}
                ]
            }
        }
    }));

    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn malformed_derived_signal_endpoint_keeps_valid_peers() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "http://127.0.0.1:4318/first/v1/traces"
                },
                {
                    "type": "gen_ai",
                    "endpoint": "http://127.0.0.1:4319/custom/traces"
                },
                {
                    "type": "openinference",
                    "endpoint": "http://127.0.0.1:4320/third/v1/traces"
                }
            ],
            "logs": {"enabled": true},
            "metrics": {"enabled": true}
        }
    }));

    let report = test_validate_static_plugin_config(&config);
    assert!(!report.has_errors());
    for signal in ["logs", "metrics"] {
        let component = format!("opentelemetry.{signal}");
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.component.as_deref() == Some(component.as_str())
                && diagnostic.field.as_deref() == Some("endpoints[1]")
                && diagnostic.level == DiagnosticLevel::Warning
                && diagnostic.code == "observability.invalid_otel_endpoint"
        }));
    }

    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn skipped_trace_endpoint_preserves_configured_indexes() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let first = test_opentelemetry_endpoint();
    let mut invalid = test_opentelemetry_endpoint();
    invalid.max_queue_size = Some(0);
    let third = test_opentelemetry_endpoint();

    let subscribers =
        build_opentelemetry_subscribers(vec![first, invalid, third], &PlaintextPolicy::default())
            .unwrap();
    assert_eq!(
        subscribers
            .iter()
            .map(|subscriber| subscriber.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        subscribers
            .iter()
            .map(|subscriber| subscriber.value.as_active().is_some())
            .collect::<Vec<_>>(),
        vec![true, false, true]
    );
    assert!(shutdown_indexed_opentelemetry_providers(&subscribers).is_empty());
}

#[test]
fn all_invalid_log_and_metric_endpoints_still_block_activation() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let log_variable = "NEMO_RELAY_TEST_MISSING_OTEL_LOG_HEADER_ENV";
    let metric_variable = "NEMO_RELAY_TEST_MISSING_OTEL_METRIC_HEADER_ENV";
    unsafe {
        std::env::remove_var(log_variable);
        std::env::remove_var(metric_variable);
    }
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "logs": {
                "enabled": true,
                "endpoints": [{
                    "endpoint": "http://localhost:4318/v1/logs",
                    "header_env": {"authorization": log_variable}
                }]
            },
            "metrics": {
                "enabled": true,
                "endpoints": [{
                    "endpoint": "http://localhost:4318/v1/metrics",
                    "header_env": {"authorization": metric_variable}
                }]
            }
        }
    }));

    let error = futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires at least one valid trace, log, or metric endpoint")
    );
    assert!(crate::test_support::plugin_host::test_plugin_host_report().is_none());
}

#[test]
fn outer_disabled_component_does_not_resolve_opentelemetry_header_env() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let variable = "NEMO_RELAY_TEST_DISABLED_OTEL_HEADER_ENV";
    unsafe { std::env::remove_var(variable) };
    let mut config = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://localhost:4318/v1/traces",
                "header_env": {"authorization": variable}
            }]
        }
    }));
    config.components[0].enabled = false;
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    test_close_plugin_host().unwrap();
}

#[test]
fn built_in_registration_is_automatic() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    assert!(list_plugin_kinds().contains(&OBSERVABILITY_PLUGIN_KIND.to_string()));
    assert!(lookup_plugin(OBSERVABILITY_PLUGIN_KIND).is_some());

    let config = plugin_config(json!({}));
    assert!(!test_validate_static_plugin_config(&config).has_errors());
}

#[test]
fn explicit_registration_helpers_are_idempotent_and_reversible() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    assert!(register_observability_component().is_ok());
    assert!(register_observability_component().is_ok());
    assert!(deregister_observability_component());
    assert!(!deregister_observability_component());
    register_observability_component().unwrap();
}

#[test]
fn empty_and_disabled_config_register_nothing() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let config = plugin_config(json!({
        "atof": {"enabled": false},
        "atif": {"enabled": false},
        "opentelemetry": {"enabled": false, "traces": []}
    }));
    assert!(!test_validate_static_plugin_config(&config).has_errors());
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let state = global_context();
    assert!(state.read().unwrap().event_subscribers.is_empty());
    crate::test_support::plugin_host::test_close_plugin_host().unwrap();
}

#[test]
fn disabled_file_sections_do_not_create_files() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-disabled-files");

    let config = plugin_config(json!({
        "atof": {
            "enabled": false,
            "sinks": [{"type": "file", "output_directory": dir, "filename": "events.jsonl"}]
        },
        "atif": {
            "enabled": false,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }));
    assert!(!test_validate_static_plugin_config(&config).has_errors());
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let agent = push_agent("disabled-agent");
    pop(&agent);
    test_close_plugin_host().unwrap();

    assert!(!dir.join("events.jsonl").exists());
    assert!(!dir.join(format!("trajectory-{}.json", agent.uuid)).exists());
}

#[test]
fn duplicate_component_is_rejected_as_singleton() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let config = PluginConfig {
        version: 1,
        components: vec![component(json!({})), component(json!({}))],
        policy: Default::default(),
    };
    let report = test_validate_static_plugin_config(&config);
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.duplicate_component")
    );
}

#[test]
fn a_file_sinks_section_is_a_recognized_opentelemetry_field() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    reset_runtime();

    // The JSON-level field check is separate from the typed validation, so a
    // new section has to be added to its allow-list or every config carrying
    // one is rejected outright.
    let report = test_validate_static_plugin_config(&plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "file_sinks": [{
                "output_directory": "/var/log/nemo-relay",
                "filename": "trace.jsonl",
                "bogus": true,
            }],
        }
    })));

    assert!(
        !report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("file_sinks")),
        "file_sinks should be recognized: {:?}",
        report.diagnostics
    );
    // Its own keys are still checked, as an endpoint's are.
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("file_sinks[0].bogus")),
        "expected the unknown sink key to be reported: {:?}",
        report.diagnostics
    );
}

#[test]
fn unknown_fields_and_bad_values_follow_policy() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let warn_report = test_validate_static_plugin_config(&plugin_config(json!({
        "atof": {"bogus": true, "sinks": [{"type": "file", "mode": "invalid"}]},
        "atif": {"filename_template": "missing-session"}
    })));
    assert!(warn_report.has_errors());
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.unknown_field")
    );
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("sinks[0].mode"))
    );
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("filename_template"))
    );

    let ignore_report = test_validate_static_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "ignore", "unsupported_value": "ignore"},
        "atof": {"bogus": true, "sinks": [{"type": "file", "mode": "invalid"}]},
        "atif": {"filename_template": "missing-session"}
    })));
    assert!(!ignore_report.has_errors());
    assert!(ignore_report.diagnostics.is_empty());
}

#[test]
fn atif_filename_template_syntax_is_rejected_before_activation() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let valid_report = test_validate_static_plugin_config(&plugin_config(json!({
        "atif": {
            "filename_template": "{metadata.workflow_id:-unassigned}/trajectory-{session_id}.json"
        }
    })));
    assert!(!valid_report.has_errors());

    let malformed = "trajectory-{session_id}.json/{metadata.tenant";
    let invalid_report = test_validate_static_plugin_config(&plugin_config(json!({
        "atif": {"filename_template": malformed}
    })));
    assert!(invalid_report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("filename_template")
            && diag.message.contains("unclosed metadata placeholder")
    }));

    let error =
        futures::executor::block_on(test_initialize_plugin_host_exact(plugin_config(json!({
            "policy": {"unsupported_value": "ignore"},
            "atif": {"enabled": true, "filename_template": malformed}
        }))))
        .unwrap_err();
    assert!(error.to_string().contains("unclosed metadata placeholder"));
}

#[test]
fn invalid_shapes_and_strict_policy_are_reported() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let invalid_shape = test_validate_static_plugin_config(&plugin_config(json!({
        "version": "one",
    })));
    assert!(invalid_shape.has_errors());
    assert!(
        invalid_shape
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.invalid_plugin_config")
    );

    let unsupported_version = test_validate_static_plugin_config(&plugin_config(json!({
        "version": 1,
    })));
    assert!(unsupported_version.has_errors());
    assert!(unsupported_version.diagnostics.iter().any(|diag| diag.code
        == "observability.unsupported_config_version"
        && diag.field.as_deref() == Some("version")));

    let strict_unknown = test_validate_static_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "error"},
        "opentelemetry": {"unexpected": true}
    })));
    assert!(strict_unknown.has_errors());
    assert!(
        strict_unknown
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.unknown_field"
                && diag.component.as_deref() == Some("opentelemetry")
                && diag.field.as_deref() == Some("unexpected"))
    );

    let strict_bad_transport = test_validate_static_plugin_config(&plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{"type": "openinference", "endpoint": "http://localhost:4318/v1/traces", "transport": "udp"}]
        }
    })));
    assert!(strict_bad_transport.has_errors());
    assert!(
        strict_bad_transport
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("endpoints[0].transport"))
    );
}

#[test]
fn atof_endpoint_validation_rejects_bad_values() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let report = test_validate_static_plugin_config(&plugin_config(json!({
        "atof": {
            "enabled": true,
            "sinks": [
                {"type": "stream", "url": "", "transport": "http_post"},
                {"type": "stream", "url": "http://localhost/events", "transport": "bogus"},
                {"type": "stream", "url": "http://localhost/events", "transport": "ndjson", "timeout_millis": 0},
                {"type": "stream", "url": "not a url", "transport": "http_post"},
                {"type": "stream", "url": "http://localhost/events", "transport": "http_post", "field_name_policy": "bogus"},
                {"type": "stream", "url": "http://localhost/events", "transport": "websocket"},
                {"type": "stream", "url": "http://localhost/events", "headers": {"invalid header": "value", "x-api-key": "value"}, "header_env": {"X-Api-Key": "NEMO_RELAY_TEST_MISSING_ATOF_HEADER_ENV"}},
                {"type": "stream", "name": "switchyard", "url": "http://localhost/first"},
                {"type": "stream", "name": "switchyard", "url": "http://localhost/second"},
                {"type": "stream", "name": " ", "url": "http://localhost/blank"}
            ]
        }
    })));

    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[0].url") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[1].transport") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[2].timeout_millis") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[3].url") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[4].field_name_policy") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[5].url") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[6].header_env.X-Api-Key") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[6].headers.invalid header") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[8].name") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("sinks[9].name") })
    );
}

#[test]
fn atof_stream_sink_name_validation_reports_each_invalid_name() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let report = test_validate_static_plugin_config(&plugin_config(json!({
        "atof": {
            "enabled": true,
            "sinks": [
                {"type": "stream", "url": "http://localhost/missing"},
                {"type": "stream", "name": "", "url": "http://localhost/empty"},
                {"type": "stream", "name": " leading", "url": "http://localhost/leading"},
                {"type": "stream", "name": "trailing ", "url": "http://localhost/trailing"},
                {"type": "stream", "name": "duplicate", "url": "http://localhost/first"},
                {"type": "stream", "name": "duplicate", "url": "http://localhost/second"},
                {"type": "stream", "name": "valid", "url": "http://localhost/valid"}
            ]
        }
    })));

    let name_diagnostics = report
        .diagnostics
        .iter()
        .filter(|diagnostic| {
            diagnostic
                .field
                .as_deref()
                .is_some_and(|field| field.ends_with(".name"))
        })
        .collect::<Vec<_>>();
    assert_eq!(name_diagnostics.len(), 4);
    assert!(name_diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("sinks[1].name")
            && diagnostic.message.contains("must be non-empty")
    }));
    assert!(name_diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("sinks[2].name")
            && diagnostic
                .message
                .contains("leading or trailing whitespace")
    }));
    assert!(name_diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("sinks[3].name")
            && diagnostic
                .message
                .contains("leading or trailing whitespace")
    }));
    assert!(name_diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("sinks[5].name")
            && diagnostic.message.contains("must be unique")
    }));
    assert!(!name_diagnostics.iter().any(|diagnostic| {
        matches!(
            diagnostic.field.as_deref(),
            Some("sinks[0].name" | "sinks[4].name" | "sinks[6].name")
        )
    }));
}

#[test]
fn initialization_fails_for_invalid_enabled_file_exporters() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-invalid-exporters");
    let not_a_directory = dir.join("not-a-directory");
    fs::write(&not_a_directory, "file").unwrap();

    let invalid_atof = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "atof": {
            "enabled": true,
            "sinks": [{"type": "file", "mode": "invalid", "output_directory": dir, "filename": "events.jsonl"}]
        }
    }));
    let error =
        futures::executor::block_on(test_initialize_plugin_host_exact(invalid_atof)).unwrap_err();
    assert!(error.to_string().contains("ATOF sinks[0].mode"));

    let invalid_atif_template = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "single-file.json"
        }
    }));
    let error =
        futures::executor::block_on(test_initialize_plugin_host_exact(invalid_atif_template))
            .unwrap_err();
    assert!(error.to_string().contains("filename_template"));

    let invalid_path = plugin_config(json!({
        "atof": {
            "enabled": true,
            "sinks": [{"type": "file", "output_directory": not_a_directory, "filename": "events.jsonl"}]
        }
    }));
    let error =
        futures::executor::block_on(test_initialize_plugin_host_exact(invalid_path)).unwrap_err();
    assert!(error.to_string().contains("registration failed"));

    let invalid_otel_transport = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{"type": "full", "endpoint": "http://localhost:4318/v1/traces", "transport": "udp"}]
        }
    }));
    let error =
        futures::executor::block_on(test_initialize_plugin_host_exact(invalid_otel_transport))
            .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires at least one valid trace, log, or metric endpoint")
    );
}

#[test]
fn atof_enabled_writes_jsonl_and_teardown_flushes() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atof");

    let config = plugin_config(json!({
        "atof": {
            "enabled": true,
            "sinks": [
                {"type": "file", "output_directory": dir, "filename": "events.jsonl", "mode": "overwrite"},
                {"type": "file", "output_directory": dir, "filename": "events-copy.jsonl", "mode": "overwrite"}
            ]
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    {
        let state = global_context();
        let names = state
            .read()
            .unwrap()
            .event_subscribers
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["nemo-relay-plugin.v1.observability:1:atof"]);
    }

    let agent = push_agent("atof-agent");
    nemo_relay_runtime::api::scope::event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("checkpoint")
            .parent(&agent)
            .data(json!({"step": 1}))
            .build(),
    )
    .unwrap();
    pop(&agent);
    test_close_plugin_host().unwrap();

    let content = fs::read_to_string(dir.join("events.jsonl")).unwrap();
    let lines = content.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("\"kind\":\"scope\""));
    assert!(lines[1].contains("\"kind\":\"mark\""));
    assert!(lines[2].contains("\"scope_category\":\"end\""));
    assert_eq!(
        fs::read_to_string(dir.join("events-copy.jsonl"))
            .unwrap()
            .lines()
            .count(),
        3
    );
}

#[test]
fn atof_stream_sinks_fan_out_and_teardown_all_workers() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let (first_url, first_captures) = start_http_capture_server(3);
    let (second_url, second_captures) = start_http_capture_server(3);

    let config = plugin_config(json!({
        "atof": {
            "enabled": true,
            "sinks": [
                {"type": "stream", "url": first_url, "transport": "http_post"},
                {"type": "stream", "url": second_url, "transport": "http_post"}
            ]
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let agent = push_agent("atof-stream-agent");
    nemo_relay_runtime::api::scope::event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("checkpoint")
            .parent(&agent)
            .data(json!({"step": 1}))
            .build(),
    )
    .unwrap();
    pop(&agent);
    test_close_plugin_host().unwrap();

    for captures in [&first_captures, &second_captures] {
        let bodies = wait_for_captures(captures, 3);
        assert_eq!(bodies.len(), 3, "captured bodies: {bodies:?}");
        let events = bodies
            .iter()
            .map(|capture| capture.body.as_str())
            .collect::<String>();
        assert!(events.contains("\"scope_category\":\"start\""));
        assert!(events.contains("\"name\":\"checkpoint\""));
        assert!(events.contains("\"scope_category\":\"end\""));
    }
}

#[test]
fn atof_stream_sink_header_env_is_snapshotted_at_activation() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let variable = format!("NEMO_RELAY_TEST_ATOF_HEADER_ENV_{}", std::process::id());
    // SAFETY: The test mutex serializes environment access for this process.
    unsafe { std::env::set_var(&variable, "Bearer relay-499") };
    let (url, captures) = start_http_capture_server(3);

    let config = plugin_config(json!({
        "atof": {
            "enabled": true,
            "sinks": [{
                "type": "stream",
                "url": url,
                "transport": "http_post",
                "header_env": {"authorization": variable.clone()}
            }]
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    // SAFETY: The active endpoint must use the header value captured at activation.
    unsafe { std::env::remove_var(&variable) };

    let agent = push_agent("atof-header-env-agent");
    nemo_relay_runtime::api::scope::event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("checkpoint")
            .parent(&agent)
            .data(json!({"step": 1}))
            .build(),
    )
    .unwrap();
    pop(&agent);
    test_close_plugin_host().unwrap();

    let captures = wait_for_captures(&captures, 3);
    assert_eq!(captures.len(), 3, "captured requests: {captures:?}");
    for capture in captures {
        assert!(capture.headers.lines().any(|line| {
            line.split_once(':').is_some_and(|(name, value)| {
                name.eq_ignore_ascii_case("authorization") && value.trim() == "Bearer relay-499"
            })
        }));
    }
}

#[test]
fn atif_remote_storage_validates_s3_configuration_and_http_access_outcomes() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let secret_var = format!("NEMO_RELAY_TEST_S3_SECRET_{}", std::process::id());
    // SAFETY: The variable name is unique to this test process and is removed before returning.
    unsafe { std::env::set_var(&secret_var, "test-secret") };
    let s3 = AtifStorageConfig::S3(S3StorageConfig {
        bucket: "test-bucket".into(),
        key_prefix: Some("trajectories".into()),
        access_key_id: Some("test-access-key".into()),
        secret_access_key_var: Some(secret_var.clone()),
        session_token_var: None,
        region: Some("us-east-1".into()),
        endpoint_url: Some("http://127.0.0.1:9".into()),
        allow_http: Some(true),
    });
    let storage = build_atif_storage(3, &s3, &PlaintextPolicy::default())
        .expect("S3 client configuration should resolve");
    drop(storage);
    // SAFETY: Cleanup of the test-only environment variable.
    unsafe { std::env::remove_var(&secret_var) };

    for (status, succeeds) in [("200 OK", true), ("503 Service Unavailable", false)] {
        let (endpoint, server) = start_http_status_server(status);
        let storage = AtifRemoteStorage::from_config(
            4,
            &AtifStorageConfig::Http(HttpStorageConfig {
                endpoint,
                headers: std::collections::HashMap::new(),
                header_env: std::collections::HashMap::new(),
                header_file: std::collections::HashMap::new(),
                timeout_millis: 5_000,
            }),
            &PlaintextPolicy::default(),
        )
        .unwrap();

        let result = storage.put("trajectory.json", "session", b"{}");
        drop(storage);
        server
            .join()
            .unwrap()
            .expect("test HTTP server should handle the upload");
        if succeeds {
            result.expect("HTTP storage upload should accept a success response");
        } else {
            assert!(
                result.is_err(),
                "HTTP storage upload should reject a {status} response"
            );
        }
    }
}

#[test]
fn atif_defaults_create_one_file_per_top_level_agent() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-defaults");

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let first = push_agent("first-agent");
    let nested = push_agent("nested-agent");
    pop(&nested);
    pop(&first);

    let second = push_agent("second-agent");
    pop(&second);
    test_close_plugin_host().unwrap();

    let first_path = dir.join(format!("nemo-relay-atif-{}.json", first.uuid));
    let second_path = dir.join(format!("nemo-relay-atif-{}.json", second.uuid));
    assert!(first_path.exists());
    assert!(second_path.exists());

    let first_json: Json = serde_json::from_str(&fs::read_to_string(first_path).unwrap()).unwrap();
    let second_json: Json =
        serde_json::from_str(&fs::read_to_string(second_path).unwrap()).unwrap();

    assert_eq!(first_json["session_id"], first.uuid.to_string());
    assert_eq!(first_json["agent"]["name"], "NeMo Relay");
    assert_eq!(first_json["agent"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(first_json["agent"]["model_name"], "unknown");
    assert_eq!(first_json["schema_version"], "ATIF-v1.7");
    assert_eq!(first_json["trajectory_id"], first.uuid.to_string());
    assert_eq!(
        first_json["subagent_trajectories"][0]["trajectory_id"],
        nested.uuid.to_string()
    );
    assert_eq!(
        first_json["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"][0]["trajectory_id"],
        nested.uuid.to_string()
    );
    let first_serialized = first_json.to_string();
    assert!(first_serialized.contains("first-agent"));
    assert!(first_serialized.contains("nested-agent"));
    assert!(!first_serialized.contains("second-agent"));

    let second_serialized = second_json.to_string();
    assert!(second_serialized.contains("second-agent"));
    assert!(!second_serialized.contains("first-agent"));
}

#[test]
fn atif_propagation_root_session_id_flows_through_plugin_e2e() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    let dir = temp_dir("observability-atif-propagation-root");
    let request_id = Uuid::parse_str("018f47a4-3af7-7d94-8e61-9f0f89b5d312").unwrap();
    let parent_id = Uuid::now_v7();
    let stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(request_id),
        parent_uuid: parent_id,
        traceparent: None,
        tracestate: None,
    })
    .unwrap();
    set_thread_scope_stack(stack);

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let first = push_agent("first-propagated-agent");
    pop(&first);
    let second = push_agent("second-propagated-agent");
    pop(&second);
    test_close_plugin_host().unwrap();

    for agent in [&first, &second] {
        let path = dir.join(format!("trajectory-{}.json", agent.uuid));
        let trajectory: Json = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(trajectory["session_id"], request_id.to_string());
        assert_eq!(trajectory["trajectory_id"], agent.uuid.to_string());
        assert_eq!(
            trajectory["extra"]["nemo_relay"]["session_instance_id"],
            request_id.to_string()
        );
    }
    assert_ne!(first.uuid, second.uuid);
}

#[test]
fn atif_filename_template_sanitizes_metadata_paths() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-metadata-template");

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "{metadata.routing.artifact_path}/trajectory-{session_id}.json"
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let sanitized = nemo_relay_runtime::api::scope::push_scope(
        PushScopeParams::builder()
            .name("sanitized-metadata-path-agent")
            .scope_type(ScopeType::Agent)
            .metadata(json!({"routing": {"artifact_path": "../escape"}}))
            .build(),
    )
    .unwrap();
    pop(&sanitized);

    let valid = nemo_relay_runtime::api::scope::push_scope(
        PushScopeParams::builder()
            .name("valid-metadata-path-agent")
            .scope_type(ScopeType::Agent)
            .metadata(json!({"routing": {"artifact_path": "tenant-a/session-123"}}))
            .build(),
    )
    .unwrap();
    pop(&valid);

    flush_subscribers().unwrap();
    assert!(
        dir.join(format!("-escape/trajectory-{}.json", sanitized.uuid))
            .exists(),
        "unsafe metadata groups should be replaced with a dash"
    );
    assert!(
        dir.join(format!(
            "tenant-a-session-123/trajectory-{}.json",
            valid.uuid
        ))
        .exists()
    );

    test_close_plugin_host().unwrap();

    futures::executor::block_on(test_initialize_plugin_host_exact(plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }))))
    .expect("ATIF teardown errors should not block later activation");
    test_close_plugin_host().expect("later ATIF teardown should succeed");
}

#[test]
fn atif_routes_global_descendant_events_by_parent_uuid() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-global-descendant");
    let root_uuid = nemo_relay_runtime::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("root-agent");
    let agent_uuid = agent.uuid;
    let child_uuid = Uuid::now_v7();
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(
        AtifSectionConfig {
            enabled: true,
            output_directory: Some(dir.clone()),
            ..AtifSectionConfig::default()
        },
        crate::observability_runtime::runtime_host(),
    )));
    let empty_storage: Arc<Vec<Arc<AtifRemoteStorage>>> = Arc::new(Vec::new());

    let mut start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent_uuid)
            .parent_uuid(root_uuid)
            .name("root-agent")
            .metadata(json!({
                "session_id": "root-session",
                "user_id": "alice"
            }))
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    start_event.set_propagation_root_uuid(Some(root_uuid));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &start_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let session_start_mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(agent_uuid)
            .name("session.start")
            .metadata(json!({
                "session_id": "root-session",
                "user_id": "alice",
                "session_instance_id": root_uuid.to_string()
            }))
            .build(),
        None,
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &session_start_mark,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let child_start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(child_uuid)
            .parent_uuid(agent_uuid)
            .name("child-agent")
            .metadata(json!({"session_id": "child-session"}))
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &child_start_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let child_end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(child_uuid)
            .parent_uuid(agent_uuid)
            .name("child-agent")
            .build(),
        ScopeCategory::End,
        vec![],
        EventCategory::agent(),
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &child_end_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let mut end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent_uuid)
            .parent_uuid(root_uuid)
            .name("root-agent")
            .build(),
        ScopeCategory::End,
        vec![],
        EventCategory::agent(),
        None,
    ));
    end_event.set_propagation_root_uuid(Some(root_uuid));
    let (pending_write, targets) = manager
        .lock()
        .unwrap()
        .observe_global(
            &end_event,
            "__test__",
            Arc::clone(&manager),
            Arc::clone(&empty_storage),
        )
        .unwrap();
    let path = dir.join(format!("nemo-relay-atif-{agent_uuid}.json"));
    let results = write_atif(&pending_write, empty_storage.as_slice(), &targets);
    for (label, result) in &results {
        assert!(result.is_ok(), "{label:?}: {result:?}");
    }
    let scope_subscriber = manager
        .lock()
        .unwrap()
        .complete_scope_write(agent_uuid, results);
    if let Some((scope_uuid, name)) = scope_subscriber {
        let _ = scope_deregister_subscriber(&scope_uuid, &name);
    }

    let value: Json = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(value["trajectory_id"], agent_uuid.to_string());
    assert_eq!(value["session_id"], root_uuid.to_string());
    assert_eq!(value["extra"]["nemo_relay"]["session_id"], "root-session");
    assert_eq!(
        value["extra"]["nemo_relay"]["session_instance_id"],
        root_uuid.to_string()
    );
    assert_eq!(value["extra"]["nemo_relay"]["user_id"], "alice");
    assert_eq!(
        value["extra"]["observed_events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|event| event["name"] == "session.start")
            .count(),
        1
    );
    assert!(
        value["steps"]
            .as_array()
            .is_some_and(|steps| steps.iter().all(|step| step["event"] != "session.start"))
    );
    assert_eq!(
        value["subagent_trajectories"][0]["session_id"],
        "child-session"
    );
    assert_eq!(
        value["subagent_trajectories"][0]["trajectory_id"],
        child_uuid.to_string()
    );
    assert_eq!(
        value["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"][0]["trajectory_id"],
        child_uuid.to_string()
    );
    pop(&agent);
}

#[test]
fn atif_writes_openclaw_child_only_fallback_without_mark_steps() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-openclaw-child-fallback");
    let root_uuid = nemo_relay_runtime::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let child_uuid = Uuid::now_v7();
    let child_mark_uuid = Uuid::now_v7();
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(
        AtifSectionConfig {
            enabled: true,
            output_directory: Some(dir.clone()),
            ..AtifSectionConfig::default()
        },
        crate::observability_runtime::runtime_host(),
    )));
    let empty_storage: Arc<Vec<Arc<AtifRemoteStorage>>> = Arc::new(Vec::new());

    let child_start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(child_uuid)
            .parent_uuid(root_uuid)
            .name("worker-agent")
            .metadata(json!({
                "session_id": "child-session",
                "nemo_relay_scope_role": "subagent"
            }))
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &child_start_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let child_mark_event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(child_mark_uuid)
            .parent_uuid(child_uuid)
            .name("worker-started")
            .data(json!({"status": "started"}))
            .build(),
        None,
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &child_mark_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let child_end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(child_uuid)
            .parent_uuid(root_uuid)
            .name("worker-agent")
            .build(),
        ScopeCategory::End,
        vec![],
        EventCategory::agent(),
        None,
    ));
    let (pending_write, targets) = manager
        .lock()
        .unwrap()
        .observe_global(
            &child_end_event,
            "__test__",
            Arc::clone(&manager),
            Arc::clone(&empty_storage),
        )
        .unwrap();
    let path = dir.join(format!("nemo-relay-atif-{child_uuid}.json"));
    let results = write_atif(&pending_write, empty_storage.as_slice(), &targets);
    for (label, result) in &results {
        assert!(result.is_ok(), "{label:?}: {result:?}");
    }
    let scope_subscriber = manager
        .lock()
        .unwrap()
        .complete_scope_write(child_uuid, results);
    if let Some((scope_uuid, name)) = scope_subscriber {
        let _ = scope_deregister_subscriber(&scope_uuid, &name);
    }

    let value: Json = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(value["trajectory_id"], child_uuid.to_string());
    assert!(value["steps"].as_array().unwrap().is_empty());
    assert!(
        value.get("subagent_trajectories").is_none() || value["subagent_trajectories"].is_null()
    );
    assert!(!value.to_string().contains("subagent_trajectory_ref"));
}

#[test]
fn atif_completed_top_level_agent_is_evicted_after_write() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-evict");
    let root_uuid = nemo_relay_runtime::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("evicted-agent");
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(
        AtifSectionConfig {
            enabled: true,
            output_directory: Some(dir.clone()),
            ..AtifSectionConfig::default()
        },
        crate::observability_runtime::runtime_host(),
    )));

    let start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent.uuid)
            .parent_uuid(root_uuid)
            .name("evicted-agent")
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    let empty_storage: Arc<Vec<Arc<AtifRemoteStorage>>> = Arc::new(Vec::new());
    manager.lock().unwrap().observe_global(
        &start_event,
        "__test__",
        Arc::clone(&manager),
        Arc::clone(&empty_storage),
    );
    {
        let dispatcher = manager.lock().unwrap();
        assert!(dispatcher.agents.contains_key(&agent.uuid));
        assert!(dispatcher.scope_subscribers.contains_key(&agent.uuid));
    }

    let end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent.uuid)
            .parent_uuid(root_uuid)
            .name("evicted-agent")
            .build(),
        ScopeCategory::End,
        vec![],
        EventCategory::agent(),
        None,
    ));
    let (pending_write, targets) = manager
        .lock()
        .unwrap()
        .observe_scope(&end_event, agent.uuid)
        .unwrap();
    let path = dir.join(format!("nemo-relay-atif-{}.json", agent.uuid));
    assert!(!path.exists());
    let results = write_atif(&pending_write, empty_storage.as_slice(), &targets);
    for (label, result) in &results {
        assert!(result.is_ok(), "{label:?}: {result:?}");
    }
    let scope_subscriber = manager
        .lock()
        .unwrap()
        .complete_scope_write(agent.uuid, results);
    if let Some((scope_uuid, name)) = scope_subscriber {
        let _ = scope_deregister_subscriber(&scope_uuid, &name);
    }

    let dispatcher = manager.lock().unwrap();
    assert!(dispatcher.fatal_error.is_none());
    assert!(dispatcher.runtime_failures.is_empty());
    assert!(!dispatcher.agents.contains_key(&agent.uuid));
    assert!(!dispatcher.scope_subscribers.contains_key(&agent.uuid));
    assert!(path.exists());
    drop(dispatcher);
    pop(&agent);
}

#[test]
fn atif_dispatcher_records_failed_agent_writes() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-write-error");
    let root_uuid = nemo_relay_runtime::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("failed-write-agent");
    let dispatcher = Arc::new(Mutex::new(AtifDispatcher::new(
        AtifSectionConfig {
            enabled: true,
            output_directory: Some(dir),
            ..AtifSectionConfig::default()
        },
        crate::observability_runtime::runtime_host(),
    )));

    let start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent.uuid)
            .parent_uuid(root_uuid)
            .name("failed-write-agent")
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    dispatcher.lock().unwrap().observe_global(
        &start_event,
        "__test__",
        Arc::clone(&dispatcher),
        Arc::new(Vec::new()),
    );

    let mut dispatcher = dispatcher.lock().unwrap();
    let scope_subscriber = dispatcher.complete_scope_write(
        agent.uuid,
        vec![(SinkLabel::Local, Err(std::io::Error::other("disk full")))],
    );
    assert!(scope_subscriber.is_some());
    assert_eq!(dispatcher.runtime_failures.len(), 1);
    assert_eq!(
        dispatcher.runtime_failures[0].code,
        "atif.local_write_failed"
    );
    assert!(dispatcher.last_error_result().is_err());
    drop(dispatcher);
    pop(&agent);
}

#[test]
fn atif_retention_options_cover_nested_steps_and_open_agent_teardown() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    for (include_observed_events, include_llm_requests, close_agent) in [false, true]
        .into_iter()
        .flat_map(|include_observed_events| {
            [false, true]
                .map(|include_llm_requests| (include_observed_events, include_llm_requests))
        })
        .flat_map(|(include_observed_events, include_llm_requests)| {
            [false, true]
                .map(|close_agent| (include_observed_events, include_llm_requests, close_agent))
        })
    {
        reset_runtime();
        let dir = temp_dir("observability-atif-retention");
        let config = plugin_config(json!({
            "policy": {"unknown_field": "error"},
            "atif": {
                "enabled": true,
                "output_directory": dir,
                "include_observed_events": include_observed_events,
                "include_llm_requests": include_llm_requests
            }
        }));
        futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
        let agent = push_agent("retention-agent");
        // Test a fresh user request and a continuation of the same turn.
        // They store requests on user and agent steps, respectively.
        for nested in [false, true] {
            let child = nested.then(|| push_agent("nested-retention-agent"));
            for continuation in [false, true] {
                let messages = if continuation {
                    json!([{"role": "tool", "tool_call_id": "call-1", "content": "result"}])
                } else {
                    json!([{"role": "user", "content": "hello"}])
                };
                let llm = nemo_relay_runtime::api::scope::push_scope(
                    PushScopeParams::builder()
                        .name("retention-llm")
                        .scope_type(ScopeType::Llm)
                        .input(json!({"messages": messages, "request_marker": "full-request"}))
                        .build(),
                )
                .unwrap();
                nemo_relay_runtime::api::scope::pop_scope(
                    PopScopeParams::builder()
                        .handle_uuid(&llm.uuid)
                        .output(json!({"content": "answer", "usage": {"prompt_tokens": 3, "completion_tokens": 1}}))
                        .build(),
                ).unwrap();
            }
            if let Some(child) = child {
                pop(&child);
            }
        }
        if close_agent {
            pop(&agent);
        }
        test_close_plugin_host().unwrap();
        let payload = fs::read(dir.join(format!("nemo-relay-atif-{}.json", agent.uuid))).unwrap();
        let value: Json = serde_json::from_slice(&payload).unwrap();
        assert!(payload.contains(&b'\n'), "local-only output stays pretty");
        assert_eq!(
            value["extra"].get("observed_events").is_some(),
            include_observed_events
        );
        let child = &value["subagent_trajectories"][0];
        assert!(child.is_object());
        for trajectory in [&value, child] {
            assert_atif_retention_projection(trajectory, include_llm_requests);
        }
        if !close_agent {
            pop(&agent);
        }
    }
}

fn assert_atif_retention_projection(trajectory: &Json, include_llm_requests: bool) {
    let steps = trajectory["steps"].as_array().unwrap();
    for source in ["user", "agent"] {
        assert!(steps.iter().any(|step| step["source"] == source));
    }
    assert!(steps.iter().any(|step| step["message"] == "answer"));
    assert_eq!(trajectory["final_metrics"]["total_prompt_tokens"], 6);
    for step in steps.iter().filter(|step| step["message"] == "answer") {
        assert!(step["extra"].get("ancestry").is_some());
    }
    let requests = steps
        .iter()
        .filter(|step| step["extra"].get("llm_request").is_some())
        .count();
    assert_eq!(requests, if include_llm_requests { 2 } else { 0 });
    assert!(
        steps
            .iter()
            .any(|step| step["extra"].get("llm_response").is_some())
    );
}

#[test]
fn atif_explicit_options_and_open_agent_teardown_are_written() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-explicit");

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "agent_name": "custom-agent",
            "agent_version": "9.9.9",
            "model_name": "demo-model",
            "tool_definitions": [{"name": "search"}],
            "extra": {"team": "runtime"},
            "output_directory": dir,
            "filename_template": "custom-{session_id}.atif.json"
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let ignored = push_function("not-an-agent");
    pop(&ignored);
    let agent = push_agent("open-agent");
    test_close_plugin_host().unwrap();

    let path = dir.join(format!("custom-{}.atif.json", agent.uuid));
    assert!(path.exists());
    let value: Json = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(value["agent"]["name"], "custom-agent");
    assert_eq!(value["agent"]["version"], "9.9.9");
    assert_eq!(value["agent"]["model_name"], "demo-model");
    assert_eq!(value["agent"]["tool_definitions"][0]["name"], "search");
    assert_eq!(value["agent"]["extra"]["team"], "runtime");
    assert!(fs::read_dir(dir).unwrap().count() == 1);
    pop(&agent);
}

#[test]
fn atif_open_agent_teardown_failure_retains_report_until_retry() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-open-agent-delivery-failure");
    let (endpoint, server) = start_http_status_server("500 Internal Server Error");
    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "http", "endpoint": endpoint}]
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    let agent = push_agent("open-agent-delivery-failure");

    let teardown = test_close_plugin_host().unwrap_err();
    assert!(teardown.to_string().contains("atif.remote_delivery_failed"));
    server.join().unwrap().unwrap();

    assert!(crate::test_support::plugin_host::test_plugin_host_report().is_some());
    test_close_plugin_host().expect("inactive test-host activation should clear on retry");
    assert!(crate::test_support::plugin_host::test_plugin_host_report().is_none());

    pop(&agent);
}

#[test]
fn atif_rejects_unsafe_template_and_ignores_non_top_level_agents() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-errors");

    let invalid_template = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "single-file.json"
        }
    }));
    assert!(test_validate_static_plugin_config(&invalid_template).has_errors());
    assert!(
        futures::executor::block_on(test_initialize_plugin_host_exact(invalid_template)).is_err()
    );

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let function = push_function("top-level-function");
    let nested_agent = push_agent("nested-under-function");
    pop(&nested_agent);
    pop(&function);
    test_close_plugin_host().unwrap();

    assert_eq!(fs::read_dir(dir).unwrap().count(), 0);
}

#[test]
fn otlp_sections_register_inferred_subscribers_with_full_config() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();

    let config = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "transport": "http_binary",
                    "endpoint": "http://127.0.0.1:4318/v1/traces",
                    "headers": {"authorization": "token"},
                    "resource_attributes": {"deployment.environment": "test"},
                    "service_name": "otel-service",
                    "service_namespace": "agents",
                    "service_version": "1.2.3",
                    "instrumentation_scope": "test-otel",
                    "timeout_millis": 1
                },
                {
                    "type": "openinference",
                    "endpoint": "http://127.0.0.1:4319/v1/traces",
                    "service_name": "oi-service"
                },
                {
                    "type": "gen_ai",
                    "endpoint": "http://127.0.0.1:4320/v1/traces"
                }
            ]
        }
    }));
    assert!(!test_validate_static_plugin_config(&config).has_errors());
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let state = global_context();
    let names = state
        .read()
        .unwrap()
        .event_subscribers
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert!(names.contains(&"nemo-relay-plugin.v1.observability:1:opentelemetry".to_string()));
    assert_eq!(
        names
            .iter()
            .filter(|name| *name == "nemo-relay-plugin.v1.observability:1:opentelemetry")
            .count(),
        1
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn opentelemetry_endpoints_fan_out_to_heterogeneous_and_repeated_types() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let (full_endpoint, full_request) = start_otlp_capture_server();
    let (gen_ai_endpoint, gen_ai_request) = start_otlp_capture_server();
    let (repeated_endpoint, repeated_request) = start_otlp_capture_server();

    let config = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {"type": "full", "endpoint": full_endpoint},
                {"type": "gen_ai", "endpoint": gen_ai_endpoint},
                {"type": "full", "endpoint": repeated_endpoint}
            ]
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let agent = push_agent("fanout-agent");
    pop(&agent);
    test_close_plugin_host().unwrap();

    for request in [full_request, gen_ai_request, repeated_request] {
        let body = request
            .recv_timeout(Duration::from_secs(5))
            .expect("each configured endpoint should receive the exported span");
        assert!(!body.is_empty());
    }
}

#[test]
fn opentelemetry_rejects_canonical_collision_during_validation_and_activation() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {"type": "full", "endpoint": " http://LOCALHOST:80//v1///traces/ "},
                {"type": "gen_ai", "endpoint": "http://127.1/v1/traces"}
            ]
        }
    }));

    let report = test_validate_static_plugin_config(&config);
    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.unsafe_otel_destination_collision"
            && diagnostic.field.as_deref() == Some("endpoints[1].endpoint")
            && diagnostic.message.contains("endpoints[0] (full)")
            && diagnostic.message.contains("endpoints[1] (gen_ai)")
            && diagnostic
                .message
                .contains("http://<loopback>:80/v1/traces")
    }));
    assert!(futures::executor::block_on(test_initialize_plugin_host_exact(config)).is_err());
    assert!(
        !global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
}

#[test]
fn opentelemetry_endpoint_delivery_failure_does_not_block_other_endpoints() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let (healthy_endpoint, healthy_request) = start_otlp_capture_server();
    let config = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "http://127.0.0.1:1/v1/traces",
                    "timeout_millis": 50
                },
                {"type": "openinference", "endpoint": healthy_endpoint}
            ]
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    let agent = push_agent("failure-isolation-agent");
    pop(&agent);
    let _ = test_close_plugin_host();

    let body = healthy_request
        .recv_timeout(Duration::from_secs(5))
        .expect("healthy endpoint should receive spans despite another endpoint failing");
    assert!(!body.is_empty());
}

#[test]
fn invalid_later_opentelemetry_endpoint_keeps_fanout_registration() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "http://127.0.0.1:4318/v1/traces"
                },
                {
                    "type": "gen_ai",
                    "endpoint": "not a valid endpoint"
                }
            ]
        }
    }));

    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    test_close_plugin_host().unwrap();
}

#[test]
fn opentelemetry_shutdown_helper_attempts_every_constructed_endpoint() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let (subscribers, exporters): (Vec<_>, Vec<_>) = (0..2)
        .map(|index| {
            let exporter = opentelemetry_sdk::trace::InMemorySpanExporterBuilder::new().build();
            let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_simple_exporter(exporter.clone())
                .build();
            (
                std::sync::Arc::new(
                    OpenTelemetrySubscriber::from_tracer_provider(
                        provider,
                        format!("rollback-{index}"),
                    )
                    .with_host(crate::observability_runtime::runtime_host()),
                ),
                exporter,
            )
        })
        .unzip();

    assert!(
        shutdown_opentelemetry_subscribers(
            &subscribers,
            crate::observability_runtime::runtime_host().as_ref()
        )
        .is_none()
    );
    for exporter in exporters {
        assert!(
            exporter.is_shutdown_called(),
            "every constructed endpoint exporter should be shut down"
        );
    }
}

#[test]
fn opentelemetry_shutdown_helper_retains_every_endpoint_failure() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let dropped_calls = Arc::new(AtomicUsize::new(0));
    let timeout_calls = Arc::new(AtomicUsize::new(0));
    let subscribers = [
        (
            format!("{OTEL_RUNTIME_DELIVERY_FAILURE_MARKER}: otel.spans_dropped (2)"),
            Arc::clone(&dropped_calls),
        ),
        (
            "endpoint shutdown timed out".to_string(),
            Arc::clone(&timeout_calls),
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (message, shutdown_calls))| {
        let processor = ShutdownFailureSpanProcessor {
            message,
            shutdown_calls,
        };
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_span_processor(processor)
            .build();
        Arc::new(
            OpenTelemetrySubscriber::from_tracer_provider(
                provider,
                format!("shutdown-failure-{index}"),
            )
            .with_host(crate::observability_runtime::runtime_host()),
        )
    })
    .collect::<Vec<_>>();

    let OpenTelemetryShutdownFailure::Other(error) = shutdown_opentelemetry_subscribers(
        &subscribers,
        crate::observability_runtime::runtime_host().as_ref(),
    )
    .expect("mixed endpoint shutdown failures should be reported") else {
        panic!("mixed endpoint shutdown failures must retain the registration failure outcome");
    };
    let error = error.to_string();

    assert_eq!(dropped_calls.load(Ordering::SeqCst), 1);
    assert_eq!(timeout_calls.load(Ordering::SeqCst), 1);
    assert!(
        error.contains(OTEL_RUNTIME_DELIVERY_FAILURE_MARKER),
        "{error}"
    );
    assert!(error.contains("endpoint shutdown timed out"), "{error}");
    assert!(
        !error.contains(&format!(
            "registration failed: {OTEL_RUNTIME_DELIVERY_FAILURE_MARKER}:"
        )),
        "mixed failures must not use the recoverable marker prefix: {error}"
    );
}

#[test]
fn log_only_plugin_reports_invalid_reserved_metric_marks_once() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "logs": {
                "enabled": true,
                "endpoints": [{
                    "endpoint": "http://127.0.0.1:1/v1/logs",
                    "timeout_millis": 50
                }]
            },
            "metrics": {"enabled": false}
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    for (version, data) in [
        ("999", json!({"measurements": [{"name": "ignored"}]})),
        ("1", json!({"measurements": []})),
    ] {
        nemo_relay_runtime::api::scope::event(
            nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
                .name("invalid-metric")
                .data(data)
                .data_schema(
                    DataSchema::builder()
                        .name(METRIC_DATA_SCHEMA_NAME)
                        .version(version)
                        .build(),
                )
                .build(),
        )
        .unwrap();
    }
    flush_subscribers().unwrap();

    let report = crate::test_support::plugin_host::test_plugin_host_report().unwrap();
    let diagnostics = report
        .runtime_diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "otel.metric_mark_invalid")
        .collect::<Vec<_>>();
    assert_eq!(diagnostics.len(), 1, "diagnostics: {diagnostics:?}");
    assert_eq!(diagnostics[0].count, 2);
    assert_eq!(diagnostics[0].field.as_deref(), None);

    test_close_plugin_host().unwrap();
}

#[test]
fn plugin_signal_rejections_record_runtime_diagnostics() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "logs": {
                "enabled": true,
                "endpoints": [{
                    "endpoint": "http://127.0.0.1:1/v1/logs",
                    "timeout_millis": 50
                }]
            },
            "metrics": {
                "enabled": true,
                "max_instruments": 1,
                "cardinality_limit": 1,
                "endpoints": [{
                    "endpoint": "http://127.0.0.1:1/v1/metrics",
                    "timeout_millis": 50
                }]
            }
        }
    }));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();

    nemo_relay_runtime::api::scope::event(
        nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
            .name("invalid-log-severity")
            .metadata(json!({"nemo_relay.log.severity": "notice"}))
            .build(),
    )
    .unwrap();
    for data in [
        json!({"measurements": [{
            "name": "example.one",
            "kind": "counter",
            "value_type": "u64",
            "value": 1,
            "attributes": {"model": "one"}
        }]}),
        json!({"measurements": [{
            "name": "example.one",
            "kind": "counter",
            "value_type": "u64",
            "value": 1,
            "attributes": {"model": "two"}
        }]}),
        json!({"measurements": [{
            "name": "example.two",
            "kind": "gauge",
            "value_type": "i64",
            "value": 1
        }]}),
        json!({"measurements": [{
            "name": "example.one",
            "kind": "gauge",
            "value_type": "i64",
            "value": 1
        }]}),
    ] {
        nemo_relay_runtime::api::scope::event(
            nemo_relay_runtime::api::scope::EmitMarkEventParams::builder()
                .name("metric-rejection")
                .data(data)
                .data_schema(
                    DataSchema::builder()
                        .name(METRIC_DATA_SCHEMA_NAME)
                        .version(METRIC_DATA_SCHEMA_VERSION)
                        .build(),
                )
                .build(),
        )
        .unwrap();
    }
    flush_subscribers().unwrap();

    let report = crate::test_support::plugin_host::test_plugin_host_report().unwrap();
    for (code, field) in [
        (
            "otel.log_mark_invalid_severity",
            "opentelemetry.logs.endpoints[0].endpoint",
        ),
        (
            "otel.metric_cardinality_limit",
            "opentelemetry.metrics.endpoints[0].endpoint",
        ),
        (
            "otel.metric_instrument_limit",
            "opentelemetry.metrics.endpoints[0].endpoint",
        ),
        (
            "otel.metric_descriptor_conflict",
            "opentelemetry.metrics.endpoints[0].endpoint",
        ),
    ] {
        assert!(
            report.runtime_diagnostics.iter().any(|diagnostic| {
                diagnostic.code == code && diagnostic.field.as_deref() == Some(field)
            }),
            "missing {code} diagnostic: {:?}",
            report.runtime_diagnostics
        );
    }

    assert!(test_close_plugin_host().is_err());
}

#[test]
fn atif_storage_private_helpers_resolve_env_and_key_prefix_branches() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let missing = "NEMO_RELAY_TEST_ATIF_HELPER_MISSING_ZZZZ";
    let empty = "NEMO_RELAY_TEST_ATIF_HELPER_EMPTY_ZZZZ";
    let secret = "NEMO_RELAY_TEST_ATIF_HELPER_SECRET_ZZZZ";
    let token = "NEMO_RELAY_TEST_ATIF_HELPER_TOKEN_ZZZZ";
    // SAFETY: these uniquely named variables are only touched by this test.
    unsafe {
        std::env::remove_var(missing);
        std::env::set_var(empty, "");
        std::env::set_var(secret, "secret-value");
        std::env::set_var(token, "token-value");
    }

    assert_eq!(resolve_env_var_field("field", None).unwrap(), None);
    assert!(
        resolve_env_var_field("field", Some(" padded "))
            .unwrap_err()
            .to_string()
            .contains("must be the name of an environment variable")
    );
    assert!(
        resolve_env_var_field("field", Some(missing))
            .unwrap_err()
            .to_string()
            .contains("is not set")
    );
    assert!(
        resolve_env_var_field("field", Some(empty))
            .unwrap_err()
            .to_string()
            .contains("set but empty")
    );
    assert_eq!(
        resolve_env_var_field("field", Some(secret)).unwrap(),
        Some("secret-value".to_string())
    );

    assert_eq!(normalize_storage_key_prefix(None), "");
    assert_eq!(
        normalize_storage_key_prefix(Some("  nested/path  ")),
        "nested/path/"
    );
    assert_eq!(
        normalize_storage_key_prefix(Some("nested/path/")),
        "nested/path/"
    );

    let overrides = S3BuilderOverrides::resolve(
        3,
        &S3StorageConfig {
            bucket: "bucket".into(),
            key_prefix: Some("prefix".into()),
            access_key_id: Some("access".into()),
            secret_access_key_var: Some(secret.into()),
            session_token_var: Some(token.into()),
            region: Some("us-west-2".into()),
            endpoint_url: Some("http://127.0.0.1:9000".into()),
            allow_http: Some(true),
        },
    )
    .unwrap();
    assert_eq!(overrides.access_key_id.as_deref(), Some("access"));
    assert_eq!(overrides.secret_access_key.as_deref(), Some("secret-value"));
    assert_eq!(overrides.session_token.as_deref(), Some("token-value"));
    assert_eq!(overrides.region.as_deref(), Some("us-west-2"));
    assert_eq!(
        overrides.endpoint_url.as_deref(),
        Some("http://127.0.0.1:9000")
    );
    assert_eq!(overrides.allow_http, Some(true));
    let _builder = overrides.apply(object_store::aws::AmazonS3Builder::from_env());

    // SAFETY: cleanup of test-only env vars.
    unsafe {
        std::env::remove_var(empty);
        std::env::remove_var(secret);
        std::env::remove_var(token);
    }
}

fn http_storage_config(endpoint: impl Into<String>) -> HttpStorageConfig {
    HttpStorageConfig {
        endpoint: endpoint.into(),
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        header_file: std::collections::HashMap::new(),
        timeout_millis: 1_000,
    }
}

#[test]
fn http_upload_config_rejects_endpoint_timeout_and_header_errors() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    for endpoint in [" http://example.com", "://", "ftp://example.com"] {
        assert!(
            HttpUploadConfig::resolve(
                2,
                &http_storage_config(endpoint),
                &PlaintextPolicy::default()
            )
            .is_err()
        );
    }

    let mut config = http_storage_config("https://example.com/atif");
    config.timeout_millis = 0;
    assert!(HttpUploadConfig::resolve(2, &config, &PlaintextPolicy::default()).is_err());

    config.timeout_millis = 1_000;
    config.headers.insert("bad header".into(), "value".into());
    assert!(HttpUploadConfig::resolve(2, &config, &PlaintextPolicy::default()).is_err());

    config.headers.clear();
    config.headers.insert("x-bad".into(), "bad\nvalue".into());
    assert!(HttpUploadConfig::resolve(2, &config, &PlaintextPolicy::default()).is_err());

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    std::fs::write(&path, "Bearer token\n").unwrap();
    let mut remote_file_header = http_storage_config("http://collector.example/atif");
    remote_file_header
        .header_file
        .insert("authorization".into(), path.to_string_lossy().into_owned());
    assert!(
        HttpUploadConfig::resolve(2, &remote_file_header, &PlaintextPolicy::default()).is_err()
    );
    remote_file_header.endpoint = "http://127.0.0.1:4318/atif".into();
    assert!(HttpUploadConfig::resolve(2, &remote_file_header, &PlaintextPolicy::default()).is_ok());
    remote_file_header.endpoint = "http://collector.example/atif".into();
    remote_file_header.header_file.clear();
    assert!(HttpUploadConfig::resolve(2, &remote_file_header, &PlaintextPolicy::default()).is_ok());

    remote_file_header
        .headers
        .insert("authorization".into(), "Bearer static".into());
    assert!(
        HttpUploadConfig::resolve(2, &remote_file_header, &PlaintextPolicy::default()).is_err()
    );
    remote_file_header.endpoint = "http://127.0.0.1:4318/atif".into();
    assert!(HttpUploadConfig::resolve(2, &remote_file_header, &PlaintextPolicy::default()).is_ok());
    remote_file_header.headers.clear();

    let variable = "NEMO_RELAY_TEST_ATIF_HTTP_RESOLVE_ZZZZ";
    // SAFETY: this uniquely named environment variable is serialized by the observability mutex.
    unsafe { std::env::set_var(variable, "Bearer resolved") };
    config.headers.clear();
    config
        .header_env
        .insert("authorization".into(), variable.into());
    config.endpoint = "http://collector.example/atif".into();
    assert!(HttpUploadConfig::resolve(2, &config, &PlaintextPolicy::default()).is_err());
    config.endpoint = "http://127.0.0.1:4318/atif".into();
    let resolved = HttpUploadConfig::resolve(2, &config, &PlaintextPolicy::default()).unwrap();
    assert_eq!(
        resolved.headers.get("authorization").map(String::as_str),
        Some("Bearer resolved")
    );
    // SAFETY: cleanup of the test-only environment variable.
    unsafe { std::env::remove_var(variable) };
}

#[tokio::test]
async fn post_atif_http_reports_transport_failure() {
    let config = HttpUploadConfig::resolve(
        0,
        &http_storage_config("http://127.0.0.1:1"),
        &PlaintextPolicy::default(),
    )
    .unwrap();
    let client = reqwest::Client::builder()
        .timeout(config.timeout)
        .build()
        .unwrap();
    assert!(
        post_atif_http(
            &client,
            &config,
            "trajectory.json".into(),
            "session".into(),
            b"{}".to_vec(),
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("HTTP ATIF upload")
    );
}

#[test]
fn s3_remote_storage_uploads_to_a_custom_http_endpoint() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let variable = "NEMO_RELAY_TEST_S3_UPLOAD_SECRET_ZZZZ";
    // SAFETY: the observability mutex serializes access to this test-only variable.
    unsafe { std::env::set_var(variable, "secret") };
    let (endpoint, server) = start_http_status_server("200 OK");
    let storage = AtifRemoteStorage::from_config(
        7,
        &AtifStorageConfig::S3(S3StorageConfig {
            bucket: "test-bucket".into(),
            key_prefix: Some("trajectories".into()),
            access_key_id: Some("access".into()),
            secret_access_key_var: Some(variable.into()),
            session_token_var: None,
            region: Some("us-east-1".into()),
            endpoint_url: Some(endpoint),
            allow_http: Some(true),
        }),
        &PlaintextPolicy::default(),
    )
    .unwrap();
    let result = storage.put("trajectory.json", "session", b"{}");
    server.join().unwrap().unwrap();
    // SAFETY: cleanup of the test-only environment variable.
    unsafe { std::env::remove_var(variable) };
    result.unwrap();
}

// --- OpenTelemetry file sinks ----------------------------------------------

fn file_sink_section(output_directory: &Path) -> OpenTelemetryFileSinkConfig {
    OpenTelemetryFileSinkConfig {
        otel_type: OpenTelemetryType::Full,
        output_directory: output_directory.to_path_buf(),
        filename: Some("trace.jsonl".to_string()),
        format: OtlpFileFormat::JsonLines,
        mode: default_otlp_file_sink_mode(),
        mark_projection: MarkProjection::default(),
        mark_exclude_names: default_mark_exclude_names(),
        attribute_mappings: Vec::new(),
        promote_metadata_prefixes: Vec::new(),
        promote_resource_metadata_prefixes: Vec::new(),
        resource_attributes: HashMap::new(),
        service_name: default_otel_service_name(),
        service_namespace: None,
        service_version: None,
        instrumentation_scope: default_otel_instrumentation_scope(),
        max_queue_size: None,
        max_export_batch_size: None,
        scheduled_delay_millis: None,
        completed_span_context_ttl_millis: None,
    }
}

#[test]
fn opentelemetry_registration_accepts_a_section_with_only_file_sinks() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let mut context = PluginRegistrationContext::new();

    register_opentelemetry(
        OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![file_sink_section(directory.path())],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
        &PlaintextPolicy::default(),
        &mut context,
        false,
        &crate::observability_runtime::runtime_host(),
    )
    .unwrap();

    assert!(directory.path().join("trace.jsonl").is_file());
    nemo_relay_plugin_host::rollback_registrations(&mut context.into_registrations());
}

#[test]
fn a_failed_signal_leaves_an_overwrite_file_sink_untouched() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");
    std::fs::write(&path, b"earlier\n").unwrap();
    let mut context = PluginRegistrationContext::new();

    // A warning policy lets an invalid severity reach registration. An
    // overwrite sink truncates as it opens, so it must not be opened until
    // every signal has been built.
    let error = register_opentelemetry(
        OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![file_sink_section(directory.path())],
            endpoints: Vec::new(),
            logs: Some(OpenTelemetryLogSectionConfig {
                enabled: true,
                endpoints: Some(vec![test_signal_endpoint()]),
                minimum_severity: "notice".to_string(),
                ..Default::default()
            }),
            metrics: None,
        },
        &PlaintextPolicy::default(),
        &mut context,
        false,
        &crate::observability_runtime::runtime_host(),
    )
    .unwrap_err();

    assert!(matches!(error, PluginError::InvalidConfig(_)), "{error}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "earlier\n");
    nemo_relay_plugin_host::rollback_registrations(&mut context.into_registrations());
}

#[test]
fn opentelemetry_registration_rejects_an_empty_endpoint_and_file_sink_list() {
    let mut context = PluginRegistrationContext::new();
    let error = register_opentelemetry(
        OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: Vec::new(),
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
        &PlaintextPolicy::default(),
        &mut context,
        false,
        &crate::observability_runtime::runtime_host(),
    )
    .unwrap_err();

    assert!(error.to_string().contains("one file sink"));
}

#[test]
fn an_invalid_file_sink_is_skipped_while_the_others_register() {
    // Registration touches the global subscriber registry.
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let mut broken = file_sink_section(directory.path());
    broken.mode = "truncate".to_string();
    broken.filename = Some("broken.jsonl".to_string());
    let mut working = file_sink_section(directory.path());
    working.filename = Some("working.jsonl".to_string());

    let mut context = PluginRegistrationContext::new();
    register_opentelemetry(
        OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![broken, working],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
        &PlaintextPolicy::default(),
        &mut context,
        false,
        &crate::observability_runtime::runtime_host(),
    )
    .unwrap();

    // One bad sink does not stop delivery to the rest.
    assert!(!directory.path().join("broken.jsonl").exists());
    assert!(directory.path().join("working.jsonl").is_file());
    nemo_relay_plugin_host::rollback_registrations(&mut context.into_registrations());
}

#[test]
fn file_sinks_register_even_when_the_environment_configures_automatic_signals() {
    const CHILD_MARKER: &str = "NEMO_RELAY_TEST_FILE_SINK_AUTOMATIC_CHILD";
    if std::env::var(CHILD_MARKER).is_ok() {
        let directory = tempfile::tempdir().unwrap();
        let mut context = PluginRegistrationContext::new();
        register_opentelemetry(
            OpenTelemetrySectionConfig {
                enabled: true,
                file_sinks: vec![file_sink_section(directory.path())],
                endpoints: Vec::new(),
                logs: None,
                metrics: None,
            },
            &PlaintextPolicy::default(),
            &mut context,
            false,
            &crate::observability_runtime::runtime_host(),
        )
        .unwrap();
        // A section carrying only file sinks is not empty, so an automatic
        // endpoint from the environment must not displace it.
        assert!(!opentelemetry_section_is_empty(
            &OpenTelemetrySectionConfig {
                enabled: true,
                file_sinks: vec![file_sink_section(directory.path())],
                endpoints: Vec::new(),
                logs: None,
                metrics: None,
            }
        ));
        assert!(directory.path().join("trace.jsonl").is_file());
        nemo_relay_plugin_host::rollback_registrations(&mut context.into_registrations());
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "--nocapture", "--test-threads", "1"])
        .arg(std::thread::current().name().unwrap())
        .env(CHILD_MARKER, "1")
        .env(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "https://collector.example:4318",
        )
        .output()
        .unwrap();
    let summary = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "child run failed: {}\n{summary}",
        output.status
    );
    // An exact filter that no longer matches leaves libtest exiting
    // successfully, so the child has to report that it ran the test.
    assert!(
        summary.contains("1 passed"),
        "child ran no test:\n{summary}"
    );
}

fn assert_automatic_exporter_signal_counter(
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
    component_key: &str,
) {
    use nemo_relay_diagnostics::DiagnosticQuery;
    let selected = diagnostics
        .snapshot(&DiagnosticQuery {
            component_key: Some(component_key.to_string()),
            code: Some("plugin.exporter_initialization_failed".to_string()),
            ..Default::default()
        })
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(selected.events.len(), 1);
    assert_eq!(selected.components.len(), 1);
    assert!(selected.components[0].closed_at.is_some());
    assert_eq!(
        selected.components[0].counters[0].occurrences.value.get(),
        1
    );
}

fn assert_automatic_exporter_observations(
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
    reports: &[ConfigDiagnostic],
) {
    use nemo_relay_diagnostics::{DiagnosticQuery, PluginPhase};
    let snapshot = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(snapshot.events.len(), 3);
    let signal_keys = [
        "opentelemetry.automatic.traces",
        "opentelemetry.automatic.logs",
        "opentelemetry.automatic.metrics",
    ];
    for ((event, report), component_key) in snapshot.events.iter().zip(reports).zip(signal_keys) {
        assert_eq!(event.component.key, component_key);
        assert_automatic_exporter_signal_counter(diagnostics, component_key);
        assert_eq!(event.code(), "plugin.exporter_initialization_failed");
        assert_eq!(event.classification.failure_class(), Some("invalid_input"));
        assert_eq!(event.context.phase, Some(PluginPhase::Activation.into()));
        assert_eq!(
            event.severity,
            nemo_relay_diagnostics::DiagnosticSeverity::Warning
        );
        assert_eq!(event.detail.as_ref().unwrap().message, report.message);
    }
    let serialized = serde_json::to_string(&snapshot).unwrap();
    for secret in ["user", "secret", "/private", "%ZZ"] {
        assert!(!serialized.contains(secret), "diagnostics exposed {secret}");
    }
}

#[test]
fn automatic_exporter_failures_reach_typed_diagnostics_and_legacy_reports_once() {
    use nemo_relay_diagnostics::{Diagnostics, DiagnosticsSettings};
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let variables = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
        "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
        "OTEL_TRACES_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_METRICS_EXPORTER",
        "OTEL_SDK_DISABLED",
        AUTOMATIC_INSECURE_HTTP_HOSTS_VARIABLE,
    ];
    let _environment = EnvironmentGuard::capture(variables);
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        for variable in variables {
            std::env::remove_var(variable);
        }
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "https://user:secret@collector.example/private?token=secret-query",
        );
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "authorization=secret-header%ZZ",
        );
    }
    for enabled in [true, false] {
        let diagnostics = Diagnostics::from_settings(
            Diagnostics::default().source().clone(),
            DiagnosticsSettings {
                enabled,
                ..Default::default()
            },
        )
        .unwrap();
        let mut ctx = PluginRegistrationContext::with_diagnostics(&diagnostics);
        let projected = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&projected);
        let _projection = diagnostics
            .subscribe_projection(move |_| {
                count.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
        let config = serde_json::from_value(json!({"version": 4})).unwrap();
        register_observability(
            config,
            &mut ctx,
            &crate::observability_runtime::runtime_host(),
        )
        .unwrap();
        let reports = ctx.activation_diagnostics();
        assert_eq!(reports.len(), 3);
        for report in reports {
            assert!(report.message.contains("invalid exporter headers"));
            assert_eq!(report.code, "observability.automatic_otel_signal_skipped");
        }
        if enabled {
            assert_automatic_exporter_observations(&diagnostics, reports);
        }
        // Existing operational logging owns output; the adapter must not duplicate it.
        assert_eq!(projected.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn automatic_exporter_unknown_causes_stay_unknown_and_healthy_signals_are_unchanged() {
    use nemo_relay_diagnostics::{DiagnosticQuery, Diagnostics, DiagnosticsSettings};
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let diagnostics = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let mut ctx = PluginRegistrationContext::with_diagnostics(&diagnostics);
    let failed = automatic_subscriber::<()>(
        true,
        "traces",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        &mut ctx,
        || {
            Err(
                crate::observability_test_support::otel::OpenTelemetryError::ExporterBuild(
                    "TLS private-key secret: invalid input unavailable".to_string(),
                )
                .into(),
            )
        },
    );
    assert!(failed.is_empty());
    let healthy = automatic_subscriber(
        true,
        "metrics",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        &mut ctx,
        || Ok(()),
    );
    assert_eq!(healthy.len(), 1);
    let disabled = automatic_subscriber::<()>(
        false,
        "logs",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        &mut ctx,
        || panic!("disabled exporter built"),
    );
    assert!(disabled.is_empty());
    assert_eq!(ctx.activation_diagnostics().len(), 1);
    let snapshot = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(snapshot.events.len(), 1);
    assert_eq!(
        snapshot.events[0].component.key,
        "opentelemetry.automatic.traces"
    );
    assert_eq!(
        snapshot.events[0].classification.failure_class(),
        Some("unknown")
    );
    assert!(
        !serde_json::to_string(&snapshot)
            .unwrap()
            .contains("private-key secret")
    );
}

// Host-level coverage uses the production initialization API with ambient config
// disabled so an embedding application's files cannot affect the assertions.
#[cfg(feature = "__skip-implicit-config")]
fn automatic_host_diagnostics_environment() -> EnvironmentGuard {
    let mut variables = vec![
        "OTEL_SDK_DISABLED".to_string(),
        "OTEL_TRACES_EXPORTER".to_string(),
        "OTEL_LOGS_EXPORTER".to_string(),
        "OTEL_METRICS_EXPORTER".to_string(),
        AUTOMATIC_INSECURE_HTTP_HOSTS_VARIABLE.to_string(),
        "NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG".to_string(),
    ];
    for signal in ["", "TRACES_", "LOGS_", "METRICS_"] {
        for suffix in [
            "ENDPOINT",
            "HEADERS",
            "PROTOCOL",
            "CERTIFICATE",
            "CLIENT_CERTIFICATE",
            "CLIENT_KEY",
        ] {
            variables.push(format!("OTEL_EXPORTER_OTLP_{signal}{suffix}"));
        }
    }
    let guard = EnvironmentGuard::capture(&variables);
    // SAFETY: callers hold the observability/runtime-owner test mutex.
    unsafe {
        for variable in variables {
            std::env::remove_var(variable);
        }
        std::env::set_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "1");
    }
    guard
}

#[cfg(feature = "__skip-implicit-config")]
fn automatic_host_failure_query(signal: &str) -> nemo_relay_diagnostics::DiagnosticQuery {
    nemo_relay_diagnostics::DiagnosticQuery {
        component_key: Some(format!("opentelemetry.automatic.{signal}")),
        code: Some("plugin.exporter_initialization_failed".into()),
        ..Default::default()
    }
}

#[cfg(feature = "__skip-implicit-config")]
fn assert_automatic_host_redacted(message: &str) {
    for secret in [
        "relay-user",
        "relay-secret",
        "/private",
        "relay-query-secret",
        "relay-private-missing-certificate",
        "relay-private-invalid-certificate",
        "relay-private-invalid-PEM",
    ] {
        assert!(!message.contains(secret), "diagnostic exposed {secret}");
    }
}

#[cfg(feature = "__skip-implicit-config")]
fn assert_automatic_host_rejection_report(diagnostics: &[ConfigDiagnostic]) -> &ConfigDiagnostic {
    let skipped = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "observability.automatic_otel_signal_skipped")
        .collect::<Vec<_>>();
    assert_eq!(skipped.len(), 1, "{diagnostics:?}");
    let legacy = skipped[0];
    assert_eq!(legacy.level, DiagnosticLevel::Warning);
    assert_eq!(
        legacy.field.as_deref(),
        Some("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    );
    assert!(!legacy.message.contains("underlying cause is unavailable"));
    assert!(
        legacy
            .message
            .contains("insecure_http_hosts or NEMO_RELAY_OTEL_INSECURE_HTTP_HOSTS")
    );
    assert_automatic_host_redacted(&legacy.message);
    legacy
}

#[cfg(feature = "__skip-implicit-config")]
fn assert_automatic_host_rejection_evidence(
    service: &nemo_relay_diagnostics::Diagnostics,
    legacy: &ConfigDiagnostic,
) {
    use nemo_relay_diagnostics::{DiagnosticSeverity, PluginPhase};
    // The production registration context has already been dropped here.
    let snapshot = service
        .snapshot(&automatic_host_failure_query("traces"))
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(snapshot.events.len(), 1);
    assert_eq!(snapshot.components.len(), 1);
    let event = &snapshot.events[0];
    assert_eq!(event.code(), "plugin.exporter_initialization_failed");
    assert_eq!(
        event.classification.failure_class(),
        Some("policy_rejected")
    );
    assert_eq!(event.context.phase, Some(PluginPhase::Activation.into()));
    assert_eq!(event.severity, DiagnosticSeverity::Warning);
    assert_eq!(
        event.origin.ownership.responsibility.as_deref(),
        Some("nemo-relay-plugin.v1.observability:1:")
    );
    assert_eq!(event.detail.as_ref().unwrap().message, legacy.message);
    let row = &snapshot.components[0];
    assert!(row.closed_at.is_some());
    assert_eq!(row.lifetime.totals.failures_total.value.get(), 1);
    assert_eq!(row.counters.len(), 1);
    assert_eq!(row.counters[0].occurrences.value.get(), 1);
    assert_automatic_host_redacted(&serde_json::to_string(&snapshot).unwrap());
}

#[cfg(feature = "__skip-implicit-config")]
fn assert_automatic_host_healthy_metrics(service: &nemo_relay_diagnostics::Diagnostics) {
    let healthy = service
        .snapshot(&automatic_host_failure_query("metrics"))
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert!(healthy.events.is_empty());
    assert!(
        healthy
            .components
            .iter()
            .all(|row| row.lifetime.totals.failures_total.value.get() == 0)
    );
}

#[cfg(feature = "__skip-implicit-config")]
#[test]
fn automatic_exporter_host_retains_rejection_after_activation_and_preserves_healthy_signal() {
    use nemo_relay_diagnostics::{DiagnosticError, Diagnostics, DiagnosticsSettings};
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let _environment = automatic_host_diagnostics_environment();
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://relay-user:relay-secret@collector.example:4318/private?token=relay-query-secret",
        );
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            "http://127.0.0.1:4318/v1/metrics",
        );
    }
    for enabled in [true, false] {
        let service = Diagnostics::from_settings(
            Diagnostics::default().source().clone(),
            DiagnosticsSettings {
                enabled,
                ..Default::default()
            },
        )
        .unwrap();
        let mut activation = futures::executor::block_on(
            nemo_relay_plugin_host::dynamic::initialize_with_diagnostics(
                plugin_config(json!({"version": 4})),
                None,
                &service,
            ),
        )
        .expect("a rejected automatic trace exporter must not fail host activation");
        let report = activation.report();
        let legacy = assert_automatic_host_rejection_report(&report.config.diagnostics);
        assert!(
            global_context()
                .read()
                .unwrap()
                .event_subscribers
                .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry.automatic",)
        );
        let query = automatic_host_failure_query("traces");
        if enabled {
            assert_automatic_host_rejection_evidence(&service, legacy);
            assert_automatic_host_healthy_metrics(&service);
        } else {
            assert!(matches!(
                service.snapshot(&query),
                Err(DiagnosticError::Disabled)
            ));
        }
        activation.close().unwrap();
        if enabled {
            let retained = service.snapshot(&query).unwrap().to_snapshot().unwrap();
            assert_eq!(retained.events.len(), 1);
            assert_eq!(
                retained.components[0].counters[0].occurrences.value.get(),
                1
            );
        }
    }
    reset_runtime();
}

#[cfg(feature = "__skip-implicit-config")]
#[test]
fn automatic_exporter_host_signal_counters_survive_without_recent_evidence() {
    use nemo_relay_diagnostics::{DiagnosticLimits, Diagnostics, DiagnosticsSettings};
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let _environment = automatic_host_diagnostics_environment();
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example");
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "authorization=relay-header-secret%ZZ",
        );
    }
    let service = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_events: 0,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let mut activation = futures::executor::block_on(
        nemo_relay_plugin_host::dynamic::initialize_with_diagnostics(
            plugin_config(json!({"version": 4})),
            None,
            &service,
        ),
    )
    .unwrap();
    let report = activation.report();
    assert_eq!(
        report
            .config
            .diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic.code == "observability.automatic_otel_signal_skipped"
            })
            .count(),
        3
    );
    activation.close().unwrap();
    for signal in ["traces", "logs", "metrics"] {
        assert_automatic_host_counter_without_evidence(&service, signal);
    }
    reset_runtime();
}

#[cfg(feature = "__skip-implicit-config")]
fn assert_automatic_host_counter_without_evidence(
    service: &nemo_relay_diagnostics::Diagnostics,
    signal: &str,
) {
    let query = automatic_host_failure_query(signal);
    let snapshot = service.snapshot(&query).unwrap().to_snapshot().unwrap();
    assert!(snapshot.events.is_empty());
    assert_eq!(snapshot.components.len(), 1, "{signal}");
    let row = &snapshot.components[0];
    assert!(row.closed_at.is_some());
    assert_eq!(
        row.identity.key,
        format!("opentelemetry.automatic.{signal}")
    );
    assert_eq!(
        row.since_refresh.ownership.responsibility.as_deref(),
        Some("nemo-relay-plugin.v1.observability:1:")
    );
    assert_eq!(row.lifetime.totals.failures_total.value.get(), 1);
    assert_eq!(row.counters.len(), 1);
    assert_eq!(
        row.counters[0].code,
        "plugin.exporter_initialization_failed"
    );
    assert_eq!(row.counters[0].occurrences.value.get(), 1);
    let reread = service.snapshot(&query).unwrap().to_snapshot().unwrap();
    assert_eq!(reread.sequence, snapshot.sequence);
    assert_eq!(reread.lifetime.totals, snapshot.lifetime.totals);
}

#[cfg(feature = "__skip-implicit-config")]
#[test]
fn automatic_exporter_host_classifies_endpoint_and_tls_failures_without_secrets() {
    use nemo_relay_diagnostics::{Diagnostics, DiagnosticsSettings};
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_runtime();
    let _environment = automatic_host_diagnostics_environment();
    let directory = tempfile::tempdir().unwrap();
    let missing_certificate = directory
        .path()
        .join("relay-private-missing-certificate.pem");
    let invalid_certificate = directory
        .path()
        .join("relay-private-invalid-certificate.pem");
    std::fs::write(&invalid_certificate, "relay-private-invalid-PEM").unwrap();
    // All three errors occur before an exporter can make a network request.
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            " https://relay-user:relay-secret@collector.example/private?token=relay-query-secret",
        );
        for signal in ["LOGS", "METRICS"] {
            std::env::set_var(
                format!("OTEL_EXPORTER_OTLP_{signal}_ENDPOINT"),
                "https://relay-user:relay-secret@collector.example/private?token=relay-query-secret",
            );
            std::env::set_var(
                format!("OTEL_EXPORTER_OTLP_{signal}_PROTOCOL"),
                "http/protobuf",
            );
        }
        std::env::set_var("OTEL_EXPORTER_OTLP_LOGS_CERTIFICATE", &missing_certificate);
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_METRICS_CERTIFICATE",
            &invalid_certificate,
        );
    }
    let service = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let mut activation = futures::executor::block_on(
        nemo_relay_plugin_host::dynamic::initialize_with_diagnostics(
            plugin_config(json!({"version": 4})),
            None,
            &service,
        ),
    )
    .expect("invalid automatic exporters must not prevent host activation");
    let report = activation.report();
    let skipped = report
        .config
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "observability.automatic_otel_signal_skipped")
        .collect::<Vec<_>>();
    assert_eq!(skipped.len(), 3, "{:?}", report.config.diagnostics);
    activation.close().unwrap();
    for (signal, field, expected_class, expected_reason) in [
        (
            "traces",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "invalid_input",
            "invalid OTLP endpoint",
        ),
        (
            "logs",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
            "io",
            "cannot read a configured TLS certificate or key file",
        ),
        (
            "metrics",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            "invalid_input",
            "invalid TLS configuration or certificate/key material",
        ),
    ] {
        let legacy = skipped
            .iter()
            .find(|diagnostic| diagnostic.field.as_deref() == Some(field))
            .unwrap();
        assert_automatic_host_source_failure(
            &service,
            legacy,
            signal,
            expected_class,
            expected_reason,
            directory.path(),
        );
    }
    reset_runtime();
}

#[cfg(feature = "__skip-implicit-config")]
fn assert_automatic_host_source_failure(
    service: &nemo_relay_diagnostics::Diagnostics,
    legacy: &ConfigDiagnostic,
    signal: &str,
    expected_class: &str,
    expected_reason: &str,
    directory: &std::path::Path,
) {
    assert!(
        legacy.message.contains(expected_reason),
        "{}",
        legacy.message
    );
    let snapshot = service
        .snapshot(&automatic_host_failure_query(signal))
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(snapshot.events.len(), 1, "{signal}");
    assert_eq!(
        snapshot.events[0].classification.failure_class(),
        Some(expected_class)
    );
    assert_eq!(
        snapshot.events[0].detail.as_ref().unwrap().message,
        legacy.message
    );
    assert_eq!(snapshot.components.len(), 1);
    assert_eq!(
        snapshot.components[0]
            .lifetime
            .totals
            .failures_total
            .value
            .get(),
        1
    );
    let serialized = serde_json::to_string(&snapshot).unwrap();
    assert_automatic_host_redacted(&serialized);
    assert_automatic_host_redacted(&legacy.message);
    assert!(!serialized.contains(&directory.to_string_lossy().to_string()));
}

#[test]
fn opentelemetry_endpoint_rejects_invalid_attribute_mappings() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let invalid_mappings = [
        (
            json!([{"key": "", "alias": "some.alias"}]),
            "attribute mapping key must not be blank",
        ),
        (
            json!([{"key": "nemo_relay.mark.metadata.source", "alias": " "}]),
            "attribute mapping alias must not be blank",
        ),
        (
            json!([{"key": "nemo_relay.mark.metadata.source", "alias": "\u{200b}"}]),
            "attribute mapping alias must not be blank",
        ),
        (
            json!([
                {"key": "nemo_relay.model_name", "alias": "duplicate.alias"},
                {"key": "nemo_relay.tool.name", "alias": "duplicate.alias"}
            ]),
            "attribute mapping alias \"duplicate.alias\" is duplicated",
        ),
    ];

    for (attribute_mappings, expected_message) in invalid_mappings {
        let config = plugin_config(json!({
            "opentelemetry": {
                "enabled": true,
                "endpoints": [{
                    "type": "full",
                    "endpoint": "http://localhost:4318/v1/traces",
                    "attribute_mappings": attribute_mappings
                }]
            }
        }));
        let report = test_validate_static_plugin_config(&config);
        assert!(report.has_errors());
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "observability.unsupported_value"
                && diagnostic.field.as_deref() == Some("endpoints[0].attribute_mappings")
                && diagnostic.message.contains(expected_message)
        }));
        assert!(futures::executor::block_on(test_initialize_plugin_host_exact(config)).is_err());
    }
}

#[test]
fn plugin_schema_contains_generic_plugin_surface() {
    let schema = plugin_config_schema();
    for field in [
        "version",
        "components",
        "policy",
        "kind",
        "enabled",
        "config",
    ] {
        assert!(
            schema_has_property(&schema, field),
            "plugin schema missing property `{field}`"
        );
    }
}
