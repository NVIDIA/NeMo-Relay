// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the built-in observability plugin component.

use super::*;
use crate::api::event::{
    BaseEvent, DataSchema, EventCategory, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION,
    MarkEvent, ScopeEvent,
};
use crate::api::runtime::{
    NemoRelayContextState, PropagationContext, ThreadScopeStackBinding, capture_thread_scope_stack,
    create_scope_stack_from_propagation, global_context, restore_thread_scope_stack,
    set_thread_scope_stack,
};
use crate::api::scope::{PopScopeParams, PushScopeParams};
use crate::api::subscriber::scope_deregister_subscriber;
use crate::config_editor::{EditorConfig, EditorFieldKind, EditorSchema};
#[cfg(feature = "schema")]
use crate::plugin::plugin_config_schema;
use crate::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins_exact,
    list_plugin_kinds, lookup_plugin, validate_plugin_config,
};
use serde_json::json;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
#[cfg(feature = "atof-streaming")]
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

#[cfg(feature = "atof-streaming")]
#[derive(Clone, Debug)]
struct HttpCapture {
    headers: String,
    body: String,
}

#[cfg(feature = "atof-streaming")]
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

#[cfg(feature = "object-store")]
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

#[cfg(feature = "atof-streaming")]
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
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
    let _ = clear_plugin_configuration();
    crate::shared_runtime::reset_runtime_owner_for_tests();
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

fn assert_signal_editor_sections(otlp: &EditorSchema) {
    for signal in ["logs", "metrics"] {
        let signal = otlp.field(signal).expect("OTLP signal section");
        assert_eq!(signal.kind, EditorFieldKind::Section);
        assert!(signal.optional);
        let signal_schema = signal.schema().expect("OTLP signal editor schema");
        let endpoints = signal_schema
            .field("endpoints")
            .expect("OTLP signal endpoints");
        assert_eq!(endpoints.kind, EditorFieldKind::List);
        assert!(endpoints.optional);
        let endpoint_schema = endpoints
            .list_item
            .and_then(|item| item.schema)
            .expect("OTLP signal endpoint schema")();
        assert!(endpoint_schema.field("type").is_none());
        assert_eq!(
            endpoint_schema
                .field("endpoint")
                .expect("OTLP signal endpoint URL")
                .kind,
            EditorFieldKind::String
        );
        assert_eq!(
            endpoint_schema
                .field("header_env")
                .expect("OTLP signal endpoint header_env")
                .kind,
            EditorFieldKind::StringMap
        );
    }
    assert_eq!(
        otlp.field("logs")
            .and_then(EditorFieldSpec::schema)
            .and_then(|logs| logs.field("minimum_severity"))
            .expect("log severity field")
            .enum_values,
        &["trace", "debug", "info", "warn", "warning", "error"]
    );
    assert_eq!(
        otlp.field("metrics")
            .and_then(EditorFieldSpec::schema)
            .and_then(|metrics| metrics.field("temporality"))
            .expect("metric temporality field")
            .enum_values,
        &["cumulative", "delta", "low_memory"]
    );
}

fn assert_trace_endpoint_editor_schema(otlp: &EditorSchema) {
    let otlp_endpoints = otlp.field("traces").expect("traces field");
    assert_eq!(otlp_endpoints.kind, EditorFieldKind::List);
    let otlp_endpoint = otlp_endpoints
        .list_item
        .expect("OTLP endpoint list metadata");
    assert_eq!(otlp_endpoint.kind, EditorFieldKind::Section);
    let otlp_endpoint_schema = otlp_endpoint.schema.expect("OTLP endpoint schema")();
    assert_eq!(
        otlp_endpoint_schema
            .field("type")
            .expect("OTLP endpoint type")
            .enum_values,
        &["full", "gen_ai", "openinference"]
    );
    assert_eq!(
        otlp_endpoint_schema
            .field("endpoint")
            .expect("OTLP endpoint URL")
            .kind,
        EditorFieldKind::String
    );
    assert_eq!(
        otlp_endpoint_schema
            .field("header_env")
            .expect("OTLP endpoint header_env")
            .kind,
        EditorFieldKind::StringMap
    );
    let promote_metadata_prefixes = otlp_endpoint_schema
        .field("promote_metadata_prefixes")
        .expect("OTLP endpoint promote_metadata_prefixes");
    assert_eq!(promote_metadata_prefixes.kind, EditorFieldKind::List);
    assert!(promote_metadata_prefixes.optional);
    for field in [
        "max_queue_size",
        "max_export_batch_size",
        "scheduled_delay_millis",
    ] {
        let field = otlp_endpoint_schema.field(field).expect("batch field");
        assert_eq!(field.kind, EditorFieldKind::Integer);
        assert!(field.optional);
    }
}

#[test]
fn editor_schema_tracks_observability_config_types() {
    let schema = ObservabilityConfig::editor_schema();
    let version = schema.field("version").expect("config version field");
    assert_eq!(version.kind, EditorFieldKind::IntegerEnum);
    assert_eq!(version.enum_values, &["3", "4"]);
    assert_eq!(
        schema
            .field("enable_full_payloads")
            .expect("full payload field")
            .kind,
        EditorFieldKind::Boolean
    );
    let atof = schema.field("atof").expect("atof section");
    assert_eq!(atof.label, "ATOF");
    assert_eq!(atof.kind, EditorFieldKind::Section);
    assert!(atof.optional);

    let atof_schema = atof.schema().expect("atof editor schema");
    let sinks = atof_schema.field("sinks").expect("atof sinks field");
    assert_eq!(sinks.kind, EditorFieldKind::List);
    let sink = sinks.list_item.expect("ATOF sink list metadata");
    assert_eq!(sink.kind, EditorFieldKind::Section);
    assert_eq!(
        sink.tagged_union.map(|metadata| metadata.discriminator),
        Some("type")
    );
    assert_eq!(
        sink.tagged_union.expect("sink tagged union").variants.len(),
        2
    );

    let otlp = schema
        .field("opentelemetry")
        .expect("opentelemetry section")
        .schema()
        .expect("opentelemetry editor schema");
    assert_trace_endpoint_editor_schema(otlp);
    assert_signal_editor_sections(otlp);
    assert_eq!(
        default_opentelemetry_endpoint_editor_value(),
        json!({
            "type": "full",
            "endpoint": "",
            "transport": "http_binary",
            "service_name": "unknown_service",
            "instrumentation_scope": "opentelemetry",
            "timeout_millis": 3000,
            "headers": {},
            "header_env": {},
            "resource_attributes": {},
        })
    );
}

#[test]
fn signal_endpoint_lists_preserve_omitted_and_explicit_empty_shapes() {
    let omitted_logs = serde_json::to_value(OpenTelemetryLogSectionConfig::default()).unwrap();
    let omitted_metrics =
        serde_json::to_value(OpenTelemetryMetricSectionConfig::default()).unwrap();
    assert!(omitted_logs.get("endpoints").is_none());
    assert!(omitted_metrics.get("endpoints").is_none());

    let explicit_logs = serde_json::to_value(OpenTelemetryLogSectionConfig {
        endpoints: Some(Vec::new()),
        ..OpenTelemetryLogSectionConfig::default()
    })
    .unwrap();
    let explicit_metrics = serde_json::to_value(OpenTelemetryMetricSectionConfig {
        endpoints: Some(Vec::new()),
        ..OpenTelemetryMetricSectionConfig::default()
    })
    .unwrap();
    assert_eq!(explicit_logs["endpoints"], json!([]));
    assert_eq!(explicit_metrics["endpoints"], json!([]));
}

#[test]
fn observability_v3_remains_trace_only_and_v4_accepts_signal_sections() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let trace_only = plugin_config(json!({
        "version": 3,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "gen_ai",
                "endpoint": "https://collector.example/v1/traces"
            }]
        }
    }));
    assert!(!validate_plugin_config(&trace_only).has_errors());

    let version_three_logs = plugin_config(json!({
        "version": 3,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "gen_ai",
                "endpoint": "https://collector.example/v1/traces"
            }],
            "logs": {"enabled": false}
        }
    }));
    let report = validate_plugin_config(&version_three_logs);
    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("logs")
            && diagnostic.message.contains("version 3 is trace-only")
    }));

    let version_three_metrics = plugin_config(json!({
        "version": 3,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "gen_ai",
                "endpoint": "https://collector.example/v1/traces"
            }],
            "metrics": {"enabled": false}
        }
    }));
    let report = validate_plugin_config(&version_three_metrics);
    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("metrics")
            && diagnostic.message.contains("version 3 is trace-only")
    }));

    let version_four = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "logs": {
                "enabled": true,
                "endpoints": [{"endpoint": "https://collector.example/v1/logs"}]
            },
            "metrics": {
                "enabled": true,
                "endpoints": [{"endpoint": "https://collector.example/v1/metrics"}]
            }
        }
    }));
    assert!(!validate_plugin_config(&version_four).has_errors());
}

#[test]
fn disabled_signal_sections_reject_duplicate_explicit_destinations() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": false,
            "logs": {
                "enabled": false,
                "endpoints": [
                    {"endpoint": "https://collector.example"},
                    {"endpoint": "https://collector.example/v1/logs"}
                ]
            },
            "metrics": {
                "enabled": false,
                "endpoints": [
                    {"endpoint": "https://collector.example"},
                    {"endpoint": "https://collector.example/v1/metrics"}
                ]
            }
        }
    }));

    let report = validate_plugin_config(&config);
    for signal in ["logs", "metrics"] {
        let component = format!("opentelemetry.{signal}");
        assert!(
            report.diagnostics.iter().any(|diagnostic| {
                diagnostic.component.as_deref() == Some(component.as_str())
                    && diagnostic.field.as_deref() == Some("endpoints")
                    && diagnostic.message.contains("endpoints[0] and")
                    && diagnostic.message.contains("endpoints[1]")
            }),
            "missing disabled {signal} destination-collision diagnostic: {:?}",
            report.diagnostics
        );
    }
}

#[test]
fn signal_endpoint_resolution_derives_or_preserves_the_expected_destination() {
    let mut trace = test_opentelemetry_endpoint();
    trace.endpoint =
        "https://collector.example/prefix/v1/traces?tenant=observability-dev".to_string();
    trace
        .headers
        .insert("x-nv-project".to_string(), "observability-dev".to_string());
    trace
        .resource_attributes
        .insert("nv.project".to_string(), "observability-dev".to_string());

    let logs = resolve_signal_endpoints("logs", None, std::slice::from_ref(&trace)).unwrap();
    assert_eq!(
        logs.endpoints[0].value.as_active().unwrap().endpoint,
        "https://collector.example/prefix/v1/logs?tenant=observability-dev"
    );
    assert_eq!(
        logs.endpoints[0].value.as_active().unwrap().headers,
        trace.headers
    );
    assert_eq!(
        logs.endpoints[0]
            .value
            .as_active()
            .unwrap()
            .resource_attributes,
        trace.resource_attributes
    );

    let metrics = resolve_signal_endpoints("metrics", None, &[trace]).unwrap();
    assert_eq!(
        metrics.endpoints[0].value.as_active().unwrap().endpoint,
        "https://collector.example/prefix/v1/metrics?tenant=observability-dev"
    );

    let mut root_trace = test_opentelemetry_endpoint();
    root_trace.endpoint = "https://collector.example/".to_string();
    assert_eq!(
        resolve_signal_endpoints("logs", None, std::slice::from_ref(&root_trace))
            .unwrap()
            .endpoints[0]
            .value
            .as_active()
            .unwrap()
            .endpoint,
        "https://collector.example/v1/logs"
    );
    assert_eq!(
        resolve_signal_endpoints("metrics", None, &[root_trace])
            .unwrap()
            .endpoints[0]
            .value
            .as_active()
            .unwrap()
            .endpoint,
        "https://collector.example/v1/metrics"
    );

    let custom = vec![OpenTelemetrySignalEndpointConfig {
        endpoint: "https://collector.example/custom/metric-intake".to_string(),
        transport: default_otlp_transport(),
        headers: HashMap::new(),
        header_env: HashMap::new(),
        resource_attributes: HashMap::new(),
        service_name: default_otel_service_name(),
        service_namespace: None,
        service_version: None,
        instrumentation_scope: default_otel_instrumentation_scope(),
        timeout_millis: default_timeout_millis(),
    }];
    assert_eq!(
        resolve_signal_endpoints("metrics", Some(&custom), &[])
            .unwrap()
            .endpoints[0]
            .value
            .as_active()
            .unwrap()
            .endpoint,
        custom[0].endpoint
    );
    assert!(resolve_signal_endpoints("metrics", Some(&Vec::new()), &[]).is_err());
}

#[test]
fn signal_endpoint_resolution_skips_ambiguous_trace_paths_and_preserves_indexes() {
    let mut first = test_opentelemetry_endpoint();
    first.endpoint = "https://collector.example/first/v1/traces".to_string();
    let mut ambiguous = test_opentelemetry_endpoint();
    ambiguous.endpoint = "https://collector.example/custom/traces".to_string();
    let mut third = test_opentelemetry_endpoint();
    third.endpoint = "https://collector.example/third/v1/traces".to_string();

    let resolution = resolve_signal_endpoints("logs", None, &[first, ambiguous, third]).unwrap();
    assert_eq!(
        resolution
            .endpoints
            .iter()
            .map(|endpoint| endpoint.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        resolution
            .endpoints
            .iter()
            .map(|endpoint| endpoint.value.as_active().is_some())
            .collect::<Vec<_>>(),
        vec![true, false, true]
    );
}

#[test]
fn signal_endpoint_resolution_rejects_explicit_wrong_signal_paths() {
    let mut explicit = OpenTelemetrySignalEndpointConfig {
        endpoint: "https://collector.example/v1/traces".to_string(),
        transport: default_otlp_transport(),
        headers: HashMap::new(),
        header_env: HashMap::new(),
        resource_attributes: HashMap::new(),
        service_name: default_otel_service_name(),
        service_namespace: None,
        service_version: None,
        instrumentation_scope: default_otel_instrumentation_scope(),
        timeout_millis: default_timeout_millis(),
    };
    assert!(resolve_signal_endpoints("logs", Some(&vec![explicit.clone()]), &[]).is_err());
    explicit.endpoint = "https://collector.example/".to_string();
    assert!(resolve_signal_endpoints("logs", Some(&vec![explicit]), &[]).is_ok());
}

#[test]
fn signal_endpoint_resolution_covers_missing_trace_grpc_and_invalid_transports() {
    assert!(resolve_signal_endpoints("logs", None, &[]).is_err());
    assert!(resolve_signal_endpoints("metrics", Some(&Vec::new()), &[]).is_err());

    let mut grpc_trace = test_opentelemetry_endpoint();
    grpc_trace.transport = "grpc".to_string();
    grpc_trace.endpoint = "http://collector.example:4317".to_string();
    let derived = resolve_signal_endpoints("metrics", None, &[grpc_trace]).unwrap();
    assert_eq!(
        derived.endpoints[0].value.as_active().unwrap().endpoint,
        "http://collector.example:4317"
    );
    assert_eq!(
        derived.endpoints[0].value.as_active().unwrap().transport,
        "grpc"
    );

    for endpoint in ["ftp://collector.example", "not a url"] {
        let mut trace = test_opentelemetry_endpoint();
        trace.endpoint = endpoint.to_string();
        let resolution = resolve_signal_endpoints("logs", None, &[trace]).unwrap();
        assert_eq!(resolution.endpoints.len(), 1);
        assert!(matches!(
            resolution.endpoints[0].value,
            OpenTelemetryResource::Skipped(_)
        ));
    }

    let mut endpoint = test_signal_endpoint();
    endpoint.transport = "udp".to_string();
    assert!(resolve_signal_endpoints("logs", Some(&vec![endpoint]), &[]).is_err());
}

fn push_agent(name: &str) -> crate::api::scope::ScopeHandle {
    crate::api::scope::push_scope(
        PushScopeParams::builder()
            .name(name)
            .scope_type(ScopeType::Agent)
            .input(json!({"agent": name}))
            .build(),
    )
    .unwrap()
}

fn push_function(name: &str) -> crate::api::scope::ScopeHandle {
    crate::api::scope::push_scope(
        PushScopeParams::builder()
            .name(name)
            .scope_type(ScopeType::Function)
            .input(json!({"function": name}))
            .build(),
    )
    .unwrap()
}

fn pop(handle: &crate::api::scope::ScopeHandle) {
    crate::api::scope::pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .output(json!({"done": handle.name}))
            .build(),
    )
    .unwrap();
}

#[cfg(feature = "schema")]
fn schema_has_property(schema: &Json, name: &str) -> bool {
    schema_property(schema, name).is_some()
}

#[cfg(feature = "schema")]
fn schema_property_has_enum(schema: &Json, name: &str, expected: &[&str]) -> bool {
    schema_property_matches(schema, name, &|property| {
        property
            .get("enum")
            .and_then(Json::as_array)
            .is_some_and(|values| {
                expected
                    .iter()
                    .all(|expected| values.iter().any(|value| value == *expected))
            })
    })
}

#[cfg(feature = "schema")]
fn schema_property_has_default(schema: &Json, name: &str, expected: Json) -> bool {
    schema_property_matches(schema, name, &|property| {
        property
            .get("default")
            .is_some_and(|default| default == &expected)
    })
}

#[cfg(feature = "schema")]
fn schema_property_matches(schema: &Json, name: &str, predicate: &impl Fn(&Json) -> bool) -> bool {
    match schema {
        Json::Object(object) => {
            if object
                .get("properties")
                .and_then(Json::as_object)
                .and_then(|properties| properties.get(name))
                .is_some_and(predicate)
            {
                return true;
            }
            object
                .values()
                .any(|value| schema_property_matches(value, name, predicate))
        }
        Json::Array(values) => values
            .iter()
            .any(|value| schema_property_matches(value, name, predicate)),
        _ => false,
    }
}

#[cfg(feature = "schema")]
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
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
        endpoints: vec![OpenTelemetryEndpointConfig {
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

#[test]
fn observability_filename_and_destination_helpers_reject_unsafe_values() {
    assert!(is_valid_atif_metadata_selector("metadata.session"));
    assert!(!is_valid_atif_metadata_selector("metadata../session"));
    assert_eq!(
        parse_atif_metadata_expression("session:-fallback").unwrap(),
        ("session", Some("fallback"))
    );
    assert!(parse_atif_metadata_expression("metadata.session|a|b").is_err());
    assert!(validate_atif_filename_template("trace-{session_id}.json").is_ok());
    assert!(validate_atif_filename_template("../escape-{session_id}.json").is_err());
    assert!(is_safe_atif_metadata_path("session-1"));
    assert!(!is_safe_atif_metadata_path("../session"));

    assert_eq!(normalize_opentelemetry_path("/v1/traces/"), "/v1/traces");
    assert_eq!(canonical_opentelemetry_host("localhost."), "<loopback>");
    assert_eq!(
        canonicalize_opentelemetry_destination("http://LOCALHOST:4318/v1/traces/").display,
        "http://<loopback>:4318/v1/traces"
    );
    assert_eq!(
        raw_opentelemetry_destination("collector:4317").display,
        "collector:4317"
    );
    assert_eq!(canonical_opentelemetry_host("127.0.0.1"), "<loopback>");
    assert_eq!(
        normalize_opentelemetry_path("//v1///traces//"),
        "/v1/traces"
    );

    let endpoints: Vec<OpenTelemetryEndpointConfig> = serde_json::from_value(json!([
        {"type": "full", "endpoint": "http://localhost:4318/v1/traces", "transport": "http_binary"},
        {"type": "gen_ai", "endpoint": "http://127.0.0.1:4318/v1/traces/", "transport": "http_binary"}
    ])).unwrap();
    assert!(validate_distinct_opentelemetry_destinations(&endpoints).is_err());
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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let config = plugin_config(json!({"enable_full_payloads": true}));
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .observability_full_payloads_enabled
    );

    clear_plugin_configuration().unwrap();
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

#[test]
fn version_three_rejects_removed_otlp_controls() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "opentelemetry": {
            "enabled": false,
            "mark_projection": "tool",
            "attribute_mappings": [],
            "endpoint": "http://localhost:4318/v1/traces"
        },
        "openinference": {
            "enabled": false,
            "endpoint": "http://localhost:4318/v1/traces"
        }
    })));
    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.legacy_opentelemetry_field"
            && diagnostic.field.as_deref() == Some("mark_projection")
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.legacy_opentelemetry_field"
            && diagnostic.field.as_deref() == Some("attribute_mappings")
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.legacy_opentelemetry_field"
            && diagnostic.field.as_deref() == Some("endpoint")
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.legacy_openinference_section"
            && diagnostic.field.as_deref() == Some("openinference")
    }));
}

#[test]
fn opentelemetry_endpoint_header_env_is_resolved_and_snapshotted() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let variable = "NEMO_RELAY_TEST_OTEL_HEADER_ENV";
    unsafe { std::env::set_var(variable, "Bearer initial") };
    let config = build_otel_config(
        0,
        OpenTelemetryEndpointConfig {
            otel_type: OpenTelemetryType::GenAi,
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
            header_env: HashMap::from([("authorization".to_string(), variable.to_string())]),
            resource_attributes: HashMap::new(),
            mark_projection: MarkProjection::default(),
            mark_exclude_names: default_mark_exclude_names(),
            attribute_mappings: Vec::new(),
            promote_metadata_prefixes: Vec::new(),
            promote_resource_metadata_prefixes: Vec::new(),
        },
    )
    .unwrap();
    unsafe { std::env::set_var(variable, "Bearer changed") };
    assert_eq!(config.header("authorization"), Some("Bearer initial"));
    unsafe { std::env::remove_var(variable) };
}

fn test_opentelemetry_endpoint() -> OpenTelemetryEndpointConfig {
    OpenTelemetryEndpointConfig {
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
        resource_attributes: HashMap::new(),
        mark_projection: MarkProjection::default(),
        mark_exclude_names: default_mark_exclude_names(),
        attribute_mappings: Vec::new(),
        promote_metadata_prefixes: Vec::new(),
        promote_resource_metadata_prefixes: Vec::new(),
    }
}

#[test]
fn trace_endpoint_completed_context_ttl_is_applied_and_must_be_positive() {
    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.completed_span_context_ttl_millis = Some(750);
    let config = build_otel_config(0, endpoint).unwrap();
    assert_eq!(
        config.completed_span_context_ttl(),
        Duration::from_millis(750)
    );

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.completed_span_context_ttl_millis = Some(0);
    assert!(
        build_otel_config(0, endpoint)
            .unwrap_err()
            .to_string()
            .contains("completed_span_context_ttl_millis must be greater than 0")
    );
}

fn test_signal_endpoint() -> OpenTelemetrySignalEndpointConfig {
    OpenTelemetrySignalEndpointConfig {
        endpoint: "http://localhost:4318/v1/logs".to_string(),
        transport: default_otlp_transport(),
        headers: HashMap::new(),
        header_env: HashMap::new(),
        resource_attributes: HashMap::new(),
        service_name: default_otel_service_name(),
        service_namespace: None,
        service_version: None,
        instrumentation_scope: default_otel_instrumentation_scope(),
        timeout_millis: default_timeout_millis(),
    }
}

#[test]
fn signal_header_activation_rejects_blank_and_padded_inline_headers() {
    for (key, value) in [
        ("", "token"),
        (" authorization", "token"),
        ("authorization", ""),
        ("authorization", " token "),
    ] {
        let mut endpoint = test_signal_endpoint();
        endpoint.headers.insert(key.to_string(), value.to_string());
        let error = resolve_signal_headers("logs", 2, &endpoint).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("OpenTelemetry logs.endpoints[2] has invalid headers")
        );
    }
}

#[test]
fn build_otel_config_rejects_each_activation_only_invalid_value() {
    let _guard = crate::observability::test_mutex().lock().unwrap();

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.endpoint = "  ".to_string();
    assert!(build_otel_config(0, endpoint).is_err());

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.transport = "udp".to_string();
    assert!(build_otel_config(0, endpoint).is_err());

    for variable in ["", " PADDED_ENV "] {
        let mut endpoint = test_opentelemetry_endpoint();
        endpoint
            .header_env
            .insert("authorization".to_string(), variable.to_string());
        assert!(build_otel_config(1, endpoint).is_err());
    }

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint
        .headers
        .insert("Authorization".to_string(), "inline".to_string());
    endpoint
        .header_env
        .insert("authorization".to_string(), "TOKEN".to_string());
    assert!(build_otel_config(2, endpoint).is_err());

    let variable = "NEMO_RELAY_TEST_BLANK_OTEL_HEADER_ENV";
    unsafe { std::env::set_var(variable, "  ") };
    let mut endpoint = test_opentelemetry_endpoint();
    endpoint
        .header_env
        .insert("authorization".to_string(), variable.to_string());
    assert!(build_otel_config(3, endpoint).is_err());
    unsafe { std::env::remove_var(variable) };

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_queue_size = Some(0);
    assert!(build_otel_config(4, endpoint).is_err());

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_export_batch_size = Some(0);
    assert!(build_otel_config(4, endpoint).is_err());

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.scheduled_delay_millis = Some(0);
    assert!(build_otel_config(4, endpoint).is_err());

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_queue_size = Some(8);
    endpoint.max_export_batch_size = Some(9);
    assert!(build_otel_config(4, endpoint).is_err());
}

#[test]
fn build_otel_config_carries_endpoint_batch_overrides() {
    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_queue_size = Some(4096);
    endpoint.max_export_batch_size = Some(256);
    endpoint.scheduled_delay_millis = Some(750);
    let config = build_otel_config(0, endpoint).unwrap();
    assert_eq!(
        config.batch_overrides(),
        (Some(4096), Some(256), Some(Duration::from_millis(750)))
    );

    let config = build_otel_config(1, test_opentelemetry_endpoint()).unwrap();
    assert_eq!(config.batch_overrides(), (None, None, None));
}

#[test]
fn build_otel_config_carries_resource_metadata_promotion_prefixes() {
    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.promote_resource_metadata_prefixes = vec!["deployment.".to_string()];

    let config = build_otel_config(0, endpoint).unwrap();

    assert_eq!(
        config.promote_resource_metadata_prefixes(),
        ["deployment.".to_string()]
    );
}

#[test]
fn validate_opentelemetry_section_reports_empty_and_malformed_endpoints() {
    let policy = ConfigPolicy::default();
    let mut diagnostics = Vec::new();
    validate_opentelemetry_section(
        &mut diagnostics,
        &policy,
        &OpenTelemetrySectionConfig {
            enabled: true,
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
    );
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints")
            && diagnostic.message.contains("at least one endpoint")
    }));

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.endpoint = " ".to_string();
    endpoint.transport = "udp".to_string();
    endpoint
        .header_env
        .insert("empty".to_string(), String::new());
    endpoint
        .header_env
        .insert("padded".to_string(), " TOKEN ".to_string());
    diagnostics.clear();
    validate_opentelemetry_section(
        &mut diagnostics,
        &policy,
        &OpenTelemetrySectionConfig {
            enabled: true,
            endpoints: vec![endpoint],
            logs: None,
            metrics: None,
        },
    );
    for field in [
        "endpoints[0].endpoint",
        "endpoints[0].transport",
        "endpoints[0].header_env.empty",
        "endpoints[0].header_env.padded",
    ] {
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.field.as_deref() == Some(field)),
            "missing diagnostic for {field}: {diagnostics:?}"
        );
    }

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_queue_size = Some(0);
    endpoint.max_export_batch_size = Some(2);
    endpoint.scheduled_delay_millis = Some(0);
    diagnostics.clear();
    validate_opentelemetry_section(
        &mut diagnostics,
        &policy,
        &OpenTelemetrySectionConfig {
            enabled: true,
            endpoints: vec![endpoint],
            logs: None,
            metrics: None,
        },
    );
    for field in [
        "endpoints[0].max_queue_size",
        "endpoints[0].max_export_batch_size",
        "endpoints[0].scheduled_delay_millis",
    ] {
        assert!(
            diagnostics.iter().any(|diagnostic| {
                diagnostic.field.as_deref() == Some(field) && diagnostic.message.contains(field)
            }),
            "missing diagnostic for {field}: {diagnostics:?}"
        );
    }
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[0].max_export_batch_size")
            && diagnostic
                .message
                .contains("endpoints[0].max_export_batch_size")
            && diagnostic
                .message
                .contains("less than or equal to max_queue_size")
    }));
}

#[test]
fn invalid_batch_config_identifies_the_endpoint_during_activation() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "version": 3,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "http://jaeger-local:4318/v1/traces",
                    "max_queue_size": 256
                },
                {
                    "type": "full",
                    "endpoint": "http://tempo-prod:4319/v1/traces",
                    "max_queue_size": 0
                },
                {
                    "type": "full",
                    "endpoint": "http://compliance:4320/v1/traces",
                    "max_queue_size": 128
                }
            ]
        }
    }));

    let report = validate_plugin_config(&config);
    assert!(!report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[1].max_queue_size")
            && diagnostic.level == DiagnosticLevel::Warning
            && diagnostic.code == "observability.invalid_otel_endpoint"
            && diagnostic
                .message
                .contains("OpenTelemetry endpoints[1].max_queue_size must be greater than 0")
    }));

    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    let report = crate::plugin::active_plugin_report().unwrap();
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[1].max_queue_size")
            && diagnostic.level == DiagnosticLevel::Warning
            && diagnostic.code == "observability.invalid_otel_endpoint"
    }));
    clear_plugin_configuration().unwrap();
}

#[test]
fn all_invalid_trace_batch_configs_still_block_activation() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let config = plugin_config(json!({
        "version": 3,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://jaeger-local:4318/v1/traces",
                "max_queue_size": 0
            }]
        }
    }));

    assert!(!validate_plugin_config(&config).has_errors());
    let error = futures::executor::block_on(initialize_plugins_exact(config)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires at least one valid trace, log, or metric endpoint")
    );
    assert!(crate::plugin::active_plugin_report().is_none());
}

#[test]
fn opentelemetry_registration_rejects_an_empty_endpoint_list() {
    let mut context = PluginRegistrationContext::new();
    let error = register_opentelemetry(
        OpenTelemetrySectionConfig {
            enabled: true,
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
        &mut context,
    )
    .unwrap_err();
    assert!(error.to_string().contains("at least one endpoint"));
}

#[test]
fn atof_stream_header_validation_reports_invalid_values_and_environment_names() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let policy = ConfigPolicy::default();
    let mut diagnostics = Vec::new();

    validate_atof_stream_header_env(&mut diagnostics, &policy, "header_env.empty", "");
    validate_atof_stream_header_env(
        &mut diagnostics,
        &policy,
        "header_env.padded",
        " PADDED_ENV ",
    );

    let blank = "NEMO_RELAY_TEST_BLANK_ATOF_STREAM_HEADER_ENV";
    // SAFETY: the observability mutex serializes access to this test-only variable.
    unsafe { std::env::set_var(blank, "  ") };
    validate_atof_stream_header_env(&mut diagnostics, &policy, "header_env.blank", blank);
    // SAFETY: cleanup of the test-only environment variable.
    unsafe { std::env::remove_var(blank) };

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("non-empty environment variable")
    }));
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("surrounding whitespace"))
    );
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("environment variable that is blank")
    }));

    #[cfg(feature = "atof-streaming")]
    {
        diagnostics.clear();
        validate_atof_stream_header(
            &mut diagnostics,
            &policy,
            "headers.invalid",
            "x-test",
            "invalid\nvalue",
        );
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("value is invalid"))
        );
    }
}

#[test]
fn atif_dispatcher_surfaces_fatal_and_runtime_failure_states() {
    let mut dispatcher = AtifDispatcher::new(AtifSectionConfig::default());
    assert_eq!(dispatcher.sink_targets(), vec![SinkLabel::Local]);
    dispatcher.fatal_error = Some("fatal export failure".into());
    assert!(
        dispatcher
            .last_error_result()
            .unwrap_err()
            .to_string()
            .contains("fatal export failure")
    );

    dispatcher.fatal_error = None;
    dispatcher.record_runtime_failure(
        "atif.local_fallback_failed",
        Some("output_directory".into()),
        "local write failed".into(),
        Some("session".into()),
    );
    assert_eq!(dispatcher.sink_targets(), vec![SinkLabel::Local]);
    assert!(dispatcher.last_error_result().is_err());
}

#[test]
fn opentelemetry_endpoint_header_env_rejects_missing_and_duplicate_headers() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let variable = "NEMO_RELAY_TEST_MISSING_OTEL_HEADER_ENV";
    unsafe { std::env::remove_var(variable) };
    let report = validate_plugin_config(&plugin_config(json!({
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
    let activation = futures::executor::block_on(initialize_plugins_exact(plugin_config(json!({
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

    futures::executor::block_on(initialize_plugins_exact(plugin_config(json!({
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
    clear_plugin_configuration().unwrap();
}

#[cfg(unix)]
#[test]
fn opentelemetry_header_env_non_unicode_errors_do_not_expose_values() {
    use std::os::unix::ffi::OsStringExt;

    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let variable = "NEMO_RELAY_TEST_NON_UNICODE_OTEL_HEADER_ENV";
    let secret = "relay-plugin-secret";
    let mut value = vec![0xff];
    value.extend_from_slice(secret.as_bytes());
    unsafe { std::env::set_var(variable, std::ffi::OsString::from_vec(value)) };

    let trace: OpenTelemetryEndpointConfig = serde_json::from_value(json!({
        "type": "full",
        "endpoint": "http://localhost:4318/v1/traces",
        "header_env": {"authorization": variable}
    }))
    .unwrap();
    let trace_error = build_otel_config(0, trace).unwrap_err().to_string();
    assert!(trace_error.contains(variable));
    assert!(!trace_error.contains(secret));

    let logs: OpenTelemetrySignalEndpointConfig = serde_json::from_value(json!({
        "endpoint": "http://localhost:4318/v1/logs",
        "header_env": {"authorization": variable}
    }))
    .unwrap();
    let log_error = resolve_signal_headers("logs", 0, &logs)
        .unwrap_err()
        .to_string();
    assert!(log_error.contains(variable));
    assert!(!log_error.contains(secret));

    unsafe { std::env::remove_var(variable) };
}

#[test]
fn invalid_log_endpoint_keeps_valid_signal_subscriber() {
    let _guard = crate::observability::test_mutex()
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

    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    clear_plugin_configuration().unwrap();
}

#[test]
fn invalid_metric_endpoint_keeps_valid_signal_subscriber() {
    let _guard = crate::observability::test_mutex()
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

    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    clear_plugin_configuration().unwrap();
}

#[test]
fn malformed_derived_signal_endpoint_keeps_valid_peers() {
    let _guard = crate::observability::test_mutex()
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

    let report = validate_plugin_config(&config);
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

    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    clear_plugin_configuration().unwrap();
}

#[test]
fn skipped_trace_endpoint_preserves_configured_indexes() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let first = test_opentelemetry_endpoint();
    let mut invalid = test_opentelemetry_endpoint();
    invalid.max_queue_size = Some(0);
    let third = test_opentelemetry_endpoint();

    let subscribers = build_opentelemetry_subscribers(vec![first, invalid, third]).unwrap();
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
    let _guard = crate::observability::test_mutex()
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

    let error = futures::executor::block_on(initialize_plugins_exact(config)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires at least one valid trace, log, or metric endpoint")
    );
    assert!(crate::plugin::active_plugin_report().is_none());
}

#[test]
fn outer_disabled_component_does_not_resolve_opentelemetry_header_env() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn opentelemetry_endpoint_accepts_legacy_projection_controls_and_rejects_unknown_fields() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "error"},
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://localhost:4318/v1/traces",
                "header_en": {"authorization": "TOKEN"},
                "mark_projection": "tool",
                "mark_exclude_names": ["notification"],
                "attribute_mappings": [{"key": "nemo_relay.model_name", "alias": "model.alias"}],
                "promote_metadata_prefixes": ["nv."],
                "max_queue_size": 4096,
                "max_export_batch_size": 256,
                "scheduled_delay_millis": 750,
                "capture_content": true
            }]
        }
    })));

    for field in ["endpoints[0].header_en", "endpoints[0].capture_content"] {
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.field.as_deref() == Some(field)),
            "missing endpoint diagnostic for {field}: {:?}",
            report.diagnostics
        );
    }
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[0].header_en")
            && diagnostic.code == "observability.unknown_field"
    }));
    assert!(!report.diagnostics.iter().any(|diagnostic| {
        matches!(
            diagnostic.field.as_deref(),
            Some("endpoints[0].mark_projection")
                | Some("endpoints[0].mark_exclude_names")
                | Some("endpoints[0].attribute_mappings")
                | Some("endpoints[0].promote_metadata_prefixes")
                | Some("endpoints[0].max_queue_size")
                | Some("endpoints[0].max_export_batch_size")
                | Some("endpoints[0].scheduled_delay_millis")
        )
    }));
}

#[test]
fn opentelemetry_endpoint_rejects_invalid_attribute_mappings() {
    let _guard = crate::observability::test_mutex()
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
        let report = validate_plugin_config(&config);
        assert!(report.has_errors());
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "observability.unsupported_value"
                && diagnostic.field.as_deref() == Some("endpoints[0].attribute_mappings")
                && diagnostic.message.contains(expected_message)
        }));
        assert!(futures::executor::block_on(initialize_plugins_exact(config)).is_err());
    }
}

#[test]
fn opentelemetry_endpoint_accepts_valid_attribute_mappings() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://localhost:4318/v1/traces",
                "attribute_mappings": [{
                    "key": "nemo_relay.model_name",
                    "alias": "model.alias"
                }]
            }]
        }
    })));

    assert!(
        !report.has_errors(),
        "valid attribute mapping produced diagnostics: {:?}",
        report.diagnostics
    );
}

#[test]
fn opentelemetry_endpoint_rejects_glob_metadata_promotion_prefix() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://localhost:4318/v1/traces",
                "promote_metadata_prefixes": ["nv.*"]
            }]
        }
    })));

    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.unsupported_value"
            && diagnostic.field.as_deref() == Some("endpoints[0].promote_metadata_prefixes")
            && diagnostic.message.contains("literal prefix, not a glob")
    }));
}

#[test]
fn opentelemetry_endpoint_rejects_invalid_and_case_duplicate_headers() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "http://localhost:4318/v1/traces",
                    "headers": {
                        "bad header": "value",
                        "x-control": "line one\nline two",
                        "Authorization": "first",
                        "authorization": "second"
                    }
                },
                {
                    "type": "gen_ai",
                    "endpoint": "http://localhost:4318/v1/traces",
                    "header_env": {
                        "X-Api-Key": "PATH",
                        "x-api-key": "PATH"
                    }
                }
            ]
        }
    })));

    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[0].headers.bad header")
            && diagnostic.message.contains("header name")
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("endpoints[0].headers.x-control")
            && diagnostic.message.contains("invalid value")
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("endpoints[0].headers contains duplicate header")
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("endpoints[1].header_env contains duplicate header")
    }));
}

#[test]
fn disabled_opentelemetry_does_not_resolve_header_env() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let variable = "NEMO_RELAY_TEST_DISABLED_OTEL_HEADER_ENV";
    unsafe { std::env::remove_var(variable) };

    let report = validate_plugin_config(&plugin_config(json!({
        "opentelemetry": {
            "enabled": false,
            "endpoints": [{
                "type": "full",
                "endpoint": "http://localhost:4318/v1/traces",
                "header_env": {"authorization": variable}
            }]
        }
    })));

    assert!(
        !report.has_errors(),
        "disabled endpoint unexpectedly resolved header_env: {:?}",
        report.diagnostics
    );
}

#[cfg(feature = "schema")]
#[test]
fn schema_contains_every_supported_observability_option() {
    let schema = observability_config_schema();
    for field in [
        "version",
        "atof",
        "atif",
        "opentelemetry",
        "logs",
        "metrics",
        "policy",
        "enabled",
        "output_directory",
        "filename",
        "mode",
        "name",
        "sinks",
        "endpoints",
        "type",
        "url",
        "field_name_policy",
        "header_env",
        "agent_name",
        "agent_version",
        "model_name",
        "tool_definitions",
        "extra",
        "filename_template",
        "transport",
        "endpoint",
        "headers",
        "resource_attributes",
        "service_name",
        "service_namespace",
        "service_version",
        "instrumentation_scope",
        "timeout_millis",
        "minimum_severity",
        "max_queue_size",
        "max_export_batch_size",
        "scheduled_delay_millis",
        "export_interval_millis",
        "temporality",
        "max_instruments",
        "cardinality_limit",
        "unknown_component",
        "unknown_field",
        "unsupported_value",
    ] {
        assert!(
            schema_has_property(&schema, field),
            "schema missing property `{field}`:\n{}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
    }
    assert!(schema_property_has_enum(
        &schema,
        "mode",
        &["append", "overwrite"]
    ));
    assert!(schema_property_has_enum(
        &schema,
        "transport",
        &["http_binary", "grpc"]
    ));
    assert!(schema_property_has_default(
        &schema,
        "mode",
        json!("append")
    ));
    assert!(schema_property_has_default(
        &schema,
        "transport",
        json!("http_binary")
    ));
    assert!(schema_property_has_enum(
        &schema,
        "minimum_severity",
        &["trace", "debug", "info", "warn", "warning", "error"]
    ));
    assert!(schema_property_has_enum(
        &schema,
        "temporality",
        &["cumulative", "delta", "low_memory"]
    ));
    assert!(schema_property_has_default(&schema, "version", json!(4)));
    assert!(schema_property_has_default(
        &schema,
        "minimum_severity",
        json!("info")
    ));
    assert!(schema_property_has_default(
        &schema,
        "temporality",
        json!("cumulative")
    ));
}

#[cfg(feature = "schema")]
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

#[test]
fn built_in_registration_is_automatic() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    assert!(list_plugin_kinds().contains(&OBSERVABILITY_PLUGIN_KIND.to_string()));
    assert!(lookup_plugin(OBSERVABILITY_PLUGIN_KIND).is_some());

    let config = plugin_config(json!({}));
    assert!(!validate_plugin_config(&config).has_errors());
}

#[test]
fn explicit_registration_helpers_are_idempotent_and_reversible() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    assert!(register_observability_component().is_ok());
    assert!(register_observability_component().is_ok());
    assert!(deregister_observability_component());
    assert!(!deregister_observability_component());
    register_observability_component().unwrap();
}

#[test]
fn empty_and_disabled_config_register_nothing() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let config = plugin_config(json!({
        "atof": {"enabled": false},
        "atif": {"enabled": false},
        "opentelemetry": {"enabled": false, "traces": []}
    }));
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let state = global_context();
    assert!(state.read().unwrap().event_subscribers.is_empty());
}

#[test]
fn disabled_file_sections_do_not_create_files() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let agent = push_agent("disabled-agent");
    pop(&agent);
    clear_plugin_configuration().unwrap();

    assert!(!dir.join("events.jsonl").exists());
    assert!(!dir.join(format!("trajectory-{}.json", agent.uuid)).exists());
}

#[test]
fn duplicate_component_is_rejected_as_singleton() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let config = PluginConfig {
        version: 1,
        components: vec![component(json!({})), component(json!({}))],
        policy: Default::default(),
    };
    let report = validate_plugin_config(&config);
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.duplicate_component")
    );
}

#[test]
fn unknown_fields_and_bad_values_follow_policy() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let warn_report = validate_plugin_config(&plugin_config(json!({
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

    let ignore_report = validate_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "ignore", "unsupported_value": "ignore"},
        "atof": {"bogus": true, "sinks": [{"type": "file", "mode": "invalid"}]},
        "atif": {"filename_template": "missing-session"}
    })));
    assert!(!ignore_report.has_errors());
    assert!(ignore_report.diagnostics.is_empty());
}

#[test]
fn atif_filename_template_syntax_is_rejected_before_activation() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let valid_report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "filename_template": "{metadata.workflow_id:-unassigned}/trajectory-{session_id}.json"
        }
    })));
    assert!(!valid_report.has_errors());

    let malformed = "trajectory-{session_id}.json/{metadata.tenant";
    let invalid_report = validate_plugin_config(&plugin_config(json!({
        "atif": {"filename_template": malformed}
    })));
    assert!(invalid_report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("filename_template")
            && diag.message.contains("unclosed metadata placeholder")
    }));

    let error = futures::executor::block_on(initialize_plugins_exact(plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "atif": {"enabled": true, "filename_template": malformed}
    }))))
    .unwrap_err();
    assert!(error.to_string().contains("unclosed metadata placeholder"));
}

#[test]
fn invalid_shapes_and_strict_policy_are_reported() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let invalid_shape = validate_plugin_config(&plugin_config(json!({
        "version": "one",
    })));
    assert!(invalid_shape.has_errors());
    assert!(
        invalid_shape
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.invalid_plugin_config")
    );

    let unsupported_version = validate_plugin_config(&plugin_config(json!({
        "version": 1,
    })));
    assert!(unsupported_version.has_errors());
    assert!(unsupported_version.diagnostics.iter().any(|diag| diag.code
        == "observability.unsupported_config_version"
        && diag.field.as_deref() == Some("version")));

    let strict_unknown = validate_plugin_config(&plugin_config(json!({
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

    let strict_bad_transport = validate_plugin_config(&plugin_config(json!({
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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
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
    #[cfg(feature = "atof-streaming")]
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
    #[cfg(feature = "atof-streaming")]
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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
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
fn build_atof_sink_config_maps_headers_timeout_and_rejects_transport() {
    let mut headers = std::collections::HashMap::new();
    headers.insert("authorization".to_string(), "token".to_string());
    let config = build_atof_sink_config(
        2,
        AtofSinkSectionConfig::Stream(AtofStreamSinkSectionConfig {
            name: Some("switchyard".into()),
            url: "ws://127.0.0.1:47632/events".into(),
            transport: "websocket".into(),
            headers: headers.clone(),
            header_env: std::collections::HashMap::from([(
                "x-api-key".into(),
                "SWITCHYARD_API_KEY".into(),
            )]),
            timeout_millis: 123,
            field_name_policy: "replace_dots".into(),
        }),
    )
    .unwrap();

    let CoreAtofSinkConfig::Stream(config) = config else {
        panic!("expected stream sink")
    };
    assert_eq!(config.url, "ws://127.0.0.1:47632/events");
    assert_eq!(
        config.transport,
        crate::observability::atof::AtofEndpointTransport::Websocket
    );
    assert_eq!(config.headers, headers);
    assert_eq!(
        config.header_env.get("x-api-key").map(String::as_str),
        Some("SWITCHYARD_API_KEY")
    );
    assert_eq!(config.timeout_millis, 123);
    assert_eq!(
        config.field_name_policy,
        crate::observability::atof::AtofEndpointFieldNamePolicy::ReplaceDots
    );

    let error = build_atof_sink_config(
        3,
        AtofSinkSectionConfig::Stream(AtofStreamSinkSectionConfig {
            name: None,
            url: "http://127.0.0.1:47632/events".into(),
            transport: "smtp".into(),
            headers: std::collections::HashMap::new(),
            header_env: std::collections::HashMap::new(),
            timeout_millis: 3_000,
            field_name_policy: "preserve".into(),
        }),
    )
    .unwrap_err();
    assert!(error.to_string().contains("sinks[3].transport"));

    let error = build_atof_sink_config(
        4,
        AtofSinkSectionConfig::Stream(AtofStreamSinkSectionConfig {
            name: None,
            url: "http://127.0.0.1:47632/events".into(),
            transport: "http_post".into(),
            headers: std::collections::HashMap::new(),
            header_env: std::collections::HashMap::new(),
            timeout_millis: 3_000,
            field_name_policy: "bogus".into(),
        }),
    )
    .unwrap_err();
    assert!(error.to_string().contains("sinks[4].field_name_policy"));
}

#[test]
fn initialization_fails_for_invalid_enabled_file_exporters() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    let error = futures::executor::block_on(initialize_plugins_exact(invalid_atof)).unwrap_err();
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
        futures::executor::block_on(initialize_plugins_exact(invalid_atif_template)).unwrap_err();
    assert!(error.to_string().contains("filename_template"));

    let invalid_path = plugin_config(json!({
        "atof": {
            "enabled": true,
            "sinks": [{"type": "file", "output_directory": not_a_directory, "filename": "events.jsonl"}]
        }
    }));
    let error = futures::executor::block_on(initialize_plugins_exact(invalid_path)).unwrap_err();
    assert!(error.to_string().contains("registration failed"));

    let invalid_otel_transport = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{"type": "full", "endpoint": "http://localhost:4318/v1/traces", "transport": "udp"}]
        }
    }));
    let error =
        futures::executor::block_on(initialize_plugins_exact(invalid_otel_transport)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires at least one valid trace, log, or metric endpoint")
    );
}

#[test]
fn atof_enabled_writes_jsonl_and_teardown_flushes() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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
    crate::api::scope::event(
        crate::api::scope::EmitMarkEventParams::builder()
            .name("checkpoint")
            .parent(&agent)
            .data(json!({"step": 1}))
            .build(),
    )
    .unwrap();
    pop(&agent);
    clear_plugin_configuration().unwrap();

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
#[cfg(feature = "atof-streaming")]
fn atof_stream_sinks_fan_out_and_teardown_all_workers() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let agent = push_agent("atof-stream-agent");
    crate::api::scope::event(
        crate::api::scope::EmitMarkEventParams::builder()
            .name("checkpoint")
            .parent(&agent)
            .data(json!({"step": 1}))
            .build(),
    )
    .unwrap();
    pop(&agent);
    clear_plugin_configuration().unwrap();

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
#[cfg(feature = "atof-streaming")]
fn atof_stream_sink_header_env_is_snapshotted_at_activation() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    // SAFETY: The active endpoint must use the header value captured at activation.
    unsafe { std::env::remove_var(&variable) };

    let agent = push_agent("atof-header-env-agent");
    crate::api::scope::event(
        crate::api::scope::EmitMarkEventParams::builder()
            .name("checkpoint")
            .parent(&agent)
            .data(json!({"step": 1}))
            .build(),
    )
    .unwrap();
    pop(&agent);
    clear_plugin_configuration().unwrap();

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
#[cfg(all(feature = "atof-streaming", feature = "object-store"))]
fn atif_remote_storage_validates_s3_configuration_and_http_access_outcomes() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    let storage = build_atif_storage(3, &s3).expect("S3 client configuration should resolve");
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
                timeout_millis: 5_000,
            }),
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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-defaults");

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir
        }
    }));
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let first = push_agent("first-agent");
    let nested = push_agent("nested-agent");
    pop(&nested);
    pop(&first);

    let second = push_agent("second-agent");
    pop(&second);
    clear_plugin_configuration().unwrap();

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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let _restore_guard = RestoreThreadScopeStackGuard(capture_thread_scope_stack());
    let dir = temp_dir("observability-atif-propagation-root");
    let request_id = Uuid::parse_str("018f47a4-3af7-7d94-8e61-9f0f89b5d312").unwrap();
    let parent_id = Uuid::now_v7();
    let stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(request_id),
        parent_uuid: parent_id,
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let first = push_agent("first-propagated-agent");
    pop(&first);
    let second = push_agent("second-propagated-agent");
    pop(&second);
    clear_plugin_configuration().unwrap();

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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-metadata-template");

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "{metadata.routing.artifact_path}/trajectory-{session_id}.json"
        }
    }));
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let sanitized = crate::api::scope::push_scope(
        PushScopeParams::builder()
            .name("sanitized-metadata-path-agent")
            .scope_type(ScopeType::Agent)
            .metadata(json!({"routing": {"artifact_path": "../escape"}}))
            .build(),
    )
    .unwrap();
    pop(&sanitized);

    let valid = crate::api::scope::push_scope(
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

    clear_plugin_configuration().unwrap();

    futures::executor::block_on(initialize_plugins_exact(plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }))))
    .expect("ATIF teardown errors should not block later activation");
    clear_plugin_configuration().expect("later ATIF teardown should succeed");
}

#[test]
fn atif_routes_global_descendant_events_by_parent_uuid() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-global-descendant");
    let root_uuid = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("root-agent");
    let agent_uuid = agent.uuid;
    let child_uuid = Uuid::now_v7();
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(AtifSectionConfig {
        enabled: true,
        output_directory: Some(dir.clone()),
        ..AtifSectionConfig::default()
    })));
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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-openclaw-child-fallback");
    let root_uuid = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let child_uuid = Uuid::now_v7();
    let child_mark_uuid = Uuid::now_v7();
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(AtifSectionConfig {
        enabled: true,
        output_directory: Some(dir.clone()),
        ..AtifSectionConfig::default()
    })));
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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-evict");
    let root_uuid = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("evicted-agent");
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(AtifSectionConfig {
        enabled: true,
        output_directory: Some(dir.clone()),
        ..AtifSectionConfig::default()
    })));

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
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-write-error");
    let root_uuid = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("failed-write-agent");
    let dispatcher = Arc::new(Mutex::new(AtifDispatcher::new(AtifSectionConfig {
        enabled: true,
        output_directory: Some(dir),
        ..AtifSectionConfig::default()
    })));

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
fn write_atif_reports_missing_local_path_and_unregistered_remote_sink() {
    let agent_uuid = Uuid::now_v7();
    let write = PendingAtifWrite {
        agent_uuid,
        session_id: agent_uuid.to_string(),
        filename: "trajectory.json".into(),
        local_root: None,
        local_path: None,
        payload: b"{}".to_vec(),
    };

    let results = write_atif(&write, &[], &[SinkLabel::Local, SinkLabel::Remote(0)]);

    assert_eq!(results.len(), 2);
    assert!(
        results[0]
            .1
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("no output path")
    );
    let remote_error = results[1].1.as_ref().unwrap_err().to_string();
    #[cfg(feature = "object-store")]
    assert!(remote_error.contains("storage[0]"));
    #[cfg(not(feature = "object-store"))]
    assert!(remote_error.contains("ATIF storage support is not enabled in this build"));
}

#[test]
fn write_atif_spills_to_local_when_all_remote_sinks_fail() {
    let dir = temp_dir("observability-atif-remote-fallback");
    let path = dir.join("trajectory.json");
    let agent_uuid = Uuid::now_v7();
    let write = PendingAtifWrite {
        agent_uuid,
        session_id: agent_uuid.to_string(),
        filename: "trajectory.json".into(),
        local_root: Some(dir.clone()),
        local_path: Some(path.clone()),
        payload: b"{}".to_vec(),
    };

    let results = write_atif(&write, &[], &[SinkLabel::Remote(0)]);

    assert_eq!(results.len(), 2);
    assert!(results[0].1.is_err());
    assert_eq!(results[1].0, SinkLabel::Local);
    assert!(results[1].1.is_ok());
    assert_eq!(fs::read(path).unwrap(), b"{}");
}

#[test]
fn atif_dispatcher_default_output_path_uses_current_directory() {
    let dispatcher = AtifDispatcher::new(AtifSectionConfig::default());
    let (filename, local_root, local_path) =
        dispatcher.prepare_destination("session-1", None).unwrap();
    assert_eq!(filename, "nemo-relay-atif-session-1.json");
    assert_eq!(local_root.unwrap(), std::env::current_dir().unwrap());
    assert_eq!(
        local_path.unwrap(),
        std::env::current_dir()
            .unwrap()
            .join("nemo-relay-atif-session-1.json")
    );
}

#[test]
fn atif_session_id_cannot_escape_the_output_directory() {
    let dispatcher = AtifDispatcher::new(AtifSectionConfig::default());
    for session_id in ["../escape", "/absolute", "nested/../../escape"] {
        assert!(
            dispatcher.prepare_destination(session_id, None).is_err(),
            "unsafe session id should be rejected: {session_id:?}"
        );
    }
}

#[cfg(unix)]
#[test]
fn atif_local_write_replaces_regular_files_privately_and_rejects_symlinks() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().canonicalize().unwrap();
    let path = dir_path.join("trajectory.json");
    write_atif_local(&dir_path, &path, b"first").unwrap();
    write_atif_local(&dir_path, &path, b"second").unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"second");
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );

    let target = dir_path.join("target.json");
    fs::write(&target, b"preserve").unwrap();
    let link = dir_path.join("link.json");
    symlink(&target, &link).unwrap();
    assert!(write_atif_local(&dir_path, &link, b"overwrite").is_err());
    assert_eq!(fs::read(target).unwrap(), b"preserve");

    let outside = tempfile::tempdir().unwrap();
    let outside_path = outside.path().canonicalize().unwrap();
    let linked_parent = dir_path.join("linked-parent");
    symlink(&outside_path, &linked_parent).unwrap();
    assert!(write_atif_local(&dir_path, &linked_parent.join("escaped.json"), b"escape").is_err());
    assert!(!outside_path.join("escaped.json").exists());
}

#[cfg(unix)]
#[test]
fn atif_local_write_resolves_symlinked_configured_root() {
    use std::os::unix::fs::symlink;

    let parent = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let parent_path = parent.path().canonicalize().unwrap();
    let outside_path = outside.path().canonicalize().unwrap();
    let link = parent_path.join("linked");
    symlink(&outside_path, &link).unwrap();
    let root = link.join("atif");
    let path = root.join("trajectory.json");

    write_atif_local(&root, &path, b"trajectory").unwrap();
    assert_eq!(
        fs::read(outside_path.join("atif/trajectory.json")).unwrap(),
        b"trajectory"
    );
}

#[test]
fn atif_metadata_template_values_are_sanitized() {
    assert_eq!(sanitize_atif_metadata_fragment("tenant-a"), "tenant-a");
    assert_eq!(sanitize_atif_metadata_fragment("../tenant/a"), "-tenant-a");
    assert_eq!(
        sanitize_atif_metadata_fragment(r"tenant\project"),
        "tenant-project"
    );
    assert_eq!(
        sanitize_atif_metadata_fragment("tenant / : project"),
        "tenant-project"
    );
    assert_eq!(sanitize_atif_metadata_fragment("run...name"), "run-name");
    assert_eq!(sanitize_atif_metadata_fragment("ténant///项目"), "t-nant-");
    assert_eq!(sanitize_atif_metadata_fragment("."), "-");
    assert_eq!(sanitize_atif_metadata_fragment(".."), "-");
}

#[test]
fn atif_metadata_template_values_must_be_safe_path_fragments() {
    assert!(
        validate_atif_filename_template(
            "{metadata.workflow_id:-unassigned}/trajectory-{session_id}.json"
        )
        .is_ok()
    );
    assert_eq!(
        render_atif_filename(
            "{metadata.workflow_id:-unassigned}/trajectory-{session_id}.json",
            "scope-id",
            None
        )
        .unwrap(),
        "unassigned/trajectory-scope-id.json"
    );

    assert!(is_safe_atif_metadata_path(
        "tenant-a/team_1.session-123~retry"
    ));

    for value in [
        "",
        "/absolute",
        "trailing/",
        "double//slash",
        ".",
        "../escape",
        "tenant/../escape",
        r"tenant\session",
        "tenant name",
        "tenant:session",
    ] {
        assert!(
            !is_safe_atif_metadata_path(value),
            "metadata path should be rejected: {value:?}"
        );
    }

    let dispatcher = AtifDispatcher::new(AtifSectionConfig {
        filename_template: "{metadata.artifact_path}/trajectory-{session_id}.json".to_string(),
        ..AtifSectionConfig::default()
    });
    assert!(dispatcher.prepare_destination("session-1", None).is_err());
    let non_string = json!({"artifact_path": 123});
    assert!(
        dispatcher
            .prepare_destination("session-1", Some(&non_string))
            .is_err()
    );
    let dispatcher_with_fallback = AtifDispatcher::new(AtifSectionConfig {
        filename_template: "{metadata.artifact_path:-unassigned}/trajectory-{session_id}.json"
            .to_string(),
        ..AtifSectionConfig::default()
    });
    let error = dispatcher_with_fallback
        .prepare_destination("session-1", Some(&non_string))
        .unwrap_err();
    assert!(error.contains("resolved to a non-string value"), "{error}");
    let nested_non_string = json!({"artifact": 123});
    let nested_dispatcher = AtifDispatcher::new(AtifSectionConfig {
        filename_template: "{metadata.artifact.path:-unassigned}/trajectory-{session_id}.json"
            .to_string(),
        ..AtifSectionConfig::default()
    });
    let error = nested_dispatcher
        .prepare_destination("session-1", Some(&nested_non_string))
        .unwrap_err();
    assert!(error.contains("traversed a non-object value"), "{error}");
    let nested_null = json!({"artifact": null});
    let destination = nested_dispatcher
        .prepare_destination("session-1", Some(&nested_null))
        .unwrap();
    assert_eq!(destination.0, "unassigned/trajectory-session-1.json");
    let nested_string = json!({"artifact": {"path": "tenant-a/team_1"}});
    let destination = nested_dispatcher
        .prepare_destination("session-1", Some(&nested_string))
        .unwrap();
    assert_eq!(destination.0, "tenant-a-team_1/trajectory-session-1.json");

    for template in [
        "/tmp/trajectory-{session_id}.json",
        "../trajectory-{session_id}.json",
        "{metadata.}/trajectory-{session_id}.json",
        "{metadata.tenant..id}/trajectory-{session_id}.json",
        "{metadata.tenant/trajectory-{session_id}.json",
        "{metadata.{tenant}}/trajectory-{session_id}.json",
        "{metadata.missing:-../escape}/trajectory-{session_id}.json",
    ] {
        assert!(
            validate_atif_filename_template(template).is_err(),
            "template should fail configuration validation: {template:?}"
        );
        let dispatcher = AtifDispatcher::new(AtifSectionConfig {
            filename_template: template.to_string(),
            ..AtifSectionConfig::default()
        });
        assert!(
            dispatcher
                .prepare_destination("session-1", Some(&json!({"tenant": "tenant-a"})))
                .is_err(),
            "template should be rejected: {template:?}"
        );
    }
}

#[test]
fn atif_payload_merges_correlation_with_existing_trajectory_extra() {
    let agent_uuid = Uuid::now_v7();
    let trajectory = crate::observability::atif::AtifTrajectory {
        schema_version: "ATIF-v1.7".to_string(),
        session_id: agent_uuid.to_string(),
        trajectory_id: Some(agent_uuid.to_string()),
        agent: AtifAgentInfo {
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
        steps: Vec::new(),
        notes: None,
        final_metrics: None,
        continued_trajectory_ref: None,
        subagent_trajectories: None,
        extra: Some(json!({
            "existing": "preserved",
            "nemo_relay": {
                "existing": "nested",
                "session_id": "untrusted"
            }
        })),
    };
    let write = prepare_atif_payload(
        agent_uuid,
        format!("trajectory-{agent_uuid}.json"),
        None,
        None,
        trajectory,
        Vec::new(),
        AtifCorrelation {
            session_id: Some("logical-session".to_string()),
            session_instance_id: Some("instance-id".to_string()),
            user_id: Some("alice".to_string()),
        },
    )
    .unwrap();
    let value: Json = serde_json::from_slice(&write.payload).unwrap();

    assert_eq!(value["session_id"], agent_uuid.to_string());
    assert_eq!(value["trajectory_id"], agent_uuid.to_string());
    assert_eq!(value["extra"]["existing"], "preserved");
    assert_eq!(value["extra"]["nemo_relay"]["existing"], "nested");
    assert_eq!(
        value["extra"]["nemo_relay"]["session_id"],
        "logical-session"
    );
    assert_eq!(
        value["extra"]["nemo_relay"]["session_instance_id"],
        "instance-id"
    );
    assert_eq!(value["extra"]["nemo_relay"]["user_id"], "alice");
    assert_eq!(value["extra"]["observed_events"], json!([]));
}

#[test]
fn atif_explicit_options_and_open_agent_teardown_are_written() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let ignored = push_function("not-an-agent");
    pop(&ignored);
    let agent = push_agent("open-agent");
    clear_plugin_configuration().unwrap();

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
#[cfg(feature = "object-store")]
fn atif_open_agent_teardown_failure_retains_runtime_diagnostic_report() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    let agent = push_agent("open-agent-delivery-failure");

    let teardown = clear_plugin_configuration().unwrap_err();
    assert!(teardown.to_string().contains("atif.remote_delivery_failed"));
    server.join().unwrap().unwrap();

    let report = crate::plugin::active_plugin_report()
        .expect("failed teardown should retain its runtime diagnostics");
    let diagnostic = report
        .runtime_diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "atif.remote_delivery_failed")
        .expect("remote failure should be retained in the report");
    assert_eq!(diagnostic.field.as_deref(), Some("storage[0]"));
    assert_eq!(
        diagnostic.session_id.as_deref(),
        Some(agent.uuid.to_string().as_str())
    );
    assert!(!diagnostic.message.is_empty());

    pop(&agent);
    clear_plugin_configuration().unwrap();
    assert!(crate::plugin::active_plugin_report().is_none());
}

#[test]
fn atif_rejects_unsafe_template_and_ignores_non_top_level_agents() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-errors");

    let invalid_template = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "single-file.json"
        }
    }));
    assert!(validate_plugin_config(&invalid_template).has_errors());
    assert!(futures::executor::block_on(initialize_plugins_exact(invalid_template)).is_err());

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }));
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let function = push_function("top-level-function");
    let nested_agent = push_agent("nested-under-function");
    pop(&nested_agent);
    pop(&function);
    clear_plugin_configuration().unwrap();

    assert_eq!(fs::read_dir(dir).unwrap().count(), 0);
}

#[test]
fn otlp_sections_register_inferred_subscribers_with_full_config() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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
    clear_plugin_configuration().unwrap();
}

#[test]
fn opentelemetry_endpoints_fan_out_to_heterogeneous_and_repeated_types() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let agent = push_agent("fanout-agent");
    pop(&agent);
    clear_plugin_configuration().unwrap();

    for request in [full_request, gen_ai_request, repeated_request] {
        let body = request
            .recv_timeout(Duration::from_secs(5))
            .expect("each configured endpoint should receive the exported span");
        assert!(!body.is_empty());
    }
}

#[test]
fn opentelemetry_rejects_canonical_equivalent_destinations() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    for (first, second) in [
        (
            "http://collector.example/v1/traces",
            "http://collector.example:80/v1/traces",
        ),
        (
            "https://collector.example/v1/traces",
            "https://collector.example:443/v1/traces",
        ),
        (
            "HTTP://COLLECTOR.EXAMPLE/v1/traces",
            "http://collector.example/v1/traces",
        ),
        (
            "http://collector.example//v1///traces",
            "http://collector.example/v1/traces/",
        ),
        ("http://localhost/v1/traces", "http://LOCALHOST/v1/traces"),
        ("http://localhost/v1/traces", "http://localhost./v1/traces"),
        (
            "http://localhost/v1/traces",
            "http://agent.localhost/v1/traces",
        ),
        ("http://localhost/v1/traces", "http://127.0.0.2/v1/traces"),
        ("http://localhost/v1/traces", "http://127.1/v1/traces"),
        ("http://localhost/v1/traces", "http://[::1]/v1/traces"),
    ] {
        let config = plugin_config(json!({
            "opentelemetry": {
                "enabled": true,
                "endpoints": [
                    {"type": "full", "endpoint": first},
                    {"type": "gen_ai", "endpoint": second}
                ]
            }
        }));

        let report = validate_plugin_config(&config);
        assert!(
            report.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == "observability.unsafe_otel_destination_collision"
            }),
            "expected equivalent destinations {first:?} and {second:?} to collide"
        );
    }

    let grpc = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "transport": "grpc",
                    "endpoint": "https://collector.example"
                },
                {
                    "type": "gen_ai",
                    "transport": "grpc",
                    "endpoint": "https://collector.example:443/"
                }
            ]
        }
    }));
    assert!(validate_plugin_config(&grpc).has_errors());
}

#[test]
fn opentelemetry_allows_distinct_canonical_destinations() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    for (first, second) in [
        ("http://collector.example", "http://collector.example/"),
        (
            "http://collector.example/",
            "http://collector.example/v1/traces",
        ),
        (
            "http://collector.example:4318/v1/traces",
            "http://collector.example:4319/v1/traces",
        ),
        (
            "http://collector.example:443/v1/traces",
            "https://collector.example/v1/traces",
        ),
        (
            "http://collector.example/v1/traces",
            "http://collector.example/custom/traces",
        ),
        (
            "http://collector.example/v1/traces",
            "http://collector.example/v1%2Ftraces",
        ),
        (
            "http://collector.example/v1/traces?tenant=one",
            "http://collector.example/v1/traces?tenant=two",
        ),
        (
            "http://localhost.example/v1/traces",
            "http://localhost/v1/traces",
        ),
        ("http://[::2]/v1/traces", "http://localhost/v1/traces"),
    ] {
        let config = plugin_config(json!({
            "opentelemetry": {
                "enabled": true,
                "endpoints": [
                    {"type": "full", "endpoint": first},
                    {"type": "gen_ai", "endpoint": second}
                ]
            }
        }));

        assert!(
            !validate_plugin_config(&config).has_errors(),
            "expected distinct destinations {first:?} and {second:?} to remain valid"
        );
    }

    let different_transports = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {
                    "type": "full",
                    "transport": "http_binary",
                    "endpoint": "http://collector.example/v1/traces"
                },
                {
                    "type": "gen_ai",
                    "transport": "grpc",
                    "endpoint": "http://collector.example/v1/traces"
                }
            ]
        }
    }));
    assert!(!validate_plugin_config(&different_transports).has_errors());
}

#[test]
fn opentelemetry_rejects_canonical_collision_during_validation_and_activation() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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

    let report = validate_plugin_config(&config);
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
    assert!(futures::executor::block_on(initialize_plugins_exact(config)).is_err());
    assert!(
        !global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
}

#[test]
fn opentelemetry_allows_repeated_projection_types_at_the_same_destination() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let config = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {"type": "full", "endpoint": "http://LOCALHOST:80//v1///traces/"},
                {"type": "full", "endpoint": "http://127.1/v1/traces"}
            ]
        }
    }));

    assert!(!validate_plugin_config(&config).has_errors());
}

#[test]
fn opentelemetry_endpoint_delivery_failure_does_not_block_other_endpoints() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    let agent = push_agent("failure-isolation-agent");
    pop(&agent);
    let _ = clear_plugin_configuration();

    let body = healthy_request
        .recv_timeout(Duration::from_secs(5))
        .expect("healthy endpoint should receive spans despite another endpoint failing");
    assert!(!body.is_empty());
}

#[test]
fn invalid_later_opentelemetry_endpoint_keeps_fanout_registration() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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

    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_subscribers
            .contains_key("nemo-relay-plugin.v1.observability:1:opentelemetry")
    );
    clear_plugin_configuration().unwrap();
}

#[test]
fn opentelemetry_shutdown_helper_attempts_every_constructed_endpoint() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let (subscribers, exporters): (Vec<_>, Vec<_>) = (0..2)
        .map(|index| {
            let exporter = opentelemetry_sdk::trace::InMemorySpanExporterBuilder::new().build();
            let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_simple_exporter(exporter.clone())
                .build();
            (
                std::sync::Arc::new(OpenTelemetrySubscriber::from_tracer_provider(
                    provider,
                    format!("rollback-{index}"),
                )),
                exporter,
            )
        })
        .unzip();

    assert!(shutdown_opentelemetry_subscribers(&subscribers).is_none());
    for exporter in exporters {
        assert!(
            exporter.is_shutdown_called(),
            "every constructed endpoint exporter should be shut down"
        );
    }
}

#[test]
fn opentelemetry_shutdown_helper_retains_every_endpoint_failure() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
        Arc::new(OpenTelemetrySubscriber::from_tracer_provider(
            provider,
            format!("shutdown-failure-{index}"),
        ))
    })
    .collect::<Vec<_>>();

    let OpenTelemetryShutdownFailure::Other(error) =
        shutdown_opentelemetry_subscribers(&subscribers)
            .expect("mixed endpoint shutdown failures should be reported")
    else {
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
fn signal_delivery_state_classifies_generic_sdk_shutdown_error_as_delivery() {
    let issue = signal_shutdown_issue(
        Err(
            crate::observability::otel::OpenTelemetryError::MetricProvider(
                "generic SDK final-export failure".to_string(),
            ),
        ),
        Some("otel.metrics_export_failed (1)".to_string()),
    )
    .expect("delivery failure should produce a shutdown issue");

    let OpenTelemetryShutdownFailure::Delivery(error) =
        shutdown_failure_from_errors(vec![issue]).expect("delivery failure should be reported")
    else {
        panic!("explicit delivery state must remain recoverable");
    };
    let error = error.to_string();
    assert!(error.contains("otel.metrics_export_failed (1)"), "{error}");
    assert!(
        error.contains("generic SDK final-export failure"),
        "raw shutdown error was lost: {error}"
    );
}

#[test]
fn metric_cardinality_limit_rejects_usize_max_in_plugin_validation() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "metrics": {
                "enabled": true,
                "cardinality_limit": usize::MAX,
                "endpoints": [{"endpoint": "https://collector.example/v1/metrics"}]
            }
        }
    }));

    let report = validate_plugin_config(&config);
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.component.as_deref() == Some("opentelemetry.metrics")
            && diagnostic.field.as_deref() == Some("cardinality_limit")
            && diagnostic.message.contains("less than usize::MAX")
    }));
}

#[test]
fn plugin_validation_reports_each_signal_specific_invalid_value() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let mut log_endpoint = test_signal_endpoint();
    log_endpoint.endpoint = "  ".to_string();
    log_endpoint.transport = "udp".to_string();
    log_endpoint
        .headers
        .insert("Authorization".to_string(), "token".to_string());
    log_endpoint
        .headers
        .insert("authorization".to_string(), "token".to_string());
    log_endpoint
        .header_env
        .insert("authorization".to_string(), "LOG_TOKEN".to_string());
    let logs = OpenTelemetryLogSectionConfig {
        enabled: true,
        endpoints: Some(vec![log_endpoint]),
        minimum_severity: "notice".to_string(),
        max_queue_size: 0,
        max_export_batch_size: 1,
        scheduled_delay_millis: 0,
        completed_span_context_ttl_millis: 0,
    };
    let metrics = OpenTelemetryMetricSectionConfig {
        enabled: true,
        endpoints: Some(vec![OpenTelemetrySignalEndpointConfig {
            endpoint: " ".to_string(),
            transport: "udp".to_string(),
            ..test_signal_endpoint()
        }]),
        export_interval_millis: 0,
        temporality: "instantaneous".to_string(),
        max_instruments: 0,
        cardinality_limit: 0,
    };
    let config = plugin_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "logs": logs,
            "metrics": metrics
        }
    }));

    let report = validate_plugin_config(&config);
    for (component, field) in [
        ("opentelemetry.logs", "minimum_severity"),
        ("opentelemetry.logs", "max_queue_size"),
        ("opentelemetry.logs", "scheduled_delay_millis"),
        ("opentelemetry.metrics", "temporality"),
        ("opentelemetry.metrics", "export_interval_millis"),
        ("opentelemetry.metrics", "max_instruments"),
        ("opentelemetry.metrics", "cardinality_limit"),
    ] {
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.component.as_deref() == Some(component)
                && diagnostic.field.as_deref() == Some(field)
        }));
    }
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.component.as_deref() == Some("opentelemetry.logs")
            && diagnostic.field.as_deref() == Some("endpoints[0].endpoint")
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.component.as_deref() == Some("opentelemetry.metrics")
            && diagnostic.field.as_deref() == Some("endpoints[0].transport")
    }));
}

fn counting_callbacks(
    counter: &Arc<AtomicUsize>,
) -> Vec<IndexedOpenTelemetryResource<EventSubscriberFn>> {
    let counter = Arc::clone(counter);
    vec![IndexedOpenTelemetryResource {
        index: 0,
        value: OpenTelemetryResource::Active(Arc::new(move |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        })),
    }]
}

fn counting_metric_callbacks(
    counter: &Arc<AtomicUsize>,
) -> Vec<IndexedOpenTelemetryResource<MetricEventCallback>> {
    let counter = Arc::clone(counter);
    vec![IndexedOpenTelemetryResource {
        index: 0,
        value: OpenTelemetryResource::Active(Arc::new(move |_, _| {
            counter.fetch_add(1, Ordering::Relaxed);
        })),
    }]
}

fn reserved_metric_mark(version: &str, data: serde_json::Value) -> crate::api::event::Event {
    crate::api::event::Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("metric")
            .data(data)
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(version)
                    .build(),
            )
            .build(),
        None,
        None,
    ))
}

#[test]
fn opentelemetry_delivery_continues_after_an_endpoint_panics() {
    let delivered = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let delivered_after_panic = std::sync::Arc::clone(&delivered);
    let callbacks: Vec<IndexedOpenTelemetryResource<EventSubscriberFn>> = vec![
        IndexedOpenTelemetryResource {
            index: 0,
            value: OpenTelemetryResource::Active(std::sync::Arc::new(|_: &Event| -> () {
                panic!("simulated endpoint failure")
            })),
        },
        IndexedOpenTelemetryResource {
            index: 1,
            value: OpenTelemetryResource::Skipped("invalid endpoint".to_string()),
        },
        IndexedOpenTelemetryResource {
            index: 2,
            value: OpenTelemetryResource::Active(std::sync::Arc::new(move |_| {
                delivered_after_panic.store(true, std::sync::atomic::Ordering::SeqCst);
            })),
        },
    ];
    let event = crate::api::event::Event::Mark(crate::api::event::MarkEvent::new(
        crate::api::event::BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("fanout")
            .build(),
        None,
        None,
    ));

    deliver_opentelemetry_event(&callbacks, &[], &[], &AtomicU64::new(0), None, &event);

    assert!(delivered.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn opentelemetry_routes_marks_by_metric_schema() {
    let traced = Arc::new(AtomicUsize::new(0));
    let logged = Arc::new(AtomicUsize::new(0));
    let metered = Arc::new(AtomicUsize::new(0));
    let trace_callbacks = counting_callbacks(&traced);
    let log_callbacks = counting_callbacks(&logged);
    let metered_for_callback = Arc::clone(&metered);
    let metric_callbacks: Vec<IndexedOpenTelemetryResource<MetricEventCallback>> =
        vec![IndexedOpenTelemetryResource {
            index: 0,
            value: OpenTelemetryResource::Active(Arc::new(
                move |_: &Event, measurements: &[ValidatedMetricMeasurement]| {
                    assert_eq!(measurements.len(), 1);
                    assert_eq!(
                        measurements[0].descriptor.name.as_str(),
                        "example.tokens.saved"
                    );
                    metered_for_callback.fetch_add(1, Ordering::Relaxed);
                },
            )),
        }];
    let rejected_metric_marks = AtomicU64::new(0);

    let ordinary_mark = crate::api::event::Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("routing-decision")
            .build(),
        None,
        None,
    ));
    deliver_opentelemetry_event(
        &trace_callbacks,
        &log_callbacks,
        &metric_callbacks,
        &rejected_metric_marks,
        Some("opentelemetry.metrics"),
        &ordinary_mark,
    );
    assert_eq!(traced.load(Ordering::Relaxed), 1);
    assert_eq!(logged.load(Ordering::Relaxed), 1);
    assert_eq!(metered.load(Ordering::Relaxed), 0);

    let valid_metric = reserved_metric_mark(
        METRIC_DATA_SCHEMA_VERSION,
        json!({
            "measurements": [{
                "name": "example.tokens.saved",
                "kind": "counter",
                "value_type": "u64",
                "value": 42
            }]
        }),
    );

    deliver_opentelemetry_event(
        &trace_callbacks,
        &log_callbacks,
        &metric_callbacks,
        &rejected_metric_marks,
        Some("opentelemetry.metrics"),
        &valid_metric,
    );
    assert_eq!(traced.load(Ordering::Relaxed), 1);
    assert_eq!(logged.load(Ordering::Relaxed), 1);
    assert_eq!(metered.load(Ordering::Relaxed), 1);
    assert_eq!(rejected_metric_marks.load(Ordering::Relaxed), 0);

    for (version, data) in [
        ("999", json!({"measurements": [{"name": "ignored"}]})),
        ("1", json!({"measurements": []})),
    ] {
        let event = reserved_metric_mark(version, data);
        deliver_opentelemetry_event(
            &trace_callbacks,
            &log_callbacks,
            &metric_callbacks,
            &rejected_metric_marks,
            Some("opentelemetry.metrics"),
            &event,
        );
    }
    assert_eq!(traced.load(Ordering::Relaxed), 1);
    assert_eq!(logged.load(Ordering::Relaxed), 1);
    assert_eq!(metered.load(Ordering::Relaxed), 1);
    assert_eq!(rejected_metric_marks.load(Ordering::Relaxed), 2);
}

#[test]
fn non_metric_schema_marks_keep_trace_and_log_routing() {
    let traced = Arc::new(AtomicUsize::new(0));
    let logged = Arc::new(AtomicUsize::new(0));
    let metered = Arc::new(AtomicUsize::new(0));
    let trace_callbacks = counting_callbacks(&traced);
    let log_callbacks = counting_callbacks(&logged);
    let metric_callbacks = counting_metric_callbacks(&metered);
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("custom-schema-mark")
            .data(json!({"measurements": []}))
            .data_schema(
                DataSchema::builder()
                    .name("example.custom")
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));

    deliver_opentelemetry_event(
        &trace_callbacks,
        &log_callbacks,
        &metric_callbacks,
        &AtomicU64::new(0),
        Some("opentelemetry.metrics"),
        &event,
    );

    assert_eq!(traced.load(Ordering::Relaxed), 1);
    assert_eq!(logged.load(Ordering::Relaxed), 1);
    assert_eq!(metered.load(Ordering::Relaxed), 0);
}

#[test]
fn log_only_plugin_reports_invalid_reserved_metric_marks_once() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    for (version, data) in [
        ("999", json!({"measurements": [{"name": "ignored"}]})),
        ("1", json!({"measurements": []})),
    ] {
        crate::api::scope::event(
            crate::api::scope::EmitMarkEventParams::builder()
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

    let report = crate::plugin::active_plugin_report().unwrap();
    let diagnostics = report
        .runtime_diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "otel.metric_mark_invalid")
        .collect::<Vec<_>>();
    assert_eq!(diagnostics.len(), 1, "diagnostics: {diagnostics:?}");
    assert_eq!(diagnostics[0].count, 2);
    assert_eq!(diagnostics[0].field.as_deref(), None);

    clear_plugin_configuration().unwrap();
}

#[test]
fn plugin_signal_rejections_record_runtime_diagnostics() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

    crate::api::scope::event(
        crate::api::scope::EmitMarkEventParams::builder()
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
        crate::api::scope::event(
            crate::api::scope::EmitMarkEventParams::builder()
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

    let report = crate::plugin::active_plugin_report().unwrap();
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

    assert!(clear_plugin_configuration().is_err());
}

#[test]
fn atif_storage_defaults_to_empty() {
    let config = AtifSectionConfig::default();
    assert!(config.storage.is_empty());
}

#[test]
fn atif_storage_section_parses_s3_variant() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": [{
            "type": "s3",
            "bucket": "my-bucket",
            "key_prefix": "openshell/"
        }]
    }))
    .expect("valid storage section should parse");
    assert_eq!(parsed.storage.len(), 1);
    match &parsed.storage[0] {
        AtifStorageConfig::Http(_) => panic!("expected s3 storage"),
        AtifStorageConfig::S3(s3) => {
            assert_eq!(s3.bucket, "my-bucket");
            assert_eq!(s3.key_prefix.as_deref(), Some("openshell/"));
        }
    }
}

#[test]
fn atif_storage_section_parses_http_variant() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": [{
            "type": "http",
            "endpoint": "https://example.com/atif",
            "timeout_millis": 1500,
            "headers": {"x-static": "value"},
            "header_env": {"authorization": "NEMO_RELAY_ATIF_HTTP_TOKEN"}
        }]
    }))
    .expect("valid HTTP storage section should parse");
    assert_eq!(parsed.storage.len(), 1);
    match &parsed.storage[0] {
        AtifStorageConfig::Http(http) => {
            assert_eq!(http.endpoint, "https://example.com/atif");
            assert_eq!(http.timeout_millis, 1500);
            assert_eq!(
                http.headers.get("x-static").map(String::as_str),
                Some("value")
            );
            assert_eq!(
                http.header_env.get("authorization").map(String::as_str),
                Some("NEMO_RELAY_ATIF_HTTP_TOKEN")
            );
        }
        AtifStorageConfig::S3(_) => panic!("expected HTTP storage"),
    }
}

#[test]
fn atif_storage_section_rejects_single_table() {
    let err = serde_json::from_value::<AtifSectionConfig>(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": {
            "type": "s3",
            "bucket": "my-bucket"
        }
    }))
    .expect_err("storage must be a list");
    assert!(
        err.to_string().contains("invalid type"),
        "unexpected error: {err}"
    );
}

#[test]
fn atif_storage_section_parses_array_of_tables() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": [
            {"type": "s3", "bucket": "primary", "key_prefix": "p/"},
            {"type": "http", "endpoint": "http://127.0.0.1:3000/atif"}
        ]
    }))
    .expect("array-of-tables form should parse");
    assert_eq!(parsed.storage.len(), 2);
    match &parsed.storage[0] {
        AtifStorageConfig::Http(_) => panic!("expected s3 storage"),
        AtifStorageConfig::S3(s3) => assert_eq!(s3.bucket, "primary"),
    }
    match &parsed.storage[1] {
        AtifStorageConfig::Http(http) => {
            assert_eq!(http.endpoint, "http://127.0.0.1:3000/atif");
        }
        AtifStorageConfig::S3(_) => panic!("expected HTTP storage"),
    }
}

#[test]
fn atif_storage_section_parses_empty_array() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": []
    }))
    .expect("empty array should parse");
    assert!(parsed.storage.is_empty());
}

#[test]
fn atif_storage_unknown_backend_type_is_rejected() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "carrier-pigeon", "bucket": "ignored"}]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.invalid_plugin_config")
    );
}

#[test]
fn disabled_atif_storage_config_does_not_report_feature_disabled() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": false,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "s3", "bucket": "configured-but-disabled"}]
        }
    })));

    assert!(
        !report.diagnostics.iter().any(|diag| {
            diag.code == "observability.feature_disabled"
                && diag.field.as_deref() == Some("storage")
        }),
        "disabled ATIF storage should not report feature-disabled diagnostics: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_empty_bucket_is_rejected() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "s3", "bucket": "  "}]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].bucket"))
    );
}

#[test]
fn atif_storage_diagnostics_carry_sink_index() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [
                {"type": "s3", "bucket": "ok"},
                {"type": "s3", "bucket": ""}
            ]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[1].bucket")),
        "diagnostic should point at the second entry: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_empty_http_endpoint_is_rejected() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "http", "endpoint": "  "}]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].endpoint")),
        "expected diagnostic for empty endpoint: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_malformed_http_endpoint_is_rejected() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "http", "endpoint": "ftp://example.com/atif"}]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].endpoint")),
        "expected diagnostic for malformed endpoint: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_timeout_must_be_positive() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "timeout_millis": 0
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].timeout_millis")),
        "expected diagnostic for non-positive timeout: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_invalid_literal_header_name_is_rejected() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "headers": {"bad header": "value"}
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].headers.bad header")),
        "expected diagnostic for invalid header name: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_invalid_literal_header_value_is_rejected() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "headers": {"x-bad": "bad\nvalue"}
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].headers.x-bad")),
        "expected diagnostic for invalid header value: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_header_env_missing_env_is_rejected() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_MISSING_ZZZZ";
    // SAFETY: tests in this binary do not concurrently observe this uniquely
    // named env var, so removing it is safe.
    unsafe {
        std::env::remove_var(var_name);
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "header_env": {"authorization": var_name}
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].header_env.authorization")),
        "expected diagnostic for missing header env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_header_env_empty_env_is_rejected() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_EMPTY_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "");
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "header_env": {"authorization": var_name}
            }]
        }
    })));
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var(var_name);
    }
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].header_env.authorization")),
        "expected diagnostic for empty header env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_header_env_whitespace_name_is_rejected() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "header_env": {"authorization": " NEMO_RELAY_TEST_ATIF_HTTP_HEADER "}
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report.diagnostics.iter().any(|diag| diag.field.as_deref()
            == Some("storage[0].header_env.authorization")
            && diag.message.contains("surrounding whitespace")),
        "expected diagnostic for whitespace header env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_header_env_present_env_is_accepted() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_OK_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "Bearer test-token");
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "header_env": {"authorization": var_name}
            }]
        }
    })));
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var(var_name);
    }
    assert!(
        !report.has_errors(),
        "validation should pass when header env var is set: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_editor_field_is_optional_typed_list() {
    let schema = AtifSectionConfig::editor_schema();
    let storage = schema.field("storage").expect("storage editor field");
    assert_eq!(storage.kind, EditorFieldKind::List);
    assert!(storage.optional);
    let tagged_union = storage
        .list_item
        .expect("storage list metadata")
        .tagged_union
        .expect("storage tagged-union metadata");
    assert_eq!(tagged_union.discriminator, "type");
    assert_eq!(
        tagged_union
            .variants
            .iter()
            .map(|variant| variant.tag)
            .collect::<Vec<_>>(),
        vec!["http", "s3"]
    );
}

#[test]
fn atif_storage_s3_parses_full_credential_block() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": [{
            "type": "s3",
            "bucket": "my-bucket",
            "key_prefix": "openshell/",
            "access_key_id": "AKIAEXAMPLE",
            "secret_access_key_var": "MY_SECRET_VAR",
            "session_token_var": "MY_TOKEN_VAR",
            "region": "us-west-2",
            "endpoint_url": "https://s3.example.com",
            "allow_http": false
        }]
    }))
    .expect("full credential block should parse");
    assert_eq!(parsed.storage.len(), 1);
    match &parsed.storage[0] {
        AtifStorageConfig::Http(_) => panic!("expected s3 storage"),
        AtifStorageConfig::S3(s3) => {
            assert_eq!(s3.bucket, "my-bucket");
            assert_eq!(s3.key_prefix.as_deref(), Some("openshell/"));
            assert_eq!(s3.access_key_id.as_deref(), Some("AKIAEXAMPLE"));
            assert_eq!(s3.secret_access_key_var.as_deref(), Some("MY_SECRET_VAR"));
            assert_eq!(s3.session_token_var.as_deref(), Some("MY_TOKEN_VAR"));
            assert_eq!(s3.region.as_deref(), Some("us-west-2"));
            assert_eq!(s3.endpoint_url.as_deref(), Some("https://s3.example.com"));
            assert_eq!(s3.allow_http, Some(false));
        }
    }
}

#[test]
fn atif_storage_secret_var_missing_env_is_rejected() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_MISSING_ZZZZ";
    // SAFETY: tests in this binary do not concurrently observe this uniquely
    // named env var, so removing it is safe.
    unsafe {
        std::env::remove_var(var_name);
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": "my-bucket",
                "secret_access_key_var": var_name
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].secret_access_key_var")),
        "expected diagnostic for missing env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_secret_var_empty_env_is_rejected() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_EMPTY_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "");
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": "my-bucket",
                "secret_access_key_var": var_name
            }]
        }
    })));
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var(var_name);
    }
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].secret_access_key_var")),
        "expected diagnostic for empty env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_secret_var_present_env_is_accepted() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_OK_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "secret-value");
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": "my-bucket",
                "secret_access_key_var": var_name
            }]
        }
    })));
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var(var_name);
    }
    assert!(
        !report.has_errors(),
        "validation should pass when env var is set: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_secret_var_empty_name_is_rejected() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": "my-bucket",
                "secret_access_key_var": "   "
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].secret_access_key_var")),
        "expected diagnostic for empty var name: {:?}",
        report.diagnostics
    );
}

#[test]
#[cfg(feature = "object-store")]
fn atif_storage_private_helpers_resolve_env_and_key_prefix_branches() {
    let _guard = crate::observability::test_mutex()
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

#[cfg(feature = "object-store")]
fn http_storage_config(endpoint: impl Into<String>) -> HttpStorageConfig {
    HttpStorageConfig {
        endpoint: endpoint.into(),
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        timeout_millis: 1_000,
    }
}

#[test]
#[cfg(feature = "object-store")]
fn http_upload_config_rejects_endpoint_timeout_and_header_errors() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    for endpoint in [" http://example.com", "://", "ftp://example.com"] {
        assert!(HttpUploadConfig::resolve(2, &http_storage_config(endpoint)).is_err());
    }

    let mut config = http_storage_config("https://example.com/atif");
    config.timeout_millis = 0;
    assert!(HttpUploadConfig::resolve(2, &config).is_err());

    config.timeout_millis = 1_000;
    config.headers.insert("bad header".into(), "value".into());
    assert!(HttpUploadConfig::resolve(2, &config).is_err());

    config.headers.clear();
    config.headers.insert("x-bad".into(), "bad\nvalue".into());
    assert!(HttpUploadConfig::resolve(2, &config).is_err());

    let variable = "NEMO_RELAY_TEST_ATIF_HTTP_RESOLVE_ZZZZ";
    // SAFETY: this uniquely named environment variable is serialized by the observability mutex.
    unsafe { std::env::set_var(variable, "Bearer resolved") };
    config.headers.clear();
    config
        .header_env
        .insert("authorization".into(), variable.into());
    let resolved = HttpUploadConfig::resolve(2, &config).unwrap();
    assert_eq!(
        resolved.headers.get("authorization").map(String::as_str),
        Some("Bearer resolved")
    );
    // SAFETY: cleanup of the test-only environment variable.
    unsafe { std::env::remove_var(variable) };
}

#[tokio::test]
#[cfg(feature = "object-store")]
async fn post_atif_http_reports_transport_failure() {
    let config = HttpUploadConfig::resolve(0, &http_storage_config("http://127.0.0.1:1")).unwrap();
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
#[cfg(feature = "object-store")]
fn s3_remote_storage_uploads_to_a_custom_http_endpoint() {
    let _guard = crate::observability::test_mutex()
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
    )
    .unwrap();
    let result = storage.put("trajectory.json", "session", b"{}");
    server.join().unwrap().unwrap();
    // SAFETY: cleanup of the test-only environment variable.
    unsafe { std::env::remove_var(variable) };
    result.unwrap();
}

#[test]
fn observability_private_editor_and_validation_helpers_cover_edge_configs() {
    assert_eq!(
        default_atof_file_sink_editor_value(),
        json!({
            "type": "file",
            "mode": "append"
        })
    );
    assert_eq!(
        default_atof_stream_sink_editor_value()["transport"],
        json!("http_post")
    );
    assert_eq!(
        default_opentelemetry_endpoint_editor_value()["service_name"],
        json!("unknown_service")
    );
    let field = otel_editor_field("optional", EditorFieldKind::String, &[], true);
    assert_eq!(field.name, "optional");
    assert!(field.optional);

    let plugin = ObservabilityPlugin;
    assert!(!plugin.allows_multiple_components());
    for value in [
        json!({"atof": {"enabled": true, "filename": "removed.jsonl"}}),
        json!({"atof": {"enabled": true, "sinks": []}}),
        json!({"opentelemetry": {"enabled": true, "endpoints": []}}),
    ] {
        let config = value.as_object().unwrap();
        assert!(!plugin.validate(config).is_empty());
    }
}

#[test]
fn atif_filename_helpers_cover_metadata_resolution_and_rejection_paths() {
    let template = "runs/{metadata.agent.name:-unknown}/{session_id}.json";
    validate_atif_filename_template(template).unwrap();
    assert_eq!(
        render_atif_filename(
            template,
            "session-1",
            Some(&json!({"agent": {"name": "planner"}})),
        )
        .unwrap(),
        "runs/planner/session-1.json"
    );
    assert_eq!(
        render_atif_filename(template, "session-1", None).unwrap(),
        "runs/unknown/session-1.json"
    );
    assert!(validate_atif_filename_template("{metadata.agent/{session_id}").is_err());
    assert!(validate_atif_filename_template("../{session_id}").is_err());
    assert!(
        render_atif_filename(
            "{metadata.agent.name}/{session_id}",
            "session-1",
            Some(&json!({"agent": "not-an-object"})),
        )
        .is_err()
    );
}
