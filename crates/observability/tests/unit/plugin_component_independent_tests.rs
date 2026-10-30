// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_plugin::component::*;
use nemo_relay_types::api::event::*;
use serde_json::json;
use serde_json::{Map, Value as Json};
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[test]
fn signal_header_resolution_validates_environment_sources_and_preserves_file_references() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let variable = "NEMO_RELAY_TEST_SIGNAL_HEADER_SOURCE";
    let _environment = EnvironmentGuard::capture([variable]);
    let mut endpoint = test_signal_endpoint();
    endpoint
        .header_env
        .insert("authorization".into(), variable.into());
    unsafe {
        std::env::remove_var(variable);
    }
    assert!(
        resolve_signal_headers("metrics", 3, &endpoint)
            .unwrap_err()
            .to_string()
            .contains("could not read environment variable")
    );
    for value in ["", " padded "] {
        unsafe {
            std::env::set_var(variable, value);
        }
        assert!(
            resolve_signal_headers("metrics", 3, &endpoint)
                .unwrap_err()
                .to_string()
                .contains("blank or padded environment variable")
        );
    }
    unsafe {
        std::env::set_var(variable, "Bearer secret");
    }
    endpoint
        .headers
        .insert("Authorization".into(), "inline".into());
    assert!(
        resolve_signal_headers("logs", 4, &endpoint)
            .unwrap_err()
            .to_string()
            .contains("cannot appear in both headers and header_env")
    );
    endpoint.headers.clear();
    endpoint
        .header_env
        .insert("x-blank".into(), " PADDED_VARIABLE ".into());
    assert!(
        resolve_signal_headers("logs", 4, &endpoint)
            .unwrap_err()
            .to_string()
            .contains("must name a nonblank environment variable")
    );
    endpoint.header_env.remove("x-blank");
    let directory = tempfile::tempdir().unwrap();
    let token_file = directory.path().join("token");
    std::fs::write(&token_file, b"file-token").unwrap();
    endpoint
        .header_file
        .insert("x-token".into(), token_file.to_string_lossy().into_owned());
    let (headers, files) = resolve_signal_headers("metrics", 3, &endpoint).unwrap();
    assert_eq!(headers["authorization"], "Bearer secret");
    assert_eq!(files["x-token"], token_file.to_string_lossy());
    endpoint.header_file.insert(
        "Authorization".into(),
        token_file.to_string_lossy().into_owned(),
    );
    assert!(resolve_signal_headers("logs", 4, &endpoint).is_err());
}

#[test]
fn signal_endpoint_configuration_reaches_exported_log_and_metric_resources() {
    use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
    use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
    use opentelemetry_proto::tonic::common::v1::any_value::Value;
    use prost::Message;

    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let token_file = directory.path().join("token");
    fs::write(&token_file, "file-token").unwrap();
    let (log_endpoint, log_request) = start_otlp_capture_server();
    let (metric_endpoint, metric_request) = start_otlp_capture_server();
    let endpoint = |address: String| {
        serde_json::from_value::<OpenTelemetrySignalEndpointConfig>(json!({
            "endpoint": address,
            "service_name": "coverage-service",
            "service_namespace": "coverage-namespace",
            "service_version": "test-version",
            "instrumentation_scope": "coverage-signal-scope",
            "headers": {"x-inline-token": "inline-token"},
            "header_file": {"x-file-token": token_file.to_string_lossy()},
            "resource_attributes": {"coverage.project": "relay"}
        }))
        .unwrap()
    };
    let logs = OpenTelemetryLogSubscriber::new(
        build_log_config(
            0,
            endpoint(log_endpoint.replace("/v1/traces", "/v1/logs")),
            &serde_json::from_value(json!({"enabled": true})).unwrap(),
            LogSeverity::Info,
            &PlaintextPolicy::default(),
        )
        .unwrap(),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .unwrap();
    let metrics = OpenTelemetryMetricSubscriber::new(
        build_metric_config(
            0,
            endpoint(metric_endpoint.replace("/v1/traces", "/v1/metrics")),
            &serde_json::from_value(json!({"enabled": true})).unwrap(),
            MetricTemporality::Cumulative,
            &PlaintextPolicy::default(),
        )
        .unwrap(),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .unwrap();
    logs.subscriber()(&Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("coverage.log")
            .data(json!({"message": "exported"}))
            .build(),
        None,
        None,
    )));
    metrics.subscriber()(&Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("coverage.metric")
            .data(json!({"measurements": [{
                "name": "coverage.requests", "kind": "counter", "value_type": "u64", "value": 3
            }]}))
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    )));
    logs.shutdown().unwrap();
    metrics.shutdown().unwrap();
    let logs = ExportLogsServiceRequest::decode(
        log_request
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let metrics = ExportMetricsServiceRequest::decode(
        metric_request
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    assert_eq!(logs.resource_logs[0].scope_logs[0].log_records.len(), 1);
    assert_eq!(
        metrics.resource_metrics[0].scope_metrics[0].metrics[0].name,
        "coverage.requests"
    );
    assert_eq!(
        logs.resource_logs[0].scope_logs[0]
            .scope
            .as_ref()
            .unwrap()
            .name,
        "coverage-signal-scope"
    );
    assert_eq!(
        metrics.resource_metrics[0].scope_metrics[0]
            .scope
            .as_ref()
            .unwrap()
            .name,
        "coverage-signal-scope"
    );
    for resource in [
        logs.resource_logs[0].resource.as_ref().unwrap(),
        metrics.resource_metrics[0].resource.as_ref().unwrap(),
    ] {
        for (key, expected) in [
            ("service.name", "coverage-service"),
            ("service.namespace", "coverage-namespace"),
            ("service.version", "test-version"),
            ("coverage.project", "relay"),
        ] {
            let value = resource
                .attributes
                .iter()
                .find(|attribute| attribute.key == key)
                .unwrap();
            assert_eq!(
                value.value.as_ref().unwrap().value,
                Some(Value::StringValue(expected.into()))
            );
        }
    }
}

#[test]
fn signal_validation_rejects_trace_types_and_invalid_resource_promotion_prefixes() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    for signal in ["logs", "metrics"] {
        let mut section = json!({"enabled": true, "endpoints": [{
            "endpoint": "http://collector.example/",
            "type": "full",
            "promote_resource_metadata_prefixes": [" padded "]
        }]});
        let config =
            component_config(json!({"opentelemetry": {"enabled": true, signal: section.take()}}));
        let report = validate_component(&config);
        for field in [
            "endpoints[0].type",
            "endpoints[0].promote_resource_metadata_prefixes",
        ] {
            assert!(
                report
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.field.as_deref() == Some(field)),
                "missing {signal}.{field}: {:?}",
                report.diagnostics
            );
        }
    }
}

#[test]
fn trace_validation_rejects_resource_promotion_prefixes_before_activation() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
        "opentelemetry": {"enabled": true, "endpoints": [{
            "type": "full", "endpoint": "http://collector.example/v1/traces",
            "promote_resource_metadata_prefixes": [""]
        }]}
    })));
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.field.as_deref()
                == Some("endpoints[0].promote_resource_metadata_prefixes"))
    );
}

#[test]
fn automatic_otel_requires_an_endpoint_and_limits_signals_to_endpoint_scope() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let endpoint = "OTEL_EXPORTER_OTLP_ENDPOINT";
    let trace_endpoint = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT";
    let log_endpoint = "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT";
    let metric_endpoint = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";
    let header = "OTEL_EXPORTER_OTLP_HEADERS";
    let traces = "OTEL_TRACES_EXPORTER";
    let disabled = "OTEL_SDK_DISABLED";
    let _environment = EnvironmentGuard::capture([
        endpoint,
        trace_endpoint,
        log_endpoint,
        metric_endpoint,
        header,
        traces,
        disabled,
    ]);

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::remove_var(endpoint);
        std::env::remove_var(trace_endpoint);
        std::env::remove_var(log_endpoint);
        std::env::remove_var(metric_endpoint);
        std::env::remove_var(disabled);
        std::env::set_var(header, "authorization=token");
    }
    assert_eq!(automatic_otlp_signals(), None);

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::set_var(trace_endpoint, "http://127.0.0.1:4318/v1/traces") };
    assert_eq!(
        automatic_otlp_signals(),
        Some(AutomaticOtlpSignals {
            traces: true,
            logs: false,
            metrics: false,
        })
    );

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::remove_var(trace_endpoint);
        std::env::set_var(endpoint, "http://127.0.0.1:4318");
        std::env::set_var(traces, "none");
    }
    assert_eq!(
        automatic_otlp_signals(),
        Some(AutomaticOtlpSignals {
            traces: true,
            logs: true,
            metrics: true,
        })
    );
    assert!(!environment_signal_enabled(traces));

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::set_var(traces, "console") };
    assert!(!environment_signal_enabled(traces));

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::set_var(disabled, "TRUE") };
    assert_eq!(automatic_otlp_signals(), None);
}

#[test]
fn automatic_otel_defaults_to_http_protobuf_without_protocol_configuration() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let generic = "OTEL_EXPORTER_OTLP_PROTOCOL";
    let traces = "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL";
    let _environment = EnvironmentGuard::capture([generic, traces]);

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::remove_var(generic);
        std::env::remove_var(traces);
    }
    assert!(crate::test_support::otel_signal::automatic_protocol_is_unset(traces));

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::set_var(generic, "grpc") };
    assert!(!crate::test_support::otel_signal::automatic_protocol_is_unset(traces));
    assert!(crate::test_support::otel_signal::automatic_protocol_is_grpc(traces));

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::set_var(traces, "http/json") };
    assert!(!crate::test_support::otel_signal::automatic_protocol_is_grpc(traces));

    // An unrecognized signal setting falls through to the generic upstream setting.
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe { std::env::set_var(traces, "invalid") };
    assert!(crate::test_support::otel_signal::automatic_protocol_is_grpc(traces));
}

#[test]
fn automatic_otel_subscribers_accept_global_otlp_headers() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let endpoint = "OTEL_EXPORTER_OTLP_ENDPOINT";
    let headers = "OTEL_EXPORTER_OTLP_HEADERS";
    let _environment = EnvironmentGuard::capture([endpoint, headers]);

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(endpoint, "http://127.0.0.1:4318");
        std::env::set_var(headers, "authorization=Bearer%20environment-token");
    }

    let traces = OpenTelemetrySubscriber::new_from_automatic_configuration_for_plugin(
        PlaintextPolicy::default(),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .expect("automatic trace exporter should accept global headers");
    let logs = OpenTelemetryLogSubscriber::new_from_automatic_configuration_for_plugin(
        PlaintextPolicy::default(),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .expect("automatic log exporter should accept global headers");
    let metrics = OpenTelemetryMetricSubscriber::new_from_automatic_configuration_for_plugin(
        PlaintextPolicy::default(),
    )
    .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
    .expect("automatic metric exporter should accept global headers");
    drop((traces, logs, metrics));
}

#[test]
fn automatic_otel_requires_protected_log_and_metric_endpoints_with_or_without_headers() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let _environment = plaintext_preflight_environment();
    for headers in [None, Some("authorization=Bearer%20environment-token")] {
        // SAFETY: the observability mutex serializes test-only environment changes.
        unsafe {
            std::env::set_var(
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://collector.example:4318",
            );
            match headers {
                Some(value) => std::env::set_var("OTEL_EXPORTER_OTLP_HEADERS", value),
                None => std::env::remove_var("OTEL_EXPORTER_OTLP_HEADERS"),
            }
        }
        let logs = OpenTelemetryLogSubscriber::new_from_automatic_configuration_for_plugin(
            PlaintextPolicy::default(),
        )
        .map(|exporter| exporter.with_host(crate::test_support::detached_host()));
        assert!(matches!(
            logs,
            Err(ExporterInitializationError::PlaintextRejected(message))
                if message.contains("OTLP log endpoints require HTTPS")
        ));
        let metrics = OpenTelemetryMetricSubscriber::new_from_automatic_configuration_for_plugin(
            PlaintextPolicy::default(),
        )
        .map(|exporter| exporter.with_host(crate::test_support::detached_host()));
        assert!(matches!(
            metrics,
            Err(ExporterInitializationError::PlaintextRejected(message))
                if message.contains("OTLP metric endpoints require HTTPS")
        ));
    }
}

#[test]
fn automatic_otel_signal_headers_override_generic_headers() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let endpoint = "OTEL_EXPORTER_OTLP_ENDPOINT";
    let generic = "OTEL_EXPORTER_OTLP_HEADERS";
    let signal_headers = [
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
        "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
    ];
    let _environment =
        EnvironmentGuard::capture([endpoint, generic].into_iter().chain(signal_headers));
    for (generic_value, signal_value, accepted) in [
        // Explicitly empty signal lists override malformed generic headers.
        ("authorization=Bearer%20generic-token,x-bad=%ZZ", "", true),
        // Malformed signal lists override valid generic headers and skip exporters.
        (
            "authorization=Bearer%20generic-token",
            "authorization=Bearer%20signal-token,x-bad=%ZZ",
            false,
        ),
    ] {
        // SAFETY: the observability mutex serializes test-only environment changes.
        unsafe {
            std::env::set_var(endpoint, "http://127.0.0.1:4318");
            std::env::set_var(generic, generic_value);
            for variable in signal_headers {
                std::env::set_var(variable, signal_value);
            }
        }
        let results = [
            OpenTelemetrySubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
            .map(drop),
            OpenTelemetryLogSubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
            .map(drop),
            OpenTelemetryMetricSubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
            .map(drop),
        ];
        for (variable, result) in signal_headers.into_iter().zip(results) {
            if accepted {
                result.expect("empty signal headers must override malformed generic headers");
            } else {
                let message = result.unwrap_err().to_string();
                assert!(message.contains(variable));
                assert!(message.contains("malformed percent escape"));
                assert!(!message.contains("generic-token"));
                assert!(!message.contains("signal-token"));
            }
        }
    }
}

#[test]
fn signal_resource_promotion_is_accepted_with_strict_unknown_field_policy() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    for signal in ["logs", "metrics"] {
        let mut value = json!({
            "version": 4,
            "policy": {"unknown_field": "error"},
            "opentelemetry": {
                "enabled": true,
                signal: {
                    "enabled": true,
                    "endpoints": [{
                        "endpoint": format!("https://collector.example/v1/{signal}"),
                        "promote_resource_metadata_prefixes": ["deployment."]
                    }]
                }
            }
        });
        if signal == "logs" {
            value["opentelemetry"][signal]["completed_span_context_ttl_millis"] = json!(60_000);
        }
        let config = component_config(value);
        let report = validate_component(&config);
        assert!(
            report.diagnostics.is_empty(),
            "{signal}: {:?}",
            report.diagnostics
        );
    }
}

#[test]
fn observability_v3_remains_trace_only_and_v4_accepts_signal_sections() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let version_three = component_config(json!({
        "version": 3,
        "opentelemetry": {
            "enabled": true,
            "endpoints": [{
                "type": "gen_ai",
                "endpoint": "https://collector.example/v1/traces"
            }]
        }
    }));
    let report = validate_component(&version_three);
    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("version")
            && diagnostic.message.contains("use version 4")
    }));

    let version_four = component_config(json!({
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
    assert!(!validate_component(&version_four).has_errors());
}

#[test]
fn disabled_signal_sections_reject_duplicate_explicit_destinations() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let config = component_config(json!({
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

    let report = validate_component(&config);
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
fn version_three_rejects_removed_otlp_controls() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
        "opentelemetry": {
            "enabled": false,
            "mark_projection": "tool",
            "attribute_mappings": [],
            "endpoint": "http://localhost:4318/v1/traces",
            "header_file": {"authorization": "/var/run/secrets/telemetry/token"}
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
        diagnostic.code == "observability.legacy_opentelemetry_field"
            && diagnostic.field.as_deref() == Some("header_file")
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.legacy_openinference_section"
            && diagnostic.field.as_deref() == Some("openinference")
    }));
}

#[test]
fn opentelemetry_endpoint_header_env_is_resolved_and_snapshotted() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let variable = "NEMO_RELAY_TEST_OTEL_HEADER_ENV";
    unsafe { std::env::set_var(variable, "Bearer initial") };
    let config = build_otel_config(
        0,
        OpenTelemetryEndpointConfig {
            tls: None,
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
            header_file: HashMap::new(),
            resource_attributes: HashMap::new(),
            mark_projection: MarkProjection::default(),
            mark_exclude_names: default_mark_exclude_names(),
            attribute_mappings: Vec::new(),
            promote_metadata_prefixes: Vec::new(),
            promote_resource_metadata_prefixes: Vec::new(),
        },
        &PlaintextPolicy::default(),
    )
    .unwrap();
    unsafe { std::env::set_var(variable, "Bearer changed") };
    assert_eq!(config.header("authorization"), Some("Bearer initial"));
    unsafe { std::env::remove_var(variable) };
}

#[test]
fn build_otel_config_rejects_each_activation_only_invalid_value() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.endpoint = "  ".to_string();
    assert!(build_otel_config(0, endpoint, &PlaintextPolicy::default()).is_err());

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.transport = "udp".to_string();
    assert!(build_otel_config(0, endpoint, &PlaintextPolicy::default()).is_err());

    for variable in ["", " PADDED_ENV "] {
        let mut endpoint = test_opentelemetry_endpoint();
        endpoint
            .header_env
            .insert("authorization".to_string(), variable.to_string());
        assert!(build_otel_config(1, endpoint, &PlaintextPolicy::default()).is_err());
    }

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint
        .headers
        .insert("Authorization".to_string(), "inline".to_string());
    endpoint
        .header_env
        .insert("authorization".to_string(), "TOKEN".to_string());
    assert!(build_otel_config(2, endpoint, &PlaintextPolicy::default()).is_err());

    let variable = "NEMO_RELAY_TEST_BLANK_OTEL_HEADER_ENV";
    unsafe { std::env::set_var(variable, "  ") };
    let mut endpoint = test_opentelemetry_endpoint();
    endpoint
        .header_env
        .insert("authorization".to_string(), variable.to_string());
    assert!(build_otel_config(3, endpoint, &PlaintextPolicy::default()).is_err());
    unsafe { std::env::remove_var(variable) };

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_queue_size = Some(0);
    assert!(build_otel_config(4, endpoint, &PlaintextPolicy::default()).is_err());

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_export_batch_size = Some(0);
    assert!(build_otel_config(4, endpoint, &PlaintextPolicy::default()).is_err());

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.scheduled_delay_millis = Some(0);
    assert!(build_otel_config(4, endpoint, &PlaintextPolicy::default()).is_err());

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_queue_size = Some(8);
    endpoint.max_export_batch_size = Some(9);
    assert!(build_otel_config(4, endpoint, &PlaintextPolicy::default()).is_err());
}

#[test]
fn validate_opentelemetry_section_reports_empty_and_malformed_endpoints() {
    // `validate_opentelemetry_section` reads the automatic-OTLP environment,
    // where a set `OTEL_EXPORTER_OTLP_*` endpoint suppresses the empty-section
    // diagnostic this asserts on. The mutex serializes it against the tests
    // that set those variables.
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let policy = ConfigPolicy::default();
    let mut diagnostics = Vec::new();
    validate_opentelemetry_section(
        &mut diagnostics,
        &policy,
        &OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: Vec::new(),
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
            file_sinks: Vec::new(),
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
            file_sinks: Vec::new(),
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
fn insecure_http_hosts_is_a_top_level_field_only() {
    let report = validate_component(&component_config(json!({
        "version": 4,
        "policy": {"unknown_field": "error"},
        "opentelemetry": {
            "enabled": true,
            "traces": [{
                "type": "gen_ai",
                "endpoint": "https://collector.example/v1/traces",
                "insecure_http_hosts": ["collector.example"]
            }]
        }
    })));
    assert!(
        report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "observability.unknown_field"
                && diagnostic
                    .field
                    .as_deref()
                    .is_some_and(|field| field.ends_with("insecure_http_hosts"))
        }),
        "{:?}",
        report.diagnostics
    );
}

#[test]
fn automatic_grpc_skips_tls_for_http_and_rejects_certificate_settings() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let signals = ["TRACES", "LOGS", "METRICS"];
    let suffixes = [
        "ENDPOINT",
        "CERTIFICATE",
        "CLIENT_CERTIFICATE",
        "CLIENT_KEY",
    ];
    let variables: Vec<_> = std::iter::once(String::new())
        .chain(signals.map(|signal| format!("{signal}_")))
        .flat_map(|prefix| suffixes.map(|suffix| format!("OTEL_EXPORTER_OTLP_{prefix}{suffix}")))
        .collect();
    let _environment = EnvironmentGuard::capture(&variables);
    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        for variable in &variables {
            std::env::remove_var(variable);
        }
    }

    for signal in signals {
        let endpoint_variable = format!("OTEL_EXPORTER_OTLP_{signal}_ENDPOINT");
        for (generic, specific) in [
            (None, None),
            (Some("http://127.0.0.1:4317"), None),
            (
                Some("https://collector.example:4317"),
                Some("http://127.0.0.1:4317"),
            ),
        ] {
            unsafe {
                for (name, value) in [
                    ("OTEL_EXPORTER_OTLP_ENDPOINT", generic),
                    (endpoint_variable.as_str(), specific),
                ] {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
            assert!(
                crate::test_support::otel_signal::automatic_otlp_grpc_tls(signal)
                    .unwrap()
                    .is_none(),
                "{signal}: HTTP must not load trusted certificates"
            );

            assert_http_rejects_tls_settings(signal);
        }
        unsafe {
            std::env::remove_var(endpoint_variable);
        }
    }
}

#[test]
fn automatic_otel_rejects_an_unparseable_signal_endpoint_instead_of_falling_back() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let variables = [
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
    ];
    let _environment = EnvironmentGuard::capture(variables);

    // SAFETY: the observability mutex serializes test-only environment changes.
    unsafe {
        std::env::set_var(variables[0], "http://host.docker.internal:4318/v1/traces ");
        std::env::set_var(variables[1], "http://host.docker.internal:4318/v1/logs\n");
        std::env::set_var(variables[2], "http://192.0.2.17:4318");
        std::env::set_var(variables[3], "authorization=Bearer%20environment-token");
    }

    let listed = PlaintextPolicy::from_hosts(&["host.docker.internal".to_string()]).unwrap();
    let traces =
        OpenTelemetrySubscriber::new_from_automatic_configuration_for_plugin(listed.clone())
            .map(|exporter| exporter.with_host(crate::test_support::detached_host()));
    let error = traces.err().unwrap().to_string();
    assert!(
        error.contains("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT must be a valid URI"),
        "{error}"
    );
    let logs = OpenTelemetryLogSubscriber::new_from_automatic_configuration_for_plugin(listed)
        .map(|exporter| exporter.with_host(crate::test_support::detached_host()));
    let error = logs.err().unwrap().to_string();
    assert!(
        error.contains("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT must be a valid URI"),
        "{error}"
    );
}

#[test]
fn atof_stream_header_validation_reports_invalid_values_and_environment_names() {
    let _guard = crate::test_support::test_mutex()
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
    let mut dispatcher = AtifDispatcher::new(
        AtifSectionConfig::default(),
        crate::test_support::detached_host(),
    );
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

#[cfg(unix)]
#[test]
fn opentelemetry_header_env_non_unicode_errors_do_not_expose_values() {
    use std::os::unix::ffi::OsStringExt;

    let _guard = crate::test_support::test_mutex()
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
    let trace_error = build_otel_config(0, trace, &PlaintextPolicy::default())
        .unwrap_err()
        .to_string();
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
fn opentelemetry_endpoint_accepts_legacy_projection_controls_and_rejects_unknown_fields() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
fn opentelemetry_endpoint_accepts_valid_attribute_mappings() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let variable = "NEMO_RELAY_TEST_DISABLED_OTEL_HEADER_ENV";
    unsafe { std::env::remove_var(variable) };

    let report = validate_component(&component_config(json!({
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

#[test]
fn atif_dispatcher_default_output_path_uses_current_directory() {
    let dispatcher = AtifDispatcher::new(
        AtifSectionConfig::default(),
        crate::test_support::detached_host(),
    );
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
    let dispatcher = AtifDispatcher::new(
        AtifSectionConfig::default(),
        crate::test_support::detached_host(),
    );
    for session_id in ["../escape", "/absolute", "nested/../../escape"] {
        assert!(
            dispatcher.prepare_destination(session_id, None).is_err(),
            "unsafe session id should be rejected: {session_id:?}"
        );
    }
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

    let dispatcher = AtifDispatcher::new(
        AtifSectionConfig {
            filename_template: "{metadata.artifact_path}/trajectory-{session_id}.json".to_string(),
            ..AtifSectionConfig::default()
        },
        crate::test_support::detached_host(),
    );
    assert!(dispatcher.prepare_destination("session-1", None).is_err());
    let non_string = json!({"artifact_path": 123});
    assert!(
        dispatcher
            .prepare_destination("session-1", Some(&non_string))
            .is_err()
    );
    let dispatcher_with_fallback = AtifDispatcher::new(
        AtifSectionConfig {
            filename_template: "{metadata.artifact_path:-unassigned}/trajectory-{session_id}.json"
                .to_string(),
            ..AtifSectionConfig::default()
        },
        crate::test_support::detached_host(),
    );
    let error = dispatcher_with_fallback
        .prepare_destination("session-1", Some(&non_string))
        .unwrap_err();
    assert!(error.contains("resolved to a non-string value"), "{error}");
    let nested_non_string = json!({"artifact": 123});
    let nested_dispatcher = AtifDispatcher::new(
        AtifSectionConfig {
            filename_template: "{metadata.artifact.path:-unassigned}/trajectory-{session_id}.json"
                .to_string(),
            ..AtifSectionConfig::default()
        },
        crate::test_support::detached_host(),
    );
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
        let dispatcher = AtifDispatcher::new(
            AtifSectionConfig {
                filename_template: template.to_string(),
                ..AtifSectionConfig::default()
            },
            crate::test_support::detached_host(),
        );
        assert!(
            dispatcher
                .prepare_destination("session-1", Some(&json!({"tenant": "tenant-a"})))
                .is_err(),
            "template should be rejected: {template:?}"
        );
    }
}

#[test]
fn opentelemetry_rejects_canonical_equivalent_destinations() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
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
        let config = component_config(json!({
            "opentelemetry": {
                "enabled": true,
                "endpoints": [
                    {"type": "full", "endpoint": first},
                    {"type": "gen_ai", "endpoint": second}
                ]
            }
        }));

        let report = validate_component(&config);
        assert!(
            report.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == "observability.unsafe_otel_destination_collision"
            }),
            "expected equivalent destinations {first:?} and {second:?} to collide"
        );
    }

    let grpc = component_config(json!({
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
    assert!(validate_component(&grpc).has_errors());
}

#[test]
fn opentelemetry_allows_distinct_canonical_destinations() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
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
        let config = component_config(json!({
            "opentelemetry": {
                "enabled": true,
                "endpoints": [
                    {"type": "full", "endpoint": first},
                    {"type": "gen_ai", "endpoint": second}
                ]
            }
        }));

        assert!(
            !validate_component(&config).has_errors(),
            "expected distinct destinations {first:?} and {second:?} to remain valid"
        );
    }

    let different_transports = component_config(json!({
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
    assert!(!validate_component(&different_transports).has_errors());
}

#[test]
fn opentelemetry_allows_repeated_projection_types_at_the_same_destination() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let config = component_config(json!({
        "opentelemetry": {
            "enabled": true,
            "endpoints": [
                {"type": "full", "endpoint": "http://LOCALHOST:80//v1///traces/"},
                {"type": "full", "endpoint": "http://127.1/v1/traces"}
            ]
        }
    }));

    assert!(!validate_component(&config).has_errors());
}

#[test]
fn metric_cardinality_limit_rejects_usize_max_in_plugin_validation() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let config = component_config(json!({
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

    let report = validate_component(&config);
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.component.as_deref() == Some("opentelemetry.metrics")
            && diagnostic.field.as_deref() == Some("cardinality_limit")
            && diagnostic.message.contains("less than usize::MAX")
    }));
}

#[test]
fn plugin_validation_reports_each_signal_specific_invalid_value() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
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
    let config = component_config(json!({
        "version": 4,
        "opentelemetry": {
            "enabled": true,
            "logs": logs,
            "metrics": metrics
        }
    }));

    let report = validate_component(&config);
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
            value: OpenTelemetryResource::Skipped(SkippedResource::for_endpoint(
                "",
                "invalid endpoint",
            )),
        },
        IndexedOpenTelemetryResource {
            index: 2,
            value: OpenTelemetryResource::Active(std::sync::Arc::new(move |_| {
                delivered_after_panic.store(true, std::sync::atomic::Ordering::SeqCst);
            })),
        },
    ];
    let event =
        nemo_relay_types::api::event::Event::Mark(nemo_relay_types::api::event::MarkEvent::new(
            nemo_relay_types::api::event::BaseEvent::builder()
                .uuid(Uuid::now_v7())
                .name("fanout")
                .build(),
            None,
            None,
        ));

    deliver_opentelemetry_event(
        &callbacks,
        &[],
        &[],
        &[],
        &AtomicU64::new(0),
        None,
        &event,
        crate::test_support::detached_host().as_ref(),
    );

    assert!(delivered.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn opentelemetry_routes_stream_timing_to_metrics_without_losing_traces_or_logs() {
    use nemo_relay_types::api::event::{CategoryProfile, MetricValue, ScopeCategory};

    let traced = Arc::new(AtomicUsize::new(0));
    let logged = Arc::new(AtomicUsize::new(0));
    let metered = Arc::new(AtomicUsize::new(0));
    let trace_callbacks = counting_callbacks(&traced);
    let log_callbacks = counting_callbacks(&logged);
    let captured = Arc::clone(&metered);
    let metric_observers: Vec<IndexedOpenTelemetryResource<EventSubscriberFn>> =
        vec![IndexedOpenTelemetryResource {
            index: 0,
            value: OpenTelemetryResource::Active(Arc::new(move |event| {
                let Some(measurement) =
                    crate::otel_metrics::gen_ai_stream_time_to_first_chunk_measurement(
                        &crate::test_support::event_view(event),
                    )
                else {
                    return;
                };
                assert_eq!(
                    measurement.descriptor.name.as_str(),
                    "gen_ai.client.operation.time_to_first_chunk"
                );
                assert!(
                    matches!(measurement.value, MetricValue::F64(value) if value.get() == 0.125)
                );
                captured.fetch_add(1, Ordering::Relaxed);
            })),
        }];
    let rejected = AtomicU64::new(0);

    for (category, name, timing) in [
        (ScopeCategory::Start, "openai.chat.completions", None),
        (ScopeCategory::End, "openai.chat.completions", Some(0.125)),
        (ScopeCategory::End, "openai.chat.completions", None),
        (ScopeCategory::End, "unknown-provider", Some(0.125)),
    ] {
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name(name).build(),
            category,
            Vec::new(),
            EventCategory::llm(),
            Some(
                CategoryProfile::builder()
                    .time_to_first_chunk_opt(timing)
                    .build(),
            ),
        ));
        deliver_opentelemetry_event(
            &trace_callbacks,
            &log_callbacks,
            &metric_observers,
            &[],
            &rejected,
            Some("opentelemetry.metrics"),
            &event,
            crate::test_support::detached_host().as_ref(),
        );
    }

    assert_eq!(traced.load(Ordering::Relaxed), 4);
    assert_eq!(logged.load(Ordering::Relaxed), 4);
    assert_eq!(metered.load(Ordering::Relaxed), 1);
    assert_eq!(rejected.load(Ordering::Relaxed), 0);
}

#[test]
fn opentelemetry_routes_marks_by_metric_schema() {
    let traced = Arc::new(AtomicUsize::new(0));
    let logged = Arc::new(AtomicUsize::new(0));
    let metered = Arc::new(AtomicUsize::new(0));
    let metric_observed = Arc::new(AtomicUsize::new(0));
    let trace_callbacks = counting_callbacks(&traced);
    let log_callbacks = counting_callbacks(&logged);
    let metric_observers = counting_callbacks(&metric_observed);
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

    let ordinary_mark = nemo_relay_types::api::event::Event::Mark(MarkEvent::new(
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
        &metric_observers,
        &metric_callbacks,
        &rejected_metric_marks,
        Some("opentelemetry.metrics"),
        &ordinary_mark,
        crate::test_support::detached_host().as_ref(),
    );
    assert_eq!(traced.load(Ordering::Relaxed), 1);
    assert_eq!(logged.load(Ordering::Relaxed), 1);
    assert_eq!(metered.load(Ordering::Relaxed), 0);
    assert_eq!(metric_observed.load(Ordering::Relaxed), 1);

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
        &metric_observers,
        &metric_callbacks,
        &rejected_metric_marks,
        Some("opentelemetry.metrics"),
        &valid_metric,
        crate::test_support::detached_host().as_ref(),
    );
    assert_eq!(traced.load(Ordering::Relaxed), 1);
    assert_eq!(logged.load(Ordering::Relaxed), 1);
    assert_eq!(metered.load(Ordering::Relaxed), 1);
    assert_eq!(metric_observed.load(Ordering::Relaxed), 1);
    assert_eq!(rejected_metric_marks.load(Ordering::Relaxed), 0);

    for (version, data) in [
        ("999", json!({"measurements": [{"name": "ignored"}]})),
        ("1", json!({"measurements": []})),
    ] {
        let event = reserved_metric_mark(version, data);
        deliver_opentelemetry_event(
            &trace_callbacks,
            &log_callbacks,
            &metric_observers,
            &metric_callbacks,
            &rejected_metric_marks,
            Some("opentelemetry.metrics"),
            &event,
            crate::test_support::detached_host().as_ref(),
        );
    }
    assert_eq!(traced.load(Ordering::Relaxed), 1);
    assert_eq!(logged.load(Ordering::Relaxed), 1);
    assert_eq!(metered.load(Ordering::Relaxed), 1);
    assert_eq!(metric_observed.load(Ordering::Relaxed), 1);
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
        &[],
        &metric_callbacks,
        &AtomicU64::new(0),
        Some("opentelemetry.metrics"),
        &event,
        crate::test_support::detached_host().as_ref(),
    );

    assert_eq!(traced.load(Ordering::Relaxed), 1);
    assert_eq!(logged.load(Ordering::Relaxed), 1);
    assert_eq!(metered.load(Ordering::Relaxed), 0);
}

#[test]
fn atif_storage_unknown_backend_type_is_rejected() {
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_MISSING_ZZZZ";
    // SAFETY: tests in this binary do not concurrently observe this uniquely
    // named env var, so removing it is safe.
    unsafe {
        std::env::remove_var(var_name);
    }
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_EMPTY_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "");
    }
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_OK_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "Bearer test-token");
    }
    let report = validate_component(&component_config(json!({
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
fn atif_storage_secret_var_missing_env_is_rejected() {
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_MISSING_ZZZZ";
    // SAFETY: tests in this binary do not concurrently observe this uniquely
    // named env var, so removing it is safe.
    unsafe {
        std::env::remove_var(var_name);
    }
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_EMPTY_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "");
    }
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_OK_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "secret-value");
    }
    let report = validate_component(&component_config(json!({
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
    let _guard = crate::test_support::test_mutex().lock().unwrap();
    let report = validate_component(&component_config(json!({
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
fn observability_private_editor_and_validation_helpers_cover_edge_configs() {
    // `validate_opentelemetry_section` reads the automatic-OTLP environment,
    // where a set `OTEL_EXPORTER_OTLP_*` endpoint suppresses the empty-section
    // diagnostic this asserts on. The mutex serializes it against the tests
    // that set those variables.
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());

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

    let plugin = ObservabilityPlugin::new(crate::test_support::detached_host());
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
fn a_file_sink_config_builds_a_subscriber_and_opens_its_file() {
    let directory = tempfile::tempdir().unwrap();
    let config = build_otel_file_config(0, file_sink_section(directory.path())).unwrap();

    // Endpoint validation must not apply: this destination has no endpoint.
    let subscriber = crate::test_support::otel::OpenTelemetrySubscriber::new_file_sink(config)
        .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
        .unwrap();
    assert!(directory.path().join("trace.jsonl").is_file());
    subscriber.shutdown().unwrap();
}

#[test]
fn a_file_sink_applies_every_optional_setting() {
    let _guard = crate::test_support::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let mut section = file_sink_section(directory.path());
    section.max_queue_size = Some(4096);
    section.max_export_batch_size = Some(512);
    section.scheduled_delay_millis = Some(1000);
    section.completed_span_context_ttl_millis = Some(30_000);
    section.service_namespace = Some("agents".to_string());
    section.service_version = Some("1.2.3".to_string());
    section
        .resource_attributes
        .insert("deployment.environment".to_string(), "test".to_string());

    let config = build_otel_file_config(0, section).unwrap();

    assert_eq!(
        config.batch_overrides(),
        (Some(4096), Some(512), Some(Duration::from_millis(1000)))
    );
    assert_eq!(
        config.completed_span_context_ttl(),
        Duration::from_millis(30_000)
    );
    crate::test_support::otel::OpenTelemetrySubscriber::new_file_sink(config)
        .map(|exporter| exporter.with_host(crate::test_support::detached_host()))
        .unwrap()
        .shutdown()
        .unwrap();
}

#[test]
fn metric_delivery_continues_after_a_panicking_or_skipped_endpoint() {
    let delivered = Arc::new(AtomicUsize::new(0));
    let remaining = Arc::clone(&delivered);
    let callbacks: Vec<IndexedOpenTelemetryResource<MetricEventCallback>> = vec![
        IndexedOpenTelemetryResource {
            index: 0,
            value: OpenTelemetryResource::Skipped(SkippedResource::for_endpoint(
                "",
                "invalid endpoint",
            )),
        },
        IndexedOpenTelemetryResource {
            index: 1,
            value: OpenTelemetryResource::Active(Arc::new(|_, _| {
                panic!("metric endpoint failure")
            })),
        },
        IndexedOpenTelemetryResource {
            index: 2,
            value: OpenTelemetryResource::Active(Arc::new(move |_, measurements| {
                assert_eq!(measurements.len(), 1);
                remaining.fetch_add(1, Ordering::Relaxed);
            })),
        },
    ];
    let event = reserved_metric_mark(
        "1",
        json!({"measurements": [{
            "name": "calls", "kind": "counter", "value_type": "u64", "value": 1
        }]}),
    );
    deliver_opentelemetry_event(
        &[],
        &[],
        &[],
        &callbacks,
        &AtomicU64::new(0),
        None,
        &event,
        crate::test_support::detached_host().as_ref(),
    );
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
}

fn assert_http_rejects_tls_settings(signal: &str) {
    for prefix in [String::new(), format!("{signal}_")] {
        for fields in [
            &["CERTIFICATE"][..],
            &["CLIENT_CERTIFICATE", "CLIENT_KEY"][..],
        ] {
            unsafe {
                for field in fields {
                    std::env::set_var(format!("OTEL_EXPORTER_OTLP_{prefix}{field}"), "unused.pem");
                }
            }
            let error = crate::test_support::otel_signal::automatic_otlp_grpc_tls(signal)
                .unwrap_err()
                .to_string();
            unsafe {
                for field in fields {
                    std::env::remove_var(format!("OTEL_EXPORTER_OTLP_{prefix}{field}"));
                }
            }
            assert!(
                error.contains("TLS settings require an HTTPS endpoint"),
                "{error}"
            );
        }
    }
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

fn reserved_metric_mark(
    version: &str,
    data: serde_json::Value,
) -> nemo_relay_types::api::event::Event {
    nemo_relay_types::api::event::Event::Mark(MarkEvent::new(
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

fn component_config(value: Json) -> Map<String, Json> {
    value.as_object().unwrap().clone()
}
struct ValidationReport {
    diagnostics: Vec<ConfigDiagnostic>,
}
impl ValidationReport {
    fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
    }
}
fn validate_component(config: &Map<String, Json>) -> ValidationReport {
    ValidationReport {
        diagnostics: crate::plugin::validate_observability_plugin_config(config),
    }
}

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
