// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::otel_logs::{OpenTelemetryLogConfig, OpenTelemetryLogSubscriber};
use crate::otel_metrics::{OpenTelemetryMetricConfig, OpenTelemetryMetricSubscriber};
use nemo_relay_types::Json;
use nemo_relay_types::api::event::{
    BaseEvent, CategoryProfile, DataSchema, Event, EventCategory, METRIC_DATA_SCHEMA_NAME,
    MarkEvent, ScopeCategory, ScopeEvent,
};
use nemo_relay_types::api::scope::ScopeType;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::{KeyValue as OtlpKeyValue, any_value};
use opentelemetry_sdk::trace::{BatchConfigBuilder, InMemorySpanExporterBuilder};
use prost::Message;
use serde_json::json;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::thread;
use uuid::Uuid;

#[test]
fn shutdown_is_idempotent_for_all_otlp_subscribers() {
    let _guard = crate::test_mutex().lock().unwrap();

    let trace = OpenTelemetrySubscriber::from_tracer_provider(
        SdkTracerProvider::builder().build(),
        "shutdown-idempotency-test",
    );
    trace.shutdown().unwrap();
    trace.shutdown().unwrap();

    let logs = OpenTelemetryLogSubscriber::new(OpenTelemetryLogConfig::new(
        "http://127.0.0.1:4318/v1/logs",
    ))
    .unwrap();
    logs.shutdown().unwrap();
    logs.shutdown().unwrap();
    assert!(logs.force_flush().is_err());

    let metrics = OpenTelemetryMetricSubscriber::new(OpenTelemetryMetricConfig::new(
        "http://127.0.0.1:4318/v1/metrics",
    ))
    .unwrap();
    metrics.shutdown().unwrap();
    metrics.shutdown().unwrap();
    assert!(metrics.force_flush().is_err());
}

#[test]
fn shutdown_normalization_preserves_provider_failures() {
    let _guard = crate::test_mutex().lock().unwrap();
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();

    assert!(normalize_shutdown_result(Err(OTelSdkError::AlreadyShutdown)).is_ok());

    let error = normalize_shutdown_result(Err(OTelSdkError::InternalFailure(
        "collector unavailable".to_string(),
    )))
    .unwrap_err();
    assert!(matches!(
        error,
        OTelSdkError::InternalFailure(message) if message == "collector unavailable"
    ));

    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        AlwaysFailingSpanExporter,
        "https://collector.example/v1/traces".to_string(),
        SignalRuntimeDiagnostics::new(Some("opentelemetry.traces[0].endpoint".to_string())),
        BatchConfigBuilder::default()
            .with_max_export_batch_size(1)
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    provider
        .tracer("first-shutdown-failure-test")
        .start("export-fails")
        .end();
    provider.force_flush().unwrap_err();
    let subscriber =
        OpenTelemetrySubscriber::from_tracer_provider(provider, "first-shutdown-failure-test");

    let error = subscriber.shutdown().unwrap_err();
    assert!(matches!(
        error,
        OpenTelemetryError::TraceProvider(message)
            if message.contains(OTEL_RUNTIME_DELIVERY_FAILURE_MARKER)
    ));
    subscriber.shutdown().unwrap();
}

#[test]
fn config_defaults_and_builder_overrides_are_applied() {
    let config =
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_service_name("demo-agent")
            .with_header("authorization", "Bearer token")
            .with_header_env("x-api-key", "NEMO_RELAY_TEST_API_KEY")
            .with_resource_attribute("deployment.environment", "test")
            .with_service_namespace("agents")
            .with_service_version("1.2.3")
            .with_instrumentation_scope("demo-scope")
            .with_mark_projection(MarkProjection::Tool)
            .with_mark_exclude_names(["notification"])
            .with_attribute_mapping("nemo_relay.model_name", "model.alias")
            .with_timeout(Duration::from_millis(1250));
    assert_config_builder_overrides(&config);
    assert_config_defaults(&OpenTelemetryConfig::default());
}

#[test]
fn subscribers_reject_configured_telemetry_sdk_resource_attributes() {
    for key in [
        "telemetry.sdk.name",
        "telemetry.sdk.language",
        "telemetry.sdk.version",
    ] {
        let trace_error = OpenTelemetrySubscriber::new(
            OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://127.0.0.1:4318/v1/traces")
                .with_resource_attribute(key, "configured"),
        )
        .err()
        .expect("configured telemetry SDK trace attribute must be rejected");
        assert!(trace_error.to_string().contains(key));

        let log_error = OpenTelemetryLogSubscriber::new(
            OpenTelemetryLogConfig::new("http://127.0.0.1:4318/v1/logs")
                .with_resource_attribute(key, "configured"),
        )
        .err()
        .expect("configured telemetry SDK log attribute must be rejected");
        assert!(log_error.to_string().contains(key));

        let metric_error = OpenTelemetryMetricSubscriber::new(
            OpenTelemetryMetricConfig::new("http://127.0.0.1:4318/v1/metrics")
                .with_resource_attribute(key, "configured"),
        )
        .err()
        .expect("configured telemetry SDK metric attribute must be rejected");
        assert!(metric_error.to_string().contains(key));
    }
}

#[test]
fn grpc_config_owns_its_tokio_runtime() {
    let subscriber = OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4317")
            .with_transport(OtlpTransport::Grpc)
            .with_service_name("demo-agent"),
    )
    .expect("gRPC construction should not require an ambient Tokio runtime");
    subscriber.shutdown().unwrap();
}

#[test]
fn direct_config_rejects_a_blank_endpoint_before_exporter_construction() {
    let error = match OpenTelemetrySubscriber::new(OpenTelemetryConfig::new(
        OpenTelemetryType::GenAi,
        "  ",
    )) {
        Ok(_) => panic!("blank endpoints must be rejected"),
        Err(error) => error,
    };
    assert!(
        matches!(error, OpenTelemetryError::ExporterBuild(message) if message.contains("nonblank"))
    );
}

#[test]
fn trace_subscriber_constructors_reject_zero_completed_context_ttl() {
    let config = OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://127.0.0.1:4318")
        .with_completed_span_context_ttl(std::time::Duration::ZERO);
    let error = match OpenTelemetrySubscriber::new(config) {
        Ok(_) => panic!("zero TTL must be rejected"),
        Err(error) => error,
    };
    assert!(
        matches!(error, OpenTelemetryError::ExporterBuild(message) if message.contains("completed_span_context_ttl must be greater than 0"))
    );

    for typed in [false, true] {
        let (provider, _exporter) = make_provider();
        let options = OpenTelemetrySubscriberOptions {
            completed_span_context_ttl: std::time::Duration::ZERO,
            ..Default::default()
        };
        let result = if typed {
            OpenTelemetrySubscriber::from_tracer_provider_with_type_and_options(
                provider,
                "test-scope",
                OpenTelemetryType::OpenInference,
                options,
            )
        } else {
            OpenTelemetrySubscriber::from_tracer_provider_with_options(
                provider,
                "test-scope",
                options,
            )
        };
        let error = match result {
            Ok(_) => panic!("zero TTL must be rejected"),
            Err(error) => error,
        };
        assert!(
            matches!(error, OpenTelemetryError::ExporterBuild(message) if message.contains("completed_span_context_ttl must be greater than 0"))
        );
    }
}

#[test]
fn subscriber_options_apply_custom_completed_context_ttl_at_the_boundary() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_options(
        provider,
        "test-scope",
        OpenTelemetrySubscriberOptions {
            completed_span_context_ttl: std::time::Duration::from_millis(10),
            ..Default::default()
        },
    )
    .unwrap();
    let callback = subscriber.subscriber();
    let uuid = Uuid::now_v7();
    let closed_at = chrono::Utc::now();
    let scope_event = |category| {
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .uuid(uuid)
                .name("completed")
                .timestamp(closed_at)
                .build(),
            category,
            Vec::new(),
            EventCategory::from(ScopeType::Tool),
            None,
        ))
    };
    let mark_event = |name, timestamp| {
        Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .parent_uuid(uuid)
                .name(name)
                .timestamp(timestamp)
                .build(),
            None,
            None,
        ))
    };

    callback(&scope_event(ScopeCategory::Start));
    callback(&scope_event(ScopeCategory::End));
    callback(&mark_event(
        "at-boundary",
        closed_at + chrono::Duration::milliseconds(10),
    ));
    callback(&mark_event(
        "after-boundary",
        closed_at + chrono::Duration::milliseconds(11),
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let parent = finished_span_named(&spans, "completed");
    let boundary_mark = finished_span_named(&spans, "mark:at-boundary");
    let expired_mark = finished_span_named(&spans, "mark:after-boundary");
    assert_eq!(
        boundary_mark.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    assert_eq!(boundary_mark.parent_span_id, parent.span_context.span_id());
    assert_ne!(
        expired_mark.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    assert_ne!(expired_mark.parent_span_id, parent.span_context.span_id());
}

#[test]
fn direct_config_rejects_invalid_and_case_duplicate_headers() {
    for config in [
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header("bad header", "value"),
        OpenTelemetryConfig::new(OpenTelemetryType::GenAi, "http://localhost:4318/v1/traces")
            .with_header("x-control", "line one\nline two"),
        OpenTelemetryConfig::new(
            OpenTelemetryType::OpenInference,
            "http://localhost:4318/v1/traces",
        )
        .with_header("Authorization", "first")
        .with_header("authorization", "second"),
    ] {
        let error = match OpenTelemetrySubscriber::new(config) {
            Ok(_) => panic!("invalid direct exporter headers should fail construction"),
            Err(error) => error,
        };
        assert!(matches!(error, OpenTelemetryError::InvalidHeader { .. }));
    }
}

#[test]
fn direct_config_rejects_invalid_header_env_without_exposing_values() {
    let _guard = crate::test_mutex().lock().unwrap();
    let missing = format!("NEMO_RELAY_TEST_MISSING_HEADER_{}", Uuid::now_v7().simple());

    for (variable, value, expected) in [
        (missing.as_str(), None, "is not set"),
        ("NEMO_RELAY_TEST_BLANK_HEADER", Some("  "), "nonblank value"),
        (
            "NEMO_RELAY_TEST_SECRET_HEADER",
            Some("relay-secret\ninvalid"),
            "valid header value",
        ),
    ] {
        if let Some(value) = value {
            unsafe { std::env::set_var(variable, value) };
        }
        let error = match OpenTelemetrySubscriber::new(
            OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
                .with_header_env("authorization", variable),
        ) {
            Ok(_) => panic!("invalid header_env should fail activation"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
        assert!(!message.contains("relay-secret"));
        unsafe { std::env::remove_var(variable) };
    }

    for variable in ["", " padded ", "INVALID=NAME", "INVALID\0NAME"] {
        let error = match OpenTelemetrySubscriber::new(
            OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
                .with_header_env("authorization", variable),
        ) {
            Ok(_) => panic!("invalid environment variable reference should fail activation"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("header_env must name"));
    }

    let error = match OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header("Authorization", "static")
            .with_header_env("authorization", &missing),
    ) {
        Ok(_) => panic!("case-insensitive duplicate should fail before environment lookup"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("unique across headers and header_env")
    );

    unsafe { std::env::set_var(&missing, "valid") };
    let exact_error = match OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header("authorization", "static")
            .with_header_env("authorization", &missing),
    ) {
        Ok(_) => panic!("exact duplicate should fail activation"),
        Err(error) => error,
    };
    assert!(
        exact_error
            .to_string()
            .contains("unique across headers and header_env")
    );
    unsafe { std::env::remove_var(&missing) };
}

#[test]
fn direct_config_rejects_case_duplicate_header_env_names() {
    let missing = format!("NEMO_RELAY_TEST_MISSING_HEADER_{}", Uuid::now_v7().simple());
    let error = match OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header_env("Authorization", &missing)
            .with_header_env("authorization", &missing),
    ) {
        Ok(_) => panic!("case-insensitive header_env duplicate should fail activation"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("unique across headers and header_env")
    );
}

#[cfg(unix)]
#[test]
fn direct_config_non_unicode_header_env_errors_do_not_expose_values() {
    use std::os::unix::ffi::OsStringExt;

    let _guard = crate::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let variable = format!(
        "NEMO_RELAY_TEST_NON_UNICODE_DIRECT_HEADER_{}",
        Uuid::now_v7().simple()
    );
    let secret = "relay-direct-secret";
    let mut value = vec![0xff];
    value.extend_from_slice(secret.as_bytes());
    unsafe { std::env::set_var(&variable, std::ffi::OsString::from_vec(value)) };

    let error = match OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_header_env("authorization", &variable),
    ) {
        Ok(_) => panic!("non-Unicode header environment value should fail activation"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains(&variable));
    assert!(!message.contains(secret));

    unsafe { std::env::remove_var(variable) };
}

#[test]
fn mapped_aliases_are_typed_and_cannot_replace_projected_span_fields() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_options(
        provider,
        "mapping-scope",
        OpenTelemetrySubscriberOptions {
            mark_projection: MarkProjection::Tool,
            mark_exclude_names: vec!["custom.mark".to_string()],
            attribute_mappings: vec![
                crate::OtlpAttributeMapping::new("nemo_relay.start.data.tenant", "tenant.id"),
                crate::OtlpAttributeMapping::new(
                    "nemo_relay.end.data.tenant",
                    "nemo_relay.start.data.tenant",
                ),
                crate::OtlpAttributeMapping::new(
                    "nemo_relay.start.data.tenant",
                    "nemo_relay.start.data.existing",
                ),
                crate::OtlpAttributeMapping::new("missing.source", "ignored.alias"),
            ],
            promote_metadata_prefixes: Vec::new(),
            completed_span_context_ttl: DEFAULT_COMPLETED_SPAN_CONTEXT_TTL,
        },
    )
    .unwrap();
    let callback = subscriber.subscriber();
    let uuid = Uuid::now_v7();
    callback(&make_start_event(
        uuid,
        None,
        "mapped-scope",
        ScopeType::Agent,
        Some(json!({"tenant": 7, "existing": 9})),
    ));
    callback(&make_end_event(
        uuid,
        None,
        "mapped-scope",
        ScopeType::Agent,
        Some(json!({"tenant": 8})),
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mapped-scope")
        .unwrap();
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.start.data.tenant")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "tenant.id")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::I64(7))
    );
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.start.data.existing")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "nemo_relay.start.data.existing")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::I64(9))
    );
    assert!(
        !span
            .attributes
            .iter()
            .any(|attribute| attribute.key.as_str() == "ignored.alias")
    );
}

#[test]
fn session_identity_is_projected_on_trace_roots_and_marks_only() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "session-identity");
    let callback = subscriber.subscriber();
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let second_root_uuid = Uuid::now_v7();
    let instance_id = root_uuid.to_string();
    let second_instance_id = second_root_uuid.to_string();
    let identity = json!({
        "session_id": "logical-session",
        "user_id": "alice",
        "agent_kind": "claude-code"
    });

    callback(&with_propagation_root(
        make_start_event_with_metadata(root_uuid, None, "identity-root", identity.clone()),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_start_event_with_metadata(
            child_uuid,
            Some(root_uuid),
            "identity-child",
            identity.clone(),
        ),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_mark_event_with_metadata(Some(root_uuid), identity.clone()),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(
            child_uuid,
            Some(root_uuid),
            "identity-child",
            ScopeType::Agent,
            None,
        ),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(root_uuid, None, "identity-root", ScopeType::Agent, None),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_start_event_with_metadata(
            second_root_uuid,
            None,
            "identity-second-root",
            json!({"session_id": "logical-session", "user_id": 42}),
        ),
        second_root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(
            second_root_uuid,
            None,
            "identity-second-root",
            ScopeType::Agent,
            None,
        ),
        second_root_uuid,
    ));
    callback(&with_propagation_root(
        make_mark_event_with_metadata(None, identity),
        root_uuid,
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_session_root_and_child_identity(&spans, &instance_id, &second_instance_id);
    assert_session_mark_identity(&spans);
}

#[test]
fn mapped_orphan_mark_alias_cannot_replace_intrinsic_mark_fields() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_attribute_mappings(
        provider,
        "mapping-scope",
        [crate::OtlpAttributeMapping::new(
            "nemo_relay.mark.data.value",
            "nemo_relay.mark.orphan",
        )],
    )
    .unwrap();
    let callback = subscriber.subscriber();
    callback(&make_mark_event(
        None,
        "mapped-orphan-mark",
        Some(json!({"value": "not-an-orphan-flag"})),
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:mapped-orphan-mark")
        .unwrap();
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.mark.orphan")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "nemo_relay.mark.orphan")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::Bool(true))
    );
}

#[test]
fn all_trace_projections_require_protected_remote_transport() {
    for endpoint in [
        "https://collector.example/v1/traces",
        "http://localhost:4318/v1/traces",
        "http://127.0.0.1:4318/v1/traces",
        "http://127.0.0.2:4318/v1/traces",
        "http://[::1]:4318/v1/traces",
    ] {
        assert!(
            validate_trace_endpoint(endpoint, &PlaintextPolicy::LoopbackOnly).is_ok(),
            "{endpoint}"
        );
    }
    for endpoint in [
        "http://collector.example/v1/traces",
        "http://10.0.0.1:4318",
        "http://[::]:4318",
        "http://localhost.example:4318",
        "http://localhost@collector.example:4318",
        "ftp://localhost/traces",
        "not a URL",
    ] {
        for transport in [OtlpTransport::HttpBinary, OtlpTransport::Grpc] {
            for otel_type in [
                OpenTelemetryType::Full,
                OpenTelemetryType::GenAi,
                OpenTelemetryType::OpenInference,
            ] {
                let result = OpenTelemetrySubscriber::new(
                    OpenTelemetryConfig::new(otel_type, endpoint).with_transport(transport),
                );
                assert!(
                    result.err().unwrap().to_string().contains("require HTTPS"),
                    "{endpoint}"
                );
            }
        }
    }
}

#[test]
fn allow_insecure_permits_plain_http_trace_export_to_any_host() {
    let _guard = crate::test_mutex().lock().unwrap();
    let config =
        || OpenTelemetryConfig::new(OpenTelemetryType::GenAi, "http://collector.example:4318");
    assert!(OpenTelemetrySubscriber::new(config()).is_err());
    let subscriber = OpenTelemetrySubscriber::new(config().with_allow_insecure(true)).unwrap();
    subscriber.shutdown().unwrap();
    assert!(OpenTelemetrySubscriber::new(config().with_allow_insecure(false)).is_err());
}

#[test]
fn gen_ai_tool_content_is_exported_with_default_options() {
    let (provider, exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_type_and_options(
        provider,
        "tool-content",
        OpenTelemetryType::GenAi,
        OpenTelemetrySubscriberOptions::default(),
    )
    .unwrap();
    let id = Uuid::now_v7();
    let start = make_start_event(
        id,
        None,
        "search",
        ScopeType::Tool,
        Some(json!({"q": "docs"})),
    );
    let end = make_end_event(
        id,
        None,
        "search",
        ScopeType::Tool,
        Some(json!({"hits": []})),
    );
    (subscriber.inner.subscriber)(&start);
    (subscriber.inner.subscriber)(&end);
    subscriber.force_flush().unwrap();
    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attrs = attr_map(&spans[0].attributes);
    assert!(attrs.contains_key("gen_ai.tool.call.arguments"));
    assert!(attrs.contains_key("gen_ai.tool.call.result"));
    subscriber.shutdown().unwrap();
}

#[test]
fn gen_ai_tool_completion_fills_missing_metadata_without_replacing_start_identity() {
    for initial_id in [None, Some("typed-id")] {
        let (provider, exporter) = make_provider();
        let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_type(
            provider,
            "tool-identity",
            OpenTelemetryType::GenAi,
        );
        let id = Uuid::now_v7();
        let start = make_scope_event_with_profile(
            ScopeCategory::Start,
            id,
            None,
            "search",
            ScopeType::Tool,
            None,
            initial_id.map(|id| CategoryProfile::builder().tool_call_id(id).build()),
        );
        let mut end = make_end_event(id, None, "end-label", ScopeType::Tool, None);
        if let Event::Scope(scope) = &mut end {
            scope.base.metadata = Some(json!({
                "tool_call_id": "late-id", "tool_description": "Search docs",
                "gen_ai.tool.name": "late-name"
            }));
        }
        (subscriber.inner.subscriber)(&start);
        (subscriber.inner.subscriber)(&end);
        subscriber.force_flush().unwrap();
        let spans = exporter.get_finished_spans().unwrap();
        let attrs = attr_map(&spans[0].attributes);
        assert_eq!(
            attrs.get("gen_ai.tool.call.id").map(String::as_str),
            Some(initial_id.unwrap_or("late-id"))
        );
        assert_eq!(
            attrs.get("gen_ai.tool.description").map(String::as_str),
            Some("Search docs")
        );
        assert_eq!(
            attrs.get("gen_ai.tool.name").map(String::as_str),
            Some("search")
        );
        subscriber.shutdown().unwrap();
    }
}

#[test]
fn gen_ai_tool_completion_refines_only_inferred_identity() {
    for inferred_start in [false, true] {
        for completion in ["canonical", "alias", "inferred", "absent"] {
            let (provider, exporter) = make_provider();
            let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_type(
                provider,
                "identity-origin",
                OpenTelemetryType::GenAi,
            );
            let id = Uuid::now_v7();
            let mut start = make_scope_event_with_profile(
                ScopeCategory::Start,
                id,
                None,
                "search",
                ScopeType::Tool,
                None,
                Some(CategoryProfile::builder().tool_call_id("typed-id").build()),
            );
            let mut metadata =
                json!({"gen_ai.tool.type": "function", "gen_ai.agent.name": "codex"});
            if inferred_start {
                metadata["nemo_relay.tool.execution.defaults"] = metadata.clone();
            }
            if let Event::Scope(scope) = &mut start {
                scope.base.metadata = Some(metadata.clone());
            }
            let mut end = make_end_event(id, None, "end-label", ScopeType::Tool, None);
            match completion {
                "canonical" => {
                    metadata["gen_ai.tool.type"] = json!("extension");
                    metadata["gen_ai.agent.name"] = json!("researcher");
                }
                "alias" => {
                    metadata["tool_type"] = json!("extension");
                    metadata["agent_name"] = json!("researcher");
                }
                "inferred" => {
                    metadata = json!({"gen_ai.tool.type": "extension", "gen_ai.agent.name": "researcher",
                        "nemo_relay.tool.execution.defaults": {"gen_ai.tool.type": "extension", "gen_ai.agent.name": "researcher"}});
                }
                "absent" => metadata = json!({}),
                _ => unreachable!("unknown completion case"),
            }
            metadata["tool_call_id"] = json!("late-id");
            if let Event::Scope(scope) = &mut end {
                scope.base.metadata = Some(metadata);
            }
            subscriber.subscriber()(&start);
            subscriber.subscriber()(&end);
            subscriber.force_flush().unwrap();
            let spans = exporter.get_finished_spans().unwrap();
            let attrs = attr_map(&spans[0].attributes);
            let refined = inferred_start && matches!(completion, "canonical" | "alias");
            assert_eq!(
                attrs["gen_ai.tool.type"],
                if refined { "extension" } else { "function" },
                "{inferred_start}/{completion}"
            );
            assert_eq!(
                attrs["gen_ai.agent.name"],
                if refined { "researcher" } else { "codex" }
            );
            assert_eq!(attrs["gen_ai.tool.call.id"], "typed-id");
            assert!(!attrs.keys().any(|key| key.starts_with("nemo_relay.")));
            subscriber.shutdown().unwrap();
        }
    }
}

#[test]
fn root_metadata_promotes_to_a_shared_otlp_resource_without_overriding_sdk_identity() {
    let _guard = crate::test_mutex().lock().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (request_tx, request_rx) = mpsc::channel();
    spawn_http_collector(listener, request_tx);

    let subscriber = OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::http_binary("resource-test")
            .with_endpoint(endpoint)
            .with_resource_attribute("tenant.region", "configured")
            .with_promote_resource_metadata_prefixes(["tenant.", "telemetry.sdk."]),
    )
    .unwrap();
    let callback = subscriber.subscriber();
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    callback(&make_start_event_with_metadata(
        root_uuid,
        None,
        "resource-root",
        json!({
            "tenant.id": "root-tenant",
            "tenant.region": "metadata",
            "telemetry.sdk.name": "metadata-name",
            "telemetry.sdk.language": "metadata-language",
            "telemetry.sdk.version": "metadata-version",
        }),
    ));
    callback(&make_start_event_with_metadata(
        child_uuid,
        Some(root_uuid),
        "resource-child",
        json!({"tenant.id": "child-tenant"}),
    ));
    callback(&make_end_event(
        child_uuid,
        Some(root_uuid),
        "resource-child",
        ScopeType::Agent,
        None,
    ));
    callback(&make_end_event(
        root_uuid,
        None,
        "resource-root",
        ScopeType::Agent,
        None,
    ));
    {
        let processor = subscriber.inner.processor.lock().unwrap();
        let pipelines = processor.dynamic_pipelines.lock().unwrap();
        assert_eq!(pipelines.len(), 1);
        let key = pipelines.keys().next().unwrap();
        assert!(key.contains("root-tenant"));
        assert!(key.contains("configured"));
        assert!(!key.contains("child-tenant"));
        assert!(!key.contains("metadata"));
    }
    subscriber.force_flush().unwrap();

    let request = request_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("expected an OTLP request for the promoted root resource");
    let request = ExportTraceServiceRequest::decode(request.body.as_slice()).unwrap();
    assert_eq!(request.resource_spans.len(), 1);
    let resource_spans = request.resource_spans.first().unwrap();
    let resource = resource_spans.resource.as_ref().unwrap();
    assert_eq!(
        otlp_string_attribute(&resource.attributes, "tenant.id"),
        Some("root-tenant")
    );
    assert_eq!(
        otlp_string_attribute(&resource.attributes, "tenant.region"),
        Some("configured")
    );
    assert_telemetry_sdk_resource(&resource.attributes);

    let spans = resource_spans
        .scope_spans
        .iter()
        .flat_map(|scope_spans| &scope_spans.spans)
        .collect::<Vec<_>>();
    assert_eq!(spans.len(), 2);
    for span in spans {
        assert!(!has_promoted_resource_metadata(&span.attributes));
        assert!(
            span.events
                .iter()
                .all(|event| !has_promoted_resource_metadata(&event.attributes))
        );
    }
}

#[test]
fn direct_trace_subscriber_exposes_runtime_diagnostics() {
    let (provider, _exporter) = make_provider();
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider(provider, "diagnostics");
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("invalid-metric")
            .data(json!({"measurements": []}))
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version("999")
                    .build(),
            )
            .build(),
        None,
        None,
    ));

    for _ in 0..3 {
        (subscriber.subscriber())(&event);
    }

    let diagnostics = subscriber.runtime_diagnostics();
    let diagnostic = diagnostics
        .get("otel.metric_mark_invalid")
        .expect("invalid metric diagnostic");
    assert_eq!(diagnostic.count, 3);
    assert!(
        diagnostic
            .message
            .contains("unsupported metric schema version")
    );
}

#[test]
fn provider_builders_cover_success_paths() {
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let http_provider = build_tracer_provider(
        &OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_service_name("demo-agent")
            .with_header("authorization", "Bearer token")
            .with_resource_attribute("deployment.environment", "test")
            .with_service_namespace("agents")
            .with_service_version("1.2.3")
            .with_max_queue_size(16)
            .with_max_export_batch_size(4)
            .with_scheduled_delay(Duration::from_millis(10)),
        SignalRuntimeDiagnostics::new(None),
    )
    .unwrap();
    http_provider.force_flush().unwrap();
    http_provider.shutdown().unwrap();

    let subscriber = OpenTelemetrySubscriber::new(
        OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318/v1/traces")
            .with_service_name("http-success"),
    )
    .unwrap();
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[test]
fn direct_trace_processor_records_cumulative_queue_drops_on_flush_and_shutdown() {
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        InMemorySpanExporterBuilder::new().build(),
        "https://collector.example/v1/traces".to_string(),
        runtime_diagnostics.clone(),
        BatchConfigBuilder::default().build(),
    );
    processor.completed_spans.store(3, Ordering::Relaxed);
    processor.accepted_spans.store(1, Ordering::Relaxed);

    processor.force_flush().unwrap();
    assert_eq!(
        runtime_diagnostics
            .snapshot()
            .get("otel.spans_dropped")
            .map(|diagnostic| diagnostic.count),
        Some(2)
    );

    processor.completed_spans.store(5, Ordering::Relaxed);
    processor
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();
    let diagnostics = runtime_diagnostics.snapshot();
    let diagnostic = diagnostics
        .get("otel.spans_dropped")
        .expect("direct trace drop diagnostic");
    assert_eq!(diagnostic.count, 4);
    assert!(diagnostic.message.contains("https://collector.example:443"));
}

#[test]
fn http_trace_client_recovers_from_connection_errors_but_stops_after_fatal_tls_errors() {
    use opentelemetry_http::{HttpClient, Request};

    let _guard = crate::test_mutex().lock().unwrap();
    for (error, terminal) in [
        (
            rustls::Error::InvalidMessage(rustls::InvalidMessage::MessageTooShort),
            false,
        ),
        (
            rustls::Error::InappropriateMessage {
                expect_types: vec![rustls::ContentType::Handshake],
                got_type: rustls::ContentType::ApplicationData,
            },
            false,
        ),
        (
            rustls::Error::InvalidCertificate(rustls::CertificateError::UnknownIssuer),
            true,
        ),
        (rustls::Error::DecryptError, true),
        (
            rustls::Error::AlertReceived(rustls::AlertDescription::HandshakeFailure),
            true,
        ),
    ] {
        let attempts = Arc::new(AtomicUsize::new(0));
        let diagnostics = SignalRuntimeDiagnostics::new(None);
        let client = DiagnosticTraceHttpClient {
            inner: RecoveringTraceHttpClient {
                first_error: error.clone(),
                attempts: attempts.clone(),
            },
            diagnostics: diagnostics.clone(),
        };
        futures::executor::block_on(async {
            let request = || {
                Request::builder()
                    .uri("https://collector.example/v1/traces")
                    .body(Default::default())
                    .unwrap()
            };
            assert!(client.send_bytes(request()).await.is_err());
            assert_eq!(attempts.load(Ordering::Relaxed), 1);
            for _ in 0..2 {
                let next = client.send_bytes(request()).await;
                assert_eq!(next.is_err(), terminal, "{error:?}");
            }
        });
        assert_eq!(
            attempts.load(Ordering::Relaxed),
            if terminal { 1 } else { 3 },
            "{error:?}"
        );
        assert_eq!(
            diagnostics.trace_tls_failure().is_some(),
            terminal,
            "{error:?}"
        );
        let snapshot = diagnostics.snapshot();
        assert_eq!(
            snapshot.get("otel.tls_failed").map(|entry| entry.count),
            terminal.then_some(1)
        );
    }
}

#[test]
fn trace_export_failures_are_diagnosed_until_a_later_export_recovers() {
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        FailingThenRecoveringSpanExporter::default(),
        "https://collector.example/v1/traces".to_string(),
        runtime_diagnostics.clone(),
        BatchConfigBuilder::default()
            .with_max_export_batch_size(1)
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let tracer = provider.tracer("trace-export-failure-diagnostics-test");

    tracer.start("first-export-fails").end();
    assert!(provider.force_flush().is_err());

    let diagnostics = runtime_diagnostics.snapshot();
    let diagnostic = diagnostics
        .get("otel.traces_export_failed")
        .expect("trace export failure diagnostic");
    assert_eq!(diagnostic.count, 1);
    assert!(diagnostic.message.contains("https://collector.example:443"));
    assert!(!diagnostic.message.contains("collector unavailable"));

    let repeated_flush = provider.force_flush().unwrap_err();
    assert!(
        repeated_flush
            .to_string()
            .contains("otel.traces_export_failed (1)")
    );

    tracer.start("recovery-export").end();
    provider.force_flush().unwrap();
    provider.shutdown().unwrap();
}

#[test]
fn trace_export_failures_use_a_safe_endpoint_identity() {
    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let endpoint = "https://user:password@collector.example:4318/private-path?access_token=url-secret#fragment-secret";
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let processor = DiagnosticBatchSpanProcessor::new_with_batch_config(
        SensitiveFailingSpanExporter,
        endpoint.to_string(),
        runtime_diagnostics.clone(),
        BatchConfigBuilder::default()
            .with_max_export_batch_size(1)
            .build(),
    );
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let tracer = provider.tracer("safe-trace-export-failure-test");

    tracer.start("export-fails").end();
    let error = provider.force_flush().unwrap_err().to_string();
    let diagnostic = runtime_diagnostics
        .snapshot()
        .get("otel.traces_export_failed")
        .expect("trace export failure diagnostic")
        .message
        .clone();

    for message in [&error, &diagnostic] {
        assert!(message.contains("https://collector.example:4318"));
        for secret in [
            "user",
            "password",
            "private-path",
            "url-secret",
            "fragment-secret",
            "exporter-secret",
            "Authorization",
        ] {
            assert!(
                !message.contains(secret),
                "trace export failure leaked {secret:?}: {message}"
            );
        }
    }

    provider.shutdown().unwrap();
}

#[test]
fn grpc_metadata_and_runtime_builder_paths_succeed() {
    let metadata = build_grpc_metadata(&HashMap::from([(
        "authorization".to_string(),
        "Bearer token".to_string(),
    )]))
    .unwrap();
    assert_eq!(
        metadata.get("authorization").unwrap().to_str().unwrap(),
        "Bearer token"
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let provider = {
        let _runtime_guard = runtime.enter();
        build_tracer_provider(
            &OpenTelemetryConfig::grpc("grpc-demo")
                .with_endpoint("http://127.0.0.1:4317")
                .with_header("authorization", "Bearer token"),
            SignalRuntimeDiagnostics::new(None),
        )
        .unwrap()
    };
    provider.force_flush().ok();
    provider.shutdown().ok();
}

#[test]
fn promoted_root_resources_share_one_file_sink_stream() {
    let _guard = crate::test_mutex().lock().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");
    let sink = OtlpFileSinkSettings {
        output_directory: directory.path().to_path_buf(),
        filename: "trace.jsonl".to_string(),
        format: crate::otel_file::OtlpFileFormat::JsonLines,
        mode: crate::otel::OtlpFileSinkMode::Overwrite,
    };

    let subscriber = OpenTelemetrySubscriber::new_file_sink(
        OpenTelemetryFileSinkConfig::new(OpenTelemetryType::Full, sink)
            .with_promote_resource_metadata_prefixes(["tenant."]),
    )
    .unwrap();
    let callback = subscriber.subscriber();
    // Two roots with different promoted resources: each opens its own pipeline,
    // and both write the one file the sink configured.
    for tenant in ["first-tenant", "second-tenant"] {
        let root = Uuid::now_v7();
        callback(&make_start_event_with_metadata(
            root,
            None,
            tenant,
            json!({"tenant.id": tenant}),
        ));
        callback(&make_end_event(root, None, tenant, ScopeType::Agent, None));
    }
    subscriber.force_flush().unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    let mut tenants = BTreeSet::new();
    for line in contents.lines() {
        // Decoded with serde_json rather than the exporter's own encoder: a
        // truncated or interleaved stream fails here.
        let request: Json = serde_json::from_str(line).expect("every record decodes");
        for resource_spans in request["resourceSpans"].as_array().unwrap() {
            for attribute in resource_spans["resource"]["attributes"].as_array().unwrap() {
                if attribute["key"] == "tenant.id" {
                    tenants.insert(
                        attribute["value"]["stringValue"]
                            .as_str()
                            .unwrap()
                            .to_string(),
                    );
                }
            }
        }
    }
    assert_eq!(
        tenants,
        BTreeSet::from(["first-tenant".to_string(), "second-tenant".to_string()]),
        "both promoted resources should survive in {contents}"
    );
}

#[test]
fn provider_build_drains_the_startup_resource_message_before_the_first_span() {
    use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};

    let runtime = test_tokio_runtime();
    let _runtime_guard = runtime.enter();
    let _held_worker = runtime.spawn(async { thread::sleep(Duration::from_millis(250)) });
    let directory = tempfile::tempdir().unwrap();
    let config = TraceConfig::File(
        OpenTelemetryFileSinkConfig::new(
            OpenTelemetryType::Full,
            OtlpFileSinkSettings {
                output_directory: directory.path().to_path_buf(),
                filename: "trace.jsonl".to_string(),
                format: crate::otel_file::OtlpFileFormat::JsonLines,
                mode: crate::otel::OtlpFileSinkMode::Overwrite,
            },
        )
        .with_max_queue_size(1)
        .with_max_export_batch_size(1),
    );
    let attributes = configured_resource_attributes(&config);
    let provider = build_tracer_provider_with_resource(
        &config,
        SignalRuntimeDiagnostics::new(None),
        attributes,
    )
    .unwrap();
    provider.tracer("drain-test").start("first").end();
    provider.force_flush().unwrap();
    provider.shutdown().unwrap();

    let contents = std::fs::read_to_string(directory.path().join("trace.jsonl")).unwrap();
    let names = contents
        .lines()
        .map(|line| serde_json::from_str::<Json>(line).unwrap())
        .flat_map(|request| request["resourceSpans"].as_array().unwrap().clone())
        .flat_map(|resource| resource["scopeSpans"].as_array().unwrap().clone())
        .flat_map(|scope| scope["spans"].as_array().unwrap().clone())
        .map(|span| span["name"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        ["first"],
        "the first span must survive the startup message: {contents}"
    );
}

fn assert_config_builder_overrides(config: &OpenTelemetryConfig) {
    let settings = config;
    assert_eq!(settings.transport, OtlpTransport::HttpBinary);
    assert_eq!(settings.endpoint, "http://localhost:4318/v1/traces");
    assert_eq!(
        settings.headers.get("authorization"),
        Some(&"Bearer token".into())
    );
    assert_eq!(
        settings.header_env.get("x-api-key"),
        Some(&"NEMO_RELAY_TEST_API_KEY".into())
    );
    assert_eq!(
        config
            .shared
            .resource_attributes
            .get("deployment.environment"),
        Some(&"test".into())
    );
    assert_eq!(config.shared.service_name.as_deref(), Some("demo-agent"));
    assert_eq!(config.shared.service_namespace.as_deref(), Some("agents"));
    assert_eq!(config.shared.service_version.as_deref(), Some("1.2.3"));
    assert_eq!(config.shared.instrumentation_scope, "demo-scope");
    assert_eq!(config.shared.mark_projection, MarkProjection::Tool);
    assert_eq!(config.shared.mark_exclude_names, vec!["notification"]);
    assert_eq!(config.shared.attribute_mappings.len(), 1);
    assert_eq!(config.timeout, Duration::from_millis(1250));
}

fn assert_config_defaults(defaults: &OpenTelemetryConfig) {
    assert_eq!(defaults.transport, OtlpTransport::HttpBinary);
    assert_eq!(defaults.shared.service_name, None);
    assert_eq!(defaults.shared.instrumentation_scope, "opentelemetry");
    assert_eq!(defaults.shared.mark_projection, MarkProjection::Inherit);
    assert_eq!(defaults.shared.mark_exclude_names, vec!["llm.chunk"]);
    assert_eq!(defaults.timeout, Duration::from_secs(3));
    assert!(defaults.headers.is_empty());
    assert!(defaults.shared.resource_attributes.is_empty());
}

fn assert_session_mark_identity(spans: &[opentelemetry_sdk::trace::SpanData]) {
    let root = finished_span_named(spans, "identity-root");
    let orphan_mark = finished_span_named(spans, "mark:session.start");
    let root_attributes = attr_map(&root.attributes);
    let mark_attributes = attr_map(&root.events.events[0].attributes);
    assert_eq!(mark_attributes["session.id"], "logical-session");
    assert_eq!(mark_attributes["user.id"], "alice");
    assert_eq!(mark_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        mark_attributes["nemo_relay.session.instance_id"],
        root_attributes["nemo_relay.session.instance_id"]
    );
    let orphan_attributes = attr_map(&orphan_mark.attributes);
    assert_eq!(orphan_attributes["session.id"], "logical-session");
    assert_eq!(orphan_attributes["user.id"], "alice");
    assert_eq!(orphan_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        orphan_attributes["nemo_relay.session.instance_id"],
        root_attributes["nemo_relay.session.instance_id"]
    );
}

fn assert_session_root_and_child_identity(
    spans: &[opentelemetry_sdk::trace::SpanData],
    instance_id: &str,
    second_instance_id: &str,
) {
    let root = finished_span_named(spans, "identity-root");
    let child = finished_span_named(spans, "identity-child");
    let second_root = finished_span_named(spans, "identity-second-root");

    let root_attributes = attr_map(&root.attributes);
    assert_eq!(root_attributes["session.id"], "logical-session");
    assert_eq!(root_attributes["user.id"], "alice");
    assert_eq!(root_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        root_attributes["nemo_relay.session.instance_id"],
        instance_id
    );
    assert_eq!(
        root.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "session.id")
            .count(),
        1
    );
    let child_attributes = attr_map(&child.attributes);
    assert!(!child_attributes.contains_key("session.id"));
    assert!(!child_attributes.contains_key("user.id"));
    assert!(!child_attributes.contains_key("nemo_relay.session.instance_id"));
    assert!(!child_attributes.contains_key("nemo_relay.agent.kind"));
    assert_eq!(
        child_attributes["nemo_relay.start.metadata.user_id"],
        "alice"
    );
    let second_root_attributes = attr_map(&second_root.attributes);
    assert_eq!(second_root_attributes["session.id"], "logical-session");
    assert!(!second_root_attributes.contains_key("user.id"));
    assert_eq!(
        second_root_attributes["nemo_relay.start.metadata.user_id"],
        "42"
    );
    assert_eq!(
        second_root_attributes["nemo_relay.session.instance_id"],
        second_instance_id
    );
    assert_ne!(
        root.span_context.trace_id(),
        second_root.span_context.trace_id()
    );
}

fn assert_telemetry_sdk_resource(attributes: &[OtlpKeyValue]) {
    assert_eq!(
        otlp_string_attribute(attributes, "telemetry.sdk.name"),
        Some("opentelemetry")
    );
    assert_eq!(
        otlp_string_attribute(attributes, "telemetry.sdk.language"),
        Some("rust")
    );
    assert_eq!(
        otlp_string_attribute(attributes, "telemetry.sdk.version"),
        Some("0.33.1")
    );
}

fn attr_map(attributes: &[KeyValue]) -> HashMap<String, String> {
    attributes
        .iter()
        .map(|attribute| {
            (
                attribute.key.as_str().to_string(),
                attribute.value.to_string(),
            )
        })
        .collect()
}

fn finished_span_named<'a>(
    spans: &'a [opentelemetry_sdk::trace::SpanData],
    name: &str,
) -> &'a opentelemetry_sdk::trace::SpanData {
    spans
        .iter()
        .find(|span| span.name.as_ref() == name)
        .unwrap_or_else(|| panic!("missing span {name}"))
}

fn has_promoted_resource_metadata(attributes: &[OtlpKeyValue]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.key.ends_with(".metadata.tenant.id")
            || attribute.key == "tenant.id"
            || matches!(
                attribute
                    .value
                    .as_ref()
                    .and_then(|value| value.value.as_ref()),
                Some(any_value::Value::StringValue(value)) if value == "child-tenant"
            )
    })
}

fn header_value(headers_text: &str, header_name: &str) -> Option<String> {
    headers_text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case(header_name)
            .then(|| value.trim().to_string())
    })
}

fn make_end_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    output: Option<Json>,
) -> Event {
    make_scope_event(
        ScopeCategory::End,
        uuid,
        parent_uuid,
        name,
        scope_type,
        output,
    )
}

fn make_mark_event(parent_uuid: Option<Uuid>, name: &str, data: Option<Json>) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data_opt(data)
            .build(),
        None,
        None,
    ))
}

fn make_mark_event_with_metadata(parent_uuid: Option<Uuid>, metadata: Json) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name("session.start")
            .metadata(metadata)
            .build(),
        None,
        None,
    ))
}

fn make_provider() -> (
    SdkTracerProvider,
    opentelemetry_sdk::trace::InMemorySpanExporter,
) {
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_id_generator(RelayIdGenerator)
        .with_simple_exporter(exporter.clone())
        .build();
    (provider, exporter)
}

fn make_scope_event(
    scope_category: ScopeCategory,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    data: Option<Json>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .data_opt(data)
            .build(),
        scope_category,
        Vec::new(),
        EventCategory::from(scope_type),
        None,
    ))
}

fn make_scope_event_with_profile(
    scope_category: ScopeCategory,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    data: Option<Json>,
    category_profile: Option<CategoryProfile>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .data_opt(data)
            .build(),
        scope_category,
        Vec::new(),
        EventCategory::from(scope_type),
        category_profile,
    ))
}

fn make_start_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    input: Option<Json>,
) -> Event {
    make_scope_event(
        ScopeCategory::Start,
        uuid,
        parent_uuid,
        name,
        scope_type,
        input,
    )
}

fn make_start_event_with_metadata(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    metadata: Json,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .metadata(metadata)
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ))
}

fn otlp_string_attribute<'a>(attributes: &'a [OtlpKeyValue], key: &str) -> Option<&'a str> {
    attributes
        .iter()
        .find(|attribute| attribute.key == key)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|value| match value.value.as_ref() {
            Some(any_value::Value::StringValue(value)) => Some(value.as_str()),
            _ => None,
        })
}

fn read_http_body(
    stream: &mut impl Read,
    bytes: &mut Vec<u8>,
    buf: &mut [u8; 4096],
    expected_len: usize,
) {
    while bytes.len() < expected_len {
        let read = stream.read(buf).unwrap();
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..read]);
    }
}

fn read_http_headers(
    stream: &mut impl Read,
    bytes: &mut Vec<u8>,
    buf: &mut [u8; 4096],
) -> (usize, usize) {
    loop {
        let read = stream.read(buf).unwrap();
        if read == 0 {
            panic!("collector closed before receiving an OTLP request");
        }
        bytes.extend_from_slice(&buf[..read]);

        if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let header_end = header_end + 4;
            let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = header_value(&headers_text, "content-length")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            return (header_end, content_length);
        }
    }
}

fn read_http_request(stream: &mut impl Read) -> CapturedHttpRequest {
    let mut bytes = Vec::new();
    let mut buf = [0_u8; 4096];
    let (header_end, content_length) = read_http_headers(stream, &mut bytes, &mut buf);
    read_http_body(stream, &mut bytes, &mut buf, header_end + content_length);

    let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
    let request_line = headers_text.lines().next().unwrap();
    CapturedHttpRequest {
        path: request_line.split_whitespace().nth(1).unwrap().to_string(),
        content_type: header_value(&headers_text, "content-type").unwrap_or_default(),
        authorization: header_value(&headers_text, "authorization"),
        body: bytes[header_end..header_end + content_length].to_vec(),
    }
}

fn spawn_http_collector(listener: TcpListener, request_tx: mpsc::Sender<CapturedHttpRequest>) {
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_http_request(&mut stream);
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        request_tx.send(request).unwrap();
    });
}

fn test_tokio_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap()
}

fn with_propagation_root(mut event: Event, root_uuid: Uuid) -> Event {
    event.set_propagation_root_uuid(Some(root_uuid));
    event
}

#[derive(Clone, Debug, Default)]
struct FailingThenRecoveringSpanExporter {
    attempts: Arc<AtomicUsize>,
}

impl SpanExporter for FailingThenRecoveringSpanExporter {
    async fn export(&self, _batch: Vec<SpanData>) -> OTelSdkResult {
        if self.attempts.fetch_add(1, Ordering::Relaxed) == 0 {
            Err(OTelSdkError::InternalFailure(
                "collector unavailable".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SensitiveFailingSpanExporter;

impl SpanExporter for SensitiveFailingSpanExporter {
    async fn export(&self, _batch: Vec<SpanData>) -> OTelSdkResult {
        Err(OTelSdkError::InternalFailure(
            "Authorization: Bearer exporter-secret".to_string(),
        ))
    }
}

#[derive(Debug)]
struct RecoveringTraceHttpClient {
    first_error: rustls::Error,
    attempts: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl opentelemetry_http::HttpClient for RecoveringTraceHttpClient {
    async fn send_bytes(
        &self,
        _request: opentelemetry_http::Request<opentelemetry_http::Bytes>,
    ) -> std::result::Result<
        opentelemetry_http::Response<opentelemetry_http::Bytes>,
        opentelemetry_http::HttpError,
    > {
        if self.attempts.fetch_add(1, Ordering::Relaxed) == 0 {
            Err(std::io::Error::other(self.first_error.clone()).into())
        } else {
            Ok(opentelemetry_http::Response::new(Default::default()))
        }
    }
}

#[derive(Clone, Debug)]
struct AlwaysFailingSpanExporter;

impl SpanExporter for AlwaysFailingSpanExporter {
    async fn export(&self, _batch: Vec<SpanData>) -> OTelSdkResult {
        Err(OTelSdkError::InternalFailure(
            "collector unavailable".to_string(),
        ))
    }
}

// Shared transport fixture retains fields used by the runtime integration tests.
#[allow(dead_code)]
struct CapturedHttpRequest {
    path: String,
    content_type: String,
    authorization: Option<String>,
    body: Vec<u8>,
}
