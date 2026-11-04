// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_types::api::event::*;
use nemo_relay_types::api::scope::ScopeType;
use nemo_relay_types::codec::response::*;
use serde_json::{Value as Json, json};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[test]
fn provider_errors_identify_their_telemetry_signal() {
    for (error, expected) in [
        (
            OpenTelemetryError::TraceProvider("trace failure".to_string()),
            "OpenTelemetry tracer provider error: trace failure",
        ),
        (
            OpenTelemetryError::LogProvider("log failure".to_string()),
            "OpenTelemetry log provider error: log failure",
        ),
        (
            OpenTelemetryError::MetricProvider("metric failure".to_string()),
            "OpenTelemetry metric provider error: metric failure",
        ),
    ] {
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn default_trace_config_leaves_service_name_to_sdk_resource_detection() {
    let config = OpenTelemetryConfig::new(OpenTelemetryType::Full, "http://localhost:4318");
    assert!(
        configured_resource_attributes(&TraceConfig::Endpoint(config.clone()))
            .iter()
            .all(|attribute| attribute.key.as_str() != "service.name")
    );

    let configured = config.with_service_name("relay-configured-service");
    assert!(
        configured_resource_attributes(&TraceConfig::Endpoint(configured.clone()))
            .iter()
            .any(|attribute| {
                attribute.key.as_str() == "service.name"
                    && attribute.value.as_str() == "relay-configured-service"
            })
    );
}

#[test]
fn optimization_summary_emits_namespaced_otel_attributes() {
    let summary: nemo_relay_types::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"complete",
            "baseline_model":{"model":"baseline"}, "effective_model":{"model":"effective"},
            "tokens_saved":{"prompt_tokens":12,"total_tokens":12},
            "baseline_cost":{"total":0.02,"currency":"USD","source":"model_pricing","pricing_as_of":"2026-07-08","pricing_source":"test"},
            "actual_cost":{"total":0.01,"currency":"USD","source":"model_pricing"},
            "estimated_cost_saved":0.01, "currency":"USD", "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_model"],
        "baseline"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.prompt_tokens_saved"],
        "12"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.estimated_cost_saved"],
        "0.01"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_as_of"],
        "2026-07-08"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_cost_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.estimated_cost_saved_currency"],
        "USD"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_pricing_source"],
        "test"
    );
}

#[test]
fn optimization_cost_attributes_preserve_independent_currency_and_provenance() {
    let summary: nemo_relay_types::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"partial",
            "limitations":["cost_currency_mismatch"],
            "tokens_saved":{},
            "baseline_cost":{"total":2.0,"currency":"EUR","source":"model_pricing","pricing_as_of":"2026-01-01"},
            "actual_cost":{"total":1.0,"currency":"GBP","source":"provider_reported","pricing_as_of":"2026-02-02","pricing_source":"provider"},
            "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);

    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_cost_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost_currency"],
        "GBP"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_source"],
        "provider"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_as_of"],
        "2026-02-02"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_source"],
        "provider"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.pricing_as_of"],
        "2026-01-01"
    );
    assert!(!attributes.contains_key("nemo_relay.llm.optimization.estimated_cost_saved"));
    assert!(!attributes.contains_key("nemo_relay.llm.optimization.estimated_cost_saved_currency"));
}

#[test]
fn complete_non_usd_optimization_costs_keep_currency_and_independent_provenance() {
    let summary: nemo_relay_types::codec::optimization::LlmOptimizationSummary =
        serde_json::from_value(json!({
            "schema_version":"1", "calculation_version":"1", "status":"complete",
            "tokens_saved":{},
            "baseline_cost":{"total":2.0,"currency":"EUR","source":"model_pricing","pricing_as_of":"2026-01-01","pricing_source":"baseline-catalog"},
            "actual_cost":{"total":1.0,"currency":"EUR","source":"provider_reported","pricing_as_of":"2026-02-02","pricing_source":"provider-meter"},
            "estimated_cost_saved":1.0, "currency":"EUR", "contributions":[]
        }))
        .unwrap();
    let mut attributes = Vec::new();
    push_optimization_attributes(&mut attributes, &summary);
    let attributes = attr_map(&attributes);

    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_cost_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_cost_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.estimated_cost_saved_currency"],
        "EUR"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_pricing_source"],
        "baseline-catalog"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_source"],
        "provider-meter"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.baseline_pricing_as_of"],
        "2026-01-01"
    );
    assert_eq!(
        attributes["nemo_relay.llm.optimization.actual_pricing_as_of"],
        "2026-02-02"
    );
}

#[test]
fn http_trace_endpoint_resolution_preserves_an_explicit_root_path() {
    for (endpoint, expected) in [
        ("http://localhost:4318", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318/", "http://localhost:4318/"),
        (
            "https://collector.example?tenant=one",
            "https://collector.example/v1/traces?tenant=one",
        ),
        (
            "https://collector.example/?tenant=one",
            "https://collector.example/?tenant=one",
        ),
        (
            "https://collector.example/#root",
            "https://collector.example/#root",
        ),
        (
            "http://localhost:4318/v1/traces",
            "http://localhost:4318/v1/traces",
        ),
        (
            "http://collector.example/custom-ingest",
            "http://collector.example/custom-ingest",
        ),
        ("not a URL", "not a URL"),
    ] {
        assert_eq!(resolve_http_trace_endpoint(endpoint), expected);
    }
}

#[test]
fn invalid_grpc_headers_are_rejected() {
    let err = build_grpc_metadata(&HashMap::from([(
        "bad key".to_string(),
        "value".to_string(),
    )]))
    .expect_err("invalid metadata key should fail");
    assert!(matches!(err, OpenTelemetryError::InvalidGrpcHeader { .. }));
}

#[test]
fn direct_config_rejects_process_global_otel_headers() {
    const CHILD_MARKER: &str = "NEMO_RELAY_TEST_GLOBAL_OTEL_HEADER_CHILD";
    if let Ok(variable) = std::env::var(CHILD_MARKER) {
        assert!(matches!(
            reject_global_header_environment(),
            Err(OpenTelemetryError::GlobalHeaderEnvironmentUnsupported {
                variable: rejected
            }) if rejected == variable
        ));
        return;
    }

    for variable in [
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(std::thread::current().name().unwrap())
            .env(CHILD_MARKER, variable)
            .env(variable, "authorization=secret")
            .env_remove(if variable == "OTEL_EXPORTER_OTLP_HEADERS" {
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
            } else {
                "OTEL_EXPORTER_OTLP_HEADERS"
            })
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("test result: ok. 1 passed"),
            "child test filter did not execute exactly one test: {stdout}"
        );
    }
}

#[test]
fn trace_endpoints_allow_plain_http_only_to_listed_insecure_hosts() {
    let trusted = PlaintextPolicy::from_hosts(&["host.docker.internal".to_string()]).unwrap();
    assert!(
        validate_trace_endpoint("http://host.docker.internal:8082/v1/traces", &trusted).is_ok()
    );
    let error = validate_trace_endpoint("http://collector.example/v1/traces", &trusted)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("hosts listed in insecure_http_hosts"),
        "{error}"
    );
}

#[test]
fn pre_epoch_timestamps_round_trip_through_system_time() {
    let timestamp = DateTime::parse_from_rfc3339("1969-12-31T23:59:58.500000000Z")
        .unwrap()
        .with_timezone(&Utc);

    assert_eq!(
        to_system_time(timestamp),
        UNIX_EPOCH - Duration::new(1, 500_000_000)
    );
}

#[test]
fn tls_failure_classification_uses_typed_causes_without_exposing_error_strings() {
    let error = std::io::Error::other(rustls::Error::InvalidCertificate(
        rustls::CertificateError::UnknownIssuer,
    ));
    assert_eq!(
        safe_tls_failure_reason(&error),
        Some("TLS certificate verification failed: unknown issuer; check the configured CA bundle")
    );
    let sensitive = std::io::Error::other("TLS certificate Authorization: Bearer secret");
    assert_eq!(safe_tls_failure_reason(&sensitive), None);
}

#[test]
fn trace_endpoint_log_identity_redacts_and_validates_urls() {
    for (endpoint, expected) in [
        (
            AUTOMATIC_OTLP_ENDPOINT_MARKER,
            "the environment-configured OTLP endpoint",
        ),
        ("not a URL", "an invalid OTLP endpoint"),
        ("ftp://collector.example/secret", "an invalid OTLP endpoint"),
        (
            "https://[::1]:4318/private?access_token=url-secret",
            "https://[::1]:4318",
        ),
        (
            "https://user:password@collector.example:4318/private?access_token=url-secret#fragment-secret",
            "https://collector.example:4318",
        ),
    ] {
        assert_eq!(trace_endpoint_log_identity(endpoint), expected);
    }
}

#[test]
fn delivery_identity_sanitizes_endpoints_and_preserves_file_paths() {
    let endpoint = TraceConfig::Endpoint(OpenTelemetryConfig::new(
        OpenTelemetryType::Full,
        "https://user:secret@collector.example:4318/v1/traces?token=abc",
    ));
    // An endpoint keeps scheme, host and port only: no credentials, no query.
    let identity = endpoint.delivery_identity();
    assert_eq!(identity, "https://collector.example:4318");
    assert!(!identity.contains("secret"));
    assert!(!identity.contains("token"));

    let sink = OtlpFileSinkSettings {
        output_directory: PathBuf::from("/var/log/nemo-relay"),
        filename: "trace.jsonl".to_string(),
        format: crate::otel_file::OtlpFileFormat::JsonLines,
        mode: crate::otel::OtlpFileSinkMode::Overwrite,
    };
    // A path is not a URL; sanitizing it as one would report
    // "an invalid OTLP endpoint" and hide which destination failed.
    let file = TraceConfig::File(OpenTelemetryFileSinkConfig::new(
        OpenTelemetryType::Full,
        sink,
    ));
    assert_eq!(
        file.delivery_identity(),
        Path::new("/var/log/nemo-relay")
            .join("trace.jsonl")
            .display()
            .to_string()
    );
}

#[test]
fn gen_ai_client_metrics_include_response_model_and_server_dimensions() {
    let event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(
            json!({"provider": "openai", "model": "requested", "server_address": "api.example.com", "server_port": 443}),
        ),
        Some(
            CategoryProfile::builder()
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    model: Some("served".into()),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    );
    assert_eq!(
        crate::otel_genai::client_metric_attributes(&crate::normalization::EventView::new(
            &event,
            &crate::host::DetachedHost
        )),
        Some(json!({
            "gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "requested", "gen_ai.response.model": "served",
            "server.address": "api.example.com", "server.port": 443
        }))
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

fn empty_annotated_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: None,
        model: None,
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: serde_json::Map::new(),
    }
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
