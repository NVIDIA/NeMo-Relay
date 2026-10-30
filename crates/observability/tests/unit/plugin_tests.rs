// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[test]
fn otel_endpoint_editor_defaults_are_typed_and_expose_editable_header_sources() {
    let trace: OpenTelemetryEndpointConfig =
        serde_json::from_value(default_opentelemetry_endpoint_editor_value()).unwrap();
    let signal: OpenTelemetrySignalEndpointConfig =
        serde_json::from_value(default_opentelemetry_signal_endpoint_editor_value()).unwrap();
    assert_eq!(trace.service_name, "unknown_service");
    assert_eq!(signal.service_name, trace.service_name);
    assert_eq!(signal.timeout_millis, 3000);
    for schema in [
        OpenTelemetryEndpointConfig::editor_schema(),
        OpenTelemetrySignalEndpointConfig::editor_schema(),
    ] {
        for name in [
            "headers",
            "header_env",
            "header_file",
            "resource_attributes",
        ] {
            assert_eq!(schema.field(name).unwrap().kind, EditorFieldKind::StringMap);
        }
        assert_eq!(
            schema.field("transport").unwrap().enum_values,
            &["http_binary", "grpc"]
        );
        assert!(schema.field("service_namespace").unwrap().optional);
    }
    let excluded = OpenTelemetryEndpointConfig::editor_schema()
        .field("mark_exclude_names")
        .unwrap();
    assert_eq!(excluded.list_item.unwrap().kind, EditorFieldKind::String);
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
    assert_eq!(
        schema
            .field("insecure_http_hosts")
            .expect("insecure HTTP hosts field")
            .kind,
        EditorFieldKind::List
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
            "header_file": {},
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
    trace.promote_resource_metadata_prefixes =
        vec!["nv.client.".to_string(), "nv.env.".to_string()];

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
    assert_eq!(
        logs.endpoints[0]
            .value
            .as_active()
            .unwrap()
            .promote_resource_metadata_prefixes,
        trace.promote_resource_metadata_prefixes
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
        tls: None,
        endpoint: "https://collector.example/custom/metric-intake".to_string(),
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
        tls: None,
        endpoint: "https://collector.example/v1/traces".to_string(),
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

#[test]
fn all_trace_plugin_endpoints_reject_remote_plaintext() {
    for (endpoint, allowed) in [
        ("http://collector.example:4318", false),
        ("https://collector.example:4318", true),
        ("http://127.0.0.1:4318", true),
        ("http://[::1]:4318", true),
    ] {
        for otel_type in [
            OpenTelemetryType::Full,
            OpenTelemetryType::GenAi,
            OpenTelemetryType::OpenInference,
        ] {
            for transport in ["http_binary", "grpc"] {
                let mut section = test_opentelemetry_endpoint();
                section.otel_type = otel_type;
                section.transport = transport.to_string();
                section.endpoint = endpoint.to_string();
                assert_eq!(
                    build_otel_config(0, section, &PlaintextPolicy::default()).is_ok(),
                    allowed,
                    "{endpoint}"
                );
            }
        }
    }
}

#[test]
fn trace_endpoint_completed_context_ttl_is_applied_and_must_be_positive() {
    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.completed_span_context_ttl_millis = Some(750);
    let config = build_otel_config(0, endpoint, &PlaintextPolicy::default()).unwrap();
    assert_eq!(
        config.completed_span_context_ttl(),
        Duration::from_millis(750)
    );

    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.completed_span_context_ttl_millis = Some(0);
    assert!(
        build_otel_config(0, endpoint, &PlaintextPolicy::default())
            .unwrap_err()
            .to_string()
            .contains("completed_span_context_ttl_millis must be greater than 0")
    );
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
fn build_otel_config_carries_endpoint_batch_overrides() {
    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.max_queue_size = Some(4096);
    endpoint.max_export_batch_size = Some(256);
    endpoint.scheduled_delay_millis = Some(750);
    let config = build_otel_config(0, endpoint, &PlaintextPolicy::default()).unwrap();
    assert_eq!(
        config.batch_overrides(),
        (Some(4096), Some(256), Some(Duration::from_millis(750)))
    );

    let config = build_otel_config(
        1,
        test_opentelemetry_endpoint(),
        &PlaintextPolicy::default(),
    )
    .unwrap();
    assert_eq!(config.batch_overrides(), (None, None, None));
}

#[test]
fn build_otel_config_carries_resource_metadata_promotion_prefixes() {
    let mut endpoint = test_opentelemetry_endpoint();
    endpoint.promote_resource_metadata_prefixes = vec!["deployment.".to_string()];

    let config = build_otel_config(0, endpoint, &PlaintextPolicy::default()).unwrap();

    assert_eq!(
        config.promote_resource_metadata_prefixes(),
        ["deployment.".to_string()]
    );
}

#[test]
fn typed_insecure_http_hosts_serialize_an_explicit_empty_list() {
    let omitted = serde_json::to_value(ObservabilityConfig::default()).unwrap();
    assert!(omitted.get("insecure_http_hosts").is_none());
    let cleared = serde_json::to_value(ObservabilityConfig {
        insecure_http_hosts: Some(Vec::new()),
        ..ObservabilityConfig::default()
    })
    .unwrap();
    assert_eq!(cleared.get("insecure_http_hosts"), Some(&json!([])));
}

#[cfg(feature = "object-store")]
#[test]
fn atif_http_storage_headers_follow_the_plaintext_policy() {
    let mut config = http_storage_config("http://release-api:8080/atif");
    config
        .headers
        .insert("authorization".to_string(), "Bearer token".to_string());
    assert!(HttpUploadConfig::resolve(0, &config, &PlaintextPolicy::LoopbackOnly).is_err());
    let listed = PlaintextPolicy::from_hosts(&["release-api".to_string()]).unwrap();
    assert!(HttpUploadConfig::resolve(0, &config, &listed).is_ok());
}

#[cfg(feature = "schema")]
#[test]
fn the_file_sink_schema_advertises_the_overwrite_default() {
    let schema = serde_json::to_value(schemars::schema_for!(OpenTelemetryFileSinkConfig)).unwrap();

    // The ATOF event sink defaults to `append`; a trace file sink does not, and
    // a generated schema that says otherwise misleads every editor reading it.
    assert!(schema_property_has_enum(
        &schema,
        "mode",
        &["append", "overwrite"]
    ));
    assert!(schema_property_has_default(
        &schema,
        "mode",
        json!("overwrite")
    ));
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
            header_file: std::collections::HashMap::new(),
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
        crate::atof::AtofEndpointTransport::Websocket
    );
    assert_eq!(config.headers, headers);
    assert_eq!(
        config.header_env.get("x-api-key").map(String::as_str),
        Some("SWITCHYARD_API_KEY")
    );
    assert_eq!(config.timeout_millis, 123);
    assert_eq!(
        config.field_name_policy,
        crate::atof::AtofEndpointFieldNamePolicy::ReplaceDots
    );

    let error = build_atof_sink_config(
        3,
        AtofSinkSectionConfig::Stream(AtofStreamSinkSectionConfig {
            name: None,
            url: "http://127.0.0.1:47632/events".into(),
            transport: "smtp".into(),
            headers: std::collections::HashMap::new(),
            header_env: std::collections::HashMap::new(),
            header_file: std::collections::HashMap::new(),
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
            header_file: std::collections::HashMap::new(),
            timeout_millis: 3_000,
            field_name_policy: "bogus".into(),
        }),
    )
    .unwrap_err();
    assert!(error.to_string().contains("sinks[4].field_name_policy"));
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
fn atif_payload_merges_correlation_with_existing_trajectory_extra() {
    let agent_uuid = Uuid::now_v7();
    let trajectory = crate::atif::AtifTrajectory {
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
        AtifOutputOptions::from(&AtifSectionConfig::default()),
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
fn atif_retention_defaults_preserve_existing_payloads() {
    let deserialized: AtifSectionConfig = serde_json::from_value(json!({})).unwrap();
    for config in [AtifSectionConfig::default(), deserialized] {
        assert!(config.include_observed_events);
        assert!(config.include_llm_requests);
    }
}

#[test]
fn atif_compact_payload_preserves_json_and_reduced_history_grows_linearly() {
    let payload = |calls: usize, compact: bool, include_llm_requests: bool| {
        let mut history = Vec::new();
        let mut steps = Vec::new();
        for index in 0..calls {
            history.push(json!({"role": "user", "content": "x".repeat(2048)}));
            steps.push(json!({
                "step_id": index + 1,
                "source": "user",
                "message": "x".repeat(2048),
                "extra": {
                    "llm_request": {"messages": history},
                    "ancestry": {"function_id": index.to_string(), "function_name": "llm"}
                }
            }));
        }
        let trajectory = serde_json::from_value(json!({
            "schema_version": "ATIF-v1.7",
            "session_id": "size-test",
            "agent": {"name": "size-test", "version": "1"},
            "steps": steps
        }))
        .unwrap();
        let write = prepare_atif_payload(
            Uuid::nil(),
            "trajectory.json".into(),
            None,
            None,
            trajectory,
            Vec::new(),
            AtifCorrelation {
                session_id: None,
                session_instance_id: None,
                user_id: None,
            },
            AtifOutputOptions {
                include_observed_events: false,
                include_llm_requests,
                compact,
            },
        )
        .unwrap();
        write.payload
    };
    let compact = payload(8, true, true);
    let pretty = payload(8, false, true);
    assert!(compact.len() < pretty.len());
    assert_eq!(
        serde_json::from_slice::<Json>(&compact).unwrap(),
        serde_json::from_slice::<Json>(&pretty).unwrap()
    );
    let full_large = payload(32, true, true);
    let reduced_small = payload(8, true, false);
    let reduced_large = payload(32, true, false);
    assert!(
        full_large.len() > compact.len() * 10,
        "full histories accumulate quadratically"
    );
    assert!(
        reduced_large.len() < reduced_small.len() * 5,
        "omitting histories keeps this fixture linear"
    );
    assert!(reduced_large.len() * 10 < full_large.len());
}

#[test]
fn signal_delivery_state_classifies_generic_sdk_shutdown_error_as_delivery() {
    let issue = signal_shutdown_issue(
        Err(crate::otel::OpenTelemetryError::MetricProvider(
            "generic SDK final-export failure".to_string(),
        )),
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

#[test]
fn a_file_sink_section_parses_with_only_an_output_directory() {
    let section: OpenTelemetryFileSinkConfig =
        toml::from_str("output_directory = \"/tmp/relay-traces\"").unwrap();

    assert_eq!(section.format, OtlpFileFormat::JsonLines);
    assert_eq!(section.mode, "overwrite");
    assert_eq!(section.otel_type, OpenTelemetryType::Full);
    assert!(section.filename.is_none());
}

#[test]
fn a_file_sink_section_parses_the_proto_format() {
    let section: OpenTelemetryFileSinkConfig =
        toml::from_str("output_directory = \"/tmp/relay-traces\"\nformat = \"proto\"").unwrap();

    assert_eq!(section.format, OtlpFileFormat::Proto);
}

#[test]
fn a_file_sink_config_resolves_its_path_under_the_output_directory() {
    let directory = tempfile::tempdir().unwrap();
    let config = build_otel_file_config(0, file_sink_section(directory.path())).unwrap();

    let settings = config.sink();
    assert_eq!(settings.path(), directory.path().join("trace.jsonl"));
    assert_eq!(settings.output_directory, directory.path());
    assert_eq!(
        settings.mode,
        crate::otel::OtlpFileSinkMode::Overwrite,
        "overwrite mode must not append"
    );
}

#[test]
fn a_file_sink_without_a_filename_names_the_file_after_its_format() {
    let directory = tempfile::tempdir().unwrap();
    for (format, extension) in [
        (OtlpFileFormat::JsonLines, ".jsonl"),
        (OtlpFileFormat::Proto, ".otlp.pb"),
    ] {
        let mut section = file_sink_section(directory.path());
        section.filename = None;
        section.format = format;
        let config = build_otel_file_config(0, section).unwrap();

        let settings = config.sink();
        let path = settings.path().display().to_string();
        assert!(
            path.ends_with(extension),
            "unexpected default filename {path}"
        );
    }
}

#[test]
fn a_file_sink_rejects_a_blank_output_directory() {
    let mut section = file_sink_section(Path::new(""));
    section.filename = None;
    let error = build_otel_file_config(3, section).unwrap_err();

    assert!(error.to_string().contains("file_sinks[3].output_directory"));
}

#[test]
fn a_file_sink_rejects_an_unknown_mode() {
    let directory = tempfile::tempdir().unwrap();
    let mut section = file_sink_section(directory.path());
    section.mode = "truncate".to_string();
    let error = build_otel_file_config(1, section).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("must be 'append' or 'overwrite'")
    );
}

#[test]
fn a_file_sink_accepts_append_mode() {
    let directory = tempfile::tempdir().unwrap();
    let mut section = file_sink_section(directory.path());
    section.mode = "append".to_string();
    let config = build_otel_file_config(0, section).unwrap();

    let settings = config.sink();
    assert_eq!(settings.mode, crate::otel::OtlpFileSinkMode::Append);
}

#[test]
fn a_file_sink_rejects_a_filename_that_leaves_its_directory() {
    let directory = tempfile::tempdir().unwrap();
    for filename in ["../escape.jsonl", "nested/trace.jsonl", "/absolute.jsonl"] {
        let mut section = file_sink_section(directory.path());
        section.filename = Some(filename.to_string());
        let error = build_otel_file_config(2, section).unwrap_err();

        assert!(
            error.to_string().contains("single path component"),
            "{filename} should be rejected, got {error}"
        );
    }
}

#[test]
fn a_file_sink_rejects_a_blank_or_padded_filename() {
    let directory = tempfile::tempdir().unwrap();
    for filename in ["", "   ", " trace.jsonl"] {
        let mut section = file_sink_section(directory.path());
        section.filename = Some(filename.to_string());
        let error = build_otel_file_config(0, section).unwrap_err();

        assert!(error.to_string().contains("nonblank and unpadded"));
    }
}

#[test]
fn a_file_sink_rejects_a_whitespace_only_output_directory() {
    let mut section = file_sink_section(Path::new(" "));

    // `OtlpFileSinkSettings::from_parts` trims the value, so a whitespace-only
    // directory the plugin accepted would be rejected by every binding, and
    // Relay would otherwise create a directory named " ".
    let error = build_otel_file_config(0, section.clone()).unwrap_err();
    assert!(error.to_string().contains("nonblank path"), "{error}");

    section.filename = Some("trace.jsonl".to_string());
    let mut diagnostics = Vec::new();
    validate_opentelemetry_section(
        &mut diagnostics,
        &ConfigPolicy::default(),
        &OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![section],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
    );
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some("file_sinks[0].output_directory")
        }),
        "expected a static diagnostic: {diagnostics:?}"
    );
}

#[test]
fn a_file_sink_rejects_a_filename_that_is_a_directory_reference() {
    let directory = tempfile::tempdir().unwrap();
    for filename in [".", ".."] {
        let mut section = file_sink_section(directory.path());
        section.filename = Some(filename.to_string());

        // Rejected as configuration rather than reaching the confinement check,
        // which reports an internal failure instead of a bad setting.
        let error = build_otel_file_config(0, section).unwrap_err();
        assert!(
            error.to_string().contains("single path component"),
            "unexpected error for {filename:?}: {error}"
        );
    }
}

#[test]
fn colliding_file_sink_paths_are_rejected_however_the_directory_is_spelled() {
    // Validation never touches the filesystem, so neither directory is created.
    let relative = PathBuf::from("relay-collision-traces");
    let absolute = std::env::current_dir().unwrap().join(&relative);
    let mut first = file_sink_section(&relative);
    first.filename = Some("trace.jsonl".to_string());
    let mut second = file_sink_section(&absolute);
    second.filename = Some("trace.jsonl".to_string());

    // A relative path and its absolute equivalent compare as different keys, so
    // an unnormalized check would accept both and open one file twice.
    let error = validate_distinct_opentelemetry_file_sinks(&[first, second]).unwrap_err();

    assert!(
        error.to_string().contains("write the same path"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_file_sink_rejects_zero_and_inverted_batch_settings() {
    let directory = tempfile::tempdir().unwrap();
    type MutateFileSink = Box<dyn Fn(&mut OpenTelemetryFileSinkConfig)>;
    let cases: Vec<(MutateFileSink, &str)> = vec![
        (Box::new(|s| s.max_queue_size = Some(0)), "max_queue_size"),
        (
            Box::new(|s| s.max_export_batch_size = Some(0)),
            "max_export_batch_size",
        ),
        (
            Box::new(|s| s.scheduled_delay_millis = Some(0)),
            "scheduled_delay_millis",
        ),
        (
            Box::new(|s| s.completed_span_context_ttl_millis = Some(0)),
            "completed_span_context_ttl_millis",
        ),
        (
            Box::new(|s| {
                s.max_queue_size = Some(1);
                s.max_export_batch_size = Some(2);
            }),
            "must be less than or equal to max_queue_size",
        ),
    ];
    for (mutate, expected) in cases {
        let mut section = file_sink_section(directory.path());
        mutate(&mut section);
        let error = build_otel_file_config(0, section).unwrap_err();
        assert!(error.to_string().contains(expected), "got {error}");
    }
}

#[test]
fn two_file_sinks_writing_one_path_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let sinks = vec![
        file_sink_section(directory.path()),
        file_sink_section(directory.path()),
    ];
    let error = validate_distinct_opentelemetry_file_sinks(&sinks).unwrap_err();

    assert!(error.to_string().contains("write the same path"));
}

#[test]
fn file_sinks_writing_distinct_paths_are_accepted() {
    let directory = tempfile::tempdir().unwrap();
    let mut second = file_sink_section(directory.path());
    second.filename = Some("other.jsonl".to_string());
    let sinks = vec![file_sink_section(directory.path()), second];

    validate_distinct_opentelemetry_file_sinks(&sinks).unwrap();
}

#[test]
fn file_sinks_are_offered_by_the_config_editor() {
    let schema = OpenTelemetrySectionConfig::editor_schema();
    let file_sinks = schema.field("file_sinks").expect("file_sinks editor field");
    assert_eq!(file_sinks.kind, EditorFieldKind::List);

    let item_schema = (file_sinks
        .list_item
        .expect("file_sinks list metadata")
        .schema
        .expect("file_sinks item schema"))();
    let format = item_schema.field("format").expect("format editor field");
    assert_eq!(format.enum_values, &["json_lines", "proto"]);
    assert!(
        item_schema.field("endpoint").is_none(),
        "a file sink has no endpoint to configure"
    );
    assert!(
        !item_schema
            .field("output_directory")
            .expect("output_directory editor field")
            .optional,
        "a file sink cannot default its output directory"
    );
}

#[test]
fn the_file_sink_editor_default_is_a_valid_section() {
    let default = default_opentelemetry_file_sink_editor_value();
    let section: OpenTelemetryFileSinkConfig = serde_json::from_value(default).unwrap();

    assert_eq!(section.format, OtlpFileFormat::JsonLines);
    assert_eq!(section.mode, "overwrite");
}

#[test]
fn validate_opentelemetry_section_accepts_a_file_sink_as_the_only_destination() {
    let directory = tempfile::tempdir().unwrap();
    let policy = ConfigPolicy::default();
    let mut diagnostics = Vec::new();

    validate_opentelemetry_section(
        &mut diagnostics,
        &policy,
        &OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![file_sink_section(directory.path())],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
    );

    // Static validation runs before activation, so a section it rejects never
    // reaches `register_opentelemetry` at all.
    assert!(
        diagnostics.is_empty(),
        "a file sink is a destination: {diagnostics:?}"
    );
}

#[test]
fn validate_opentelemetry_section_reports_malformed_file_sinks() {
    let directory = tempfile::tempdir().unwrap();
    let policy = ConfigPolicy::default();
    let mut blank_directory = file_sink_section(directory.path());
    blank_directory.output_directory = PathBuf::new();
    let mut bad_mode = file_sink_section(directory.path());
    bad_mode.mode = "truncate".to_string();
    let mut diagnostics = Vec::new();

    validate_opentelemetry_section(
        &mut diagnostics,
        &policy,
        &OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![blank_directory, bad_mode],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
    );

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("file_sinks[0].output_directory")
    }));
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.field.as_deref() == Some("file_sinks[1].mode"))
    );
}

#[test]
fn a_default_file_sink_service_name_is_left_to_sdk_resource_detection() {
    let directory = tempfile::tempdir().unwrap();
    let section = file_sink_section(directory.path());

    // The endpoint path leaves the default unset so OTEL_SERVICE_NAME and
    // OTEL_RESOURCE_ATTRIBUTES can supply it. A file sink and an endpoint in
    // the same section would otherwise carry different resources.
    let config = build_otel_file_config(0, section).unwrap();
    assert_eq!(config.service_name(), None);

    let mut named = file_sink_section(directory.path());
    named.service_name = "relay-configured-service".to_string();
    assert_eq!(
        build_otel_file_config(0, named).unwrap().service_name(),
        Some("relay-configured-service"),
    );
}

#[test]
fn validate_opentelemetry_section_names_the_batch_field_that_failed() {
    let directory = tempfile::tempdir().unwrap();
    let mut section = file_sink_section(directory.path());
    section.scheduled_delay_millis = Some(0);
    let mut diagnostics = Vec::new();

    validate_opentelemetry_section(
        &mut diagnostics,
        &ConfigPolicy::default(),
        &OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![section],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
    );

    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some("file_sinks[0].scheduled_delay_millis")
        }),
        "the failing field should be named: {diagnostics:?}"
    );
}

#[test]
fn validate_opentelemetry_section_reports_file_sink_mappings_and_prefixes() {
    let directory = tempfile::tempdir().unwrap();
    let mut section = file_sink_section(directory.path());
    section.attribute_mappings = vec![OtlpAttributeMapping {
        key: String::new(),
        alias: "alias".to_string(),
    }];
    section.promote_metadata_prefixes = vec![String::new()];
    section.promote_resource_metadata_prefixes = vec![String::new()];
    let mut diagnostics = Vec::new();

    validate_opentelemetry_section(
        &mut diagnostics,
        &ConfigPolicy::default(),
        &OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![section],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
    );

    // These otherwise fail only at activation, where the sink is skipped with a
    // warning instead of failing configuration.
    for field in [
        "file_sinks[0].attribute_mappings",
        "file_sinks[0].promote_metadata_prefixes",
        "file_sinks[0].promote_resource_metadata_prefixes",
    ] {
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.field.as_deref() == Some(field)),
            "expected a diagnostic for {field}: {diagnostics:?}"
        );
    }
}

#[test]
fn validate_opentelemetry_section_reports_colliding_file_sink_paths() {
    let directory = tempfile::tempdir().unwrap();
    let mut diagnostics = Vec::new();

    validate_opentelemetry_section(
        &mut diagnostics,
        &ConfigPolicy::default(),
        &OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![
                file_sink_section(directory.path()),
                file_sink_section(directory.path()),
            ],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
    );

    // A collision fails the whole section at activation, so it is reported
    // statically as well, like a colliding pair of endpoints.
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "observability.unsafe_otel_destination_collision"
                && diagnostic.field.as_deref() == Some("file_sinks")
        }),
        "expected a collision diagnostic: {diagnostics:?}"
    );
}

#[test]
fn file_sinks_without_filenames_collide_on_the_generated_name() {
    let directory = tempfile::tempdir().unwrap();
    let mut first = file_sink_section(directory.path());
    first.filename = None;
    let mut second = file_sink_section(directory.path());
    second.filename = None;

    // Both fall back to the same format-derived default.
    let error = validate_distinct_opentelemetry_file_sinks(&[first, second]).unwrap_err();

    assert!(error.to_string().contains("write the same path"));
}

#[test]
fn validate_opentelemetry_section_reports_file_sink_filename_and_batch_errors() {
    let directory = tempfile::tempdir().unwrap();
    let policy = ConfigPolicy::default();
    let mut bad_filename = file_sink_section(directory.path());
    bad_filename.filename = Some("../escape.jsonl".to_string());
    let mut bad_batch = file_sink_section(directory.path());
    bad_batch.filename = Some("second.jsonl".to_string());
    bad_batch.max_queue_size = Some(0);
    let mut diagnostics = Vec::new();

    validate_opentelemetry_section(
        &mut diagnostics,
        &policy,
        &OpenTelemetrySectionConfig {
            enabled: true,
            file_sinks: vec![bad_filename, bad_batch],
            endpoints: Vec::new(),
            logs: None,
            metrics: None,
        },
    );

    // Reported before activation, as the endpoint path already does, instead of
    // only as a skipped-sink warning once the plugin starts.
    assert!(
        diagnostics
            .iter()
            .any(|d| d.field.as_deref() == Some("file_sinks[0].filename"))
    );
    assert!(
        diagnostics
            .iter()
            .any(|d| d.field.as_deref() == Some("file_sinks[1].max_queue_size"))
    );
}

#[test]
fn signal_shutdown_distinguishes_delivery_failures_from_provider_failures() {
    use crate::otel::OpenTelemetryError;
    assert!(signal_shutdown_issue(Ok(()), None).is_none());
    let delivery = signal_shutdown_issue(
        Err(OpenTelemetryError::ExporterBuild("provider stopped".into())),
        Some("records lost".into()),
    )
    .unwrap();
    match shutdown_failure_from_errors(vec![delivery]).unwrap() {
        OpenTelemetryShutdownFailure::Delivery(error) => {
            let message = error.to_string();
            assert!(message.contains("records lost"));
            assert!(message.contains("provider stopped"));
        }
        OpenTelemetryShutdownFailure::Other(_) => {
            panic!("delivery failure must retain its classification")
        }
    }
    let provider = signal_shutdown_issue(
        Err(OpenTelemetryError::ExporterBuild("flush failed".into())),
        None,
    )
    .unwrap();
    match shutdown_failure_from_errors(vec![provider]).unwrap() {
        OpenTelemetryShutdownFailure::Other(error) => {
            assert!(error.to_string().contains("flush failed"))
        }
        OpenTelemetryShutdownFailure::Delivery(_) => {
            panic!("provider failure must not imply lost records")
        }
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

#[cfg(feature = "object-store")]
fn http_storage_config(endpoint: impl Into<String>) -> HttpStorageConfig {
    HttpStorageConfig {
        endpoint: endpoint.into(),
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        header_file: std::collections::HashMap::new(),
        timeout_millis: 1_000,
    }
}

#[cfg(feature = "schema")]
fn schema_has_property(schema: &Json, name: &str) -> bool {
    schema_property(schema, name).is_some()
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

#[cfg(feature = "schema")]
fn schema_property_has_default(schema: &Json, name: &str, expected: Json) -> bool {
    schema_property_matches(schema, name, &|property| {
        property
            .get("default")
            .is_some_and(|default| default == &expected)
    })
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

fn temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("nemo-relay-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path.canonicalize().unwrap()
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
