// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_types::api::event::*;
use nemo_relay_types::api::event::{
    BaseEvent, DataSchema, METRIC_DATA_SCHEMA_NAME, MarkEvent, ScopeCategory, ScopeEvent,
};
use nemo_relay_types::api::scope::ScopeType;
use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::common::v1::any_value;
use opentelemetry_sdk::error::OTelSdkError;
use opentelemetry_sdk::logs::{InMemoryLogExporter, SdkLoggerProvider};
use prost::Message;
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

#[test]
fn log_config_rejects_zero_timeout() {
    let error = OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
        .with_timeout(Duration::ZERO)
        .validate()
        .unwrap_err();

    assert!(error.to_string().contains("timeout must be greater than 0"));
}

#[test]
fn log_config_rejects_blank_and_padded_headers() {
    for (key, value) in [
        ("", "token"),
        (" authorization", "token"),
        ("authorization", ""),
        ("authorization", " token "),
    ] {
        let error = OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
            .with_header(key, value)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("surrounding whitespace"));
    }
}

#[test]
fn log_configured_headers_require_https_except_for_loopback() {
    for transport in [OtlpTransport::HttpBinary, OtlpTransport::Grpc] {
        for config in [
            OpenTelemetryLogConfig::new("http://collector.example/v1/logs")
                .with_transport(transport)
                .with_header("authorization", "Bearer static"),
            OpenTelemetryLogConfig::new("http://collector.example/v1/logs")
                .with_transport(transport)
                .with_header_env("authorization", "NEMO_RELAY_TEST_LOG_TOKEN"),
            OpenTelemetryLogConfig::new("http://collector.example/v1/logs")
                .with_transport(transport)
                .with_header_file("authorization", "/var/run/secrets/telemetry/token"),
        ] {
            let error = config.validate().unwrap_err();
            assert!(error.to_string().contains("require https"), "{error}");
        }

        assert!(
            OpenTelemetryLogConfig::new("http://127.0.0.1:4318/v1/logs")
                .with_transport(transport)
                .with_header("authorization", "Bearer static")
                .validate()
                .is_ok()
        );
    }
    assert!(
        OpenTelemetryLogConfig::new("http://collector.example/v1/logs")
            .validate()
            .is_ok()
    );
}

#[test]
fn log_configured_headers_allow_plain_http_only_to_listed_insecure_hosts() {
    let _guard = crate::test_mutex().lock().unwrap();
    let config = |endpoint: &str| {
        OpenTelemetryLogConfig::new(endpoint)
            .with_header("authorization", "Bearer static")
            .with_plaintext_policy(
                PlaintextPolicy::from_hosts(&["release-api".to_string()]).unwrap(),
            )
    };
    assert!(config("http://release-api:4318/v1/logs").validate().is_ok());
    let error = config("http://collector.example/v1/logs")
        .validate()
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("hosts listed in insecure_http_hosts"),
        "{error}"
    );
    assert!(
        OpenTelemetryLogConfig::new("http://collector.example/v1/logs")
            .with_header("authorization", "Bearer static")
            .with_allow_insecure(true)
            .validate()
            .is_ok()
    );
}

#[test]
fn log_config_validates_batch_limits_and_retains_resource_identity() {
    for config in [
        OpenTelemetryLogConfig::new("   "),
        OpenTelemetryLogConfig::new("https://collector.example/v1/logs").with_max_queue_size(0),
        OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
            .with_max_queue_size(1)
            .with_max_export_batch_size(2),
        OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
            .with_scheduled_delay(Duration::ZERO),
        OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
            .with_promote_resource_metadata_prefixes(["nv.*"]),
    ] {
        assert!(config.validate().is_err());
    }

    let config = OpenTelemetryLogConfig::new("https://collector.example/v1/logs")
        .with_service_namespace("relay")
        .with_service_version("0.8.0")
        .with_promote_resource_metadata_prefixes(["nv.client.", "nv.env."])
        .with_resource_attribute("deployment.environment", "test");
    assert_eq!(config.service_namespace.as_deref(), Some("relay"));
    assert_eq!(config.service_version.as_deref(), Some("0.8.0"));
    assert_eq!(
        config.resource_attributes.get("deployment.environment"),
        Some(&"test".to_string())
    );
    assert_eq!(
        config.promote_resource_metadata_prefixes,
        ["nv.client.", "nv.env."]
    );
}

#[test]
fn direct_log_subscriber_recovers_a_poisoned_processor_lock() {
    let subscriber =
        OpenTelemetryLogSubscriber::new(OpenTelemetryLogConfig::new("http://127.0.0.1:4318"))
            .unwrap();
    let processor = Arc::clone(&subscriber.inner._processor);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _guard = processor.lock().unwrap();
            panic!("poison log processor");
        }))
        .is_err()
    );

    let uuid = Uuid::now_v7();
    (subscriber.subscriber())(&scope(uuid, ScopeCategory::Start));

    let Err(poisoned) = processor.lock() else {
        panic!("log processor lock should remain poisoned after recovery");
    };
    let processor = poisoned.into_inner();
    assert!(processor.lineage.active.contains_key(&uuid));
}

#[test]
fn direct_log_subscriber_exposes_runtime_diagnostics() {
    let subscriber =
        OpenTelemetryLogSubscriber::new(OpenTelemetryLogConfig::new("http://127.0.0.1:4318"))
            .unwrap();
    let event = mark(
        None,
        "invalid-metric",
        Some(json!({"measurements": []})),
        Some(
            DataSchema::builder()
                .name(METRIC_DATA_SCHEMA_NAME)
                .version("999")
                .build(),
        ),
        None,
    );

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
fn signal_endpoint_resolution_replaces_trace_suffix_without_nested_paths() {
    for (input, expected) in [
        (
            "https://collector.example",
            "https://collector.example/v1/logs",
        ),
        (
            "https://collector.example/prefix/v1/traces?tenant=one",
            "https://collector.example/prefix/v1/logs?tenant=one",
        ),
        (
            "https://collector.example/custom",
            "https://collector.example/custom",
        ),
        (
            "https://collector.example/",
            "https://collector.example/v1/logs",
        ),
    ] {
        let resolved = resolve_http_log_endpoint(input);
        assert_eq!(resolved, expected);
        assert!(!resolved.contains("/v1/traces/v1/logs"));
    }
}

#[test]
fn log_delivery_state_reports_queue_and_export_failures_independently() {
    let diagnostics = LogDeliveryDiagnostics::new(
        "https://collector.example/v1/logs".to_string(),
        SignalRuntimeDiagnostics::new(Some("opentelemetry.logs.endpoints[0].endpoint".to_string())),
    );
    diagnostics.emitted.store(3, Ordering::Relaxed);
    diagnostics.accepted.store(2, Ordering::Relaxed);
    diagnostics.export_failures.store(2, Ordering::Relaxed);

    assert_eq!(
        diagnostics.failure_summary().as_deref(),
        Some("otel.logs_dropped (1), otel.logs_export_failed (2)")
    );
}

#[test]
fn direct_log_processor_records_cumulative_queue_drops_on_flush_and_shutdown() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let runtime_guard = runtime.enter();
    let runtime_diagnostics = SignalRuntimeDiagnostics::new(None);
    let delivery_diagnostics = Arc::new(LogDeliveryDiagnostics::new(
        "https://collector.example/v1/logs".to_string(),
        runtime_diagnostics.clone(),
    ));
    delivery_diagnostics.emitted.store(3, Ordering::Relaxed);
    delivery_diagnostics.accepted.store(1, Ordering::Relaxed);
    let processor = DiagnosticBatchLogProcessor {
        inner: AsyncBatchLogProcessor::builder(
            InMemoryLogExporter::default(),
            opentelemetry_sdk::runtime::Tokio,
        )
        .build(),
        diagnostics: Arc::clone(&delivery_diagnostics),
        retry_timeout: Duration::from_secs(1),
    };
    drop(runtime_guard);

    processor.force_flush().unwrap();

    let diagnostics = runtime_diagnostics.snapshot();
    let diagnostic = diagnostics
        .get("otel.logs_dropped")
        .expect("direct subscriber flush drop diagnostic");
    assert_eq!(diagnostic.count, 2);
    assert!(
        diagnostic
            .message
            .contains("https://collector.example/v1/logs")
    );

    delivery_diagnostics.emitted.store(5, Ordering::Relaxed);
    processor
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let diagnostics = runtime_diagnostics.snapshot();
    let diagnostic = diagnostics
        .get("otel.logs_dropped")
        .expect("direct subscriber shutdown drop diagnostic");
    assert_eq!(diagnostic.count, 4);
    assert!(
        diagnostic
            .message
            .contains("https://collector.example/v1/logs")
    );
}

#[test]
fn batch_control_retry_waits_for_a_transiently_full_queue() {
    let attempts = AtomicU64::new(0);

    retry_batch_processor_channel_full(Duration::from_secs(1), || {
        if attempts.fetch_add(1, Ordering::Relaxed) < 2 {
            Err(OTelSdkError::InternalFailure("ChannelFull".to_string()))
        } else {
            Ok(())
        }
    })
    .unwrap();

    assert_eq!(attempts.load(Ordering::Relaxed), 3);
}

#[test]
fn json_conversion_preserves_top_level_absence_and_nested_null() {
    assert_eq!(json_body(&Json::Null), None);
    let body = json_body(&json!([null, i64::MAX as u64 + 1])).unwrap();
    assert_eq!(
        body,
        AnyValue::ListAny(Box::new(vec![
            AnyValue::from("null"),
            AnyValue::from((i64::MAX as u64 + 1).to_string()),
        ]))
    );
}

#[test]
fn severity_and_json_conversion_cover_remaining_scalar_variants() {
    assert_eq!(
        otel_severity(LogSeverity::Trace),
        (Severity::Trace, "TRACE")
    );
    assert_eq!(
        otel_severity(LogSeverity::Debug),
        (Severity::Debug, "DEBUG")
    );
    assert_eq!(
        otel_severity(LogSeverity::Error),
        (Severity::Error, "ERROR")
    );
    assert_eq!(
        json_any_value(&json!(true), false),
        Some(AnyValue::Boolean(true))
    );
    assert_eq!(
        json_any_value(&json!(1.25), false),
        Some(AnyValue::Double(1.25))
    );
}

#[test]
fn provider_delivery_counters_do_not_report_other_providers_pending_logs_as_dropped() {
    #[derive(Debug)]
    struct RetainedExporter(InMemoryLogExporter);
    impl LogExporter for RetainedExporter {
        async fn export(&self, batch: LogBatch<'_>) -> OTelSdkResult {
            self.0.export(batch).await
        }
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let guard = runtime.enter();
    let diagnostics = Arc::new(LogDeliveryDiagnostics::new(
        "in-memory".to_string(),
        SignalRuntimeDiagnostics::new(Some("opentelemetry.logs".to_string())),
    ));
    let mut providers = Vec::new();
    let mut exporters = Vec::new();
    for count in [1, 3] {
        let provider_diagnostics = Arc::new(LogDeliveryDiagnostics::new(
            diagnostics.endpoint.clone(),
            diagnostics.runtime_diagnostics.clone(),
        ));
        diagnostics
            .children
            .lock()
            .unwrap()
            .push(Arc::clone(&provider_diagnostics));
        let capture = InMemoryLogExporter::default();
        let exporter = DiagnosticLogExporter {
            inner: RetainedExporter(capture.clone()),
            diagnostics: Arc::clone(&provider_diagnostics),
        };
        let processor = DiagnosticBatchLogProcessor {
            inner: AsyncBatchLogProcessor::builder(exporter, runtime::Tokio)
                .with_batch_config(
                    BatchConfigBuilder::default()
                        .with_scheduled_delay(Duration::from_secs(3600))
                        .build(),
                )
                .build(),
            diagnostics: Arc::clone(&provider_diagnostics),
            retry_timeout: Duration::from_secs(1),
        };
        let provider = SdkLoggerProvider::builder()
            .with_log_processor(processor)
            .build();
        let logger = provider.logger("regression");
        for _ in 0..count {
            logger.emit(logger.create_log_record());
        }
        providers.push(provider);
        exporters.push(capture);
    }
    drop(guard);
    let first = providers[0].shutdown();
    let second = providers[1].shutdown();
    let delivered: usize = exporters
        .iter()
        .map(|exporter| exporter.get_emitted_logs().unwrap().len())
        .sum();
    assert_eq!(delivered, 4, "all records must reach exporters");
    assert_eq!(diagnostics.failure_summary(), None);
    assert!(
        diagnostics
            .runtime_diagnostics
            .snapshot()
            .get("otel.logs_dropped")
            .is_none()
    );
    assert!(
        first.is_ok(),
        "all four records delivered, but first shutdown failed: {first:?}"
    );
    assert!(second.is_ok());
}

#[test]
fn resource_provider_failures_remain_visible_in_endpoint_summary() {
    let diagnostics =
        LogDeliveryDiagnostics::new("endpoint".to_string(), SignalRuntimeDiagnostics::new(None));
    let child = Arc::new(LogDeliveryDiagnostics::new(
        "endpoint".to_string(),
        diagnostics.runtime_diagnostics.clone(),
    ));
    child.emitted.store(3, Ordering::Relaxed);
    child.accepted.store(2, Ordering::Relaxed);
    child.export_failures.store(1, Ordering::Relaxed);
    diagnostics.children.lock().unwrap().push(child);
    assert_eq!(
        diagnostics.failure_summary().as_deref(),
        Some("otel.logs_dropped (1), otel.logs_export_failed (1)")
    );
}

#[test]
fn provider_build_drains_the_startup_resource_message_before_the_first_record() {
    use opentelemetry::logs::{LogRecord as _, Logger as _, LoggerProvider as _};

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let _runtime_guard = runtime.enter();
    let _held_worker = runtime.spawn(async { thread::sleep(Duration::from_millis(250)) });
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/logs", listener.local_addr().unwrap());
    let receiver = capture_one_request(listener);
    let config = OpenTelemetryLogConfig::new(endpoint.clone())
        .with_max_queue_size(1)
        .with_max_export_batch_size(1);
    let diagnostics = Arc::new(LogDeliveryDiagnostics::new(
        endpoint,
        SignalRuntimeDiagnostics::new(None),
    ));
    let provider = build_log_provider(&config, Arc::clone(&diagnostics), None).unwrap();
    let logger = provider.logger("drain-test");
    let mut record = logger.create_log_record();
    record.set_body(AnyValue::String("first".into()));
    logger.emit(record);
    provider.force_flush().unwrap();

    let body = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    let request = ExportLogsServiceRequest::decode(body.as_slice()).unwrap();
    let bodies = request
        .resource_logs
        .iter()
        .flat_map(|resource| resource.scope_logs.iter())
        .flat_map(|scope| scope.log_records.iter())
        .map(|record| record.body.clone().unwrap().value.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(bodies, [any_value::Value::StringValue("first".to_string())]);
    assert_eq!(diagnostics.record_queue_drops(), 0);
    provider.shutdown().unwrap();
}

fn capture_one_request(listener: TcpListener) -> mpsc::Receiver<Vec<u8>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 4_096];
        let (header_end, content_length) = loop {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0, "collector closed before request headers");
            bytes.extend_from_slice(&buffer[..count]);
            if let Some(offset) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                let header_end = offset + 4;
                let content_length = String::from_utf8_lossy(&bytes[..header_end])
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                break (header_end, content_length);
            }
        };
        while bytes.len() < header_end + content_length {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0, "collector closed before request body");
            bytes.extend_from_slice(&buffer[..count]);
        }
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        sender
            .send(bytes[header_end..header_end + content_length].to_vec())
            .unwrap();
    });
    receiver
}

fn mark(
    parent_uuid: Option<Uuid>,
    name: &str,
    data: Option<Json>,
    data_schema: Option<DataSchema>,
    metadata: Option<Json>,
) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data_opt(data)
            .data_schema_opt(data_schema)
            .metadata_opt(metadata)
            .build(),
        None,
        None,
    ))
}

fn scope(uuid: Uuid, category: ScopeCategory) -> Event {
    scope_with_parent(uuid, None, category)
}

fn scope_with_parent(uuid: Uuid, parent_uuid: Option<Uuid>, category: ScopeCategory) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .parent_uuid_opt(parent_uuid)
            .name("agent")
            .build(),
        category,
        Vec::new(),
        ScopeType::Agent.into(),
        None,
    ))
}
