// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Focused public-boundary coverage for diagnostic adapters and startup disablement.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use nemo_relay::diagnostics::{
    BuiltinDiagnosticCode, Classification, CommonPluginFailureCode, ComponentRegistration,
    DiagnosticAvailability, DiagnosticClassifier, DiagnosticDetailRef, DiagnosticError,
    DiagnosticQuery, Diagnostics, DiagnosticsSettings, FailureClass, FailureCode, Observation,
    PluginErrorClassifier, PluginFailureCode, RecordOutcome,
};
use nemo_relay::logging::{
    FileLogRotationConfig, FileLogSinkConfig, LogLevel, LogSinkConfig, LoggingConfig,
    LoggingRuntime,
};
use nemo_relay::plugin::PluginError;
use serde_json::Value;

#[path = "support/diagnostic_allocations.rs"]
mod diagnostic_allocations;
use diagnostic_allocations::{allocation_stats_during, allocations_during};

static LOGGING: Mutex<()> = Mutex::new(());
const PLUGIN_ERROR: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(
    PluginFailureCode::Common(CommonPluginFailureCode::Internal),
));

fn enabled() -> Diagnostics {
    Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap()
}

fn file_logging(service: &Diagnostics, path: std::path::PathBuf) -> LoggingRuntime {
    LoggingRuntime::configure_with_diagnostics(
        LoggingConfig {
            level: LogLevel::Error,
            stderr_enabled: false,
            flush_interval_millis: 0,
            sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
                path,
                ..Default::default()
            })],
            ..Default::default()
        },
        service,
    )
    .unwrap()
}

fn internal(message: &str) -> Observation<'_> {
    Observation::new(
        Classification::failure(
            FailureCode::Plugin(PluginFailureCode::Common(CommonPluginFailureCode::Internal)),
            FailureClass::Internal,
        )
        .unwrap(),
        message,
        NonZeroU64::MIN,
    )
}

#[cfg(windows)]
#[test]
fn windows_sharing_violation_records_required_sink_initialization_failure_once() {
    use std::os::windows::fs::OpenOptionsExt;

    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    for rotation in [None, Some(FileLogRotationConfig::new(1, 2).unwrap())] {
        let service = enabled();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("locked.jsonl");
        let exclusive = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .share_mode(0)
            .open(&path)
            .unwrap();
        let error = LoggingRuntime::configure_with_diagnostics(
            LoggingConfig {
                stderr_enabled: false,
                sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
                    path: path.clone(),
                    rotation,
                    ..Default::default()
                })],
                ..Default::default()
            },
            &service,
        )
        .err()
        .expect("a required sink locked by another handle must fail startup");
        assert!(error.to_string().contains("failed to open"), "{error}");
        let view = service
            .snapshot(&DiagnosticQuery::default())
            .unwrap()
            .to_snapshot()
            .unwrap();
        for (key, code) in [
            ("logging", "logging.initialize_failed"),
            ("logging.file.0", "logging.sink.initialize_failed"),
        ] {
            let component = view
                .components
                .iter()
                .find(|row| row.identity.key == key)
                .unwrap();
            assert!(component.closed_at.is_some());
            assert_eq!(
                component
                    .counters
                    .iter()
                    .find(|row| row.code == code)
                    .unwrap()
                    .occurrences
                    .value
                    .get(),
                1
            );
            assert_eq!(
                view.events
                    .iter()
                    .filter(|event| event.component == component.identity && event.code() == code)
                    .count(),
                1
            );
        }
        assert!(
            !serde_json::to_string(&view)
                .unwrap()
                .contains("locked.jsonl")
        );
        drop(exclusive);
        // A failed initialization must leave the logging facade usable.
        file_logging(&service, path).shutdown();
    }
}

#[cfg(windows)]
#[test]
fn windows_rotation_sharing_violation_records_once_without_projection_recursion() {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{FILE_SHARE_READ, FILE_SHARE_WRITE};

    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    let service = enabled();
    let producer = service
        .register_component(ComponentRegistration::new("test.producer", &[PLUGIN_ERROR]))
        .unwrap();
    let deliveries = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&deliveries);
    let _projection = service
        .subscribe_projection(move |_| {
            captured.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("rotation.jsonl");
    // Permit normal file writes but deny the delete sharing required by rename.
    let locked = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(&path)
        .unwrap();
    let runtime = LoggingRuntime::configure_with_diagnostics(
        LoggingConfig {
            level: LogLevel::Error,
            stderr_enabled: false,
            flush_interval_millis: 0,
            sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
                path: path.clone(),
                rotation: Some(FileLogRotationConfig::new(1, 2).unwrap()),
                ..Default::default()
            })],
            ..Default::default()
        },
        &service,
    )
    .unwrap();
    producer.record_lazy(|| internal("first record")).unwrap();
    producer.record_lazy(|| internal("rotation fails")).unwrap();
    // Draining is the synchronization barrier; no scheduler timing assumptions.
    runtime.shutdown();
    let view = service
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let sink = view
        .components
        .iter()
        .find(|row| row.identity.key == "logging.file.0")
        .unwrap();
    assert_eq!(
        sink.counters
            .iter()
            .find(|row| row.code == "logging.sink.write_failed")
            .unwrap()
            .occurrences
            .value
            .get(),
        1
    );
    assert_eq!(
        view.events
            .iter()
            .filter(|event| event.component == sink.identity
                && event.code() == "logging.sink.write_failed")
            .count(),
        1
    );
    assert_eq!(deliveries.load(Ordering::Relaxed), 2);
    assert_eq!(
        view.coverage.loss.projection_deliveries_omitted.value.get(),
        0
    );
    let output = std::fs::read_to_string(&path).unwrap();
    let rows: Vec<Value> = output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["message"], "Diagnostic observation");
    assert_eq!(rows[0]["event"], "plugin.internal");
    assert!(rows[0]["fields"]["diagnostic"].get("detail").is_none());
    assert!(!output.contains("first record"));
    assert!(!output.contains("rotation fails"));
    assert!(sink.closed_at.is_some());
    drop(locked);
    file_logging(&service, path).shutdown();
}

#[test]
fn unicode_paths_preserve_summary_logs_and_retained_details_through_rotation_and_shutdown() {
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    for rotation in [None, Some(FileLogRotationConfig::new(1, 2).unwrap())] {
        let service = enabled();
        let producer = service
            .register_component(ComponentRegistration::new("test.producer", &[PLUGIN_ERROR]))
            .unwrap();
        let temp = tempfile::tempdir().unwrap();
        // Canonicalization supplies a verbatim prefix on Windows, so paths over
        // MAX_PATH work without a machine-wide long-path policy change.
        let path = std::fs::canonicalize(temp.path())
            .unwrap()
            .join("long directory ".repeat(14))
            .join("logs with spaces 日本語🙂")
            .join("relay café.jsonl");
        assert!(path.as_os_str().len() > 260);
        let runtime = LoggingRuntime::configure_with_diagnostics(
            LoggingConfig {
                level: LogLevel::Error,
                stderr_enabled: false,
                flush_interval_millis: 0,
                sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
                    path: path.clone(),
                    rotation,
                    ..Default::default()
                })],
                ..Default::default()
            },
            &service,
        )
        .unwrap();
        let messages = ["first\r\n日本語🙂", "second\r\n café"];
        for message in messages {
            producer.record_lazy(|| internal(message)).unwrap();
        }
        runtime.shutdown();
        let files = if rotation.is_some() {
            vec![path.with_file_name("relay café.1.jsonl"), path.clone()]
        } else {
            vec![path.clone()]
        };
        let instance_id = producer.identity().unwrap().instance_id;
        let rows = read_unicode_log_rows(&files, &messages);
        assert_unicode_log_projection(&rows, instance_id, messages.len());
        assert_unicode_diagnostic_history(&service, instance_id, &messages);
    }
}

fn read_unicode_log_rows(files: &[std::path::PathBuf], messages: &[&str]) -> Vec<Value> {
    let mut rows = Vec::new();
    for file in files {
        let output = std::fs::read_to_string(file).unwrap();
        // JSON escaping must not hide raw detail leaking into either log file.
        for message in messages {
            assert!(!output.contains(&serde_json::to_string(message).unwrap()));
        }
        rows.extend(
            output
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap()),
        );
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // An exclusive open verifies all worker-owned file handles closed.
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .share_mode(0)
                .open(file)
                .unwrap();
        }
    }
    rows
}

fn assert_unicode_log_projection(rows: &[Value], instance_id: uuid::Uuid, expected: usize) {
    assert_eq!(rows.len(), expected);
    for row in rows {
        assert_eq!(row["message"], "Diagnostic observation");
        assert_eq!(row["event"], "plugin.internal");
        assert!(row["fields"]["diagnostic"].get("detail").is_none());
        assert_eq!(
            row["fields"]["diagnostic"]["component_instance_id"],
            instance_id.to_string()
        );
    }
}

fn assert_unicode_diagnostic_history(
    service: &Diagnostics,
    instance_id: uuid::Uuid,
    messages: &[&str],
) {
    let view = service
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let retained: Vec<_> = view
        .events
        .iter()
        .filter(|event| event.component.instance_id == instance_id)
        .collect();
    assert_eq!(retained.len(), messages.len());
    for (event, message) in retained.iter().zip(messages) {
        let detail = event.detail.as_ref().unwrap();
        assert_eq!(detail.message, *message);
        assert!(!detail.truncated);
    }
    assert!(
        !view
            .events
            .iter()
            .any(|event| event.code().starts_with("logging.sink.")
                && event.classification.failure_class().is_some())
    );
    assert_eq!(
        view.coverage.loss.projection_deliveries_omitted.value.get(),
        0
    );
}

#[test]
fn disabled_diagnostics_preserve_operational_file_output() {
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    let service = Diagnostics::default();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("disabled.jsonl");
    let runtime = file_logging(&service, path.clone());
    let recorder = service
        .register_component(ComponentRegistration::new("test.disabled", &[PLUGIN_ERROR]))
        .unwrap();
    assert!(recorder.identity().is_none());
    assert_eq!(
        recorder
            .record_lazy(|| panic!("disabled factory ran"))
            .unwrap(),
        RecordOutcome::Disabled
    );
    assert_eq!(
        service.reader().availability(),
        DiagnosticAvailability::Disabled
    );
    assert_eq!(
        service.snapshot(&DiagnosticQuery::default()).unwrap_err(),
        DiagnosticError::Disabled
    );
    assert_eq!(service.memory_usage().reserved_bytes, 0);
    log::error!(target: "nemo_relay.test", event = "disabled_service_operational_output"; "Ordinary operational output remains available");
    runtime.shutdown();
    let output = std::fs::read_to_string(path).unwrap();
    let matching = output
        .lines()
        .filter(|line| {
            serde_json::from_str::<Value>(line).unwrap()["event"]
                == "disabled_service_operational_output"
        })
        .count();
    assert_eq!(matching, 1);
    assert!(!output.contains("plugin.internal"));
}

#[test]
fn typed_plugin_error_records_once_without_projections_or_legacy_replay() {
    let service = enabled();
    let code = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Common(
        CommonPluginFailureCode::RegistrationFailed,
    )));
    let recorder = service
        .register_component(ComponentRegistration::new("test.plugin", &[code]))
        .unwrap();
    let projected = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&projected);
    let _projection = service
        .subscribe_projection(move |_| {
            captured.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let original = PluginError::RegistrationFailed("private registration context".into());
    let detail = || DiagnosticDetailRef::Text("Plugin registration failed");
    let outcome = recorder
        .record_without_projection_lazy(|| {
            Observation::new(
                PluginErrorClassifier.classify(&original),
                detail(),
                NonZeroU64::new(3).unwrap(),
            )
        })
        .unwrap();
    assert!(matches!(outcome, RecordOutcome::Recorded(_)));
    assert!(matches!(original, PluginError::RegistrationFailed(_)));
    let page = service
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].code(), "plugin.registration_failed");
    assert_eq!(
        page.events[0].classification.failure_class(),
        Some("unknown")
    );
    assert_eq!(page.events[0].count.get(), 3);
    assert_eq!(
        page.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        1
    );
    assert_eq!(
        page.components[0]
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        3
    );
    assert!(
        !serde_json::to_string(&page)
            .unwrap()
            .contains("private registration context")
    );
    assert_eq!(projected.load(Ordering::Relaxed), 0);
}

struct DisabledClassifier;
impl DiagnosticClassifier<PluginError> for DisabledClassifier {
    fn classify(&self, _error: &PluginError) -> Classification {
        panic!("disabled classifier ran")
    }
}

#[test]
fn disabled_plugin_recording_skips_classifier_and_detail_factory() {
    let service = Diagnostics::default();
    let recorder = service
        .register_component(ComponentRegistration::new("test.plugin", &[]))
        .unwrap();
    let error = PluginError::Internal("operation error remains available".into());
    let detail = || -> DiagnosticDetailRef<'_> { panic!("disabled detail factory ran") };
    let outcome = recorder
        .record_without_projection_lazy(|| {
            Observation::new(
                DisabledClassifier.classify(&error),
                detail(),
                NonZeroU64::MIN,
            )
        })
        .unwrap();
    assert_eq!(outcome, RecordOutcome::Disabled);
    assert!(matches!(error, PluginError::Internal(_)));
    assert_eq!(service.memory_usage().components, 0);
}

#[test]
fn typed_projection_reuses_logging_without_replaying_history_or_pinning_reader() {
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    let service = enabled();
    let producer = service
        .register_component(ComponentRegistration::new("test.producer", &[PLUGIN_ERROR]))
        .unwrap();
    producer
        .record_without_projection_lazy(|| internal("Earlier retained evidence"))
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("projected.jsonl");
    let runtime = file_logging(&service, path.clone());
    producer
        .record_lazy(|| internal("One projected observation"))
        .unwrap();
    runtime.shutdown();
    let output = std::fs::read_to_string(&path).unwrap();
    let records: Vec<Value> = output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|record: &Value| record["event"] == "plugin.internal")
        .collect();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["message"], "Diagnostic observation");
    assert!(!output.contains("One projected observation"));
    assert_eq!(records[0]["target"], "nemo_relay.diagnostics");
    assert_eq!(
        records[0]["fields"]["diagnostic"]["classification"]["canonical_code"],
        "plugin.internal"
    );
    assert_eq!(
        records[0]["fields"]["diagnostic"]["component_instance_id"],
        producer.identity().unwrap().instance_id.to_string()
    );
    assert_eq!(
        records[0]["root_relay_id"],
        service.source().instance_id.to_string()
    );
    assert!(!output.contains("Earlier retained evidence"));
    producer
        .record_lazy(|| internal("After logger shutdown"))
        .unwrap();
    assert_eq!(std::fs::read_to_string(path).unwrap(), output);
    let page = service
        .snapshot(&DiagnosticQuery {
            component_key: Some("test.producer".into()),
            ..Default::default()
        })
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(
        page.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        3
    );
    assert_eq!(page.events.len(), 3);
}

#[test]
fn restricted_detail_does_not_consume_logging_projection_payload_budget() {
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    let service = enabled();
    let producer = service
        .register_component(ComponentRegistration::new("test.producer", &[PLUGIN_ERROR]))
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("capped.jsonl");
    let runtime = file_logging(&service, path.clone());
    // Raw detail stays in retention; its escaping cost cannot exhaust output payloads.
    let message = "\u{1}".repeat(4096);
    producer.record_lazy(|| internal(&message)).unwrap();
    log::error!(target: "nemo_relay.test", event = "operational_after_projection_limit"; "The ordinary logger still works");
    runtime.shutdown();
    let page = service
        .snapshot(&DiagnosticQuery {
            component_key: Some("test.producer".into()),
            ..Default::default()
        })
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(
        page.components[0]
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        1
    );
    assert_eq!(
        page.components[0]
            .lifetime
            .totals
            .failures_total
            .value
            .get(),
        1
    );
    assert_eq!(page.events.len(), 1);
    assert_eq!(
        page.coverage.loss.projection_deliveries_omitted.value.get(),
        0
    );
    let output = std::fs::read_to_string(path).unwrap();
    assert!(
        output
            .lines()
            .any(|line| serde_json::from_str::<Value>(line).unwrap()["event"]
                == "operational_after_projection_limit")
    );
    assert!(output.contains("plugin.internal"));
    assert!(!output.contains("\\u0001"));
}

#[test]
fn warmed_projection_avoids_owned_field_conversions_without_sinks() {
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    let service = enabled();
    let producer = service
        .register_component(ComponentRegistration::new("test.producer", &[PLUGIN_ERROR]))
        .unwrap();
    let runtime = LoggingRuntime::configure_with_diagnostics(
        LoggingConfig {
            level: LogLevel::Error,
            stderr_enabled: false,
            flush_interval_millis: 0,
            ..Default::default()
        },
        &service,
    )
    .unwrap();
    producer
        .record_lazy(|| internal("Projected observation"))
        .unwrap();
    let (_, calls) = allocations_during(|| {
        for _ in 0..100 {
            producer
                .record_lazy(|| internal("Projected observation"))
                .unwrap();
        }
    });
    // The native backend formats the message once. Borrowed fields require no
    // intermediate vector or owned ValueBag conversion on this synchronous path.
    assert!(
        calls <= 100,
        "projection allocated {calls} times for 100 events"
    );
    runtime.shutdown();
}

/// Complete producer-to-backend path, including queued ownership when enabled.
/// File worker formatting/I/O allocations run on other threads and are excluded
/// from these producer-thread traffic measurements; draining/output are verified.
#[test]
#[ignore = "informational complete logging adapter allocation/throughput profile"]
fn diagnostic_logging_projection_profile() {
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    eprintln!(
        "logging profile: debug_assertions={}, producer-thread request traffic only",
        cfg!(debug_assertions)
    );
    for asynchronous in [false, true] {
        let service = enabled();
        let producer = service
            .register_component(ComponentRegistration::new(
                "profile.producer",
                &[PLUGIN_ERROR],
            ))
            .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("profile.jsonl");
        let runtime = LoggingRuntime::configure_with_diagnostics(
            LoggingConfig {
                level: LogLevel::Error,
                stderr_enabled: false,
                flush_interval_millis: 0,
                sinks: if asynchronous {
                    vec![LogSinkConfig::File(FileLogSinkConfig {
                        path: path.clone(),
                        queue_capacity: 256,
                        ..Default::default()
                    })]
                } else {
                    vec![]
                },
                ..Default::default()
            },
            &service,
        )
        .unwrap();
        producer.record_lazy(|| internal("warm-up")).unwrap();
        let reserved = service.memory_usage().reserved_bytes;
        let begin = std::time::Instant::now();
        let (_, stats) = allocation_stats_during(|| {
            for _ in 0..100 {
                producer
                    .record_lazy(|| internal("projected observation"))
                    .unwrap();
            }
        });
        eprintln!(
            "logging async={asynchronous}: events=100 elapsed={:?} traffic={stats:?}",
            begin.elapsed()
        );
        assert_eq!(service.memory_usage().reserved_bytes, reserved);
        if !asynchronous {
            assert_eq!(stats.allocation_calls, 100);
            assert_eq!(stats.free_calls, 100);
            assert_eq!(stats.allocated_bytes, stats.freed_bytes);
            assert_eq!(stats.realloc_calls, 0);
        } else {
            assert!(
                stats.allocation_calls > 100,
                "async queues must own their records"
            );
        }
        runtime.shutdown();
        if asynchronous {
            let output = std::fs::read_to_string(&path).unwrap();
            let rows: Vec<Value> = output
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(rows.len(), 101);
            assert!(
                rows.iter()
                    .all(|row| row["target"] == "nemo_relay.diagnostics")
            );
            assert_eq!(rows[100]["message"], "projected observation");
            assert_eq!(
                rows[100]["fields"]["diagnostic"]["classification"]["canonical_code"],
                "plugin.internal"
            );
        }
    }
}

#[test]
fn diagnostic_reads_and_recording_do_not_feed_observability_subscribers() {
    use nemo_relay::api::scope::{EmitMarkEventParams, event};
    use nemo_relay::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber,
    };
    use nemo_relay::diagnostics::{ComponentOwnership, DiagnosticCorrelation};
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    let observed = Arc::new(Mutex::new(Vec::new()));
    let output = observed.clone();
    register_subscriber(
        "diagnostic-isolation",
        Arc::new(move |value| {
            output.lock().unwrap().push(value.name().to_owned());
        }),
    )
    .unwrap();
    let service = enabled();
    let mut registration = ComponentRegistration::new("restricted.component", &[PLUGIN_ERROR]);
    registration.ownership = ComponentOwnership {
        configuration_id: Some("restricted-path".into()),
        ..Default::default()
    };
    let producer = service.register_component(registration).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("output.jsonl");
    let logger = file_logging(&service, path.clone());
    for (project, message) in [
        (true, "restricted-projected-payload"),
        (false, "restricted-state-only-payload"),
    ] {
        let observation = || {
            internal(message).with_correlation(DiagnosticCorrelation {
                request_id: Some("restricted-request".into()),
                session_id: Some("restricted-session".into()),
            })
        };
        let outcome = if project {
            producer.record_lazy(observation).unwrap()
        } else {
            producer
                .record_without_projection_lazy(observation)
                .unwrap()
        };
        assert!(matches!(outcome, RecordOutcome::Recorded(_)));
    }
    let page = service
        .snapshot(&DiagnosticQuery {
            component_key: Some("restricted.component".into()),
            ..Default::default()
        })
        .unwrap();
    let snapshot = page.to_snapshot().unwrap();
    assert_eq!(
        snapshot
            .events
            .iter()
            .map(|event| event.message())
            .collect::<Vec<_>>(),
        [
            Some("restricted-projected-payload"),
            Some("restricted-state-only-payload")
        ]
    );
    let raw = std::str::from_utf8(page.as_bytes()).unwrap();
    for marker in [
        "restricted.component",
        "restricted-path",
        "restricted-request",
        "restricted-session",
    ] {
        assert!(raw.contains(marker));
    }
    drop(page);
    // The event path remains usable and independent; no diagnostic-generated event
    // should be queued before this explicit application mark.
    event(
        EmitMarkEventParams::builder()
            .name("application-mark")
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();
    deregister_subscriber("diagnostic-isolation").unwrap();
    logger.shutdown();
    assert_eq!(*observed.lock().unwrap(), ["application-mark"]);
    let output = std::fs::read_to_string(path).unwrap();
    let component_id = producer.identity().unwrap().instance_id.to_string();
    let projected: Vec<Value> = output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|row: &Value| {
            row["target"] == "nemo_relay.diagnostics"
                && row["fields"]["diagnostic"]["component_instance_id"] == component_id
        })
        .collect();
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0]["event"], "plugin.internal");
    assert!(!output.contains("restricted"));
}
