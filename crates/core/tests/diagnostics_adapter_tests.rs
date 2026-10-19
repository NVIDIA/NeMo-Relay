// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Focused public-boundary coverage for diagnostic adapters and startup disablement.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use nemo_relay::diagnostics::{
    BuiltinDiagnosticCode, Classification, CommonPluginFailureCode, ComponentRegistration,
    DiagnosticAvailability, DiagnosticClassifier, DiagnosticDetailRef, DiagnosticError,
    DiagnosticQuery, Diagnostics, DiagnosticsSettings, FailureClass, FailureCode, Observation,
    PluginErrorClassifier, PluginFailureCode, RecordOutcome, RuntimeFailureCode,
};
use nemo_relay::logging::{
    FileLogSinkConfig, LogLevel, LogSinkConfig, LoggingConfig, LoggingRuntime,
};
use nemo_relay::plugin::PluginError;
use serde_json::Value;

static LOGGING: Mutex<()> = Mutex::new(());
const RUNTIME_ERROR: BuiltinDiagnosticCode =
    BuiltinDiagnosticCode::Failure(FailureCode::Runtime(RuntimeFailureCode::Internal));

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
            FailureCode::Runtime(RuntimeFailureCode::Internal),
            FailureClass::Internal,
        )
        .unwrap(),
        message,
        1,
    )
}

#[test]
fn disabled_diagnostics_preserve_operational_file_output() {
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    let service = Diagnostics::default();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("disabled.jsonl");
    let runtime = file_logging(&service, path.clone());
    let recorder = service
        .register_component(ComponentRegistration::new(
            "test.disabled",
            &[RUNTIME_ERROR],
        ))
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
        service.snapshot(DiagnosticQuery::default()).unwrap_err(),
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
    assert!(!output.contains("runtime.internal"));
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
            Observation::new(PluginErrorClassifier.classify(&original), detail(), 3)
        })
        .unwrap();
    assert!(matches!(outcome, RecordOutcome::Recorded(_)));
    assert!(matches!(original, PluginError::RegistrationFailed(_)));
    let page = service
        .snapshot(DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].code(), "plugin.registration_failed");
    assert_eq!(
        page.events[0].classification.failure_class.as_deref(),
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
            Observation::new(DisabledClassifier.classify(&error), detail(), 1)
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
        .register_component(ComponentRegistration::new(
            "test.producer",
            &[RUNTIME_ERROR],
        ))
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
        .filter(|record: &Value| record["event"] == "runtime.internal")
        .collect();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["message"], "One projected observation");
    assert_eq!(
        records[0]["fields"]["diagnostic"]["classification"]["canonical_code"],
        "runtime.internal"
    );
    assert_eq!(
        records[0]["fields"]["diagnostic"]["component"]["instance_id"],
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
        .snapshot(DiagnosticQuery {
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
fn capped_logging_projection_reports_loss_without_undoing_counts_or_other_logs() {
    let _guard = LOGGING.lock().unwrap_or_else(|error| error.into_inner());
    let service = enabled();
    let producer = service
        .register_component(ComponentRegistration::new(
            "test.producer",
            &[RUNTIME_ERROR],
        ))
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("capped.jsonl");
    let runtime = file_logging(&service, path.clone());
    // JSON escaping exceeds the adapter's fixed serialization ceiling despite bounded input.
    let message = "\u{1}".repeat(4096);
    producer.record_lazy(|| internal(&message)).unwrap();
    log::error!(target: "nemo_relay.test", event = "operational_after_projection_limit"; "The ordinary logger still works");
    runtime.shutdown();
    let page = service
        .snapshot(DiagnosticQuery {
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
        1
    );
    let output = std::fs::read_to_string(path).unwrap();
    assert!(
        output
            .lines()
            .any(|line| serde_json::from_str::<Value>(line).unwrap()["event"]
                == "operational_after_projection_limit")
    );
    assert!(!output.contains("runtime.internal"));
}
