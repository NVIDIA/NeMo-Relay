// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostics and logging stay independent from the SDK event subscriber pipeline.

use nemo_relay_diagnostics::{
    BuiltinDiagnosticCode, Classification, CommonPluginFailureCode, ComponentRegistration,
    DiagnosticQuery, Diagnostics, DiagnosticsSettings, FailureClass, FailureCode, Observation,
    PluginFailureCode, RecordOutcome,
};
use nemo_relay_logging::{
    FileLogSinkConfig, LogLevel, LogSinkConfig, LoggingConfig, LoggingRuntime,
};
use serde_json::Value;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

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

#[test]
fn diagnostic_reads_and_recording_do_not_feed_observability_subscribers() {
    use nemo_relay_diagnostics::{ComponentOwnership, DiagnosticCorrelation};
    use nemo_relay_runtime::api::scope::{EmitMarkEventParams, event};
    use nemo_relay_runtime::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber,
    };
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
