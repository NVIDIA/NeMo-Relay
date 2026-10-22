// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public embedded startup contract. Each test requires process isolation (nextest).
use nemo_relay::{
    diagnostics::{DiagnosticSnapshot, Diagnostics, embedded},
    logging::{LoggingConfig, LoggingRuntime},
    plugin::dynamic::{PluginHostConfig, initialize, validate},
};

#[tokio::test]
async fn explicit_logging_does_not_select_embedded_diagnostics() {
    let logging_config = LoggingConfig {
        stderr_enabled: false,
        ..Default::default()
    };
    let logging = LoggingRuntime::configure(logging_config.clone()).unwrap();
    assert!(Diagnostics::selected_process_default().is_none());
    assert_eq!(embedded::reader().unwrap_err().reason(), "not_initialized");

    let config = PluginHostConfig {
        diagnostics: Some(nemo_relay::diagnostics::DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut activation = initialize(config, None).await.unwrap();
    let reader = embedded::reader().unwrap();
    assert!(reader.snapshot_json("{}").is_ok());
    drop(logging);

    let logging = LoggingRuntime::configure(logging_config).unwrap();
    let page = reader.snapshot_json("{}").unwrap();
    let snapshot: DiagnosticSnapshot = serde_json::from_str(page.as_str()).unwrap();
    assert_eq!(
        logging.root_relay_id(),
        snapshot.source.instance_id.to_string()
    );
    assert!(
        snapshot
            .components
            .iter()
            .any(|component| component.identity.key == "logging")
    );
    activation.close().unwrap();
    drop(logging);
}

#[tokio::test]
async fn file_settings_are_reused_and_absent_from_plugin_reports() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("plugins.toml");
    std::fs::write(
        &path,
        "version = 1\n[diagnostics]\nenabled = true\n[diagnostics.limits]\nmax_events = 3\n",
    )
    .unwrap();
    let validation = validate(PluginHostConfig::default(), Some(path.clone())).unwrap();
    assert!(validation.resolved_config.get("diagnostics").is_none());
    assert_eq!(embedded::reader().unwrap_err().reason(), "not_initialized");
    let mut activation = initialize(PluginHostConfig::default(), Some(path.clone()))
        .await
        .unwrap();
    let reader = embedded::reader().unwrap();
    let info = reader.info().unwrap();
    assert_eq!(info.limits.max_events, 3);
    assert!(
        activation
            .report()
            .resolved_config
            .get("diagnostics")
            .is_none()
    );
    activation.close().unwrap();
    std::fs::write(path.clone(), "[diagnostics]\nenabled = false\n").unwrap();
    assert!(
        initialize(PluginHostConfig::default(), Some(path))
            .await
            .is_err()
    );
    assert_eq!(
        reader.info().unwrap().discovery.source.instance_id,
        info.discovery.source.instance_id
    );
    assert!(reader.snapshot_json("{}").is_ok());
}

#[tokio::test]
async fn programmatic_service_profile_replaces_invalid_file_settings() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("plugins.toml");
    std::fs::write(&path, "[diagnostics]\nenabled = 'true'\n").unwrap();
    assert!(validate(PluginHostConfig::default(), Some(path.clone())).is_err());
    assert!(
        initialize(PluginHostConfig::default(), Some(path.clone()))
            .await
            .is_err()
    );

    let mut settings = nemo_relay::diagnostics::DiagnosticsSettings {
        enabled: true,
        ..Default::default()
    };
    settings.limits.max_events = 3;
    let config = PluginHostConfig {
        diagnostics: Some(settings),
        ..Default::default()
    };
    let report = validate(config.clone(), Some(path.clone())).unwrap();
    assert!(report.resolved_config.get("diagnostics").is_none());

    let mut invalid = config.clone();
    invalid.diagnostics.as_mut().unwrap().limits.max_owned_bytes = 1;
    assert!(validate(invalid.clone(), Some(path.clone())).is_err());
    assert!(initialize(invalid, Some(path.clone())).await.is_err());

    let mut activation = initialize(config, Some(path)).await.unwrap();
    assert_eq!(
        embedded::reader()
            .unwrap()
            .info()
            .unwrap()
            .limits
            .max_events,
        3
    );
    activation.close().unwrap();
}

#[test]
fn invalid_service_settings_do_not_select_a_default() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("plugins.toml");
    std::fs::write(&path, "[diagnostics]\nenabled = true\nunknown = 1\n").unwrap();
    assert!(validate(PluginHostConfig::default(), Some(path)).is_err());
    assert_eq!(embedded::reader().unwrap_err().reason(), "not_initialized");
}

#[tokio::test]
async fn application_owned_logger_does_not_block_diagnostics() {
    struct ApplicationLogger;
    impl log::Log for ApplicationLogger {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, _: &log::Record<'_>) {}
        fn flush(&self) {}
    }
    static LOGGER: ApplicationLogger = ApplicationLogger;
    log::set_logger(&LOGGER).unwrap();
    let config = PluginHostConfig {
        diagnostics: Some(nemo_relay::diagnostics::DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut activation = initialize(config, None).await.unwrap();
    let reader = embedded::reader().unwrap();
    assert_eq!(
        reader.info().unwrap().logging_coverage,
        "not_owned_or_closed"
    );
    assert!(reader.snapshot_json("{}").is_ok());
    assert!(std::ptr::addr_eq(log::logger(), &LOGGER as &dyn log::Log));
    activation.close().unwrap();
}
