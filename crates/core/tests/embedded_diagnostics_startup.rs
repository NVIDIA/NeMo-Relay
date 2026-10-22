// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public embedded startup contract. Each test requires process isolation (nextest).
use nemo_relay::{
    diagnostics::embedded,
    plugin::dynamic::{PluginHostConfig, initialize, validate},
};

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
