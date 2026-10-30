// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded diagnostics and plugin-host startup integration.

use nemo_relay_diagnostics::embedded::EmbeddedDiagnosticError;
use nemo_relay_diagnostics::*;
use nemo_relay_runtime::{logging, plugin};
#[tokio::test]
async fn embedded_host_selects_service_before_plugin_failure_and_reuses_it() {
    // nextest runs this process-global scenario in its own process.
    assert!(matches!(
        nemo_relay_diagnostics::embedded::reader(),
        Err(EmbeddedDiagnosticError::NotInitialized)
    ));
    logging::initialize_default_logging().unwrap();
    assert!(Diagnostics::selected_process_default().is_none());
    let config: plugin::dynamic::PluginHostConfig = serde_json::from_value(serde_json::json!({
        "diagnostics": {"enabled": true},
        "components": [{"kind": "missing.phase4.component"}],
        "policy": {"unknown_component": "error"}
    }))
    .unwrap();
    assert!(plugin::dynamic::initialize(config, None).await.is_err());
    let retained = nemo_relay_diagnostics::embedded::reader().unwrap();
    assert!(retained.info().unwrap().discovery.collection_enabled);
    let source = retained.info().unwrap().discovery.source.instance_id;
    let config: plugin::dynamic::PluginHostConfig =
        serde_json::from_value(serde_json::json!({"diagnostics":{"enabled":true}})).unwrap();
    let mut activation = plugin::dynamic::initialize(config, None).await.unwrap();
    activation.close().unwrap();
    assert_eq!(
        nemo_relay_diagnostics::embedded::reader()
            .unwrap()
            .info()
            .unwrap()
            .discovery
            .source
            .instance_id,
        source
    );
    assert!(nemo_relay_diagnostics::embedded::start(DiagnosticsSettings::default()).is_err());
    assert!(retained.snapshot_json("{}").is_ok());
    logging::shutdown_default_logging().unwrap();
    assert!(retained.snapshot_json("{}").is_ok());
}
