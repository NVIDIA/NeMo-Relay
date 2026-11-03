// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SDK initialization composes built-ins and selects a process-owned diagnostic service.

mod support;

use nemo_relay::plugin::{
    self, ConfigPolicy, PluginComponentSpec, PluginConfig, UnsupportedBehavior,
};

#[tokio::test]
async fn initialization_recognizes_builtins_and_keeps_diagnostics_after_close() {
    support::require_isolated_plugin_config();
    use nemo_relay::diagnostics::{self, DiagnosticQuery, DiagnosticsSettings};
    assert_eq!(
        diagnostics::reader().unwrap_err().reason(),
        "not_initialized"
    );
    let configuration = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(configuration.path(), "version = 1\n").unwrap();
    let kinds = [
        "adaptive",
        "pii_redaction",
        "observability",
        "pricing",
        "resource_metrics",
    ];
    let components = kinds
        .iter()
        .map(|kind| {
            let mut component = PluginComponentSpec::new(*kind);
            // Disabled components are still validated, without starting transports or polling.
            component.enabled = false;
            if *kind == "pii_redaction" {
                component.config = serde_json::from_value(serde_json::json!({
                    "builtin": {"action": "redact", "detector": "email"}
                }))
                .unwrap();
            }
            component
        })
        .collect();
    let mut activation = plugin::initialize(
        PluginConfig {
            components,
            policy: ConfigPolicy {
                unknown_component: UnsupportedBehavior::Error,
                ..Default::default()
            },
            ..Default::default()
        },
        Some(configuration.path().to_owned()),
    )
    .await
    .unwrap();
    let report = activation.report();
    assert!(
        report.config.diagnostics.is_empty(),
        "{:?}",
        report.config.diagnostics
    );
    let registered = plugin::list_plugin_kinds();
    for kind in kinds {
        assert!(registered.iter().any(|value| value == kind), "{kind}");
    }
    let reader = diagnostics::reader().unwrap();
    let source = reader.info().unwrap().discovery.source;
    assert_eq!(
        reader
            .snapshot(&DiagnosticQuery::default())
            .err()
            .unwrap()
            .reason(),
        "disabled"
    );
    activation.close().unwrap();
    assert_eq!(reader.info().unwrap().discovery.source, source);
    assert_eq!(
        diagnostics::reader()
            .unwrap()
            .info()
            .unwrap()
            .discovery
            .source,
        source
    );
    let mut reloaded = plugin::initialize(
        PluginConfig::default(),
        Some(configuration.path().to_owned()),
    )
    .await
    .unwrap();
    let conflict = plugin::initialize(
        plugin::PluginHostConfig {
            plugins: PluginConfig::default(),
            diagnostics: Some(DiagnosticsSettings {
                enabled: true,
                ..Default::default()
            }),
        },
        Some(configuration.path().to_owned()),
    )
    .await
    .err()
    .unwrap();
    assert!(conflict.to_string().contains("configuration_conflict"));
    assert!(reloaded.is_active());
    assert_eq!(reader.info().unwrap().discovery.source, source);
    reloaded.close().unwrap();
}
