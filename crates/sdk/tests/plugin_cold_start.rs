// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SDK initialization composes every built-in before validating an exact configuration.

use nemo_relay::plugin::{
    self, ConfigPolicy, PluginComponentSpec, PluginConfig, UnsupportedBehavior,
};

#[tokio::test]
async fn exact_initialization_recognizes_every_builtin_without_prior_bootstrap() {
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
    let mut activation = plugin::initialize_exact(PluginConfig {
        components,
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Error,
            ..Default::default()
        },
        ..Default::default()
    })
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
    activation.close().unwrap();
}
