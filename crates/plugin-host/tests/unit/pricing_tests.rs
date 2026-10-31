// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pricing component validation and host activation.
use crate::test_support::plugin_host::{
    test_close_plugin_host, test_initialize_plugin_host_exact, test_validate_static_plugin_config,
};
use crate::test_support::pricing_test_mutex;
use nemo_relay_model_pricing::estimate_cost;
use nemo_relay_plugin_host::{DiagnosticLevel, PluginComponentSpec, PluginConfig};
use nemo_relay_types::codec::response::Usage;
use serde_json::json;
struct ClearPluginConfigurationGuard;
impl Drop for ClearPluginConfigurationGuard {
    fn drop(&mut self) {
        let _ = test_close_plugin_host();
    }
}

fn register_pricing_component() -> nemo_relay_plugin_host::Result<()> {
    nemo_relay_plugin_host::register_builtin_plugin(std::sync::Arc::new(
        nemo_relay_model_pricing::plugin::PricingPlugin,
    ))
}

#[test]
fn test_pricing_plugin_configures_process_resolver_and_clears_to_default() {
    log::set_max_level(log::LevelFilter::Info);
    let _runtime_guard = crate::test_support::runtime_owner_test_mutex()
        .lock()
        .unwrap();
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let mut component = PluginComponentSpec::new("pricing");
    component.config = serde_json::from_value(json!({
        "sources": [
            {
                "type": "inline",
                "catalog": {
                    "version": 1,
                    "entries": [
                        {
                            "provider": "plugin-inline",
                            "model_id": "plugin-model",
                            "pricing_as_of": "2026-06-04",
                            "pricing_source": "plugins.toml",
                            "rates": {
                                "input_per_million": 1.0,
                                "output_per_million": 2.0
                            },
                            "prompt_cache": {
                                "read_accounting": "separate"
                            }
                        }
                    ]
                }
            }
        ]
    }))
    .unwrap();
    let mut config = PluginConfig::default();
    config.components.push(component);

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { test_initialize_plugin_host_exact(config).await.unwrap() });
    let _clear_guard = ClearPluginConfigurationGuard;
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    let configured = estimate_cost("plugin-model", &usage).unwrap();
    assert_eq!(configured.total, Some(0.002));
    assert!(estimate_cost("gpt-4o-mini", &usage).is_none());

    test_close_plugin_host().unwrap();

    assert!(estimate_cost("plugin-model", &usage).is_none());
    assert!(estimate_cost("gpt-4o-mini", &usage).is_none());
}

#[test]
fn test_pricing_plugin_validation_reports_invalid_json_and_catalog_errors() {
    let _runtime_guard = crate::test_support::runtime_owner_test_mutex()
        .lock()
        .unwrap();
    register_pricing_component().unwrap();
    register_pricing_component().unwrap();

    let mut malformed = PluginComponentSpec::new("pricing");
    malformed.config = serde_json::from_value(json!({
        "sources": [],
        "unexpected": true
    }))
    .unwrap();
    let report = test_validate_static_plugin_config(&PluginConfig {
        components: vec![malformed],
        ..PluginConfig::default()
    });
    assert!(report.has_errors());
    assert_eq!(report.diagnostics[0].level, DiagnosticLevel::Error);
    assert_eq!(report.diagnostics[0].code, "pricing.invalid_config");
    assert!(
        report.diagnostics[0]
            .message
            .contains("unknown field `unexpected`")
    );

    let mut invalid_catalog = PluginComponentSpec::new("pricing");
    invalid_catalog.config = serde_json::from_value(json!({
        "sources": [
            {
                "type": "inline",
                "catalog": {
                    "version": 2,
                    "entries": []
                }
            }
        ]
    }))
    .unwrap();
    let report = test_validate_static_plugin_config(&PluginConfig {
        components: vec![invalid_catalog],
        ..PluginConfig::default()
    });
    assert!(report.has_errors());
    assert!(
        report.diagnostics[0]
            .message
            .contains("unsupported model pricing catalog version 2")
    );

    let duplicate = test_validate_static_plugin_config(&PluginConfig {
        components: vec![
            PluginComponentSpec::new("pricing"),
            PluginComponentSpec::new("pricing"),
        ],
        ..PluginConfig::default()
    });
    assert!(duplicate.has_errors());
    assert!(
        duplicate
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "plugin.duplicate_component")
    );
}
