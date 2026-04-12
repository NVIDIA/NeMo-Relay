// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::sync::{Mutex, OnceLock};

use nemo_flow::plugin::{DiagnosticLevel, UnsupportedBehavior, clear_plugin_configuration};
use serde_json::json;

static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn test_mutex() -> &'static Mutex<()> {
    TEST_MUTEX.get_or_init(|| Mutex::new(()))
}

#[test]
fn component_spec_conversion_preserves_kind_and_config_payload() {
    let spec = ComponentSpec::new(AdaptiveConfig {
        agent_id: Some("agent-1".to_string()),
        ..AdaptiveConfig::default()
    });
    let plugin_spec: PluginComponentSpec = spec.into();

    assert_eq!(plugin_spec.kind, ADAPTIVE_PLUGIN_KIND);
    assert!(plugin_spec.enabled);
    assert_eq!(plugin_spec.config.get("agent_id"), Some(&json!("agent-1")));
}

#[test]
fn validate_adaptive_plugin_config_reports_unknown_fields_and_backend_errors() {
    let config = json!({
        "version": 1,
        "state": {
            "backend": {
                "kind": "bogus",
                "config": {"surprise": true}
            }
        },
        "tool_parallelism": {
            "mode": "invalid",
            "extra": 1
        },
        "extra_root": true,
        "policy": {
            "unknown_component": "warn",
            "unknown_field": "warn",
            "unsupported_value": "error"
        }
    });

    let diagnostics = validate_adaptive_plugin_config(config.as_object().unwrap());
    assert!(
        diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.unknown_field")
    );
    assert!(
        diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.unknown_backend")
    );
    assert!(
        diagnostics
            .iter()
            .any(|diag| diag.code == "adaptive.unsupported_value")
    );
    assert!(
        diagnostics
            .iter()
            .any(|diag| diag.level == DiagnosticLevel::Error)
    );
}

#[test]
fn register_adaptive_component_is_idempotent_and_deregisters_cleanly() {
    let _guard = test_mutex().lock().unwrap();
    let _ = clear_plugin_configuration();
    let _ = deregister_adaptive_component();

    register_adaptive_component().unwrap();
    register_adaptive_component().unwrap();
    assert!(lookup_plugin(ADAPTIVE_PLUGIN_KIND).is_some());

    assert!(deregister_adaptive_component());
    assert!(!deregister_adaptive_component());
}

#[test]
fn parse_adaptive_config_preserves_policy_behavior() {
    let config = json!({
        "version": 1,
        "policy": {
            "unknown_component": "ignore",
            "unknown_field": "warn",
            "unsupported_value": "error"
        }
    });

    let parsed = parse_adaptive_config(config.as_object().unwrap()).unwrap();
    assert_eq!(parsed.policy.unknown_component, UnsupportedBehavior::Ignore);
    assert_eq!(parsed.policy.unknown_field, UnsupportedBehavior::Warn);
    assert_eq!(parsed.policy.unsupported_value, UnsupportedBehavior::Error);
}
