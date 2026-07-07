// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for plugin component in the NeMo Relay adaptive crate.

use super::*;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::llm::llm_request_intercepts;
use nemo_relay::api::runtime::NemoRelayContextState;
use nemo_relay::api::runtime::global_context;
use nemo_relay::plugin::{DiagnosticLevel, UnsupportedBehavior, clear_plugin_configuration};
use nemo_relay::plugin::{Plugin, PluginRegistrationContext, rollback_registrations};
use serde_json::json;

fn reset_global() {
    let _ = clear_plugin_configuration();
    let _ = deregister_adaptive_component();
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
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

#[tokio::test(flavor = "current_thread")]
async fn register_adaptive_component_is_idempotent_and_deregisters_cleanly() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
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

#[test]
fn parse_adaptive_config_rejects_invalid_shapes() {
    let config = json!({
        "version": "wrong-type",
    });

    let err = parse_adaptive_config(config.as_object().unwrap()).unwrap_err();
    assert!(err.to_string().contains("invalid adaptive plugin config"));
}

#[test]
fn acg_component_parse_adaptive_config_preserves_existing_acg_surface() {
    let config = json!({
        "version": 1,
        "acg": {
            "provider": "openai",
            "observation_window": 24,
            "priority": 17,
            "stability_thresholds": {
                "stable_threshold": 0.99,
                "semi_stable_threshold": 0.75,
                "min_observations_for_full_confidence": 12
            }
        }
    });

    let parsed = parse_adaptive_config(config.as_object().unwrap()).unwrap();
    let acg = parsed.acg.expect("acg config should parse");

    assert_eq!(acg.provider, "openai");
    assert_eq!(acg.observation_window, 24);
    assert_eq!(acg.priority, 17);
    assert_eq!(acg.stability_thresholds.stable_threshold, 0.99);
    assert_eq!(acg.stability_thresholds.semi_stable_threshold, 0.75);
    assert_eq!(
        acg.stability_thresholds
            .min_observations_for_full_confidence,
        12
    );
}

#[test]
fn acg_component_config_rejects_new_economics_or_breakpoint_knobs() {
    let config = json!({
        "version": 1,
        "acg": {
            "provider": "anthropic",
            "observation_window": 24,
            "priority": 17,
            "economics_window": 60,
            "breakpoint_budget": 3
        }
    });

    let diagnostics = validate_adaptive_plugin_config(config.as_object().unwrap());
    assert!(diagnostics.iter().any(|diag| {
        diag.code == "adaptive.unknown_field"
            && diag.component.as_deref() == Some("acg")
            && diag.field.as_deref() == Some("economics_window")
    }));
    assert!(diagnostics.iter().any(|diag| {
        diag.code == "adaptive.unknown_field"
            && diag.component.as_deref() == Some("acg")
            && diag.field.as_deref() == Some("breakpoint_budget")
    }));
}

#[test]
fn validate_unknown_fields_honors_policy_levels() {
    let mut diagnostics = vec![];
    let config = serde_json::Map::from_iter([("extra".to_string(), json!(true))]);

    validate_unknown_fields(
        &mut diagnostics,
        &ConfigPolicy {
            unknown_field: UnsupportedBehavior::Ignore,
            ..ConfigPolicy::default()
        },
        Some("adaptive".to_string()),
        &config,
        &[],
    );
    assert!(diagnostics.is_empty());

    validate_unknown_fields(
        &mut diagnostics,
        &ConfigPolicy {
            unknown_field: UnsupportedBehavior::Warn,
            ..ConfigPolicy::default()
        },
        Some("adaptive".to_string()),
        &config,
        &[],
    );
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].level, DiagnosticLevel::Warning);

    diagnostics.clear();
    validate_unknown_fields(
        &mut diagnostics,
        &ConfigPolicy {
            unknown_field: UnsupportedBehavior::Error,
            ..ConfigPolicy::default()
        },
        Some("adaptive".to_string()),
        &config,
        &[],
    );
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].level, DiagnosticLevel::Error);
}

#[test]
fn validate_backend_config_fields_only_flags_known_backend_extras() {
    let policy = ConfigPolicy {
        unknown_field: UnsupportedBehavior::Warn,
        ..ConfigPolicy::default()
    };
    let backend_config = serde_json::Map::from_iter([("surprise".to_string(), json!(true))]);
    let mut diagnostics = vec![];

    validate_backend_config_fields(&mut diagnostics, &policy, "redis", &backend_config);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].component.as_deref(), Some("redis"));

    diagnostics.clear();
    validate_backend_config_fields(&mut diagnostics, &policy, "in_memory", &backend_config);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].component.as_deref(), Some("in_memory"));

    diagnostics.clear();
    validate_backend_config_fields(&mut diagnostics, &policy, "unknown", &backend_config);
    assert!(diagnostics.is_empty());
}

#[test]
fn adaptive_to_plugin_error_maps_all_non_redis_variants() {
    assert!(matches!(
        adaptive_to_plugin_error(AdaptiveError::InvalidConfig("bad".into())),
        nemo_relay::plugin::PluginError::InvalidConfig(message) if message == "bad"
    ));
    assert!(matches!(
        adaptive_to_plugin_error(AdaptiveError::NotFound("missing".into())),
        nemo_relay::plugin::PluginError::NotFound(message) if message == "missing"
    ));
    assert!(matches!(
        adaptive_to_plugin_error(AdaptiveError::Storage("store".into())),
        nemo_relay::plugin::PluginError::Internal(message) if message == "store"
    ));
    assert!(matches!(
        adaptive_to_plugin_error(AdaptiveError::Internal("internal".into())),
        nemo_relay::plugin::PluginError::Internal(message) if message == "internal"
    ));
    assert!(matches!(
        adaptive_to_plugin_error(AdaptiveError::RegistrationFailed("register".into())),
        nemo_relay::plugin::PluginError::RegistrationFailed(message) if message == "register"
    ));
    assert!(matches!(
        adaptive_to_plugin_error(AdaptiveError::ChannelClosed("closed".into())),
        nemo_relay::plugin::PluginError::Internal(message) if message == "closed"
    ));
    let serde_error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    assert!(matches!(
        adaptive_to_plugin_error(AdaptiveError::Serialization(serde_error)),
        nemo_relay::plugin::PluginError::Serialization(_)
    ));
}

#[cfg(feature = "redis-backend")]
#[test]
fn adaptive_to_plugin_error_maps_redis_variant() {
    let redis_error = redis::Client::open("redis://bad host").unwrap_err();
    assert!(matches!(
        adaptive_to_plugin_error(AdaptiveError::Redis(redis_error)),
        nemo_relay::plugin::PluginError::Internal(message) if message.contains("Redis URL")
    ));
}

#[test]
fn adaptive_plugin_reports_invalid_plugin_config_diagnostics() {
    let plugin = AdaptivePlugin;
    let diagnostics = plugin.validate(
        json!({
            "version": "wrong-type",
        })
        .as_object()
        .unwrap(),
    );

    assert_eq!(plugin.plugin_kind(), ADAPTIVE_PLUGIN_KIND);
    assert!(!plugin.allows_multiple_components());
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, "adaptive.invalid_plugin_config");
    assert_eq!(diagnostics[0].level, DiagnosticLevel::Error);
}

#[test]
fn validate_adaptive_plugin_config_reports_component_specific_unknown_fields() {
    let config = json!({
        "version": 1,
        "telemetry": {
            "subscriber_name": "adaptive-sub",
            "extra": true
        },
        "adaptive_hints": {
            "inject_header": true,
            "extra": true
        },
        "tool_parallelism": {
            "mode": "observe_only"
        },
        "policy": {
            "unknown_field": "warn"
        }
    });

    let diagnostics = validate_adaptive_plugin_config(config.as_object().unwrap());
    assert!(diagnostics.iter().any(|diag| {
        diag.code == "adaptive.unknown_field"
            && diag.component.as_deref() == Some("telemetry")
            && diag.field.as_deref() == Some("extra")
    }));
    assert!(diagnostics.iter().any(|diag| {
        diag.code == "adaptive.unknown_field"
            && diag.component.as_deref() == Some("adaptive_hints")
            && diag.field.as_deref() == Some("extra")
    }));
}

#[test]
fn validate_adaptive_plugin_config_accepts_topology_sections() {
    let config = json!({
        "version": 1,
        "adaptive_hints": {
            "governor": {"enabled": true, "epsilon": 2.0}
        },
        "tool_parallelism": {
            "mode": "observe_only",
            "drift": {"enabled": true, "threshold": 0.5}
        },
        "acg": {
            "provider": "anthropic",
            "convergence": {"enabled": true, "epsilon": 0.01, "stability_window": 3}
        },
        "convergence": {"enabled": true, "epsilon": 0.01, "stability_window": 3},
        "policy": {
            "unknown_field": "error",
            "unsupported_value": "error"
        }
    });

    let diagnostics = validate_adaptive_plugin_config(config.as_object().unwrap());
    assert!(
        diagnostics
            .iter()
            .all(|diag| diag.code != "adaptive.unknown_field"),
        "topology config fields should not be reported as unknown: {diagnostics:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn adaptive_plugin_registers_runtime_and_rolls_back_registration() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_global();

    let plugin = AdaptivePlugin;
    let config = json!({
        "adaptive_hints": {
            "priority": 7,
            "inject_header": true
        }
    });
    let mut ctx = PluginRegistrationContext::with_namespace("adaptive.test.");

    plugin
        .register(config.as_object().unwrap(), &mut ctx)
        .await
        .unwrap();

    let request = llm_request_intercepts(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({}),
        },
    )
    .unwrap();
    assert!(request.request.headers.is_empty());

    let mut registrations = ctx.into_registrations();
    assert_eq!(registrations.len(), 1);
    rollback_registrations(&mut registrations);
}
