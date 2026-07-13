// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for config in the NeMo Relay adaptive crate.

use super::*;
use nemo_relay::config_editor::{EditorConfig, EditorFieldKind};
use serde_json::json;

#[test]
fn test_adaptive_config_defaults() {
    let config = AdaptiveConfig::default();
    assert_eq!(config.version, 1);
    assert!(config.telemetry.is_none());
    assert!(config.adaptive_hints.is_none());
    assert!(config.tool_parallelism.is_none());
    assert_eq!(
        config.policy.unknown_component,
        nemo_relay::plugin::UnsupportedBehavior::Warn
    );
}

#[test]
fn test_typed_section_helpers_default() {
    let adaptive_hints = AdaptiveHintsComponentConfig::default();
    assert_eq!(adaptive_hints.priority, 100);
    assert!(adaptive_hints.inject_header);

    let tool_parallelism = ToolParallelismComponentConfig::default();
    assert_eq!(tool_parallelism.mode, "observe_only");
}

#[test]
fn test_backend_spec_in_memory_helper_uses_empty_config() {
    let backend = BackendSpec::in_memory();
    assert_eq!(backend.kind, "in_memory");
    assert!(backend.config.is_empty());
}

#[cfg(feature = "redis-backend")]
#[test]
fn test_backend_spec_redis_helper_sets_expected_fields() {
    let backend = BackendSpec::redis("redis://127.0.0.1/", "adaptive:");
    assert_eq!(backend.kind, "redis");
    assert_eq!(
        backend.config.get("url"),
        Some(&json!("redis://127.0.0.1/"))
    );
    assert_eq!(backend.config.get("key_prefix"), Some(&json!("adaptive:")));
}

#[test]
fn test_adaptive_config_deserialization_applies_field_defaults() {
    let config: AdaptiveConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(config.version, 1);
    assert!(config.state.is_none());
    assert!(config.telemetry.is_none());
    assert!(config.adaptive_hints.is_none());
    assert!(config.tool_parallelism.is_none());
}

#[test]
fn test_component_configs_deserialize_with_default_helpers() {
    let adaptive_hints: AdaptiveHintsComponentConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(adaptive_hints.priority, 100);
    assert!(!adaptive_hints.break_chain);
    assert!(adaptive_hints.inject_header);
    assert_eq!(adaptive_hints.inject_body_path, "nvext.agent_hints");

    let tool_parallelism: ToolParallelismComponentConfig =
        serde_json::from_value(json!({})).unwrap();
    assert_eq!(tool_parallelism.priority, 100);
    assert_eq!(tool_parallelism.mode, "observe_only");
}

#[test]
fn test_adaptive_editor_schema_covers_canonical_options() {
    let schema = AdaptiveConfig::editor_schema();
    let fields = schema
        .fields
        .iter()
        .map(|field| field.name)
        .collect::<Vec<_>>();
    assert_eq!(
        fields,
        vec![
            "agent_id",
            "state",
            "telemetry",
            "adaptive_hints",
            "tool_parallelism",
            "acg",
            "policy",
        ]
    );

    let state = schema.field("state").unwrap().schema().unwrap();
    let backend = state.field("backend").unwrap().schema().unwrap();
    assert_eq!(backend.field("kind").unwrap().kind, EditorFieldKind::Enum);
    assert_eq!(backend.field("config").unwrap().kind, EditorFieldKind::Json);

    let telemetry = schema.field("telemetry").unwrap().schema().unwrap();
    assert_eq!(
        telemetry.field("learners").unwrap().kind,
        EditorFieldKind::List
    );

    let acg = schema.field("acg").unwrap().schema().unwrap();
    let thresholds = acg.field("stability_thresholds").unwrap().schema().unwrap();
    assert_eq!(
        thresholds.field("stable_threshold").unwrap().kind,
        EditorFieldKind::Float
    );
    assert_eq!(
        thresholds
            .field("min_observations_for_full_confidence")
            .unwrap()
            .kind,
        EditorFieldKind::Integer
    );
}
