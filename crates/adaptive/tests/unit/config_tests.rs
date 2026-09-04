// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for config in the NeMo Relay adaptive crate.

use super::*;
use nemo_relay::config_editor::{EditorConfig, EditorFieldKind};
use serde_json::json;

use crate::response_cache::config::{ResponseCacheKeyStrategy, ToolCacheConfig, ToolClass};

#[test]
fn test_adaptive_config_defaults() {
    let config = AdaptiveConfig::default();
    assert_eq!(config.version, 1);
    assert!(config.telemetry.is_none());
    assert!(config.adaptive_hints.is_none());
    assert!(config.tool_parallelism.is_none());
    assert!(config.response_cache.is_none());
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

    let response_cache = ResponseCacheConfig::default();
    assert!(!response_cache.cache_nondeterministic);
    assert_eq!(
        response_cache.key_strategy,
        ResponseCacheKeyStrategy::ExactRequest
    );

    let tools = ToolCacheConfig::default();
    assert!(!tools.enabled);
    assert!(!tools.cache_errors);
    assert_eq!(tools.priority, 100);
}

#[test]
fn test_tool_cache_deserializes_explicit_error_caching_opt_in() {
    let tools: ToolCacheConfig = serde_json::from_value(json!({
        "enabled": true,
        "cache_errors": true,
    }))
    .unwrap();
    assert!(tools.enabled);
    assert!(tools.cache_errors);
}

#[test]
fn test_response_cache_key_strategy_roundtrips_and_preserves_unknown() {
    let logical: ResponseCacheConfig = serde_json::from_value(json!({
        "key_strategy": "logical"
    }))
    .unwrap();
    assert_eq!(logical.key_strategy, ResponseCacheKeyStrategy::Logical);
    assert_eq!(
        serde_json::to_value(logical).unwrap()["key_strategy"],
        json!("logical")
    );

    let unsupported: ResponseCacheConfig = serde_json::from_value(json!({
        "key_strategy": "future"
    }))
    .unwrap();
    assert_eq!(
        unsupported.key_strategy,
        ResponseCacheKeyStrategy::Unknown("future".to_string())
    );
}

#[test]
fn test_backend_spec_in_memory_helper_uses_empty_config() {
    let backend = BackendSpec::in_memory();
    assert_eq!(backend.kind, "in_memory");
    assert!(backend.config.is_empty());

    let default_backend = BackendSpec::default();
    assert_eq!(default_backend.kind, "in_memory");
    assert!(default_backend.config.is_empty());
}

#[cfg(not(feature = "redis-backend"))]
#[test]
fn test_response_cache_redis_backend_requires_the_redis_feature() {
    let mut response_cache = ResponseCacheConfig {
        namespace: "cache-tests".to_string(),
        ..ResponseCacheConfig::default()
    };
    response_cache.backend.kind = "redis".to_string();
    response_cache
        .backend
        .config
        .insert("url".to_string(), json!("redis://127.0.0.1/"));

    let report = crate::runtime::features::AdaptiveRuntime::validate_config(&AdaptiveConfig {
        response_cache: Some(response_cache),
        ..AdaptiveConfig::default()
    });

    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.backend_unavailable")
    );
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
    assert!(config.response_cache.is_none());
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
            "response_cache",
            "policy",
        ]
    );

    let state = schema.field("state").unwrap().schema().unwrap();
    let backend = state.field("backend").unwrap().schema().unwrap();
    assert_eq!(backend.field("kind").unwrap().kind, EditorFieldKind::Enum);

    let telemetry = schema.field("telemetry").unwrap().schema().unwrap();
    let learners = telemetry.field("learners").unwrap();
    assert_eq!(learners.kind, EditorFieldKind::List);
    assert_eq!(
        learners.list_item.expect("learners item metadata").kind,
        EditorFieldKind::String
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

    let response_cache = schema.field("response_cache").unwrap().schema().unwrap();
    assert_eq!(
        response_cache.field("ttl_seconds").unwrap().kind,
        EditorFieldKind::Integer
    );
    assert_eq!(
        response_cache.field("bypass_rate").unwrap().kind,
        EditorFieldKind::Float
    );
    assert!(
        response_cache.field("skip_keys").is_none(),
        "exact-match cache config must not expose arbitrary key omission"
    );
    let response_cache_backend = response_cache.field("backend").unwrap().schema().unwrap();
    assert_eq!(
        response_cache_backend.field("kind").unwrap().kind,
        EditorFieldKind::Enum
    );
    #[cfg(not(feature = "redis-backend"))]
    assert_eq!(
        response_cache_backend.field("kind").unwrap().enum_values,
        &["in_memory"]
    );
    #[cfg(feature = "redis-backend")]
    assert_eq!(
        response_cache_backend.field("kind").unwrap().enum_values,
        &["in_memory", "redis"]
    );

    let tools = response_cache.field("tools").unwrap();
    assert_eq!(tools.kind, EditorFieldKind::Section);
    assert!(tools.optional);
    let tools = tools.schema().unwrap();
    assert_eq!(
        tools.field("enabled").unwrap().kind,
        EditorFieldKind::Boolean
    );
    assert_eq!(
        tools.field("priority").unwrap().kind,
        EditorFieldKind::Integer
    );
    assert_eq!(
        tools.field("default").unwrap().kind,
        EditorFieldKind::Section
    );
}

#[test]
fn test_response_cache_key_strategy_editor_field_is_typed() {
    let schema = AdaptiveConfig::editor_schema();
    let response_cache = schema.field("response_cache").unwrap().schema().unwrap();
    let key_strategy = response_cache.field("key_strategy").unwrap();

    assert_eq!(key_strategy.kind, EditorFieldKind::Enum);
    assert_eq!(key_strategy.enum_values, &["exact_request", "logical"]);
}

#[test]
fn adaptive_editor_schema_describes_structured_collections() {
    let schema = AdaptiveConfig::editor_schema();
    let backend = schema
        .field("state")
        .unwrap()
        .schema()
        .unwrap()
        .field("backend")
        .unwrap()
        .schema()
        .unwrap();
    let backend_config = backend.field("config").unwrap();
    assert_eq!(backend_config.kind, EditorFieldKind::DiscriminatedSection);
    assert_eq!(backend_config.tagged_union.unwrap().discriminator, "kind");

    let response_cache = schema.field("response_cache").unwrap().schema().unwrap();
    let header_allowlist = response_cache.field("header_allowlist").unwrap();
    assert_eq!(header_allowlist.kind, EditorFieldKind::List);
    assert_eq!(
        header_allowlist.list_item.unwrap().kind,
        EditorFieldKind::String
    );

    let tools = response_cache.field("tools").unwrap().schema().unwrap();
    for field_name in ["classes", "overrides"] {
        let field = tools.field(field_name).unwrap();
        assert_eq!(field.kind, EditorFieldKind::Map);
        assert_eq!(field.list_item.unwrap().kind, EditorFieldKind::Section);
    }
}

#[test]
fn tool_class_editor_schema_exposes_optional_version() {
    let tool_class = ToolClass::editor_schema();
    assert_eq!(
        tool_class.field("tool_version").unwrap().kind,
        EditorFieldKind::String
    );
    assert!(tool_class.field("tool_version").unwrap().optional);
    for field_name in ["arg_skip", "members"] {
        let field = tool_class.field(field_name).unwrap();
        assert_eq!(field.kind, EditorFieldKind::List);
        assert_eq!(
            field.list_item.expect("string-list item metadata").kind,
            EditorFieldKind::String
        );
    }

    let arg_skip = crate::response_cache::config::ToolOverride::editor_schema()
        .field("arg_skip")
        .unwrap();
    assert_eq!(arg_skip.kind, EditorFieldKind::List);
    assert!(arg_skip.optional);
    assert_eq!(
        arg_skip.list_item.expect("string-list item metadata").kind,
        EditorFieldKind::String
    );
}
