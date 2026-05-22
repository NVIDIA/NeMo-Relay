// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for anthropic plugin in the NeMo Relay adaptive crate.

use std::sync::Arc;

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use crate::acg::capability::CapabilityRegistry;
use crate::acg::plugin::{PluginInput, ProviderPlugin};
use crate::acg::plugin_registry::PluginRegistry;
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
    TokenizationMetadata,
};
use crate::acg::translation::{AnthropicHintDirective, HintTarget};
use crate::acg::types::{
    AgentIdentity, CacheStabilityIntent, CompressionIntent, ModelClass, ModelRoutingIntent,
    OptimizationIntent, OptimizationIntentBundle, RetentionIntent, RetentionTier, SharingScope,
    TranslationStatus,
};
use nemo_relay::api::llm::LlmRequest;

use super::AnthropicCachePlugin;

// -------------------------------------------------------------------
// Test helpers
// -------------------------------------------------------------------

fn sample_agent_identity() -> AgentIdentity {
    AgentIdentity {
        agent_id: "test-agent".to_string(),
        template_version: "1.0.0".to_string(),
        toolset_hash: "abc123".to_string(),
        model_family: "claude".to_string(),
        tenant_scope: "test-tenant".to_string(),
    }
}

fn sample_intent_bundle(intents: Vec<OptimizationIntent>) -> OptimizationIntentBundle {
    OptimizationIntentBundle {
        request_id: Uuid::new_v4(),
        agent_identity: sample_agent_identity(),
        policy_version: "1.0.0".to_string(),
        intents,
        created_at: Utc::now(),
    }
}

/// Build a large system prompt string (>= 4096 characters for token estimation).
fn large_system_text(min_chars: usize) -> String {
    let base = "You are a helpful AI assistant. You must follow instructions carefully. ";
    base.repeat((min_chars / base.len()) + 1)
}

fn sample_anthropic_request_with_system_array(model: &str) -> LlmRequest {
    let system_text = large_system_text(5000);
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": model,
            "system": [
                {
                    "type": "text",
                    "text": system_text
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                }
            ]
        }),
    }
}

fn sample_anthropic_request_with_system_string(model: &str) -> LlmRequest {
    let system_text = large_system_text(5000);
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": model,
            "system": system_text,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        }),
    }
}

fn sample_prompt_ir_with_system(char_count: usize) -> PromptIR {
    let text = large_system_text(char_count);
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![
            PromptBlock {
                span_id: SpanId("system-0".to_string()),
                sequence_index: 0,
                role: PromptRole::System,
                content: text,
                content_type: BlockContentType::Text,
                provenance: ProvenanceLabel::System,
                sensitivity: SensitivityLabel::Public,
                token_metadata: Some(TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count: 2000,
                }),
            },
            PromptBlock {
                span_id: SpanId("user-1".to_string()),
                sequence_index: 1,
                role: PromptRole::User,
                content: "Hello, how are you?".to_string(),
                content_type: BlockContentType::Text,
                provenance: ProvenanceLabel::User,
                sensitivity: SensitivityLabel::Public,
                token_metadata: Some(TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count: 6,
                }),
            },
        ],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

fn sample_prompt_ir_no_token_metadata(char_count: usize) -> PromptIR {
    let text = large_system_text(char_count);
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![
            PromptBlock {
                span_id: SpanId("system-0".to_string()),
                sequence_index: 0,
                role: PromptRole::System,
                content: text,
                content_type: BlockContentType::Text,
                provenance: ProvenanceLabel::System,
                sensitivity: SensitivityLabel::Public,
                token_metadata: None,
            },
            PromptBlock {
                span_id: SpanId("user-1".to_string()),
                sequence_index: 1,
                role: PromptRole::User,
                content: "Hello".to_string(),
                content_type: BlockContentType::Text,
                provenance: ProvenanceLabel::User,
                sensitivity: SensitivityLabel::Public,
                token_metadata: None,
            },
        ],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

// -------------------------------------------------------------------
// plugin_id() returns "anthropic"
// -------------------------------------------------------------------

#[test]
fn test_plugin_id() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);
    assert_eq!(plugin.plugin_id(), "anthropic");
}

// -------------------------------------------------------------------
// plugin_name() returns "Anthropic Cache Plugin"
// -------------------------------------------------------------------

#[test]
fn test_plugin_name() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);
    assert_eq!(plugin.plugin_name(), "Anthropic Cache Plugin");
}

// -------------------------------------------------------------------
// AnthropicCachePlugin is Send + Sync and object-safe
// -------------------------------------------------------------------

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn test_anthropic_plugin_is_send_sync() {
    assert_send_sync::<AnthropicCachePlugin>();
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);
    let _: Arc<dyn ProviderPlugin> = Arc::new(plugin);
}

// -------------------------------------------------------------------
// capabilities() returns correct features
// -------------------------------------------------------------------

#[test]
fn test_capabilities() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);
    let caps = plugin.capabilities();

    assert_eq!(caps.backend_id, "anthropic");
    assert!(caps.supports(crate::acg::capability::ProviderFeature::ExplicitCacheBreakpoints));
    assert!(caps.supports(crate::acg::capability::ProviderFeature::RetentionTiers));
    assert!(caps.supports(crate::acg::capability::ProviderFeature::StreamingTokenCounts));
    assert!(!caps.supports(crate::acg::capability::ProviderFeature::AutomaticPrefixCaching));
}

#[test]
fn test_build_hint_translation_uses_surface_agnostic_breakpoint_targets() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        },
    )]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let translation = plugin.build_hint_translation(&input).unwrap();
    assert_eq!(translation.hint_plan.provider, "anthropic");
    assert!(matches!(
        translation.hint_plan.directives[0],
        crate::acg::translation::HintDirective::Anthropic(
            AnthropicHintDirective::CanonicalizeToolSchemas
        )
    ));
    assert!(matches!(
        translation.hint_plan.directives[1],
        crate::acg::translation::HintDirective::Anthropic(
            AnthropicHintDirective::CacheBreakpoint {
                target: HintTarget::StablePrefix {
                    end_exclusive: 1,
                    ..
                },
                scope: SharingScope::Session,
            }
        )
    ));

    let debug = format!("{:?}", translation.hint_plan);
    assert!(!debug.contains("\"messages\""));
    assert!(!debug.contains("\"input\""));
    assert!(!debug.contains("\"system\""));
}

#[test]
fn test_anthropic_plugin_source_routes_through_request_surface_appliers() {
    let source = include_str!("../../../src/acg/anthropic_plugin.rs");

    assert!(
        source.contains("request_surfaces::apply_request_surface"),
        "Anthropic plugin should delegate raw request mutation to request surfaces"
    );
    assert!(
        !source.contains("fn inject_cache_control_system("),
        "Anthropic raw request mutation helpers should move out of the plugin"
    );
    assert!(
        !source.contains("fn inject_cache_control_message("),
        "Anthropic raw request mutation helpers should move out of the plugin"
    );
    assert!(
        !source.contains("fn inject_cache_control_tool("),
        "Anthropic raw request mutation helpers should move out of the plugin"
    );
}

#[test]
fn test_plugin_translate_preserves_semantic_translation_report() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        },
    )]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let semantic = plugin.build_hint_translation(&input).unwrap();
    let output = plugin.translate(&input).unwrap();

    assert_eq!(
        output.translation_report.plugin_id,
        semantic.translation_report.plugin_id
    );
    assert_eq!(
        output.translation_report.request_id,
        semantic.translation_report.request_id
    );
    assert_eq!(
        output.translation_report.outcomes.len(),
        semantic.translation_report.outcomes.len()
    );
    assert_eq!(
        output.translation_report.outcomes[0].intent_type,
        semantic.translation_report.outcomes[0].intent_type
    );
    assert_eq!(
        output.translation_report.outcomes[0].status,
        semantic.translation_report.outcomes[0].status
    );
    assert_eq!(
        output.translation_report.outcomes[0].reason,
        semantic.translation_report.outcomes[0].reason
    );
    assert_eq!(
        output.translation_report.outcomes[0].detail,
        semantic.translation_report.outcomes[0].detail
    );
}

// -------------------------------------------------------------------
// translate with single CacheStability intent on system message array
// -------------------------------------------------------------------

#[test]
fn test_translate_single_cache_stability_system_array() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1, // system block only
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        },
    )]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 1);
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Applied
    );

    // Verify cache_control is present on the system content block
    let system = output.translated_request.content.get("system").unwrap();
    let arr = system.as_array().unwrap();
    let last_block = arr.last().unwrap();
    assert!(last_block.get("cache_control").is_some());
    assert_eq!(
        last_block["cache_control"]["type"].as_str().unwrap(),
        "ephemeral"
    );
}

// -------------------------------------------------------------------
// translate with system as plain string converts to array-of-blocks
// -------------------------------------------------------------------

#[test]
fn test_translate_system_string_to_array() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_string("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        },
    )]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Applied
    );

    // System should now be an array
    let system = output.translated_request.content.get("system").unwrap();
    assert!(system.is_array(), "system should be converted to array");
    let arr = system.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["type"].as_str().unwrap(), "text");
    assert!(arr[0].get("cache_control").is_some());
}

// -------------------------------------------------------------------
// translate with 5 CacheStability intents: only 4 breakpoints, 5th Degraded
// -------------------------------------------------------------------

#[test]
fn test_translate_breakpoint_budget_cap_at_4() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    // Build a request with system + 4 user messages
    let system_text = large_system_text(5000);
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4",
            "system": [{"type": "text", "text": system_text}],
            "messages": [
                {"role": "user", "content": "msg1"},
                {"role": "user", "content": "msg2"},
                {"role": "user", "content": "msg3"},
                {"role": "user", "content": "msg4"}
            ]
        }),
    };

    // 5 blocks in the IR: system + 4 user messages, all with enough tokens
    let ir = PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: (0..5)
            .map(|i| PromptBlock {
                span_id: SpanId(format!("block-{i}")),
                sequence_index: i as u32,
                role: if i == 0 {
                    PromptRole::System
                } else {
                    PromptRole::User
                },
                content: large_system_text(5000),
                content_type: BlockContentType::Text,
                provenance: if i == 0 {
                    ProvenanceLabel::System
                } else {
                    ProvenanceLabel::User
                },
                sensitivity: SensitivityLabel::Public,
                token_metadata: Some(TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count: 2000,
                }),
            })
            .collect(),
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    };

    // 5 CacheStability intents
    let intents: Vec<OptimizationIntent> = (1..=5)
        .map(|i| {
            OptimizationIntent::CacheStability(CacheStabilityIntent {
                stability_score: 0.95,
                stable_prefix_end: i,
                recommended_retention_tier: None,
                scope_label: SharingScope::Session,
                confidence: 0.9,
                evidence_count: 50,
            })
        })
        .collect();

    let bundle = sample_intent_bundle(intents);
    let identity = sample_agent_identity();
    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 5);

    // First 4 should be Applied
    for outcome in &output.translation_report.outcomes[..4] {
        assert_eq!(outcome.status, TranslationStatus::Applied);
    }

    // 5th should be Degraded/BackendLimitReached
    let fifth = &output.translation_report.outcomes[4];
    assert_eq!(fifth.status, TranslationStatus::Degraded);
    assert_eq!(
        fifth.reason,
        crate::acg::types::ReasonCode::BackendLimitReached
    );
    assert!(fifth.detail.as_ref().unwrap().contains("max reached"));
}

// -------------------------------------------------------------------
// translate with block below model token minimum -> Degraded
// -------------------------------------------------------------------

#[test]
fn test_translate_token_minimum_enforcement() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    // Use a model with high token minimum (claude-opus-4.5 has 4096)
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-opus-4.5",
            "system": [{"type": "text", "text": "Short system."}],
            "messages": [{"role": "user", "content": "Hi"}]
        }),
    };

    // IR with very small blocks (token_count = 100, well below 4096)
    let ir = PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![PromptBlock {
            span_id: SpanId("system-0".to_string()),
            sequence_index: 0,
            role: PromptRole::System,
            content: "Short system.".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::System,
            sensitivity: SensitivityLabel::Public,
            token_metadata: Some(TokenizationMetadata {
                model_family: "claude".to_string(),
                token_count: 100,
            }),
        }],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    };

    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        },
    )]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 1);
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Degraded
    );
    assert_eq!(
        output.translation_report.outcomes[0].reason,
        crate::acg::types::ReasonCode::BackendLimitReached
    );
    assert!(
        output.translation_report.outcomes[0]
            .detail
            .as_ref()
            .unwrap()
            .contains("below model minimum")
    );
}

// -------------------------------------------------------------------
// translate with RetentionIntent: Ephemeral -> no TTL, SessionDuration -> 1h
// -------------------------------------------------------------------

#[test]
fn test_translate_retention_ephemeral_uses_default_ttl() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![OptimizationIntent::Retention(RetentionIntent {
        recommended_tier: RetentionTier::Ephemeral,
        expected_session_duration_secs: None,
        inter_call_gap_p50_ms: None,
        scope_label: SharingScope::Session,
    })]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 1);
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Applied
    );
    assert!(
        output.translation_report.outcomes[0]
            .detail
            .as_ref()
            .unwrap()
            .contains("5m TTL")
    );
}

#[test]
fn test_translate_retention_short_lived_uses_default_ttl() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![OptimizationIntent::Retention(RetentionIntent {
        recommended_tier: RetentionTier::ShortLived,
        expected_session_duration_secs: None,
        inter_call_gap_p50_ms: None,
        scope_label: SharingScope::Session,
    })]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Applied
    );
    assert!(
        output.translation_report.outcomes[0]
            .detail
            .as_ref()
            .unwrap()
            .contains("5m TTL")
    );
}

#[test]
fn test_translate_retention_session_duration_injects_1h_ttl() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    // First place a breakpoint (CacheStability), then apply retention
    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![
        OptimizationIntent::CacheStability(CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        }),
        OptimizationIntent::Retention(RetentionIntent {
            recommended_tier: RetentionTier::SessionDuration,
            expected_session_duration_secs: Some(3600.0),
            inter_call_gap_p50_ms: Some(5000.0),
            scope_label: SharingScope::Session,
        }),
    ]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 2);

    // CacheStability -> Applied
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Applied
    );

    // Retention -> Applied with 1h TTL
    assert_eq!(
        output.translation_report.outcomes[1].status,
        TranslationStatus::Applied
    );
    assert!(
        output.translation_report.outcomes[1]
            .detail
            .as_ref()
            .unwrap()
            .contains("1h")
    );

    // Verify TTL is in the JSON
    let system = output.translated_request.content.get("system").unwrap();
    let arr = system.as_array().unwrap();
    let cc = arr.last().unwrap().get("cache_control").unwrap();
    assert_eq!(cc["ttl"].as_str().unwrap(), "1h");
}

#[test]
fn test_translate_retention_long_lived_injects_1h_ttl() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![
        OptimizationIntent::CacheStability(CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        }),
        OptimizationIntent::Retention(RetentionIntent {
            recommended_tier: RetentionTier::LongLived,
            expected_session_duration_secs: None,
            inter_call_gap_p50_ms: None,
            scope_label: SharingScope::Session,
        }),
    ]);
    let identity = sample_agent_identity();
    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(
        output.translation_report.outcomes[1].status,
        TranslationStatus::Applied
    );
    assert!(
        output.translation_report.outcomes[1]
            .detail
            .as_ref()
            .unwrap()
            .contains("1h")
    );
}

#[test]
fn test_translate_retention_permanent_injects_1h_ttl() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![
        OptimizationIntent::CacheStability(CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        }),
        OptimizationIntent::Retention(RetentionIntent {
            recommended_tier: RetentionTier::Permanent,
            expected_session_duration_secs: None,
            inter_call_gap_p50_ms: None,
            scope_label: SharingScope::Session,
        }),
    ]);
    let identity = sample_agent_identity();
    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(
        output.translation_report.outcomes[1].status,
        TranslationStatus::Applied
    );
}

// -------------------------------------------------------------------
// translate with retention but no breakpoints -> Degraded
// -------------------------------------------------------------------

#[test]
fn test_translate_retention_no_breakpoints_degrades() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    // Only retention intent, no cache stability -> no breakpoints placed
    let bundle = sample_intent_bundle(vec![OptimizationIntent::Retention(RetentionIntent {
        recommended_tier: RetentionTier::SessionDuration,
        expected_session_duration_secs: Some(3600.0),
        inter_call_gap_p50_ms: None,
        scope_label: SharingScope::Session,
    })]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 1);
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Degraded
    );
    assert!(
        output.translation_report.outcomes[0]
            .detail
            .as_ref()
            .unwrap()
            .contains("no breakpoints")
    );
}

// -------------------------------------------------------------------
// translate with non-cache intents -> Ignored/NotRelevant
// -------------------------------------------------------------------

#[test]
fn test_translate_non_cache_intents_ignored() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![
        OptimizationIntent::ModelRouting(ModelRoutingIntent {
            model_class: ModelClass::Premium,
            complexity_score: 0.7,
            criticality: 0.9,
            fallback_allowed: true,
        }),
        OptimizationIntent::Compression(CompressionIntent {
            block_id: "block-0".to_string(),
            compression_ratio: 0.5,
            reversible: true,
            contribution_score: 0.8,
        }),
    ]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 2);
    for outcome in &output.translation_report.outcomes {
        assert_eq!(outcome.status, TranslationStatus::Ignored);
        assert_eq!(outcome.reason, crate::acg::types::ReasonCode::NotRelevant);
    }
}

// -------------------------------------------------------------------
// translate with empty intent bundle -> 0 outcomes, cloned request
// -------------------------------------------------------------------

#[test]
fn test_translate_empty_bundle() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 0);
}

// -------------------------------------------------------------------
// translate with missing token_metadata -> chars/4 heuristic, Degraded
// -------------------------------------------------------------------

#[test]
fn test_translate_missing_token_metadata_fallback() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    // Use a model with 1024 min tokens (claude-sonnet-4)
    // The system text is ~5000 chars -> chars/4 ~= 1250, which is above 1024
    let request = sample_anthropic_request_with_system_array("claude-sonnet-4");
    let ir = sample_prompt_ir_no_token_metadata(5000);
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        },
    )]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translation_report.outcomes.len(), 1);
    // Should be Degraded because token metadata was missing (fallback used)
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Degraded
    );
    assert_eq!(
        output.translation_report.outcomes[0].reason,
        crate::acg::types::ReasonCode::InsufficientEvidence
    );
    assert!(
        output.translation_report.outcomes[0]
            .detail
            .as_ref()
            .unwrap()
            .contains("chars/4")
    );
}

// -------------------------------------------------------------------
// translate with tools -> canonicalize tool schemas
// -------------------------------------------------------------------

#[test]
fn test_translate_canonicalizes_tool_schemas() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);

    // Request with tools that have non-canonical key ordering
    let system_text = large_system_text(5000);
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4",
            "system": [{"type": "text", "text": system_text}],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            ],
            "messages": [
                {"role": "user", "content": "Search for rust"}
            ]
        }),
    };

    let ir = sample_prompt_ir_with_system(5000);
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.9,
            evidence_count: 50,
        },
    )]);
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();

    // Verify tools are still present and have been processed
    let tools = output
        .translated_request
        .content
        .get("tools")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(tools.len(), 1);

    // Verify the tool schema is now canonicalized (keys sorted)
    let tool = &tools[0];
    let tool_str = serde_json::to_string(tool).unwrap();
    // In RFC 8785 canonical form, "description" comes before "input_schema" (alphabetical)
    let desc_pos = tool_str.find("description").unwrap();
    let schema_pos = tool_str.find("input_schema").unwrap();
    assert!(
        desc_pos < schema_pos,
        "keys should be in alphabetical order after canonicalization"
    );
}

// -------------------------------------------------------------------
// Constructor takes &CapabilityRegistry
// -------------------------------------------------------------------

#[test]
fn test_constructor_takes_registry() {
    let registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&registry);
    // Verify it can resolve model capabilities
    let caps = plugin.capabilities();
    assert_eq!(caps.backend_id, "anthropic");
}

// -------------------------------------------------------------------
// Integration: plugin in PluginRegistry alongside PassthroughPlugin
// -------------------------------------------------------------------

#[test]
fn test_anthropic_plugin_in_registry_with_passthrough() {
    let mut plugin_registry = PluginRegistry::new();
    let cap_registry = CapabilityRegistry::with_defaults();

    let anthropic: Arc<dyn ProviderPlugin> = Arc::new(AnthropicCachePlugin::new(&cap_registry));
    let passthrough: Arc<dyn ProviderPlugin> = Arc::new(crate::acg::passthrough::PassthroughPlugin);

    plugin_registry.register(anthropic).unwrap();
    plugin_registry.register(passthrough).unwrap();

    assert_eq!(plugin_registry.len(), 2);

    let retrieved = plugin_registry.get("anthropic").unwrap();
    assert_eq!(retrieved.plugin_id(), "anthropic");
    assert_eq!(retrieved.plugin_name(), "Anthropic Cache Plugin");

    let pass = plugin_registry.get("passthrough").unwrap();
    assert_eq!(pass.plugin_id(), "passthrough");
}

// -------------------------------------------------------------------
// Full round-trip integration test
// -------------------------------------------------------------------

#[test]
fn test_full_round_trip_anthropic_breakpoint() {
    let cap_registry = CapabilityRegistry::with_defaults();
    let plugin = AnthropicCachePlugin::new(&cap_registry);

    // 1. Build a PromptIR with 3 blocks: large system (~2000 chars), tool schema, user msg
    let system_text = large_system_text(8000);
    let ir = PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![
            PromptBlock {
                span_id: SpanId("system-0".to_string()),
                sequence_index: 0,
                role: PromptRole::System,
                content: system_text.clone(),
                content_type: BlockContentType::Text,
                provenance: ProvenanceLabel::System,
                sensitivity: SensitivityLabel::Public,
                token_metadata: Some(TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count: 2500,
                }),
            },
            PromptBlock {
                span_id: SpanId("tool-1".to_string()),
                sequence_index: 1,
                role: PromptRole::System,
                content: r#"{"name":"search","description":"Web search"}"#.to_string(),
                content_type: BlockContentType::ToolSchema,
                provenance: ProvenanceLabel::System,
                sensitivity: SensitivityLabel::Public,
                token_metadata: Some(TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count: 20,
                }),
            },
            PromptBlock {
                span_id: SpanId("user-2".to_string()),
                sequence_index: 2,
                role: PromptRole::User,
                content: "Search for Rust programming language".to_string(),
                content_type: BlockContentType::Text,
                provenance: ProvenanceLabel::User,
                sensitivity: SensitivityLabel::Public,
                token_metadata: Some(TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count: 7,
                }),
            },
        ],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    };

    // 2. Create a CacheStabilityIntent with stable_prefix_end=2 (system + tool stable)
    let intents = vec![OptimizationIntent::CacheStability(CacheStabilityIntent {
        stability_score: 0.98,
        stable_prefix_end: 2,
        recommended_retention_tier: Some(RetentionTier::SessionDuration),
        scope_label: SharingScope::Session,
        confidence: 0.95,
        evidence_count: 100,
    })];

    // 3. Build an Anthropic-format request
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4",
            "system": [
                {"type": "text", "text": system_text}
            ],
            "tools": [
                {
                    "name": "search",
                    "description": "Web search",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    }
                }
            ],
            "messages": [
                {"role": "user", "content": "Search for Rust programming language"}
            ]
        }),
    };

    let bundle = sample_intent_bundle(intents);
    let identity = sample_agent_identity();
    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    // 4. Call translate
    let output = plugin.translate(&input).unwrap();

    // 5. Verify cache_control present on the tool definition targeted by the stable prefix
    let tools = output.translated_request.content.get("tools").unwrap();
    let arr = tools.as_array().unwrap();
    let last = arr.last().unwrap();
    assert!(
        last.get("cache_control").is_some(),
        "cache_control should be present on the stable tool definition"
    );
    assert_eq!(last["cache_control"]["type"].as_str().unwrap(), "ephemeral");

    // 6. Verify TranslationReport has Applied status
    assert_eq!(output.translation_report.outcomes.len(), 1);
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Applied
    );
    assert_eq!(output.translation_report.plugin_id, "anthropic");
}
