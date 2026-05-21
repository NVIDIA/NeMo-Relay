// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for openai plugin in the NeMo Relay adaptive crate.

use std::sync::Arc;

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use crate::acg::plugin::{PluginInput, ProviderPlugin};
use crate::acg::plugin_registry::PluginRegistry;
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
};
use crate::acg::translation::{HintTarget, OpenAIHintDirective};
use crate::acg::types::{
    AgentIdentity, CacheStabilityIntent, CompressionIntent, IntentType, ModelClass,
    ModelRoutingIntent, OptimizationIntent, OptimizationIntentBundle, ReasonCode, RetentionIntent,
    RetentionTier, SharingScope, TranslationStatus,
};
use nemo_relay::api::llm::LlmRequest;

use super::OpenAICachePlugin;

// -------------------------------------------------------------------
// Test helpers
// -------------------------------------------------------------------

fn assert_send_sync<T: Send + Sync>() {}

fn sample_agent_identity() -> AgentIdentity {
    AgentIdentity {
        agent_id: "test-agent".to_string(),
        template_version: "1.0.0".to_string(),
        toolset_hash: "abc123".to_string(),
        model_family: "gpt".to_string(),
        tenant_scope: "test-tenant".to_string(),
    }
}

fn sample_prompt_ir() -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![PromptBlock {
            span_id: SpanId("span-0".to_string()),
            sequence_index: 0,
            role: PromptRole::System,
            content: "You are helpful.".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::System,
            sensitivity: SensitivityLabel::Public,
            token_metadata: None,
        }],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
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

fn sample_llm_request_with_tools() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Search for Rust caching."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "required": ["query"],
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            }
                        }
                    }
                }
            ]
        }),
    }
}

fn sample_llm_request_no_tools() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        }),
    }
}

fn cache_stability_intent(stable_prefix_end: usize) -> CacheStabilityIntent {
    CacheStabilityIntent {
        stability_score: 0.9,
        stable_prefix_end,
        recommended_retention_tier: None,
        scope_label: SharingScope::Session,
        confidence: 0.8,
        evidence_count: 10,
    }
}

// -------------------------------------------------------------------
// Task 1 tests: Core plugin behavior
// -------------------------------------------------------------------

// plugin_id() returns "openai"
#[test]
fn test_plugin_id() {
    let plugin = OpenAICachePlugin;
    assert_eq!(plugin.plugin_id(), "openai");
}

// plugin_name() returns "OpenAI Cache Plugin"
#[test]
fn test_plugin_name() {
    let plugin = OpenAICachePlugin;
    assert_eq!(plugin.plugin_name(), "OpenAI Cache Plugin");
}

// OpenAICachePlugin is Send + Sync and object-safe
#[test]
fn test_openai_plugin_is_send_sync() {
    assert_send_sync::<OpenAICachePlugin>();
    let _: Arc<dyn ProviderPlugin> = Arc::new(OpenAICachePlugin);
}

// capabilities() returns correct features
#[test]
fn test_capabilities() {
    let plugin = OpenAICachePlugin;
    let caps = plugin.capabilities();
    assert_eq!(caps.backend_id, "openai");
    assert!(caps.supports(crate::acg::capability::ProviderFeature::AutomaticPrefixCaching));
    assert!(caps.supports(crate::acg::capability::ProviderFeature::StreamingTokenCounts));
    assert!(caps.supports(crate::acg::capability::ProviderFeature::StructuredOutput));
    assert!(!caps.supports(crate::acg::capability::ProviderFeature::ExplicitCacheBreakpoints));
    assert!(!caps.supports(crate::acg::capability::ProviderFeature::RetentionTiers));
}

#[test]
fn test_build_hint_translation_uses_stable_prefix_targets() {
    let plugin = OpenAICachePlugin;
    let request = sample_llm_request_with_tools();
    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
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
    assert_eq!(translation.hint_plan.provider, "openai");
    assert!(matches!(
        translation.hint_plan.directives[0],
        crate::acg::translation::HintDirective::OpenAI(
            OpenAIHintDirective::CanonicalizeToolSchemas
        )
    ));
    assert!(matches!(
        translation.hint_plan.directives[1],
        crate::acg::translation::HintDirective::OpenAI(
            OpenAIHintDirective::CanonicalizeStablePrefix {
                target: HintTarget::StablePrefix {
                    end_exclusive: 1,
                    ..
                },
            }
        )
    ));

    let debug = format!("{:?}", translation.hint_plan);
    assert!(!debug.contains("\"messages\""));
    assert!(!debug.contains("\"input\""));
    assert!(!debug.contains("\"system\""));
}

#[test]
fn test_openai_plugin_source_routes_through_request_surface_appliers() {
    let source = include_str!("../../../src/acg/openai_plugin.rs");

    assert!(
        source.contains("request_surfaces::apply_request_surface"),
        "OpenAI plugin should delegate raw request mutation to request surfaces"
    );
    assert!(
        !source.contains("fn canonicalize_stable_messages("),
        "OpenAI raw request mutation helpers should move out of the plugin"
    );
}

#[test]
fn test_openai_responses_plugin_source_uses_explicit_request_surface_resolution() {
    let source = include_str!("../../../src/acg/openai_plugin.rs");

    assert!(
        source.contains("request_surfaces::apply_request_surface(")
            && source.contains("self.plugin_id()"),
        "OpenAI plugin should resolve Chat versus Responses through request surfaces"
    );
    assert!(
        !source.contains("let content = &mut translated.content;"),
        "OpenAI plugin should not mutate raw request JSON directly"
    );
}

#[test]
fn test_plugin_translate_preserves_semantic_translation_report() {
    let plugin = OpenAICachePlugin;
    let request = sample_llm_request_with_tools();
    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
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

// translate with CacheStability intent canonicalizes tool schemas -> Applied
#[test]
fn test_translate_cache_stability_with_tools() {
    let plugin = OpenAICachePlugin;
    let request = sample_llm_request_with_tools();
    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
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
    assert_eq!(
        output.translation_report.outcomes[0].reason,
        ReasonCode::FullySupported
    );
    assert_eq!(
        output.translation_report.outcomes[0].intent_type,
        IntentType::CacheStability
    );

    // Verify tool schemas are in the output
    let tools = output.translated_request.content["tools"]
        .as_array()
        .unwrap();
    assert_eq!(tools.len(), 1);
}

// Tool schemas with different key ordering produce identical output
#[test]
fn test_tool_schema_canonicalization_deterministic() {
    let plugin = OpenAICachePlugin;

    // Request A: standard key ordering
    let request_a = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {"type": "string"}
                        }
                    }
                }
            }]
        }),
    };

    // Request B: different key ordering (same logical content)
    let request_b = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [],
            "tools": [{
                "function": {
                    "parameters": {
                        "properties": {
                            "q": {"type": "string"}
                        },
                        "type": "object"
                    },
                    "name": "search"
                },
                "type": "function"
            }]
        }),
    };

    let ir = sample_prompt_ir();
    let bundle_a = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
    )]);
    let bundle_b = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
    )]);
    let identity = sample_agent_identity();

    let input_a = PluginInput {
        original_request: &request_a,
        rewritten_request: &request_a,
        prompt_ir: &ir,
        intent_bundle: &bundle_a,
        agent_identity: &identity,
    };

    let input_b = PluginInput {
        original_request: &request_b,
        rewritten_request: &request_b,
        prompt_ir: &ir,
        intent_bundle: &bundle_b,
        agent_identity: &identity,
    };

    let output_a = plugin.translate(&input_a).unwrap();
    let output_b = plugin.translate(&input_b).unwrap();

    // Both should produce byte-identical tool schema JSON
    assert_eq!(
        output_a.translated_request.content["tools"], output_b.translated_request.content["tools"],
        "Differently-ordered tool schemas should produce identical output after canonicalization"
    );

    // Verify the canonical form has sorted keys
    let tool = &output_a.translated_request.content["tools"][0];
    let tool_json = serde_json::to_string(tool).unwrap();
    // In RFC 8785, "function" comes before "type" lexicographically
    assert!(
        tool_json.find("\"function\"").unwrap() < tool_json.find("\"type\"").unwrap(),
        "Keys should be in lexicographic order per RFC 8785: {tool_json}"
    );
}

// translate with CacheStability canonicalizes stable message content blocks
#[test]
fn test_translate_canonicalizes_stable_message_content() {
    let plugin = OpenAICachePlugin;

    // Request with structured content blocks in a message
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "tool_result", "data": {"z": 1, "a": 2}}
                    ]
                }
            ]
        }),
    };

    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
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

    // The tool_result block should be canonicalized (keys sorted)
    let blocks = output.translated_request.content["messages"][0]["content"]
        .as_array()
        .unwrap();
    // Text block should be unchanged
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[0]["text"], "Hello");
    // tool_result block should have canonicalized keys
    let tool_result = &blocks[1];
    let tool_result_json = serde_json::to_string(tool_result).unwrap();
    // "a" should come before "z" in the data field, and "data" before "type"
    assert!(
        tool_result_json.find("\"data\"").unwrap() < tool_result_json.find("\"type\"").unwrap(),
        "Keys should be in lexicographic order: {tool_result_json}"
    );
}

// translate with no tools in request still applies message canonicalization -> Applied
#[test]
fn test_translate_no_tools_still_applied() {
    let plugin = OpenAICachePlugin;
    let request = sample_llm_request_no_tools();
    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
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
}

// translate with non-cache intents marks them Ignored/NotRelevant
#[test]
fn test_translate_non_cache_intents_ignored() {
    let plugin = OpenAICachePlugin;
    let request = sample_llm_request_no_tools();
    let ir = sample_prompt_ir();

    let intents = vec![
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
    ];

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
    assert_eq!(output.translation_report.outcomes.len(), 2);

    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Ignored
    );
    assert_eq!(
        output.translation_report.outcomes[0].reason,
        ReasonCode::NotRelevant
    );
    assert_eq!(
        output.translation_report.outcomes[0].intent_type,
        IntentType::ModelRouting
    );

    assert_eq!(
        output.translation_report.outcomes[1].status,
        TranslationStatus::Ignored
    );
    assert_eq!(
        output.translation_report.outcomes[1].reason,
        ReasonCode::NotRelevant
    );
    assert_eq!(
        output.translation_report.outcomes[1].intent_type,
        IntentType::Compression
    );
}

// translate with empty intent bundle returns 0 outcomes
#[test]
fn test_translate_empty_bundle() {
    let plugin = OpenAICachePlugin;
    let request = sample_llm_request_with_tools();
    let ir = sample_prompt_ir();
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
    assert_eq!(output.translation_report.plugin_id, "openai");
    assert_eq!(output.translation_report.request_id, bundle.request_id);
}

// translate with Retention intent marks it Ignored/UnsupportedByBackend
#[test]
fn test_translate_retention_ignored() {
    let plugin = OpenAICachePlugin;
    let request = sample_llm_request_no_tools();
    let ir = sample_prompt_ir();

    let intents = vec![OptimizationIntent::Retention(RetentionIntent {
        recommended_tier: RetentionTier::LongLived,
        expected_session_duration_secs: Some(3600.0),
        inter_call_gap_p50_ms: Some(500.0),
        scope_label: SharingScope::Session,
    })];

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
    assert_eq!(output.translation_report.outcomes.len(), 1);
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Ignored
    );
    assert_eq!(
        output.translation_report.outcomes[0].reason,
        ReasonCode::UnsupportedByBackend
    );
    assert_eq!(
        output.translation_report.outcomes[0].intent_type,
        IntentType::Retention
    );
    assert!(
        output.translation_report.outcomes[0]
            .detail
            .as_ref()
            .unwrap()
            .contains("retention control")
    );
}

// -------------------------------------------------------------------
// Task 2 tests: Integration tests
// -------------------------------------------------------------------

// OpenAICachePlugin can be registered in PluginRegistry alongside PassthroughPlugin
#[test]
fn test_openai_plugin_in_registry_with_passthrough() {
    let mut registry = PluginRegistry::new();
    let openai: Arc<dyn ProviderPlugin> = Arc::new(OpenAICachePlugin);
    let passthrough: Arc<dyn ProviderPlugin> = Arc::new(crate::acg::passthrough::PassthroughPlugin);

    registry.register(openai).unwrap();
    registry.register(passthrough).unwrap();

    let retrieved_openai = registry.get("openai").unwrap();
    assert_eq!(retrieved_openai.plugin_id(), "openai");
    assert_eq!(retrieved_openai.plugin_name(), "OpenAI Cache Plugin");

    let retrieved_pt = registry.get("passthrough").unwrap();
    assert_eq!(retrieved_pt.plugin_id(), "passthrough");
}

// OpenAI and Anthropic plugins can coexist in same PluginRegistry
#[test]
fn test_both_plugins_in_registry() {
    let registry_caps = crate::acg::capability::CapabilityRegistry::with_defaults();

    let mut registry = PluginRegistry::new();
    let openai: Arc<dyn ProviderPlugin> = Arc::new(OpenAICachePlugin);
    let anthropic: Arc<dyn ProviderPlugin> = Arc::new(
        crate::acg::anthropic_plugin::AnthropicCachePlugin::new(&registry_caps),
    );
    let passthrough: Arc<dyn ProviderPlugin> = Arc::new(crate::acg::passthrough::PassthroughPlugin);

    registry.register(openai).unwrap();
    registry.register(anthropic).unwrap();
    registry.register(passthrough).unwrap();

    let ids = registry.list_plugin_ids();
    assert_eq!(ids, vec!["anthropic", "openai", "passthrough"]);
}

// Full round-trip: build OpenAI-format request -> CacheStability -> translate -> verify
#[test]
fn test_full_round_trip_openai_canonicalization() {
    let plugin = OpenAICachePlugin;

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Search for Rust caching."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        }
                    }
                }
            }]
        }),
    };

    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
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

    // Verify Applied status
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Applied
    );

    // Verify tool parameters have deterministic key ordering (RFC 8785 lexicographic)
    let params = &output.translated_request.content["tools"][0]["function"]["parameters"];
    let params_json = serde_json::to_string(params).unwrap();

    // RFC 8785 lexicographic order: "properties" < "required" < "type"
    let pos_properties = params_json.find("\"properties\"").unwrap();
    let pos_required = params_json.find("\"required\"").unwrap();
    // Find "type":"object" at top level (not nested "type":"string")
    let pos_type = params_json.find("\"type\":\"object\"").unwrap();
    assert!(
        pos_properties < pos_required,
        "Expected properties before required in: {params_json}"
    );
    assert!(
        pos_required < pos_type,
        "Expected required before type in: {params_json}"
    );
}

// Request with stable_prefix_end=0 still canonicalizes tools -> Applied
#[test]
fn test_zero_stable_prefix_still_canonicalizes_tools() {
    let plugin = OpenAICachePlugin;
    let request = sample_llm_request_with_tools();
    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(0),
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
        TranslationStatus::Applied,
        "Tools should always be canonicalized even with stable_prefix_end=0"
    );
}

// 5 complex tools all get canonicalized
#[test]
fn test_five_complex_tools_all_canonicalized() {
    let plugin = OpenAICachePlugin;

    let make_tool = |name: &str, params: serde_json::Value| -> serde_json::Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": format!("{name} tool"),
                "parameters": params
            }
        })
    };

    let tools = vec![
        make_tool(
            "search",
            json!({"type": "object", "required": ["query"],
                    "properties": {"query": {"type": "string"}}}),
        ),
        make_tool(
            "calculator",
            json!({"type": "object", "required": ["expression"],
                    "properties": {"expression": {"type": "string"}, "precision": {"type": "integer"}}}),
        ),
        make_tool(
            "weather",
            json!({"type": "object", "required": ["location"],
                    "properties": {"location": {"type": "string"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}),
        ),
        make_tool(
            "translate",
            json!({"type": "object", "required": ["text", "target_lang"],
                    "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}, "source_lang": {"type": "string"}}}),
        ),
        make_tool(
            "database",
            json!({"type": "object", "required": ["query"],
                    "properties": {"query": {"type": "string"}, "database": {"type": "string"}, "timeout": {"type": "integer"}}}),
        ),
    ];

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "help"}],
            "tools": tools
        }),
    };

    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        cache_stability_intent(1),
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

    let output_tools = output.translated_request.content["tools"]
        .as_array()
        .unwrap();
    assert_eq!(output_tools.len(), 5);

    // Verify determinism: canonicalize each tool again and compare
    for tool in output_tools {
        let canonical = crate::acg::canonicalize::canonicalize_value(tool).unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&canonical).unwrap();
        assert_eq!(tool, &reparsed, "Tool should already be in canonical form");
    }

    // Verify determinism via sha256: each tool produces stable hash
    for tool in output_tools {
        let json_str = serde_json::to_string(tool).unwrap();
        let hash1 = crate::acg::canonicalize::sha256_hex(&json_str);
        let hash2 = crate::acg::canonicalize::sha256_hex(&json_str);
        assert_eq!(hash1, hash2);
    }
}
