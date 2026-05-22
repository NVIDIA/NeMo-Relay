// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for passthrough in the NeMo Relay adaptive crate.

use std::sync::Arc;

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use crate::acg::plugin::{PluginInput, ProviderPlugin};
use crate::acg::plugin_registry::PluginRegistry;
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
};
use crate::acg::types::{
    AgentIdentity, CacheStabilityIntent, CompressionIntent, ContentExtractionIntent, IntentType,
    ModelClass, ModelRoutingIntent, OptimizationIntent, OptimizationIntentBundle, ReasonCode,
    SharingScope, TranslationStatus,
};
use nemo_relay::api::llm::LlmRequest;

use super::PassthroughPlugin;

fn assert_send_sync<T: Send + Sync>() {}

fn sample_llm_request() -> LlmRequest {
    LlmRequest {
        headers: {
            let mut m = serde_json::Map::new();
            m.insert(
                "x-api-key".to_string(),
                serde_json::Value::String("sk-test".to_string()),
            );
            m
        },
        content: json!({
            "model": "claude-3.5-sonnet",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello."}
            ]
        }),
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

// -------------------------------------------------------------------
// plugin_id() returns "passthrough"
// -------------------------------------------------------------------

#[test]
fn test_passthrough_plugin_id() {
    let plugin = PassthroughPlugin;
    assert_eq!(plugin.plugin_id(), "passthrough");
}

// -------------------------------------------------------------------
// plugin_name() returns "Passthrough (No-Op)"
// -------------------------------------------------------------------

#[test]
fn test_passthrough_plugin_name() {
    let plugin = PassthroughPlugin;
    assert_eq!(plugin.plugin_name(), "Passthrough (No-Op)");
}

#[test]
fn test_passthrough_capabilities_are_empty() {
    let plugin = PassthroughPlugin;
    let capabilities = plugin.capabilities();

    assert_eq!(capabilities.backend_id, "passthrough");
    assert!(capabilities.supported_features.is_empty());
    assert!(capabilities.model_families.is_empty());
}

// -------------------------------------------------------------------
// translate() with empty intent bundle returns 0 outcomes and cloned request
// -------------------------------------------------------------------

#[test]
fn test_translate_empty_bundle() {
    let plugin = PassthroughPlugin;
    let request = sample_llm_request();
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
    assert_eq!(output.translated_request.content, request.content);
}

// -------------------------------------------------------------------
// translate() with 3 mixed intents returns 3 outcomes, all Ignored/NotRelevant
// -------------------------------------------------------------------

#[test]
fn test_translate_three_mixed_intents() {
    let plugin = PassthroughPlugin;
    let request = sample_llm_request();
    let ir = sample_prompt_ir();

    let intents = vec![
        OptimizationIntent::CacheStability(CacheStabilityIntent {
            stability_score: 0.9,
            stable_prefix_end: 100,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.8,
            evidence_count: 10,
        }),
        OptimizationIntent::ContentExtraction(ContentExtractionIntent {
            block_id: "block-0".to_string(),
            variable_pattern: ".*".to_string(),
            extraction_strategy: "regex".to_string(),
            scope_label: SharingScope::Session,
        }),
        OptimizationIntent::ModelRouting(ModelRoutingIntent {
            model_class: ModelClass::Premium,
            complexity_score: 0.7,
            criticality: 0.9,
            fallback_allowed: true,
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
    assert_eq!(output.translation_report.outcomes.len(), 3);
    for outcome in &output.translation_report.outcomes {
        assert_eq!(outcome.status, TranslationStatus::Ignored);
        assert_eq!(outcome.reason, ReasonCode::NotRelevant);
    }
}

// -------------------------------------------------------------------
// translate() preserves exact request content (deep equality)
// -------------------------------------------------------------------

#[test]
fn test_translate_preserves_exact_request() {
    let plugin = PassthroughPlugin;
    let request = sample_llm_request();
    let ir = sample_prompt_ir();
    let bundle = sample_intent_bundle(vec![OptimizationIntent::CacheStability(
        CacheStabilityIntent {
            stability_score: 0.5,
            stable_prefix_end: 50,
            recommended_retention_tier: None,
            scope_label: SharingScope::Request,
            confidence: 0.6,
            evidence_count: 3,
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
    // Deep equality: headers and content must match exactly
    assert_eq!(output.translated_request.headers, request.headers);
    assert_eq!(output.translated_request.content, request.content);
}

// -------------------------------------------------------------------
// translate() outcomes have correct IntentType discriminants
// -------------------------------------------------------------------

#[test]
fn test_translate_outcomes_have_correct_discriminants() {
    let plugin = PassthroughPlugin;
    let request = sample_llm_request();
    let ir = sample_prompt_ir();

    let intents = vec![
        OptimizationIntent::CacheStability(CacheStabilityIntent {
            stability_score: 0.9,
            stable_prefix_end: 100,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.8,
            evidence_count: 10,
        }),
        OptimizationIntent::ContentExtraction(ContentExtractionIntent {
            block_id: "b".to_string(),
            variable_pattern: "p".to_string(),
            extraction_strategy: "s".to_string(),
            scope_label: SharingScope::Tenant,
        }),
        OptimizationIntent::Compression(CompressionIntent {
            block_id: "c".to_string(),
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
    assert_eq!(
        output.translation_report.outcomes[0].intent_type,
        IntentType::CacheStability
    );
    assert_eq!(
        output.translation_report.outcomes[1].intent_type,
        IntentType::ContentExtraction
    );
    assert_eq!(
        output.translation_report.outcomes[2].intent_type,
        IntentType::Compression
    );
}

// -------------------------------------------------------------------
// translate() report has correct plugin_id and request_id
// -------------------------------------------------------------------

#[test]
fn test_translate_report_ids() {
    let plugin = PassthroughPlugin;
    let request = sample_llm_request();
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
    assert_eq!(output.translation_report.plugin_id, "passthrough");
    assert_eq!(output.translation_report.request_id, bundle.request_id);
}

// -------------------------------------------------------------------
// PassthroughPlugin is Send + Sync
// -------------------------------------------------------------------

#[test]
fn test_passthrough_is_send_sync() {
    assert_send_sync::<PassthroughPlugin>();
    // Also verify it works as Arc<dyn ProviderPlugin>
    let _: Arc<dyn ProviderPlugin> = Arc::new(PassthroughPlugin);
}

// -------------------------------------------------------------------
// PassthroughPlugin can be registered in PluginRegistry
// -------------------------------------------------------------------

#[test]
fn test_passthrough_registerable_in_plugin_registry() {
    let mut registry = PluginRegistry::new();
    let plugin: Arc<dyn ProviderPlugin> = Arc::new(PassthroughPlugin);
    registry.register(plugin).unwrap();

    let retrieved = registry.get("passthrough").unwrap();
    assert_eq!(retrieved.plugin_id(), "passthrough");
    assert_eq!(retrieved.plugin_name(), "Passthrough (No-Op)");
}
