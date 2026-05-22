// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for plugin in the NeMo Relay adaptive crate.

use super::*;
use std::sync::Arc;

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use crate::acg::prompt_ir::{BlockContentType, PromptBlock, PromptRole, ProvenanceLabel, SpanId};
use crate::acg::types::{
    CacheStabilityIntent, OptimizationIntent, ReasonCode, SharingScope, TranslationReport,
    TranslationStatus,
};

/// A minimal mock plugin for testing.
struct MockPlugin;

impl ProviderPlugin for MockPlugin {
    fn plugin_id(&self) -> &str {
        "mock"
    }

    fn plugin_name(&self) -> &str {
        "Mock Plugin"
    }

    fn translate(&self, input: &PluginInput<'_>) -> crate::acg::error::Result<PluginOutput> {
        Ok(PluginOutput {
            translated_request: input.rewritten_request.clone(),
            translation_report: TranslationReport::all_ignored(
                input.intent_bundle,
                self.plugin_id(),
                ReasonCode::NotRelevant,
                None,
            ),
        })
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::none("mock")
    }
}

fn assert_send_sync<T: Send + Sync>() {}

fn sample_llm_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "test", "messages": []}),
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
            sensitivity: crate::acg::prompt_ir::SensitivityLabel::Public,
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

fn sample_bundle() -> OptimizationIntentBundle {
    OptimizationIntentBundle {
        request_id: Uuid::new_v4(),
        agent_identity: sample_agent_identity(),
        policy_version: "1.0.0".to_string(),
        intents: vec![OptimizationIntent::CacheStability(CacheStabilityIntent {
            stability_score: 0.9,
            stable_prefix_end: 100,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.8,
            evidence_count: 10,
        })],
        created_at: Utc::now(),
    }
}

// -------------------------------------------------------------------
// Object safety: Arc<dyn ProviderPlugin> compiles
// -------------------------------------------------------------------

#[test]
fn test_provider_plugin_is_object_safe() {
    let _: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin);
}

// -------------------------------------------------------------------
// Send + Sync compile-time assertions
// -------------------------------------------------------------------

#[test]
fn test_provider_plugin_is_send_sync() {
    assert_send_sync::<Arc<dyn ProviderPlugin>>();
}

// -------------------------------------------------------------------
// PluginInput can be constructed with borrowed references
// -------------------------------------------------------------------

#[test]
fn test_plugin_input_borrows_all_fields() {
    let request = sample_llm_request();
    let ir = sample_prompt_ir();
    let bundle = sample_bundle();
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    // Verify the references are correct
    assert_eq!(input.original_request.content, request.content);
    assert_eq!(input.prompt_ir.blocks.len(), 1);
    assert_eq!(input.intent_bundle.intents.len(), 1);
    assert_eq!(input.agent_identity.agent_id, "test-agent");
}

// -------------------------------------------------------------------
// PluginOutput contains owned data
// -------------------------------------------------------------------

#[test]
fn test_plugin_output_owns_data() {
    let request = sample_llm_request();
    let bundle = sample_bundle();

    let output = PluginOutput {
        translated_request: request.clone(),
        translation_report: TranslationReport::all_ignored(
            &bundle,
            "test",
            ReasonCode::NotRelevant,
            None,
        ),
    };

    assert_eq!(output.translated_request.content, request.content);
    assert_eq!(output.translation_report.plugin_id, "test");
    assert_eq!(output.translation_report.outcomes.len(), 1);
    assert_eq!(
        output.translation_report.outcomes[0].status,
        TranslationStatus::Ignored
    );
}

// -------------------------------------------------------------------
// MockPlugin translate works end-to-end
// -------------------------------------------------------------------

#[test]
fn test_mock_plugin_translate() {
    let plugin: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin);
    let request = sample_llm_request();
    let ir = sample_prompt_ir();
    let bundle = sample_bundle();
    let identity = sample_agent_identity();

    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &ir,
        intent_bundle: &bundle,
        agent_identity: &identity,
    };

    let output = plugin.translate(&input).unwrap();
    assert_eq!(output.translated_request.content, request.content);
    assert_eq!(output.translation_report.plugin_id, "mock");
    assert_eq!(output.translation_report.outcomes.len(), 1);
}
