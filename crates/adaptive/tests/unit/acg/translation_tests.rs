// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for translation in the NeMo Relay adaptive crate.

use super::{
    AnthropicHintDirective, HintPlan, HintTarget, HintTranslation, HintTranslator,
    OpenAIHintDirective,
};
use crate::acg::plugin::{HintPlanApplier, PluginInput, translate_with_hint_plan};
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
};
use crate::acg::types::{
    AgentIdentity, CacheStabilityIntent, OptimizationIntent, OptimizationIntentBundle,
    SharingScope, TranslationReport,
};
use chrono::Utc;
use nemo_relay::api::llm::LlmRequest;
use serde_json::json;
use uuid::Uuid;

struct MockTranslator;

struct NoopApplier;

#[test]
fn hint_plan_targets_prompt_ir_without_surface_field_names() {
    let mut plan = HintPlan::new("anthropic");
    plan.push(AnthropicHintDirective::CacheBreakpoint {
        target: HintTarget::stable_prefix(2, Some(SpanId("span-1".to_string()))),
        scope: SharingScope::Session,
    });
    plan.push(OpenAIHintDirective::CanonicalizeStablePrefix {
        target: HintTarget::span(SpanId("span-0".to_string())),
    });

    let debug = format!("{plan:?}");
    assert!(!debug.contains("messages"));
    assert!(!debug.contains("input"));
    assert!(!debug.contains("system"));
}

#[test]
fn translators_preserve_translation_report_semantics_when_no_directives_emit() {
    let input = sample_input();
    let translation = MockTranslator.translate(&input).unwrap();

    assert!(translation.hint_plan.directives.is_empty());
    assert_eq!(translation.translation_report.outcomes.len(), 1);
    assert!(matches!(
        translation.translation_report.outcomes[0].reason,
        crate::acg::types::ReasonCode::UnsupportedByBackend
    ));
}

#[test]
fn provider_plugin_facade_can_delegate_through_translator_and_applier() {
    let input = sample_input();
    let output = translate_with_hint_plan(&MockTranslator, &NoopApplier, &input).unwrap();
    assert_eq!(
        output.translated_request.content,
        input.rewritten_request.content
    );
}

fn sample_input() -> PluginInput<'static> {
    let request = Box::leak(Box::new(LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    }));
    let prompt_ir = Box::leak(Box::new(PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![PromptBlock {
            span_id: SpanId("span-0".to_string()),
            sequence_index: 0,
            role: PromptRole::User,
            content: "hello".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::User,
            sensitivity: SensitivityLabel::Public,
            token_metadata: None,
        }],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }));
    let agent_identity = Box::leak(Box::new(AgentIdentity {
        agent_id: "agent".to_string(),
        template_version: "1".to_string(),
        toolset_hash: "hash".to_string(),
        model_family: "claude".to_string(),
        tenant_scope: "tenant".to_string(),
    }));
    let bundle = Box::leak(Box::new(OptimizationIntentBundle {
        request_id: Uuid::new_v4(),
        agent_identity: (*agent_identity).clone(),
        policy_version: "1".to_string(),
        intents: vec![OptimizationIntent::CacheStability(CacheStabilityIntent {
            stability_score: 0.9,
            stable_prefix_end: 1,
            recommended_retention_tier: None,
            scope_label: SharingScope::Session,
            confidence: 0.8,
            evidence_count: 4,
        })],
        created_at: Utc::now(),
    }));

    PluginInput {
        original_request: request,
        rewritten_request: request,
        prompt_ir,
        intent_bundle: bundle,
        agent_identity,
    }
}

impl HintPlanApplier for NoopApplier {
    fn apply_hint_plan(
        &self,
        request: &LlmRequest,
        _prompt_ir: &PromptIR,
        _hint_plan: &HintPlan,
    ) -> crate::acg::Result<LlmRequest> {
        Ok(request.clone())
    }
}

impl HintTranslator for MockTranslator {
    fn provider_id(&self) -> &str {
        "mock"
    }

    fn translate(&self, input: &PluginInput<'_>) -> crate::acg::Result<HintTranslation> {
        Ok(HintTranslation {
            hint_plan: HintPlan {
                provider: self.provider_id().to_string(),
                directives: Vec::new(),
            },
            translation_report: TranslationReport::all_ignored(
                input.intent_bundle,
                self.provider_id(),
                crate::acg::types::ReasonCode::UnsupportedByBackend,
                None,
            ),
        })
    }
}
