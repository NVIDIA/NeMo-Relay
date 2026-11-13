// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic semantic hint translation.

use std::sync::Arc;

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use crate::acg::capability::{CapabilityRegistry, ModelFamilyCapabilities};
use crate::acg::debug as acg_debug;
use crate::acg::plugin::PluginInput;
use crate::acg::prompt_ir::PromptBlock;
use crate::acg::translation::{
    AnthropicCacheTtl, AnthropicHintDirective, HintPlan, HintTranslation, HintTranslator,
    stable_prefix_target,
};
use crate::acg::types::{
    CacheStabilityIntent, IntentOutcome, IntentType, OptimizationIntent, ReasonCode,
    RetentionIntent, RetentionTier, TranslationReport, TranslationStatus,
};

/// Default maximum number of cache breakpoints when the model family is unknown.
const DEFAULT_MAX_CACHE_BREAKPOINTS: u32 = 4;

/// Default minimum cacheable tokens when model is not found in the registry.
const DEFAULT_MIN_CACHEABLE_TOKENS: u32 = 1024;

#[derive(Debug, Clone)]
struct AnthropicTranslationContext {
    model_name: String,
    resolved_model_family: Option<String>,
    max_cache_breakpoints: u32,
    min_cacheable_tokens: u32,
}

#[derive(Debug, Clone, Copy)]
struct PrefixTokenEstimate {
    cumulative_tokens: u32,
    used_fallback: bool,
}

/// Anthropic semantic hint translator.
pub(crate) struct AnthropicHintTranslator {
    registry: Arc<CapabilityRegistry>,
}

impl AnthropicHintTranslator {
    pub fn new(registry: &CapabilityRegistry) -> Self {
        Self {
            registry: Arc::new(registry.clone()),
        }
    }

    fn resolve_model_capabilities(&self, model_name: &str) -> Option<ModelFamilyCapabilities> {
        let backend = self.registry.get_backend("anthropic")?;
        backend.model_families.get(model_name).cloned().or_else(|| {
            backend
                .model_families
                .iter()
                .filter(|(family, _)| model_name.starts_with(family.as_str()))
                .max_by_key(|(family, _)| family.len())
                .map(|(_, capabilities)| capabilities.clone())
        })
    }

    fn build_translation_context(&self, input: &PluginInput<'_>) -> AnthropicTranslationContext {
        let model_name = input
            .rewritten_request
            .content
            .get("model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let model_capabilities = self.resolve_model_capabilities(&model_name);

        AnthropicTranslationContext {
            model_name,
            resolved_model_family: model_capabilities
                .as_ref()
                .map(|capabilities| capabilities.model_family.clone()),
            max_cache_breakpoints: model_capabilities
                .as_ref()
                .and_then(|capabilities| capabilities.max_cache_breakpoints)
                .unwrap_or(DEFAULT_MAX_CACHE_BREAKPOINTS),
            min_cacheable_tokens: model_capabilities
                .as_ref()
                .and_then(|capabilities| capabilities.min_cacheable_tokens)
                .unwrap_or(DEFAULT_MIN_CACHEABLE_TOKENS),
        }
    }
}

impl HintTranslator for AnthropicHintTranslator {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn translate(&self, input: &PluginInput<'_>) -> crate::acg::Result<HintTranslation> {
        let mut outcomes = Vec::new();
        let mut hint_plan = HintPlan::new(self.provider_id());
        hint_plan.push(AnthropicHintDirective::CanonicalizeToolSchemas);
        let context = self.build_translation_context(input);
        acg_debug::emit(
            "anthropic_translation_begin",
            json!({
                "model_name": context.model_name,
                "resolved_model_capabilities": context.resolved_model_family,
                "max_cache_breakpoints": context.max_cache_breakpoints,
                "min_cacheable_tokens": context.min_cacheable_tokens,
                "intent_count": input.intent_bundle.intents.len(),
            }),
        );

        let mut breakpoints_placed: u32 = 0;

        for intent in &input.intent_bundle.intents {
            translate_intent(
                intent,
                input,
                &context,
                &mut hint_plan,
                &mut outcomes,
                &mut breakpoints_placed,
            );
        }

        acg_debug::emit(
            "anthropic_translation_result",
            json!({
                "model_name": context.model_name,
                "directive_count": hint_plan.directives.len(),
                "directives": hint_plan
                    .directives
                    .iter()
                    .map(|directive| format!("{directive:?}"))
                    .collect::<Vec<_>>(),
                "outcome_count": outcomes.len(),
            }),
        );

        Ok(HintTranslation {
            hint_plan,
            translation_report: TranslationReport {
                request_id: input.intent_bundle.request_id,
                plugin_id: self.provider_id().to_string(),
                outcomes,
                created_at: Utc::now(),
            },
        })
    }
}

fn translate_intent(
    intent: &OptimizationIntent,
    input: &PluginInput<'_>,
    context: &AnthropicTranslationContext,
    hint_plan: &mut HintPlan,
    outcomes: &mut Vec<IntentOutcome>,
    breakpoints_placed: &mut u32,
) {
    match intent {
        OptimizationIntent::CacheStability(cache_stability) => handle_cache_stability_intent(
            cache_stability,
            input,
            context,
            hint_plan,
            outcomes,
            breakpoints_placed,
        ),
        OptimizationIntent::Retention(retention) => {
            handle_retention_intent(retention, hint_plan, outcomes)
        }
        other => outcomes.push(ignored_intent_outcome(other.discriminant())),
    }
}

fn handle_cache_stability_intent(
    cache_stability: &CacheStabilityIntent,
    input: &PluginInput<'_>,
    context: &AnthropicTranslationContext,
    hint_plan: &mut HintPlan,
    outcomes: &mut Vec<IntentOutcome>,
    breakpoints_placed: &mut u32,
) {
    if *breakpoints_placed >= context.max_cache_breakpoints {
        acg_debug::emit(
            "anthropic_translation_degraded",
            json!({
                "reason": "backend_limit_reached",
                "model_name": context.model_name,
                "max_cache_breakpoints": context.max_cache_breakpoints,
                "breakpoints_placed": breakpoints_placed,
                "stable_prefix_end": cache_stability.stable_prefix_end,
            }),
        );
        outcomes.push(new_intent_outcome(
            IntentType::CacheStability,
            TranslationStatus::Degraded,
            ReasonCode::BackendLimitReached,
            Some(format!(
                "{} breakpoints already placed, max reached",
                context.max_cache_breakpoints
            )),
        ));
        return;
    }

    let end_index = cache_stability
        .stable_prefix_end
        .min(input.prompt_ir.blocks.len());
    let Some(token_estimate) =
        estimate_prefix_tokens(input.prompt_ir.blocks.iter().take(end_index))
    else {
        outcomes.push(new_intent_outcome(
            IntentType::CacheStability,
            TranslationStatus::Degraded,
            ReasonCode::InsufficientEvidence,
            Some("token count unavailable for media or opaque prefix content".into()),
        ));
        return;
    };

    if token_estimate.cumulative_tokens < context.min_cacheable_tokens {
        acg_debug::emit(
            "anthropic_translation_degraded",
            json!({
                "reason": "below_min_cacheable_tokens",
                "model_name": context.model_name,
                "stable_prefix_end": end_index,
                "cumulative_tokens": token_estimate.cumulative_tokens,
                "min_cacheable_tokens": context.min_cacheable_tokens,
            }),
        );
        outcomes.push(new_intent_outcome(
            IntentType::CacheStability,
            TranslationStatus::Degraded,
            ReasonCode::BackendLimitReached,
            Some(format!(
                "cumulative tokens ({}) below model minimum ({})",
                token_estimate.cumulative_tokens, context.min_cacheable_tokens
            )),
        ));
        return;
    }

    let target = stable_prefix_target(input.prompt_ir, end_index);
    acg_debug::emit(
        "anthropic_translation_applied",
        json!({
            "model_name": context.model_name,
            "stable_prefix_end": end_index,
            "cumulative_tokens": token_estimate.cumulative_tokens,
            "target": format!("{target:?}"),
            "used_fallback_token_estimate": token_estimate.used_fallback,
        }),
    );
    hint_plan.push(AnthropicHintDirective::CacheBreakpoint {
        target,
        scope: cache_stability.scope_label,
    });
    *breakpoints_placed += 1;
    outcomes.push(cache_stability_outcome(
        cache_stability,
        end_index,
        token_estimate.used_fallback,
    ));
}

fn handle_retention_intent(
    retention: &RetentionIntent,
    hint_plan: &mut HintPlan,
    outcomes: &mut Vec<IntentOutcome>,
) {
    match retention.recommended_tier {
        RetentionTier::Ephemeral | RetentionTier::ShortLived => outcomes.push(new_intent_outcome(
            IntentType::Retention,
            TranslationStatus::Applied,
            ReasonCode::FullySupported,
            Some("using Anthropic default 5m TTL".to_string()),
        )),
        RetentionTier::SessionDuration | RetentionTier::LongLived | RetentionTier::Permanent => {
            if hint_plan.has_anthropic_breakpoint() {
                hint_plan.push(AnthropicHintDirective::ApplyTtl {
                    ttl: AnthropicCacheTtl::OneHour,
                });
                outcomes.push(new_intent_outcome(
                    IntentType::Retention,
                    TranslationStatus::Applied,
                    ReasonCode::FullySupported,
                    Some("applied 1h extended TTL to all cache_control annotations".to_string()),
                ));
            } else {
                outcomes.push(new_intent_outcome(
                    IntentType::Retention,
                    TranslationStatus::Degraded,
                    ReasonCode::BackendLimitReached,
                    Some("no breakpoints to apply TTL to".to_string()),
                ));
            }
        }
    }
}

fn cache_stability_outcome(
    cache_stability: &CacheStabilityIntent,
    end_index: usize,
    used_fallback: bool,
) -> IntentOutcome {
    new_intent_outcome(
        IntentType::CacheStability,
        if used_fallback {
            TranslationStatus::Degraded
        } else {
            TranslationStatus::Applied
        },
        if used_fallback {
            ReasonCode::InsufficientEvidence
        } else {
            ReasonCode::FullySupported
        },
        Some(if used_fallback {
            "token count estimated via chars/4 heuristic; actual tokenization unavailable"
                .to_string()
        } else {
            format!(
                "breakpoint planned at stable prefix {}, scope={}",
                end_index,
                format_scope(&cache_stability.scope_label)
            )
        }),
    )
}

fn ignored_intent_outcome(intent_type: IntentType) -> IntentOutcome {
    new_intent_outcome(
        intent_type,
        TranslationStatus::Ignored,
        ReasonCode::NotRelevant,
        Some(format!(
            "Anthropic plugin does not handle {intent_type:?} intents"
        )),
    )
}

fn new_intent_outcome(
    intent_type: IntentType,
    status: TranslationStatus,
    reason: ReasonCode,
    detail: Option<String>,
) -> IntentOutcome {
    IntentOutcome {
        intent_id: Uuid::new_v4(),
        intent_type,
        status,
        reason,
        detail,
    }
}

fn estimate_prefix_tokens<'a>(
    blocks: impl Iterator<Item = &'a PromptBlock>,
) -> Option<PrefixTokenEstimate> {
    let mut cumulative_tokens = 0_u32;
    let mut used_fallback = false;

    for block in blocks {
        let (tokens, fallback) = block.estimated_tokens()?;
        cumulative_tokens = cumulative_tokens.saturating_add(tokens);
        used_fallback |= fallback;
    }

    Some(PrefixTokenEstimate {
        cumulative_tokens,
        used_fallback,
    })
}

fn format_scope(scope: &crate::acg::types::SharingScope) -> &'static str {
    match scope {
        crate::acg::types::SharingScope::Request => "request",
        crate::acg::types::SharingScope::Session => "session",
        crate::acg::types::SharingScope::Tenant => "tenant",
        crate::acg::types::SharingScope::Global => "global",
    }
}
