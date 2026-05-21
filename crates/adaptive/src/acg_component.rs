// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive Cache Governor (ACG) request and execution intercept helpers for
//! the adaptive runtime.

use std::fmt::Display;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use serde_json::json;

use crate::acg::economics;
use crate::acg::plugin::{PluginInput, ProviderPlugin};
use crate::acg::request_surfaces::{RequestSurface, resolve_request_surface_from_request};
use crate::acg::translation::anthropic::AnthropicHintTranslator;
use crate::acg::translation::openai::OpenAIHintTranslator;
use crate::acg::translation::{HintPlan, HintTranslation, HintTranslator};
use crate::acg::{
    AgentIdentity, AnthropicCachePlugin, CacheStabilityIntent, CapabilityRegistry,
    MIN_ACG_OBSERVATIONS, OpenAICachePlugin, OptimizationIntent, OptimizationIntentBundle,
    PassthroughPlugin, SharingScope, StabilityAnalysisResult, debug as acg_debug,
};
use chrono::Utc;
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    LlmExecutionFn, LlmExecutionNextFn, LlmRequestInterceptFn, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn,
};
use nemo_relay::codec::anthropic::AnthropicMessagesCodec;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::openai_responses::OpenAIResponsesCodec;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::traits::LlmCodec;
use nemo_relay::json::Json;
use uuid::Uuid;

use crate::acg_profile::{derive_acg_learning_key, derive_acg_profile_key};
use crate::error::{AdaptiveError, Result};
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;

struct SemanticRequestView {
    request_surface: RequestSurface,
    annotated_request: AnnotatedLlmRequest,
}

pub(crate) async fn load_persisted_acg_state(
    agent_id: &str,
    backend: &dyn StorageBackendDyn,
    hot_cache: &Arc<RwLock<HotCache>>,
) -> Result<()> {
    let stability = backend.load_stability(agent_id).await?;
    let observation_count = match stability.as_ref() {
        Some(result) => result.total_observations,
        None => backend
            .load_observations(agent_id)
            .await?
            .map(|observations| observations.len() as u32)
            .unwrap_or(0),
    };

    if stability.is_none() && observation_count == 0 {
        return Ok(());
    }

    let mut guard = hot_cache
        .write()
        .map_err(|error| AdaptiveError::Internal(format!("hot cache lock poisoned: {error}")))?;
    guard.acg_stability = stability;
    guard.acg_observation_count = observation_count;
    Ok(())
}

pub(crate) fn build_provider_plugin(provider: &str) -> Result<Arc<dyn ProviderPlugin>> {
    match provider {
        "anthropic" => {
            let registry = CapabilityRegistry::with_defaults();
            Ok(Arc::new(AnthropicCachePlugin::new(&registry)))
        }
        "openai" => Ok(Arc::new(OpenAICachePlugin)),
        "passthrough" => Ok(Arc::new(PassthroughPlugin)),
        other => Err(AdaptiveError::InvalidConfig(format!(
            "unsupported acg provider '{other}'"
        ))),
    }
}

fn decode_request_for_surface(
    request_surface: RequestSurface,
    request: &LlmRequest,
) -> Result<AnnotatedLlmRequest> {
    match request_surface {
        RequestSurface::AnthropicMessages => {
            AnthropicMessagesCodec.decode(request).map_err(|error| {
                AdaptiveError::Internal(format!("failed to decode anthropic request: {error}"))
            })
        }
        RequestSurface::OpenAIChat => OpenAIChatCodec.decode(request).map_err(|error| {
            AdaptiveError::Internal(format!("failed to decode openai chat request: {error}"))
        }),
        RequestSurface::OpenAIResponses => OpenAIResponsesCodec.decode(request).map_err(|error| {
            AdaptiveError::Internal(format!(
                "failed to decode openai responses request: {error}"
            ))
        }),
    }
}

fn build_semantic_request_view(request: &LlmRequest) -> Result<SemanticRequestView> {
    let request_surface = resolve_request_surface_from_request(request)
        .map_err(|error| AdaptiveError::Internal(error.to_string()))?;
    let annotated_request = decode_request_for_surface(request_surface, request)?;

    Ok(SemanticRequestView {
        request_surface,
        annotated_request,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_intent_bundle(
    agent_id: &str,
    provider: &str,
    plugin: &dyn ProviderPlugin,
    request_surface: RequestSurface,
    annotated_request: &AnnotatedLlmRequest,
    prompt_ir: &crate::acg::PromptIR,
    stability: &StabilityAnalysisResult,
    observation_count: u32,
) -> Option<OptimizationIntentBundle> {
    if observation_count < MIN_ACG_OBSERVATIONS {
        acg_debug::emit(
            "build_intent_bundle_skipped",
            json!({
                "reason": "insufficient_observations",
                "agent_id": agent_id,
                "provider": provider,
                "observation_count": observation_count,
                "minimum_observations": MIN_ACG_OBSERVATIONS,
                "stable_prefix_length": stability.stable_prefix_length,
            }),
        );
        return None;
    }

    let toolset_hash = annotated_request
        .tools
        .as_ref()
        .map(|tools| format!("tool-count-{}", tools.len()))
        .unwrap_or_else(|| "tool-count-0".to_string());

    let agent_identity = AgentIdentity {
        agent_id: agent_id.to_string(),
        template_version: "unknown".to_string(),
        toolset_hash,
        model_family: annotated_request
            .model
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        tenant_scope: "default".to_string(),
    };

    let intents = match provider {
        "anthropic" => build_anthropic_cache_intents(
            plugin,
            request_surface,
            annotated_request,
            prompt_ir,
            stability,
            observation_count,
        )?,
        "openai" => vec![build_cache_stability_intent(
            stability,
            stability.stable_prefix_length,
            SharingScope::Session,
        )?],
        _ => {
            acg_debug::emit(
                "build_intent_bundle_skipped",
                json!({
                    "reason": "unsupported_provider",
                    "agent_id": agent_id,
                    "provider": provider,
                    "observation_count": observation_count,
                }),
            );
            return None;
        }
    };

    acg_debug::emit(
        "build_intent_bundle_ready",
        json!({
            "agent_id": agent_id,
            "provider": provider,
            "observation_count": observation_count,
            "intent_count": intents.len(),
            "stable_prefix_length": stability.stable_prefix_length,
        }),
    );

    Some(OptimizationIntentBundle {
        request_id: Uuid::new_v4(),
        agent_identity: agent_identity.clone(),
        policy_version: "phase-1004-economics-acg".to_string(),
        intents,
        created_at: Utc::now(),
    })
}

fn build_cache_stability_intent(
    stability: &StabilityAnalysisResult,
    stable_prefix_end: usize,
    scope_label: SharingScope,
) -> Option<OptimizationIntent> {
    let prefix_scores = stability
        .scores
        .iter()
        .take(stable_prefix_end)
        .collect::<Vec<_>>();
    if prefix_scores.is_empty() {
        return None;
    }

    let stability_score = prefix_scores
        .iter()
        .map(|score| score.score)
        .fold(1.0_f64, f64::min);
    let confidence = prefix_scores
        .iter()
        .map(|score| score.confidence)
        .fold(1.0_f64, f64::min);

    Some(OptimizationIntent::CacheStability(CacheStabilityIntent {
        stability_score,
        stable_prefix_end,
        recommended_retention_tier: None,
        scope_label,
        confidence,
        evidence_count: stability.total_observations,
    }))
}

fn build_anthropic_cache_intents(
    plugin: &dyn ProviderPlugin,
    request_surface: RequestSurface,
    annotated_request: &AnnotatedLlmRequest,
    prompt_ir: &crate::acg::PromptIR,
    stability: &StabilityAnalysisResult,
    observation_count: u32,
) -> Option<Vec<OptimizationIntent>> {
    if request_surface != RequestSurface::AnthropicMessages {
        acg_debug::emit(
            "anthropic_cache_intents_skipped",
            json!({
                "reason": "request_surface_not_anthropic_messages",
                "request_surface": format!("{request_surface:?}"),
            }),
        );
        return None;
    }

    let model_name = annotated_request.model.as_deref()?;
    let backend_capabilities = plugin.capabilities();
    let capabilities = match resolve_model_family_capabilities(&backend_capabilities, model_name) {
        Some(capabilities) => capabilities,
        None => {
            acg_debug::emit(
                "anthropic_cache_intents_skipped",
                json!({
                    "reason": "model_capabilities_not_found",
                    "model_name": model_name,
                }),
            );
            return None;
        }
    };
    let plan = economics::plan_breakpoints(prompt_ir, stability, observation_count, &capabilities);

    if plan.planned_breakpoints.is_empty() {
        acg_debug::emit(
            "anthropic_cache_intents_skipped",
            json!({
                "reason": "economics_plan_empty",
                "model_name": model_name,
                "observation_count": observation_count,
                "stable_prefix_length": stability.stable_prefix_length,
                "minimum_cacheable_tokens": plan.minimum_cacheable_tokens,
                "observed_reuse_horizon": plan.observed_reuse_horizon,
            }),
        );
        return None;
    }

    plan.planned_breakpoints
        .iter()
        .map(|breakpoint| {
            build_cache_stability_intent(stability, breakpoint.stable_prefix_end, breakpoint.scope)
        })
        .collect()
}

fn resolve_model_family_capabilities(
    backend: &crate::acg::BackendCapabilities,
    model_name: &str,
) -> Option<crate::acg::ModelFamilyCapabilities> {
    backend.model_families.get(model_name).cloned().or_else(|| {
        backend
            .model_families
            .iter()
            .filter(|(family, _)| model_name.starts_with(family.as_str()))
            .max_by_key(|(family, _)| family.len())
            .map(|(_, caps)| caps.clone())
    })
}

fn build_hint_translation(
    provider: &str,
    input: &PluginInput<'_>,
) -> crate::acg::Result<HintTranslation> {
    match provider {
        "anthropic" => {
            let registry = CapabilityRegistry::with_defaults();
            AnthropicHintTranslator::new(&registry).translate(input)
        }
        "openai" => OpenAIHintTranslator.translate(input),
        "passthrough" => Ok(HintTranslation {
            hint_plan: HintPlan::new("passthrough"),
            translation_report: crate::acg::TranslationReport::all_ignored(
                input.intent_bundle,
                "passthrough",
                crate::acg::ReasonCode::NotRelevant,
                Some("passthrough provider applies no semantic hint translation".to_string()),
            ),
        }),
        other => Err(crate::acg::AcgError::Internal(format!(
            "unsupported semantic provider '{other}'"
        ))),
    }
}

fn apply_hint_translation(
    request: &LlmRequest,
    provider: &str,
    request_surface: RequestSurface,
    prompt_ir: &crate::acg::PromptIR,
    hint_plan: &HintPlan,
) -> crate::acg::Result<LlmRequest> {
    if !request_surface.supports_provider(provider) {
        return Err(crate::acg::AcgError::Internal(format!(
            "provider '{provider}' is incompatible with request surface {request_surface:?}"
        )));
    }

    request_surface.apply(request, prompt_ir, hint_plan)
}

fn translate_request_error(
    agent_id: &str,
    provider: &str,
    learning_key: &str,
    profile_key: &str,
    reason: &str,
    error: &dyn Display,
) -> Option<LlmRequest> {
    acg_debug::emit(
        "translate_request_skipped",
        json!({
            "reason": reason,
            "agent_id": agent_id,
            "provider": provider,
            "learning_key": learning_key,
            "profile_key": profile_key,
            "error": error.to_string(),
        }),
    );
    None
}

fn translate_request(
    request: &LlmRequest,
    agent_id: &str,
    provider: &str,
    plugin: &dyn ProviderPlugin,
    hot_cache: &Arc<RwLock<HotCache>>,
) -> Option<LlmRequest> {
    // Response codecs stay on the observability path after provider execution.
    let semantic_request_view = match build_semantic_request_view(request) {
        Ok(view) => view,
        Err(error) => {
            acg_debug::emit(
                "translate_request_skipped",
                json!({
                    "reason": "semantic_request_view_failed",
                    "provider": provider,
                    "error": error.to_string(),
                }),
            );
            return None;
        }
    };
    let learning_key = derive_acg_learning_key(agent_id, &semantic_request_view.annotated_request);
    let profile_key = derive_acg_profile_key(agent_id, &semantic_request_view.annotated_request);
    let Some((stability, observation_count)) = hot_cache.read().ok().and_then(|guard| {
        let profile_stability = guard.acg_profiles.get(&learning_key).cloned();
        let profile_observation_count = guard
            .acg_profile_observation_counts
            .get(&learning_key)
            .copied();

        profile_stability
            .map(|stability| {
                let observation_count =
                    profile_observation_count.unwrap_or(stability.total_observations);
                (stability, observation_count)
            })
            .or_else(|| {
                guard
                    .acg_stability
                    .clone()
                    .map(|stability| (stability, guard.acg_observation_count))
            })
    }) else {
        acg_debug::emit(
            "translate_request_skipped",
            json!({
                "reason": "no_stability_in_hot_cache",
                "agent_id": agent_id,
                "provider": provider,
                "learning_key": learning_key,
                "profile_key": profile_key,
                "model": semantic_request_view.annotated_request.model,
            }),
        );
        return None;
    };
    acg_debug::emit(
        "translate_request_context",
        json!({
            "agent_id": agent_id,
            "provider": provider,
            "learning_key": learning_key,
            "profile_key": profile_key,
            "request_surface": format!("{:?}", semantic_request_view.request_surface),
            "model": semantic_request_view.annotated_request.model,
            "observation_count": observation_count,
            "stable_prefix_length": stability.stable_prefix_length,
            "stability_total_observations": stability.total_observations,
        }),
    );
    let prompt_ir =
        match crate::acg::ir_builder::build_prompt_ir(&semantic_request_view.annotated_request) {
            Ok(prompt_ir) => prompt_ir,
            Err(error) => {
                return translate_request_error(
                    agent_id,
                    provider,
                    &learning_key,
                    &profile_key,
                    "prompt_ir_build_failed",
                    &error,
                );
            }
        };
    let Some(intent_bundle) = build_intent_bundle(
        agent_id,
        provider,
        plugin,
        semantic_request_view.request_surface,
        &semantic_request_view.annotated_request,
        &prompt_ir,
        &stability,
        observation_count,
    ) else {
        acg_debug::emit(
            "translate_request_skipped",
            json!({
                "reason": "intent_bundle_empty",
                "agent_id": agent_id,
                "provider": provider,
                "learning_key": learning_key,
                "profile_key": profile_key,
                "observation_count": observation_count,
                "stable_prefix_length": stability.stable_prefix_length,
                "prompt_block_count": prompt_ir.blocks.len(),
            }),
        );
        return None;
    };

    let input = PluginInput {
        original_request: request,
        rewritten_request: request,
        prompt_ir: &prompt_ir,
        intent_bundle: &intent_bundle,
        agent_identity: &intent_bundle.agent_identity,
    };

    let HintTranslation {
        hint_plan,
        translation_report,
    } = match build_hint_translation(provider, &input) {
        Ok(translation) => translation,
        Err(error) => {
            return translate_request_error(
                agent_id,
                provider,
                &learning_key,
                &profile_key,
                "hint_translation_failed",
                &error,
            );
        }
    };
    acg_debug::emit(
        "translate_request_hint_plan",
        json!({
            "agent_id": agent_id,
            "provider": provider,
            "learning_key": learning_key,
            "profile_key": profile_key,
            "directive_count": hint_plan.directives.len(),
            "directives": hint_plan
                .directives
                .iter()
                .map(|directive| format!("{directive:?}"))
                .collect::<Vec<_>>(),
            "translation_outcomes": translation_report
                .outcomes
                .iter()
                .map(|outcome| json!({
                    "intent_type": format!("{:?}", outcome.intent_type),
                    "status": format!("{:?}", outcome.status),
                    "reason": format!("{:?}", outcome.reason),
                    "detail": outcome.detail,
                }))
                .collect::<Vec<_>>(),
        }),
    );
    let translated = match apply_hint_translation(
        request,
        provider,
        semantic_request_view.request_surface,
        &prompt_ir,
        &hint_plan,
    ) {
        Ok(translated) => translated,
        Err(error) => {
            return translate_request_error(
                agent_id,
                provider,
                &learning_key,
                &profile_key,
                "apply_hint_translation_failed",
                &error,
            );
        }
    };
    acg_debug::emit(
        "translate_request_applied",
        json!({
            "agent_id": agent_id,
            "provider": provider,
            "learning_key": learning_key,
            "profile_key": profile_key,
            "directive_count": hint_plan.directives.len(),
        }),
    );
    Some(translated)
}

/// Rewrite a provider-native request using a seeded `HotCache` and live agent
/// identity.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn rewrite_request_with_hot_cache(
    request: &LlmRequest,
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: &str,
    provider: &str,
) -> Result<LlmRequest> {
    let plugin = build_provider_plugin(provider)?;

    Ok(
        translate_request(request, agent_id, provider, plugin.as_ref(), &hot_cache)
            .unwrap_or_else(|| request.clone()),
    )
}

pub(crate) fn create_acg_llm_request_intercept(
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    provider: String,
    plugin: Arc<dyn ProviderPlugin>,
) -> LlmRequestInterceptFn {
    Arc::new(move |_name: &str, request: LlmRequest, annotated| {
        let translated =
            translate_request(&request, &agent_id, &provider, plugin.as_ref(), &hot_cache)
                .unwrap_or(request);
        Ok((translated, annotated))
    })
}

pub(crate) fn create_acg_llm_execution_intercept(
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    provider: String,
    plugin: Arc<dyn ProviderPlugin>,
) -> LlmExecutionFn {
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let cache = hot_cache.clone();
            let agent_id = agent_id.clone();
            let provider = provider.clone();
            let plugin = plugin.clone();
            Box::pin(async move {
                // Planner/runtime request surface mismatches pass through unchanged.
                let translated =
                    translate_request(&request, &agent_id, &provider, plugin.as_ref(), &cache)
                        .unwrap_or(request);
                next(translated).await
            }) as Pin<Box<dyn Future<Output = nemo_relay::error::Result<Json>> + Send>>
        },
    )
}

pub(crate) fn create_acg_llm_stream_execution_intercept(
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    provider: String,
    plugin: Arc<dyn ProviderPlugin>,
) -> LlmStreamExecutionFn {
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let cache = hot_cache.clone();
            let agent_id = agent_id.clone();
            let provider = provider.clone();
            let plugin = plugin.clone();
            Box::pin(async move {
                let translated =
                    translate_request(&request, &agent_id, &provider, plugin.as_ref(), &cache)
                        .unwrap_or(request);
                next(translated).await
            })
        },
    )
}

#[cfg(test)]
#[path = "../tests/unit/acg_component_tests.rs"]
mod tests;
