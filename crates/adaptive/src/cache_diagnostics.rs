// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-local cache miss request facts and diagnostics tracking.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::acg::canonicalize::sha256_hex;
use crate::acg::ir_builder::build_prompt_ir;
use crate::acg::prompt_ir::PromptIR;
use crate::acg::{CacheRequestFacts, CapabilityRegistry};
use chrono::{DateTime, Utc};
use nemo_relay::codec::request::AnnotatedLlmRequest;

use crate::acg_profile::derive_acg_learning_key;
use crate::types::cache::HotCache;

const DEFAULT_ANTHROPIC_MIN_TOKENS: u32 = 1024;
const OPENAI_MIN_TOKENS: u32 = 1024;
const ANTHROPIC_RETENTION_WINDOW_SECS: f64 = 300.0;
const HASH_PREFIX_LEN: usize = 12;

type StablePrefixKey = (String, String, String);
type StablePrefixExemplar = Vec<(String, u32, String)>;
type AgentProviderKey = (String, String);

struct CacheFactsBuildInput<'a> {
    agent_id: &'a str,
    provider: &'a str,
    model: Option<&'a str>,
    prompt_ir: &'a PromptIR,
    hot_cache: &'a HotCache,
    profile_key: &'a str,
    now: DateTime<Utc>,
}

/// Runtime-local miss diagnosis tracker.
#[derive(Debug, Default)]
pub struct CacheDiagnosticsTracker {
    /// Last time a specific stable prefix hash was observed for an agent/provider pair.
    pub last_seen_by_prefix: HashMap<StablePrefixKey, DateTime<Utc>>,
    /// Last retained stable prefix exemplar for an agent/provider pair.
    pub last_exemplar_by_agent: HashMap<AgentProviderKey, StablePrefixExemplar>,
}

/// Builds canonical request facts for cache miss diagnosis from the live runtime state.
#[must_use]
pub fn build_cache_request_facts(
    agent_id: &str,
    provider: &str,
    annotated_request: &AnnotatedLlmRequest,
    hot_cache: &Arc<RwLock<HotCache>>,
    tracker: &Arc<RwLock<CacheDiagnosticsTracker>>,
) -> Option<CacheRequestFacts> {
    let prompt_ir = build_prompt_ir(annotated_request).ok()?;
    let hot_cache = hot_cache.read().ok()?;
    let mut tracker = tracker.write().ok()?;
    let profile_key = derive_acg_learning_key(agent_id, annotated_request);

    Some(build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id,
            provider,
            model: annotated_request.model.as_deref(),
            prompt_ir: &prompt_ir,
            hot_cache: &hot_cache,
            profile_key: &profile_key,
            now: Utc::now(),
        },
        &mut tracker,
    ))
}

fn build_cache_request_facts_from_prompt_ir(
    input: CacheFactsBuildInput<'_>,
    tracker: &mut CacheDiagnosticsTracker,
) -> CacheRequestFacts {
    let CacheFactsBuildInput {
        agent_id,
        provider,
        model,
        prompt_ir,
        hot_cache,
        profile_key,
        now,
    } = input;

    let Some(stability) = hot_cache
        .acg_profiles
        .get(profile_key)
        .or(hot_cache.acg_stability.as_ref())
    else {
        return CacheRequestFacts {
            provider: provider.to_string(),
            stable_prefix_length: 0,
            stable_prefix_tokens: None,
            required_min_tokens: None,
            first_mismatch_span_id: None,
            first_mismatch_sequence_index: None,
            expected_hash_prefix: None,
            actual_hash_prefix: None,
            retention_window_secs: None,
            observed_gap_secs: None,
            missing_facts: vec!["acg_stability_unavailable".to_string()],
        };
    };

    let stable_prefix_length = stability.stable_prefix_length;
    let stable_blocks = prompt_ir
        .blocks
        .iter()
        .take(stable_prefix_length)
        .collect::<Vec<_>>();
    let mut missing_facts = Vec::new();

    let stable_prefix_tokens = if stable_blocks.len() < stable_prefix_length {
        missing_facts.push("stable_prefix_tokens_unavailable".to_string());
        None
    } else {
        stable_blocks
            .iter()
            .try_fold(0_u32, |acc, block| {
                block
                    .token_metadata
                    .as_ref()
                    .and_then(|meta| acc.checked_add(meta.token_count))
            })
            .or_else(|| {
                if stable_blocks
                    .iter()
                    .any(|block| block.token_metadata.is_none())
                {
                    missing_facts.push("stable_prefix_tokens_unavailable".to_string());
                }
                None
            })
    };

    let current_exemplar = stable_blocks
        .iter()
        .map(|block| {
            (
                block.span_id.0.clone(),
                block.sequence_index,
                short_hash_prefix(&block.content),
            )
        })
        .collect::<Vec<_>>();
    let current_prefix_hash =
        if stable_prefix_length == 0 || stable_blocks.len() < stable_prefix_length {
            None
        } else {
            Some(prefix_hash(
                stable_blocks.iter().map(|block| block.content.as_str()),
            ))
        };

    let agent_provider_key = (agent_id.to_string(), provider.to_string());
    let first_mismatch = tracker
        .last_exemplar_by_agent
        .get(&agent_provider_key)
        .and_then(|previous| first_mismatch(previous, &current_exemplar));

    let (
        first_mismatch_span_id,
        first_mismatch_sequence_index,
        expected_hash_prefix,
        actual_hash_prefix,
    ) = if let Some((span_id, sequence_index, expected_hash_prefix, actual_hash_prefix)) =
        first_mismatch
    {
        (
            Some(span_id),
            Some(sequence_index),
            Some(expected_hash_prefix),
            Some(actual_hash_prefix),
        )
    } else {
        (None, None, None, None)
    };

    let required_min_tokens = match provider {
        "anthropic" => Some(resolve_anthropic_min_tokens(model)),
        "openai" => Some(OPENAI_MIN_TOKENS),
        _ => None,
    };
    let retention_window_secs = if provider == "anthropic" {
        Some(ANTHROPIC_RETENTION_WINDOW_SECS)
    } else {
        None
    };

    let observed_gap_secs = current_prefix_hash.as_ref().and_then(|stable_prefix_hash| {
        tracker
            .last_seen_by_prefix
            .get(&(
                agent_id.to_string(),
                provider.to_string(),
                stable_prefix_hash.clone(),
            ))
            .map(|last_seen| {
                (now.signed_duration_since(*last_seen).num_milliseconds() as f64 / 1000.0).max(0.0)
            })
    });

    if let Some(stable_prefix_hash) = current_prefix_hash {
        tracker.last_seen_by_prefix.insert(
            (
                agent_id.to_string(),
                provider.to_string(),
                stable_prefix_hash,
            ),
            now,
        );
    }
    tracker
        .last_exemplar_by_agent
        .insert(agent_provider_key, current_exemplar);

    CacheRequestFacts {
        provider: provider.to_string(),
        stable_prefix_length,
        stable_prefix_tokens,
        required_min_tokens,
        first_mismatch_span_id,
        first_mismatch_sequence_index,
        expected_hash_prefix,
        actual_hash_prefix,
        retention_window_secs,
        observed_gap_secs,
        missing_facts,
    }
}

fn resolve_anthropic_min_tokens(model: Option<&str>) -> u32 {
    let registry = CapabilityRegistry::with_defaults();
    model
        .and_then(|model| {
            registry.get_backend("anthropic").and_then(|backend| {
                backend
                    .model_families
                    .get(model)
                    .or_else(|| {
                        backend
                            .model_families
                            .iter()
                            .filter(|(family, _)| model.starts_with(family.as_str()))
                            .max_by_key(|(family, _)| family.len())
                            .map(|(_, caps)| caps)
                    })
                    .and_then(|family| family.min_cacheable_tokens)
            })
        })
        .unwrap_or(DEFAULT_ANTHROPIC_MIN_TOKENS)
}

fn first_mismatch(
    previous: &[(String, u32, String)],
    current: &[(String, u32, String)],
) -> Option<(String, u32, String, String)> {
    previous.iter().zip(current.iter()).find_map(
        |(
            (expected_span_id, expected_sequence_index, expected_hash),
            (actual_span_id, actual_sequence_index, actual_hash),
        )| {
            if expected_span_id != actual_span_id
                || expected_sequence_index != actual_sequence_index
                || expected_hash != actual_hash
            {
                Some((
                    actual_span_id.clone(),
                    *actual_sequence_index,
                    expected_hash.clone(),
                    actual_hash.clone(),
                ))
            } else {
                None
            }
        },
    )
}

fn prefix_hash<'a>(stable_contents: impl Iterator<Item = &'a str>) -> String {
    let joined = stable_contents
        .map(sha256_hex)
        .collect::<Vec<_>>()
        .join("|");
    sha256_hex(&joined)
}

fn short_hash_prefix(content: &str) -> String {
    let full_hash = sha256_hex(content);
    format!(
        "sha256:{}",
        &full_hash["sha256:".len()..][..HASH_PREFIX_LEN]
    )
}

#[cfg(test)]
#[path = "../tests/unit/cache_diagnostics_tests.rs"]
mod tests;
