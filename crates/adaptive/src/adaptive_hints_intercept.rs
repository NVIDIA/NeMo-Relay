// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AdaptiveHintsIntercept: opt-in LLM request intercept that injects AgentHints
//! from HotCache trie.
//!
//! This module provides [`AdaptiveHintsIntercept`], which builds [`AgentHints`] from
//! the prediction trie in [`HotCache`] and injects them into LLM request
//! headers as a request intercept. AdaptiveHintsIntercept is opt-in and synchronously
//! transforms the [`LLMRequest`] before it reaches the callable.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use nemo_flow::codec::request::AnnotatedLLMRequest;
use nemo_flow::context::callbacks::LlmRequestInterceptFn;
use nemo_flow::types::llm::LLMRequest;

use crate::context_helpers::{
    extract_scope_path, read_manual_latency_sensitivity, resolve_agent_id,
};
use crate::intercepts::AGENT_HINTS_HEADER_KEY;
use crate::trie::builder::SensitivityConfig;
use crate::trie::lookup::PredictionTrieLookup;
use crate::types::cache::HotCache;
use crate::types::metadata::AgentHints;

/// Builds [`AgentHints`] from a trie prediction and optional default hints.
///
/// Falls back to `default_hints` if no prediction is available.
/// Sets `prefix_id` to `"{agent_id}-d{scope_depth}"` per architecture doc.
pub(crate) fn build_agent_hints(
    prediction: Option<&crate::trie::data_models::LlmCallPrediction>,
    default_hints: &Option<AgentHints>,
    agent_id: &str,
    call_index: u32,
    scope_depth: usize,
) -> Option<AgentHints> {
    if let Some(pred) = prediction {
        let scale = SensitivityConfig::default().sensitivity_scale;
        let ls = pred.latency_sensitivity.unwrap_or(1);
        Some(AgentHints {
            osl: pred.output_tokens.p90.round() as u32,
            iat: pred.interarrival_ms.mean.round() as u32,
            priority: (scale as i32 - ls as i32).max(0),
            latency_sensitivity: ls as f64,
            prefix_id: format!("{agent_id}-d{scope_depth}"),
            total_requests: pred.remaining_calls.mean.round() as u32 + call_index,
        })
    } else {
        default_hints.clone()
    }
}

/// Opt-in LLM request intercept that injects [`AgentHints`] into request
/// headers from the prediction trie in [`HotCache`].
///
/// Constructed via [`AdaptiveHintsIntercept::new`] and converted to an
/// [`LlmRequestInterceptFn`] via [`AdaptiveHintsIntercept::into_request_fn`] for
/// registration with the NeMo Flow runtime.
pub struct AdaptiveHintsIntercept {
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    call_counter: AtomicU32,
}

impl AdaptiveHintsIntercept {
    /// Creates a new `AdaptiveHintsIntercept`.
    pub fn new(hot_cache: Arc<RwLock<HotCache>>, agent_id: String) -> Self {
        Self {
            hot_cache,
            agent_id,
            call_counter: AtomicU32::new(1),
        }
    }

    /// Converts this intercept into an [`LlmRequestInterceptFn`] suitable for
    /// registration with [`register_llm_request_intercept`].
    ///
    /// The returned closure reads the HotCache trie, builds AgentHints,
    /// injects them into the request headers and body, and returns the
    /// transformed request.
    pub fn into_request_fn(self) -> LlmRequestInterceptFn {
        let this = Arc::new(self);
        Box::new(
            move |_name: &str, mut request: LLMRequest, annotated: Option<AnnotatedLLMRequest>| {
                // LOCK ORDERING: scope_stack first, hot_cache second.
                let scope_path = extract_scope_path();
                let manual_ls = read_manual_latency_sensitivity();
                let scope_depth = scope_path.len();
                let call_index = this.call_counter.fetch_add(1, Ordering::Relaxed);

                // Resolve agent ID: scope metadata → root scope name → adaptive config
                let effective_agent_id =
                    resolve_agent_id().unwrap_or_else(|| this.agent_id.clone());

                // Read hot cache
                let final_hints = match this.hot_cache.read() {
                    Ok(cache_guard) => {
                        // Build hints from trie or defaults
                        let hints = if let Some(ref trie) = cache_guard.trie {
                            let lookup = PredictionTrieLookup::new(trie);
                            let prediction = lookup.find(&scope_path, call_index);
                            build_agent_hints(
                                prediction,
                                &cache_guard.agent_hints_default,
                                &effective_agent_id,
                                call_index,
                                scope_depth,
                            )
                        } else {
                            cache_guard.agent_hints_default.clone()
                        };

                        // Apply manual latency sensitivity override (max-merge)
                        match (hints, manual_ls) {
                            (Some(mut h), Some(manual)) => {
                                let manual_f = manual as f64;
                                if manual_f > h.latency_sensitivity {
                                    let scale = SensitivityConfig::default().sensitivity_scale;
                                    h.latency_sensitivity = manual_f;
                                    h.priority = (scale as i32 - manual_f.round() as i32).max(0);
                                }
                                Some(h)
                            }
                            (Some(h), None) => Some(h),
                            (None, Some(manual)) => {
                                let scale = SensitivityConfig::default().sensitivity_scale;
                                Some(AgentHints {
                                    osl: 0,
                                    iat: 0,
                                    priority: (scale as i32 - manual as i32).max(0),
                                    latency_sensitivity: manual as f64,
                                    prefix_id: format!("{effective_agent_id}-d{scope_depth}"),
                                    total_requests: 0,
                                })
                            }
                            (None, None) => None,
                        }
                    }
                    _ => {
                        None // Lock poisoned -- pass through
                    }
                };

                // Inject hints into request body at content.nvext.agent_hints
                // (matches NAT's adaptive transport injection point)
                if let Some(hints) = final_hints
                    && let Ok(val) = serde_json::to_value(&hints)
                {
                    // Ensure content is an object
                    if let Some(body) = request.content.as_object_mut() {
                        // Ensure nvext is an object
                        if !body.contains_key("nvext") {
                            body.insert(
                                "nvext".to_string(),
                                serde_json::Value::Object(serde_json::Map::new()),
                            );
                        }
                        if let Some(nvext) = body.get_mut("nvext").and_then(|v| v.as_object_mut()) {
                            nvext.insert("agent_hints".to_string(), val);
                        }
                    }
                    // Also set the header for backward compat with adaptive consumers
                    if let Ok(header_val) = serde_json::to_value(&hints) {
                        request
                            .headers
                            .insert(AGENT_HINTS_HEADER_KEY.to_string(), header_val);
                    }
                }

                Ok((request, annotated))
            },
        )
    }
}

#[cfg(test)]
#[path = "../tests/unit/adaptive_hints_intercept_tests.rs"]
mod tests;
