// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DynamoIntercept: opt-in LLM request intercept that injects AgentHints
//! from HotCache trie.
//!
//! This module provides [`DynamoIntercept`], which builds [`AgentHints`] from
//! the prediction trie in [`HotCache`] and injects them into LLM request
//! headers as a request intercept. DynamoIntercept is opt-in and synchronously
//! transforms the [`LLMRequest`] before it reaches the callable.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use nvidia_nat_nexus_core::{LLMRequest, LlmRequestInterceptFn};

use crate::context_helpers::{
    extract_scope_path, read_manual_latency_sensitivity, resolve_agent_id,
};
use crate::intercepts::AGENT_HINTS_HEADER_KEY;
use crate::trie::lookup::PredictionTrieLookup;
use crate::trie::SensitivityConfig;
use crate::types::{AgentHints, HotCache};

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
/// Constructed via [`DynamoIntercept::new`] and converted to an
/// [`LlmRequestInterceptFn`] via [`DynamoIntercept::into_request_fn`] for
/// registration with the Nexus runtime.
pub struct DynamoIntercept {
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    call_counter: AtomicU32,
}

impl DynamoIntercept {
    /// Creates a new `DynamoIntercept`.
    pub fn new(hot_cache: Arc<RwLock<HotCache>>, agent_id: String) -> Self {
        Self {
            hot_cache,
            agent_id,
            call_counter: AtomicU32::new(1),
        }
    }

    /// Converts this intercept into an [`LlmRequestInterceptFn`] suitable for
    /// registration with [`nat_nexus_register_llm_request_intercept`].
    ///
    /// The returned closure reads the HotCache trie, builds AgentHints,
    /// injects them into the request headers and body, and returns the
    /// transformed request.
    pub fn into_request_fn(self) -> LlmRequestInterceptFn {
        let this = Arc::new(self);
        Box::new(move |_name: &str, mut request: LLMRequest| {
            // LOCK ORDERING: scope_stack first, hot_cache second.
            let scope_path = extract_scope_path();
            let manual_ls = read_manual_latency_sensitivity();
            let scope_depth = scope_path.len();
            let call_index = this.call_counter.fetch_add(1, Ordering::Relaxed);

            // Resolve agent ID: scope metadata → root scope name → proxy config
            let effective_agent_id = resolve_agent_id().unwrap_or_else(|| this.agent_id.clone());

            // Read hot cache
            let final_hints = if let Ok(cache_guard) = this.hot_cache.read() {
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
            } else {
                None // Lock poisoned -- pass through
            };

            // Inject hints into request body at content.nvext.agent_hints
            // (matches NAT's DynamoTransport injection point)
            if let Some(hints) = final_hints {
                if let Ok(val) = serde_json::to_value(&hints) {
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
                    // Also set the header for backward compat with proxy consumers
                    if let Ok(header_val) = serde_json::to_value(&hints) {
                        request
                            .headers
                            .insert(AGENT_HINTS_HEADER_KEY.to_string(), header_val);
                    }
                }
            }

            Ok(request)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trie::data_models::{LlmCallPrediction, PredictionMetrics};

    #[test]
    fn test_build_agent_hints_from_prediction() {
        let pred = LlmCallPrediction {
            remaining_calls: PredictionMetrics {
                sample_count: 10,
                mean: 5.0,
                p50: 5.0,
                p90: 8.0,
                p95: 9.0,
            },
            interarrival_ms: PredictionMetrics {
                sample_count: 10,
                mean: 200.0,
                p50: 180.0,
                p90: 300.0,
                p95: 350.0,
            },
            output_tokens: PredictionMetrics {
                sample_count: 10,
                mean: 100.0,
                p50: 90.0,
                p90: 150.0,
                p95: 180.0,
            },
            latency_sensitivity: Some(4),
        };

        let hints = build_agent_hints(Some(&pred), &None, "test-agent", 2, 3).unwrap();
        assert_eq!(hints.osl, 150, "osl = output_tokens.p90");
        assert_eq!(hints.iat, 200, "iat = interarrival_ms.mean");
        assert_eq!(hints.priority, 1, "priority = 5 - 4 = 1");
        assert!((hints.latency_sensitivity - 4.0).abs() < f64::EPSILON);
        assert_eq!(hints.prefix_id, "test-agent-d3");
        assert_eq!(hints.total_requests, 7, "total_requests = 5 + 2 = 7");
    }

    #[test]
    fn test_build_agent_hints_falls_back_to_defaults() {
        let defaults = AgentHints {
            osl: 42,
            iat: 99,
            priority: 1,
            latency_sensitivity: 4.0,
            prefix_id: "fallback".into(),
            total_requests: 10,
        };
        let hints = build_agent_hints(None, &Some(defaults.clone()), "agent", 1, 0).unwrap();
        assert_eq!(hints.osl, 42);
        assert_eq!(hints.prefix_id, "fallback");
    }

    #[test]
    fn test_build_agent_hints_none_when_no_prediction_and_no_defaults() {
        let hints = build_agent_hints(None, &None, "agent", 1, 0);
        assert!(hints.is_none());
    }

    #[test]
    fn test_dynamo_intercept_new() {
        let hot_cache = Arc::new(RwLock::new(HotCache {
            plan: None,
            trie: None,
            agent_hints_default: None,
        }));
        let intercept = DynamoIntercept::new(hot_cache, "test".to_string());
        assert_eq!(intercept.call_counter.load(Ordering::Relaxed), 1);
        assert_eq!(intercept.agent_id, "test");
    }

    #[test]
    fn test_dynamo_intercept_into_request_fn_compiles() {
        let hot_cache = Arc::new(RwLock::new(HotCache {
            plan: None,
            trie: None,
            agent_hints_default: None,
        }));
        let intercept = DynamoIntercept::new(hot_cache, "test".to_string());
        let _req_fn: LlmRequestInterceptFn = intercept.into_request_fn();
        // If this compiles and runs, the type is correct.
    }
}
