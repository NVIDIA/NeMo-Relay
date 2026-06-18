// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AdaptiveHintsIntercept: opt-in LLM request intercept that injects AgentHints
//! from HotCache trie.
//!
//! This module provides [`AdaptiveHintsIntercept`], which builds [`AgentHints`] from
//! the prediction trie in [`HotCache`] and injects them into LLM request
//! headers as a request intercept. AdaptiveHintsIntercept is opt-in and synchronously
//! transforms the [`LlmRequest`] before it reaches the callable.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::LlmRequestInterceptFn;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay_adaptive_topology::GeometricGovernor;

use crate::config::GovernorConfig;
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

fn apply_manual_latency_override(
    hints: Option<AgentHints>,
    manual_ls: Option<u32>,
    effective_agent_id: &str,
    scope_depth: usize,
) -> Option<AgentHints> {
    match (hints, manual_ls) {
        (Some(mut hints), Some(manual)) => {
            let manual_f = manual as f64;
            if manual_f > hints.latency_sensitivity {
                let scale = SensitivityConfig::default().sensitivity_scale;
                hints.latency_sensitivity = manual_f;
                hints.priority = (scale as i32 - manual_f.round() as i32).max(0);
            }
            Some(hints)
        }
        (Some(hints), None) => Some(hints),
        (None, Some(manual)) => Some(manual_agent_hints(manual, effective_agent_id, scope_depth)),
        (None, None) => None,
    }
}

fn manual_agent_hints(manual: u32, effective_agent_id: &str, scope_depth: usize) -> AgentHints {
    let scale = SensitivityConfig::default().sensitivity_scale;
    AgentHints {
        osl: 0,
        iat: 0,
        priority: (scale as i32 - manual as i32).max(0),
        latency_sensitivity: manual as f64,
        prefix_id: format!("{effective_agent_id}-d{scope_depth}"),
        total_requests: 0,
    }
}

fn inject_agent_hints(request: &mut LlmRequest, hints: &AgentHints) {
    let Ok(serialized_hints) = serde_json::to_value(hints) else {
        return;
    };

    if let Some(body) = request.content.as_object_mut() {
        if !body.contains_key("nvext") {
            body.insert(
                "nvext".to_string(),
                serde_json::Value::Object(serde_json::Map::new()),
            );
        }
        if let Some(nvext) = body
            .get_mut("nvext")
            .and_then(|value| value.as_object_mut())
        {
            nvext.insert("agent_hints".to_string(), serialized_hints.clone());
        }
    }

    request
        .headers
        .insert(AGENT_HINTS_HEADER_KEY.to_string(), serialized_hints);
}

/// Opt-in LLM request intercept that injects [`AgentHints`] into request
/// headers from the prediction trie in [`HotCache`].
///
/// Constructed via [`AdaptiveHintsIntercept::new`] and converted to an
/// [`LlmRequestInterceptFn`] via [`AdaptiveHintsIntercept::into_request_fn`] for
/// registration with the NeMo Relay runtime.
pub struct AdaptiveHintsIntercept {
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    call_counter: AtomicU32,
    governor: Option<Arc<Mutex<HintGovernor>>>,
}

impl AdaptiveHintsIntercept {
    /// Creates a new `AdaptiveHintsIntercept`.
    pub fn new(hot_cache: Arc<RwLock<HotCache>>, agent_id: String) -> Self {
        Self::with_governor(hot_cache, agent_id, None)
    }

    /// Creates a new `AdaptiveHintsIntercept` with optional load shedding.
    pub fn with_governor(
        hot_cache: Arc<RwLock<HotCache>>,
        agent_id: String,
        governor: Option<GovernorConfig>,
    ) -> Self {
        let governor = governor
            .filter(|config| config.enabled)
            .map(|config| Arc::new(Mutex::new(HintGovernor::new(config.epsilon))));
        Self {
            hot_cache,
            agent_id,
            call_counter: AtomicU32::new(1),
            governor,
        }
    }

    fn effective_agent_id(&self) -> String {
        resolve_agent_id().unwrap_or_else(|| self.agent_id.clone())
    }

    fn load_hints(
        &self,
        scope_path: &[String],
        effective_agent_id: &str,
        call_index: u32,
        scope_depth: usize,
    ) -> Option<AgentHints> {
        let Ok(cache_guard) = self.hot_cache.read() else {
            return None;
        };

        if let Some(ref trie) = cache_guard.trie {
            let lookup = PredictionTrieLookup::new(trie);
            let prediction = lookup.find(scope_path, call_index);
            build_agent_hints(
                prediction,
                &cache_guard.agent_hints_default,
                effective_agent_id,
                call_index,
                scope_depth,
            )
        } else {
            cache_guard.agent_hints_default.clone()
        }
    }

    fn should_inject_hints(&self, hints: &AgentHints, manual_ls: Option<u32>) -> bool {
        if manual_ls.is_some() {
            return true;
        }

        let Some(governor) = &self.governor else {
            return true;
        };

        governor
            .lock()
            .map(|mut governor| governor.allow(hints.latency_sensitivity))
            .unwrap_or(true)
    }

    /// Converts this intercept into an [`LlmRequestInterceptFn`] suitable for
    /// registration with [`register_llm_request_intercept`].
    ///
    /// The returned closure reads the HotCache trie, builds AgentHints,
    /// injects them into the request headers and body, and returns the
    /// transformed request.
    pub fn into_request_fn(self) -> LlmRequestInterceptFn {
        let this = Arc::new(self);
        Arc::new(
            move |_name: &str, mut request: LlmRequest, annotated: Option<AnnotatedLlmRequest>| {
                let scope_path = extract_scope_path();
                let manual_ls = read_manual_latency_sensitivity();
                let scope_depth = scope_path.len();
                let call_index = this.call_counter.fetch_add(1, Ordering::Relaxed);

                let effective_agent_id = this.effective_agent_id();
                let cached_hints =
                    this.load_hints(&scope_path, &effective_agent_id, call_index, scope_depth);
                let final_hints = apply_manual_latency_override(
                    cached_hints,
                    manual_ls,
                    &effective_agent_id,
                    scope_depth,
                );

                if let Some(hints) = final_hints
                    && this.should_inject_hints(&hints, manual_ls)
                {
                    inject_agent_hints(&mut request, &hints);
                }

                Ok((request, annotated))
            },
        )
    }
}

struct HintGovernor {
    governor: GeometricGovernor,
    last_seen: Option<Instant>,
}

impl HintGovernor {
    fn new(epsilon: f64) -> Self {
        Self {
            governor: GeometricGovernor::with_epsilon(epsilon),
            last_seen: None,
        }
    }

    fn allow(&mut self, latency_sensitivity: f64) -> bool {
        let allow = self.governor.should_trigger(latency_sensitivity);
        let now = Instant::now();
        if let Some(last_seen) = self.last_seen {
            let dt = now.duration_since(last_seen).as_secs_f64().max(0.000_001);
            let observed_rate = 1.0 / dt;
            self.governor.adapt(observed_rate, dt);
        }
        self.last_seen = Some(now);
        allow
    }
}

#[cfg(test)]
#[path = "../tests/unit/adaptive_hints_intercept_tests.rs"]
mod tests;
