// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `response_cache` mark: every cache decision, its metadata, and the
//! savings a reuse reports.

use nemo_relay::api::scope::{EmitMarkEventParams, event};
use nemo_relay::codec::model_pricing::{
    attach_estimated_cost_for_provider, estimate_cost_for_provider,
};
use nemo_relay::codec::resolve::{detect_response_surface, response_codec};
use nemo_relay::codec::response::Usage;
use serde_json::{Map, Value as Json, json};

use crate::response_cache::store::CacheEntry;

/// Mark-event name emitted on every cache decision.
pub const RESPONSE_CACHE_MARK: &str = "response_cache";

/// Pulls saved token count and cost out of a stored entry (the aggregate
/// response object — buffered and streaming both store this shape).
///
/// A body of a built-in provider shape is decoded through its response codec,
/// which owns the usage normalization and cost precedence (a provider-reported
/// cost on the body wins; otherwise the cost is derived from the token counts
/// via the active pricing resolver). Shapes no codec recognizes fall back to
/// raw JSON probing — savings reporting is best-effort and fails open.
pub(crate) fn savings_from(entry: &CacheEntry) -> (Option<u64>, Option<f64>) {
    match normalized_savings(entry) {
        Some(savings) => savings,
        None => probed_savings(entry),
    }
}

/// Savings read from the normalized decode of a built-in provider response
/// shape. `None` when nothing detects/decodes or the decode carries neither a
/// token count nor a cost (the raw probe may still read an unmodeled shape).
fn normalized_savings(entry: &CacheEntry) -> Option<(Option<u64>, Option<f64>)> {
    let surface = detect_response_surface(&entry.response)?;
    let mut decoded = response_codec(surface)
        .decode_response(&entry.response)
        .ok()?;
    if decoded.model.is_none() {
        decoded.model = entry.model_name.clone();
    }
    attach_estimated_cost_for_provider(&mut decoded, entry.provider_name.as_deref());
    let usage = decoded.usage?;
    let tokens = usage.total_tokens.or_else(|| {
        Some(
            usage
                .prompt_tokens?
                .saturating_add(usage.completion_tokens?),
        )
    });
    let cost = usage.cost.and_then(|cost| cost.total);
    if tokens.is_none() && cost.is_none() {
        return None;
    }
    Some((tokens, cost))
}

/// Raw JSON probing fallback for stored bodies no built-in codec decodes.
fn probed_savings(entry: &CacheEntry) -> (Option<u64>, Option<f64>) {
    let response = &entry.response;
    let usage = response
        .get("usage")
        .or_else(|| response.get("token_usage"));
    let tokens = usage.and_then(|usage| {
        usage
            .get("total_tokens")
            .and_then(Json::as_u64)
            .or_else(|| {
                let prompt = usage.get("prompt_tokens").and_then(Json::as_u64)?;
                let completion = usage.get("completion_tokens").and_then(Json::as_u64)?;
                Some(prompt.saturating_add(completion))
            })
            .or_else(|| {
                // Anthropic Messages usage uses input_tokens / output_tokens.
                let input = usage.get("input_tokens").and_then(Json::as_u64)?;
                let output = usage.get("output_tokens").and_then(Json::as_u64)?;
                Some(input.saturating_add(output))
            })
    });
    // 1. Provider-reported cost on the body wins.
    let body_cost = usage
        .and_then(|usage| {
            usage
                .get("cost_usd")
                .and_then(Json::as_f64)
                .or_else(|| usage.pointer("/cost/total").and_then(Json::as_f64))
        })
        .or_else(|| response.get("cost_usd").and_then(Json::as_f64));
    // 2. Otherwise price the token counts with the active resolver.
    let cost = body_cost.or_else(|| {
        let model = entry.model_name.as_deref()?;
        let usage: Usage = serde_json::from_value(usage?.clone()).ok()?;
        estimate_cost_for_provider(entry.provider_name.as_deref(), model, &usage)
            .and_then(|estimate| estimate.total)
    });
    (tokens, cost)
}

pub(crate) struct CacheMark<'a> {
    status: &'a str,
    reason: Option<&'a str>,
    backend: &'a str,
    key_hash: Option<&'a str>,
    age_ms: Option<u64>,
    ttl_ms: Option<u64>,
    saved_tokens: Option<u64>,
    saved_cost_usd: Option<f64>,
}

impl<'a> CacheMark<'a> {
    pub(crate) fn new(status: &'a str, backend: &'a str) -> Self {
        Self {
            status,
            reason: None,
            backend,
            key_hash: None,
            age_ms: None,
            ttl_ms: None,
            saved_tokens: None,
            saved_cost_usd: None,
        }
    }

    pub(crate) fn reason(mut self, reason: &'a str) -> Self {
        self.reason = Some(reason);
        self
    }

    pub(crate) fn key_hash(mut self, key_hash: &'a str) -> Self {
        self.key_hash = Some(key_hash);
        self
    }

    pub(crate) fn age_ms(mut self, age_ms: u64) -> Self {
        self.age_ms = Some(age_ms);
        self
    }

    pub(crate) fn ttl_ms(mut self, ttl_ms: u64) -> Self {
        self.ttl_ms = Some(ttl_ms);
        self
    }

    pub(crate) fn savings(mut self, tokens: Option<u64>, cost: Option<f64>) -> Self {
        self.saved_tokens = tokens;
        self.saved_cost_usd = cost;
        self
    }
}

/// Emits the `response_cache` mark. Only the key fingerprint is ever recorded —
/// never raw prompts, answers, or credentials. Fails open.
pub(crate) fn emit_cache_mark(mark: CacheMark<'_>) {
    let mut metadata = Map::new();
    metadata.insert(
        "nemo_relay.response_cache.surface".to_string(),
        json!("llm"),
    );
    metadata.insert(
        "nemo_relay.response_cache.backend".to_string(),
        json!(mark.backend),
    );
    if let Some(reason) = mark.reason {
        metadata.insert(
            "nemo_relay.response_cache.reason".to_string(),
            json!(reason),
        );
    }
    if let Some(key_hash) = mark.key_hash {
        metadata.insert(
            "nemo_relay.response_cache.key_hash".to_string(),
            json!(key_hash),
        );
    }
    if let Some(age_ms) = mark.age_ms {
        metadata.insert(
            "nemo_relay.response_cache.age_ms".to_string(),
            json!(age_ms),
        );
    }
    if let Some(ttl_ms) = mark.ttl_ms {
        metadata.insert(
            "nemo_relay.response_cache.ttl_ms".to_string(),
            json!(ttl_ms),
        );
    }
    if let Some(saved_tokens) = mark.saved_tokens {
        metadata.insert(
            "nemo_relay.response_cache.saved_tokens".to_string(),
            json!(saved_tokens),
        );
    }
    if let Some(saved_cost_usd) = mark.saved_cost_usd {
        metadata.insert(
            "nemo_relay.response_cache.saved_cost_usd".to_string(),
            json!(saved_cost_usd),
        );
    }

    let _ = event(
        EmitMarkEventParams::builder()
            .name(RESPONSE_CACHE_MARK)
            .data(json!({ "status": mark.status }))
            .metadata(Json::Object(metadata))
            .build(),
    );
}

#[cfg(test)]
#[path = "../../tests/unit/response_cache/mark_tests.rs"]
mod tests;
