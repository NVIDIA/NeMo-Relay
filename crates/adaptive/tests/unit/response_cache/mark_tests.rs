// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache savings marks in the NeMo Relay adaptive crate.

use std::{hint::black_box, time::Duration};

use nemo_relay::codec::model_pricing::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};

use super::*;

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

#[test]
fn cache_surfaces_have_stable_metadata_values() {
    assert_eq!(CacheSurface::Llm.as_str(), "llm");
    assert_eq!(CacheSurface::Tool.as_str(), "tool");
}

#[test]
fn cache_mark_labels_have_stable_telemetry_values() {
    assert_eq!(black_box(CacheMarkStatus::Bypass).as_str(), "bypass");
    assert_eq!(black_box(CacheMarkStatus::Hit).as_str(), "hit");
    assert_eq!(black_box(CacheMarkStatus::Miss).as_str(), "miss");
    assert_eq!(
        black_box(CacheReason::CanonicalizationFailed).as_str(),
        "canonicalization_failed"
    );
    assert_eq!(black_box(CacheReason::CachedError).as_str(), "cached_error");
    assert_eq!(
        black_box(CacheReason::NondeterministicTemperature).as_str(),
        "nondeterministic_temperature"
    );
    assert_eq!(black_box(CacheReason::ReplayLossy).as_str(), "replay_lossy");
    assert_eq!(black_box(CacheReason::Sampled).as_str(), "sampled");
    assert_eq!(
        black_box(CacheReason::StatefulConversation).as_str(),
        "stateful_conversation"
    );
    assert_eq!(
        black_box(CacheReason::StatefulPreviousResponseId).as_str(),
        "stateful_previous_response_id"
    );
    assert_eq!(
        black_box(CacheReason::StatefulStore).as_str(),
        "stateful_store"
    );
    assert_eq!(black_box(CacheReason::StoreError).as_str(), "store_error");
    assert_eq!(
        black_box(CacheReason::StreamNoCodec).as_str(),
        "stream_no_codec"
    );
    assert_eq!(
        black_box(CacheReason::UnparseableBody).as_str(),
        "unparseable_body"
    );
    assert_eq!(black_box(CacheReason::Uncacheable).as_str(), "uncacheable");
    assert_eq!(
        black_box(CacheReason::UnrepresentableNumber).as_str(),
        "unrepresentable_number"
    );
}

#[test]
fn anthropic_shaped_bodies_price_through_the_catalog() {
    // A real Anthropic response body must yield a dollar figure when the
    // model is in the pricing catalog: its usage carries only
    // input_tokens/output_tokens, which `Usage` has no aliases for, so raw
    // probing can count tokens but never price them.
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [{
                "provider": "anthropic",
                "model_id": "claude-cache-price-test",
                "pricing_as_of": "2026-07-01",
                "pricing_source": "test",
                "rates": {"input_per_million": 3.0, "output_per_million": 15.0},
                "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
            }]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
    let _reset_pricing = ResetPricingResolverGuard;
    let entry = CacheEntry::new(
        json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-cache-price-test",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 900, "output_tokens": 100}
        }),
        Duration::from_secs(60),
        "sha256:x".to_string(),
        Some("claude-cache-price-test".to_string()),
        Some("anthropic.messages".to_string()),
    );
    let (tokens, cost) = savings_from(&entry);
    assert_eq!(tokens, Some(1000));
    let cost = cost.expect("a cataloged anthropic hit must report saved cost");
    assert!(
        (cost - 0.0042).abs() < 1e-12,
        "900 input + 100 output at 3.0/15.0 per million must price at 0.0042, got {cost}"
    );
}

#[test]
fn savings_from_counts_anthropic_input_output_tokens() {
    // A bare content+usage body no built-in codec detects must still count
    // input_tokens/output_tokens through the raw-probing fallback, or such
    // hits report zero avoided tokens.
    let entry = CacheEntry {
        response: json!({
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 100, "output_tokens": 25}
        }),
        created_unix_ms: 0,
        expires_unix_ms: 0,
        key_hash: "sha256:x".to_string(),
        model_name: Some("claude-x".to_string()),
        provider_name: Some("anthropic.messages".to_string()),
    };
    let (tokens, _cost) = savings_from(&entry);
    assert_eq!(
        tokens,
        Some(125),
        "anthropic input+output tokens must be counted for savings"
    );
}

#[test]
fn normalized_savings_uses_entry_model_and_derives_missing_total_tokens() {
    // Providers can omit a model in the payload while the cache knows the
    // request model. A Chat response with prompt/completion tokens but no
    // total must still report its complete saved-token count.
    let entry = CacheEntry::new(
        json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5}
        }),
        Duration::from_secs(60),
        "sha256:chat".to_string(),
        Some("model-recorded-with-request".to_string()),
        Some("openai".to_string()),
    );

    assert_eq!(savings_from(&entry).0, Some(12));
}

#[test]
fn normalized_empty_usage_falls_back_to_no_savings() {
    // A recognized response with an empty usage object is not a zero-token
    // hit: it is missing accounting, so diagnostics must leave savings unset.
    let entry = CacheEntry::new(
        json!({
            "id": "chatcmpl_2",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {}
        }),
        Duration::from_secs(60),
        "sha256:empty-usage".to_string(),
        None,
        None,
    );

    assert_eq!(normalized_savings(&entry), None);
    assert_eq!(savings_from(&entry), (None, None));
}

#[test]
fn raw_usage_probe_derives_total_from_prompt_and_completion_tokens() {
    // Unknown provider shapes still expose standard OpenAI-style usage fields;
    // raw fallback must preserve their useful savings diagnostics.
    let entry = CacheEntry::new(
        json!({"usage": {"prompt_tokens": 11, "completion_tokens": 4}}),
        Duration::from_secs(60),
        "sha256:raw".to_string(),
        None,
        None,
    );

    assert_eq!(savings_from(&entry), (Some(15), None));
}
