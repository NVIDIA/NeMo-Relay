// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache savings marks in the NeMo Relay adaptive crate.

use std::time::Duration;

use nemo_relay::codec::model_pricing::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};

use super::*;

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
    reset_active_pricing_resolver().unwrap();
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
