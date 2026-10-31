// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::test_support::pricing_test_mutex;
use nemo_relay_model_pricing::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};

use serde_json::json;

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

#[test]
fn test_decode_response_omits_estimated_cost_for_cross_model_iterations() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [{
                "provider": "anthropic",
                "model_id": "claude-fallback",
                "pricing_as_of": "2026-09-25",
                "pricing_source": "test",
                "rates": {"input_per_million": 1.0, "output_per_million": 2.0},
                "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
            }]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let response = json!({
        "id": "msg_fallback",
        "model": "claude-fallback",
        "content": [{"type": "text", "text": "Hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 412,
            "output_tokens": 264,
            "iterations": [
                {"type": "message", "model": "claude-primary", "input_tokens": 535, "output_tokens": 0},
                {"type": "fallback_message", "model": "claude-fallback", "input_tokens": 412, "output_tokens": 264}
            ]
        }
    });

    let usage = crate::codec::response_codec(
        nemo_relay_codecs::resolve::ProviderSurface::AnthropicMessages,
    )
    .decode_response(&response)
    .unwrap()
    .usage
    .unwrap();

    assert_eq!(usage.prompt_tokens, Some(947));
    assert_eq!(usage.completion_tokens, Some(264));
    assert_eq!(usage.cost, None);
}
