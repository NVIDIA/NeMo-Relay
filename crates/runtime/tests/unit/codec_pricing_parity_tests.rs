// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::test_support::pricing_test_mutex;
use nemo_relay_model_pricing::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use nemo_relay_types::{Json, codec::response::*};
use serde_json::json;

use nemo_relay_types::codec::normalization::LlmNormalizer;
fn decode(raw: &Json) -> AnnotatedLlmResponse {
    crate::codec::RuntimeLlmNormalizer
        .normalize_response(raw)
        .unwrap()
}
fn chat_response_with_usage(model: &str, extra_usage: Json) -> Json {
    let mut raw = chat_text_response(model);
    let mut usage = json!({
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 200}
    });
    merge_object(&mut usage, extra_usage);
    raw.as_object_mut().unwrap().insert("usage".into(), usage);
    raw
}

fn anthropic_response_with_usage(model: &str, extra_usage: Json) -> Json {
    let mut raw = anthropic_text_response(model);
    // Anthropic reports no total_tokens; the codec computes prompt + completion.
    let mut usage = json!({
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 200
    });
    merge_object(&mut usage, extra_usage);
    raw.as_object_mut().unwrap().insert("usage".into(), usage);
    raw
}

fn responses_response_with_usage(model: &str, extra_usage: Json) -> Json {
    let mut raw = responses_text_response(model);
    let mut usage = json!({
        "input_tokens": 1000,
        "output_tokens": 500,
        "total_tokens": 1500,
        "input_tokens_details": {"cached_tokens": 200}
    });
    merge_object(&mut usage, extra_usage);
    raw.as_object_mut().unwrap().insert("usage".into(), usage);
    raw
}

fn merge_object(target: &mut Json, extra: Json) {
    if let (Some(target), Json::Object(extra)) = (target.as_object_mut(), extra) {
        target.extend(extra);
    }
}
struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

fn install_parity_pricing(model_id: &str) {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "test",
                    "model_id": model_id,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.60,
                        "cache_read_per_million": 0.075
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

#[test]
fn test_estimated_cost_parity_for_identical_model_and_usage() {
    let _pricing_guard = pricing_test_mutex()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    install_parity_pricing("parity-priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    // Each codec infers its own default provider ("openai" vs "anthropic"),
    // but pricing lookup falls back to the model-only key, so an identical
    // model + request must estimate identically everywhere.
    // Anthropic counts cache reads separately from input tokens.
    let chat = decode(&chat_response_with_usage("parity-priced-model", json!({})));
    let anthropic = decode(&anthropic_response_with_usage(
        "parity-priced-model",
        json!({"input_tokens": 800}),
    ));
    let responses = decode(&responses_response_with_usage(
        "parity-priced-model",
        json!({}),
    ));

    let chat_cost = chat.usage.unwrap().cost.expect("chat estimated cost");
    let anthropic_cost = anthropic
        .usage
        .unwrap()
        .cost
        .expect("anthropic estimated cost");
    let responses_cost = responses
        .usage
        .unwrap()
        .cost
        .expect("responses estimated cost");

    assert_eq!(chat_cost, anthropic_cost);
    assert_eq!(chat_cost, responses_cost);
    // 800 billable prompt (1000 - 200 cached) * 0.15/M
    //   + 500 completion * 0.60/M + 200 cache-read * 0.075/M
    let total = chat_cost.total.expect("estimated total");
    assert!(
        (total - 0.000_435).abs() < 1e-9,
        "unexpected estimated total: {total}"
    );
    assert_eq!(chat_cost.currency, "USD");
    assert_eq!(chat_cost.source, CostSource::ModelPricing);
    assert_eq!(chat_cost.pricing_provider.as_deref(), Some("test"));
    assert_eq!(
        chat_cost.pricing_model.as_deref(),
        Some("parity-priced-model")
    );
}

fn chat_text_response(model: &str) -> Json {
    json!({
        "id": "chatcmpl-parity",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    })
}

fn anthropic_text_response(model: &str) -> Json {
    json!({
        "id": "msg_parity",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn"
    })
}

fn responses_text_response(model: &str) -> Json {
    json!({
        "id": "resp_parity",
        "model": model,
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}]
        }]
    })
}
