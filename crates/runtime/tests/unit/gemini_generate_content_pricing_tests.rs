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

fn gemini_test_catalog() -> PricingCatalog {
    PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [{
                "provider": "google",
                "model_id": "gemini-test",
                "pricing_as_of": "2026-10-20",
                "pricing_source": "test",
                "rates": {"input_per_million": 1.0, "output_per_million": 2.0},
                "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
            }]
        })
        .to_string(),
    )
    .unwrap()
}

#[test]
fn test_decode_response_bills_tool_use_prompt_tokens_at_the_input_rate() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![gemini_test_catalog()]))
        .unwrap();
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "Hi"}]},
            "finishReason": "STOP",
            "urlContextMetadata": {"urlMetadata": [{"retrievedUrl": "https://example.test"}]}
        }],
        "modelVersion": "gemini-test",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "toolUsePromptTokenCount": 30,
            "totalTokenCount": 45
        }
    });

    let usage = crate::codec::response_codec(
        nemo_relay_codecs::resolve::ProviderSurface::GeminiGenerateContent,
    )
    .decode_response(&response)
    .unwrap()
    .usage
    .unwrap();

    assert_eq!(usage.prompt_tokens, Some(10));
    let cost = usage.cost.unwrap();
    assert!((cost.input.unwrap() - 0.00004).abs() < 1e-12, "{cost:?}");
    assert!((cost.total.unwrap() - 0.00005).abs() < 1e-12, "{cost:?}");
}

#[test]
fn test_decode_response_leaves_search_grounded_tool_use_prompt_tokens_unbilled() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![gemini_test_catalog()]))
        .unwrap();
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "Hi"}]},
            "finishReason": "STOP",
            "groundingMetadata": {
                "webSearchQueries": ["weather in Rome"],
                "groundingChunks": [{"web": {"uri": "https://example.test", "title": "Rome"}}]
            }
        }],
        "modelVersion": "gemini-test",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "toolUsePromptTokenCount": 30,
            "totalTokenCount": 45
        }
    });

    let decoded = crate::codec::response_codec(
        nemo_relay_codecs::resolve::ProviderSurface::GeminiGenerateContent,
    )
    .decode_response(&response)
    .unwrap();

    let cost = decoded.usage.unwrap().cost.unwrap();
    assert!((cost.input.unwrap() - 0.00001).abs() < 1e-12, "{cost:?}");
    assert!((cost.total.unwrap() - 0.00002).abs() < 1e-12, "{cost:?}");
    assert_eq!(
        serde_json::to_value(decoded.api_specific).unwrap()["tool_use_prompt_tokens"],
        30
    );
}

#[test]
fn test_decode_response_bills_tool_use_prompt_tokens_by_tool_evidence() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![gemini_test_catalog()]))
        .unwrap();
    let text = json!({"text": "Hi"});
    let code = json!({"executableCode": {"language": "PYTHON", "code": "print(1)"}});
    let code_result = json!({"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1"}});
    let url_context = json!({"urlMetadata": [{"retrievedUrl": "https://example.test"}]});
    let file_search_chunk = json!({"retrievedContext": {"fileSearchStore": "fileSearchStores/1"}});
    let vertex_search_chunk = json!({"retrievedContext": {"uri": "gs://bucket/doc"}});
    let unnamed_store_chunk = json!({"retrievedContext": {"fileSearchStore": ""}});

    for (parts, candidate_fields, expected_total) in [
        (json!([code, code_result]), json!({}), Some(0.00005)),
        (
            json!([text]),
            json!({"groundingMetadata": {"groundingChunks": [file_search_chunk]}}),
            Some(0.00005),
        ),
        (
            json!([text]),
            json!({"groundingMetadata": {"imageSearchQueries": ["Rome skyline"]}}),
            Some(0.00002),
        ),
        (
            json!([text]),
            json!({
                "groundingMetadata": {"webSearchQueries": ["weather in Rome"]},
                "urlContextMetadata": url_context
            }),
            None,
        ),
        (
            json!([code, code_result]),
            json!({"groundingMetadata": {"webSearchQueries": ["weather in Rome"]}}),
            None,
        ),
        (
            json!([text]),
            json!({"groundingMetadata": {"groundingChunks": [vertex_search_chunk]}}),
            None,
        ),
        (
            json!([text]),
            json!({"groundingMetadata": {"groundingChunks": [unnamed_store_chunk]}}),
            None,
        ),
        (
            json!([text]),
            json!({"urlContextMetadata": {"urlMetadata": []}}),
            None,
        ),
        (json!([text]), json!({}), None),
    ] {
        let mut candidate = json!({
            "content": {"role": "model", "parts": parts},
            "finishReason": "STOP"
        });
        candidate
            .as_object_mut()
            .unwrap()
            .extend(candidate_fields.as_object().unwrap().clone());
        let response = json!({
            "candidates": [candidate],
            "modelVersion": "gemini-test",
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "toolUsePromptTokenCount": 30,
                "totalTokenCount": 45
            }
        });

        let usage = crate::codec::response_codec(
            nemo_relay_codecs::resolve::ProviderSurface::GeminiGenerateContent,
        )
        .decode_response(&response)
        .unwrap()
        .usage
        .unwrap();

        let total = usage.cost.map(|cost| cost.total.unwrap());
        match (total, expected_total) {
            (Some(total), Some(expected)) => {
                assert!((total - expected).abs() < 1e-12, "{response}: {total}");
            }
            (total, expected) => assert_eq!(total, expected, "{response}"),
        }
    }
}

#[test]
fn test_decode_response_prices_zero_tool_use_prompt_tokens_without_tool_evidence() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![gemini_test_catalog()]))
        .unwrap();
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "Hi"}]},
            "finishReason": "STOP"
        }],
        "modelVersion": "gemini-test",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "toolUsePromptTokenCount": 0,
            "totalTokenCount": 15
        }
    });

    let cost = crate::codec::response_codec(
        nemo_relay_codecs::resolve::ProviderSurface::GeminiGenerateContent,
    )
    .decode_response(&response)
    .unwrap()
    .usage
    .unwrap()
    .cost
    .unwrap();

    assert!((cost.total.unwrap() - 0.00002).abs() < 1e-12, "{cost:?}");
}

#[test]
fn test_decode_response_omits_cost_when_billed_input_tokens_overflow() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![gemini_test_catalog()]))
        .unwrap();
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "Hi"}]},
            "finishReason": "STOP",
            "urlContextMetadata": {"urlMetadata": [{"retrievedUrl": "https://example.test"}]}
        }],
        "modelVersion": "gemini-test",
        "usageMetadata": {
            "promptTokenCount": u64::MAX,
            "candidatesTokenCount": 5,
            "toolUsePromptTokenCount": 1,
            "totalTokenCount": 20
        }
    });

    let usage = crate::codec::response_codec(
        nemo_relay_codecs::resolve::ProviderSurface::GeminiGenerateContent,
    )
    .decode_response(&response)
    .unwrap()
    .usage
    .unwrap();

    assert_eq!(usage.prompt_tokens, Some(u64::MAX));
    assert_eq!(usage.cost, None);
}

#[test]
fn test_decode_response_bills_thoughts_only_usage_at_the_output_rate() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![gemini_test_catalog()]))
        .unwrap();
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": []},
            "finishReason": "STOP"
        }],
        "modelVersion": "gemini-test",
        "usageMetadata": {
            "promptTokenCount": 10,
            "thoughtsTokenCount": 30
        }
    });

    let usage = crate::codec::response_codec(
        nemo_relay_codecs::resolve::ProviderSurface::GeminiGenerateContent,
    )
    .decode_response(&response)
    .unwrap()
    .usage
    .unwrap();

    assert_eq!(usage.completion_tokens, None);
    let total = usage.cost.unwrap().total.unwrap();
    assert!((total - 0.00007).abs() < 1e-12, "{total}");
}

#[test]
fn test_decode_response_omits_cost_when_billed_output_tokens_overflow() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![gemini_test_catalog()]))
        .unwrap();
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "Hi"}]},
            "finishReason": "STOP"
        }],
        "modelVersion": "gemini-test",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": u64::MAX,
            "thoughtsTokenCount": 1,
            "totalTokenCount": 20
        }
    });

    let usage = crate::codec::response_codec(
        nemo_relay_codecs::resolve::ProviderSurface::GeminiGenerateContent,
    )
    .decode_response(&response)
    .unwrap()
    .usage
    .unwrap();

    assert_eq!(usage.completion_tokens, Some(u64::MAX));
    assert_eq!(usage.cost, None);
}
