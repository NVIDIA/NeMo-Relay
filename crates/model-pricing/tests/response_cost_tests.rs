// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog pricing, provider inference, and checked token billing.
use nemo_relay_model_pricing::*;
use nemo_relay_types::Json;
use nemo_relay_types::codec::request::{ContentPart, MessageContent};
use nemo_relay_types::codec::response::*;
use serde_json::{Value, json};
#[path = "support/mod.rs"]
mod test_support;
use test_support::pricing_test_mutex;
struct ResetPricingResolverGuard;
impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

fn url_context_extra() -> serde_json::Map<String, Json> {
    serde_json::from_value(json!({
        "urlContextMetadata": {"urlMetadata": [{"retrievedUrl": "https://example.test"}]}
    }))
    .unwrap()
}

fn full_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: Some("chatcmpl-abc123".into()),
        model: Some("gpt-4".into()),
        message: Some(MessageContent::Text("Hello, world!".into())),
        tool_calls: Some(vec![ResponseToolCall {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: json!({"city": "NYC"}),
        }]),
        finish_reason: Some(FinishReason::Complete),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
            cache_read_tokens: Some(5),
            cache_write_tokens: Some(3),
            uncached_input_tokens: None,
            cost: None,
        }),
        optimization_summary: None,
        api_specific: Some(ApiSpecificResponse::OpenAIChat {
            logprobs: None,
            system_fingerprint: Some("fp_abc123".into()),
            service_tier: Some("default".into()),
        }),
        extra: serde_json::Map::new(),
    }
}

fn minimal_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: None,
        model: None,
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: serde_json::Map::new(),
    }
}

fn pricing_catalog(entries: Value) -> PricingCatalog {
    PricingCatalog::from_json_str(&json!({ "version": 1, "entries": entries }).to_string()).unwrap()
}

fn flat_pricing_entry(
    provider: &str,
    model_id: &str,
    input_per_million: f64,
    output_per_million: f64,
) -> Value {
    json!({
        "provider": provider,
        "model_id": model_id,
        "pricing_as_of": "2026-06-04",
        "pricing_source": format!("https://example.test/{provider}"),
        "rates": {
            "input_per_million": input_per_million,
            "output_per_million": output_per_million
        },
        "prompt_cache": {
            "read_accounting": "separate"
        }
    })
}

fn threshold_pricing_catalog(read_accounting: &str) -> PricingCatalog {
    pricing_catalog(json!([
        {
            "provider": "threshold-ai",
            "model_id": "threshold-model",
            "pricing_as_of": "2026-06-05",
            "pricing_source": "https://example.test/pricing",
            "rate_schedule": {
                "type": "prompt_token_threshold",
                "applies_to": "full_request",
                "tiers": [
                    {
                        "max_prompt_tokens": 200000,
                        "rates": {
                            "input_per_million": 1.0,
                            "output_per_million": 2.0,
                            "cache_read_per_million": 0.1,
                            "cache_write_per_million": 1.25
                        }
                    },
                    {
                        "min_prompt_tokens": 200001,
                        "rates": {
                            "input_per_million": 10.0,
                            "output_per_million": 20.0,
                            "cache_read_per_million": 1.0,
                            "cache_write_per_million": 12.5
                        }
                    }
                ]
            },
            "prompt_cache": {
                "read_accounting": read_accounting
            }
        }
    ]))
}

#[test]
fn test_attach_estimated_cost_uses_event_provider() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let catalog = pricing_catalog(json!([
        flat_pricing_entry("openai", "same-model", 1.0, 2.0),
        flat_pricing_entry("azure/openai", "same-model", 10.0, 20.0)
    ]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
    let _reset_guard = ResetPricingResolverGuard;

    let mut response = AnnotatedLlmResponse {
        model: Some("same-model".into()),
        usage: Some(Usage {
            prompt_tokens: Some(1_000),
            completion_tokens: Some(500),
            ..Usage::default()
        }),
        ..minimal_response()
    };

    attach_estimated_cost_for_provider(&mut response, Some("azure/openai"));

    let cost = response.usage.unwrap().cost.unwrap();
    assert_eq!(cost.total, Some(0.02));
    assert_eq!(cost.pricing_provider.as_deref(), Some("azure/openai"));
}

#[test]
fn test_pricing_public_helpers_and_provider_inference_cover_edge_branches() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([
        {
            "provider": "provider-a",
            "model_id": "priced-model",
            "aliases": ["alias-model"],
            "pricing_as_of": "2026-06-04",
            "pricing_source": "test",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0,
                "cache_read_per_million": 0.25,
                "cache_write_per_million": 3.0
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        }
    ]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog.clone()])).unwrap();
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        cache_read_tokens: Some(100),
        cache_write_tokens: Some(10),
        uncached_input_tokens: None,
        ..Usage::default()
    };

    assert!(catalog.pricing_for_model(" ").is_none());
    assert_eq!(
        pricing_for_model("ALIAS-MODEL").unwrap().model_id,
        "priced-model"
    );
    assert_eq!(
        pricing_for_provider(Some("/provider-a/"), "provider-a/priced-model")
            .unwrap()
            .provider,
        "provider-a"
    );
    assert_eq!(
        estimate_cost("priced-model", &usage).unwrap().total,
        Some(0.001955)
    );
    assert_eq!(
        estimate_cost_for_provider(Some("provider-a"), "priced-model", &usage)
            .unwrap()
            .pricing_provider
            .as_deref(),
        Some("provider-a")
    );
    assert_eq!(
        estimate_cost_with_catalog(&catalog, "priced-model", &usage)
            .unwrap()
            .pricing_model
            .as_deref(),
        Some("priced-model")
    );
    assert_eq!(
        estimate_cost_with_provider(&catalog, Some("provider-a"), "priced-model", &usage)
            .unwrap()
            .pricing_provider
            .as_deref(),
        Some("provider-a")
    );
    assert_eq!(
        infer_model_provider(" OpenAI ", Some("azure/openai/gpt-4o-mini")),
        Some("azure/openai".to_string())
    );
    assert_eq!(
        infer_model_provider(" /OpenAI/ ", Some("gpt-4o-mini")),
        Some("openai".into())
    );
    assert_eq!(infer_model_provider(" / ", None), None);
}

#[test]
fn test_attach_estimated_cost_preserves_existing_or_incomplete_responses() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "configured",
        "configured-model",
        1.0,
        2.0
    )]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let mut with_cost = full_response();
    with_cost.model = Some("configured-model".into());
    with_cost.usage.as_mut().unwrap().cost = Some(CostEstimate {
        total: Some(9.0),
        currency: "USD".into(),
        input: None,
        output: None,
        cache_read: None,
        cache_write: None,
        source: CostSource::ProviderReported,
        pricing_provider: None,
        pricing_model: None,
        pricing_as_of: None,
        pricing_source: None,
    });
    attach_estimated_cost(&mut with_cost);
    assert_eq!(
        with_cost
            .usage
            .as_ref()
            .unwrap()
            .cost
            .as_ref()
            .unwrap()
            .total,
        Some(9.0)
    );

    let mut without_model = full_response();
    without_model.model = None;
    attach_estimated_cost(&mut without_model);
    assert!(without_model.usage.as_ref().unwrap().cost.is_none());

    let mut without_usage = minimal_response();
    without_usage.model = Some("configured-model".into());
    attach_estimated_cost_for_provider(&mut without_usage, Some("configured"));
    assert!(without_usage.usage.is_none());

    let mut priced = full_response();
    priced.model = Some("configured-model".into());
    priced.usage.as_mut().unwrap().cache_read_tokens = None;
    priced.usage.as_mut().unwrap().cache_write_tokens = None;
    attach_estimated_cost_for_provider(&mut priced, Some("configured"));
    assert_eq!(
        priced.usage.as_ref().unwrap().cost.as_ref().unwrap().total,
        Some(0.00005)
    );
}

#[test]
fn test_attach_estimated_cost_bills_gemini_thinking_tokens_as_output() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "configured",
        "configured-model",
        1.0,
        2.0
    )]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let mut response = full_response();
    response.model = Some("configured-model".into());
    let usage = response.usage.as_mut().unwrap();
    usage.cache_read_tokens = None;
    usage.cache_write_tokens = None;
    response.api_specific = Some(ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: Some(30),
        tool_use_prompt_tokens: None,
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: serde_json::Map::new(),
    });
    attach_estimated_cost_for_provider(&mut response, Some("configured"));

    let usage = response.usage.as_ref().unwrap();
    assert_eq!(usage.completion_tokens, Some(20));
    let total = usage.cost.as_ref().unwrap().total.unwrap();
    assert!((total - 0.00011).abs() < 1e-12, "{total}");
}

#[test]
fn test_attach_estimated_cost_bills_gemini_tool_use_prompt_tokens_as_input() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "configured",
        "configured-model",
        1.0,
        2.0
    )]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let mut response = full_response();
    response.model = Some("configured-model".into());
    let usage = response.usage.as_mut().unwrap();
    usage.cache_read_tokens = None;
    usage.cache_write_tokens = None;
    response.api_specific = Some(ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: None,
        tool_use_prompt_tokens: Some(30),
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: url_context_extra(),
    });
    attach_estimated_cost_for_provider(&mut response, Some("configured"));

    let usage = response.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(10));
    let cost = usage.cost.as_ref().unwrap();
    assert!((cost.input.unwrap() - 0.00004).abs() < 1e-12, "{cost:?}");
    assert!((cost.total.unwrap() - 0.00008).abs() < 1e-12, "{cost:?}");
}

#[test]
fn test_attach_estimated_cost_bills_gemini_tool_use_prompt_tokens_with_uncached_input() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([{
        "provider": "configured",
        "model_id": "configured-model",
        "pricing_as_of": "2026-06-04",
        "pricing_source": "https://example.test/configured",
        "rates": {
            "input_per_million": 1.0,
            "output_per_million": 2.0,
            "cache_read_per_million": 0.5
        },
        "prompt_cache": {
            "read_accounting": "included_in_prompt_tokens"
        }
    }]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let mut response = full_response();
    response.model = Some("configured-model".into());
    let usage = response.usage.as_mut().unwrap();
    usage.cache_write_tokens = None;
    usage.uncached_input_tokens = Some(5);
    response.api_specific = Some(ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: None,
        tool_use_prompt_tokens: Some(30),
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: url_context_extra(),
    });
    attach_estimated_cost_for_provider(&mut response, Some("configured"));

    let cost = response.usage.unwrap().cost.unwrap();
    assert!((cost.input.unwrap() - 0.000_035).abs() < 1e-12, "{cost:?}");
    assert!(
        (cost.total.unwrap() - 0.000_077_5).abs() < 1e-12,
        "{cost:?}"
    );
}

#[test]
fn test_attach_estimated_cost_bills_gemini_tool_use_prompt_tokens_without_prompt_tokens() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "configured",
        "configured-model",
        1.0,
        2.0
    )]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let mut response = full_response();
    response.model = Some("configured-model".into());
    response.usage = Some(Usage {
        completion_tokens: Some(20),
        uncached_input_tokens: Some(50),
        ..Usage::default()
    });
    response.api_specific = Some(ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: None,
        tool_use_prompt_tokens: Some(7),
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: url_context_extra(),
    });
    attach_estimated_cost_for_provider(&mut response, Some("configured"));

    let cost = response.usage.unwrap().cost.unwrap();
    assert!((cost.input.unwrap() - 0.000_057).abs() < 1e-12, "{cost:?}");
    assert!((cost.total.unwrap() - 0.000_097).abs() < 1e-12, "{cost:?}");
}

#[test]
fn test_attach_estimated_cost_bills_gemini_tool_use_prompt_tokens_by_tool_evidence() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "configured",
        "configured-model",
        1.0,
        2.0
    )]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
    let web_search = json!({"webSearchQueries": ["weather in Rome"]});
    let code_execution = MessageContent::Parts(vec![ContentPart::ProviderNative {
        provider: "gemini".into(),
        kind: "executableCode".into(),
        value: json!({"executableCode": {"language": "PYTHON", "code": "print(1)"}}),
    }]);
    let adk_code_execution = MessageContent::Parts(vec![ContentPart::ProviderNative {
        provider: "google_adk".into(),
        kind: "executable_code".into(),
        value: json!({"executable_code": {"language": "PYTHON", "code": "print(1)"}}),
    }]);
    let text = MessageContent::Text("Hello, world!".into());

    for (grounding_metadata, extra, message, expected_total) in [
        (
            Some(web_search.clone()),
            serde_json::Map::new(),
            text.clone(),
            Some(0.00005),
        ),
        (
            Some(
                json!({"groundingChunks": [{"maps": {"uri": "https://maps.example", "placeId": "places/1"}}]}),
            ),
            serde_json::Map::new(),
            text.clone(),
            Some(0.00005),
        ),
        (
            Some(json!({"googleMapsWidgetContextToken": "token"})),
            serde_json::Map::new(),
            text.clone(),
            Some(0.00005),
        ),
        (None, url_context_extra(), text.clone(), Some(0.00008)),
        (
            None,
            serde_json::Map::new(),
            code_execution.clone(),
            Some(0.00008),
        ),
        (
            None,
            serde_json::Map::new(),
            adk_code_execution.clone(),
            Some(0.00008),
        ),
        (
            Some(web_search.clone()),
            serde_json::Map::new(),
            adk_code_execution,
            None,
        ),
        (
            Some(web_search.clone()),
            url_context_extra(),
            text.clone(),
            None,
        ),
        (
            Some(web_search),
            serde_json::Map::new(),
            code_execution,
            None,
        ),
        (
            Some(
                json!({"groundingChunks": [{"web": {"uri": "https://example.test", "title": "Rome"}}]}),
            ),
            serde_json::Map::new(),
            text,
            None,
        ),
    ] {
        let mut response = full_response();
        response.model = Some("configured-model".into());
        response.message = Some(message);
        let usage = response.usage.as_mut().unwrap();
        usage.cache_read_tokens = None;
        usage.cache_write_tokens = None;
        response.api_specific = Some(ApiSpecificResponse::GeminiGenerateContent {
            thoughts_tokens: None,
            tool_use_prompt_tokens: Some(30),
            safety_ratings: None,
            grounding_metadata,
            citation_metadata: None,
            extra,
        });
        let case = format!("{:?} {:?}", response.api_specific, response.message);
        attach_estimated_cost_for_provider(&mut response, Some("configured"));

        let total = response.usage.unwrap().cost.map(|cost| cost.total.unwrap());
        match (total, expected_total) {
            (Some(total), Some(expected)) => {
                assert!((total - expected).abs() < 1e-12, "{case}: {total}");
            }
            (total, expected) => assert_eq!(total, expected, "{case}"),
        }
    }
}

#[test]
fn test_attach_estimated_cost_selects_rate_tier_from_reported_prompt_tokens() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![
        threshold_pricing_catalog("separate"),
    ]))
    .unwrap();

    let mut response = full_response();
    response.model = Some("threshold-model".into());
    let usage = response.usage.as_mut().unwrap();
    usage.prompt_tokens = Some(199_990);
    usage.cache_read_tokens = None;
    usage.cache_write_tokens = None;
    response.api_specific = Some(ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: None,
        tool_use_prompt_tokens: Some(20),
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: url_context_extra(),
    });
    attach_estimated_cost_for_provider(&mut response, Some("threshold-ai"));

    let cost = response.usage.unwrap().cost.unwrap();
    assert!((cost.input.unwrap() - 0.20001).abs() < 1e-12, "{cost:?}");
    assert!((cost.total.unwrap() - 0.20005).abs() < 1e-12, "{cost:?}");
}

#[test]
fn test_attach_estimated_cost_omits_cost_when_billed_input_tokens_overflow() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "configured",
        "configured-model",
        1.0,
        2.0
    )]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let mut response = full_response();
    response.model = Some("configured-model".into());
    let usage = response.usage.as_mut().unwrap();
    usage.prompt_tokens = Some(u64::MAX);
    usage.cache_read_tokens = None;
    usage.cache_write_tokens = None;
    response.api_specific = Some(ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: None,
        tool_use_prompt_tokens: Some(1),
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: url_context_extra(),
    });
    attach_estimated_cost_for_provider(&mut response, Some("configured"));

    let usage = response.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(u64::MAX));
    assert_eq!(usage.cost, None);
}

#[test]
fn test_attach_estimated_cost_omits_cost_when_billed_output_tokens_overflow() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let _reset_guard = ResetPricingResolverGuard;
    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "configured",
        "configured-model",
        1.0,
        2.0
    )]));
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let mut response = full_response();
    response.model = Some("configured-model".into());
    let usage = response.usage.as_mut().unwrap();
    usage.completion_tokens = Some(u64::MAX);
    usage.cache_read_tokens = None;
    usage.cache_write_tokens = None;
    response.api_specific = Some(ApiSpecificResponse::GeminiGenerateContent {
        thoughts_tokens: Some(1),
        tool_use_prompt_tokens: None,
        safety_ratings: None,
        grounding_metadata: None,
        citation_metadata: None,
        extra: serde_json::Map::new(),
    });
    attach_estimated_cost_for_provider(&mut response, Some("configured"));

    let usage = response.usage.as_ref().unwrap();
    assert_eq!(usage.completion_tokens, Some(u64::MAX));
    assert_eq!(usage.cost, None);
}
