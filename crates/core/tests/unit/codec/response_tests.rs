// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response in the NeMo Relay core crate.

use super::*;
use serde_json::{Value, json};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use super::super::request::{ContentPart, MessageContent};
use super::super::traits::LlmResponseCodec;
use crate::codec::model_pricing::pricing_test_mutex;
use crate::error::FlowError;
use crate::json::Json;
use crate::plugin::{
    DiagnosticLevel, PluginComponentSpec, PluginConfig, clear_plugin_configuration,
    initialize_plugins, validate_plugin_config,
};
use crate::plugins::model_pricing::register_pricing_component;

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

struct ClearPluginConfigurationGuard;

impl Drop for ClearPluginConfigurationGuard {
    fn drop(&mut self) {
        let _ = clear_plugin_configuration();
    }
}

/// Helper: build a fully-populated AnnotatedLlmResponse.
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

/// Helper: build a minimal AnnotatedLlmResponse (all None + empty extra).
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

fn pricing_catalog_error(entries: Value) -> PricingCatalogError {
    PricingCatalog::from_json_str(&json!({ "version": 1, "entries": entries }).to_string())
        .unwrap_err()
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
                            "cache_read_per_million": 0.1
                        }
                    },
                    {
                        "min_prompt_tokens": 200001,
                        "rates": {
                            "input_per_million": 10.0,
                            "output_per_million": 20.0,
                            "cache_read_per_million": 1.0
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

// -------------------------------------------------------------------
// AnnotatedLlmResponse serialization
// -------------------------------------------------------------------

#[test]
fn test_annotated_llm_response_full_round_trip() {
    let resp = full_response();
    let json_val = serde_json::to_value(&resp).unwrap();
    let deserialized: AnnotatedLlmResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(resp, deserialized);
}

#[test]
fn test_annotated_llm_response_minimal_round_trip() {
    let resp = minimal_response();
    let json_val = serde_json::to_value(&resp).unwrap();
    // Minimal response should serialize to just `{}`
    assert_eq!(json_val, json!({}));
    let deserialized: AnnotatedLlmResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(resp, deserialized);
}

// -------------------------------------------------------------------
// Usage serialization
// -------------------------------------------------------------------

#[test]
fn test_usage_all_none_deserializes_from_empty() {
    let usage: Usage = serde_json::from_value(json!({})).unwrap();
    assert_eq!(usage, Usage::default());
    assert!(usage.prompt_tokens.is_none());
    assert!(usage.completion_tokens.is_none());
    assert!(usage.total_tokens.is_none());
    assert!(usage.cache_read_tokens.is_none());
    assert!(usage.cache_write_tokens.is_none());
}

#[test]
fn test_usage_all_populated_round_trip() {
    let usage = Usage {
        prompt_tokens: Some(100),
        completion_tokens: Some(50),
        total_tokens: Some(150),
        cache_read_tokens: Some(20),
        cache_write_tokens: Some(10),
        cost: None,
    };
    let json_val = serde_json::to_value(&usage).unwrap();
    let deserialized: Usage = serde_json::from_value(json_val).unwrap();
    assert_eq!(usage, deserialized);
}

#[test]
fn test_default_pricing_resolver_has_no_model_prices() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    reset_active_pricing_resolver().unwrap();
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        total_tokens: Some(1_500),
        cache_read_tokens: Some(200),
        cache_write_tokens: None,
        cost: None,
    };

    assert_eq!(estimate_cost("configured-model", &usage), None);
}

#[test]
fn test_configured_model_pricing_estimates_total_cost() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "configured",
            "model_id": "configured-model",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "file:///tmp/pricing.json",
            "rates": {
                "input_per_million": 0.15,
                "output_per_million": 0.60,
                "cache_read_per_million": 0.075
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        }
    ]));
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        total_tokens: Some(1_500),
        cache_read_tokens: Some(200),
        cache_write_tokens: None,
        cost: None,
    };

    let cost = estimate_cost_with_catalog(&catalog, "configured-model", &usage).unwrap();

    assert_eq!(cost.total, Some(0.000_435));
    assert_eq!(cost.currency, "USD");
    assert_eq!(cost.input, Some(0.000_12));
    assert_eq!(cost.output, Some(0.000_3));
    assert_eq!(cost.cache_read, Some(0.000_015));
    assert_eq!(cost.cache_write, None);
    assert_eq!(cost.source, CostSource::ModelPricing);
    assert_eq!(cost.pricing_provider.as_deref(), Some("configured"));
    assert_eq!(cost.pricing_model.as_deref(), Some("configured-model"));
    assert_eq!(cost.pricing_as_of.as_deref(), Some("2026-06-04"));
    assert_eq!(
        cost.pricing_source.as_deref(),
        Some("file:///tmp/pricing.json")
    );
}

#[test]
fn test_pricing_catalog_uses_data_driven_alias_entries() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "configured",
            "model_id": "configured-model",
            "aliases": ["configured-model-2026-06-04"],
            "pricing_as_of": "2026-06-04",
            "pricing_source": "file:///tmp/pricing.json",
            "rates": {
                "input_per_million": 0.15,
                "output_per_million": 0.60
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        }
    ]));
    let pricing = catalog
        .pricing_for_model("CONFIGURED-MODEL-2026-06-04")
        .expect("alias should resolve from configured catalog");

    assert_eq!(pricing.provider, "configured");
    assert_eq!(pricing.model_id, "configured-model");
    assert_eq!(pricing.currency, "USD");
    assert_eq!(pricing.unit, PricingUnit::PerToken);
    assert_eq!(pricing.pricing_as_of, "2026-06-04");
    let rates = pricing.rates.as_ref().unwrap();
    assert_eq!(rates.input_per_million, 0.15);
    assert_eq!(rates.output_per_million, 0.60);
    assert_eq!(
        pricing.prompt_cache.read_accounting,
        CacheReadAccounting::IncludedInPromptTokens
    );
}

#[test]
fn test_pricing_catalog_preserves_currency_and_unit() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "enterprise",
            "model_id": "regional-model",
            "currency": "EUR",
            "unit": "per_token",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "postgres://pricing/model_prices",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0
            },
            "prompt_cache": {
                "read_accounting": "separate"
            }
        }
    ]));
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    let pricing = catalog.pricing_for_model("regional-model").unwrap();
    let cost = estimate_cost_with_catalog(&catalog, "regional-model", &usage).unwrap();

    assert_eq!(pricing.currency, "EUR");
    assert_eq!(pricing.unit, PricingUnit::PerToken);
    assert_eq!(cost.currency, "EUR");
    assert_eq!(cost.total, Some(0.002));
}

#[test]
fn test_non_token_pricing_units_are_representable_but_not_estimated() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "self-hosted",
            "model_id": "nemotron-owned",
            "unit": "gpu_hour",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "internal-owned-fleet-snapshot",
            "prompt_cache": {
                "read_accounting": "separate"
            }
        }
    ]));
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    let pricing = catalog.pricing_for_model("nemotron-owned").unwrap();

    assert_eq!(pricing.unit, PricingUnit::GpuHour);
    assert_eq!(pricing.rates, None);
    assert_eq!(
        estimate_cost_with_catalog(&catalog, "nemotron-owned", &usage),
        None
    );
}

#[test]
fn test_per_token_pricing_requires_token_rates() {
    let err = pricing_catalog_error(json!([
        {
            "provider": "broken",
            "model_id": "missing-rates",
            "unit": "per_token",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "test",
            "prompt_cache": {
                "read_accounting": "separate"
            }
        }
    ]));

    assert!(err.to_string().contains("empty rates or rate_schedule"));
}

#[test]
fn test_pricing_catalog_normalizes_routed_model_names() {
    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "openai",
        "gpt-4o-mini",
        0.15,
        0.6
    )]));

    let azure_pricing = catalog
        .pricing_for_model("azure/openai/gpt-4o-mini")
        .expect("routed provider/model name should resolve");
    let openai_pricing = catalog
        .pricing_for_model("openai/openai/gpt-4o-mini")
        .expect("routed provider/model-owner name should resolve");

    assert_eq!(azure_pricing.provider, "openai");
    assert_eq!(azure_pricing.model_id, "gpt-4o-mini");
    assert_eq!(openai_pricing.provider, "openai");
    assert_eq!(openai_pricing.model_id, "gpt-4o-mini");
}

#[test]
fn test_pricing_resolver_prefers_exact_routed_model_before_suffix_fallback() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "https://example.test/openai",
            "rates": {
                "input_per_million": 0.15,
                "output_per_million": 0.6
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        },
        {
            "provider": "azure-openai",
            "model_id": "azure/openai/gpt-4o-mini",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "https://example.test/azure-openai",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        }
    ]));
    let resolver = PricingResolver::from_catalogs(vec![catalog]);
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    let cost = resolver
        .estimate_cost("azure/openai/gpt-4o-mini", &usage)
        .unwrap();

    assert_eq!(cost.total, Some(0.002));
    assert_eq!(cost.pricing_provider.as_deref(), Some("azure-openai"));
    assert_eq!(
        cost.pricing_model.as_deref(),
        Some("azure/openai/gpt-4o-mini")
    );
}

#[test]
fn test_pricing_catalog_allows_same_model_id_for_distinct_providers() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "https://example.test/openai",
            "rates": {
                "input_per_million": 0.15,
                "output_per_million": 0.6
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        },
        {
            "provider": "azure/openai",
            "model_id": "gpt-4o-mini",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "https://example.test/azure-openai",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        }
    ]));
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    let openai = estimate_cost_with_provider(&catalog, Some("openai"), "gpt-4o-mini", &usage)
        .expect("openai provider price should resolve");
    let azure = estimate_cost_with_provider(&catalog, Some("azure/openai"), "gpt-4o-mini", &usage)
        .expect("azure/openai provider price should resolve");

    assert_eq!(openai.total, Some(0.000_45));
    assert_eq!(openai.pricing_provider.as_deref(), Some("openai"));
    assert_eq!(azure.total, Some(0.002));
    assert_eq!(azure.pricing_provider.as_deref(), Some("azure/openai"));
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
fn test_custom_pricing_catalog_supports_future_models_without_code_changes() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "future-ai",
            "model_id": "future-model",
            "aliases": ["future-model-latest"],
            "pricing_as_of": "2026-06-04",
            "pricing_source": "https://example.test/pricing",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0,
                "cache_read_per_million": 0.25,
                "cache_write_per_million": 1.5
            },
            "prompt_cache": {
                "read_accounting": "separate"
            }
        }
    ]));
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(2_000),
        cache_read_tokens: Some(3_000),
        cache_write_tokens: Some(4_000),
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "future-model-latest", &usage).unwrap();

    assert_eq!(cost.total, Some(0.011_75));
    assert_eq!(cost.input, Some(0.001));
    assert_eq!(cost.output, Some(0.004));
    assert_eq!(cost.cache_read, Some(0.000_75));
    assert_eq!(cost.cache_write, Some(0.006));
    assert_eq!(cost.pricing_provider.as_deref(), Some("future-ai"));
    assert_eq!(cost.pricing_model.as_deref(), Some("future-model"));
}

#[test]
fn test_prompt_threshold_pricing_applies_selected_tier_to_full_request() {
    let catalog = threshold_pricing_catalog("included_in_prompt_tokens");
    let usage = Usage {
        prompt_tokens: Some(200_001),
        completion_tokens: Some(1_000),
        cache_read_tokens: Some(1_000),
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "threshold-model", &usage).unwrap();

    assert_eq!(cost.input, Some(1.990_01));
    assert_eq!(cost.output, Some(0.02));
    assert_eq!(cost.cache_read, Some(0.001));
    assert_eq!(cost.total, Some(2.011_01));
}

#[test]
fn test_prompt_threshold_pricing_uses_lower_tier_at_boundary() {
    let catalog = threshold_pricing_catalog("separate");
    let usage = Usage {
        prompt_tokens: Some(200_000),
        completion_tokens: Some(1_000),
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "threshold-model", &usage).unwrap();

    assert_eq!(cost.input, Some(0.2));
    assert_eq!(cost.output, Some(0.002));
    assert_eq!(cost.total, Some(0.202));
}

#[test]
fn test_prompt_threshold_pricing_requires_prompt_tokens() {
    let catalog = threshold_pricing_catalog("separate");
    let usage = Usage {
        completion_tokens: Some(1_000),
        ..Usage::default()
    };

    assert!(estimate_cost_with_catalog(&catalog, "threshold-model", &usage).is_none());
}

#[test]
fn test_pricing_resolver_uses_first_matching_source() {
    let override_catalog = pricing_catalog(json!([
        {
            "provider": "local-override",
            "model_id": "gpt-4o-mini",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "file:///tmp/local-pricing.json",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0,
                "cache_read_per_million": 0.5
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        }
    ]));
    let resolver = PricingResolver::from_catalogs(vec![override_catalog]);
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        cache_read_tokens: Some(200),
        ..Usage::default()
    };

    let cost = resolver.estimate_cost("gpt-4o-mini", &usage).unwrap();

    assert_eq!(cost.total, Some(0.001_9));
    assert_eq!(cost.pricing_provider.as_deref(), Some("local-override"));
    assert!(resolver.estimate_cost("missing-model", &usage).is_none());
}

#[test]
fn test_pricing_resolver_loads_inline_and_file_sources_in_order() {
    let temp = std::env::temp_dir().join(format!(
        "nemo-relay-pricing-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&temp).unwrap();
    let file_path = temp.join("pricing.json");
    fs::write(
        &file_path,
        json!({
            "version": 1,
            "entries": [flat_pricing_entry("global-file", "file-model", 4.0, 8.0)]
        })
        .to_string(),
    )
    .unwrap();
    let inline_catalog = pricing_catalog(json!([flat_pricing_entry(
        "project-inline",
        "inline-model",
        1.0,
        2.0
    )]));
    let config = PricingConfig {
        sources: vec![
            PricingSourceConfig::Inline {
                catalog: inline_catalog,
            },
            PricingSourceConfig::File { path: file_path },
        ],
    };
    let resolver = PricingResolver::from_config(&config).unwrap();
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    let inline = resolver.estimate_cost("inline-model", &usage).unwrap();
    let file = resolver.estimate_cost("file-model", &usage).unwrap();

    assert_eq!(inline.total, Some(0.002));
    assert_eq!(inline.pricing_provider.as_deref(), Some("project-inline"));
    assert_eq!(file.total, Some(0.008));
    assert_eq!(file.pricing_provider.as_deref(), Some("global-file"));
    assert!(resolver.estimate_cost("gpt-4o-mini", &usage).is_none());
    fs::remove_dir_all(temp).unwrap();
}

#[test]
fn test_pricing_resolver_validates_inline_catalogs() {
    let config = PricingConfig {
        sources: vec![PricingSourceConfig::Inline {
            catalog: PricingCatalog {
                version: 2,
                entries: vec![],
            },
        }],
    };

    let err = PricingResolver::from_config(&config).unwrap_err();

    assert!(
        err.to_string()
            .contains("unsupported model pricing catalog version 2")
    );
}

#[test]
fn test_pricing_resolver_accepts_custom_database_backed_sources() {
    struct TestDatabasePricingSource {
        catalog: PricingCatalog,
    }

    impl PricingSource for TestDatabasePricingSource {
        fn source_name(&self) -> &str {
            "test-db"
        }

        fn load_catalog(&self) -> Result<Option<PricingCatalog>, PricingCatalogError> {
            Ok(Some(self.catalog.clone()))
        }
    }

    let catalog = pricing_catalog(json!([flat_pricing_entry(
        "database", "db-model", 10.0, 20.0
    )]));
    let resolver =
        PricingResolver::from_sources(vec![Box::new(TestDatabasePricingSource { catalog })])
            .unwrap();
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    let cost = resolver.estimate_cost("db-model", &usage).unwrap();

    assert_eq!(cost.total, Some(0.02));
    assert_eq!(cost.pricing_provider.as_deref(), Some("database"));
}

#[test]
fn test_pricing_resolver_validates_custom_source_catalogs() {
    struct InvalidDatabasePricingSource;

    impl PricingSource for InvalidDatabasePricingSource {
        fn source_name(&self) -> &str {
            "invalid-test-db"
        }

        fn load_catalog(&self) -> Result<Option<PricingCatalog>, PricingCatalogError> {
            Ok(Some(PricingCatalog {
                version: 2,
                entries: vec![],
            }))
        }
    }

    let err =
        PricingResolver::from_sources(vec![Box::new(InvalidDatabasePricingSource)]).unwrap_err();

    assert!(
        err.to_string()
            .contains("unsupported model pricing catalog version 2")
    );
}

#[test]
fn test_pricing_plugin_configures_process_resolver_and_clears_to_default() {
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
    let _runtime_guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap();
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    let mut component = PluginComponentSpec::new("pricing");
    component.config = serde_json::from_value(json!({
        "sources": [
            {
                "type": "inline",
                "catalog": {
                    "version": 1,
                    "entries": [
                        {
                            "provider": "plugin-inline",
                            "model_id": "plugin-model",
                            "pricing_as_of": "2026-06-04",
                            "pricing_source": "plugins.toml",
                            "rates": {
                                "input_per_million": 1.0,
                                "output_per_million": 2.0
                            },
                            "prompt_cache": {
                                "read_accounting": "separate"
                            }
                        }
                    ]
                }
            }
        ]
    }))
    .unwrap();
    let mut config = PluginConfig::default();
    config.components.push(component);

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { initialize_plugins(config).await.unwrap() });
    let _clear_guard = ClearPluginConfigurationGuard;
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    let configured = estimate_cost("plugin-model", &usage).unwrap();
    assert_eq!(configured.total, Some(0.002));
    assert!(estimate_cost("gpt-4o-mini", &usage).is_none());

    clear_plugin_configuration().unwrap();

    assert!(estimate_cost("plugin-model", &usage).is_none());
    assert!(estimate_cost("gpt-4o-mini", &usage).is_none());
}

#[test]
fn test_pricing_plugin_validation_reports_invalid_json_and_catalog_errors() {
    let _runtime_guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap();
    register_pricing_component().unwrap();
    register_pricing_component().unwrap();

    let mut malformed = PluginComponentSpec::new("pricing");
    malformed.config = serde_json::from_value(json!({
        "sources": [],
        "unexpected": true
    }))
    .unwrap();
    let report = validate_plugin_config(&PluginConfig {
        components: vec![malformed],
        ..PluginConfig::default()
    });
    assert!(report.has_errors());
    assert_eq!(report.diagnostics[0].level, DiagnosticLevel::Error);
    assert_eq!(report.diagnostics[0].code, "pricing.invalid_config");
    assert!(
        report.diagnostics[0]
            .message
            .contains("unknown field `unexpected`")
    );

    let mut invalid_catalog = PluginComponentSpec::new("pricing");
    invalid_catalog.config = serde_json::from_value(json!({
        "sources": [
            {
                "type": "inline",
                "catalog": {
                    "version": 2,
                    "entries": []
                }
            }
        ]
    }))
    .unwrap();
    let report = validate_plugin_config(&PluginConfig {
        components: vec![invalid_catalog],
        ..PluginConfig::default()
    });
    assert!(report.has_errors());
    assert!(
        report.diagnostics[0]
            .message
            .contains("unsupported model pricing catalog version 2")
    );

    let duplicate = validate_plugin_config(&PluginConfig {
        components: vec![
            PluginComponentSpec::new("pricing"),
            PluginComponentSpec::new("pricing"),
        ],
        ..PluginConfig::default()
    });
    assert!(duplicate.has_errors());
    assert!(
        duplicate
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "plugin.duplicate_component")
    );
}

#[test]
fn test_pricing_catalog_rejects_duplicate_model_aliases() {
    let err = pricing_catalog_error(json!([
        flat_pricing_entry("a", "same-model", 1.0, 1.0),
        {
            "provider": "a",
            "model_id": "other-model",
            "aliases": ["SAME-MODEL"],
            "pricing_as_of": "2026-06-04",
            "pricing_source": "https://example.test/b",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 1.0
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        }
    ]));

    assert!(
        err.to_string()
            .contains("duplicate model pricing alias 'a/same-model'")
    );
}

#[test]
fn test_pricing_catalog_rejects_unsupported_schema_version() {
    let err = PricingCatalog::from_json_str(
        r#"{
            "version": 2,
            "entries": []
        }"#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("unsupported model pricing catalog version 2")
    );
}

#[test]
fn test_pricing_catalog_rejects_empty_required_fields_and_invalid_rates() {
    for (field, expected) in [
        ("provider", "empty provider"),
        ("model_id", "empty model_id"),
        ("currency", "empty currency"),
        ("pricing_as_of", "empty pricing_as_of"),
        ("pricing_source", "empty pricing_source"),
    ] {
        let mut entry = flat_pricing_entry("configured", "configured-model", 1.0, 2.0);
        entry[field] = json!(" ");

        let err = pricing_catalog_error(json!([entry]));

        assert!(
            err.to_string().contains(expected),
            "expected {expected}, got {err}"
        );
    }

    for (field, expected) in [
        ("input_per_million", "rates.input_per_million"),
        ("output_per_million", "rates.output_per_million"),
        ("cache_read_per_million", "rates.cache_read_per_million"),
        ("cache_write_per_million", "rates.cache_write_per_million"),
    ] {
        let mut entry = flat_pricing_entry("configured", "configured-model", 1.0, 2.0);
        entry["rates"][field] = json!(-0.1);

        let err = pricing_catalog_error(json!([entry]));

        assert!(
            err.to_string().contains(expected),
            "expected {expected}, got {err}"
        );
    }
}

#[test]
fn test_pricing_catalog_rejects_invalid_rate_schedules() {
    let empty_tiers = pricing_catalog_error(json!([
        {
            "provider": "configured",
            "model_id": "configured-model",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "test",
            "rate_schedule": {
                "type": "prompt_token_threshold",
                "tiers": []
            },
            "prompt_cache": {
                "read_accounting": "separate"
            }
        }
    ]));
    assert!(empty_tiers.to_string().contains("rate_schedule.tiers"));

    let reversed_bounds = pricing_catalog_error(json!([
        {
            "provider": "configured",
            "model_id": "configured-model",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "test",
            "rate_schedule": {
                "type": "prompt_token_threshold",
                "tiers": [
                    {
                        "min_prompt_tokens": 20,
                        "max_prompt_tokens": 10,
                        "rates": {
                            "input_per_million": 1.0,
                            "output_per_million": 2.0
                        }
                    }
                ]
            },
            "prompt_cache": {
                "read_accounting": "separate"
            }
        }
    ]));
    assert!(
        reversed_bounds
            .to_string()
            .contains("rate_schedule.tiers.prompt_tokens")
    );

    let invalid_tier_rate = pricing_catalog_error(json!([
        {
            "provider": "configured",
            "model_id": "configured-model",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "test",
            "rate_schedule": {
                "type": "prompt_token_threshold",
                "tiers": [
                    {
                        "rates": {
                            "input_per_million": -1.0,
                            "output_per_million": 2.0
                        }
                    }
                ]
            },
            "prompt_cache": {
                "read_accounting": "separate"
            }
        }
    ]));
    assert!(
        invalid_tier_rate
            .to_string()
            .contains("rate_schedule.tiers[0].rates.input_per_million")
    );
}

#[test]
fn test_pricing_resolver_file_and_source_error_branches() {
    let missing_path = std::env::temp_dir().join(format!(
        "nemo-relay-missing-pricing-{}-{}.json",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let err = PricingResolver::from_config(&PricingConfig {
        sources: vec![PricingSourceConfig::File {
            path: missing_path.clone(),
        }],
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("could not read model pricing catalog file")
    );
    assert!(
        err.to_string()
            .contains(&missing_path.display().to_string())
    );

    struct EmptyPricingSource;

    impl PricingSource for EmptyPricingSource {
        fn source_name(&self) -> &str {
            "empty"
        }

        fn load_catalog(&self) -> Result<Option<PricingCatalog>, PricingCatalogError> {
            Ok(None)
        }
    }

    let source = EmptyPricingSource;
    assert_eq!(source.source_name(), "empty");
    let resolver = PricingResolver::from_sources(vec![Box::new(source)]).unwrap();
    assert!(resolver.pricing_for_model("anything").is_none());

    struct ErrorPricingSource;

    impl PricingSource for ErrorPricingSource {
        fn source_name(&self) -> &str {
            "error"
        }

        fn load_catalog(&self) -> Result<Option<PricingCatalog>, PricingCatalogError> {
            Err(PricingCatalogError::UnsupportedVersion { version: 99 })
        }
    }

    let err = PricingResolver::from_sources(vec![Box::new(ErrorPricingSource)]).unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported model pricing catalog version 99")
    );
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
    attach_estimated_cost_for_provider(&mut priced, Some("configured"));
    assert_eq!(
        priced.usage.as_ref().unwrap().cost.as_ref().unwrap().total,
        Some(0.00005)
    );
}

#[test]
fn test_missing_token_pricing_returns_none_without_fabricating_zero_cost() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    reset_active_pricing_resolver().unwrap();
    assert_eq!(estimate_cost("gpt-4o-mini", &Usage::default()), None);
    assert_eq!(estimate_cost("gpt-4o-mini", &Usage::default()), None);
}

#[test]
fn test_provider_reported_cost_sums_components_and_defaults_currency() {
    let cost = provider_reported_cost(
        None,
        Some(RawUsageCost {
            input: Some(0.12),
            output: Some(0.30),
            cache_read: Some(0.01),
            cache_write: Some(0.02),
            ..RawUsageCost::default()
        }),
    )
    .expect("component-only provider cost should be retained");

    assert_eq!(cost.total, Some(0.45));
    assert_eq!(cost.currency, "USD");
    assert_eq!(cost.source, CostSource::ProviderReported);
}

#[test]
fn test_provider_reported_cost_keeps_top_level_provider_usd_currency() {
    let cost = provider_reported_cost(
        Some(0.42),
        Some(RawUsageCost {
            currency: Some("EUR".to_string()),
            input: Some(0.10),
            output: Some(0.20),
            cache_read: Some(0.01),
            cache_write: Some(0.02),
            ..RawUsageCost::default()
        }),
    )
    .expect("top-level provider USD cost should be retained");

    assert_eq!(cost.total, Some(0.42));
    assert_eq!(cost.currency, "USD");
    assert_eq!(cost.input, None);
    assert_eq!(cost.output, None);
    assert_eq!(cost.cache_read, None);
    assert_eq!(cost.cache_write, None);
    assert_eq!(cost.total_for_currency("USD"), Some(0.42));
}

#[test]
fn test_provider_reported_cost_keeps_usd_components_with_top_level_usd_cost() {
    let cost = provider_reported_cost(
        Some(0.42),
        Some(RawUsageCost {
            currency: Some("usd".to_string()),
            input: Some(0.10),
            output: Some(0.20),
            cache_read: Some(0.01),
            cache_write: Some(0.02),
            ..RawUsageCost::default()
        }),
    )
    .expect("top-level and nested USD cost fields should be retained");

    assert_eq!(cost.total, Some(0.42));
    assert_eq!(cost.currency, "USD");
    assert_eq!(cost.input, Some(0.10));
    assert_eq!(cost.output, Some(0.20));
    assert_eq!(cost.cache_read, Some(0.01));
    assert_eq!(cost.cache_write, Some(0.02));
}

#[test]
fn test_cost_estimate_total_helpers_sum_components_and_match_currency() {
    let component_cost = CostEstimate {
        total: None,
        currency: "usd".to_string(),
        input: Some(0.12),
        output: Some(0.30),
        cache_read: Some(0.01),
        cache_write: Some(0.02),
        source: CostSource::ProviderReported,
        pricing_provider: None,
        pricing_model: None,
        pricing_as_of: None,
        pricing_source: None,
    };

    assert_eq!(component_cost.total_or_component_sum(), Some(0.45));
    assert_eq!(
        component_cost.total_or_component_sum_for_currency("USD"),
        Some(0.45)
    );
    assert_eq!(component_cost.total_for_currency("USD"), None);
    assert_eq!(
        component_cost.total_or_component_sum_for_currency("EUR"),
        None
    );

    let empty_cost = CostEstimate {
        total: None,
        currency: "USD".to_string(),
        input: None,
        output: None,
        cache_read: None,
        cache_write: None,
        source: CostSource::ProviderReported,
        pricing_provider: None,
        pricing_model: None,
        pricing_as_of: None,
        pricing_source: None,
    };
    assert_eq!(empty_cost.total_or_component_sum(), None);
}

#[test]
fn test_usage_cost_round_trip_preserves_model_pricing_codec_compatibility() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "configured",
            "model_id": "configured-model",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "file:///tmp/pricing.json",
            "rates": {
                "input_per_million": 0.15,
                "output_per_million": 0.60,
                "cache_read_per_million": 0.075
            },
            "prompt_cache": {
                "read_accounting": "included_in_prompt_tokens"
            }
        }
    ]));
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        total_tokens: Some(1_500),
        cache_read_tokens: Some(200),
        cache_write_tokens: None,
        cost: estimate_cost_with_catalog(
            &catalog,
            "configured-model",
            &Usage {
                prompt_tokens: Some(1_000),
                completion_tokens: Some(500),
                total_tokens: Some(1_500),
                cache_read_tokens: Some(200),
                cache_write_tokens: None,
                cost: None,
            },
        ),
    };

    let json_val = serde_json::to_value(&usage).unwrap();

    assert_eq!(json_val["cost"]["total"], json!(0.000_435));
    assert_eq!(json_val["cost"]["source"], json!("model_pricing"));
    assert_eq!(json_val["cost"]["pricing_as_of"], json!("2026-06-04"));
    let deserialized: Usage = serde_json::from_value(json_val).unwrap();
    assert_eq!(usage, deserialized);
}

#[test]
fn test_usage_cost_round_trip_preserves_provider_reported_codec_compatibility() {
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        total_tokens: Some(1_500),
        cache_read_tokens: Some(200),
        cache_write_tokens: None,
        cost: Some(CostEstimate {
            total: Some(0.42),
            currency: "USD".into(),
            input: Some(0.12),
            output: Some(0.30),
            cache_read: None,
            cache_write: None,
            source: CostSource::ProviderReported,
            pricing_provider: None,
            pricing_model: None,
            pricing_as_of: None,
            pricing_source: None,
        }),
    };

    let json_val = serde_json::to_value(&usage).unwrap();

    assert_eq!(json_val["cost"]["total"], json!(0.42));
    assert_eq!(json_val["cost"]["source"], json!("provider_reported"));
    let deserialized: Usage = serde_json::from_value(json_val).unwrap();
    assert_eq!(usage, deserialized);
}

#[test]
fn test_unknown_model_pricing_returns_none_without_blocking_usage() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    reset_active_pricing_resolver().unwrap();
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(500),
        ..Usage::default()
    };

    assert_eq!(estimate_cost("unknown-model", &usage), None);
    assert_eq!(usage.prompt_tokens, Some(1_000));
}

#[test]
fn test_usage_ignores_unmodeled_provider_subfields() {
    let usage: Usage = serde_json::from_value(json!({
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "some_future_field": 99
    }))
    .unwrap();
    assert_eq!(usage.prompt_tokens, Some(5));
    assert_eq!(usage.completion_tokens, Some(7));
    let reserialized = serde_json::to_value(&usage).unwrap();
    assert!(reserialized.get("some_future_field").is_none());
}

// -------------------------------------------------------------------
// FinishReason serialization
// -------------------------------------------------------------------

#[test]
fn test_finish_reason_complete_serializes_to_complete() {
    let reason = FinishReason::Complete;
    let json_val = serde_json::to_value(&reason).unwrap();
    assert_eq!(json_val, json!("complete"));
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::Complete);
}

#[test]
fn test_finish_reason_length_round_trip() {
    let reason = FinishReason::Length;
    let json_val = serde_json::to_value(&reason).unwrap();
    assert_eq!(json_val, json!("length"));
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::Length);
}

#[test]
fn test_finish_reason_tool_use_round_trip() {
    let reason = FinishReason::ToolUse;
    let json_val = serde_json::to_value(&reason).unwrap();
    assert_eq!(json_val, json!("tool_use"));
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::ToolUse);
}

#[test]
fn test_finish_reason_content_filter_round_trip() {
    let reason = FinishReason::ContentFilter;
    let json_val = serde_json::to_value(&reason).unwrap();
    assert_eq!(json_val, json!("content_filter"));
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::ContentFilter);
}

#[test]
fn test_finish_reason_unknown_round_trip() {
    let reason = FinishReason::Unknown("custom_reason".into());
    let json_val = serde_json::to_value(&reason).unwrap();
    let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
    assert_eq!(deserialized, FinishReason::Unknown("custom_reason".into()));
}

// -------------------------------------------------------------------
// ResponseToolCall serialization
// -------------------------------------------------------------------

#[test]
fn test_response_tool_call_json_arguments_round_trip() {
    let tc = ResponseToolCall {
        id: "call_abc".into(),
        name: "search".into(),
        arguments: json!({"query": "weather", "limit": 5}),
    };
    let json_val = serde_json::to_value(&tc).unwrap();
    assert_eq!(json_val["arguments"]["query"], json!("weather"));
    assert_eq!(json_val["arguments"]["limit"], json!(5));
    let deserialized: ResponseToolCall = serde_json::from_value(json_val).unwrap();
    assert_eq!(tc, deserialized);
}

// -------------------------------------------------------------------
// ApiSpecificResponse serialization
// -------------------------------------------------------------------

#[test]
fn test_api_specific_openai_chat_round_trip() {
    let api = ApiSpecificResponse::OpenAIChat {
        logprobs: Some(json!({"content": []})),
        system_fingerprint: Some("fp_abc".into()),
        service_tier: Some("default".into()),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("openai_chat"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_api_specific_openai_responses_round_trip() {
    let api = ApiSpecificResponse::OpenAIResponses {
        output_items: Some(vec![json!({"type": "message", "content": []})]),
        status: Some("completed".into()),
        incomplete_details: None,
        previous_response_id: None,
        store: None,
        service_tier: None,
        truncation: None,
        reasoning: None,
        input_tokens_details: None,
        output_tokens_details: None,
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("openai_responses"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_api_specific_anthropic_messages_round_trip() {
    let api = ApiSpecificResponse::AnthropicMessages {
        object_type: Some("message".into()),
        role: Some("assistant".into()),
        stop_reason: Some("end_turn".into()),
        stop_sequence: Some("\n\nHuman:".into()),
        service_tier: Some("default".into()),
        container: Some(json!({"id":"container_123"})),
        content_blocks: Some(vec![json!({"type": "text", "text": "Hello"})]),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("anthropic_messages"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

#[test]
fn test_api_specific_custom_round_trip() {
    let api = ApiSpecificResponse::Custom {
        api_name: "my_custom_llm".into(),
        data: json!({"version": "2.0", "extra_field": true}),
    };
    let json_val = serde_json::to_value(&api).unwrap();
    assert_eq!(json_val["api"], json!("custom"));
    assert_eq!(json_val["api_name"], json!("my_custom_llm"));
    let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(api, deserialized);
}

// -------------------------------------------------------------------
// Helper: response_text()
// -------------------------------------------------------------------

#[test]
fn test_response_text_returns_some_for_text_content() {
    let resp = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("Hello!".into())),
        ..minimal_response()
    };
    assert_eq!(resp.response_text(), Some("Hello!"));
}

#[test]
fn test_response_text_returns_none_when_message_is_none() {
    let resp = minimal_response();
    assert_eq!(resp.response_text(), None);
}

#[test]
fn test_response_text_extracts_first_text_from_parts() {
    let resp = AnnotatedLlmResponse {
        message: Some(MessageContent::Parts(vec![ContentPart::Text {
            text: "Part text".into(),
            extra: serde_json::Map::new(),
        }])),
        ..minimal_response()
    };
    assert_eq!(resp.response_text(), Some("Part text"));
}

// -------------------------------------------------------------------
// Helper: has_tool_calls()
// -------------------------------------------------------------------

#[test]
fn test_has_tool_calls_true_when_present() {
    let resp = AnnotatedLlmResponse {
        tool_calls: Some(vec![ResponseToolCall {
            id: "tc_1".into(),
            name: "search".into(),
            arguments: json!({}),
        }]),
        ..minimal_response()
    };
    assert!(resp.has_tool_calls());
}

#[test]
fn test_has_tool_calls_false_when_none() {
    let resp = minimal_response();
    assert!(!resp.has_tool_calls());
}

#[test]
fn test_has_tool_calls_false_when_empty_vec() {
    let resp = AnnotatedLlmResponse {
        tool_calls: Some(vec![]),
        ..minimal_response()
    };
    assert!(!resp.has_tool_calls());
}

// -------------------------------------------------------------------
// Helper: FinishReason::is_complete()
// -------------------------------------------------------------------

#[test]
fn test_is_complete_true_for_complete() {
    assert!(FinishReason::Complete.is_complete());
}

#[test]
fn test_is_complete_false_for_other_variants() {
    assert!(!FinishReason::Length.is_complete());
    assert!(!FinishReason::ToolUse.is_complete());
    assert!(!FinishReason::ContentFilter.is_complete());
    assert!(!FinishReason::Unknown("other".into()).is_complete());
}

// -------------------------------------------------------------------
// LlmResponseCodec trait: mock implementation
// -------------------------------------------------------------------

struct MockResponseCodec;

impl LlmResponseCodec for MockResponseCodec {
    fn decode_response(&self, _response: &Json) -> crate::error::Result<AnnotatedLlmResponse> {
        Ok(AnnotatedLlmResponse {
            id: Some("mock-id".into()),
            model: Some("mock-model".into()),
            message: Some(MessageContent::Text("mock response".into())),
            tool_calls: None,
            finish_reason: Some(FinishReason::Complete),
            usage: None,
            optimization_summary: None,
            api_specific: None,
            extra: serde_json::Map::new(),
        })
    }
}

#[test]
fn test_mock_response_codec_compiles_and_returns_ok() {
    let codec = MockResponseCodec;
    let result = codec.decode_response(&json!({}));
    assert!(result.is_ok());
    let resp = result.unwrap();
    assert_eq!(resp.id, Some("mock-id".into()));
    assert_eq!(resp.model, Some("mock-model".into()));
}

struct FailingMockResponseCodec;

impl LlmResponseCodec for FailingMockResponseCodec {
    fn decode_response(&self, _response: &Json) -> crate::error::Result<AnnotatedLlmResponse> {
        Err(FlowError::Internal("decode failed".into()))
    }
}

#[test]
fn test_failing_mock_codec_demonstrates_non_fatal_pattern() {
    let codec = FailingMockResponseCodec;
    let result = codec.decode_response(&json!({"choices": []}));
    assert!(result.is_err());

    // Non-fatal pattern: callers use .ok() to convert Err to None
    let annotated: Option<AnnotatedLlmResponse> =
        codec.decode_response(&json!({"choices": []})).ok();
    assert!(annotated.is_none());
}

// -------------------------------------------------------------------
// Extra field (flatten) captures unmodeled keys
// -------------------------------------------------------------------

#[test]
fn test_annotated_llm_response_extra_captures_unmodeled_keys() {
    let json_val = json!({
        "id": "test-123",
        "model": "gpt-4",
        "custom_field": "custom_value",
        "another_field": 42
    });
    let resp: AnnotatedLlmResponse = serde_json::from_value(json_val).unwrap();
    assert_eq!(resp.id, Some("test-123".into()));
    assert_eq!(resp.model, Some("gpt-4".into()));
    assert_eq!(resp.extra.get("custom_field"), Some(&json!("custom_value")));
    assert_eq!(resp.extra.get("another_field"), Some(&json!(42)));

    // Round-trip: extra fields should appear as top-level keys
    let serialized = serde_json::to_value(&resp).unwrap();
    assert_eq!(serialized["custom_field"], json!("custom_value"));
    assert_eq!(serialized["another_field"], json!(42));
}
