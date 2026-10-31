// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::{Value, json};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

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
        cache_write_1h_tokens: None,
        uncached_input_tokens: None,
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
        cache_write_1h_tokens: None,
        uncached_input_tokens: None,
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
                "cache_write_per_million": 1.5,
                "cache_write_1h_per_million": 2.4
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
        cache_write_1h_tokens: Some(1_000),
        uncached_input_tokens: None,
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "future-model-latest", &usage).unwrap();

    assert_eq!(cost.total, Some(0.012_65));
    assert_eq!(cost.input, Some(0.001));
    assert_eq!(cost.output, Some(0.004));
    assert_eq!(cost.cache_read, Some(0.000_75));
    assert_eq!(cost.cache_write, Some(0.0069));
    assert_eq!(cost.pricing_provider.as_deref(), Some("future-ai"));
    assert_eq!(cost.pricing_model.as_deref(), Some("future-model"));
}

#[test]
fn test_model_pricing_omits_total_when_nonzero_usage_lacks_a_rate() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "future-ai",
            "model_id": "future-model",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "https://example.test/pricing",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0,
                "cache_read_per_million": 0.25
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
        uncached_input_tokens: None,
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "future-model", &usage).unwrap();

    assert_eq!(cost.total, None);
    assert_eq!(cost.input, Some(0.001));
    assert_eq!(cost.output, Some(0.004));
    assert_eq!(cost.cache_read, Some(0.000_75));
    assert_eq!(cost.cache_write, None);
    assert_eq!(cost.source, CostSource::ModelPricing);
}

#[test]
fn test_model_pricing_omits_total_when_nonzero_cache_read_usage_lacks_a_rate() {
    let catalog = pricing_catalog(json!([
        {
            "provider": "future-ai",
            "model_id": "future-model",
            "pricing_as_of": "2026-06-04",
            "pricing_source": "https://example.test/pricing",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0,
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
        uncached_input_tokens: None,
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "future-model", &usage).unwrap();

    assert_eq!(cost.total, None);
    assert_eq!(cost.input, Some(0.001));
    assert_eq!(cost.output, Some(0.004));
    assert_eq!(cost.cache_read, None);
    assert_eq!(cost.cache_write, Some(0.006));
    assert_eq!(cost.source, CostSource::ModelPricing);
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
fn test_split_usage_prices_the_same_under_either_read_accounting() {
    let inclusive = Usage {
        prompt_tokens: Some(100),
        completion_tokens: Some(20),
        cache_read_tokens: Some(40),
        cache_write_tokens: Some(10),
        uncached_input_tokens: Some(50),
        ..Usage::default()
    };
    let separate = Usage {
        prompt_tokens: Some(50),
        ..inclusive.clone()
    };

    for read_accounting in ["included_in_prompt_tokens", "separate"] {
        let catalog = cache_pricing_catalog(read_accounting);
        for usage in [&inclusive, &separate] {
            let cost = estimate_cost_with_catalog(&catalog, "cache-model", usage).unwrap();

            assert_eq!(cost.input, Some(0.000_15), "{read_accounting} {usage:?}");
            assert_eq!(cost.cache_read, Some(0.000_012), "{read_accounting}");
            assert_eq!(cost.cache_write, Some(0.000_037_5), "{read_accounting}");
            assert!(
                (cost.total.unwrap() - 0.000_499_5).abs() < 1e-12,
                "{read_accounting} {cost:?}"
            );
        }
    }
}

#[test]
fn test_usage_with_inconsistent_input_counts_is_not_priced() {
    let catalog = cache_pricing_catalog("included_in_prompt_tokens");
    let usages = [
        Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(20),
            cache_read_tokens: Some(140),
            ..Usage::default()
        },
        Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(20),
            uncached_input_tokens: Some(999),
            ..Usage::default()
        },
        Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(20),
            cache_read_tokens: Some(40),
            cache_write_tokens: Some(10),
            uncached_input_tokens: Some(60),
            ..Usage::default()
        },
    ];

    for usage in &usages {
        assert!(
            estimate_cost_with_catalog(&catalog, "cache-model", usage).is_none(),
            "{usage:?}"
        );
    }
}

#[test]
fn test_split_usage_without_prompt_tokens_bills_its_uncached_input() {
    let catalog = cache_pricing_catalog("included_in_prompt_tokens");
    let usage = Usage {
        completion_tokens: Some(20),
        cache_read_tokens: Some(40),
        uncached_input_tokens: Some(50),
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "cache-model", &usage).unwrap();

    assert_eq!(cost.input, Some(0.000_15));
    assert_eq!(cost.cache_read, Some(0.000_012));
    assert_eq!(cost.output, Some(0.000_3));
    assert!((cost.total.unwrap() - 0.000_462).abs() < 1e-12, "{cost:?}");
}

#[test]
fn test_rate_tier_of_split_usage_counts_cache_reads_and_writes() {
    let catalog = threshold_pricing_catalog("separate");
    let usage = Usage {
        prompt_tokens: Some(150_000),
        completion_tokens: Some(1_000),
        cache_read_tokens: Some(40_000),
        cache_write_tokens: Some(20_000),
        uncached_input_tokens: Some(150_000),
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "threshold-model", &usage).unwrap();

    assert_eq!(cost.input, Some(1.5));
    assert_eq!(cost.output, Some(0.02));
    assert_eq!(cost.cache_read, Some(0.04));
    assert_eq!(cost.cache_write, Some(0.25));
    assert!((cost.total.unwrap() - 1.81).abs() < 1e-12, "{cost:?}");
}

#[test]
fn test_rate_tier_matches_for_split_and_unsplit_usage_of_one_request() {
    let catalog = threshold_pricing_catalog("separate");
    let unsplit = Usage {
        prompt_tokens: Some(150_000),
        completion_tokens: Some(1_000),
        cache_read_tokens: Some(60_000),
        ..Usage::default()
    };
    let split = Usage {
        uncached_input_tokens: Some(150_000),
        ..unsplit.clone()
    };

    let unsplit_cost = estimate_cost_with_catalog(&catalog, "threshold-model", &unsplit).unwrap();
    let split_cost = estimate_cost_with_catalog(&catalog, "threshold-model", &split).unwrap();

    assert_eq!(unsplit_cost, split_cost);
    assert_eq!(unsplit_cost.input, Some(1.5));
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
        (
            "cache_write_1h_per_million",
            "rates.cache_write_1h_per_million",
        ),
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
fn test_pricing_catalog_rejects_unknown_fields_in_nested_structures() {
    let valid_entry = flat_pricing_entry("configured", "configured-model", 1.0, 2.0);

    let mut catalog_with_unknown = json!({
        "version": 1,
        "entries": [valid_entry.clone()]
    });
    catalog_with_unknown["unexpected_catalog"] = json!(true);

    let mut entry_with_unknown = valid_entry.clone();
    entry_with_unknown["unexpected_entry"] = json!(true);

    let mut rates_with_unknown = valid_entry.clone();
    rates_with_unknown["rates"]["cache_writ_per_million"] = json!(0.1);

    let mut prompt_cache_with_unknown = valid_entry.clone();
    prompt_cache_with_unknown["prompt_cache"]["read_accountng"] = json!("separate");

    let schedule_with_unknown = json!([{
        "provider": "configured",
        "model_id": "scheduled-model",
        "pricing_as_of": "2026-06-04",
        "pricing_source": "test",
        "rate_schedule": {
            "type": "prompt_token_threshold",
            "tierz": [{
                "min_prompt_tokens": 0,
                "rates": {
                    "input_per_million": 1.0,
                    "output_per_million": 2.0
                }
            }],
            "tiers": [{
                "min_prompt_tokens": 0,
                "rates": {
                    "input_per_million": 1.0,
                    "output_per_million": 2.0
                }
            }]
        },
        "prompt_cache": {
            "read_accounting": "separate"
        }
    }]);

    let tier_with_unknown = json!([{
        "provider": "configured",
        "model_id": "tiered-model",
        "pricing_as_of": "2026-06-04",
        "pricing_source": "test",
        "rate_schedule": {
            "type": "prompt_token_threshold",
            "tiers": [{
                "min_prompt_toknes": 0,
                "rates": {
                    "input_per_million": 1.0,
                    "output_per_million": 2.0
                }
            }]
        },
        "prompt_cache": {
            "read_accounting": "separate"
        }
    }]);

    for (payload, expected_field) in [
        (catalog_with_unknown, "unexpected_catalog"),
        (
            json!({ "version": 1, "entries": [entry_with_unknown] }),
            "unexpected_entry",
        ),
        (
            json!({ "version": 1, "entries": [rates_with_unknown] }),
            "cache_writ_per_million",
        ),
        (
            json!({ "version": 1, "entries": [prompt_cache_with_unknown] }),
            "read_accountng",
        ),
        (
            json!({ "version": 1, "entries": schedule_with_unknown }),
            "tierz",
        ),
        (
            json!({ "version": 1, "entries": tier_with_unknown }),
            "min_prompt_toknes",
        ),
    ] {
        let err = PricingCatalog::from_json_str(&payload.to_string()).unwrap_err();
        assert!(
            err.to_string()
                .contains(&format!("unknown field `{expected_field}`")),
            "expected unknown field `{expected_field}`, got {err}"
        );
    }
}

#[test]
fn test_pricing_source_config_rejects_unknown_fields_on_variants() {
    let valid_entry = flat_pricing_entry("configured", "configured-model", 1.0, 2.0);

    for source in [
        json!({
            "type": "inline",
            "catalog": {
                "version": 1,
                "entries": [valid_entry.clone()]
            },
            "unexpected": true
        }),
        json!({
            "type": "file",
            "path": "pricing.json",
            "unexpected": true
        }),
    ] {
        let err = serde_json::from_value::<PricingSourceConfig>(source).unwrap_err();
        assert!(
            err.to_string().contains("unknown field `unexpected`"),
            "expected unknown field `unexpected`, got {err}"
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
fn test_missing_token_pricing_returns_none_without_fabricating_zero_cost() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    reset_active_pricing_resolver().unwrap();
    assert_eq!(estimate_cost("gpt-4o-mini", &Usage::default()), None);
    assert_eq!(estimate_cost("gpt-4o-mini", &Usage::default()), None);
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

    let partial_model_pricing_cost = CostEstimate {
        total: None,
        currency: "USD".to_string(),
        input: Some(0.12),
        output: Some(0.30),
        cache_read: Some(0.01),
        cache_write: None,
        source: CostSource::ModelPricing,
        pricing_provider: Some("test".to_string()),
        pricing_model: Some("model".to_string()),
        pricing_as_of: Some("2026-06-04".to_string()),
        pricing_source: Some("test".to_string()),
    };
    assert_eq!(partial_model_pricing_cost.total_or_component_sum(), None);
    assert_eq!(
        partial_model_pricing_cost.total_or_component_sum_for_currency("USD"),
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
        cache_write_1h_tokens: None,
        uncached_input_tokens: None,
        cost: estimate_cost_with_catalog(
            &catalog,
            "configured-model",
            &Usage {
                prompt_tokens: Some(1_000),
                completion_tokens: Some(500),
                total_tokens: Some(1_500),
                cache_read_tokens: Some(200),
                cache_write_tokens: None,
                cache_write_1h_tokens: None,
                uncached_input_tokens: None,
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

fn cache_pricing_catalog(read_accounting: &str) -> PricingCatalog {
    pricing_catalog(json!([
        {
            "provider": "configured",
            "model_id": "cache-model",
            "pricing_as_of": "2026-06-05",
            "pricing_source": "https://example.test/pricing",
            "rates": {
                "input_per_million": 3.0,
                "output_per_million": 15.0,
                "cache_read_per_million": 0.3,
                "cache_write_per_million": 3.75
            },
            "prompt_cache": {
                "read_accounting": read_accounting
            }
        }
    ]))
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

fn pricing_catalog(entries: Value) -> PricingCatalog {
    PricingCatalog::from_json_str(&json!({ "version": 1, "entries": entries }).to_string()).unwrap()
}

fn pricing_catalog_error(entries: Value) -> PricingCatalogError {
    PricingCatalog::from_json_str(&json!({ "version": 1, "entries": entries }).to_string())
        .unwrap_err()
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
fn test_model_pricing_rejects_one_hour_writes_above_combined_writes() {
    let catalog = pricing_catalog(json!([{
        "provider": "future-ai",
        "model_id": "future-model",
        "pricing_as_of": "2026-06-04",
        "pricing_source": "https://example.test/pricing",
        "rates": {
            "input_per_million": 1.0,
            "output_per_million": 2.0,
            "cache_write_per_million": 1.5,
            "cache_write_1h_per_million": 2.4
        },
        "prompt_cache": {"read_accounting": "separate"}
    }]));
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(2_000),
        cache_write_tokens: Some(1_000),
        cache_write_1h_tokens: Some(1_001),
        ..Usage::default()
    };

    assert!(estimate_cost_with_catalog(&catalog, "future-model", &usage).is_none());
}

#[test]
fn test_model_pricing_omits_cache_write_component_when_one_hour_rate_is_missing() {
    let catalog = pricing_catalog(json!([{
        "provider": "future-ai",
        "model_id": "future-model",
        "pricing_as_of": "2026-06-04",
        "pricing_source": "https://example.test/pricing",
        "rates": {
            "input_per_million": 1.0,
            "output_per_million": 2.0,
            "cache_write_per_million": 1.5
        },
        "prompt_cache": {"read_accounting": "separate"}
    }]));
    let usage = Usage {
        prompt_tokens: Some(1_000),
        completion_tokens: Some(2_000),
        cache_write_tokens: Some(4_000),
        cache_write_1h_tokens: Some(1_000),
        ..Usage::default()
    };

    let cost = estimate_cost_with_catalog(&catalog, "future-model", &usage).unwrap();

    assert_eq!(cost.total, None);
    assert_eq!(cost.input, Some(0.001));
    assert_eq!(cost.output, Some(0.004));
    assert_eq!(cost.cache_write, None);
}
