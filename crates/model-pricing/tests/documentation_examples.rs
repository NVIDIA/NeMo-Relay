// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Check the catalog and expected costs displayed in the Model Pricing reference.

use nemo_relay_model_pricing::{PricingCatalog, PricingResolver};
use nemo_relay_types::codec::response::Usage;
use serde_json::{Value, json};

#[test]
fn documentation_catalog_matches_expected_threshold_costs() {
    let page = include_str!("../../../docs/reference/api/model-pricing.mdx").replace("\r\n", "\n");
    for line_ending in ["\n", "\r\n"] {
        assert_documentation_catalog_matches_threshold_costs(&page.replace('\n', line_ending));
    }
}

fn assert_documentation_catalog_matches_threshold_costs(page: &str) {
    let page = page.replace("\r\n", "\n");
    let entry = page
        .split("```json title=\"Example catalog entry\"\n")
        .nth(1)
        .unwrap()
        .split("\n```")
        .next()
        .unwrap();
    let entry: Value = serde_json::from_str(entry).unwrap();
    let catalog =
        PricingCatalog::from_json_str(&json!({"version": 1, "entries": [entry]}).to_string())
            .unwrap();
    let resolver = PricingResolver::from_catalogs(vec![catalog]);
    let results = page.lines().filter(|line| line.starts_with("| 200,"));
    let mut cases = 0;
    for row in results {
        let fields: Vec<_> = row.split('|').map(str::trim).collect();
        let input_tokens: u64 = fields[1].replace(',', "").parse().unwrap();
        let expected: f64 = fields[2].parse().unwrap();
        let usage = Usage {
            prompt_tokens: Some(input_tokens),
            completion_tokens: Some(1000),
            ..Usage::default()
        };
        let estimate = resolver
            .estimate_cost_for_provider(Some("google"), "gemini-3.1-pro-preview", &usage)
            .unwrap();
        assert_eq!(estimate.currency, "USD");
        assert!((estimate.total.unwrap() - expected).abs() < 1e-12);
        cases += 1;
    }
    assert_eq!(
        cases, 2,
        "both documented threshold outcomes must be exercised"
    );
}
