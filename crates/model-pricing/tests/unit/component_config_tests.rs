// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::catalog::{pricing_catalog_sources_from_config, resolve_pricing};
use crate::{PricingCatalog, PricingSourceConfig};
use serde_json::{Value, json};

fn catalog_json() -> Value {
    json!({
        "version": 1,
        "entries": [{
            "provider": "test",
            "model_id": "model",
            "aliases": ["alias-model"],
            "pricing_as_of": "2026-06-04",
            "pricing_source": "unit-test",
            "rates": {
                "input_per_million": 1.0,
                "output_per_million": 2.0
            },
            "prompt_cache": {
                "read_accounting": "separate"
            }
        }]
    })
}

fn catalog() -> PricingCatalog {
    PricingCatalog::from_json_str(&catalog_json().to_string()).unwrap()
}

#[test]
fn component_config_and_source_resolution_preserve_precedence() {
    let mut components = Vec::new();
    let created = ensure_pricing_component(&mut components).unwrap();
    assert_eq!(created, 0);
    assert!(components[created].enabled);
    assert_eq!(ensure_pricing_component(&mut components).unwrap(), created);
    let parsed = pricing_config_from_component(&components[created]).unwrap();
    assert!(parsed.sources.is_empty());

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("pricing.json");
    std::fs::write(&file, catalog_json().to_string()).unwrap();
    let sources = pricing_catalog_sources_from_config(
        &PricingConfig {
            sources: vec![
                PricingSourceConfig::Inline { catalog: catalog() },
                PricingSourceConfig::File { path: file.clone() },
            ],
        },
        |path| PricingCatalog::from_json_str(&std::fs::read_to_string(path).unwrap()),
    )
    .unwrap();
    assert_eq!(sources[0].label, "inline:0");
    assert_eq!(sources[1].label, format!("file:{}", file.display()));
    assert_eq!(
        resolve_pricing(&sources, Some("test"), "alias-model")
            .unwrap()
            .pricing
            .model_id,
        "model"
    );
    assert!(resolve_pricing(&sources, Some("other"), "missing-model").is_none());
}

#[test]
fn pricing_component_rejects_malformed_component_config() {
    let mut component = PluginComponentSpec::new(PRICING_PLUGIN_KIND);
    component.config.insert("sources".into(), json!("bad"));
    let error = pricing_config_from_component(&component)
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid model pricing config"));
}
