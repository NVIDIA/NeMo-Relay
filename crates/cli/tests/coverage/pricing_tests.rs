// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use super::*;

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
                "read_accounting": "separate",
                "read_per_million": 0.1,
                "write_per_million": 0.2
            }
        }]
    })
}

fn catalog() -> PricingCatalog {
    PricingCatalog::from_json_str(&catalog_json().to_string()).unwrap()
}

#[test]
fn pricing_helpers_cover_scopes_components_sources_and_usage() {
    assert_eq!(
        target_pricing_scope(&PricingScopeArgs::default()).unwrap(),
        TargetScope::User
    );
    assert_eq!(
        target_pricing_scope(&PricingScopeArgs {
            project: true,
            ..PricingScopeArgs::default()
        })
        .unwrap(),
        TargetScope::Project
    );
    assert_eq!(
        target_pricing_scope(&PricingScopeArgs {
            global: true,
            ..PricingScopeArgs::default()
        })
        .unwrap(),
        TargetScope::Global
    );
    assert!(
        target_pricing_scope(&PricingScopeArgs {
            user: true,
            project: true,
            ..PricingScopeArgs::default()
        })
        .unwrap_err()
        .to_string()
        .contains("choose only one")
    );

    let mut plugin_config = PluginConfig::default();
    let created = ensure_pricing_component(&mut plugin_config).unwrap();
    assert_eq!(created, 0);
    assert!(plugin_config.components[created].enabled);
    assert_eq!(
        ensure_pricing_component(&mut plugin_config).unwrap(),
        created
    );
    let parsed = pricing_config_from_component(&plugin_config.components[created]).unwrap();
    assert!(parsed.sources.is_empty());

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("pricing.json");
    std::fs::write(&file, catalog_json().to_string()).unwrap();
    let sources = pricing_catalog_sources_from_config(&PricingConfig {
        sources: vec![
            PricingSourceConfig::Inline { catalog: catalog() },
            PricingSourceConfig::File { path: file.clone() },
        ],
    })
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

    assert!(!usage_has_tokens(&Usage::default()));
    assert!(usage_has_tokens(&Usage {
        prompt_tokens: Some(1),
        ..Usage::default()
    }));
    assert_eq!(plural(1, "entry", "entries"), "entry");
    assert_eq!(plural(2, "entry", "entries"), "entries");
}

#[test]
fn pricing_component_rejects_malformed_component_config() {
    let mut component = PluginComponentSpec::new(PRICING_PLUGIN_KIND);
    component.config.insert("sources".into(), json!("bad"));
    let error = pricing_config_from_component(&component)
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid pricing config"));
}
