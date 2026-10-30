// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn pricing_cli_helpers_cover_scopes_usage_and_labels() {
    assert_eq!(
        target_pricing_scope(&ConfigurationScope::default()).unwrap(),
        TargetScope::User
    );
    assert_eq!(
        target_pricing_scope(&ConfigurationScope::User).unwrap(),
        TargetScope::User
    );
    assert_eq!(
        target_pricing_scope(&ConfigurationScope::Global).unwrap(),
        TargetScope::Global
    );
    assert!(
        target_pricing_scope(&ConfigurationScope::Invalid)
            .unwrap_err()
            .to_string()
            .contains("choose only one")
    );

    assert!(!usage_has_tokens(&Usage::default()));
    assert!(usage_has_tokens(&Usage {
        prompt_tokens: Some(1),
        ..Usage::default()
    }));
    assert_eq!(plural(1, "entry", "entries"), "entry");
    assert_eq!(plural(2, "entry", "entries"), "entries");
}

#[test]
fn pricing_catalog_reads_are_bounded_and_require_utf8() {
    let temp = tempfile::tempdir().unwrap();
    let invalid_utf8 = temp.path().join("invalid.json");
    std::fs::write(&invalid_utf8, [0xff]).unwrap();
    let error = read_pricing_catalog(&invalid_utf8).unwrap_err().to_string();
    assert!(error.contains("not valid UTF-8"), "{error}");

    let oversized = temp.path().join("oversized.json");
    let file = std::fs::File::create(&oversized).unwrap();
    file.set_len(crate::filesystem::bounded::MAX_BOUNDED_FILE_BYTES + 1)
        .unwrap();
    let error = read_pricing_catalog(&oversized).unwrap_err().to_string();
    assert!(error.contains("exceeds"), "{error}");
}

#[test]
fn pricing_document_update_preserves_dynamic_and_host_sections() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    std::fs::write(
        &path,
        r#"host_setting = "preserve-me"

[[plugins.dynamic]]
manifest = "./relay-plugin.toml"
config = {}

[plugins.policy]
allow_unsigned = true

[host.extra]
value = 42
"#,
    )
    .unwrap();

    update_plugin_config_document(&path, |plugin_config| {
        let index = ensure_pricing_component(&mut plugin_config.components)
            .map_err(pricing_config_error)?;
        plugin_config.components[index].enabled = true;
        Ok(())
    })
    .unwrap();

    let root = std::fs::read_to_string(&path)
        .unwrap()
        .parse::<toml::Table>()
        .unwrap();
    assert_eq!(root["host_setting"].as_str(), Some("preserve-me"));
    assert_eq!(root["host"]["extra"]["value"].as_integer(), Some(42));
    assert_eq!(
        root["plugins"]["policy"]["allow_unsigned"].as_bool(),
        Some(true)
    );
    let dynamic = root["plugins"]["dynamic"].as_array().unwrap();
    assert_eq!(dynamic[0]["manifest"].as_str(), Some("./relay-plugin.toml"));
    assert!(dynamic[0]["config"].as_table().unwrap().is_empty());
    let pricing = root["components"]
        .as_array()
        .unwrap()
        .iter()
        .find(|component| component["kind"].as_str() == Some(PRICING_PLUGIN_KIND))
        .expect("pricing component should be present");
    assert!(
        pricing
            .get("enabled")
            .and_then(toml::Value::as_bool)
            .unwrap_or(true)
    );
}
