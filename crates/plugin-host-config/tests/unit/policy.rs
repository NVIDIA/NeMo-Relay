// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::plugin::dynamic::{DynamicPluginKind, DynamicPluginManifest};

use super::*;

#[test]
fn policy_rules_require_every_configured_selector_to_match() {
    let manifest = DynamicPluginManifest::parse_toml(
        r#"manifest_version = 1
[plugin]
id = "fixture"
kind = "rust_dynamic"
[compat]
relay = ">=0.5,<1.0"
native_api = "v1"
[capabilities]
items = ["plugin_native"]
[defaults]
enabled = false
[load]
library = "plugin.so"
symbol = "nemo_relay_plugin_entrypoint_v1"
"#,
    )
    .unwrap();
    let effect = DynamicPluginHostPolicyEffect {
        allowed: Some(false),
        ..Default::default()
    };

    let wrong_kind = DynamicPluginHostPolicyRule {
        match_kind: Some(DynamicPluginKind::Worker),
        effect: effect.clone(),
        ..Default::default()
    };
    assert!(!policy_rule_matches(&wrong_kind, &manifest));

    let wrong_id = DynamicPluginHostPolicyRule {
        match_plugin_id: Some("other".into()),
        effect: effect.clone(),
        ..Default::default()
    };
    assert!(!policy_rule_matches(&wrong_id, &manifest));

    let matching = DynamicPluginHostPolicyRule {
        match_kind: Some(DynamicPluginKind::RustDynamic),
        match_plugin_id: Some("fixture".into()),
        effect,
    };
    assert!(policy_rule_matches(&matching, &manifest));
}
