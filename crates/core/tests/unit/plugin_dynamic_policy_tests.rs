// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn manifest() -> DynamicPluginManifest {
    DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1
[plugin]
id = "fixture.policy"
kind = "worker"
[compat]
relay = ">=0.8.0,<1.0"
worker_protocol = "grpc-v1"
[defaults]
enabled = false
[capabilities]
items = ["plugin_worker"]
[load]
runtime = "command"
entrypoint = "fixture"
"#,
    )
    .unwrap()
}

#[test]
fn matching_rules_and_id_overrides_apply_in_order() {
    let mut policy = DynamicPluginHostPolicy::default();
    policy.rules.push(DynamicPluginHostPolicyRule {
        match_kind: Some(DynamicPluginKind::Worker),
        match_plugin_id: None,
        effect: DynamicPluginHostPolicyEffect {
            allowed: Some(false),
            ..Default::default()
        },
    });
    let denied = evaluate_dynamic_plugin_host_policy(&policy, &manifest());
    assert!(
        !denied.policy_satisfied,
        "kind rule must deny before override"
    );

    policy.overrides.insert(
        "fixture.policy".into(),
        DynamicPluginHostPolicyEffect {
            allowed: Some(true),
            startup: Some(DynamicPluginStartupClass::Required),
            ..Default::default()
        },
    );

    let evaluated = evaluate_dynamic_plugin_host_policy(&policy, &manifest());
    assert!(evaluated.policy_satisfied);
    assert_eq!(evaluated.startup_class, DynamicPluginStartupClass::Required);
}
