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

#[test]
fn native_diagnostic_read_grant_uses_policy_layers_and_never_grants_worker_reads() {
    let lower: FileDynamicPluginHostPolicy = toml::from_str(
        r#"
[defaults]
native_diagnostics_read_context = true
"#,
    )
    .unwrap();
    let upper: FileDynamicPluginHostPolicy = toml::from_str(
        r#"
[[rules]]
match_kind = "rust_dynamic"
native_diagnostics_read_context = false
[overrides."fixture.policy"]
native_diagnostics_read_context = true
"#,
    )
    .unwrap();
    let mut policy: DynamicPluginHostPolicy = lower.into();
    policy.merge_from(upper.into());
    let mut native = manifest();
    native.plugin.kind = DynamicPluginKind::RustDynamic;
    assert!(evaluate_dynamic_plugin_host_policy(&policy, &native).native_diagnostics_read_context);
    assert!(
        !evaluate_dynamic_plugin_host_policy(&policy, &manifest()).native_diagnostics_read_context
    );
    policy.overrides.clear();
    assert!(!evaluate_dynamic_plugin_host_policy(&policy, &native).native_diagnostics_read_context);
    assert!(
        !evaluate_dynamic_plugin_host_policy(&DynamicPluginHostPolicy::default(), &native)
            .native_diagnostics_read_context
    );
}

#[test]
fn native_diagnostic_read_grant_preserves_explicit_denials_across_file_layers() {
    let mut native = manifest();
    native.plugin.kind = DynamicPluginKind::RustDynamic;
    for table in ["[defaults]", "[[rules]]", "[overrides.\"fixture.policy\"]"] {
        let lower: FileDynamicPluginHostPolicy = toml::from_str(&format!(
            "{table}\nnative_diagnostics_read_context = true\n"
        ))
        .unwrap();
        let upper: FileDynamicPluginHostPolicy = toml::from_str(&format!(
            "{table}\nnative_diagnostics_read_context = false\n"
        ))
        .unwrap();
        let mut policy: DynamicPluginHostPolicy = lower.into();
        assert!(
            evaluate_dynamic_plugin_host_policy(&policy, &native).native_diagnostics_read_context
        );
        policy.merge_from(upper.into());
        assert!(
            !evaluate_dynamic_plugin_host_policy(&policy, &native).native_diagnostics_read_context,
            "explicit denial must survive merging {table}"
        );
    }
}

#[test]
fn native_diagnostic_read_grant_requires_a_boolean_in_every_file_representation() {
    for table in ["[defaults]", "[[rules]]", "[overrides.\"fixture.policy\"]"] {
        for value in ["\"true\"", "1", "[]", "{}"] {
            let input = format!("{table}\nnative_diagnostics_read_context = {value}\n");
            assert!(toml::from_str::<FileDynamicPluginHostPolicy>(&input).is_err());
        }
        let misspelled = format!("{table}\nnative_diagnostic_read_context = true\n");
        assert!(toml::from_str::<FileDynamicPluginHostPolicy>(&misspelled).is_err());
    }
}
