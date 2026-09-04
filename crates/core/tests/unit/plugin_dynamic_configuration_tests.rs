// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, json};

use super::*;
use crate::plugin::dynamic::{
    DynamicPluginAttestationMode, DynamicPluginRecord, DynamicPluginStartupClass,
};

fn worker_manifest(plugin_id: &str, schema: Option<&str>) -> DynamicPluginManifest {
    let capabilities = if schema.is_some() {
        "[\"plugin_worker\", \"config_schema\"]"
    } else {
        "[\"plugin_worker\"]"
    };
    let schema = schema
        .map(|path| format!("[config_schema]\npath = {path:?}\n"))
        .unwrap_or_default();
    DynamicPluginManifest::parse_toml(&format!(
        r#"
manifest_version = 1

[plugin]
id = {plugin_id:?}
kind = "worker"

[compat]
relay = ">=0.8.0,<1.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = {capabilities}

{schema}
[load]
runtime = "command"
entrypoint = "fixture-worker"
"#
    ))
    .unwrap()
}

fn lifecycle_record(plugin_id: &str, manifest_ref: &str) -> DynamicPluginRecord {
    let mut record = worker_manifest(plugin_id, None)
        .into_record(Some(manifest_ref.to_owned()))
        .unwrap();
    record.spec.present = true;
    record.spec.enabled = true;
    record.source.environment_ref = Some("fixture-environment".into());
    record
}

fn write_state(path: &Path, records: &[DynamicPluginRecord], schema_version: Option<u32>) {
    let mut value = json!({"records": records});
    if let Some(schema_version) = schema_version {
        value["schema_version"] = json!(schema_version);
    }
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn validation_request_deserializes_every_target_and_rejects_invalid_shapes() {
    let all: PluginHostValidationRequest = serde_json::from_value(json!({})).unwrap();
    assert_eq!(all.target, PluginHostValidationTarget::All);

    let plugin: PluginHostValidationRequest = serde_json::from_value(json!({
        "target": "plugin_id",
        "value": "fixture.plugin",
        "additional_plugins_toml": "plugins.toml"
    }))
    .unwrap();
    assert_eq!(
        plugin.target,
        PluginHostValidationTarget::PluginId("fixture.plugin".into())
    );
    assert_eq!(
        plugin.additional_plugins_toml,
        Some(PathBuf::from("plugins.toml"))
    );

    let manifest: PluginHostValidationRequest = serde_json::from_value(json!({
        "target": "manifest_path",
        "value": "relay-plugin.toml"
    }))
    .unwrap();
    assert_eq!(
        manifest.target,
        PluginHostValidationTarget::ManifestPath(PathBuf::from("relay-plugin.toml"))
    );

    for invalid in [
        json!({"target": "plugin_id"}),
        json!({"target": "manifest_path"}),
        json!({"target": "unsupported", "value": "fixture"}),
    ] {
        assert!(
            serde_json::from_value::<PluginHostValidationRequest>(invalid).is_err(),
            "invalid target must fail closed"
        );
    }
}

#[test]
fn lifecycle_state_selection_covers_absent_disabled_and_enabled_records() {
    let temp = tempfile::tempdir().unwrap();
    let plugins_toml = temp.path().join("plugins.toml");
    let state_path = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let manifest_ref = temp.path().join("relay-plugin.toml");
    let manifest_ref = manifest_ref.to_string_lossy().into_owned();

    assert!(
        state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
            .unwrap()
            .is_none()
    );

    write_state(&state_path, &[], None);
    let missing = state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
        .unwrap()
        .unwrap();
    assert!(!missing.selected);
    assert!(missing.environment_ref.is_none());

    let mut record = lifecycle_record("fixture.plugin", &manifest_ref);
    write_state(&state_path, &[record.clone()], Some(1));
    let enabled = state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
        .unwrap()
        .unwrap();
    assert!(enabled.selected);
    assert_eq!(
        enabled.environment_ref.as_deref(),
        Some("fixture-environment")
    );

    record.spec.enabled = false;
    write_state(&state_path, &[record.clone()], Some(1));
    assert!(
        !state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
            .unwrap()
            .unwrap()
            .selected
    );

    record.spec.enabled = true;
    record.spec.present = false;
    write_state(&state_path, &[record], Some(1));
    assert!(
        !state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
            .unwrap()
            .unwrap()
            .selected
    );
}

#[test]
fn lifecycle_state_selection_rejects_malformed_ambiguous_and_mismatched_state() {
    let temp = tempfile::tempdir().unwrap();
    let plugins_toml = temp.path().join("plugins.toml");
    let state_path = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let manifest_ref = temp.path().join("relay-plugin.toml");
    let manifest_ref = manifest_ref.to_string_lossy().into_owned();

    fs::write(&state_path, "{").unwrap();
    let malformed = state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
        .err()
        .expect("malformed state must fail")
        .to_string();
    assert!(malformed.contains("invalid dynamic plugin lifecycle state"));

    write_state(&state_path, &[], Some(2));
    let version = state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
        .err()
        .expect("unsupported state version must fail")
        .to_string();
    assert!(version.contains("unsupported dynamic plugin lifecycle schema_version"));

    let record = lifecycle_record("fixture.plugin", &manifest_ref);
    write_state(&state_path, &[record.clone(), record], Some(1));
    let duplicate = state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
        .err()
        .expect("duplicate state records must fail")
        .to_string();
    assert!(duplicate.contains("duplicate record"));

    let mismatch = lifecycle_record("fixture.plugin", "different-manifest.toml");
    write_state(&state_path, &[mismatch], Some(1));
    let error = state_for_plugin(&plugins_toml, &manifest_ref, "fixture.plugin")
        .err()
        .expect("manifest identity mismatch must fail")
        .to_string();
    assert!(error.contains("manifest identity mismatch"));
}

#[test]
fn policy_and_utf8_file_loading_fail_closed_with_secure_defaults() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing.toml");
    let policy = resolve_plugin_host_config_inner(PluginConfig::default(), Some(&missing), false)
        .unwrap()
        .policy;
    assert_eq!(
        policy.defaults.startup,
        Some(DynamicPluginStartupClass::Required)
    );
    assert_eq!(
        policy.defaults.attestation,
        Some(DynamicPluginAttestationMode::SignatureRequired)
    );

    let plugins_toml = temp.path().join("plugins.toml");
    fs::write(
        &plugins_toml,
        r#"
[plugins.policy.defaults]
allowed = true
startup = "optional"
attestation = "integrity_only"
"#,
    )
    .unwrap();
    let policy =
        resolve_plugin_host_config_inner(PluginConfig::default(), Some(&plugins_toml), false)
            .unwrap()
            .policy;
    assert_eq!(policy.defaults.allowed, Some(true));
    assert_eq!(
        policy.defaults.startup,
        Some(DynamicPluginStartupClass::Optional)
    );
    assert_eq!(
        policy.defaults.attestation,
        Some(DynamicPluginAttestationMode::IntegrityOnly)
    );

    fs::write(&plugins_toml, "{").unwrap();
    assert!(
        resolve_plugin_host_config_inner(PluginConfig::default(), Some(&plugins_toml), false)
            .is_err()
    );

    fs::write(&plugins_toml, [0xff, 0xfe]).unwrap();
    let utf8 = read_utf8_plugin_file(&plugins_toml)
        .unwrap_err()
        .to_string();
    assert!(utf8.contains("is not UTF-8"), "{utf8}");

    let directory = read_utf8_plugin_file(temp.path()).unwrap_err().to_string();
    assert!(directory.contains("failed to read"), "{directory}");
}

#[test]
fn static_plugin_policy_does_not_override_dynamic_attestation_policy() {
    let temp = tempfile::tempdir().unwrap();
    let plugins_toml = temp.path().join("plugins.toml");
    fs::write(
        &plugins_toml,
        r#"
[plugins.policy.defaults]
attestation = "integrity_only"
"#,
    )
    .unwrap();
    let programmatic = PluginConfig {
        policy: crate::plugin::ConfigPolicy {
            unsupported_value: crate::plugin::UnsupportedBehavior::Ignore,
            ..crate::plugin::ConfigPolicy::default()
        },
        ..PluginConfig::default()
    };

    let resolved = resolve_plugin_host_config_inner(programmatic, Some(&plugins_toml), false)
        .expect("static and dynamic policies should resolve independently");

    assert_eq!(
        resolved.config.policy.unsupported_value,
        crate::plugin::UnsupportedBehavior::Ignore
    );
    assert_eq!(
        resolved.policy.defaults.attestation,
        Some(DynamicPluginAttestationMode::IntegrityOnly)
    );
}

#[test]
fn manifest_paths_and_validation_reports_preserve_fail_closed_selection() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("config/plugins.toml");
    let relative = resolve_manifest_path(&source, "plugins/relay-plugin.toml");
    assert_eq!(
        relative,
        temp.path().join("config/plugins/relay-plugin.toml")
    );
    let absolute = temp.path().join("absolute/relay-plugin.toml");
    assert_eq!(
        resolve_manifest_path(&source, absolute.to_string_lossy().as_ref()),
        absolute
    );

    let manifest = worker_manifest("fixture.invalid", Some("missing.schema.json"));
    let mut policy = DynamicPluginHostPolicy::default();
    policy.apply_secure_defaults();
    let evaluated_policy = evaluate_dynamic_plugin_host_policy(&policy, &manifest);
    let report = validate_declaration(
        &manifest,
        temp.path()
            .join("relay-plugin.toml")
            .to_string_lossy()
            .into_owned(),
        true,
        &Map::new(),
        &evaluated_policy,
    );
    assert!(!report.selected);
    assert_eq!(report.status.manifest, DynamicPluginCheckState::Invalid);
    assert!(report.failure.is_some());
}
