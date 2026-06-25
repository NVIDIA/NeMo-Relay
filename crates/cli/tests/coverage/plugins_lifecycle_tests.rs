// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::{ffi::OsString, sync::MutexGuard};

use super::*;
use crate::config::{
    PluginsAddCommand, PluginsDisableCommand, PluginsEnableCommand, PluginsInspectCommand,
    PluginsListCommand, PluginsRemoveCommand, PluginsScopeArgs, PluginsValidateCommand, ServerArgs,
};
use crate::error::PluginLifecycleFailureKind;
use base64::Engine;
use nemo_relay::plugin::dynamic::DynamicPluginFailurePhase;
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair};
use sha2::{Digest, Sha256};

struct CurrentDirGuard {
    original: PathBuf,
}

impl CurrentDirGuard {
    fn enter(path: &Path) -> Self {
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(path).unwrap();
        Self { original }
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.original).unwrap();
    }
}

struct EnvScope {
    _guard: MutexGuard<'static, ()>,
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn hermetic(temp: &tempfile::TempDir) -> Self {
        let xdg = temp.path().join("xdg");
        std::fs::create_dir_all(&xdg).unwrap();
        Self::set(&[
            ("HOME", Some(temp.path().as_os_str())),
            ("XDG_CONFIG_HOME", Some(xdg.as_os_str())),
        ])
    }

    fn set(values: &[(&'static str, Option<&std::ffi::OsStr>)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();
        for (key, value) in values {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
        Self {
            _guard: guard,
            values: previous,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (key, value) in self.values.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn write_dynamic_manifest(dir: &Path, plugin_id: &str) -> PathBuf {
    write_dynamic_manifest_with_options(dir, plugin_id, &["plugin_worker"], None)
}

fn write_dynamic_manifest_with_options(
    dir: &Path,
    plugin_id: &str,
    capabilities: &[&str],
    signature_ref: Option<&str>,
) -> PathBuf {
    let artifact_body = format!("def register():\n    return {plugin_id:?}\n");
    std::fs::write(dir.join("plugin.py"), &artifact_body).unwrap();
    let digest = format!(
        "sha256:{}",
        Sha256::digest(artifact_body.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let capabilities = capabilities
        .iter()
        .map(|capability| format!("\"{capability}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let signature_line = signature_ref
        .map(|signature_ref| format!("signature = \"{signature_ref}\"\n"))
        .unwrap_or_default();
    let manifest_path = dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"
manifest_version = 1

[plugin]
id = "{plugin_id}"
kind = "worker"

[compat]
relay = "0.5"
worker_protocol = "1"

[defaults]
enabled = false

[capabilities]
items = [{capabilities}]

[source]
artifact = "plugin.py"

[integrity]
sha256 = "{digest}"
{signature_line}

[load]
runtime = "python"
entrypoint = "{plugin_id}.plugin:register"
"#,
            capabilities = capabilities,
            signature_line = signature_line,
        ),
    )
    .unwrap();
    manifest_path
}

fn write_detached_ed25519_signature(dir: &Path, signature_name: &str) -> String {
    std::fs::create_dir_all(dir).unwrap();
    let artifact = std::fs::read(dir.join("plugin.py")).unwrap();
    let pkcs8 =
        Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).expect("generate ed25519 keypair");
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).expect("parse ed25519 keypair");
    let signature = key_pair.sign(&artifact);
    let signature_text = format!(
        "ed25519:{}\n",
        base64::engine::general_purpose::STANDARD.encode(signature.as_ref())
    );
    std::fs::write(dir.join(signature_name), signature_text).unwrap();
    format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key_pair.public_key().as_ref())
    )
}

fn generate_ed25519_public_key() -> String {
    let pkcs8 =
        Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).expect("generate ed25519 keypair");
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).expect("parse ed25519 keypair");
    format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key_pair.public_key().as_ref())
    )
}

#[test]
fn trust_failure_messages_and_codes_cover_all_variants() {
    let path = PathBuf::from("/tmp/plugin.py");
    let signature_path = PathBuf::from("/tmp/plugin.py.sig");
    let cases = vec![
        (
            trust::DynamicPluginTrustFailure::MissingArtifact,
            "integrity_failed",
            "missing source.artifact",
        ),
        (
            trust::DynamicPluginTrustFailure::MissingIntegrityDigest,
            "integrity_failed",
            "missing integrity.sha256",
        ),
        (
            trust::DynamicPluginTrustFailure::ArtifactRead {
                path: path.clone(),
                error: "boom".into(),
            },
            "integrity_failed",
            "could not be read for trust verification",
        ),
        (
            trust::DynamicPluginTrustFailure::IntegrityMismatch {
                path: path.clone(),
                expected: "sha256:expected".into(),
                actual: "sha256:actual".into(),
            },
            "integrity_failed",
            "failed integrity verification",
        ),
        (
            trust::DynamicPluginTrustFailure::MissingSignature,
            "attestation_failed",
            "requires integrity.signature",
        ),
        (
            trust::DynamicPluginTrustFailure::MissingTrustedKeys,
            "attestation_failed",
            "no trusted_public_keys",
        ),
        (
            trust::DynamicPluginTrustFailure::SignatureRead {
                path: signature_path.clone(),
                error: "nope".into(),
            },
            "attestation_failed",
            "signature /tmp/plugin.py.sig could not be read",
        ),
        (
            trust::DynamicPluginTrustFailure::InvalidTrustedKey {
                key: "ed25519:bad".into(),
                error: "invalid".into(),
            },
            "attestation_failed",
            "invalid trusted public key",
        ),
        (
            trust::DynamicPluginTrustFailure::SignatureVerification {
                path: signature_path,
                parse_errors: vec!["bad key".into()],
            },
            "attestation_failed",
            "key parse errors: bad key",
        ),
    ];

    for (failure, code, snippet) in cases {
        assert_eq!(failure.refusal_code(), code);
        let rendered = failure.display("acme.coverage").to_string();
        assert!(rendered.contains("acme.coverage"), "{rendered}");
        assert!(rendered.contains(snippet), "{rendered}");
    }
}

#[test]
fn trust_last_error_preserves_integrity_code_under_signature_policy() {
    let trust = EvaluatedDynamicPluginTrust {
        integrity: DynamicPluginCheckState::Invalid,
        authenticity: DynamicPluginCheckState::Unknown,
        failure: Some(trust::DynamicPluginTrustFailure::IntegrityMismatch {
            path: PathBuf::from("/tmp/plugin.py"),
            expected: "sha256:expected".into(),
            actual: "sha256:actual".into(),
        }),
    };

    let error = trust
        .last_error("acme.coverage")
        .expect("integrity mismatch should persist an error");
    assert_eq!(error.phase, DynamicPluginFailurePhase::Validation);
    assert_eq!(error.code, "integrity_failed");
    assert!(error.message.contains("acme.coverage"));
    assert!(error.message.contains("failed integrity verification"));
}

#[test]
fn trust_evaluation_short_circuits_when_policy_is_blocked() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.blocked-short-circuit");

    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();
    let blocked_policy = crate::plugins::policy::EvaluatedDynamicPluginHostPolicy {
        policy_satisfied: false,
        startup_class: nemo_relay::plugin::dynamic::DynamicPluginStartupClass::Required,
        attestation_mode:
            nemo_relay::plugin::dynamic::DynamicPluginAttestationMode::SignatureRequired,
        trusted_public_keys: Vec::new(),
        failure: Some(crate::plugins::policy::DynamicPluginHostPolicyFailure::Blocked),
    };

    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &blocked_policy);

    assert_eq!(trust.integrity, DynamicPluginCheckState::Unknown);
    assert_eq!(trust.authenticity, DynamicPluginCheckState::Unknown);
    assert!(trust.failure().is_none());
}

#[test]
fn add_registers_dynamic_plugin_in_project_plugins_toml() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir.clone(),
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(rendered.contains("[[plugins.dynamic]]"));
    assert!(rendered.contains("relay-plugin.toml"));

    let resolved = resolve_plugins_config(None).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.guardrail");
}

#[test]
fn add_rejects_duplicate_dynamic_plugin_ids() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir.clone(),
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap();

    let error = add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("already registered"));
}

#[test]
fn add_rejects_scope_flags_when_explicit_config_is_set() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join("custom-config");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.explicit-conflict");

    let server = ServerArgs {
        config: Some(config_dir.join("gateway.toml")),
        ..ServerArgs::default()
    };

    let error = add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("--config cannot be combined"));
}

#[test]
fn add_refuses_dynamic_plugins_blocked_by_host_policy() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.blocked");
    std::fs::write(
        config_dir.join("plugins.toml"),
        r#"
[plugins.policy.defaults]
allowed = false
"#,
    )
    .unwrap();

    let error = add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap_err();

    match error {
        CliError::PluginLifecycle {
            kind: PluginLifecycleFailureKind::Refused,
            message,
            ..
        } => assert!(message.contains("blocked by host policy")),
        other => panic!("unexpected policy add error: {other}"),
    }

    let rendered = std::fs::read_to_string(config_dir.join("plugins.toml")).unwrap();
    assert!(!rendered.contains("[[plugins.dynamic]]"));
}

#[test]
fn validate_path_reports_integrity_mismatch() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.integrity");
    std::fs::write(
        plugin_dir.join("plugin.py"),
        "def register():\n    return 'tampered'\n",
    )
    .unwrap();
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();
    let policy = evaluate_dynamic_plugin_host_policy(
        &ResolvedConfig::default().dynamic_plugin_policy,
        &manifest,
    );
    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
    let summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: None,
        host_config: None,
        policy: &policy,
        trust: &trust,
    }
    .to_string();

    assert_eq!(trust.integrity, DynamicPluginCheckState::Invalid);
    assert!(summary.contains("trust verification blocks it"));
    assert!(summary.contains("integrity_state: invalid"));
    assert!(
        summary
            .contains("trust_error: dynamic plugin 'acme.integrity' failed integrity verification")
    );
}

#[test]
fn list_and_inspect_render_discovered_dynamic_plugins() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let records = collect_records(&scopes, false);
    let list = PluginListView {
        records: &records,
        host_config_by_id: &host_config_by_id,
    }
    .to_string();
    assert!(list.contains("POLICY"));
    assert!(list.contains("acme.guardrail"));
    assert!(list.contains("absent"));
    assert!(list.contains("false"));
    assert!(
        list.lines()
            .any(|line| line.contains("acme.guardrail") && line.contains(" valid "))
    );

    let entry = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("plugin record");
    let (manifest, manifest_ref) =
        DynamicPluginManifest::load_from_path(entry.record.source.manifest_ref.clone().unwrap())
            .map_err(|error| CliError::Config(error.to_string()))
            .unwrap();
    let inspect = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.guardrail"),
    }
    .to_string();
    let inspect_value: serde_yaml::Value = serde_yaml::from_str(&inspect).unwrap();
    assert_eq!(
        inspect_value["metadata"]["id"].as_str(),
        Some("acme.guardrail")
    );
    assert_eq!(inspect_value["metadata"]["kind"].as_str(), Some("worker"));
    assert_eq!(inspect_value["host_config_status"].as_str(), Some("absent"));
    assert!(
        inspect_value["source"]["manifest_ref"]
            .as_str()
            .unwrap()
            .contains("relay-plugin.toml")
    );
    assert_eq!(
        inspect_value["load"]["entrypoint"].as_str(),
        Some("acme.guardrail.plugin:register")
    );
}

#[test]
fn validate_renders_summary_for_path_and_id_targets() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap();

    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();
    let default_policy = evaluate_dynamic_plugin_host_policy(
        &ResolvedConfig::default().dynamic_plugin_policy,
        &manifest,
    );
    let default_trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &default_policy);
    let path_summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: None,
        host_config: None,
        policy: &default_policy,
        trust: &default_trust,
    }
    .to_string();
    assert!(path_summary.contains("Dynamic plugin 'acme.guardrail' is valid."));
    assert!(path_summary.contains("policy_state: valid"));

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("plugin record");
    let policy = evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
    let id_summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: Some(&entry),
        host_config: host_config_by_id.get("acme.guardrail"),
        policy: &policy,
        trust: &trust,
    }
    .to_string();
    assert!(id_summary.contains("host_config: absent"));
    assert!(id_summary.contains("desired.enabled: false"));

    let missing_validate = validate(
        PluginsValidateCommand {
            target: "missing.plugin".into(),
            json: false,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing_validate.contains("not registered"));

    let missing_inspect = inspect(
        PluginsInspectCommand {
            id: "missing.plugin".into(),
            json: false,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing_inspect.contains("not registered"));

    assert_eq!(
        list(
            PluginsListCommand::default(),
            &crate::config::ServerArgs::default()
        )
        .unwrap(),
        ()
    );
}

#[test]
fn enable_disable_and_remove_persist_lifecycle_state() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");
    let server = crate::config::ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    enable(
        PluginsEnableCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let enabled = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("enabled record");
    assert!(enabled.record.spec.enabled);

    disable(
        PluginsDisableCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();
    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let disabled = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("disabled record");
    assert!(!disabled.record.spec.enabled);

    remove(
        PluginsRemoveCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();
    let resolved = resolve_plugins_config(None).unwrap();
    assert!(resolved.dynamic_plugins.is_empty());
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let removed = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("removed record");
    assert!(removed.record.is_tombstoned());

    let all_records = collect_records(&scopes, true);
    let host_config_by_id = host_config_by_id(&resolved);
    let all_list = PluginListView {
        records: &all_records,
        host_config_by_id: &host_config_by_id,
    }
    .to_string();
    assert!(all_list.contains("acme.guardrail"));
    assert!(all_list.contains("tombstoned"));

    let error = enable(
        PluginsEnableCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .expect_err("tombstoned plugin should not enable");
    match error {
        CliError::PluginLifecycle {
            kind: PluginLifecycleFailureKind::Refused,
            message,
            ..
        } => assert!(message.contains("tombstoned")),
        other => panic!("unexpected tombstone enable error: {other}"),
    }
}

#[test]
fn add_with_explicit_config_uses_sibling_plugins_and_state_files() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join("custom-config");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.explicit");

    let server = ServerArgs {
        config: Some(config_dir.join("gateway.toml")),
        ..ServerArgs::default()
    };

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs::default(),
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = config_dir.join("plugins.toml");
    let state_path = config_dir.join(".dynamic-plugins.json");
    assert!(plugins_toml.exists());
    assert!(state_path.exists());

    let resolved = resolve_plugins_config(server.config.as_ref()).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.explicit");

    let scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.explicit")
        .unwrap()
        .expect("explicit-scope record");
    assert_eq!(entry.scope.to_string(), "explicit");
    assert_eq!(entry.plugins_toml_path, plugins_toml);
    assert_eq!(entry.state_path, state_path);
}

#[test]
fn hydrate_bootstraps_registry_records_from_existing_dynamic_plugin_refs() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.bootstrap");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);

    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.bootstrap")
        .unwrap()
        .expect("hydrated record");
    assert_eq!(entry.scope.to_string(), "project");
    assert_eq!(entry.record.metadata.id, "acme.bootstrap");
    assert!(entry.record.spec.present);
    assert!(!entry.record.spec.enabled);
    let canonical_manifest_path = std::fs::canonicalize(&manifest_path).unwrap();
    assert_eq!(
        entry.record.source.manifest_ref.as_deref(),
        Some(canonical_manifest_path.to_string_lossy().as_ref())
    );
}

#[test]
fn hydrate_applies_host_policy_status_to_discovered_dynamic_plugins() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.policy");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n"
            ),
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.policy")
        .unwrap()
        .expect("hydrated record");

    assert_eq!(
        entry.record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        entry
            .record
            .status
            .startup_class
            .map(|value| value.to_string()),
        Some("required".into())
    );
    assert_eq!(
        entry
            .record
            .status
            .attestation_mode
            .map(|value| value.to_string()),
        Some("signature_required".into())
    );
    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("signature verification")
            || entry
                .record
                .status
                .last_error
                .as_ref()
                .unwrap()
                .message
                .contains("integrity.signature")
    );
}

#[test]
fn hydrate_persists_updated_policy_and_error_state() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.persist-blocked");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir.clone(),
        },
        &ServerArgs::default(),
    )
    .unwrap();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "allowed = false\n"
            ),
            plugin_dir.join("relay-plugin.toml").to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let _ = load_and_hydrate_scopes(None, &resolved).unwrap();

    let state_path = config_dir.join(".dynamic-plugins.json");
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
    let record = &state["records"][0];
    assert_eq!(
        record["metadata"]["id"],
        serde_json::json!("acme.persist-blocked")
    );
    assert_eq!(
        record["status"]["validation"]["policy_satisfied"],
        serde_json::json!("invalid")
    );
    assert_eq!(
        record["status"]["last_error"]["phase"],
        serde_json::json!("policy")
    );
}

#[test]
fn hydrate_verifies_signatures_when_host_policy_provides_trusted_keys() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    let trusted_public_key = write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n",
                "trusted_public_keys = [{:?}]\n"
            ),
            manifest_path.to_string_lossy(),
            trusted_public_key
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.signed")
        .unwrap()
        .expect("hydrated signed record");

    assert_eq!(
        entry.record.status.validation.integrity,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Valid
    );
    assert!(entry.record.status.last_error.is_none());
}

#[test]
fn hydrate_marks_signature_required_plugins_invalid_without_trusted_keys() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed-without-trust",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "attestation = \"signature_required\"\n"
            ),
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.signed-without-trust")
        .unwrap()
        .expect("hydrated signed record");

    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("no trusted_public_keys")
    );
}

#[test]
fn hydrate_marks_signature_required_plugins_invalid_with_wrong_trusted_key() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed-wrong-key",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");
    let wrong_public_key = generate_ed25519_public_key();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "attestation = \"signature_required\"\n",
                "trusted_public_keys = [{:?}]\n"
            ),
            manifest_path.to_string_lossy(),
            wrong_public_key
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.signed-wrong-key")
        .unwrap()
        .expect("hydrated signed record");

    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("failed signature verification")
    );
}

#[test]
fn hydrate_marks_malformed_signature_files_invalid_when_signature_is_present() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed-malformed",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    std::fs::write(plugin_dir.join("plugin.py.sig"), "ed25519:not-base64\n").unwrap();
    let trusted_public_key = generate_ed25519_public_key();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "attestation = \"signature_if_present\"\n",
                "trusted_public_keys = [{:?}]\n"
            ),
            manifest_path.to_string_lossy(),
            trusted_public_key
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.signed-malformed")
        .unwrap()
        .expect("hydrated signed record");

    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("invalid base64 signature")
    );
}

#[test]
fn enable_refuses_dynamic_plugins_blocked_by_host_policy_and_persists_status() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.enable-blocked");
    let server = crate::config::ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "allowed = false\n"
            ),
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let error = enable(
        PluginsEnableCommand {
            id: "acme.enable-blocked".into(),
        },
        &server,
    )
    .unwrap_err();

    match error {
        CliError::PluginLifecycle {
            kind: PluginLifecycleFailureKind::Refused,
            ref message,
            ..
        } => assert!(message.contains("blocked by host policy")),
        other => panic!("unexpected enable policy error: {other}"),
    }
    let (command, target, kind, code, _) = error
        .as_plugin_lifecycle_error_context()
        .expect("plugin lifecycle error context");
    assert_eq!(command, "plugins enable");
    assert_eq!(target, Some("acme.enable-blocked"));
    assert_eq!(kind, PluginLifecycleFailureKind::Refused);
    assert_eq!(code, Some("policy_blocked"));

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.enable-blocked")
        .unwrap()
        .expect("policy-updated record");
    assert!(!entry.record.spec.enabled);
    assert_eq!(
        entry.record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Invalid
    );
    assert_eq!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .map(|error| error.phase.to_string()),
        Some("policy".into())
    );
}

#[test]
fn disable_succeeds_when_registered_plugin_manifest_is_unreadable() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");
    let server = crate::config::ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir.clone(),
        },
        &server,
    )
    .unwrap();

    enable(
        PluginsEnableCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    std::fs::remove_file(plugin_dir.join("relay-plugin.toml")).unwrap();

    disable(
        PluginsDisableCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    let scopes = load_scoped_registries(None).unwrap();
    let entry = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("disabled plugin record");
    assert!(!entry.record.spec.enabled);
}

#[test]
fn validate_marks_registered_plugins_invalid_when_host_policy_blocks_them() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.validate-blocked");
    let server = crate::config::ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n",
                "allowed = false\n"
            ),
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    validate(
        PluginsValidateCommand {
            target: "acme.validate-blocked".into(),
            json: false,
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.validate-blocked")
        .unwrap()
        .expect("policy-updated record");

    assert_eq!(
        entry.record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Invalid
    );
    assert_eq!(
        entry.record.status.validation.message.as_deref(),
        Some("validated by CLI")
    );
    let (blocked_manifest, blocked_manifest_ref) =
        DynamicPluginManifest::load_from_path(&manifest_path)
            .map_err(|error| CliError::Config(error.to_string()))
            .unwrap();
    let blocked_policy =
        evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &blocked_manifest);
    let blocked_trust =
        evaluate_dynamic_plugin_trust(&blocked_manifest, &blocked_manifest_ref, &blocked_policy);
    let blocked_summary = PluginValidationSummaryView {
        manifest: &blocked_manifest,
        manifest_ref: &blocked_manifest_ref,
        entry: Some(&entry),
        host_config: None,
        policy: &blocked_policy,
        trust: &blocked_trust,
    }
    .to_string();
    assert!(blocked_summary.contains("host policy blocks it"));
    assert!(blocked_summary.contains("policy_state: invalid"));
    let blocked_list = PluginListView {
        records: std::slice::from_ref(&entry),
        host_config_by_id: &std::collections::HashMap::new(),
    }
    .to_string();
    assert!(blocked_list.contains("POLICY"));
    assert!(blocked_list.contains("invalid"));
    let blocked_validate_value = serde_json::to_value(responses::validate_success(
        responses::ValidateResponseInput {
            command: "plugins validate",
            target: Some("acme.validate-blocked"),
            target_kind: "plugin_id",
            resolved_plugin_id: Some("acme.validate-blocked"),
            manifest: &blocked_manifest,
            manifest_ref: &blocked_manifest_ref,
            entry: Some(&entry),
            host_config: None,
            policy: &blocked_policy,
            trust: &blocked_trust,
        },
    ))
    .unwrap();
    assert_eq!(
        blocked_validate_value["data"]["valid"],
        serde_json::json!(false)
    );
    assert_eq!(
        blocked_validate_value["data"]["policy_state"],
        serde_json::json!("invalid")
    );
    assert!(
        blocked_validate_value["data"]["errors"][0]
            .as_str()
            .unwrap()
            .contains("blocked by host policy")
    );
    assert_eq!(
        entry
            .record
            .status
            .startup_class
            .map(|value| value.to_string()),
        Some("required".into())
    );
    assert_eq!(
        entry
            .record
            .status
            .attestation_mode
            .map(|value| value.to_string()),
        Some("signature_required".into())
    );
    assert_eq!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .map(|error| error.phase.to_string()),
        Some("policy".into())
    );
}

#[test]
fn add_can_revive_tombstoned_records() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.revive");
    let server = crate::config::ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir.clone(),
        },
        &server,
    )
    .unwrap();

    remove(
        PluginsRemoveCommand {
            id: "acme.revive".into(),
        },
        &server,
    )
    .unwrap();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let revived = find_record_by_id(&scopes, "acme.revive")
        .unwrap()
        .expect("revived record");
    assert!(!revived.record.is_tombstoned());
    assert!(revived.record.spec.present);
}

#[test]
fn json_helpers_emit_stable_success_and_failure_shapes() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.json");
    let server = ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let records = collect_records(&scopes, false);
    let entry = find_record_by_id(&scopes, "acme.json")
        .unwrap()
        .expect("json record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let list_value = serde_json::to_value(responses::list_success(
        "plugins list",
        None,
        &records,
        &host_config_by_id,
    ))
    .unwrap();
    assert_eq!(list_value["schema_version"], serde_json::json!(1));
    assert_eq!(list_value["ok"], serde_json::json!(true));
    assert_eq!(list_value["data"][0]["id"], serde_json::json!("acme.json"));

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.json",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.json"),
    ))
    .unwrap();
    assert_eq!(inspect_value["data"]["id"], serde_json::json!("acme.json"));
    assert_eq!(
        inspect_value["data"]["source"]["manifest_ref"],
        serde_json::json!(manifest_ref)
    );
    assert_eq!(
        inspect_value["data"]["host_config"],
        serde_json::Value::Null
    );

    let validate_policy =
        evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
    let validate_trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &validate_policy);
    let validate_value = serde_json::to_value(responses::validate_success(
        responses::ValidateResponseInput {
            command: "plugins validate",
            target: Some("acme.json"),
            target_kind: "plugin_id",
            resolved_plugin_id: Some("acme.json"),
            manifest: &manifest,
            manifest_ref: &manifest_ref,
            entry: Some(&entry),
            host_config: host_config_by_id.get("acme.json"),
            policy: &validate_policy,
            trust: &validate_trust,
        },
    ))
    .unwrap();
    assert_eq!(
        validate_value["data"]["target_kind"],
        serde_json::json!("plugin_id")
    );
    assert_eq!(validate_value["data"]["valid"], serde_json::json!(true));
    assert_eq!(
        validate_value["data"]["policy_state"],
        serde_json::json!("valid")
    );
    assert_eq!(
        validate_value["data"]["startup_class"],
        serde_json::json!("optional")
    );
    assert_eq!(
        validate_value["data"]["attestation_mode"],
        serde_json::json!("integrity_only")
    );

    let failure = serde_json::to_value(responses::failure(
        "plugins inspect",
        Some("missing.plugin"),
        PluginLifecycleFailureKind::NotFound,
        None,
        "missing plugin",
    ))
    .unwrap();
    assert_eq!(failure["ok"], serde_json::json!(false));
    assert_eq!(failure["error"]["code"], serde_json::json!("not_found"));

    let refused = serde_json::to_value(responses::failure(
        "plugins add",
        Some("acme.blocked"),
        PluginLifecycleFailureKind::Refused,
        Some("policy_blocked"),
        "blocked by host policy",
    ))
    .unwrap();
    assert_eq!(
        refused["error"]["code"],
        serde_json::json!("policy_blocked")
    );
}

#[test]
fn remove_tolerates_unreadable_non_target_manifest_entries() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let broken_dir = temp.path().join("plugins").join("broken");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&broken_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.guardrail");
    let server = ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n\n[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy(),
            broken_dir.join("missing.toml").to_string_lossy()
        ),
    )
    .unwrap();

    remove(
        PluginsRemoveCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(!rendered.contains("acme.guardrail"));
    assert!(rendered.contains("missing.toml"));
}

#[test]
fn remove_reports_malformed_dynamic_plugin_containers() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugins_toml = temp.path().join("plugins.toml");

    std::fs::write(&plugins_toml, "[plugins]\ndynamic = \"oops\"\n").unwrap();
    let error = remove_dynamic_plugin_reference(&plugins_toml, "acme.guardrail", None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("plugins.dynamic must be an array of tables"));

    std::fs::write(&plugins_toml, "plugins = \"oops\"\n").unwrap();
    let error = remove_dynamic_plugin_reference(&plugins_toml, "acme.guardrail", None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("[plugins] must be a table"));
}

#[test]
fn append_reports_malformed_dynamic_plugin_containers() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugins_toml = temp.path().join("plugins.toml");

    std::fs::write(&plugins_toml, "plugins = \"oops\"\n").unwrap();
    let error = append_dynamic_plugin_reference(&plugins_toml, "/tmp/plugin/relay-plugin.toml")
        .unwrap_err()
        .to_string();
    assert!(error.contains("[plugins] must be a table"));

    std::fs::write(&plugins_toml, "[plugins]\ndynamic = \"oops\"\n").unwrap();
    let error = append_dynamic_plugin_reference(&plugins_toml, "/tmp/plugin/relay-plugin.toml")
        .unwrap_err()
        .to_string();
    assert!(error.contains("plugins.dynamic must be an array of tables"));
}

#[test]
fn remove_matches_relative_target_manifest_refs_without_loading_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let config_dir = temp.path().join(".nemo-relay");
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();

    let manifest_path = plugin_dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrail"
kind = "worker"
"#,
    )
    .unwrap();

    let plugins_toml = config_dir.join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        "[[plugins.dynamic]]\nmanifest = \"../plugins/acme/relay-plugin.toml\"\n",
    )
    .unwrap();

    std::fs::remove_file(&manifest_path).unwrap();

    let removed = remove_dynamic_plugin_reference(
        &plugins_toml,
        "acme.guardrail",
        Some("../plugins/acme/relay-plugin.toml"),
    )
    .unwrap();
    assert!(removed);
    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(!rendered.contains("relay-plugin.toml"));
}

#[test]
fn inspect_redacts_host_config_values() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.redacted");
    let server = ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\nconfig = {{ api_key = \"secret-token\", region = \"us-west-2\" }}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.redacted")
        .unwrap()
        .expect("redacted record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let inspect_output = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.redacted"),
    }
    .to_string();
    assert!(!inspect_output.contains("secret-token"));
    let inspect_output: serde_yaml::Value = serde_yaml::from_str(&inspect_output).unwrap();
    assert_eq!(
        inspect_output["host_config"]["api_key"].as_str(),
        Some("<redacted>")
    );
    assert_eq!(
        inspect_output["host_config"]["region"].as_str(),
        Some("<redacted>")
    );

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.redacted",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.redacted"),
    ))
    .unwrap();
    assert_eq!(
        inspect_value["data"]["host_config"]["api_key"],
        serde_json::json!("<redacted>")
    );
    assert_eq!(
        inspect_value["data"]["host_config"]["region"],
        serde_json::json!("<redacted>")
    );
    assert_eq!(inspect_value["data"]["host_config_status"], "present");
}

#[test]
fn inspect_distinguishes_empty_host_config_from_missing_host_config() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.empty-config");
    let server = ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\nconfig = {{}}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.empty-config")
        .unwrap()
        .expect("empty-config record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let inspect_output = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.empty-config"),
    }
    .to_string();
    let inspect_output: serde_yaml::Value = serde_yaml::from_str(&inspect_output).unwrap();
    assert_eq!(
        inspect_output["host_config_status"].as_str(),
        Some("present")
    );
    assert_eq!(
        inspect_output["host_config"]
            .as_mapping()
            .expect("empty host config should render as an object")
            .len(),
        0
    );

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.empty-config",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.empty-config"),
    ))
    .unwrap();
    assert_eq!(inspect_value["data"]["host_config_status"], "present");
    assert_eq!(
        inspect_value["data"]["host_config"]
            .as_object()
            .expect("empty host config should serialize as an object")
            .len(),
        0
    );
}
