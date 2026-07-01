// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::http::HeaderValue;
use base64::Engine;
use nemo_relay::plugin::dynamic::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginKind,
    DynamicPluginManifest, DynamicPluginStartupClass,
};
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::plugins::policy::{
    DynamicPluginHostPolicy, DynamicPluginHostPolicyEffect, DynamicPluginHostPolicyRule,
    evaluate_dynamic_plugin_host_policy,
};

fn config() -> GatewayConfig {
    GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),

        anthropic_base_url: "http://anthropic".into(),
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::config::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::config::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    }
}

fn isolated_config_path(temp: &tempfile::TempDir) -> std::path::PathBuf {
    temp.path().join("config.toml")
}

fn write_dynamic_manifest(dir: &std::path::Path, plugin_id: &str) -> std::path::PathBuf {
    write_dynamic_manifest_with_options(dir, plugin_id, &["plugin_worker"], None)
}

fn write_dynamic_manifest_with_options(
    dir: &std::path::Path,
    plugin_id: &str,
    capabilities: &[&str],
    signature_ref: Option<&str>,
) -> std::path::PathBuf {
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
relay = "0.1"
worker_protocol = "grpc-v1"

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

fn write_detached_ed25519_signature(dir: &std::path::Path, signature_name: &str) -> String {
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

fn write_dynamic_plugin_state(plugins_toml_path: &std::path::Path, plugin_id: &str, enabled: bool) {
    let manifest_ref = plugins_toml_path
        .parent()
        .unwrap()
        .join("plugins/acme/relay-plugin.toml");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_ref).unwrap();
    let mut record = manifest.into_record(Some(manifest_ref)).unwrap();
    assert_eq!(record.metadata.id, plugin_id);
    record.spec.enabled = enabled;
    record.status.validation.policy_satisfied = DynamicPluginCheckState::Unknown;
    std::fs::write(
        plugins_toml_path
            .parent()
            .unwrap()
            .join(".dynamic-plugins.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "records": [record],
        }))
        .unwrap(),
    )
    .unwrap();
}

fn read_dynamic_plugin_state(
    plugins_toml_path: &std::path::Path,
) -> nemo_relay::plugin::dynamic::DynamicPluginRecord {
    let persisted: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            plugins_toml_path
                .parent()
                .unwrap()
                .join(".dynamic-plugins.json"),
        )
        .unwrap(),
    )
    .unwrap();
    serde_json::from_value(persisted["records"][0].clone()).unwrap()
}

#[test]
fn session_config_prefers_headers_and_parses_json() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-relay-config-profile",
        HeaderValue::from_static("profile-a"),
    );
    headers.insert(
        "x-nemo-relay-session-metadata",
        HeaderValue::from_static(r#"{"team":"obs"}"#),
    );
    headers.insert(
        "x-nemo-relay-plugin-config",
        HeaderValue::from_static(r#"{"components":[]}"#),
    );
    headers.insert(
        "x-nemo-relay-gateway-mode",
        HeaderValue::from_static("required"),
    );

    let session = config().session_config_from_headers(&headers);

    assert_eq!(session.profile.as_deref(), Some("profile-a"));
    assert_eq!(session.metadata, Some(json!({ "team": "obs" })));
    assert_eq!(session.plugin_config, Some(json!({ "components": [] })));
    assert_eq!(session.gateway_mode.as_deref(), Some("required"));
}

#[test]
fn session_config_uses_defaults_and_ignores_bad_json() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-relay-session-metadata",
        HeaderValue::from_static("not-json"),
    );
    headers.insert("x-empty", HeaderValue::from_static(""));

    let session = config().session_config_from_headers(&headers);

    assert_eq!(session.metadata, None);
    assert_eq!(header_string(&headers, "x-empty"), None);
}

#[test]
fn agent_and_gateway_mode_arguments_are_stable() {
    assert_eq!(CodingAgent::ClaudeCode.hook_path(), "/hooks/claude-code");
    assert_eq!(CodingAgent::Codex.hook_path(), "/hooks/codex");
    assert_eq!(CodingAgent::Hermes.hook_path(), "/hooks/hermes");
    assert_eq!(GatewayMode::HookOnly.as_arg(), "hook-only");
    assert_eq!(GatewayMode::Passthrough.as_arg(), "passthrough");
    assert_eq!(GatewayMode::Required.as_arg(), "required");
}

#[test]
fn agent_inference_uses_executable_basename() {
    assert_eq!(
        CodingAgent::infer("/opt/bin/claude"),
        Some(CodingAgent::ClaudeCode)
    );
    assert_eq!(CodingAgent::infer("codex"), Some(CodingAgent::Codex));
    assert_eq!(CodingAgent::infer("cursor-agent"), None);
    assert_eq!(CodingAgent::infer("hermes"), Some(CodingAgent::Hermes));
    assert_eq!(CodingAgent::infer("wrapper"), None);
}

#[test]
fn explicit_toml_config_maps_supported_sections() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://openai"
anthropic_base_url = "http://anthropic"

[gateway]
max_hook_payload_bytes = 12345
max_passthrough_body_bytes = 67890

[agents.claude]
command = "claude"

[agents.codex]
command = "codex --approval-mode never"

[agents.hermes]
command = "hermes --yolo chat"
"#,
    )
    .unwrap();
    let command = RunCommand {
        agent: None,
        config: Some(path),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec![],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.gateway.bind.to_string(), "127.0.0.1:0");
    assert_eq!(resolved.gateway.openai_base_url, "http://openai");
    assert_eq!(resolved.gateway.anthropic_base_url, "http://anthropic");
    assert_eq!(resolved.gateway.max_hook_payload_bytes, 12345);
    assert_eq!(resolved.gateway.max_passthrough_body_bytes, 67890);
    assert_eq!(resolved.gateway.metadata, None);
    assert_eq!(resolved.gateway.plugin_config, None);
    assert_eq!(
        resolved.agents.codex.command.as_deref(),
        Some("codex --approval-mode never")
    );
    assert_eq!(
        resolved.agents.hermes.command.as_deref(),
        Some("hermes --yolo chat")
    );
}

#[test]
fn legacy_observability_config_sections_fail_clearly() {
    let temp = tempfile::tempdir().unwrap();
    for (name, contents, expected) in [
        (
            "exporters.toml",
            "[exporters]\natof_dir = \"atof\"\n",
            "[exporters]",
        ),
        (
            "observability.toml",
            "[observability]\natif_dir = \"atif\"\n",
            "[observability]",
        ),
        (
            "openinference.toml",
            "[export.openinference]\nendpoint = \"http://localhost:4318\"\n",
            "[export.openinference]",
        ),
    ] {
        let path = temp.path().join(name);
        std::fs::write(&path, contents).unwrap();
        let command = RunCommand {
            agent: None,
            config: Some(path),
            openai_base_url: None,
            anthropic_base_url: None,
            session_metadata: None,
            plugin_config_path: None,
            dry_run: false,
            print: false,
            command: vec![],
        };

        let error = resolve_run_config(&command, None).unwrap_err().to_string();

        assert!(error.contains("legacy observability config"));
        assert!(error.contains(expected));
        assert!(error.contains("plugins.toml"));
        assert!(error.contains("nemo-relay plugins edit"));
    }
}

#[test]
fn explicit_plugins_toml_maps_root_plugin_config() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
[upstream]
openai_base_url = "http://openai"
"#,
    )
    .unwrap();
    std::fs::write(
        temp.path().join("plugins.toml"),
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "atof"
filename = "events.jsonl"
mode = "overwrite"
"#,
    )
    .unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(config_path),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(
        resolved.gateway.plugin_config,
        Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "atof": {
                            "enabled": true,
                            "output_directory": "atof",
                            "filename": "events.jsonl",
                            "mode": "overwrite"
                        }
                    }
                }
            ]
        }))
    );
}

#[test]
fn plugins_toml_path_resolution_tracks_config_scope() {
    let temp = tempfile::tempdir().unwrap();
    let explicit = temp.path().join("custom-config.toml");
    assert_eq!(
        plugin_config_paths(Some(&explicit), None),
        vec![temp.path().join("plugins.toml")]
    );

    let project = temp.path().join("workspace");
    let nested = project.join("a/b/c");
    std::fs::create_dir_all(project.join(".nemo-relay")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    let plugin_path = project.join(".nemo-relay/plugins.toml");
    std::fs::write(&plugin_path, "version = 1").unwrap();
    let user_config = temp.path().join("xdg/nemo-relay");

    assert_eq!(find_project_plugin_config(&nested), Some(plugin_path));
    assert_eq!(
        project_plugin_config_path(&nested),
        project.join(".nemo-relay/plugins.toml")
    );
    assert_eq!(
        implicit_plugin_config_paths(Some(&nested), Some(user_config.clone())),
        vec![
            PathBuf::from("/etc/nemo-relay/plugins.toml"),
            project.join(".nemo-relay/plugins.toml"),
            user_config.join("plugins.toml"),
        ]
    );

    std::fs::remove_file(project.join(".nemo-relay/plugins.toml")).unwrap();
    std::fs::write(project.join(".nemo-relay/config.toml"), "").unwrap();
    assert_eq!(find_project_plugin_config(&nested), None);
    assert_eq!(
        project_plugin_config_path(&nested),
        project.join(".nemo-relay/plugins.toml")
    );
}

#[test]
fn discovered_plugins_toml_upserts_components_by_kind() {
    let temp = tempfile::tempdir().unwrap();
    let project_plugin = temp.path().join("project-plugins.toml");
    let user_plugin = temp.path().join("user-plugins.toml");
    std::fs::write(
        &project_plugin,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
filename = "project.jsonl"

[[components]]
kind = "adaptive"
enabled = true

[components.config]
mode = "project-only"
"#,
    )
    .unwrap();
    std::fs::write(
        &user_plugin,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true

[components.config.atif]
enabled = true
filename_template = "user-{session_id}.json"

[[components]]
kind = "custom"
enabled = true

[components.config]
source = "user"
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![project_plugin, user_plugin]).unwrap();

    assert_eq!(
        resolved.map(|config| config.value),
        Some(Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "atof": {
                            "enabled": true,
                            "filename": "project.jsonl"
                        },
                        "atif": {
                            "enabled": true,
                            "filename_template": "user-{session_id}.json"
                        }
                    }
                },
                {
                    "kind": "adaptive",
                    "enabled": true,
                    "config": {
                        "mode": "project-only"
                    }
                },
                {
                    "kind": "custom",
                    "enabled": true,
                    "config": {
                        "source": "user"
                    }
                }
            ]
        })))
    );
}

#[test]
fn discovered_pricing_plugin_sources_layer_user_before_lower_priority_sources() {
    let temp = tempfile::tempdir().unwrap();
    let system_plugin = temp.path().join("system-plugins.toml");
    let user_plugin = temp.path().join("user-plugins.toml");
    std::fs::write(
        &system_plugin,
        r#"
version = 1

[[components]]
kind = "pricing"
enabled = true

[[components.config.sources]]
type = "file"
path = "/etc/nemo-relay/pricing.json"
"#,
    )
    .unwrap();
    std::fs::write(
        &user_plugin,
        r#"
version = 1

[[components]]
kind = "pricing"
enabled = true

[[components.config.sources]]
type = "file"
path = "/home/user/.config/nemo-relay/pricing.json"
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![system_plugin, user_plugin]).unwrap();

    assert_eq!(
        resolved.map(|config| config.value),
        Some(Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "pricing",
                    "enabled": true,
                    "config": {
                        "sources": [
                            {
                                "type": "file",
                                "path": "/home/user/.config/nemo-relay/pricing.json"
                            },
                            {
                                "type": "file",
                                "path": "/etc/nemo-relay/pricing.json"
                            }
                        ]
                    }
                }
            ]
        })))
    );
}

#[test]
fn discovered_plugins_toml_can_disable_lower_priority_observability_section() {
    let temp = tempfile::tempdir().unwrap();
    let project_plugin = temp.path().join("project-plugins.toml");
    let user_plugin = temp.path().join("user-plugins.toml");
    std::fs::write(
        &project_plugin,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "project-atof"
mode = "overwrite"
"#,
    )
    .unwrap();
    std::fs::write(
        &user_plugin,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = false
mode = "append"
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![project_plugin, user_plugin]).unwrap();

    assert_eq!(
        resolved.map(|config| config.value),
        Some(Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "atof": {
                            "enabled": false,
                            "output_directory": "project-atof",
                            "mode": "append"
                        }
                    }
                }
            ]
        })))
    );
}

#[test]
fn plugins_toml_resolves_dynamic_plugin_refs_without_polluting_runtime_plugin_config() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.worker");
    let plugins_path = temp.path().join("plugins.toml");
    std::fs::write(
        &plugins_path,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"

[plugins.dynamic.config]
mode = "strict"
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![plugins_path])
        .unwrap()
        .unwrap();

    assert_eq!(
        resolved.value,
        Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1
                    }
                }
            ]
        }))
    );
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.worker");
    assert_eq!(
        resolved.dynamic_plugins[0].manifest_ref,
        manifest_path.canonicalize().unwrap().to_string_lossy()
    );
    assert_eq!(
        resolved.dynamic_plugins[0].config,
        serde_json::Map::from_iter([("mode".into(), json!("strict"))])
    );
}

#[test]
fn plugins_toml_resolves_dynamic_plugin_refs_from_absolute_manifest_paths() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.worker");
    let plugins_path = temp.path().join("plugins.toml");
    std::fs::write(
        &plugins_path,
        format!(
            r#"
[[plugins.dynamic]]
manifest = '{}'
"#,
            manifest_path.display()
        ),
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![plugins_path])
        .unwrap()
        .unwrap();

    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.worker");
    assert_eq!(
        resolved.dynamic_plugins[0].manifest_ref,
        manifest_path.canonicalize().unwrap().to_string_lossy()
    );
}

#[test]
fn plugins_toml_resolves_dynamic_plugin_host_policy_without_polluting_runtime_plugin_config() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let plugins_path = temp.path().join("plugins.toml");
    std::fs::write(
        &plugins_path,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[plugins.policy.defaults]
startup = "optional"
attestation = "integrity_only"
trusted_public_keys = ["ed25519:ZmFrZS1rZXk="]

[[plugins.policy.rules]]
match_kind = "worker"
startup = "required"

[plugins.policy.overrides."acme.worker"]
attestation = "signature_required"
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![plugins_path])
        .unwrap()
        .unwrap();

    assert_eq!(
        resolved.value,
        Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1
                    }
                }
            ]
        }))
    );
    assert_eq!(
        resolved.dynamic_plugin_policy.defaults.startup,
        Some(DynamicPluginStartupClass::Optional)
    );
    assert_eq!(
        resolved.dynamic_plugin_policy.defaults.attestation,
        Some(DynamicPluginAttestationMode::IntegrityOnly)
    );
    assert_eq!(
        resolved.dynamic_plugin_policy.defaults.trusted_public_keys,
        Some(vec!["ed25519:ZmFrZS1rZXk=".into()])
    );
    assert_eq!(resolved.dynamic_plugin_policy.rules.len(), 1);
    assert_eq!(
        resolved.dynamic_plugin_policy.rules[0].match_kind,
        Some(DynamicPluginKind::Worker)
    );
    assert_eq!(
        resolved.dynamic_plugin_policy.rules[0].effect.startup,
        Some(DynamicPluginStartupClass::Required)
    );
    assert_eq!(
        resolved
            .dynamic_plugin_policy
            .overrides
            .get("acme.worker")
            .and_then(|effect| effect.attestation),
        Some(DynamicPluginAttestationMode::SignatureRequired)
    );
}

#[test]
fn dynamic_plugin_host_policy_evaluator_applies_rules_before_plugin_overrides() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.worker");
    let (manifest, _) = DynamicPluginManifest::load_from_path(&manifest_path).unwrap();
    let policy = DynamicPluginHostPolicy {
        defaults: DynamicPluginHostPolicyEffect {
            allowed: Some(true),
            startup: Some(DynamicPluginStartupClass::Optional),
            attestation: Some(DynamicPluginAttestationMode::IntegrityOnly),
            trusted_public_keys: None,
        },
        rules: vec![DynamicPluginHostPolicyRule {
            match_kind: Some(DynamicPluginKind::Worker),
            match_plugin_id: None,
            effect: DynamicPluginHostPolicyEffect {
                allowed: None,
                startup: Some(DynamicPluginStartupClass::Required),
                attestation: None,
                trusted_public_keys: None,
            },
        }],
        overrides: std::iter::once((
            "acme.worker".into(),
            DynamicPluginHostPolicyEffect {
                allowed: Some(false),
                startup: None,
                attestation: Some(DynamicPluginAttestationMode::SignatureRequired),
                trusted_public_keys: None,
            },
        ))
        .collect(),
    };

    let evaluated = evaluate_dynamic_plugin_host_policy(&policy, &manifest);

    assert!(!evaluated.policy_satisfied);
    assert_eq!(evaluated.startup_class, DynamicPluginStartupClass::Required);
    assert_eq!(
        evaluated.attestation_mode,
        DynamicPluginAttestationMode::SignatureRequired
    );
    assert!(
        evaluated
            .failure()
            .map(|failure| failure.display(manifest.plugin.id.as_str()).to_string())
            .unwrap()
            .contains("blocked by host policy")
    );
}

#[test]
fn plugins_toml_layers_dynamic_plugin_host_policy_across_sources() {
    let temp = tempfile::tempdir().unwrap();
    let project_plugins = temp.path().join("project-plugins.toml");
    let user_plugins = temp.path().join("user-plugins.toml");
    std::fs::write(
        &project_plugins,
        r#"
[plugins.policy.defaults]
startup = "required"

[[plugins.policy.rules]]
match_kind = "worker"
startup = "required"

[plugins.policy.overrides."acme.worker"]
attestation = "signature_if_present"
"#,
    )
    .unwrap();
    std::fs::write(
        &user_plugins,
        r#"
[plugins.policy.defaults]
attestation = "signature_required"

[[plugins.policy.rules]]
match_plugin_id = "acme.worker"
allowed = false

[plugins.policy.overrides."acme.worker"]
allowed = true
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![project_plugins, user_plugins])
        .unwrap()
        .unwrap();

    assert_eq!(resolved.value, None);
    assert_eq!(
        resolved.dynamic_plugin_policy.defaults.startup,
        Some(DynamicPluginStartupClass::Required)
    );
    assert_eq!(
        resolved.dynamic_plugin_policy.defaults.attestation,
        Some(DynamicPluginAttestationMode::SignatureRequired)
    );
    assert_eq!(resolved.dynamic_plugin_policy.rules.len(), 2);
    assert_eq!(
        resolved.dynamic_plugin_policy.rules[0].match_kind,
        Some(DynamicPluginKind::Worker)
    );
    assert_eq!(
        resolved.dynamic_plugin_policy.rules[1]
            .match_plugin_id
            .as_deref(),
        Some("acme.worker")
    );
    let override_effect = resolved
        .dynamic_plugin_policy
        .overrides
        .get("acme.worker")
        .expect("merged override");
    assert_eq!(
        override_effect.attestation,
        Some(DynamicPluginAttestationMode::SignatureIfPresent)
    );
    assert_eq!(override_effect.allowed, Some(true));
}

#[test]
fn plugins_toml_rejects_duplicate_dynamic_plugin_ids_across_sources() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let project_plugin = temp.path().join("project-plugins.toml");
    let user_plugin = temp.path().join("user-plugins.toml");
    std::fs::write(
        &project_plugin,
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"
"#,
    )
    .unwrap();
    std::fs::write(
        &user_plugin,
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme"
"#,
    )
    .unwrap();

    let error = load_plugin_toml_config_from_paths(vec![project_plugin, user_plugin])
        .unwrap_err()
        .to_string();

    assert!(error.contains("duplicate dynamic plugin id"));
    assert!(error.contains("acme.worker"));
}

#[test]
fn plugins_toml_rejects_duplicate_dynamic_plugin_ids_within_one_file() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_a = temp.path().join("plugins/a");
    let plugin_b = temp.path().join("plugins/b");
    std::fs::create_dir_all(&plugin_a).unwrap();
    std::fs::create_dir_all(&plugin_b).unwrap();
    write_dynamic_manifest(&plugin_a, "acme.worker");
    write_dynamic_manifest(&plugin_b, "acme.worker");
    let plugins_path = temp.path().join("plugins.toml");
    std::fs::write(
        &plugins_path,
        r#"
[[plugins.dynamic]]
manifest = "plugins/a/relay-plugin.toml"

[[plugins.dynamic]]
manifest = "plugins/b/relay-plugin.toml"
"#,
    )
    .unwrap();

    let error = load_plugin_toml_config_from_paths(vec![plugins_path])
        .unwrap_err()
        .to_string();

    assert!(error.contains("duplicate dynamic plugin id"));
    assert!(error.contains("acme.worker"));
}

#[test]
fn plugins_toml_rejects_dynamic_plugin_lifecycle_fields() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let plugins_path = temp.path().join("plugins.toml");
    std::fs::write(
        &plugins_path,
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"
enabled = true
"#,
    )
    .unwrap();

    let error = load_plugin_toml_config_from_paths(vec![plugins_path])
        .unwrap_err()
        .to_string();

    assert!(error.contains("invalid dynamic plugin config"));
    assert!(error.contains("enabled"));
}

#[test]
fn plugins_toml_layers_runtime_plugin_config_and_dynamic_only_sources_independently() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let lower_priority = temp.path().join("lower-plugins.toml");
    let higher_priority = temp.path().join("higher-plugins.toml");
    std::fs::write(
        &lower_priority,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1
"#,
    )
    .unwrap();
    std::fs::write(
        &higher_priority,
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"
"#,
    )
    .unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![lower_priority, higher_priority])
        .unwrap()
        .unwrap();

    assert_eq!(
        resolved.value,
        Some(json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1
                    }
                }
            ]
        }))
    );
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.worker");
}

#[test]
fn plugins_toml_rejects_duplicate_component_kinds_per_file() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_path = temp.path().join("plugins.toml");
    std::fs::write(
        &plugin_path,
        r#"
version = 1

[[components]]
kind = "observability"

[[components]]
kind = "observability"
"#,
    )
    .unwrap();

    let error = load_plugin_toml_config_from_paths(vec![plugin_path])
        .unwrap_err()
        .to_string();

    assert!(error.contains("duplicate plugin component kind"));
    assert!(error.contains("observability"));
}

#[test]
fn config_toml_plugin_configuration_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
[plugins]
config = { version = 1, components = [] }
"#,
    )
    .unwrap();
    let args = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("plugin configuration"));
    assert!(error.contains("no longer supported"));
    assert!(error.contains("plugins.toml"));
}

#[test]
fn plugin_config_path_overrides_sibling_plugin_file() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    let sibling_path = temp.path().join("plugins.toml");
    let override_path = temp.path().join("override.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(&sibling_path, "version = 1\n").unwrap();
    std::fs::write(&override_path, "version = 2\n").unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(config_path),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: Some(override_path),
        dry_run: true,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(
        resolved.gateway.plugin_config,
        Some(json!({ "version": 2 }))
    );
}

#[test]
fn cli_run_overrides_config_values() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://file-openai"
"#,
    )
    .unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(path),
        openai_base_url: Some("http://cli-openai".into()),
        anthropic_base_url: None,
        session_metadata: Some(r#"{"team":"cli"}"#.into()),
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://cli-openai");
    assert_eq!(resolved.gateway.metadata, Some(json!({ "team": "cli" })));
}

#[test]
fn run_inherits_top_level_server_flags_when_subcommand_flags_are_absent() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://file-openai"
"#,
    )
    .unwrap();
    let server = ServerArgs {
        config: Some(path),
        openai_base_url: Some("http://top-level-openai".into()),
        ..ServerArgs::default()
    };
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: None,
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, Some(&server)).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://top-level-openai");
}

#[test]
fn server_resolution_applies_all_server_overrides() {
    let temp = tempfile::tempdir().unwrap();
    let args = ServerArgs {
        config: Some(isolated_config_path(&temp)),
        bind: Some("127.0.0.1:0".parse().unwrap()),
        openai_base_url: Some("http://cli-openai".into()),
        anthropic_base_url: Some("http://cli-anthropic".into()),
        plugin_config_path: None,
        max_hook_payload_bytes: Some(222),
        max_passthrough_body_bytes: Some(333),
    };

    let resolved = resolve_server_config(&args).unwrap();

    assert_eq!(resolved.gateway.bind.to_string(), "127.0.0.1:0");
    assert_eq!(resolved.gateway.openai_base_url, "http://cli-openai");
    assert_eq!(resolved.gateway.anthropic_base_url, "http://cli-anthropic");
    assert_eq!(resolved.gateway.max_hook_payload_bytes, 222);
    assert_eq!(resolved.gateway.max_passthrough_body_bytes, 333);
    assert_eq!(resolved.gateway.plugin_config, None);
    assert!(args.requested_daemon_mode());
}

#[test]
fn server_resolution_fails_when_required_enabled_dynamic_plugin_is_blocked_by_policy() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let config_path = temp.path().join("config.toml");
    let plugins_toml_path = temp.path().join("plugins.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        &plugins_toml_path,
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"

[plugins.policy.defaults]
startup = "required"
allowed = false
"#,
    )
    .unwrap();
    write_dynamic_plugin_state(&plugins_toml_path, "acme.worker", true);
    let args = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("required dynamic plugin startup preflight failed"));
    assert!(error.contains("acme.worker"));
    assert!(error.contains("blocked by host policy"));
}

#[test]
fn server_resolution_fails_when_required_enabled_dynamic_plugin_fails_integrity() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    std::fs::write(
        plugin_dir.join("plugin.py"),
        "def register():\n    return 'tampered'\n",
    )
    .unwrap();
    let config_path = temp.path().join("config.toml");
    let plugins_toml_path = temp.path().join("plugins.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        &plugins_toml_path,
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"

[plugins.policy.defaults]
startup = "required"
"#,
    )
    .unwrap();
    write_dynamic_plugin_state(&plugins_toml_path, "acme.worker", true);

    let args = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("required dynamic plugin startup preflight failed"));
    assert!(error.contains("acme.worker"));
    assert!(error.contains("integrity verification"));

    let record = read_dynamic_plugin_state(&plugins_toml_path);
    assert_eq!(
        record.status.validation.integrity,
        DynamicPluginCheckState::Invalid
    );
    assert_eq!(
        record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        record.status.startup_class,
        Some(DynamicPluginStartupClass::Required)
    );
    assert_eq!(
        record.status.attestation_mode,
        Some(DynamicPluginAttestationMode::IntegrityOnly)
    );
    assert!(
        record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("integrity verification")
    );
}

#[test]
fn server_resolution_fails_when_required_enabled_dynamic_plugin_lacks_trusted_keys() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.worker",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");
    let config_path = temp.path().join("config.toml");
    let plugins_toml_path = temp.path().join("plugins.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        &plugins_toml_path,
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"

[plugins.policy.defaults]
startup = "required"
attestation = "signature_required"
"#,
    )
    .unwrap();
    write_dynamic_plugin_state(&plugins_toml_path, "acme.worker", true);

    let args = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("required dynamic plugin startup preflight failed"));
    assert!(error.contains("acme.worker"));
    assert!(error.contains("no trusted_public_keys"));

    let record = read_dynamic_plugin_state(&plugins_toml_path);
    assert_eq!(
        record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert_eq!(
        record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        record.status.startup_class,
        Some(DynamicPluginStartupClass::Required)
    );
    assert_eq!(
        record.status.attestation_mode,
        Some(DynamicPluginAttestationMode::SignatureRequired)
    );
    assert!(
        record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("no trusted_public_keys")
    );
}

#[test]
fn server_resolution_fails_when_required_enabled_dynamic_plugin_has_wrong_trusted_key() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.worker",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");
    let wrong_public_key = generate_ed25519_public_key();
    let config_path = temp.path().join("config.toml");
    let plugins_toml_path = temp.path().join("plugins.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        &plugins_toml_path,
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = \"plugins/acme/relay-plugin.toml\"\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n",
                "trusted_public_keys = [{:?}]\n"
            ),
            wrong_public_key
        ),
    )
    .unwrap();
    write_dynamic_plugin_state(&plugins_toml_path, "acme.worker", true);

    let args = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("required dynamic plugin startup preflight failed"));
    assert!(error.contains("acme.worker"));
    assert!(error.contains("failed signature verification"));

    let record = read_dynamic_plugin_state(&plugins_toml_path);
    assert_eq!(
        record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert_eq!(
        record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        record.status.startup_class,
        Some(DynamicPluginStartupClass::Required)
    );
    assert_eq!(
        record.status.attestation_mode,
        Some(DynamicPluginAttestationMode::SignatureRequired)
    );
    assert!(
        record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("failed signature verification")
    );
}

#[test]
fn server_resolution_fails_when_required_enabled_dynamic_plugin_has_malformed_signature() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.worker",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    std::fs::write(plugin_dir.join("plugin.py.sig"), "ed25519:not-base64\n").unwrap();
    let trusted_public_key = generate_ed25519_public_key();
    let config_path = temp.path().join("config.toml");
    let plugins_toml_path = temp.path().join("plugins.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        &plugins_toml_path,
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = \"plugins/acme/relay-plugin.toml\"\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_if_present\"\n",
                "trusted_public_keys = [{:?}]\n"
            ),
            trusted_public_key
        ),
    )
    .unwrap();
    write_dynamic_plugin_state(&plugins_toml_path, "acme.worker", true);

    let args = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("required dynamic plugin startup preflight failed"));
    assert!(error.contains("acme.worker"));
    assert!(error.contains("invalid base64 signature"));

    let record = read_dynamic_plugin_state(&plugins_toml_path);
    assert_eq!(
        record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert_eq!(
        record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        record.status.startup_class,
        Some(DynamicPluginStartupClass::Required)
    );
    assert_eq!(
        record.status.attestation_mode,
        Some(DynamicPluginAttestationMode::SignatureIfPresent)
    );
    assert!(
        record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("invalid base64 signature")
    );
}

#[test]
fn gateway_body_limit_defaults_are_stable() {
    let gateway = GatewayConfig::default();

    assert_eq!(
        gateway.max_hook_payload_bytes,
        crate::config::DEFAULT_MAX_HOOK_PAYLOAD_BYTES
    );
    assert_eq!(
        gateway.max_passthrough_body_bytes,
        crate::config::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES
    );
}

#[test]
fn gateway_body_limit_file_values_must_be_nonzero() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    for (field, expected) in [
        ("max_hook_payload_bytes", "gateway.max_hook_payload_bytes"),
        (
            "max_passthrough_body_bytes",
            "gateway.max_passthrough_body_bytes",
        ),
    ] {
        std::fs::write(&path, format!("[gateway]\n{field} = 0\n")).unwrap();
        let args = ServerArgs {
            config: Some(path.clone()),
            ..ServerArgs::default()
        };

        let error = resolve_server_config(&args).unwrap_err().to_string();

        assert!(error.contains(expected));
        assert!(error.contains("greater than 0"));
    }
}

#[test]
fn run_resolution_applies_all_run_overrides() {
    let temp = tempfile::tempdir().unwrap();
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(isolated_config_path(&temp)),
        openai_base_url: Some("http://run-openai".into()),
        anthropic_base_url: Some("http://run-anthropic".into()),
        session_metadata: Some(r#"{"team":"run"}"#.into()),
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let resolved = resolve_run_config(&command, None).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://run-openai");
    assert_eq!(resolved.gateway.anthropic_base_url, "http://run-anthropic");
    assert_eq!(resolved.gateway.metadata, Some(json!({ "team": "run" })));
}

#[test]
fn run_resolution_fails_when_required_enabled_dynamic_plugin_is_blocked_by_policy() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let config_path = temp.path().join("config.toml");
    let plugins_toml_path = temp.path().join("plugins.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        &plugins_toml_path,
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"

[plugins.policy.defaults]
startup = "required"
allowed = false
"#,
    )
    .unwrap();
    write_dynamic_plugin_state(&plugins_toml_path, "acme.worker", true);
    let command = RunCommand {
        agent: Some(CodingAgent::Codex),
        config: Some(config_path),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: vec!["codex".into()],
    };

    let error = resolve_run_config(&command, None).unwrap_err().to_string();

    assert!(error.contains("required dynamic plugin startup preflight failed"));
    assert!(error.contains("acme.worker"));
    assert!(error.contains("blocked by host policy"));
}

#[test]
fn malformed_shared_config_reports_context() {
    let temp = tempfile::tempdir().unwrap();
    let invalid_toml = temp.path().join("invalid.toml");
    std::fs::write(&invalid_toml, "server = [").unwrap();
    let args = ServerArgs {
        config: Some(invalid_toml),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid TOML"));

    let invalid_shape = temp.path().join("invalid-shape.toml");
    std::fs::write(&invalid_shape, "upstream = \"not-a-table\"").unwrap();
    let args = ServerArgs {
        config: Some(invalid_shape),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid gateway configuration shape"));

    let plugin_config = temp.path().join("config-with-invalid-plugins.toml");
    std::fs::write(&plugin_config, "").unwrap();
    std::fs::write(temp.path().join("plugins.toml"), "version = [").unwrap();
    let args = ServerArgs {
        config: Some(plugin_config),
        ..ServerArgs::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid plugin TOML"));
}

#[test]
fn recursive_toml_merge_replaces_scalars_and_preserves_tables() {
    let mut left: toml::Value = r#"
[upstream]
openai_base_url = "http://old"
anthropic_base_url = "http://anthropic"

[runtime.policy]
version = 1
policy = { unknown_component = "warn", unknown_field = "warn" }
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();
    let right: toml::Value = r#"
[upstream]
openai_base_url = "http://new"

[runtime.policy.policy]
unknown_component = "error"
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();

    merge_toml(&mut left, right);

    assert_eq!(
        left["upstream"]["openai_base_url"].as_str(),
        Some("http://new")
    );
    assert_eq!(
        left["upstream"]["anthropic_base_url"].as_str(),
        Some("http://anthropic")
    );
    assert_eq!(
        left["runtime"]["policy"]["policy"]["unknown_component"].as_str(),
        Some("error")
    );
    assert_eq!(
        left["runtime"]["policy"]["policy"]["unknown_field"].as_str(),
        Some("warn")
    );
}
