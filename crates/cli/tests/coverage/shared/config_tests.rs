// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::hooks::GatewayMode;
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
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::MutexGuard;

use crate::plugins::policy::{
    DynamicPluginHostPolicy, DynamicPluginHostPolicyEffect, DynamicPluginHostPolicyRule,
    evaluate_dynamic_plugin_host_policy,
};

struct PluginConfigDiscoveryScope {
    _cwd_guard: crate::test_support::CwdTestScope,
    _guard: MutexGuard<'static, ()>,
    previous_cwd: PathBuf,
    previous_xdg_config_home: Option<OsString>,
    previous_config_scope: Option<OsString>,
    previous_openai_api_key: Option<OsString>,
    previous_bootstrap_fingerprint: Option<OsString>,
    previous_plugin_idle_timeout: Option<OsString>,
}

impl PluginConfigDiscoveryScope {
    fn enter(cwd: &std::path::Path, xdg_config_home: &std::path::Path) -> Self {
        let cwd_guard = crate::test_support::CwdTestScope::locked();
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous_cwd = std::env::current_dir().unwrap();
        let previous_xdg_config_home = std::env::var_os("XDG_CONFIG_HOME");
        let previous_config_scope = std::env::var_os("NEMO_RELAY_CONFIG_SCOPE");
        let previous_openai_api_key = std::env::var_os("OPENAI_API_KEY");
        let previous_bootstrap_fingerprint = std::env::var_os(BOOTSTRAP_FINGERPRINT_ENV);
        let previous_plugin_idle_timeout = std::env::var_os(PLUGIN_IDLE_TIMEOUT_ENV);
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", xdg_config_home);
            std::env::remove_var("NEMO_RELAY_CONFIG_SCOPE");
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var(BOOTSTRAP_FINGERPRINT_ENV);
            std::env::remove_var(PLUGIN_IDLE_TIMEOUT_ENV);
        }
        std::env::set_current_dir(cwd).unwrap();
        Self {
            _cwd_guard: cwd_guard,
            _guard: guard,
            previous_cwd,
            previous_xdg_config_home,
            previous_config_scope,
            previous_openai_api_key,
            previous_bootstrap_fingerprint,
            previous_plugin_idle_timeout,
        }
    }

    fn enable_user_scope(&self) {
        // SAFETY: This scope holds the process-wide environment mutex.
        unsafe {
            std::env::set_var("NEMO_RELAY_CONFIG_SCOPE", "user");
        }
    }

    fn set_bootstrap_fingerprint(&self, fingerprint: &str) {
        // SAFETY: This scope holds the process-wide environment mutex.
        unsafe {
            std::env::set_var(BOOTSTRAP_FINGERPRINT_ENV, fingerprint);
        }
    }
}

impl Drop for PluginConfigDiscoveryScope {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.previous_cwd).unwrap();
        unsafe {
            match self.previous_xdg_config_home.take() {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
            match self.previous_config_scope.take() {
                Some(value) => std::env::set_var("NEMO_RELAY_CONFIG_SCOPE", value),
                None => std::env::remove_var("NEMO_RELAY_CONFIG_SCOPE"),
            }
            match self.previous_openai_api_key.take() {
                Some(value) => std::env::set_var("OPENAI_API_KEY", value),
                None => std::env::remove_var("OPENAI_API_KEY"),
            }
            match self.previous_bootstrap_fingerprint.take() {
                Some(value) => std::env::set_var(BOOTSTRAP_FINGERPRINT_ENV, value),
                None => std::env::remove_var(BOOTSTRAP_FINGERPRINT_ENV),
            }
            match self.previous_plugin_idle_timeout.take() {
                Some(value) => std::env::set_var(PLUGIN_IDLE_TIMEOUT_ENV, value),
                None => std::env::remove_var(PLUGIN_IDLE_TIMEOUT_ENV),
            }
        }
    }
}

fn config() -> GatewayConfig {
    GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),

        anthropic_base_url: "http://anthropic".into(),
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::configuration::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    }
}

#[test]
fn effective_plugin_toml_sources_reports_empty_and_sorted_contributors() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&xdg).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&project, &xdg);

    assert_eq!(
        effective_plugin_toml_sources().unwrap(),
        Vec::<PathBuf>::new()
    );

    let project_plugins = project.join(".nemo-relay/plugins.toml");
    let user_plugins = xdg.join("nemo-relay/plugins.toml");
    std::fs::create_dir_all(project_plugins.parent().unwrap()).unwrap();
    std::fs::create_dir_all(user_plugins.parent().unwrap()).unwrap();
    std::fs::write(&project_plugins, "version = 1\ncomponents = []\n").unwrap();
    std::fs::write(&user_plugins, "version = 1\ncomponents = []\n").unwrap();

    let sources = effective_plugin_toml_sources().unwrap();
    assert!(sources.is_sorted());
    assert!(sources.windows(2).all(|paths| paths[0] != paths[1]));

    let mut actual = sources
        .iter()
        .map(|path| path.canonicalize().unwrap())
        .collect::<Vec<_>>();
    actual.sort();
    let mut expected = [project_plugins, user_plugins]
        .iter()
        .map(|path| path.canonicalize().unwrap())
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(actual, expected);
}

fn isolated_config_path(temp: &tempfile::TempDir) -> std::path::PathBuf {
    temp.path().join("config.toml")
}

fn write_attested_python_environment(path: &std::path::Path, manifest_path: &std::path::Path) {
    let interpreter = if cfg!(windows) {
        path.join("Scripts/python.exe")
    } else {
        path.join("bin/python")
    };
    std::fs::create_dir_all(interpreter.parent().unwrap()).unwrap();
    std::fs::write(interpreter, b"fixture interpreter").unwrap();
    let installed = path.join("site-packages/fixture.py");
    std::fs::create_dir_all(installed.parent().unwrap()).unwrap();
    std::fs::write(installed, b"fixture = True\n").unwrap();
    let (manifest, _) = DynamicPluginManifest::load_from_path(manifest_path).unwrap();
    let source_artifact_sha256 = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
        .unwrap();
    crate::plugins::lifecycle::attest_test_python_environment(path, source_artifact_sha256)
        .unwrap();
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
manifest_root = "."
artifact = "plugin.py"

[integrity]
sha256 = "{digest}"
{signature_line}

[load]
runtime = "python"
entrypoint = "plugin:register"
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
    let command = RunOverrides {
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
fn explicit_config_must_exist() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("missing-config.toml");
    let command = RunOverrides {
        agent: None,
        config: Some(path.clone()),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: true,
        print: false,
        command: vec![],
    };

    let error = resolve_run_config(&command, None).unwrap_err().to_string();

    assert!(error.contains("does not exist"), "{error}");
    assert!(error.contains(path.to_string_lossy().as_ref()), "{error}");
}

#[test]
fn absent_optional_plugin_config_is_ignored() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("plugins.toml");

    let loaded = load_plugin_toml_config_from_paths(vec![missing]).unwrap();

    assert!(loaded.is_none());
}

#[cfg(unix)]
#[test]
fn unreadable_config_errors_include_the_source_path() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::set_permissions(&config_path, std::fs::Permissions::from_mode(0o000)).unwrap();
    let command = RunOverrides {
        agent: None,
        config: Some(config_path.clone()),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: true,
        print: false,
        command: vec![],
    };
    let config_error = resolve_run_config(&command, None).unwrap_err().to_string();
    std::fs::set_permissions(&config_path, std::fs::Permissions::from_mode(0o600)).unwrap();

    assert!(config_error.contains("failed to read configuration file"));
    assert!(
        config_error.contains(config_path.to_string_lossy().as_ref()),
        "{config_error}"
    );

    let plugins_path = temp.path().join("plugins.toml");
    std::fs::write(&plugins_path, "version = 1\n").unwrap();
    std::fs::set_permissions(&plugins_path, std::fs::Permissions::from_mode(0o000)).unwrap();
    let plugin_error = load_plugin_toml_config_from_paths(vec![plugins_path.clone()])
        .unwrap_err()
        .to_string();
    std::fs::set_permissions(&plugins_path, std::fs::Permissions::from_mode(0o600)).unwrap();

    assert!(plugin_error.contains("failed to read plugin configuration file"));
    assert!(
        plugin_error.contains(plugins_path.to_string_lossy().as_ref()),
        "{plugin_error}"
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
        let command = RunOverrides {
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
    let command = RunOverrides {
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
fn persistent_user_scope_excludes_project_gateway_and_plugin_layers() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("workspace");
    let nested = project.join("nested");
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(project.join(".nemo-relay")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(project.join(".nemo-relay/config.toml"), "").unwrap();
    std::fs::write(project.join(".nemo-relay/plugins.toml"), "version = 1\n").unwrap();
    let scope = PluginConfigDiscoveryScope::enter(&nested, &xdg);
    scope.enable_user_scope();

    assert_eq!(
        config_paths(None),
        vec![
            PathBuf::from("/etc/nemo-relay/config.toml"),
            xdg.join("nemo-relay/config.toml"),
        ]
    );
    assert_eq!(
        plugin_config_paths(None, None),
        vec![
            PathBuf::from("/etc/nemo-relay/plugins.toml"),
            xdg.join("nemo-relay/plugins.toml"),
        ]
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

    let resolved = load_plugin_toml_config_from_paths(vec![plugins_path.clone()])
        .unwrap()
        .unwrap();

    assert!(resolved.contributing_sources.contains(&plugins_path));

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

    let resolved = load_plugin_toml_config_from_paths(vec![plugins_path.clone()])
        .unwrap()
        .unwrap();

    assert!(resolved.contributing_sources.contains(&plugins_path));

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

    let resolved =
        load_plugin_toml_config_from_paths(vec![project_plugins.clone(), user_plugins.clone()])
            .unwrap()
            .unwrap();

    assert_eq!(resolved.value, None);
    assert_eq!(
        resolved.contributing_sources,
        vec![project_plugins, user_plugins],
        "policy-only layers still affect runtime dynamic-plugin behavior"
    );
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
    let args = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
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
    let command = RunOverrides {
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
    let command = RunOverrides {
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
    let server = GatewayOverrides {
        config: Some(path),
        openai_base_url: Some("http://top-level-openai".into()),
        ..GatewayOverrides::default()
    };
    let command = RunOverrides {
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
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let config_path = isolated_config_path(&temp);
    std::fs::write(&config_path, "").unwrap();
    let args = GatewayOverrides {
        config: Some(config_path),
        bind: Some("127.0.0.1:0".parse().unwrap()),
        openai_base_url: Some("http://cli-openai".into()),
        anthropic_base_url: Some("http://cli-anthropic".into()),
        plugin_config_path: None,
        ready_file: None,
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
    assert_eq!(resolved.bootstrap_fingerprint, None);
    assert!(
        !xdg.join("nemo-relay/bootstrap/fingerprint-hmac.key")
            .exists()
    );
    assert!(args.requested_daemon_mode());
}

#[test]
fn ordinary_server_ignores_managed_bootstrap_fingerprint_environment() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    scope.set_bootstrap_fingerprint("opaque-parent-fingerprint");
    let config_path = isolated_config_path(&temp);
    std::fs::write(&config_path, "").unwrap();

    let args = GatewayOverrides {
        config: Some(config_path),
        bind: Some("127.0.0.1:0".parse().unwrap()),
        ..GatewayOverrides::default()
    };
    let resolved = resolve_server_config(&args).unwrap();

    assert_eq!(resolved.bootstrap_fingerprint, None);
    assert!(
        managed_bootstrap_identity(&args, &resolved, &[])
            .unwrap()
            .is_none()
    );
    assert!(
        !xdg.join("nemo-relay/bootstrap/fingerprint-hmac.key")
            .exists()
    );

    scope.set_bootstrap_fingerprint("");
    let managed_args = GatewayOverrides {
        ready_file: Some(temp.path().join("managed.ready.json")),
        ..args
    };
    assert!(
        managed_bootstrap_identity(&managed_args, &resolved, &[])
            .unwrap()
            .is_none()
    );
}

#[test]
fn managed_bootstrap_environment_is_not_forwarded_from_codex() {
    let names = crate::mcp_environment::forwarded_names(
        [
            "NEMO_RELAY_BOOTSTRAP_AGENT".to_string(),
            "NEMO_RELAY_BOOTSTRAP_FINGERPRINT".to_string(),
            "NEMO_RELAY_BOOTSTRAP_STATE_DIR".to_string(),
            "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN".to_string(),
        ],
        None,
    );

    assert!(!names.iter().any(|name| name.contains("BOOTSTRAP")));
}

#[test]
fn mcp_environment_policy_handles_unresolved_values_and_historical_names_per_platform() {
    assert!(
        crate::mcp_environment::unresolved_self_placeholder_for_platform(
            "AWS_ROLE_ARN",
            "${AWS_ROLE_ARN}",
            false,
        )
    );
    assert!(
        !crate::mcp_environment::unresolved_self_placeholder_for_platform(
            "AWS_ROLE_ARN",
            "${aws_role_arn}",
            false,
        )
    );
    assert!(
        crate::mcp_environment::unresolved_self_placeholder_for_platform(
            "AWS_ROLE_ARN",
            "${aws_role_arn}",
            true,
        )
    );
    assert!(
        !crate::mcp_environment::unresolved_self_placeholder_for_platform(
            "AWS_ROLE_ARN",
            "real-value",
            true,
        )
    );

    for allowed in ["AWS_PROFILE", "NEMO_RELAY_CUSTOM", "OTEL_CUSTOM"] {
        assert!(
            crate::mcp_environment::previously_forwardable_name_for_platform(allowed, false),
            "rejected {allowed}"
        );
    }
    assert!(crate::mcp_environment::previously_forwardable_name_for_platform("Aws_Custom", true,));
    for rejected in [
        "UNRELATED_SECRET",
        "NEMO_RELAY_WORKER_TOKEN",
        "NEMO_RELAY_TEST_CAPTURE",
    ] {
        assert!(
            !crate::mcp_environment::previously_forwardable_name_for_platform(rejected, true),
            "accepted {rejected}"
        );
    }
}

#[test]
fn transparent_gateway_fingerprint_is_stable_and_endpoint_specific() {
    let first = transparent_gateway_fingerprint("http://127.0.0.1:41001");
    let repeated = transparent_gateway_fingerprint("http://127.0.0.1:41001");
    let second = transparent_gateway_fingerprint("http://127.0.0.1:41002");

    assert_eq!(first, repeated);
    assert_ne!(first, second);
    assert!(first.starts_with("transparent-sha256:"));
    assert_eq!(first.len(), "transparent-sha256:".len() + 64);
}

#[test]
fn bootstrap_health_proofs_and_client_tokens_reject_every_malformed_shape() {
    let key = BootstrapChallengeKey::from_bytes(&[7_u8; BOOTSTRAP_HMAC_KEY_BYTES]);
    let other = BootstrapChallengeKey::from_bytes(&[8_u8; BOOTSTRAP_HMAC_KEY_BYTES]);
    let fingerprint = "hmac-sha256:fixture";
    let nonce = "nonce";
    let proof = key.proof(fingerprint, nonce);

    assert!(key.verify(fingerprint, nonce, &proof));
    assert!(!key.verify("other", nonce, &proof));
    assert!(!key.verify(fingerprint, "other", &proof));
    for malformed in [
        "missing-prefix",
        "hmac-sha256:short",
        "hmac-sha256:zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
    ] {
        assert!(!key.verify(fingerprint, nonce, malformed));
    }

    let token = key.client_token();
    assert!(key.verify_client_token(&token));
    assert!(!other.verify_client_token(&token));
    for malformed in [
        "missing-prefix",
        "hmac-sha256:short",
        "hmac-sha256:zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
    ] {
        assert!(!key.verify_client_token(malformed));
    }

    assert!(
        !verify_python_environment_attestation("source", "environment", "missing-prefix").unwrap()
    );
    assert!(
        !verify_python_environment_attestation("source", "environment", "hmac-sha256:short")
            .unwrap()
    );
}

#[test]
fn bootstrap_hmac_state_reports_invalid_path_and_existing_key_shapes() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);

    let root_error = load_or_create_bootstrap_hmac_key_at(Path::new("/")).unwrap_err();
    assert!(
        root_error.to_string().contains("no parent directory"),
        "{root_error}"
    );

    let parent_file = temp.path().join("not-a-directory");
    std::fs::write(&parent_file, b"file").unwrap();
    let parent_error = load_or_create_bootstrap_hmac_key_at(&parent_file.join("key")).unwrap_err();
    assert!(
        parent_error.to_string().contains("failed to create"),
        "{parent_error}"
    );

    let directory_key = temp.path().join("directory-key");
    std::fs::create_dir(&directory_key).unwrap();
    let open_error = load_or_create_bootstrap_hmac_key_at(&directory_key).unwrap_err();
    assert!(
        open_error.to_string().contains("failed to open"),
        "{open_error}"
    );

    let configured_key = xdg
        .join("nemo-relay")
        .join("bootstrap")
        .join("fingerprint-hmac.key");
    std::fs::create_dir_all(configured_key.parent().unwrap()).unwrap();
    std::fs::write(&configured_key, b"short").unwrap();
    let existing_error = BootstrapChallengeKey::load_existing()
        .err()
        .expect("corrupt existing bootstrap key was accepted");
    assert!(
        existing_error.to_string().contains("invalid length 5"),
        "{existing_error}"
    );
}

#[cfg(unix)]
#[test]
fn bounded_identity_reader_reports_missing_unreadable_and_invalid_utf8_inputs() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing");
    let missing_error = read_bounded_regular_file(&missing, "fixture").unwrap_err();
    assert!(
        missing_error.contains("failed to inspect"),
        "{missing_error}"
    );

    let unreadable = temp.path().join("unreadable");
    std::fs::write(&unreadable, b"contents").unwrap();
    std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o000)).unwrap();
    let unreadable_result = read_bounded_regular_file(&unreadable, "fixture");
    std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o600)).unwrap();
    let unreadable_error = unreadable_result.unwrap_err();
    assert!(
        unreadable_error.contains("failed to read"),
        "{unreadable_error}"
    );

    let manifest = temp.path().join("invalid-utf8.toml");
    std::fs::write(&manifest, [0xff_u8]).unwrap();
    let utf8_error = load_bounded_dynamic_plugin_manifest_bytes(&manifest).unwrap_err();
    assert!(
        utf8_error.to_string().contains("is not UTF-8"),
        "{utf8_error}"
    );

    assert_eq!(
        resolve_dynamic_plugin_relative_path(Path::new("/manifest.toml"), "/artifact"),
        PathBuf::from("/artifact")
    );
}

#[test]
fn persistent_server_resolution_excludes_project_config_and_fingerprints_credentials() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(project.join(".nemo-relay")).unwrap();
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::write(
        project.join(".nemo-relay/config.toml"),
        "[upstream]\nopenai_base_url = \"http://project-only\"\n",
    )
    .unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&project, &xdg);
    let args = GatewayOverrides {
        bind: Some("127.0.0.1:47632".parse().unwrap()),
        ..GatewayOverrides::default()
    };

    unsafe { std::env::set_var("OPENAI_API_KEY", "credential-one") };
    let first = resolve_persistent_server_config(&args).unwrap();
    assert_ne!(first.gateway.openai_base_url, "http://project-only");
    assert!(
        first
            .bootstrap_fingerprint
            .as_deref()
            .unwrap()
            .starts_with("hmac-sha256:")
    );
    let key_path = xdg
        .join("nemo-relay")
        .join("bootstrap")
        .join("fingerprint-hmac.key");
    assert_eq!(std::fs::metadata(&key_path).unwrap().len(), 32);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&key_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    unsafe { std::env::set_var("OPENAI_API_KEY", "credential-two") };
    let second = resolve_persistent_server_config(&args).unwrap();
    assert_ne!(first.bootstrap_fingerprint, second.bootstrap_fingerprint);
}

#[test]
fn managed_bootstrap_canonicalizes_unset_and_zero_padded_default_idle_timeout() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let parent = resolve_persistent_server_config(&GatewayOverrides::default()).unwrap();
    let expected = parent.bootstrap_fingerprint.unwrap();
    scope.set_bootstrap_fingerprint(&expected);
    unsafe {
        std::env::set_var(PLUGIN_IDLE_TIMEOUT_ENV, "0300");
    }
    let child_args = GatewayOverrides {
        ready_file: Some(temp.path().join("managed.ready.json")),
        ..GatewayOverrides::default()
    };
    let child = resolve_server_config(&child_args).unwrap();
    let active = active_dynamic_plugin_components(None, &child).unwrap();
    let identity = managed_bootstrap_identity(&child_args, &child, &active)
        .unwrap()
        .unwrap();

    assert_eq!(identity.fingerprint(), expected);
    identity.verify_current().unwrap();
}

#[test]
fn plugin_launch_carries_effective_hook_limit_below_and_above_default() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let user_config = xdg.join("nemo-relay/config.toml");
    std::fs::create_dir_all(user_config.parent().unwrap()).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let bind = "127.0.0.1:47632".parse().unwrap();

    for limit in [1024, DEFAULT_MAX_HOOK_PAYLOAD_BYTES + 4096] {
        std::fs::write(
            &user_config,
            format!("[gateway]\nmax_hook_payload_bytes = {limit}\n"),
        )
        .unwrap();
        let launch =
            crate::bootstrap::resolve_plugin_gateway(&GatewayOverrides::default(), bind).unwrap();
        assert_eq!(launch.max_hook_payload_bytes, limit);
    }
}

#[test]
fn bootstrap_hmac_key_creation_is_concurrency_safe_and_stable() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("state/fingerprint-hmac.key");
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let handles = (0..8)
        .map(|_| {
            let path = path.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                load_or_create_bootstrap_hmac_key_at(&path).unwrap()
            })
        })
        .collect::<Vec<_>>();
    let keys = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();

    assert!(keys.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(std::fs::metadata(path).unwrap().len(), 32);
}

#[cfg(windows)]
#[test]
fn bootstrap_hmac_key_uses_and_repairs_a_private_windows_dacl() {
    let temp = tempfile::tempdir().unwrap();
    set_test_windows_dacl(temp.path(), "D:P(A;;FA;;;WD)");
    let path = temp.path().join("state/fingerprint-hmac.key");

    let original = load_or_create_bootstrap_hmac_key_at(&path).unwrap();

    assert!(crate::filesystem::windows_path_is_private(path.parent().unwrap()).unwrap());
    assert!(crate::filesystem::windows_path_is_private(&path).unwrap());

    set_test_windows_dacl(&path, "D:P(A;;FA;;;WD)");
    assert!(!crate::filesystem::windows_path_is_private(&path).unwrap());

    let reloaded = load_or_create_bootstrap_hmac_key_at(&path).unwrap();

    assert_eq!(reloaded, original);
    assert!(crate::filesystem::windows_path_is_private(&path).unwrap());
}

#[test]
fn bootstrap_hmac_key_lock_wait_is_bounded_under_synchronized_contention() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("state/fingerprint-hmac.key");
    load_or_create_bootstrap_hmac_key_at(&path).unwrap();
    let owner = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    fs2::FileExt::lock_exclusive(&owner).unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let waiter = {
        let barrier = barrier.clone();
        let path = path.clone();
        std::thread::spawn(move || {
            barrier.wait();
            load_or_create_bootstrap_hmac_key_at_with_timeout(&path, Duration::from_millis(75))
                .unwrap_err()
        })
    };
    barrier.wait();

    let error = waiter.join().unwrap();

    assert!(error.to_string().contains("timed out waiting"));
    drop(owner);
}

#[test]
fn persistent_fingerprint_tracks_active_dynamic_plugin_and_file_identity() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let plugin_dir = temp.path().join("plugin");
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.bootstrap-identity");
    let environment_a = temp.path().join("managed/environment-a");
    let environment_b = temp.path().join("managed/environment-b");
    write_attested_python_environment(&environment_a, &manifest_path);
    write_attested_python_environment(&environment_b, &manifest_path);
    let resolved = ResolvedConfig {
        gateway: config(),
        ..ResolvedConfig::default()
    };
    let active = ActiveDynamicPluginComponent {
        plugin_id: "acme.bootstrap-identity".into(),
        kind: DynamicPluginKind::Worker,
        lifecycle_generation: 7,
        manifest_ref: Some(manifest_path.to_string_lossy().into_owned()),
        environment_ref: Some(environment_a.to_string_lossy().into_owned()),
        config: Map::new(),
        activation_snapshot: None,
    };

    let inactive = persistent_bootstrap_fingerprint(&resolved, &[]).unwrap();
    let enabled =
        persistent_bootstrap_fingerprint(&resolved, std::slice::from_ref(&active)).unwrap();
    assert_ne!(
        inactive, enabled,
        "enable/disable or tombstone must conflict"
    );

    std::fs::write(plugin_dir.join("plugin.py"), b"changed artifact").unwrap();
    let artifact_changed =
        persistent_bootstrap_fingerprint(&resolved, std::slice::from_ref(&active)).unwrap();
    assert_ne!(enabled, artifact_changed);

    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    std::fs::write(
        &manifest_path,
        manifest.replace(
            "id = \"acme.bootstrap-identity\"",
            "id = \"acme.bootstrap-identity\"\nname = \"changed manifest\"",
        ),
    )
    .unwrap();
    let manifest_changed =
        persistent_bootstrap_fingerprint(&resolved, std::slice::from_ref(&active)).unwrap();
    assert_ne!(artifact_changed, manifest_changed);

    let mut rebuilt_environment = active.clone();
    rebuilt_environment.lifecycle_generation += 1;
    let rebuilt_environment_fingerprint =
        persistent_bootstrap_fingerprint(&resolved, &[rebuilt_environment]).unwrap();
    assert_ne!(
        manifest_changed, rebuilt_environment_fingerprint,
        "a same-path managed environment rebuild must conflict through lifecycle generation"
    );

    let mut environment_changed = active;
    environment_changed.environment_ref = Some(environment_b.to_string_lossy().into_owned());
    let environment_changed =
        persistent_bootstrap_fingerprint(&resolved, &[environment_changed]).unwrap();
    assert_ne!(manifest_changed, environment_changed);
}

#[test]
fn persistent_hook_identity_authenticates_python_marker_without_rehashing_environment() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let xdg = temp.path().join("xdg");
    let user_config = xdg.join("nemo-relay");
    let plugin_dir = temp.path().join("python-plugin");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&user_config).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&project, &xdg);
    let plugin_id = "acme.read-only-hook-identity";
    let manifest_path = write_dynamic_manifest(&plugin_dir, plugin_id);
    let plugins_toml = user_config.join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "version = 1\n\n[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();
    let environment_name = Sha256::digest(plugin_id.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let environment = user_config
        .join(".dynamic-plugin-environments")
        .join(environment_name);
    write_attested_python_environment(&environment, &manifest_path);
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path).unwrap();
    let mut record = manifest.into_record(Some(manifest_ref)).unwrap();
    record.spec.enabled = true;
    record.source.environment_ref = Some(environment.to_string_lossy().into_owned());
    std::fs::write(
        user_config.join(".dynamic-plugins.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "records": [record],
        }))
        .unwrap(),
    )
    .unwrap();

    crate::plugins::lifecycle::reset_test_python_environment_digest_calls();
    let before = resolve_persistent_server_config(&GatewayOverrides::default()).unwrap();
    assert_eq!(
        crate::plugins::lifecycle::test_python_environment_digest_calls(),
        0,
        "persistent hook identity must trust only the authenticated environment marker"
    );

    let resolved = load_shared_config_scoped(None, None, true).unwrap();
    let active = active_dynamic_plugin_components(None, &resolved).unwrap();
    assert_eq!(active.len(), 1);
    assert!(active[0].activation_snapshot.is_some());
    let snapshot_fingerprint = persistent_bootstrap_fingerprint(&resolved, &active).unwrap();
    assert!(snapshot_fingerprint.starts_with("hmac-sha256:"));
    assert!(
        crate::plugins::lifecycle::test_python_environment_digest_calls() > 0,
        "activation must verify the complete environment before snapshotting it"
    );
    crate::plugins::lifecycle::reset_test_python_environment_digest_calls();

    std::fs::write(
        environment.join("site-packages/fixture.py"),
        b"fixture = 'mutated'\n",
    )
    .unwrap();
    let after = resolve_persistent_server_config(&GatewayOverrides::default()).unwrap();
    assert_eq!(
        crate::plugins::lifecycle::test_python_environment_digest_calls(),
        0,
        "mutating environment content must not make hook preflight traverse it"
    );
    assert_eq!(before.bootstrap_fingerprint, after.bootstrap_fingerprint);

    let error = active_dynamic_plugin_components(None, &resolved)
        .unwrap_err()
        .to_string();
    assert!(error.contains("changed after provisioning"), "{error}");
    assert!(
        crate::plugins::lifecycle::test_python_environment_digest_calls() > 0,
        "sidecar activation must still perform the full environment verification"
    );
}

#[test]
fn managed_server_rejects_config_and_artifact_changes_after_parent_resolution() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let plugin_dir = temp.path().join("plugin");
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.bootstrap-race");
    let environment = temp.path().join("managed/environment");
    write_attested_python_environment(&environment, &manifest_path);
    let resolved = ResolvedConfig {
        gateway: config(),
        ..ResolvedConfig::default()
    };
    let active = ActiveDynamicPluginComponent {
        plugin_id: "acme.bootstrap-race".into(),
        kind: DynamicPluginKind::Worker,
        lifecycle_generation: 3,
        manifest_ref: Some(manifest_path.to_string_lossy().into_owned()),
        environment_ref: Some(environment.to_string_lossy().into_owned()),
        config: Map::new(),
        activation_snapshot: None,
    };
    let expected =
        persistent_bootstrap_fingerprint(&resolved, std::slice::from_ref(&active)).unwrap();
    scope.set_bootstrap_fingerprint(&expected);
    let args = GatewayOverrides {
        ready_file: Some(temp.path().join("managed.ready.json")),
        ..GatewayOverrides::default()
    };

    let identity = managed_bootstrap_identity(&args, &resolved, std::slice::from_ref(&active))
        .unwrap()
        .unwrap();
    assert_eq!(identity.fingerprint(), expected);

    let mut changed_config = resolved.clone();
    changed_config.gateway.openai_base_url = "https://changed.invalid/v1".into();
    let config_error =
        managed_bootstrap_identity(&args, &changed_config, std::slice::from_ref(&active))
            .unwrap_err();
    assert!(config_error.to_string().contains("identity changed"));

    std::fs::write(plugin_dir.join("plugin.py"), b"changed during bootstrap").unwrap();
    let artifact_error = identity.verify_current().unwrap_err();
    assert!(artifact_error.to_string().contains("identity changed"));
}

#[test]
fn bootstrap_file_digest_streams_across_internal_buffer_boundaries() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("large-artifact.bin");
    let bytes = (0..(128 * 1024 + 17))
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    std::fs::write(&path, &bytes).unwrap();
    let expected = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    assert_eq!(
        bootstrap_file_digest(&path, "test artifact").unwrap(),
        expected
    );
}

#[test]
fn bootstrap_file_digest_rejects_non_regular_and_oversized_inputs() {
    let temp = tempfile::tempdir().unwrap();
    let non_regular = bootstrap_file_digest(temp.path(), "test artifact").unwrap_err();
    assert!(non_regular.to_string().contains("must be a regular file"));

    let oversized = temp.path().join("oversized-artifact.bin");
    let file = std::fs::File::create(&oversized).unwrap();
    file.set_len(MAX_BOOTSTRAP_IDENTITY_FILE_BYTES + 1).unwrap();
    let oversized = bootstrap_file_digest(&oversized, "test artifact").unwrap_err();
    assert!(oversized.to_string().contains("identity budget"));
}

#[test]
fn persistent_server_resolution_rejects_oversized_sparse_dynamic_plugin_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let user_config_dir = xdg.join("nemo-relay");
    let plugin_dir = user_config_dir.join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.worker");
    std::fs::write(
        user_config_dir.join("plugins.toml"),
        r#"
[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"
"#,
    )
    .unwrap();
    std::fs::File::options()
        .write(true)
        .open(&manifest_path)
        .unwrap()
        .set_len(MAX_BOOTSTRAP_IDENTITY_FILE_BYTES + 1)
        .unwrap();

    let error = resolve_persistent_server_config(&GatewayOverrides::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("dynamic plugin manifest"));
    assert!(error.contains("identity budget"));
}

#[test]
fn persistent_server_resolution_rejects_oversized_sparse_dynamic_plugin_artifact() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let user_config_dir = xdg.join("nemo-relay");
    let plugin_dir = user_config_dir.join("plugins/acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let plugins_toml_path = user_config_dir.join("plugins.toml");
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
    std::fs::File::options()
        .write(true)
        .open(plugin_dir.join("plugin.py"))
        .unwrap()
        .set_len(MAX_BOOTSTRAP_IDENTITY_FILE_BYTES + 1)
        .unwrap();

    let error = resolve_persistent_server_config(&GatewayOverrides::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("dynamic plugin artifact"));
    assert!(error.contains("identity budget"));
}

#[test]
fn bootstrap_hmac_key_rejects_corrupt_persistent_state() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("state/fingerprint-hmac.key");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"short").unwrap();

    let error = load_or_create_bootstrap_hmac_key_at(&path).unwrap_err();

    assert!(error.to_string().contains("invalid length 5"));
}

#[test]
fn persistent_server_resolution_rejects_project_specific_flags() {
    let args = GatewayOverrides {
        config: Some(PathBuf::from("project-config.toml")),
        ..GatewayOverrides::default()
    };

    assert!(
        resolve_persistent_server_config(&args)
            .unwrap_err()
            .to_string()
            .contains("nemo-relay run")
    );
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
    let args = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
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

    let args = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
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

    let args = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
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

    let args = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
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

    let args = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
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
        crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES
    );
    assert_eq!(
        gateway.max_passthrough_body_bytes,
        crate::configuration::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES
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
        let args = GatewayOverrides {
            config: Some(path.clone()),
            ..GatewayOverrides::default()
        };

        let error = resolve_server_config(&args).unwrap_err().to_string();

        assert!(error.contains(expected));
        assert!(error.contains("greater than 0"));
    }
}

#[test]
fn run_resolution_applies_all_run_overrides() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = isolated_config_path(&temp);
    std::fs::write(&config_path, "").unwrap();
    let command = RunOverrides {
        agent: Some(CodingAgent::Codex),
        config: Some(config_path),
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
    let command = RunOverrides {
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
    let args = GatewayOverrides {
        config: Some(invalid_toml),
        ..GatewayOverrides::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid TOML"));

    let invalid_shape = temp.path().join("invalid-shape.toml");
    std::fs::write(&invalid_shape, "upstream = \"not-a-table\"").unwrap();
    let args = GatewayOverrides {
        config: Some(invalid_shape),
        ..GatewayOverrides::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid gateway configuration shape"));

    let plugin_config = temp.path().join("config-with-invalid-plugins.toml");
    std::fs::write(&plugin_config, "").unwrap();
    std::fs::write(temp.path().join("plugins.toml"), "version = [").unwrap();
    let args = GatewayOverrides {
        config: Some(plugin_config),
        ..GatewayOverrides::default()
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

#[cfg(windows)]
fn set_test_windows_dacl(path: &std::path::Path, sddl: &str) {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{
        DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR,
        SetFileSecurityW,
    };

    let sddl = std::ffi::OsStr::new(sddl)
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
    // SAFETY: The SDDL is NUL-terminated and the output pointer is valid.
    assert_ne!(
        unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                std::ptr::null_mut(),
            )
        },
        0,
        "{}",
        std::io::Error::last_os_error()
    );
    let path = path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    // SAFETY: The path and descriptor remain valid for the duration of the call.
    let result = unsafe {
        SetFileSecurityW(
            path.as_ptr(),
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            descriptor,
        )
    };
    // SAFETY: The descriptor was allocated by ConvertStringSecurityDescriptor... above.
    unsafe { LocalFree(descriptor.cast()) };
    assert_ne!(result, 0, "{}", std::io::Error::last_os_error());
}
