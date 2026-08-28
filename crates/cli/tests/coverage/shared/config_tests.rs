// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::agents::CodingAgent;
use crate::filesystem::bounded::MAX_BOUNDED_FILE_BYTES as MAX_BOOTSTRAP_IDENTITY_FILE_BYTES;
#[cfg(unix)]
use crate::filesystem::bounded::read_bounded_regular_file;
use crate::hooks::GatewayMode;
use axum::http::HeaderValue;
use base64::Engine;
use nemo_relay::logging::{
    DEFAULT_FILE_SINK_QUEUE_ENTRIES, LogFormat, LogLevel, LogSinkConfig,
    MAX_FILE_SINK_QUEUE_ENTRIES,
};
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

#[test]
fn transparent_gateway_fingerprints_are_stable_and_url_scoped() {
    let first = transparent_gateway_fingerprint("http://127.0.0.1:47632");
    assert_eq!(
        first,
        transparent_gateway_fingerprint("http://127.0.0.1:47632")
    );
    assert_ne!(
        first,
        transparent_gateway_fingerprint("http://127.0.0.1:47633")
    );
    assert_eq!(
        first,
        "transparent-sha256:bb0bbeaad4270d196ff14af97fc00e330c9d1385224b0b26dfbba9e626c44717"
    );
    assert!(first.starts_with("transparent-sha256:"));
}

#[test]
fn configuration_json_and_header_helpers_retain_only_valid_values() {
    assert_eq!(
        parse_json_option("metadata", r#"{"enabled":true}"#).unwrap(),
        json!({"enabled": true})
    );
    assert!(
        parse_json_option("metadata", "{")
            .unwrap_err()
            .to_string()
            .contains("invalid metadata")
    );

    let mut headers = http::HeaderMap::new();
    headers.insert("x-value", http::HeaderValue::from_static("text"));
    headers.insert("x-json", http::HeaderValue::from_static(r#"{"ok":true}"#));
    headers.insert("x-empty", http::HeaderValue::from_static(""));
    assert_eq!(header_string(&headers, "x-value").as_deref(), Some("text"));
    assert_eq!(header_string(&headers, "x-empty"), None);
    assert_eq!(header_string(&headers, "missing"), None);
    assert_eq!(header_json(&headers, "x-json"), Some(json!({"ok": true})));
    assert_eq!(header_json(&headers, "x-value"), None);
}

#[test]
fn configuration_legacy_observability_detection_reports_each_supported_section() {
    assert!(legacy_observability_sections(&toml::Value::Table(toml::Table::new())).is_empty());
    let value = toml::from_str(
        r#"
[exporters]
[observability]
[export.openinference]
"#,
    )
    .unwrap();
    assert_eq!(
        legacy_observability_sections(&value),
        vec!["[exporters]", "[observability]", "[export.openinference]"]
    );
}

#[test]
fn bounded_dynamic_manifest_loader_accepts_directories_and_rejects_invalid_utf8() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join(DYNAMIC_PLUGIN_MANIFEST_FILENAME);
    std::fs::write(
        &manifest_path,
        r#"
manifest_version = 1

[plugin]
id = "acme.coverage-worker"
kind = "worker"

[compat]
relay = ">=0.8.0,<1.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "plugin:register"
"#,
    )
    .unwrap();
    let (manifest, normalized, bytes) =
        load_bounded_dynamic_plugin_manifest_bytes(dir.path()).unwrap();
    assert_eq!(manifest.plugin.id, "acme.coverage-worker");
    assert!(normalized.ends_with(DYNAMIC_PLUGIN_MANIFEST_FILENAME));
    assert!(!bytes.is_empty());
    assert!(load_bounded_dynamic_plugin_manifest(dir.path()).is_ok());

    let invalid = dir.path().join("invalid.toml");
    std::fs::write(&invalid, [0xff_u8]).unwrap();
    assert!(
        load_bounded_dynamic_plugin_manifest_bytes(&invalid)
            .unwrap_err()
            .to_string()
            .contains("not UTF-8")
    );
}

#[test]
fn explicit_plugin_config_path_resolves_runtime_target() {
    let config = PathBuf::from("/managed/config.toml");
    let sibling = PathBuf::from("/managed/plugins.toml");
    let override_path = PathBuf::from("/override/plugins.toml");

    assert_eq!(
        explicit_plugin_config_path(Some(&config), None),
        Some(sibling)
    );
    assert_eq!(
        explicit_plugin_config_path(Some(&config), Some(&override_path)),
        Some(override_path)
    );
    assert_eq!(explicit_plugin_config_path(None, None), None);
}

#[test]
fn config_paths_layer_explicit_or_user_then_system_and_ignore_project() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let child = project.join("nested");
    let xdg = temp.path().join("xdg");
    let project_config = project.join(".nemo-relay").join("config.toml");
    std::fs::create_dir_all(&child).unwrap();
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::create_dir_all(project_config.parent().unwrap()).unwrap();
    std::fs::write(&project_config, "").unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&child, &xdg);
    let system_config = system_config_dir().join("config.toml");

    assert_eq!(
        config_paths(None),
        vec![
            xdg.join("nemo-relay").join("config.toml"),
            system_config.clone(),
        ]
    );

    let explicit_config = temp.path().join("managed").join("config.toml");
    assert_eq!(
        config_paths(Some(&explicit_config)),
        vec![explicit_config, system_config]
    );
}

#[test]
fn plugin_config_paths_layer_explicit_or_user_then_system_and_ignore_project() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let child = project.join("nested");
    let xdg = temp.path().join("xdg");
    let project_plugins = project.join(".nemo-relay").join("plugins.toml");
    std::fs::create_dir_all(&child).unwrap();
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::create_dir_all(project_plugins.parent().unwrap()).unwrap();
    std::fs::write(&project_plugins, "version = 1\n").unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&child, &xdg);
    let system_plugins = system_config_dir().join("plugins.toml");

    assert_eq!(
        plugin_config_paths(None, None),
        vec![
            xdg.join("nemo-relay").join("plugins.toml"),
            system_plugins.clone(),
        ]
    );

    let explicit_config = temp.path().join("managed").join("config.toml");
    let explicit_plugins = explicit_config.parent().unwrap().join("plugins.toml");
    assert_eq!(
        plugin_config_paths(Some(&explicit_config), None),
        vec![explicit_plugins.clone(), system_plugins.clone()]
    );

    let override_plugins = temp.path().join("override").join("plugins.toml");
    assert_eq!(
        plugin_config_paths(Some(&explicit_config), Some(&override_plugins)),
        vec![override_plugins, system_plugins]
    );
}

struct PluginConfigDiscoveryScope {
    _cwd_guard: crate::test_support::CwdTestScope,
    _guard: MutexGuard<'static, ()>,
    previous_cwd: PathBuf,
    previous_xdg_config_home: Option<OsString>,
    previous_openai_api_key: Option<OsString>,
    previous_openai_base_url: Option<OsString>,
    previous_openai_auth_header: Option<OsString>,
    previous_anthropic_base_url: Option<OsString>,
    previous_anthropic_auth_header: Option<OsString>,
    previous_bootstrap_fingerprint: Option<OsString>,
    previous_plugin_idle_timeout: Option<OsString>,
    previous_plugin_heartbeat_interval: Option<OsString>,
    previous_test_skip_implicit_config: Option<OsString>,
}

impl PluginConfigDiscoveryScope {
    fn enter(cwd: &std::path::Path, xdg_config_home: &std::path::Path) -> Self {
        let cwd_guard = crate::test_support::CwdTestScope::locked();
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous_cwd = std::env::current_dir().unwrap();
        let previous_xdg_config_home = std::env::var_os("XDG_CONFIG_HOME");
        let previous_openai_api_key = std::env::var_os("OPENAI_API_KEY");
        let previous_openai_base_url = std::env::var_os("NEMO_RELAY_OPENAI_BASE_URL");
        let previous_openai_auth_header = std::env::var_os("NEMO_RELAY_OPENAI_AUTH_HEADER");
        let previous_anthropic_base_url = std::env::var_os("NEMO_RELAY_ANTHROPIC_BASE_URL");
        let previous_anthropic_auth_header = std::env::var_os("NEMO_RELAY_ANTHROPIC_AUTH_HEADER");
        let previous_bootstrap_fingerprint = std::env::var_os(BOOTSTRAP_FINGERPRINT_ENV);
        let previous_plugin_idle_timeout = std::env::var_os(PLUGIN_IDLE_TIMEOUT_ENV);
        let previous_plugin_heartbeat_interval = std::env::var_os(PLUGIN_HEARTBEAT_INTERVAL_ENV);
        let previous_test_skip_implicit_config =
            std::env::var_os("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG");
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", xdg_config_home);
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("NEMO_RELAY_OPENAI_BASE_URL");
            std::env::remove_var("NEMO_RELAY_OPENAI_AUTH_HEADER");
            std::env::remove_var("NEMO_RELAY_ANTHROPIC_BASE_URL");
            std::env::remove_var("NEMO_RELAY_ANTHROPIC_AUTH_HEADER");
            std::env::remove_var(BOOTSTRAP_FINGERPRINT_ENV);
            std::env::remove_var(PLUGIN_IDLE_TIMEOUT_ENV);
            std::env::remove_var(PLUGIN_HEARTBEAT_INTERVAL_ENV);
            std::env::remove_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG");
        }
        std::env::set_current_dir(cwd).unwrap();
        Self {
            _cwd_guard: cwd_guard,
            _guard: guard,
            previous_cwd,
            previous_xdg_config_home,
            previous_openai_api_key,
            previous_openai_base_url,
            previous_openai_auth_header,
            previous_anthropic_base_url,
            previous_anthropic_auth_header,
            previous_bootstrap_fingerprint,
            previous_plugin_idle_timeout,
            previous_plugin_heartbeat_interval,
            previous_test_skip_implicit_config,
        }
    }

    fn set_bootstrap_fingerprint(&self, fingerprint: &str) {
        // SAFETY: This scope holds the process-wide environment mutex.
        unsafe {
            std::env::set_var(BOOTSTRAP_FINGERPRINT_ENV, fingerprint);
        }
    }

    fn set_auth_headers(&self, openai: &str, anthropic: &str) {
        // SAFETY: This scope holds the process-wide environment mutex.
        unsafe {
            std::env::set_var("NEMO_RELAY_OPENAI_AUTH_HEADER", openai);
            std::env::set_var("NEMO_RELAY_ANTHROPIC_AUTH_HEADER", anthropic);
        }
    }

    fn set_base_urls(&self, openai: &str, anthropic: &str) {
        // SAFETY: This scope holds the process-wide environment mutex.
        unsafe {
            std::env::set_var("NEMO_RELAY_OPENAI_BASE_URL", openai);
            std::env::set_var("NEMO_RELAY_ANTHROPIC_BASE_URL", anthropic);
        }
    }

    #[cfg(feature = "__skip-implicit-config")]
    fn skip_implicit_config(&self) {
        // SAFETY: This scope holds the process-wide environment mutex.
        unsafe {
            std::env::set_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "1");
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
            match self.previous_openai_api_key.take() {
                Some(value) => std::env::set_var("OPENAI_API_KEY", value),
                None => std::env::remove_var("OPENAI_API_KEY"),
            }
            match self.previous_openai_base_url.take() {
                Some(value) => std::env::set_var("NEMO_RELAY_OPENAI_BASE_URL", value),
                None => std::env::remove_var("NEMO_RELAY_OPENAI_BASE_URL"),
            }
            match self.previous_openai_auth_header.take() {
                Some(value) => std::env::set_var("NEMO_RELAY_OPENAI_AUTH_HEADER", value),
                None => std::env::remove_var("NEMO_RELAY_OPENAI_AUTH_HEADER"),
            }
            match self.previous_anthropic_base_url.take() {
                Some(value) => std::env::set_var("NEMO_RELAY_ANTHROPIC_BASE_URL", value),
                None => std::env::remove_var("NEMO_RELAY_ANTHROPIC_BASE_URL"),
            }
            match self.previous_anthropic_auth_header.take() {
                Some(value) => std::env::set_var("NEMO_RELAY_ANTHROPIC_AUTH_HEADER", value),
                None => std::env::remove_var("NEMO_RELAY_ANTHROPIC_AUTH_HEADER"),
            }
            match self.previous_bootstrap_fingerprint.take() {
                Some(value) => std::env::set_var(BOOTSTRAP_FINGERPRINT_ENV, value),
                None => std::env::remove_var(BOOTSTRAP_FINGERPRINT_ENV),
            }
            match self.previous_plugin_idle_timeout.take() {
                Some(value) => std::env::set_var(PLUGIN_IDLE_TIMEOUT_ENV, value),
                None => std::env::remove_var(PLUGIN_IDLE_TIMEOUT_ENV),
            }
            match self.previous_plugin_heartbeat_interval.take() {
                Some(value) => std::env::set_var(PLUGIN_HEARTBEAT_INTERVAL_ENV, value),
                None => std::env::remove_var(PLUGIN_HEARTBEAT_INTERVAL_ENV),
            }
            match self.previous_test_skip_implicit_config.take() {
                Some(value) => std::env::set_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", value),
                None => std::env::remove_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG"),
            }
        }
    }
}

#[cfg(feature = "__skip-implicit-config")]
#[test]
fn test_hook_skips_implicit_config_but_retains_explicit_config_and_environment() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let user_config = xdg.join("nemo-relay/config.toml");
    let user_plugins = xdg.join("nemo-relay").join(PLUGINS_TOML);
    let explicit_config = temp.path().join("explicit/config.toml");
    let explicit_plugins = explicit_config.parent().unwrap().join(PLUGINS_TOML);
    std::fs::create_dir_all(user_config.parent().unwrap()).unwrap();
    std::fs::create_dir_all(explicit_config.parent().unwrap()).unwrap();
    std::fs::write(
        &user_config,
        "[upstream]\nopenai_base_url = 'http://user'\n",
    )
    .unwrap();
    std::fs::write(&user_plugins, "not valid TOML = [").unwrap();
    std::fs::write(
        &explicit_config,
        "[upstream]\nopenai_base_url = 'http://explicit'\n",
    )
    .unwrap();
    let explicit_plugin_dir = explicit_config.parent().unwrap().join("plugins/acme");
    std::fs::create_dir_all(&explicit_plugin_dir).unwrap();
    let explicit_manifest = write_dynamic_manifest(&explicit_plugin_dir, "acme.explicit");
    std::fs::write(
        &explicit_plugins,
        r#"
version = 1
components = []

[[plugins.dynamic]]
manifest = "plugins/acme/relay-plugin.toml"
"#,
    )
    .unwrap();

    let scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    scope.set_base_urls("http://environment", "http://environment-anthropic");
    scope.skip_implicit_config();

    assert!(config_paths(None).is_empty());
    assert!(plugin_config_paths(None, None).is_empty());
    assert!(!any_config_file_exists());
    assert_eq!(
        config_paths(Some(&explicit_config)),
        vec![explicit_config.clone()]
    );
    assert_eq!(
        plugin_config_paths(Some(&explicit_config), None),
        vec![explicit_plugins.clone()]
    );

    let implicit = load_shared_config(None, None).unwrap();
    assert_eq!(implicit.gateway.openai_base_url, "http://environment");
    let explicit = load_shared_config(Some(&explicit_config), None).unwrap();
    assert_eq!(explicit.gateway.openai_base_url, "http://environment");
    let explicit_plugin_config = load_shared_config(None, Some(&explicit_plugins)).unwrap();
    assert_eq!(explicit_plugin_config.dynamic_plugins.len(), 1);
    assert_eq!(
        explicit_plugin_config.dynamic_plugins[0].manifest_ref,
        explicit_manifest.canonicalize().unwrap().to_string_lossy()
    );
}

fn config() -> GatewayConfig {
    GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),
        openai_auth_header: None,
        anthropic_base_url: "http://anthropic".into(),
        anthropic_auth_header: None,
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::configuration::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    }
}

#[test]
fn provider_auth_headers_default_to_unset() {
    let config = GatewayConfig::default();

    assert!(config.openai_auth_header.is_none());
    assert!(config.anthropic_auth_header.is_none());
}

fn effective_plugin_toml_sources_without_system(
    explicit: Option<&PathBuf>,
    plugin_config_path: Option<&PathBuf>,
) -> Result<Vec<PathBuf>, CliError> {
    // System-path discovery has dedicated coverage; source-list tests must not read host policy.
    let system = global_plugin_config_path();
    effective_plugin_toml_sources_from_paths(
        plugin_config_paths(explicit, plugin_config_path)
            .into_iter()
            .filter(|path| path != &system),
    )
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
        effective_plugin_toml_sources_without_system(None, None).unwrap(),
        Vec::<PathBuf>::new()
    );

    let project_plugins = project.join(".nemo-relay/plugins.toml");
    let user_plugins = xdg.join("nemo-relay").join("plugins.toml");
    std::fs::create_dir_all(project_plugins.parent().unwrap()).unwrap();
    std::fs::create_dir_all(user_plugins.parent().unwrap()).unwrap();
    std::fs::write(&project_plugins, "version = 1\ncomponents = []\n").unwrap();
    std::fs::write(&user_plugins, "version = 1\ncomponents = []\n").unwrap();

    let sources = effective_plugin_toml_sources_without_system(None, None).unwrap();
    assert!(sources.is_sorted());
    assert!(sources.windows(2).all(|paths| paths[0] != paths[1]));

    let mut actual = sources
        .iter()
        .map(|path| path.canonicalize().unwrap())
        .collect::<Vec<_>>();
    actual.sort();
    let mut expected = [user_plugins]
        .iter()
        .map(|path| path.canonicalize().unwrap())
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(actual, expected);
}

#[test]
fn effective_plugin_toml_sources_replace_user_with_explicit_and_ignore_project() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let xdg = temp.path().join("xdg");
    let explicit_dir = temp.path().join("explicit");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::create_dir_all(&explicit_dir).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&project, &xdg);
    let explicit_config = explicit_dir.join("config.toml");
    let explicit_plugins = explicit_dir.join("plugins.toml");
    std::fs::write(&explicit_config, "").unwrap();
    std::fs::write(&explicit_plugins, "version = 1\ncomponents = []\n").unwrap();
    std::fs::create_dir_all(project.join(".nemo-relay")).unwrap();
    let project_plugins = project.join(".nemo-relay").join("plugins.toml");
    std::fs::write(&project_plugins, "version = 1\ncomponents = []\n").unwrap();
    let user_plugins = xdg.join("nemo-relay/plugins.toml");
    std::fs::create_dir_all(user_plugins.parent().unwrap()).unwrap();
    std::fs::write(&user_plugins, "components = [\n").unwrap();

    assert_eq!(
        effective_plugin_toml_sources_without_system(Some(&explicit_config), None).unwrap(),
        vec![explicit_plugins]
    );
}

fn isolated_config_path(temp: &tempfile::TempDir) -> std::path::PathBuf {
    temp.path().join("config.toml")
}

// Escapes a path for embedding in a TOML basic string (Windows `\U` sequences are invalid otherwise).
fn toml_basic_string(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
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
relay = ">=0.8.0,<1.0"
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
    assert_eq!(CodingAgent::infer("hermes"), None);
    assert_eq!(CodingAgent::infer("wrapper"), None);
}

#[test]
fn explicit_toml_config_maps_supported_sections() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://openai"
openai_auth_header = "Bearer openai-file"
anthropic_base_url = "http://anthropic"
anthropic_auth_header = "Basic anthropic-file"

[gateway]
max_hook_payload_bytes = 12345
max_passthrough_body_bytes = 67890

[agents.claude]
command = "claude"

[agents.codex]
command = "codex --approval-mode never"

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
    assert_eq!(
        resolved.gateway.openai_auth_header.as_deref(),
        Some("Bearer openai-file")
    );
    assert_eq!(resolved.gateway.anthropic_base_url, "http://anthropic");
    assert_eq!(
        resolved.gateway.anthropic_auth_header.as_deref(),
        Some("Basic anthropic-file")
    );
    assert_eq!(resolved.gateway.max_hook_payload_bytes, 12345);
    assert_eq!(resolved.gateway.max_passthrough_body_bytes, 67890);
    assert_eq!(resolved.gateway.metadata, None);
    assert_eq!(resolved.gateway.plugin_config, None);
    assert_eq!(
        resolved.agents.codex.command.as_deref(),
        Some("codex --approval-mode never")
    );
}

#[test]
fn stale_hermes_agent_config_is_rejected() {
    let value = toml::from_str::<toml::Value>(
        r#"
[agents.hermes]
command = "hermes"
"#,
    )
    .unwrap();

    let error = validate_shared_config_shape(value).unwrap_err().to_string();

    assert!(error.contains("unknown field `hermes`"));
}

#[test]
fn provider_auth_environment_overrides_file_values() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_auth_header = "Bearer openai-file"
anthropic_auth_header = "Basic anthropic-file"
"#,
    )
    .unwrap();
    scope.set_auth_headers("  Bearer openai-env  ", "  Basic anthropic-env  ");

    let resolved = resolve_server_config(&GatewayOverrides {
        config: Some(path),
        ..GatewayOverrides::default()
    })
    .unwrap();

    assert_eq!(
        resolved.gateway.openai_auth_header.as_deref(),
        Some("Bearer openai-env")
    );
    assert_eq!(
        resolved.gateway.anthropic_auth_header.as_deref(),
        Some("Basic anthropic-env")
    );
}

#[test]
fn ignored_project_endpoints_do_not_clear_user_provider_auth_headers() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let nested = project.join("nested");
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(project.join(".nemo-relay")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::create_dir_all(xdg.join("nemo-relay")).unwrap();
    std::fs::write(
        project.join(".nemo-relay/config.toml"),
        r#"
[upstream]
openai_base_url = "http://project-openai"
anthropic_base_url = "http://project-anthropic"
"#,
    )
    .unwrap();
    std::fs::write(
        xdg.join("nemo-relay/config.toml"),
        r#"
[upstream]
openai_base_url = "http://user-openai"
openai_auth_header = "Bearer user-openai"
anthropic_base_url = "http://user-anthropic"
anthropic_auth_header = "Basic user-anthropic"
"#,
    )
    .unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&nested, &xdg);

    let resolved = resolve_server_config(&GatewayOverrides::default()).unwrap();

    assert_eq!(resolved.gateway.openai_base_url, "http://user-openai");
    assert_eq!(
        resolved.gateway.openai_auth_header.as_deref(),
        Some("Bearer user-openai")
    );
    assert_eq!(resolved.gateway.anthropic_base_url, "http://user-anthropic");
    assert_eq!(
        resolved.gateway.anthropic_auth_header.as_deref(),
        Some("Basic user-anthropic")
    );
}

#[test]
fn endpoint_environment_overrides_clear_file_provider_auth_headers() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_auth_header = "Bearer file-openai"
anthropic_auth_header = "Basic file-anthropic"
"#,
    )
    .unwrap();
    scope.set_base_urls("http://environment-openai", "http://environment-anthropic");

    let resolved = resolve_server_config(&GatewayOverrides {
        config: Some(path),
        ..GatewayOverrides::default()
    })
    .unwrap();

    assert_eq!(
        resolved.gateway.openai_base_url,
        "http://environment-openai"
    );
    assert!(resolved.gateway.openai_auth_header.is_none());
    assert_eq!(
        resolved.gateway.anthropic_base_url,
        "http://environment-anthropic"
    );
    assert!(resolved.gateway.anthropic_auth_header.is_none());
}

#[test]
fn matching_endpoint_environment_overrides_preserve_file_provider_auth_headers() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://same-openai"
openai_auth_header = "Bearer file-openai"
anthropic_base_url = "http://same-anthropic"
anthropic_auth_header = "Basic file-anthropic"
"#,
    )
    .unwrap();
    scope.set_base_urls("http://same-openai", "http://same-anthropic");

    let resolved = resolve_server_config(&GatewayOverrides {
        config: Some(path),
        ..GatewayOverrides::default()
    })
    .unwrap();

    assert_eq!(
        resolved.gateway.openai_auth_header.as_deref(),
        Some("Bearer file-openai")
    );
    assert_eq!(
        resolved.gateway.anthropic_auth_header.as_deref(),
        Some("Basic file-anthropic")
    );
}

#[test]
fn invalid_provider_auth_header_errors_do_not_expose_secret_values() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        "[upstream]\nopenai_auth_header = \"\"\"\\\nBearer private\nsecret\"\"\"\n",
    )
    .unwrap();

    let error = resolve_server_config(&GatewayOverrides {
        config: Some(path),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();

    assert!(error.contains("upstream.openai_auth_header"), "{error}");
    assert!(error.contains("valid HTTP header value"), "{error}");
    assert!(!error.contains("private"), "{error}");
    assert!(!error.contains("secret"), "{error}");
}

#[test]
fn invalid_anthropic_provider_auth_header_errors_do_not_expose_secret_values() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        "[upstream]\nanthropic_auth_header = \"\"\"\\\nBasic private\nsecret\"\"\"\n",
    )
    .unwrap();

    let error = resolve_server_config(&GatewayOverrides {
        config: Some(path),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();

    assert!(error.contains("upstream.anthropic_auth_header"), "{error}");
    assert!(error.contains("valid HTTP header value"), "{error}");
    assert!(!error.contains("private"), "{error}");
    assert!(!error.contains("secret"), "{error}");
}

#[test]
fn invalid_provider_auth_environment_errors_do_not_expose_secret_values() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let path = temp.path().join("config.toml");
    std::fs::write(&path, "").unwrap();
    scope.set_auth_headers("Bearer private\nsecret", "Basic valid");

    let error = resolve_server_config(&GatewayOverrides {
        config: Some(path),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();

    assert!(error.contains("NEMO_RELAY_OPENAI_AUTH_HEADER"), "{error}");
    assert!(!error.contains("private"), "{error}");
    assert!(!error.contains("secret"), "{error}");
}

#[test]
fn explicit_config_must_exist() {
    let _cwd = crate::test_support::CwdTestScope::locked();
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
fn plugin_config_loader_deduplicates_aliases_at_highest_precedence() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let physical = temp.path().join("system-plugins.toml");
    let alias = temp.path().join("explicit-plugins.toml");
    std::fs::write(
        &physical,
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
    symlink(&physical, &alias).unwrap();

    let resolved = load_plugin_toml_config_from_paths(vec![alias, physical.clone()])
        .unwrap()
        .expect("the physical file exists");

    assert_eq!(resolved.contributing_sources, vec![physical]);
    assert_eq!(
        resolved.value.unwrap()["components"][0]["config"]["sources"]
            .as_array()
            .unwrap()
            .len(),
        1,
        "the aliased file must not duplicate list entries"
    );
}

#[cfg(unix)]
#[test]
fn unreadable_config_errors_include_the_source_path() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    use std::os::unix::fs::PermissionsExt;

    if unsafe { libc::geteuid() } == 0 {
        return;
    }

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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let xdg = temp.path().join("xdg");
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let explicit = temp.path().join("custom-config.toml");
    assert_eq!(
        plugin_config_paths(Some(&explicit), None),
        vec![
            temp.path().join("plugins.toml"),
            system_config_dir().join("plugins.toml"),
        ]
    );

    let project = temp.path().join("workspace");
    let nested = project.join("a/b/c");
    std::fs::create_dir_all(project.join(".nemo-relay")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    let plugin_path = project.join(".nemo-relay/plugins.toml");
    std::fs::write(&plugin_path, "version = 1").unwrap();
    let user_config = temp.path().join("xdg/nemo-relay");

    assert_eq!(
        implicit_plugin_config_paths(Some(user_config.clone())),
        vec![
            user_config.join("plugins.toml"),
            system_config_dir().join("plugins.toml"),
        ]
    );
    assert!(
        plugin_path.exists(),
        "project file remains present but ignored"
    );
}

#[test]
fn all_runtime_scopes_exclude_project_gateway_and_plugin_layers() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("workspace");
    let nested = project.join("nested");
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(project.join(".nemo-relay")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(project.join(".nemo-relay/config.toml"), "").unwrap();
    std::fs::write(project.join(".nemo-relay/plugins.toml"), "version = 1\n").unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&nested, &xdg);

    assert_eq!(
        config_paths(None),
        vec![
            xdg.join("nemo-relay/config.toml"),
            system_config_dir().join("config.toml"),
        ]
    );
    assert_eq!(
        plugin_config_paths(None, None),
        vec![
            xdg.join("nemo-relay/plugins.toml"),
            system_config_dir().join("plugins.toml"),
        ]
    );
}

#[test]
fn logging_resolution_ignores_project_config() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("workspace");
    let nested = project.join("nested");
    let xdg = temp.path().join("xdg");
    let project_config_dir = project.join(".nemo-relay");
    let user_config_dir = xdg.join("nemo-relay");
    let project_sink = temp.path().join("project.log.jsonl");
    std::fs::create_dir_all(&project_config_dir).unwrap();
    std::fs::create_dir_all(&user_config_dir).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(
        project_config_dir.join("config.toml"),
        format!(
            r#"
[[logging.sinks]]
path = {}
"#,
            toml_basic_string(project_sink.to_string_lossy().as_ref())
        ),
    )
    .unwrap();
    std::fs::write(
        user_config_dir.join("config.toml"),
        r#"
[logging]
level = "warn"
"#,
    )
    .unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&nested, &xdg);
    let has_project_sink = |config: &LoggingConfig| {
        config.sinks.iter().any(
            |sink| matches!(sink, LogSinkConfig::File(file) if file.path == project_sink.as_path()),
        )
    };

    let resolved = resolve_logging_config(None).unwrap();
    assert!(!has_project_sink(&resolved));
}

#[test]
fn operational_logging_uses_explicit_config_and_ignores_project_layer() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("workspace");
    let nested = project.join("nested");
    let xdg = temp.path().join("xdg");
    let project_config_dir = project.join(".nemo-relay");
    let explicit_config_dir = temp.path().join("explicit");
    let project_sink = temp.path().join("project.log.jsonl");
    let explicit_sink = temp.path().join("explicit.log.jsonl");
    std::fs::create_dir_all(&project_config_dir).unwrap();
    std::fs::create_dir_all(&explicit_config_dir).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(
        project_config_dir.join("config.toml"),
        format!(
            "[[logging.sinks]]\npath = {}\n",
            toml_basic_string(project_sink.to_string_lossy().as_ref())
        ),
    )
    .unwrap();
    let explicit_config = explicit_config_dir.join("config.toml");
    std::fs::write(
        &explicit_config,
        format!(
            "[[logging.sinks]]\npath = {}\n",
            toml_basic_string(explicit_sink.to_string_lossy().as_ref())
        ),
    )
    .unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&nested, &xdg);

    let logging = resolve_logging_config(Some(&explicit_config)).unwrap();
    let paths = logging
        .sinks
        .iter()
        .map(|sink| match sink {
            LogSinkConfig::File(file) => file.path.as_path(),
        })
        .collect::<Vec<_>>();

    assert_eq!(paths, vec![explicit_sink.as_path()]);
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    let sibling_path = temp.path().join("plugins.toml");
    let override_path = temp.path().join("override.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        &sibling_path,
        "version = 1\n[policy]\nunknown_field = \"warn\"\n",
    )
    .unwrap();
    std::fs::write(
        &override_path,
        "version = 1\n[policy]\nunknown_field = \"error\"\n",
    )
    .unwrap();
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
        Some(json!({
            "version": 1,
            "policy": { "unknown_field": "error" }
        }))
    );
}

#[test]
fn cli_run_overrides_config_values() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://file-openai"
openai_auth_header = "Bearer file-openai"
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
    assert!(resolved.gateway.openai_auth_header.is_none());
    assert_eq!(resolved.gateway.metadata, Some(json!({ "team": "cli" })));
}

#[test]
fn run_inherits_top_level_server_flags_when_subcommand_flags_are_absent() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[upstream]
openai_base_url = "http://file-openai"
openai_auth_header = "Bearer file-openai"
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
    assert!(resolved.gateway.openai_auth_header.is_none());
}

#[test]
fn server_resolution_applies_all_server_overrides() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let config_path = isolated_config_path(&temp);
    std::fs::write(
        &config_path,
        r#"
[upstream]
openai_base_url = "http://file-openai"
openai_auth_header = "Bearer file-openai"
anthropic_base_url = "http://file-anthropic"
anthropic_auth_header = "Basic file-anthropic"
"#,
    )
    .unwrap();
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
    assert!(resolved.gateway.openai_auth_header.is_none());
    assert_eq!(resolved.gateway.anthropic_base_url, "http://cli-anthropic");
    assert!(resolved.gateway.anthropic_auth_header.is_none());
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
    let error = managed_bootstrap_identity(&managed_args, &resolved, &[]).unwrap_err();
    assert!(error.to_string().contains("must be set and non-empty"));
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
    if unsafe { libc::geteuid() } != 0 {
        std::fs::write(&unreadable, b"contents").unwrap();
        std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o000)).unwrap();
        let unreadable_result = read_bounded_regular_file(&unreadable, "fixture");
        std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o600)).unwrap();
        let unreadable_error = unreadable_result.unwrap_err();
        assert!(
            unreadable_error.contains("failed to read"),
            "{unreadable_error}"
        );
    }

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
fn persistent_fingerprint_tracks_provider_auth_headers() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&project).unwrap();
    let user_config_dir = xdg.join("nemo-relay");
    std::fs::create_dir_all(&user_config_dir).unwrap();
    let _scope = PluginConfigDiscoveryScope::enter(&project, &xdg);
    let config_path = user_config_dir.join("config.toml");
    std::fs::write(
        &config_path,
        "[upstream]\nopenai_auth_header = \"Bearer one\"\nanthropic_auth_header = \"Basic one\"\n",
    )
    .unwrap();

    let first = resolve_persistent_server_config(&GatewayOverrides::default())
        .unwrap()
        .bootstrap_fingerprint
        .unwrap();
    std::fs::write(
        &config_path,
        "[upstream]\nopenai_auth_header = \"Bearer two\"\nanthropic_auth_header = \"Basic one\"\n",
    )
    .unwrap();
    let openai_changed = resolve_persistent_server_config(&GatewayOverrides::default())
        .unwrap()
        .bootstrap_fingerprint
        .unwrap();
    std::fs::write(
        &config_path,
        "[upstream]\nopenai_auth_header = \"Bearer two\"\nanthropic_auth_header = \"Basic two\"\n",
    )
    .unwrap();
    let anthropic_changed = resolve_persistent_server_config(&GatewayOverrides::default())
        .unwrap()
        .bootstrap_fingerprint
        .unwrap();

    assert_ne!(first, openai_changed);
    assert_ne!(openai_changed, anthropic_changed);
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

#[cfg(unix)]
#[test]
fn bootstrap_hmac_key_completes_empty_private_state() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("state/fingerprint-hmac.key");
    let parent = path.parent().unwrap();
    std::fs::create_dir_all(parent).unwrap();
    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::File::create(&path).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

    let key = load_or_create_bootstrap_hmac_key_at(&path).unwrap();

    assert_eq!(std::fs::metadata(&path).unwrap().len(), 32);
    assert_eq!(load_or_create_bootstrap_hmac_key_at(&path).unwrap(), key);
}

#[cfg(unix)]
#[test]
fn bootstrap_hmac_key_reuses_private_state_without_changing_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("state/fingerprint-hmac.key");
    let original = load_or_create_bootstrap_hmac_key_at(&path).unwrap();
    let parent = path.parent().unwrap();
    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o500)).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o400)).unwrap();

    let reloaded = load_or_create_bootstrap_hmac_key_at(&path).unwrap();

    assert_eq!(reloaded, original);
    assert_eq!(
        std::fs::metadata(parent).unwrap().permissions().mode() & 0o777,
        0o500
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o400
    );
}

#[cfg(unix)]
#[test]
fn bootstrap_hmac_key_repairs_group_or_other_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("state/fingerprint-hmac.key");
    let original = load_or_create_bootstrap_hmac_key_at(&path).unwrap();
    let parent = path.parent().unwrap();

    for (directory_mode, key_mode) in [(0o750, 0o640), (0o701, 0o604)] {
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(directory_mode)).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(key_mode)).unwrap();

        assert_eq!(
            load_or_create_bootstrap_hmac_key_at(&path).unwrap(),
            original
        );
        assert_eq!(
            std::fs::metadata(parent).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
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
            "version = 1\n\n[plugins.policy.defaults]\nstartup = \"optional\"\nattestation = \"integrity_only\"\n\n[[plugins.dynamic]]\nmanifest = {:?}\n",
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

    let resolved = load_shared_config(None, None).unwrap();
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
    assert!(oversized.to_string().contains("exceeds"));
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
    assert!(error.contains("byte limit"));
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
    assert!(error.contains("byte limit"));
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
fn persistent_server_resolution_rejects_explicit_config_flags() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let args = GatewayOverrides {
        config: Some(PathBuf::from("explicit-config.toml")),
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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
    let _cwd = crate::test_support::CwdTestScope::locked();
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

    let invalid_upstream_key = temp.path().join("invalid-upstream-key.toml");
    std::fs::write(
        &invalid_upstream_key,
        "[upstream]\nopenai_baseurl = \"https://example.test/v1\"\n",
    )
    .unwrap();
    let args = GatewayOverrides {
        config: Some(invalid_upstream_key),
        ..GatewayOverrides::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid gateway configuration shape"));
    assert!(error.contains("openai_baseurl"));

    let invalid_nested_upstream = temp.path().join("invalid-nested-upstream.toml");
    std::fs::write(
        &invalid_nested_upstream,
        "[upstream.openai]\nbase_url = \"https://example.test/v1\"\n",
    )
    .unwrap();
    let args = GatewayOverrides {
        config: Some(invalid_nested_upstream),
        ..GatewayOverrides::default()
    };

    let error = resolve_server_config(&args).unwrap_err().to_string();

    assert!(error.contains("invalid gateway configuration shape"));
    assert!(error.contains("openai"));

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

#[test]
fn uphill_config_layers_replace_scalars_and_aggregate_distinct_agent_tables() {
    let layers = [
        r#"
[gateway]
max_hook_payload_bytes = 100

[agents.codex]
command = "user-codex"
"#,
        r#"
[gateway]
max_passthrough_body_bytes = 200

[agents.claude]
command = "project-claude"
"#,
        r#"
[gateway]
max_hook_payload_bytes = 300

[agents.codex]
command = "system-codex"
"#,
    ];
    let mut merged = toml::Value::Table(toml::map::Map::new());
    for layer in layers {
        let layer = layer
            .parse::<toml::Table>()
            .map(toml::Value::Table)
            .unwrap();
        merge_gateway_config_toml(&mut merged, layer);
    }

    assert_eq!(
        merged["gateway"]["max_hook_payload_bytes"].as_integer(),
        Some(300)
    );
    assert_eq!(
        merged["gateway"]["max_passthrough_body_bytes"].as_integer(),
        Some(200)
    );
    assert_eq!(
        merged["agents"]["codex"]["command"].as_str(),
        Some("system-codex")
    );
    assert_eq!(
        merged["agents"]["claude"]["command"].as_str(),
        Some("project-claude")
    );
}

#[test]
fn upstream_base_url_identity_controls_credential_inheritance() {
    fn merge_upstream(lower: &str, higher: &str) -> toml::Value {
        let mut merged = lower
            .parse::<toml::Table>()
            .map(toml::Value::Table)
            .unwrap();
        let higher = higher
            .parse::<toml::Table>()
            .map(toml::Value::Table)
            .unwrap();
        merge_gateway_config_toml(&mut merged, higher);
        merged
    }

    let lower = r#"
[upstream]
openai_base_url = "https://gateway.example/v1"
openai_auth_header = "Bearer lower"
"#;
    let same_identity = merge_upstream(
        lower,
        r#"
[upstream]
openai_base_url = "https://gateway.example/v1"
"#,
    );
    assert_eq!(
        same_identity["upstream"]["openai_auth_header"].as_str(),
        Some("Bearer lower")
    );

    let replaced_credential = merge_upstream(
        lower,
        r#"
[upstream]
openai_base_url = "https://gateway.example/v1"
openai_auth_header = "Bearer higher"
"#,
    );
    assert_eq!(
        replaced_credential["upstream"]["openai_auth_header"].as_str(),
        Some("Bearer higher")
    );

    let changed_identity = merge_upstream(
        lower,
        r#"
[upstream]
openai_base_url = "https://other.example/v1"
"#,
    );
    assert!(
        changed_identity["upstream"]
            .get("openai_auth_header")
            .is_none()
    );

    let replaced_identity_and_credential = merge_upstream(
        lower,
        r#"
[upstream]
openai_base_url = "https://other.example/v1"
openai_auth_header = "Bearer replacement"
"#,
    );
    assert_eq!(
        replaced_identity_and_credential["upstream"]["openai_auth_header"].as_str(),
        Some("Bearer replacement")
    );
}

#[test]
fn logging_sinks_aggregate_by_path_with_higher_layers_first() {
    let layers = [
        r#"
[[logging.sinks]]
path = "shared.log"
level = "debug"
queue_capacity = 128

[[logging.sinks]]
path = "user.log"
level = "debug"
"#,
        r#"
[[logging.sinks]]
path = "shared.log"
level = "info"
format = "jsonl"

[[logging.sinks]]
path = "project.log"
level = "info"
"#,
        r#"
[[logging.sinks]]
path = "system.log"
level = "warn"

[[logging.sinks]]
path = "shared.log"
format = "human"
"#,
    ];
    let mut merged = toml::Value::Table(toml::map::Map::new());
    for layer in layers {
        let layer = layer
            .parse::<toml::Table>()
            .map(toml::Value::Table)
            .unwrap();
        merge_gateway_config_toml(&mut merged, layer);
    }

    let sinks = merged["logging"]["sinks"].as_array().unwrap();
    let paths = sinks
        .iter()
        .map(|sink| logging_sink_path(sink).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        vec!["system.log", "shared.log", "project.log", "user.log"]
    );

    let shared = sinks
        .iter()
        .find(|sink| logging_sink_path(sink) == Some("shared.log"))
        .unwrap();
    assert_eq!(shared["level"].as_str(), Some("info"));
    assert_eq!(shared["format"].as_str(), Some("human"));
    assert_eq!(shared["queue_capacity"].as_integer(), Some(128));
}

#[test]
fn logging_sink_path_aliases_coalesce_by_runtime_destination() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let _scope = PluginConfigDiscoveryScope::enter(temp.path(), &xdg);
    let mut merged = r#"
[[logging.sinks]]
path = "relay.log"
level = "debug"
queue_capacity = 128
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();
    let higher = r#"
[[logging.sinks]]
path = "./relay.log"
level = "warn"
format = "human"
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();

    merge_gateway_config_toml(&mut merged, higher);

    let sinks = merged["logging"]["sinks"].as_array().unwrap();
    assert_eq!(sinks.len(), 1);
    assert_eq!(logging_sink_path(&sinks[0]), Some("./relay.log"));
    assert_eq!(sinks[0]["level"].as_str(), Some("warn"));
    assert_eq!(sinks[0]["format"].as_str(), Some("human"));
    assert_eq!(sinks[0]["queue_capacity"].as_integer(), Some(128));
}

#[test]
fn empty_higher_logging_sink_list_preserves_lower_sinks() {
    let mut merged = r#"
[[logging.sinks]]
path = "lower.log"
level = "info"
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();
    let higher = r#"
[logging]
sinks = []
"#
    .parse::<toml::Table>()
    .map(toml::Value::Table)
    .unwrap();

    merge_gateway_config_toml(&mut merged, higher);

    let sinks = merged["logging"]["sinks"].as_array().unwrap();
    assert_eq!(sinks.len(), 1);
    assert_eq!(logging_sink_path(&sinks[0]), Some("lower.log"));
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

#[test]
fn logging_defaults_when_section_absent() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let config_path = isolated_config_path(&temp);
    std::fs::write(&config_path, "").unwrap();
    let args = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
    };

    let resolved = resolve_server_config(&args).unwrap();

    assert_eq!(resolved.logging, LoggingConfig::default());
    assert_eq!(resolved.logging.level, LogLevel::Error);
    assert_eq!(resolved.logging.stderr_format, LogFormat::Human);
    assert!(resolved.logging.sinks.is_empty());
}

#[test]
fn logging_parses_global_settings_and_file_sinks() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let config_path = isolated_config_path(&temp);
    let log_a = temp.path().join("a.log.jsonl");
    let log_b = temp.path().join("b.log.jsonl");
    std::fs::write(
        &config_path,
        format!(
            r#"
[logging]
level = "debug"
stderr_enabled = false
stderr_format = "human"
flush_interval_millis = 250

[[logging.sinks]]
path = {}
level = "trace"
format = "jsonl"
queue_capacity = 32

[[logging.sinks]]
path = {}
format = "human"
"#,
            toml_basic_string(log_a.to_string_lossy().as_ref()),
            toml_basic_string(log_b.to_string_lossy().as_ref())
        ),
    )
    .unwrap();
    let args = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
    };

    let resolved = resolve_server_config(&args).unwrap();

    assert_eq!(resolved.logging.level, LogLevel::Debug);
    assert!(!resolved.logging.stderr_enabled);
    assert_eq!(resolved.logging.stderr_format, LogFormat::Human);
    assert_eq!(resolved.logging.flush_interval_millis, 250);
    assert_eq!(resolved.logging.sinks.len(), 2);
    match &resolved.logging.sinks[0] {
        LogSinkConfig::File(sink) => {
            assert_eq!(sink.path, log_a);
            assert_eq!(sink.level, LogLevel::Trace);
            assert_eq!(sink.format, LogFormat::Jsonl);
            assert_eq!(sink.queue_capacity, 32);
        }
    }
    match &resolved.logging.sinks[1] {
        LogSinkConfig::File(sink) => {
            assert_eq!(sink.path, log_b);
            assert_eq!(sink.level, LogLevel::Debug);
            assert_eq!(sink.format, LogFormat::Human);
            assert_eq!(sink.queue_capacity, DEFAULT_FILE_SINK_QUEUE_ENTRIES);
        }
    }
}

#[test]
fn logging_rotation_cli_config_preserves_pair_and_rejects_incomplete_pair() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let config_path = isolated_config_path(&temp);
    let log_path = temp.path().join("relay.log.jsonl");
    std::fs::write(
        &config_path,
        format!(
            r#"
[[logging.sinks]]
path = {}
max_file_size_bytes = 1024
retained_files = 2
"#,
            toml_basic_string(log_path.to_string_lossy().as_ref())
        ),
    )
    .unwrap();

    let resolved = resolve_server_config(&GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
    })
    .unwrap();
    let LogSinkConfig::File(sink) = &resolved.logging.sinks[0];
    let rotation = sink.rotation.expect("complete rotation configuration");
    assert_eq!(rotation.max_file_size_bytes(), 1024);
    assert_eq!(rotation.retained_files(), 2);

    let incomplete_path = isolated_config_path(&temp);
    std::fs::write(
        &incomplete_path,
        r#"
[[logging.sinks]]
path = "relay.log.jsonl"
max_file_size_bytes = 1024
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(incomplete_path),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains(
        "logging sink max_file_size_bytes and retained_files must be configured together"
    ));
}

#[test]
fn logging_rejects_invalid_level_format_missing_path_and_zero_queue() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();

    let bad_level = temp.path().join("bad-level.toml");
    std::fs::write(
        &bad_level,
        r#"
[logging]
level = "verbose"
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(bad_level),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("invalid logging level"));

    let bad_format = temp.path().join("bad-format.toml");
    std::fs::write(
        &bad_format,
        r#"
[logging]
stderr_format = "yaml"
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(bad_format),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("invalid logging format"));

    let missing_path = temp.path().join("missing-path.toml");
    std::fs::write(
        &missing_path,
        r#"
[[logging.sinks]]
format = "jsonl"
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(missing_path),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("requires path"));

    let zero_queue = temp.path().join("zero-queue.toml");
    std::fs::write(
        &zero_queue,
        r#"
[[logging.sinks]]
path = "relay.log"
queue_capacity = 0
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(zero_queue),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("queue_capacity must be greater than 0"));

    let over_max = temp.path().join("over-max.toml");
    std::fs::write(
        &over_max,
        format!(
            r#"
[[logging.sinks]]
path = "relay.log"
queue_capacity = {}
"#,
            MAX_FILE_SINK_QUEUE_ENTRIES + 1
        ),
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(over_max),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("exceeds maximum"));
}

#[test]
fn logging_rejects_invalid_sink_level_and_format() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();

    let bad_sink_level = temp.path().join("bad-sink-level.toml");
    std::fs::write(
        &bad_sink_level,
        r#"
[[logging.sinks]]
path = "relay.log"
level = "verbose"
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(bad_sink_level),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("invalid logging level"));

    let bad_sink_format = temp.path().join("bad-sink-format.toml");
    std::fs::write(
        &bad_sink_format,
        r#"
[[logging.sinks]]
path = "relay.log"
format = "yaml"
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(bad_sink_format),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("invalid logging format"));
}

#[test]
fn logging_rejects_unknown_section_and_sink_fields() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();

    let unknown_logging_field = temp.path().join("unknown-logging-field.toml");
    std::fs::write(
        &unknown_logging_field,
        r#"
[logging]
levle = "debug"
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(unknown_logging_field),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("unknown field `levle`"));

    let unknown_sink_field = temp.path().join("unknown-sink-field.toml");
    std::fs::write(
        &unknown_sink_field,
        r#"
[[logging.sinks]]
path = "relay.log"
queue_capcity = 32
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(unknown_sink_field),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("unknown field `queue_capcity`"));
}

#[test]
fn logging_rejects_empty_sink_path() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let config_path = isolated_config_path(&temp);
    std::fs::write(
        &config_path,
        r#"
[[logging.sinks]]
path = ""
"#,
    )
    .unwrap();
    let error = resolve_server_config(&GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("path must not be empty"));
}

#[test]
fn logging_config_does_not_read_rust_log() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let _env = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let previous = std::env::var_os("RUST_LOG");
    unsafe {
        std::env::set_var("RUST_LOG", "trace");
    }

    let temp = tempfile::tempdir().unwrap();
    let config_path = isolated_config_path(&temp);
    std::fs::write(
        &config_path,
        r#"
[logging]
level = "error"
"#,
    )
    .unwrap();
    let resolved = resolve_server_config(&GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
    })
    .unwrap();

    assert_eq!(resolved.logging.level, LogLevel::Error);
    assert!(resolved.logging.sinks.is_empty());

    unsafe {
        match previous {
            Some(value) => std::env::set_var("RUST_LOG", value),
            None => std::env::remove_var("RUST_LOG"),
        }
    }
}

#[test]
fn configuration_value_helpers_cover_empty_and_invalid_shapes() {
    let source = Path::new("plugins.toml");
    let mut scalar = toml::Value::String("not a table".into());
    let resolved =
        resolve_dynamic_plugin_refs(source, &mut scalar, &mut std::collections::HashSet::new())
            .unwrap();
    assert!(resolved.dynamic_plugins.is_empty());
    assert_eq!(
        resolved.dynamic_plugin_policy,
        DynamicPluginHostPolicy::default()
    );

    let value: toml::Value = toml::from_str(
        r#"
[plugins]
dynamic = []
[plugins.policy]
rules = []
[other]
enabled = true
"#,
    )
    .unwrap();
    let cleaned = remove_dynamic_plugin_sections(value);
    assert!(cleaned.get("plugins").is_none());
    assert_eq!(cleaned["other"]["enabled"].as_bool(), Some(true));
    assert_eq!(plugin_toml_runtime_value(json!({})), None);
    assert_eq!(
        plugin_toml_runtime_value(json!({"enabled": true})),
        Some(json!({"enabled": true}))
    );

    assert!(validate_auth_header("AUTH", "   ".into()).is_err());
    assert!(parse_env_body_limit("LIMIT", "not-a-number").is_err());
    assert_eq!(parse_env_body_limit("LIMIT", "17").unwrap(), 17);
}

#[test]
fn logging_path_and_sink_helpers_cover_lexical_fallbacks() {
    let temp = tempfile::tempdir().unwrap();
    let existing_parent = temp.path().join("logs");
    std::fs::create_dir_all(&existing_parent).unwrap();
    let canonical = std::fs::canonicalize(&existing_parent)
        .unwrap()
        .join("relay.log");
    assert_eq!(
        logging_path_identity(&existing_parent.join("relay.log")),
        canonical
    );

    let lexical = logging_path_identity(&temp.path().join("missing/../nested/relay.log"));
    assert!(lexical.ends_with("nested/relay.log"));
    assert_eq!(
        normalize_path_components(Path::new("a/./b/../c")),
        PathBuf::from("a/c")
    );
    assert_eq!(
        normalize_path_components(Path::new("single")),
        PathBuf::from("single")
    );

    let path = temp.path().join("coalesced.log");
    let lower = toml::Value::Table(toml::Table::from_iter([
        (
            "path".into(),
            toml::Value::String(path.display().to_string()),
        ),
        ("level".into(), toml::Value::String("info".into())),
    ]));
    let higher = toml::Value::Table(toml::Table::from_iter([
        (
            "path".into(),
            toml::Value::String(path.display().to_string()),
        ),
        ("format".into(), toml::Value::String("json".into())),
    ]));
    let coalesced = coalesce_logging_sinks(vec![lower, higher]);
    assert_eq!(coalesced.len(), 1);
    assert_eq!(coalesced[0]["level"].as_str(), Some("info"));
    assert_eq!(coalesced[0]["format"].as_str(), Some("json"));

    let mut invalid_higher: toml::Value = toml::from_str("[logging]\nsinks = 'invalid'").unwrap();
    let lower = toml::Value::Table(toml::Table::new());
    merge_logging_sinks_by_path(&lower, &mut invalid_higher);
    assert_eq!(invalid_higher["logging"]["sinks"].as_str(), Some("invalid"));
}

#[test]
fn dynamic_plugin_identity_allows_worker_without_manifest() {
    let component = ActiveDynamicPluginComponent {
        plugin_id: "acme.manual-worker".into(),
        kind: DynamicPluginKind::Worker,
        lifecycle_generation: 7,
        manifest_ref: None,
        environment_ref: None,
        config: serde_json::Map::new(),
        activation_snapshot: None,
    };
    let identity = dynamic_plugin_bootstrap_identity(&component).unwrap();
    assert_eq!(identity["plugin_id"], "acme.manual-worker");
    assert_eq!(identity["manifest"], Value::Null);
    assert_eq!(identity["lifecycle_generation"], 7);
}
