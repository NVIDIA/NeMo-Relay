// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::io::Write;
use std::net::TcpListener;
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tempfile::tempdir;

use super::*;

fn home_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct HomeScope<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
    prev_home: Option<std::ffi::OsString>,
    prev_userprofile: Option<std::ffi::OsString>,
}

impl<'a> HomeScope<'a> {
    fn enter(path: &std::path::Path) -> Self {
        let guard = home_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let prev_home = std::env::var_os("HOME");
        let prev_userprofile = std::env::var_os("USERPROFILE");
        // SAFETY: This test holds a process-wide mutex for the lifetime of the env override.
        unsafe {
            std::env::set_var("HOME", path);
            std::env::remove_var("USERPROFILE");
        }
        Self {
            _guard: guard,
            prev_home,
            prev_userprofile,
        }
    }
}

impl<'a> Drop for HomeScope<'a> {
    fn drop(&mut self) {
        // SAFETY: This restores the process environment while the mutex is still held.
        unsafe {
            match self.prev_home.take() {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match self.prev_userprofile.take() {
                Some(value) => std::env::set_var("USERPROFILE", value),
                None => std::env::remove_var("USERPROFILE"),
            }
        }
    }
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set_path(key: &'static str, value: &std::path::Path) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: Callers hold the process-wide environment mutex through HomeScope.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: This restores the process environment while HomeScope still holds the mutex.
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[test]
fn backup_preserves_first_snapshot() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();

    backup(&path).unwrap();
    fs::write(&path, "model_provider = \"nemo-relay-openai\"\n").unwrap();
    backup(&path).unwrap();

    assert_eq!(
        fs::read_to_string(backup_path(&path)).unwrap(),
        "model_provider = \"openai\"\n"
    );
}

#[test]
fn atomic_write_replaces_existing_destination() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "old\n").unwrap();

    atomic_write(&path, b"new\n").unwrap();

    assert_eq!(fs::read_to_string(&path).unwrap(), "new\n");
}

#[test]
fn repeated_codex_install_does_not_overwrite_original_backup() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();

    install_codex_config(&path, DEFAULT_URL).unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();

    assert_eq!(
        fs::read_to_string(backup_path(&path)).unwrap(),
        "model_provider = \"openai\"\n"
    );
}

#[test]
fn codex_install_backs_up_when_relay_provider_table_is_not_active() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "openai"

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    install_codex_config(&path, DEFAULT_URL).unwrap();

    assert!(
        fs::read_to_string(backup_path(&path))
            .unwrap()
            .contains("model_provider = \"openai\"")
    );
}

#[test]
fn codex_install_backs_up_when_hooks_flag_changes_even_with_managed_provider() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = false

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    install_codex_config(&path, DEFAULT_URL).unwrap();

    let backup = fs::read_to_string(backup_path(&path)).unwrap();
    assert!(backup.contains("hooks = false"));
    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();
    assert!(updated.contains("hooks = false"));
}

#[test]
fn codex_provider_installed_requires_active_managed_provider() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let path = codex_dir.join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "openai"

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    assert!(!codex_provider_installed(DEFAULT_URL));
    install_codex_config(&path, DEFAULT_URL).unwrap();
    assert!(codex_provider_installed(DEFAULT_URL));
    assert!(!codex_provider_installed("http://127.0.0.1:47633"));
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = false

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();
    assert!(!codex_provider_installed(DEFAULT_URL));
}

#[test]
fn codex_hooks_installed_requires_generated_plugin_local_groups() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let path = codex_dir.join("hooks.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "nemo-relay plugin-shim hook codex --gateway-url http://127.0.0.1:47632",
                                "timeout": 30
                            }
                        ]
                    }
                ]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    assert!(!codex_hooks_installed(DEFAULT_URL).unwrap());
    install_codex_hooks(&path, DEFAULT_URL).unwrap();
    assert!(codex_hooks_installed(DEFAULT_URL).unwrap());
    assert!(!codex_hooks_installed("http://127.0.0.1:47633").unwrap());
}

#[test]
fn codex_doctor_allows_stopped_lazy_sidecar_when_static_setup_is_valid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    install_codex_config(&codex_dir.join("config.toml"), DEFAULT_URL).unwrap();
    install_codex_hooks(&codex_dir.join("hooks.json"), DEFAULT_URL).unwrap();

    let status = doctor(PluginShimDoctorCommand {
        agent: CodingAgent::Codex,
        gateway_url: DEFAULT_URL.into(),
    })
    .unwrap();

    assert_eq!(status, std::process::ExitCode::SUCCESS);
}

#[test]
fn codex_doctor_requires_enabled_hooks_feature() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = false

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();
    install_codex_hooks(&codex_dir.join("hooks.json"), DEFAULT_URL).unwrap();

    let status = doctor(PluginShimDoctorCommand {
        agent: CodingAgent::Codex,
        gateway_url: DEFAULT_URL.into(),
    })
    .unwrap();

    assert_eq!(status, std::process::ExitCode::FAILURE);
}

#[test]
fn codex_setup_persists_path_based_launcher_when_sidecar_binary_override_is_set() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let sidecar_override = dir.path().join("sidecar").join("nemo-relay");
    fs::create_dir_all(sidecar_override.parent().unwrap()).unwrap();
    fs::write(&sidecar_override, b"sidecar override").unwrap();
    let _binary_override = EnvVarGuard::set_path("NEMO_RELAY_PLUGIN_BINARY", &sidecar_override);

    install_codex(DEFAULT_URL).unwrap();

    let hooks_path = codex_dir.join("hooks.json");
    let hooks: Value = serde_json::from_str(&fs::read_to_string(&hooks_path).unwrap()).unwrap();
    let launcher_command = codex_hook_command(DEFAULT_URL);
    let sidecar_command = codex_hook_command_for_platform(&sidecar_override, DEFAULT_URL, false);
    assert!(event_contains_command(
        &hooks,
        "SessionStart",
        &launcher_command
    ));
    assert!(!event_contains_command(
        &hooks,
        "SessionStart",
        &sidecar_command
    ));
    assert!(codex_hooks_installed(DEFAULT_URL).unwrap());
    assert_eq!(
        doctor(PluginShimDoctorCommand {
            agent: CodingAgent::Codex,
            gateway_url: DEFAULT_URL.into(),
        })
        .unwrap(),
        std::process::ExitCode::SUCCESS
    );

    uninstall_codex(DEFAULT_URL).unwrap();
    let hooks: Value = serde_json::from_str(&fs::read_to_string(&hooks_path).unwrap()).unwrap();
    assert!(!event_contains_command(
        &hooks,
        "SessionStart",
        &launcher_command
    ));
}

#[test]
fn relay_binary_prefers_sidecar_binary_override() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let sidecar_override = dir.path().join("sidecar").join("nemo-relay");
    fs::create_dir_all(sidecar_override.parent().unwrap()).unwrap();
    fs::write(&sidecar_override, b"sidecar override").unwrap();
    let _binary_override = EnvVarGuard::set_path("NEMO_RELAY_PLUGIN_BINARY", &sidecar_override);

    assert_eq!(relay_binary().unwrap(), sidecar_override);
}

#[test]
fn codex_uninstall_without_backup_removes_managed_hooks_flag() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(!updated.contains("model_provider"));
    assert!(!updated.contains("nemo-relay-openai"));
    assert!(!updated.contains("hooks = true"));
}

#[test]
fn codex_uninstall_with_backup_preserves_user_changed_model_provider() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();
    fs::write(
        &path,
        r#"
model_provider = "local"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("model_provider = \"local\""));
    assert!(!updated.contains("nemo-relay-openai"));
    assert!(!backup_path(&path).exists());
}

#[test]
fn codex_uninstall_with_backup_preserves_user_changed_provider_table() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "Custom Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("model_provider = \"nemo-relay-openai\""));
    assert!(updated.contains("name = \"Custom Relay\""));
    assert!(updated.contains("nemo-relay-openai"));
    assert!(!backup_path(&path).exists());
}

#[test]
fn codex_uninstall_preserves_user_changed_provider_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:49999"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("model_provider = \"nemo-relay-openai\""));
    assert!(updated.contains("base_url = \"http://127.0.0.1:49999\""));
    assert!(!backup_path(&path).exists());
}

#[test]
fn codex_uninstall_without_backup_preserves_user_changed_provider_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:49999"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("model_provider = \"nemo-relay-openai\""));
    assert!(updated.contains("base_url = \"http://127.0.0.1:49999\""));
}

#[test]
fn codex_uninstall_preserves_hooks_feature_when_user_hooks_remain() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        r#"
model_provider = "openai"

[features]
hooks = false
"#,
    )
    .unwrap();

    install_codex(DEFAULT_URL).unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let mut hooks: Value = serde_json::from_str(&fs::read_to_string(&hooks_path).unwrap()).unwrap();
    hooks["hooks"]["SessionStart"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "hooks": [
                {
                    "type": "command",
                    "command": "custom-hook",
                    "timeout": 30
                }
            ]
        }));
    fs::write(&hooks_path, serde_json::to_vec_pretty(&hooks).unwrap()).unwrap();

    uninstall_codex(DEFAULT_URL).unwrap();

    let updated_config = fs::read_to_string(codex_dir.join("config.toml")).unwrap();
    assert!(updated_config.contains("hooks = true"));
    let updated_hooks: Value =
        serde_json::from_str(&fs::read_to_string(&hooks_path).unwrap()).unwrap();
    assert!(event_contains_command(
        &updated_hooks,
        "SessionStart",
        "custom-hook"
    ));
    assert!(
        !serde_json::to_string(&updated_hooks)
            .unwrap()
            .contains("plugin-shim hook codex")
    );
}

#[test]
fn codex_reinstall_uses_fresh_backup_after_prior_uninstall() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();

    install_codex_config(&path, DEFAULT_URL).unwrap();
    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    assert!(!backup_path(&path).exists());

    fs::write(&path, "model_provider = \"local\"\n").unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();
    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "model_provider = \"local\"\n"
    );
    assert!(!backup_path(&path).exists());
}

#[test]
fn claude_restore_without_backup_preserves_matching_user_relay_url() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": DEFAULT_URL,
                "OTHER": "kept"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    claude_provider(PluginShimProviderAction::Restore, DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some(DEFAULT_URL)
    );
    assert_eq!(json_env_string(&updated, "OTHER"), Some("kept"));
    assert!(!backup_path(&settings).exists());
}

#[test]
fn claude_enable_rolls_back_backup_when_settings_write_fails() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::create_dir(settings.with_extension("json.tmp")).unwrap();

    let error = claude_provider(PluginShimProviderAction::Enable, DEFAULT_URL).unwrap_err();

    assert!(error.contains("failed to write"));
    assert!(!backup_path(&settings).exists());
}

#[test]
fn claude_enable_does_not_back_up_when_env_shape_is_invalid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": "invalid"
        }))
        .unwrap(),
    )
    .unwrap();

    let error = claude_provider(PluginShimProviderAction::Enable, DEFAULT_URL).unwrap_err();

    assert!(error.contains("non-object env field"));
    assert!(!backup_path(&settings).exists());
    let unchanged: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(unchanged["env"], json!("invalid"));
}

#[test]
fn claude_restore_with_backup_preserves_user_settings_added_after_install() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                "ORIGINAL": "kept"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    claude_provider(PluginShimProviderAction::Enable, DEFAULT_URL).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": DEFAULT_URL,
                "ORIGINAL": "updated",
                "ADDED": "kept"
            },
            "theme": "dark"
        }))
        .unwrap(),
    )
    .unwrap();

    claude_provider(PluginShimProviderAction::Restore, DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some("https://api.anthropic.com")
    );
    assert_eq!(json_env_string(&updated, "ORIGINAL"), Some("updated"));
    assert_eq!(json_env_string(&updated, "ADDED"), Some("kept"));
    assert_eq!(updated["theme"], json!("dark"));
    assert!(!backup_path(&settings).exists());
}

#[test]
fn claude_restore_with_backup_preserves_user_changed_provider_url() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    claude_provider(PluginShimProviderAction::Enable, DEFAULT_URL).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:49999"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    claude_provider(PluginShimProviderAction::Restore, DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some("http://127.0.0.1:49999")
    );
    assert!(!backup_path(&settings).exists());
}

#[test]
fn claude_reinstall_uses_fresh_backup_after_prior_restore() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    claude_provider(PluginShimProviderAction::Enable, DEFAULT_URL).unwrap();
    claude_provider(PluginShimProviderAction::Restore, DEFAULT_URL).unwrap();
    assert!(!backup_path(&settings).exists());

    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://custom.example"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    claude_provider(PluginShimProviderAction::Enable, DEFAULT_URL).unwrap();
    claude_provider(PluginShimProviderAction::Restore, DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some("https://custom.example")
    );
    assert!(!backup_path(&settings).exists());
}

#[test]
fn stale_lock_is_repaired_after_grace_period_even_when_pid_file_exists() {
    let dir = tempdir().unwrap();
    let lock = dir.path().join("codex-sidecar.lock");
    fs::create_dir(&lock).unwrap();
    fs::write(
        dir.path().join("codex-sidecar.pid"),
        std::process::id().to_string(),
    )
    .unwrap();

    assert!(repair_stale_lock_after(&lock, Duration::ZERO));
    assert!(!lock.exists());
}

#[test]
fn codex_hook_command_uses_cmd_quoting_for_windows_paths() {
    let relay = std::path::PathBuf::from(r"C:\Program Files\NeMo 100%\bin\nemo-relay.exe");
    let command = codex_hook_command_for_platform(&relay, DEFAULT_URL, true);

    assert_eq!(
        command,
        r#""C:\Program Files\NeMo 100%%\bin\nemo-relay.exe" plugin-shim hook codex --gateway-url http://127.0.0.1:47632"#
    );
    assert_eq!(
        shell_quote_arg_for_platform("foo&bar", true),
        r#""foo^&bar""#
    );
}

#[test]
fn codex_hook_command_uses_posix_single_quote_escaping() {
    let relay = std::path::PathBuf::from("/tmp/NeMo $Relay`test'/bin/nemo-relay");
    let command = codex_hook_command_for_platform(&relay, DEFAULT_URL, false);

    assert_eq!(
        command,
        "'/tmp/NeMo $Relay`test'\\''/bin/nemo-relay' plugin-shim hook codex --gateway-url http://127.0.0.1:47632"
    );
    assert_eq!(shell_quote_arg_for_platform("", false), "''");
    assert_eq!(
        shell_quote_arg_for_platform(r"/tmp/path\with-backslash", false),
        r#"'/tmp/path\with-backslash'"#
    );
}

#[test]
fn hook_forward_connect_attempt_is_bounded() {
    let error = post_hook(CodingAgent::Codex, "http://127.0.0.1:9", b"{}").unwrap_err();

    assert!(error.contains("hook forward failed"));
}

#[test]
fn hook_http_response_requires_numeric_2xx_status() {
    assert_eq!(
        parse_http_response(b"HTTP/1.1 204 No Content\r\n\r\npayload").unwrap(),
        b"payload"
    );
    assert!(
        parse_http_response(b"HTTP/1.1 500 upstream 2 bad\r\n\r\npayload")
            .unwrap_err()
            .contains("HTTP/1.1 500 upstream 2 bad")
    );
    assert!(
        parse_http_response(b"HTTP/1.1 OK 2\r\n\r\npayload")
            .unwrap_err()
            .contains("HTTP/1.1 OK 2")
    );
}

#[test]
fn unready_sidecar_child_is_terminated_and_pid_removed() {
    let dir = tempdir().unwrap();
    let pid_path = dir.path().join("codex-sidecar.pid");
    let mut command = long_lived_command();
    let child = command.spawn().unwrap();
    fs::write(&pid_path, child.id().to_string()).unwrap();

    let error = terminate_unready_sidecar(child, &pid_path, DEFAULT_URL).unwrap_err();

    assert!(error.contains("terminated startup process"));
    assert!(!pid_path.exists());
}

#[test]
fn codex_uninstall_removes_only_exact_generated_hook_groups() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let command = codex_hook_command("http://127.0.0.1:47633");
    let generated = generated_hooks(CodingAgent::Codex, &command);
    let user_command = "custom-user-codex-hook";
    let config = json!({
        "hooks": {
            "SessionStart": [
                generated["hooks"]["SessionStart"][0].clone(),
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": user_command,
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    });
    fs::write(&path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

    uninstall_codex_hooks(&path, "http://127.0.0.1:47633").unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(event_contains_command(
        &updated,
        "SessionStart",
        user_command
    ));
    assert!(!generated_event_contains_group(
        &updated,
        "SessionStart",
        &generated["hooks"]["SessionStart"][0]
    ));
}

#[test]
fn codex_install_hooks_removes_prior_non_default_generated_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let old_command = codex_hook_command("http://127.0.0.1:47633");
    let new_command = codex_hook_command("http://127.0.0.1:47634");

    install_codex_hooks(&path, "http://127.0.0.1:47633").unwrap();
    install_codex_hooks(&path, "http://127.0.0.1:47634").unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &old_command
    ));
    assert!(event_contains_command(
        &updated,
        "SessionStart",
        &new_command
    ));
}

#[test]
fn codex_uninstall_hooks_removes_all_generated_url_variants_for_launcher() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let old_command = codex_hook_command("http://127.0.0.1:47633");
    let new_command = codex_hook_command("http://127.0.0.1:47634");
    let mut old_generated = generated_hooks(CodingAgent::Codex, &old_command);
    let new_generated = generated_hooks(CodingAgent::Codex, &new_command);
    old_generated["hooks"]["SessionStart"]
        .as_array_mut()
        .unwrap()
        .push(new_generated["hooks"]["SessionStart"][0].clone());
    fs::write(&path, serde_json::to_vec_pretty(&old_generated).unwrap()).unwrap();

    uninstall_codex_hooks(&path, "http://127.0.0.1:47634").unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &old_command
    ));
    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &new_command
    ));
}

#[cfg(windows)]
fn long_lived_command() -> std::process::Command {
    let mut command = std::process::Command::new("cmd");
    command.args(["/C", "ping -n 60 127.0.0.1 >NUL"]);
    command
}

#[cfg(not(windows))]
fn long_lived_command() -> std::process::Command {
    let mut command = std::process::Command::new("sh");
    command.args(["-c", "sleep 60"]);
    command
}

#[test]
fn codex_install_hooks_persist_custom_gateway_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");

    install_codex_hooks(&path, "http://127.0.0.1:47633").unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let command = updated["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        .as_str()
        .unwrap();

    assert!(command.contains("plugin-shim hook codex"));
    assert!(command.contains("--gateway-url http://127.0.0.1:47633"));
}

#[test]
fn codex_install_hooks_replaces_legacy_generated_command() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let relay = current_exe().unwrap();
    let legacy_command = legacy_codex_hook_command(&relay);
    let legacy = generated_hooks(CodingAgent::Codex, &legacy_command);
    fs::write(&path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();

    install_codex_hooks(&path, DEFAULT_URL).unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &legacy_command
    ));
    assert!(event_contains_command(
        &updated,
        "SessionStart",
        &codex_hook_command(DEFAULT_URL)
    ));
}

#[test]
fn codex_install_does_not_write_provider_config_when_hooks_are_invalid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        "model_provider = \"openai\"\n",
    )
    .unwrap();
    fs::write(codex_dir.join("hooks.json"), "{ invalid json").unwrap();

    let error = install_codex(DEFAULT_URL).unwrap_err();
    assert!(error.contains("invalid JSON"));

    assert_eq!(
        fs::read_to_string(codex_dir.join("config.toml")).unwrap(),
        "model_provider = \"openai\"\n"
    );
    assert!(!backup_path(&codex_dir.join("config.toml")).exists());
}

#[test]
fn codex_install_does_not_write_hooks_when_config_is_invalid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(codex_dir.join("config.toml"), "model_provider = [").unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let original_hooks = serde_json::to_vec_pretty(&json!({
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "custom-hook",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }))
    .unwrap();
    fs::write(&hooks_path, &original_hooks).unwrap();

    let error = install_codex(DEFAULT_URL).unwrap_err();
    assert!(error.contains("invalid TOML"));

    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert!(!backup_path(&hooks_path).exists());
}

#[test]
fn codex_install_does_not_write_hooks_when_config_is_not_readable() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::create_dir(codex_dir.join("config.toml")).unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let original_hooks = serde_json::to_vec_pretty(&json!({
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "custom-hook",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }))
    .unwrap();
    fs::write(&hooks_path, &original_hooks).unwrap();

    let error = install_codex(DEFAULT_URL).unwrap_err();
    assert!(error.contains("failed to read"));

    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert!(!backup_path(&hooks_path).exists());
}

#[test]
fn codex_install_config_rolls_back_backup_when_write_fails() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();
    fs::create_dir(path.with_extension("toml.tmp")).unwrap();

    let error = install_codex_config(&path, DEFAULT_URL).unwrap_err();

    assert!(error.contains("failed to write"));
    assert!(!backup_path(&path).exists());
}

#[test]
fn codex_install_rolls_back_hooks_backup_when_hook_merge_fails() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        "model_provider = \"openai\"\n",
    )
    .unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let original_hooks = serde_json::to_vec_pretty(&json!({
        "hooks": {
            "SessionStart": "invalid"
        }
    }))
    .unwrap();
    fs::write(&hooks_path, &original_hooks).unwrap();

    let error = install_codex(DEFAULT_URL).unwrap_err();

    assert!(error.contains("SessionStart hooks must be an array"));
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert!(!backup_path(&hooks_path).exists());
    assert_eq!(
        fs::read_to_string(codex_dir.join("config.toml")).unwrap(),
        "model_provider = \"openai\"\n"
    );
}

#[test]
fn codex_uninstall_rolls_back_hooks_when_provider_config_is_invalid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(codex_dir.join("config.toml"), "model_provider = [").unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    install_codex_hooks(&hooks_path, DEFAULT_URL).unwrap();
    let original_hooks = fs::read(&hooks_path).unwrap();

    let error = uninstall_codex(DEFAULT_URL).unwrap_err();

    assert!(error.contains("invalid TOML"));
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
}

#[test]
fn codex_install_rolls_back_hooks_when_provider_config_write_fails() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        "model_provider = \"openai\"\n",
    )
    .unwrap();
    fs::create_dir(codex_dir.join("config.toml.tmp")).unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let original_hooks = serde_json::to_vec_pretty(&json!({
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "custom-hook",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }))
    .unwrap();
    fs::write(&hooks_path, &original_hooks).unwrap();

    let error = install_codex(DEFAULT_URL).unwrap_err();

    assert!(error.contains("failed to write"));
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert!(!backup_path(&hooks_path).exists());
    assert_eq!(
        fs::read_to_string(codex_dir.join("config.toml")).unwrap(),
        "model_provider = \"openai\"\n"
    );
}

#[test]
fn codex_uninstall_hooks_removes_legacy_generated_command() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let relay = current_exe().unwrap();
    let legacy_command = legacy_codex_hook_command(&relay);
    let legacy = generated_hooks(CodingAgent::Codex, &legacy_command);
    fs::write(&path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();

    uninstall_codex_hooks(&path, DEFAULT_URL).unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &legacy_command
    ));
}

#[test]
fn codex_provider_gateway_url_reads_managed_provider_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
[model_providers.nemo-relay-openai]
base_url = "http://127.0.0.1:47633"
"#,
    )
    .unwrap();

    assert_eq!(
        codex_provider_gateway_url(&path).as_deref(),
        Some("http://127.0.0.1:47633")
    );
}

#[test]
fn healthz_times_out_for_bad_port_occupant() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        thread::sleep(Duration::from_secs(2));
        let _ = stream.write_all(b"HTTP/1.1 200 OK\r\n\r\n");
    });

    let started = Instant::now();
    assert!(!healthz(&format!("http://127.0.0.1:{port}")));
    assert!(started.elapsed() < Duration::from_secs(2));
    handle.join().unwrap();
}

fn event_contains_command(config: &Value, event: &str, command: &str) -> bool {
    config
        .get("hooks")
        .and_then(Value::as_object)
        .and_then(|hooks| hooks.get(event))
        .and_then(Value::as_array)
        .is_some_and(|groups| {
            groups.iter().any(|group| {
                group
                    .get("hooks")
                    .and_then(Value::as_array)
                    .is_some_and(|hooks| {
                        hooks.iter().any(|hook| {
                            hook.get("command").and_then(Value::as_str) == Some(command)
                        })
                    })
            })
        })
}
