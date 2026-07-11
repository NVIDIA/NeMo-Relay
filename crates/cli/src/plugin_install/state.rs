// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Install directory layout, persisted state, and filesystem helpers.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::config::IntegrationHost;
use crate::install_generation::GENERATION_FILE_NAME;

use super::PLUGIN_NAME;

#[derive(Debug, Clone)]
pub(super) struct PluginInstallOptions {
    pub(super) install_dir: PathBuf,
    pub(super) operation_lock_dir: PathBuf,
    pub(super) force: bool,
    pub(super) dry_run: bool,
    pub(super) skip_doctor: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum HostSelectionMode {
    Install,
    InstalledState,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct HostRegistrationProgress {
    pub(super) host_plugin_added: bool,
    pub(super) host_marketplace_added: bool,
}

impl HostRegistrationProgress {
    pub(super) fn any_added(self) -> bool {
        self.host_plugin_added || self.host_marketplace_added
    }
}

#[derive(Debug, Clone)]
pub(super) struct PluginLayout {
    pub(super) host: IntegrationHost,
    pub(super) marketplace_root: PathBuf,
    pub(super) marketplace_manifest: PathBuf,
    pub(super) plugin_root: PathBuf,
    pub(super) plugin_manifest: PathBuf,
    pub(super) mcp_config: PathBuf,
    pub(super) generation_fence: PathBuf,
    pub(super) generation_lock: PathBuf,
    pub(super) hooks_path: PathBuf,
    pub(super) state_path: PathBuf,
}

impl PluginLayout {
    pub(super) fn new(host: IntegrationHost, install_dir: &Path) -> Self {
        let marketplace_root = install_dir.join(format!("{}-marketplace", host.as_arg()));
        let marketplace_manifest = match host {
            IntegrationHost::Codex => marketplace_root
                .join(".agents")
                .join("plugins")
                .join("marketplace.json"),
            IntegrationHost::ClaudeCode => marketplace_root
                .join(".claude-plugin")
                .join("marketplace.json"),
            IntegrationHost::Hermes | IntegrationHost::All => {
                unreachable!("all is expanded before layout resolution")
            }
        };
        let plugin_root = marketplace_root.join("plugins").join(PLUGIN_NAME);
        let plugin_manifest = match host {
            IntegrationHost::Codex => plugin_root.join(".codex-plugin").join("plugin.json"),
            IntegrationHost::ClaudeCode => plugin_root.join(".claude-plugin").join("plugin.json"),
            IntegrationHost::Hermes | IntegrationHost::All => {
                unreachable!("all is expanded before layout resolution")
            }
        };
        let mcp_config = plugin_root.join(".mcp.json");
        let generation_fence = plugin_root.join(GENERATION_FILE_NAME);
        let generation_lock =
            install_dir.join(format!(".nemo-relay-{}-mcp-generation.lock", host.as_arg()));
        let hooks_path = plugin_root.join("hooks").join("hooks.json");
        let state_path = state_path(host, install_dir);
        Self {
            host,
            marketplace_root,
            marketplace_manifest,
            plugin_root,
            plugin_manifest,
            mcp_config,
            generation_fence,
            generation_lock,
            hooks_path,
            state_path,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct PluginState {
    pub(super) marketplace_root: PathBuf,
    pub(super) plugin_root: PathBuf,
    pub(super) host_plugin_removed: bool,
    pub(super) host_marketplace_removed: bool,
    pub(super) plugin_setup_installed: bool,
}

pub(super) trait CanonicalizeOrSelf {
    fn canonicalize_or_self(self) -> Self;
}

impl CanonicalizeOrSelf for PathBuf {
    fn canonicalize_or_self(self) -> Self {
        self.canonicalize().unwrap_or(self)
    }
}

pub(super) fn default_install_dir() -> PathBuf {
    default_install_dir_for(
        env::consts::OS,
        env::var_os("HOME"),
        env::var_os("USERPROFILE"),
        env::var_os("LOCALAPPDATA"),
        env::var_os("XDG_DATA_HOME"),
    )
}

pub(super) fn default_install_dir_for(
    os: &str,
    home: Option<OsString>,
    userprofile: Option<OsString>,
    localappdata: Option<OsString>,
    xdg_data_home: Option<OsString>,
) -> PathBuf {
    let home = home
        .or(userprofile)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    match os {
        "macos" => home
            .join("Library")
            .join("Application Support")
            .join("nemo-relay")
            .join("plugins"),
        "windows" => localappdata
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join("AppData").join("Local"))
            .join("nemo-relay")
            .join("plugins"),
        _ => xdg_data_home
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local").join("share"))
            .join("nemo-relay")
            .join("plugins"),
    }
}

pub(super) fn write_state(
    layout: &PluginLayout,
    options: &PluginInstallOptions,
) -> Result<(), String> {
    write_state_for_host(
        layout.host,
        &PluginState {
            marketplace_root: layout.marketplace_root.clone(),
            plugin_root: layout.plugin_root.clone(),
            host_plugin_removed: false,
            host_marketplace_removed: false,
            plugin_setup_installed: false,
        },
        layout
            .state_path
            .parent()
            .expect("state_path should have a parent directory"),
        options,
    )
}

pub(super) fn mark_plugin_setup_installed(
    host: IntegrationHost,
    layout: &PluginLayout,
    options: &PluginInstallOptions,
) -> Result<(), String> {
    let mut state = read_state(host, &options.install_dir).unwrap_or_else(|| PluginState {
        marketplace_root: layout.marketplace_root.clone(),
        plugin_root: layout.plugin_root.clone(),
        host_plugin_removed: false,
        host_marketplace_removed: false,
        plugin_setup_installed: false,
    });
    state.plugin_setup_installed = true;
    write_state_for_host(host, &state, &options.install_dir, options)
}

pub(super) fn write_state_for_host(
    host: IntegrationHost,
    state: &PluginState,
    install_dir: &Path,
    options: &PluginInstallOptions,
) -> Result<(), String> {
    let path = state_path(host, install_dir);
    if options.dry_run {
        println!("write {}", path.display());
        return Ok(());
    }
    write_json(
        &path,
        &json!({
            "host": host.as_arg(),
            "marketplaceRoot": state.marketplace_root,
            "pluginRoot": state.plugin_root,
            "hostUnregistered": state.host_plugin_removed && state.host_marketplace_removed,
            "hostPluginRemoved": state.host_plugin_removed,
            "hostMarketplaceRemoved": state.host_marketplace_removed,
            "pluginSetupInstalled": state.plugin_setup_installed
        }),
    )
}

pub(super) fn read_state(host: IntegrationHost, install_dir: &Path) -> Option<PluginState> {
    let raw = fs::read_to_string(state_path(host, install_dir)).ok()?;
    let value = serde_json::from_str::<Value>(&raw).ok()?;
    let legacy_host_unregistered = value
        .get("hostUnregistered")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Some(PluginState {
        marketplace_root: PathBuf::from(value.get("marketplaceRoot")?.as_str()?),
        plugin_root: PathBuf::from(value.get("pluginRoot")?.as_str()?),
        host_plugin_removed: value
            .get("hostPluginRemoved")
            .and_then(Value::as_bool)
            .unwrap_or(legacy_host_unregistered),
        host_marketplace_removed: value
            .get("hostMarketplaceRemoved")
            .and_then(Value::as_bool)
            .unwrap_or(legacy_host_unregistered),
        plugin_setup_installed: value
            .get("pluginSetupInstalled")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    })
}

pub(super) fn state_path(host: IntegrationHost, install_dir: &Path) -> PathBuf {
    install_dir.join(format!("{}.json", host.as_arg()))
}

pub(super) fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    crate::file_io::atomic_write(path, &bytes)
}

pub(super) fn remove_path(path: &Path, options: &PluginInstallOptions) -> Result<(), String> {
    if options.dry_run {
        println!("remove {}", path.display());
        return Ok(());
    }
    fs::remove_dir_all(path)
        .or_else(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(error)
            }
        })
        .or_else(|_| fs::remove_file(path))
        .or_else(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(format!("failed to remove {}: {error}", path.display()))
            }
        })
}
