// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hermes-owned MCP and lifecycle-hook configuration.

mod config;
mod files;
mod trust;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde_json::{Map, Value, json};

#[cfg(test)]
use self::config::persistent_hook_command_for_platform;
use self::config::{
    MCP_SERVER_NAME, expected_mcp_server, forwarded_environment_names, is_managed_hook_command,
    is_managed_mcp_server, parse_yaml_object, persistent_config, relay_is_executable,
    remove_managed_mcp, strip_managed_hooks, user_config_path_with_override, yaml_bytes,
};
use self::files::{
    FileSnapshot, INSTALL_LOCK_TIMEOUT, PersistentPaths, acquire_allowlist_lock,
    acquire_install_lock, read_optional_utf8, remove_optional_file, replace_optional_file,
};
use self::trust::{json_bytes, parse_json_object, trusted_hooks, verify_trust};
use crate::error::CliError;
use crate::file_io::atomic_write;
use crate::install_generation::GENERATION_FILE_ENV;
#[cfg(test)]
use crate::install_generation::GENERATION_FILE_NAME;
use crate::installer::HERMES_HOOK_EVENTS;
use crate::sidecar::DEFAULT_BIND;
pub(crate) use config::{persistent_hook_command, transparent_config};

/// Hermes host configuration is user-owned even when Relay itself uses project configuration.
/// Project-specific Relay behavior remains available through transparent `nemo-relay run`.
pub(crate) fn user_config_path(default_home: &Path) -> PathBuf {
    user_config_path_with_override(default_home, env::var_os("HERMES_HOME"))
}

pub(crate) fn install_persistent(config: &Path, relay: &Path) -> Result<Vec<PathBuf>, CliError> {
    let relay = relay.canonicalize().unwrap_or_else(|_| relay.to_path_buf());
    let relay = crate::plugin_host::portable_executable_path(relay);
    if !relay_is_executable(&relay) {
        return Err(CliError::Install(format!(
            "nemo-relay executable is missing or not executable at {}",
            relay.display()
        )));
    }
    let paths = PersistentPaths::for_config(config.to_path_buf())?;
    let _lock =
        acquire_install_lock(&paths.config, INSTALL_LOCK_TIMEOUT).map_err(CliError::Install)?;
    let _allowlist_lock = acquire_allowlist_lock(&paths.allowlist, INSTALL_LOCK_TIMEOUT)
        .map_err(CliError::Install)?;
    let plugin_config = crate::config::user_plugin_runtime_config()?;
    let environment = env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .collect::<Vec<_>>();
    install_persistent_with(
        paths,
        &relay,
        &environment,
        plugin_config.as_ref(),
        SystemTime::now(),
        atomic_write,
    )
}

pub(crate) fn persistent_state_exists(config: &Path) -> bool {
    PersistentPaths::for_config(config.to_path_buf())
        .ok()
        .and_then(|paths| persistent_paths_have_managed_state(&paths).ok())
        .unwrap_or(false)
}

pub(crate) fn uninstall_persistent(config: &Path) -> Result<Vec<PathBuf>, CliError> {
    let paths = PersistentPaths::for_config(config.to_path_buf())?;
    if !persistent_paths_have_managed_state(&paths)? {
        return Ok(Vec::new());
    }
    let _lock =
        acquire_install_lock(&paths.config, INSTALL_LOCK_TIMEOUT).map_err(CliError::Install)?;
    let _allowlist_lock = acquire_allowlist_lock(&paths.allowlist, INSTALL_LOCK_TIMEOUT)
        .map_err(CliError::Install)?;
    if !persistent_paths_have_managed_state(&paths)? {
        return Ok(Vec::new());
    }
    uninstall_persistent_with(paths, atomic_write)
}

pub(crate) fn retire_persistent_gateway() -> Result<(), CliError> {
    crate::plugin_host::stop_plugin_gateway().map_err(CliError::Install)
}

fn persistent_paths_have_managed_state(paths: &PersistentPaths) -> Result<bool, CliError> {
    if paths.generation.exists() {
        return Ok(true);
    }
    if let Some(raw) = read_optional_utf8(&paths.config)? {
        let config = parse_yaml_object(Some(&raw), "Hermes config")?;
        if config_has_managed_state(&config) {
            return Ok(true);
        }
    }
    if let Some(raw) = read_optional_utf8(&paths.allowlist)? {
        let allowlist = parse_json_object(Some(&raw), "Hermes shell-hook allowlist")?;
        if allowlist_has_managed_state(&allowlist) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn config_has_managed_state(config: &Value) -> bool {
    config
        .get("mcp_servers")
        .and_then(|servers| servers.get(MCP_SERVER_NAME))
        .is_some_and(is_managed_mcp_server)
        || config
            .get("hooks")
            .and_then(Value::as_object)
            .into_iter()
            .flat_map(Map::values)
            .filter_map(Value::as_array)
            .flatten()
            .filter_map(|entry| entry.get("command").and_then(Value::as_str))
            .any(is_managed_hook_command)
}

fn allowlist_has_managed_state(allowlist: &Value) -> bool {
    allowlist
        .get("approvals")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("command").and_then(Value::as_str))
        .any(is_managed_hook_command)
}

pub(crate) fn diagnose_persistent(config_path: &Path) -> Result<String, String> {
    let paths = PersistentPaths::for_config(config_path.to_path_buf())
        .map_err(|error| error.to_string())?;
    let raw = fs::read_to_string(&paths.config)
        .map_err(|error| format!("failed to read {}: {error}", paths.config.display()))?;
    let config = parse_yaml_object(Some(&raw), "Hermes config").map_err(|e| e.to_string())?;
    let relay = relay_executable_from_config(&config)?;
    if !relay_is_executable(&relay) {
        return Err(format!(
            "configured nemo-relay executable is missing or not executable at {}",
            relay.display()
        ));
    }
    let command = persistent_hook_command(&relay);
    verify_hook_definitions(&config, &command)?;
    verify_trust(&paths.allowlist, &command)?;

    let mcp_env = config["mcp_servers"][MCP_SERVER_NAME]
        .get("env")
        .and_then(Value::as_object)
        .ok_or_else(|| "Hermes Relay MCP environment is missing".to_string())?;
    if mcp_env.get("NEMO_RELAY_GATEWAY_BIND") != Some(&json!(DEFAULT_BIND)) {
        return Err(format!(
            "Hermes Relay MCP must use the shared gateway bind {DEFAULT_BIND}"
        ));
    }
    let configured_generation = mcp_env
        .get(GENERATION_FILE_ENV)
        .and_then(Value::as_str)
        .ok_or_else(|| "Hermes Relay MCP generation fence is missing".to_string())?;
    if Path::new(configured_generation) != paths.generation {
        return Err("Hermes Relay MCP generation fence points at the wrong file".into());
    }
    let generation = fs::read_to_string(&paths.generation)
        .map_err(|error| format!("failed to read {}: {error}", paths.generation.display()))?;
    if generation.trim().is_empty() || generation.trim().starts_with("retired:") {
        return Err("Hermes Relay MCP generation fence is not active".into());
    }

    let plugin_config = crate::config::user_plugin_runtime_config().map_err(|e| e.to_string())?;
    let environment = env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .collect::<Vec<_>>();
    let missing = forwarded_environment_names(&environment, plugin_config.as_ref())
        .into_iter()
        .filter(|name| !mcp_env.contains_key(name))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Hermes Relay MCP is missing environment names {}; run `nemo-relay install hermes --force`",
            missing.join(", ")
        ));
    }
    Ok(format!(
        "MCP lifecycle and {} hooks trusted at {}",
        HERMES_HOOK_EVENTS.len(),
        paths.config.display()
    ))
}

/// Returns the exact Relay binary configured for Hermes's managed MCP client.
///
/// Doctor uses this path instead of the currently running binary so it verifies the executable
/// that Hermes will actually launch.
pub(crate) fn configured_relay_executable(config_path: &Path) -> Result<PathBuf, String> {
    let raw = fs::read_to_string(config_path)
        .map_err(|error| format!("failed to read {}: {error}", config_path.display()))?;
    let config = parse_yaml_object(Some(&raw), "Hermes config").map_err(|e| e.to_string())?;
    let relay = relay_executable_from_config(&config)?;
    if !relay_is_executable(&relay) {
        return Err(format!(
            "configured nemo-relay executable is missing or not executable at {}",
            relay.display()
        ));
    }
    Ok(relay)
}

fn relay_executable_from_config(config: &Value) -> Result<PathBuf, String> {
    let server = config
        .pointer("/mcp_servers/nemo-relay")
        .ok_or_else(|| "Hermes MCP server `nemo-relay` is missing".to_string())?;
    if !is_managed_mcp_server(server) {
        return Err("Hermes MCP server `nemo-relay` is not a managed Relay MCP client".into());
    }
    Ok(PathBuf::from(
        server
            .get("command")
            .and_then(Value::as_str)
            .expect("managed MCP server has a string command"),
    ))
}

fn install_persistent_with<W>(
    paths: PersistentPaths,
    relay: &Path,
    environment: &[String],
    plugin_config: Option<&Value>,
    now: SystemTime,
    mut write: W,
) -> Result<Vec<PathBuf>, CliError>
where
    W: FnMut(&Path, &[u8]) -> Result<(), String>,
{
    let snapshots = paths
        .all()
        .iter()
        .map(|path| FileSnapshot::capture(path))
        .collect::<Result<Vec<_>, _>>()?;
    let existing_config = read_optional_utf8(&paths.config)?;
    let existing_allowlist = read_optional_utf8(&paths.allowlist)?;
    let command = persistent_hook_command(relay);
    let environment = forwarded_environment_names(environment, plugin_config);
    let token = uuid::Uuid::now_v7().to_string();
    let config = persistent_config(
        existing_config.as_deref(),
        relay,
        &command,
        &paths.generation,
        &environment,
    )?;
    let allowlist = trusted_hooks(existing_allowlist.as_deref(), &command, relay, now)?;
    let config = yaml_bytes(&config)?;
    let allowlist = json_bytes(&allowlist)?;
    let generation = format!("{token}\n").into_bytes();

    let result = (|| {
        // Trust is published before config so Hermes never observes a configured hook without
        // its exact approval. The config write is the transaction's commit point.
        write(&paths.generation, &generation)?;
        write(&paths.allowlist, &allowlist)?;
        write(&paths.config, &config)?;
        verify_install(&paths, relay, &command, &environment, &token)
    })();
    if let Err(error) = result {
        return rollback_error("install", error, &snapshots, &mut write);
    }
    Ok(paths.all().into_iter().collect())
}

fn uninstall_persistent_with<W>(
    paths: PersistentPaths,
    mut write: W,
) -> Result<Vec<PathBuf>, CliError>
where
    W: FnMut(&Path, &[u8]) -> Result<(), String>,
{
    let affected = paths
        .all()
        .into_iter()
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    let snapshots = paths
        .all()
        .iter()
        .map(|path| FileSnapshot::capture(path))
        .collect::<Result<Vec<_>, _>>()?;
    let config = read_optional_utf8(&paths.config)?
        .map(|raw| {
            let mut root = parse_yaml_object(Some(&raw), "Hermes config")?;
            strip_managed_hooks(&mut root)?;
            remove_managed_mcp(&mut root)?;
            if root.as_object().is_some_and(Map::is_empty) {
                Ok(None)
            } else {
                yaml_bytes(&root).map(Some)
            }
        })
        .transpose()?
        .flatten();
    let allowlist = read_optional_utf8(&paths.allowlist)?
        .map(|raw| {
            let mut root = parse_json_object(Some(&raw), "Hermes shell-hook allowlist")?;
            let object = root
                .as_object_mut()
                .expect("allowlist root checked as object");
            if let Some(approvals) = object.get_mut("approvals") {
                let approvals = approvals.as_array_mut().ok_or_else(|| {
                    CliError::Install(
                        "Hermes shell-hook allowlist approvals must be an array".into(),
                    )
                })?;
                approvals.retain(|entry| {
                    entry
                        .get("command")
                        .and_then(Value::as_str)
                        .is_none_or(|command| !is_managed_hook_command(command))
                });
                if approvals.is_empty() {
                    object.remove("approvals");
                }
            }
            if object.is_empty() {
                Ok(None)
            } else {
                json_bytes(&root).map(Some)
            }
        })
        .transpose()?
        .flatten();

    let result = (|| {
        remove_optional_file(&paths.generation)?;
        replace_optional_file(&paths.allowlist, allowlist.as_deref(), &mut write)?;
        replace_optional_file(&paths.config, config.as_deref(), &mut write)?;
        verify_uninstall(&paths)
    })();
    if let Err(error) = result {
        return rollback_error("uninstall", error, &snapshots, &mut write);
    }
    Ok(affected)
}

fn rollback_error<T, W>(
    operation: &str,
    error: String,
    snapshots: &[FileSnapshot],
    write: &mut W,
) -> Result<T, CliError>
where
    W: FnMut(&Path, &[u8]) -> Result<(), String>,
{
    let rollback_errors = snapshots
        .iter()
        .rev()
        .filter_map(|snapshot| snapshot.restore(write).err())
        .collect::<Vec<_>>();
    let rollback = if rollback_errors.is_empty() {
        String::new()
    } else {
        format!("; rollback also failed: {}", rollback_errors.join("; "))
    };
    Err(CliError::Install(format!(
        "failed to {operation} Hermes MCP integration: {error}{rollback}"
    )))
}

fn verify_install(
    paths: &PersistentPaths,
    relay: &Path,
    command: &str,
    environment: &[String],
    token: &str,
) -> Result<(), String> {
    let raw = fs::read_to_string(&paths.config)
        .map_err(|error| format!("failed to verify {}: {error}", paths.config.display()))?;
    let config = parse_yaml_object(Some(&raw), "Hermes config").map_err(|e| e.to_string())?;
    let expected = expected_mcp_server(relay, &paths.generation, environment);
    if config.pointer("/mcp_servers/nemo-relay") != Some(&expected) {
        return Err("Hermes MCP server did not persist exactly".into());
    }
    verify_hook_definitions(&config, command)?;
    verify_trust(&paths.allowlist, command)?;

    let raw = fs::read_to_string(&paths.allowlist)
        .map_err(|error| format!("failed to verify {}: {error}", paths.allowlist.display()))?;
    let allowlist =
        parse_json_object(Some(&raw), "Hermes shell-hook allowlist").map_err(|e| e.to_string())?;
    let approvals = allowlist["approvals"]
        .as_array()
        .expect("trust verification checked approvals");
    if approvals
        .iter()
        .filter_map(|entry| entry.get("command").and_then(Value::as_str))
        .any(|candidate| is_managed_hook_command(candidate) && candidate != command)
    {
        return Err("stale Hermes Relay hook approvals remain".into());
    }

    let actual = fs::read_to_string(&paths.generation)
        .map_err(|error| format!("failed to verify {}: {error}", paths.generation.display()))?;
    if actual.trim() != token {
        return Err("Hermes MCP generation did not persist exactly".into());
    }
    Ok(())
}

fn verify_hook_definitions(config: &Value, command: &str) -> Result<(), String> {
    for event in HERMES_HOOK_EVENTS {
        let groups = config
            .pointer(&format!("/hooks/{event}"))
            .and_then(Value::as_array)
            .ok_or_else(|| format!("Hermes hook {event} is missing"))?;
        let matching = groups
            .iter()
            .filter(|group| group.get("command").and_then(Value::as_str) == Some(command))
            .count();
        let managed = groups
            .iter()
            .filter_map(|group| group.get("command").and_then(Value::as_str))
            .filter(|candidate| is_managed_hook_command(candidate))
            .count();
        if matching != 1 || managed != 1 {
            return Err(format!(
                "Hermes hook {event} expected exactly one trusted Relay handler"
            ));
        }
    }
    let mut managed = config
        .get("hooks")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|hooks| hooks.iter())
        .flat_map(|(event, groups)| {
            groups.as_array().into_iter().flatten().filter_map(|group| {
                let candidate = group.get("command").and_then(Value::as_str)?;
                is_managed_hook_command(candidate).then_some((event.as_str(), candidate))
            })
        })
        .collect::<Vec<_>>();
    managed.sort_unstable();
    let mut expected = HERMES_HOOK_EVENTS
        .iter()
        .map(|event| (*event, command))
        .collect::<Vec<_>>();
    expected.sort_unstable();
    if managed != expected {
        return Err("Hermes config contains an unexpected Relay hook handler".into());
    }
    Ok(())
}

fn verify_uninstall(paths: &PersistentPaths) -> Result<(), String> {
    if paths.generation.exists() {
        return Err("Hermes MCP generation fence still exists".into());
    }
    if let Some(raw) = read_optional_utf8(&paths.config).map_err(|error| error.to_string())? {
        let config = parse_yaml_object(Some(&raw), "Hermes config").map_err(|e| e.to_string())?;
        if config_has_managed_state(&config) {
            return Err("managed Hermes Relay config still exists".into());
        }
    }
    if let Some(raw) = read_optional_utf8(&paths.allowlist).map_err(|error| error.to_string())? {
        let allowlist = parse_json_object(Some(&raw), "Hermes shell-hook allowlist")
            .map_err(|e| e.to_string())?;
        if allowlist_has_managed_state(&allowlist) {
            return Err("managed Hermes Relay trust approval still exists".into());
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/coverage/hermes_tests.rs"]
mod tests;
