// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hermes-owned MCP and lifecycle-hook configuration.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde_json::{Map, Value, json};

#[cfg(test)]
use super::config::persistent_hook_command_for_platform;
use super::config::{
    MCP_SERVER_NAME, expected_mcp_server, forwarded_environment_names, owned_install_command,
    parse_yaml_object, persistent_config, relay_is_executable, remove_owned_mcp, strip_owned_hooks,
    user_config_path_with_override, yaml_bytes,
};
pub(crate) use super::config::{persistent_hook_command, transparent_config};
use super::files::{
    FileSnapshot, INSTALL_LOCK_TIMEOUT, PersistentPaths, acquire_allowlist_lock,
    acquire_install_lock, read_optional_utf8, remove_optional_file, replace_optional_file,
};
use super::trust::{json_bytes, parse_json_object, trusted_hooks, verify_trust};
use crate::agents::CodingAgent;
use crate::bootstrap::DEFAULT_BIND;
use crate::error::CliError;
use crate::filesystem::atomic_write;
#[cfg(test)]
use crate::installation::generation::GENERATION_FILE_NAME;
use crate::installation::generation::{
    GENERATION_FILE_ENV, GENERATION_TOKEN_ENV, GenerationRetirement, InstallGeneration,
};

/// Hermes host configuration is user-owned even when Relay itself uses project configuration.
/// Project-specific Relay behavior remains available through transparent `nemo-relay run`.
pub(crate) fn user_config_path(default_home: &Path) -> PathBuf {
    user_config_path_with_override(default_home, env::var_os("HERMES_HOME"))
}

pub(crate) fn install_persistent(config: &Path, relay: &Path) -> Result<Vec<PathBuf>, CliError> {
    let relay = relay.canonicalize().unwrap_or_else(|_| relay.to_path_buf());
    let relay = crate::agents::portable_executable_path(relay);
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
    let plugin_config = crate::configuration::user_plugin_runtime_config()?;
    let environment = env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .collect::<Vec<_>>();
    let mut retirement = retire_generation_before_gateway_stop(&paths)?;
    let result = install_persistent_with_generation(
        paths,
        &relay,
        &environment,
        plugin_config.as_ref(),
        retirement.as_ref(),
        SystemTime::now(),
        atomic_write,
    );
    finish_generation_mutation(result, retirement.as_mut(), "install")
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
    let mut retirement = retire_generation_before_gateway_stop(&paths)?;
    let result = uninstall_persistent_with(paths, atomic_write);
    finish_generation_mutation(result, retirement.as_mut(), "uninstall")
}

fn retire_generation_before_gateway_stop(
    paths: &PersistentPaths,
) -> Result<Option<GenerationRetirement>, CliError> {
    let mut retirement =
        GenerationRetirement::acquire(&paths.generation).map_err(CliError::Install)?;
    if let Some(retirement) = retirement.as_mut() {
        retirement
            .invalidate_for_replacement()
            .map_err(CliError::Install)?;
    }
    if let Err(error) = crate::agents::stop_plugin_gateway() {
        if let Some(retirement) = retirement.as_mut()
            && let Err(restore_error) = retirement.restore_after_rollback()
        {
            return Err(CliError::Install(format!(
                "{error}; additionally failed to restore the Hermes MCP generation: {restore_error}"
            )));
        }
        return Err(CliError::Install(error));
    }
    Ok(retirement)
}

fn finish_generation_mutation<T>(
    result: Result<T, CliError>,
    retirement: Option<&mut GenerationRetirement>,
    operation: &str,
) -> Result<T, CliError> {
    match result {
        Ok(value) => {
            if let Some(retirement) = retirement {
                retirement.commit_replacement();
            }
            Ok(value)
        }
        Err(error) => {
            let Some(retirement) = retirement else {
                return Err(error);
            };
            match retirement.restore_after_rollback() {
                Ok(()) => Err(error),
                Err(restore_error) => Err(CliError::Install(format!(
                    "{error}; additionally failed to restore the Hermes MCP generation after {operation}: {restore_error}"
                ))),
            }
        }
    }
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
        if allowlist_has_owned_command(&allowlist, None) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn config_has_managed_state(config: &Value) -> bool {
    owned_command_from_config(config, None).is_some()
}

fn allowlist_has_owned_command(allowlist: &Value, command: Option<&str>) -> bool {
    allowlist
        .get("approvals")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("command").and_then(Value::as_str))
        .any(|candidate| {
            command == Some(candidate)
                || (command.is_none() && is_persistent_relay_hook_command(candidate))
        })
}

fn is_persistent_relay_hook_command(command: &str) -> bool {
    #[cfg(any(windows, test))]
    if let Some(arguments) = crate::hooks::decode_windows_hook_command(command) {
        return matches!(
            arguments.as_slice(),
            [
                _,
                hook_forward,
                agent,
                gateway_flag,
                gateway_url,
                generation_file_flag,
                _,
                generation_token_flag,
                generation_token,
            ] if hook_forward == "hook-forward"
                && agent == "hermes"
                && gateway_flag == "--gateway-url"
                && gateway_url == crate::bootstrap::DEFAULT_URL
                && generation_file_flag == "--generation-file"
                && generation_token_flag == "--generation-token"
                && !generation_token.is_empty()
        );
    }
    command.contains("hook-forward")
        && command.contains("hermes")
        && command.contains("--gateway-url")
        && command.contains(crate::bootstrap::DEFAULT_URL)
        && command.contains("--generation-file")
        && command.contains("--generation-token")
}

fn owned_command_from_config(config: &Value, generation: Option<&Path>) -> Option<String> {
    let relay = config
        .pointer(&format!("/mcp_servers/{MCP_SERVER_NAME}/command"))
        .and_then(Value::as_str)
        .map(PathBuf::from)?;
    owned_install_command(config, &relay, generation)
        .ok()
        .flatten()
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
    let generation = InstallGeneration::capture(paths.generation.clone())?;
    let command = persistent_hook_command(&relay, &paths.generation, generation.token())?;
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
    let configured_token = mcp_env
        .get(GENERATION_TOKEN_ENV)
        .and_then(Value::as_str)
        .ok_or_else(|| "Hermes Relay MCP expected generation identity is missing".to_string())?;
    if configured_token != generation.token() {
        return Err("Hermes Relay MCP expected generation identity is stale".into());
    }

    let plugin_config =
        crate::configuration::user_plugin_runtime_config().map_err(|e| e.to_string())?;
    let environment = env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .collect::<Vec<_>>();
    let environment = forwarded_environment_names(&environment, plugin_config.as_ref());
    let expected = expected_mcp_server(&relay, &paths.generation, generation.token(), &environment);
    let expected_env = expected
        .get("env")
        .and_then(Value::as_object)
        .expect("expected MCP environment is an object");
    let missing = environment
        .into_iter()
        .filter(|name| mcp_env.get(name) != expected_env.get(name))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Hermes Relay MCP is missing environment names {}; run `nemo-relay install hermes --force`",
            missing.join(", ")
        ));
    }
    Ok(format!(
        "MCP lifecycle and {} hooks trusted at {}",
        CodingAgent::Hermes.hook_events().len(),
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
        .get("mcp_servers")
        .and_then(|servers| servers.get(MCP_SERVER_NAME))
        .ok_or_else(|| format!("Hermes MCP server `{MCP_SERVER_NAME}` is missing"))?;
    let relay = PathBuf::from(
        server
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| "Hermes Relay MCP command is missing".to_string())?,
    );
    if owned_install_command(config, &relay, None)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        return Err(format!(
            "Hermes MCP server `{MCP_SERVER_NAME}` is not a managed Relay MCP client"
        ));
    }
    Ok(relay)
}

#[cfg(test)]
fn install_persistent_with<W>(
    paths: PersistentPaths,
    relay: &Path,
    environment: &[String],
    plugin_config: Option<&Value>,
    now: SystemTime,
    write: W,
) -> Result<Vec<PathBuf>, CliError>
where
    W: FnMut(&Path, &[u8]) -> Result<(), String>,
{
    install_persistent_with_generation(paths, relay, environment, plugin_config, None, now, write)
}

fn install_persistent_with_generation<W>(
    paths: PersistentPaths,
    relay: &Path,
    environment: &[String],
    plugin_config: Option<&Value>,
    generation_transaction: Option<&GenerationRetirement>,
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
    let previous_command = match existing_config.as_deref() {
        Some(raw) => {
            let root = parse_yaml_object(Some(raw), "Hermes config")?;
            owned_install_command(&root, relay, Some(&paths.generation))?
        }
        None => None,
    };
    let environment = forwarded_environment_names(environment, plugin_config);
    let token = uuid::Uuid::now_v7().to_string();
    let command =
        persistent_hook_command(relay, &paths.generation, &token).map_err(CliError::Install)?;
    let config = persistent_config(
        existing_config.as_deref(),
        relay,
        &command,
        &paths.generation,
        &token,
        &environment,
    )?;
    let allowlist = trusted_hooks(
        existing_allowlist.as_deref(),
        previous_command.as_deref(),
        &command,
        relay,
        now,
    )?;
    let config = yaml_bytes(&config)?;
    let allowlist = json_bytes(&allowlist)?;
    let generation = format!("{token}\n").into_bytes();

    let result = (|| {
        // Trust is published before config so Hermes never observes a configured hook without
        // its exact approval. The config write is the transaction's commit point.
        write(&paths.generation, &generation)?;
        write(&paths.allowlist, &allowlist)?;
        write(&paths.config, &config)?;
        verify_install(
            &paths,
            relay,
            &command,
            &environment,
            &token,
            generation_transaction,
        )
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
            let owned = owned_command_from_config(&root, Some(&paths.generation));
            strip_owned_hooks(&mut root, owned.as_deref())?;
            remove_owned_mcp(&mut root, owned.is_some())?;
            if root.as_object().is_some_and(Map::is_empty) {
                Ok(None)
            } else {
                yaml_bytes(&root).map(Some)
            }
        })
        .transpose()?
        .flatten();
    let owned = read_optional_utf8(&paths.config)?
        .and_then(|raw| parse_yaml_object(Some(&raw), "Hermes config").ok())
        .and_then(|root| owned_command_from_config(&root, Some(&paths.generation)));
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
                        .is_none_or(|command| Some(command) != owned.as_deref())
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
        verify_uninstall(&paths, owned.as_deref())
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
    generation_transaction: Option<&GenerationRetirement>,
) -> Result<(), String> {
    let raw = fs::read_to_string(&paths.config)
        .map_err(|error| format!("failed to verify {}: {error}", paths.config.display()))?;
    let config = parse_yaml_object(Some(&raw), "Hermes config").map_err(|e| e.to_string())?;
    let expected = expected_mcp_server(relay, &paths.generation, token, environment);
    if config.pointer("/mcp_servers/nemo-relay") != Some(&expected) {
        return Err("Hermes MCP server did not persist exactly".into());
    }
    verify_hook_definitions(&config, command)?;
    verify_trust(&paths.allowlist, command)?;

    let actual_token = match generation_transaction {
        Some(transaction) => transaction.active_visible_token()?,
        None => InstallGeneration::capture(paths.generation.clone())?
            .token()
            .to_owned(),
    };
    if actual_token != token {
        return Err("Hermes MCP generation did not persist exactly".into());
    }
    Ok(())
}

fn verify_hook_definitions(config: &Value, command: &str) -> Result<(), String> {
    for event in CodingAgent::Hermes.hook_events() {
        let groups = config
            .pointer(&format!("/hooks/{event}"))
            .and_then(Value::as_array)
            .ok_or_else(|| format!("Hermes hook {event} is missing"))?;
        let matching = groups
            .iter()
            .filter(|group| group.get("command").and_then(Value::as_str) == Some(command))
            .count();
        if matching != 1 {
            return Err(format!(
                "Hermes hook {event} expected exactly one trusted Relay handler"
            ));
        }
    }
    for (event, groups) in config
        .get("hooks")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(Map::iter)
    {
        let groups = groups
            .as_array()
            .ok_or_else(|| format!("Hermes {event} hooks must be an array"))?;
        if !CodingAgent::Hermes.hook_events().contains(&event.as_str())
            && groups
                .iter()
                .any(|group| group.get("command").and_then(Value::as_str) == Some(command))
        {
            return Err("Hermes config contains an unexpected Relay hook handler".into());
        }
    }
    Ok(())
}

fn verify_uninstall(paths: &PersistentPaths, owned_command: Option<&str>) -> Result<(), String> {
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
        if allowlist_has_owned_command(&allowlist, owned_command) {
            return Err("managed Hermes Relay trust approval still exists".into());
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/hermes_tests.rs"]
mod tests;
