// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure Hermes YAML generation, migration, and ownership recognition.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::error::CliError;
use crate::hooks::{generated_hooks, merge_hooks};

pub(super) use crate::mcp::SERVER_NAME as MCP_SERVER_NAME;

pub(super) fn user_config_path_with_override(
    default_home: &Path,
    hermes_home: Option<std::ffi::OsString>,
) -> PathBuf {
    hermes_home
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| default_home.join(".hermes"))
        .join("config.yaml")
}

/// Rewrites the Relay-owned portion of a Hermes config for a transparent run. The fixed MCP
/// client is removed because the wrapper already owns a dynamic gateway.
pub(crate) fn transparent_config(
    existing: &str,
    relay: &Path,
    gateway_url: &str,
) -> Result<String, CliError> {
    let mut root = parse_yaml_object(Some(existing), "Hermes config")?;
    let owned = owned_install_command(&root, relay, None)?;
    strip_owned_hooks(&mut root, owned.as_deref())?;
    remove_owned_mcp(&mut root, owned.is_some())?;
    let command = crate::hooks::transparent_hook_forward_command(
        relay,
        crate::configuration::CodingAgent::Hermes,
        gateway_url,
    )
    .map_err(CliError::Install)?;
    let root = merge_hooks(
        root,
        generated_hooks(crate::configuration::CodingAgent::Hermes, &command),
    )?;
    serde_yaml::to_string(&root).map_err(|error| CliError::Install(error.to_string()))
}

pub(crate) fn persistent_hook_command(
    relay: &Path,
    generation: &Path,
    generation_token: &str,
) -> Result<String, String> {
    crate::hooks::persistent_hook_forward_command(
        relay,
        crate::configuration::CodingAgent::Hermes,
        generation,
        generation_token,
    )
}

#[cfg(test)]
pub(super) fn persistent_hook_command_for_platform(
    relay: &Path,
    generation: &Path,
    generation_token: &str,
    windows: bool,
) -> String {
    crate::hooks::persistent_hook_forward_command_for_platform(
        relay,
        crate::configuration::CodingAgent::Hermes,
        generation,
        generation_token,
        windows,
    )
}

pub(super) fn persistent_config(
    existing: Option<&str>,
    relay: &Path,
    command: &str,
    generation: &Path,
    generation_token: &str,
    environment: &[String],
) -> Result<Value, CliError> {
    let mut root = parse_yaml_object(existing, "Hermes config")?;
    let owned = owned_install_command(&root, relay, Some(generation))?;
    if root
        .pointer(&format!("/mcp_servers/{MCP_SERVER_NAME}"))
        .is_some()
        && owned.is_none()
    {
        return Err(CliError::Install(format!(
            "Hermes MCP server `{MCP_SERVER_NAME}` already exists and is not managed by Relay; rename or remove it before installing the Relay integration"
        )));
    }
    strip_owned_hooks(&mut root, owned.as_deref())?;
    root = merge_hooks(
        root,
        generated_hooks(crate::configuration::CodingAgent::Hermes, command),
    )?;
    let servers = object_field_mut(&mut root, "mcp_servers", "mcp_servers")?;
    servers.insert(
        MCP_SERVER_NAME.into(),
        expected_mcp_server(relay, generation, generation_token, environment),
    );
    Ok(root)
}

pub(super) fn expected_mcp_server(
    relay: &Path,
    generation: &Path,
    generation_token: &str,
    environment: &[String],
) -> Value {
    let mut server = crate::mcp::persistent_server(relay, generation, generation_token);
    let forwarded = server
        .get_mut("env")
        .and_then(Value::as_object_mut)
        .expect("persistent MCP server environment is an object");
    for name in environment {
        forwarded.insert(name.clone(), json!(format!("${{{name}}}")));
    }
    server
}

pub(super) fn forwarded_environment_names(
    environment: &[String],
    plugin_config: Option<&Value>,
) -> Vec<String> {
    crate::mcp_environment::forwarded_names(environment.iter().cloned(), plugin_config)
}

pub(super) fn strip_owned_hooks(
    root: &mut Value,
    owned_command: Option<&str>,
) -> Result<(), CliError> {
    let Some(hooks) = root.get_mut("hooks") else {
        return Ok(());
    };
    let remove_hooks = {
        let hooks = hooks
            .as_object_mut()
            .ok_or_else(|| CliError::Install("Hermes hooks must be an object".into()))?;
        let mut empty = Vec::new();
        for (event, groups) in hooks.iter_mut() {
            let groups = groups.as_array_mut().ok_or_else(|| {
                CliError::Install(format!("Hermes {event} hooks must be an array"))
            })?;
            groups.retain(|group| {
                group
                    .get("command")
                    .and_then(Value::as_str)
                    .is_none_or(|command| Some(command) != owned_command)
            });
            if groups.is_empty() {
                empty.push(event.clone());
            }
        }
        for event in empty {
            hooks.remove(&event);
        }
        hooks.is_empty()
    };
    if remove_hooks {
        root.as_object_mut()
            .expect("Hermes config root checked as object")
            .remove("hooks");
    }
    Ok(())
}

pub(super) fn remove_owned_mcp(root: &mut Value, owned: bool) -> Result<(), CliError> {
    let Some(servers) = root.get_mut("mcp_servers") else {
        return Ok(());
    };
    let servers = servers
        .as_object_mut()
        .ok_or_else(|| CliError::Install("Hermes mcp_servers must be an object".into()))?;
    if owned {
        servers.remove(MCP_SERVER_NAME);
    }
    if servers.is_empty() {
        root.as_object_mut()
            .expect("Hermes config root checked as object")
            .remove("mcp_servers");
    }
    Ok(())
}

pub(super) fn owned_install_command(
    root: &Value,
    relay: &Path,
    expected_generation: Option<&Path>,
) -> Result<Option<String>, CliError> {
    let Some(server) = root.pointer(&format!("/mcp_servers/{MCP_SERVER_NAME}")) else {
        return Ok(None);
    };
    if server.get("command") != Some(&json!(relay)) {
        return Ok(None);
    }
    let env = server.get("env").and_then(Value::as_object);
    if server.get("args") == Some(&json!(["mcp"]))
        && env.and_then(|env| env.get("NEMO_RELAY_GATEWAY_BIND"))
            == Some(&json!(crate::sidecar::DEFAULT_BIND))
    {
        let generation = env
            .and_then(|env| env.get(crate::installation::generation::GENERATION_FILE_ENV))
            .and_then(Value::as_str);
        let token = env
            .and_then(|env| env.get(crate::installation::generation::GENERATION_TOKEN_ENV))
            .and_then(Value::as_str);
        if let (Some(generation), Some(token)) = (generation, token)
            && !token.is_empty()
            && expected_generation.is_none_or(|expected| Path::new(generation) == expected)
        {
            let command = persistent_hook_command(relay, Path::new(generation), token)
                .map_err(CliError::Install)?;
            return Ok(has_complete_hook_set(root, &command).then_some(command));
        }
    }
    legacy_owned_command(root, relay)
}

fn has_complete_hook_set(root: &Value, command: &str) -> bool {
    crate::configuration::CodingAgent::Hermes
        .hook_events()
        .iter()
        .all(|event| {
            root.pointer(&format!("/hooks/{event}"))
                .and_then(Value::as_array)
                .is_some_and(|groups| {
                    groups
                        .iter()
                        .filter(|entry| {
                            entry.get("command").and_then(Value::as_str) == Some(command)
                        })
                        .count()
                        == 1
                })
        })
}

fn legacy_owned_command(root: &Value, relay: &Path) -> Result<Option<String>, CliError> {
    let server = &root["mcp_servers"][MCP_SERVER_NAME];
    if server.get("args") != Some(&json!(["mcp", "--agent", "hermes"])) {
        return Ok(None);
    }
    let Some(hooks) = root.get("hooks").and_then(Value::as_object) else {
        return Ok(None);
    };
    let mut common = None;
    for event in crate::configuration::CodingAgent::Hermes.hook_events() {
        let commands = hooks
            .get(*event)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("command").and_then(Value::as_str))
            .filter(|command| legacy_command_uses_relay(command, relay))
            .collect::<Vec<_>>();
        if commands.len() != 1 || common.is_some_and(|value| value != commands[0]) {
            return Ok(None);
        }
        common = Some(commands[0]);
    }
    Ok(common.map(str::to_owned))
}

fn legacy_command_uses_relay(command: &str, relay: &Path) -> bool {
    let relay = relay.to_string_lossy();
    let quoted = crate::agents::host::shell_quote_arg_for_platform(&relay, cfg!(windows));
    [relay.as_ref(), quoted.as_str()].into_iter().any(|prefix| {
        command.strip_prefix(prefix).is_some_and(|arguments| {
            [" hook-forward hermes", " plugin-shim hook hermes"]
                .iter()
                .any(|marker| arguments.starts_with(marker))
        })
    })
}

pub(super) fn relay_is_executable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

pub(super) fn parse_yaml_object(raw: Option<&str>, description: &str) -> Result<Value, CliError> {
    let value = match raw.filter(|raw| !raw.trim().is_empty()) {
        Some(raw) => serde_yaml::from_str(raw)
            .map_err(|error| CliError::Install(format!("invalid {description}: {error}")))?,
        None => json!({}),
    };
    if value.is_object() {
        Ok(value)
    } else {
        Err(CliError::Install(format!(
            "{description} must contain an object"
        )))
    }
}

pub(super) fn yaml_bytes(value: &Value) -> Result<Vec<u8>, CliError> {
    serde_yaml::to_string(value)
        .map(String::into_bytes)
        .map_err(|error| CliError::Install(error.to_string()))
}

fn object_field_mut<'a>(
    root: &'a mut Value,
    field: &str,
    description: &str,
) -> Result<&'a mut Map<String, Value>, CliError> {
    root.as_object_mut()
        .expect("config root checked as object")
        .entry(field)
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| CliError::Install(format!("Hermes {description} must be an object")))
}
