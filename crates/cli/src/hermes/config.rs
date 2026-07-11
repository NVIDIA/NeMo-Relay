// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure Hermes YAML generation, migration, and ownership recognition.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::error::CliError;
use crate::installer::{generated_hooks, merge_hooks};

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
    strip_managed_hooks(&mut root)?;
    remove_managed_mcp(&mut root)?;
    let command = crate::installer::transparent_hook_forward_command(
        relay,
        crate::config::CodingAgent::Hermes,
        gateway_url,
    )
    .map_err(CliError::Install)?;
    let root = merge_hooks(
        root,
        generated_hooks(crate::config::CodingAgent::Hermes, &command),
    )?;
    serde_yaml::to_string(&root).map_err(|error| CliError::Install(error.to_string()))
}

pub(crate) fn persistent_hook_command(
    relay: &Path,
    generation: &Path,
    generation_token: &str,
) -> Result<String, String> {
    crate::installer::persistent_hook_forward_command(
        relay,
        crate::config::CodingAgent::Hermes,
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
    crate::installer::persistent_hook_forward_command_for_platform(
        relay,
        crate::config::CodingAgent::Hermes,
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
    if let Some(server) = root.pointer(&format!("/mcp_servers/{MCP_SERVER_NAME}"))
        && !is_managed_mcp_server(server)
    {
        return Err(CliError::Install(format!(
            "Hermes MCP server `{MCP_SERVER_NAME}` already exists and is not managed by Relay; rename or remove it before installing the Relay integration"
        )));
    }
    strip_managed_hooks(&mut root)?;
    root = merge_hooks(
        root,
        generated_hooks(crate::config::CodingAgent::Hermes, command),
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

pub(super) fn strip_managed_hooks(root: &mut Value) -> Result<(), CliError> {
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
                    .is_none_or(|command| !is_managed_hook_command(command))
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

pub(super) fn remove_managed_mcp(root: &mut Value) -> Result<(), CliError> {
    let Some(servers) = root.get_mut("mcp_servers") else {
        return Ok(());
    };
    let servers = servers
        .as_object_mut()
        .ok_or_else(|| CliError::Install("Hermes mcp_servers must be an object".into()))?;
    if servers
        .get(MCP_SERVER_NAME)
        .is_some_and(is_managed_mcp_server)
    {
        servers.remove(MCP_SERVER_NAME);
    }
    if servers.is_empty() {
        root.as_object_mut()
            .expect("Hermes config root checked as object")
            .remove("mcp_servers");
    }
    Ok(())
}

pub(super) fn is_managed_mcp_server(server: &Value) -> bool {
    crate::mcp::is_managed_server(server, is_relay_executable)
}

pub(crate) fn is_managed_hook_command(command: &str) -> bool {
    if let Some(arguments) = crate::installer::decode_windows_hook_command(command) {
        if arguments.len() < 3
            || !is_relay_executable(&arguments[0])
            || arguments[1] != "hook-forward"
            || arguments[2] != "hermes"
        {
            return false;
        }
        let options = &arguments[3..];
        return options.is_empty()
            || (options.len() == 3
                && options[0] == "--gateway-url"
                && options[2] == "--transparent-run")
            || (options.len() == 6
                && options[0] == "--gateway-url"
                && options[2] == "--generation-file"
                && options[4] == "--generation-token");
    }
    [" hook-forward hermes", " plugin-shim hook hermes"]
        .into_iter()
        .any(|separator| {
            command
                .trim()
                .rsplit_once(separator)
                .is_some_and(|(executable, arguments)| {
                    is_relay_executable(executable)
                        && (arguments.is_empty()
                            || arguments.starts_with(" --gateway-url ")
                            || arguments.starts_with(" --fail-closed"))
                })
        })
}

fn is_relay_executable(raw: &str) -> bool {
    let mut candidate = raw.trim().to_string();
    if candidate.starts_with('\'') && candidate.ends_with('\'') && candidate.len() >= 2 {
        candidate = candidate[1..candidate.len() - 1].replace("'\\''", "'");
    } else if candidate.starts_with('"') && candidate.ends_with('"') && candidate.len() >= 2 {
        candidate = candidate[1..candidate.len() - 1].to_string();
    }
    candidate = candidate.replace('^', "").replace("%%", "%");
    let normalized = candidate.replace('\\', "/");
    matches!(
        normalized.rsplit('/').next().map(str::to_ascii_lowercase),
        Some(name) if name == "nemo-relay" || name == "nemo-relay.exe"
    )
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
