// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure Hermes YAML generation, migration, and ownership recognition.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::error::CliError;
use crate::install_generation::GENERATION_FILE_ENV;
use crate::installer::{hermes_hooks, merge_hooks};
use crate::sidecar::DEFAULT_BIND;

pub(super) const MCP_SERVER_NAME: &str = "nemo-relay";
const ALWAYS_FORWARDED_CREDENTIALS: &[&str] = &["ANTHROPIC_API_KEY", "OPENAI_API_KEY"];

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
pub(crate) fn transparent_config(existing: &str, relay: &Path) -> Result<String, CliError> {
    let mut root = parse_yaml_object(Some(existing), "Hermes config")?;
    strip_managed_hooks(&mut root)?;
    remove_managed_mcp(&mut root)?;
    let root = merge_hooks(root, hermes_hooks(&persistent_hook_command(relay)))?;
    serde_yaml::to_string(&root).map_err(|error| CliError::Install(error.to_string()))
}

pub(crate) fn persistent_hook_command(relay: &Path) -> String {
    persistent_hook_command_for_platform(relay, cfg!(windows))
}

pub(super) fn persistent_hook_command_for_platform(relay: &Path, windows: bool) -> String {
    format!(
        "{} plugin-shim hook hermes",
        crate::plugin_shim::shell_quote_for_platform(relay, windows)
    )
}

pub(super) fn persistent_config(
    existing: Option<&str>,
    relay: &Path,
    command: &str,
    generation: &Path,
    environment: &[String],
) -> Result<Value, CliError> {
    let mut root = parse_yaml_object(existing, "Hermes config")?;
    strip_managed_hooks(&mut root)?;
    root = merge_hooks(root, hermes_hooks(command))?;
    let servers = object_field_mut(&mut root, "mcp_servers", "mcp_servers")?;
    servers.insert(
        MCP_SERVER_NAME.into(),
        expected_mcp_server(relay, generation, environment),
    );
    Ok(root)
}

pub(super) fn expected_mcp_server(
    relay: &Path,
    generation: &Path,
    environment: &[String],
) -> Value {
    let mut forwarded = Map::from_iter([
        ("NEMO_RELAY_GATEWAY_BIND".into(), json!(DEFAULT_BIND)),
        (
            GENERATION_FILE_ENV.into(),
            json!(generation.display().to_string()),
        ),
    ]);
    for name in environment {
        forwarded.insert(name.clone(), json!(format!("${{{name}}}")));
    }
    json!({
        "command": relay.display().to_string(),
        "args": ["mcp", "--agent", "hermes"],
        "env": forwarded,
    })
}

pub(super) fn forwarded_environment_names(
    environment: &[String],
    plugin_config: Option<&Value>,
) -> Vec<String> {
    let present = environment.iter().cloned().collect::<BTreeSet<_>>();
    let referenced = crate::mcp_environment::config_referenced_names(plugin_config)
        .into_iter()
        .collect::<BTreeSet<_>>();
    crate::mcp_environment::forwarded_names(environment.iter().cloned(), plugin_config)
        .into_iter()
        .filter(|name| {
            present.contains(name)
                || referenced.contains(name)
                || ALWAYS_FORWARDED_CREDENTIALS.contains(&name.as_str())
        })
        .collect()
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
    server
        .get("command")
        .and_then(Value::as_str)
        .is_some_and(is_relay_executable)
        && server.get("args") == Some(&json!(["mcp", "--agent", "hermes"]))
}

pub(crate) fn is_managed_hook_command(command: &str) -> bool {
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
        fs::metadata(path)
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
