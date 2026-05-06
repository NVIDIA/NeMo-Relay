// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use toml_edit::{DocumentMut, table, value};

use crate::config::{
    CodingAgent, GatewayMode, HookForwardCommand, InstallCommand, InstallScope, InstallTarget,
};
use crate::error::SidecarError;

const HOOK_EVENTS: &[&str] = &[
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "PreCompact",
    "SessionEnd",
];

const CURSOR_HOOK_EVENTS: &[&str] = &[
    "sessionStart",
    "beforeSubmitPrompt",
    "preToolUse",
    "beforeShellExecution",
    "beforeMCPExecution",
    "postToolUse",
    "afterShellExecution",
    "afterMCPExecution",
    "subagentStart",
    "subagentStop",
    "afterAgentResponse",
    "preCompact",
    "stop",
    "sessionEnd",
];

#[derive(Debug, Clone)]
struct PlannedFile {
    path: PathBuf,
    contents: String,
}

pub(crate) fn install(command: InstallCommand) -> Result<(), SidecarError> {
    validate_optional_json("session metadata", command.session_metadata.as_deref())?;
    validate_optional_json("plugin config", command.plugin_config.as_deref())?;
    let files = planned_files(&command)?;
    if command.print {
        for file in &files {
            println!("--- {}", file.path.display());
            print!("{}", file.contents);
            if !file.contents.ends_with('\n') {
                println!();
            }
        }
    }
    if command.dry_run {
        println!(
            "Dry run: would install {} integration for {:?} {:?}.",
            command.agent.as_arg(),
            command.scope,
            command.target
        );
        return Ok(());
    }
    for file in &files {
        write_planned_file(file)?;
        println!("Installed {}", file.path.display());
    }
    print_target_note(command.agent, command.target);
    Ok(())
}

pub(crate) async fn hook_forward(command: HookForwardCommand) -> Result<(), SidecarError> {
    validate_optional_json("session metadata", command.session_metadata.as_deref())?;
    validate_optional_json("plugin config", command.plugin_config.as_deref())?;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    if input.trim().is_empty() {
        input = "{}".to_string();
    }

    let Some(sidecar_url) = command
        .sidecar_url
        .clone()
        .or_else(|| std::env::var("NEMO_FLOW_SIDECAR_URL").ok())
    else {
        eprintln!(
            "nemo-flow-sidecar hook forward failed: missing sidecar URL; pass --sidecar-url or set NEMO_FLOW_SIDECAR_URL"
        );
        if command.fail_closed {
            return Err(SidecarError::Install(
                "missing sidecar URL; pass --sidecar-url or set NEMO_FLOW_SIDECAR_URL".into(),
            ));
        }
        return Ok(());
    };
    let url = format!(
        "{}{}",
        sidecar_url.trim_end_matches('/'),
        command.agent.hook_path()
    );
    let response = reqwest::Client::new()
        .post(url)
        .headers(sidecar_headers(
            command.atif_dir.as_deref(),
            command.openinference_endpoint.as_deref(),
            command.profile.as_deref(),
            command.session_metadata.as_deref(),
            command.plugin_config.as_deref(),
            command.gateway_mode,
        )?)
        .header(CONTENT_TYPE, "application/json")
        .body(input)
        .send()
        .await;

    match response {
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                eprintln!("nemo-flow-sidecar hook forward failed with HTTP {status}");
                if command.fail_closed {
                    return Err(SidecarError::Install(format!(
                        "hook forward failed with HTTP {status}"
                    )));
                }
                return Ok(());
            }
            if !body.is_empty() {
                println!("{body}");
            }
            Ok(())
        }
        Err(error) => {
            eprintln!("nemo-flow-sidecar hook forward failed: {error}");
            if command.fail_closed {
                Err(SidecarError::Upstream(error))
            } else {
                Ok(())
            }
        }
    }
}

fn planned_files(command: &InstallCommand) -> Result<Vec<PlannedFile>, SidecarError> {
    let base = install_base(command)?;
    match command.agent {
        CodingAgent::ClaudeCode => {
            let path = base.join(".claude/settings.json");
            let existing = read_json_file(&path)?;
            let contents = serde_json::to_string_pretty(&merge_hooks(
                existing,
                claude_hooks(&hook_command(command, CodingAgent::ClaudeCode)),
            )?)
            .map_err(|error| SidecarError::Install(error.to_string()))?;
            Ok(vec![PlannedFile { path, contents }])
        }
        CodingAgent::Codex => {
            let config_path = base.join(".codex/config.toml");
            let hooks_path = base.join(".codex/hooks.json");
            let config =
                merge_codex_config(&std::fs::read_to_string(&config_path).unwrap_or_default())?;
            let hooks = serde_json::to_string_pretty(&merge_hooks(
                read_json_file(&hooks_path)?,
                codex_hooks(&hook_command(command, CodingAgent::Codex)),
            )?)
            .map_err(|error| SidecarError::Install(error.to_string()))?;
            Ok(vec![
                PlannedFile {
                    path: config_path,
                    contents: config,
                },
                PlannedFile {
                    path: hooks_path,
                    contents: hooks,
                },
            ])
        }
        CodingAgent::Cursor => {
            let path = base.join(".cursor/hooks.json");
            let existing = read_json_file(&path)?;
            let contents = serde_json::to_string_pretty(&merge_hooks(
                existing,
                cursor_hooks(&hook_command(command, CodingAgent::Cursor)),
            )?)
            .map_err(|error| SidecarError::Install(error.to_string()))?;
            Ok(vec![PlannedFile { path, contents }])
        }
    }
}

fn install_base(command: &InstallCommand) -> Result<PathBuf, SidecarError> {
    match command.scope {
        InstallScope::User => command
            .home_dir
            .clone()
            .or_else(home_dir)
            .ok_or_else(|| SidecarError::Install("could not resolve home directory".into())),
        InstallScope::Project => command
            .project_dir
            .clone()
            .map(Ok)
            .unwrap_or_else(std::env::current_dir)
            .map_err(SidecarError::from),
    }
}

fn hook_command(command: &InstallCommand, agent: CodingAgent) -> String {
    let mut args = vec![
        "nemo-flow-sidecar".to_string(),
        "hook-forward".to_string(),
        agent.as_arg().to_string(),
        "--sidecar-url".to_string(),
        command.sidecar_url.clone(),
    ];
    push_optional_path(&mut args, "--atif-dir", command.atif_dir.as_deref());
    push_optional(
        &mut args,
        "--openinference-endpoint",
        command.openinference_endpoint.as_deref(),
    );
    push_optional(&mut args, "--profile", command.profile.as_deref());
    push_optional(
        &mut args,
        "--session-metadata",
        command.session_metadata.as_deref(),
    );
    push_optional(
        &mut args,
        "--plugin-config",
        command.plugin_config.as_deref(),
    );
    push_optional_gateway_mode(&mut args, command.gateway_mode);
    args.into_iter()
        .map(|arg| shell_quote(&arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn push_optional(args: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        args.push(flag.to_string());
        args.push(value.to_string());
    }
}

fn push_optional_path(args: &mut Vec<String>, flag: &str, value: Option<&Path>) {
    if let Some(value) = value {
        args.push(flag.to_string());
        args.push(value.display().to_string());
    }
}

fn push_optional_gateway_mode(args: &mut Vec<String>, gateway_mode: Option<GatewayMode>) {
    if let Some(gateway_mode) = gateway_mode {
        args.push("--gateway-mode".to_string());
        args.push(gateway_mode.as_arg().to_string());
    }
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || "-_./:=,".contains(character))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

pub(crate) fn generated_hooks(agent: CodingAgent, command: &str) -> Value {
    match agent {
        CodingAgent::ClaudeCode => claude_hooks(command),
        CodingAgent::Codex => codex_hooks(command),
        CodingAgent::Cursor => cursor_hooks(command),
    }
}

pub(crate) fn hook_forward_command(agent: CodingAgent) -> String {
    format!("nemo-flow-sidecar hook-forward {}", agent.as_arg())
}

fn claude_hooks(command: &str) -> Value {
    hooks_for_events(HOOK_EVENTS, command, true)
}

fn codex_hooks(command: &str) -> Value {
    hooks_for_events(HOOK_EVENTS, command, true)
}

fn cursor_hooks(command: &str) -> Value {
    hooks_for_events(CURSOR_HOOK_EVENTS, command, true)
}

fn hooks_for_events(events: &[&str], command: &str, matcher_for_tools: bool) -> Value {
    let hooks: serde_json::Map<String, Value> = events
        .iter()
        .map(|event| {
            let mut group = serde_json::Map::new();
            if matcher_for_tools && event_matches_tools(event) {
                group.insert("matcher".into(), json!("*"));
            }
            group.insert(
                "hooks".into(),
                json!([{
                    "type": "command",
                    "command": command,
                    "timeout": 30
                }]),
            );
            (
                (*event).to_string(),
                Value::Array(vec![Value::Object(group)]),
            )
        })
        .collect();
    json!({ "hooks": Value::Object(hooks) })
}

fn event_matches_tools(event: &str) -> bool {
    matches!(
        event,
        "PreToolUse"
            | "PostToolUse"
            | "PostToolUseFailure"
            | "PermissionRequest"
            | "preToolUse"
            | "postToolUse"
            | "beforeShellExecution"
            | "afterShellExecution"
            | "beforeMCPExecution"
            | "afterMCPExecution"
    )
}

pub(crate) fn merge_hooks(existing: Value, generated: Value) -> Result<Value, SidecarError> {
    let mut root = match existing {
        Value::Null => json!({}),
        Value::Object(object) => Value::Object(object),
        _ => {
            return Err(SidecarError::Install(
                "hook config must be a JSON object".into(),
            ));
        }
    };
    let root_object = root.as_object_mut().expect("root checked as object");
    let hooks = root_object
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| SidecarError::Install("hooks must be a JSON object".into()))?;
    let generated_hooks = generated
        .get("hooks")
        .and_then(Value::as_object)
        .ok_or_else(|| SidecarError::Install("generated hooks were malformed".into()))?;
    for (event, groups) in generated_hooks {
        let groups = groups
            .as_array()
            .ok_or_else(|| SidecarError::Install("generated hook groups were malformed".into()))?;
        let event_groups = hooks.entry(event.clone()).or_insert_with(|| json!([]));
        let event_groups = event_groups
            .as_array_mut()
            .ok_or_else(|| SidecarError::Install(format!("{event} hooks must be an array")))?;
        for group in groups {
            if !event_groups.iter().any(|existing| existing == group) {
                event_groups.push(group.clone());
            }
        }
    }
    Ok(root)
}

fn merge_codex_config(existing: &str) -> Result<String, SidecarError> {
    let mut document = if existing.trim().is_empty() {
        DocumentMut::new()
    } else {
        existing
            .parse::<DocumentMut>()
            .map_err(|error| SidecarError::Install(format!("invalid TOML: {error}")))?
    };
    if !document.as_table().contains_key("features") {
        document["features"] = table();
    }
    document["features"]["codex_hooks"] = value(true);
    Ok(document.to_string())
}

pub(crate) fn read_json_file(path: &Path) -> Result<Value, SidecarError> {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).map_err(|error| {
            SidecarError::Install(format!("invalid JSON in {}: {error}", path.display()))
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(error) => Err(SidecarError::Io(error)),
    }
}

fn write_planned_file(file: &PlannedFile) -> Result<(), SidecarError> {
    if let Some(parent) = file.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if file.path.exists() {
        std::fs::copy(&file.path, backup_path(&file.path)?)?;
    }
    std::fs::write(&file.path, &file.contents)?;
    Ok(())
}

fn backup_path(path: &Path) -> Result<PathBuf, SidecarError> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| SidecarError::Install(error.to_string()))?
        .as_secs();
    Ok(path.with_extension(format!(
        "{}.bak.{timestamp}",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("config")
    )))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn validate_optional_json(name: &str, value: Option<&str>) -> Result<(), SidecarError> {
    if let Some(value) = value {
        serde_json::from_str::<Value>(value)
            .map_err(|error| SidecarError::Install(format!("invalid {name}: {error}")))?;
    }
    Ok(())
}

fn sidecar_headers(
    atif_dir: Option<&Path>,
    openinference_endpoint: Option<&str>,
    profile: Option<&str>,
    session_metadata: Option<&str>,
    plugin_config: Option<&str>,
    gateway_mode: Option<GatewayMode>,
) -> Result<HeaderMap, SidecarError> {
    let mut headers = HeaderMap::new();
    insert_header_path(&mut headers, "x-nemo-flow-atif-dir", atif_dir)?;
    insert_header(
        &mut headers,
        "x-nemo-flow-openinference-endpoint",
        openinference_endpoint,
    )?;
    insert_header(&mut headers, "x-nemo-flow-config-profile", profile)?;
    insert_header(
        &mut headers,
        "x-nemo-flow-session-metadata",
        session_metadata,
    )?;
    insert_header(&mut headers, "x-nemo-flow-plugin-config", plugin_config)?;
    insert_header(
        &mut headers,
        "x-nemo-flow-gateway-mode",
        gateway_mode.map(GatewayMode::as_arg),
    )?;
    Ok(headers)
}

fn insert_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value: Option<&str>,
) -> Result<(), SidecarError> {
    if let Some(value) = value {
        headers.insert(
            HeaderName::from_static(name),
            HeaderValue::from_str(value).map_err(|error| {
                SidecarError::Install(format!("invalid header {name}: {error}"))
            })?,
        );
    }
    Ok(())
}

fn insert_header_path(
    headers: &mut HeaderMap,
    name: &'static str,
    value: Option<&Path>,
) -> Result<(), SidecarError> {
    if let Some(value) = value {
        let value = value.to_string_lossy();
        insert_header(headers, name, Some(value.as_ref()))
    } else {
        Ok(())
    }
}

fn print_target_note(agent: CodingAgent, target: InstallTarget) {
    match (agent, target) {
        (CodingAgent::ClaudeCode, InstallTarget::Gui | InstallTarget::Both) => {
            println!(
                "Note: Claude application/web sessions are not configured by Claude Code hooks."
            );
        }
        (CodingAgent::Codex, InstallTarget::Gui | InstallTarget::Both) => {
            println!(
                "Note: Codex GUI local sessions can use local config; cloud tasks need separate gateway support."
            );
        }
        (CodingAgent::Cursor, InstallTarget::Cli | InstallTarget::Both) => {
            println!(
                "Note: run the Cursor CLI smoke test to confirm cursor-agent loads hooks in your version."
            );
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "../tests/coverage/installer_tests.rs"]
mod tests;
