// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde_json::Value;

use crate::agents::CodingAgent;
use crate::configuration::{RELAY_PLUGIN_ID, RELAY_SOURCE_PLUGIN_ID};
use crate::error::CliError;
use crate::hooks::{generated_hooks, transparent_hook_forward_command};
use crate::process::{PreparedAgentLaunch, insert_after_host};

pub(crate) fn prepare(launch: &mut PreparedAgentLaunch, gateway_url: &str) -> Result<(), CliError> {
    let has_openai_key = std::env::var("OPENAI_API_KEY")
        .ok()
        .is_some_and(|value| !value.is_empty());
    let has_codex_auth = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| PathBuf::from(home).join(".codex/auth.json").exists())
        .unwrap_or(false);
    if !has_openai_key && !has_codex_auth {
        eprintln!(
            "warning: No OpenAI credentials found. Either export OPENAI_API_KEY \
             (e.g. `export OPENAI_API_KEY=sk-...`), log in to codex (`codex --login`), \
             or pass `--openai-base-url` to an upstream that needs no key."
        );
    }
    let hook_command = transparent_hook_forward_command(
        &transparent_hook_executable(),
        CodingAgent::Codex,
        gateway_url,
    )
    .map_err(CliError::Launch)?;
    let hook_groups = generated_hooks(CodingAgent::Codex, &hook_command);
    let mut args = vec![
        "--config".to_string(),
        "features.hooks=true".to_string(),
        "--config".to_string(),
        "features.multi_agent_v2.enabled=false".to_string(),
        "--config".to_string(),
        "model_provider=\"nemo-relay-openai\"".to_string(),
        "--config".to_string(),
        gateway_provider_config(gateway_url),
    ];
    for (event, groups) in hook_groups["hooks"].as_object().into_iter().flatten() {
        args.push("--config".to_string());
        args.push(format!("hooks.{event}={}", hook_groups_toml(groups)));
    }
    args.push("--config".to_string());
    args.push(session_hook_state_override(&hook_groups)?);
    insert_after_host(&mut launch.argv, launch.host_index, args);
    Ok(())
}

pub(crate) fn session_hook_state_override(generated: &Value) -> Result<String, CliError> {
    let events = generated
        .get("hooks")
        .and_then(Value::as_object)
        .ok_or_else(|| CliError::Launch("generated Codex hooks were malformed".into()))?;
    let mut states = Vec::new();
    for (event, groups) in events {
        let groups = groups.as_array().ok_or_else(|| {
            CliError::Launch(format!(
                "generated Codex {event} hook groups were malformed"
            ))
        })?;
        let event_key = hook_event_key(event);
        for (group_index, group) in groups.iter().enumerate() {
            let group = group.as_object().ok_or_else(|| {
                CliError::Launch(format!("generated Codex {event} hook group was malformed"))
            })?;
            let handlers = group
                .get("hooks")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    CliError::Launch(format!(
                        "generated Codex {event} hook handlers were malformed"
                    ))
                })?;
            for (handler_index, handler) in handlers.iter().enumerate() {
                let hash = command_hook_hash(&event_key, group, handler)?;
                let key = format!(
                    "/<session-flags>/config.toml:{event_key}:{group_index}:{handler_index}"
                );
                states.push(format!(
                    "{}={{trusted_hash={},enabled=true}}",
                    toml_string(&key),
                    toml_string(&hash)
                ));
                for plugin_id in [RELAY_PLUGIN_ID, RELAY_SOURCE_PLUGIN_ID] {
                    let key = format!(
                        "{plugin_id}:hooks/hooks.json:{event_key}:{group_index}:{handler_index}"
                    );
                    states.push(format!("{}={{enabled=false}}", toml_string(&key)));
                }
            }
        }
    }
    Ok(format!("hooks.state={{{}}}", states.join(",")))
}

fn hook_event_key(event: &str) -> String {
    let mut normalized = String::with_capacity(event.len() + 2);
    for (index, character) in event.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index > 0 {
                normalized.push('_');
            }
            normalized.push(character.to_ascii_lowercase());
        } else {
            normalized.push(character);
        }
    }
    normalized
}

pub(crate) fn command_hook_hash(
    event_key: &str,
    group: &serde_json::Map<String, Value>,
    handler: &Value,
) -> Result<String, CliError> {
    use sha2::{Digest, Sha256};

    let handler = handler.as_object().ok_or_else(|| {
        CliError::Launch(format!(
            "generated Codex {event_key} command hook was malformed"
        ))
    })?;
    if handler.get("type").and_then(Value::as_str) != Some("command") {
        return Err(CliError::Launch(format!(
            "generated Codex {event_key} hook was not a command"
        )));
    }
    let command = handler
        .get(if cfg!(windows) {
            "commandWindows"
        } else {
            "command"
        })
        .or_else(|| handler.get("command"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            CliError::Launch(format!(
                "generated Codex {event_key} hook command was missing"
            ))
        })?;
    let timeout = handler
        .get("timeout")
        .and_then(Value::as_u64)
        .unwrap_or(600)
        .max(1);
    let mut normalized_handler = serde_json::Map::new();
    normalized_handler.insert("type".into(), Value::String("command".into()));
    normalized_handler.insert("command".into(), Value::String(command.into()));
    normalized_handler.insert("timeout".into(), Value::Number(timeout.into()));
    normalized_handler.insert("async".into(), Value::Bool(false));
    if let Some(status) = handler.get("statusMessage").and_then(Value::as_str) {
        normalized_handler.insert("statusMessage".into(), Value::String(status.into()));
    }
    let mut identity = serde_json::Map::new();
    identity.insert("event_name".into(), Value::String(event_key.into()));
    if let Some(matcher) = group.get("matcher").and_then(Value::as_str) {
        identity.insert("matcher".into(), Value::String(matcher.into()));
    }
    identity.insert(
        "hooks".into(),
        Value::Array(vec![Value::Object(normalized_handler)]),
    );
    let bytes = serde_json::to_vec(&canonical_json(Value::Object(identity)))
        .map_err(|error| CliError::Launch(format!("failed to hash Codex hook: {error}")))?;
    let digest = Sha256::digest(bytes);
    Ok(format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn canonical_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, canonical_json(value)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_json).collect()),
        other => other,
    }
}

fn gateway_provider_config(gateway_url: &str) -> String {
    format!(
        "model_providers.nemo-relay-openai={{name=\"NeMo Relay OpenAI\",base_url={},wire_api=\"responses\",requires_openai_auth=true,supports_websockets=false}}",
        toml_string(gateway_url)
    )
}

pub(crate) fn hook_groups_toml(value: &Value) -> String {
    let mut groups = Vec::new();
    for group in value.as_array().into_iter().flatten() {
        let matcher = group
            .get("matcher")
            .and_then(Value::as_str)
            .map(|matcher| format!("matcher={},", toml_string(matcher)))
            .unwrap_or_default();
        let command = group["hooks"][0]["command"].as_str().unwrap_or_default();
        groups.push(format!(
            "{{{matcher}hooks=[{{type=\"command\",command={},timeout=30}}]}}",
            toml_string(command)
        ));
    }
    format!("[{}]", groups.join(","))
}

pub(crate) fn toml_string(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn transparent_hook_executable() -> PathBuf {
    std::env::current_exe()
        .map(|path| path.canonicalize().unwrap_or(path))
        .map(crate::agents::portable_executable_path)
        .unwrap_or_else(|_| PathBuf::from("nemo-relay"))
}
