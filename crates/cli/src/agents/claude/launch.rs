// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::agents::CodingAgent;
use crate::error::CliError;
use crate::hooks::{generated_hooks, transparent_hook_forward_command};
use crate::process::{PreparedAgentLaunch, insert_after_host};

pub(crate) fn prepare(
    launch: &mut PreparedAgentLaunch,
    gateway_url: &str,
    dry_run: bool,
) -> Result<(), CliError> {
    if dry_run {
        insert_after_host(
            &mut launch.argv,
            launch.host_index,
            [
                "--plugin-dir".into(),
                "<temporary-claude-plugin-dir>".into(),
                "--settings".into(),
                "<temporary-claude-settings>".into(),
            ],
        );
        launch
            .env
            .push(("ANTHROPIC_BASE_URL".into(), gateway_url.to_string()));
        launch
            .notes
            .push("would generate a temporary Claude Code plugin directory".into());
        return Ok(());
    }

    let root = temp_dir("nemo-relay-claude-plugin")?;
    std::fs::create_dir_all(root.join(".claude-plugin"))?;
    std::fs::create_dir_all(root.join("hooks"))?;
    std::fs::write(
        root.join(".claude-plugin/plugin.json"),
        serde_json::to_vec_pretty(&json!({
            "name": "nemo-relay-cli",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Temporary NeMo Relay gateway hooks"
        }))
        .map_err(|error| CliError::Launch(error.to_string()))?,
    )?;
    let hook_command = transparent_hook_forward_command(
        &transparent_hook_executable(),
        CodingAgent::ClaudeCode,
        gateway_url,
    )
    .map_err(CliError::Launch)?;
    write_hooks(
        &root.join("hooks/hooks.json"),
        generated_hooks(CodingAgent::ClaudeCode, &hook_command),
    )?;
    let settings_path = root.join("settings.json");
    let settings = settings_overlay(&launch.argv, launch.host_index, gateway_url)?;
    let settings_bytes = serde_json::to_vec_pretty(&settings)
        .map_err(|error| CliError::Launch(error.to_string()))?;
    crate::filesystem::atomic_write_private(&settings_path, &settings_bytes)
        .map_err(CliError::Launch)?;
    insert_after_host(
        &mut launch.argv,
        launch.host_index,
        [
            "--plugin-dir".into(),
            root.display().to_string(),
            "--settings".into(),
            settings_path.display().to_string(),
        ],
    );
    launch
        .env
        .push(("ANTHROPIC_BASE_URL".into(), gateway_url.to_string()));
    launch.temp_dirs.push(root);
    Ok(())
}

pub(crate) fn settings_overlay(
    argv: &[String],
    host_index: usize,
    gateway_url: &str,
) -> Result<Value, CliError> {
    let mut settings = match first_settings(argv, host_index)? {
        Some(source) => read_settings(source)?,
        None => json!({}),
    };
    let object = settings.as_object_mut().ok_or_else(|| {
        CliError::Launch("Claude Code --settings must contain a JSON object".into())
    })?;
    let environment = object.entry("env").or_insert_with(|| json!({}));
    let environment = environment.as_object_mut().ok_or_else(|| {
        CliError::Launch("Claude Code --settings field `env` must be a JSON object".into())
    })?;
    environment.insert(
        "ANTHROPIC_BASE_URL".into(),
        Value::String(gateway_url.into()),
    );
    Ok(settings)
}

fn first_settings(argv: &[String], host_index: usize) -> Result<Option<&str>, CliError> {
    let boundary = argv
        .iter()
        .skip(host_index + 1)
        .position(|argument| argument == "--")
        .map_or(argv.len(), |offset| host_index + 1 + offset);
    let mut index = host_index + 1;
    while index < boundary {
        if argv[index] == "--settings" {
            if index + 1 >= boundary || argv[index + 1].is_empty() {
                return Err(CliError::Launch(
                    "Claude Code --settings is missing its value".into(),
                ));
            }
            return Ok(Some(argv[index + 1].as_str()));
        }
        if let Some(value) = argv[index].strip_prefix("--settings=") {
            if value.is_empty() {
                return Err(CliError::Launch(
                    "Claude Code --settings is missing its value".into(),
                ));
            }
            return Ok(Some(value));
        }
        index += 1;
    }
    Ok(None)
}

fn read_settings(source: &str) -> Result<Value, CliError> {
    let raw = if source.trim_start().starts_with('{') {
        source.to_string()
    } else {
        std::fs::read_to_string(source).map_err(|error| {
            CliError::Launch(format!(
                "failed to read Claude Code settings {}: {error}",
                Path::new(source).display()
            ))
        })?
    };
    serde_json::from_str(&raw).map_err(|error| {
        CliError::Launch(format!(
            "failed to parse Claude Code --settings JSON: {error}"
        ))
    })
}

fn transparent_hook_executable() -> PathBuf {
    std::env::current_exe()
        .map(|path| path.canonicalize().unwrap_or(path))
        .map(crate::agents::portable_executable_path)
        .unwrap_or_else(|_| PathBuf::from("nemo-relay"))
}

pub(crate) fn write_hooks(path: &Path, hooks: Value) -> Result<(), CliError> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&hooks).map_err(|error| CliError::Launch(error.to_string()))?,
    )?;
    Ok(())
}

fn temp_dir(prefix: &str) -> Result<PathBuf, CliError> {
    let path = std::env::temp_dir().join(format!("{prefix}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}
