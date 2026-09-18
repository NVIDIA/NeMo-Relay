// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::agents::CodingAgent;
use crate::error::CliError;
use crate::hooks::{generated_policy_hooks, transparent_hook_forward_commands_with_config};
use crate::process::{PreparedAgentLaunch, insert_after_host};

pub(crate) fn prepare(
    launch: &mut PreparedAgentLaunch,
    gateway_url: &str,
    proxy_credential: &crate::provider_auth::TransparentProxyCredential,
    dry_run: bool,
) -> Result<(), CliError> {
    let proxy_header = format!(
        "{}: {}",
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
        proxy_credential.expose()
    );
    let custom_headers = std::env::var("ANTHROPIC_CUSTOM_HEADERS")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map_or_else(
            || proxy_header.clone(),
            |value| replace_custom_header(&value, &proxy_header),
        );
    launch.set_secret_env("ANTHROPIC_CUSTOM_HEADERS", custom_headers);
    let hook_integrity_risk = HookIntegrityRisk::detect(&launch.argv, launch.host_index);
    if dry_run {
        insert_after_host(
            &mut launch.argv,
            launch.host_index,
            [
                "--plugin-dir".into(),
                "<temporary-claude-plugin-dir>".into(),
            ],
        );
        insert_before_argument_boundary(
            &mut launch.argv,
            launch.host_index,
            ["--settings".into(), "<temporary-claude-settings>".into()],
        );
        launch
            .env
            .push(("ANTHROPIC_BASE_URL".into(), gateway_url.to_string()));
        launch
            .notes
            .push("would generate a temporary Claude Code plugin directory".into());
        record_hook_integrity_risk(launch, hook_integrity_risk);
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
    let hook_config = root.join(".nemo-relay-hook-config.json");
    crate::hooks::HookCommandConfig::transparent(CodingAgent::ClaudeCode, gateway_url)
        .write(&hook_config)
        .map_err(CliError::Launch)?;
    let hook_commands = transparent_hook_forward_commands_with_config(
        &transparent_hook_executable(),
        CodingAgent::ClaudeCode,
        &hook_config,
    )
    .map_err(CliError::Launch)?;
    write_hooks(
        &root.join("hooks/hooks.json"),
        generated_policy_hooks(CodingAgent::ClaudeCode, &hook_commands),
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
        ["--plugin-dir".into(), root.display().to_string()],
    );
    insert_before_argument_boundary(
        &mut launch.argv,
        launch.host_index,
        ["--settings".into(), settings_path.display().to_string()],
    );
    launch
        .env
        .push(("ANTHROPIC_BASE_URL".into(), gateway_url.to_string()));
    launch.temp_dirs.push(root);
    record_hook_integrity_risk(launch, hook_integrity_risk);
    Ok(())
}

#[derive(Default)]
struct HookIntegrityRisk {
    safe_mode: bool,
    bare_mode: bool,
}

impl HookIntegrityRisk {
    // Claude leaves ANTHROPIC_BASE_URL active in these modes while suppressing plugin hooks.
    // Match launcher-visible inputs only; user/managed settings and pre-host wrapper env are opaque.
    fn detect(argv: &[String], host_index: usize) -> Self {
        let arguments = claude_arguments(argv, host_index);
        // Claude itself scans raw argv for these modes, independent of option-value parsing.
        let informational_only = arguments.iter().any(|argument| {
            matches!(
                argument.as_str(),
                "-h" | "--help" | "-v" | "-V" | "--version"
            )
        }) && arguments.iter().all(|argument| {
            matches!(
                argument.as_str(),
                "--safe-mode" | "--bare" | "-h" | "--help" | "-v" | "-V" | "--version"
            )
        });
        if informational_only {
            return Self::default();
        }

        Self {
            safe_mode: arguments.iter().any(|argument| argument == "--safe-mode")
                || inherited_env_flag_enabled("CLAUDE_CODE_SAFE_MODE"),
            bare_mode: arguments.iter().any(|argument| argument == "--bare")
                || inherited_env_flag_enabled("CLAUDE_CODE_SIMPLE"),
        }
    }

    const fn detected(&self) -> bool {
        self.safe_mode || self.bare_mode
    }

    const fn sources(&self) -> &'static str {
        match (self.safe_mode, self.bare_mode) {
            (true, true) => "Claude safe and bare modes",
            (true, false) => "Claude safe mode",
            (false, true) => "Claude bare mode",
            (false, false) => "",
        }
    }
}

fn inherited_env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .as_deref()
        .is_some_and(claude_env_flag_enabled)
}

fn claude_env_flag_enabled(value: &str) -> bool {
    let value = value.trim();
    value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
}

fn record_hook_integrity_risk(launch: &mut PreparedAgentLaunch, risk: HookIntegrityRisk) {
    if !risk.detected() {
        return;
    }
    let note = format!(
        "Claude hooks at risk ({}): Relay routing stays configured, but hook observability and enforcement are not guaranteed",
        risk.sources(),
    );
    log::warn!(
        target: "nemo_relay.cli",
        event = "agent_invocation_warning",
        diagnostic_code = "claude_relay_hook_integrity_at_risk",
        agent = "claude",
        safe_mode_signal = risk.safe_mode,
        bare_mode_signal = risk.bare_mode,
        model_routing = "configured",
        hook_integrity = "at_risk",
        action = "remove_hook_disabling_claude_mode_if_hook_integrity_is_required",
        command_modified = false,
        arguments_redacted = true;
        "{note}"
    );
    launch.non_tty_warnings.push(note);
}

fn claude_arguments(argv: &[String], host_index: usize) -> &[String] {
    let arguments = argv.get(host_index + 1..).unwrap_or_default();
    let boundary = arguments
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(arguments.len());
    &arguments[..boundary]
}

fn insert_before_argument_boundary(
    argv: &mut Vec<String>,
    host_index: usize,
    values: impl IntoIterator<Item = String>,
) {
    let boundary = argv
        .iter()
        .skip(host_index + 1)
        .position(|argument| argument == "--")
        .map_or(argv.len(), |offset| host_index + 1 + offset);
    argv.splice(boundary..boundary, values);
}

fn replace_custom_header(existing: &str, replacement: &str) -> String {
    let replacement_name = replacement
        .split_once(':')
        .map_or(replacement, |(name, _)| name)
        .trim();
    existing
        .lines()
        .filter(|line| {
            line.split_once(':')
                .is_none_or(|(name, _)| !name.trim().eq_ignore_ascii_case(replacement_name))
        })
        .chain(std::iter::once(replacement))
        .collect::<Vec<_>>()
        .join("\n")
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
