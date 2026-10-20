// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Terminal-only prompt adapter for the interactive config editor.

use std::io::IsTerminal;
use std::path::PathBuf;

use dialoguer::theme::ColorfulTheme;
use dialoguer::{Input, Password, Select};

use super::{
    ConfigDocument, ConfigEditCommand, LOG_FORMATS, LOG_LEVELS, ensure_tty_with,
    resolve_edit_target,
};
use crate::error::CliError;

fn require_nonempty(value: &str) -> Result<(), &'static str> {
    if value.trim().is_empty() {
        Err("value must not be empty")
    } else {
        Ok(())
    }
}

const EDIT_CANCELLED_MESSAGE: &str = "configuration edit cancelled — no config saved";
pub(super) fn edit(
    command: ConfigEditCommand,
    explicit_path: Option<PathBuf>,
) -> Result<(), CliError> {
    ensure_tty()?;
    let (scope, path) = resolve_edit_target(&command, explicit_path)?;
    let mut document = ConfigDocument::read(path)?;
    let theme = ColorfulTheme::default();

    crate::banner::print_intro();
    println!("  Editing config at {}", document.path().display());
    println!("  Secrets are never displayed. Choose Save to write changes.");
    println!();

    loop {
        let choices = [
            format!("Gateway limits ({})", document.gateway_summary()),
            format!("Provider upstreams ({})", document.upstream_summary()),
            format!("Operational logging ({})", document.logging_summary()),
            "Agent commands".into(),
            "Daemon logging".into(),
            "Preview".into(),
            "Save".into(),
            "Cancel".into(),
        ];
        match select(&theme, "config.toml", &choices)? {
            0 => edit_gateway(&theme, &mut document)?,
            1 => edit_upstream(&theme, &mut document)?,
            2 => edit_logging(&theme, &mut document, "logging")?,
            3 => edit_agents(&theme, &mut document)?,
            4 => edit_logging(&theme, &mut document, "daemon.logging")?,
            5 => print_preview(&document),
            6 => match document.write(scope) {
                Ok(()) => {
                    println!("  ✓ Saved {}", document.path().display());
                    return Ok(());
                }
                Err(error) => println!("  Could not save: {error}"),
            },
            7 => return Err(CliError::Config(EDIT_CANCELLED_MESSAGE.into())),
            _ => unreachable!("select returns an in-range index"),
        }
    }
}

fn ensure_tty() -> Result<(), CliError> {
    ensure_tty_with(std::io::stdin().is_terminal())
}

fn select(theme: &ColorfulTheme, prompt: &str, choices: &[String]) -> Result<usize, CliError> {
    Select::with_theme(theme)
        .with_prompt(prompt)
        .items(choices)
        .default(0)
        .interact()
        .map_err(prompt_error)
}

fn choose_action(theme: &ColorfulTheme, configured: bool) -> Result<usize, CliError> {
    let choices = if configured {
        vec!["Set or replace".into(), "Clear".into(), "Back".into()]
    } else {
        vec!["Set".into(), "Back".into()]
    };
    select(theme, "Action", &choices)
}

fn edit_gateway(theme: &ColorfulTheme, document: &mut ConfigDocument) -> Result<(), CliError> {
    loop {
        let choices = [
            format!(
                "Maximum hook payload bytes: {}",
                document.integer_summary("gateway", "max_hook_payload_bytes")
            ),
            format!(
                "Maximum passthrough body bytes: {}",
                document.integer_summary("gateway", "max_passthrough_body_bytes")
            ),
            "Back".into(),
        ];
        match select(theme, "Gateway limits", &choices)? {
            0 => edit_positive_integer(theme, document, "gateway", "max_hook_payload_bytes")?,
            1 => edit_positive_integer(theme, document, "gateway", "max_passthrough_body_bytes")?,
            2 => return Ok(()),
            _ => unreachable!(),
        }
    }
}

fn edit_upstream(theme: &ColorfulTheme, document: &mut ConfigDocument) -> Result<(), CliError> {
    loop {
        let choices = [
            format!(
                "OpenAI base URL: {}",
                document.string_summary("upstream", "openai_base_url")
            ),
            format!(
                "OpenAI authorization header: {}",
                document.secret_summary("openai_auth_header")
            ),
            format!(
                "Anthropic base URL: {}",
                document.string_summary("upstream", "anthropic_base_url")
            ),
            format!(
                "Anthropic authorization header: {}",
                document.secret_summary("anthropic_auth_header")
            ),
            format!(
                "ChatGPT/Codex base URL: {}",
                document.string_summary("upstream", "chatgpt_codex_base_url")
            ),
            format!(
                "Response timeout (seconds, 0 disables): {}",
                document.integer_summary("upstream", "response_timeout_secs")
            ),
            format!(
                "OpenAI API key: {}",
                document.secret_summary("openai_api_key")
            ),
            format!(
                "Anthropic API key: {}",
                document.secret_summary("anthropic_api_key")
            ),
            format!(
                "OpenAI API key helper: {}",
                document.secret_summary("openai_api_key_helper")
            ),
            format!(
                "Anthropic API key helper: {}",
                document.secret_summary("anthropic_api_key_helper")
            ),
            "Back".into(),
        ];
        match select(theme, "Provider upstreams", &choices)? {
            0 => edit_string(theme, document, "upstream", "openai_base_url")?,
            1 => edit_secret(theme, document, "openai_auth_header")?,
            2 => edit_string(theme, document, "upstream", "anthropic_base_url")?,
            3 => edit_secret(theme, document, "anthropic_auth_header")?,
            4 => edit_string(theme, document, "upstream", "chatgpt_codex_base_url")?,
            5 => edit_nonnegative_integer(
                theme,
                document,
                "upstream",
                "response_timeout_secs",
                "Seconds (0 disables the response timeout)",
            )?,
            6 => edit_secret(theme, document, "openai_api_key")?,
            7 => edit_secret(theme, document, "anthropic_api_key")?,
            8 => edit_api_key_helper(theme, document, "openai")?,
            9 => edit_api_key_helper(theme, document, "anthropic")?,
            10 => return Ok(()),
            _ => unreachable!(),
        }
    }
}

fn edit_agents(theme: &ColorfulTheme, document: &mut ConfigDocument) -> Result<(), CliError> {
    const AGENTS: &[&str] = &["claude", "codex", "pi"];
    loop {
        let mut choices = AGENTS
            .iter()
            .map(|agent| {
                format!(
                    "{agent} command: {}",
                    document.string_summary(&format!("agents.{agent}"), "command")
                )
            })
            .collect::<Vec<_>>();
        choices.push("Back".into());
        match select(theme, "Agent commands", &choices)? {
            index if index < AGENTS.len() => edit_string(
                theme,
                document,
                &format!("agents.{}", AGENTS[index]),
                "command",
            )?,
            _ => return Ok(()),
        }
    }
}

fn edit_boolean(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    section: &str,
    key: &str,
) -> Result<(), CliError> {
    let configured = document.has_key(section, key);
    match choose_action(theme, configured)? {
        0 => {
            let selected = Select::with_theme(theme)
                .with_prompt("Value")
                .items(["true", "false"])
                .default(usize::from(document.boolean(section, key) == Some(false)))
                .interact()
                .map_err(prompt_error)?;
            document.set_value(section, key, toml_edit::Value::from(selected == 0))?;
        }
        1 if configured => document.clear_key(section, key)?,
        _ => {}
    }
    Ok(())
}

fn edit_logging(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    section: &'static str,
) -> Result<(), CliError> {
    document.logging_section = section;
    loop {
        let choices = [
            format!("Level: {}", document.string_summary(section, "level")),
            format!(
                "Stderr format: {}",
                document.string_summary(section, "stderr_format")
            ),
            format!(
                "Flush interval (ms): {}",
                document.integer_summary(section, "flush_interval_millis")
            ),
            format!(
                "Stderr enabled: {}",
                document.boolean_summary(section, "stderr_enabled")
            ),
            format!("File sinks ({})", document.sink_count()),
            "Back".into(),
        ];
        match select(theme, section, &choices)? {
            0 => edit_enum(theme, document, section, "level", LOG_LEVELS)?,
            1 => edit_enum(theme, document, section, "stderr_format", LOG_FORMATS)?,
            2 => edit_nonnegative_integer(
                theme,
                document,
                section,
                "flush_interval_millis",
                "Milliseconds (0 flushes on shutdown)",
            )?,
            3 => edit_boolean(theme, document, section, "stderr_enabled")?,
            4 => edit_sinks(theme, document)?,
            5 => return Ok(()),
            _ => unreachable!(),
        }
    }
}

fn edit_positive_integer(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    section: &str,
    key: &str,
) -> Result<(), CliError> {
    let configured = document.has_key(section, key);
    match choose_action(theme, configured)? {
        0 => {
            let value = prompt_u64(theme, "Value in bytes", document.integer(section, key))?;
            document.set_positive_integer(section, key, value)?;
        }
        1 if configured => document.clear_key(section, key)?,
        _ => {}
    }
    Ok(())
}

fn edit_nonnegative_integer(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    section: &str,
    key: &str,
    prompt: &str,
) -> Result<(), CliError> {
    let configured = document.has_key(section, key);
    match choose_action(theme, configured)? {
        0 => {
            let value = prompt_u64(theme, prompt, document.integer(section, key))?;
            document.set_integer(section, key, value)?;
        }
        1 if configured => document.clear_key(section, key)?,
        _ => {}
    }
    Ok(())
}

fn edit_string(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    section: &str,
    key: &str,
) -> Result<(), CliError> {
    let configured = document.has_key(section, key);
    match choose_action(theme, configured)? {
        0 => {
            let default = document.string(section, key).unwrap_or_default();
            let value = Input::<String>::with_theme(theme)
                .with_prompt("Value")
                .with_initial_text(default)
                .validate_with(|value: &String| require_nonempty(value))
                .interact_text()
                .map_err(prompt_error)?;
            document.set_string(section, key, value)?;
        }
        1 if configured => document.clear_key(section, key)?,
        _ => {}
    }
    Ok(())
}

fn edit_secret(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    key: &str,
) -> Result<(), CliError> {
    let configured = document.has_key("upstream", key);
    match choose_action(theme, configured)? {
        0 => {
            println!(
                "  Setting this credential replaces the provider's other credential sources in this file."
            );
            let is_api_key = key.ends_with("_api_key");
            let value = Password::with_theme(theme)
                .with_prompt(if is_api_key {
                    "API key (bare token)"
                } else {
                    "Authorization header value"
                })
                .allow_empty_password(false)
                .validate_with(|input: &String| {
                    if is_api_key {
                        crate::upstream_auth::validate_api_key("API key", input.clone())
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    } else {
                        require_nonempty(input).map_err(str::to_owned)?;
                        axum::http::HeaderValue::from_str(input.trim())
                            .map(|_| ())
                            .map_err(|_| "value must be a valid HTTP header value".to_owned())
                    }
                })
                .interact()
                .map_err(prompt_error)?;
            if is_api_key {
                document.set_api_key(key, value)?;
            } else {
                document.set_auth_header(key, value)?;
            }
        }
        1 if configured => document.clear_key("upstream", key)?,
        _ => {}
    }
    Ok(())
}

fn edit_api_key_helper(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    provider: &str,
) -> Result<(), CliError> {
    let key = format!("{provider}_api_key_helper");
    let configured = document.has_key("upstream", &key);
    match choose_action(theme, configured)? {
        0 => {
            println!(
                "  Replace all helper fields. Command and arguments are hidden because they may contain secrets."
            );
            println!(
                "  Setting this helper replaces the provider's other credential sources in this file."
            );
            let command = Password::with_theme(theme)
                .with_prompt("Helper command")
                .allow_empty_password(false)
                .validate_with(|value: &String| require_nonempty(value))
                .interact()
                .map_err(prompt_error)?;
            let args = Password::with_theme(theme)
                .with_prompt("Arguments (JSON string array; empty means no arguments)")
                .allow_empty_password(true)
                .validate_with(|value: &String| parse_helper_args(value).map(|_| ()))
                .interact()
                .map_err(prompt_error)?;
            let args = parse_helper_args(&args).map_err(|error| CliError::Config(error.into()))?;
            let section = format!("upstream.{key}");
            let timeout_ms = Input::<u64>::with_theme(theme)
                .with_prompt("Timeout (milliseconds)")
                .default(
                    document
                        .integer(&section, "timeout_ms")
                        .filter(|value| *value > 0 && *value <= i64::MAX as u64)
                        .unwrap_or(10_000),
                )
                .validate_with(|value: &u64| {
                    if *value == 0 || *value > i64::MAX as u64 {
                        Err("timeout must be positive and fit a TOML integer")
                    } else {
                        Ok(())
                    }
                })
                .interact_text()
                .map_err(prompt_error)?;
            let schemes: &[&str] = if provider == "openai" {
                &["Provider default", "bearer"]
            } else {
                &["Provider default", "bearer", "api_key"]
            };
            let current = document.string(&section, "auth_scheme");
            let selected = Select::with_theme(theme)
                .with_prompt("Authentication scheme")
                .items(schemes)
                .default(
                    current
                        .as_deref()
                        .and_then(|current| schemes.iter().position(|scheme| *scheme == current))
                        .unwrap_or(0),
                )
                .interact()
                .map_err(prompt_error)?;
            let auth_scheme = match selected {
                0 => None,
                1 => Some(crate::upstream_auth::AuthScheme::Bearer),
                2 => Some(crate::upstream_auth::AuthScheme::ApiKey),
                _ => unreachable!(),
            };
            document.set_api_key_helper(
                provider,
                crate::upstream_auth::ApiKeyHelper {
                    command,
                    args,
                    timeout_ms,
                    auth_scheme,
                },
            )?;
        }
        1 if configured => document.clear_key("upstream", &key)?,
        _ => {}
    }
    Ok(())
}

fn parse_helper_args(value: &str) -> Result<Vec<String>, &'static str> {
    if value.trim().is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(value).map_err(|_| "arguments must be a JSON array of strings")
}

fn edit_enum(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    section: &str,
    key: &str,
    values: &[&str],
) -> Result<(), CliError> {
    let configured = document.has_key(section, key);
    match choose_action(theme, configured)? {
        0 => {
            let current = document.string(section, key);
            let default = current
                .as_deref()
                .and_then(|current| values.iter().position(|value| *value == current))
                .unwrap_or(0);
            let selected = Select::with_theme(theme)
                .with_prompt("Value")
                .items(values)
                .default(default)
                .interact()
                .map_err(prompt_error)?;
            document.set_enum(section, key, values[selected], values)?;
        }
        1 if configured => document.clear_key(section, key)?,
        _ => {}
    }
    Ok(())
}

fn edit_sinks(theme: &ColorfulTheme, document: &mut ConfigDocument) -> Result<(), CliError> {
    loop {
        let mut choices = document
            .sink_labels()
            .into_iter()
            .map(|label| format!("Edit {label}"))
            .collect::<Vec<_>>();
        let sink_count = choices.len();
        choices.push("Add file sink".into());
        choices.push("Back".into());
        match select(theme, "File sinks", &choices)? {
            index if index < sink_count => edit_sink(theme, document, index)?,
            index if index == sink_count => {
                let path = Input::<String>::with_theme(theme)
                    .with_prompt("File path")
                    .validate_with(|value: &String| require_nonempty(value))
                    .interact_text()
                    .map_err(prompt_error)?;
                document.add_sink(path)?;
            }
            _ => return Ok(()),
        }
    }
}

fn edit_sink(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    index: usize,
) -> Result<(), CliError> {
    loop {
        let choices = [
            format!("Path: {}", document.sink_string_summary(index, "path")),
            format!("Level: {}", document.sink_string_summary(index, "level")),
            format!("Format: {}", document.sink_string_summary(index, "format")),
            format!(
                "Queue capacity: {}",
                document.sink_integer_summary(index, "queue_capacity")
            ),
            format!("Rotation: {}", document.sink_rotation_summary(index)),
            "Remove sink".into(),
            "Back".into(),
        ];
        match select(theme, "File sink", &choices)? {
            0 => edit_sink_path(theme, document, index)?,
            1 => edit_sink_enum(theme, document, index, "level", LOG_LEVELS)?,
            2 => edit_sink_enum(theme, document, index, "format", LOG_FORMATS)?,
            3 => edit_sink_queue_capacity(theme, document, index)?,
            4 => edit_sink_rotation(theme, document, index)?,
            5 => {
                document.remove_sink(index)?;
                return Ok(());
            }
            _ => return Ok(()),
        }
    }
}

fn edit_sink_path(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    index: usize,
) -> Result<(), CliError> {
    let current = document.sink_string(index, "path").unwrap_or_default();
    let value = Input::<String>::with_theme(theme)
        .with_prompt("File path")
        .with_initial_text(current)
        .validate_with(|value: &String| require_nonempty(value))
        .interact_text()
        .map_err(prompt_error)?;
    document.set_sink_string(index, "path", value)
}

fn edit_sink_enum(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    index: usize,
    key: &str,
    values: &[&str],
) -> Result<(), CliError> {
    let configured = document.sink_has_key(index, key)?;
    match choose_action(theme, configured)? {
        0 => {
            let default = document
                .sink_string(index, key)
                .as_deref()
                .and_then(|current| values.iter().position(|value| *value == current))
                .unwrap_or(0);
            let selected = Select::with_theme(theme)
                .with_prompt("Value")
                .items(values)
                .default(default)
                .interact()
                .map_err(prompt_error)?;
            document.set_sink_enum(index, key, values[selected], values)?;
        }
        1 if configured => document.clear_sink_key(index, key)?,
        _ => {}
    }
    Ok(())
}

fn edit_sink_queue_capacity(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    index: usize,
) -> Result<(), CliError> {
    let configured = document.sink_has_key(index, "queue_capacity")?;
    match choose_action(theme, configured)? {
        0 => {
            let value = prompt_u64(
                theme,
                "Queue entries",
                document.sink_integer(index, "queue_capacity"),
            )?;
            document.set_sink_queue_capacity(index, value)?;
        }
        1 if configured => document.clear_sink_key(index, "queue_capacity")?,
        _ => {}
    }
    Ok(())
}

fn edit_sink_rotation(
    theme: &ColorfulTheme,
    document: &mut ConfigDocument,
    index: usize,
) -> Result<(), CliError> {
    let configured = document.sink_has_key(index, "max_file_size_bytes")?
        || document.sink_has_key(index, "retained_files")?;
    match choose_action(theme, configured)? {
        0 => {
            let size = prompt_u64(
                theme,
                "Maximum file size in bytes",
                document.sink_integer(index, "max_file_size_bytes"),
            )?;
            let retained = prompt_u64(
                theme,
                "Retained backup files",
                document.sink_integer(index, "retained_files"),
            )?;
            document.set_sink_rotation(index, size, retained)?;
        }
        1 if configured => document.clear_sink_rotation(index)?,
        _ => {}
    }
    Ok(())
}

fn prompt_u64(theme: &ColorfulTheme, prompt: &str, current: Option<u64>) -> Result<u64, CliError> {
    let mut input = Input::<u64>::with_theme(theme).with_prompt(prompt);
    if let Some(current) = current {
        input = input.with_initial_text(current.to_string());
    }
    input.interact_text().map_err(prompt_error)
}

fn prompt_error(error: dialoguer::Error) -> CliError {
    CliError::Config(format!("configuration edit error: {error}"))
}

fn print_preview(document: &ConfigDocument) {
    println!();
    println!("  ─── Preview ─────────────────────────────────────────────");
    for line in document.preview().lines() {
        println!("  {line}");
    }
    println!();
}
