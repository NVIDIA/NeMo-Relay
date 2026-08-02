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
            "Preview".into(),
            "Save".into(),
            "Cancel".into(),
        ];
        match select(&theme, "config.toml", &choices)? {
            0 => edit_gateway(&theme, &mut document)?,
            1 => edit_upstream(&theme, &mut document)?,
            2 => edit_logging(&theme, &mut document)?,
            3 => print_preview(&document),
            4 => {
                document.write(scope)?;
                println!("  ✓ Saved {}", document.path().display());
                return Ok(());
            }
            5 => return Err(CliError::Config(EDIT_CANCELLED_MESSAGE.into())),
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
            "Back".into(),
        ];
        match select(theme, "Provider upstreams", &choices)? {
            0 => edit_string(theme, document, "upstream", "openai_base_url")?,
            1 => edit_secret(theme, document, "openai_auth_header")?,
            2 => edit_string(theme, document, "upstream", "anthropic_base_url")?,
            3 => edit_secret(theme, document, "anthropic_auth_header")?,
            4 => return Ok(()),
            _ => unreachable!(),
        }
    }
}

fn edit_logging(theme: &ColorfulTheme, document: &mut ConfigDocument) -> Result<(), CliError> {
    loop {
        let choices = [
            format!("Level: {}", document.string_summary("logging", "level")),
            format!(
                "Stderr format: {}",
                document.string_summary("logging", "stderr_format")
            ),
            format!(
                "Flush interval (ms): {}",
                document.integer_summary("logging", "flush_interval_millis")
            ),
            format!("File sinks ({})", document.sink_count()),
            "Back".into(),
        ];
        match select(theme, "Operational logging", &choices)? {
            0 => edit_enum(theme, document, "logging", "level", LOG_LEVELS)?,
            1 => edit_enum(theme, document, "logging", "stderr_format", LOG_FORMATS)?,
            2 => edit_nonnegative_integer(theme, document, "logging", "flush_interval_millis")?,
            3 => edit_sinks(theme, document)?,
            4 => return Ok(()),
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
) -> Result<(), CliError> {
    let configured = document.has_key(section, key);
    match choose_action(theme, configured)? {
        0 => {
            let value = prompt_u64(
                theme,
                "Milliseconds (0 flushes on shutdown)",
                document.integer(section, key),
            )?;
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
            let value = Password::with_theme(theme)
                .with_prompt("Authorization header value")
                .allow_empty_password(false)
                .interact()
                .map_err(prompt_error)?;
            document.set_auth_header(key, value)?;
        }
        1 if configured => document.clear_key("upstream", key)?,
        _ => {}
    }
    Ok(())
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
