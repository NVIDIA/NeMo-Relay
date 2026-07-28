// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interactive editor for the non-agent sections of Relay's `config.toml`.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use dialoguer::theme::ColorfulTheme;
use dialoguer::{Input, Password, Select};
use nemo_relay::logging::MAX_FILE_SINK_QUEUE_ENTRIES;
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table, Value, value};

use super::ConfigEditCommand;
use crate::error::CliError;

const EDIT_CANCELLED_MESSAGE: &str = "configuration edit cancelled — no config saved";
const LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace"];
const LOG_FORMATS: &[&str] = &["human", "jsonl"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetScope {
    User,
    Project,
    Global,
}

impl From<&ConfigEditCommand> for TargetScope {
    fn from(command: &ConfigEditCommand) -> Self {
        if command.project {
            Self::Project
        } else if command.global {
            Self::Global
        } else {
            Self::User
        }
    }
}

pub(super) fn edit(command: ConfigEditCommand) -> Result<(), CliError> {
    ensure_tty()?;
    let scope = TargetScope::from(&command);
    let path = target_path(scope)?;
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

fn ensure_tty_with(stdin_is_terminal: bool) -> Result<(), CliError> {
    if stdin_is_terminal {
        Ok(())
    } else {
        Err(CliError::Config(
            "interactive configuration editing requires a TTY".into(),
        ))
    }
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
                .validate_with(|value: &String| {
                    if value.trim().is_empty() {
                        Err("value must not be empty; use Clear to remove it")
                    } else {
                        Ok(())
                    }
                })
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
                    .validate_with(|value: &String| {
                        if value.trim().is_empty() {
                            Err("value must not be empty".to_owned())
                        } else {
                            Ok(())
                        }
                    })
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
        .validate_with(|value: &String| {
            if value.trim().is_empty() {
                Err("value must not be empty".to_owned())
            } else {
                Ok(())
            }
        })
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

struct ConfigDocument {
    path: PathBuf,
    document: DocumentMut,
}

impl ConfigDocument {
    fn read(path: PathBuf) -> Result<Self, CliError> {
        let document = if path.exists() {
            std::fs::read_to_string(&path)?.parse().map_err(|error| {
                CliError::Config(format!("invalid TOML in {}: {error}", path.display()))
            })?
        } else {
            DocumentMut::new()
        };
        Ok(Self { path, document })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn write(&self, scope: TargetScope) -> Result<(), CliError> {
        let contents = self.document.to_string();
        match scope {
            TargetScope::Global => {
                if self.has_auth_headers() {
                    return Err(CliError::Config(
                        "global config cannot include upstream authorization headers; configure credentials in a user config".into(),
                    ));
                }
                crate::filesystem::atomic_write_system_readable(&self.path, contents.as_bytes())
            }
            TargetScope::User | TargetScope::Project => {
                crate::filesystem::atomic_write_private(&self.path, contents.as_bytes())
            }
        }
        .map_err(CliError::Config)
    }

    fn has_auth_headers(&self) -> bool {
        ["openai_auth_header", "anthropic_auth_header"]
            .into_iter()
            .any(|key| self.has_key("upstream", key))
    }

    fn preview(&self) -> String {
        let mut document = self.document.clone();
        if let Some(upstream) = document.get_mut("upstream") {
            for key in ["openai_auth_header", "anthropic_auth_header"] {
                if let Some(table) = upstream.as_table_mut() {
                    if table.contains_key(key) {
                        table[key] = value("<redacted>");
                    }
                } else if let Some(inline) =
                    upstream.as_value_mut().and_then(Value::as_inline_table_mut)
                    && inline.contains_key(key)
                {
                    inline.insert(key, Value::from("<redacted>"));
                }
            }
        }
        document.to_string()
    }

    fn item(&self, section: &str, key: &str) -> Option<&Item> {
        self.document.get(section)?.as_table()?.get(key)
    }

    fn has_key(&self, section: &str, key: &str) -> bool {
        self.item(section, key).is_some()
            || self
                .document
                .get(section)
                .and_then(Item::as_value)
                .and_then(Value::as_inline_table)
                .is_some_and(|table| table.contains_key(key))
    }

    fn string(&self, section: &str, key: &str) -> Option<String> {
        self.item(section, key)
            .and_then(Item::as_value)
            .and_then(Value::as_str)
            .or_else(|| {
                self.document
                    .get(section)
                    .and_then(Item::as_value)
                    .and_then(Value::as_inline_table)
                    .and_then(|table| table.get(key))
                    .and_then(Value::as_str)
            })
            .map(str::to_owned)
    }

    fn integer(&self, section: &str, key: &str) -> Option<u64> {
        self.item(section, key)
            .and_then(Item::as_value)
            .and_then(Value::as_integer)
            .or_else(|| {
                self.document
                    .get(section)
                    .and_then(Item::as_value)
                    .and_then(Value::as_inline_table)
                    .and_then(|table| table.get(key))
                    .and_then(Value::as_integer)
            })
            .and_then(|value| u64::try_from(value).ok())
    }

    fn string_summary(&self, section: &str, key: &str) -> String {
        match (self.has_key(section, key), self.string(section, key)) {
            (false, _) => "unset".into(),
            (true, Some(value)) => value,
            (true, None) => "invalid".into(),
        }
    }

    fn integer_summary(&self, section: &str, key: &str) -> String {
        match (self.has_key(section, key), self.integer(section, key)) {
            (false, _) => "unset".into(),
            (true, Some(value)) => value.to_string(),
            (true, None) => "invalid".into(),
        }
    }

    fn secret_summary(&self, key: &str) -> &'static str {
        if self.has_key("upstream", key) {
            "configured"
        } else {
            "unset"
        }
    }

    fn gateway_summary(&self) -> &'static str {
        if self.has_key("gateway", "max_hook_payload_bytes")
            || self.has_key("gateway", "max_passthrough_body_bytes")
        {
            "configured"
        } else {
            "defaults"
        }
    }

    fn upstream_summary(&self) -> &'static str {
        if self.document.get("upstream").is_some() {
            "configured"
        } else {
            "defaults"
        }
    }

    fn logging_summary(&self) -> &'static str {
        if self.document.get("logging").is_some() {
            "configured"
        } else {
            "defaults"
        }
    }

    fn table_mut(&mut self, section: &str) -> Result<&mut Table, CliError> {
        if self.document.get(section).is_none() {
            self.document[section] = Item::Table(Table::new());
        }
        self.document[section].as_table_mut().ok_or_else(|| {
            CliError::Config(format!(
                "[{section}] must be a TOML table before it can be edited"
            ))
        })
    }

    fn set_string(&mut self, section: &str, key: &str, new_value: String) -> Result<(), CliError> {
        self.set_value(section, key, Value::from(new_value))
    }

    fn set_integer(&mut self, section: &str, key: &str, new_value: u64) -> Result<(), CliError> {
        let numeric = i64::try_from(new_value)
            .map_err(|_| CliError::Config(format!("{section}.{key} is too large")))?;
        self.set_value(section, key, Value::from(numeric))
    }

    fn set_positive_integer(
        &mut self,
        section: &str,
        key: &str,
        new_value: u64,
    ) -> Result<(), CliError> {
        if new_value == 0 {
            return Err(CliError::Config(format!(
                "{section}.{key} must be greater than 0"
            )));
        }
        self.set_integer(section, key, new_value)
    }

    fn set_enum(
        &mut self,
        section: &str,
        key: &str,
        new_value: &str,
        allowed: &[&str],
    ) -> Result<(), CliError> {
        if !allowed.contains(&new_value) {
            return Err(CliError::Config(format!(
                "invalid {section}.{key}: {new_value}"
            )));
        }
        self.set_string(section, key, new_value.into())
    }

    fn set_auth_header(&mut self, key: &str, new_value: String) -> Result<(), CliError> {
        let value = new_value.trim();
        if value.is_empty() {
            return Err(CliError::Config(format!(
                "upstream.{key} must not be empty"
            )));
        }
        axum::http::HeaderValue::from_str(value).map_err(|_| {
            CliError::Config(format!("upstream.{key} must be a valid HTTP header value"))
        })?;
        self.set_string("upstream", key, value.into())
    }

    fn clear_key(&mut self, section: &str, key: &str) -> Result<(), CliError> {
        let empty = match self.document.get_mut(section) {
            Some(item) => {
                if let Some(table) = item.as_table_mut() {
                    table.remove(key);
                    table.is_empty()
                } else if let Some(table) = item.as_value_mut().and_then(Value::as_inline_table_mut)
                {
                    table.remove(key);
                    table.is_empty()
                } else {
                    return Err(CliError::Config(format!(
                        "[{section}] must be a TOML table before it can be edited"
                    )));
                }
            }
            None => false,
        };
        if empty {
            self.document.remove(section);
        }
        Ok(())
    }

    fn sinks(&self) -> Option<&ArrayOfTables> {
        self.document
            .get("logging")?
            .as_table()?
            .get("sinks")?
            .as_array_of_tables()
    }

    fn sinks_mut(&mut self) -> Result<&mut ArrayOfTables, CliError> {
        let logging = self.table_mut("logging")?;
        if logging.get("sinks").is_none() {
            logging["sinks"] = Item::ArrayOfTables(ArrayOfTables::new());
        }
        logging["sinks"].as_array_of_tables_mut().ok_or_else(|| {
            CliError::Config(
                "logging.sinks must be an array of tables before it can be edited".into(),
            )
        })
    }

    fn sink_count(&self) -> usize {
        self.sinks().map_or(0, ArrayOfTables::len)
    }

    fn sink_labels(&self) -> Vec<String> {
        self.sinks()
            .map(|sinks| {
                sinks
                    .iter()
                    .enumerate()
                    .map(|(index, sink)| {
                        let path = sink
                            .get("path")
                            .and_then(Item::as_value)
                            .and_then(Value::as_str)
                            .unwrap_or("invalid path");
                        format!("sink {} ({path})", index + 1)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn sink(&self, index: usize) -> Option<&Table> {
        self.sinks()?.get(index)
    }

    fn sink_mut(&mut self, index: usize) -> Result<&mut Table, CliError> {
        self.sinks_mut()?
            .get_mut(index)
            .ok_or_else(|| CliError::Config(format!("logging sink {} does not exist", index + 1)))
    }

    fn add_sink(&mut self, path: String) -> Result<(), CliError> {
        let mut sink = Table::new();
        sink["path"] = value(path);
        self.sinks_mut()?.push(sink);
        Ok(())
    }

    fn remove_sink(&mut self, index: usize) -> Result<(), CliError> {
        let empty = {
            let sinks = self.sinks_mut()?;
            if index >= sinks.len() {
                return Err(CliError::Config(format!(
                    "logging sink {} does not exist",
                    index + 1
                )));
            }
            sinks.remove(index);
            sinks.is_empty()
        };
        if empty {
            self.clear_key("logging", "sinks")?;
        }
        Ok(())
    }

    fn sink_has_key(&self, index: usize, key: &str) -> Result<bool, CliError> {
        Ok(self
            .sink(index)
            .ok_or_else(|| CliError::Config(format!("logging sink {} does not exist", index + 1)))?
            .contains_key(key))
    }

    fn sink_string(&self, index: usize, key: &str) -> Option<String> {
        self.sink(index)?
            .get(key)?
            .as_value()?
            .as_str()
            .map(str::to_owned)
    }

    fn sink_integer(&self, index: usize, key: &str) -> Option<u64> {
        self.sink(index)?
            .get(key)?
            .as_value()?
            .as_integer()
            .and_then(|value| u64::try_from(value).ok())
    }

    fn sink_string_summary(&self, index: usize, key: &str) -> String {
        match (
            self.sink(index).is_some_and(|sink| sink.contains_key(key)),
            self.sink_string(index, key),
        ) {
            (false, _) => "unset".into(),
            (true, Some(value)) => value,
            (true, None) => "invalid".into(),
        }
    }

    fn sink_integer_summary(&self, index: usize, key: &str) -> String {
        match (
            self.sink(index).is_some_and(|sink| sink.contains_key(key)),
            self.sink_integer(index, key),
        ) {
            (false, _) => "unset".into(),
            (true, Some(value)) => value.to_string(),
            (true, None) => "invalid".into(),
        }
    }

    fn sink_rotation_summary(&self, index: usize) -> String {
        match (
            self.sink_integer(index, "max_file_size_bytes"),
            self.sink_integer(index, "retained_files"),
        ) {
            (None, None) => "unset".into(),
            (Some(size), Some(retained)) => format!("{size} bytes, {retained} backups"),
            _ => "incomplete".into(),
        }
    }

    fn set_sink_string(
        &mut self,
        index: usize,
        key: &str,
        new_value: String,
    ) -> Result<(), CliError> {
        self.sink_mut(index)?[key] = value(new_value);
        Ok(())
    }

    fn set_sink_enum(
        &mut self,
        index: usize,
        key: &str,
        new_value: &str,
        allowed: &[&str],
    ) -> Result<(), CliError> {
        if !allowed.contains(&new_value) {
            return Err(CliError::Config(format!(
                "invalid logging sink {key}: {new_value}"
            )));
        }
        self.set_sink_string(index, key, new_value.into())
    }

    fn set_sink_queue_capacity(&mut self, index: usize, capacity: u64) -> Result<(), CliError> {
        if capacity == 0 {
            return Err(CliError::Config(
                "logging sink queue_capacity must be greater than 0".into(),
            ));
        }
        if capacity > MAX_FILE_SINK_QUEUE_ENTRIES as u64 {
            return Err(CliError::Config(format!(
                "logging sink queue_capacity {capacity} exceeds maximum {MAX_FILE_SINK_QUEUE_ENTRIES} entries per file sink"
            )));
        }
        let capacity = i64::try_from(capacity)
            .map_err(|_| CliError::Config("logging sink queue_capacity is too large".into()))?;
        self.sink_mut(index)?["queue_capacity"] = value(capacity);
        Ok(())
    }

    fn set_sink_rotation(
        &mut self,
        index: usize,
        max_size: u64,
        retained: u64,
    ) -> Result<(), CliError> {
        let max_size = i64::try_from(max_size).map_err(|_| {
            CliError::Config("logging sink max_file_size_bytes is too large".into())
        })?;
        let retained_i64 = i64::try_from(retained)
            .map_err(|_| CliError::Config("logging sink retained_files is too large".into()))?;
        let retained = usize::try_from(retained)
            .map_err(|_| CliError::Config("logging sink retained_files is too large".into()))?;
        nemo_relay::logging::FileLogRotationConfig::new(max_size as u64, retained)
            .map_err(|error| CliError::Config(error.to_string()))?;
        let sink = self.sink_mut(index)?;
        sink["max_file_size_bytes"] = value(max_size);
        sink["retained_files"] = value(retained_i64);
        Ok(())
    }

    fn clear_sink_key(&mut self, index: usize, key: &str) -> Result<(), CliError> {
        self.sink_mut(index)?.remove(key);
        Ok(())
    }

    fn clear_sink_rotation(&mut self, index: usize) -> Result<(), CliError> {
        let sink = self.sink_mut(index)?;
        sink.remove("max_file_size_bytes");
        sink.remove("retained_files");
        Ok(())
    }

    fn set_value(&mut self, section: &str, key: &str, new_value: Value) -> Result<(), CliError> {
        if self.document.get(section).is_none() {
            self.document[section] = Item::Table(Table::new());
        }
        let item = &mut self.document[section];
        if let Some(table) = item.as_table_mut() {
            table[key] = Item::Value(new_value);
            Ok(())
        } else if let Some(table) = item.as_value_mut().and_then(Value::as_inline_table_mut) {
            table.insert(key, new_value);
            Ok(())
        } else {
            Err(CliError::Config(format!(
                "[{section}] must be a TOML table before it can be edited"
            )))
        }
    }
}

fn target_path(scope: TargetScope) -> Result<PathBuf, CliError> {
    match scope {
        TargetScope::User => crate::configuration::user_config_dir()
            .map(|directory| directory.join("config.toml"))
            .ok_or_else(|| {
                CliError::Config(
                    "cannot determine user config directory; set HOME or XDG_CONFIG_HOME".into(),
                )
            }),
        TargetScope::Project => Ok(project_config_path(&std::env::current_dir()?)),
        TargetScope::Global => Ok(PathBuf::from("/etc/nemo-relay/config.toml")),
    }
}

fn project_config_path(start: &Path) -> PathBuf {
    for ancestor in start.ancestors() {
        let candidate = ancestor.join(".nemo-relay/config.toml");
        if candidate.exists() {
            return candidate;
        }
    }
    start.join(".nemo-relay/config.toml")
}

#[cfg(test)]
#[path = "../../../tests/coverage/commands/configure_editor_tests.rs"]
mod tests;
