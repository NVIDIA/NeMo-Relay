// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interactive editor for the non-agent sections of Relay's `config.toml`.

use std::path::{Path, PathBuf};

use nemo_relay::logging::MAX_FILE_SINK_QUEUE_ENTRIES;
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table, Value, value};

use super::ConfigEditCommand;
use crate::error::CliError;

const LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace"];
const LOG_FORMATS: &[&str] = &["human", "jsonl"];

mod prompt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetScope {
    User,
    Global,
}

impl From<&ConfigEditCommand> for TargetScope {
    fn from(command: &ConfigEditCommand) -> Self {
        if command.global {
            Self::Global
        } else {
            Self::User
        }
    }
}

pub(super) fn edit(
    command: ConfigEditCommand,
    explicit_path: Option<PathBuf>,
) -> Result<(), CliError> {
    prompt::edit(command, explicit_path)
}

fn resolve_edit_target(
    command: &ConfigEditCommand,
    explicit_path: Option<PathBuf>,
) -> Result<(TargetScope, PathBuf), CliError> {
    let scope = TargetScope::from(command);
    let path = if command.global {
        target_path(scope)?
    } else {
        match explicit_path {
            Some(path) => path,
            None => target_path(scope)?,
        }
    };
    Ok((scope, path))
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
            TargetScope::User => {
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
        TargetScope::Global => Ok(crate::configuration::system_config_dir().join("config.toml")),
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/commands/configure_editor_tests.rs"]
mod tests;
