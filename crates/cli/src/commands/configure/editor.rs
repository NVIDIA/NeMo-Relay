// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interactive editor for Relay's `config.toml` settings.

use std::path::{Path, PathBuf};

use nemo_relay::logging::MAX_FILE_SINK_QUEUE_ENTRIES;
use toml_edit::{
    Array, ArrayOfTables, DocumentMut, InlineTable, Item, Table, TableLike, Value, value,
};

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
    logging_section: &'static str,
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
        Ok(Self {
            path,
            document,
            logging_section: "logging",
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn write(&self, scope: TargetScope) -> Result<(), CliError> {
        let contents = self.document.to_string();
        // Reject credentials in system-readable files before validating their contents.
        if scope == TargetScope::Global && self.has_auth_headers() {
            return Err(CliError::Config(
                "global config cannot include upstream authorization headers or API keys; configure credentials in a user config".into(),
            ));
        }
        let parsed = toml::from_str::<toml::Value>(&contents).map_err(|_| {
            CliError::Config("configuration must be valid TOML before saving".into())
        })?;
        crate::configuration::validate_config_document(parsed)?;
        match scope {
            TargetScope::Global => {
                crate::filesystem::atomic_write_system_readable(&self.path, contents.as_bytes())
            }
            TargetScope::User => {
                crate::filesystem::atomic_write_private(&self.path, contents.as_bytes())
            }
        }
        .map_err(CliError::Config)
    }

    fn has_auth_headers(&self) -> bool {
        [
            "openai_auth_header",
            "anthropic_auth_header",
            "openai_api_key",
            "anthropic_api_key",
        ]
        .into_iter()
        .any(|key| self.has_key("upstream", key))
    }

    fn preview(&self) -> String {
        let mut document = self.document.clone();
        if let Some(upstream) = document.get_mut("upstream") {
            for key in [
                "openai_auth_header",
                "anthropic_auth_header",
                "openai_api_key",
                "anthropic_api_key",
                "openai_api_key_helper",
                "anthropic_api_key_helper",
            ] {
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

    fn section(&self, section: &str) -> Option<&dyn toml_edit::TableLike> {
        let mut table: &dyn toml_edit::TableLike = self.document.as_table();
        for part in section.split('.') {
            table = table.get(part)?.as_table_like()?;
        }
        Some(table)
    }

    fn section_mut(&mut self, section: &str) -> Result<&mut dyn toml_edit::TableLike, CliError> {
        let mut table: &mut dyn toml_edit::TableLike = self.document.as_table_mut();
        for part in section.split('.') {
            if !table.contains_key(part) {
                table.insert(part, Item::Table(Table::new()));
            }
            table = table
                .get_mut(part)
                .and_then(Item::as_table_like_mut)
                .ok_or_else(|| {
                    CliError::Config(format!(
                        "[{section}] must be a TOML table before it can be edited"
                    ))
                })?;
        }
        Ok(table)
    }

    fn item(&self, section: &str, key: &str) -> Option<&Item> {
        self.section(section)?.get(key)
    }

    fn has_key(&self, section: &str, key: &str) -> bool {
        self.item(section, key).is_some()
    }

    fn string(&self, section: &str, key: &str) -> Option<String> {
        self.item(section, key)?.as_str().map(str::to_owned)
    }

    fn integer(&self, section: &str, key: &str) -> Option<u64> {
        self.item(section, key)?
            .as_integer()
            .and_then(|value| u64::try_from(value).ok())
    }

    fn boolean(&self, section: &str, key: &str) -> Option<bool> {
        self.item(section, key)?.as_bool()
    }

    fn boolean_summary(&self, section: &str, key: &str) -> String {
        match (self.has_key(section, key), self.boolean(section, key)) {
            (false, _) => "unset".into(),
            (true, Some(value)) => value.to_string(),
            (true, None) => "invalid".into(),
        }
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
        self.set_credential(key, Value::from(value))
    }

    fn set_api_key(&mut self, key: &str, new_value: String) -> Result<(), CliError> {
        let token = crate::upstream_auth::validate_api_key(&format!("upstream.{key}"), new_value)?;
        self.set_credential(key, Value::from(token))
    }

    fn set_api_key_helper(
        &mut self,
        provider: &str,
        helper: crate::upstream_auth::ApiKeyHelper,
    ) -> Result<(), CliError> {
        let mut config = crate::configuration::GatewayConfig::default();
        match provider {
            "openai" => config.openai_api_key_helper = Some(helper.clone()),
            "anthropic" => config.anthropic_api_key_helper = Some(helper.clone()),
            _ => return Err(CliError::Config("unknown credential provider".into())),
        }
        crate::upstream_auth::validate_config(&config)?;
        let mut table = InlineTable::new();
        table.insert("command", Value::from(helper.command));
        table.insert(
            "args",
            Value::Array(helper.args.into_iter().collect::<Array>()),
        );
        let timeout = i64::try_from(helper.timeout_ms)
            .map_err(|_| CliError::Config("helper timeout_ms is too large".into()))?;
        table.insert("timeout_ms", Value::from(timeout));
        if let Some(scheme) = helper.auth_scheme {
            table.insert(
                "auth_scheme",
                Value::from(match scheme {
                    crate::upstream_auth::AuthScheme::Bearer => "bearer",
                    crate::upstream_auth::AuthScheme::ApiKey => "api_key",
                }),
            );
        }
        self.set_credential(
            &format!("{provider}_api_key_helper"),
            Value::InlineTable(table),
        )
    }

    fn set_credential(&mut self, key: &str, new_value: Value) -> Result<(), CliError> {
        let provider = key.split('_').next().unwrap_or_default();
        let upstream = self.section_mut("upstream")?;
        for suffix in ["auth_header", "api_key", "api_key_helper"] {
            let other = format!("{provider}_{suffix}");
            if other != key {
                upstream.remove(&other);
            }
        }
        upstream.insert(key, Item::Value(new_value));
        Ok(())
    }

    fn clear_key(&mut self, section: &str, key: &str) -> Result<(), CliError> {
        if self.section(section).is_none() {
            // Missing sections are already clear; malformed sections still report an error.
            let mut table = self.document.as_item();
            for part in section.split('.') {
                match table.get(part) {
                    Some(item) => table = item,
                    None => return Ok(()),
                }
            }
        }
        let table = self.section_mut(section)?;
        table.remove(key);
        if table.is_empty() {
            if let Some((parent, child)) = section.rsplit_once('.') {
                self.clear_key(parent, child)?;
            } else {
                self.document.remove(section);
            }
        }
        Ok(())
    }

    fn sinks(&self) -> Option<&Item> {
        self.section(self.logging_section)?.get("sinks")
    }

    fn sinks_mut(&mut self) -> Result<&mut Item, CliError> {
        let section = self.logging_section;
        let logging = self.section_mut(section)?;
        if logging.get("sinks").is_none() {
            logging.insert("sinks", Item::ArrayOfTables(ArrayOfTables::new()));
        }
        let sinks = logging.get_mut("sinks").expect("sinks initialized above");
        if sinks.is_array_of_tables()
            || sinks
                .as_array()
                .is_some_and(|array| array.iter().all(Value::is_inline_table))
        {
            Ok(sinks)
        } else {
            Err(CliError::Config(format!(
                "{section}.sinks must be an array of tables before it can be edited"
            )))
        }
    }

    fn sink_count(&self) -> usize {
        match self.sinks() {
            Some(Item::ArrayOfTables(sinks)) => sinks.len(),
            Some(Item::Value(Value::Array(sinks))) => sinks.len(),
            _ => 0,
        }
    }

    fn sink_labels(&self) -> Vec<String> {
        (0..self.sink_count())
            .map(|index| {
                let path = self
                    .sink_string(index, "path")
                    .unwrap_or_else(|| "invalid path".into());
                format!("sink {} ({path})", index + 1)
            })
            .collect()
    }

    fn sink(&self, index: usize) -> Option<&dyn TableLike> {
        match self.sinks()? {
            Item::ArrayOfTables(sinks) => sinks.get(index).map(|sink| sink as &dyn TableLike),
            Item::Value(Value::Array(sinks)) => sinks
                .get(index)?
                .as_inline_table()
                .map(|sink| sink as &dyn TableLike),
            _ => None,
        }
    }

    fn sink_mut(&mut self, index: usize) -> Result<&mut dyn TableLike, CliError> {
        let sink: Option<&mut dyn TableLike> = match self.sinks_mut()? {
            Item::ArrayOfTables(sinks) => {
                sinks.get_mut(index).map(|sink| sink as &mut dyn TableLike)
            }
            Item::Value(Value::Array(sinks)) => sinks
                .get_mut(index)
                .and_then(Value::as_inline_table_mut)
                .map(|sink| sink as &mut dyn TableLike),
            _ => unreachable!("sinks_mut checks the array shape"),
        };
        sink.ok_or_else(|| CliError::Config(format!("logging sink {} does not exist", index + 1)))
    }

    fn add_sink(&mut self, path: String) -> Result<(), CliError> {
        let mut sink = Table::new();
        sink["path"] = value(path);
        match self.sinks_mut()? {
            Item::ArrayOfTables(sinks) => sinks.push(sink),
            Item::Value(Value::Array(sinks)) => sinks.push(sink.into_inline_table()),
            _ => unreachable!("sinks_mut checks the array shape"),
        }
        Ok(())
    }

    fn remove_sink(&mut self, index: usize) -> Result<(), CliError> {
        if index >= self.sink_count() {
            return Err(CliError::Config(format!(
                "logging sink {} does not exist",
                index + 1
            )));
        }
        let empty = match self.sinks_mut()? {
            Item::ArrayOfTables(sinks) => {
                sinks.remove(index);
                sinks.is_empty()
            }
            Item::Value(Value::Array(sinks)) => {
                sinks.remove(index);
                sinks.is_empty()
            }
            _ => unreachable!("sinks_mut checks the array shape"),
        };
        if empty {
            self.clear_key(self.logging_section, "sinks")?;
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
        self.sink_mut(index)?.insert(key, value(new_value));
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
        self.sink_mut(index)?
            .insert("queue_capacity", value(capacity));
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
        sink.insert("max_file_size_bytes", value(max_size));
        sink.insert("retained_files", value(retained_i64));
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
        self.section_mut(section)?
            .insert(key, Item::Value(new_value));
        Ok(())
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
