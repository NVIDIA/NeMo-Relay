// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operational log record rendering (human/JSONL).

use std::fmt::Write as _;

use chrono::{SecondsFormat, Utc};
use serde_json::{Map, Value, json};
use spdlog::formatter::{Formatter, FormatterContext};
use spdlog::{Level, Record, StringBuf};

use super::config::LogFormat;

#[derive(Clone)]
pub(super) struct RelayFormatter {
    pub(super) format: LogFormat,
    pub(super) root_relay_id: String,
}

impl Formatter for RelayFormatter {
    fn format(
        &self,
        record: &Record<'_>,
        dest: &mut StringBuf,
        _ctx: &mut FormatterContext<'_>,
    ) -> spdlog::Result<()> {
        let rendered = render_record(
            self.format,
            &self.root_relay_id,
            &system_time_to_rfc3339(record.time()),
            record.level(),
            record.logger_name().unwrap_or(""),
            record.payload(),
            &collect_fields(record),
        );
        dest.write_str(&rendered)
            .map_err(spdlog::Error::FormatRecord)?;
        Ok(())
    }
}

#[derive(Default)]
struct CollectedFields {
    event_name: Option<String>,
    fields: Map<String, Value>,
}

fn collect_fields(record: &Record<'_>) -> CollectedFields {
    let mut collected = CollectedFields::default();
    for (key, value) in record.key_values() {
        let name = key.as_str();
        let rendered = format!("{value}");
        match name {
            "event" => collected.event_name = Some(rendered),
            "message" => {
                // Message already lives in record.payload(); ignore KV duplicates.
            }
            _ => {
                collected.fields.insert(name.to_owned(), json!(rendered));
            }
        }
    }
    collected
}

fn render_record(
    format: LogFormat,
    root_relay_id: &str,
    timestamp: &str,
    level: Level,
    target: &str,
    message: &str,
    fields: &CollectedFields,
) -> String {
    let event_name = fields.event_name.as_deref().unwrap_or("");
    match format {
        LogFormat::Jsonl => {
            let mut body = Map::new();
            body.insert("timestamp".into(), json!(timestamp));
            body.insert("level".into(), json!(json_level_name(level)));
            body.insert("root_relay_id".into(), json!(root_relay_id));
            body.insert("target".into(), json!(target));
            body.insert("event".into(), json!(event_name));
            body.insert("message".into(), json!(message));
            body.insert("fields".into(), Value::Object(fields.fields.clone()));
            format!(
                "{}\n",
                serde_json::to_string(&body).unwrap_or_else(|_| {
                    r#"{"timestamp":"","level":"error","root_relay_id":"","target":"","event":"format_error","message":"failed to serialize log record","fields":{}}"#.into()
                })
            )
        }
        LogFormat::Human => {
            let root_short = short_root_id(root_relay_id);
            let mut line = format!(
                "{timestamp} {} root={root_short} target={target}",
                human_level_name(level)
            );
            if !event_name.is_empty() {
                line.push_str(&format!(" event={event_name}"));
            }
            for (key, value) in &fields.fields {
                line.push_str(&format!(" {key}={}", format_human_value(value)));
            }
            if !message.is_empty() {
                line.push(' ');
                line.push_str(message);
            }
            line.push('\n');
            line
        }
    }
}

fn system_time_to_rfc3339(time: std::time::SystemTime) -> String {
    let datetime: chrono::DateTime<Utc> = time.into();
    datetime.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn json_level_name(level: Level) -> &'static str {
    match level {
        Level::Critical | Level::Error => "error",
        Level::Warn => "warn",
        Level::Info => "info",
        Level::Debug => "debug",
        Level::Trace => "trace",
    }
}

fn human_level_name(level: Level) -> &'static str {
    match level {
        Level::Critical | Level::Error => "ERROR",
        Level::Warn => "WARN",
        Level::Info => "INFO",
        Level::Debug => "DEBUG",
        Level::Trace => "TRACE",
    }
}

fn short_root_id(root_relay_id: &str) -> &str {
    root_relay_id.split('-').next().unwrap_or(root_relay_id)
}

fn format_human_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Renders an event through the production [`render_record`] path with a fixed timestamp, so
/// formatter tests exercise the real rendering code rather than a parallel copy.
#[cfg(test)]
pub(crate) fn format_event_for_test(
    format: LogFormat,
    root_relay_id: &str,
    level: Level,
    target: &str,
    event_name: Option<&str>,
    message: &str,
    extra_fields: &[(&str, &str)],
) -> String {
    let mut fields = CollectedFields::default();
    if let Some(event_name) = event_name {
        fields.event_name = Some(event_name.to_owned());
    }
    for (name, value) in extra_fields {
        fields.fields.insert((*name).to_owned(), json!(*value));
    }
    render_record(
        format,
        root_relay_id,
        "2026-07-10T14:22:31.123Z",
        level,
        target,
        message,
        &fields,
    )
}
