// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operational process logging for Relay (stderr + optional file sinks).
//!
//! Call sites emit through the `log` facade (`log::info!`, …). This module owns the
//! `spdlog-rs` backend, `LogCrateProxy` installation, formatters, and sink lifetime.

use std::fmt::Write as _;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{SecondsFormat, Utc};
use serde_json::{Map, Value, json};
use spdlog::formatter::{Formatter, FormatterContext};
use spdlog::sink::{AsyncPoolSink, FileSink, OverflowPolicy, Sink, StdStreamSink};
use spdlog::terminal_style::StyleMode;
use spdlog::{Level, LevelFilter, Logger, Record, StringBuf, ThreadPool};
use uuid::Uuid;

use crate::config::{LogFormat, LogLevel, LogSinkConfig, LoggingConfig};
use crate::error::CliError;

/// Owns logging resources that must remain alive for the process / run lifetime.
pub(crate) struct LoggingRuntime {
    root_relay_id: String,
    pub(crate) logger: Arc<Logger>,
    /// Keeps per-sink async thread pools alive until shutdown.
    _thread_pools: Vec<Arc<ThreadPool>>,
}

impl LoggingRuntime {
    /// Returns the process root Relay ID attached to operational records after initialization.
    #[allow(dead_code)]
    pub(crate) fn root_relay_id(&self) -> &str {
        &self.root_relay_id
    }

    /// Flushes buffered sinks.
    #[allow(dead_code)]
    pub(crate) fn shutdown(self) {
        drop(self);
    }
}

impl Drop for LoggingRuntime {
    fn drop(&mut self) {
        // Periodic flusher must stop before exit flush so it cannot race teardown.
        self.logger.set_flush_period(None);
        // `Logger::flush` only enqueues AsyncPoolSink work. `flush_on_exit` destroys the
        // pool (draining pending tasks) then flushes the underlying FileSink on this thread.
        // LogCrateProxy loggers are outside spdlog's atexit default-logger path, so we must
        // do this explicitly while `_thread_pools` is still alive.
        for sink in self.logger.sinks() {
            if let Err(error) = Sink::flush_on_exit(sink.as_ref()) {
                let _ = writeln!(
                    io::stderr(),
                    "nemo-relay: logging shutdown flush failed: {error}"
                );
            }
        }
    }
}

/// Installs process-wide operational logging from resolved config.
///
/// Stderr is always enabled. Explicit file sinks fail startup if they cannot be opened.
///
/// Verbosity is owned by `[logging].level` (a minimum severity threshold). Call sites may emit any
/// level; less-severe records are discarded. `RUST_LOG` is intentionally not consulted.
pub(crate) fn init_logging(config: &LoggingConfig) -> Result<LoggingRuntime, CliError> {
    let root_relay_id = Uuid::now_v7().to_string();
    let (logger, thread_pools) = build_logger(config, root_relay_id.clone())?;

    // Install once per process. Subsequent calls (tests / re-entry) reuse the proxy and swap
    // the receiver logger.
    let _ = spdlog::init_log_crate_proxy();
    spdlog::log_crate_proxy().set_logger(Some(Arc::clone(&logger)));
    // No env-filter / RUST_LOG: Relay config alone owns operational verbosity.
    spdlog::log_crate_proxy().set_filter(None);
    // `log::set_max_level` is poorly named: it installs the minimum severity threshold from
    // config (e.g. Info keeps error/warn/info and drops debug/trace).
    log::set_max_level(log_level_filter(config.level));

    Ok(LoggingRuntime {
        root_relay_id,
        logger,
        _thread_pools: thread_pools,
    })
}

fn build_logger(
    config: &LoggingConfig,
    root_relay_id: String,
) -> Result<(Arc<Logger>, Vec<Arc<ThreadPool>>), CliError> {
    let mut sinks: Vec<Arc<dyn spdlog::sink::Sink>> = Vec::new();
    let mut thread_pools = Vec::new();
    let mut resolved_paths: Vec<PathBuf> = Vec::new();
    let mut flush_interval_millis: Option<u64> = None;

    let stderr_sink = StdStreamSink::builder()
        .stderr()
        .style_mode(StyleMode::Never)
        .formatter(RelayFormatter {
            format: config.stderr_format,
            root_relay_id: root_relay_id.clone(),
        })
        .level_filter(spdlog_level_filter(config.level))
        .error_handler(stderr_error_handler("stderr"))
        .build_arc()
        .map_err(|error| {
            CliError::Config(format!("failed to create stderr logging sink: {error}"))
        })?;
    sinks.push(stderr_sink);

    for sink in &config.sinks {
        let LogSinkConfig::File(file_sink) = sink;
        let resolved_path = resolve_log_path(&file_sink.path)?;
        if resolved_paths
            .iter()
            .any(|existing| existing == &resolved_path)
        {
            return Err(CliError::Config(format!(
                "duplicate logging sink path {}",
                resolved_path.display()
            )));
        }
        resolved_paths.push(resolved_path.clone());

        // FileSink performs the real open/append. AsyncPoolSink is spdlog's stock bounded queue +
        // worker pool in front of that file so hot paths enqueue instead of blocking on disk I/O.
        // Overflow drops incoming records so a stuck disk cannot stall the process.
        let file = FileSink::builder()
            .path(&resolved_path)
            .truncate(false)
            .formatter(RelayFormatter {
                format: file_sink.format,
                root_relay_id: root_relay_id.clone(),
            })
            .level_filter(spdlog_level_filter(file_sink.level))
            .error_handler(stderr_error_handler(&resolved_path.display().to_string()))
            .build_arc()
            .map_err(|error| {
                CliError::Config(format!(
                    "failed to open logging sink {}: {error}",
                    resolved_path.display()
                ))
            })?;

        let capacity = NonZeroUsize::new(file_sink.queue_capacity).ok_or_else(|| {
            CliError::Config("logging sink queue_capacity must be greater than 0".into())
        })?;
        let mut pool_builder = ThreadPool::builder();
        let pool = pool_builder
            .capacity(capacity)
            .build_arc()
            .map_err(|error| {
                CliError::Config(format!(
                    "failed to create logging thread pool for {}: {error}",
                    resolved_path.display()
                ))
            })?;

        let async_sink = AsyncPoolSink::builder()
            .sink(file)
            .thread_pool(Arc::clone(&pool))
            .overflow_policy(OverflowPolicy::DropIncoming)
            .level_filter(spdlog_level_filter(file_sink.level))
            .error_handler(stderr_error_handler(&resolved_path.display().to_string()))
            .build_arc()
            .map_err(|error| {
                CliError::Config(format!(
                    "failed to create async logging sink for {}: {error}",
                    resolved_path.display()
                ))
            })?;

        thread_pools.push(pool);
        sinks.push(async_sink);

        if file_sink.flush_interval_millis > 0 {
            flush_interval_millis = Some(
                flush_interval_millis
                    .map(|current| current.min(file_sink.flush_interval_millis))
                    .unwrap_or(file_sink.flush_interval_millis),
            );
        }
    }

    // Leave the logger unnamed so LogCrateProxy maps log::target into record.logger_name().
    let logger = Logger::builder()
        .level_filter(spdlog_level_filter(config.level))
        .sinks(sinks)
        .build_arc()
        .map_err(|error| CliError::Config(format!("failed to build logging runtime: {error}")))?;

    if let Some(millis) = flush_interval_millis {
        logger.set_flush_period(Some(Duration::from_millis(millis)));
    }

    Ok((logger, thread_pools))
}

fn resolve_log_path(path: &Path) -> Result<PathBuf, CliError> {
    // Least astonishment for CLIs: relative paths are relative to process CWD (same as opening
    // `./foo` yourself). Absolute paths are unchanged. No `~` or env expansion.
    if path.as_os_str().is_empty() {
        return Err(CliError::Config(
            "logging sink path must not be empty".into(),
        ));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        let cwd = std::env::current_dir().map_err(|error| {
            CliError::Config(format!(
                "failed to resolve relative logging path {}: {error}",
                path.display()
            ))
        })?;
        cwd.join(path)
    };
    Ok(logging_path_identity(&absolute))
}

/// Builds a stable path identity for duplicate-sink detection.
///
/// Prefer filesystem canonicalization when the path (or its parent) exists so `./a` and `a`
/// collapse, and symlink parents resolve. When nothing on disk exists yet, normalize `.` / `..`
/// components only. Distinct basenames that are symlink aliases remain distinct until the
/// destination exists and can be canonicalized.
fn logging_path_identity(path: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            let file_name = path.file_name().unwrap_or_default();
            if let Ok(canonical_parent) = std::fs::canonicalize(parent) {
                return canonical_parent.join(file_name);
            }
            normalize_path_components(parent).join(file_name)
        }
        _ => normalize_path_components(path),
    }
}

fn normalize_path_components(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn stderr_error_handler(sink_label: &str) -> impl Fn(spdlog::Error) + Send + Sync + 'static {
    let sink_label = sink_label.to_owned();
    move |error| {
        let _ = writeln!(
            io::stderr(),
            "nemo-relay: logging sink error ({sink_label}): {error}"
        );
    }
}

fn spdlog_level_filter(level: LogLevel) -> LevelFilter {
    LevelFilter::MoreSevereEqual(spdlog_level(level))
}

fn spdlog_level(level: LogLevel) -> Level {
    match level {
        LogLevel::Error => Level::Error,
        LogLevel::Warn => Level::Warn,
        LogLevel::Info => Level::Info,
        LogLevel::Debug => Level::Debug,
        LogLevel::Trace => Level::Trace,
    }
}

fn log_level_filter(level: LogLevel) -> log::LevelFilter {
    match level {
        LogLevel::Error => log::LevelFilter::Error,
        LogLevel::Warn => log::LevelFilter::Warn,
        LogLevel::Info => log::LevelFilter::Info,
        LogLevel::Debug => log::LevelFilter::Debug,
        LogLevel::Trace => log::LevelFilter::Trace,
    }
}

#[derive(Clone)]
struct RelayFormatter {
    format: LogFormat,
    root_relay_id: String,
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
            record.time(),
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
            _ if is_sensitive_field_name(name) => {
                collected
                    .fields
                    .insert(name.to_owned(), json!("[redacted]"));
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
    time: std::time::SystemTime,
    level: Level,
    target: &str,
    message: &str,
    fields: &CollectedFields,
) -> String {
    let timestamp = system_time_to_rfc3339(time);
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
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        Value::Null => "null".into(),
        other => other.to_string(),
    }
}

fn is_sensitive_field_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "password"
            | "passwd"
            | "secret"
            | "token"
            | "api_key"
            | "apikey"
            | "access_token"
            | "refresh_token"
            | "authorization"
            | "auth"
            | "openai_api_key"
            | "anthropic_api_key"
    )
}

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
        if is_sensitive_field_name(name) {
            fields
                .fields
                .insert((*name).to_owned(), json!("[redacted]"));
        } else {
            fields.fields.insert((*name).to_owned(), json!(*value));
        }
    }
    // Fixed timestamp for deterministic formatter assertions.
    let timestamp = "2026-07-10T14:22:31.123Z";
    match format {
        LogFormat::Jsonl => {
            let mut body = Map::new();
            body.insert("timestamp".into(), json!(timestamp));
            body.insert("level".into(), json!(json_level_name(level)));
            body.insert("root_relay_id".into(), json!(root_relay_id));
            body.insert("target".into(), json!(target));
            body.insert(
                "event".into(),
                json!(fields.event_name.as_deref().unwrap_or("")),
            );
            body.insert("message".into(), json!(message));
            body.insert("fields".into(), Value::Object(fields.fields));
            format!("{}\n", serde_json::to_string(&body).expect("json"))
        }
        LogFormat::Human => {
            let root_short = short_root_id(root_relay_id);
            let mut line = format!(
                "{timestamp} {} root={root_short} target={target}",
                human_level_name(level)
            );
            if let Some(event_name) = fields.event_name.as_deref()
                && !event_name.is_empty()
            {
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

/// Builds sinks/logger without installing the process-global `log` facade (unit tests).
#[cfg(test)]
pub(crate) fn build_logger_for_test(config: &LoggingConfig) -> Result<LoggingRuntime, CliError> {
    let root_relay_id = Uuid::now_v7().to_string();
    let (logger, thread_pools) = build_logger(config, root_relay_id.clone())?;
    Ok(LoggingRuntime {
        root_relay_id,
        logger,
        _thread_pools: thread_pools,
    })
}

#[cfg(test)]
#[path = "../tests/coverage/logging_tests.rs"]
mod tests;
