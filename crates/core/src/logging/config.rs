// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolved operational logging configuration types and source parsing.

use std::env::VarError;
use std::path::{Path, PathBuf};
use std::{env, fs};

use crate::error::{FlowError, Result};
use serde::Deserialize;

const LOG_LEVEL_ENV: &str = "NEMO_RELAY_LOG";
const LOG_STDERR_FORMAT_ENV: &str = "NEMO_RELAY_LOG_STDERR_FORMAT";
const LOG_CONFIG_PATH_ENV: &str = "NEMO_RELAY_LOG_CONFIG_PATH";

/// Default number of pending asynchronous queue entries per file sink when `queue_capacity` is
/// omitted.
pub const DEFAULT_FILE_SINK_QUEUE_ENTRIES: usize = 1024;

/// Default periodic flush interval when [`LoggingConfig::flush_interval_millis`] is omitted.
pub const DEFAULT_FILE_FLUSH_INTERVAL_MILLIS: u64 = 1000;

/// Fixed hard maximum number of pending asynchronous queue entries per file sink.
///
/// This is a non-configurable safety limit, not the queue size itself. The async queue
/// preallocates every slot, so an oversized `queue_capacity` can panic the process at startup;
/// configuration above this bound is rejected with a config error. It cannot be raised.
pub const MAX_FILE_SINK_QUEUE_ENTRIES: usize = 8_192;

/// Operational logging configuration for [`LoggingRuntime::configure`](super::LoggingRuntime::configure).
///
/// `level` is the process-wide **minimum severity**: call sites may emit any level, but records
/// less severe than this threshold are discarded. Per-file sinks may raise their own minimum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggingConfig {
    /// Minimum severity for operational logs.
    pub level: LogLevel,
    /// Encoding for the always-on stderr sink.
    pub stderr_format: LogFormat,
    /// Additional file sinks beyond stderr.
    pub sinks: Vec<LogSinkConfig>,
    /// Periodic flush cadence in milliseconds applied to all file sinks. `0` disables periodic
    /// flush (shutdown flush only). Defaults to [`DEFAULT_FILE_FLUSH_INTERVAL_MILLIS`].
    pub flush_interval_millis: u64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            stderr_format: LogFormat::Human,
            sinks: Vec::new(),
            flush_interval_millis: DEFAULT_FILE_FLUSH_INTERVAL_MILLIS,
        }
    }
}

impl LoggingConfig {
    /// Resolves logging configuration from the supported process environment.
    ///
    /// Returns `None` when no logging environment variables are present. Direct level and stderr
    /// format settings may be combined. `NEMO_RELAY_LOG_CONFIG_PATH` selects an absolute TOML file
    /// instead and is mutually exclusive with both direct settings.
    pub fn from_environment() -> Result<Option<Self>> {
        let level = environment_value(LOG_LEVEL_ENV)?;
        let stderr_format = environment_value(LOG_STDERR_FORMAT_ENV)?;
        let config_path = environment_value(LOG_CONFIG_PATH_ENV)?;

        if config_path.is_some() && (level.is_some() || stderr_format.is_some()) {
            return Err(FlowError::InvalidArgument(format!(
                "{LOG_CONFIG_PATH_ENV} cannot be combined with {LOG_LEVEL_ENV} or \
                 {LOG_STDERR_FORMAT_ENV}"
            )));
        }
        if let Some(path) = config_path {
            return Self::from_file_path(path).map(Some);
        }
        if level.is_none() && stderr_format.is_none() {
            return Ok(None);
        }

        let mut config = Self::default();
        if let Some(level) = level {
            config.level = LogLevel::parse(&level)?;
        }
        if let Some(stderr_format) = stderr_format {
            config.stderr_format = LogFormat::parse(&stderr_format)?;
        }
        Ok(Some(config))
    }

    /// Loads logging configuration from an absolute TOML file containing `[logging]`.
    pub fn from_file_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_absolute() {
            return Err(FlowError::InvalidArgument(format!(
                "logging configuration path must be absolute: {}",
                path.display()
            )));
        }
        if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
            return Err(FlowError::InvalidArgument(format!(
                "logging configuration path must identify a .toml file: {}",
                path.display()
            )));
        }

        let contents = fs::read_to_string(path).map_err(|error| {
            FlowError::InvalidArgument(format!(
                "failed to read logging configuration {}: {error}",
                path.display()
            ))
        })?;
        Self::from_toml_document(&contents).map_err(|error| {
            FlowError::InvalidArgument(format!(
                "invalid logging configuration in {}: {error}",
                path.display()
            ))
        })
    }

    /// Parses a TOML document containing Relay's existing `[logging]` schema.
    ///
    /// This is exposed for Relay frontends that already own TOML discovery and merging. Most
    /// callers should use [`Self::from_file_path`] or construct [`LoggingConfig`] directly.
    #[doc(hidden)]
    pub fn from_toml_document(contents: &str) -> Result<Self> {
        let document: LoggingDocument = toml::from_str(contents).map_err(|error| {
            FlowError::InvalidArgument(format!("invalid logging TOML: {error}"))
        })?;
        document
            .logging
            .ok_or_else(|| {
                FlowError::InvalidArgument(
                    "logging configuration requires a [logging] section".into(),
                )
            })?
            .resolve()
    }
}

/// Global / per-sink minimum severity for operational logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Error and above.
    Error,
    /// Warning and above.
    Warn,
    /// Informational and above.
    Info,
    /// Debug and above.
    Debug,
    /// Trace and above (most verbose).
    Trace,
}

impl LogLevel {
    /// Parses a config string into a [`LogLevel`].
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "warn" | "warning" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            other => Err(FlowError::InvalidArgument(format!(
                "invalid logging level '{other}'; expected error, warn, info, debug, or trace"
            ))),
        }
    }
}

/// Output encoding for an operational log sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Single-line human-readable text.
    Human,
    /// One JSON object per line.
    Jsonl,
}

impl LogFormat {
    /// Parses a config string into a [`LogFormat`].
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "human" => Ok(Self::Human),
            "jsonl" | "json" => Ok(Self::Jsonl),
            other => Err(FlowError::InvalidArgument(format!(
                "invalid logging format '{other}'; expected human or jsonl"
            ))),
        }
    }
}

/// Additional operational log sink beyond always-on stderr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogSinkConfig {
    /// Append-only file sink with an async delivery queue.
    File(FileLogSinkConfig),
}

/// File sink settings for non-blocking operational logging.
///
/// Relative `path` values are resolved against the process current working directory at sink open
/// time. Absolute paths are used as-is. `~` and env expansion are not applied.
///
/// File sinks write through an async queue so logging cannot stall the process on disk I/O.
/// `queue_capacity` is an optional advanced override; an omitted value uses
/// [`DEFAULT_FILE_SINK_QUEUE_ENTRIES`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileLogSinkConfig {
    /// Destination file path.
    pub path: PathBuf,
    /// Minimum severity for this file sink.
    pub level: LogLevel,
    /// Output encoding for this file sink.
    pub format: LogFormat,
    /// Maximum pending asynchronous queue entries for this file sink. Must be greater than 0 and
    /// at most [`MAX_FILE_SINK_QUEUE_ENTRIES`].
    pub queue_capacity: usize,
}

impl Default for FileLogSinkConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from(".nemo-relay/logs/relay.log.jsonl"),
            level: LogLevel::Info,
            format: LogFormat::Jsonl,
            queue_capacity: DEFAULT_FILE_SINK_QUEUE_ENTRIES,
        }
    }
}

#[derive(Debug, Deserialize)]
struct LoggingDocument {
    logging: Option<RawLoggingConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLoggingConfig {
    level: Option<String>,
    stderr_format: Option<String>,
    flush_interval_millis: Option<u64>,
    #[serde(default)]
    sinks: Vec<RawFileLogSinkConfig>,
}

impl RawLoggingConfig {
    fn resolve(self) -> Result<LoggingConfig> {
        let mut config = LoggingConfig::default();
        if let Some(level) = self.level {
            config.level = LogLevel::parse(&level)?;
        }
        if let Some(stderr_format) = self.stderr_format {
            config.stderr_format = LogFormat::parse(&stderr_format)?;
        }
        if let Some(flush_interval_millis) = self.flush_interval_millis {
            config.flush_interval_millis = flush_interval_millis;
        }
        if !self.sinks.is_empty() {
            config.sinks = self
                .sinks
                .into_iter()
                .map(|sink| sink.resolve(config.level))
                .collect::<Result<Vec<_>>>()?;
        }
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFileLogSinkConfig {
    path: Option<PathBuf>,
    level: Option<String>,
    format: Option<String>,
    queue_capacity: Option<usize>,
}

impl RawFileLogSinkConfig {
    fn resolve(self, default_level: LogLevel) -> Result<LogSinkConfig> {
        let path = self
            .path
            .ok_or_else(|| FlowError::InvalidArgument("logging sink requires path".into()))?;
        if path.as_os_str().is_empty() {
            return Err(FlowError::InvalidArgument(
                "logging sink path must not be empty".into(),
            ));
        }

        let level = self
            .level
            .as_deref()
            .map(LogLevel::parse)
            .transpose()?
            .unwrap_or(default_level);
        let format = self
            .format
            .as_deref()
            .map(LogFormat::parse)
            .transpose()?
            .unwrap_or(LogFormat::Jsonl);
        let queue_capacity = match self.queue_capacity {
            Some(0) => {
                return Err(FlowError::InvalidArgument(
                    "logging sink queue_capacity must be greater than 0".into(),
                ));
            }
            Some(capacity) if capacity > MAX_FILE_SINK_QUEUE_ENTRIES => {
                return Err(FlowError::InvalidArgument(format!(
                    "logging sink queue_capacity {capacity} exceeds maximum \
                     {MAX_FILE_SINK_QUEUE_ENTRIES} entries per file sink"
                )));
            }
            Some(capacity) => capacity,
            None => DEFAULT_FILE_SINK_QUEUE_ENTRIES,
        };

        Ok(LogSinkConfig::File(FileLogSinkConfig {
            path,
            level,
            format,
            queue_capacity,
        }))
    }
}

fn environment_value(name: &str) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) if value.is_empty() => Err(FlowError::InvalidArgument(format!(
            "{name} must not be empty when set"
        ))),
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(VarError::NotUnicode(_)) => Err(FlowError::InvalidArgument(format!(
            "{name} must contain valid Unicode"
        ))),
    }
}
