// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TOML input, validation, and default inheritance for operational logging.

use std::path::PathBuf;

use nemo_relay::error::FlowError;
use nemo_relay::logging::{
    DEFAULT_FILE_FLUSH_INTERVAL_MILLIS, DEFAULT_FILE_SINK_QUEUE_ENTRIES, FileLogSinkConfig,
    LogFormat, LogLevel, LogSinkConfig, LoggingConfig, MAX_FILE_SINK_QUEUE_ENTRIES,
};
use serde::Deserialize;

use super::CliError;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FileLoggingConfig {
    level: Option<String>,
    stderr_format: Option<String>,
    /// Optional advanced: periodic flush interval in ms applied to all file sinks (default
    /// [`DEFAULT_FILE_FLUSH_INTERVAL_MILLIS`]). `0` means flush only on shutdown.
    flush_interval_millis: Option<u64>,
    #[serde(default)]
    sinks: Vec<RawFileLogSinkConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFileLogSinkConfig {
    /// Required. Every `[[logging.sinks]]` entry is a file sink.
    path: Option<PathBuf>,
    level: Option<String>,
    format: Option<String>,
    /// Optional advanced: pending async queue entries per file sink (default
    /// [`DEFAULT_FILE_SINK_QUEUE_ENTRIES`]).
    queue_capacity: Option<usize>,
}

pub(super) fn apply_file_logging_config(
    logging: &mut LoggingConfig,
    config: Option<FileLoggingConfig>,
) -> Result<(), CliError> {
    let Some(config) = config else {
        return Ok(());
    };
    if let Some(level) = config.level.as_deref() {
        logging.level = LogLevel::parse(level).map_err(logging_parse_error)?;
    }
    if let Some(stderr_format) = config.stderr_format.as_deref() {
        logging.stderr_format = LogFormat::parse(stderr_format).map_err(logging_parse_error)?;
    }
    logging.flush_interval_millis = config
        .flush_interval_millis
        .unwrap_or(DEFAULT_FILE_FLUSH_INTERVAL_MILLIS);
    if !config.sinks.is_empty() {
        let default_sink_level = logging.level;
        logging.sinks = config
            .sinks
            .into_iter()
            .map(|sink| parse_file_log_sink(sink, default_sink_level))
            .collect::<Result<Vec<_>, _>>()?;
    }
    Ok(())
}

fn parse_file_log_sink(
    config: RawFileLogSinkConfig,
    default_level: LogLevel,
) -> Result<LogSinkConfig, CliError> {
    let path = config
        .path
        .ok_or_else(|| CliError::Config("logging sink requires path".into()))?;
    if path.as_os_str().is_empty() {
        return Err(CliError::Config(
            "logging sink path must not be empty".into(),
        ));
    }
    let level = match config.level.as_deref() {
        Some(raw) => LogLevel::parse(raw).map_err(logging_parse_error)?,
        None => default_level,
    };
    let format = match config.format.as_deref() {
        Some(raw) => LogFormat::parse(raw).map_err(logging_parse_error)?,
        None => LogFormat::Jsonl,
    };
    let queue_capacity = match config.queue_capacity {
        Some(0) => {
            return Err(CliError::Config(
                "logging sink queue_capacity must be greater than 0".into(),
            ));
        }
        Some(capacity) if capacity > MAX_FILE_SINK_QUEUE_ENTRIES => {
            return Err(CliError::Config(format!(
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

fn logging_parse_error(error: FlowError) -> CliError {
    match error {
        FlowError::InvalidArgument(message) => CliError::Config(message),
        other => CliError::Flow(other),
    }
}
