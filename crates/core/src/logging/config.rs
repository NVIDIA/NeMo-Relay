// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolved operational logging configuration types and string parsing.

use std::path::PathBuf;

use crate::error::{FlowError, Result};

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

/// Operational logging configuration for [`init_logging`](super::init_logging).
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
