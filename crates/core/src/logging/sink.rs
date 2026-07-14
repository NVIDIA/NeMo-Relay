// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logger and sink assembly, including sink path resolution.

use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use spdlog::sink::{AsyncPoolSink, FileSink, OverflowPolicy, StdStreamSink};
use spdlog::terminal_style::StyleMode;
use spdlog::{Level, LevelFilter, Logger, ThreadPool};

use super::config::{LogLevel, LogSinkConfig, LoggingConfig};
use super::format::RelayFormatter;
use crate::error::{FlowError, Result};

pub(crate) fn build_logger(
    config: &LoggingConfig,
    root_relay_id: String,
) -> Result<(Arc<Logger>, Vec<Arc<ThreadPool>>)> {
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
            FlowError::InvalidArgument(format!("failed to create stderr logging sink: {error}"))
        })?;
    sinks.push(stderr_sink);

    for sink in &config.sinks {
        let LogSinkConfig::File(file_sink) = sink;
        let resolved_path = resolve_log_path(&file_sink.path)?;
        if resolved_paths
            .iter()
            .any(|existing| existing == &resolved_path)
        {
            return Err(FlowError::InvalidArgument(format!(
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
                FlowError::InvalidArgument(format!(
                    "failed to open logging sink {}: {error}",
                    resolved_path.display()
                ))
            })?;

        if file_sink.queue_capacity > config.max_queue_capacity {
            return Err(FlowError::InvalidArgument(format!(
                "logging sink queue_capacity {} exceeds maximum {}",
                file_sink.queue_capacity, config.max_queue_capacity
            )));
        }
        let capacity = NonZeroUsize::new(file_sink.queue_capacity).ok_or_else(|| {
            FlowError::InvalidArgument("logging sink queue_capacity must be greater than 0".into())
        })?;
        let mut pool_builder = ThreadPool::builder();
        let pool = pool_builder
            .capacity(capacity)
            .build_arc()
            .map_err(|error| {
                FlowError::InvalidArgument(format!(
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
                FlowError::InvalidArgument(format!(
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
        .map_err(|error| {
            FlowError::InvalidArgument(format!("failed to build logging runtime: {error}"))
        })?;

    if let Some(millis) = flush_interval_millis {
        logger.set_flush_period(Some(Duration::from_millis(millis)));
    }

    Ok((logger, thread_pools))
}

fn resolve_log_path(path: &Path) -> Result<PathBuf> {
    // Relative paths resolve against process CWD. Absolute paths are unchanged. No `~` or env
    // expansion.
    if path.as_os_str().is_empty() {
        return Err(FlowError::InvalidArgument(
            "logging sink path must not be empty".into(),
        ));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        let cwd = std::env::current_dir().map_err(|error| {
            FlowError::InvalidArgument(format!(
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

pub(super) fn log_level_filter(level: LogLevel) -> log::LevelFilter {
    match level {
        LogLevel::Error => log::LevelFilter::Error,
        LogLevel::Warn => log::LevelFilter::Warn,
        LogLevel::Info => log::LevelFilter::Info,
        LogLevel::Debug => log::LevelFilter::Debug,
        LogLevel::Trace => log::LevelFilter::Trace,
    }
}
