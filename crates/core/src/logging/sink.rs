// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logger and sink assembly, including sink path resolution.

use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use spdlog::sink::{
    AsyncPoolSink, FileSink, OverflowPolicy, Sink, SinkPropAccess, StdStreamSink, WriteSink,
};
use spdlog::terminal_style::StyleMode;
use spdlog::{ErrorHandler, Level, LevelFilter, Logger, Record, ThreadPool};

use super::config::{
    FileLogSinkConfig, FileLogSinkFailurePolicy, LogLevel, LogSinkConfig, LoggingConfig,
    MAX_FILE_SINK_QUEUE_ENTRIES,
};
use super::diagnostics::{LoggingDiagnostics, SinkDiagnostics, file_open_failure_reason};
use super::format::RelayFormatter;
use super::rotation::{SizeRotatingFileWriter, rotated_log_path};
use crate::error::{FlowError, Result};

#[cfg(test)]
pub(crate) fn build_logger(
    config: &LoggingConfig,
    root_relay_id: String,
) -> Result<(Arc<Logger>, Vec<Arc<ThreadPool>>)> {
    build_logger_with_diagnostics(config, root_relay_id, None)
        .map(|built| (built.logger, built.thread_pools))
}

pub(super) struct BuiltLogger {
    pub logger: Arc<Logger>,
    pub thread_pools: Vec<Arc<ThreadPool>>,
    pub sink_indices: Vec<usize>,
    pub unavailable_files: Vec<(usize, io::ErrorKind)>,
}

pub(super) fn build_logger_with_diagnostics(
    config: &LoggingConfig,
    root_relay_id: String,
    diagnostics: Option<&LoggingDiagnostics>,
) -> Result<BuiltLogger> {
    // Validate every configuration and reserve every destination before opening any files.
    // An unavailable optional file must not hide invalid limits or path collisions.
    let resolved_paths = validate_file_sinks(config, diagnostics)?;
    let mut sinks: Vec<Arc<dyn spdlog::sink::Sink>> = Vec::new();
    let mut thread_pools = Vec::new();
    let mut sink_indices = Vec::new();
    let mut unavailable_files = Vec::new();

    if config.stderr_enabled {
        let diagnostic = diagnostics.and_then(LoggingDiagnostics::stderr);
        let initialization = SinkInitialization::new(diagnostic.clone());
        let error_handler: ErrorHandler = stderr_error_handler("stderr", diagnostic).into();
        let stderr_sink = StdStreamSink::builder()
            .stderr()
            .style_mode(StyleMode::Never)
            .formatter(RelayFormatter {
                format: config.stderr_format,
                root_relay_id: root_relay_id.clone(),
            })
            .level_filter(spdlog_level_filter(config.level))
            .error_handler(error_handler.clone())
            .build()
            .map_err(|error| {
                FlowError::InvalidArgument(format!("failed to create stderr logging sink: {error}"))
            })?;
        sinks.push(Arc::new(SinkErrorBoundary::new(stderr_sink, error_handler)));
        sink_indices.push(0);
        initialization.complete();
    }

    for (index, sink) in config.sinks.iter().enumerate() {
        let diagnostic = diagnostics.and_then(|diagnostics| diagnostics.file(index));
        let initialization = SinkInitialization::new(diagnostic.clone());
        let LogSinkConfig::File(file_sink) = sink;
        let resolved_path = &resolved_paths[index];
        let sink_label = resolved_path.display().to_string();
        let file = match open_file_sink(
            file_sink,
            resolved_path,
            &root_relay_id,
            diagnostic.clone(),
        ) {
            Ok(file) => file,
            Err(error) => {
                // Match structured filesystem failures only. Thread pool, async sink,
                // formatter, and logger failures are never covered by this policy.
                if let Some(reason) = file_open_error_kind(&error) {
                    initialization.open_failed(reason);
                    if file_sink.failure_policy == FileLogSinkFailurePolicy::Optional {
                        unavailable_files.push((index, reason));
                        // Available even before logging/diagnostics starts, or with stderr disabled.
                        let _ = writeln!(
                            io::stderr(),
                            "nemo-relay: optional logging sink logging.file.{index} disabled: {}; records for this sink are discarded until logging is reinitialized",
                            file_open_failure_reason(reason)
                        );
                        continue;
                    }
                }
                let kind = if file_sink.rotation.is_some() {
                    "rotating logging"
                } else {
                    "logging"
                };
                return Err(FlowError::InvalidArgument(format!(
                    "failed to open {kind} sink {}: {error}",
                    resolved_path.display()
                )));
            }
        };

        // Already validated, including programmatically constructed configurations.
        let capacity =
            NonZeroUsize::new(file_sink.queue_capacity).expect("validated queue capacity");
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

        let async_error_handler: ErrorHandler =
            async_sink_error_handler(&sink_label, diagnostic).into();
        let async_sink = AsyncPoolSink::builder()
            .sink(file)
            .thread_pool(Arc::clone(&pool))
            .overflow_policy(OverflowPolicy::DropIncoming)
            .level_filter(spdlog_level_filter(file_sink.level))
            .error_handler(async_error_handler.clone())
            .build()
            .map_err(|error| {
                FlowError::InvalidArgument(format!(
                    "failed to create async logging sink for {}: {error}",
                    resolved_path.display()
                ))
            })?;
        let async_sink = Arc::new(SinkErrorBoundary::new(async_sink, async_error_handler));

        thread_pools.push(pool);
        sinks.push(async_sink);
        sink_indices.push(index + usize::from(config.stderr_enabled));
        initialization.complete();
    }

    // Leave the logger unnamed so LogCrateProxy maps log::target into record.logger_name().
    let logger = Logger::builder()
        .level_filter(spdlog_level_filter(config.level))
        .sinks(sinks)
        .build_arc()
        .map_err(|error| {
            FlowError::InvalidArgument(format!("failed to build logging runtime: {error}"))
        })?;

    if !thread_pools.is_empty() && config.flush_interval_millis > 0 {
        logger.set_flush_period(Some(Duration::from_millis(config.flush_interval_millis)));
    }

    Ok(BuiltLogger {
        logger,
        thread_pools,
        sink_indices,
        unavailable_files,
    })
}

fn validate_file_sinks(
    config: &LoggingConfig,
    diagnostics: Option<&LoggingDiagnostics>,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    let mut reserved_paths: Vec<PathBuf> = Vec::new();
    for (index, sink) in config.sinks.iter().enumerate() {
        (|| {
            let LogSinkConfig::File(file) = sink;
            if file.queue_capacity == 0 {
                return Err(FlowError::InvalidArgument(
                    "logging sink queue_capacity must be greater than 0".into(),
                ));
            }
            if file.queue_capacity > MAX_FILE_SINK_QUEUE_ENTRIES {
                return Err(FlowError::InvalidArgument(format!(
                    "logging sink queue_capacity {} exceeds maximum {MAX_FILE_SINK_QUEUE_ENTRIES} entries per file sink",
                    file.queue_capacity
                )));
            }
            let path = resolve_log_path(&file.path)?;
            if paths.contains(&path) {
                return Err(FlowError::InvalidArgument(format!(
                    "duplicate logging sink path {}",
                    path.display()
                )));
            }
            let candidates = reserved_sink_paths(&path, file.rotation);
            if let Some(collision) = candidates
                .iter()
                .find(|candidate| reserved_paths.contains(candidate))
            {
                return Err(FlowError::InvalidArgument(format!(
                    "logging sink path conflicts with another active or rotated file: {}",
                    collision.display()
                )));
            }
            paths.push(path);
            reserved_paths.extend(candidates);
            Ok(())
        })()
        .inspect_err(|_| {
            if let Some(diagnostic) = diagnostics.and_then(|diagnostics| diagnostics.file(index)) {
                diagnostic.initialization_failed();
            }
        })?;
    }
    Ok(paths)
}

fn open_file_sink(
    config: &FileLogSinkConfig,
    path: &Path,
    root_relay_id: &str,
    diagnostic: Option<SinkDiagnostics>,
) -> spdlog::Result<Arc<dyn Sink>> {
    let formatter = RelayFormatter {
        format: config.format,
        root_relay_id: root_relay_id.to_owned(),
    };
    let error_handler = stderr_error_handler(&path.display().to_string(), diagnostic);
    match config.rotation {
        None => Ok(FileSink::builder()
            .path(path)
            .truncate(false)
            .formatter(formatter)
            .level_filter(spdlog_level_filter(config.level))
            .error_handler(error_handler)
            .build_arc()?),
        Some(rotation) => {
            let writer = SizeRotatingFileWriter::new(
                path.to_owned(),
                rotation.max_file_size_bytes(),
                rotation.retained_files(),
            )
            .map_err(spdlog::Error::OpenFile)?;
            Ok(WriteSink::builder()
                .target(writer)
                .formatter(formatter)
                .level_filter(spdlog_level_filter(config.level))
                .error_handler(error_handler)
                .build_arc()?)
        }
    }
}

fn file_open_error_kind(error: &spdlog::Error) -> Option<io::ErrorKind> {
    match error {
        spdlog::Error::CreateDirectory(error)
        | spdlog::Error::OpenFile(error)
        | spdlog::Error::QueryFileMetadata(error) => Some(error.kind()),
        _ => None,
    }
}

/// Records early returns during sink construction before a logger is installed.
struct SinkInitialization {
    diagnostics: Option<SinkDiagnostics>,
    recorded: bool,
}

impl SinkInitialization {
    fn new(diagnostics: Option<SinkDiagnostics>) -> Self {
        Self {
            diagnostics,
            recorded: false,
        }
    }

    fn open_failed(mut self, reason: io::ErrorKind) {
        self.recorded = true;
        if let Some(diagnostics) = &self.diagnostics {
            diagnostics.open_failed(reason);
        }
    }

    fn complete(mut self) {
        self.recorded = true;
        if let Some(diagnostics) = &self.diagnostics {
            diagnostics.initialized();
        }
    }
}

impl Drop for SinkInitialization {
    fn drop(&mut self) {
        if !self.recorded
            && let Some(diagnostics) = &self.diagnostics
        {
            diagnostics.initialization_failed();
        }
    }
}

/// Routes returned sink errors through the sink-specific handler instead of the logger handler.
struct SinkErrorBoundary<S: Sink> {
    inner: S,
    error_handler: RwLock<ErrorHandler>,
}

impl<S: Sink> SinkErrorBoundary<S> {
    fn new(inner: S, error_handler: ErrorHandler) -> Self {
        Self {
            inner,
            error_handler: RwLock::new(error_handler),
        }
    }

    fn report_immediate_error(&self, result: spdlog::Result<()>) -> spdlog::Result<()> {
        if let Err(error) = result {
            let error_handler = self
                .error_handler
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            error_handler.call(error);
        }
        Ok(())
    }
}

impl<S: Sink> SinkPropAccess for SinkErrorBoundary<S> {
    fn level_filter(&self) -> LevelFilter {
        self.inner.level_filter()
    }

    fn set_level_filter(&self, level_filter: LevelFilter) {
        self.inner.set_level_filter(level_filter);
    }

    fn set_formatter(&self, formatter: Box<dyn spdlog::formatter::Formatter>) {
        self.inner.set_formatter(formatter);
    }

    fn set_error_handler(&self, error_handler: ErrorHandler) {
        self.inner.set_error_handler(error_handler.clone());
        *self
            .error_handler
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = error_handler;
    }
}

impl<S: Sink> Sink for SinkErrorBoundary<S> {
    fn log(&self, record: &Record) -> spdlog::Result<()> {
        self.report_immediate_error(self.inner.log(record))
    }

    fn flush(&self) -> spdlog::Result<()> {
        self.report_immediate_error(self.inner.flush())
    }

    fn flush_on_exit(&self) -> spdlog::Result<()> {
        // Preserve final flush and drain errors for `LoggingRuntime`.
        self.inner.flush_on_exit()
    }
}

fn reserved_sink_paths(
    resolved_path: &Path,
    rotation: Option<super::config::FileLogRotationConfig>,
) -> Vec<PathBuf> {
    let mut paths = vec![resolved_path.to_path_buf()];
    if let Some(rotation) = rotation {
        paths.reserve(rotation.retained_files());
        for index in 1..=rotation.retained_files() {
            paths.push(logging_path_identity(&rotated_log_path(
                resolved_path,
                index,
            )));
        }
    }
    paths
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

fn stderr_error_handler(
    sink_label: &str,
    diagnostics: Option<SinkDiagnostics>,
) -> impl Fn(spdlog::Error) + Send + Sync + 'static {
    let sink_label = sink_label.to_owned();
    move |error| {
        if let Some(diagnostics) = &diagnostics {
            diagnostics.runtime_error(&error);
        }
        let _ = writeln!(
            io::stderr(),
            "nemo-relay: logging sink error ({sink_label}): {error}"
        );
    }
}

/// Minimum spacing between queue-full notices written to stderr.
const DROP_REPORT_INTERVAL_MILLIS: u64 = 1000;

/// Rate-limits async-sink queue-full notices so a full queue cannot flood or block stderr.
///
/// `DropIncoming` reports a queue-full error per dropped record on the enqueue (caller) thread.
/// Writing each one synchronously would defeat the non-blocking queue during a burst, so Relay
/// reports at most one notice per [`DROP_REPORT_INTERVAL_MILLIS`].
struct DropNoticeRateLimiter {
    last_report_millis: AtomicU64,
}

impl DropNoticeRateLimiter {
    fn new() -> Self {
        Self {
            last_report_millis: AtomicU64::new(0),
        }
    }

    fn should_report(&self, now_millis: u64) -> bool {
        loop {
            let last = self.last_report_millis.load(Ordering::Relaxed);
            if now_millis.saturating_sub(last) < DROP_REPORT_INTERVAL_MILLIS {
                return false;
            }
            if self
                .last_report_millis
                .compare_exchange_weak(last, now_millis, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

fn async_sink_error_handler(
    sink_label: &str,
    diagnostics: Option<SinkDiagnostics>,
) -> impl Fn(spdlog::Error) + Send + Sync + 'static {
    let sink_label = sink_label.to_owned();
    let dropped_record_rate_limiter = DropNoticeRateLimiter::new();
    let dropped_flush_rate_limiter = DropNoticeRateLimiter::new();
    move |error| {
        // Retain every loss while the independent emergency notice stays rate-limited.
        // Sink failures never project back into logging, including errors on async workers.
        if let Some(diagnostics) = &diagnostics {
            diagnostics.runtime_error(&error);
        }
        // A full queue can reject records and periodic flush requests at producer speed. Rate-limit
        // them separately so neither failure class can flood stderr or hide the other. Other errors
        // (disconnected channel and worker-side write failures) are rare and reported immediately.
        match &error {
            spdlog::Error::SendToChannel(
                spdlog::error::SendToChannelError::Full,
                spdlog::error::SendToChannelErrorDropped::Record(_),
            ) => {
                if dropped_record_rate_limiter.should_report(now_millis()) {
                    let _ = writeln!(
                        io::stderr(),
                        "nemo-relay: logging sink ({sink_label}): records are being dropped \
                         because the queue is full; repeated notices are limited to once per \
                         {DROP_REPORT_INTERVAL_MILLIS}ms"
                    );
                }
            }
            spdlog::Error::SendToChannel(
                spdlog::error::SendToChannelError::Full,
                spdlog::error::SendToChannelErrorDropped::Flush,
            ) => {
                if dropped_flush_rate_limiter.should_report(now_millis()) {
                    let _ = writeln!(
                        io::stderr(),
                        "nemo-relay: logging sink ({sink_label}): flush requests are being \
                         dropped because the queue is full; repeated notices are limited to once \
                         per {DROP_REPORT_INTERVAL_MILLIS}ms"
                    );
                }
            }
            other => {
                let _ = writeln!(
                    io::stderr(),
                    "nemo-relay: logging sink error ({sink_label}): {other}"
                );
            }
        }
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

#[cfg(test)]
#[path = "../../tests/coverage/logging_sink_tests.rs"]
mod tests;
