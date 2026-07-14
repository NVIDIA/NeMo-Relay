// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operational process logging for Relay (stderr + optional file sinks).
//!
//! Call sites emit through the `log` facade (`log::info!`, …). This module owns the
//! `spdlog-rs` backend, `LogCrateProxy` installation, formatters, and sink lifetime.

mod config;
mod format;
mod sink;

use std::io::{self, Write};
use std::sync::Arc;

use spdlog::sink::Sink;
use spdlog::{Logger, ThreadPool};
use uuid::Uuid;

use crate::error::Result;

pub use config::{
    DEFAULT_FILE_FLUSH_INTERVAL_MILLIS, DEFAULT_FILE_QUEUE_CAPACITY,
    DEFAULT_MAX_FILE_QUEUE_CAPACITY, FileLogSinkConfig, LogFormat, LogLevel, LogSinkConfig,
    LoggingConfig,
};
pub(crate) use sink::build_logger;
use sink::log_level_filter;

#[cfg(test)]
pub(crate) use format::format_event_for_test;

/// Owns logging resources that must remain alive for the process / run lifetime.
///
/// When created by [`init_logging`], dropping this value flushes sinks and detaches this
/// logger from the process-global spdlog `log` proxy if it is still installed.
pub struct LoggingRuntime {
    root_relay_id: String,
    /// Underlying spdlog logger (also installed into the `log` facade by [`init_logging`]).
    pub(crate) logger: Arc<Logger>,
    /// Keeps per-sink async thread pools alive until shutdown.
    _thread_pools: Vec<Arc<ThreadPool>>,
}

impl LoggingRuntime {
    /// Returns the process root Relay ID attached to operational records after initialization.
    pub fn root_relay_id(&self) -> &str {
        &self.root_relay_id
    }

    /// Flushes buffered sinks and detaches global proxy wiring by dropping the runtime.
    pub fn shutdown(self) {
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

        // Detach only if we are still the installed receiver. A later `init_logging` may have
        // replaced us; do not clear that newer install.
        let detached = spdlog::log_crate_proxy().swap_logger(None);
        if let Some(logger) = detached
            && !Arc::ptr_eq(&logger, &self.logger)
        {
            spdlog::log_crate_proxy().set_logger(Some(logger));
        }
    }
}

/// Installs process-wide operational logging from resolved config.
///
/// Stderr is always enabled. Explicit file sinks fail startup if they cannot be opened.
///
/// Verbosity comes from [`LoggingConfig::level`]: records below that minimum severity are discarded.
/// Dropping the returned [`LoggingRuntime`] flushes sinks and detaches this logger from the
/// process-global `log` proxy when it is still installed.
pub fn init_logging(config: &LoggingConfig) -> Result<LoggingRuntime> {
    let root_relay_id = Uuid::now_v7().to_string();
    let (logger, thread_pools) = build_logger(config, root_relay_id.clone())?;

    // Install once per process. Subsequent calls (tests / re-entry) reuse the proxy and swap
    // the receiver logger.
    let _ = spdlog::init_log_crate_proxy();
    spdlog::log_crate_proxy().set_logger(Some(Arc::clone(&logger)));
    spdlog::log_crate_proxy().set_filter(None);
    log::set_max_level(log_level_filter(config.level));

    Ok(LoggingRuntime {
        root_relay_id,
        logger,
        _thread_pools: thread_pools,
    })
}

#[cfg(test)]
#[path = "../../tests/coverage/logging_tests.rs"]
mod tests;
