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
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use spdlog::sink::Sink;
use spdlog::{Logger, ThreadPool};
use uuid::Uuid;

use crate::error::{FlowError, Result};

pub use config::{
    DEFAULT_FILE_FLUSH_INTERVAL_MILLIS, DEFAULT_FILE_SINK_QUEUE_ENTRIES, FileLogSinkConfig,
    LogFormat, LogLevel, LogSinkConfig, LoggingConfig, MAX_FILE_SINK_QUEUE_ENTRIES,
};
pub(crate) use sink::build_logger;
use sink::log_level_filter;

#[cfg(test)]
pub(crate) use format::format_event_for_test;

static LOGGER_LIFECYCLE_LOCK: Mutex<()> = Mutex::new(());

fn lock_logger_lifecycle() -> MutexGuard<'static, ()> {
    LOGGER_LIFECYCLE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn log_crate_proxy_is_installed() -> bool {
    std::ptr::addr_eq(log::logger(), spdlog::log_crate_proxy() as &dyn log::Log)
}

fn install_log_crate_proxy() -> Result<()> {
    match spdlog::init_log_crate_proxy() {
        Ok(()) => Ok(()),
        Err(_) if log_crate_proxy_is_installed() => Ok(()),
        Err(_) => Err(FlowError::AlreadyExists(
            "process-global log facade is already initialized by another logger; Relay logging cannot install its log proxy"
                .into(),
        )),
    }
}

/// Owns logging resources that must remain alive for the process / run lifetime.
///
/// When created by [`LoggingRuntime::configure`], dropping this value flushes sinks and detaches this
/// logger from the process-global spdlog `log` proxy if it is still installed.
pub struct LoggingRuntime {
    root_relay_id: String,
    /// Underlying spdlog logger (also installed into the `log` facade by
    /// [`LoggingRuntime::configure`]).
    pub(crate) logger: Arc<Logger>,
    /// Keeps per-sink async thread pools alive until shutdown.
    _thread_pools: Vec<Arc<ThreadPool>>,
}

impl LoggingRuntime {
    /// Installs process-wide operational logging from resolved configuration.
    ///
    /// Stderr is always enabled. Explicit file sinks fail initialization if they cannot be
    /// opened. Dropping the returned runtime flushes sinks and detaches its logger from the
    /// process-global `log` proxy when it is still installed.
    pub fn configure(config: LoggingConfig) -> Result<Self> {
        let root_relay_id = Uuid::now_v7().to_string();
        let (logger, thread_pools) = build_logger(&config, root_relay_id.clone())?;

        // Install once per process. Subsequent calls (tests / re-entry) reuse the proxy and swap
        // the receiver logger. A different global logger would prevent Relay sinks from receiving
        // `log` facade records, so fail instead of returning a nonfunctional runtime.
        let _lifecycle = lock_logger_lifecycle();
        install_log_crate_proxy()?;
        spdlog::log_crate_proxy().set_logger(Some(Arc::clone(&logger)));
        spdlog::log_crate_proxy().set_filter(None);
        log::set_max_level(log_level_filter(config.level));

        Ok(Self {
            root_relay_id,
            logger,
            _thread_pools: thread_pools,
        })
    }

    /// Loads logging configuration from an absolute TOML path and installs it process-wide.
    pub fn configure_from_file_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::configure(LoggingConfig::from_file_path(path)?)
    }

    /// Resolves supported logging environment variables and installs the resulting configuration.
    ///
    /// Built-in defaults are used when no logging environment variables are present.
    pub fn configure_from_environment() -> Result<Self> {
        Self::configure(LoggingConfig::from_environment()?.unwrap_or_default())
    }

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

        // Detach only if we are still the installed receiver. A later configuration may have
        // replaced us; do not clear that newer install. Installation and teardown are serialized
        // because swap-and-restore is a multi-step operation.
        let _lifecycle = lock_logger_lifecycle();
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
    LoggingRuntime::configure(config.clone())
}

#[cfg(test)]
#[path = "../../tests/coverage/logging_tests.rs"]
mod tests;
