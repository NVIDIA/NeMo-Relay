// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operational process logging for Relay (stderr + optional file sinks).
//!
//! Call sites emit through the `log` facade (`log::info!`, …). This module owns the
//! `spdlog-rs` backend, `LogCrateProxy` installation, formatters, and sink lifetime.

mod config;
mod diagnostics;
mod format;
mod rotation;
mod sink;

use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use serde_json::{Map, Value as Json};
use spdlog::sink::Sink;
use spdlog::{Logger, ThreadPool};

use self::diagnostics::{LoggingDiagnostics, LoggingProjection};
use crate::diagnostics::Diagnostics;
use crate::error::{FlowError, Result};

pub use config::{
    DEFAULT_FILE_FLUSH_INTERVAL_MILLIS, DEFAULT_FILE_SINK_QUEUE_ENTRIES, FileLogRotationConfig,
    FileLogSinkConfig, FileLogSinkFailurePolicy, LogFormat, LogLevel, LogSinkConfig, LoggingConfig,
    MAX_FILE_SINK_QUEUE_ENTRIES, MAX_FILE_SINK_RETAINED_FILES,
};
#[cfg(test)]
pub(crate) use sink::build_logger;
use sink::log_level_filter;

#[cfg(test)]
pub(crate) use format::format_event_for_test;

static LOGGER_LIFECYCLE_LOCK: Mutex<()> = Mutex::new(());
static DEFAULT_LOGGING_RUNTIME: Mutex<Option<LoggingRuntime>> = Mutex::new(None);
static ACTIVE_RELAY_LOGGER: Mutex<Option<Weak<Logger>>> = Mutex::new(None);
#[cfg(test)]
static TEST_LOGGING_LOCK: Mutex<()> = Mutex::new(());

fn lock_logger_lifecycle() -> MutexGuard<'static, ()> {
    LOGGER_LIFECYCLE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
pub(crate) fn lock_test_logging() -> MutexGuard<'static, ()> {
    TEST_LOGGING_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn log_crate_proxy_is_installed() -> bool {
    std::ptr::addr_eq(log::logger(), spdlog::log_crate_proxy() as &dyn log::Log)
}

fn active_relay_logger_exists() -> bool {
    ACTIVE_RELAY_LOGGER
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .as_ref()
        .is_some_and(|logger| logger.upgrade().is_some())
}

fn is_active_relay_logger(logger: &Arc<Logger>) -> bool {
    ACTIVE_RELAY_LOGGER
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .as_ref()
        .is_some_and(|active| Weak::ptr_eq(active, &Arc::downgrade(logger)))
}

fn set_active_relay_logger(logger: &Arc<Logger>) {
    *ACTIVE_RELAY_LOGGER
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = Some(Arc::downgrade(logger));
}

fn clear_active_relay_logger(logger: &Arc<Logger>) {
    let mut active = ACTIVE_RELAY_LOGGER
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if active
        .as_ref()
        .is_some_and(|current| Weak::ptr_eq(current, &Arc::downgrade(logger)))
    {
        *active = None;
    }
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
    /// Original configured sink positions, including stderr, for each installed sink.
    sink_indices: Vec<usize>,
    /// One fixed reason per unavailable file, retained for late diagnostics attachment.
    unavailable_files: Vec<(usize, io::ErrorKind)>,
    diagnostics: LoggingDiagnostics,
    projection: Option<LoggingProjection>,
    config: LoggingConfig,
    diagnostic_source: Option<uuid::Uuid>,
}

impl LoggingRuntime {
    /// Installs process-wide operational logging from resolved configuration.
    ///
    /// Stderr is enabled by default and can be disabled with [`LoggingConfig::stderr_enabled`].
    /// Required file sinks fail initialization if they cannot be opened. Optional sinks are
    /// disabled on filesystem opening failures until logging is reinitialized. Dropping the returned
    /// runtime flushes sinks and detaches its logger from the process-global `log` proxy when it is
    /// still installed.
    /// Reuses an already selected diagnostics context. Otherwise, logging uses an inert context
    /// without selecting the process default, leaving diagnostics enablement to runtime startup.
    pub fn configure(config: LoggingConfig) -> Result<Self> {
        Self::configure_with_diagnostics(
            config,
            Diagnostics::selected_process_default().unwrap_or(&Diagnostics::default()),
        )
    }

    /// Installs operational logging with diagnostics owned by an independent diagnostic context.
    ///
    /// Sink initialization and delivery failures remain queryable when logging is unavailable or
    /// filtered. File sinks use their configured failure policy, matching [`Self::configure`].
    /// Diagnostic events are projected through a detachable subscription. Sink failures update
    /// diagnostic state directly without being sent back through the failing logging pipeline.
    /// Exhausting diagnostic component or projection budgets does not prevent operational logging
    /// from starting; the diagnostic snapshot reports the available evidence and coverage loss.
    /// Dropping this runtime detaches the projection before draining sinks.
    pub fn configure_with_diagnostics(
        config: LoggingConfig,
        context: &Diagnostics,
    ) -> Result<Self> {
        let _lifecycle = lock_logger_lifecycle();
        Self::configure_with_lifecycle_lock(config, context)
    }

    fn configure_with_lifecycle_lock(config: LoggingConfig, context: &Diagnostics) -> Result<Self> {
        let diagnostics = LoggingDiagnostics::new(context, &config);
        let root_relay_id = context.source().instance_id.to_string();
        let built =
            sink::build_logger_with_diagnostics(&config, root_relay_id.clone(), Some(&diagnostics));
        let sink::BuiltLogger {
            logger,
            thread_pools,
            sink_indices,
            unavailable_files,
        } = match built {
            Ok(built) => built,
            Err(error) => {
                diagnostics.initialization_failed();
                return Err(error);
            }
        };
        let projection = match self::diagnostics::attach_projection(context, &logger) {
            Ok(projection) => Some(projection),
            Err(_) => {
                diagnostics.projection_unavailable();
                None
            }
        };
        let installed = {
            install_log_crate_proxy().map(|()| {
                spdlog::log_crate_proxy().set_logger(Some(Arc::clone(&logger)));
                spdlog::log_crate_proxy().set_filter(None);
                log::set_max_level(log_level_filter(config.level));
                set_active_relay_logger(&logger);
            })
        };
        if let Err(error) = installed {
            diagnostics.initialization_failed();
            return Err(error);
        }
        diagnostics.initialized();
        log::info!(
            target: "nemo_relay.logging",
            event = "logging_initialized",
            file_sink_count = thread_pools.len(),
            unavailable_file_sink_count = unavailable_files.len();
            "Operational logging initialized"
        );
        Ok(Self {
            root_relay_id,
            logger,
            _thread_pools: thread_pools,
            sink_indices,
            unavailable_files,
            diagnostics,
            projection,
            config,
            diagnostic_source: (context.reader().availability()
                == crate::diagnostics::DiagnosticAvailability::Available)
                .then_some(context.source().instance_id),
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
        // No new diagnostic projections may enqueue work while the sinks are being drained.
        self.projection.take();
        log::info!(
            target: "nemo_relay.logging",
            event = "logging_shutdown_started";
            "Operational logging shutdown started"
        );
        // Periodic flusher must stop before exit flush so it cannot race teardown.
        self.logger.set_flush_period(None);
        // `Logger::flush` only enqueues AsyncPoolSink work. `flush_on_exit` destroys the
        // pool (draining pending tasks) then flushes the underlying FileSink on this thread.
        // LogCrateProxy loggers are outside spdlog's atexit default-logger path, so we must
        // do this explicitly while `_thread_pools` is still alive.
        let mut shutdown_failed = false;
        for (index, sink) in self.logger.sinks().iter().enumerate() {
            if let Err(error) = Sink::flush_on_exit(sink.as_ref()) {
                shutdown_failed = true;
                if let Some(diagnostics) = self.diagnostics.sink(self.sink_indices[index]) {
                    diagnostics.shutdown_failed();
                }
                let _ = writeln!(
                    io::stderr(),
                    "nemo-relay: logging shutdown flush failed: {error}"
                );
            }
        }
        self.diagnostics.shutdown(shutdown_failed);

        // Detach only if we are still the installed receiver. A later configuration may have
        // replaced us; do not clear that newer install. Installation and teardown are serialized
        // because swap-and-restore is a multi-step operation.
        let _lifecycle = lock_logger_lifecycle();
        let detached = spdlog::log_crate_proxy().swap_logger(None);
        if let Some(logger) = detached
            && !Arc::ptr_eq(&logger, &self.logger)
        {
            spdlog::log_crate_proxy().set_logger(Some(Arc::clone(&logger)));
            set_active_relay_logger(&logger);
        } else {
            clear_active_relay_logger(&self.logger);
        }
    }
}

/// Installs process-wide operational logging from resolved config.
///
/// Stderr is enabled by default and can be disabled with [`LoggingConfig::stderr_enabled`].
/// Required file sinks fail startup if they cannot be opened. Optional sinks are disabled on
/// filesystem opening failures until logging is reinitialized.
///
/// Verbosity comes from [`LoggingConfig::level`]: records below that minimum severity are discarded.
/// Dropping the returned [`LoggingRuntime`] flushes sinks and detaches this logger from the
/// process-global `log` proxy when it is still installed.
pub fn init_logging(config: &LoggingConfig) -> Result<LoggingRuntime> {
    LoggingRuntime::configure(config.clone())
}

/// Installs and retains the default process-wide logging runtime for a language binding.
///
/// Configuration is resolved from the supported logging environment variables, with built-in
/// defaults when none are present. Repeated initialization in the same linked runtime is a no-op.
#[doc(hidden)]
pub fn initialize_default_logging() -> Result<()> {
    let mut runtime = DEFAULT_LOGGING_RUNTIME.lock().map_err(|error| {
        FlowError::Internal(format!("default logging runtime lock poisoned: {error}"))
    })?;
    if runtime.is_none() {
        let config = LoggingConfig::from_environment()?;
        let uses_default_config = config.is_none();
        let _lifecycle = lock_logger_lifecycle();
        if uses_default_config && active_relay_logger_exists() {
            return Ok(());
        }
        match LoggingRuntime::configure_with_lifecycle_lock(
            config.unwrap_or_default(),
            Diagnostics::selected_process_default().unwrap_or(&Diagnostics::default()),
        ) {
            Ok(configured) => *runtime = Some(configured),
            // Language bindings initialize logging automatically. When Relay was not explicitly
            // configured, defer to an application logger that already owns the process facade.
            Err(FlowError::AlreadyExists(_)) if uses_default_config => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Attach a newly selected embedded service without replacing the logger,
/// reopening sinks, resetting queues/rate limits, or replaying previous records.
#[doc(hidden)]
pub fn attach_default_diagnostics(context: &Diagnostics) -> Result<()> {
    let mut owner = DEFAULT_LOGGING_RUNTIME
        .lock()
        .map_err(|_| FlowError::Internal("logging owner unavailable".into()))?;
    let Some(runtime) = owner.as_mut() else {
        return Ok(());
    };
    let _lifecycle = lock_logger_lifecycle();
    if !is_active_relay_logger(&runtime.logger)
        || runtime.diagnostic_source.is_some()
        || context.reader().availability() != crate::diagnostics::DiagnosticAvailability::Available
    {
        return Ok(());
    }
    runtime.diagnostics.attach(context);
    runtime.diagnostic_source = Some(context.source().instance_id);
    runtime.root_relay_id = context.source().instance_id.to_string();
    // These sinks remain unavailable; observe that state when a service attaches later.
    // No log payloads or earlier delivery history are replayed.
    for &(index, reason) in &runtime.unavailable_files {
        if let Some(diagnostic) = runtime.diagnostics.file(index) {
            diagnostic.open_failed(reason);
        }
    }
    let formats = runtime.sink_indices.iter().map(|&index| {
        if runtime.config.stderr_enabled && index == 0 {
            runtime.config.stderr_format
        } else {
            let LogSinkConfig::File(file) =
                &runtime.config.sinks[index - usize::from(runtime.config.stderr_enabled)];
            file.format
        }
    });
    for (sink, format) in runtime.logger.sinks().iter().zip(formats) {
        sink.set_formatter(Box::new(self::format::RelayFormatter {
            format,
            root_relay_id: runtime.root_relay_id.clone(),
        }));
    }
    runtime.projection = match self::diagnostics::attach_projection(context, &runtime.logger) {
        Ok(projection) => Some(projection),
        Err(_) => {
            runtime.diagnostics.projection_unavailable();
            None
        }
    };
    Ok(())
}

/// Explicit adapter coverage; application-owned loggers remain unsupported.
#[doc(hidden)]
pub fn default_diagnostics_coverage() -> &'static str {
    let owner = DEFAULT_LOGGING_RUNTIME
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    match owner.as_ref() {
        Some(runtime) if !is_active_relay_logger(&runtime.logger) => "not_active_owner",
        Some(runtime) if runtime.diagnostic_source.is_some() => "attached_since_service_start",
        Some(_) => "not_attached",
        None => "not_owned_or_closed",
    }
}

/// Shuts down and releases the default process-wide logging runtime for a language binding.
///
/// Repeated shutdown in the same linked runtime is a no-op. The runtime is removed from shared
/// state before its sinks are drained so shutdown does not hold the default-runtime lock.
#[doc(hidden)]
pub fn shutdown_default_logging() -> Result<()> {
    let runtime = DEFAULT_LOGGING_RUNTIME
        .lock()
        .map_err(|error| {
            FlowError::Internal(format!("default logging runtime lock poisoned: {error}"))
        })?
        .take();
    if let Some(runtime) = runtime {
        runtime.shutdown();
    }
    Ok(())
}

/// Emits a structured operational record through Relay's configured logger.
pub fn emit(level: log::Level, target: &str, message: &str, fields: Map<String, Json>) {
    if level > log::max_level() {
        return;
    }

    struct Fields(Map<String, Json>);
    impl log::kv::Source for Fields {
        fn visit<'kvs>(
            &'kvs self,
            visitor: &mut dyn log::kv::VisitSource<'kvs>,
        ) -> std::result::Result<(), log::kv::Error> {
            for (key, value) in &self.0 {
                visitor.visit_pair(
                    log::kv::Key::from_str(key),
                    log::kv::Value::from_serde(value),
                )?;
            }
            Ok(())
        }
    }
    let fields = Fields(fields);
    let args = format_args!("{message}");
    let record = log::Record::builder()
        .args(args)
        .level(level)
        .target(target)
        .key_values(&fields)
        .build();
    log::logger().log(&record);
}

/// Emits a structured record using a binding-facing level name.
pub fn emit_str(level: &str, target: &str, message: &str, fields: Map<String, Json>) -> Result<()> {
    // Accept the same level names as logging configuration.
    let level = LogLevel::parse(level)?;
    emit(level.into(), target, message, fields);
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/coverage/logging_tests.rs"]
mod tests;
