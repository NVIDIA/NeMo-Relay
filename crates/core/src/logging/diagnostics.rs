// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic recording stays independent of operational log delivery.

use std::sync::{Arc, Mutex};

use spdlog::Logger;

use crate::diagnostics::{ComponentRecorder, Diagnostics, Observation, ProjectionGuard};
use crate::error::{FlowError, Result};
use nemo_relay_types::diagnostics::{DiagnosticOutcome, DiagnosticSeverity};

use super::LoggingConfig;

pub(super) struct LoggingDiagnostics {
    runtime: Option<ComponentRecorder>,
    stderr_enabled: bool,
    stderr: Option<SinkDiagnostics>,
    files: Vec<Option<SinkDiagnostics>>,
}

impl LoggingDiagnostics {
    pub(super) fn new(diagnostics: &Diagnostics, config: &LoggingConfig) -> Self {
        let register = |key: String| {
            diagnostics
                .register_component(key)
                .ok()
                .map(|recorder| SinkDiagnostics { recorder })
        };
        let runtime = diagnostics.register_component("logging").ok();
        let stderr = config
            .stderr_enabled
            .then(|| register("logging.stderr".into()))
            .flatten();
        let files = config
            .sinks
            .iter()
            .enumerate()
            .map(|(index, _)| register(format!("logging.file.{index}")))
            .collect();
        Self {
            runtime,
            stderr_enabled: config.stderr_enabled,
            stderr,
            files,
        }
    }

    pub(super) fn stderr(&self) -> Option<SinkDiagnostics> {
        self.stderr.clone()
    }

    pub(super) fn file(&self, index: usize) -> Option<SinkDiagnostics> {
        self.files.get(index).cloned().flatten()
    }

    pub(super) fn sink(&self, index: usize) -> Option<&SinkDiagnostics> {
        match (self.stderr_enabled, index) {
            (true, 0) => self.stderr.as_ref(),
            (true, index) => self.files.get(index - 1).and_then(Option::as_ref),
            (false, index) => self.files.get(index).and_then(Option::as_ref),
        }
    }

    pub(super) fn initialized(&self) {
        record(
            self.runtime.as_ref(),
            "logging.initialized",
            DiagnosticSeverity::Info,
            DiagnosticOutcome::Success,
            "Operational logging initialized",
            false,
        );
    }

    pub(super) fn initialization_failed(&self) {
        record(
            self.runtime.as_ref(),
            "logging.initialize_failed",
            DiagnosticSeverity::Error,
            DiagnosticOutcome::Failure,
            "Operational logging initialization failed",
            true,
        );
    }

    pub(super) fn shutdown(&self, failed: bool) {
        record(
            self.runtime.as_ref(),
            if failed {
                "logging.shutdown_failed"
            } else {
                "logging.shutdown_completed"
            },
            if failed {
                DiagnosticSeverity::Error
            } else {
                DiagnosticSeverity::Info
            },
            if failed {
                DiagnosticOutcome::Failure
            } else {
                DiagnosticOutcome::Success
            },
            if failed {
                "Operational logging shutdown encountered sink failures"
            } else {
                "Operational logging shutdown completed"
            },
            failed,
        );
    }

    pub(super) fn projection_unavailable(&self) {
        record(
            self.runtime.as_ref(),
            "logging.projection_unavailable",
            DiagnosticSeverity::Warning,
            DiagnosticOutcome::Failure,
            "Operational logging is available but the diagnostic projection could not be attached",
            true,
        );
    }
}

#[derive(Clone)]
pub(super) struct SinkDiagnostics {
    recorder: ComponentRecorder,
}

impl SinkDiagnostics {
    pub(super) fn initialized(&self) {
        record(
            Some(&self.recorder),
            "logging.sink.initialized",
            DiagnosticSeverity::Info,
            DiagnosticOutcome::Success,
            "Logging sink initialized",
            false,
        );
    }

    pub(super) fn initialization_failed(&self) {
        record(
            Some(&self.recorder),
            "logging.sink.initialize_failed",
            DiagnosticSeverity::Error,
            DiagnosticOutcome::Failure,
            "Logging sink initialization failed",
            true,
        );
    }

    pub(super) fn runtime_error(&self, error: &spdlog::Error) {
        use spdlog::error::{SendToChannelError, SendToChannelErrorDropped};
        let (code, message, activate_condition) = match error {
            spdlog::Error::SendToChannel(
                SendToChannelError::Full,
                SendToChannelErrorDropped::Record(_),
            ) => (
                "logging.sink.record_dropped",
                "Operational log record dropped because the sink queue is full",
                false,
            ),
            spdlog::Error::SendToChannel(
                SendToChannelError::Full,
                SendToChannelErrorDropped::Flush,
            ) => (
                "logging.sink.flush_dropped",
                "Operational log flush request dropped because the sink queue is full",
                false,
            ),
            spdlog::Error::FlushBuffer(_) => (
                "logging.sink.flush_failed",
                "Operational log sink flush failed",
                true,
            ),
            _ => (
                "logging.sink.write_failed",
                "Operational log sink failed to deliver a record",
                true,
            ),
        };
        // Neither a dropped record's payload nor backend error text belongs in diagnostics.
        record(
            Some(&self.recorder),
            code,
            DiagnosticSeverity::Error,
            DiagnosticOutcome::Failure,
            message,
            activate_condition,
        );
    }

    pub(super) fn shutdown_failed(&self) {
        record(
            Some(&self.recorder),
            "logging.sink.shutdown_failed",
            DiagnosticSeverity::Error,
            DiagnosticOutcome::Failure,
            "Operational log sink final drain or flush failed",
            true,
        );
    }
}

fn record(
    recorder: Option<&ComponentRecorder>,
    code: &str,
    severity: DiagnosticSeverity,
    outcome: DiagnosticOutcome,
    message: &str,
    activate_condition: bool,
) {
    let Some(recorder) = recorder else {
        return;
    };
    // Direct state updates cannot recursively enqueue work into the sink that is failing.
    let _ = recorder.record_without_projection(Observation {
        code: code.into(),
        severity,
        outcome,
        message: message.into(),
        count: 1,
        correlation: Default::default(),
        activate_condition,
    });
}

pub(super) fn attach_projection(
    diagnostics: &Diagnostics,
    logger: &Arc<Logger>,
) -> Result<LoggingProjection> {
    let logger = Arc::downgrade(logger);
    let active = Arc::new(Mutex::new(true));
    let callback_active = Arc::clone(&active);
    let subscription = diagnostics
        .subscribe_projection(move |event| {
            let active = callback_active
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !*active {
                return;
            }
            let Some(logger) = logger.upgrade() else {
                return;
            };
            if !super::is_active_relay_logger(&logger) {
                return;
            }
            let level = match event.severity {
                DiagnosticSeverity::Trace => log::Level::Trace,
                DiagnosticSeverity::Debug => log::Level::Debug,
                DiagnosticSeverity::Info => log::Level::Info,
                DiagnosticSeverity::Warning => log::Level::Warn,
                DiagnosticSeverity::Error => log::Level::Error,
            };
            if !logger.should_log(level.into()) {
                return;
            }
            let Ok(diagnostic) = serde_json::to_value(event) else {
                return;
            };
            let fields = serde_json::Map::from_iter([
                (
                    "event".into(),
                    serde_json::Value::String(event.code.clone()),
                ),
                ("diagnostic".into(), diagnostic),
            ]);
            // This local adapter uses this logger's filters and sinks without consulting or
            // mutating the process-global logger. It lives only for this callback invocation.
            let proxy = spdlog::LogCrateProxy::default();
            proxy.set_logger(Some(logger));
            super::emit_to_logger(
                &proxy,
                level,
                "nemo_relay.diagnostics",
                &event.message,
                fields,
            );
        })
        .map_err(|error| FlowError::InvalidArgument(error.to_string()))?;
    Ok(LoggingProjection {
        subscription: Some(subscription),
        active,
    })
}

pub(super) struct LoggingProjection {
    subscription: Option<ProjectionGuard>,
    active: Arc<Mutex<bool>>,
}

impl Drop for LoggingProjection {
    fn drop(&mut self) {
        self.subscription.take();
        // A publisher can have cloned the callback before unsubscribe. Synchronize with that
        // delivery before the owning logging runtime begins draining the async sink pools.
        *self
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = false;
    }
}
