// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic recording stays independent of operational log delivery.

use std::io;
use std::sync::{Arc, Mutex};

use spdlog::Logger;

use crate::diagnostics::{
    ComponentRecorder, ComponentRegistration, Diagnostics, Observation, ProjectionError,
    ProjectionGuard,
};
use crate::error::{FlowError, Result};
use nemo_relay_types::diagnostics::{
    BuiltinDiagnosticCode, Classification, DiagnosticClassifier, DiagnosticEvent,
    DiagnosticSeverity, FailureClass, FailureCode, LoggingFailureCode, SuccessCode,
};

use super::LoggingConfig;

const RUNTIME_CODES: &[BuiltinDiagnosticCode] = &[
    BuiltinDiagnosticCode::Success(SuccessCode::LoggingInitialized),
    BuiltinDiagnosticCode::Success(SuccessCode::LoggingShutdownCompleted),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::InitializeFailed)),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::ShutdownFailed)),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(
        LoggingFailureCode::ProjectionUnavailable,
    )),
];
const SINK_CODES: &[BuiltinDiagnosticCode] = &[
    BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(
        LoggingFailureCode::SinkInitializeFailed,
    )),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkRecordDropped)),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkFlushDropped)),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkFlushFailed)),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkWriteFailed)),
    BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkShutdownFailed)),
];

pub(super) struct LoggingDiagnostics {
    // These Options describe best-effort admission, not service enablement. A disabled
    // service returns inert recorders through the same registration interface.
    runtime: Option<ComponentRecorder>,
    stderr_enabled: bool,
    stderr: Option<SinkDiagnostics>,
    files: Vec<Option<SinkDiagnostics>>,
}

impl LoggingDiagnostics {
    pub(super) fn new(diagnostics: &Diagnostics, config: &LoggingConfig) -> Self {
        let register = |key: &str| {
            diagnostics
                .register_component(ComponentRegistration::new(key, SINK_CODES))
                .ok()
                .map(|recorder| SinkDiagnostics { recorder })
        };
        let runtime = diagnostics
            .register_component(ComponentRegistration::new("logging", RUNTIME_CODES))
            .ok();
        let stderr = config
            .stderr_enabled
            .then(|| register("logging.stderr"))
            .flatten();
        let files = config
            .sinks
            .iter()
            .enumerate()
            .map(|(index, _)| register(&format!("logging.file.{index}")))
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
            BuiltinDiagnosticCode::Success(SuccessCode::LoggingInitialized),
            None,
            DiagnosticSeverity::Info,
            "Operational logging initialized",
        );
    }

    pub(super) fn initialization_failed(&self) {
        self.failure(
            LoggingFailureCode::InitializeFailed,
            DiagnosticSeverity::Error,
            "Operational logging initialization failed",
        );
    }

    pub(super) fn shutdown(&self, failed: bool) {
        if failed {
            self.failure(
                LoggingFailureCode::ShutdownFailed,
                DiagnosticSeverity::Error,
                "Operational logging shutdown encountered sink failures",
            );
        } else {
            record(
                self.runtime.as_ref(),
                BuiltinDiagnosticCode::Success(SuccessCode::LoggingShutdownCompleted),
                None,
                DiagnosticSeverity::Info,
                "Operational logging shutdown completed",
            );
        }
    }

    pub(super) fn projection_unavailable(&self) {
        self.failure(
            LoggingFailureCode::ProjectionUnavailable,
            DiagnosticSeverity::Warning,
            "Operational logging is available but the diagnostic projection could not be attached",
        );
    }

    fn failure(
        &self,
        code: LoggingFailureCode,
        severity: DiagnosticSeverity,
        message: &'static str,
    ) {
        record(
            self.runtime.as_ref(),
            BuiltinDiagnosticCode::Failure(FailureCode::Logging(code)),
            Some(FailureClass::Unknown),
            severity,
            message,
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
            BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized),
            None,
            DiagnosticSeverity::Info,
            "Logging sink initialized",
        );
    }

    pub(super) fn initialization_failed(&self) {
        self.failure(
            LoggingFailureCode::SinkInitializeFailed,
            "Logging sink initialization failed",
        );
    }

    pub(super) fn runtime_error(&self, error: &spdlog::Error) {
        let _ = self.recorder.record_without_projection_lazy(|| {
            let classification = LoggingErrorClassifier.classify(error);
            let message = match BuiltinDiagnosticCode::from_canonical_name(classification.code()) {
                Some(BuiltinDiagnosticCode::Failure(FailureCode::Logging(
                    LoggingFailureCode::SinkRecordDropped,
                ))) => "Operational log record dropped because the sink queue is full",
                Some(BuiltinDiagnosticCode::Failure(FailureCode::Logging(
                    LoggingFailureCode::SinkFlushDropped,
                ))) => "Operational log flush request dropped because the sink queue is full",
                Some(BuiltinDiagnosticCode::Failure(FailureCode::Logging(
                    LoggingFailureCode::SinkFlushFailed,
                ))) => "Operational log sink flush failed",
                _ => "Operational log sink failed to deliver a record",
            };
            // Neither a dropped payload nor backend error text belongs in diagnostics.
            Observation::new(classification, message, 1)
        });
    }

    pub(super) fn shutdown_failed(&self) {
        self.failure(
            LoggingFailureCode::SinkShutdownFailed,
            "Operational log sink final drain or flush failed",
        );
    }

    fn failure(&self, code: LoggingFailureCode, message: &'static str) {
        record(
            Some(&self.recorder),
            BuiltinDiagnosticCode::Failure(FailureCode::Logging(code)),
            Some(FailureClass::Unknown),
            DiagnosticSeverity::Error,
            message,
        );
    }
}

/// Classifies structured logging backend evidence without formatting payloads or errors.
///
/// Generic backend failures stay Unknown: their Display strings cannot establish
/// a more specific cause. This producer-local adapter retains no source error.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoggingErrorClassifier;

impl DiagnosticClassifier<spdlog::Error> for LoggingErrorClassifier {
    fn classify(&self, error: &spdlog::Error) -> Classification {
        use spdlog::error::{SendToChannelError, SendToChannelErrorDropped};
        let (code, class) = match error {
            spdlog::Error::SendToChannel(
                SendToChannelError::Full,
                SendToChannelErrorDropped::Record(_),
            ) => (
                LoggingFailureCode::SinkRecordDropped,
                FailureClass::ResourceExhausted,
            ),
            spdlog::Error::SendToChannel(
                SendToChannelError::Full,
                SendToChannelErrorDropped::Flush,
            ) => (
                LoggingFailureCode::SinkFlushDropped,
                FailureClass::ResourceExhausted,
            ),
            spdlog::Error::FlushBuffer(_) => {
                (LoggingFailureCode::SinkFlushFailed, FailureClass::Unknown)
            }
            _ => (LoggingFailureCode::SinkWriteFailed, FailureClass::Unknown),
        };
        Classification::failure(FailureCode::Logging(code), class)
            .expect("logging classifier uses permitted catalogue classes")
    }
}

fn record(
    recorder: Option<&ComponentRecorder>,
    code: BuiltinDiagnosticCode,
    class: Option<FailureClass>,
    severity: DiagnosticSeverity,
    message: &'static str,
) {
    let Some(recorder) = recorder else {
        return;
    };
    // Direct updates bypass every projection, including non-logging subscribers.
    let _ = recorder.record_without_projection_lazy(|| {
        let classification = Classification::new(code, class)
            .expect("logging observations use permitted catalogue classes");
        Observation::new(classification, message, 1).with_severity(severity)
    });
}

// A fixed adapter ceiling, independent of retained detail capacity. Core bounds
// concurrent callback delivery; an oversized conversion reports projection loss.
const MAX_PROJECTED_EVENT_BYTES: usize = 16 * 1_024;

struct ProjectionBuffer {
    bytes: [u8; MAX_PROJECTED_EVENT_BYTES],
    len: usize,
    full: bool,
}

impl ProjectionBuffer {
    fn new() -> Self {
        Self {
            bytes: [0; MAX_PROJECTED_EVENT_BYTES],
            len: 0,
            full: false,
        }
    }
    fn as_str(&self) -> std::result::Result<&str, ProjectionError> {
        std::str::from_utf8(&self.bytes[..self.len])
            .map_err(|_| ProjectionError::SerializationFailed)
    }
}

impl io::Write for ProjectionBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.bytes.len() - self.len {
            self.full = true;
            return Err(io::Error::from(io::ErrorKind::WriteZero));
        }
        self.bytes[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct ProjectionFields<'a> {
    code: &'a str,
    json: &'a str,
}

impl log::kv::Source for ProjectionFields<'_> {
    fn visit<'kvs>(
        &'kvs self,
        visitor: &mut dyn log::kv::VisitSource<'kvs>,
    ) -> std::result::Result<(), log::kv::Error> {
        visitor.visit_pair(
            log::kv::Key::from_str("event"),
            log::kv::Value::from(self.code),
        )?;
        visitor.visit_pair(
            log::kv::Key::from_str("diagnostic"),
            log::kv::Value::from_display(&self.json),
        )
    }
}

fn project_event(
    logger: Arc<Logger>,
    event: &DiagnosticEvent,
) -> std::result::Result<(), ProjectionError> {
    let level = match event.severity {
        DiagnosticSeverity::Trace => log::Level::Trace,
        DiagnosticSeverity::Debug => log::Level::Debug,
        DiagnosticSeverity::Info => log::Level::Info,
        DiagnosticSeverity::Warning => log::Level::Warn,
        DiagnosticSeverity::Error => log::Level::Error,
    };
    if !logger.should_log(level.into()) {
        return Ok(());
    }
    let mut buffer = ProjectionBuffer::new();
    if serde_json::to_writer(&mut buffer, event).is_err() {
        return Err(if buffer.full {
            ProjectionError::PayloadTooLarge
        } else {
            ProjectionError::SerializationFailed
        });
    }
    let fields = ProjectionFields {
        code: event.code(),
        json: buffer.as_str()?,
    };
    let message = event.message().unwrap_or("Diagnostic observation");
    let args = format_args!("{message}");
    let record = log::Record::builder()
        .args(args)
        .level(level)
        .target("nemo_relay.diagnostics")
        .key_values(&fields)
        .build();
    // Reuse the existing logger/filter/formatter/sinks. The proxy borrows this
    // capped serialization synchronously; no history slot or detail is pinned.
    let proxy = spdlog::LogCrateProxy::default();
    proxy.set_logger(Some(logger));
    log::Log::log(&proxy, &record);
    Ok(())
}

pub(super) fn attach_projection(
    diagnostics: &Diagnostics,
    logger: &Arc<Logger>,
) -> Result<LoggingProjection> {
    let logger = Arc::downgrade(logger);
    let active = Arc::new(Mutex::new(true));
    let callback_active = Arc::clone(&active);
    let subscription = diagnostics
        .subscribe_fallible_projection(move |event| {
            let active = callback_active
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !*active {
                return Ok(());
            }
            let Some(logger) = logger.upgrade() else {
                return Ok(());
            };
            if !super::is_active_relay_logger(&logger) {
                return Ok(());
            }
            project_event(logger, event)
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
        // A copied callback can still run after unsubscribe. Quiesce it before
        // the owning runtime drains the existing asynchronous sink pools.
        *self
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = false;
    }
}
