// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic recording stays independent of operational log delivery.

use std::io;
use std::num::NonZeroU64;
use std::sync::{Arc, OnceLock, RwLock};

use spdlog::Logger;

use crate::diagnostics::{
    ComponentRecorder, ComponentRegistration, Diagnostics, Observation, ProjectionError,
    ProjectionGuard,
};
use crate::error::{FlowError, Result};
use nemo_relay_types::diagnostics::{
    BuiltinDiagnosticCode, Classification, DiagnosticClassifier, DiagnosticOutputSummary,
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
            let sink = SinkDiagnostics {
                recorder: Arc::new(OnceLock::new()),
            };
            if diagnostics.reader().availability()
                == crate::diagnostics::DiagnosticAvailability::Available
                && let Ok(recorder) =
                    diagnostics.register_component(ComponentRegistration::new(key, SINK_CODES))
            {
                let _ = sink.recorder.set(recorder);
            }
            Some(sink)
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

    pub(super) fn attach(&mut self, context: &Diagnostics) {
        self.runtime = context
            .register_component(ComponentRegistration::new("logging", RUNTIME_CODES))
            .ok();
        let attach = |slot: &SinkDiagnostics, key: &str| {
            if let Ok(recorder) =
                context.register_component(ComponentRegistration::new(key, SINK_CODES))
            {
                let _ = slot.recorder.set(recorder);
            }
        };
        if let Some(stderr) = &self.stderr {
            attach(stderr, "logging.stderr");
        }
        for (index, file) in self.files.iter().enumerate() {
            if let Some(file) = file {
                attach(file, &format!("logging.file.{index}"));
            }
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
    recorder: Arc<OnceLock<ComponentRecorder>>,
}

impl SinkDiagnostics {
    pub(super) fn initialized(&self) {
        record(
            self.recorder.get(),
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

    pub(super) fn open_failed(&self, reason: io::ErrorKind) {
        record(
            self.recorder.get(),
            BuiltinDiagnosticCode::Failure(FailureCode::Logging(
                LoggingFailureCode::SinkInitializeFailed,
            )),
            Some(FailureClass::Io),
            DiagnosticSeverity::Error,
            file_open_failure_reason(reason),
        );
    }

    pub(super) fn runtime_error(&self, error: &spdlog::Error) {
        let Some(recorder) = self.recorder.get() else {
            return;
        };
        let _ = recorder.record_without_projection_lazy(|| {
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
            Observation::new(classification, message, NonZeroU64::MIN)
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
            self.recorder.get(),
            BuiltinDiagnosticCode::Failure(FailureCode::Logging(code)),
            Some(FailureClass::Unknown),
            DiagnosticSeverity::Error,
            message,
        );
    }
}

/// Fixed reasons shared by startup diagnostics and the emergency stderr notice.
/// Neither file paths nor backend error strings enter diagnostic evidence.
pub(super) fn file_open_failure_reason(reason: io::ErrorKind) -> &'static str {
    match reason {
        io::ErrorKind::PermissionDenied => "Logging sink could not open: permission_denied",
        io::ErrorKind::NotFound => "Logging sink could not open: not_found",
        io::ErrorKind::NotADirectory => "Logging sink could not open: not_a_directory",
        io::ErrorKind::IsADirectory => "Logging sink could not open: is_a_directory",
        io::ErrorKind::ReadOnlyFilesystem => "Logging sink could not open: read_only_filesystem",
        io::ErrorKind::StorageFull => "Logging sink could not open: storage_full",
        _ => "Logging sink could not open: io_error",
    }
}

/// Classifies structured logging backend evidence without formatting payloads or errors.
///
/// Generic backend failures stay Unknown: their Display strings cannot establish
/// a more specific cause. This producer-local adapter retains no source error.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LoggingErrorClassifier;

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
        Observation::new(classification, message, NonZeroU64::MIN).with_severity(severity)
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

fn project_event(
    logger: Arc<Logger>,
    event: &DiagnosticOutputSummary,
) -> std::result::Result<(), ProjectionError> {
    let facade_level = match event.severity {
        DiagnosticSeverity::Trace => log::Level::Trace,
        DiagnosticSeverity::Debug => log::Level::Debug,
        DiagnosticSeverity::Info => log::Level::Info,
        DiagnosticSeverity::Warning => log::Level::Warn,
        DiagnosticSeverity::Error => log::Level::Error,
    };
    if facade_level > log::max_level() {
        return Ok(());
    }
    let level = spdlog::Level::from(facade_level);
    if !logger.should_log(level) {
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
    let json = buffer.as_str()?;
    let message = "Diagnostic observation";
    // Native backend fields borrow the capped buffer through synchronous delivery.
    // The backend formats the payload once; asynchronous sinks make their own
    // required ownership copies before this buffer goes out of scope.
    spdlog::log!(logger: logger, level, "{message}", kv: {
        event = event.code(),
        diagnostic: = json,
    });
    Ok(())
}

pub(super) fn attach_projection(
    diagnostics: &Diagnostics,
    logger: &Arc<Logger>,
) -> Result<LoggingProjection> {
    // Preserve the facade target in RelayFormatter without converting every
    // event through LogCrateProxy. This logger shares the existing sinks.
    let projection_logger = logger
        .fork_with_name(Some("nemo_relay.diagnostics"))
        .map_err(|error| FlowError::InvalidArgument(error.to_string()))?;
    // Only the runtime owns periodic flushing and final sink drain.
    projection_logger.set_flush_period(None);
    let logger = Arc::downgrade(logger);
    let active = Arc::new(RwLock::new(true));
    let callback_active = Arc::clone(&active);
    let subscription = diagnostics
        .subscribe_fallible_projection(move |event| {
            let active = callback_active
                .read()
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
            project_event(Arc::clone(&projection_logger), event)
        })
        .map_err(|error| FlowError::InvalidArgument(error.to_string()))?;
    Ok(LoggingProjection {
        subscription: Some(subscription),
        active,
    })
}

pub(super) struct LoggingProjection {
    subscription: Option<ProjectionGuard>,
    active: Arc<RwLock<bool>>,
}

impl Drop for LoggingProjection {
    fn drop(&mut self) {
        self.subscription.take();
        // A copied callback can still run after unsubscribe. Quiesce it before
        // the owning runtime drains the existing asynchronous sink pools.
        *self
            .active
            .write()
            .unwrap_or_else(|error| error.into_inner()) = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spdlog::sink::{GetSinkProp, Sink, SinkProp};
    use std::sync::{Barrier, Condvar, Mutex, mpsc};
    use std::time::Duration;

    struct BlockingSink {
        prop: SinkProp,
        entered: mpsc::Sender<()>,
        released: Mutex<bool>,
        release: Condvar,
    }
    impl GetSinkProp for BlockingSink {
        fn prop(&self) -> &SinkProp {
            &self.prop
        }
    }
    impl Sink for BlockingSink {
        fn log(&self, _record: &spdlog::Record<'_>) -> spdlog::Result<()> {
            self.entered.send(()).unwrap();
            let released = self.released.lock().unwrap();
            drop(
                self.release
                    .wait_while(released, |released| !*released)
                    .unwrap(),
            );
            Ok(())
        }
        fn flush(&self) -> spdlog::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn projections_overlap_and_shutdown_waits_for_inflight_delivery() {
        let _logging = super::super::lock_test_logging();
        struct RestoreMaxLevel(log::LevelFilter);
        impl Drop for RestoreMaxLevel {
            fn drop(&mut self) {
                log::set_max_level(self.0);
            }
        }
        let _restore_max_level = RestoreMaxLevel(log::max_level());
        log::set_max_level(log::LevelFilter::Trace);
        let diagnostics = Diagnostics::from_settings(
            Diagnostics::default().source().clone(),
            crate::diagnostics::DiagnosticsSettings {
                enabled: true,
                ..Default::default()
            },
        )
        .unwrap();
        let code = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
        let producer = diagnostics
            .register_component(ComponentRegistration::new("test.projection", &[code]))
            .unwrap();
        let (entered, entries) = mpsc::channel();
        let sink = Arc::new(BlockingSink {
            prop: SinkProp::default(),
            entered,
            released: Mutex::new(false),
            release: Condvar::new(),
        });
        let logger = Logger::builder().sink(sink.clone()).build_arc().unwrap();
        super::super::set_active_relay_logger(&logger);
        let projection = attach_projection(&diagnostics, &logger).unwrap();
        let active = projection.active.clone();
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let producer = &producer;
                scope.spawn(move || {
                    producer
                        .record_lazy(|| {
                            Observation::new(
                                Classification::success(SuccessCode::SinkInitialized),
                                "delivery",
                                NonZeroU64::MIN,
                            )
                        })
                        .unwrap()
                });
            }
            let first = entries.recv_timeout(Duration::from_secs(5));
            let second = entries.recv_timeout(Duration::from_secs(5));
            let exclusive_blocked = active.try_write().is_err();
            let (done, finished) = mpsc::channel();
            let teardown_started = Arc::new(Barrier::new(2));
            let teardown_started_thread = Arc::clone(&teardown_started);
            scope.spawn(move || {
                teardown_started_thread.wait();
                drop(projection);
                done.send(()).unwrap();
            });
            teardown_started.wait();
            let shutdown_waiting = matches!(
                finished.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout)
            );
            // Always release blocked deliveries before assertions or scope joining.
            *sink.released.lock().unwrap() = true;
            sink.release.notify_all();
            finished.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(
                first.is_ok() && second.is_ok(),
                "both deliveries must enter before either leaves"
            );
            assert!(exclusive_blocked && shutdown_waiting);
        });
        assert!(!*active.read().unwrap());
        producer
            .record_lazy(|| {
                Observation::new(
                    Classification::success(SuccessCode::SinkInitialized),
                    "after detach",
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
        assert!(entries.try_recv().is_err());
        super::super::clear_active_relay_logger(&logger);
    }
}
