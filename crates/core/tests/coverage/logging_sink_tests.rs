// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    DROP_REPORT_INTERVAL_MILLIS, DropNoticeRateLimiter, SinkErrorBoundary,
    async_sink_error_handler, build_logger, log_level_filter, logging_path_identity,
    normalize_path_components, now_millis, reserved_sink_paths, resolve_log_path, spdlog_level,
    stderr_error_handler,
};
use crate::logging::{
    FileLogRotationConfig, FileLogSinkConfig, LogLevel, LogSinkConfig, LoggingConfig,
    MAX_FILE_SINK_QUEUE_ENTRIES,
};
use spdlog::sink::{AsyncPoolSink, GetSinkProp, OverflowPolicy, Sink, SinkProp, SinkPropAccess};
use spdlog::{Error, ErrorHandler, Level, LevelFilter, Logger, Record, ThreadPool};
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

fn enabled_diagnostics() -> crate::diagnostics::Diagnostics {
    use crate::diagnostics::{Diagnostics, DiagnosticsSettings};
    Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap()
}

fn diagnostic_test_producer(
    service: &crate::diagnostics::Diagnostics,
) -> crate::diagnostics::ComponentRecorder {
    use crate::diagnostics::{
        BuiltinDiagnosticCode, CommonPluginFailureCode, ComponentRegistration, FailureCode,
        PluginFailureCode,
    };
    let code = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Common(
        CommonPluginFailureCode::Internal,
    )));
    service
        .register_component(ComponentRegistration::new("test.exporter", &[code]))
        .unwrap()
}

fn record_test_failure(recorder: &crate::diagnostics::ComponentRecorder, message: &str) {
    use crate::diagnostics::{
        Classification, CommonPluginFailureCode, FailureClass, FailureCode, Observation,
        PluginFailureCode,
    };
    recorder
        .record_lazy(|| {
            Observation::new(
                Classification::failure(
                    FailureCode::Plugin(PluginFailureCode::Common(
                        CommonPluginFailureCode::Internal,
                    )),
                    FailureClass::Internal,
                )
                .unwrap(),
                message,
                NonZeroU64::MIN,
            )
        })
        .unwrap();
}

struct BlockingTestSink {
    prop: SinkProp,
    blocked_once: AtomicBool,
    entered: mpsc::Sender<()>,
    release: Mutex<mpsc::Receiver<()>>,
    fail_exit_flush: bool,
}

fn blocking_test_sink(
    fail_exit_flush: bool,
) -> (Arc<BlockingTestSink>, mpsc::Receiver<()>, mpsc::Sender<()>) {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    (
        Arc::new(BlockingTestSink {
            prop: SinkProp::default(),
            blocked_once: AtomicBool::new(false),
            entered: entered_tx,
            release: Mutex::new(release_rx),
            fail_exit_flush,
        }),
        entered_rx,
        release_tx,
    )
}

impl GetSinkProp for BlockingTestSink {
    fn prop(&self) -> &SinkProp {
        &self.prop
    }
}

impl Sink for BlockingTestSink {
    fn log(&self, _record: &Record) -> spdlog::Result<()> {
        if !self.blocked_once.swap(true, Ordering::Relaxed) {
            let _ = self.entered.send(());
            let _ = self.release.lock().unwrap().recv();
        }
        Ok(())
    }

    fn flush(&self) -> spdlog::Result<()> {
        Ok(())
    }

    fn flush_on_exit(&self) -> spdlog::Result<()> {
        if self.fail_exit_flush {
            Err(Error::FlushBuffer(std::io::Error::other(
                "expected exit flush failure",
            )))
        } else {
            Ok(())
        }
    }
}

struct ReleaseBlockedSinks([mpsc::Sender<()>; 2]);

impl ReleaseBlockedSinks {
    fn release(&self) {
        for release in &self.0 {
            let _ = release.send(());
        }
    }
}

impl Drop for ReleaseBlockedSinks {
    fn drop(&mut self) {
        self.release();
    }
}

fn test_async_sink(backend: Arc<dyn Sink>, error_handler: ErrorHandler) -> AsyncPoolSink {
    let pool = ThreadPool::builder()
        .capacity(NonZeroUsize::new(1).unwrap())
        .build_arc()
        .unwrap();
    AsyncPoolSink::builder()
        .sink(backend)
        .thread_pool(pool)
        .overflow_policy(OverflowPolicy::DropIncoming)
        .error_handler(error_handler)
        .build()
        .unwrap()
}

#[test]
fn drop_notice_rate_limiter_reports_immediately_then_once_per_interval() {
    let rate_limiter = DropNoticeRateLimiter::new();
    let interval = DROP_REPORT_INTERVAL_MILLIS;
    let first_timestamp = 10 * interval;

    assert!(rate_limiter.should_report(first_timestamp));
    assert!(!rate_limiter.should_report(first_timestamp + interval - 1));
    assert!(rate_limiter.should_report(first_timestamp + interval));
}

#[test]
fn sink_helpers_cover_boundary_levels_time_and_emergency_handlers() {
    assert_eq!(spdlog_level(LogLevel::Error), spdlog::Level::Error);
    assert_eq!(spdlog_level(LogLevel::Trace), spdlog::Level::Trace);
    assert_eq!(log_level_filter(LogLevel::Error), log::LevelFilter::Error);
    assert_eq!(log_level_filter(LogLevel::Trace), log::LevelFilter::Trace);
    assert!(now_millis() > 0);

    stderr_error_handler("test", None)(spdlog::Error::WriteRecord(std::io::Error::other(
        "expected test error",
    )));
    async_sink_error_handler("test", None)(spdlog::Error::WriteRecord(std::io::Error::other(
        "expected test error",
    )));
}

#[test]
fn sink_path_helpers_cover_rotation_and_normalization_edges() {
    assert!(resolve_log_path(Path::new("")).is_err());
    assert_eq!(
        normalize_path_components(Path::new("alpha/./beta/../gamma")),
        PathBuf::from("alpha/gamma")
    );
    assert_eq!(
        logging_path_identity(Path::new("relay.log")),
        PathBuf::from("relay.log")
    );

    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("relay.log");
    std::fs::write(&base, "existing").unwrap();
    assert_eq!(
        logging_path_identity(&base),
        std::fs::canonicalize(&base).unwrap()
    );

    let rotation = FileLogRotationConfig::new(1_024, 2).unwrap();
    let paths = reserved_sink_paths(&base, Some(rotation));
    assert_eq!(paths.len(), 3);
    assert_eq!(paths[0], base);
    assert!(paths[1].ends_with("relay.1.log"));
    assert!(paths[2].ends_with("relay.2.log"));
    assert_eq!(reserved_sink_paths(&paths[0], None), vec![paths[0].clone()]);
}

#[test]
fn sink_level_helpers_cover_all_intermediate_levels() {
    for (level, spdlog_level_expected, log_level_expected) in [
        (LogLevel::Warn, spdlog::Level::Warn, log::LevelFilter::Warn),
        (LogLevel::Info, spdlog::Level::Info, log::LevelFilter::Info),
        (
            LogLevel::Debug,
            spdlog::Level::Debug,
            log::LevelFilter::Debug,
        ),
    ] {
        assert_eq!(spdlog_level(level), spdlog_level_expected);
        assert_eq!(log_level_filter(level), log_level_expected);
    }
}

fn file_sink(path: PathBuf) -> FileLogSinkConfig {
    FileLogSinkConfig {
        path,
        ..FileLogSinkConfig::default()
    }
}

fn build_logger_error(config: &LoggingConfig) -> String {
    match build_logger(config, "root".into()) {
        Ok(_) => panic!("expected logger construction to fail"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn logger_builder_rejects_duplicate_reserved_and_invalid_queue_sinks() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("relay.log");

    let mut config = LoggingConfig {
        sinks: vec![
            LogSinkConfig::File(file_sink(path.clone())),
            LogSinkConfig::File(file_sink(path.clone())),
        ],
        ..LoggingConfig::default()
    };
    assert!(build_logger_error(&config).contains("duplicate"));

    let mut rotating = file_sink(path.clone());
    rotating.rotation = Some(FileLogRotationConfig::new(1_024, 1).unwrap());
    config.sinks = vec![
        LogSinkConfig::File(rotating),
        LogSinkConfig::File(file_sink(temp.path().join("relay.1.log"))),
    ];
    assert!(build_logger_error(&config).contains("conflicts"));

    for (capacity, expected) in [
        (0, "must be greater than 0"),
        (MAX_FILE_SINK_QUEUE_ENTRIES + 1, "exceeds maximum"),
    ] {
        let mut sink = file_sink(temp.path().join(format!("queue-{capacity}.log")));
        sink.queue_capacity = capacity;
        config.sinks = vec![LogSinkConfig::File(sink)];
        assert!(build_logger_error(&config).contains(expected));
    }
}

#[test]
fn logger_builder_omits_stderr_when_disabled() {
    let config = LoggingConfig {
        stderr_enabled: false,
        ..LoggingConfig::default()
    };
    let (logger, _) = build_logger(&config, "root".into()).unwrap();
    assert!(logger.sinks().is_empty());
}

#[test]
fn logger_builder_reports_file_and_rotating_file_open_errors() {
    let temp = tempfile::tempdir().unwrap();
    let blocked_parent = temp.path().join("not-a-directory");
    std::fs::write(&blocked_parent, "file").unwrap();
    let mut config = LoggingConfig {
        sinks: vec![LogSinkConfig::File(file_sink(
            blocked_parent.join("relay.log"),
        ))],
        ..LoggingConfig::default()
    };
    assert!(build_logger_error(&config).contains("failed to open logging sink"));

    let mut rotating = file_sink(blocked_parent.join("rotating.log"));
    rotating.rotation = Some(FileLogRotationConfig::new(1_024, 1).unwrap());
    config.sinks = vec![LogSinkConfig::File(rotating)];
    assert!(build_logger_error(&config).contains("failed to open rotating logging sink"));
}

#[test]
fn async_sink_boundary_preserves_shutdown_flush_errors() {
    let (backend, _entered, _release) = blocking_test_sink(true);
    let (sink_tx, sink_rx) = mpsc::channel();
    let sink_handler = ErrorHandler::new(move |error| sink_tx.send(error).unwrap());
    let inner = test_async_sink(backend.clone(), sink_handler.clone());
    let boundary = SinkErrorBoundary::new(inner, sink_handler);

    let result = Sink::flush_on_exit(&boundary);

    assert!(matches!(result, Err(Error::FlushBuffer(_))));
    assert!(matches!(sink_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
}

#[test]
fn async_sink_boundaries_report_each_full_sink_without_blocking_the_caller() {
    let (backend_a, entered_a, release_a) = blocking_test_sink(false);
    let (backend_b, entered_b, release_b) = blocking_test_sink(false);
    let (sink_tx, sink_rx) = mpsc::channel();

    let handler_a = ErrorHandler::new({
        let sink_tx = sink_tx.clone();
        move |error| sink_tx.send(("sink-a", error)).unwrap()
    });
    let handler_b = ErrorHandler::new(move |error| sink_tx.send(("sink-b", error)).unwrap());
    let boundary_a = Arc::new(SinkErrorBoundary::new(
        test_async_sink(backend_a.clone(), ErrorHandler::default()),
        ErrorHandler::default(),
    ));
    let boundary_b = Arc::new(SinkErrorBoundary::new(
        test_async_sink(backend_b.clone(), ErrorHandler::default()),
        ErrorHandler::default(),
    ));
    boundary_a.set_error_handler(handler_a);
    boundary_b.set_error_handler(handler_b);
    boundary_a.set_level_filter(LevelFilter::Off);
    assert!(!boundary_a.should_log(Level::Info));
    boundary_a.set_level_filter(LevelFilter::All);
    let (logger_tx, logger_rx) = mpsc::channel();
    let logger = Logger::builder()
        .sinks([
            boundary_a.clone() as Arc<dyn Sink>,
            boundary_b.clone() as Arc<dyn Sink>,
        ])
        .error_handler(move |error| logger_tx.send(error).unwrap())
        .build_arc()
        .unwrap();
    // Declared after the logger so unwinding releases blocked workers before dropping their pools.
    let release_sinks = ReleaseBlockedSinks([release_a, release_b]);

    spdlog::info!(logger: logger, "block both workers");
    entered_a.recv_timeout(Duration::from_secs(5)).unwrap();
    entered_b.recv_timeout(Duration::from_secs(5)).unwrap();
    spdlog::info!(logger: logger, "fill both queues");

    let producer_logger = logger.clone();
    let (done_tx, done_rx) = mpsc::channel();
    let producer = std::thread::spawn(move || {
        spdlog::info!(logger: producer_logger, "drop in both queues");
        producer_logger.flush();
        done_tx.send(()).unwrap();
    });
    let producer_remained_nonblocking = done_rx.recv_timeout(Duration::from_secs(5)).is_ok();

    release_sinks.release();
    producer.join().unwrap();
    assert!(
        producer_remained_nonblocking,
        "full logging queues must not block the caller"
    );
    let mut record_sinks = Vec::new();
    let mut flush_sinks = Vec::new();
    for (sink, error) in sink_rx.try_iter() {
        match error {
            Error::SendToChannel(
                spdlog::error::SendToChannelError::Full,
                spdlog::error::SendToChannelErrorDropped::Record(_),
            ) => record_sinks.push(sink),
            Error::SendToChannel(
                spdlog::error::SendToChannelError::Full,
                spdlog::error::SendToChannelErrorDropped::Flush,
            ) => flush_sinks.push(sink),
            other => panic!("unexpected async sink error: {other}"),
        }
    }
    record_sinks.sort_unstable();
    flush_sinks.sort_unstable();
    assert_eq!(record_sinks, ["sink-a", "sink-b"]);
    assert_eq!(flush_sinks, ["sink-a", "sink-b"]);
    assert!(matches!(sink_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    assert!(matches!(
        logger_rx.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
}

#[test]
fn diagnostics_count_each_queue_loss_even_when_emergency_notices_are_limited() {
    use crate::diagnostics::{DiagnosticOutcome, DiagnosticQuery};
    use crate::logging::diagnostics::LoggingDiagnostics;

    let diagnostics = enabled_diagnostics();
    let config = LoggingConfig {
        stderr_enabled: false,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig::default())],
        ..LoggingConfig::default()
    };
    let logging = LoggingDiagnostics::new(&diagnostics, &config);
    let (backend, entered, release) = blocking_test_sink(false);
    let handler = ErrorHandler::new(async_sink_error_handler("test", logging.file(0)));
    let boundary = Arc::new(SinkErrorBoundary::new(
        test_async_sink(backend, handler.clone()),
        handler,
    ));
    let logger = Logger::builder().sink(boundary).build_arc().unwrap();
    // Always unblock the worker on assertion failure before the logger drops.
    let release_sinks = ReleaseBlockedSinks([release.clone(), release]);
    spdlog::info!(logger: logger, "block worker");
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    spdlog::info!(logger: logger, "fill queue");
    spdlog::info!(logger: logger, "first dropped record");
    spdlog::info!(logger: logger, "second dropped record");
    logger.flush();

    let snapshot = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let sink = snapshot
        .components
        .iter()
        .find(|component| component.identity.key == "logging.file.0")
        .unwrap();
    for (code, expected) in [
        ("logging.sink.record_dropped", 2),
        ("logging.sink.flush_dropped", 1),
    ] {
        assert_eq!(
            sink.counters
                .iter()
                .find(|counter| counter.code == code)
                .unwrap()
                .occurrences
                .value
                .get(),
            expected
        );
        let observations = snapshot
            .events
            .iter()
            .filter(|event| event.component == sink.identity && event.code() == code)
            .collect::<Vec<_>>();
        assert_eq!(observations.len() as u64, expected);
        assert!(
            observations
                .iter()
                .all(|event| event.outcome == DiagnosticOutcome::Failure)
        );
    }
    release_sinks.release();
}

#[derive(Default)]
struct FailingRecordSink {
    prop: SinkProp,
    fail_flush: bool,
}

impl GetSinkProp for FailingRecordSink {
    fn prop(&self) -> &SinkProp {
        &self.prop
    }
}

impl Sink for FailingRecordSink {
    fn log(&self, _record: &Record) -> spdlog::Result<()> {
        Err(Error::WriteRecord(std::io::Error::other(
            "private backend detail",
        )))
    }

    fn flush(&self) -> spdlog::Result<()> {
        if self.fail_flush {
            Err(Error::FlushBuffer(std::io::Error::other(
                "private flush detail",
            )))
        } else {
            Ok(())
        }
    }
}

#[test]
fn async_projection_failure_records_sink_evidence_without_reenqueuing_the_failure() {
    use crate::diagnostics::DiagnosticQuery;
    use crate::logging::{
        LoggingRuntime,
        diagnostics::{LoggingDiagnostics, attach_projection},
    };

    let _guard = crate::logging::lock_test_logging();
    struct RestoreMaxLevel(log::LevelFilter);
    impl Drop for RestoreMaxLevel {
        fn drop(&mut self) {
            log::set_max_level(self.0);
        }
    }
    let _restore_max_level = RestoreMaxLevel(log::max_level());
    log::set_max_level(log::LevelFilter::Trace);
    let diagnostics = enabled_diagnostics();
    let producer = diagnostic_test_producer(&diagnostics);
    let config = LoggingConfig {
        stderr_enabled: false,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig::default())],
        ..LoggingConfig::default()
    };
    let logging = LoggingDiagnostics::new(&diagnostics, &config);
    let sink_diagnostics = logging.file(0).unwrap();
    let (failed_tx, failed_rx) = mpsc::channel();
    let handler = ErrorHandler::new(move |error| {
        sink_diagnostics.runtime_error(&error);
        failed_tx.send(()).unwrap();
    });
    let boundary = Arc::new(SinkErrorBoundary::new(
        test_async_sink(Arc::new(FailingRecordSink::default()), handler.clone()),
        handler,
    ));
    let logger = Logger::builder().sink(boundary).build_arc().unwrap();
    crate::logging::set_active_relay_logger(&logger);
    let projection = attach_projection(&diagnostics, &logger).unwrap();
    let runtime = LoggingRuntime {
        root_relay_id: diagnostics.source().instance_id.to_string(),
        logger,
        _thread_pools: Vec::new(),
        sink_indices: vec![0],
        unavailable_files: Vec::new(),
        diagnostics: logging,
        config,
        diagnostic_source: Some(diagnostics.source().instance_id),
        projection: Some(projection),
    };

    record_test_failure(&producer, "Exporter failed");
    failed_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    runtime.shutdown();

    let snapshot = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let sink = snapshot
        .components
        .iter()
        .find(|component| component.identity.key == "logging.file.0")
        .unwrap();
    assert_eq!(
        sink.counters
            .iter()
            .find(|counter| counter.code == "logging.sink.write_failed")
            .unwrap()
            .occurrences
            .value
            .get(),
        1
    );
    assert!(failed_rx.try_recv().is_err());
    assert!(
        !serde_json::to_string(&snapshot)
            .unwrap()
            .contains("private backend detail")
    );
}

#[test]
fn logging_shutdown_records_final_flush_failures_before_closing_component_handles() {
    use crate::diagnostics::{DiagnosticOutcome, DiagnosticQuery};
    use crate::logging::{LoggingRuntime, diagnostics::LoggingDiagnostics};

    let _guard = crate::logging::lock_test_logging();
    let diagnostics = enabled_diagnostics();
    let config = LoggingConfig {
        stderr_enabled: false,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig::default()); 2],
        ..LoggingConfig::default()
    };
    let logging = LoggingDiagnostics::new(&diagnostics, &config);
    let (backend, _entered, _release) = blocking_test_sink(true);
    let boundary = Arc::new(SinkErrorBoundary::new(
        test_async_sink(backend, ErrorHandler::default()),
        ErrorHandler::default(),
    ));
    let logger = Logger::builder().sink(boundary).build_arc().unwrap();
    LoggingRuntime {
        root_relay_id: diagnostics.source().instance_id.to_string(),
        logger,
        _thread_pools: Vec::new(),
        // Only the second configured file was installed.
        sink_indices: vec![1],
        unavailable_files: Vec::new(),
        diagnostics: logging,
        config,
        diagnostic_source: Some(diagnostics.source().instance_id),
        projection: None,
    }
    .shutdown();

    let snapshot = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    for (component_key, code) in [
        ("logging", "logging.shutdown_failed"),
        ("logging.file.1", "logging.sink.shutdown_failed"),
    ] {
        let component = snapshot
            .components
            .iter()
            .find(|component| component.identity.key == component_key)
            .unwrap();
        assert!(component.closed_at.is_some());
        assert!(snapshot.events.iter().any(|event| {
            event.component == component.identity
                && event.code() == code
                && event.outcome == DiagnosticOutcome::Failure
        }));
        assert_eq!(
            component
                .counters
                .iter()
                .find(|counter| counter.code == code)
                .unwrap()
                .occurrences
                .value
                .get(),
            1
        );
    }
}

#[test]
fn synchronous_sink_boundary_reports_once_and_preserves_exit_flush_errors() {
    use crate::diagnostics::{DiagnosticOutcome, DiagnosticQuery};
    use crate::logging::diagnostics::LoggingDiagnostics;

    let diagnostics = enabled_diagnostics();
    let logging = LoggingDiagnostics::new(&diagnostics, &LoggingConfig::default());
    let sink_diagnostics = logging.stderr().unwrap();
    let (sink_tx, sink_rx) = mpsc::channel();
    let boundary = Arc::new(SinkErrorBoundary::new(
        FailingRecordSink {
            fail_flush: true,
            ..Default::default()
        },
        ErrorHandler::new(move |error| {
            sink_diagnostics.runtime_error(&error);
            sink_tx.send(error).unwrap();
        }),
    ));
    let (logger_tx, logger_rx) = mpsc::channel();
    let logger = Logger::builder()
        .sink(boundary.clone())
        .error_handler(move |error| logger_tx.send(error).unwrap())
        .build_arc()
        .unwrap();

    spdlog::info!(logger: logger, "A synchronous record fails");
    logger.flush();
    // Exit failures must reach LoggingRuntime instead of being consumed or reported twice.
    assert!(matches!(
        Sink::flush_on_exit(boundary.as_ref()),
        Err(Error::FlushBuffer(_))
    ));

    let errors = sink_rx.try_iter().collect::<Vec<_>>();
    assert_eq!(errors.len(), 2);
    assert!(matches!(&errors[0], Error::WriteRecord(_)));
    assert!(matches!(&errors[1], Error::FlushBuffer(_)));
    assert!(matches!(
        logger_rx.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    let snapshot = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let sink = snapshot
        .components
        .iter()
        .find(|component| component.identity.key == "logging.stderr")
        .unwrap();
    for code in ["logging.sink.write_failed", "logging.sink.flush_failed"] {
        assert_eq!(
            sink.counters
                .iter()
                .find(|counter| counter.code == code)
                .unwrap()
                .occurrences
                .value
                .get(),
            1
        );
        let events = snapshot
            .events
            .iter()
            .filter(|event| event.component == sink.identity && event.code() == code)
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].outcome, DiagnosticOutcome::Failure);
    }
}

#[test]
fn stderr_backend_failures_record_once_without_projection_recursion() {
    use crate::diagnostics::{DiagnosticOutcome, DiagnosticQuery};
    use crate::logging::LoggingRuntime;

    const CHILD_ENV: &str = "NEMO_RELAY_TEST_STDERR_DIAGNOSTICS_CHILD";
    if std::env::var_os(CHILD_ENV).is_none() {
        let (stderr_reader, stderr_writer) = std::io::pipe().unwrap();
        drop(stderr_reader);
        let current_thread = std::thread::current();
        let test_name = current_thread.name().expect("test thread should be named");
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD_ENV, "1")
            // A closed pipe reader fails writes on Unix and Windows. An absent
            // stderr handle would be silently ignored by Rust instead.
            .stderr(stderr_writer)
            .output()
            .expect("stderr diagnostic child test should start");
        assert!(
            output.status.success(),
            "stderr diagnostic child test failed\nstdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
        return;
    }

    // Stderr is intentionally unwritable; keep child assertion failures visible.
    std::panic::set_hook(Box::new(|info| println!("{info}")));
    let diagnostics = enabled_diagnostics();
    let producer = diagnostic_test_producer(&diagnostics);
    let runtime = LoggingRuntime::configure_with_diagnostics(
        LoggingConfig {
            level: LogLevel::Error,
            flush_interval_millis: 0,
            ..LoggingConfig::default()
        },
        &diagnostics,
    )
    .unwrap();
    let logger_failures = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let callback_logger_failures = Arc::clone(&logger_failures);
    runtime.logger.set_error_handler(move |_| {
        callback_logger_failures.fetch_add(1, Ordering::Relaxed);
    });
    let projected = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let callback_projected = Arc::clone(&projected);
    let _projection = diagnostics
        .subscribe_projection(move |_| {
            callback_projected.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();

    log::error!(target: "test.stderr", "An ordinary stderr write fails");
    record_test_failure(&producer, "The projected stderr write also fails");

    assert_eq!(logger_failures.load(Ordering::Relaxed), 0);
    assert_eq!(projected.load(Ordering::Relaxed), 1);
    runtime.shutdown();
    let snapshot = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let sink = snapshot
        .components
        .iter()
        .find(|component| component.identity.key == "logging.stderr")
        .unwrap();
    assert!(sink.closed_at.is_some());
    assert_eq!(
        sink.counters
            .iter()
            .find(|counter| counter.code == "logging.sink.write_failed")
            .unwrap()
            .occurrences
            .value
            .get(),
        2
    );
    let events = snapshot
        .events
        .iter()
        .filter(|event| {
            event.component == sink.identity && event.code() == "logging.sink.write_failed"
        })
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 2);
    assert!(
        events
            .iter()
            .all(|event| event.outcome == DiagnosticOutcome::Failure)
    );
}

#[test]
fn late_diagnostic_attachment_reaches_existing_sink_error_handlers() {
    use crate::logging::diagnostics::LoggingDiagnostics;
    let disabled = crate::diagnostics::Diagnostics::default();
    let config = LoggingConfig::default();
    let mut adapter = LoggingDiagnostics::new(&disabled, &config);
    // This clone is what the logger's existing handler captures at import.
    let retained_handler = adapter.stderr().unwrap();
    let enabled = enabled_diagnostics();
    adapter.attach(&enabled);
    retained_handler.runtime_error(&Error::FlushBuffer(std::io::Error::other(
        "restricted backend detail",
    )));
    let snapshot = enabled
        .snapshot(&crate::diagnostics::DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(snapshot.lifetime.totals.failures_total.value.get(), 1);
    assert!(
        snapshot
            .events
            .iter()
            .any(|event| event.component.key == "logging.stderr")
    );
    assert!(
        !serde_json::to_string(&snapshot)
            .unwrap()
            .contains("restricted backend detail")
    );
}

#[test]
fn optional_file_policy_only_accepts_structured_initialization_io_errors() {
    use std::io;
    for error in [
        Error::CreateDirectory(io::ErrorKind::PermissionDenied.into()),
        Error::OpenFile(io::ErrorKind::PermissionDenied.into()),
        Error::QueryFileMetadata(io::ErrorKind::PermissionDenied.into()),
    ] {
        assert_eq!(
            super::file_open_error_kind(&error),
            Some(io::ErrorKind::PermissionDenied)
        );
    }
    for error in [
        Error::FormatRecord(std::fmt::Error),
        Error::Downstream(Box::new(io::Error::other("permission denied"))),
        Error::WriteRecord(io::ErrorKind::PermissionDenied.into()),
        Error::FlushBuffer(io::ErrorKind::PermissionDenied.into()),
    ] {
        assert_eq!(super::file_open_error_kind(&error), None);
    }
}
