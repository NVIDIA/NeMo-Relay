// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    AsyncSinkErrorBoundary, DROP_REPORT_INTERVAL_MILLIS, DropNoticeRateLimiter,
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
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

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

    stderr_error_handler("test")(spdlog::Error::WriteRecord(std::io::Error::other(
        "expected test error",
    )));
    async_sink_error_handler("test")(spdlog::Error::WriteRecord(std::io::Error::other(
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
    let boundary = AsyncSinkErrorBoundary::new(inner, sink_handler);

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
    let boundary_a = Arc::new(AsyncSinkErrorBoundary::new(
        test_async_sink(backend_a.clone(), ErrorHandler::default()),
        ErrorHandler::default(),
    ));
    let boundary_b = Arc::new(AsyncSinkErrorBoundary::new(
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
