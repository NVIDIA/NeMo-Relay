// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    DROP_REPORT_INTERVAL_MILLIS, DropNoticeRateLimiter, build_logger, dropped_record_error_handler,
    log_level_filter, logging_path_identity, normalize_path_components, now_millis,
    reserved_sink_paths, resolve_log_path, spdlog_level, stderr_error_handler,
};
use crate::logging::{
    FileLogRotationConfig, FileLogSinkConfig, LogLevel, LogSinkConfig, LoggingConfig,
    MAX_FILE_SINK_QUEUE_ENTRIES,
};
use std::path::{Path, PathBuf};

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
    dropped_record_error_handler("test")(spdlog::Error::WriteRecord(std::io::Error::other(
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
