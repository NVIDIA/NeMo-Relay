// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    DROP_REPORT_INTERVAL_MILLIS, DropNoticeRateLimiter, build_logger, dropped_record_error_handler,
    log_level_filter, now_millis, reserved_sink_paths, resolve_log_path, rotated_log_path,
    spdlog_level, stderr_error_handler,
};
use crate::logging::LogLevel;
use crate::logging::{
    FileLogRotationConfig, FileLogSinkConfig, LogFormat, LogSinkConfig, LoggingConfig,
    MAX_FILE_SINK_QUEUE_ENTRIES,
};
use std::path::PathBuf;

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
fn logger_builder_rejects_duplicate_conflicting_and_invalid_file_sinks() {
    let directory = tempfile::tempdir().unwrap();
    let log_path = directory.path().join("relay.log");
    let file_sink = |path: PathBuf, rotation| {
        LogSinkConfig::File(FileLogSinkConfig {
            path,
            level: LogLevel::Info,
            format: LogFormat::Jsonl,
            queue_capacity: 8,
            rotation,
        })
    };

    assert!(resolve_log_path(std::path::Path::new("")).is_err());
    let rotation = FileLogRotationConfig::new(32, 1).unwrap();
    assert_eq!(reserved_sink_paths(&log_path, Some(rotation)).len(), 2);

    let duplicate = LoggingConfig {
        sinks: vec![
            file_sink(log_path.clone(), None),
            file_sink(log_path.clone(), None),
        ],
        ..LoggingConfig::default()
    };
    let error = match build_logger(&duplicate, "root".into()) {
        Ok(_) => panic!("duplicate file sinks must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("duplicate logging sink path"));

    let conflict = LoggingConfig {
        sinks: vec![
            file_sink(log_path.clone(), Some(rotation)),
            file_sink(rotated_log_path(&log_path, 1), None),
        ],
        ..LoggingConfig::default()
    };
    let error = match build_logger(&conflict, "root".into()) {
        Ok(_) => panic!("active and rotated file paths must not overlap"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("conflicts"));

    let mut invalid_capacity = LoggingConfig {
        sinks: vec![file_sink(log_path, None)],
        ..LoggingConfig::default()
    };
    let LogSinkConfig::File(file_sink) = &mut invalid_capacity.sinks[0];
    file_sink.queue_capacity = MAX_FILE_SINK_QUEUE_ENTRIES + 1;
    let error = match build_logger(&invalid_capacity, "root".into()) {
        Ok(_) => panic!("oversized async queues must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("queue_capacity"));
}
