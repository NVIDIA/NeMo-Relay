// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    DROP_REPORT_INTERVAL_MILLIS, DropNoticeRateLimiter, dropped_record_error_handler,
    log_level_filter, logging_path_identity, normalize_path_components, now_millis,
    reserved_sink_paths, resolve_log_path, spdlog_level, stderr_error_handler,
};
use crate::logging::{FileLogRotationConfig, LogLevel};
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
