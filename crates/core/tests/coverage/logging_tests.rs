// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::logging::{
    FileLogSinkConfig, LogFormat, LogLevel, LogSinkConfig, LoggingConfig, build_logger,
    format_event_for_test, init_logging,
};
use serde_json::Value;
use spdlog::Level;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

static LOGGING_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_logging_tests() -> MutexGuard<'static, ()> {
    LOGGING_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn default_config() -> LoggingConfig {
    LoggingConfig::default()
}

#[test]
fn human_formatter_includes_correlation_and_event_context() {
    let line = format_event_for_test(
        LogFormat::Human,
        "018f3d7c-aaaa-bbbb-cccc-ddddeeeeffff",
        Level::Info,
        "nemo_relay.server",
        Some("server_started"),
        "Relay server started",
        &[("bind", "127.0.0.1:4040")],
    );
    assert!(line.contains("INFO"));
    assert!(line.contains("root=018f3d7c"));
    assert!(line.contains("target=nemo_relay.server"));
    assert!(line.contains("event=server_started"));
    assert!(line.contains("bind=127.0.0.1:4040"));
    assert!(line.contains("Relay server started"));
    assert!(line.ends_with('\n'));
}

#[test]
fn jsonl_formatter_emits_required_schema_without_duplicating_event_or_message() {
    let line = format_event_for_test(
        LogFormat::Jsonl,
        "018f3d7c-aaaa-bbbb-cccc-ddddeeeeffff",
        Level::Info,
        "nemo_relay.server",
        Some("server_started"),
        "Relay server started",
        &[("bind", "127.0.0.1:4040")],
    );
    let record: Value = serde_json::from_str(line.trim_end()).unwrap();
    assert_eq!(record["timestamp"], "2026-07-10T14:22:31.123Z");
    assert_eq!(record["level"], "info");
    assert_eq!(
        record["root_relay_id"],
        "018f3d7c-aaaa-bbbb-cccc-ddddeeeeffff"
    );
    assert_eq!(record["target"], "nemo_relay.server");
    assert_eq!(record["event"], "server_started");
    assert_eq!(record["message"], "Relay server started");
    assert_eq!(record["fields"]["bind"], "127.0.0.1:4040");
    assert!(record["fields"].get("event").is_none());
    assert!(record["fields"].get("message").is_none());
}

#[test]
fn jsonl_formatter_redacts_sensitive_field_names() {
    let line = format_event_for_test(
        LogFormat::Jsonl,
        "root-id",
        Level::Warn,
        "nemo_relay.config",
        Some("config_warning"),
        "sanitized",
        &[
            ("token", "secret-token"),
            ("api_key", "sk-test"),
            ("bind", "127.0.0.1:1"),
        ],
    );
    let record: Value = serde_json::from_str(line.trim_end()).unwrap();
    assert_eq!(record["fields"]["token"], "[redacted]");
    assert_eq!(record["fields"]["api_key"], "[redacted]");
    assert_eq!(record["fields"]["bind"], "127.0.0.1:1");
}

#[test]
fn multiple_jsonl_records_are_one_object_per_line_with_shared_root_id() {
    let root = "018f3d7c-1111-2222-3333-444455556666";
    let first = format_event_for_test(
        LogFormat::Jsonl,
        root,
        Level::Info,
        "nemo_relay.server",
        Some("a"),
        "one",
        &[],
    );
    let second = format_event_for_test(
        LogFormat::Jsonl,
        root,
        Level::Info,
        "nemo_relay.gateway",
        Some("b"),
        "two",
        &[],
    );
    let combined = format!("{first}{second}");
    let lines: Vec<&str> = combined.lines().collect();
    assert_eq!(lines.len(), 2);
    let left: Value = serde_json::from_str(lines[0]).unwrap();
    let right: Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(left["root_relay_id"], root);
    assert_eq!(right["root_relay_id"], root);
    assert_eq!(left["event"], "a");
    assert_eq!(right["event"], "b");
}

#[test]
fn file_sink_receives_jsonl_and_preserves_existing_content() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("relay.log.jsonl");
    std::fs::write(&path, "{\"preexisting\":true}\n").unwrap();

    let config = LoggingConfig {
        level: LogLevel::Info,
        stderr_format: LogFormat::Human,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: path.clone(),
            level: LogLevel::Info,
            format: LogFormat::Jsonl,
            flush_interval_millis: 0,
            ..FileLogSinkConfig::default()
        })],
    };
    let runtime = init_logging(&config).unwrap();
    let root = runtime.root_relay_id().to_owned();

    log::info!(
        target: "nemo_relay.server",
        event = "server_started",
        bind = "127.0.0.1:4040";
        "Relay server started"
    );
    runtime.logger.flush();
    // AsyncPoolSink flush queues work on the pool; wait briefly for the append to land.
    let contents = wait_for_log_line(&path, |contents| contents.lines().count() >= 2);
    runtime.shutdown();

    assert!(contents.starts_with("{\"preexisting\":true}\n"));
    let mut lines = contents.lines();
    let _preexisting = lines.next().unwrap();
    let record: Value = serde_json::from_str(lines.next().expect("logged line")).unwrap();
    assert_eq!(record["root_relay_id"], root);
    assert_eq!(record["target"], "nemo_relay.server");
    assert_eq!(record["event"], "server_started");
    assert_eq!(record["message"], "Relay server started");
    assert_eq!(record["fields"]["bind"], "127.0.0.1:4040");
    assert!(record["fields"].get("event").is_none());
    assert!(record["fields"].get("message").is_none());
}

fn wait_for_log_line(path: &std::path::Path, ready: impl Fn(&str) -> bool) -> String {
    for _ in 0..50 {
        if let Ok(contents) = std::fs::read_to_string(path)
            && ready(&contents)
        {
            return contents;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    std::fs::read_to_string(path).unwrap_or_default()
}

#[test]
fn sink_level_filter_drops_events_below_sink_minimum() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("filtered.log.jsonl");

    let config = LoggingConfig {
        level: LogLevel::Debug,
        stderr_format: LogFormat::Human,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: path.clone(),
            level: LogLevel::Warn,
            format: LogFormat::Jsonl,
            flush_interval_millis: 0,
            ..FileLogSinkConfig::default()
        })],
    };
    let runtime = init_logging(&config).unwrap();

    log::info!(
        target: "nemo_relay.server",
        event = "info_only";
        "should not reach warn sink"
    );
    log::warn!(
        target: "nemo_relay.server",
        event = "warn_event";
        "should reach warn sink"
    );
    runtime.logger.flush();
    let contents = wait_for_log_line(&path, |contents| contents.contains("warn_event"));
    runtime.shutdown();

    assert!(!contents.contains("info_only"));
    assert!(contents.contains("warn_event"));
}

#[test]
fn init_logging_errors_when_sink_cannot_be_opened() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let blocker = temp.path().join("not-a-directory");
    std::fs::write(&blocker, "file").unwrap();
    let path = blocker.join("relay.log.jsonl");

    let config = LoggingConfig {
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path,
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
    };
    let error = build_logger(&config, "root".into())
        .err()
        .expect("open should fail")
        .to_string();
    assert!(error.contains("failed to open logging sink") || error.contains("failed to create"));
}

#[test]
fn init_logging_rejects_duplicate_resolved_paths() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("dup.log.jsonl");
    let config = LoggingConfig {
        sinks: vec![
            LogSinkConfig::File(FileLogSinkConfig {
                path: path.clone(),
                ..FileLogSinkConfig::default()
            }),
            LogSinkConfig::File(FileLogSinkConfig {
                path,
                ..FileLogSinkConfig::default()
            }),
        ],
        ..default_config()
    };
    let error = build_logger(&config, "root".into())
        .err()
        .expect("duplicate paths should fail")
        .to_string();
    assert!(error.contains("duplicate logging sink path"));
}

#[test]
fn init_logging_rejects_dot_slash_duplicate_relative_paths() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let previous_cwd = std::env::current_dir().unwrap();
    struct RestoreCwd(PathBuf);
    impl Drop for RestoreCwd {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }
    let _restore_cwd = RestoreCwd(previous_cwd);
    std::env::set_current_dir(temp.path()).unwrap();

    let config = LoggingConfig {
        sinks: vec![
            LogSinkConfig::File(FileLogSinkConfig {
                path: PathBuf::from("dup.log.jsonl"),
                ..FileLogSinkConfig::default()
            }),
            LogSinkConfig::File(FileLogSinkConfig {
                path: PathBuf::from("./dup.log.jsonl"),
                ..FileLogSinkConfig::default()
            }),
        ],
        ..default_config()
    };
    let error = build_logger(&config, "root".into())
        .err()
        .expect("dot-slash duplicate paths should fail")
        .to_string();
    assert!(error.contains("duplicate logging sink path"));
}

#[test]
fn shutdown_drains_async_file_sink_without_waiting() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("shutdown-drain.log.jsonl");

    let config = LoggingConfig {
        level: LogLevel::Info,
        stderr_format: LogFormat::Human,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: path.clone(),
            level: LogLevel::Info,
            format: LogFormat::Jsonl,
            flush_interval_millis: 0,
            ..FileLogSinkConfig::default()
        })],
    };
    let runtime = init_logging(&config).unwrap();

    log::info!(
        target: "nemo_relay.server",
        event = "shutdown_drain";
        "must land via flush_on_exit"
    );
    // Deliberately skip logger.flush() / sleep: Drop must drain AsyncPoolSink.
    runtime.shutdown();

    let contents = std::fs::read_to_string(&path).expect("log file after shutdown");
    assert!(
        contents.contains("shutdown_drain"),
        "expected drained log contents, got: {contents:?}"
    );
}

#[test]
fn default_logging_config_has_stderr_defaults_and_no_sinks() {
    let config = LoggingConfig::default();
    assert_eq!(config.level, LogLevel::Info);
    assert_eq!(config.stderr_format, LogFormat::Human);
    assert!(config.sinks.is_empty());
}

#[test]
fn human_and_jsonl_share_root_id_across_destinations_in_formatter() {
    let root = "shared-root-id";
    let human = format_event_for_test(
        LogFormat::Human,
        root,
        Level::Info,
        "nemo_relay.server",
        Some("server_started"),
        "",
        &[],
    );
    let jsonl = format_event_for_test(
        LogFormat::Jsonl,
        root,
        Level::Info,
        "nemo_relay.server",
        Some("server_started"),
        "",
        &[],
    );
    assert!(human.contains("root=shared"));
    let record: Value = serde_json::from_str(jsonl.trim_end()).unwrap();
    assert_eq!(record["root_relay_id"], root);
}

#[test]
fn stderr_only_logger_builds_without_file_sinks() {
    let _lock = lock_logging_tests();
    let runtime = init_logging(&default_config()).unwrap();
    runtime.shutdown();
}

#[test]
fn global_level_filter_drops_events_below_process_minimum() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("global-filter.log.jsonl");

    // Process minimum is warn; sink would accept info, but global filter must drop it first.
    let config = LoggingConfig {
        level: LogLevel::Warn,
        stderr_format: LogFormat::Human,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: path.clone(),
            level: LogLevel::Info,
            format: LogFormat::Jsonl,
            flush_interval_millis: 0,
            ..FileLogSinkConfig::default()
        })],
    };
    let runtime = init_logging(&config).unwrap();

    log::info!(
        target: "nemo_relay.server",
        event = "info_should_drop";
        "below process minimum"
    );
    log::warn!(
        target: "nemo_relay.server",
        event = "warn_should_keep";
        "at process minimum"
    );
    runtime.logger.flush();
    let contents = wait_for_log_line(&path, |contents| contents.contains("warn_should_keep"));
    runtime.shutdown();

    assert!(!contents.contains("info_should_drop"));
    assert!(contents.contains("warn_should_keep"));
}

#[test]
fn relative_sink_path_resolves_against_process_cwd() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let previous_cwd = std::env::current_dir().unwrap();
    struct RestoreCwd(PathBuf);
    impl Drop for RestoreCwd {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }
    let _restore_cwd = RestoreCwd(previous_cwd);
    std::env::set_current_dir(temp.path()).unwrap();

    let config = LoggingConfig {
        level: LogLevel::Info,
        stderr_format: LogFormat::Human,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: PathBuf::from("relay.log.jsonl"),
            level: LogLevel::Info,
            format: LogFormat::Jsonl,
            flush_interval_millis: 0,
            ..FileLogSinkConfig::default()
        })],
    };
    let expected = temp.path().join("relay.log.jsonl");
    let runtime = init_logging(&config).unwrap();

    log::info!(
        target: "nemo_relay.server",
        event = "relative_path_ok";
        "wrote via relative path"
    );
    runtime.logger.flush();
    let contents = wait_for_log_line(&expected, |contents| contents.contains("relative_path_ok"));
    runtime.shutdown();

    assert!(expected.is_file());
    assert!(contents.contains("relative_path_ok"));
}

#[test]
fn multiple_file_sinks_receive_same_event() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path_a = temp.path().join("a.log.jsonl");
    let path_b = temp.path().join("b.log.jsonl");

    let config = LoggingConfig {
        level: LogLevel::Info,
        stderr_format: LogFormat::Human,
        sinks: vec![
            LogSinkConfig::File(FileLogSinkConfig {
                path: path_a.clone(),
                level: LogLevel::Info,
                format: LogFormat::Jsonl,
                flush_interval_millis: 0,
                ..FileLogSinkConfig::default()
            }),
            LogSinkConfig::File(FileLogSinkConfig {
                path: path_b.clone(),
                level: LogLevel::Info,
                format: LogFormat::Human,
                flush_interval_millis: 0,
                ..FileLogSinkConfig::default()
            }),
        ],
    };
    let runtime = init_logging(&config).unwrap();
    let root = runtime.root_relay_id().to_owned();

    log::info!(
        target: "nemo_relay.server",
        event = "fanout";
        "delivered to both sinks"
    );
    runtime.logger.flush();
    let jsonl = wait_for_log_line(&path_a, |contents| contents.contains("fanout"));
    let human = wait_for_log_line(&path_b, |contents| contents.contains("fanout"));
    runtime.shutdown();

    let record: Value = serde_json::from_str(jsonl.lines().next().unwrap()).unwrap();
    assert_eq!(record["root_relay_id"], root);
    assert_eq!(record["event"], "fanout");
    assert!(human.contains("event=fanout"));
    assert!(human.contains("delivered to both sinks"));
}

#[test]
fn empty_sink_path_fails_at_logger_build() {
    let _lock = lock_logging_tests();
    let config = LoggingConfig {
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: PathBuf::from(""),
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
    };
    let error = build_logger(&config, "root".into())
        .err()
        .expect("empty path should fail")
        .to_string();
    assert!(error.contains("path must not be empty"));
}
