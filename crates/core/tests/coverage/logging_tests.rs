// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::logging::{
    FileLogSinkConfig, LogFormat, LogLevel, LogSinkConfig, LoggingConfig, LoggingRuntime,
    MAX_FILE_SINK_QUEUE_ENTRIES, build_logger, format_event_for_test, init_logging,
};
use serde_json::Value;
use spdlog::Level;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

static LOGGING_TEST_LOCK: Mutex<()> = Mutex::new(());
const FOREIGN_LOGGER_CHILD_ENV: &str = "NEMO_RELAY_TEST_FOREIGN_LOGGER_CHILD";

struct ForeignLogger;

impl log::Log for ForeignLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, _record: &log::Record<'_>) {}

    fn flush(&self) {}
}

static FOREIGN_LOGGER: ForeignLogger = ForeignLogger;

fn lock_logging_tests() -> MutexGuard<'static, ()> {
    LOGGING_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

struct LoggingEnvScope {
    _guard: MutexGuard<'static, ()>,
    previous: Vec<(&'static str, Option<OsString>)>,
}

impl LoggingEnvScope {
    fn set(values: &[(&'static str, Option<&OsStr>)]) -> Self {
        let guard = lock_logging_tests();
        let mut previous = Vec::with_capacity(values.len());
        for &(name, value) in values {
            previous.push((name, std::env::var_os(name)));
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for LoggingEnvScope {
    fn drop(&mut self) {
        for (name, value) in self.previous.drain(..).rev() {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

fn default_config() -> LoggingConfig {
    LoggingConfig::default()
}

fn toml_basic_string(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

#[test]
fn configure_from_file_path_loads_toml_and_writes_file_sink() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("logging.toml");
    let log_path = temp.path().join("relay.log.jsonl");
    std::fs::write(
        &config_path,
        format!(
            r#"
[logging]
level = "debug"
stderr_format = "human"

[[logging.sinks]]
path = {}
level = "debug"
format = "jsonl"
queue_capacity = 16
"#,
            toml_basic_string(log_path.to_string_lossy().as_ref())
        ),
    )
    .unwrap();

    let runtime = LoggingRuntime::configure_from_file_path(&config_path).unwrap();
    let root_relay_id = runtime.root_relay_id().to_owned();
    log::debug!(
        target: "nemo_relay.logging_test",
        event = "configured_from_file",
        source = "toml";
        "Configured logging from a TOML file"
    );
    runtime.logger.flush();
    let contents = wait_for_log_line(&log_path, |contents| {
        contents.contains("configured_from_file")
    });
    runtime.shutdown();

    let line = contents
        .lines()
        .find(|line| line.contains("configured_from_file"))
        .expect("configured event should reach the file sink");
    let record: Value = serde_json::from_str(line).unwrap();
    assert_eq!(record["root_relay_id"], root_relay_id);
    assert_eq!(record["level"], "debug");
    assert_eq!(record["target"], "nemo_relay.logging_test");
    assert_eq!(record["event"], "configured_from_file");
    assert_eq!(record["fields"]["source"], "toml");
}

#[test]
fn logging_config_from_environment_resolves_direct_settings() {
    let _environment = LoggingEnvScope::set(&[
        ("NEMO_RELAY_LOG", Some(OsStr::new("debug"))),
        ("NEMO_RELAY_LOG_STDERR_FORMAT", Some(OsStr::new("jsonl"))),
        ("NEMO_RELAY_LOG_CONFIG_PATH", None),
    ]);

    let config = LoggingConfig::from_environment()
        .unwrap()
        .expect("direct environment settings should select a configuration");

    assert_eq!(config.level, LogLevel::Debug);
    assert_eq!(config.stderr_format, LogFormat::Jsonl);
    assert!(config.sinks.is_empty());
}

#[test]
fn logging_environment_rejects_mixed_sources() {
    let unused_path = std::env::current_dir().unwrap().join("unused-logging.toml");
    let _environment = LoggingEnvScope::set(&[
        ("NEMO_RELAY_LOG", Some(OsStr::new("info"))),
        ("NEMO_RELAY_LOG_STDERR_FORMAT", None),
        ("NEMO_RELAY_LOG_CONFIG_PATH", Some(unused_path.as_os_str())),
    ]);

    let error = LoggingConfig::from_environment().unwrap_err().to_string();

    assert!(error.contains("cannot be combined"), "{error}");
}

#[test]
fn logging_environment_rejects_empty_values() {
    let _environment = LoggingEnvScope::set(&[
        ("NEMO_RELAY_LOG", Some(OsStr::new(""))),
        ("NEMO_RELAY_LOG_STDERR_FORMAT", None),
        ("NEMO_RELAY_LOG_CONFIG_PATH", None),
    ]);

    let error = LoggingConfig::from_environment().unwrap_err().to_string();

    assert!(
        error.contains("NEMO_RELAY_LOG must not be empty"),
        "{error}"
    );
}

#[test]
fn logging_config_path_must_be_absolute() {
    let error = LoggingConfig::from_file_path("relative-logging.toml")
        .unwrap_err()
        .to_string();

    assert!(error.contains("path must be absolute"), "{error}");
}

#[test]
fn logging_config_file_requires_logging_section() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("logging.toml");
    std::fs::write(&config_path, "title = \"not logging configuration\"\n").unwrap();

    let error = LoggingConfig::from_file_path(&config_path)
        .unwrap_err()
        .to_string();

    assert!(error.contains("requires a [logging] section"), "{error}");
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
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
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
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
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
fn build_logger_rejects_queue_capacity_above_fixed_maximum() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let config = LoggingConfig {
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: temp.path().join("over-max.log.jsonl"),
            queue_capacity: MAX_FILE_SINK_QUEUE_ENTRIES + 1,
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
    };
    let error = build_logger(&config, "root".into())
        .err()
        .expect("queue_capacity above the fixed maximum should fail")
        .to_string();
    assert!(error.contains("exceeds maximum"));
}

#[test]
fn build_logger_allows_queue_capacity_at_fixed_maximum() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let config = LoggingConfig {
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: temp.path().join("within-max.log.jsonl"),
            queue_capacity: MAX_FILE_SINK_QUEUE_ENTRIES,
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
    };
    assert!(
        build_logger(&config, "root".into()).is_ok(),
        "queue_capacity at the fixed maximum should build"
    );
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
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
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
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
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
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
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
                ..FileLogSinkConfig::default()
            }),
            LogSinkConfig::File(FileLogSinkConfig {
                path: path_b.clone(),
                level: LogLevel::Info,
                format: LogFormat::Human,
                ..FileLogSinkConfig::default()
            }),
        ],
        ..default_config()
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

#[test]
fn periodic_flush_applies_to_all_file_sinks() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path_a = temp.path().join("flush-a.log.jsonl");
    let path_b = temp.path().join("flush-b.log.jsonl");
    // Small global interval; rely solely on the periodic timer (no explicit flush/shutdown).
    let config = LoggingConfig {
        flush_interval_millis: 20,
        sinks: vec![
            LogSinkConfig::File(FileLogSinkConfig {
                path: path_a.clone(),
                ..FileLogSinkConfig::default()
            }),
            LogSinkConfig::File(FileLogSinkConfig {
                path: path_b.clone(),
                ..FileLogSinkConfig::default()
            }),
        ],
        ..default_config()
    };
    let runtime = init_logging(&config).unwrap();
    log::info!(target: "nemo_relay.server", event = "periodic_flush"; "written to both sinks");

    // A single logger-level flush period must drain every sink without an explicit flush.
    let contents_a = wait_for_log_line(&path_a, |c| c.contains("periodic_flush"));
    let contents_b = wait_for_log_line(&path_b, |c| c.contains("periodic_flush"));
    assert!(
        contents_a.contains("periodic_flush"),
        "sink A was not periodically flushed"
    );
    assert!(
        contents_b.contains("periodic_flush"),
        "sink B was not periodically flushed"
    );
    runtime.shutdown();
}

#[test]
fn file_sink_creates_missing_parent_directories() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("nested/dir/relay.log.jsonl");
    let config = LoggingConfig {
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: path.clone(),
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
    };
    let result = build_logger(&config, "root".into());
    assert!(
        result.is_ok(),
        "expected nested sink path to build; got {:?}",
        result.err()
    );
}

#[test]
fn log_level_parse_maps_valid_inputs_and_rejects_unknown() {
    let valid = [
        ("error", LogLevel::Error),
        ("warn", LogLevel::Warn),
        ("warning", LogLevel::Warn),
        ("INFO", LogLevel::Info),
        (" debug ", LogLevel::Debug),
        ("trace", LogLevel::Trace),
    ];
    for (input, expected) in valid {
        assert_eq!(
            LogLevel::parse(input).unwrap(),
            expected,
            "input={input:?} should map to {expected:?}"
        );
    }
    assert!(LogLevel::parse("verbose").is_err());
}

#[test]
fn log_format_parse_maps_valid_inputs_and_rejects_unknown() {
    let valid = [
        ("human", LogFormat::Human),
        ("jsonl", LogFormat::Jsonl),
        ("json", LogFormat::Jsonl),
        ("JSONL", LogFormat::Jsonl),
        (" human ", LogFormat::Human),
    ];
    for (input, expected) in valid {
        assert_eq!(
            LogFormat::parse(input).unwrap(),
            expected,
            "input={input:?} should map to {expected:?}"
        );
    }
    assert!(LogFormat::parse("yaml").is_err());
}

#[test]
fn toml_logging_configuration_rejects_unknown_fields() {
    let unknown_logging_field = LoggingConfig::from_toml_document(
        r#"
[logging]
levle = "debug"
"#,
    )
    .unwrap_err()
    .to_string();
    assert!(unknown_logging_field.contains("unknown field `levle`"));

    let unknown_sink_field = LoggingConfig::from_toml_document(
        r#"
[logging]

[[logging.sinks]]
path = "relay.log.jsonl"
queue_capcity = 32
"#,
    )
    .unwrap_err()
    .to_string();
    assert!(unknown_sink_field.contains("unknown field `queue_capcity`"));
}

#[test]
fn non_string_fields_are_coerced_to_json_strings() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("typed-fields.log.jsonl");
    let config = LoggingConfig {
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: path.clone(),
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
    };
    let runtime = init_logging(&config).unwrap();
    // Numeric and boolean fields go through the same stringifying path as everything else.
    log::info!(target: "nemo_relay.server", event = "typed_fields", count = 42, ok = true; "coercion");
    runtime.logger.flush();
    let contents = wait_for_log_line(&path, |c| c.contains("typed_fields"));
    runtime.shutdown();

    let line = contents
        .lines()
        .find(|line| line.contains("typed_fields"))
        .expect("typed_fields record should be present");
    let record: Value = serde_json::from_str(line).unwrap();
    // Operational logging coerces every field value to a string by design.
    assert_eq!(record["fields"]["count"], Value::String("42".into()));
    assert_eq!(record["fields"]["ok"], Value::String("true".into()));
}

#[test]
fn build_logger_rejects_zero_queue_capacity() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let config = LoggingConfig {
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: temp.path().join("zero-queue.log.jsonl"),
            queue_capacity: 0,
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
    };
    let error = build_logger(&config, "root".into())
        .err()
        .expect("zero queue_capacity should fail")
        .to_string();
    assert!(error.contains("queue_capacity must be greater than 0"));
}

#[test]
fn shutdown_flushes_file_sink_when_periodic_flush_disabled() {
    let _lock = lock_logging_tests();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("no-periodic-flush.log.jsonl");
    let config = LoggingConfig {
        // Periodic flush disabled: only the shutdown drain should reach disk.
        flush_interval_millis: 0,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: path.clone(),
            ..FileLogSinkConfig::default()
        })],
        ..default_config()
    };
    let runtime = init_logging(&config).unwrap();
    log::info!(target: "nemo_relay.server", event = "no_periodic"; "flushed on shutdown");
    // No explicit flush() and no periodic flush; shutdown must still drain the record.
    runtime.shutdown();
    let contents = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        contents.contains("no_periodic"),
        "shutdown should drain the sink even when periodic flush is disabled"
    );
}

#[test]
fn dropping_stale_runtime_preserves_current_logger_and_final_drop_detaches() {
    let _lock = lock_logging_tests();
    let stale = init_logging(&default_config()).unwrap();
    let current = init_logging(&default_config()).unwrap();

    // Dropping the older runtime must not detach the newer (currently installed) logger.
    drop(stale);
    let installed = spdlog::log_crate_proxy().swap_logger(None);
    match &installed {
        Some(logger) => assert!(
            std::sync::Arc::ptr_eq(logger, &current.logger),
            "proxy should still hold the current runtime's logger after a stale drop"
        ),
        None => panic!("proxy lost its logger when a stale runtime was dropped"),
    }
    // Restore so the current runtime still sees itself as the installed receiver.
    spdlog::log_crate_proxy().set_logger(installed);

    // Dropping the current runtime detaches the proxy.
    drop(current);
    assert!(
        spdlog::log_crate_proxy().swap_logger(None).is_none(),
        "proxy should be empty after the current runtime is dropped"
    );
}

#[test]
fn configure_rejects_preinstalled_foreign_logger() {
    if std::env::var_os(FOREIGN_LOGGER_CHILD_ENV).is_some() {
        log::set_logger(&FOREIGN_LOGGER).expect("foreign logger should install in child process");
        let error = LoggingRuntime::configure(default_config())
            .err()
            .expect("Relay logging should reject a foreign global logger");
        assert!(
            error
                .to_string()
                .contains("process-global log facade is already initialized by another logger"),
            "{error}"
        );
        return;
    }

    let current_thread = std::thread::current();
    let test_name = current_thread.name().expect("test thread should be named");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env(FOREIGN_LOGGER_CHILD_ENV, "1")
        .output()
        .expect("foreign logger child test should start");

    assert!(
        output.status.success(),
        "foreign logger child test failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
