// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_support::EnvScope;
use clap::Parser;
use std::ffi::OsStr;

struct SystemConfigGuard(Option<std::path::PathBuf>);
impl Drop for SystemConfigGuard {
    fn drop(&mut self) {
        runtime_configuration::set_test_system_config_dir(self.0.take());
    }
}
#[test]
fn diagnostic_boolean_spellings_preserve_hook_parsing_and_cli_precedence() {
    for (value, expected) in [
        ("1", true),
        ("yes", true),
        ("on", true),
        ("TRUE", true),
        ("0", false),
        ("no", false),
        ("off", false),
        ("FALSE", false),
    ] {
        let _env = EnvScope::set(&[
            ("NEMO_RELAY_DIAGNOSTICS", Some(OsStr::new(value))),
            ("NEMO_RELAY_DIAGNOSTICS_HTTP", Some(OsStr::new(value))),
        ]);
        let cli = Cli::try_parse_from(["nemo-relay", "hook-forward", "claude"]).unwrap();
        assert_eq!(cli.diagnostics, Some(expected));
        assert_eq!(cli.diagnostics_http, Some(expected));
        let cli = Cli::try_parse_from([
            "nemo-relay",
            "--diagnostics=off",
            "--diagnostics-http=on",
            "hook-forward",
            "claude",
        ])
        .unwrap();
        assert_eq!(cli.diagnostics, Some(false));
        assert_eq!(cli.diagnostics_http, Some(true));
        let cli = Cli::try_parse_from([
            "nemo-relay",
            "--diagnostics",
            "--diagnostics-http",
            "hook-forward",
            "claude",
        ])
        .unwrap();
        assert_eq!(cli.diagnostics, Some(true));
        assert_eq!(cli.diagnostics_http, Some(true));
    }
    let _env = EnvScope::set(&[
        ("NEMO_RELAY_DIAGNOSTICS", Some(OsStr::new("maybe"))),
        ("NEMO_RELAY_DIAGNOSTICS_HTTP", None),
    ]);
    assert!(Cli::try_parse_from(["nemo-relay", "hook-forward", "claude"]).is_err());
}
#[test]
fn startup_uses_run_selection_then_system_then_explicit_diagnostic_overrides() {
    let temp = tempfile::tempdir().unwrap();
    let explicit = temp.path().join("explicit.toml");
    let run = temp.path().join("run.toml");
    let system = temp.path().join("system");
    std::fs::create_dir(&system).unwrap();
    let _system = SystemConfigGuard(runtime_configuration::set_test_system_config_dir(Some(
        system.clone(),
    )));
    std::fs::write(
        &explicit,
        "[diagnostics]\nenabled = false\n[diagnostics.limits]\nmax_events = 2",
    )
    .unwrap();
    std::fs::write(
        &run,
        "[diagnostics]\nenabled = true\n[diagnostics.limits]\nmax_events = 3",
    )
    .unwrap();
    std::fs::write(
        system.join("config.toml"),
        "[diagnostics.limits]\nmax_events = 5",
    )
    .unwrap();
    let _env = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        (
            "NEMO_RELAY_TEST_SYSTEM_CONFIG_DIR",
            Some(system.as_os_str()),
        ),
        ("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", None),
        ("NEMO_RELAY_DIAGNOSTICS", Some(OsStr::new("false"))),
        ("NEMO_RELAY_DIAGNOSTICS_HTTP", None),
    ]);
    let mut cli = Cli::parse_from([
        "nemo-relay",
        "--config",
        explicit.to_str().unwrap(),
        "run",
        "--config",
        run.to_str().unwrap(),
        "--diagnostics=true",
        "--dry-run",
        "--",
        "example",
    ]);
    let config = prepare_diagnostic_configuration(&mut cli).unwrap().unwrap();
    assert!(config.enabled);
    assert_eq!(config.limits.max_events, 5);
    assert!(cli.server.prepared.is_some());
}
#[test]
fn inspection_and_repair_skip_ambient_configuration() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("nemo-relay")).unwrap();
    std::fs::write(temp.path().join("nemo-relay/config.toml"), "malformed[").unwrap();
    let _env = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("NEMO_RELAY_DIAGNOSTICS", None),
        ("NEMO_RELAY_DIAGNOSTICS_HTTP", None),
    ]);
    for command in ["config", "diagnostics"] {
        let mut cli = Cli::parse_from(["nemo-relay", command]);
        let config = prepare_diagnostic_configuration(&mut cli).unwrap().unwrap();
        assert!(!config.enabled);
        assert!(cli.server.prepared.is_none());
    }
}
#[test]
fn doctor_keeps_reporting_after_bad_startup_inputs() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("bad.toml");
    std::fs::write(&file, "[diagnostics]\nenabled = 'wrong'").unwrap();
    let _env = EnvScope::set(&[
        ("NEMO_RELAY_DIAGNOSTICS", None),
        ("NEMO_RELAY_DIAGNOSTICS_HTTP", None),
    ]);
    let mut cli = Cli::parse_from([
        "nemo-relay",
        "--config",
        file.to_str().unwrap(),
        "doctor",
        "--offline",
    ]);
    assert!(
        prepare_diagnostic_configuration(&mut cli)
            .unwrap()
            .is_none()
    );
}

#[test]
fn daemon_helpers_preserve_no_ambient_policy() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("nemo-relay")).unwrap();
    std::fs::write(temp.path().join("nemo-relay/config.toml"), "malformed[").unwrap();
    let _env = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("NEMO_RELAY_DIAGNOSTICS", None),
        ("NEMO_RELAY_DIAGNOSTICS_HTTP", None),
    ]);
    let mut cli = Cli::parse_from([
        "nemo-relay",
        "daemon",
        "mcp",
        "--daemon-address",
        "http://127.0.0.1:47632",
    ]);
    let config = prepare_diagnostic_configuration(&mut cli).unwrap().unwrap();
    assert!(!config.enabled);
    assert!(cli.server.prepared.is_none());
}
#[test]
fn managed_worker_ignores_diagnostic_environment_and_cli_overrides() {
    let temp = tempfile::tempdir().unwrap();
    let system = temp.path().join("system");
    std::fs::create_dir(&system).unwrap();
    std::fs::write(
        system.join("config.toml"),
        "[diagnostics]\nenabled = false\nhttp_read = false",
    )
    .unwrap();
    let _env = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        (
            "NEMO_RELAY_TEST_SYSTEM_CONFIG_DIR",
            Some(system.as_os_str()),
        ),
        ("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", None),
        ("NEMO_RELAY_DIAGNOSTICS", Some(OsStr::new("true"))),
        ("NEMO_RELAY_DIAGNOSTICS_HTTP", Some(OsStr::new("true"))),
    ]);
    let mut cli = Cli::parse_from([
        "nemo-relay",
        "--diagnostics=true",
        "daemon",
        "worker",
        "--daemon-address",
        "http://127.0.0.1:47632",
    ]);
    let config = prepare_diagnostic_configuration(&mut cli).unwrap().unwrap();
    assert!(!config.enabled);
    assert!(!config.http_read);
    assert!(cli.server.prepared.is_some());
}

#[test]
fn diagnostic_inspector_requires_explicit_daemon_selection_for_worker_pages() {
    let uuid = uuid::Uuid::now_v7().to_string();
    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "diagnostics",
            "--daemon-url",
            "http://127.0.0.1:48400",
            "--worker",
            &uuid
        ])
        .is_ok()
    );
    assert!(Cli::try_parse_from(["nemo-relay", "diagnostics", "--worker", &uuid]).is_err());
    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "diagnostics",
            "--daemon-url",
            "http://127.0.0.1:48400",
            "--worker",
            &uuid,
            "--capabilities"
        ])
        .is_err()
    );
    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "diagnostics",
            "--daemon-url",
            "http://127.0.0.1:48400",
            "--gateway-url",
            "http://127.0.0.1:4040"
        ])
        .is_err()
    );
}
