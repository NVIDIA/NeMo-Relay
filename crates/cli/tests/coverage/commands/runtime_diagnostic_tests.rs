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
