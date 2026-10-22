// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public executable behavior: diagnostics must preserve ordinary command output and results.
use nemo_relay::diagnostics::DiagnosticSnapshot;
use std::process::Command;

fn command(directory: &std::path::Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_nemo-relay"));
    command
        .current_dir(directory)
        .env("XDG_CONFIG_HOME", directory.join("xdg"));
    for name in [
        "NEMO_RELAY_DIAGNOSTICS",
        "NEMO_RELAY_DIAGNOSTICS_HTTP",
        "NEMO_RELAY_LOG_CONFIG_PATH",
        "NEMO_RELAY_LOG",
        "NEMO_RELAY_LOG_STDERR",
        "NEMO_RELAY_LOG_STDERR_FORMAT",
    ] {
        command.env_remove(name);
    }
    command
}
fn snapshot(stderr: &[u8]) -> DiagnosticSnapshot {
    String::from_utf8_lossy(stderr)
        .lines()
        .find_map(|line| serde_json::from_str(line).ok())
        .expect("explicit final diagnostic snapshot")
}

#[test]
fn final_inspection_preserves_successful_stdout_and_command_exit() {
    let directory = tempfile::tempdir().unwrap();
    let baseline = command(directory.path())
        .args(["completions", "bash"])
        .output()
        .unwrap();
    let inspected = command(directory.path())
        .args([
            "--diagnostics=true",
            "--diagnostics-final",
            "completions",
            "bash",
        ])
        .output()
        .unwrap();
    assert!(baseline.status.success());
    assert_eq!(baseline.status.code(), inspected.status.code());
    assert_eq!(baseline.stdout, inspected.stdout);
    assert!(baseline.stderr.is_empty());
    let page = snapshot(&inspected.stderr);
    assert!(
        page.events
            .iter()
            .any(|event| event.code() == "cli.command_completed")
    );
}

#[test]
fn final_inspection_retains_configuration_failure_and_logging_teardown() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("config.toml");
    std::fs::write(&config, "[upstream]\nresponse_timeout_secs = -1\n").unwrap();
    let run = |enabled: bool| {
        command(directory.path())
            .args([
                "--config",
                config.to_str().unwrap(),
                if enabled {
                    "--diagnostics=true"
                } else {
                    "--diagnostics=false"
                },
                "--diagnostics-final",
                "run",
                "--dry-run",
                "--",
                "example",
            ])
            .output()
            .unwrap()
    };
    let baseline = run(false);
    let inspected = run(true);
    assert!(!baseline.status.success());
    assert_eq!(baseline.status.code(), inspected.status.code());
    assert_eq!(baseline.stdout, inspected.stdout);
    let page = snapshot(&inspected.stderr);
    for code in [
        "cli.configuration_failed",
        "cli.command_failed",
        "logging.shutdown_completed",
    ] {
        assert!(
            page.events.iter().any(|event| event.code() == code),
            "missing {code}"
        );
    }
}
