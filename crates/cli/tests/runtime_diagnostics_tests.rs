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

#[cfg(unix)]
#[tokio::test]
async fn daemon_optional_file_sink_preserves_health_and_reports_diagnostic_failure() {
    use std::process::Stdio;
    use std::time::{Duration, Instant};
    struct Child(std::process::Child);
    impl Drop for Child {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    for rotation in ["", "max_file_size_bytes = 4096\nretained_files = 1"] {
        let directory = tempfile::tempdir().unwrap();
        let blocked = directory.path().join("private-path-sentinel");
        std::fs::write(&blocked, "file").unwrap();
        let healthy = directory.path().join("healthy.jsonl");
        let config = directory.path().join("config.toml");
        std::fs::write(&config, format!(
            "[diagnostics]\nenabled = true\n[daemon.logging]\nlevel = 'info'\nstderr_enabled = false\n[[daemon.logging.sinks]]\npath = {}\nfailure_policy = 'optional'\n{rotation}\n[[daemon.logging.sinks]]\npath = {}\n",
            serde_json::to_string(&blocked.join("relay.jsonl")).unwrap(),
            serde_json::to_string(&healthy).unwrap(),
        )).unwrap();
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let stderr_path = directory.path().join("stderr");
        let mut child = Child(
            command(directory.path())
                .args([
                    "--config",
                    config.to_str().unwrap(),
                    "--diagnostics-final",
                    "daemon",
                    "--port",
                    &port.to_string(),
                ])
                .stdout(Stdio::null())
                .stderr(std::fs::File::create(&stderr_path).unwrap())
                .spawn()
                .unwrap(),
        );
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Ok(response) = client
                .get(format!("http://127.0.0.1:{port}/healthz"))
                .send()
                .await
            {
                assert!(response.status().is_success());
                let health: serde_json::Value = response.json().await.unwrap();
                assert_eq!(health["status"], "ok");
                break;
            }
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "{}",
                std::fs::read_to_string(&stderr_path).unwrap()
            );
            assert!(Instant::now() < deadline, "daemon did not become ready");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        // SAFETY: This is the live daemon child owned by this test. SIGINT exercises normal shutdown.
        assert_eq!(unsafe { libc::kill(child.0.id() as i32, libc::SIGINT) }, 0);
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            assert!(Instant::now() < deadline, "daemon did not shut down");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let stderr = std::fs::read(&stderr_path).unwrap();
        let text = String::from_utf8_lossy(&stderr);
        assert_eq!(
            text.matches("optional logging sink logging.file.0 disabled")
                .count(),
            1
        );
        assert!(!text.contains("private-path-sentinel"));
        let page = snapshot(&stderr);
        assert!(
            page.events
                .iter()
                .any(|event| event.component.key == "logging.file.0"
                    && event.code() == "logging.sink.initialize_failed")
        );
        assert!(
            std::fs::read_to_string(healthy)
                .unwrap()
                .contains("logging_initialized")
        );
    }
}
