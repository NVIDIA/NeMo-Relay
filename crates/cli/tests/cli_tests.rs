// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI-level gateway coverage tests.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

fn gateway_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nemo-flow")
}

#[test]
fn cli_help_exits_successfully() {
    let output = Command::new(gateway_bin()).arg("--help").output().unwrap();

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Coding-agent gateway"));
}

#[test]
fn cli_help_lists_easy_path_agent_shortcuts() {
    let output = Command::new(gateway_bin()).arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for agent in ["claude", "codex", "cursor", "hermes"] {
        assert!(
            stdout.contains(&format!("  {agent}")),
            "expected `--help` to list `{agent}` subcommand, got:\n{stdout}"
        );
    }
}

#[test]
fn cli_easy_path_invokes_setup_when_no_config_found() {
    // When no config exists anywhere, the easy path fires setup. In a non-TTY test
    // context the setup errors with a clear "requires a TTY" message; that's the contract
    // we lock in here. Interactive testing of setup itself lives in the unit tests
    // (build_config, save_config) since spawning real prompt UI from cargo-test is brittle.
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(&cwd).unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
        .arg("claude")
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "easy path should exit non-zero when no config + no TTY for setup"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("setup requires a TTY"),
        "expected non-TTY setup error in stderr, got:\n{stderr}"
    );
}

#[test]
fn cli_bare_invocation_invokes_setup_when_no_config_found() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(&cwd).unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "bare invocation should enter non-TTY setup when no config exists"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("setup requires a TTY"),
        "expected non-TTY setup error in stderr, got:\n{stderr}"
    );
}

#[test]
fn cli_bare_invocation_runs_doctor_when_config_exists() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(cwd.join(".nemo-flow")).unwrap();
    std::fs::write(cwd.join(".nemo-flow/config.toml"), "[observability]\n").unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "bare invocation should run doctor when config exists: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Environment"));
    assert!(stdout.contains("Configuration"));
    assert!(stdout.contains("Agents detected"));
}

#[test]
fn cli_run_dry_run_resolves_config_and_command() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config.toml");
    std::fs::write(
        &config,
        r#"
[upstream]
openai_base_url = "http://file-openai"
anthropic_base_url = "http://file-anthropic"

[observability]
atif_dir = "file-atif"

[export.openinference]
endpoint = "http://otel"

[agents.hermes]
command = "hermes --yolo chat"
"#,
    )
    .unwrap();

    let output = Command::new(gateway_bin())
        .args([
            "--config",
            config.to_str().unwrap(),
            "run",
            "--agent",
            "hermes",
            "--dry-run",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("agent = hermes"));
    assert!(stdout.contains("openai_base_url = http://file-openai"));
    assert!(stdout.contains("argv = hermes --yolo chat"));
}

#[test]
fn cli_run_dry_run_uses_project_user_and_env_config_layers() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    let nested = project.join("nested");
    let xdg = temp.path().join("xdg/nemo-flow");
    std::fs::create_dir_all(project.join(".nemo-flow")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::write(
        project.join(".nemo-flow/config.toml"),
        r#"
[upstream]
openai_base_url = "http://project-openai"
"#,
    )
    .unwrap();
    std::fs::write(
        xdg.join("config.toml"),
        r#"
[upstream]
anthropic_base_url = "http://user-anthropic"

[agents.codex]
command = "codex --full-auto"
"#,
    )
    .unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&nested)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("NEMO_FLOW_GATEWAY_BIND", "127.0.0.1:0")
        .env("NEMO_FLOW_OPENAI_BASE_URL", "http://env-openai")
        .env("NEMO_FLOW_ANTHROPIC_BASE_URL", "http://env-anthropic")
        .env("NEMO_FLOW_ATIF_DIR", "env-atif")
        .env("NEMO_FLOW_OPENINFERENCE_ENDPOINT", "http://env-otel")
        .args(["run", "--agent", "codex", "--dry-run"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("openai_base_url = http://env-openai"));
    assert!(stdout.contains("anthropic_base_url = http://env-anthropic"));
    assert!(stdout.contains("atif_dir = env-atif"));
    assert!(stdout.contains("openinference_endpoint = http://env-otel"));
    assert!(stdout.contains("argv = codex"));
}

#[test]
fn cli_hook_forward_fails_open_without_gateway_url() {
    let mut child = Command::new(gateway_bin())
        .env_remove("NEMO_FLOW_GATEWAY_URL")
        .args(["hook-forward", "codex"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"").unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing gateway URL"));
}

#[test]
fn cli_hook_forward_fails_closed_without_gateway_url() {
    let mut child = Command::new(gateway_bin())
        .env_remove("NEMO_FLOW_GATEWAY_URL")
        .args(["hook-forward", "codex", "--fail-closed"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{}").unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing gateway URL"));
}

#[test]
fn cli_hook_forward_posts_payload_headers_and_prints_response() {
    let (server_url, received) = spawn_single_request_server(200, r#"{"continue":true}"#);
    let mut child = Command::new(gateway_bin())
        .args([
            "hook-forward",
            "codex",
            "--gateway-url",
            &server_url,
            "--atif-dir",
            "atif",
            "--openinference-endpoint",
            "http://otel",
            "--profile",
            "coverage",
            "--session-metadata",
            r#"{"team":"cli"}"#,
            "--plugin-config",
            r#"{"components":[]}"#,
            "--gateway-mode",
            "passthrough",
            "--fail-closed",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"hook_event_name":"sessionStart"}"#)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let request = received.recv().unwrap();

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        r#"{"continue":true}"#
    );
    assert!(request.contains("POST /hooks/codex HTTP/1.1"));
    assert!(request.contains("x-nemo-flow-atif-dir: atif"));
    assert!(request.contains("x-nemo-flow-openinference-endpoint: http://otel"));
    assert!(request.contains("x-nemo-flow-config-profile: coverage"));
    assert!(request.contains("x-nemo-flow-gateway-mode: passthrough"));
    assert!(request.contains(r#"{"hook_event_name":"sessionStart"}"#));
}

#[test]
fn cli_hook_forward_reports_http_failure_when_fail_closed() {
    let (server_url, received) = spawn_single_request_server(503, "unavailable");
    let mut child = Command::new(gateway_bin())
        .args([
            "hook-forward",
            "cursor",
            "--gateway-url",
            &server_url,
            "--fail-closed",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{}").unwrap();
    let output = child.wait_with_output().unwrap();
    let request = received.recv().unwrap();

    assert!(!output.status.success());
    assert!(request.contains("POST /hooks/cursor HTTP/1.1"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("HTTP 503"));
}

#[test]
fn cli_hook_forward_reports_transport_failure_when_fail_closed() {
    let mut child = Command::new(gateway_bin())
        .args([
            "hook-forward",
            "codex",
            "--gateway-url",
            "http://127.0.0.1:1",
            "--fail-closed",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{}").unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("hook forward failed"));
}

fn spawn_single_request_server(
    status: u16,
    body: &'static str,
) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_http_request(&mut stream);
        sender.send(request).unwrap();
        let response = format!(
            "HTTP/1.1 {status} OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
    (format!("http://{address}"), receiver)
}

fn read_http_request(stream: &mut std::net::TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut scratch = [0; 1024];
    loop {
        let read = stream.read(&mut scratch).unwrap();
        assert_ne!(read, 0);
        buffer.extend_from_slice(&scratch[..read]);
        if let Some(header_end) = find_header_end(&buffer) {
            let headers = String::from_utf8_lossy(&buffer[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let expected = header_end + 4 + content_length;
            while buffer.len() < expected {
                let read = stream.read(&mut scratch).unwrap();
                assert_ne!(read, 0);
                buffer.extend_from_slice(&scratch[..read]);
            }
            return String::from_utf8(buffer).unwrap();
        }
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}
