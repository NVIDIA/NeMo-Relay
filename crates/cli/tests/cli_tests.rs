// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI-level gateway coverage tests.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

fn gateway_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nemo-relay")
}

fn toml_basic_string(value: &str) -> String {
    let escaped = value
        .chars()
        .map(|character| match character {
            '\\' => "\\\\".to_string(),
            '"' => "\\\"".to_string(),
            '\n' => "\\n".to_string(),
            '\t' => "\\t".to_string(),
            '\r' => "\\r".to_string(),
            '\u{08}' => "\\b".to_string(),
            '\u{0c}' => "\\f".to_string(),
            '\u{00}'..='\u{1f}' | '\u{7f}' => {
                format!("\\u{:04X}", character as u32)
            }
            character => character.to_string(),
        })
        .collect::<String>();
    format!("\"{escaped}\"")
}

#[test]
fn toml_basic_string_escapes_toml_control_characters() {
    assert_eq!(
        toml_basic_string("a\\b\"c\nd\te\rf\u{08}g\u{0c}h\u{01}\u{7f}"),
        "\"a\\\\b\\\"c\\nd\\te\\rf\\bg\\fh\\u0001\\u007F\""
    );
}

#[test]
fn cli_help_exits_successfully() {
    let output = Command::new(gateway_bin()).arg("--help").output().unwrap();

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Coding-agent gateway"));
}

#[test]
fn cli_version_exits_successfully() {
    let output = Command::new(gateway_bin())
        .arg("--version")
        .output()
        .unwrap();

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("nemo-relay "));
}

#[test]
fn cli_agents_json_emits_supported_agent_shapes() {
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(gateway_bin())
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["agents", "--json"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let agents = parsed.as_array().unwrap();
    assert!(agents.iter().any(|agent| agent["name"] == "codex"));
    assert!(agents.iter().all(|agent| agent["status"].is_string()));
}

#[test]
fn cli_doctor_json_emits_versioned_report() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(&cwd).unwrap();
    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["doctor", "--json"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert!(parsed["environment"].is_object());
    assert!(parsed["configuration"].is_object());
    assert!(parsed["agents"].is_array());
}

#[test]
fn cli_completions_prints_script_for_requested_shell() {
    let output = Command::new(gateway_bin())
        .args(["completions", "zsh"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("#compdef nemo-relay") || stdout.contains("_nemo-relay"));
}

#[test]
fn cli_plugins_edit_requires_tty() {
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(gateway_bin())
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "edit", "--user"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("requires a TTY"),
        "stderr was:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_pricing_validate_accepts_valid_catalog() {
    let temp = tempfile::tempdir().unwrap();
    let catalog = temp.path().join("pricing.json");
    std::fs::write(&catalog, pricing_catalog_json("test-model")).unwrap();

    let output = Command::new(gateway_bin())
        .args(["pricing", "validate"])
        .arg(&catalog)
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Valid pricing catalog"));
    assert!(stdout.contains("1 entry"));
}

#[test]
fn cli_pricing_validate_rejects_invalid_catalog() {
    let temp = tempfile::tempdir().unwrap();
    let catalog = temp.path().join("pricing.json");
    std::fs::write(
        &catalog,
        r#"{
  "version": 1,
  "entries": [{
    "provider": "test",
    "model_id": "bad-model",
    "prompt_cache": { "read_accounting": "included_in_prompt_tokens" },
    "pricing_as_of": "2026-06-05",
    "pricing_source": "test"
  }]
}"#,
    )
    .unwrap();

    let output = Command::new(gateway_bin())
        .args(["pricing", "validate"])
        .arg(&catalog)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid pricing catalog"));
    assert!(stderr.contains("rates or rate_schedule"));
}

#[test]
fn cli_pricing_init_creates_project_pricing_component() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    std::fs::create_dir_all(&project).unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&project)
        .args(["pricing", "init", "--project"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let path = project.join(".nemo-relay/plugins.toml");
    let rendered = std::fs::read_to_string(path).unwrap();
    assert!(rendered.contains("kind = \"pricing\""));
    assert!(!rendered.contains("include_bundled"));
}

#[test]
fn cli_pricing_add_source_validates_and_updates_user_plugin_config() {
    let temp = tempfile::tempdir().unwrap();
    let catalog = temp.path().join("pricing.json");
    std::fs::write(&catalog, pricing_catalog_json("custom-model")).unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::copy(&catalog, cwd.join("pricing.json")).unwrap();
    let canonical = std::fs::canonicalize(cwd.join("pricing.json")).unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["pricing", "add-source"])
        .arg("pricing.json")
        .output()
        .unwrap();

    assert!(output.status.success());
    let rendered = std::fs::read_to_string(
        temp.path()
            .join("xdg")
            .join("nemo-relay")
            .join("plugins.toml"),
    )
    .unwrap();
    assert!(rendered.contains("kind = \"pricing\""));
    assert!(rendered.contains("type = \"file\""));
    assert!(rendered.contains(canonical.to_str().unwrap()));
}

#[test]
fn cli_pricing_resolve_reports_source_match_and_estimate() {
    let temp = tempfile::tempdir().unwrap();
    let catalog = temp.path().join("pricing.json");
    let xdg = temp.path().join("xdg/nemo-relay");
    let project = temp.path().join("project");
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(&catalog, pricing_catalog_json("custom-model")).unwrap();
    std::fs::write(
        xdg.join("plugins.toml"),
        format!(
            r#"
[[components]]
kind = "pricing"

[components.config]
[[components.config.sources]]
type = "file"
path = {}
"#,
            toml_basic_string(&catalog.display().to_string())
        ),
    )
    .unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&project)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args([
            "pricing",
            "resolve",
            "custom-model",
            "--provider",
            "test",
            "--prompt-tokens",
            "1000",
            "--completion-tokens",
            "500",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr was:\n{}\nstdout was:\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Resolved pricing"));
    assert!(stdout.contains(&format!("source = file:{}", catalog.display())));
    assert!(stdout.contains("provider = test"));
    assert!(stdout.contains("model = custom-model"));
    assert!(stdout.contains("estimated_total"));
    assert!(stdout.contains("currency = USD"));
}

#[test]
fn cli_pricing_resolve_reports_missing_sources_distinctly() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(&cwd).unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["pricing", "resolve", "custom-model"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no pricing sources configured"),
        "expected missing pricing source error, got:\n{stderr}"
    );
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
fn cli_help_lists_plugin_install_commands() {
    let output = Command::new(gateway_bin()).arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for command in ["install", "uninstall"] {
        assert!(
            stdout.contains(&format!("  {command}")),
            "expected `--help` to list `{command}` subcommand, got:\n{stdout}"
        );
    }
}

#[test]
fn cli_install_dry_run_plans_local_codex_marketplace() {
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(gateway_bin())
        .args([
            "install",
            "codex",
            "--dry-run",
            "--skip-doctor",
            "--install-dir",
        ])
        .arg(temp.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("codex-marketplace"),
        "stdout was:\n{stdout}"
    );
    assert!(
        stdout.contains("codex plugin marketplace add"),
        "stdout was:\n{stdout}"
    );
    assert!(
        stdout.contains("configure Codex provider and hook-supervised lazy startup"),
        "stdout was:\n{stdout}"
    );
}

#[test]
fn cli_doctor_plugin_help_accepts_plugin_flag() {
    let output = Command::new(gateway_bin())
        .args(["doctor", "--help"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--plugin"), "stdout was:\n{stdout}");
}

#[test]
fn cli_doctor_plugin_accepts_json_flag() {
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(gateway_bin())
        .args(["doctor", "--plugin", "codex", "--json", "--install-dir"])
        .arg(temp.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("cannot be used with"),
        "stderr was:\n{stderr}"
    );
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
fn cli_hermes_easy_path_invokes_setup_when_no_config_found() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(&cwd).unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
        .arg("hermes")
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "Hermes easy path should exit non-zero when no config + no TTY for setup"
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
    std::fs::create_dir_all(cwd.join(".nemo-relay")).unwrap();
    std::fs::write(cwd.join(".nemo-relay/config.toml"), "[upstream]\n").unwrap();

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
fn cli_bare_invocation_reports_invalid_config_resolution() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(cwd.join(".nemo-relay")).unwrap();
    std::fs::write(cwd.join(".nemo-relay/config.toml"), "[upstream]\n").unwrap();
    std::fs::write(cwd.join(".nemo-relay/plugins.toml"), "components = [\n").unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "bare invocation should fail doctor when config resolution fails"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Configuration"));
    assert!(stdout.contains("Resolution"));
    assert!(stdout.contains("invalid plugin TOML"));
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
    let xdg = temp.path().join("xdg/nemo-relay");
    std::fs::create_dir_all(project.join(".nemo-relay")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::create_dir_all(&xdg).unwrap();
    std::fs::write(
        project.join(".nemo-relay/config.toml"),
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
        .env("NEMO_RELAY_GATEWAY_BIND", "127.0.0.1:0")
        .env("NEMO_RELAY_OPENAI_BASE_URL", "http://env-openai")
        .env("NEMO_RELAY_ANTHROPIC_BASE_URL", "http://env-anthropic")
        .args(["run", "--agent", "codex", "--dry-run"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("openai_base_url = http://env-openai"));
    assert!(stdout.contains("anthropic_base_url = http://env-anthropic"));
    assert!(!stdout.contains("atif_dir"));
    assert!(!stdout.contains("openinference_endpoint"));
    assert!(stdout.contains("argv = codex"));
}

#[test]
fn cli_hook_forward_fails_open_without_gateway_url() {
    let mut child = Command::new(gateway_bin())
        .env_remove("NEMO_RELAY_GATEWAY_URL")
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
        .env_remove("NEMO_RELAY_GATEWAY_URL")
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
    assert!(request.contains("x-nemo-relay-config-profile: coverage"));
    assert!(request.contains("x-nemo-relay-gateway-mode: passthrough"));
    assert!(request.contains(r#"{"hook_event_name":"sessionStart"}"#));
}

#[test]
fn cli_hook_forward_hermes_shell_hook_returns_empty_object() {
    let (server_url, received) = spawn_single_request_server(200, r#"{}"#);
    let mut child = Command::new(gateway_bin())
        .args([
            "hook-forward",
            "hermes",
            "--gateway-url",
            &server_url,
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
        .write_all(br#"{"session_id":"smoke-hermes","hook_event_name":"on_session_start"}"#)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let request = received.recv().unwrap();

    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), r#"{}"#);
    assert!(request.contains("POST /hooks/hermes HTTP/1.1"));
    assert!(
        request.contains(r#"{"session_id":"smoke-hermes","hook_event_name":"on_session_start"}"#)
    );
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
fn cli_hook_forward_exits_two_for_guardrail_rejection() {
    let (server_url, received) = spawn_single_request_server(
        403,
        r#"{"error":{"message":"guardrail rejected: blocked by policy","type":"nemo_relay_guardrail_rejected","reason":"blocked by policy"}}"#,
    );
    let mut child = Command::new(gateway_bin())
        .args(["hook-forward", "codex", "--gateway-url", &server_url])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{}").unwrap();
    let output = child.wait_with_output().unwrap();
    let request = received.recv().unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(request.contains("POST /hooks/codex HTTP/1.1"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("blocked by policy"));
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

fn pricing_catalog_json(model_id: &str) -> String {
    format!(
        r#"{{
  "version": 1,
  "entries": [{{
    "provider": "test",
    "model_id": "{model_id}",
    "rates": {{
      "input_per_million": 1.0,
      "output_per_million": 2.0,
      "cache_read_per_million": 0.1
    }},
    "prompt_cache": {{ "read_accounting": "included_in_prompt_tokens" }},
    "pricing_as_of": "2026-06-05",
    "pricing_source": "test"
  }}]
}}"#
    )
}
