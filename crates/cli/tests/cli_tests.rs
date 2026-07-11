// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI-level gateway coverage tests.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, ChildStdin, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine;
use ring::hmac;
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair};
use sha2::{Digest, Sha256};

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

fn write_dynamic_plugin_manifest(dir: &std::path::Path, plugin_id: &str) {
    write_dynamic_plugin_manifest_with_options(dir, plugin_id, &["plugin_worker"], None);
}

fn write_dynamic_plugin_manifest_with_options(
    dir: &std::path::Path,
    plugin_id: &str,
    capabilities: &[&str],
    signature_ref: Option<&str>,
) {
    std::fs::create_dir_all(dir).unwrap();
    let artifact_body = format!("def register():\n    return {plugin_id:?}\n");
    std::fs::write(dir.join("plugin.py"), &artifact_body).unwrap();
    let digest = format!(
        "sha256:{}",
        Sha256::digest(artifact_body.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let capabilities = capabilities
        .iter()
        .map(|capability| toml_basic_string(capability))
        .collect::<Vec<_>>()
        .join(", ");
    let signature_line = signature_ref
        .map(|signature_ref| format!("signature = {}\n", toml_basic_string(signature_ref)))
        .unwrap_or_default();
    std::fs::write(
        dir.join("relay-plugin.toml"),
        format!(
            r#"manifest_version = 1

[plugin]
id = {plugin_id}
kind = "worker"

[compat]
relay = "0.5"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = [{capabilities}]

[source]
artifact = "plugin.py"

[integrity]
sha256 = {digest}
{signature_line}

[load]
runtime = "command"
entrypoint = "plugin.py"
"#,
            capabilities = capabilities,
            signature_line = signature_line,
            digest = toml_basic_string(&digest),
            plugin_id = toml_basic_string(plugin_id),
        ),
    )
    .unwrap();
}

fn write_detached_ed25519_signature(dir: &std::path::Path, signature_name: &str) -> String {
    std::fs::create_dir_all(dir).unwrap();
    let artifact = std::fs::read(dir.join("plugin.py")).unwrap();
    let pkcs8 =
        Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).expect("generate ed25519 keypair");
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).expect("parse ed25519 keypair");
    let signature = key_pair.sign(&artifact);
    let signature_text = format!(
        "ed25519:{}\n",
        base64::engine::general_purpose::STANDARD.encode(signature.as_ref())
    );
    std::fs::write(dir.join(signature_name), signature_text).unwrap();
    format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key_pair.public_key().as_ref())
    )
}

fn generate_ed25519_public_key() -> String {
    let pkcs8 =
        Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).expect("generate ed25519 keypair");
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).expect("parse ed25519 keypair");
    format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key_pair.public_key().as_ref())
    )
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
    assert!(!String::from_utf8_lossy(&output.stdout).contains("plugin-config"));
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
fn cli_mcp_help_describes_lifecycle_bound_native_gateway() {
    let output = Command::new(gateway_bin())
        .args(["mcp", "--help"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Multiple MCP clients share the gateway"));
    assert!(stdout.contains("127.0.0.1:47632"));
}

#[test]
fn cli_mcp_initializes_and_exits_cleanly_when_stdio_closes() {
    let temp = tempfile::tempdir().unwrap();
    let mut child = Command::new(gateway_bin())
        .args(["--bind", "127.0.0.1:0", "mcp"])
        .env("HOME", temp.path())
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("XDG_RUNTIME_DIR", temp.path().join("runtime"))
        .env("TMPDIR", temp.path())
        .env("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(
            b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\"}}\n",
        )
        .unwrap();

    let output = wait_child_with_output(child);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let response = serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap();
    assert_eq!(response["id"], serde_json::json!(1));
    assert_eq!(
        response["result"]["serverInfo"]["name"],
        serde_json::json!("nemo-relay")
    );
    let log = std::fs::read_to_string(
        find_runtime_file(temp.path(), "codex-sidecar.log")
            .expect("Codex sidecar log should exist"),
    )
    .unwrap();
    assert!(log.contains("Gateway        http://127.0.0.1:"));
}

#[test]
fn cli_mcp_does_not_launch_gateway_when_stdio_closes_before_request() {
    let temp = tempfile::tempdir().unwrap();
    let mut child = Command::new(gateway_bin())
        .args(["mcp"])
        .env("HOME", temp.path())
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("XDG_RUNTIME_DIR", temp.path().join("runtime"))
        .env("TMPDIR", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdin.take());

    let output = wait_child_with_output(child);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert!(find_runtime_file(temp.path(), "codex-sidecar.log").is_none());
}

#[cfg(unix)]
#[test]
fn cli_internal_hermes_install_writes_mcp_hooks_trust_and_doctor_ready_state() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let hermes_home = temp.path().join("hermes");
    let xdg = temp.path().join("xdg");
    let runtime = temp.path().join("runtime");
    let bin = temp.path().join("bin");
    for directory in [&home, &hermes_home, &xdg, &runtime, &bin] {
        std::fs::create_dir_all(directory).unwrap();
    }
    let hermes = bin.join("hermes");
    std::fs::write(&hermes, "#!/bin/sh\necho 'Hermes 1.0.0'\n").unwrap();
    std::fs::set_permissions(&hermes, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = std::env::join_paths(std::iter::once(bin.clone()).chain(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    )))
    .unwrap();

    let install = Command::new(gateway_bin())
        .args(["plugin-shim", "install", "hermes"])
        .env("HOME", &home)
        .env("HERMES_HOME", &hermes_home)
        .env("XDG_CONFIG_HOME", &xdg)
        .env("XDG_RUNTIME_DIR", &runtime)
        .env("PATH", &path)
        .env("OPENAI_API_KEY", "not-written-to-config")
        .output()
        .unwrap();
    assert!(
        install.status.success(),
        "{}",
        String::from_utf8_lossy(&install.stderr)
    );

    let config_path = hermes_home.join("config.yaml");
    let config: serde_json::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    let server = &config["mcp_servers"]["nemo-relay"];
    assert_eq!(server["command"], gateway_bin());
    assert_eq!(
        server["args"],
        serde_json::json!(["mcp", "--agent", "hermes"])
    );
    assert_eq!(server["env"]["NEMO_RELAY_GATEWAY_BIND"], "127.0.0.1:47632");
    assert_eq!(server["env"]["OPENAI_API_KEY"], "${OPENAI_API_KEY}");
    assert!(
        !std::fs::read_to_string(&config_path)
            .unwrap()
            .contains("not-written-to-config")
    );
    let command = config["hooks"]["on_session_start"][0]["command"]
        .as_str()
        .unwrap();
    assert!(command.contains("plugin-shim hook hermes"));
    let approvals: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(hermes_home.join("shell-hooks-allowlist.json")).unwrap(),
    )
    .unwrap();
    let approvals = approvals["approvals"].as_array().unwrap();
    assert_eq!(approvals.len(), 13);
    assert!(approvals.iter().all(|entry| entry["command"] == command));

    let relay_config_dir = xdg.join("nemo-relay");
    std::fs::create_dir_all(&relay_config_dir).unwrap();
    std::fs::write(
        relay_config_dir.join("config.toml"),
        format!(
            "[agents.hermes]\ncommand = {:?}\nhooks_path = {:?}\n",
            hermes.display().to_string(),
            config_path.display().to_string()
        ),
    )
    .unwrap();
    let doctor = Command::new(gateway_bin())
        .args(["doctor", "hermes", "--json"])
        .env("HOME", &home)
        .env("HERMES_HOME", &hermes_home)
        .env("XDG_CONFIG_HOME", &xdg)
        .env("XDG_RUNTIME_DIR", &runtime)
        .env("PATH", &path)
        .env("OPENAI_API_KEY", "runtime-only")
        .output()
        .unwrap();
    assert!(
        doctor.status.success(),
        "{}",
        String::from_utf8_lossy(&doctor.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&doctor.stdout).unwrap();
    assert_eq!(report["agents"][0]["name"], "hermes");
    assert_eq!(report["agents"][0]["status"], "pass");
    assert!(
        report["agents"][0]["annotation"]
            .as_str()
            .unwrap()
            .contains("MCP lifecycle")
    );

    let uninstall = Command::new(gateway_bin())
        .args(["plugin-shim", "uninstall", "hermes"])
        .env("HOME", &home)
        .env("HERMES_HOME", &hermes_home)
        .env("XDG_CONFIG_HOME", &xdg)
        .output()
        .unwrap();
    assert!(
        uninstall.status.success(),
        "{}",
        String::from_utf8_lossy(&uninstall.stderr)
    );
    assert!(!config_path.exists());
    assert!(!hermes_home.join("shell-hooks-allowlist.json").exists());
    assert!(!hermes_home.join(".nemo-relay-generation").exists());
}

fn start_mcp_client(temp: &std::path::Path, bind: SocketAddr) -> (Child, ChildStdin) {
    start_mcp_client_for_agent(temp, bind, "codex")
}

fn start_mcp_client_for_agent(
    temp: &std::path::Path,
    bind: SocketAddr,
    agent: &str,
) -> (Child, ChildStdin) {
    start_mcp_client_with_idle_timeout(temp, bind, agent, "1")
}

fn start_mcp_client_with_idle_timeout(
    temp: &std::path::Path,
    bind: SocketAddr,
    agent: &str,
    idle_timeout_secs: &str,
) -> (Child, ChildStdin) {
    let mut child = Command::new(gateway_bin())
        .args(["--bind", &bind.to_string(), "mcp", "--agent", agent])
        .env("HOME", temp)
        .env("XDG_CONFIG_HOME", temp.join("xdg"))
        .env("XDG_RUNTIME_DIR", temp.join("runtime"))
        .env("TMPDIR", temp)
        .env("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", idle_timeout_secs)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    stdin
        .write_all(
            b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\"}}\n",
        )
        .unwrap();
    let (response_tx, response_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut response = String::new();
        let result = BufReader::new(stdout)
            .read_line(&mut response)
            .map(|_| response);
        let _ = response_tx.send(result);
    });
    let response = match response_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(response) => response.unwrap(),
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            panic!("MCP initialization response timed out: {error}");
        }
    };
    let response: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(response["result"]["serverInfo"]["name"], "nemo-relay");
    (child, stdin)
}

#[test]
fn cli_hooks_and_mcp_share_the_same_persistent_identity_for_each_host() {
    for agent in ["codex", "claude", "hermes"] {
        let temp = tempfile::tempdir().unwrap();
        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = probe.local_addr().unwrap();
        drop(probe);
        let gateway_url = format!("http://{address}");
        let mut hook = Command::new(gateway_bin())
            .args(["plugin-shim", "hook", agent, "--gateway-url", &gateway_url])
            .env("HOME", temp.path())
            .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
            .env("XDG_RUNTIME_DIR", temp.path().join("runtime"))
            .env("TMPDIR", temp.path())
            .env("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", "10")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        hook.stdin
            .take()
            .unwrap()
            .write_all(b"{\"session_id\":\"cold-hook\",\"hook_event_name\":\"SessionStart\"}")
            .unwrap();
        let hook_output = wait_child_with_output(hook);
        assert!(
            hook_output.status.success(),
            "{agent} cold hook recovery failed: {}",
            String::from_utf8_lossy(&hook_output.stderr)
        );

        let (mut mcp, mcp_stdin) =
            start_mcp_client_with_idle_timeout(temp.path(), address, agent, "10");
        drop(mcp_stdin);
        assert!(wait_child(&mut mcp).success());

        let owner = wait_for_owned_sidecar(temp.path(), agent, None);
        assert_eq!(owner["url"], gateway_url);
        stop_owned_sidecar(&owner);
        wait_for_port_closed(address);
    }
}

#[derive(Clone, Copy)]
enum FakeBootstrapProof {
    Missing,
    Wrong,
    Valid,
}

fn bootstrap_request_header<'a>(request: &'a str, name: &str) -> Option<&'a str> {
    request.lines().find_map(|line| {
        let (candidate, value) = line.split_once(':')?;
        candidate.eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

fn fake_bootstrap_proof(key: &[u8], fingerprint: &str, nonce: &str) -> String {
    let key = hmac::Key::new(hmac::HMAC_SHA256, key);
    let mut context = hmac::Context::with_key(&key);
    context.update(b"nemo-relay/bootstrap-health/v1\0");
    context.update(fingerprint.as_bytes());
    context.update(&[0]);
    context.update(nonce.as_bytes());
    format!(
        "hmac-sha256:{}",
        context
            .sign()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn run_fake_bootstrap_listener(proof: FakeBootstrapProof) -> (Output, Vec<String>) {
    let temp = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let stopped = Arc::new(AtomicBool::new(false));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_stopped = stopped.clone();
    let server_requests = requests.clone();
    let key_path = temp
        .path()
        .join("xdg")
        .join("nemo-relay")
        .join("bootstrap")
        .join("fingerprint-hmac.key");
    let server = thread::spawn(move || {
        while !server_stopped.load(Ordering::Relaxed) {
            let (mut stream, _) = match listener.accept() {
                Ok(connection) => connection,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                    continue;
                }
                Err(error) => panic!("fake bootstrap listener failed: {error}"),
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let request = read_http_request(&mut stream);
            server_requests.lock().unwrap().push(request.clone());
            if request.starts_with("GET /healthz ") {
                let fingerprint =
                    bootstrap_request_header(&request, "x-nemo-relay-bootstrap-fingerprint")
                        .unwrap();
                let nonce =
                    bootstrap_request_header(&request, "x-nemo-relay-bootstrap-nonce").unwrap();
                let proof_header = match proof {
                    FakeBootstrapProof::Missing => String::new(),
                    FakeBootstrapProof::Wrong => {
                        "X-NeMo-Relay-Bootstrap-Proof: hmac-sha256:0000000000000000000000000000000000000000000000000000000000000000\r\n".into()
                    }
                    FakeBootstrapProof::Valid => {
                        let key = std::fs::read(&key_path).unwrap();
                        format!(
                            "X-NeMo-Relay-Bootstrap-Proof: {}\r\n",
                            fake_bootstrap_proof(&key, fingerprint, nonce)
                        )
                    }
                };
                let body = format!(
                    r#"{{"status":"ok","service":"nemo-relay","version":"{}","bootstrap_protocol":1}}"#,
                    env!("CARGO_PKG_VERSION")
                );
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\n{proof_header}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .unwrap();
            } else {
                let body = r#"{"continue":true}"#;
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .unwrap();
            }
        }
    });

    let mut child = Command::new(gateway_bin())
        .args([
            "plugin-shim",
            "hook",
            "codex",
            "--gateway-url",
            &format!("http://{address}"),
        ])
        .env("HOME", temp.path())
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("XDG_RUNTIME_DIR", temp.path().join("runtime"))
        .env("TMPDIR", temp.path())
        .env("NEMO_RELAY_FAIL_CLOSED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"{\"session_id\":\"challenge\",\"hook_event_name\":\"SessionStart\"}")
        .unwrap();
    let output = wait_child_with_output(child);
    stopped.store(true, Ordering::Relaxed);
    server.join().unwrap();
    let requests = Arc::try_unwrap(requests).unwrap().into_inner().unwrap();
    (output, requests)
}

#[test]
fn cli_codex_hook_rejects_compatible_json_without_bootstrap_proof() {
    let (output, requests) = run_fake_bootstrap_listener(FakeBootstrapProof::Missing);
    assert!(!output.status.success());
    assert!(requests.iter().all(|request| !request.starts_with("POST ")));
}

#[test]
fn cli_codex_hook_rejects_an_invalid_bootstrap_proof() {
    let (output, requests) = run_fake_bootstrap_listener(FakeBootstrapProof::Wrong);
    assert!(!output.status.success());
    assert!(requests.iter().all(|request| !request.starts_with("POST ")));
}

#[test]
fn cli_codex_hook_reuses_a_listener_with_a_valid_bootstrap_proof() {
    let (output, requests) = run_fake_bootstrap_listener(FakeBootstrapProof::Valid);
    assert!(
        output.status.success(),
        "authenticated hook failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        requests
            .iter()
            .any(|request| request.starts_with("POST /hooks/codex "))
    );
}

fn run_codex_hook_with_launch_resolution_error(
    temp: &std::path::Path,
    fail_closed: bool,
    payload: &[u8],
) -> Output {
    let mut command = Command::new(gateway_bin());
    command
        .args([
            "plugin-shim",
            "hook",
            "codex",
            "--gateway-url",
            "http://127.0.0.1:1",
        ])
        .env("HOME", temp)
        .env("XDG_CONFIG_HOME", temp.join("xdg"))
        .env("XDG_RUNTIME_DIR", temp.join("runtime"))
        .env("TMPDIR", temp)
        .env("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS", "not-a-number")
        .env_remove("NEMO_RELAY_FAIL_CLOSED")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if fail_closed {
        command.env("NEMO_RELAY_FAIL_CLOSED", "1");
    }
    let mut child = command.spawn().unwrap();
    child.stdin.take().unwrap().write_all(payload).unwrap();
    wait_child_with_output(child)
}

#[test]
fn cli_codex_hook_launch_resolution_error_respects_forwarding_policy() {
    let temp = tempfile::tempdir().unwrap();

    for fail_closed in [false, true] {
        let output = run_codex_hook_with_launch_resolution_error(temp.path(), fail_closed, b"{}");
        assert_eq!(output.status.success(), !fail_closed);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("gateway identity preflight failed"));
        assert!(stderr.contains("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS"));
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn cli_codex_hook_launch_resolution_error_retains_default_payload_cap() {
    const DEFAULT_HOOK_PAYLOAD_BYTES: usize = 20 * 1024 * 1024;
    let temp = tempfile::tempdir().unwrap();
    let payload = vec![b'x'; DEFAULT_HOOK_PAYLOAD_BYTES + 1];

    for fail_closed in [false, true] {
        let output =
            run_codex_hook_with_launch_resolution_error(temp.path(), fail_closed, &payload);

        assert_eq!(output.status.success(), !fail_closed);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("hook payload exceeds the 20971520-byte limit"));
        assert!(!stderr.contains("gateway identity preflight failed"));
        assert!(output.stdout.is_empty());
    }
}

fn sidecar_address(temp: &std::path::Path, agent: &str) -> SocketAddr {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let log_path = find_runtime_file(temp, &format!("{agent}-sidecar.log"));
        if let Some(log_path) = log_path.as_ref()
            && let Ok(log) = std::fs::read_to_string(log_path)
            && let Some(address) = log.lines().find_map(|line| {
                line.split("Gateway        http://")
                    .nth(1)
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse().ok())
            })
        {
            return address;
        }
        assert!(
            Instant::now() < deadline,
            "sidecar address was not logged under {}; log: {:?}",
            temp.display(),
            log_path.and_then(|path| std::fs::read_to_string(path).ok())
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn find_runtime_file(root: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(directory).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.file_name().and_then(|value| value.to_str()) == Some(name) {
                return Some(path);
            }
            if path.is_dir() {
                pending.push(path);
            }
        }
    }
    None
}

fn find_runtime_files_matching(
    root: &std::path::Path,
    prefix: &str,
    suffix: &str,
) -> Vec<std::path::PathBuf> {
    let mut matches = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(suffix))
            {
                matches.push(path);
            }
        }
    }
    matches
}

fn wait_child(child: &mut Child) -> ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let mut stderr = String::new();
            if let Some(mut child_stderr) = child.stderr.take() {
                let _ = child_stderr.read_to_string(&mut stderr);
            }
            panic!("child process did not exit within 10 seconds; stderr: {stderr}");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_child_with_output(mut child: Child) -> Output {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "child process did not exit within 10 seconds; stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_port_closed(address: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if TcpStream::connect_timeout(&address, Duration::from_millis(100)).is_err() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "shared gateway remained bound after the final MCP client and idle timeout"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_owned_sidecar(
    temp: &std::path::Path,
    agent: &str,
    previous_pid: Option<u64>,
) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        for path in find_runtime_files_matching(temp, &format!("{agent}-sidecar"), ".owner.json") {
            if let Ok(raw) = std::fs::read(path)
                && let Ok(owner) = serde_json::from_slice::<serde_json::Value>(&raw)
                && owner["pid"]
                    .as_u64()
                    .is_some_and(|pid| Some(pid) != previous_pid)
            {
                return owner;
            }
        }
        assert!(
            Instant::now() < deadline,
            "owned {agent} sidecar was not published under {}",
            temp.display()
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn stop_owned_sidecar(owner: &serde_json::Value) {
    let address = owner["url"]
        .as_str()
        .unwrap()
        .strip_prefix("http://")
        .unwrap()
        .parse::<SocketAddr>()
        .unwrap();
    let token = owner["shutdown_token"].as_str().unwrap();
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .write_all(
            format!(
                "POST /bootstrap/shutdown HTTP/1.1\r\nHost: {address}\r\nX-NeMo-Relay-Bootstrap-Token: {token}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .as_bytes(),
        )
        .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 204"), "{response}");
}

fn relay_health(address: SocketAddr) -> serde_json::Value {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .write_all(
            format!("GET /healthz HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    serde_json::from_str(response.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

#[test]
fn cli_mcp_clients_share_gateway_until_final_idle_shutdown() {
    for (first_agent, second_agent) in [
        ("codex", "claude"),
        ("claude", "codex"),
        ("codex", "hermes"),
        ("hermes", "codex"),
        ("claude", "hermes"),
        ("hermes", "claude"),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let (mut first, first_stdin) =
            start_mcp_client_for_agent(temp.path(), "127.0.0.1:0".parse().unwrap(), first_agent);
        let address = sidecar_address(temp.path(), first_agent);
        let (mut second, second_stdin) =
            start_mcp_client_for_agent(temp.path(), address, second_agent);

        drop(first_stdin);
        assert!(wait_child(&mut first).success());
        let health = relay_health(address);
        assert_eq!(health["service"], "nemo-relay");
        assert_eq!(health["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(health["bootstrap_protocol"], 1);
        assert!(
            find_runtime_file(temp.path(), &format!("{second_agent}-sidecar.log")).is_none(),
            "the second MCP client should adopt the first host's gateway"
        );

        drop(second_stdin);
        assert!(wait_child(&mut second).success());
        wait_for_port_closed(address);
    }
}

#[test]
fn cli_mcp_restarts_one_stopped_gateway_then_fails_after_the_second_stop() {
    let temp = tempfile::tempdir().unwrap();
    let (mut client, _stdin) = start_mcp_client(temp.path(), "127.0.0.1:0".parse().unwrap());
    let first = wait_for_owned_sidecar(temp.path(), "codex", None);
    let first_pid = first["pid"].as_u64().unwrap();

    stop_owned_sidecar(&first);
    let second = wait_for_owned_sidecar(temp.path(), "codex", Some(first_pid));
    assert_ne!(second["pid"], first["pid"]);

    stop_owned_sidecar(&second);
    let status = wait_child(&mut client);
    assert!(
        !status.success(),
        "MCP client unexpectedly restarted the shared gateway twice"
    );
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
fn cli_plugins_validate_json_emits_versioned_success_output() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins").join("acme");
    write_dynamic_plugin_manifest(&plugin_dir, "acme.cli-json");

    let output = Command::new(gateway_bin())
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "validate"])
        .arg(&plugin_dir)
        .arg("--json")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["command"], "plugins validate");
    assert_eq!(parsed["data"]["target_kind"], "path");
    assert_eq!(parsed["data"]["resolved_plugin_id"], "acme.cli-json");
    assert_eq!(parsed["data"]["valid"], true);
    assert_eq!(parsed["data"]["policy_state"], "valid");
    assert_eq!(parsed["data"]["startup_class"], "optional");
    assert_eq!(parsed["data"]["attestation_mode"], "integrity_only");
}

#[test]
fn cli_plugins_list_json_emits_empty_versioned_success_output() {
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(gateway_bin())
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "list", "--json"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["command"], "plugins list");
    assert_eq!(parsed["data"], serde_json::json!([]));
}

#[test]
fn cli_plugins_inspect_json_missing_plugin_emits_not_found_error() {
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(gateway_bin())
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "inspect", "missing.plugin", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["ok"], false);
    assert_eq!(parsed["command"], "plugins inspect");
    assert_eq!(parsed["error"]["code"], "not_found");
    assert_eq!(parsed["error"]["kind"], "not_found");
}

#[test]
fn cli_plugins_list_all_json_includes_tombstoned_records() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    let plugin_dir = cwd.join("plugins").join("acme");
    std::fs::create_dir_all(&cwd).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.tombstoned");

    let add = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&add.stderr)
    );

    let remove = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "remove", "acme.tombstoned"])
        .output()
        .unwrap();
    assert!(
        remove.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&remove.stderr)
    );

    let list = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "list", "--all", "--json"])
        .output()
        .unwrap();

    assert!(
        list.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&list.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&list.stdout).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["command"], "plugins list");
    assert_eq!(parsed["data"][0]["id"], "acme.tombstoned");
    assert_eq!(parsed["data"][0]["tombstoned"], true);
    assert_eq!(parsed["data"][0]["runtime_state"], "tombstoned");
    assert_eq!(parsed["data"][0]["policy_state"], "valid");
    assert_eq!(parsed["data"][0]["startup_class"], "optional");
    assert_eq!(parsed["data"][0]["attestation_mode"], "integrity_only");
}

#[test]
fn cli_plugins_validate_json_reports_blocked_policy_for_path_target() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins").join("acme");
    let xdg = temp.path().join("xdg");
    let user_config_dir = xdg.join("nemo-relay");
    std::fs::create_dir_all(&user_config_dir).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.cli-blocked-path");
    std::fs::write(
        user_config_dir.join("plugins.toml"),
        r#"
[plugins.policy.defaults]
allowed = false
"#,
    )
    .unwrap();

    let output = Command::new(gateway_bin())
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
        .args(["plugins", "validate"])
        .arg(&plugin_dir)
        .arg("--json")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["data"]["target_kind"], "path");
    assert_eq!(parsed["data"]["valid"], false);
    assert_eq!(parsed["data"]["policy_state"], "invalid");
    assert_eq!(parsed["data"]["startup_class"], "optional");
    assert_eq!(parsed["data"]["attestation_mode"], "integrity_only");
    assert!(
        parsed["data"]["errors"][0]
            .as_str()
            .unwrap()
            .contains("blocked by host policy")
    );
}

#[test]
fn cli_plugins_validate_json_reports_verified_signature_for_path_target() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins").join("acme");
    let xdg = temp.path().join("xdg");
    let user_config_dir = xdg.join("nemo-relay");
    std::fs::create_dir_all(&user_config_dir).unwrap();
    write_dynamic_plugin_manifest_with_options(
        &plugin_dir,
        "acme.cli-signed-path",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    let trusted_public_key = write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");
    std::fs::write(
        user_config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[plugins.policy.defaults]\n",
                "attestation = \"signature_required\"\n",
                "trusted_public_keys = [{}]\n"
            ),
            toml_basic_string(&trusted_public_key)
        ),
    )
    .unwrap();

    let output = Command::new(gateway_bin())
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
        .args(["plugins", "validate"])
        .arg(&plugin_dir)
        .arg("--json")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["data"]["valid"], true);
    assert_eq!(parsed["data"]["attestation_mode"], "signature_required");
    assert_eq!(parsed["data"]["authenticity_state"], "valid");
}

#[test]
fn cli_plugins_validate_json_reports_invalid_signature_for_wrong_trusted_key() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins").join("acme");
    let xdg = temp.path().join("xdg");
    let user_config_dir = xdg.join("nemo-relay");
    std::fs::create_dir_all(&user_config_dir).unwrap();
    write_dynamic_plugin_manifest_with_options(
        &plugin_dir,
        "acme.cli-signed-wrong-key",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");
    let wrong_public_key = generate_ed25519_public_key();
    std::fs::write(
        user_config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[plugins.policy.defaults]\n",
                "attestation = \"signature_required\"\n",
                "trusted_public_keys = [{}]\n"
            ),
            toml_basic_string(&wrong_public_key)
        ),
    )
    .unwrap();

    let output = Command::new(gateway_bin())
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
        .args(["plugins", "validate"])
        .arg(&plugin_dir)
        .arg("--json")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["data"]["valid"], false);
    assert_eq!(parsed["data"]["attestation_mode"], "signature_required");
    assert_eq!(parsed["data"]["authenticity_state"], "invalid");
    assert!(
        parsed["data"]["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value
                .as_str()
                .unwrap()
                .contains("failed signature verification"))
    );
}

#[test]
fn cli_plugins_list_json_reports_blocked_policy_for_installed_plugin() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    let plugin_dir = cwd.join("plugins").join("acme");
    let config_dir = cwd.join(".nemo-relay");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.cli-blocked-list");

    let add = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&add.stderr)
    );

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n",
                "allowed = false\n"
            ),
            toml_basic_string(plugin_dir.to_string_lossy().as_ref())
        ),
    )
    .unwrap();

    let list = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "list", "--json"])
        .output()
        .unwrap();

    assert!(
        list.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&list.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&list.stdout).unwrap();
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["data"][0]["id"], "acme.cli-blocked-list");
    assert_eq!(parsed["data"][0]["validation_state"], "invalid");
    assert_eq!(parsed["data"][0]["policy_state"], "invalid");
    assert_eq!(parsed["data"][0]["startup_class"], "required");
    assert_eq!(parsed["data"][0]["attestation_mode"], "signature_required");
    assert_eq!(parsed["data"][0]["last_error"]["phase"], "policy");

    let state_path = config_dir.join(".dynamic-plugins.json");
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
    assert_eq!(
        state["records"][0]["status"]["validation"]["policy_satisfied"],
        "invalid"
    );
    assert_eq!(
        state["records"][0]["status"]["last_error"]["phase"],
        "policy"
    );
}

#[test]
fn cli_plugins_list_json_reports_invalid_trust_in_validation_state() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    let plugin_dir = cwd.join("plugins").join("acme");
    let config_dir = cwd.join(".nemo-relay");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.cli-trust-list");

    let add = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&add.stderr)
    );

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n"
            ),
            toml_basic_string(plugin_dir.to_string_lossy().as_ref())
        ),
    )
    .unwrap();

    let list = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "list", "--json"])
        .output()
        .unwrap();

    assert!(
        list.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&list.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&list.stdout).unwrap();
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["data"][0]["id"], "acme.cli-trust-list");
    assert_eq!(parsed["data"][0]["validation_state"], "invalid");
    assert_eq!(parsed["data"][0]["policy_state"], "valid");
    assert_eq!(parsed["data"][0]["attestation_mode"], "signature_required");
    assert_eq!(parsed["data"][0]["last_error"]["phase"], "validation");
}

#[test]
fn cli_plugins_validate_json_reports_blocked_policy_for_installed_id_target() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    let plugin_dir = cwd.join("plugins").join("acme");
    let config_dir = cwd.join(".nemo-relay");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.cli-blocked-id");

    let add = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&add.stderr)
    );

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n",
                "allowed = false\n"
            ),
            toml_basic_string(plugin_dir.to_string_lossy().as_ref())
        ),
    )
    .unwrap();

    let validate = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "validate", "acme.cli-blocked-id", "--json"])
        .output()
        .unwrap();

    assert!(
        validate.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&validate.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&validate.stdout).unwrap();
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["data"]["target_kind"], "plugin_id");
    assert_eq!(parsed["data"]["valid"], false);
    assert_eq!(parsed["data"]["policy_state"], "invalid");
    assert_eq!(parsed["data"]["startup_class"], "required");
    assert_eq!(parsed["data"]["attestation_mode"], "signature_required");
    assert_eq!(parsed["data"]["desired_enabled"], false);
    assert!(
        parsed["data"]["errors"][0]
            .as_str()
            .unwrap()
            .contains("blocked by host policy")
    );
}

#[test]
fn cli_plugins_inspect_json_emits_installed_plugin_details() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    let plugin_dir = cwd.join("plugins").join("acme");
    std::fs::create_dir_all(&cwd).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.inspect-json");

    let add = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&add.stderr)
    );

    let inspect = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "inspect", "acme.inspect-json", "--json"])
        .output()
        .unwrap();

    assert!(
        inspect.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&inspect.stdout).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["command"], "plugins inspect");
    assert_eq!(parsed["target"], "acme.inspect-json");
    assert_eq!(parsed["data"]["id"], "acme.inspect-json");
    assert_eq!(parsed["data"]["kind"], "worker");
    assert_eq!(parsed["data"]["scope"], "project");
    assert_eq!(parsed["data"]["policy_state"], "valid");
    assert_eq!(parsed["data"]["startup_class"], "optional");
    assert_eq!(parsed["data"]["attestation_mode"], "integrity_only");
    assert_eq!(parsed["data"]["host_config_status"], "absent");
    assert!(parsed["data"]["source"]["manifest_ref"].is_string());
}

#[test]
fn cli_plugins_inspect_json_reports_blocked_policy_for_installed_plugin() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    let plugin_dir = cwd.join("plugins").join("acme");
    let config_dir = cwd.join(".nemo-relay");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.inspect-blocked");

    let add = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&add.stderr)
    );

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n",
                "allowed = false\n"
            ),
            toml_basic_string(plugin_dir.to_string_lossy().as_ref())
        ),
    )
    .unwrap();

    let validate = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "validate", "acme.inspect-blocked"])
        .output()
        .unwrap();
    assert!(
        validate.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&validate.stderr)
    );

    let inspect = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "inspect", "acme.inspect-blocked", "--json"])
        .output()
        .unwrap();

    assert!(
        inspect.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&inspect.stdout).unwrap();
    assert_eq!(parsed["data"]["id"], "acme.inspect-blocked");
    assert_eq!(parsed["data"]["policy_state"], "invalid");
    assert_eq!(parsed["data"]["startup_class"], "required");
    assert_eq!(parsed["data"]["attestation_mode"], "signature_required");
    assert_eq!(parsed["data"]["status"]["last_error"]["phase"], "policy");
}

#[test]
fn cli_plugins_mutation_commands_emit_terse_confirmation_output() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    let plugin_dir = cwd.join("plugins").join("acme");
    std::fs::create_dir_all(&cwd).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.mutate-output");

    let add = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&add.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&add.stdout).trim(),
        "Added dynamic plugin acme.mutate-output"
    );

    let enable = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "enable", "acme.mutate-output"])
        .output()
        .unwrap();
    assert!(
        enable.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&enable.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&enable.stdout).trim(),
        "Enabled dynamic plugin acme.mutate-output"
    );

    let disable = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "disable", "acme.mutate-output"])
        .output()
        .unwrap();
    assert!(
        disable.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&disable.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&disable.stdout).trim(),
        "Disabled dynamic plugin acme.mutate-output"
    );

    let remove = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "remove", "acme.mutate-output"])
        .output()
        .unwrap();
    assert!(
        remove.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&remove.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&remove.stdout).trim(),
        "Removed dynamic plugin acme.mutate-output"
    );

    let revive = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        revive.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&revive.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&revive.stdout).trim(),
        "Revived dynamic plugin acme.mutate-output"
    );
}

#[test]
fn cli_plugins_enable_tombstoned_plugin_returns_refused_exit_code() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    let plugin_dir = cwd.join("plugins").join("acme");
    std::fs::create_dir_all(&cwd).unwrap();
    write_dynamic_plugin_manifest(&plugin_dir, "acme.tombstone-enable");

    let add = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "add", "--project"])
        .arg(&plugin_dir)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&add.stderr)
    );

    let remove = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "remove", "acme.tombstone-enable"])
        .output()
        .unwrap();
    assert!(
        remove.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&remove.stderr)
    );

    let enable = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["plugins", "enable", "acme.tombstone-enable"])
        .output()
        .unwrap();
    assert_eq!(enable.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&enable.stderr).contains("tombstoned"),
        "stderr was:\n{}",
        String::from_utf8_lossy(&enable.stderr)
    );
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
fn cli_model_pricing_validate_accepts_valid_catalog() {
    let temp = tempfile::tempdir().unwrap();
    let catalog = temp.path().join("pricing.json");
    std::fs::write(&catalog, pricing_catalog_json("test-model")).unwrap();

    let output = Command::new(gateway_bin())
        .args(["model-pricing", "validate"])
        .arg(&catalog)
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Valid model pricing catalog"));
    assert!(stdout.contains("1 entry"));
}

#[test]
fn cli_model_pricing_validate_rejects_invalid_catalog() {
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
        .args(["model-pricing", "validate"])
        .arg(&catalog)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid model pricing catalog"));
    assert!(stderr.contains("rates or rate_schedule"));
}

#[test]
fn cli_model_pricing_init_creates_project_pricing_component() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    std::fs::create_dir_all(&project).unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&project)
        .args(["model-pricing", "init", "--project"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let path = project.join(".nemo-relay/plugins.toml");
    let rendered = std::fs::read_to_string(path).unwrap();
    assert!(rendered.contains("kind = \"pricing\""));
    assert!(!rendered.contains("include_bundled"));
}

#[test]
fn cli_model_pricing_add_source_validates_and_updates_user_plugin_config() {
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
        .args(["model-pricing", "add-source"])
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
fn cli_model_pricing_resolve_reports_source_match_and_estimate() {
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
            "model-pricing",
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
    assert!(stdout.contains("Resolved model pricing"));
    assert!(stdout.contains(&format!("source = file:{}", catalog.display())));
    assert!(stdout.contains("provider = test"));
    assert!(stdout.contains("model = custom-model"));
    assert!(stdout.contains("estimated_total"));
    assert!(stdout.contains("currency = USD"));
}

#[test]
fn cli_model_pricing_resolve_reports_missing_sources_distinctly() {
    let temp = tempfile::tempdir().unwrap();
    let cwd = temp.path().join("workdir");
    std::fs::create_dir_all(&cwd).unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&cwd)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .args(["model-pricing", "resolve", "custom-model"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no model pricing sources configured"),
        "expected missing model pricing source error, got:\n{stderr}"
    );
}

#[test]
fn cli_help_lists_easy_path_agent_shortcuts() {
    let output = Command::new(gateway_bin()).arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for agent in ["claude", "codex", "hermes"] {
        assert!(
            stdout.contains(&format!("  {agent}")),
            "expected `--help` to list `{agent}` subcommand, got:\n{stdout}"
        );
    }
    assert!(!stdout.contains("  cursor"));
}

#[test]
fn cli_rejects_removed_cursor_entry_points() {
    let output = Command::new(gateway_bin()).arg("cursor").output().unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand 'cursor'"));

    let output = Command::new(gateway_bin())
        .args(["hook-forward", "cursor"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value 'cursor'"));
}

#[test]
fn cli_help_lists_model_pricing_command_only() {
    let output = Command::new(gateway_bin()).arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("  model-pricing"),
        "expected `--help` to list `model-pricing` subcommand, got:\n{stdout}"
    );
    assert!(
        !stdout.lines().any(|line| line.starts_with("  pricing")),
        "expected `--help` not to list the old `pricing` subcommand, got:\n{stdout}"
    );

    let old_command = Command::new(gateway_bin()).arg("pricing").output().unwrap();
    assert!(!old_command.status.success());
    assert!(String::from_utf8_lossy(&old_command.stderr).contains("unrecognized subcommand"));

    let model_pricing_help = Command::new(gateway_bin())
        .args(["model-pricing", "--help"])
        .output()
        .unwrap();
    let model_pricing_stdout = String::from_utf8_lossy(&model_pricing_help.stdout);
    for description in [
        "Validate a model pricing catalog JSON file",
        "Initialize model pricing in",
        "Add a model pricing catalog file source",
        "Resolve which model pricing entry matches a model",
    ] {
        assert!(
            model_pricing_stdout.contains(description),
            "expected `model-pricing --help` to include `{description}`, got:\n{model_pricing_stdout}"
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
        stdout.contains("configure Codex provider and trust plugin-owned hooks"),
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
    let xdg = temp.path().join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
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
        .env("XDG_CONFIG_HOME", &xdg)
        .env("HOME", temp.path())
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
fn cli_run_dry_run_rejects_missing_explicit_config() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing-config.toml");

    let output = Command::new(gateway_bin())
        .args([
            "run",
            "--config",
            missing.to_str().unwrap(),
            "--agent",
            "codex",
            "--dry-run",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("does not exist"), "{stderr}");
    assert!(
        stderr.contains(missing.to_string_lossy().as_ref()),
        "{stderr}"
    );
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
    let plugin_config = temp.path().join("override-plugins.toml");
    std::fs::write(
        &plugin_config,
        r#"
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "logs"
filename = "events.jsonl"
mode = "append"
"#,
    )
    .unwrap();

    let output = Command::new(gateway_bin())
        .current_dir(&nested)
        .env("XDG_CONFIG_HOME", temp.path().join("xdg"))
        .env("HOME", temp.path())
        .env("NEMO_RELAY_GATEWAY_BIND", "127.0.0.1:0")
        .env("NEMO_RELAY_OPENAI_BASE_URL", "http://env-openai")
        .env("NEMO_RELAY_ANTHROPIC_BASE_URL", "http://env-anthropic")
        .env("NEMO_RELAY_MAX_HOOK_PAYLOAD_BYTES", "444")
        .env("NEMO_RELAY_MAX_PASSTHROUGH_BODY_BYTES", "555")
        .env("NEMO_RELAY_PLUGIN_CONFIG_PATH", &plugin_config)
        .args(["run", "--agent", "codex", "--dry-run"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("openai_base_url = http://env-openai"));
    assert!(stdout.contains("anthropic_base_url = http://env-anthropic"));
    assert!(stdout.contains("max_hook_payload_bytes = 444"));
    assert!(stdout.contains("max_passthrough_body_bytes = 555"));
    assert!(!stdout.contains("atif_dir"));
    assert!(!stdout.contains("openinference_endpoint"));
    assert!(stdout.contains("argv = codex"));
    let expected_atof_path = std::path::Path::new("logs").join("events.jsonl");
    assert!(stdout.contains(&format!("ATOF {}", expected_atof_path.display())));
}

#[test]
fn cli_run_rejects_zero_body_limit_env() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config.toml");
    std::fs::write(&config, "").unwrap();

    let output = Command::new(gateway_bin())
        .env("NEMO_RELAY_MAX_HOOK_PAYLOAD_BYTES", "0")
        .args([
            "--config",
            config.to_str().unwrap(),
            "run",
            "--agent",
            "codex",
            "--dry-run",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("NEMO_RELAY_MAX_HOOK_PAYLOAD_BYTES"));
    assert!(stderr.contains("greater than 0"));
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
    child.stdin.take().unwrap().write_all(b"{}").unwrap();
    let output = child.wait_with_output().unwrap();
    let request = received.recv().unwrap();

    assert!(!output.status.success());
    assert!(request.contains("POST /hooks/hermes HTTP/1.1"));
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
