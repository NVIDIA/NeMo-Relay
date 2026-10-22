// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Managed hook credential failures at the CLI boundary.

use std::io::Write;
use std::net::TcpListener;
use std::process::{Command, Stdio};

#[test]
fn missing_hook_credential_honors_flags_events_and_environment() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let temp = tempfile::tempdir().unwrap();

    for (agent, pass_through) in [
        ("claude", &br#"{"continue":true}"#[..]),
        ("codex", &b"{}"[..]),
        ("pi", &b"{}"[..]),
    ] {
        for credential in [None, Some("invalid-hook-credential-sentinel")] {
            for (flag, environment, event, blocks) in [
                (Some("--fail-closed"), None, "Stop", true),
                (Some("--fail-open"), Some("1"), "PreToolUse", false),
                (None, None, "PreToolUse", true),
                (None, None, "pre_tool_call", true),
                (None, Some("1"), "Stop", true),
                (None, None, "Stop", false),
            ] {
                let mut command = Command::new(env!("CARGO_BIN_EXE_nemo-relay"));
                command
                    .args(["daemon", "hook", agent, "--daemon-address", &address])
                    .env("XDG_CONFIG_HOME", temp.path())
                    .env("APPDATA", temp.path())
                    .env("LOCALAPPDATA", temp.path())
                    .env_remove("NEMO_RELAY_CLIENT_TOKEN")
                    .env_remove("NEMO_RELAY_FAIL_CLOSED")
                    .env(
                        "ANTHROPIC_CUSTOM_HEADERS",
                        "x-nemo-relay-client-token: model-credential-sentinel",
                    )
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                if let Some(flag) = flag {
                    command.arg(flag);
                }
                if let Some(environment) = environment {
                    command.env("NEMO_RELAY_FAIL_CLOSED", environment);
                }
                if let Some(credential) = credential {
                    command.env("NEMO_RELAY_CLIENT_TOKEN", credential);
                }
                let mut child = command.spawn().unwrap();
                let payload = format!(r#"{{"hook_event_name":"{event}"}}"#);
                child
                    .stdin
                    .take()
                    .unwrap()
                    .write_all(payload.as_bytes())
                    .unwrap();
                let output = child.wait_with_output().unwrap();
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert_eq!(
                    output.status.success(),
                    !blocks,
                    "{agent} {flag:?} {event}: {stderr}"
                );
                if blocks {
                    assert!(output.stdout.is_empty());
                    assert!(stderr.contains("client credential"), "{stderr}");
                } else if matches!(event, "PreToolUse" | "pre_tool_call") {
                    assert!(
                        output.stdout.is_empty(),
                        "permission events must not receive an allow response"
                    );
                } else {
                    assert_eq!(output.stdout, pass_through);
                }
                assert!(!stderr.contains("model-credential-sentinel"), "{stderr}");
                assert!(
                    !stderr.contains("invalid-hook-credential-sentinel"),
                    "{stderr}"
                );
            }
        }
    }
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

/// A daemon running under one config home, standing in for a service account, with hooks sent
/// from other config homes, standing in for the users. The test system config directory plays the
/// administrator-owned `/etc/nemo-relay`.
#[cfg(feature = "__skip-implicit-config")]
struct CrossAccountDaemon {
    temp: tempfile::TempDir,
    daemon: std::process::Child,
    address: String,
    token: String,
}

#[cfg(feature = "__skip-implicit-config")]
impl CrossAccountDaemon {
    fn start() -> Self {
        use base64::Engine;

        let temp = tempfile::tempdir().unwrap();
        for directory in ["daemon", "user", "other", "system"] {
            std::fs::create_dir_all(temp.path().join(directory)).unwrap();
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let system = temp.path().join("system");
            std::fs::set_permissions(system, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let daemon = account_command(temp.path(), "daemon")
            .args(["daemon", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let setup = Self {
            temp,
            daemon,
            address: format!("http://127.0.0.1:{port}"),
            token: base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x5a_u8; 32]),
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(std::time::Instant::now() < deadline, "daemon did not start");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        setup
    }

    fn home(&self, account: &str) -> std::path::PathBuf {
        self.temp.path().join(account)
    }

    fn published(&self) -> std::path::PathBuf {
        self.home("system").join("hook-tls-certificate.pem")
    }

    fn as_account(&self, account: &str) -> Command {
        account_command(self.temp.path(), account)
    }

    fn certificate_of(&self, account: &str) -> String {
        let output = self
            .as_account(account)
            .args(["daemon", "tls-certificate"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    }

    /// Sends one fail-closed hook as the user: exit code 0 proves delivery to the daemon.
    fn hook(&self, agent: &str, event: &str) -> (Option<i32>, String) {
        let mut child = self
            .as_account("user")
            .env("NEMO_RELAY_CLIENT_TOKEN", &self.token)
            .args(["daemon", "hook", agent, "--daemon-address", &self.address])
            .arg("--fail-closed")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let payload = format!(r#"{{"session_id":"cross-account","hook_event_name":"{event}"}}"#);
        child
            .stdin
            .take()
            .unwrap()
            .write_all(payload.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        (output.status.code(), stderr)
    }

    fn user_has_bootstrap_state(&self) -> bool {
        self.home("user")
            .join(".config/nemo-relay/bootstrap")
            .exists()
    }
}

#[cfg(feature = "__skip-implicit-config")]
impl Drop for CrossAccountDaemon {
    fn drop(&mut self) {
        let _ = self.daemon.kill();
        let _ = self.daemon.wait();
    }
}

/// Runs the CLI as `account`: its own home and config directory, sharing the test system config
/// directory that stands in for the administrator-owned one.
#[cfg(feature = "__skip-implicit-config")]
fn account_command(root: &std::path::Path, account: &str) -> Command {
    let home = root.join(account);
    let mut command = Command::new(env!("CARGO_BIN_EXE_nemo-relay"));
    command
        .env("HOME", &home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("APPDATA", &home)
        .env("LOCALAPPDATA", &home)
        .env("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "1")
        .env("NEMO_RELAY_TEST_SYSTEM_CONFIG_DIR", root.join("system"))
        .env_remove("NEMO_RELAY_CLIENT_TOKEN")
        .env_remove("NEMO_RELAY_FAIL_CLOSED");
    command
}

/// Before an administrator publishes the certificate, a fail-closed hook to another account's
/// daemon blocks, says how to fix it, and creates no per-user bootstrap state.
#[cfg(feature = "__skip-implicit-config")]
#[test]
fn hooks_to_another_accounts_daemon_explain_the_missing_published_certificate() {
    let setup = CrossAccountDaemon::start();

    let (code, stderr) = setup.hook("claude", "SessionStart");

    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("no pinned certificate for the daemon"),
        "{stderr}"
    );
    assert!(
        stderr.contains(&setup.published().display().to_string()),
        "{stderr}"
    );
    assert!(
        stderr.contains("nemo-relay daemon tls-certificate"),
        "{stderr}"
    );
    assert!(!setup.user_has_bootstrap_state());
}

/// The exported certificate is the one the running daemon serves. Once it is published, every
/// agent's hooks reach the daemon, still without per-user bootstrap state.
#[cfg(feature = "__skip-implicit-config")]
#[test]
fn hooks_reach_another_accounts_daemon_through_its_published_certificate() {
    let setup = CrossAccountDaemon::start();
    let certificate = setup.certificate_of("daemon");
    assert!(
        certificate.starts_with("-----BEGIN CERTIFICATE-----\n"),
        "{certificate}"
    );
    assert!(!certificate.contains("PRIVATE KEY"));
    assert_eq!(setup.certificate_of("daemon"), certificate);
    std::fs::write(setup.published(), &certificate).unwrap();

    for (agent, event) in [
        ("claude", "SessionStart"),
        ("claude", "PreToolUse"),
        ("codex", "SessionStart"),
        ("codex", "PreToolUse"),
        ("pi", "session_start"),
    ] {
        let (code, stderr) = setup.hook(agent, event);
        assert_eq!(code, Some(0), "{agent} {event}: {stderr}");
        assert!(!stderr.contains("hook failed"), "{agent} {event}: {stderr}");
    }
    assert!(!setup.user_has_bootstrap_state());
}

/// A certificate of any other account does not authenticate the daemon, and on Unix a published
/// certificate that other users could replace is refused.
#[cfg(feature = "__skip-implicit-config")]
#[test]
fn hooks_refuse_a_wrong_or_unprotected_published_certificate() {
    let setup = CrossAccountDaemon::start();
    let certificate = setup.certificate_of("daemon");

    std::fs::write(setup.published(), setup.certificate_of("other")).unwrap();
    let (code, stderr) = setup.hook("codex", "PreToolUse");
    assert_eq!(code, Some(2), "{stderr}");
    assert!(stderr.contains("certificate"), "{stderr}");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(setup.published(), &certificate).unwrap();
        let unprotected = std::fs::Permissions::from_mode(0o666);
        std::fs::set_permissions(setup.published(), unprotected).unwrap();
        let (code, stderr) = setup.hook("codex", "SessionStart");
        assert_eq!(code, Some(2), "{stderr}");
        assert!(
            stderr.contains("must not be writable by group or other users"),
            "{stderr}"
        );
    }
    #[cfg(not(unix))]
    drop(certificate);
}
