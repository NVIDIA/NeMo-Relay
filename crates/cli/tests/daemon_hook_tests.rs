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
