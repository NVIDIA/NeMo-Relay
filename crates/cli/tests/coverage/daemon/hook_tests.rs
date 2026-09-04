// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use base64::Engine;

use super::*;

fn valid_token() -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([7_u8; 32])
}

fn capture_server(response: Vec<u8>) -> (String, Arc<Mutex<Vec<u8>>>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let request = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&request);
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut bytes = Vec::new();
        let mut byte = [0_u8; 1];
        while !bytes.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            bytes.push(byte[0]);
        }
        let headers = String::from_utf8_lossy(&bytes);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
            })
            .unwrap();
        let mut body = vec![0_u8; content_length];
        stream.read_exact(&mut body).unwrap();
        bytes.extend_from_slice(&body);
        *captured.lock().unwrap() = bytes;
        stream.write_all(&response).unwrap();
    });
    (format!("http://{address}"), request, handle)
}

#[test]
fn hook_payload_is_bounded_and_empty_input_is_normalized() {
    assert_eq!(read_hook_payload(&b" \n\t"[..]).unwrap(), b"{}");
    assert_eq!(read_hook_payload(&b"{\"x\":1}"[..]).unwrap(), b"{\"x\":1}");

    let oversized = vec![b'x'; crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES + 1];
    let error = read_hook_payload(oversized.as_slice()).unwrap_err();
    assert!(error.to_string().contains("exceeds"), "{error}");
}

#[test]
fn default_failure_policy_is_event_specific() {
    assert!(effective_fail_closed(
        HookFailurePolicy::Default,
        Some(br#"{"hook_event_name":"PreToolUse"}"#),
    ));
    assert!(effective_fail_closed(
        HookFailurePolicy::Default,
        Some(br#"{"hook_event_name":"pre_tool_call"}"#),
    ));
    for event in ["tool_call", "toolCall", "user_bash", "userBash"] {
        let payload = format!(r#"{{"hook_event_name":"{event}"}}"#);
        assert!(
            effective_fail_closed(HookFailurePolicy::Default, Some(payload.as_bytes())),
            "managed Pi policy event must fail closed: {event}"
        );
    }
    assert!(!effective_fail_closed(
        HookFailurePolicy::Default,
        Some(br#"{"hook_event_name":"PostToolUse"}"#),
    ));
    assert!(!effective_fail_closed(
        HookFailurePolicy::FailOpen,
        Some(br#"{"hook_event_name":"PreToolUse"}"#),
    ));
    assert!(effective_fail_closed(
        HookFailurePolicy::FailClosed,
        Some(br#"{"hook_event_name":"Stop"}"#),
    ));
}

#[test]
fn route_token_requires_exactly_256_bits_without_exposing_the_value() {
    let secret = "not-a-route-credential";
    let error = route_token(secret).unwrap_err().to_string();
    assert!(error.contains(CLIENT_TOKEN_ENV), "{error}");
    assert!(!error.contains(secret), "{error}");
    assert!(route_token(&valid_token()).is_ok());
}

#[test]
fn managed_pi_hook_uses_the_existing_root_path() {
    let endpoint = hook_endpoint("https://relay.example.com:443", CodingAgent::Pi).unwrap();
    assert_eq!(endpoint.as_str(), "https://relay.example.com/hooks/pi");
}

#[tokio::test]
async fn hook_forward_uses_the_exact_agent_path_and_one_route_header() {
    let response_body = b"{\"continue\":true}\n";
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response_body.len(),
        String::from_utf8_lossy(response_body)
    );
    let (daemon_address, captured, server) = capture_server(response.into_bytes());
    let options = Options {
        agent: CodingAgent::ClaudeCode,
        daemon_address,
        failure_policy: HookFailurePolicy::FailClosed,
    };

    let body = forward(
        &options,
        b"{\"hook_event_name\":\"Stop\"}".to_vec(),
        route_token(&valid_token()).unwrap(),
    )
    .await
    .unwrap();
    server.join().unwrap();

    assert_eq!(body, response_body);
    let request = String::from_utf8(captured.lock().unwrap().clone()).unwrap();
    assert!(
        request.starts_with("POST /hooks/claude-code HTTP/1.1\r\n"),
        "{request}"
    );
    assert_eq!(
        request
            .lines()
            .filter(|line| {
                line.split_once(':')
                    .is_some_and(|(name, _)| name.eq_ignore_ascii_case(CLIENT_TOKEN_HEADER))
            })
            .count(),
        1,
        "{request}"
    );
    assert!(
        request.ends_with("{\"hook_event_name\":\"Stop\"}"),
        "{request}"
    );
}

#[tokio::test]
async fn guardrail_rejections_are_never_failed_open() {
    let body = r#"{"error":{"type":"nemo_relay_guardrail_rejected","reason":"blocked"}}"#;
    let response = format!(
        "HTTP/1.1 403 Forbidden\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let (daemon_address, _captured, server) = capture_server(response.into_bytes());
    let options = Options {
        agent: CodingAgent::Codex,
        daemon_address,
        failure_policy: HookFailurePolicy::FailOpen,
    };

    let error = forward(
        &options,
        b"{}".to_vec(),
        route_token(&valid_token()).unwrap(),
    )
    .await
    .unwrap_err();
    server.join().unwrap();
    assert_eq!(error.guardrail_rejection_reason(), Some("blocked"));
}
