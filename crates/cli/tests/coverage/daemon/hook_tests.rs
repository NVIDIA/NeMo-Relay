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
        if let Err(error) = stream.write_all(&response) {
            assert!(
                matches!(
                    error.kind(),
                    std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
                ),
                "response write failed: {error}"
            );
        }
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

#[test]
fn managed_hook_never_sends_its_route_credential_over_remote_cleartext() {
    assert!(hook_endpoint("http://relay.example.com:47632", CodingAgent::Pi).is_err());
    assert!(hook_endpoint("http://127.0.0.1:47632", CodingAgent::Pi).is_ok());
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
        &crate::operational::OperationalContext::new(),
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
        &crate::operational::OperationalContext::new(),
    )
    .await
    .unwrap_err();
    server.join().unwrap();
    assert_eq!(error.guardrail_rejection_reason(), Some("blocked"));
}

#[test]
fn hook_validation_covers_malformed_payloads_urls_and_default_events() {
    assert!(read_hook_payload(&[0xff][..]).is_err());
    assert!(!effective_fail_closed(HookFailurePolicy::Default, None));
    assert!(!effective_fail_closed(
        HookFailurePolicy::Default,
        Some(b"not-json")
    ));
    assert!(!effective_fail_closed(
        HookFailurePolicy::Default,
        Some(br#"{"unrelated":true}"#)
    ));

    for address in [
        "not a URL",
        "https://user@relay.example:443",
        "https://relay.example:443/path",
        "https://relay.example:443?query=true",
        "https://relay.example:443/#fragment",
    ] {
        assert!(
            hook_endpoint(address, CodingAgent::Pi).is_err(),
            "accepted {address}"
        );
    }
}

#[test]
fn guardrail_error_decoder_accepts_message_fallback_and_rejects_other_shapes() {
    assert_eq!(
        guardrail_rejection_reason(
            br#"{"error":{"type":"nemo_relay_guardrail_rejected","message":"fallback"}}"#
        ),
        Some("fallback".into())
    );
    for body in [
        &b"not-json"[..],
        &br#"{}"#[..],
        &br#"{"error":{"type":"other","reason":"no"}}"#[..],
        &br#"{"error":{"type":"nemo_relay_guardrail_rejected"}}"#[..],
    ] {
        assert_eq!(guardrail_rejection_reason(body), None);
    }
}

#[test]
fn delivery_failure_policy_wraps_closed_errors_and_swallows_open_errors() {
    assert!(handle_delivery_failure(CliError::Config("closed".into()), true).is_err());
    assert!(handle_delivery_failure(CliError::Config("open".into()), false).is_ok());
}

#[tokio::test]
async fn non_guardrail_http_failures_and_oversized_responses_are_rejected() {
    let response =
        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
    let (daemon_address, _captured, server) = capture_server(response.to_vec());
    let error = forward(
        &Options {
            agent: CodingAgent::Pi,
            daemon_address,
            failure_policy: HookFailurePolicy::FailClosed,
        },
        b"{}".to_vec(),
        route_token(&valid_token()).unwrap(),
        &crate::operational::OperationalContext::new(),
    )
    .await
    .unwrap_err();
    server.join().unwrap();
    assert!(error.to_string().contains("503"));

    let large = vec![b'x'; MAX_HOOK_RESPONSE_BYTES + 1];
    let mut response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        large.len()
    )
    .into_bytes();
    response.extend_from_slice(&large);
    let (daemon_address, _captured, server) = capture_server(response);
    let error = forward(
        &Options {
            agent: CodingAgent::Pi,
            daemon_address,
            failure_policy: HookFailurePolicy::FailClosed,
        },
        b"{}".to_vec(),
        route_token(&valid_token()).unwrap(),
        &crate::operational::OperationalContext::new(),
    )
    .await
    .unwrap_err();
    server.join().unwrap();
    assert!(matches!(error, CliError::PayloadTooLarge(_)));
}
