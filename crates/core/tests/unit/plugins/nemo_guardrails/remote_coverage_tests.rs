// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Focused remote runtime coverage tests for the NeMo Guardrails plugin component.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::plugins::nemo_guardrails::component::{RailSelector, RemoteBackendConfig};
use tokio_stream::StreamExt;

fn runtime_config(remote: RemoteBackendConfig) -> NeMoGuardrailsConfig {
    NeMoGuardrailsConfig {
        remote: Some(remote),
        ..NeMoGuardrailsConfig::default()
    }
}

fn valid_remote() -> RemoteBackendConfig {
    RemoteBackendConfig {
        endpoint: Some("http://127.0.0.1:1/base/".to_string()),
        config_id: Some("default".to_string()),
        ..RemoteBackendConfig::default()
    }
}

fn valid_runtime() -> RemoteBackendRuntime {
    RemoteBackendRuntime::new(&runtime_config(valid_remote())).unwrap()
}

fn runtime_with_endpoint(endpoint: String) -> RemoteBackendRuntime {
    RemoteBackendRuntime::new(&runtime_config(RemoteBackendConfig {
        endpoint: Some(endpoint),
        timeout_millis: 5_000,
        ..valid_remote()
    }))
    .unwrap()
}

fn simple_chat_request() -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
        }),
    }
}

fn unused_local_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}")
}

fn spawn_json_response(response: Json) -> String {
    spawn_http_response("200 OK", "application/json", response.to_string())
}

fn spawn_http_response(
    status: &'static str,
    content_type: &'static str,
    body: impl Into<Vec<u8>>,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let body = body.into();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        read_http_request(&mut stream);
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        stream.write_all(&body).unwrap();
    });
    format!("http://{address}")
}

fn read_http_request(stream: &mut std::net::TcpStream) {
    let mut request = Vec::new();
    let mut buffer = [0; 1024];
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                request.extend_from_slice(&buffer[..n]);
                if http_request_body_complete(&request) {
                    break;
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                break;
            }
            Err(error) => panic!("failed to read local HTTP request: {error}"),
        }
    }
}

fn http_request_body_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let header_end = header_end + 4;
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    request.len() >= header_end + content_length
}

fn assert_flow_error_contains<T>(result: crate::error::Result<T>, expected: &str) {
    let error = match result {
        Ok(_) => panic!("expected FlowError"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains(expected),
        "expected '{error}' to contain '{expected}'"
    );
}

fn expect_plugin_error_contains<T>(result: PluginResult<T>, expected: &str) {
    let error = match result {
        Ok(_) => panic!("expected PluginError"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains(expected),
        "expected '{error}' to contain '{expected}'"
    );
}

#[test]
fn remote_runtime_new_reports_missing_and_invalid_config() {
    expect_plugin_error_contains(
        RemoteBackendRuntime::new(&NeMoGuardrailsConfig::default()),
        "remote config is required",
    );

    expect_plugin_error_contains(
        RemoteBackendRuntime::new(&runtime_config(RemoteBackendConfig::default())),
        "remote.endpoint is required",
    );

    let mut headers = HashMap::new();
    headers.insert("bad header".to_string(), "value".to_string());
    expect_plugin_error_contains(
        RemoteBackendRuntime::new(&runtime_config(RemoteBackendConfig {
            headers,
            ..valid_remote()
        })),
        "remote.headers contains invalid header name",
    );

    let mut headers = HashMap::new();
    headers.insert("x-valid".to_string(), "bad\r\nvalue".to_string());
    expect_plugin_error_contains(
        RemoteBackendRuntime::new(&runtime_config(RemoteBackendConfig {
            headers,
            ..valid_remote()
        })),
        "remote.headers[x-valid] has an invalid value",
    );
}

#[test]
fn request_body_and_guardrails_config_helpers_cover_defaults() {
    let runtime = valid_runtime();
    assert_eq!(
        runtime.chat_completions_url(),
        "http://127.0.0.1:1/base/v1/chat/completions"
    );

    let invalid_request = LlmRequest {
        headers: Map::new(),
        content: Json::Null,
    };
    assert_flow_error_contains(
        runtime.build_request_body(&invalid_request, false),
        "request content is not an object",
    );

    let defaults = RequestDefaultsConfig {
        context: Some(json!({"tenant": "test"})),
        thread_id: Some("thread-1234567890".to_string()),
        state: Some(json!({"events": []})),
        rails: Some(RequestRailsConfig {
            input: Some(RailSelector::Enabled(true)),
            output: Some(RailSelector::Enabled(true)),
            retrieval: Some(RailSelector::Named(vec!["kb".to_string()])),
            dialog: Some(true),
            tool_input: Some(RailSelector::Named(vec!["tool-in".to_string()])),
            tool_output: Some(RailSelector::Named(vec!["tool-out".to_string()])),
        }),
        llm_params: Some(json!({"temperature": 0.1})),
        llm_output: Some(true),
        output_vars: Some(json!(["answer"])),
        log: Some(json!({"activated_rails": false, "details": true})),
    };

    let llm_guardrails = build_llm_guardrails_config(
        &Some("primary".to_string()),
        &["fallback".to_string()],
        Some(&defaults),
        false,
        true,
    )
    .expect("guardrails config");
    assert_eq!(llm_guardrails["config_id"], json!("primary"));
    assert_eq!(llm_guardrails["config_ids"], json!(["fallback"]));
    assert_eq!(llm_guardrails["context"], json!({"tenant": "test"}));
    assert_eq!(llm_guardrails["thread_id"], json!("thread-1234567890"));
    assert_eq!(
        llm_guardrails["options"]["rails"]["input"],
        Json::Bool(false)
    );
    assert_eq!(
        llm_guardrails["options"]["rails"]["retrieval"],
        json!(["kb"])
    );
    assert_eq!(
        llm_guardrails["options"]["llm_params"],
        json!({"temperature": 0.1})
    );
    assert_eq!(llm_guardrails["options"]["output_vars"], json!(["answer"]));
    assert_eq!(
        build_llm_guardrails_config(&None, &[], None, true, true),
        None
    );

    let tool_input =
        build_tool_check_guardrails_config(RemoteCheckKind::Input, &None, &[], Some(&defaults));
    assert_eq!(
        tool_input["options"]["rails"]["tool_output"],
        json!(["tool-in"])
    );
    assert_eq!(
        tool_input["options"]["log"]["activated_rails"],
        Json::Bool(true)
    );

    let tool_output =
        build_tool_check_guardrails_config(RemoteCheckKind::Output, &None, &[], Some(&defaults));
    assert_eq!(
        tool_output["options"]["rails"]["tool_input"],
        json!(["tool-out"])
    );
}

#[test]
fn named_rail_selector_combinations_are_preserved_for_llm_and_tool_checks() {
    let defaults = RequestDefaultsConfig {
        rails: Some(RequestRailsConfig {
            input: Some(RailSelector::Named(vec![
                "input-a".to_string(),
                "input-b".to_string(),
            ])),
            output: Some(RailSelector::Named(vec!["output-a".to_string()])),
            retrieval: Some(RailSelector::Enabled(false)),
            dialog: Some(false),
            tool_input: Some(RailSelector::Enabled(false)),
            tool_output: Some(RailSelector::Named(vec![
                "tool-output-a".to_string(),
                "tool-output-b".to_string(),
            ])),
        }),
        ..RequestDefaultsConfig::default()
    };

    let llm_guardrails = build_llm_guardrails_config(
        &None,
        &["named-a".to_string(), "named-b".to_string()],
        Some(&defaults),
        true,
        true,
    )
    .expect("guardrails config");
    assert_eq!(llm_guardrails["config_ids"], json!(["named-a", "named-b"]));
    assert_eq!(
        llm_guardrails["options"]["rails"]["input"],
        json!(["input-a", "input-b"])
    );
    assert_eq!(
        llm_guardrails["options"]["rails"]["output"],
        json!(["output-a"])
    );
    assert_eq!(
        llm_guardrails["options"]["rails"]["retrieval"],
        Json::Bool(false)
    );
    assert_eq!(
        llm_guardrails["options"]["rails"]["dialog"],
        Json::Bool(false)
    );

    let tool_input =
        build_tool_check_guardrails_config(RemoteCheckKind::Input, &None, &[], Some(&defaults));
    assert_eq!(
        tool_input["options"]["rails"]["tool_input"],
        Json::Bool(false)
    );
    assert_eq!(
        tool_input["options"]["rails"]["tool_output"],
        Json::Bool(false)
    );

    let tool_output =
        build_tool_check_guardrails_config(RemoteCheckKind::Output, &None, &[], Some(&defaults));
    assert_eq!(
        tool_output["options"]["rails"]["tool_input"],
        json!(["tool-output-a", "tool-output-b"])
    );
    assert_eq!(
        tool_output["options"]["rails"]["tool_output"],
        Json::Bool(false)
    );
}

#[test]
fn request_body_rejects_tool_definitions_and_sets_stream_flag() {
    let runtime = valid_runtime();
    let with_tools = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {"name": "search", "parameters": {"type": "object"}}
            }],
        }),
    };
    assert_flow_error_contains(
        runtime.build_request_body(&with_tools, false),
        "does not support OpenAI tool definitions",
    );

    let with_tool_choice = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": "auto",
        }),
    };
    assert_flow_error_contains(
        runtime.build_request_body(&with_tool_choice, false),
        "does not support OpenAI tool definitions",
    );

    let body = runtime
        .build_request_body(&simple_chat_request(), true)
        .expect("valid request body");
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["guardrails"]["config_id"], json!("default"));
}

#[test]
fn tool_message_helpers_build_guardrails_compatible_chat_payloads() {
    let args = json!({"city": "Phoenix"});
    let result = json!({"forecast": "sunny"});

    let input_messages = tool_input_messages("weather_lookup", &args);
    assert_eq!(
        input_messages[0]["content"],
        json!("Run the tool 'weather_lookup' and validate the result.")
    );
    assert_eq!(
        input_messages[1]["tool_calls"][0]["id"],
        json!("nemo_guardrails_weather_lookup_call")
    );
    assert_eq!(
        input_messages[1]["tool_calls"][0]["function"]["arguments"],
        json!("{\"city\":\"Phoenix\"}")
    );

    let output_messages = tool_output_messages("weather_lookup", &args, &result);
    assert_eq!(output_messages[2]["role"], json!("tool"));
    assert_eq!(
        output_messages[2]["content"],
        json!("{\"forecast\":\"sunny\"}")
    );
}

#[test]
fn modified_tool_argument_parsing_covers_success_and_error_shapes() {
    let response = json!({
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "weather_lookup",
                        "arguments": "{\"city\":\"Paris\"}"
                    }
                }]
            }
        }]
    });
    assert_eq!(
        modified_tool_arguments(&response, "weather_lookup").unwrap(),
        Some(json!({"city": "Paris"}))
    );

    assert_flow_error_contains(
        modified_tool_arguments(&json!({"choices": []}), "weather_lookup"),
        "did not contain choices[0].message",
    );
    assert_flow_error_contains(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"tool_calls": [{}]}}]}),
            "weather_lookup",
        ),
        "without a function payload",
    );
    assert_flow_error_contains(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"tool_calls": [{"function": {}}]}}]}),
            "weather_lookup",
        ),
        "without a function name",
    );
    assert_flow_error_contains(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"tool_calls": [{"function": {"name": "other"}}]}}]}),
            "weather_lookup",
        ),
        "unexpected tool 'other'",
    );
    assert_flow_error_contains(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"tool_calls": [{"function": {"name": "weather_lookup"}}]}}]}),
            "weather_lookup",
        ),
        "without function.arguments",
    );
    assert_flow_error_contains(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"tool_calls": [{"function": {"name": "weather_lookup", "arguments": "not json"}}]}}]}),
            "weather_lookup",
        ),
        "not valid JSON",
    );

    let legacy = json!({
        "choices": [{
            "message": {
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Berlin\"}}"
            }
        }]
    });
    assert_eq!(
        modified_tool_arguments(&legacy, "weather_lookup").unwrap(),
        Some(json!({"city": "Berlin"}))
    );
    assert_eq!(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"content": "not json"}}]}),
            "weather_lookup",
        )
        .unwrap(),
        None
    );
    assert_flow_error_contains(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"content": "{\"tool_name\":\"other\",\"arguments\":{}}"}}]}),
            "weather_lookup",
        ),
        "unexpected tool 'other'",
    );
    assert_eq!(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"content": "[]"}}]}),
            "weather_lookup",
        )
        .unwrap(),
        None
    );
    assert_eq!(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"content": "{\"tool_name\":\"weather_lookup\"}"}}]}),
            "weather_lookup",
        )
        .unwrap(),
        None
    );
}

#[test]
fn modified_tool_payload_helpers_cover_odd_remote_payload_shapes() {
    assert_flow_error_contains(
        first_choice_message(&json!({"choices": [{"message": []}]})).map(|_| ()),
        "did not contain choices[0].message",
    );
    assert_eq!(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"tool_calls": ["not-an-object"]}}]}),
            "weather_lookup",
        )
        .unwrap(),
        None
    );
    assert_flow_error_contains(
        modified_tool_arguments(
            &json!({"choices": [{"message": {"tool_calls": [{"function": {"name": "weather_lookup", "arguments": {"city": "Paris"}}}]}}]}),
            "weather_lookup",
        ),
        "without function.arguments",
    );
    assert_eq!(
        modified_tool_result(
            &json!({"choices": [{"message": {"role": "assistant", "content": "{\"tool_name\":\"weather_lookup\",\"result\":null}"}}]}),
            "weather_lookup",
        )
        .unwrap(),
        Some(Json::Null)
    );
    assert_flow_error_contains(
        modified_tool_result(
            &json!({"choices": [{"message": {"role": "tool", "name": "weather_lookup", "content": {"forecast": "rain"}}}]}),
            "weather_lookup",
        ),
        "without message.content",
    );
}

#[test]
fn modified_tool_result_parsing_covers_success_and_error_shapes() {
    let response = json!({
        "choices": [{
            "message": {
                "role": "tool",
                "name": "weather_lookup",
                "content": "{\"forecast\":\"cloudy\"}"
            }
        }]
    });
    assert_eq!(
        modified_tool_result(&response, "weather_lookup").unwrap(),
        Some(json!({"forecast": "cloudy"}))
    );

    assert_flow_error_contains(
        modified_tool_result(
            &json!({"choices": [{"message": {"role": "tool", "name": "other", "content": "{}"}}]}),
            "weather_lookup",
        ),
        "unexpected tool 'other'",
    );
    assert_flow_error_contains(
        modified_tool_result(
            &json!({"choices": [{"message": {"role": "tool", "name": "weather_lookup"}}]}),
            "weather_lookup",
        ),
        "without message.content",
    );
    assert_flow_error_contains(
        modified_tool_result(
            &json!({"choices": [{"message": {"role": "tool", "name": "weather_lookup", "content": "not json"}}]}),
            "weather_lookup",
        ),
        "not valid JSON",
    );

    let legacy = json!({
        "choices": [{
            "message": {
                "content": "{\"tool_name\":\"weather_lookup\",\"result\":{\"forecast\":\"rain\"}}"
            }
        }]
    });
    assert_eq!(
        modified_tool_result(&legacy, "weather_lookup").unwrap(),
        Some(json!({"forecast": "rain"}))
    );
    assert_eq!(
        modified_tool_result(
            &json!({"choices": [{"message": {"content": "{\"tool_name\":\"weather_lookup\"}"}}]}),
            "weather_lookup",
        )
        .unwrap(),
        None
    );
    assert_eq!(
        modified_tool_result(
            &json!({"choices": [{"message": {"content": "[]"}}]}),
            "weather_lookup",
        )
        .unwrap(),
        None
    );
    assert_eq!(
        modified_tool_result(
            &json!({"choices": [{"message": {"content": "not json"}}]}),
            "weather_lookup",
        )
        .unwrap(),
        None
    );
}

#[test]
fn blocking_and_mark_helpers_cover_optional_payload_shapes() {
    let stopped = json!({
        "guardrails": {"log": {"activated_rails": [{"name": "stop rail", "stop": true}]}}
    });
    assert_eq!(blocking_rail_name(&stopped), Some("stop rail".to_string()));

    let refused = json!({
        "guardrails": {
            "log": {
                "activated_rails": [{
                    "name": "refuse rail",
                    "decisions": ["refuse answer"]
                }]
            }
        }
    });
    assert_eq!(
        blocking_rail_name(&refused),
        Some("refuse rail".to_string())
    );
    assert_eq!(
        blocking_rail_name(
            &json!({"guardrails": {"log": {"activated_rails": [{"name": "allow"}]}}})
        ),
        None
    );

    let mark = remote_mark_data(
        true,
        &Some("primary".to_string()),
        &["fallback".to_string()],
        Some(503),
        Some("redacted".to_string()),
    );
    assert_eq!(mark["stream"], Json::Bool(true));
    assert_eq!(mark["config_id"], json!("primary"));
    assert_eq!(mark["config_ids"], json!(["fallback"]));
    assert_eq!(mark["http_status"], json!(503));
    assert_eq!(mark["error"], json!("redacted"));

    let tool_mark = tool_remote_mark_data(
        RemoteCheckKind::Output,
        "weather_lookup",
        &None,
        &[],
        Some(200),
        None,
    );
    assert_eq!(tool_mark["surface"], json!("tool_output"));
    assert_eq!(tool_mark["tool_name"], json!("weather_lookup"));
    assert_eq!(
        redact_remote_error_payload(500, "sensitive body"),
        "remote request failed with status 500; error body omitted from marks (14 bytes)"
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn remote_execute_reports_non_stream_success_http_errors_and_invalid_json() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    crate::api::runtime::set_thread_scope_stack(crate::api::runtime::create_scope_stack());

    let success = runtime_with_endpoint(spawn_json_response(json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }]
    })));
    let response = success
        .execute(simple_chat_request(), false)
        .await
        .expect("remote success");
    assert_eq!(
        response["choices"][0]["message"]["content"],
        json!("guarded")
    );

    let invalid_json = runtime_with_endpoint(spawn_http_response(
        "200 OK",
        "application/json",
        "not json",
    ));
    assert_flow_error_contains(
        invalid_json.execute(simple_chat_request(), false).await,
        "failed to parse remote response JSON",
    );

    let http_error = runtime_with_endpoint(spawn_http_response(
        "502 Bad Gateway",
        "application/json",
        r#"{"error":"backend unavailable"}"#,
    ));
    assert_flow_error_contains(
        http_error.execute(simple_chat_request(), false).await,
        "status 502",
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn remote_execute_transport_and_stream_status_errors_are_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    crate::api::runtime::set_thread_scope_stack(crate::api::runtime::create_scope_stack());

    let runtime = RemoteBackendRuntime::new(&runtime_config(RemoteBackendConfig {
        endpoint: Some(unused_local_endpoint()),
        timeout_millis: 50,
        ..valid_remote()
    }))
    .unwrap();
    assert_flow_error_contains(
        runtime.execute(simple_chat_request(), false).await,
        "remote request failed",
    );
    assert_flow_error_contains(
        runtime.execute_stream(simple_chat_request()).await,
        "remote stream request failed",
    );

    let stream_status_error = runtime_with_endpoint(spawn_http_response(
        "503 Service Unavailable",
        "text/plain",
        "downstream unavailable",
    ));
    assert_flow_error_contains(
        stream_status_error
            .execute_stream(simple_chat_request())
            .await,
        "status 503",
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn remote_execute_stream_yields_completed_events_and_reports_malformed_final_event() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    crate::api::runtime::set_thread_scope_stack(crate::api::runtime::create_scope_stack());

    let runtime = runtime_with_endpoint(spawn_http_response(
        "200 OK",
        "text/event-stream",
        concat!(
            "data: {\"chunk\":\"first\"}\n\n",
            "event: done\ndata: {\"chunk\":\"final\"}\n\n"
        ),
    ));
    let mut stream = runtime
        .execute_stream(simple_chat_request())
        .await
        .expect("remote stream");
    assert_eq!(
        stream.next().await.unwrap().unwrap()["chunk"],
        json!("first")
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap()["chunk"],
        json!("final")
    );
    assert!(stream.next().await.is_none());

    let malformed = runtime_with_endpoint(spawn_http_response(
        "200 OK",
        "text/event-stream",
        "data: {\"chunk\":",
    ));
    let mut stream = malformed
        .execute_stream(simple_chat_request())
        .await
        .expect("malformed stream opens");
    assert_flow_error_contains(
        stream
            .next()
            .await
            .expect("decoder should report final frame"),
        "failed to parse SSE data payload",
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn remote_execute_stream_reports_malformed_named_final_event() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    crate::api::runtime::set_thread_scope_stack(crate::api::runtime::create_scope_stack());

    let runtime = runtime_with_endpoint(spawn_http_response(
        "200 OK",
        "text/event-stream",
        concat!(": keep-alive\n\n", "event: done\ndata: not-json\n\n"),
    ));
    let mut stream = runtime
        .execute_stream(simple_chat_request())
        .await
        .expect("remote stream");
    assert_flow_error_contains(
        stream
            .next()
            .await
            .expect("decoder should report malformed done event"),
        "failed to parse SSE data payload",
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn remote_tool_input_checks_cover_rewrite_block_noop_and_invalid_json() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    crate::api::runtime::set_thread_scope_stack(crate::api::runtime::create_scope_stack());

    let rewritten = runtime_with_endpoint(spawn_json_response(json!({
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "weather_lookup",
                        "arguments": "{\"city\":\"Paris\"}"
                    }
                }]
            }
        }]
    })))
    .check_tool_input("weather_lookup", &json!({"city": "Phoenix"}))
    .await
    .expect("modified tool input");
    assert_eq!(rewritten, json!({"city": "Paris"}));

    let blocked = runtime_with_endpoint(spawn_json_response(json!({
        "guardrails": {
            "log": {
                "activated_rails": [{"name": "input rail", "stop": true}]
            }
        }
    })));
    assert_flow_error_contains(
        blocked
            .check_tool_input("weather_lookup", &json!({"city": "Phoenix"}))
            .await,
        "tool_input rail blocked",
    );

    let original = json!({"city": "Phoenix"});
    let noop = runtime_with_endpoint(spawn_json_response(json!({
        "choices": [{"message": {"role": "assistant", "content": ""}}],
        "guardrails": {"log": {"activated_rails": []}}
    })))
    .check_tool_input("weather_lookup", &original)
    .await
    .expect("noop tool input");
    assert_eq!(noop, original);

    let invalid_json = runtime_with_endpoint(spawn_http_response(
        "200 OK",
        "application/json",
        "not json",
    ));
    assert_flow_error_contains(
        invalid_json
            .check_tool_input("weather_lookup", &json!({"city": "Phoenix"}))
            .await,
        "failed to parse remote response JSON",
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn remote_tool_output_checks_cover_rewrite_block_and_noop() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    crate::api::runtime::set_thread_scope_stack(crate::api::runtime::create_scope_stack());

    let rewritten = runtime_with_endpoint(spawn_json_response(json!({
        "choices": [{
            "message": {
                "role": "tool",
                "name": "weather_lookup",
                "content": "{\"forecast\":\"rain\"}"
            }
        }]
    })))
    .check_tool_output(
        "weather_lookup",
        &json!({"city": "Phoenix"}),
        &json!({"forecast": "sunny"}),
    )
    .await
    .expect("modified tool output");
    assert_eq!(rewritten, json!({"forecast": "rain"}));

    let blocked = runtime_with_endpoint(spawn_json_response(json!({
        "guardrails": {
            "log": {
                "activated_rails": [{
                    "name": "output rail",
                    "decisions": ["execute check", "refuse answer"]
                }]
            }
        }
    })));
    assert_flow_error_contains(
        blocked
            .check_tool_output(
                "weather_lookup",
                &json!({"city": "Phoenix"}),
                &json!({"forecast": "sunny"}),
            )
            .await,
        "tool_output rail blocked",
    );

    let original = json!({"forecast": "sunny"});
    let noop = runtime_with_endpoint(spawn_json_response(json!({
        "choices": [{"message": {"role": "assistant", "content": ""}}],
        "guardrails": {"log": {"activated_rails": []}}
    })))
    .check_tool_output("weather_lookup", &json!({"city": "Phoenix"}), &original)
    .await
    .expect("noop tool output");
    assert_eq!(noop, original);
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn tool_remote_check_transport_failures_are_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let stack = crate::api::runtime::create_scope_stack();
    crate::api::runtime::set_thread_scope_stack(stack);

    let runtime = RemoteBackendRuntime::new(&runtime_config(RemoteBackendConfig {
        endpoint: Some(unused_local_endpoint()),
        timeout_millis: 50,
        ..valid_remote()
    }))
    .unwrap();
    assert_flow_error_contains(
        runtime
            .check_tool_input("weather_lookup", &json!({"city": "Phoenix"}))
            .await,
        "remote request failed",
    );
    assert_flow_error_contains(
        runtime
            .check_tool_output(
                "weather_lookup",
                &json!({"city": "Phoenix"}),
                &json!({"forecast": "sunny"}),
            )
            .await,
        "remote request failed",
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn tool_remote_check_http_status_failures_are_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    crate::api::runtime::set_thread_scope_stack(crate::api::runtime::create_scope_stack());

    let input_status_error = runtime_with_endpoint(spawn_http_response(
        "429 Too Many Requests",
        "application/json",
        r#"{"error":"limited"}"#,
    ));
    assert_flow_error_contains(
        input_status_error
            .check_tool_input("weather_lookup", &json!({"city": "Phoenix"}))
            .await,
        "status 429",
    );

    let output_status_error = runtime_with_endpoint(spawn_http_response(
        "503 Service Unavailable",
        "application/json",
        r#"{"error":"maintenance"}"#,
    ));
    assert_flow_error_contains(
        output_status_error
            .check_tool_output(
                "weather_lookup",
                &json!({"city": "Phoenix"}),
                &json!({"forecast": "sunny"}),
            )
            .await,
        "status 503",
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn registered_remote_tool_input_intercept_rewrites_args_and_skips_output_check() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    crate::api::runtime::set_thread_scope_stack(crate::api::runtime::create_scope_stack());

    let endpoint = spawn_json_response(json!({
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "weather_lookup",
                        "arguments": "{\"city\":\"Paris\"}"
                    }
                }]
            }
        }],
        "guardrails": {"log": {"activated_rails": []}}
    }));
    let mut ctx = crate::plugin::PluginRegistrationContext::new();
    register_remote_backend(
        NeMoGuardrailsConfig {
            input: false,
            output: false,
            tool_input: true,
            tool_output: false,
            remote: Some(RemoteBackendConfig {
                endpoint: Some(endpoint),
                config_id: Some("safety-default".to_string()),
                timeout_millis: 5_000,
                ..RemoteBackendConfig::default()
            }),
            ..NeMoGuardrailsConfig::default()
        },
        &mut ctx,
    )
    .unwrap();
    let mut registrations = ctx.into_registrations();

    let callback_args = Arc::new(std::sync::Mutex::new(Json::Null));
    let seen = Arc::clone(&callback_args);
    let callback: crate::api::runtime::ToolExecutionNextFn = Arc::new(move |args| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            *seen.lock().unwrap() = args;
            Ok(json!({"forecast": "sunny"}))
        })
    });

    let result = crate::api::tool::tool_call_execute(
        crate::api::tool::ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(callback)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(*callback_args.lock().unwrap(), json!({"city": "Paris"}));
    assert_eq!(result, json!({"forecast": "sunny"}));

    crate::plugin::rollback_registrations(&mut registrations);
}
