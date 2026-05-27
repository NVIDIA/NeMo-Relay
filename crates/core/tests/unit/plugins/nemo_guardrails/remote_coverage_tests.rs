// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Focused remote runtime coverage tests for the NeMo Guardrails plugin component.

use std::collections::HashMap;
use std::net::TcpListener;

use super::*;
use crate::plugins::nemo_guardrails::component::{RailSelector, RemoteBackendConfig};

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

fn unused_local_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}")
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
