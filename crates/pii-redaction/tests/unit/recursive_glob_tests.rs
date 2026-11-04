// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn recursive_globs_sanitize_managed_llm_observations_without_changing_execution() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();
    test_initialize_plugin_host_exact(plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "redact", "detector": "email",
            "target_path_globs": ["/messages/**/content", "/messages/**/text", "/**/message"]
        }
    })))
    .await
    .unwrap();
    let events = capture_events("recursive-glob-managed-llm");
    let input = json!({"model": "test-model", "messages": [
        {"role": "user", "content": "alice@example.com"},
        {"role": "user", "content": [{"type": "text", "text": "bob@example.com"}]}
    ]});
    let response = json!({"id": "response-id", "model": "test-model", "choices": [{
        "index": 0, "message": {"role": "assistant", "content": "carol@example.com"},
        "finish_reason": "stop"
    }]});
    let expected_input = input.clone();
    let provider_response = response.clone();
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("recursive-globs")
            .request(LlmRequest {
                headers: Default::default(),
                content: input,
            })
            .func(Arc::new(move |request| {
                assert_eq!(request.content, expected_input);
                let response = provider_response.clone();
                Box::pin(async move { Ok(response) })
            }))
            .codec(Arc::new(OpenAIChatCodec))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result, response);
    let captured = captured_events_snapshot(&events);
    assert_eq!(captured.len(), 2);
    let request = &captured[0].input().unwrap()["content"];
    assert_eq!(request["messages"][0]["content"], "[REDACTED]");
    assert_eq!(request["messages"][1]["content"][0]["text"], "[REDACTED]");
    assert_eq!(request["model"], "test-model");
    assert_eq!(
        captured[1].output().unwrap()["choices"][0]["message"]["content"],
        "[REDACTED]"
    );
    let serialized = serde_json::to_string(&captured).unwrap();
    for secret in ["alice@example.com", "bob@example.com", "carol@example.com"] {
        assert!(!serialized.contains(secret));
    }
    deregister_subscriber("recursive-glob-managed-llm").unwrap();
    test_close_plugin_host().unwrap();
}

#[tokio::test]
async fn recursive_globs_keep_multi_projection_responses_fail_closed() {
    for (identity, payload) in [
        (
            BuiltinLlmCodec::OpenAiChat,
            json!({"choices": [
                {"index": 0, "message": {"role": "assistant", "content": "alice@example.com"}},
                {"index": 1, "message": {"role": "assistant", "content": "bob@example.com"}}
            ]}),
        ),
        (
            BuiltinLlmCodec::GeminiGenerateContent,
            json!({"candidates": [
                {"content": {"role": "model", "parts": [{"text": "alice@example.com"}]}},
                {"content": {"role": "model", "parts": [{"text": "bob@example.com"}]}}
            ]}),
        ),
    ] {
        for glob in ["/**/message", "/**/text", "/**"] {
            let backend = crate::builtin::CompiledBuiltinBackend::new(
                BuiltinBackendConfig {
                    action: "redact".into(),
                    detector: Some("email".into()),
                    target_path_globs: vec![glob.into()],
                    ..Default::default()
                },
                None,
            )
            .unwrap();
            let callback = crate::builtin::llm_sanitize_response_callback(backend);
            assert!(
                callback(
                    payload.clone(),
                    LlmSanitizeResponseContext::with_identity(LlmCodecIdentity::BuiltIn(identity))
                )
                .await
                .unwrap()
                .is_none(),
                "{identity:?}: {glob}"
            );
        }
    }
}

#[tokio::test]
async fn recursive_globs_sanitize_tool_and_mark_fields_at_different_depths() {
    let backend = crate::builtin::CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "redact".into(),
            detector: Some("email".into()),
            target_path_globs: vec!["/**/email".into()],
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let payload = json!({"email": "alice@example.com", "nested": [{"email": "bob@example.com"}],
        "unselected": "carol@example.com"});
    let expected = json!({"email": "[REDACTED]", "nested": [{"email": "[REDACTED]"}],
        "unselected": "carol@example.com"});
    let tool = crate::builtin::tool_sanitize_callback(backend.clone());
    assert_eq!(
        tool("lookup".into(), payload.clone()).await.unwrap(),
        expected
    );
    let mark = crate::builtin::event_sanitize_callback(backend);
    let fields = mark(
        Arc::new(Event::Mark(MarkEvent::new(
            BaseEvent::builder().name("recursive-globs").build(),
            None,
            None,
        ))),
        EventSanitizeFields {
            data: Some(payload.clone()),
            metadata: Some(payload),
            category_profile: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(fields.data.unwrap(), expected);
    assert_eq!(fields.metadata.unwrap(), expected);
}

#[tokio::test]
async fn recursive_globs_sanitize_metric_text_without_changing_analytical_values() {
    let data = json!({"measurements": [{
        "name": "example.requests", "kind": "counter", "value_type": "u64", "value": 7,
        "description": "alice@example.com",
        "attributes": {"owners": ["bob@example.com"], "attempts": 2, "ok": true}
    }]});
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("example.metrics")
            .data(data.clone())
            .data_schema(
                DataSchema::builder()
                    .name(METRIC_DATA_SCHEMA_NAME)
                    .version(METRIC_DATA_SCHEMA_VERSION)
                    .build(),
            )
            .build(),
        None,
        None,
    ));
    for action in ["redact", "remove"] {
        let backend = crate::builtin::CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                action: action.into(),
                detector: (action == "redact").then(|| "email".into()),
                target_path_globs: vec!["/**".into()],
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let callback = crate::builtin::event_sanitize_callback(backend);
        let sanitized = callback(
            Arc::new(event.clone()),
            EventSanitizeFields {
                data: Some(data.clone()),
                metadata: None,
                category_profile: None,
            },
        )
        .await
        .unwrap()
        .data
        .unwrap();
        serde_json::from_value::<MetricEnvelope>(sanitized.clone())
            .unwrap()
            .validate()
            .unwrap();
        let measurement = &sanitized["measurements"][0];
        assert_eq!(measurement["name"], "example.requests");
        assert_eq!(measurement["value"], 7);
        assert_eq!(measurement["attributes"]["attempts"], 2);
        assert_eq!(measurement["attributes"]["ok"], true);
        if action == "redact" {
            assert_eq!(measurement["description"], "[REDACTED]");
            assert_eq!(measurement["attributes"]["owners"], json!(["[REDACTED]"]));
        } else {
            assert!(measurement["description"].is_null());
            assert!(measurement["attributes"].get("owners").is_none());
        }
    }
}
