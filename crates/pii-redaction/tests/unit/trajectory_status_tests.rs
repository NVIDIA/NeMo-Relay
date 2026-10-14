// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Status and privacy regression tests for trajectory context redaction.

use super::*;
use nemo_relay::error::{FlowError, UpstreamFailure, UpstreamFailureClass};
use opentelemetry::trace::Status;

#[tokio::test]
async fn trajectory_status_metadata_is_bounded() {
    let callback =
        crate::builtin::event_sanitize_callback(trajectory_backend(None, "redact_all_leaves"));
    for (metadata, expected) in [
        (
            json!({"otel.status_code": "OK", "error.type": "internal_error"}),
            json!({"otel.status_code": "OK"}),
        ),
        (
            json!({"otel.status_code": "UNSET"}),
            json!({"otel.status_code": "UNSET"}),
        ),
        (
            json!({"otel.status_code": "ERROR", "error.type": "internal_error"}),
            json!({"otel.status_code": "ERROR", "error.type": "internal_error"}),
        ),
        (
            json!({"otel.status_code": "ERROR", "error.type": "private error category"}),
            json!({"otel.status_code": "ERROR"}),
        ),
        (
            json!({"otel.status_code": "ERROR", "error.type": {"private": "value"}}),
            json!({"otel.status_code": "ERROR"}),
        ),
        (
            json!({"otel.status_code": "private status", "error.type": "internal_error"}),
            json!({}),
        ),
        (
            json!({"otel.status_code": ["ERROR"], "error.type": "internal_error"}),
            json!({}),
        ),
        (json!({"error.type": "internal_error"}), json!({})),
    ] {
        let mut metadata = metadata;
        metadata["otel.status_description"] = json!("private error message");
        metadata["exception.type"] = json!("PrivateApplicationException");
        metadata["nested"] = json!({"otel.status_code": "ERROR"});
        for scope_category in [ScopeCategory::Start, ScopeCategory::End] {
            let event = Event::Scope(ScopeEvent::new(
                BaseEvent::builder().name("test-scope").build(),
                scope_category,
                Vec::new(),
                EventCategory::agent(),
                None,
            ));
            let sanitized = callback(
                Arc::new(event),
                EventSanitizeFields {
                    data: None,
                    category_profile: None,
                    metadata: Some(metadata.clone()),
                },
            )
            .await
            .unwrap();
            assert_eq!(sanitized.metadata, Some(expected.clone()));
        }
        let mark = Event::Mark(MarkEvent::new(
            BaseEvent::builder().name("llm.chunk").build(),
            Some(EventCategory::custom()),
            None,
        ));
        let sanitized = callback(
            Arc::new(mark),
            EventSanitizeFields {
                data: None,
                category_profile: None,
                metadata: Some(metadata),
            },
        )
        .await
        .unwrap();
        assert_eq!(sanitized.metadata, Some(json!({})));
    }
}

#[test]
fn trajectory_redaction_preserves_exported_success_and_error_status() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();
    futures::executor::block_on(test_initialize_plugin_host_exact(plugin_config(json!({
        "profiles": [{"mode": "builtin", "builtin": {"preset": "trajectory_context"}}]
    }))))
    .unwrap();

    let captured = capture_events("trajectory-status-events");
    let exporters: Vec<_> = [
        ("trajectory-status-full", OpenTelemetryType::Full),
        ("trajectory-status-genai", OpenTelemetryType::GenAi),
        (
            "trajectory-status-openinference",
            OpenTelemetryType::OpenInference,
        ),
    ]
    .into_iter()
    .map(|(name, projection)| {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber =
            OpenTelemetrySubscriber::from_tracer_provider_with_type(provider, "test", projection);
        subscriber.register(name).unwrap();
        (name, subscriber, exporter)
    })
    .collect();
    let private_content = "synthetic private content canary";
    let request = LlmRequest {
        headers: [("authorization".into(), json!(private_content))]
            .into_iter()
            .collect(),
        content: json!({"model": "test-model", "messages": [{"role": "user", "content": private_content}]}),
    };
    let response = json!({
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": private_content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    });
    futures::executor::block_on(async {
        let result = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("test-provider")
                .request(request.clone())
                .func(noop_openai_chat_exec_fn(response.clone()))
                .codec(Arc::new(OpenAIChatCodec))
                .response_codec(Arc::new(OpenAIChatCodec))
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(result, response);

        for failure in [
            FlowError::Upstream(UpstreamFailure {
                status: Some(400),
                body: private_content.into(),
                headers: BTreeMap::new(),
                class: UpstreamFailureClass::InvalidRequest,
            }),
            FlowError::CallbackException {
                message: private_content.into(),
                exception_type: private_content.into(),
                source: None,
            },
        ] {
            let original_error = failure.to_string();
            let provider: LlmExecutionNextFn = Arc::new(move |_| {
                let failure = failure.clone();
                Box::pin(async move { Err(failure) })
            });
            let result = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name("test-provider")
                    .request(request.clone())
                    .func(provider)
                    .codec(Arc::new(OpenAIChatCodec))
                    .response_codec(Arc::new(OpenAIChatCodec))
                    .build(),
            )
            .await;
            assert_eq!(result.unwrap_err().to_string(), original_error);
        }

        let stream_provider: LlmStreamExecutionNextFn = Arc::new(move |_| {
            Box::pin(async move {
                Ok(LlmJsonStream::new(futures::stream::iter(vec![Err(
                    FlowError::Upstream(UpstreamFailure {
                        status: None,
                        body: private_content.into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::Timeout,
                    }),
                )])))
            })
        });
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("test-provider")
                .request(request)
                .func(stream_provider)
                .collector(Box::new(|_| Ok(())))
                .finalizer(Box::new(|| json!({})))
                .codec(Arc::new(OpenAIChatCodec))
                .response_codec(Arc::new(OpenAIChatCodec))
                .build(),
        )
        .await
        .unwrap();
        assert!(
            stream
                .next()
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains(private_content)
        );
        assert!(stream.next().await.is_none());

        let result = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("test-tool")
                .args(json!({"input": private_content}))
                .func(Arc::new(move |_| {
                    Box::pin(async move { Err(FlowError::Internal(private_content.into())) })
                }))
                .build(),
        )
        .await;
        assert!(result.unwrap_err().to_string().contains(private_content));
    });

    let events = captured_events_snapshot(&captured);
    assert!(
        !serde_json::to_string(&events)
            .unwrap()
            .contains(private_content)
    );
    for (name, subscriber, exporter) in &exporters {
        subscriber.force_flush().unwrap();
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 5, "{name}: {spans:?}");
        assert_eq!(
            spans
                .iter()
                .filter(|span| span.status == Status::Ok)
                .count(),
            1,
            "{name}: {spans:?}"
        );
        let mut error_types = Vec::new();
        for span in &spans {
            if let Status::Error { description } = &span.status {
                assert!(description.is_empty(), "{name}: error description retained");
                let error_key = if *name == "trajectory-status-openinference" {
                    "openinference.metadata.error.type"
                } else {
                    "error.type"
                };
                let error_type = span
                    .attributes
                    .iter()
                    .find(|attr| attr.key.as_str() == error_key)
                    .unwrap_or_else(|| {
                        panic!("{name}: failed span retains a bounded error category: {span:?}")
                    });
                error_types.push(error_type.value.to_string());
            }
        }
        error_types.sort();
        assert_eq!(
            error_types,
            [
                "internal_error",
                "internal_error",
                "invalid_request",
                "timeout"
            ],
            "{name}"
        );
        assert!(!format!("{spans:?}").contains(private_content), "{name}");
    }
    for (name, subscriber, _) in exporters {
        subscriber.deregister(name).unwrap();
    }
    deregister_subscriber("trajectory-status-events").unwrap();
    test_close_plugin_host().unwrap();
}
