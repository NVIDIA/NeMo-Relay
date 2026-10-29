// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exporter regression coverage for trajectory tool-call correlation.

use super::*;

#[tokio::test]
async fn trajectory_preserves_distinct_tool_calls_and_observations() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    // Also exercise repeated sanitization by registering two trajectory profiles.
    for profile_count in 0..=2 {
        reset_runtime();
        setup_isolated_thread();
        let config = if profile_count == 0 {
            PluginConfig {
                version: 1,
                components: vec![],
                policy: Default::default(),
            }
        } else {
            plugin_config(json!({"profiles": vec![json!({
                "mode": "builtin", "builtin": {"preset": "trajectory_context"}
            }); profile_count]}))
        };
        test_initialize_plugin_host_exact(config).await.unwrap();
        let atif = AtifExporter::new(
            "tool-correlation".into(),
            AtifAgentInfo {
                name: "test-agent".into(),
                version: "1".into(),
                model_name: None,
                tool_definitions: None,
                extra: None,
            },
        );
        register_subscriber("tool-correlation-atif", atif.subscriber()).unwrap();
        let exporters: Vec<_> = [
            ("tool-correlation-full", OpenTelemetryType::Full),
            ("tool-correlation-genai", OpenTelemetryType::GenAi),
            (
                "tool-correlation-openinference",
                OpenTelemetryType::OpenInference,
            ),
        ]
        .into_iter()
        .map(|(name, projection)| {
            let memory = InMemorySpanExporterBuilder::new().build();
            let provider = SdkTracerProvider::builder()
                .with_simple_exporter(memory.clone())
                .build();
            let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_type(
                provider, "test", projection,
            );
            subscriber.register(name).unwrap();
            (name, subscriber, memory)
        })
        .collect();
        let captured = capture_events("tool-correlation-events");
        let agent = push_scope(
            PushScopeParams::builder()
                .name("agent")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        let response = json!({
            "id": "response-private", "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call-A", "type": "function", "function": {"name": "lookup", "arguments": "{\"city\":\"NYC\"}"}},
                {"id": "call-B", "type": "function", "function": {"name": "lookup", "arguments": "{\"city\":\"SF\"}"}}
            ]}, "finish_reason": "tool_calls"}]
        });
        let client_response = llm_call_execute(LlmCallExecuteParams::builder()
            .name("openai").request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "test-model", "messages": [{"role": "user", "content": "lookup two cities"}]})
            })
            .func(noop_openai_chat_exec_fn(response.clone()))
            .codec(Arc::new(OpenAIChatCodec)).response_codec(Arc::new(OpenAIChatCodec)).build()
        ).await.unwrap();
        assert_eq!(client_response, response);
        let normalized = OpenAIChatCodec.decode_response(&client_response).unwrap();
        // Use the provider's normalized calls, completing the same-name tools in reverse order.
        for call in normalized.tool_calls.unwrap().into_iter().rev() {
            let args = call.arguments.clone();
            let result = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name(call.name)
                    .args(call.arguments)
                    .tool_call_id(call.id)
                    .func(Arc::new(|args| {
                        Box::pin(
                            async move { Ok(ToolExecutionResult::new(json!({"answer": args}))) },
                        )
                    }))
                    .build(),
            )
            .await
            .unwrap();
            assert_eq!(result.result, json!({"answer": args}));
        }
        pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
        crate::api::subscriber::flush_subscribers().unwrap();
        let trajectory = atif.export().unwrap();
        let calls: Vec<_> = trajectory
            .steps
            .iter()
            .flat_map(|s| s.tool_calls.iter().flatten())
            .collect();
        let observations: Vec<_> = trajectory
            .steps
            .iter()
            .flat_map(|s| s.observation.iter().flat_map(|o| o.results.iter()))
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_call_id, "call-A");
        assert_eq!(calls[1].tool_call_id, "call-B");
        assert_eq!(observations.len(), 2, "profile_count={profile_count}");
        assert_eq!(observations[0].source_call_id.as_deref(), Some("call-B"));
        assert_eq!(observations[1].source_call_id.as_deref(), Some("call-A"));
        let events = captured_events_snapshot(&captured);
        let tool_events: Vec<_> = events
            .iter()
            .filter(|e| e.category().is_some_and(|c| c.as_str() == "tool"))
            .collect();
        assert_eq!(tool_events.len(), 4);
        for (event, id) in tool_events
            .iter()
            .zip(["call-B", "call-B", "call-A", "call-A"])
        {
            assert_eq!(event.tool_call_id(), Some(id));
            if profile_count > 0 {
                assert_eq!(event.data(), Some(&json!({})));
            }
        }
        if profile_count > 0 {
            assert!(calls.iter().all(|call| call.arguments == json!({})));
            let serialized = serde_json::to_string(&trajectory).unwrap();
            for private in ["NYC", "SF", "response-private", "lookup two cities"] {
                assert!(!serialized.contains(private), "{private}: {serialized}");
            }
        }
        for (name, subscriber, memory) in exporters {
            subscriber.force_flush().unwrap();
            let spans = memory.get_finished_spans().unwrap();
            let id_key = match name {
                "tool-correlation-genai" => "gen_ai.tool.call.id",
                "tool-correlation-openinference" => "tool_call.id",
                _ => "nemo_relay.tool_call_id",
            };
            let tool_spans: Vec<_> = spans
                .iter()
                .filter(|s| s.attributes.iter().any(|a| a.key.as_str() == id_key))
                .collect();
            assert_eq!(tool_spans.len(), 2, "{name}");
            for (span, id) in tool_spans.iter().zip(["call-B", "call-A"]) {
                assert!(
                    span.attributes
                        .iter()
                        .any(|a| a.key.as_str() == id_key && a.value.to_string() == id)
                );
            }
            assert_ne!(
                tool_spans[0].span_context.span_id(),
                tool_spans[1].span_context.span_id()
            );
            if profile_count > 0 && name == "tool-correlation-genai" {
                for span in &tool_spans {
                    for key in ["gen_ai.tool.call.arguments", "gen_ai.tool.call.result"] {
                        assert!(
                            span.attributes
                                .iter()
                                .any(|a| a.key.as_str() == key && a.value.to_string() == "{}")
                        );
                    }
                }
                let messages = spans
                    .iter()
                    .flat_map(|s| &s.attributes)
                    .find(|a| a.key.as_str() == "gen_ai.output.messages")
                    .unwrap();
                let messages: Json = serde_json::from_str(&messages.value.to_string()).unwrap();
                assert_eq!(messages[0]["parts"][0]["id"], "call-A");
                assert_eq!(messages[0]["parts"][1]["id"], "call-B");
            }
            if profile_count > 0 {
                let serialized = format!("{spans:?}");
                for private in ["NYC", "SF", "response-private", "lookup two cities"] {
                    assert!(!serialized.contains(private), "{name}: {serialized}");
                }
            }
            subscriber.deregister(name).unwrap();
        }
        println!(
            "CORRELATION_SMOKE profile_count={profile_count} ATIF_calls=call-A,call-B ATIF_observations=call-B,call-A OTEL_tool_spans_per_projection=2"
        );
        deregister_subscriber("tool-correlation-events").unwrap();
        deregister_subscriber("tool-correlation-atif").unwrap();
        test_close_plugin_host().unwrap();
    }
}
