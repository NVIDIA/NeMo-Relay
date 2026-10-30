// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use opentelemetry_proto::tonic::common::v1::any_value::Value as OtlpValue;

#[tokio::test]
async fn recursive_selectors_reach_atif_and_otlp_without_changing_execution() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    for enabled in [false, true] {
        reset_runtime();
        setup_isolated_thread();
        let mut config = plugin_config(json!({"mode": "builtin", "builtin": {
            "action": "redact", "detector": "email",
            "target_path_globs": ["/**/email", "/messages/**/content", "/messages/**/text", "/**/message"]
        }}));
        if !enabled {
            config.components.clear();
        }
        config.components.push(PluginComponentSpec {
            kind: "observability".into(),
            enabled: true,
            config: json!({"enable_full_payloads": true})
                .as_object()
                .unwrap()
                .clone(),
        });
        test_initialize_plugin_host_exact(config).await.unwrap();
        let atif = nemo_relay_plugin_host::exporter::atif(
            "recursive-export".into(),
            AtifAgentInfo {
                name: "test-agent".into(),
                version: "1".into(),
                model_name: None,
                tool_definitions: None,
                extra: None,
            },
        );
        register_subscriber("recursive-export-atif", atif.subscriber()).unwrap();
        let exporters: Vec<_> = [
            (
                "recursive-export-full",
                OpenTelemetryType::Full,
                "nemo_relay.tool_call_id",
            ),
            (
                "recursive-export-genai",
                OpenTelemetryType::GenAi,
                "gen_ai.tool.call.id",
            ),
            (
                "recursive-export-openinference",
                OpenTelemetryType::OpenInference,
                "tool_call.id",
            ),
        ]
        .into_iter()
        .map(|(name, projection, id_key)| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let receiver = capture_one_otlp_request(listener);
            let subscriber = nemo_relay_plugin_host::exporter::opentelemetry(
                OpenTelemetryConfig::new(projection, endpoint),
            )
            .unwrap();
            subscriber.register(name).unwrap();
            (name, id_key, subscriber, receiver)
        })
        .collect();
        let captured = capture_events("recursive-export-events");
        let agent = push_scope(
            PushScopeParams::builder()
                .name("recursive-export-agent")
                .scope_type(ScopeType::Agent)
                .metadata(json!({"email": "metadata@example.com", "note": "keep@example.com"}))
                .build(),
        )
        .unwrap();
        let input = json!({"model": "test-model", "messages": [
            {"role": "user", "content": "request@example.com"},
            {"role": "user", "content": [{"type": "text", "text": "part@example.com"}]}
        ]});
        let response = json!({"id": "response-id", "model": "test-model", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": "reply@example.com", "tool_calls": [
                {"id": "call-A", "type": "function", "function": {"name": "lookup", "arguments": "{\"email\":\"arg-a@example.com\",\"nested\":{\"email\":\"nested-a@example.com\"},\"note\":\"keep@example.com\"}"}},
                {"id": "call-B", "type": "function", "function": {"name": "lookup", "arguments": "{\"email\":\"arg-b@example.com\",\"nested\":{\"email\":\"nested-b@example.com\"},\"note\":\"keep@example.com\"}"}}
            ]}, "finish_reason": "tool_calls"
        }]});
        let expected_input = input.clone();
        let provider_response = response.clone();
        let actual = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("recursive-export-llm")
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
        assert_eq!(actual, response);
        for call in OpenAIChatCodec
            .decode_response(&actual)
            .unwrap()
            .tool_calls
            .unwrap()
            .into_iter()
            .rev()
        {
            let args = call.arguments.clone();
            let suffix = if call.id == "call-A" { "a" } else { "b" };
            let result = json!({"email": format!("result-{suffix}@example.com"),
                "rows": [{"contact": {"email": format!("deep-{suffix}@example.com")}}], "note": "keep@example.com"});
            let annotation = json!({"contact": {"email": format!("annotation-{suffix}@example.com")}, "note": "keep@example.com"});
            let expected_args = args.clone();
            let tool_result =
                ToolExecutionResult::new(result.clone()).with_annotation(annotation.clone());
            let expected_result = tool_result.clone();
            let actual = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name(call.name)
                    .tool_call_id(call.id)
                    .args(args)
                    .func(Arc::new(move |args| {
                        assert_eq!(args, expected_args);
                        let result = tool_result.clone();
                        Box::pin(async move { Ok(result) })
                    }))
                    .build(),
            )
            .await
            .unwrap();
            assert_eq!(actual, expected_result);
        }
        event(EmitMarkEventParams::builder().name("recursive-export-mark")
            .data(json!({"email": "mark@example.com", "rows": [{"contact": {"email": "deep-mark@example.com"}}], "note": "keep@example.com"}))
            .build()).unwrap();
        pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
        let events = captured_events_snapshot(&captured);
        let selected = [
            "request@example.com",
            "part@example.com",
            "reply@example.com",
            "arg-a@example.com",
            "arg-b@example.com",
            "nested-a@example.com",
            "nested-b@example.com",
            "result-a@example.com",
            "result-b@example.com",
            "deep-a@example.com",
            "deep-b@example.com",
            "annotation-a@example.com",
            "annotation-b@example.com",
            "metadata@example.com",
            "mark@example.com",
            "deep-mark@example.com",
        ];
        let canonical = serde_json::to_string(&events).unwrap();
        for secret in selected {
            assert_eq!(
                canonical.contains(secret),
                !enabled,
                "canonical: {secret}, enabled={enabled}"
            );
        }
        assert!(canonical.contains("keep@example.com"));
        let trajectory = atif.export().unwrap();
        let calls: Vec<_> = trajectory
            .steps
            .iter()
            .flat_map(|s| s.tool_calls.iter().flatten())
            .collect();
        let observations: Vec<_> = trajectory
            .steps
            .iter()
            .flat_map(|s| s.observation.iter().flat_map(|o| &o.results))
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls
                .iter()
                .map(|c| c.tool_call_id.as_str())
                .collect::<Vec<_>>(),
            ["call-A", "call-B"]
        );
        assert_eq!(
            observations
                .iter()
                .map(|o| o.source_call_id.as_deref().unwrap())
                .collect::<Vec<_>>(),
            ["call-B", "call-A"]
        );
        for call in &calls {
            assert_eq!(call.arguments["note"], "keep@example.com");
            if enabled {
                assert_eq!(call.arguments["email"], "[REDACTED]");
                assert_eq!(call.arguments["nested"]["email"], "[REDACTED]");
            }
        }
        for observation in &observations {
            let extra = observation.extra.as_ref().unwrap();
            assert_eq!(extra["tool_result"]["note"], "keep@example.com");
            assert_eq!(extra["tool_result_annotation"]["note"], "keep@example.com");
            if enabled {
                assert_eq!(extra["tool_result"]["email"], "[REDACTED]");
                assert_eq!(
                    extra["tool_result"]["rows"][0]["contact"]["email"],
                    "[REDACTED]"
                );
                assert_eq!(
                    extra["tool_result_annotation"]["contact"]["email"],
                    "[REDACTED]"
                );
            }
        }
        let atif_json = serde_json::to_string(&trajectory).unwrap();
        if enabled {
            for secret in selected {
                assert!(!atif_json.contains(secret), "ATIF: {secret}");
            }
        } else {
            for secret in [
                "arg-a@example.com",
                "result-a@example.com",
                "annotation-a@example.com",
            ] {
                assert!(atif_json.contains(secret), "ATIF control omitted {secret}");
            }
        }
        for (name, id_key, subscriber, receiver) in exporters {
            subscriber.force_flush().unwrap();
            subscriber.deregister(name).unwrap();
            subscriber.shutdown().unwrap();
            let request = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(request.path, "/v1/traces");
            let trace = ExportTraceServiceRequest::decode(request.body.as_slice()).unwrap();
            let spans: Vec<_> = trace
                .resource_spans
                .iter()
                .flat_map(|r| &r.scope_spans)
                .flat_map(|s| &s.spans)
                .collect();
            let tools: Vec<_> = spans
                .iter()
                .filter(|s| s.attributes.iter().any(|a| a.key == id_key))
                .collect();
            assert_eq!(tools.len(), 2, "{name}");
            for id in ["call-A", "call-B"] {
                assert_eq!(
                    tools
                        .iter()
                        .filter(|s| s.attributes.iter().any(|a| a.key == id_key
                            && matches!(
                                a.value.as_ref().and_then(|value| value.value.as_ref()),
                                Some(OtlpValue::StringValue(value)) if value == id
                            )))
                        .count(),
                    1,
                    "{name}: {id}"
                );
            }
            assert_ne!(tools[0].span_id, tools[1].span_id);
            for tool in &tools {
                let string_attr = |key: &str| {
                    tool.attributes.iter().find(|a| a.key == key).and_then(|a| {
                        match a.value.as_ref()?.value.as_ref()? {
                            OtlpValue::StringValue(value) => Some(value.as_str()),
                            _ => None,
                        }
                    })
                };
                if name == "recursive-export-genai" {
                    let result: Json =
                        serde_json::from_str(string_attr("gen_ai.tool.call.result").unwrap())
                            .unwrap();
                    assert_eq!(result["note"], "keep@example.com");
                    if enabled {
                        assert_eq!(result["email"], "[REDACTED]");
                        assert_eq!(result["rows"][0]["contact"]["email"], "[REDACTED]");
                    }
                } else {
                    assert_eq!(
                        string_attr("nemo_relay.end.output.note"),
                        Some("keep@example.com")
                    );
                    let annotation: Json = serde_json::from_str(
                        string_attr("nemo_relay.tool.result.annotation").unwrap(),
                    )
                    .unwrap();
                    assert_eq!(annotation["note"], "keep@example.com");
                    if enabled {
                        assert_eq!(
                            string_attr("nemo_relay.end.output.email"),
                            Some("[REDACTED]")
                        );
                        assert_eq!(annotation["contact"]["email"], "[REDACTED]");
                    }
                }
            }
            let wire = format!("{trace:?}");
            assert!(
                wire.contains("keep@example.com"),
                "{name}: unselected value omitted"
            );
            if enabled {
                assert!(wire.contains("[REDACTED]"), "{name}");
                for secret in selected {
                    assert!(!wire.contains(secret), "{name}: {secret}");
                }
            } else {
                for secret in ["arg-a@example.com", "result-a@example.com"] {
                    assert!(wire.contains(secret), "{name}: control omitted {secret}");
                }
            }
            println!(
                "EXPORT_SMOKE enabled={enabled} projection={name} spans={} tool_spans={} bytes={}",
                spans.len(),
                tools.len(),
                request.body.len()
            );
        }
        println!(
            "EXPORT_SMOKE enabled={enabled} ATIF_calls={} ATIF_observations={} selected_values={} canonical_events={}",
            calls.len(),
            observations.len(),
            selected.len(),
            events.len()
        );
        deregister_subscriber("recursive-export-events").unwrap();
        deregister_subscriber("recursive-export-atif").unwrap();
        test_close_plugin_host().unwrap();
    }
}
