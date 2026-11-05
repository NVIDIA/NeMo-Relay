// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_observability::OtlpAttributeMapping;
use nemo_relay_observability::otel::OpenTelemetrySubscriberOptions;

const CLASS_ALIAS: &str = "client.claude_code.request_class";
const PROMPT_ALIAS: &str = "client.turn.prompt_observed";

async fn export_activity_call(
    manager: &SessionManager,
    session_id: &str,
    model: &str,
    request_class: Option<&str>,
    input_tokens: Option<u64>,
) {
    let mut headers = HeaderMap::new();
    if let Some(request_class) = request_class {
        headers.insert(
            "x-claude-code-request-class",
            request_class.parse().unwrap(),
        );
    }
    let start = llm_start_with_content(
        session_id,
        "anthropic.messages",
        model,
        json!({
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "inspect the export fixture"}]
        }),
    );
    let prep = manager.prepare_gateway_call(&headers, start).await.unwrap();
    let mut response = json!({
        "id": format!("msg-{model}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "fixture response"}],
        "stop_reason": "end_turn"
    });
    if let Some(input_tokens) = input_tokens {
        response["usage"] = json!({"input_tokens": input_tokens, "output_tokens": 1});
    }
    execute_prepared_llm(manager, prep, response).await;
}

#[tokio::test]
async fn claude_activity_genai_aliases_preserve_class_prompt_evidence_and_usage() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atof = make_atof_test_exporter(temp.path(), "events.jsonl");
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let mapped = OpenTelemetrySubscriber::from_tracer_provider_with_type_and_options(
        provider,
        "claude-activity-mapped-genai",
        OpenTelemetryType::GenAi,
        OpenTelemetrySubscriberOptions {
            attribute_mappings: vec![
                OtlpAttributeMapping::new(
                    "nemo_relay.start.metadata.claude_code_request_class",
                    CLASS_ALIAS,
                ),
                OtlpAttributeMapping::new(
                    "nemo_relay.end.metadata.turn_prompt_observed",
                    PROMPT_ALIAS,
                ),
            ],
            ..Default::default()
        },
    )
    .unwrap();
    let (unmapped, unmapped_exporter) =
        make_gen_ai_test_subscriber("claude-activity-default-genai");
    let ids = tracked_sessions(&[
        "activity-export-idle",
        "activity-export-turn",
        "activity-export-compaction-hook",
    ]);
    for (name, subscriber) in [
        ("activity-export-atof", atof.subscriber()),
        ("activity-export-mapped", mapped.subscriber()),
        ("activity-export-default", unmapped.subscriber()),
    ] {
        register_filtered_session_subscriber(name, ids.clone(), subscriber);
    }

    let manager = SessionManager::new(session_test_config());
    export_activity_call(
        &manager,
        "activity-export-idle",
        "claude-export-idle",
        Some("auxiliary"),
        Some(2),
    )
    .await;
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::PromptSubmitted(session_event(
                "activity-export-turn",
                "UserPromptSubmit",
            ))],
        )
        .await
        .unwrap();
    for (model, request_class, usage) in [
        ("claude-export-main", Some("main"), Some(3)),
        ("claude-export-no-usage", Some("compaction"), None),
        ("claude-export-unknown", None, Some(5)),
    ] {
        export_activity_call(
            &manager,
            "activity-export-turn",
            model,
            request_class,
            usage,
        )
        .await;
    }
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::Compaction(session_event(
                "activity-export-compaction-hook",
                "PreCompact",
            ))],
        )
        .await
        .unwrap();
    manager.close_all("test_shutdown").await.unwrap();
    flush_subscribers().unwrap();
    atof.force_flush().unwrap();
    mapped.force_flush().unwrap();
    unmapped.force_flush().unwrap();
    for name in [
        "activity-export-atof",
        "activity-export-mapped",
        "activity-export-default",
    ] {
        assert!(deregister_subscriber(name).unwrap());
    }
    let spans = exporter.get_finished_spans().unwrap();
    let default_spans = unmapped_exporter.get_finished_spans().unwrap();
    mapped.shutdown().unwrap();
    unmapped.shutdown().unwrap();

    let expected = [
        ("claude-export-idle", "auxiliary"),
        ("claude-export-main", "main"),
        ("claude-export-no-usage", "compaction"),
        ("claude-export-unknown", "unknown"),
    ];
    let modeled = unique_span_attributes_by_model(
        &spans,
        "gen_ai.request.model",
        &expected.map(|(model, _)| model),
    );
    for (model, request_class) in expected {
        assert_eq!(modeled[model][CLASS_ALIAS], request_class);
        assert!(!modeled[model].contains_key(PROMPT_ALIAS));
        assert!(!modeled[model].contains_key("client.request.headers"));
    }
    assert!(!modeled["claude-export-no-usage"].contains_key("gen_ai.usage.input_tokens"));
    assert!(!modeled["claude-export-no-usage"].contains_key("gen_ai.usage.output_tokens"));
    let token_total: u64 = modeled
        .values()
        .flat_map(|attributes| {
            ["gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"]
                .into_iter()
                .filter_map(move |key| attributes.get(key))
                .map(|value| value.parse::<u64>().unwrap())
        })
        .sum();
    assert_eq!(
        token_total, 13,
        "internal and unknown usage must remain in the total"
    );
    let prompt_turns = spans
        .iter()
        .filter(|span| {
            let attributes = attr_map(&span.attributes);
            attributes.get("gen_ai.operation.name").map(String::as_str) == Some("invoke_agent")
                && span.attributes.iter().any(|attribute| {
                    attribute.key.as_str() == PROMPT_ALIAS
                        && attribute.value == opentelemetry::Value::Bool(true)
                })
        })
        .count();
    assert_eq!(
        prompt_turns, 1,
        "the documented selector must not count model calls"
    );
    let turns = spans
        .iter()
        .map(|span| attr_map(&span.attributes))
        .filter(|attributes| {
            attributes.get("gen_ai.operation.name").map(String::as_str) == Some("invoke_agent")
        })
        .collect::<Vec<_>>();
    assert_eq!(turns.len(), 2);
    assert!(
        turns
            .iter()
            .all(|attributes| !attributes.contains_key(CLASS_ALIAS))
    );
    assert!(
        turns
            .iter()
            .any(|attributes| attributes[PROMPT_ALIAS] == "false")
    );
    assert_eq!(default_spans.len(), spans.len());
    for span in &default_spans {
        assert!(span.attributes.iter().all(|attribute| {
            let key = attribute.key.as_str();
            !key.starts_with("client.")
                && !key.starts_with("nemo_relay.")
                && !key.contains("request_class")
                && !key.contains("prompt_observed")
        }));
    }

    let events = read_atof_events(atof.path().unwrap());
    let model_ends = events
        .iter()
        .filter(|event| event["name"] == "anthropic.messages" && event["scope_category"] == "end")
        .collect::<Vec<_>>();
    assert_eq!(model_ends.len(), 4);
    for (model, request_class) in expected {
        let end = model_ends
            .iter()
            .find(|event| event["data"]["model"] == model)
            .unwrap();
        assert_eq!(end["metadata"]["claude_code_request_class"], request_class);
    }
    let idle = model_ends
        .iter()
        .find(|event| event["data"]["model"] == "claude-export-idle")
        .unwrap();
    assert_eq!(idle["data"]["usage"]["input_tokens"], 2);
    assert_eq!(idle["data"]["usage"]["output_tokens"], 1);
    let turn_ends = events
        .iter()
        .filter(|event| event["name"] == "claude-code-turn" && event["scope_category"] == "end")
        .collect::<Vec<_>>();
    assert_eq!(turn_ends.len(), 2);
    for (session_id, prompt_observed) in [
        ("activity-export-turn", true),
        ("activity-export-compaction-hook", false),
    ] {
        let end = turn_ends
            .iter()
            .find(|event| event["metadata"]["session_id"] == session_id)
            .unwrap();
        assert_eq!(end["metadata"]["turn_prompt_observed"], prompt_observed);
    }
}

#[tokio::test]
async fn claude_activity_atif_class_requires_observed_events_and_an_enclosing_turn() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    for include_observed_events in [true, false] {
        let directory = temp.path().join(include_observed_events.to_string());
        std::fs::create_dir_all(&directory).unwrap();
        let config: PluginConfig = serde_json::from_value(json!({
            "version": 1,
            "components": [{
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 4,
                    "atif": {
                        "enabled": true,
                        "output_directory": directory,
                        "filename_template": "trajectory-{session_id}.json",
                        "include_observed_events": include_observed_events
                    }
                }
            }]
        }))
        .unwrap();
        test_initialize_plugin_host_exact(config).await.unwrap();
        let manager = SessionManager::new(session_test_config());
        export_activity_call(
            &manager,
            "activity-atif-idle",
            "claude-atif-idle",
            Some("auxiliary"),
            Some(2),
        )
        .await;
        flush_subscribers().unwrap();
        assert!(
            std::fs::read_dir(&directory).unwrap().next().is_none(),
            "an idle internal LLM must not fabricate an ATIF trajectory"
        );
        manager
            .apply_events(
                &HeaderMap::new(),
                vec![NormalizedEvent::PromptSubmitted(session_event(
                    "activity-atif-turn",
                    "UserPromptSubmit",
                ))],
            )
            .await
            .unwrap();
        for request_class in ["main", "auxiliary"] {
            export_activity_call(
                &manager,
                "activity-atif-turn",
                &format!("claude-atif-{request_class}"),
                Some(request_class),
                Some(3),
            )
            .await;
        }
        manager.close_all("test_shutdown").await.unwrap();
        flush_subscribers().unwrap();
        test_close_plugin_host().unwrap();
        let trajectories = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| {
                serde_json::from_slice::<Value>(&std::fs::read(entry.unwrap().path()).unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            trajectories.len(),
            1,
            "only the prompt's enclosing turn owns a trajectory"
        );
        let trajectory = &trajectories[0];
        assert_eq!(trajectory["final_metrics"]["total_prompt_tokens"], 6);
        assert!(
            trajectory["steps"]
                .as_array()
                .unwrap()
                .iter()
                .all(|step| { step["extra"].get("claude_code_request_class").is_none() })
        );
        if include_observed_events {
            let observed = trajectory["extra"]["observed_events"].as_array().unwrap();
            let internal = observed
                .iter()
                .filter(|event| {
                    event["name"] == "anthropic.messages"
                        && event["metadata"]["claude_code_request_class"] == "auxiliary"
                })
                .collect::<Vec<_>>();
            assert_eq!(
                internal.len(),
                2,
                "the internal call retains its start and end metadata"
            );
            assert!(
                internal
                    .iter()
                    .all(|event| event["metadata"]["session_id"] == "activity-atif-turn")
            );
        } else {
            assert!(trajectory["extra"].get("observed_events").is_none());
            assert!(
                !serde_json::to_string(trajectory)
                    .unwrap()
                    .contains("claude_code_request_class"),
                "ordinary ATIF steps must not be described as retaining event metadata"
            );
        }
    }
}
