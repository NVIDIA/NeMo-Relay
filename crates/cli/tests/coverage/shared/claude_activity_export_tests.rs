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
    request_class: &str,
    input_tokens: u64,
) {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-claude-code-request-class",
        request_class.parse().unwrap(),
    );
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
    let response = json!({
        "id": format!("msg-{model}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "fixture response"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": 1}
    });
    execute_prepared_llm(manager, prep, response).await;
}

#[tokio::test]
async fn claude_activity_genai_aliases_preserve_class_prompt_evidence_and_usage() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
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
    let ids = tracked_sessions(&[
        "activity-export-idle",
        "activity-export-turn",
        "activity-export-compaction-hook",
    ]);
    register_filtered_session_subscriber("activity-export-mapped", ids, mapped.subscriber());

    let manager = SessionManager::new(session_test_config());
    export_activity_call(
        &manager,
        "activity-export-idle",
        "claude-export-idle",
        "auxiliary",
        2,
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
    export_activity_call(
        &manager,
        "activity-export-turn",
        "claude-export-main",
        "main",
        3,
    )
    .await;
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
    mapped.force_flush().unwrap();
    assert!(deregister_subscriber("activity-export-mapped").unwrap());
    let spans = exporter.get_finished_spans().unwrap();
    mapped.shutdown().unwrap();

    let expected = [
        ("claude-export-idle", "auxiliary"),
        ("claude-export-main", "main"),
    ];
    let modeled = unique_span_attributes_by_model(
        &spans,
        "gen_ai.request.model",
        &expected.map(|(model, _)| model),
    );
    for (model, request_class) in expected {
        assert_eq!(modeled[model][CLASS_ALIAS], request_class);
        assert!(!modeled[model].contains_key(PROMPT_ALIAS));
    }
    let token_total: u64 = modeled
        .values()
        .flat_map(|attributes| {
            ["gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"]
                .into_iter()
                .filter_map(move |key| attributes.get(key))
                .map(|value| value.parse::<u64>().unwrap())
        })
        .sum();
    assert_eq!(token_total, 7, "internal usage must remain in the total");
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
}
