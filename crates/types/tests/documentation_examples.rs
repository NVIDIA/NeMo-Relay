// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile and exercise the source files displayed in documentation guides.

mod framework_response_codec {
    include!("../../../docs/integrate-into-frameworks/_code/framework_response_codec.rs");

    #[test]
    fn decodes_framework_response() {
        let response = FrameworkResponseCodec
            .decode_response(&serde_json::json!({
                "id": "response-1",
                "model_name": "example-model",
                "text": "Hello",
                "input_tokens": 4,
                "output_tokens": 2,
            }))
            .unwrap();

        assert_eq!(response.id.as_deref(), Some("response-1"));
        assert_eq!(response.model.as_deref(), Some("example-model"));
        assert_eq!(response.message, Some(MessageContent::Text("Hello".into())));
        assert_eq!(response.finish_reason, Some(FinishReason::Complete));
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(4));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(6));
    }

    #[test]
    fn keeps_missing_token_counts_unknown() {
        let response = FrameworkResponseCodec
            .decode_response(&serde_json::json!({"input_tokens": 4}))
            .unwrap();

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(4));
        assert_eq!(usage.completion_tokens, None);
        assert_eq!(usage.total_tokens, None);
    }

    #[test]
    fn keeps_overflowing_total_unknown() {
        let response = FrameworkResponseCodec
            .decode_response(&serde_json::json!({
                "input_tokens": u64::MAX,
                "output_tokens": 1,
            }))
            .unwrap();

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(u64::MAX));
        assert_eq!(usage.completion_tokens, Some(1));
        assert_eq!(usage.total_tokens, None);
    }

    #[test]
    fn rejects_malformed_token_counts() {
        let result =
            FrameworkResponseCodec.decode_response(&serde_json::json!({"input_tokens": "invalid"}));

        assert!(matches!(result, Err(FlowError::Internal(_))));
    }
}

#[test]
fn documentation_resource_metrics_snapshot_matches_the_public_schema() {
    let page =
        include_str!("../../../docs/reference/api/resource-metrics.mdx").replace("\r\n", "\n");
    for line_ending in ["\n", "\r\n"] {
        assert_documentation_snapshot_matches_public_schema(&page.replace('\n', line_ending));
    }
}

fn assert_documentation_snapshot_matches_public_schema(page: &str) {
    use nemo_relay_types::api::resource_metrics::ResourceMetricsSnapshot;

    let page = page.replace("\r\n", "\n");
    let example = page
        .split("```json title=\"Illustrative snapshot\"\n")
        .nth(1)
        .unwrap()
        .split("\n```")
        .next()
        .unwrap();
    let expected: serde_json::Value = serde_json::from_str(example).unwrap();
    let snapshot: ResourceMetricsSnapshot = serde_json::from_str(example).unwrap();
    assert_eq!(serde_json::to_value(&snapshot).unwrap(), expected);
    assert_eq!(snapshot.measurement_scope.as_str(), "process_tree");
    assert!(snapshot.cpu.unwrap().consumption_rate.is_none());
    assert!(snapshot.memory.is_none());
}
