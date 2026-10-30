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
