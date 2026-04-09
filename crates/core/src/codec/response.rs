// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalized LLM response types produced by response codecs.
//!
//! This module defines [`AnnotatedLLMResponse`] and its supporting types
//! for structured, API-agnostic access to LLM response data.

use serde::{Deserialize, Serialize};

use crate::json::Json;

use super::request::MessageContent;

// ---------------------------------------------------------------------------
// AnnotatedLLMResponse type hierarchy
// ---------------------------------------------------------------------------

/// Structured view of an LLM response, produced by a response codec from
/// raw JSON API output.
///
/// The `extra` field captures any top-level keys not modeled by the known
/// fields, ensuring lossless round-trip through serde.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedLLMResponse {
    /// Response ID from the API (e.g., "chatcmpl-abc123", "resp_abc123", "msg_abc123").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// The model that actually served the request (may differ from requested model).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The assistant's response content, reusing [`MessageContent`] from request types.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<MessageContent>,

    /// Tool calls requested by the model, normalized across APIs.
    ///
    /// Uses [`ResponseToolCall`] (arguments as [`Json`]) NOT the request-side
    /// `ToolCall` (arguments as `String`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,

    /// Why generation stopped, normalized across APIs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// API-specific response data that cannot be normalized across providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_specific: Option<ApiSpecificResponse>,

    /// Catch-all for unmodeled top-level fields, ensuring lossless round-trip.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Json>,
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Token usage statistics from an LLM API response.
///
/// All fields are `Option<u64>` because not every provider supplies every
/// field. For example, cache token counts are only available from providers
/// that support prompt caching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Tokens consumed by the prompt/input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Tokens generated in the completion/output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Total tokens (prompt + completion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Tokens served from prompt cache (read).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Tokens written to prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

/// Normalized reason why the model stopped generating.
///
/// Maps from provider-specific stop reasons:
/// - **Complete**: OpenAI Chat `"stop"`, Anthropic `"end_turn"`, Responses `"completed"`
/// - **Length**: OpenAI Chat `"length"`, Anthropic `"max_tokens"`, Responses incomplete+max_output_tokens
/// - **ToolUse**: OpenAI Chat `"tool_calls"`, Anthropic `"tool_use"`
/// - **ContentFilter**: OpenAI Chat `"content_filter"`, Responses incomplete+content_filter
/// - **Unknown**: Forward-compatible catch-all for unrecognized reasons
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Model naturally completed its response.
    Complete,
    /// Maximum token limit reached.
    Length,
    /// Model requested a tool call.
    ToolUse,
    /// Content was filtered by safety systems.
    ContentFilter,
    /// Unknown or forward-compatible reason.
    Unknown(String),
}

impl FinishReason {
    /// Returns `true` if the model naturally completed its response.
    ///
    /// Only the [`FinishReason::Complete`] variant returns `true`.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, FinishReason::Complete)
    }
}

// ---------------------------------------------------------------------------
// ResponseToolCall
// ---------------------------------------------------------------------------

/// A tool call requested by the model in its response.
///
/// Unlike the request-side `ToolCall` (which stores arguments as a JSON
/// string per OpenAI convention), response tool calls store arguments as
/// parsed [`Json`]. Codecs parse OpenAI's string arguments during decode;
/// Anthropic's `input` is already parsed JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The function/tool name.
    pub name: String,
    /// The arguments as parsed JSON (not a string).
    pub arguments: Json,
}

// ---------------------------------------------------------------------------
// ApiSpecificResponse
// ---------------------------------------------------------------------------

/// API-specific response data that cannot be normalized across providers.
///
/// Each variant captures fields unique to a particular LLM API, stored via
/// internal tagging on the `"api"` key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "api")]
pub enum ApiSpecificResponse {
    /// OpenAI Chat Completions-specific fields.
    #[serde(rename = "openai_chat")]
    OpenAIChat {
        /// Token-level log probabilities (raw JSON, too complex to normalize).
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Json>,
        /// System fingerprint for reproducibility.
        #[serde(skip_serializing_if = "Option::is_none")]
        system_fingerprint: Option<String>,
        /// Processing tier used (e.g., "default").
        #[serde(skip_serializing_if = "Option::is_none")]
        service_tier: Option<String>,
    },

    /// OpenAI Responses API-specific fields.
    #[serde(rename = "openai_responses")]
    OpenAIResponses {
        /// Full output items array for direct access.
        #[serde(skip_serializing_if = "Option::is_none")]
        output_items: Option<Vec<Json>>,
        /// Response status (e.g., "completed", "incomplete").
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        /// Details about why the response is incomplete.
        #[serde(skip_serializing_if = "Option::is_none")]
        incomplete_details: Option<Json>,
    },

    /// Anthropic Messages API-specific fields.
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages {
        /// Which stop sequence was matched (if any).
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_sequence: Option<String>,
        /// Full content blocks array for direct access.
        #[serde(skip_serializing_if = "Option::is_none")]
        content_blocks: Option<Vec<Json>>,
    },

    /// Custom/unknown API -- catch-all for user-implemented codecs.
    #[serde(rename = "custom")]
    Custom {
        /// API identifier.
        api_name: String,
        /// Opaque API-specific data.
        data: Json,
    },
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

impl AnnotatedLLMResponse {
    /// Extract the text content of the response message.
    ///
    /// For [`MessageContent::Text`], returns the string directly.
    /// For [`MessageContent::Parts`], returns the text of the first
    /// [`ContentPart::Text`] part.
    /// Returns `None` if `message` is `None`.
    #[must_use]
    pub fn response_text(&self) -> Option<&str> {
        match self.message.as_ref()? {
            MessageContent::Text(s) => Some(s.as_str()),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| {
                    let super::request::ContentPart::Text { text } = p;
                    text.as_str()
                })
                .next(),
        }
    }

    /// Check if the response contains any tool calls.
    ///
    /// Returns `true` if `tool_calls` is `Some` with at least one element.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use super::super::request::ContentPart;
    use super::super::traits::LlmResponseCodec;
    use crate::error::NexusError;

    /// Helper: build a fully-populated AnnotatedLLMResponse.
    fn full_response() -> AnnotatedLLMResponse {
        AnnotatedLLMResponse {
            id: Some("chatcmpl-abc123".into()),
            model: Some("gpt-4".into()),
            message: Some(MessageContent::Text("Hello, world!".into())),
            tool_calls: Some(vec![ResponseToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: json!({"city": "NYC"}),
            }]),
            finish_reason: Some(FinishReason::Complete),
            usage: Some(Usage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
                cache_read_tokens: Some(5),
                cache_write_tokens: Some(3),
            }),
            api_specific: Some(ApiSpecificResponse::OpenAIChat {
                logprobs: None,
                system_fingerprint: Some("fp_abc123".into()),
                service_tier: Some("default".into()),
            }),
            extra: serde_json::Map::new(),
        }
    }

    /// Helper: build a minimal AnnotatedLLMResponse (all None + empty extra).
    fn minimal_response() -> AnnotatedLLMResponse {
        AnnotatedLLMResponse {
            id: None,
            model: None,
            message: None,
            tool_calls: None,
            finish_reason: None,
            usage: None,
            api_specific: None,
            extra: serde_json::Map::new(),
        }
    }

    // -------------------------------------------------------------------
    // AnnotatedLLMResponse serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_annotated_llm_response_full_round_trip() {
        let resp = full_response();
        let json_val = serde_json::to_value(&resp).unwrap();
        let deserialized: AnnotatedLLMResponse = serde_json::from_value(json_val).unwrap();
        assert_eq!(resp, deserialized);
    }

    #[test]
    fn test_annotated_llm_response_minimal_round_trip() {
        let resp = minimal_response();
        let json_val = serde_json::to_value(&resp).unwrap();
        // Minimal response should serialize to just `{}`
        assert_eq!(json_val, json!({}));
        let deserialized: AnnotatedLLMResponse = serde_json::from_value(json_val).unwrap();
        assert_eq!(resp, deserialized);
    }

    // -------------------------------------------------------------------
    // Usage serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_usage_all_none_deserializes_from_empty() {
        let usage: Usage = serde_json::from_value(json!({})).unwrap();
        assert_eq!(usage, Usage::default());
        assert!(usage.prompt_tokens.is_none());
        assert!(usage.completion_tokens.is_none());
        assert!(usage.total_tokens.is_none());
        assert!(usage.cache_read_tokens.is_none());
        assert!(usage.cache_write_tokens.is_none());
    }

    #[test]
    fn test_usage_all_populated_round_trip() {
        let usage = Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            total_tokens: Some(150),
            cache_read_tokens: Some(20),
            cache_write_tokens: Some(10),
        };
        let json_val = serde_json::to_value(&usage).unwrap();
        let deserialized: Usage = serde_json::from_value(json_val).unwrap();
        assert_eq!(usage, deserialized);
    }

    // -------------------------------------------------------------------
    // FinishReason serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_finish_reason_complete_serializes_to_complete() {
        let reason = FinishReason::Complete;
        let json_val = serde_json::to_value(&reason).unwrap();
        assert_eq!(json_val, json!("complete"));
        let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
        assert_eq!(deserialized, FinishReason::Complete);
    }

    #[test]
    fn test_finish_reason_length_round_trip() {
        let reason = FinishReason::Length;
        let json_val = serde_json::to_value(&reason).unwrap();
        assert_eq!(json_val, json!("length"));
        let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
        assert_eq!(deserialized, FinishReason::Length);
    }

    #[test]
    fn test_finish_reason_tool_use_round_trip() {
        let reason = FinishReason::ToolUse;
        let json_val = serde_json::to_value(&reason).unwrap();
        assert_eq!(json_val, json!("tool_use"));
        let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
        assert_eq!(deserialized, FinishReason::ToolUse);
    }

    #[test]
    fn test_finish_reason_content_filter_round_trip() {
        let reason = FinishReason::ContentFilter;
        let json_val = serde_json::to_value(&reason).unwrap();
        assert_eq!(json_val, json!("content_filter"));
        let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
        assert_eq!(deserialized, FinishReason::ContentFilter);
    }

    #[test]
    fn test_finish_reason_unknown_round_trip() {
        let reason = FinishReason::Unknown("custom_reason".into());
        let json_val = serde_json::to_value(&reason).unwrap();
        let deserialized: FinishReason = serde_json::from_value(json_val).unwrap();
        assert_eq!(deserialized, FinishReason::Unknown("custom_reason".into()));
    }

    // -------------------------------------------------------------------
    // ResponseToolCall serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_response_tool_call_json_arguments_round_trip() {
        let tc = ResponseToolCall {
            id: "call_abc".into(),
            name: "search".into(),
            arguments: json!({"query": "weather", "limit": 5}),
        };
        let json_val = serde_json::to_value(&tc).unwrap();
        assert_eq!(json_val["arguments"]["query"], json!("weather"));
        assert_eq!(json_val["arguments"]["limit"], json!(5));
        let deserialized: ResponseToolCall = serde_json::from_value(json_val).unwrap();
        assert_eq!(tc, deserialized);
    }

    // -------------------------------------------------------------------
    // ApiSpecificResponse serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_api_specific_openai_chat_round_trip() {
        let api = ApiSpecificResponse::OpenAIChat {
            logprobs: Some(json!({"content": []})),
            system_fingerprint: Some("fp_abc".into()),
            service_tier: Some("default".into()),
        };
        let json_val = serde_json::to_value(&api).unwrap();
        assert_eq!(json_val["api"], json!("openai_chat"));
        let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
        assert_eq!(api, deserialized);
    }

    #[test]
    fn test_api_specific_openai_responses_round_trip() {
        let api = ApiSpecificResponse::OpenAIResponses {
            output_items: Some(vec![json!({"type": "message", "content": []})]),
            status: Some("completed".into()),
            incomplete_details: None,
        };
        let json_val = serde_json::to_value(&api).unwrap();
        assert_eq!(json_val["api"], json!("openai_responses"));
        let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
        assert_eq!(api, deserialized);
    }

    #[test]
    fn test_api_specific_anthropic_messages_round_trip() {
        let api = ApiSpecificResponse::AnthropicMessages {
            stop_sequence: Some("\n\nHuman:".into()),
            content_blocks: Some(vec![json!({"type": "text", "text": "Hello"})]),
        };
        let json_val = serde_json::to_value(&api).unwrap();
        assert_eq!(json_val["api"], json!("anthropic_messages"));
        let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
        assert_eq!(api, deserialized);
    }

    #[test]
    fn test_api_specific_custom_round_trip() {
        let api = ApiSpecificResponse::Custom {
            api_name: "my_custom_llm".into(),
            data: json!({"version": "2.0", "extra_field": true}),
        };
        let json_val = serde_json::to_value(&api).unwrap();
        assert_eq!(json_val["api"], json!("custom"));
        assert_eq!(json_val["api_name"], json!("my_custom_llm"));
        let deserialized: ApiSpecificResponse = serde_json::from_value(json_val).unwrap();
        assert_eq!(api, deserialized);
    }

    // -------------------------------------------------------------------
    // Helper: response_text()
    // -------------------------------------------------------------------

    #[test]
    fn test_response_text_returns_some_for_text_content() {
        let resp = AnnotatedLLMResponse {
            message: Some(MessageContent::Text("Hello!".into())),
            ..minimal_response()
        };
        assert_eq!(resp.response_text(), Some("Hello!"));
    }

    #[test]
    fn test_response_text_returns_none_when_message_is_none() {
        let resp = minimal_response();
        assert_eq!(resp.response_text(), None);
    }

    #[test]
    fn test_response_text_extracts_first_text_from_parts() {
        let resp = AnnotatedLLMResponse {
            message: Some(MessageContent::Parts(vec![ContentPart::Text {
                text: "Part text".into(),
            }])),
            ..minimal_response()
        };
        assert_eq!(resp.response_text(), Some("Part text"));
    }

    // -------------------------------------------------------------------
    // Helper: has_tool_calls()
    // -------------------------------------------------------------------

    #[test]
    fn test_has_tool_calls_true_when_present() {
        let resp = AnnotatedLLMResponse {
            tool_calls: Some(vec![ResponseToolCall {
                id: "tc_1".into(),
                name: "search".into(),
                arguments: json!({}),
            }]),
            ..minimal_response()
        };
        assert!(resp.has_tool_calls());
    }

    #[test]
    fn test_has_tool_calls_false_when_none() {
        let resp = minimal_response();
        assert!(!resp.has_tool_calls());
    }

    #[test]
    fn test_has_tool_calls_false_when_empty_vec() {
        let resp = AnnotatedLLMResponse {
            tool_calls: Some(vec![]),
            ..minimal_response()
        };
        assert!(!resp.has_tool_calls());
    }

    // -------------------------------------------------------------------
    // Helper: FinishReason::is_complete()
    // -------------------------------------------------------------------

    #[test]
    fn test_is_complete_true_for_complete() {
        assert!(FinishReason::Complete.is_complete());
    }

    #[test]
    fn test_is_complete_false_for_other_variants() {
        assert!(!FinishReason::Length.is_complete());
        assert!(!FinishReason::ToolUse.is_complete());
        assert!(!FinishReason::ContentFilter.is_complete());
        assert!(!FinishReason::Unknown("other".into()).is_complete());
    }

    // -------------------------------------------------------------------
    // LlmResponseCodec trait: mock implementation
    // -------------------------------------------------------------------

    struct MockResponseCodec;

    impl LlmResponseCodec for MockResponseCodec {
        fn decode_response(&self, _response: &Json) -> crate::error::Result<AnnotatedLLMResponse> {
            Ok(AnnotatedLLMResponse {
                id: Some("mock-id".into()),
                model: Some("mock-model".into()),
                message: Some(MessageContent::Text("mock response".into())),
                tool_calls: None,
                finish_reason: Some(FinishReason::Complete),
                usage: None,
                api_specific: None,
                extra: serde_json::Map::new(),
            })
        }
    }

    #[test]
    fn test_mock_response_codec_compiles_and_returns_ok() {
        let codec = MockResponseCodec;
        let result = codec.decode_response(&json!({}));
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.id, Some("mock-id".into()));
        assert_eq!(resp.model, Some("mock-model".into()));
    }

    struct FailingMockResponseCodec;

    impl LlmResponseCodec for FailingMockResponseCodec {
        fn decode_response(&self, _response: &Json) -> crate::error::Result<AnnotatedLLMResponse> {
            Err(NexusError::Internal("decode failed".into()))
        }
    }

    #[test]
    fn test_failing_mock_codec_demonstrates_non_fatal_pattern() {
        let codec = FailingMockResponseCodec;
        let result = codec.decode_response(&json!({"choices": []}));
        assert!(result.is_err());

        // Non-fatal pattern: callers use .ok() to convert Err to None
        let annotated: Option<AnnotatedLLMResponse> =
            codec.decode_response(&json!({"choices": []})).ok();
        assert!(annotated.is_none());
    }

    // -------------------------------------------------------------------
    // Extra field (flatten) captures unmodeled keys
    // -------------------------------------------------------------------

    #[test]
    fn test_annotated_llm_response_extra_captures_unmodeled_keys() {
        let json_val = json!({
            "id": "test-123",
            "model": "gpt-4",
            "custom_field": "custom_value",
            "another_field": 42
        });
        let resp: AnnotatedLLMResponse = serde_json::from_value(json_val).unwrap();
        assert_eq!(resp.id, Some("test-123".into()));
        assert_eq!(resp.model, Some("gpt-4".into()));
        assert_eq!(resp.extra.get("custom_field"), Some(&json!("custom_value")));
        assert_eq!(resp.extra.get("another_field"), Some(&json!(42)));

        // Round-trip: extra fields should appear as top-level keys
        let serialized = serde_json::to_value(&resp).unwrap();
        assert_eq!(serialized["custom_field"], json!("custom_value"));
        assert_eq!(serialized["another_field"], json!(42));
    }
}
