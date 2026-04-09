// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the OpenAI Responses API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the OpenAI Responses API format.
//!
//! The Responses API differs significantly from Chat Completions:
//! - **Response**: Heterogeneous `output` array (message, function_call, reasoning)
//!   instead of `choices[0].message`.
//! - **Finish reason**: Derived from `status` + `incomplete_details.reason`
//!   instead of `finish_reason` field.
//! - **Request**: Uses `input` (string or array) instead of `messages`, and
//!   `instructions` (top-level) instead of system message.
//! - **Max tokens**: `max_output_tokens` instead of `max_tokens`.

use serde::Deserialize;

use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::types::LLMRequest;

use super::request::{
    AnnotatedLLMRequest, GenerationParams, Message, MessageContent, ToolChoice, ToolDefinition,
};
use super::response::{
    AnnotatedLLMResponse, ApiSpecificResponse, FinishReason, ResponseToolCall, Usage,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Responses API.
pub struct OpenAIResponsesCodec;

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawResponsesResponse {
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    output: Option<Vec<Json>>,
    usage: Option<RawResponsesUsage>,
    incomplete_details: Option<Json>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawResponsesUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    input_tokens_details: Option<RawInputTokensDetails>,
}

#[derive(Deserialize)]
struct RawInputTokensDetails {
    cached_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map Responses API `status` + `incomplete_details` to normalized [`FinishReason`].
fn map_responses_finish_reason(
    status: Option<&str>,
    incomplete_details: Option<&Json>,
) -> Option<FinishReason> {
    let incomplete_reason = incomplete_details
        .and_then(|d| d.get("reason"))
        .and_then(|r| r.as_str());

    match status {
        Some("completed") => Some(FinishReason::Complete),
        Some("incomplete") => match incomplete_reason {
            Some("max_output_tokens") => Some(FinishReason::Length),
            Some("content_filter") => Some(FinishReason::ContentFilter),
            Some(other) => Some(FinishReason::Unknown(other.to_string())),
            None => Some(FinishReason::Unknown("incomplete".to_string())),
        },
        Some(other) => Some(FinishReason::Unknown(other.to_string())),
        None => None,
    }
}

/// Parse OpenAI tool call arguments from JSON string to [`Json`] value.
///
/// Falls back to [`Json::String`] if parsing fails (malformed model output).
fn parse_arguments(arguments: &str) -> Json {
    serde_json::from_str(arguments).unwrap_or_else(|_| Json::String(arguments.to_string()))
}

/// Keys that are modeled in [`AnnotatedLLMRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "input",
    "instructions",
    "model",
    "max_output_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
];

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for OpenAIResponsesCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLLMResponse> {
        let raw: RawResponsesResponse = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("OpenAI Responses response decode: {e}")))?;

        // Process heterogeneous output items.
        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ResponseToolCall> = Vec::new();
        let all_output_items = raw.output.clone();

        if let Some(ref items) = raw.output {
            for item in items {
                let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match item_type {
                    "message" => {
                        // Extract text from content[type=output_text].text
                        if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                            for block in content {
                                let block_type =
                                    block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                if block_type == "output_text"
                                    && let Some(text) = block.get("text").and_then(|t| t.as_str())
                                {
                                    text_parts.push(text.to_string());
                                }
                            }
                        }
                    }
                    "function_call" => {
                        // CRITICAL: use call_id, NOT id
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .map(parse_arguments)
                            .unwrap_or(Json::Object(serde_json::Map::new()));
                        tool_calls.push(ResponseToolCall {
                            id: call_id,
                            name,
                            arguments,
                        });
                    }
                    // "reasoning" and other types: captured in api_specific.output_items
                    _ => {}
                }
            }
        }

        // Build message content from collected text parts.
        let message = if text_parts.is_empty() {
            None
        } else if text_parts.len() == 1 {
            Some(MessageContent::Text(text_parts.into_iter().next().unwrap()))
        } else {
            Some(MessageContent::Text(text_parts.join("\n")))
        };

        // Map tool calls: None if empty, Some if present.
        let tool_calls = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };

        // Map finish reason from status + incomplete_details.
        let finish_reason =
            map_responses_finish_reason(raw.status.as_deref(), raw.incomplete_details.as_ref());

        // Map usage.
        let usage = raw.usage.map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.total_tokens,
            cache_read_tokens: u.input_tokens_details.and_then(|d| d.cached_tokens),
            cache_write_tokens: None,
        });

        // Build API-specific fields.
        let api_specific = Some(ApiSpecificResponse::OpenAIResponses {
            output_items: all_output_items,
            status: raw.status,
            incomplete_details: raw.incomplete_details,
        });

        Ok(AnnotatedLLMResponse {
            id: raw.id,
            model: raw.model,
            message,
            tool_calls,
            finish_reason,
            usage,
            api_specific,
            extra: raw.extra,
        })
    }
}

// ---------------------------------------------------------------------------
// LlmCodec implementation
// ---------------------------------------------------------------------------

impl LlmCodec for OpenAIResponsesCodec {
    fn decode(&self, request: &LLMRequest) -> Result<AnnotatedLLMRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;

        let mut messages: Vec<Message> = Vec::new();

        // Extract instructions -> system message (first).
        if let Some(instructions) = obj.get("instructions").and_then(|v| v.as_str()) {
            messages.push(Message::System {
                content: MessageContent::Text(instructions.to_string()),
                name: None,
            });
        }

        // Extract input.
        if let Some(input) = obj.get("input") {
            if let Some(s) = input.as_str() {
                // Input is a simple string -> single User message.
                messages.push(Message::User {
                    content: MessageContent::Text(s.to_string()),
                    name: None,
                });
            } else if input.is_array() {
                // Input is an array of message items.
                let input_messages: Vec<Message> =
                    serde_json::from_value(input.clone()).unwrap_or_default();
                messages.extend(input_messages);
            }
        }

        // Extract model.
        let model = obj.get("model").and_then(|v| v.as_str()).map(String::from);

        // Extract generation params.
        let temperature = obj.get("temperature").and_then(|v| v.as_f64());
        let top_p = obj.get("top_p").and_then(|v| v.as_f64());
        let max_tokens = obj.get("max_output_tokens").and_then(|v| v.as_u64());
        // Responses API does not support stop sequences.

        let params = if temperature.is_some() || max_tokens.is_some() || top_p.is_some() {
            Some(GenerationParams {
                temperature,
                max_tokens,
                top_p,
                stop: None,
            })
        } else {
            None
        };

        // Extract tools.
        let tools: Option<Vec<ToolDefinition>> = obj
            .get("tools")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| FlowError::Internal(format!("OpenAI Responses tools decode: {e}")))?;

        // Extract tool_choice.
        let tool_choice: Option<ToolChoice> = obj
            .get("tool_choice")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| {
                FlowError::Internal(format!("OpenAI Responses tool_choice decode: {e}"))
            })?;

        // Collect extra fields (keys not in MODELED_REQUEST_KEYS).
        let extra: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| !MODELED_REQUEST_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Ok(AnnotatedLLMRequest {
            messages,
            model,
            params,
            tools,
            tool_choice,
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLLMRequest, original: &LLMRequest) -> Result<LLMRequest> {
        let mut content = original.content.clone();
        let obj = content
            .as_object_mut()
            .ok_or_else(|| FlowError::Internal("original content is not an object".into()))?;

        // Extract system message -> write as instructions (top-level string).
        // Remaining messages -> write as input array.
        let mut system_text: Option<String> = None;
        let mut input_messages: Vec<&Message> = Vec::new();

        for msg in &annotated.messages {
            match msg {
                Message::System { content, .. } => {
                    if let MessageContent::Text(s) = content {
                        system_text = Some(s.clone());
                    }
                }
                other => {
                    input_messages.push(other);
                }
            }
        }

        if let Some(instructions) = system_text {
            obj.insert("instructions".into(), Json::String(instructions));
        } else {
            obj.remove("instructions");
        }

        // Write input from non-system messages.
        let input_val = serde_json::to_value(&input_messages)
            .map_err(|e| FlowError::Internal(format!("OpenAI Responses input encode: {e}")))?;
        obj.insert("input".into(), input_val);

        // Overlay model if present.
        if let Some(ref model) = annotated.model {
            obj.insert("model".into(), Json::String(model.clone()));
        }

        // Overlay generation params.
        if let Some(ref params) = annotated.params {
            if let Some(temp) = params.temperature {
                obj.insert("temperature".into(), json_f64(temp));
            }
            if let Some(top_p) = params.top_p {
                obj.insert("top_p".into(), json_f64(top_p));
            }
            if let Some(max_tokens) = params.max_tokens {
                // Responses API uses max_output_tokens.
                obj.insert("max_output_tokens".into(), Json::from(max_tokens));
                obj.remove("max_tokens");
            }
        }

        // Overlay tools if present.
        if let Some(ref tools) = annotated.tools {
            obj.insert(
                "tools".into(),
                serde_json::to_value(tools).map_err(|e| {
                    FlowError::Internal(format!("OpenAI Responses tools encode: {e}"))
                })?,
            );
        }

        // Overlay tool_choice if present.
        if let Some(ref tool_choice) = annotated.tool_choice {
            obj.insert(
                "tool_choice".into(),
                serde_json::to_value(tool_choice).map_err(|e| {
                    FlowError::Internal(format!("OpenAI Responses tool_choice encode: {e}"))
                })?,
            );
        }

        // Merge extra fields back.
        for (k, v) in &annotated.extra {
            obj.insert(k.clone(), v.clone());
        }

        Ok(LLMRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use super::super::request::MessageContent;
    use super::super::response::{ApiSpecificResponse, FinishReason};

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    fn make_request(content: Json) -> LLMRequest {
        LLMRequest {
            headers: serde_json::Map::new(),
            content,
        }
    }

    /// Full Responses API response with message, function_call, reasoning, and usage.
    fn full_responses_response() -> Json {
        json!({
            "id": "resp_abc123",
            "object": "response",
            "created_at": 1746989954.0,
            "model": "gpt-4o-2024-08-06",
            "status": "completed",
            "output": [
                {
                    "id": "rs_abc123",
                    "type": "reasoning",
                    "summary": [],
                    "status": null,
                    "encrypted_content": "gAAAAABo..."
                },
                {
                    "id": "msg_abc123",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello!",
                            "annotations": []
                        }
                    ]
                },
                {
                    "type": "function_call",
                    "id": "fc_abc123",
                    "name": "get_weather",
                    "call_id": "call_abc123",
                    "arguments": "{\"city\":\"NYC\"}",
                    "status": "completed"
                }
            ],
            "usage": {
                "input_tokens": 75,
                "output_tokens": 1186,
                "total_tokens": 1261,
                "input_tokens_details": { "cached_tokens": 10 },
                "output_tokens_details": { "reasoning_tokens": 1024 }
            }
        })
    }

    // ===================================================================
    // Response decode tests
    // ===================================================================

    #[test]
    fn test_decode_full_response() {
        let codec = OpenAIResponsesCodec;
        let resp = codec.decode_response(&full_responses_response()).unwrap();

        assert_eq!(resp.id, Some("resp_abc123".into()));
        assert_eq!(resp.model, Some("gpt-4o-2024-08-06".into()));

        // Text from output_text items
        assert_eq!(resp.message, Some(MessageContent::Text("Hello!".into())));

        // Tool calls from function_call items
        let tool_calls = resp.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc123"); // call_id, NOT id
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));

        // Finish reason from status
        assert_eq!(resp.finish_reason, Some(FinishReason::Complete));

        // Usage mapping
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(75)); // input_tokens -> prompt_tokens
        assert_eq!(usage.completion_tokens, Some(1186)); // output_tokens -> completion_tokens
        assert_eq!(usage.total_tokens, Some(1261));
        assert_eq!(usage.cache_read_tokens, Some(10));
        assert_eq!(usage.cache_write_tokens, None);

        // API specific fields
        match resp.api_specific.unwrap() {
            ApiSpecificResponse::OpenAIResponses {
                output_items,
                status,
                incomplete_details,
            } => {
                assert_eq!(status, Some("completed".into()));
                assert!(output_items.is_some());
                assert_eq!(output_items.unwrap().len(), 3);
                assert!(incomplete_details.is_none());
            }
            other => panic!("Expected OpenAIResponses, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_response_status_completed() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "status": "completed",
            "output": []
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::Complete));
    }

    #[test]
    fn test_decode_response_status_incomplete_max_output_tokens() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "status": "incomplete",
            "output": [],
            "incomplete_details": { "reason": "max_output_tokens" }
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn test_decode_response_status_incomplete_content_filter() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "status": "incomplete",
            "output": [],
            "incomplete_details": { "reason": "content_filter" }
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::ContentFilter));
    }

    #[test]
    fn test_decode_response_status_failed() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "status": "failed",
            "output": []
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(
            resp.finish_reason,
            Some(FinishReason::Unknown("failed".into()))
        );
    }

    #[test]
    fn test_decode_response_status_incomplete_no_details() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "status": "incomplete",
            "output": []
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(
            resp.finish_reason,
            Some(FinishReason::Unknown("incomplete".into()))
        );
    }

    #[test]
    fn test_decode_response_function_call_uses_call_id() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "output": [{
                "type": "function_call",
                "id": "fc_should_not_be_used",
                "name": "search",
                "call_id": "call_correct_id",
                "arguments": "{}",
                "status": "completed"
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        let tc = &resp.tool_calls.unwrap()[0];
        assert_eq!(tc.id, "call_correct_id");
        assert_ne!(tc.id, "fc_should_not_be_used");
    }

    #[test]
    fn test_decode_response_tool_call_arguments_parsed() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "name": "search",
                "call_id": "call_1",
                "arguments": "{\"query\":\"weather\",\"limit\":5}",
                "status": "completed"
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        let tc = &resp.tool_calls.unwrap()[0];
        assert_eq!(tc.arguments, json!({"query": "weather", "limit": 5}));
        assert!(tc.arguments.is_object());
    }

    #[test]
    fn test_decode_response_usage_mapping() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "output": [],
            "usage": {
                "input_tokens": 75,
                "output_tokens": 1186,
                "total_tokens": 1261,
                "input_tokens_details": { "cached_tokens": 42 }
            }
        });
        let resp = codec.decode_response(&response).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(75));
        assert_eq!(usage.completion_tokens, Some(1186));
        assert_eq!(usage.total_tokens, Some(1261));
        assert_eq!(usage.cache_read_tokens, Some(42));
        assert_eq!(usage.cache_write_tokens, None);
    }

    #[test]
    fn test_decode_response_multiple_output_text_items() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        { "type": "output_text", "text": "First part." },
                        { "type": "output_text", "text": "Second part." }
                    ]
                }
            ]
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(
            resp.message,
            Some(MessageContent::Text("First part.\nSecond part.".into()))
        );
    }

    #[test]
    fn test_decode_response_only_reasoning_items() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "output": [{
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "gAAAAABo..."
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        // No message content when there's only reasoning
        assert_eq!(resp.message, None);
        // Reasoning items captured in api_specific
        match resp.api_specific.unwrap() {
            ApiSpecificResponse::OpenAIResponses { output_items, .. } => {
                let items = output_items.unwrap();
                assert_eq!(items.len(), 1);
                assert_eq!(items[0]["type"], "reasoning");
            }
            other => panic!("Expected OpenAIResponses, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_response_extra_fields_preserved() {
        let codec = OpenAIResponsesCodec;
        let response = json!({
            "id": "resp_test",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "gpt-4o",
            "status": "completed",
            "output": [],
            "custom_future_field": "preserved_value",
            "another_field": 42
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.extra.get("object"), Some(&json!("response")));
        assert_eq!(resp.extra.get("created_at"), Some(&json!(1234567890.0)));
        assert_eq!(
            resp.extra.get("custom_future_field"),
            Some(&json!("preserved_value"))
        );
        assert_eq!(resp.extra.get("another_field"), Some(&json!(42)));
    }

    #[test]
    fn test_decode_minimal_response() {
        let codec = OpenAIResponsesCodec;
        let response = json!({});
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.id, None);
        assert_eq!(resp.model, None);
        assert_eq!(resp.message, None);
        assert!(resp.tool_calls.is_none() || resp.tool_calls.as_ref().unwrap().is_empty());
        assert_eq!(resp.usage, None);
    }

    #[test]
    fn test_decode_invalid_json() {
        let codec = OpenAIResponsesCodec;
        let response = json!("not an object");
        let result = codec.decode_response(&response);
        assert!(result.is_err());
    }

    // ===================================================================
    // Request decode tests
    // ===================================================================

    #[test]
    fn test_decode_request_with_input_array() {
        let codec = OpenAIResponsesCodec;
        let request = make_request(json!({
            "model": "gpt-4o",
            "instructions": "Be helpful and concise.",
            "input": [
                { "role": "user", "content": "What is 2+2?" },
                { "role": "assistant", "content": "4" },
                { "role": "user", "content": "And 3+3?" }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate math",
                    "parameters": {"type": "object"}
                }
            }]
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(annotated.model, Some("gpt-4o".into()));

        // instructions becomes system message (first)
        assert!(annotated.messages.len() >= 2);
        assert_eq!(annotated.system_prompt(), Some("Be helpful and concise."));

        // input items become messages (after system)
        // System + 3 input items = 4 total messages
        assert_eq!(annotated.messages.len(), 4);

        // Tools present
        let tools = annotated.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "calculate");
    }

    #[test]
    fn test_decode_request_with_input_string() {
        let codec = OpenAIResponsesCodec;
        let request = make_request(json!({
            "model": "gpt-4o",
            "input": "Hello, world!"
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(annotated.messages.len(), 1);
        assert_eq!(annotated.last_user_message(), Some("Hello, world!"));
    }

    #[test]
    fn test_decode_request_max_output_tokens() {
        let codec = OpenAIResponsesCodec;
        let request = make_request(json!({
            "model": "gpt-4o",
            "input": "Hi",
            "max_output_tokens": 500
        }));
        let annotated = codec.decode(&request).unwrap();
        let params = annotated.params.unwrap();
        assert_eq!(params.max_tokens, Some(500));
    }

    #[test]
    fn test_decode_request_extra_fields() {
        let codec = OpenAIResponsesCodec;
        let request = make_request(json!({
            "model": "gpt-4o",
            "input": "Hi",
            "store": true,
            "metadata": { "key": "value" },
            "tool_choice": "auto"
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(annotated.extra.get("store"), Some(&json!(true)));
        assert_eq!(
            annotated.extra.get("metadata"),
            Some(&json!({"key": "value"}))
        );
    }

    // ===================================================================
    // Request encode tests
    // ===================================================================

    #[test]
    fn test_encode_round_trip_preserves_unmodeled_fields() {
        let codec = OpenAIResponsesCodec;
        let original = make_request(json!({
            "model": "gpt-4o",
            "instructions": "Be helpful.",
            "input": [
                { "role": "user", "content": "Hello" }
            ],
            "store": true,
            "metadata": { "session": "abc" },
            "max_output_tokens": 100,
            "temperature": 0.7
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // Unmodeled fields preserved
        assert_eq!(obj.get("store"), Some(&json!(true)));
        assert_eq!(obj.get("metadata"), Some(&json!({"session": "abc"})));
    }

    #[test]
    fn test_encode_writes_instructions_and_input() {
        let codec = OpenAIResponsesCodec;
        let original = make_request(json!({
            "model": "gpt-4o",
            "instructions": "Be concise.",
            "input": [
                { "role": "user", "content": "Hello" }
            ]
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // instructions should be present
        assert!(obj.contains_key("instructions"));
        // input should be present
        assert!(obj.contains_key("input"));
        // Should NOT contain "messages"
        assert!(!obj.contains_key("messages"));
    }

    #[test]
    fn test_encode_writes_max_output_tokens() {
        let codec = OpenAIResponsesCodec;
        let original = make_request(json!({
            "model": "gpt-4o",
            "input": "Hi",
            "max_output_tokens": 200
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // Should use max_output_tokens, not max_tokens
        assert_eq!(obj.get("max_output_tokens"), Some(&json!(200)));
        assert!(!obj.contains_key("max_tokens"));
    }
}
