// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the OpenAI Chat Completions API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the OpenAI Chat Completions format.

use serde::Deserialize;

use crate::error::{NexusError, Result};
use crate::json::Json;
use crate::types::LLMRequest;

use super::request::{AnnotatedLLMRequest, GenerationParams, Message, ToolChoice, ToolDefinition};
use super::response::{
    AnnotatedLLMResponse, ApiSpecificResponse, FinishReason, ResponseToolCall, Usage,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
pub struct OpenAIChatCodec;

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawChatCompletion {
    id: Option<String>,
    model: Option<String>,
    choices: Option<Vec<RawChoice>>,
    usage: Option<RawChatUsage>,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawChoice {
    message: Option<RawMessage>,
    finish_reason: Option<String>,
    logprobs: Option<Json>,
}

#[derive(Deserialize)]
struct RawMessage {
    content: Option<String>,
    tool_calls: Option<Vec<RawToolCall>>,
}

#[derive(Deserialize)]
struct RawToolCall {
    id: String,
    function: RawFunction,
}

#[derive(Deserialize)]
struct RawFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct RawChatUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    prompt_tokens_details: Option<RawPromptTokensDetails>,
}

#[derive(Deserialize)]
struct RawPromptTokensDetails {
    cached_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map OpenAI Chat finish_reason string to normalized [`FinishReason`].
fn map_chat_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Complete,
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Unknown(other.to_string()),
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
    "messages",
    "model",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "stop",
    "tools",
    "tool_choice",
];

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for OpenAIChatCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLLMResponse> {
        let raw: RawChatCompletion = serde_json::from_value(response.clone())
            .map_err(|e| NexusError::Internal(format!("OpenAI Chat response decode: {e}")))?;

        // Extract first choice (if any).
        let choice = raw.choices.as_ref().and_then(|c| c.first());

        // Map message content.
        let message = choice
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.content.as_ref())
            .map(|s| super::request::MessageContent::Text(s.clone()));

        // Map tool calls.
        let tool_calls = choice
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.tool_calls.as_ref())
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| ResponseToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: parse_arguments(&tc.function.arguments),
                    })
                    .collect::<Vec<_>>()
            });

        // Map finish reason.
        let finish_reason = choice
            .and_then(|c| c.finish_reason.as_deref())
            .map(map_chat_finish_reason);

        // Map usage.
        let usage = raw.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cache_read_tokens: u.prompt_tokens_details.and_then(|d| d.cached_tokens),
            cache_write_tokens: None,
        });

        // Build API-specific fields.
        let logprobs = choice.and_then(|c| c.logprobs.clone());
        let api_specific = Some(ApiSpecificResponse::OpenAIChat {
            logprobs,
            system_fingerprint: raw.system_fingerprint,
            service_tier: raw.service_tier,
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

impl LlmCodec for OpenAIChatCodec {
    fn decode(&self, request: &LLMRequest) -> Result<AnnotatedLLMRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| NexusError::Internal("request content is not an object".into()))?;

        // Extract messages (default to empty vec if absent).
        let messages: Vec<Message> = obj
            .get("messages")
            .map(|v| serde_json::from_value(v.clone()).unwrap_or_default())
            .unwrap_or_default();

        // Extract model.
        let model = obj.get("model").and_then(|v| v.as_str()).map(String::from);

        // Extract generation params.
        let temperature = obj.get("temperature").and_then(|v| v.as_f64());
        let top_p = obj.get("top_p").and_then(|v| v.as_f64());
        let stop = obj
            .get("stop")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());

        // max_completion_tokens takes priority over max_tokens (newer API key).
        let max_tokens = obj
            .get("max_completion_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| obj.get("max_tokens").and_then(|v| v.as_u64()));

        let params =
            if temperature.is_some() || max_tokens.is_some() || top_p.is_some() || stop.is_some() {
                Some(GenerationParams {
                    temperature,
                    max_tokens,
                    top_p,
                    stop,
                })
            } else {
                None
            };

        // Extract tools.
        let tools: Option<Vec<ToolDefinition>> = obj
            .get("tools")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| NexusError::Internal(format!("OpenAI Chat tools decode: {e}")))?;

        // Extract tool_choice.
        let tool_choice: Option<ToolChoice> = obj
            .get("tool_choice")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| NexusError::Internal(format!("OpenAI Chat tool_choice decode: {e}")))?;

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
            .ok_or_else(|| NexusError::Internal("original content is not an object".into()))?;

        // Overlay messages.
        obj.insert(
            "messages".into(),
            serde_json::to_value(&annotated.messages)
                .map_err(|e| NexusError::Internal(format!("OpenAI Chat messages encode: {e}")))?,
        );

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
            if let Some(ref stop) = params.stop {
                obj.insert(
                    "stop".into(),
                    serde_json::to_value(stop).map_err(|e| {
                        NexusError::Internal(format!("OpenAI Chat stop encode: {e}"))
                    })?,
                );
            }
            if let Some(max_tokens) = params.max_tokens {
                // Use the same key that the original request used.
                if obj.contains_key("max_completion_tokens") {
                    obj.insert("max_completion_tokens".into(), Json::from(max_tokens));
                } else {
                    obj.insert("max_tokens".into(), Json::from(max_tokens));
                }
            }
        }

        // Overlay tools if present.
        if let Some(ref tools) = annotated.tools {
            obj.insert(
                "tools".into(),
                serde_json::to_value(tools)
                    .map_err(|e| NexusError::Internal(format!("OpenAI Chat tools encode: {e}")))?,
            );
        }

        // Overlay tool_choice if present.
        if let Some(ref tool_choice) = annotated.tool_choice {
            obj.insert(
                "tool_choice".into(),
                serde_json::to_value(tool_choice).map_err(|e| {
                    NexusError::Internal(format!("OpenAI Chat tool_choice encode: {e}"))
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

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
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

    /// Full Chat Completions response with text + tool calls + usage + cached tokens.
    fn full_chat_response() -> Json {
        json!({
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o-2024-08-06",
            "service_tier": "default",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"NYC\"}"
                        }
                    }]
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [{
                        "token": "Hello",
                        "logprob": -0.317
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
                "prompt_tokens_details": {
                    "cached_tokens": 5
                }
            }
        })
    }

    // ===================================================================
    // Response decode tests
    // ===================================================================

    #[test]
    fn test_decode_full_response() {
        let codec = OpenAIChatCodec;
        let resp = codec.decode_response(&full_chat_response()).unwrap();

        assert_eq!(resp.id, Some("chatcmpl-abc123".into()));
        assert_eq!(resp.model, Some("gpt-4o-2024-08-06".into()));
        assert_eq!(resp.message, Some(MessageContent::Text("Hello!".into())));
        assert_eq!(resp.finish_reason, Some(FinishReason::Complete));

        let tool_calls = resp.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc123");
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));

        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(9));
        assert_eq!(usage.completion_tokens, Some(12));
        assert_eq!(usage.total_tokens, Some(21));
        assert_eq!(usage.cache_read_tokens, Some(5));
        assert_eq!(usage.cache_write_tokens, None);
    }

    #[test]
    fn test_decode_response_cached_tokens() {
        let codec = OpenAIChatCodec;
        let response = json!({
            "id": "chatcmpl-cached",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "Hi" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 42
                }
            }
        });
        let resp = codec.decode_response(&response).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.cache_read_tokens, Some(42));
    }

    #[test]
    fn test_decode_response_finish_reason_stop() {
        let codec = OpenAIChatCodec;
        let response = json!({
            "choices": [{
                "message": { "content": "done" },
                "finish_reason": "stop"
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::Complete));
    }

    #[test]
    fn test_decode_response_finish_reason_tool_calls() {
        let codec = OpenAIChatCodec;
        let response = json!({
            "choices": [{
                "message": { "content": null },
                "finish_reason": "tool_calls"
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::ToolUse));
    }

    #[test]
    fn test_decode_response_finish_reason_length() {
        let codec = OpenAIChatCodec;
        let response = json!({
            "choices": [{
                "message": { "content": "truncated" },
                "finish_reason": "length"
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn test_decode_response_finish_reason_content_filter() {
        let codec = OpenAIChatCodec;
        let response = json!({
            "choices": [{
                "message": { "content": "" },
                "finish_reason": "content_filter"
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::ContentFilter));
    }

    #[test]
    fn test_decode_response_finish_reason_unknown() {
        let codec = OpenAIChatCodec;
        let response = json!({
            "choices": [{
                "message": { "content": "" },
                "finish_reason": "some_new_reason"
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(
            resp.finish_reason,
            Some(FinishReason::Unknown("some_new_reason".into()))
        );
    }

    #[test]
    fn test_decode_response_tool_call_arguments_parsed() {
        let codec = OpenAIChatCodec;
        let response = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{\"query\":\"weather\",\"limit\":5}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let resp = codec.decode_response(&response).unwrap();
        let tc = &resp.tool_calls.unwrap()[0];
        assert_eq!(tc.arguments, json!({"query": "weather", "limit": 5}));
        // Arguments should be a Json object, not a Json::String
        assert!(tc.arguments.is_object());
    }

    #[test]
    fn test_decode_response_api_specific_fields() {
        let codec = OpenAIChatCodec;
        let resp = codec.decode_response(&full_chat_response()).unwrap();
        match resp.api_specific.unwrap() {
            ApiSpecificResponse::OpenAIChat {
                logprobs,
                system_fingerprint,
                service_tier,
            } => {
                assert!(logprobs.is_some());
                assert_eq!(system_fingerprint, Some("fp_44709d6fcb".into()));
                assert_eq!(service_tier, Some("default".into()));
            }
            other => panic!("Expected OpenAIChat, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_response_extra_fields_preserved() {
        let codec = OpenAIChatCodec;
        let response = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "message": { "content": "hi" },
                "finish_reason": "stop"
            }],
            "custom_future_field": "preserved_value",
            "another_field": 42
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.extra.get("object"), Some(&json!("chat.completion")));
        assert_eq!(resp.extra.get("created"), Some(&json!(1234567890)));
        assert_eq!(
            resp.extra.get("custom_future_field"),
            Some(&json!("preserved_value"))
        );
        assert_eq!(resp.extra.get("another_field"), Some(&json!(42)));
    }

    #[test]
    fn test_decode_minimal_response() {
        let codec = OpenAIChatCodec;
        let response = json!({});
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.id, None);
        assert_eq!(resp.model, None);
        assert_eq!(resp.message, None);
        assert_eq!(resp.tool_calls, None);
        assert_eq!(resp.finish_reason, None);
        assert_eq!(resp.usage, None);
    }

    #[test]
    fn test_decode_invalid_json_type() {
        let codec = OpenAIChatCodec;
        // A JSON string (not an object) should fail to decode
        let response = json!("not an object");
        let result = codec.decode_response(&response);
        assert!(result.is_err());
    }

    // ===================================================================
    // Request decode tests
    // ===================================================================

    #[test]
    fn test_decode_request_full() {
        let codec = OpenAIChatCodec;
        let request = make_request(json!({
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 100,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": "auto"
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(annotated.messages.len(), 2);
        assert_eq!(annotated.model, Some("gpt-4o".into()));

        let params = annotated.params.unwrap();
        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.max_tokens, Some(100));

        let tools = annotated.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");

        assert_eq!(annotated.tool_choice, Some(ToolChoice::Auto));
    }

    #[test]
    fn test_decode_request_max_completion_tokens() {
        let codec = OpenAIChatCodec;
        let request = make_request(json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt-4o",
            "max_completion_tokens": 200
        }));
        let annotated = codec.decode(&request).unwrap();
        let params = annotated.params.unwrap();
        assert_eq!(params.max_tokens, Some(200));
    }

    #[test]
    fn test_decode_request_extra_fields() {
        let codec = OpenAIChatCodec;
        let request = make_request(json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt-4o",
            "stream": true,
            "seed": 42,
            "response_format": {"type": "json_object"}
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(annotated.extra.get("stream"), Some(&json!(true)));
        assert_eq!(annotated.extra.get("seed"), Some(&json!(42)));
        assert_eq!(
            annotated.extra.get("response_format"),
            Some(&json!({"type": "json_object"}))
        );
    }

    #[test]
    fn test_decode_request_no_messages_key() {
        let codec = OpenAIChatCodec;
        let request = make_request(json!({
            "model": "gpt-4o"
        }));
        let annotated = codec.decode(&request).unwrap();
        assert!(annotated.messages.is_empty());
    }

    // ===================================================================
    // Request encode tests
    // ===================================================================

    #[test]
    fn test_encode_round_trip_preserves_unmodeled_fields() {
        let codec = OpenAIChatCodec;
        let original = make_request(json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
            "stream": true,
            "seed": 42,
            "temperature": 0.7
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // Unmodeled fields preserved
        assert_eq!(obj.get("stream"), Some(&json!(true)));
        assert_eq!(obj.get("seed"), Some(&json!(42)));
        // Modeled fields present
        assert!(obj.contains_key("messages"));
        assert_eq!(obj.get("model"), Some(&json!("gpt-4o")));
    }

    #[test]
    fn test_encode_with_modified_model() {
        let codec = OpenAIChatCodec;
        let original = make_request(json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o"
        }));
        let mut annotated = codec.decode(&original).unwrap();
        annotated.model = Some("gpt-4o-mini".into());
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        assert_eq!(obj.get("model"), Some(&json!("gpt-4o-mini")));
    }

    #[test]
    fn test_encode_restores_max_completion_tokens_key() {
        let codec = OpenAIChatCodec;
        let original = make_request(json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
            "max_completion_tokens": 200
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // Should write back to max_completion_tokens, not max_tokens
        assert_eq!(obj.get("max_completion_tokens"), Some(&json!(200)));
        assert!(!obj.contains_key("max_tokens"));
    }
}
