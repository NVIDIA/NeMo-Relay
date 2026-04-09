// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the Anthropic Messages API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the Anthropic Messages API format.
//!
//! # Anthropic-specific patterns handled
//!
//! - **Content blocks**: Heterogeneous array of `text`, `tool_use`, `thinking`,
//!   `redacted_thinking`, `mcp_tool_use`, `server_tool_use` blocks
//! - **Top-level system**: System prompt is a top-level field, not inside messages
//! - **stop_reason**: Maps to [`FinishReason`] (not `finish_reason`)
//! - **Tool definitions**: Uses `input_schema` instead of `parameters`
//! - **Tool choice**: `{"type":"auto"}` / `{"type":"any"}` / `{"type":"tool","name":"..."}`
//! - **Cache tokens**: `cache_read_input_tokens` / `cache_creation_input_tokens`

use serde::Deserialize;

use crate::error::{NexusError, Result};
use crate::json::Json;
use crate::types::LLMRequest;

use super::request::{
    AnnotatedLLMRequest, FunctionDefinition, GenerationParams, Message, MessageContent, ToolChoice,
    ToolChoiceFunction, ToolChoiceFunctionName, ToolDefinition,
};
use super::response::{
    AnnotatedLLMResponse, ApiSpecificResponse, FinishReason, ResponseToolCall, Usage,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the Anthropic Messages API.
pub struct AnthropicMessagesCodec;

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawAnthropicResponse {
    id: Option<String>,
    model: Option<String>,
    content: Option<Vec<Json>>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: Option<RawAnthropicUsage>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawAnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map Anthropic `stop_reason` string to normalized [`FinishReason`].
fn map_anthropic_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" => FinishReason::Complete,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolUse,
        other => FinishReason::Unknown(other.to_string()),
    }
}

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

/// Keys that are modeled in [`AnnotatedLLMRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "system",
    "messages",
    "model",
    "max_tokens",
    "temperature",
    "top_p",
    "stop_sequences",
    "tools",
    "tool_choice",
];

/// Decode the Anthropic `tool_choice` JSON value into a normalized [`ToolChoice`].
///
/// Anthropic format:
/// - `{"type": "auto"}` -> `ToolChoice::Auto`
/// - `{"type": "any"}` -> `ToolChoice::Required`
/// - `{"type": "tool", "name": "X"}` -> `ToolChoice::Specific`
fn decode_anthropic_tool_choice(val: &Json) -> Option<ToolChoice> {
    let obj = val.as_object()?;
    let tc_type = obj.get("type")?.as_str()?;
    match tc_type {
        "auto" => Some(ToolChoice::Auto),
        "any" => Some(ToolChoice::Required),
        "tool" => {
            let name = obj.get("name")?.as_str()?.to_string();
            Some(ToolChoice::Specific(ToolChoiceFunction {
                choice_type: "function".into(),
                function: ToolChoiceFunctionName { name },
            }))
        }
        _ => None,
    }
}

/// Encode a normalized [`ToolChoice`] back into Anthropic JSON format.
fn encode_anthropic_tool_choice(tc: &ToolChoice) -> Json {
    match tc {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::Required => serde_json::json!({"type": "any"}),
        ToolChoice::None => serde_json::json!({"type": "auto"}), // Anthropic has no "none"; fall back to auto
        ToolChoice::Specific(func) => {
            serde_json::json!({"type": "tool", "name": func.function.name})
        }
    }
}

/// Extract the system prompt from an Anthropic top-level `system` field.
///
/// Handles both string and array-of-content-blocks formats.
fn extract_system_message(system_val: &Json) -> Option<Message> {
    if let Some(s) = system_val.as_str() {
        Some(Message::System {
            content: MessageContent::Text(s.to_string()),
            name: None,
        })
    } else if let Some(arr) = system_val.as_array() {
        // Array of content blocks -- extract text from each "text" block.
        let texts: Vec<&str> = arr
            .iter()
            .filter_map(|block| {
                let block_type = block.get("type")?.as_str()?;
                if block_type == "text" {
                    block.get("text")?.as_str()
                } else {
                    None
                }
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(Message::System {
                content: MessageContent::Text(texts.join("\n")),
                name: None,
            })
        }
    } else {
        None
    }
}

/// Extract system text from a [`Message::System`] for encoding back to top-level.
fn extract_system_text(msg: &Message) -> Option<String> {
    match msg {
        Message::System {
            content: MessageContent::Text(s),
            ..
        } => Some(s.clone()),
        Message::System {
            content: MessageContent::Parts(parts),
            ..
        } => {
            let texts: Vec<&str> = parts
                .iter()
                .map(|p| {
                    let super::request::ContentPart::Text { text } = p;
                    text.as_str()
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for AnthropicMessagesCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLLMResponse> {
        let raw: RawAnthropicResponse = serde_json::from_value(response.clone()).map_err(|e| {
            NexusError::Internal(format!("Anthropic Messages response decode: {e}"))
        })?;

        // Process content blocks.
        let content_blocks = raw.content.as_ref();

        // Extract text from all "text" blocks, concatenated with newline.
        let text_parts: Vec<&str> = content_blocks
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| {
                        let block_type = block.get("type")?.as_str()?;
                        if block_type == "text" {
                            block.get("text")?.as_str()
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let message = if text_parts.is_empty() {
            None
        } else {
            Some(MessageContent::Text(text_parts.join("\n")))
        };

        // Extract tool_use blocks (only "tool_use" type, NOT mcp_tool_use or server_tool_use).
        let tool_calls: Vec<ResponseToolCall> = content_blocks
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| {
                        let block_type = block.get("type")?.as_str()?;
                        if block_type == "tool_use" {
                            let id = block.get("id")?.as_str()?.to_string();
                            let name = block.get("name")?.as_str()?.to_string();
                            // CRITICAL: input is already parsed JSON -- clone directly.
                            let arguments = block.get("input")?.clone();
                            Some(ResponseToolCall {
                                id,
                                name,
                                arguments,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let tool_calls = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };

        // Map stop_reason to FinishReason.
        let finish_reason = raw.stop_reason.as_deref().map(map_anthropic_stop_reason);

        // Map usage.
        let usage = raw.usage.map(|u| {
            let prompt = u.input_tokens;
            let completion = u.output_tokens;
            Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                // Anthropic does not supply total_tokens; compute it.
                total_tokens: match (prompt, completion) {
                    (Some(p), Some(c)) => Some(p + c),
                    _ => None,
                },
                cache_read_tokens: u.cache_read_input_tokens,
                cache_write_tokens: u.cache_creation_input_tokens,
            }
        });

        // Build API-specific fields: all content blocks + stop_sequence.
        let api_specific_content_blocks = raw.content.clone();
        let api_specific = Some(ApiSpecificResponse::AnthropicMessages {
            stop_sequence: raw.stop_sequence,
            content_blocks: api_specific_content_blocks,
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

impl LlmCodec for AnthropicMessagesCodec {
    fn decode(&self, request: &LLMRequest) -> Result<AnnotatedLLMRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| NexusError::Internal("request content is not an object".into()))?;

        // Extract system from top-level field.
        let system_msg = obj.get("system").and_then(extract_system_message);

        // Extract messages (default to empty vec if absent).
        let mut messages: Vec<Message> = obj
            .get("messages")
            .map(|v| serde_json::from_value(v.clone()).unwrap_or_default())
            .unwrap_or_default();

        // Prepend system message if present.
        if let Some(sys) = system_msg {
            messages.insert(0, sys);
        }

        // Extract model.
        let model = obj.get("model").and_then(|v| v.as_str()).map(String::from);

        // Extract generation params.
        let temperature = obj.get("temperature").and_then(|v| v.as_f64());
        let top_p = obj.get("top_p").and_then(|v| v.as_f64());
        let max_tokens = obj.get("max_tokens").and_then(|v| v.as_u64());
        // Anthropic uses stop_sequences (not stop).
        let stop = obj
            .get("stop_sequences")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());

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

        // Extract tools: Anthropic uses flat structure (name, description, input_schema).
        // Normalize to ToolDefinition { type: "function", function: { name, description, parameters } }.
        let tools: Option<Vec<ToolDefinition>> = obj.get("tools").and_then(|v| {
            let arr = v.as_array()?;
            let defs: Vec<ToolDefinition> = arr
                .iter()
                .filter_map(|tool| {
                    let name = tool.get("name")?.as_str()?.to_string();
                    let description = tool
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(String::from);
                    let parameters = tool.get("input_schema").cloned();
                    Some(ToolDefinition {
                        tool_type: "function".into(),
                        function: FunctionDefinition {
                            name,
                            description,
                            parameters,
                        },
                    })
                })
                .collect();
            if defs.is_empty() {
                None
            } else {
                Some(defs)
            }
        });

        // Extract tool_choice: Anthropic format.
        let tool_choice = obj
            .get("tool_choice")
            .and_then(decode_anthropic_tool_choice);

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

        // Extract system message from annotated.messages, write as top-level field.
        // Remaining (non-system) messages go into the messages array.
        let mut non_system_messages = Vec::new();
        let mut system_text: Option<String> = None;

        for msg in &annotated.messages {
            if let Some(text) = extract_system_text(msg) {
                system_text = Some(text);
            } else {
                non_system_messages.push(msg);
            }
        }

        if let Some(text) = system_text {
            obj.insert("system".into(), Json::String(text));
        }

        // Overlay messages (non-system only).
        obj.insert(
            "messages".into(),
            serde_json::to_value(&non_system_messages).map_err(|e| {
                NexusError::Internal(format!("Anthropic Messages messages encode: {e}"))
            })?,
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
            // Write max_tokens (Anthropic key name).
            if let Some(max_tokens) = params.max_tokens {
                obj.insert("max_tokens".into(), Json::from(max_tokens));
            }
            // Write stop_sequences (Anthropic key name, not "stop").
            if let Some(ref stop) = params.stop {
                obj.insert(
                    "stop_sequences".into(),
                    serde_json::to_value(stop).map_err(|e| {
                        NexusError::Internal(format!(
                            "Anthropic Messages stop_sequences encode: {e}"
                        ))
                    })?,
                );
            }
        }

        // Overlay tools in Anthropic format: { name, description, input_schema }.
        // Denormalize from ToolDefinition (drop type/function wrapper, rename parameters -> input_schema).
        if let Some(ref tools) = annotated.tools {
            let anthropic_tools: Vec<Json> = tools
                .iter()
                .map(|td| {
                    let mut tool = serde_json::Map::new();
                    tool.insert("name".into(), Json::String(td.function.name.clone()));
                    if let Some(ref desc) = td.function.description {
                        tool.insert("description".into(), Json::String(desc.clone()));
                    }
                    if let Some(ref params) = td.function.parameters {
                        tool.insert("input_schema".into(), params.clone());
                    }
                    Json::Object(tool)
                })
                .collect();
            obj.insert(
                "tools".into(),
                serde_json::to_value(&anthropic_tools).map_err(|e| {
                    NexusError::Internal(format!("Anthropic Messages tools encode: {e}"))
                })?,
            );
        }

        // Overlay tool_choice in Anthropic format.
        if let Some(ref tool_choice) = annotated.tool_choice {
            obj.insert(
                "tool_choice".into(),
                encode_anthropic_tool_choice(tool_choice),
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

    use super::super::request::{Message, MessageContent, ToolChoice};
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

    /// Full Anthropic Messages response with text, tool_use, thinking, usage, etc.
    fn full_anthropic_response() -> Json {
        json!({
            "id": "msg_abc123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me analyze...",
                    "signature": "sig_xxx"
                },
                {
                    "type": "text",
                    "text": "Here is my answer."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "get_weather",
                    "input": { "city": "NYC" }
                },
                {
                    "type": "redacted_thinking",
                    "data": "gAAAAABo..."
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 1024,
                "output_tokens": 256,
                "cache_creation_input_tokens": 512,
                "cache_read_input_tokens": 0
            }
        })
    }

    // ===================================================================
    // Response decode tests
    // ===================================================================

    #[test]
    fn test_decode_full_response() {
        let codec = AnthropicMessagesCodec;
        let resp = codec.decode_response(&full_anthropic_response()).unwrap();

        assert_eq!(resp.id, Some("msg_abc123".into()));
        assert_eq!(resp.model, Some("claude-sonnet-4-20250514".into()));
        assert_eq!(
            resp.message,
            Some(MessageContent::Text("Here is my answer.".into()))
        );
        assert_eq!(resp.finish_reason, Some(FinishReason::Complete));

        let tool_calls = resp.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "toolu_abc123");
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));

        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(1024));
        assert_eq!(usage.completion_tokens, Some(256));
        assert_eq!(usage.total_tokens, Some(1280));
        assert_eq!(usage.cache_read_tokens, Some(0));
        assert_eq!(usage.cache_write_tokens, Some(512));
    }

    #[test]
    fn test_decode_response_multiple_text_blocks() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "id": "msg_multi",
            "model": "claude-sonnet-4-20250514",
            "content": [
                { "type": "text", "text": "First paragraph." },
                { "type": "text", "text": "Second paragraph." }
            ],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 20 }
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(
            resp.message,
            Some(MessageContent::Text(
                "First paragraph.\nSecond paragraph.".into()
            ))
        );
    }

    #[test]
    fn test_decode_response_tool_use_input_stored_as_json() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "id": "msg_tool",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_xyz",
                    "name": "search",
                    "input": { "query": "weather", "limit": 5 }
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 10, "output_tokens": 20 }
        });
        let resp = codec.decode_response(&response).unwrap();
        let tc = &resp.tool_calls.unwrap()[0];
        assert_eq!(tc.id, "toolu_xyz");
        assert_eq!(tc.name, "search");
        assert_eq!(tc.arguments, json!({"query": "weather", "limit": 5}));
        // Arguments should be a Json object, not a Json::String
        assert!(tc.arguments.is_object());
    }

    #[test]
    fn test_decode_response_finish_reason_end_turn() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "content": [{ "type": "text", "text": "done" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::Complete));
    }

    #[test]
    fn test_decode_response_finish_reason_max_tokens() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "content": [{ "type": "text", "text": "truncated" }],
            "stop_reason": "max_tokens",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn test_decode_response_finish_reason_tool_use() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "fn",
                "input": {}
            }],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::ToolUse));
    }

    #[test]
    fn test_decode_response_finish_reason_stop_sequence() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "content": [{ "type": "text", "text": "stopped" }],
            "stop_reason": "stop_sequence",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(
            resp.finish_reason,
            Some(FinishReason::Unknown("stop_sequence".into()))
        );
    }

    #[test]
    fn test_decode_response_usage_mapping() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "id": "msg_usage",
            "model": "claude-sonnet-4-20250514",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 25,
                "cache_creation_input_tokens": 10
            }
        });
        let resp = codec.decode_response(&response).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(50));
        assert_eq!(usage.total_tokens, Some(150));
        assert_eq!(usage.cache_read_tokens, Some(25));
        assert_eq!(usage.cache_write_tokens, Some(10));
    }

    #[test]
    fn test_decode_response_thinking_blocks_in_api_specific() {
        let codec = AnthropicMessagesCodec;
        let resp = codec.decode_response(&full_anthropic_response()).unwrap();
        match resp.api_specific.unwrap() {
            ApiSpecificResponse::AnthropicMessages {
                content_blocks,
                stop_sequence,
            } => {
                let blocks = content_blocks.unwrap();
                // Should contain ALL content blocks
                assert_eq!(blocks.len(), 4);
                // Verify thinking and redacted_thinking are present
                let types: Vec<&str> = blocks
                    .iter()
                    .filter_map(|b| b.get("type").and_then(|t| t.as_str()))
                    .collect();
                assert!(types.contains(&"thinking"));
                assert!(types.contains(&"redacted_thinking"));
                assert!(types.contains(&"text"));
                assert!(types.contains(&"tool_use"));
                assert_eq!(stop_sequence, None);
            }
            other => panic!("Expected AnthropicMessages, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_response_stop_sequence_value() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "id": "msg_seq",
            "model": "claude-sonnet-4-20250514",
            "content": [{ "type": "text", "text": "stopped" }],
            "stop_reason": "stop_sequence",
            "stop_sequence": "\n\nHuman:",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        });
        let resp = codec.decode_response(&response).unwrap();
        match resp.api_specific.unwrap() {
            ApiSpecificResponse::AnthropicMessages {
                stop_sequence,
                content_blocks: _,
            } => {
                assert_eq!(stop_sequence, Some("\n\nHuman:".into()));
            }
            other => panic!("Expected AnthropicMessages, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_response_extra_fields_preserved() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "id": "msg_extra",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{ "type": "text", "text": "hi" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 },
            "container": { "id": "container_abc123" }
        });
        let resp = codec.decode_response(&response).unwrap();
        // type, role, container should be in extra
        assert_eq!(resp.extra.get("type"), Some(&json!("message")));
        assert_eq!(resp.extra.get("role"), Some(&json!("assistant")));
        assert_eq!(
            resp.extra.get("container"),
            Some(&json!({"id": "container_abc123"}))
        );
    }

    #[test]
    fn test_decode_minimal_response() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "content": [],
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(resp.id, None);
        assert_eq!(resp.model, None);
        assert_eq!(resp.message, None);
        assert!(
            resp.tool_calls.is_none() || resp.tool_calls.as_ref().is_some_and(|t| t.is_empty())
        );
        assert_eq!(resp.finish_reason, None);
    }

    #[test]
    fn test_decode_invalid_json() {
        let codec = AnthropicMessagesCodec;
        let response = json!("not an object");
        let result = codec.decode_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_response_mcp_tool_use_not_in_tool_calls() {
        let codec = AnthropicMessagesCodec;
        let response = json!({
            "id": "msg_mcp",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "mcp_tool_use",
                    "id": "mcptoolu_abc123",
                    "name": "search",
                    "server_name": "my_server",
                    "input": { "query": "test" }
                },
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_abc123",
                    "name": "code_execution",
                    "input": { "code": "print(1+1)" }
                }
            ],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        });
        let resp = codec.decode_response(&response).unwrap();
        // mcp_tool_use and server_tool_use should NOT appear in normalized tool_calls
        assert!(
            resp.tool_calls.is_none() || resp.tool_calls.as_ref().is_some_and(|t| t.is_empty())
        );
        // But they should appear in api_specific content_blocks
        match resp.api_specific.unwrap() {
            ApiSpecificResponse::AnthropicMessages { content_blocks, .. } => {
                let blocks = content_blocks.unwrap();
                assert_eq!(blocks.len(), 2);
                let types: Vec<&str> = blocks
                    .iter()
                    .filter_map(|b| b.get("type").and_then(|t| t.as_str()))
                    .collect();
                assert!(types.contains(&"mcp_tool_use"));
                assert!(types.contains(&"server_tool_use"));
            }
            other => panic!("Expected AnthropicMessages, got {other:?}"),
        }
    }

    // ===================================================================
    // Request decode tests
    // ===================================================================

    #[test]
    fn test_decode_request_full() {
        let codec = AnthropicMessagesCodec;
        let request = make_request(json!({
            "system": "Be helpful",
            "messages": [
                { "role": "user", "content": "Hello" }
            ],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
            }],
            "tool_choice": { "type": "auto" }
        }));
        let annotated = codec.decode(&request).unwrap();

        // System should be prepended as Message::System
        assert_eq!(annotated.messages.len(), 2);
        assert!(
            matches!(&annotated.messages[0], Message::System { content: MessageContent::Text(t), .. } if t == "Be helpful")
        );
        assert!(matches!(&annotated.messages[1], Message::User { .. }));

        assert_eq!(annotated.model, Some("claude-sonnet-4-20250514".into()));

        let params = annotated.params.unwrap();
        assert_eq!(params.max_tokens, Some(1024));

        // Tools should be normalized: input_schema -> parameters
        let tools = annotated.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(tools[0].function.description, Some("Get weather".into()));
        assert!(tools[0].function.parameters.is_some());

        assert_eq!(annotated.tool_choice, Some(ToolChoice::Auto));
    }

    #[test]
    fn test_decode_request_system_array() {
        let codec = AnthropicMessagesCodec;
        let request = make_request(json!({
            "system": [
                { "type": "text", "text": "First instruction." },
                { "type": "text", "text": "Second instruction." }
            ],
            "messages": [
                { "role": "user", "content": "Hello" }
            ],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(annotated.messages.len(), 2);
        assert!(matches!(
            &annotated.messages[0],
            Message::System { content: MessageContent::Text(t), .. }
            if t == "First instruction.\nSecond instruction."
        ));
    }

    #[test]
    fn test_decode_request_stop_sequences() {
        let codec = AnthropicMessagesCodec;
        let request = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "stop_sequences": ["\n\nHuman:", "END"]
        }));
        let annotated = codec.decode(&request).unwrap();
        let params = annotated.params.unwrap();
        assert_eq!(params.stop, Some(vec!["\n\nHuman:".into(), "END".into()]));
    }

    #[test]
    fn test_decode_request_tool_choice_auto() {
        let codec = AnthropicMessagesCodec;
        let request = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": { "type": "auto" }
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(annotated.tool_choice, Some(ToolChoice::Auto));
    }

    #[test]
    fn test_decode_request_tool_choice_any() {
        let codec = AnthropicMessagesCodec;
        let request = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": { "type": "any" }
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(annotated.tool_choice, Some(ToolChoice::Required));
    }

    #[test]
    fn test_decode_request_tool_choice_specific() {
        let codec = AnthropicMessagesCodec;
        let request = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": { "type": "tool", "name": "get_weather" }
        }));
        let annotated = codec.decode(&request).unwrap();
        match annotated.tool_choice.unwrap() {
            ToolChoice::Specific(tc) => {
                assert_eq!(tc.function.name, "get_weather");
            }
            other => panic!("Expected Specific, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_request_extra_fields() {
        let codec = AnthropicMessagesCodec;
        let request = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "metadata": { "user_id": "abc" },
            "stream": true
        }));
        let annotated = codec.decode(&request).unwrap();
        assert_eq!(
            annotated.extra.get("metadata"),
            Some(&json!({"user_id": "abc"}))
        );
        assert_eq!(annotated.extra.get("stream"), Some(&json!(true)));
    }

    // ===================================================================
    // Request encode tests
    // ===================================================================

    #[test]
    fn test_encode_round_trip_preserves_unmodeled_fields() {
        let codec = AnthropicMessagesCodec;
        let original = make_request(json!({
            "system": "Be helpful",
            "messages": [{ "role": "user", "content": "Hello" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "metadata": { "user_id": "abc" },
            "stream": true
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // Unmodeled fields preserved
        assert_eq!(obj.get("metadata"), Some(&json!({"user_id": "abc"})));
        assert_eq!(obj.get("stream"), Some(&json!(true)));
    }

    #[test]
    fn test_encode_system_as_top_level() {
        let codec = AnthropicMessagesCodec;
        let original = make_request(json!({
            "system": "Original system",
            "messages": [{ "role": "user", "content": "Hello" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // System should be a top-level field, not in messages
        assert_eq!(obj.get("system"), Some(&json!("Original system")));
        // Messages array should not contain a system role message
        let messages = obj.get("messages").unwrap().as_array().unwrap();
        for msg in messages {
            assert_ne!(msg.get("role").and_then(|r| r.as_str()), Some("system"));
        }
    }

    #[test]
    fn test_encode_stop_sequences() {
        let codec = AnthropicMessagesCodec;
        let original = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "stop_sequences": ["\n\nHuman:"]
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // Should write stop_sequences (not stop)
        assert_eq!(obj.get("stop_sequences"), Some(&json!(["\n\nHuman:"])));
        assert!(!obj.contains_key("stop"));
    }

    #[test]
    fn test_encode_tools_with_input_schema() {
        let codec = AnthropicMessagesCodec;
        let original = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": { "type": "object" }
            }]
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        let tools = obj.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 1);
        // Should write input_schema (not parameters), and no function wrapper
        assert!(tools[0].get("input_schema").is_some());
        assert!(!tools[0].as_object().unwrap().contains_key("parameters"));
        assert!(!tools[0].as_object().unwrap().contains_key("type"));
        assert!(!tools[0].as_object().unwrap().contains_key("function"));
    }

    #[test]
    fn test_encode_tool_choice_anthropic_format() {
        let codec = AnthropicMessagesCodec;
        let original = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": { "type": "auto" }
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        assert_eq!(obj.get("tool_choice"), Some(&json!({"type": "auto"})));
    }

    #[test]
    fn test_encode_max_tokens() {
        let codec = AnthropicMessagesCodec;
        let original = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        // Should write max_tokens (not max_completion_tokens or max_output_tokens)
        assert_eq!(obj.get("max_tokens"), Some(&json!(200)));
    }

    #[test]
    fn test_encode_tool_choice_any_to_anthropic() {
        let codec = AnthropicMessagesCodec;
        let original = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": { "type": "any" }
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        assert_eq!(obj.get("tool_choice"), Some(&json!({"type": "any"})));
    }

    #[test]
    fn test_encode_tool_choice_specific_to_anthropic() {
        let codec = AnthropicMessagesCodec;
        let original = make_request(json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": { "type": "tool", "name": "my_func" }
        }));
        let annotated = codec.decode(&original).unwrap();
        let encoded = codec.encode(&annotated, &original).unwrap();
        let obj = encoded.content.as_object().unwrap();
        assert_eq!(
            obj.get("tool_choice"),
            Some(&json!({"type": "tool", "name": "my_func"}))
        );
    }
}
