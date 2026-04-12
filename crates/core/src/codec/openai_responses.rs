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
use crate::types::llm::LLMRequest;

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
#[path = "../../tests/unit/codec/openai_responses_tests.rs"]
mod tests;
