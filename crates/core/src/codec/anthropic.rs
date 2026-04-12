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

use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::types::llm::LLMRequest;

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
        let raw: RawAnthropicResponse = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("Anthropic Messages response decode: {e}")))?;

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
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;

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
            if defs.is_empty() { None } else { Some(defs) }
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
            .ok_or_else(|| FlowError::Internal("original content is not an object".into()))?;

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
                FlowError::Internal(format!("Anthropic Messages messages encode: {e}"))
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
                        FlowError::Internal(format!(
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
                    FlowError::Internal(format!("Anthropic Messages tools encode: {e}"))
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
#[path = "../../tests/unit/codec/anthropic_tests.rs"]
mod tests;
