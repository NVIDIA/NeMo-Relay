// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the OpenAI Chat Completions API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the OpenAI Chat Completions format.

use serde::Deserialize;

use crate::error::{FlowError, Result};
use crate::json::Json;
use crate::types::llm::LLMRequest;

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
            .map_err(|e| FlowError::Internal(format!("OpenAI Chat response decode: {e}")))?;

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
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;

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
            .map_err(|e| FlowError::Internal(format!("OpenAI Chat tools decode: {e}")))?;

        // Extract tool_choice.
        let tool_choice: Option<ToolChoice> = obj
            .get("tool_choice")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| FlowError::Internal(format!("OpenAI Chat tool_choice decode: {e}")))?;

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

        // Overlay messages.
        obj.insert(
            "messages".into(),
            serde_json::to_value(&annotated.messages)
                .map_err(|e| FlowError::Internal(format!("OpenAI Chat messages encode: {e}")))?,
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
                        FlowError::Internal(format!("OpenAI Chat stop encode: {e}"))
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
                    .map_err(|e| FlowError::Internal(format!("OpenAI Chat tools encode: {e}")))?,
            );
        }

        // Overlay tool_choice if present.
        if let Some(ref tool_choice) = annotated.tool_choice {
            obj.insert(
                "tool_choice".into(),
                serde_json::to_value(tool_choice).map_err(|e| {
                    FlowError::Internal(format!("OpenAI Chat tool_choice encode: {e}"))
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
#[path = "../../tests/unit/codec/openai_chat_tests.rs"]
mod tests;
