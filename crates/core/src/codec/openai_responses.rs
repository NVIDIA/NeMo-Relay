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

use crate::api::llm::LlmRequest;
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, ApiSpecificRequest, ContentPart, FunctionDefinition, GenerationParams,
    Message, MessageContent, ProviderNativeComponent, ToolChoice, ToolChoiceFunction,
    ToolChoiceFunctionName, ToolDefinition,
};
use super::resolve::{ProviderSurface, ProviderSurfaceDescriptor};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, RawUsageCost, ResponseToolCall, Usage,
    estimate_cost_for_provider, infer_model_provider, provider_reported_cost,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Responses API.
pub struct OpenAIResponsesCodec;

pub(crate) const PROVIDER_SURFACE: ProviderSurfaceDescriptor = ProviderSurfaceDescriptor {
    surface: ProviderSurface::OpenAIResponses,
    detect_request: |obj, _| obj.contains_key("input") || obj.contains_key("instructions"),
    detect_response: |obj| {
        obj.get("output").is_some_and(Json::is_array)
            || obj.get("output_text").is_some_and(Json::is_string)
    },
    decode_request: |request| OpenAIResponsesCodec.decode(request),
    decode_response: |raw| OpenAIResponsesCodec.decode_response(raw),
    codec_name: "openai_responses",
    request_codec: || std::sync::Arc::new(OpenAIResponsesCodec),
    response_codec: || std::sync::Arc::new(OpenAIResponsesCodec),
    streaming_codec: || Box::new(OpenAIResponsesStreamingCodec::new()),
};

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
    previous_response_id: Option<String>,
    store: Option<bool>,
    service_tier: Option<String>,
    truncation: Option<Json>,
    reasoning: Option<Json>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawResponsesUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    input_tokens_details: Option<RawInputTokensDetails>,
    output_tokens_details: Option<RawOutputTokensDetails>,
    #[serde(rename = "cost_usd")]
    provider_cost: Option<f64>,
    cost: Option<RawUsageCost>,
}

#[derive(Deserialize, Clone)]
struct RawInputTokensDetails {
    cached_tokens: Option<u64>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize, Clone)]
struct RawOutputTokensDetails {
    reasoning_tokens: Option<u64>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
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

fn input_tokens_details_to_json(details: &RawInputTokensDetails) -> Json {
    let mut obj = serde_json::Map::new();
    if let Some(cached_tokens) = details.cached_tokens {
        obj.insert("cached_tokens".into(), Json::from(cached_tokens));
    }
    obj.extend(details.extra.clone());
    Json::Object(obj)
}

fn output_tokens_details_to_json(details: &RawOutputTokensDetails) -> Json {
    let mut obj = serde_json::Map::new();
    if let Some(reasoning_tokens) = details.reasoning_tokens {
        obj.insert("reasoning_tokens".into(), Json::from(reasoning_tokens));
    }
    obj.extend(details.extra.clone());
    Json::Object(obj)
}

/// Keys that are modeled in [`AnnotatedLlmRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "input",
    "instructions",
    "model",
    "max_output_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "store",
    "previous_response_id",
    "truncation",
    "reasoning",
    "include",
    "user",
    "metadata",
    "service_tier",
    "parallel_tool_calls",
    "max_tool_calls",
    "top_logprobs",
    "stream",
    "background",
    "context_management",
    "conversation",
    "moderation",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "safety_identifier",
    "stream_options",
    "text",
];

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

fn collect_output_parts(items: Option<&[Json]>) -> (Vec<String>, Vec<ResponseToolCall>) {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(items) = items {
        for item in items {
            collect_output_item(item, &mut text_parts, &mut tool_calls);
        }
    }

    (text_parts, tool_calls)
}

fn collect_output_item(
    item: &Json,
    text_parts: &mut Vec<String>,
    tool_calls: &mut Vec<ResponseToolCall>,
) {
    match item
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("")
    {
        "message" => collect_message_text_parts(item, text_parts),
        "output_text" => {
            if let Some(text) = output_text_block(item) {
                text_parts.push(text);
            }
        }
        "function_call" => tool_calls.push(parse_function_call(item)),
        _ => {}
    }
}

fn collect_message_text_parts(item: &Json, text_parts: &mut Vec<String>) {
    let Some(content) = item.get("content").and_then(|value| value.as_array()) else {
        return;
    };

    for block in content {
        if let Some(text) = output_text_block(block) {
            text_parts.push(text);
        }
    }
}

fn output_text_block(block: &Json) -> Option<String> {
    (block.get("type").and_then(|value| value.as_str()) == Some("output_text"))
        .then(|| block.get("text").and_then(|value| value.as_str()))
        .flatten()
        .map(str::to_string)
}

fn parse_function_call(item: &Json) -> ResponseToolCall {
    ResponseToolCall {
        id: item
            .get("call_id")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string(),
        name: item
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string(),
        arguments: item
            .get("arguments")
            .and_then(|value| value.as_str())
            .map(parse_arguments)
            .unwrap_or(Json::Object(serde_json::Map::new())),
    }
}

fn message_from_text_parts(text_parts: Vec<String>) -> Option<MessageContent> {
    match text_parts.as_slice() {
        [] => None,
        [text] => Some(MessageContent::Text(text.clone())),
        _ => Some(MessageContent::Text(text_parts.join("\n"))),
    }
}

fn top_level_output_text(response: &Json) -> Option<MessageContent> {
    response
        .get("output_text")
        .and_then(|value| value.as_str())
        .filter(|text| !text.is_empty())
        .map(|text| MessageContent::Text(text.to_string()))
}

fn optional_vec<T>(items: Vec<T>) -> Option<Vec<T>> {
    (!items.is_empty()).then_some(items)
}

fn responses_native(kind: &str, value: &Json) -> ProviderNativeComponent {
    ProviderNativeComponent {
        provider: "openai_responses".into(),
        kind: kind.to_string(),
        value: value.clone(),
    }
}

fn decode_responses_content(value: &Json) -> Result<MessageContent> {
    if let Some(text) = value.as_str() {
        return Ok(MessageContent::Text(text.to_string()));
    }
    let parts = value.as_array().ok_or_else(|| {
        FlowError::InvalidArgument(
            "OpenAI Responses message content must be a string or array".into(),
        )
    })?;
    Ok(MessageContent::Parts(
        parts
            .iter()
            .map(decode_responses_content_part)
            .collect::<Result<Vec<_>>>()?,
    ))
}

fn decode_responses_content_part(value: &Json) -> Result<ContentPart> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Responses content part must be an object".into())
    })?;
    let kind = obj.get("type").and_then(Json::as_str).unwrap_or("unknown");
    match kind {
        "input_text" | "output_text" => Ok(ContentPart::Text {
            text: obj
                .get("text")
                .and_then(Json::as_str)
                .ok_or_else(|| {
                    FlowError::InvalidArgument("OpenAI Responses text part is missing text".into())
                })?
                .to_string(),
            extra: obj
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "text"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        "input_image" => Ok(ContentPart::Image {
            image: Json::Object(
                obj.iter()
                    .filter(|(key, _)| matches!(key.as_str(), "image_url" | "file_id" | "detail"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
            extra: obj
                .iter()
                .filter(|(key, _)| {
                    !matches!(key.as_str(), "type" | "image_url" | "file_id" | "detail")
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        "input_file" => Ok(ContentPart::File {
            file: Json::Object(
                obj.iter()
                    .filter(|(key, _)| {
                        matches!(
                            key.as_str(),
                            "file_data" | "file_id" | "file_url" | "filename"
                        )
                    })
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
            extra: obj
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "type" | "file_data" | "file_id" | "file_url" | "filename"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        "refusal" => Ok(ContentPart::Refusal {
            refusal: obj
                .get("refusal")
                .and_then(Json::as_str)
                .ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "OpenAI Responses refusal part is missing refusal".into(),
                    )
                })?
                .to_string(),
            extra: obj
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "refusal"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        _ => Ok(ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: kind.to_string(),
            value: value.clone(),
        }),
    }
}

fn decode_responses_input_item(value: &Json) -> Result<Message> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Responses input item must be an object".into())
    })?;
    if let Some(role) = obj.get("role").and_then(Json::as_str) {
        if obj
            .keys()
            .any(|key| !matches!(key.as_str(), "type" | "role" | "content"))
        {
            return Ok(Message::ProviderNative {
                provider: "openai_responses".into(),
                kind: "message".into(),
                value: value.clone(),
            });
        }
        let content = decode_responses_content(obj.get("content").ok_or_else(|| {
            FlowError::InvalidArgument("OpenAI Responses message is missing content".into())
        })?)?;
        return Ok(match role {
            "user" => Message::User {
                content,
                name: None,
            },
            "system" => Message::System {
                content,
                name: None,
            },
            "developer" => Message::Developer {
                content,
                name: None,
            },
            "assistant" => Message::Assistant {
                content: Some(content),
                tool_calls: None,
                name: None,
            },
            _ => Message::ProviderNative {
                provider: "openai_responses".into(),
                kind: "message".into(),
                value: value.clone(),
            },
        });
    }

    let kind = obj.get("type").and_then(Json::as_str).unwrap_or("unknown");
    match kind {
        "function_call" => {
            let call_id = obj.get("call_id").and_then(Json::as_str).ok_or_else(|| {
                FlowError::InvalidArgument(
                    "OpenAI Responses function_call is missing call_id".into(),
                )
            })?;
            let name = obj.get("name").and_then(Json::as_str).ok_or_else(|| {
                FlowError::InvalidArgument("OpenAI Responses function_call is missing name".into())
            })?;
            let arguments = obj.get("arguments").and_then(Json::as_str).ok_or_else(|| {
                FlowError::InvalidArgument(
                    "OpenAI Responses function_call is missing arguments".into(),
                )
            })?;
            let id = match obj.get("id") {
                Some(Json::String(id)) => Some(id.clone()),
                Some(Json::Null) | None => None,
                Some(_) => {
                    return Err(FlowError::InvalidArgument(
                        "OpenAI Responses function_call id must be a string or null".into(),
                    ));
                }
            };
            Ok(Message::ToolCallItem {
                id,
                call_id: call_id.to_string(),
                name: name.to_string(),
                arguments: parse_arguments(arguments),
                extra: obj
                    .iter()
                    .filter(|(key, _)| {
                        !matches!(
                            key.as_str(),
                            "type" | "id" | "call_id" | "name" | "arguments"
                        )
                    })
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            })
        }
        "function_call_output" => {
            let call_id = obj.get("call_id").and_then(Json::as_str).ok_or_else(|| {
                FlowError::InvalidArgument(
                    "OpenAI Responses function_call_output is missing call_id".into(),
                )
            })?;
            let output = obj.get("output").ok_or_else(|| {
                FlowError::InvalidArgument(
                    "OpenAI Responses function_call_output is missing output".into(),
                )
            })?;
            let id = match obj.get("id") {
                Some(Json::String(id)) => Some(id.clone()),
                Some(Json::Null) | None => None,
                Some(_) => {
                    return Err(FlowError::InvalidArgument(
                        "OpenAI Responses function_call_output id must be a string or null".into(),
                    ));
                }
            };
            Ok(Message::ToolResultItem {
                id,
                call_id: call_id.to_string(),
                output: output.clone(),
                extra: obj
                    .iter()
                    .filter(|(key, _)| {
                        !matches!(key.as_str(), "type" | "id" | "call_id" | "output")
                    })
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            })
        }
        _ => Ok(Message::ProviderNative {
            provider: "openai_responses".into(),
            kind: kind.into(),
            value: value.clone(),
        }),
    }
}

fn encode_responses_content(content: &MessageContent, assistant: bool) -> Result<Json> {
    match content {
        MessageContent::Text(text) => Ok(Json::String(text.clone())),
        MessageContent::Parts(parts) => Ok(Json::Array(
            parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text, extra } => {
                        let mut obj = extra.clone();
                        obj.insert(
                            "type".into(),
                            Json::String(
                                if assistant {
                                    "output_text"
                                } else {
                                    "input_text"
                                }
                                .into(),
                            ),
                        );
                        obj.insert("text".into(), Json::String(text.clone()));
                        Ok(Json::Object(obj))
                    }
                    ContentPart::ImageUrl { image_url, extra } => {
                        let mut obj = extra.clone();
                        obj.insert("type".into(), Json::String("input_image".into()));
                        obj.insert("image_url".into(), Json::String(image_url.url.clone()));
                        if let Some(detail) = &image_url.detail {
                            obj.insert("detail".into(), Json::String(detail.clone()));
                        }
                        Ok(Json::Object(obj))
                    }
                    ContentPart::Image { image, extra } => {
                        let mut obj = image.as_object().cloned().ok_or_else(|| {
                            FlowError::InvalidArgument(
                                "OpenAI Responses image content must be an object".into(),
                            )
                        })?;
                        obj.extend(extra.clone());
                        obj.insert("type".into(), Json::String("input_image".into()));
                        Ok(Json::Object(obj))
                    }
                    ContentPart::File { file, extra } => {
                        let mut obj = file.as_object().cloned().ok_or_else(|| {
                            FlowError::InvalidArgument(
                                "OpenAI Responses file content must be an object".into(),
                            )
                        })?;
                        obj.extend(extra.clone());
                        obj.insert("type".into(), Json::String("input_file".into()));
                        Ok(Json::Object(obj))
                    }
                    ContentPart::Refusal { refusal, extra } if assistant => {
                        let mut obj = extra.clone();
                        obj.insert("type".into(), Json::String("refusal".into()));
                        obj.insert("refusal".into(), Json::String(refusal.clone()));
                        Ok(Json::Object(obj))
                    }
                    ContentPart::ProviderNative {
                        provider, value, ..
                    } if provider == "openai_responses" => Ok(value.clone()),
                    other => Err(FlowError::InvalidArgument(format!(
                        "content part {other:?} cannot be encoded for OpenAI Responses"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        )),
    }
}

fn encode_responses_input_item(message: &Message) -> Result<Json> {
    match message {
        Message::User { content, .. }
        | Message::System { content, .. }
        | Message::Developer { content, .. } => {
            let role = match message {
                Message::User { .. } => "user",
                Message::System { .. } => "system",
                Message::Developer { .. } => "developer",
                _ => unreachable!(),
            };
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), Json::String("message".into()));
            obj.insert("role".into(), Json::String(role.into()));
            obj.insert("content".into(), encode_responses_content(content, false)?);
            Ok(Json::Object(obj))
        }
        Message::Assistant {
            content: Some(content),
            ..
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), Json::String("message".into()));
            obj.insert("role".into(), Json::String("assistant".into()));
            obj.insert("content".into(), encode_responses_content(content, true)?);
            Ok(Json::Object(obj))
        }
        Message::ToolCallItem {
            id,
            call_id,
            name,
            arguments,
            extra,
        } => {
            let mut obj = extra.clone();
            obj.insert("type".into(), Json::String("function_call".into()));
            if let Some(id) = id {
                obj.insert("id".into(), Json::String(id.clone()));
            }
            obj.insert("call_id".into(), Json::String(call_id.clone()));
            obj.insert("name".into(), Json::String(name.clone()));
            let arguments = match arguments {
                Json::String(raw) => raw.clone(),
                value => serde_json::to_string(value).map_err(|error| {
                    FlowError::Internal(format!(
                        "OpenAI Responses function arguments encode: {error}"
                    ))
                })?,
            };
            obj.insert("arguments".into(), Json::String(arguments));
            Ok(Json::Object(obj))
        }
        Message::ToolResultItem {
            id,
            call_id,
            output,
            extra,
        } => {
            let mut obj = extra.clone();
            obj.insert("type".into(), Json::String("function_call_output".into()));
            if let Some(id) = id {
                obj.insert("id".into(), Json::String(id.clone()));
            }
            obj.insert("call_id".into(), Json::String(call_id.clone()));
            obj.insert("output".into(), output.clone());
            Ok(Json::Object(obj))
        }
        Message::ProviderNative {
            provider, value, ..
        } if provider == "openai_responses" => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "message {other:?} cannot be encoded for OpenAI Responses"
        ))),
    }
}

fn decode_responses_tool(value: &Json) -> Result<ToolDefinition> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Responses tool must be an object".into())
    })?;
    if obj.get("type").and_then(Json::as_str) != Some("function") {
        let kind = obj.get("type").and_then(Json::as_str).unwrap_or("unknown");
        return Ok(ToolDefinition::ProviderNative {
            provider: "openai_responses".into(),
            kind: kind.into(),
            value: value.clone(),
        });
    }
    let (function, wrapper_extra) =
        if let Some(function) = obj.get("function").and_then(Json::as_object) {
            (
                function,
                obj.iter()
                    .filter(|(key, _)| !matches!(key.as_str(), "type" | "function"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            )
        } else {
            (obj, serde_json::Map::new())
        };
    let name = function.get("name").and_then(Json::as_str).ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Responses function tool is missing name".into())
    })?;
    let description =
        super::optional_string(function, "description", "OpenAI Responses function tool")?;
    let strict = super::optional_bool(function, "strict", "OpenAI Responses function tool")?;
    Ok(ToolDefinition::Function {
        function: FunctionDefinition {
            name: name.into(),
            description,
            parameters: function.get("parameters").cloned(),
            strict,
            extra: function
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "type" | "name" | "description" | "parameters" | "strict" | "function"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        },
        extra: wrapper_extra,
    })
}

fn encode_responses_tool(tool: &ToolDefinition) -> Result<Json> {
    match tool {
        ToolDefinition::Function { function, extra } => {
            let mut obj = extra.clone();
            let Json::Object(function) = encode_responses_function(function) else {
                unreachable!("function definition encodes as an object")
            };
            obj.extend(function);
            obj.insert("type".into(), Json::String("function".into()));
            Ok(Json::Object(obj))
        }
        ToolDefinition::ProviderNative {
            provider, value, ..
        } if provider == "openai_responses" => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "tool {other:?} cannot be encoded for OpenAI Responses"
        ))),
    }
}

fn encode_responses_function(function: &FunctionDefinition) -> Json {
    let mut obj = function.extra.clone();
    obj.insert("name".into(), Json::String(function.name.clone()));
    if let Some(description) = &function.description {
        obj.insert("description".into(), Json::String(description.clone()));
    }
    if let Some(parameters) = &function.parameters {
        obj.insert("parameters".into(), parameters.clone());
    }
    if let Some(strict) = function.strict {
        obj.insert("strict".into(), Json::Bool(strict));
    }
    Json::Object(obj)
}

fn patch_responses_tool(
    original: &Json,
    baseline: &ToolDefinition,
    edited: &ToolDefinition,
    baseline_value: &Json,
    edited_value: &Json,
) -> Result<Json> {
    if let (
        Some(original),
        ToolDefinition::Function {
            function: baseline_function,
            extra: baseline_extra,
        },
        ToolDefinition::Function {
            function: edited_function,
            extra: edited_extra,
        },
    ) = (original.as_object(), baseline, edited)
        && let Some(original_function) = original.get("function")
    {
        let mut patched = original.clone();
        patch_extra_fields(&mut patched, baseline_extra, edited_extra);
        patched.insert(
            "function".into(),
            super::patch_changed_json(
                original_function,
                &encode_responses_function(baseline_function),
                &encode_responses_function(edited_function),
            )?,
        );
        return Ok(Json::Object(patched));
    }

    super::patch_changed_json(original, baseline_value, edited_value)
}

fn decode_openai_or_anthropic_tool_choice(value: &Json) -> ToolChoice {
    match value.as_str() {
        Some("auto") => ToolChoice::Auto,
        Some("none") => ToolChoice::None,
        Some("required") => ToolChoice::Required,
        _ => match value.as_object().and_then(|obj| {
            let choice_type = obj.get("type").and_then(Json::as_str)?;
            match choice_type {
                "auto" => Some(ToolChoice::Auto),
                "any" => Some(ToolChoice::Required),
                "none" => Some(ToolChoice::None),
                "tool" | "function" => obj
                    .get("name")
                    .and_then(Json::as_str)
                    .or_else(|| {
                        obj.get("function")
                            .and_then(Json::as_object)
                            .and_then(|function| function.get("name"))
                            .and_then(Json::as_str)
                    })
                    .map(|name| {
                        ToolChoice::Specific(ToolChoiceFunction {
                            choice_type: "function".into(),
                            function: ToolChoiceFunctionName { name: name.into() },
                        })
                    }),
                _ => None,
            }
        }) {
            Some(choice) => choice,
            None => ToolChoice::ProviderNative(responses_native("tool_choice", value)),
        },
    }
}

fn encode_responses_tool_choice(choice: &ToolChoice) -> Result<Json> {
    match choice {
        ToolChoice::Auto => Ok(Json::String("auto".into())),
        ToolChoice::None => Ok(Json::String("none".into())),
        ToolChoice::Required => Ok(Json::String("required".into())),
        ToolChoice::Specific(choice) => Ok(serde_json::json!({
            "type":"function",
            "name":choice.function.name,
        })),
        ToolChoice::ProviderNative(native) if native.provider == "openai_responses" => {
            Ok(native.value.clone())
        }
        ToolChoice::ProviderNative(native) => Err(FlowError::InvalidArgument(format!(
            "tool choice for {} cannot be encoded for OpenAI Responses",
            native.provider
        ))),
    }
}

fn patch_extra_fields(
    obj: &mut serde_json::Map<String, Json>,
    baseline: &serde_json::Map<String, Json>,
    edited: &serde_json::Map<String, Json>,
) {
    for key in baseline.keys().filter(|key| !edited.contains_key(*key)) {
        obj.remove(key);
    }
    for (key, value) in edited {
        if baseline.get(key) != Some(value) {
            obj.insert(key.clone(), value.clone());
        }
    }
}

fn set_or_remove_json(obj: &mut serde_json::Map<String, Json>, key: &str, value: Option<Json>) {
    if let Some(value) = value {
        obj.insert(key.into(), value);
    } else {
        obj.remove(key);
    }
}

fn patch_responses_api_specific(
    obj: &mut serde_json::Map<String, Json>,
    edited: &Option<ApiSpecificRequest>,
    baseline: &Option<ApiSpecificRequest>,
) -> Result<()> {
    match (edited, baseline) {
        (
            Some(ApiSpecificRequest::OpenAIResponses {
                background,
                context_management,
                conversation,
                moderation,
                prompt,
                prompt_cache_key,
                prompt_cache_options,
                prompt_cache_retention,
                safety_identifier,
                stream_options,
                text,
            }),
            Some(ApiSpecificRequest::OpenAIResponses {
                background: old_background,
                context_management: old_context_management,
                conversation: old_conversation,
                moderation: old_moderation,
                prompt: old_prompt,
                prompt_cache_key: old_prompt_cache_key,
                prompt_cache_options: old_prompt_cache_options,
                prompt_cache_retention: old_prompt_cache_retention,
                safety_identifier: old_safety_identifier,
                stream_options: old_stream_options,
                text: old_text,
            }),
        ) => {
            if background != old_background {
                set_or_remove_json(obj, "background", background.map(Json::Bool));
            }
            for (key, value, old_value) in [
                (
                    "context_management",
                    context_management,
                    old_context_management,
                ),
                ("conversation", conversation, old_conversation),
                ("moderation", moderation, old_moderation),
                ("prompt", prompt, old_prompt),
                (
                    "prompt_cache_options",
                    prompt_cache_options,
                    old_prompt_cache_options,
                ),
                ("stream_options", stream_options, old_stream_options),
                ("text", text, old_text),
            ] {
                if value != old_value {
                    set_or_remove_json(obj, key, value.clone());
                }
            }
            for (key, value, old_value) in [
                ("prompt_cache_key", prompt_cache_key, old_prompt_cache_key),
                (
                    "prompt_cache_retention",
                    prompt_cache_retention,
                    old_prompt_cache_retention,
                ),
                (
                    "safety_identifier",
                    safety_identifier,
                    old_safety_identifier,
                ),
            ] {
                if value != old_value {
                    set_or_remove_json(obj, key, value.clone().map(Json::String));
                }
            }
            Ok(())
        }
        (None, Some(ApiSpecificRequest::OpenAIResponses { .. })) => {
            for key in [
                "background",
                "context_management",
                "conversation",
                "moderation",
                "prompt",
                "prompt_cache_key",
                "prompt_cache_options",
                "prompt_cache_retention",
                "safety_identifier",
                "stream_options",
                "text",
            ] {
                obj.remove(key);
            }
            Ok(())
        }
        (Some(_), _) => Err(FlowError::InvalidArgument(
            "api_specific provider does not match OpenAI Responses".into(),
        )),
        (None, Some(_)) => Err(FlowError::InvalidArgument(
            "api_specific provider does not match OpenAI Responses".into(),
        )),
        (None, None) => Ok(()),
    }
}

fn decode_openai_or_anthropic_parallel_tool_calls(
    obj: &serde_json::Map<String, Json>,
) -> Result<Option<bool>> {
    if let Some(value) = super::optional_bool(obj, "parallel_tool_calls", "OpenAI Responses")? {
        return Ok(Some(value));
    }
    let Some(tool_choice) = obj.get("tool_choice").and_then(Json::as_object) else {
        return Ok(None);
    };
    Ok(super::optional_bool(
        tool_choice,
        "disable_parallel_tool_use",
        "OpenAI Responses tool_choice",
    )?
    .map(|disabled| !disabled))
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for OpenAIResponsesCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: RawResponsesResponse = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("OpenAI Responses response decode: {e}")))?;

        let all_output_items = raw.output.clone();
        let (text_parts, tool_calls) = collect_output_parts(raw.output.as_deref());
        let message =
            message_from_text_parts(text_parts).or_else(|| top_level_output_text(response));
        let tool_calls = optional_vec(tool_calls);

        // Map finish reason from status + incomplete_details.
        let finish_reason =
            map_responses_finish_reason(raw.status.as_deref(), raw.incomplete_details.as_ref());

        let input_tokens_details = raw.usage.as_ref().and_then(|u| {
            u.input_tokens_details
                .as_ref()
                .map(input_tokens_details_to_json)
        });
        let output_tokens_details = raw.usage.as_ref().and_then(|u| {
            u.output_tokens_details
                .as_ref()
                .map(output_tokens_details_to_json)
        });

        // Map usage.
        let model_for_pricing = raw.model.as_deref();
        let model_provider = infer_model_provider("openai", model_for_pricing);
        let usage = raw.usage.map(|u| {
            let mut usage = Usage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.total_tokens,
                cache_read_tokens: u
                    .input_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens),
                cache_write_tokens: None,
                cost: provider_reported_cost(u.provider_cost, u.cost),
            };
            if usage.cost.is_none() {
                usage.cost = model_for_pricing.and_then(|model| {
                    estimate_cost_for_provider(model_provider.as_deref(), model, &usage)
                });
            }
            usage
        });

        // Build API-specific fields.
        let api_specific = Some(ApiSpecificResponse::OpenAIResponses {
            output_items: all_output_items,
            status: raw.status,
            incomplete_details: raw.incomplete_details,
            previous_response_id: raw.previous_response_id,
            store: raw.store,
            service_tier: raw.service_tier,
            truncation: raw.truncation,
            reasoning: raw.reasoning,
            input_tokens_details,
            output_tokens_details,
        });

        Ok(AnnotatedLlmResponse {
            id: raw.id,
            model: raw.model,
            message,
            tool_calls,
            finish_reason,
            usage,
            optimization_summary: None,
            api_specific,
            extra: raw.extra,
        })
    }
}

// ---------------------------------------------------------------------------
// LlmCodec implementation
// ---------------------------------------------------------------------------

impl LlmCodec for OpenAIResponsesCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;
        let input = obj.get("input").ok_or_else(|| {
            FlowError::InvalidArgument("OpenAI Responses request is missing input".into())
        })?;
        let messages = if let Some(input) = input.as_str() {
            vec![Message::User {
                content: MessageContent::Text(input.to_string()),
                name: None,
            }]
        } else {
            input
                .as_array()
                .ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "OpenAI Responses input must be a string or an array".into(),
                    )
                })?
                .iter()
                .map(decode_responses_input_item)
                .collect::<Result<Vec<_>>>()?
        };
        let instructions = match obj.get("instructions") {
            Some(Json::String(instructions)) => Some(MessageContent::Text(instructions.clone())),
            Some(Json::Null) | None => None,
            Some(_) => {
                return Err(FlowError::InvalidArgument(
                    "OpenAI Responses instructions must be a string or null".into(),
                ));
            }
        };
        let model = super::optional_string(obj, "model", "OpenAI Responses")?;
        let temperature = super::optional_f64(obj, "temperature", "OpenAI Responses")?;
        let top_p = super::optional_f64(obj, "top_p", "OpenAI Responses")?;
        let max_tokens = super::optional_u64(obj, "max_output_tokens", "OpenAI Responses")?;
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
        let tools = obj
            .get("tools")
            .map(|value| {
                value
                    .as_array()
                    .ok_or_else(|| {
                        FlowError::InvalidArgument("OpenAI Responses tools must be an array".into())
                    })?
                    .iter()
                    .map(decode_responses_tool)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let tool_choice = obj
            .get("tool_choice")
            .map(decode_openai_or_anthropic_tool_choice);
        let store = super::optional_bool(obj, "store", "OpenAI Responses")?;
        let previous_response_id =
            super::optional_string(obj, "previous_response_id", "OpenAI Responses")?;
        let user = super::optional_string(obj, "user", "OpenAI Responses")?;
        let service_tier = super::optional_string(obj, "service_tier", "OpenAI Responses")?;
        let parallel_tool_calls = decode_openai_or_anthropic_parallel_tool_calls(obj)?;
        let max_tool_calls = super::optional_u64(obj, "max_tool_calls", "OpenAI Responses")?;
        let top_logprobs = super::optional_u64(obj, "top_logprobs", "OpenAI Responses")?;
        let stream = super::optional_bool(obj, "stream", "OpenAI Responses")?;
        let background = super::optional_bool(obj, "background", "OpenAI Responses")?;
        let prompt_cache_key = super::optional_string(obj, "prompt_cache_key", "OpenAI Responses")?;
        let prompt_cache_retention =
            super::optional_string(obj, "prompt_cache_retention", "OpenAI Responses")?;
        let safety_identifier =
            super::optional_string(obj, "safety_identifier", "OpenAI Responses")?;
        let reasoning = super::optional_object(obj, "reasoning", "OpenAI Responses")?;
        let include = super::optional_array(obj, "include", "OpenAI Responses")?;
        let metadata = super::optional_object(obj, "metadata", "OpenAI Responses")?;
        let context_management =
            super::optional_array(obj, "context_management", "OpenAI Responses")?;
        let moderation = super::optional_object(obj, "moderation", "OpenAI Responses")?;
        let prompt = super::optional_object(obj, "prompt", "OpenAI Responses")?;
        let prompt_cache_options =
            super::optional_object(obj, "prompt_cache_options", "OpenAI Responses")?;
        let stream_options = super::optional_object(obj, "stream_options", "OpenAI Responses")?;
        let text = super::optional_object(obj, "text", "OpenAI Responses")?;
        let extra: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| !MODELED_REQUEST_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(AnnotatedLlmRequest {
            messages,
            instructions,
            model,
            params,
            tools,
            tool_choice,
            store,
            previous_response_id,
            truncation: obj.get("truncation").cloned(),
            reasoning,
            include,
            user,
            metadata,
            service_tier,
            parallel_tool_calls,
            max_output_tokens: max_tokens,
            max_tool_calls,
            top_logprobs,
            stream,
            api_specific: Some(ApiSpecificRequest::OpenAIResponses {
                background,
                context_management,
                conversation: obj.get("conversation").cloned(),
                moderation,
                prompt,
                prompt_cache_key,
                prompt_cache_options,
                prompt_cache_retention,
                safety_identifier,
                stream_options,
                text,
            }),
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let baseline = self.decode(original)?;
        let mut content = original.content.clone();
        let obj = content
            .as_object_mut()
            .ok_or_else(|| FlowError::Internal("original content is not an object".into()))?;
        if annotated.messages != baseline.messages {
            let input = if original.content.get("input").is_some_and(Json::is_string)
                && matches!(
                    annotated.messages.as_slice(),
                    [Message::User {
                        content: MessageContent::Text(_),
                        name: None
                    }]
                ) {
                match &annotated.messages[0] {
                    Message::User {
                        content: MessageContent::Text(text),
                        ..
                    } => Json::String(text.clone()),
                    _ => unreachable!(),
                }
            } else {
                Json::Array(super::encode_changed_items(
                    &annotated.messages,
                    &baseline.messages,
                    original
                        .content
                        .get("input")
                        .and_then(Json::as_array)
                        .map(Vec::as_slice),
                    encode_responses_input_item,
                )?)
            };
            obj.insert("input".into(), input);
        }
        if annotated.instructions != baseline.instructions {
            let instructions = match &annotated.instructions {
                Some(MessageContent::Text(text)) => Some(Json::String(text.clone())),
                Some(MessageContent::Parts(_)) => {
                    return Err(FlowError::InvalidArgument(
                        "OpenAI Responses instructions cannot contain content parts".into(),
                    ));
                }
                None => None,
            };
            set_or_remove_json(obj, "instructions", instructions);
        }
        if annotated.model != baseline.model {
            set_or_remove_json(obj, "model", annotated.model.clone().map(Json::String));
        }
        if annotated.params != baseline.params {
            let edited = annotated.params.as_ref();
            let before = baseline.params.as_ref();
            if edited.and_then(|params| params.stop.as_ref())
                != before.and_then(|params| params.stop.as_ref())
                && edited.and_then(|params| params.stop.as_ref()).is_some()
            {
                return Err(FlowError::InvalidArgument(
                    "OpenAI Responses does not support stop sequences".into(),
                ));
            }
            for (key, value, old_value) in [
                (
                    "temperature",
                    edited.and_then(|params| params.temperature),
                    before.and_then(|params| params.temperature),
                ),
                (
                    "top_p",
                    edited.and_then(|params| params.top_p),
                    before.and_then(|params| params.top_p),
                ),
            ] {
                if value != old_value {
                    set_or_remove_json(obj, key, value.map(json_f64));
                }
            }
            let max_tokens = edited.and_then(|params| params.max_tokens);
            let old_max_tokens = before.and_then(|params| params.max_tokens);
            if max_tokens != old_max_tokens {
                set_or_remove_json(obj, "max_output_tokens", max_tokens.map(Json::from));
            }
        }
        if annotated.tools != baseline.tools {
            let tools = annotated
                .tools
                .as_deref()
                .map(|tools| {
                    super::encode_changed_items_with_patch(
                        tools,
                        baseline.tools.as_deref().unwrap_or(&[]),
                        obj.get("tools").and_then(Json::as_array).map(Vec::as_slice),
                        encode_responses_tool,
                        patch_responses_tool,
                    )
                })
                .transpose()?
                .map(Json::Array);
            set_or_remove_json(obj, "tools", tools);
        }
        if annotated.tool_choice != baseline.tool_choice {
            let tool_choice = match (&annotated.tool_choice, &baseline.tool_choice) {
                (Some(edited), Some(before)) => {
                    let edited = encode_responses_tool_choice(edited)?;
                    let before = encode_responses_tool_choice(before)?;
                    Some(match obj.get("tool_choice") {
                        Some(original) => super::patch_changed_json(original, &before, &edited)?,
                        None => edited,
                    })
                }
                (Some(edited), None) => Some(encode_responses_tool_choice(edited)?),
                (None, _) => None,
            };
            set_or_remove_json(obj, "tool_choice", tool_choice);
        }
        for (key, value, old_value) in [
            ("truncation", &annotated.truncation, &baseline.truncation),
            ("reasoning", &annotated.reasoning, &baseline.reasoning),
            ("include", &annotated.include, &baseline.include),
            ("metadata", &annotated.metadata, &baseline.metadata),
        ] {
            if value != old_value {
                set_or_remove_json(obj, key, value.clone());
            }
        }
        for (key, value, old_value) in [
            (
                "previous_response_id",
                &annotated.previous_response_id,
                &baseline.previous_response_id,
            ),
            ("user", &annotated.user, &baseline.user),
            (
                "service_tier",
                &annotated.service_tier,
                &baseline.service_tier,
            ),
        ] {
            if value != old_value {
                set_or_remove_json(obj, key, value.clone().map(Json::String));
            }
        }
        for (key, value, old_value) in [
            ("store", annotated.store, baseline.store),
            (
                "parallel_tool_calls",
                annotated.parallel_tool_calls,
                baseline.parallel_tool_calls,
            ),
            ("stream", annotated.stream, baseline.stream),
        ] {
            if value != old_value {
                set_or_remove_json(obj, key, value.map(Json::Bool));
            }
        }
        for (key, value, old_value) in [
            (
                "max_output_tokens",
                annotated.max_output_tokens,
                baseline.max_output_tokens,
            ),
            (
                "max_tool_calls",
                annotated.max_tool_calls,
                baseline.max_tool_calls,
            ),
            (
                "top_logprobs",
                annotated.top_logprobs,
                baseline.top_logprobs,
            ),
        ] {
            if value != old_value {
                set_or_remove_json(obj, key, value.map(Json::from));
            }
        }
        patch_responses_api_specific(obj, &annotated.api_specific, &baseline.api_specific)?;
        patch_extra_fields(obj, &baseline.extra, &annotated.extra);

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`OpenAIResponsesCodec`].
///
/// Replays the OpenAI Responses SSE event sequence into the same JSON shape the API returns for a
/// non-streaming request (`{id, model, status, output, usage, incomplete_details, ...}`). Once
/// finalized, the assembled JSON can be fed back through [`OpenAIResponsesCodec::decode_response`]
/// to produce the canonical [`AnnotatedLlmResponse`].
///
/// # Strategy
///
/// The Responses API is a relatively forgiving streaming target because every event carries
/// either the full `response` snapshot (`response.created`, `response.in_progress`,
/// `response.completed`, `response.failed`, `response.incomplete`) or the final-state output item
/// (`response.output_item.done`). We:
///
/// 1. Track the latest `response` snapshot — terminal events (`completed`/`failed`/`incomplete`)
///    typically carry the complete state including `output`, so we prefer those when present.
/// 2. Track output items by `output_index` — `output_item.done` events deliver the final per-item
///    state, used as a fallback when the terminal `response.output` is missing or empty.
/// 3. Per-token `output_text.delta` and `function_call_arguments.delta` events are ignored
///    because their content is redelivered in the matching `output_item.done` event. Skipping
///    deltas keeps the codec resilient to schema additions and avoids double-accumulation.
///
/// Internal state lives behind `Arc<Mutex<...>>` so the `&self`-produced collector and finalizer
/// closures share access. Each instance is single-use because [`LlmFinalizerFn`] consumes the
/// finalize step.
///
/// [`AnnotatedLlmResponse`]: crate::codec::response::AnnotatedLlmResponse
/// [`LlmFinalizerFn`]: crate::api::runtime::LlmFinalizerFn
pub struct OpenAIResponsesStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<OpenAIResponsesStreamingState>>,
}

impl OpenAIResponsesStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(
                OpenAIResponsesStreamingState::default(),
            )),
        }
    }
}

impl Default for OpenAIResponsesStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for OpenAIResponsesStreamingCodec {
    fn collector(&self) -> crate::api::runtime::LlmCollectorFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move |event: Json| -> Result<()> {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.observe(&event);
            Ok(())
        })
    }

    fn finalizer(&self) -> crate::api::runtime::LlmFinalizerFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move || -> Json {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *guard).finalize()
        })
    }
}

#[derive(Debug, Default)]
struct OpenAIResponsesStreamingState {
    /// Latest `response` snapshot from any event that carries one. Last write wins, so terminal
    /// events with the complete state will end up here when they fire.
    response: Option<serde_json::Map<String, Json>>,
    /// Items keyed by `output_index`. Captured from `response.output_item.added` (initial) and
    /// replaced on `response.output_item.done` (final). Used as a fallback for `output` when the
    /// terminal `response` snapshot lacks it.
    items: std::collections::BTreeMap<usize, Json>,
}

impl OpenAIResponsesStreamingState {
    fn observe(&mut self, event: &Json) {
        let event_type = event.get("type").and_then(Json::as_str).unwrap_or("");
        match event_type {
            "response.created"
            | "response.in_progress"
            | "response.completed"
            | "response.failed"
            | "response.incomplete" => self.observe_response_snapshot(event),
            "response.output_item.added" | "response.output_item.done" => {
                self.observe_output_item(event);
            }
            // response.output_text.delta, response.function_call_arguments.delta,
            // response.content_part.added/done — content is redelivered in output_item.done, so we
            // don't accumulate deltas. Unknown events are ignored.
            _ => {}
        }
    }

    fn observe_response_snapshot(&mut self, event: &Json) {
        let Some(response) = event.get("response") else {
            return;
        };
        if let Json::Object(map) = response {
            self.response = Some(map.clone());
        }
    }

    fn observe_output_item(&mut self, event: &Json) {
        let Some(index) = event.get("output_index").and_then(Json::as_u64) else {
            return;
        };
        let Some(item) = event.get("item") else {
            return;
        };
        self.items.insert(index as usize, item.clone());
    }

    fn finalize(self) -> Json {
        let mut output = self.response.unwrap_or_default();
        // If the latest snapshot lacked `output` (or has an empty array because it came from an
        // early `response.created` event), backfill from per-item accumulator. Terminal events
        // typically carry the complete output, so this branch is a safety net for truncated
        // streams or schemas that drop output from terminal events.
        let snapshot_output_empty = output
            .get("output")
            .and_then(Json::as_array)
            .map(|arr| arr.is_empty())
            .unwrap_or(true);
        if snapshot_output_empty && !self.items.is_empty() {
            let items: Vec<Json> = self.items.into_values().collect();
            output.insert("output".to_string(), Json::Array(items));
        }
        Json::Object(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/openai_responses_tests.rs"]
mod tests;
